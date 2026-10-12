package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Протокол FERET (PLAN.md п. 5.5; принят Хозяином 11.10.2026, план кода подтверждён). Кадры — FeretData (fa, fb; YuNet;
 * кадр «а» и alignCrop как у GT); отбор по позе |r| ≤ R (fei_selection.tsv). Роли — по датам с анфасом до отбора:
 * <ul>
 *   <li>(1) межсессионно — люди с ≥ 2 датами: галерея — снимки первой даты, пробы — более поздних; опора внутри сессии —
 *       галерея fa первой даты, пробы fb той же даты (люди с fa и fb после отбора);</li>
 *   <li>(2) регистрация по нескольким дням — люди с ≥ 3 датами, одни и те же во всех вариантах и одни и те же пробы (даты
 *       3 и позже): G1 — дата 1, G2-i — даты 1 и 2 до среднего человека, G2-ii — даты 1 и 2 до ближайшего шаблона сессии
 *       (проекция MLDA по людям, шаблоны — средние по датам, оценка — наименьшая по датам человека), G2' — по одному снимку
 *       дат 1 и 2 (fa без флага, иначе первый fa по имени, иначе первый fb);</li>
 *   <li>(3) контрольные чужие — 200 людей с одной датой (перестановка Random(seed + 9600), первые 200 с отобранными
 *       снимками), все их отобранные снимки;</li>
 *   <li>(4) когорта MLDA — встроенная FEI 20 (как на GT).</li>
 * </ul>
 * Человек остаётся в роли, если после детекции и отбора у него есть снимки галереи и проба (в (2) — снимки дат 1, 2 и проба).
 * Методы: L1, L1-tt, LTP-in-tt (t = faces.ltp.t), A справочно, SFace — верхняя граница; зеркала и выбор модели по позе — как
 * на GT; порог П2 / П3 по «чужим» галереи своей конфигурации (5 частей, XBaseOther.cvParts, seed 100 + s); рядом —
 * разделимость при равном фактическом FAR контроля; бутстреп по людям роли и контроля. Отчёт, файлы оценок, контрольная точка
 * — только в каталоге базы (reports/, derived/checkpoint_feret/).
 *
 * @author ssv
 */
final class XBaseFeret {

    static final long CTRL_SEED_OFFSET = 9600;
    static final int CTRL = 200;
    /** Интервалы между датой пробы и первой датой человека, дни: ≤ 30, 31–365, > 365. */
    static final long[] BINS = {30, 365};
    static final String[] BIN_NAMES = {"≤ 30 дн.", "31–365 дн.", "> 365 дн."};

    private XBaseFeret() {
    }

    // ---------------------------------------------------------------- выбор снимков и ролей

    /** Даты с анфасом человека (до отбора), по возрастанию. */
    static Map<String, TreeSet<String>> dates(List<Img> imgs) {
        Map<String, TreeSet<String>> d = new TreeMap<>();
        for (Img im : imgs) d.computeIfAbsent(im.person(), p -> new TreeSet<>()).add(im.date());
        return d;
    }

    /** Снимок G2' даты: fa без флага, иначе первый fa по имени, иначе первый fb по имени; null — снимков нет. */
    static String g2pChoice(List<Img> dateImgs) {
        List<Img> s = new ArrayList<>(dateImgs);
        s.sort((a, b) -> a.key().compareTo(b.key()));
        for (Img im : s) if (im.pose().equals("fa") && im.flag().equals("-")) return im.key();
        for (Img im : s) if (im.pose().equals("fa")) return im.key();
        for (Img im : s) if (im.pose().equals("fb")) return im.key();
        return null;
    }

    /** Контрольные чужие: люди с одной датой по возрастанию, Random(seed + 9600), первые 200 с отобранными снимками; по возрастанию. */
    static List<String> controls(List<String> oneDate, Set<String> withKept, long seed) {
        List<String> c = new ArrayList<>(oneDate.stream().sorted().toList());
        Collections.shuffle(c, new Random(seed + CTRL_SEED_OFFSET));
        List<String> out = new ArrayList<>();
        for (String p : c) {
            if (out.size() == CTRL) break;
            if (withKept.contains(p)) out.add(p);
        }
        if (out.size() < CTRL) throw new IllegalStateException("FERET: контрольных чужих " + out.size() + " < " + CTRL);
        Collections.sort(out);
        return out;
    }

    /** Номер интервала (BINS) для разности дней. */
    static int bin(long days) {
        return days <= BINS[0] ? 0 : days <= BINS[1] ? 1 : 2;
    }

    /**
     * Модель «до ближайшего шаблона сессии»: внутренняя модель с шаблоном на каждую сессию (owner[j] — человек шаблона j);
     * оценка человека — наименьшая по его шаблонам.
     */
    static GalleryScorer.ScoreModel nearestSession(GalleryScorer.ScoreModel inner, int[] owner, int persons) {
        return x -> {
            double[] s = inner.scores(x);
            double[] out = new double[persons];
            Arrays.fill(out, Double.POSITIVE_INFINITY);
            for (int j = 0; j < s.length; j++) out[owner[j]] = Math.min(out[owner[j]], s[j]);
            return out;
        };
    }

    // ---------------------------------------------------------------- конфигурации

    /** Конфигурация: имя, номер, свои, галерея, пробы; sessions — галерея по датам (G2-ii) или null. */
    record Conf(String name, int s, List<String> own, Map<String, List<String>> gal, Map<String, List<String>> pro,
                Map<String, List<List<String>>> sessions) {}

    /** Методы: подпись, набор векторов (0 — none, 1 — tt, 2 — SFace), вариант. */
    record Method(String label, int set, V v) {}

    static final List<Method> METHODS = List.of(new Method("L1", 0, V.L1), new Method("L1-tt", 1, V.L1), new Method("LTP-in-tt", 1, V.LTPIN),
            new Method("A", 0, V.A), new Method("SFace", 2, V.SFACE));
    static final List<Set<V>> SETS = List.of(EnumSet.of(V.L1, V.A), EnumSet.of(V.L1, V.LTPIN), EnumSet.of(V.SFACE));

    /** Строки оценок G2-ii: как XBaseEval.fitLines, но модель — nearestSession (шаблоны по датам, проекция MLDA по людям). */
    static String fitLinesSessions(XBaseEval.Gt gt, XBaseEval x, Set<V> vs, Map<String, List<List<String>>> sessions,
                                   Map<String, List<String>> galleryKeys, List<XBaseEval.Probe> mainProbes, List<List<String>> parts, int s) {
        List<String> own = gt.own;
        List<Callable<String>> tasks = new ArrayList<>();
        for (V v : vs) {
            for (boolean m : v == V.SFACE ? new boolean[] {false} : new boolean[] {false, true}) {
                String method = v.id + (m ? FeiMethods.MIRROR : "");
                tasks.add(() -> x.lines(method, s, 0, sessionModel(x, v, m, own, sessions, gt.cohort), v, own, mainProbes));
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                    List<XBaseEval.Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new XBaseEval.Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                    tasks.add(() -> x.lines(method, s, 0, sessionModel(x, v, m, g, sessions, gt.cohort), v, g, pr));
                }
            }
        }
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, gt.threads).forEach(all::append);
        return all.toString();
    }

    /** Модель G2-ii людей gallery: MLDA — обучение по людям (+ когорта), шаблоны по датам; SFace — шаблоны по датам. */
    static GalleryScorer.ScoreModel sessionModel(XBaseEval x, V v, boolean mirror, List<String> gallery, Map<String, List<List<String>>> sessions,
                                                 List<List<String>> cohort) {
        List<List<double[]>> train = new ArrayList<>();
        List<List<double[]>> templates = new ArrayList<>();
        List<Integer> owner = new ArrayList<>();
        for (int i = 0; i < gallery.size(); i++) {
            List<double[]> person = new ArrayList<>();
            for (List<String> sess : sessions.get(gallery.get(i))) {
                List<double[]> t = new ArrayList<>();
                for (String k : sess) t.add(x.vec.get(v).get(k));
                if (mirror) for (String k : sess) t.add(x.mir.get(v).get(k));
                templates.add(t);
                owner.add(i);
                person.addAll(t);
            }
            train.add(person);
        }
        int[] own = owner.stream().mapToInt(Integer::intValue).toArray();
        if (v == V.SFACE) return nearestSession(new SFaceScorer(1).fit(templates, templates.size()), own, gallery.size());
        for (List<String> keys : cohort) {
            List<double[]> list = new ArrayList<>();
            for (String k : keys) list.add(x.vec.get(v).get(k));
            if (mirror) for (String k : keys) list.add(x.mir.get(v).get(k));
            train.add(list);
        }
        return nearestSession(new MldaScorer().fit(train, templates, templates.size(), 1e-10), own, gallery.size());
    }

    // ---------------------------------------------------------------- прогон

    static void run(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        double t = XBaseLtp.threshold(settings.loadFacesLtpT());
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval feret: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        tpl.r0 = FeiPose.r0(rawR0, "+pose");
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);
        Path out = dir.resolve("reports");
        Files.createDirectories(out);

        // Снимки, детекции, кадры.
        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        Path sfaceModel = models.resolve(FaceAlignment.SFACE_FILE);
        FaceRecognizerSF sface = FaceRecognizerSF.create(sfaceModel.toString(), "");
        long t0 = System.nanoTime();
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        double detS = (System.nanoTime() - t0) / 1e9;
        Map<String, Img> byKey = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> kept = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        int notFound = 0;
        int poseOut = 0;
        for (Img im : imgs) {
            byKey.put(im.key(), im);
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) {
                notFound++;
                continue;
            }
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) {
                poseOut++;
                continue;
            }
            kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
        }
        Map<String, TreeSet<String>> dates = dates(imgs);

        // Роли.
        List<String> own1 = new ArrayList<>();
        List<String> own1in = new ArrayList<>();
        List<String> own2 = new ArrayList<>();
        Map<String, List<String>> gal1 = new LinkedHashMap<>();
        Map<String, List<String>> pro1 = new LinkedHashMap<>();
        Map<String, List<String>> gal1in = new LinkedHashMap<>();
        Map<String, List<String>> pro1in = new LinkedHashMap<>();
        Map<String, List<String>> g1 = new LinkedHashMap<>();
        Map<String, List<String>> g2 = new LinkedHashMap<>();
        Map<String, List<List<String>>> g2s = new LinkedHashMap<>();
        Map<String, List<String>> g2p = new LinkedHashMap<>();
        Map<String, List<String>> pro2 = new LinkedHashMap<>();
        int[] dropped = new int[2];
        List<String> oneDate = new ArrayList<>();
        for (Map.Entry<String, TreeSet<String>> e : dates.entrySet()) {
            String p = e.getKey();
            List<String> ds = new ArrayList<>(e.getValue());
            Map<String, List<Img>> k = kept.getOrDefault(p, Map.of());
            if (ds.size() == 1) {
                oneDate.add(p);
                continue;
            }
            List<String> d1 = keys(k.get(ds.get(0)));
            List<String> later = new ArrayList<>();
            for (String d : ds.subList(1, ds.size())) later.addAll(keys(k.get(d)));
            if (d1.isEmpty() || later.isEmpty()) {
                dropped[0]++;
            } else {
                own1.add(p);
                gal1.put(p, d1);
                pro1.put(p, sorted(later));
                List<String> fa = new ArrayList<>();
                List<String> fb = new ArrayList<>();
                for (Img im : k.get(ds.get(0))) (im.pose().equals("fa") ? fa : fb).add(im.key());
                if (!fa.isEmpty() && !fb.isEmpty()) {
                    own1in.add(p);
                    gal1in.put(p, sorted(fa));
                    pro1in.put(p, sorted(fb));
                }
            }
            if (ds.size() >= 3) {
                List<String> d2 = keys(k.get(ds.get(1)));
                List<String> rest = new ArrayList<>();
                for (String d : ds.subList(2, ds.size())) rest.addAll(keys(k.get(d)));
                if (d1.isEmpty() || d2.isEmpty() || rest.isEmpty()) {
                    dropped[1]++;
                } else {
                    own2.add(p);
                    g1.put(p, d1);
                    List<String> both = new ArrayList<>(d1);
                    both.addAll(d2);
                    g2.put(p, sorted(both));
                    g2s.put(p, List.of(d1, d2));
                    g2p.put(p, sorted(List.of(g2pChoice(k.get(ds.get(0))), g2pChoice(k.get(ds.get(1))))));
                    pro2.put(p, sorted(rest));
                }
            }
        }
        List<String> ctrl = controls(oneDate, kept.keySet(), tpl.seed);
        Map<String, List<String>> ctrlKeys = new TreeMap<>();
        for (String p : ctrl) {
            List<String> ks = new ArrayList<>();
            kept.get(p).values().forEach(l -> ks.addAll(keys(l)));
            ctrlKeys.put(p, sorted(ks));
        }

        // Когорта FEI 20.
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }

        // Векторы: отобранные снимки FERET и когорта.
        Set<String> used = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(gal1, pro1, g2, pro2, ctrlKeys)) m.values().forEach(used::addAll);
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String k : used) {
            fr.put(k, FeretData.frame(dir, "a", k));
            sf.put(k, FeretData.frame(dir, "sface", k));
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1, V.A), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        xs[1].ltpVectors(fr, tpl.params, t);
        xs[2].sfaceMin = 1;
        xs[2].sfaceVectors(sf, sface);
        fr.values().forEach(Mat::release);
        sf.values().forEach(Mat::release);

        List<Conf> confs1 = List.of(new Conf("межс.", 0, own1, gal1, pro1, null), new Conf("внутри", 1, own1in, gal1in, pro1in, null));
        List<Conf> confs2 = List.of(new Conf("G1", 2, own2, g1, pro2, null), new Conf("G2-i", 3, own2, g2, pro2, null),
                new Conf("G2-ii", 4, own2, g2, pro2, g2s), new Conf("G2'", 5, own2, g2p, pro2, null));

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret %s; снимков %d; детекций найдено %d; seed %d; r0 %s; t %s", dir,
                imgs.size(), imgs.size() - notFound, tpl.seed, tpl.r0, t));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret"), fp);
        StringBuilder r = new StringBuilder();
        header(r, dir, exportDir, imgs, notFound, poseOut, tpl, t, cohortP, ctrl, ctrlKeys, own1, own1in, own2, dropped, gal1, pro1, gal1in, pro1in, g1, g2, g2p,
                pro2, detS, sfaceModel);
        StringBuilder lines = new StringBuilder();
        for (int ex = 0; ex < 2; ex++) {
            List<Conf> confs = ex == 0 ? confs1 : confs2;
            List<String> ownAll = ex == 0 ? own1 : own2;
            TreeSet<String> ps = new TreeSet<>(ownAll);
            ps.addAll(ctrl);
            List<String> persons = new ArrayList<>(ps);
            XBaseEval.Gt base = gt(tpl, persons, ctrl, ctrlKeys, cohort, rOf, imageNo);
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            Map<String, List<Integer>> bins = new LinkedHashMap<>();
            Map<String, FeiThreshold.Imp> imps = new HashMap<>();
            for (Conf c : confs) {
                XBaseEval.Gt g = withOwn(base, c.own());
                List<XBaseEval.Probe> probes = XBaseEval.probes(g, c.pro());
                List<List<String>> parts = XBaseOther.cvParts(c.own(), tpl.seed, XBaseEval.CV_S + c.s());
                for (int set = 0; set < SETS.size(); set++) {
                    String name = String.format(Locale.ROOT, "conf_%d_%d", c.s(), set);
                    String ls = cp.read(name);
                    if (ls == null) {
                        ls = c.sessions() == null ? XBaseEval.fitLines(g, xs[set], SETS.get(set), c.gal(), probes, parts, c.s())
                                : fitLinesSessions(g, xs[set], SETS.get(set), c.sessions(), c.gal(), probes, parts, c.s());
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "FERET: конфигурация %s, набор %d готов%n", c.name(), set);
                    }
                    lines.append("# конфигурация ").append(c.name()).append(", набор ").append(set == 0 ? "none" : set == 1 ? "tt" : "SFace").append('\n').append(ls);
                    XBaseEval.Parsed p = XBaseEval.parse(g, SETS.get(set), c.gal(), probes, parts, ls);
                    for (Method m : METHODS) {
                        if (m.set() != set) continue;
                        String label = c.name() + " " + m.label();
                        imps.put(label, p.imp.get(m.v()));
                        res.put(label, FeiThreshold.evaluate(label, m.v().ratioId(), List.of(p.cfg.get(m.v())), List.of(p.imp.get(m.v())), persons.size(),
                                tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    }
                }
                if (c.s() != 1) {
                    List<Integer> b = new ArrayList<>();
                    for (XBaseEval.Probe q : probes) {
                        if (!q.role().equals(FeiScores.OWN)) continue;
                        b.add(bin(FeretData.days(dates.get(q.person()).first(), byKey.get(q.key()).date())));
                    }
                    bins.put(c.name(), b);
                }
            }
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            int[] unit = FeiBootstrap.unit(persons.size());
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
            List<String[]> pairs = new ArrayList<>();
            if (ex == 0) {
                for (Conf c : confs) {
                    pairs.add(new String[] {c.name() + " L1-tt", c.name() + " L1"});
                    pairs.add(new String[] {c.name() + " LTP-in-tt", c.name() + " L1-tt"});
                }
            } else {
                for (Method m : METHODS) {
                    pairs.add(new String[] {"G2' " + m.label(), "G1 " + m.label()});
                    pairs.add(new String[] {"G2-i " + m.label(), "G1 " + m.label()});
                    pairs.add(new String[] {"G2-ii " + m.label(), "G2-i " + m.label()});
                }
            }
            r.append("\n\n######## ").append(ex == 0 ? "Роль (1): межсессионно и опора внутри сессии" : "Роль (2): регистрация по нескольким дням")
                    .append(String.format(Locale.ROOT, " — бутстреп по %d людям (свои и контроль) ########%n", persons.size()));
            XBaseOther.tables(r, res, sep, tpl.b, pairs);
            binTable(r, res, imps, bins, persons.size(), tpl);
        }
        eyeTable(r, imgs, det, own1, own2, ctrl);
        XBaseOther.writeScores(out.resolve("xbase_feret_scores.tsv.gz"), "# FERET (XBaseFeret, PLAN.md п. 5.5), только локально; # конфигурация — блоки\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    static List<String> keys(List<Img> l) {
        return l == null ? new ArrayList<>() : sorted(l.stream().map(Img::key).toList());
    }

    static List<String> sorted(List<String> l) {
        List<String> s = new ArrayList<>(l);
        Collections.sort(s);
        return s;
    }

    /** Gt эксперимента: люди бутстрепа, контроль, когорта, позы, номера снимков. */
    static XBaseEval.Gt gt(XBaseEval.Gt t, List<String> persons, List<String> ctrl, Map<String, List<String>> ctrlKeys, List<List<String>> cohort,
                           Map<String, Double> rOf, Map<String, Integer> imageNo) {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.seed = t.seed;
        g.r0 = t.r0;
        g.b = t.b;
        g.threads = t.threads;
        g.rLimit = t.rLimit;
        g.dir = t.dir;
        g.params = t.params;
        g.rOf = rOf;
        g.imageNo = imageNo;
        g.persons = persons;
        for (int i = 0; i < persons.size(); i++) g.pIdx.put(persons.get(i), i);
        g.ctrl = ctrl;
        for (String p : ctrl) g.byPerson.put(p, new TreeMap<>(Map.of(0, ctrlKeys.get(p))));
        g.cohort = cohort;
        return g;
    }

    static XBaseEval.Gt withOwn(XBaseEval.Gt b, List<String> own) {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.seed = b.seed;
        g.r0 = b.r0;
        g.b = b.b;
        g.threads = b.threads;
        g.rLimit = b.rLimit;
        g.dir = b.dir;
        g.params = b.params;
        g.rOf = b.rOf;
        g.imageNo = b.imageNo;
        g.persons = b.persons;
        g.pIdx = b.pIdx;
        g.ctrl = b.ctrl;
        g.byPerson = b.byPerson;
        g.cohort = b.cohort;
        g.own = own;
        return g;
    }

    // ---------------------------------------------------------------- отчёт

    static int sum(Map<String, List<String>> m) {
        return m.values().stream().mapToInt(List::size).sum();
    }

    static void header(StringBuilder r, Path dir, Path exportDir, List<Img> imgs, int notFound, int poseOut, XBaseEval.Gt t, double ltpT, List<String> cohortP,
                       List<String> ctrl, Map<String, List<String>> ctrlKeys, List<String> own1, List<String> own1in, List<String> own2, int[] dropped,
                       Map<String, List<String>> gal1, Map<String, List<String>> pro1, Map<String, List<String>> gal1in, Map<String, List<String>> pro1in,
                       Map<String, List<String>> g1, Map<String, List<String>> g2, Map<String, List<String>> g2p, Map<String, List<String>> pro2, double detS,
                       Path sfaceModel) throws IOException {
        r.append("PLAN.md п. 5.5, протокол FERET (XBaseEval feret): Color FERET — только локально (Release Agreement v1)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (dvd1, dvd2; fa и fb; распаковка в памяти); когорта FEI — экспорт ").append(exportDir).append(", коммит данных ")
                .append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(String.format(Locale.ROOT, "Снимков fa/fb: %d; YuNet (порог %.1f, два прохода) не нашёл лица: %d; отбор по позе |r| ≤ R = %.4f исключил: %d; "
                + "детекция и кадры — %.0f с (из кэша derived/ — быстрее).%n", imgs.size(), FeretData.SCORE, notFound, t.rLimit, poseOut, detS));
        r.append(String.format(Locale.ROOT, "Роль (1): своих %d (выбыли после отбора %d), галерея %d снимков, пробы %d; опора внутри сессии — своих %d, галерея fa %d, "
                + "пробы fb %d.%n", own1.size(), dropped[0], sum(gal1), sum(pro1), own1in.size(), sum(gal1in), sum(pro1in)));
        r.append(String.format(Locale.ROOT, "Роль (2): своих %d (выбыли %d); G1 %d снимков, G2 %d, G2' %d; пробы %d.%n", own2.size(), dropped[1], sum(g1), sum(g2), sum(g2p),
                sum(pro2)));
        r.append(String.format(Locale.ROOT, "Роль (3): контрольных чужих %d (люди с одной датой, Random(seed + %d)), снимков %d.%n", ctrl.size(), CTRL_SEED_OFFSET, sum(ctrlKeys)));
        r.append("Когорта MLDA — FEI 20 (seed faces.seed + 9000): ").append(String.join(", ", cohortP)).append(".\n");
        r.append(String.format(Locale.ROOT, "Методы: L1 = LBP u2 (8, 2) 7×7 + MLDA (none); L1-tt — то же на выходе Tan – Triggs; LTP-in-tt — LTP t = %s по выходу tt, "
                + "внутренние блоки;%nA = CLAHE + MLDA (справочно); SFace — alignCrop 112×112, %s (верхняя граница). Зеркала и выбор модели по позе r0 = %s (кроме SFace); "
                + "оценка d₁/d₂.%n", ltpT, sfaceModel.getFileName(), t.r0));
        r.append("G2-i — шаблон человека — среднее снимков дат 1 и 2; G2-ii — шаблоны по датам (проекция MLDA обучена по людям), оценка — наименьшая по датам\n"
                + "человека, второй кандидат — ближайший другой человек; G2' — по одному снимку дат 1 и 2 (fa без флага, иначе fa, иначе fb).\n");
        r.append("Порог: П2 / П3 по «чужим» галереи своей конфигурации (5 частей своих, seed + 8000 + 100 + s; s: межс. 0, внутри 1, G1 2, G2-i 3, G2-ii 4, G2' 5).\n");
        if (t.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям роли и контроля: B = %d, seed = %d; одни веса для всех строк роли.%n", t.b,
                t.seed + FeiBootstrap.SEED_OFFSET));
    }

    /** Ошибка своих по интервалам дат (порог своей конфигурации; те же «чужие» и контроль). */
    static void binTable(StringBuilder r, Map<String, FeiThreshold.Res> res, Map<String, FeiThreshold.Imp> imps, Map<String, List<Integer>> bins, int persons,
                         XBaseEval.Gt t) {
        r.append("\n=== Ошибка своих по интервалу между датой пробы и первой датой человека (порог своей конфигурации) ===\n");
        r.append("строка | интервал | проб | П2 FAR 1 % | П3 FAR 1 % | П2 FAR 5 % | П3 FAR 5 %\n");
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) {
            String conf = e.getKey().substring(0, e.getKey().indexOf(' '));
            List<Integer> b = bins.get(conf);
            if (b == null) continue;
            FeiDecompose.Cfg c = e.getValue().cfgs.get(0);
            for (int k = 0; k < BIN_NAMES.length; k++) {
                List<Integer> idx = new ArrayList<>();
                for (int i = 0; i < b.size(); i++) if (b.get(i) == k) idx.add(i);
                if (idx.isEmpty()) continue;
                FeiDecompose.Cfg s = subset(c, idx);
                FeiThreshold.Res sr = FeiThreshold.evaluate(e.getKey(), e.getValue().id, List.of(s), List.of(imps.get(e.getKey())), persons, t.b,
                        t.seed + FeiBootstrap.SEED_OFFSET);
                r.append(String.join(" | ", e.getKey(), BIN_NAMES[k], Integer.toString(idx.size()), FeiThreshold.errCell(sr, 2, 0, 0, t.b),
                        FeiThreshold.errCell(sr, 3, 0, 0, t.b), FeiThreshold.errCell(sr, 2, 1, 0, t.b), FeiThreshold.errCell(sr, 3, 1, 0, t.b))).append('\n');
            }
        }
    }

    /** Конфигурация с частью проб своих (индексы idx), контроль и прочее — те же. */
    static FeiDecompose.Cfg subset(FeiDecompose.Cfg c, List<Integer> idx) {
        FeiDecompose.Cfg s = new FeiDecompose.Cfg();
        int n = idx.size();
        s.ownP = new int[n];
        s.ownR = new double[n];
        s.ownOk = new boolean[n];
        s.noSecond = new boolean[n];
        s.trueInf = new boolean[n];
        s.poseFrontal = new boolean[n];
        for (int i = 0; i < n; i++) {
            int j = idx.get(i);
            s.ownP[i] = c.ownP[j];
            s.ownR[i] = c.ownR[j];
            s.ownOk[i] = c.ownOk[j];
            s.noSecond[i] = c.noSecond[j];
            s.trueInf[i] = c.trueInf[j];
            s.poseFrontal[i] = c.poseFrontal[j];
        }
        s.thrP = c.thrP;
        s.thrR = c.thrR;
        s.ctrlP = c.ctrlP;
        s.ctrlR = c.ctrlR;
        return s;
    }

    /** Диагностика выравнивания: расхождение точек глаз YuNet с ручными (доли ручного межглазья), по ролям. */
    static void eyeTable(StringBuilder r, List<Img> imgs, Map<String, double[]> det, List<String> own1, List<String> own2, List<String> ctrl) {
        r.append("\n=== Диагностика выравнивания: точки глаз YuNet против ручных координат разметки (доли ручного межглазья) ===\n");
        r.append("Ручные координаты — только для диагностики; все снимки fa/fb с разметкой и найденным лицом (до отбора по позе).\n");
        r.append("группа | снимков | среднее по глазам: медиана | 95 % | макс | доля > 5 % | доля > 10 % | межглазье, пикс.: медиана\n");
        Set<String> s1 = new HashSet<>(own1);
        Set<String> s2 = new HashSet<>(own2);
        Set<String> s3 = new HashSet<>(ctrl);
        Map<String, List<double[]>> groups = new LinkedHashMap<>();
        for (String g : List.of("все", "роль (1)", "роль (2)", "контроль")) groups.put(g, new ArrayList<>());
        int noTruth = 0;
        for (Img im : imgs) {
            double[] e = FeretData.eyeError(im, det.get(im.key()));
            if (e == null) {
                if (im.leftEye() == null) noTruth++;
                continue;
            }
            groups.get("все").add(e);
            if (s1.contains(im.person())) groups.get("роль (1)").add(e);
            if (s2.contains(im.person())) groups.get("роль (2)").add(e);
            if (s3.contains(im.person())) groups.get("контроль").add(e);
        }
        for (Map.Entry<String, List<double[]>> g : groups.entrySet()) {
            List<double[]> l = g.getValue();
            if (l.isEmpty()) continue;
            double[] m = FeiBootstrap.sorted(l.stream().mapToDouble(a -> a[2]).toArray());
            double[] iod = FeiBootstrap.sorted(l.stream().mapToDouble(a -> a[3]).toArray());
            long g5 = Arrays.stream(m).filter(v -> v > 0.05).count();
            long g10 = Arrays.stream(m).filter(v -> v > 0.10).count();
            r.append(String.format(Locale.ROOT, "%s | %d | %.4f | %.4f | %.4f | %.1f %% | %.1f %% | %.1f%n", g.getKey(), l.size(), FeiBootstrap.quantile(m, 0.5),
                    FeiBootstrap.quantile(m, 0.95), m[m.length - 1], 100.0 * g5 / m.length, 100.0 * g10 / m.length, FeiBootstrap.quantile(iod, 0.5)));
        }
        r.append("Снимков fa/fb без ручных координат глаз: ").append(noTruth).append(".\n");
    }
}
