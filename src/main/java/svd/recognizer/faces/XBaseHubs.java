package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseExcl.Data;
import svd.recognizer.faces.XBaseExcl.G;
import svd.recognizer.faces.XBaseExcl.Model;
import svd.recognizer.storage.SettingsStore;

/**
 * Симметричные галереи и защита от хабов на FERET (PLAN.md п. 5.5, (а2в); план подтверждён Хозяином 12.10.2026).
 * <ul>
 *   <li>Часть 1 — симметричные галереи: у всех своих ровно s сессий (первые s дат, все fa/fb), пробы — даты после s-й;
 *       шаблон (м) — средний, при s ≥ 2 ещё (с) — по сессиям, до ближайшего (как G2-ii). (i) по каждому s — люди с ≥ s + 1
 *       датами; (ii) парно — одни и те же люди с ≥ 4 датами, пробы — даты с 4-й, s = 1, 2, 3.</li>
 *   <li>Часть 2 — несимметричная галерея Г-все (схема (а2)): (м) средний шаблон — строки из файла оценок (а2) с проверкой
 *       сводки; (с) — шаблоны по сессиям, до ближайшего; (с3) — у человека в галерее и обучении MLDA только 3 последние по
 *       времени сессии галереи (ограничение хранения), шаблоны по сессиям.</li>
 * </ul>
 * Методы L1, L1-tt, LTP-in-tt, SFace (минимум снимков шаблона 1). Мера хабовости — сколько ошибочных проб своих (неверный
 * лучший кандидат) перехватил каждый человек. Режим XBaseEval feret-hubs; отчёт — только в каталоге базы FERET.
 *
 * @author ssv
 */
final class XBaseHubs {

    /** Наибольшее число сессий человека в (с3), задано заранее. */
    static final int M = 3;
    /** Номера конфигураций (seed частей П2/П3 — faces.seed + 8000 + 100 + s): (i) s = 1 — 0 (как роль (1) feret), иначе — ниже. */
    static final int[] CONF_I = {0, 10, 11};
    static final int[] CONF_II = {12, 13, 14};
    /** Номера s строк части 2: (с) полная модель и раунды, (с3) полная модель и раунды. */
    static final int S_C_FULL = 21;
    static final int S_C = 22;
    static final int S_C3_FULL = 23;
    static final int S_C3 = 24;

    static final List<XBaseFeret.Method> METHODS = List.of(new XBaseFeret.Method("L1", 0, V.L1), new XBaseFeret.Method("L1-tt", 1, V.L1),
            new XBaseFeret.Method("LTP-in-tt", 1, V.LTPIN), new XBaseFeret.Method("SFace", 2, V.SFACE));
    static final List<Set<V>> SETS = List.of(EnumSet.of(V.L1), EnumSet.of(V.L1, V.LTPIN), EnumSet.of(V.SFACE));
    static final String[] SET_NAMES = {"none", "tt", "SFace"};

    private XBaseHubs() {
    }

    // ---------------------------------------------------------------- выбор снимков

    /** Последние по времени не больше cap номеров дат (по возрастанию); cap ≤ 0 — все. */
    static List<Integer> last(List<Integer> dates, int cap) {
        return cap <= 0 || dates.size() <= cap ? dates : new ArrayList<>(dates.subList(dates.size() - cap, dates.size()));
    }

    /** Сессии человека: номера дат js → списки ключей. */
    static List<List<String>> sessions(Data d, String p, List<Integer> js) {
        List<List<String>> out = new ArrayList<>();
        for (int j : js) out.add(d.gsess().get(p).get(j));
        return out;
    }

    static List<Integer> all(int n) {
        List<Integer> out = new ArrayList<>();
        for (int j = 0; j < n; j++) out.add(j);
        return out;
    }

    /** Галерея по сессиям человека → ключи. */
    static Map<String, List<String>> keys(Map<String, List<List<String>>> sess) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        sess.forEach((p, ss) -> {
            List<String> ks = new ArrayList<>();
            ss.forEach(ks::addAll);
            out.put(p, XBaseExcl.sorted(ks));
        });
        return out;
    }

    /** Часть 1: сессии галереи — первые s дат людей own. */
    static Map<String, List<List<String>>> firstSessions(Data d, List<String> own, int s) {
        Map<String, List<List<String>>> out = new LinkedHashMap<>();
        for (String p : own) out.put(p, sessions(d, p, all(s)));
        return out;
    }

    /** Часть 1: пробы — снимки дат с номера from (с 0). */
    static Map<String, List<String>> probesFrom(Data d, List<String> own, int from) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (String p : own) {
            List<String> ks = new ArrayList<>();
            List<List<String>> ds = d.sess().get(p);
            for (int j = from; j < ds.size(); j++) ks.addAll(ds.get(j));
            out.put(p, XBaseExcl.sorted(ks));
        }
        return out;
    }

    /** Часть 2: сессии всех своих в полной модели (все даты, при cap — последние cap). */
    static Map<String, List<List<String>>> fullSessions(Data d, int cap) {
        Map<String, List<List<String>>> out = new LinkedHashMap<>();
        for (String p : d.own()) out.put(p, sessions(d, p, last(all(d.sess().get(p).size()), cap)));
        return out;
    }

    /** Часть 2: сессии своих в модели схемы (а2) Г-все: у проб модели нет k-й даты; при cap — последние cap дат. */
    static Map<String, List<List<String>>> modelSessions(Data d, Model md, int cap) {
        Map<String, List<List<String>>> out = new LinkedHashMap<>();
        for (String p : d.own()) {
            int n = d.sess().get(p).size();
            List<Integer> js = md.pro().containsKey(p) ? XBaseExcl.galleryDates(G.ALL, md.k() - 1, n) : all(n);
            out.put(p, sessions(d, p, last(js, cap)));
        }
        return out;
    }

    /** Разбивка пробы с даты i при датах галереи gd: {сессий, снимков, интервал} — как XBaseExcl.bins. */
    static int[] bins(Data d, String p, int i, List<Integer> gd) {
        List<List<String>> ds = d.gsess().get(p);
        int imgs = 0;
        long near = Long.MAX_VALUE;
        for (int j : gd) {
            imgs += ds.get(j).size();
            near = Math.min(near, Math.abs(ChronoUnit.DAYS.between(d.when().get(p).get(j), d.when().get(p).get(i))));
        }
        int s = gd.size();
        return new int[] {Math.min(s, 4) - 1, imgs <= 2 ? 0 : imgs <= 4 ? 1 : imgs <= 6 ? 2 : 3, XBaseFeret.bin(near)};
    }

    // ---------------------------------------------------------------- хабовость

    /** Хабовость: ошибочных проб с неверным лучшим, людей-перехватчиков, доля первого, доля первых 5, первые 5 (человек → проб). */
    record Hubs(int wrong, int hubs, double top1, double top5, List<Map.Entry<String, Integer>> top) {}

    static Hubs hubs(List<String> bestOfWrong) {
        Map<String, Integer> c = new TreeMap<>();
        for (String b : bestOfWrong) c.merge(b, 1, Integer::sum);
        List<Map.Entry<String, Integer>> s = new ArrayList<>(c.entrySet());
        s.sort((a, b) -> b.getValue().equals(a.getValue()) ? a.getKey().compareTo(b.getKey()) : b.getValue() - a.getValue());
        int n = bestOfWrong.size();
        int t5 = 0;
        for (int i = 0; i < Math.min(5, s.size()); i++) t5 += s.get(i).getValue();
        return new Hubs(n, s.size(), n == 0 ? 0 : (double) s.get(0).getValue() / n, n == 0 ? 0 : (double) t5 / n, s.subList(0, Math.min(5, s.size())));
    }

    /** Лучшие кандидаты ошибочных проб своих (модель по позе пробы) варианта v по строкам оценок. */
    static List<String> wrongBest(XBaseEval.Gt g, V v, List<Probe> probes, String lines) {
        Map<String, Row> rows = new HashMap<>();
        for (String l : lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        List<String> out = new ArrayList<>();
        for (Probe p : probes) {
            if (!p.role().equals(FeiScores.OWN)) continue;
            Row r = XBaseEval.choose(rows, v, p.role(), p.person(), p.image(), g.rOf.get(p.key()) <= g.r0);
            if (!r.bestId().equals(r.trueId())) out.add(r.bestId());
        }
        return out;
    }

    // ---------------------------------------------------------------- обучение

    /** Строки оценок проб своих моделей схемы (а2) с шаблонами по сессиям (s — номер строк, j — раунд). */
    static String fitModelsSessions(XBaseEval.Gt g, XBaseEval x, Set<V> vs, Data d, List<Model> models, int cap, int s) {
        List<String> own = g.own;
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < own.size(); i++) idx.put(own.get(i), i);
        List<Callable<String>> tasks = new ArrayList<>();
        for (V v : vs) {
            for (boolean m : v == V.SFACE ? new boolean[] {false} : new boolean[] {false, true}) {
                String method = v.id + (m ? FeiMethods.MIRROR : "");
                for (Model md : models) {
                    Map<String, List<List<String>>> sess = modelSessions(d, md, cap);
                    List<Probe> pr = new ArrayList<>();
                    md.pro().forEach((p, ks) -> {
                        for (String k : ks) pr.add(new Probe(FeiScores.OWN, p, g.img(k), k, idx.get(p)));
                    });
                    tasks.add(() -> x.lines(method, s, md.k(), XBaseFeret.sessionModel(x, v, m, own, sess, g.cohort), v, own, pr));
                }
            }
        }
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, g.threads).forEach(all::append);
        return all.toString();
    }

    /** Блоки файла оценок (а2): «блок, набор» → строки. */
    static Map<String, String> blocks(Path scores) throws IOException {
        Map<String, StringBuilder> b = new LinkedHashMap<>();
        String cur = null;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(scores), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("# блок ")) {
                    cur = l.substring("# блок ".length());
                    b.put(cur, new StringBuilder());
                } else if (!l.startsWith("#") && !l.startsWith("method\t") && cur != null) {
                    b.get(cur).append(l).append('\n');
                }
            }
        }
        Map<String, String> out = new LinkedHashMap<>();
        b.forEach((k, v) -> out.put(k, v.toString()));
        return out;
    }

    // ---------------------------------------------------------------- группа строк

    /** Группа строк одного бутстрепа: итоги, «чужие», лучшие кандидаты ошибочных проб, разбивки. */
    static final class Group {
        final Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        final Map<String, FeiThreshold.Imp> imps = new HashMap<>();
        final Map<String, Hubs> hubs = new LinkedHashMap<>();
        final Map<String, List<int[]>> bins = new LinkedHashMap<>();
        final List<String[]> pairs = new ArrayList<>();
        final StringBuilder lines = new StringBuilder();
    }

    static void add(Group gr, XBaseEval.Gt g, String prefix, int set, XBaseEval.Parsed p, List<Probe> probes, String lines, List<int[]> bins) {
        for (XBaseFeret.Method m : METHODS) {
            if (m.set() != set) continue;
            String label = prefix + " " + m.label();
            gr.imps.put(label, p.imp.get(m.v()));
            gr.res.put(label, FeiThreshold.evaluate(label, m.v().ratioId(), List.of(p.cfg.get(m.v())), List.of(p.imp.get(m.v())), g.persons.size(), g.b,
                    g.seed + FeiBootstrap.SEED_OFFSET));
            gr.hubs.put(label, hubs(wrongBest(g, m.v(), probes, lines)));
            if (bins != null) gr.bins.put(label, bins);
        }
    }

    static void tables(StringBuilder r, Group gr, XBaseEval.Gt g, Data d) {
        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(g.persons.size());
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : gr.res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
        XBaseOther.tables(r, gr.res, sep, g.b, gr.pairs);
        r.append("\n=== Хабовость: ошибочные пробы своих с неверным лучшим кандидатом (модель по позе пробы; ошибка при любом пороге) ===\n");
        r.append("строка | таких проб | людей-перехватчиков | доля первого | доля первых 5 | первые 5: человек (дат) — проб\n");
        for (Map.Entry<String, Hubs> e : gr.hubs.entrySet()) {
            Hubs h = e.getValue();
            List<String> top = new ArrayList<>();
            for (Map.Entry<String, Integer> t : h.top()) {
                List<List<String>> ds = d.sess().get(t.getKey());
                top.add(t.getKey() + " (" + (ds == null ? "?" : ds.size()) + ") — " + t.getValue());
            }
            r.append(String.format(Locale.ROOT, "%s | %d | %d | %.1f %% | %.1f %% | %s%n", e.getKey(), h.wrong(), h.hubs(), 100 * h.top1(), 100 * h.top5(),
                    String.join(", ", top)));
        }
        if (gr.bins.isEmpty()) return;
        String[][] names = {XBaseExcl.SESS_NAMES, XBaseExcl.IMG_NAMES, XBaseFeret.BIN_NAMES};
        String[] titles = {"по числу сессий (дат) в галерее человека для пробы (в (с3) — после ограничения)",
                "по числу снимков в галерее человека для пробы", "по интервалу от ближайшей даты галереи до даты пробы"};
        for (int t = 0; t < names.length; t++) {
            r.append("\n=== Ошибка своих ").append(titles[t]).append(" (порог своей строки; «мало» — меньше ").append(XBaseExcl.FEW).append(" людей) ===\n");
            r.append("строка | группа | проб | людей | П2 FAR 1 % | П3 FAR 1 % | П2 FAR 5 % | П3 FAR 5 %\n");
            for (Map.Entry<String, FeiThreshold.Res> e : gr.res.entrySet()) {
                List<int[]> b = gr.bins.get(e.getKey());
                if (b == null) continue;
                FeiDecompose.Cfg c = e.getValue().cfgs.get(0);
                for (int k = 0; k < names[t].length; k++) {
                    List<Integer> idx = new ArrayList<>();
                    for (int i = 0; i < b.size(); i++) if (b.get(i)[t] == k) idx.add(i);
                    if (idx.isEmpty()) continue;
                    long people = idx.stream().map(i -> c.ownP[i]).distinct().count();
                    FeiThreshold.Res sr = FeiThreshold.evaluate(e.getKey(), e.getValue().id, List.of(XBaseFeret.subset(c, idx)), List.of(gr.imps.get(e.getKey())),
                            g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET);
                    r.append(String.join(" | ", e.getKey(), names[t][k], Integer.toString(idx.size()), people + (people < XBaseExcl.FEW ? " (мало)" : ""),
                            FeiThreshold.errCell(sr, 2, 0, 0, g.b), FeiThreshold.errCell(sr, 3, 0, 0, g.b), FeiThreshold.errCell(sr, 2, 1, 0, g.b),
                            FeiThreshold.errCell(sr, 3, 1, 0, g.b))).append('\n');
                }
            }
        }
    }

    /** Блок оценок из контрольной точки или расчётом. */
    interface Fit {
        String fit() throws IOException;
    }

    static String block(XBaseOther.Checkpoint cp, String name, Fit f, double[] fitS) throws IOException {
        String l = cp.read(name);
        if (l != null) {
            System.out.println("feret-hubs: блок " + name + " — из контрольной точки");
            return l;
        }
        long t1 = System.nanoTime();
        l = f.fit();
        double s = (System.nanoTime() - t1) / 1e9;
        fitS[0] += s;
        cp.save(name, l);
        System.out.printf(Locale.ROOT, "feret-hubs: блок %s готов (%.0f с)%n", name, s);
        return l;
    }

    // ---------------------------------------------------------------- прогон

    static void run(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-hubs: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        double t = XBaseLtp.threshold(settings.loadFacesLtpT());
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval feret-hubs: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        tpl.r0 = FeiPose.r0(rawR0, "+pose");
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);
        boolean eq = settings.loadFacesFeretGalleryEqual(); // №9: равные галереи — по умолчанию (отчёт и файл оценок (а2) — тоже из eq\), old — прежние
        Path out = eq ? dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : dir.resolve("reports");
        Path exclScores = out.resolve("xbase_feret_excl_scores.tsv.gz");
        Path exclReport = out.resolve("xbase_feret_excl.txt");
        if (!Files.exists(exclScores) || !Files.exists(exclReport)) throw new IllegalStateException("feret-hubs: нужны отчёт и файл оценок (а2) в " + out);

        // Снимки, детекции, кадры — как XBaseExcl.feret.
        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        Path sfaceModel = models.resolve(FaceAlignment.SFACE_FILE);
        FaceRecognizerSF sface = FaceRecognizerSF.create(sfaceModel.toString(), "");
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> kept = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        for (Img im : imgs) {
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) continue;
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) continue;
            kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), x -> new ArrayList<>()).add(im);
        }
        XBaseFeret.Roles ro = XBaseFeret.roles(XBaseFeret.dates(imgs), kept, tpl.seed);
        Map<String, List<List<String>>> sess = new LinkedHashMap<>();
        Map<String, List<LocalDate>> when = new LinkedHashMap<>();
        for (String p : ro.own1) {
            List<List<String>> ds = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (Map.Entry<String, List<Img>> e : kept.get(p).entrySet()) {
                ds.add(XBaseFeret.keys(e.getValue()));
                ws.add(FeretData.date(e.getKey()));
            }
            sess.put(p, ds);
            when.put(p, ws);
        }
        Map<String, Img> byKey = new HashMap<>();
        for (Img im : imgs) byKey.put(im.key(), im);
        Data d = new Data(ro.own1, sess, when, eq ? XBaseExcl.pairSessions(sess, byKey) : sess);

        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        sess.values().forEach(ds -> ds.forEach(used::addAll));
        ro.ctrlKeys.values().forEach(used::addAll);
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
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        xs[1].ltpVectors(fr, tpl.params, t);
        xs[2].sfaceMin = 1;
        xs[2].sfaceVectors(sf, sface);
        fr.values().forEach(Mat::release);
        sf.values().forEach(Mat::release);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-hubs %s; снимков %d; seed %d; r0 %s; t %s; M %d; (а2) %s", dir,
                imgs.size(), tpl.seed, tpl.r0, t, M, FeiThreshold.sha(exclScores)) + (eq ? "; №8 галерея fa/fb" : ""));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve(eq ? "checkpoint_feret_hubs_eq" : "checkpoint_feret_hubs"), fp);
        double[] fitS = {0};
        StringBuilder r = new StringBuilder();
        StringBuilder allLines = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а2в) симметричные галереи и защита от хабов на FERET (XBaseEval feret-hubs): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры — как feret); когорта FEI — экспорт ").append(exportDir).append(", коммит данных ")
                .append(FarMethods.dataCommit(exportDir)).append("; строки (м) части 2 — файл оценок (а2) ").append(exclScores.getFileName()).append(" (")
                .append(FeiThreshold.sha(exclScores)).append(")\n");
        r.append(String.format(Locale.ROOT, "Свои с ≥ 2 датами: %d; контрольные чужие — те же 200 (%d снимков); когорта MLDA — FEI 20. Методы: L1, L1-tt, LTP-in-tt, "
                + "SFace (минимум снимков шаблона 1); зеркала и выбор модели по позе r0 = %s (кроме SFace); оценка d₁/d₂.%n", d.own().size(),
                XBaseFeret.sum(ro.ctrlKeys), tpl.r0));
        r.append("Шаблон (м) — среднее снимков галереи человека; (с) — шаблон на каждую сессию (дату) галереи, проекция MLDA обучена по людям, оценка\n"
                + "человека — наименьшая по его сессиям, второй кандидат — ближайший другой человек (как G2-ii); (с3) — у человека в галерее и обучении\n"
                + "только " + M + " последние по времени сессии его галереи (M задан заранее), шаблоны по сессиям.\n");
        if (eq) r.append(XBaseExcl.EQ_NOTE);
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям группы (свои и контроль): B = %d, seed = %d; «*» — интервал разности не содержит 0.%n",
                tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        // ---- Часть 2, (м): строки (а2) и проверка сводки.
        TreeSet<String> ps2 = new TreeSet<>(d.own());
        ps2.addAll(ro.ctrl);
        XBaseEval.Gt g2 = XBaseFeret.withOwn(XBaseFeret.gt(tpl, new ArrayList<>(ps2), ro.ctrl, ro.ctrlKeys, cohort, rOf, imageNo), d.own());
        List<List<String>> parts0 = XBaseOther.cvParts(d.own(), tpl.seed, XBaseEval.CV_S);
        List<Probe> main2 = XBaseEval.probes(g2, XBaseExcl.probeKeys(d, G.ALL));
        List<Probe> ctrl2 = main2.stream().filter(p -> p.role().equals(FeiScores.CTRL)).toList();
        Map<String, Integer> dateOf = new HashMap<>();
        for (String p : d.own()) for (int j = 0; j < d.sess().get(p).size(); j++) for (String k : d.sess().get(p).get(j)) dateOf.put(k, j);
        Map<String, String> excl = blocks(exclScores);
        Group gr2 = new Group();
        Map<String, List<int[]>> bins2 = new LinkedHashMap<>();
        for (int cap : new int[] {-1, 0, M}) {
            List<int[]> b = new ArrayList<>();
            for (Probe q : main2) {
                if (!q.role().equals(FeiScores.OWN)) continue;
                int i = dateOf.get(q.key());
                b.add(bins(d, q.person(), i, last(XBaseExcl.galleryDates(G.ALL, i, d.sess().get(q.person()).size()), cap)));
            }
            bins2.put(cap < 0 ? "(м)" : cap == 0 ? "(с)" : "(с3)", b);
        }
        Map<String, List<String>> fullGal = XBaseExcl.gallery(d, null);
        for (int set = 0; set < SETS.size(); set++) {
            String ls = excl.get("full, набор " + SET_NAMES[set]) + excl.get("all, набор " + SET_NAMES[set]);
            XBaseEval.Parsed p = XBaseEval.parse(g2, SETS.get(set), fullGal, main2, parts0, ls);
            add(gr2, g2, "Г-все (м)", set, p, main2, ls, bins2.get("(м)"));
        }
        StringBuilder check = new StringBuilder();
        Group chk = new Group();
        for (Map.Entry<String, FeiThreshold.Res> e : gr2.res.entrySet()) chk.res.put(e.getKey().replace(" (м)", ""), e.getValue());
        int[][] w2 = tpl.b > 0 ? FeiBootstrap.weights(g2.persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sepChk = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : chk.res.entrySet()) sepChk.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(g2.persons.size()), w2));
        XBaseOther.tables(check, chk.res, sepChk, tpl.b, List.of());
        List<String> exclLines = Files.readAllLines(exclReport, StandardCharsets.UTF_8);
        int matched = 0;
        for (String l : check.toString().lines().toList()) {
            if (!l.startsWith("Г-все ")) continue;
            if (!exclLines.contains(l)) throw new IllegalStateException("feret-hubs: строка (м) не совпала с xbase_feret_excl.txt — пересчитать (м):\n" + l);
            matched++;
        }
        System.out.println("feret-hubs: проверка (м) — совпало строк " + matched);

        // ---- Часть 1.
        StringBuilder part1 = new StringBuilder();
        for (int view = 0; view < 2; view++) {
            for (int s = 1; s <= (view == 0 ? 3 : 1); s++) {
                int sMin = view == 0 ? s + 1 : 4;
                List<String> own = d.own().stream().filter(p -> d.sess().get(p).size() >= sMin).toList();
                TreeSet<String> ps = new TreeSet<>(own);
                ps.addAll(ro.ctrl);
                XBaseEval.Gt g = XBaseFeret.withOwn(XBaseFeret.gt(tpl, new ArrayList<>(ps), ro.ctrl, ro.ctrlKeys, cohort, rOf, imageNo), own);
                Group gr = new Group();
                List<Integer> svals = view == 0 ? List.of(s) : List.of(1, 2, 3);
                for (int sv : svals) {
                    int cs = view == 0 ? CONF_I[sv - 1] : CONF_II[sv - 1];
                    Map<String, List<List<String>>> fs = firstSessions(d, own, sv);
                    Map<String, List<String>> gal = keys(fs);
                    List<Probe> probes = XBaseEval.probes(g, probesFrom(d, own, view == 0 ? sv : 3));
                    List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + cs);
                    String tag = view == 0 ? "(i) s" + sv : "(ii) s" + sv;
                    for (int set = 0; set < SETS.size(); set++) {
                        int st = set;
                        String name = String.format(Locale.ROOT, "p1_%d_%d", cs, set);
                        String lm = block(cp, name + "_m", () -> XBaseEval.fitLines(g, xs[st], SETS.get(st), gal, probes, parts, cs), fitS);
                        allLines.append("# часть 1 ").append(tag).append(" (м), набор ").append(SET_NAMES[set]).append('\n').append(lm);
                        add(gr, g, tag + (sv == 1 ? "" : " (м)"), set, XBaseEval.parse(g, SETS.get(set), gal, probes, parts, lm), probes, lm, null);
                        if (sv >= 2) {
                            String lc = block(cp, name + "_c", () -> XBaseFeret.fitLinesSessions(g, xs[st], SETS.get(st), fs, gal, probes, parts, cs), fitS);
                            allLines.append("# часть 1 ").append(tag).append(" (с), набор ").append(SET_NAMES[set]).append('\n').append(lc);
                            add(gr, g, tag + " (с)", set, XBaseEval.parse(g, SETS.get(set), gal, probes, parts, lc), probes, lc, null);
                        }
                    }
                }
                // Порядок строк: по s, затем (м), (с), по методам.
                Map<String, FeiThreshold.Res> ordered = new LinkedHashMap<>();
                for (int sv : svals) {
                    String tag = view == 0 ? "(i) s" + sv : "(ii) s" + sv;
                    for (String mode : sv == 1 ? List.of("") : List.of(" (м)", " (с)")) {
                        for (XBaseFeret.Method m : METHODS) ordered.put(tag + mode + " " + m.label(), gr.res.get(tag + mode + " " + m.label()));
                    }
                }
                gr.res.clear();
                gr.res.putAll(ordered);
                Map<String, Hubs> oh = new LinkedHashMap<>();
                for (String k : ordered.keySet()) oh.put(k, gr.hubs.get(k));
                gr.hubs.clear();
                gr.hubs.putAll(oh);
                for (XBaseFeret.Method m : METHODS) {
                    String l = " " + m.label();
                    if (view == 0) {
                        if (s >= 2) gr.pairs.add(new String[] {"(i) s" + s + " (с)" + l, "(i) s" + s + " (м)" + l});
                    } else {
                        gr.pairs.add(new String[] {"(ii) s2 (м)" + l, "(ii) s1" + l});
                        gr.pairs.add(new String[] {"(ii) s2 (с)" + l, "(ii) s1" + l});
                        gr.pairs.add(new String[] {"(ii) s3 (м)" + l, "(ii) s2 (м)" + l});
                        gr.pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s2 (с)" + l});
                        gr.pairs.add(new String[] {"(ii) s3 (м)" + l, "(ii) s1" + l});
                        gr.pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s1" + l});
                        gr.pairs.add(new String[] {"(ii) s2 (с)" + l, "(ii) s2 (м)" + l});
                        gr.pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s3 (м)" + l});
                    }
                }
                int nPro = (int) XBaseEval.probes(g, probesFrom(d, own, view == 0 ? s : 3)).stream().filter(q -> q.role().equals(FeiScores.OWN)).count();
                part1.append("\n\n######## Часть 1 ").append(view == 0 ? "(i), s = " + s : "(ii), s = 1, 2, 3 — одни и те же люди")
                        .append(String.format(Locale.ROOT, ": своих %d (≥ %d дат), проб своих %d, люди бутстрепа %d; части П2/П3 — seed + 8000 + 100 + %s ########%n",
                                own.size(), sMin, nPro, g.persons.size(), view == 0 ? Integer.toString(CONF_I[s - 1]) : "12, 13, 14"));
                if (view == 1) part1.append("Главная мера — разделимость (25 своих: П2/П3 по «чужим» галереи 5 × 5 человек ненадёжны, справочно).\n");
                if (view == 0) part1.append("Оговорка: при разных s — разные люди и галереи разного размера; сравнение между s — только в (ii).\n");
                tables(part1, gr, g, d);
            }
        }

        // ---- Часть 2, (с) и (с3).
        List<Model> modelsAll = XBaseExcl.models(d, parts0, G.ALL);
        for (int cap : new int[] {0, M}) {
            String v = cap == 0 ? "(с)" : "(с3)";
            Map<String, List<List<String>>> fsess = fullSessions(d, cap);
            Map<String, List<String>> fk = keys(fsess);
            for (int set = 0; set < SETS.size(); set++) {
                int st = set;
                int sFull = cap == 0 ? S_C_FULL : S_C3_FULL;
                int sRounds = cap == 0 ? S_C : S_C3;
                String full = block(cp, "p2_" + sFull + "_" + set, () -> XBaseFeret.fitLinesSessions(g2, xs[st], SETS.get(st), fsess, fk, ctrl2, parts0, sFull),
                        fitS);
                String rounds = block(cp, "p2_" + sRounds + "_" + set, () -> fitModelsSessions(g2, xs[st], SETS.get(st), d, modelsAll, cap, sRounds), fitS);
                allLines.append("# часть 2 Г-все ").append(v).append(" полная модель, набор ").append(SET_NAMES[set]).append('\n').append(full);
                allLines.append("# часть 2 Г-все ").append(v).append(" раунды, набор ").append(SET_NAMES[set]).append('\n').append(rounds);
                String ls = full + rounds;
                add(gr2, g2, "Г-все " + v, set, XBaseEval.parse(g2, SETS.get(set), fk, main2, parts0, ls), main2, ls, bins2.get(v));
            }
        }
        Map<String, FeiThreshold.Res> o2 = new LinkedHashMap<>();
        Map<String, Hubs> h2 = new LinkedHashMap<>();
        for (String v : List.of("(м)", "(с)", "(с3)")) {
            for (XBaseFeret.Method m : METHODS) {
                String k = "Г-все " + v + " " + m.label();
                o2.put(k, gr2.res.get(k));
                h2.put(k, gr2.hubs.get(k));
            }
        }
        gr2.res.clear();
        gr2.res.putAll(o2);
        gr2.hubs.clear();
        gr2.hubs.putAll(h2);
        for (XBaseFeret.Method m : METHODS) {
            String l = " " + m.label();
            gr2.pairs.add(new String[] {"Г-все (с)" + l, "Г-все (м)" + l});
            gr2.pairs.add(new String[] {"Г-все (с3)" + l, "Г-все (м)" + l});
            gr2.pairs.add(new String[] {"Г-все (с3)" + l, "Г-все (с)" + l});
        }
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d. Проверка (м): %d строк сводки совпали с "
                + "xbase_feret_excl.txt побайтно.%n", fitS[0], tpl.threads, matched));
        r.append(part1);
        r.append("\n\n######## Часть 2: несимметричная галерея Г-все (схема и модели (а2)), ").append(d.own().size()).append(" своих, проб ")
                .append(main2.stream().filter(q -> q.role().equals(FeiScores.OWN)).count()).append(", люди бутстрепа ").append(g2.persons.size())
                .append(" ########\n");
        r.append("Порог — одна калибровка П2/П3 на метод и вариант по полной галерее (для (с3) — 3 последние сессии каждого), контроль — против полной\n"
                + "модели варианта. (м) — строки файла оценок (а2) (блоки full и all). Ошибка своих с одной сессией в галерее для сравнения — строка\n"
                + "«(i) s1» части 1 (= Г-первая (а2)).\n");
        tables(r, gr2, g2, d);
        XBaseOther.writeScores(out.resolve("xbase_feret_hubs_scores.tsv.gz"), "# FERET (а2в), XBaseHubs, только локально; блоки — «# часть …»; (м) части 2 — "
                + "в файле оценок (а2)\n# Код: коммит " + FarMethods.commit() + "\n", allLines.toString());
        FarMethods.write(out.resolve("xbase_feret_hubs.txt"), r);
        System.out.print(r);
        cp.delete();
    }
}
