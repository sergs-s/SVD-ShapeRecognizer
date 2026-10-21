package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * (П-д) Межсессионная изменчивость FERET в обучении MLDA (PLAN.md п. 5.5; задание №1 от 14.10.2026, план подтверждён
 * Хозяином). Пул изменчивости — люди FERET с ≥ 2 датами (все вне 200 контрольных: контроль — люди с одной датой), снимки
 * всех дат по правилу равных галерей (№8: на дату один fa и один fb). Варианты обучения MLDA (признак L1 или L1-tt):
 * <ul>
 *   <li>L1 — классы галереи своих и когорта FEI 20 (без добавок);</li>
 *   <li>L1-xs-F — снимки пула вне галереи добавлены в класс самого человека (человек вне своих роли — один новый класс по
 *       всем датам);</li>
 *   <li>L1-xs0-F — те же снимки, класс = человек и дата: снимок даты, уже представленной в классе галереи человека, — в
 *       класс галереи, каждая остальная дата — отдельный новый класс (поправка Жорика 14.10.2026; класс галереи не режется,
 *       L1-xs0-F содержит L1).</li>
 * </ul>
 * Шаблоны (центры в пространстве MLDA) — только по снимкам галереи. Режим feret-xs: свои роли делятся на 5 частей по людям
 * (XBaseOther.cvParts, seed faces.seed + 9900); модель части q — без добавок людей части q, пробы — люди части q и контроль
 * (всеми 5 моделями); порог П2/П3 — снимки галереи людей части q против модели без части q, один порог по 5 частям; роли —
 * межсессионно, внутри сессии, G1, G2'. Режим gt-xs-f — перенос: протокол GT прежний (XBaseEval.fitLines), добавки — весь
 * пул FERET (новые классы) во всех моделях.
 *
 * @author ssv
 */
final class XBaseXsF {

    /** Части своих: XBaseOther.cvParts с s = PART_S, то есть seed faces.seed + 8000 + 1900 = faces.seed + 9900. */
    static final int PART_S = 1900;
    static final int PARTS = FeiMethods.CV_PARTS;

    enum Kind {
        L1("", ""), XS("-xsF", "-xs-F"), XS0("-xs0F", "-xs0-F");

        final String suffix;
        final String label;

        Kind(String suffix, String label) {
            this.suffix = suffix;
            this.label = label;
        }
    }

    /** Признаки: подпись и векторы (V.L1 набора none или tt). */
    static final String[] FEATS = {"L1", "L1-tt"};

    private XBaseXsF() {
    }

    // ---------------------------------------------------------------- добавки

    /** Обучение варианта: классы своих (по порядку own; галерея с добавками) и новые классы. */
    record Train(List<List<String>> own, List<List<String>> extra) {
        int added(Map<String, List<String>> gal, List<String> ownP) {
            int n = 0;
            for (int i = 0; i < ownP.size(); i++) n += own.get(i).size() - gal.get(ownP.get(i)).size();
            for (List<String> e : extra) n += e.size();
            return n;
        }
    }

    /**
     * Классы обучения варианта k: own — свои роли, gal — их галерея, pool — человек → даты (по возрастанию) → ключи (правило
     * равных галерей), excluded — люди без добавок (проверяемая часть), date — дата снимка.
     */
    static Train train(Kind k, List<String> own, Map<String, List<String>> gal, Map<String, List<List<String>>> pool, Set<String> excluded,
                       Map<String, String> date) {
        List<List<String>> ownCls = new ArrayList<>();
        List<List<String>> extra = new ArrayList<>();
        Set<String> ownSet = new HashSet<>(own);
        for (String p : own) {
            List<String> cls = new ArrayList<>(gal.get(p));
            if (k != Kind.L1 && pool.containsKey(p) && !excluded.contains(p)) {
                Set<String> inGal = new HashSet<>(gal.get(p));
                Set<String> galDates = new HashSet<>();
                for (String key : gal.get(p)) galDates.add(date.get(key));
                for (List<String> d : pool.get(p)) {
                    List<String> rest = new ArrayList<>();
                    for (String key : d) if (!inGal.contains(key)) rest.add(key);
                    if (rest.isEmpty()) continue;
                    if (k == Kind.XS || galDates.contains(date.get(rest.get(0)))) {
                        cls.addAll(rest);
                    } else {
                        extra.add(rest);
                    }
                }
            }
            ownCls.add(cls);
        }
        if (k != Kind.L1) {
            for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) {
                if (ownSet.contains(e.getKey()) || excluded.contains(e.getKey())) continue;
                if (k == Kind.XS) {
                    List<String> all = new ArrayList<>();
                    e.getValue().forEach(all::addAll);
                    extra.add(all);
                } else {
                    for (List<String> d : e.getValue()) extra.add(new ArrayList<>(d));
                }
            }
        }
        return new Train(ownCls, extra);
    }

    // ---------------------------------------------------------------- модели и строки

    static List<double[]> vectors(XBaseEval x, List<String> keys, boolean mirror) {
        List<double[]> out = new ArrayList<>();
        for (String k : keys) out.add(x.vec.get(V.L1).get(k));
        if (mirror) for (String k : keys) out.add(x.mir.get(V.L1).get(k));
        return out;
    }

    /** MLDA: обучение — классы train, центры галереи — классы templ (первые templ.size() классов train — галерея). */
    static GalleryScorer.ScoreModel fit(XBaseEval x, boolean mirror, List<List<String>> train, List<List<String>> templ) {
        List<List<double[]>> tr = new ArrayList<>();
        for (List<String> ks : train) tr.add(vectors(x, ks, mirror));
        List<List<double[]>> te = new ArrayList<>();
        for (List<String> ks : templ) te.add(vectors(x, ks, mirror));
        return new MldaScorer().fit(tr, te, templ.size(), 1e-10);
    }

    static String lines(XBaseEval x, String method, int q, GalleryScorer.ScoreModel m, List<String> gallery, List<Probe> probes) {
        StringBuilder t = new StringBuilder();
        for (Probe p : probes) {
            t.append(FeiScores.line(method, q, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, m.scores(x.vec.get(V.L1).get(p.key())), false));
        }
        return t.toString();
    }

    /** Конфигурация роли: подпись, номер, свои, галерея, пробы. */
    record Conf(String name, int s, List<String> own, Map<String, List<String>> gal, Map<String, List<String>> pro) {}

    /** Строки оценок конфигурации c для векторов x: все варианты, 5 частей, зеркала; модели частей и модели порога. */
    static String fitConf(XBaseEval.Gt g, XBaseEval x, Conf c, List<List<String>> parts, Map<String, List<List<String>>> pool, Map<String, String> date,
                          List<String> ctrlKeys) {
        List<String> own = c.own();
        List<Callable<String>> tasks = new ArrayList<>();
        for (Kind kind : Kind.values()) {
            for (int q = 0; q < PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                Train tr = train(kind, own, c.gal(), pool, part, date);
                List<List<String>> ownTempl = own.stream().map(c.gal()::get).toList();
                List<List<String>> mainTrain = new ArrayList<>(tr.own());
                mainTrain.addAll(tr.extra());
                mainTrain.addAll(g.cohort);
                List<String> rest = new ArrayList<>();
                List<List<String>> restTrain = new ArrayList<>();
                List<List<String>> restTempl = new ArrayList<>();
                for (int i = 0; i < own.size(); i++) {
                    if (part.contains(own.get(i))) continue;
                    rest.add(own.get(i));
                    restTrain.add(tr.own().get(i));
                    restTempl.add(c.gal().get(own.get(i)));
                }
                restTrain.addAll(tr.extra());
                restTrain.addAll(g.cohort);
                List<Probe> mainProbes = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : c.pro().get(p)) mainProbes.add(new Probe(FeiScores.OWN, p, g.img(k), k, own.indexOf(p)));
                for (String k : ctrlKeys) mainProbes.add(new Probe(FeiScores.CTRL, person(g, k), g.img(k), k, -1));
                List<Probe> calProbes = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : c.gal().get(p)) calProbes.add(new Probe(FeiMethods.CV_ROLE + q, p, g.img(k), k, -1));
                int qq = q;
                for (boolean m : new boolean[] {false, true}) {
                    String method = FeiMethods.LBP_MLDA + kind.suffix + (m ? FeiMethods.MIRROR : "");
                    tasks.add(() -> lines(x, method, qq, fit(x, m, mainTrain, ownTempl), own, mainProbes));
                    tasks.add(() -> lines(x, method, qq, fit(x, m, restTrain, restTempl), rest, calProbes));
                }
            }
        }
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, g.threads).forEach(all::append);
        return all.toString();
    }

    /** Человек контрольного снимка (по byPerson). */
    static String person(XBaseEval.Gt g, String key) {
        for (Map.Entry<String, Map<Integer, List<String>>> e : g.byPerson.entrySet()) for (List<String> ks : e.getValue().values()) if (ks.contains(key)) return e.getKey();
        throw new IllegalStateException("Контрольный снимок без человека: " + key);
    }

    /** Строка модели части q по позе: анфас — X, иначе X+mirror. */
    static Row choose(Map<String, Row> rows, String id, int q, String role, String person, int image, boolean frontal) {
        String method = id + (frontal ? "" : FeiMethods.MIRROR);
        Row r = rows.get(method + "\t" + q + "\t" + role + "\t" + person + "\t" + image);
        if (r == null) throw new IllegalStateException("Нет строки " + method + " s=" + q + " " + role + " " + person + " №" + image);
        return r;
    }

    /** Итог варианта kind по строкам конфигурации (порог П2/П3 — по «чужим» 5 частей). */
    static FeiThreshold.Res result(XBaseEval.Gt g, Conf c, List<List<String>> parts, List<String> ctrlKeys, Map<String, Row> rows, Kind kind, String label) {
        String id = FeiMethods.LBP_MLDA + kind.suffix;
        FeiDecompose.Cfg cf = new FeiDecompose.Cfg();
        List<Row> ownRows = new ArrayList<>();
        List<Boolean> front = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        List<double[]> cv = new ArrayList<>();
        int cvFront = 0;
        for (int q = 0; q < PARTS; q++) {
            for (String p : parts.get(q)) {
                for (String k : c.pro().get(p)) {
                    boolean f = g.rOf.get(k) <= g.r0;
                    ownRows.add(choose(rows, id, q, FeiScores.OWN, p, g.img(k), f));
                    front.add(f);
                }
                for (String k : c.gal().get(p)) {
                    boolean f = g.rOf.get(k) <= g.r0;
                    if (f) cvFront++;
                    Row r = choose(rows, id, q, FeiMethods.CV_ROLE + q, p, g.img(k), f);
                    cv.add(new double[] {g.pIdx.get(p), FeiScores.ratio(r, null)});
                }
            }
            for (String k : ctrlKeys) {
                String p = person(g, k);
                Row r = choose(rows, id, q, FeiScores.CTRL, p, g.img(k), g.rOf.get(k) <= g.r0);
                ctrl.add(new double[] {g.pIdx.get(p), FeiScores.ratio(r, null)});
            }
        }
        int n = ownRows.size();
        cf.ownP = new int[n];
        cf.ownR = new double[n];
        cf.ownOk = new boolean[n];
        cf.noSecond = new boolean[n];
        cf.trueInf = new boolean[n];
        cf.poseFrontal = new boolean[n];
        for (int i = 0; i < n; i++) {
            Row r = ownRows.get(i);
            int[] cs = new int[FeiScores.CASES];
            cf.ownP[i] = g.pIdx.get(r.person());
            cf.ownR[i] = FeiScores.ratio(r, cs);
            cf.noSecond[i] = cs[FeiScores.CASE_NO_SECOND] > 0;
            cf.trueInf[i] = Double.isInfinite(r.trueScore());
            cf.ownOk[i] = r.bestId().equals(r.trueId());
            cf.poseFrontal[i] = front.get(i);
        }
        cf.thrP = new int[0];
        cf.thrR = new double[0];
        cf.ctrlP = ctrl.stream().mapToInt(d -> (int) d[0]).toArray();
        cf.ctrlR = ctrl.stream().mapToDouble(d -> d[1]).toArray();
        return FeiThreshold.evaluate(label, id + "-ratio+pose", List.of(cf), List.of(FeiThreshold.Imp.of(cv, cvFront)), g.persons.size(), g.b,
                g.seed + FeiBootstrap.SEED_OFFSET);
    }

    static Map<String, Row> rows(String lines) {
        Map<String, Row> rows = new HashMap<>();
        for (String l : lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.s() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        return rows;
    }

    static String label(int feat, Kind kind) {
        return FEATS[feat] + kind.label;
    }

    static final String INTERPRETATION = "Толкование (Жорик, 14.10.2026; записано до прогона): L1-xs-F − L1-xs0-F < 0 значимо – помогает именно знание о смене "
            + "дней; L1-xs0-F − L1 < 0 при L1-xs-F − L1-xs0-F ≈ 0 – помогает просто больше людей. Внутри сессии – «не хуже»: верхняя граница\n"
            + "L1-xs-F − L1 (П2 FAR 1 %) ≤ +3 п. п.\n";

    // ---------------------------------------------------------------- feret-xs

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-xs: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(out);
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, String> date = new HashMap<>();
        su.byKey.forEach((k, im) -> date.put(k, im.date()));
        Map<String, List<List<String>>> pool = pool(roAll, su.byKey);

        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.g2, roAll.g2p, roAll.pro2, roAll.ctrlKeys)) m.values().forEach(used::addAll);
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);

        List<List<Conf>> groups = List.of(List.of(new Conf("межс.", 0, ro.own1, ro.gal1, ro.pro1), new Conf("внутри", 1, ro.own1in, ro.gal1in, ro.pro1in)),
                List.of(new Conf("G1", 2, ro.own2, ro.g1, ro.pro2), new Conf("G2'", 5, ro.own2, ro.g2p, ro.pro2)));
        List<String> ctrlKeys = new ArrayList<>();
        ro.ctrlKeys.values().forEach(ctrlKeys::addAll);
        ctrlKeys.sort(null);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-xs %s; снимков %d; seed %d; r0 %s; пул %d; №8 галерея fa/fb", dir,
                su.imgs.size(), tpl.seed, tpl.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_xs_eq"), fp);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (П-д) межсессионная изменчивость FERET в обучении MLDA (XBaseEval feret-xs): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры — как feret, кэш derived/); когорта FEI — экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(XBaseFeret.EQ_NOTE);
        int poolImages = pool.values().stream().mapToInt(ds -> ds.stream().mapToInt(List::size).sum()).sum();
        long pool3 = pool.values().stream().filter(ds -> ds.size() >= 3).count();
        r.append(String.format(Locale.ROOT, "Пул изменчивости: людей с ≥ 2 датами %d (с ≥ 3 датами %d), снимков по правилу равных галерей %d; в контрольных 200 "
                + "из них 0 (контроль — люди с одной датой).%n", pool.size(), pool3, poolImages));
        r.append("Варианты (признак L1 – LBP u2 (8, 2) 7×7 + MLDA, none; L1-tt – то же на выходе Tan – Triggs; когорта FEI 20; зеркала способом а и\n"
                + "выбор X / X+mirror по |r| пробы, r0 = " + tpl.r0 + "; шаблоны – только по снимкам галереи):\n"
                + "  L1 – классы галереи своих роли + когорта;\n"
                + "  L1-xs-F – снимки пула вне галереи (все даты) – в класс самого человека; человек пула вне своих роли – один новый класс;\n"
                + "  L1-xs0-F – те же снимки: дата, уже представленная в галерее человека, – в класс галереи, каждая остальная дата – новый\n"
                + "  класс (класс галереи не режется; L1-xs0-F содержит L1).\n"
                + "Свои роли – 5 частей по людям (XBaseOther.cvParts, seed faces.seed + 9900); модель части q – без добавок людей части q;\n"
                + "пробы – люди части q и контрольные чужие (всеми 5 моделями); порог П2/П3 – снимки галереи людей части q против модели без\n"
                + "части q, один порог по 5 частям. Контрольные 200 – никогда в обучении. L1 здесь – со своей схемой порога (не равна\n"
                + "строке L1 отчёта feret).\n");
        r.append(INTERPRETATION);
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям группы (свои и контроль): B = %d, seed = %d; «*» — интервал разности не содержит 0.%n",
                tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        StringBuilder adds = new StringBuilder("\n=== Добавки по частям: снимков вне галереи и новых классов (L1-xs-F / L1-xs0-F) ===\n"
                + "роль | часть | людей части | L1-xs-F: снимков, новых классов | L1-xs0-F: снимков, новых классов\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double[] fitS = {0};
        for (List<Conf> group : groups) {
            TreeSet<String> ps = new TreeSet<>(group.get(0).own());
            ps.addAll(ro.ctrl);
            List<String> persons = new ArrayList<>(ps);
            XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            for (Conf c : group) {
                XBaseEval.Gt g = XBaseFeret.withOwn(base, c.own());
                List<List<String>> parts = XBaseOther.cvParts(c.own(), tpl.seed, PART_S);
                for (int q = 0; q < PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    Train t1 = train(Kind.XS, c.own(), c.gal(), pool, part, date);
                    Train t0 = train(Kind.XS0, c.own(), c.gal(), pool, part, date);
                    adds.append(String.format(Locale.ROOT, "%s | %d | %d | %d, %d | %d, %d%n", c.name(), q, parts.get(q).size(), t1.added(c.gal(), c.own()),
                            t1.extra().size(), t0.added(c.gal(), c.own()), t0.extra().size()));
                }
                for (int f = 0; f < FEATS.length; f++) {
                    String name = String.format(Locale.ROOT, "conf_%d_%d", c.s(), f);
                    String ls = cp.read(name);
                    if (ls == null) {
                        long t0 = System.nanoTime();
                        ls = fitConf(g, xs[f], c, parts, pool, date, ctrlKeys);
                        double s = (System.nanoTime() - t0) / 1e9;
                        fitS[0] += s;
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "feret-xs: %s, %s готово (%.0f s)%n", c.s() == 5 ? "G2p" : c.s() == 2 ? "G1" : c.s() == 1 ? "in" : "cross",
                                FEATS[f], s);
                    }
                    lines.append("# конфигурация ").append(c.name()).append(", признак ").append(FEATS[f]).append('\n').append(ls);
                    Map<String, Row> rows = rows(ls);
                    for (Kind kind : Kind.values()) {
                        String label = c.name() + " " + label(f, kind);
                        res.put(label, result(g, c, parts, ctrlKeys, rows, kind, label));
                    }
                    String b = c.name() + " " + FEATS[f];
                    pairs.add(new String[] {b + Kind.XS.label, b});
                    pairs.add(new String[] {b + Kind.XS.label, b + Kind.XS0.label});
                    pairs.add(new String[] {b + Kind.XS0.label, b});
                }
            }
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            int[] unit = FeiBootstrap.unit(persons.size());
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
            body.append("\n\n######## ").append(group.get(0).s() == 0 ? "Роль (1): межсессионно и внутри сессии" : "Роль (2): G1 и G2'")
                    .append(String.format(Locale.ROOT, " — бутстреп по %d людям (свои и контроль) ########%n", persons.size()));
            XBaseOther.tables(body, res, sep, tpl.b, pairs);
        }
        r.append(adds);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS[0], tpl.threads));
        r.append(body);
        XBaseOther.writeScores(out.resolve("xbase_feret_xs_scores.tsv.gz"), "# FERET (П-д), XBaseXsF, только локально; s — часть q; блоки — конфигурация и "
                + "признак\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_xs.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /** Пул: люди роли (1) (≥ 2 дат) → даты по возрастанию → ключи по правилу равных галерей (XBaseFeret.pairOf). */
    static Map<String, List<List<String>>> pool(XBaseFeret.Roles ro, Map<String, Img> byKey) {
        Map<String, List<List<String>>> out = new TreeMap<>();
        for (String p : ro.own1) {
            Map<String, List<String>> byDate = new TreeMap<>();
            List<String> all = new ArrayList<>(ro.gal1.get(p));
            all.addAll(ro.pro1.get(p));
            for (String k : all) byDate.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ds = new ArrayList<>();
            for (List<String> d : byDate.values()) ds.add(XBaseFeret.sorted(XBaseFeret.pairOf(d, byKey, false)));
            out.put(p, ds);
        }
        return out;
    }

    // ---------------------------------------------------------------- gt-xs-f

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-xs-f: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = pool(su.ro, su.byKey);

        // Векторы: GT и когорта (none — g.x, tt — заново), пул FERET (кадры «а» derived/).
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        xs[1].vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);
        Map<String, Mat> pf = new LinkedHashMap<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) pf.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval p0 = new XBaseEval();
        p0.vectors(pf, EnumSet.of(V.L1), g.params);
        XBaseEval p1 = new XBaseEval();
        p1.vectors(pf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        pf.values().forEach(Mat::release);
        xs[0].vec.get(V.L1).putAll(p0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(p0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(p1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(p1.mir.get(V.L1));

        Map<Kind, List<List<String>>> extra = new LinkedHashMap<>();
        extra.put(Kind.L1, List.of());
        extra.put(Kind.XS, train(Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra());
        Map<String, String> date = new HashMap<>();
        su.byKey.forEach((k, im) -> date.put(k, im.date()));
        extra.put(Kind.XS0, train(Kind.XS0, List.of(), Map.of(), pool, Set.of(), date).extra());

        List<Map<String, List<String>>> split = XBaseEval.within(g);
        Object[][] modes = {{"межс.", g.galleryKeys, g.probeKeys}, {"внутри", split.get(0), split.get(1)}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        List<List<String>> cohort0 = g.cohort;
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        StringBuilder lines = new StringBuilder();
        long t0 = System.nanoTime();
        // Контрольная точка – после каждого режима и варианта (решение Хозяина 20.10.2026: прогон без неё терял до 80 мин).
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_xs_f"), FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT,
                "gt-xs-f; seed %d; B %d; r0 %s; пул %d", g.seed, g.b, g.r0, pool.size())));
        try {
            for (Object[] md : modes) {
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Probe> probes = XBaseEval.probes(g, pro);
                for (int f = 0; f < FEATS.length; f++) {
                    for (Kind kind : Kind.values()) {
                        List<List<String>> aug = new ArrayList<>(cohort0);
                        aug.addAll(extra.get(kind));
                        g.cohort = aug;
                        String part = (md[0].equals("межс.") ? "x" : "i") + "_" + f + "_" + kind.ordinal();
                        String ls = cp.read(part);
                        if (ls == null) {
                            ls = XBaseEval.fitLines(g, xs[f], EnumSet.of(V.L1), gal, probes, parts, 0);
                            cp.save(part, ls);
                        }
                        lines.append("# режим ").append(md[0]).append(", вариант ").append(label(f, kind)).append('\n').append(ls);
                        XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, ls);
                        String label = md[0] + " " + label(f, kind);
                        res.put(label, FeiThreshold.evaluate(label, V.L1.ratioId(), List.of(p.cfg.get(V.L1)), List.of(p.imp.get(V.L1)), g.persons.size(), g.b,
                                g.seed + FeiBootstrap.SEED_OFFSET));
                        System.out.printf(Locale.ROOT, "gt-xs-f: %s, %s готово%n", md[0].equals("межс.") ? "cross" : "in", label(f, kind));
                    }
                    String b = md[0] + " " + FEATS[f];
                    pairs.add(new String[] {b + Kind.XS.label, b});
                    pairs.add(new String[] {b + Kind.XS.label, b + Kind.XS0.label});
                    pairs.add(new String[] {b + Kind.XS0.label, b});
                }
            }
        } finally {
            g.cohort = cohort0;
        }
        double fitS = (System.nanoTime() - t0) / 1e9;

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, (П-д) (2): перенос межсессионной изменчивости FERET → GT (XBaseEval gt-xs-f)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir))
                .append(" (a/gt; когорта – a/fei); пул – Color FERET (кадры «а» derived/, только в обучении; имён снимков FERET в отчёте и оценках нет)\n");
        int poolImages = pool.values().stream().mapToInt(ds -> ds.stream().mapToInt(List::size).sum()).sum();
        t.append(String.format(Locale.ROOT, "Пул FERET: людей с ≥ 2 датами %d, снимков по правилу равных галерей %d; новых классов: L1-xs-F %d, L1-xs0-F %d "
                + "(снимков поровну, %d).%n", pool.size(), poolImages, extra.get(Kind.XS).size(), extra.get(Kind.XS0).size(),
                extra.get(Kind.XS0).stream().mapToInt(List::size).sum()));
        t.append("Протокол GT прежний (gt / gt-in: галерея, пробы, контроль, когорта FEI 20, порог П2/П3 по «чужим» галереи, 5 частей seed\n"
                + "faces.seed + 8100); добавки пула – новыми классами во всех моделях (основной и моделях порога). L1-xs-F – человек FERET\n"
                + "– один класс по всем датам; L1-xs0-F – класс на человека и дату; L1 – без добавок (= gt). L1-tt – то же на выходе Tan – Triggs.\n");
        t.append(INTERPRETATION);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d; «*» — интервал разности не содержит 0.%n", g.b,
                g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с, потоков %d.%n", fitS, g.threads));
        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(g.persons.size());
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
        XBaseOther.tables(t, res, sep, g.b, pairs);
        XBaseOther.writeScores(out.resolve("xbase_gt_xs_f_scores.tsv.gz"), "# GT (П-д) (2), XBaseXsF gt-xs-f; блоки — режим и вариант; роли own, ctrl, cv<q>\n"
                + "# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_xs_f.txt"), t);
        cp.delete();
        System.out.print(t);
    }

    // ---------------------------------------------------------------- feret-xs2

    /** feret-xs2: деление пула пополам — java.util.Random(faces.seed + SPLIT_SEED_OFFSET). */
    static final long SPLIT_SEED_OFFSET = 9910;
    /** Хаб — человек пула с не меньше чем HUB_DATES датами. */
    static final int HUB_DATES = 5;

    /**
     * (П-д) внутри FERET с непересекающимися людьми (задание №2 от 14.10.2026, план подтверждён Хозяином): люди пула делятся
     * пополам (A — первые ⌈n/2⌉ после перемешивания, B — остальные); проход 1 — свои A, пул B, проход 2 — наоборот. Свои и пул
     * не пересекаются — добавки пула (новые классы) одни для всех моделей прохода, модель — XBaseEval.fitLines с когортой,
     * расширенной пулом; порог П2/П3 — по «чужим» галереи своих прохода (5 частей, seed faces.seed + 8000 + 100 + s), свой на
     * проход; итог — объединение проходов (FeiThreshold.evaluate по двум конфигурациям).
     */
    static void feret2(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-xs2: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(out);
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = pool(roAll, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        Set<String> hubs = new TreeSet<>();
        pool.forEach((p, ds) -> {
            if (ds.size() >= HUB_DATES) hubs.add(p);
        });

        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.g2, roAll.g2p, roAll.pro2, roAll.ctrlKeys)) m.values().forEach(used::addAll);
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);

        // Пул прохода: человек → даты (люди другой половины).
        List<Map<String, List<List<String>>>> passPool = new ArrayList<>();
        for (int ps = 0; ps < 2; ps++) {
            Map<String, List<List<String>>> m = new TreeMap<>();
            for (String p : halves.get(1 - ps)) m.put(p, pool.get(p));
            passPool.add(m);
        }
        List<Conf> confs = List.of(new Conf("межс.", 0, ro.own1, ro.gal1, ro.pro1), new Conf("внутри", 1, ro.own1in, ro.gal1in, ro.pro1in),
                new Conf("G1", 2, ro.own2, ro.g1, ro.pro2), new Conf("G2'", 5, ro.own2, ro.g2p, ro.pro2));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-xs2 %s; снимков %d; seed %d; r0 %s; пул %d; половины %d/%d; "
                + "№8 галерея fa/fb", dir, su.imgs.size(), tpl.seed, tpl.r0, pool.size(), halves.get(0).size(), halves.get(1).size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_xs2_eq"), fp);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (П-д) внутри FERET с непересекающимися людьми (XBaseEval feret-xs2): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры — как feret, кэш derived/); когорта FEI — экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(XBaseFeret.EQ_NOTE);
        r.append(String.format(Locale.ROOT, "Пул: людей с ≥ 2 датами %d; половины (java.util.Random(seed %d + %d)): A %d, B %d. Проход 1 – свои A (во всех ролях "
                + "только люди A), пул B; проход 2 – наоборот; контрольные 200 – в обоих проходах, никогда в обучении. Хабы – люди пула с ≥ %d датами: "
                + "%d (%s).%n", pool.size(), tpl.seed, SPLIT_SEED_OFFSET, halves.get(0).size(), halves.get(1).size(), HUB_DATES, hubs.size(),
                String.join(", ", hubs)));
        r.append("Строки (признак L1 – LBP u2 (8, 2) 7×7 + MLDA, none; L1-tt – то же на выходе Tan – Triggs; когорта FEI 20; зеркала и выбор X /\n"
                + "X+mirror по |r| пробы, r0 = " + tpl.r0 + "; шаблоны – по галерее своих): L1 – галерея своих + когорта; L1-xs-F – + пул (все даты,\n"
                + "равные галереи), человек пула – один класс; L1-xs0-F – + пул, каждая дата человека пула – отдельный класс. Порог П2/П3 – по\n"
                + "«чужим» галереи своих прохода (5 частей, seed faces.seed + 8000 + 100 + s), свой на проход; итог – объединение проходов.\n");
        r.append("Оговорка: галерея вдвое меньше, чем в ролях №8–№9 – числа с ними напрямую не сравнивать, сравнивать только строки между собой; в G1\n"
                + "и G2' своих около 30 на проход – различимы только крупные эффекты.\n");
        r.append(INTERPRETATION);
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для обоих проходов и всех строк: B = %d, seed = %d; «*» — "
                + "интервал разности не содержит 0.%n", persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        StringBuilder vol = new StringBuilder("\n=== Объём по проходам ===\nроль | проход | своих | проб своих | новых классов L1-xs-F / L1-xs0-F | снимков пула\n");
        StringBuilder hubT = new StringBuilder("\n=== Лучший кандидат – хаб (человек пула с ≥ " + HUB_DATES + " датами; модель по позе пробы; оба прохода) ===\n"
                + "строка | проб своих: лучший – хаб | из них хаб – не свой | контроль: лучший – хаб\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double[] fitS = {0};
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        for (Conf c : confs) {
            if (c.s() == 2) {
                flush(body, res, pairs, persons.size(), tpl, "Роль (1): межсессионно и внутри сессии");
                res = new LinkedHashMap<>();
                pairs = new ArrayList<>();
            }
            for (int f = 0; f < FEATS.length; f++) {
                for (Kind kind : Kind.values()) {
                    List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
                    List<FeiThreshold.Imp> imps = new ArrayList<>();
                    int[] hub = new int[4];
                    for (int ps = 0; ps < 2; ps++) {
                        Set<String> mine = halves.get(ps);
                        List<String> own = c.own().stream().filter(mine::contains).toList();
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        Map<String, List<String>> pro = new LinkedHashMap<>();
                        for (String p : own) {
                            gal.put(p, c.gal().get(p));
                            pro.put(p, c.pro().get(p));
                        }
                        List<List<String>> extra = train(kind, List.of(), Map.of(), passPool.get(ps), Set.of(), Map.of()).extra();
                        XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                        List<List<String>> aug = new ArrayList<>(cohort);
                        aug.addAll(extra);
                        g.cohort = aug;
                        List<Probe> probes = XBaseEval.probes(g, pro);
                        List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                        String name = String.format(Locale.ROOT, "p%d_%d_%d_%s", ps, c.s(), f, kind.name());
                        String ls = cp.read(name);
                        if (ls == null) {
                            long t0 = System.nanoTime();
                            ls = XBaseEval.fitLines(g, xs[f], EnumSet.of(V.L1), gal, probes, parts, c.s());
                            double s = (System.nanoTime() - t0) / 1e9;
                            fitS[0] += s;
                            cp.save(name, ls);
                            System.out.printf(Locale.ROOT, "feret-xs2: pass %d, s %d, %s, %s ready (%.0f s)%n", ps + 1, c.s(), FEATS[f], kind.name(), s);
                        }
                        lines.append("# проход ").append(ps + 1).append(", конфигурация ").append(c.name()).append(", строка ").append(label(f, kind))
                                .append('\n').append(ls);
                        XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, ls);
                        cfgs.add(p.cfg.get(V.L1));
                        imps.add(p.imp.get(V.L1));
                        Map<String, Row> rows = new HashMap<>();
                        for (String l : ls.lines().toList()) {
                            Row row = FeiScores.parse(l);
                            rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
                        }
                        for (Probe q : probes) {
                            Row row = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), g.rOf.get(q.key()) <= g.r0);
                            boolean isHub = hubs.contains(row.bestId());
                            if (q.role().equals(FeiScores.OWN)) {
                                hub[0]++;
                                if (isHub) hub[1]++;
                                if (isHub && !row.bestId().equals(row.trueId())) hub[2]++;
                            } else if (isHub) {
                                hub[3]++;
                            }
                        }
                        if (f == 0 && kind == Kind.XS0) {
                            List<List<String>> e1 = train(Kind.XS, List.of(), Map.of(), passPool.get(ps), Set.of(), Map.of()).extra();
                            vol.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d / %d | %d%n", c.name(), ps + 1, own.size(),
                                    probes.stream().filter(q -> q.role().equals(FeiScores.OWN)).count(), e1.size(), extra.size(),
                                    extra.stream().mapToInt(List::size).sum()));
                        }
                    }
                    String label = c.name() + " " + label(f, kind);
                    res.put(label, FeiThreshold.evaluate(label, V.L1.ratioId(), cfgs, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    int ctrlN = cfgs.stream().mapToInt(cf -> cf.ctrlR.length).sum();
                    hubT.append(String.format(Locale.ROOT, "%s | %d/%d = %.1f %% | %d | %d/%d = %.1f %%%n", label, hub[1], hub[0], 100.0 * hub[1] / hub[0], hub[2],
                            hub[3], ctrlN, 100.0 * hub[3] / ctrlN));
                }
                String b = c.name() + " " + FEATS[f];
                pairs.add(new String[] {b + Kind.XS.label, b});
                pairs.add(new String[] {b + Kind.XS.label, b + Kind.XS0.label});
                pairs.add(new String[] {b + Kind.XS0.label, b});
            }
        }
        flush(body, res, pairs, persons.size(), tpl, "Роль (2): G1 и G2'");
        r.append(vol);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS[0], tpl.threads));
        r.append(hubT);
        r.append(body);
        XBaseOther.writeScores(out.resolve("xbase_feret_xs2_scores.tsv.gz"), "# FERET (П-д), XBaseXsF feret-xs2, только локально; блоки — проход, "
                + "конфигурация, строка; s — номер конфигурации\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_xs2.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /** Таблицы группы ролей (разделимость, EER, парные разности; бутстреп по persons людям). */
    static void flush(StringBuilder body, Map<String, FeiThreshold.Res> res, List<String[]> pairs, int persons, XBaseEval.Gt tpl, String title) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons);
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
        body.append("\n\n######## ").append(title).append(" — оба прохода ########\n");
        XBaseOther.tables(body, res, sep, tpl.b, pairs);
    }
}
