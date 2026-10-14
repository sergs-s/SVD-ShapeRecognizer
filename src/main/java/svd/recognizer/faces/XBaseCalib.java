package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Калибровка порога межсессионно (PLAN.md п. 5.5, (2а); задание №1 от 15.10.2026, план подтверждён Хозяином). Способы
 * (Нейман – Пирсон, FAR 1 %): К0 — П2 по «чужим» галереи (как сейчас), К0' — П3; К1 — П2 по внешним межсессионным чужим —
 * снимкам калибровочных людей (не в галерее и не в обучении проекции); К2 — П3 по ним же; К3 — П2 по объединению «чужих»
 * галереи и К1. Модель и пробы прежние, меняется только набор «чужих» порога; оценки К1 — дополнительные пробы основной
 * модели (роль cal). Режимы feret-calib (сценарий 1 — половины №2, свои A, пул B → проекция B1 и калибровка B2; сценарий 2 —
 * смешанная галерея D = 2, строки (а) и (в2)) и gt-calib (сценарий 3 — GT, пул FERET → проекция и калибровка; строки (а) и
 * (в) gt-synth). Деления пула — java.util.Random(faces.seed + 9960).
 *
 * @author ssv
 */
final class XBaseCalib {

    static final long CAL_SEED_OFFSET = 9960;
    static final String CAL_ROLE = "cal";
    /** Меньше стольких чужих К1 на проход — К2 (П3 по К1) вырожден: верхняя 95 % граница КП ≤ 1 % недостижима и при нуле. */
    static final int K2_MIN = 299;

    /** Метод: подпись, набор векторов (0 — none, 1 — tt), с пулом проекции. */
    record Meth(String label, int feat, boolean pool) {}

    static final List<Meth> FERET_METHS = List.of(new Meth("L1", 0, false), new Meth("L1-tt", 1, false), new Meth("L1-xs-F", 0, true),
            new Meth("L1-tt-xs-F", 1, true));
    static final List<Meth> POOL_METHS = List.of(new Meth("L1-xs-F", 0, true), new Meth("L1-tt-xs-F", 1, true), new Meth("L1", 0, false));

    /** Итоги одного прохода: данные своих и контроля, «чужие» галереи, чужие К1. */
    record Pass(FeiDecompose.Cfg cfg, FeiThreshold.Imp cv, FeiThreshold.Imp cal) {}

    private XBaseCalib() {
    }

    /** Прогон прохода: строки оценок (основная модель — свои, контроль, К1; модели частей — «чужие» галереи) и разбор. */
    static Pass pass(XBaseEval.Gt g, XBaseEval x, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, int s,
                     List<String> calKeys, Map<String, String> calPerson, XBaseOther.Checkpoint cp, String name, StringBuilder lines, double[] fitS)
            throws IOException {
        List<Probe> probes = XBaseEval.probes(g, pro);
        List<Probe> all = new ArrayList<>(probes);
        for (String k : calKeys) all.add(new Probe(CAL_ROLE, calPerson.get(k), g.img(k), k, -1));
        String ls = cp.read(name);
        if (ls == null) {
            long t0 = System.nanoTime();
            ls = XBaseEval.fitLines(g, x, EnumSet.of(V.L1), gal, all, parts, s);
            double sec = (System.nanoTime() - t0) / 1e9;
            fitS[0] += sec;
            cp.save(name, ls);
            System.out.printf(Locale.ROOT, "calib: %s ready (%.0f s)%n", name, sec);
        }
        lines.append("# ").append(name).append('\n').append(ls);
        XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, ls);
        Map<String, Row> rows = new HashMap<>();
        for (String l : ls.lines().toList()) {
            Row row = FeiScores.parse(l);
            rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
        }
        List<double[]> cal = new ArrayList<>();
        int front = 0;
        for (String k : calKeys) {
            boolean f = g.rOf.get(k) <= g.r0;
            if (f) front++;
            Row row = XBaseEval.choose(rows, V.L1, CAL_ROLE, calPerson.get(k), g.img(k), f);
            cal.add(new double[] {g.pIdx.get(calPerson.get(k)), FeiScores.ratio(row, null)});
        }
        return new Pass(p.cfg.get(V.L1), p.imp.get(V.L1), FeiThreshold.Imp.of(cal, front));
    }

    /** Объединение «чужих» галереи и К1. */
    static FeiThreshold.Imp union(FeiThreshold.Imp a, FeiThreshold.Imp b) {
        List<double[]> v = new ArrayList<>();
        for (int i = 0; i < a.r().length; i++) v.add(new double[] {a.p()[i], a.r()[i]});
        for (int i = 0; i < b.r().length; i++) v.add(new double[] {b.p()[i], b.r()[i]});
        return FeiThreshold.Imp.of(v, a.frontal() + b.frontal());
    }

    /** Способы: подпись, источник «чужих» (0 — галерея, 1 — К1, 2 — объединение), m (2 — П2, 3 — П3). */
    record Way(String label, int src, int m) {}

    static final List<Way> WAYS = List.of(new Way("К0 (П2)", 0, 2), new Way("К0' (П3)", 0, 3), new Way("К1 (П2 по К1)", 1, 2), new Way("К2 (П3 по К1)", 1, 3),
            new Way("К3 (П2 по объединению)", 2, 2));
    static final String[] SRC = {"К0", "К1", "К3"};

    /** Итоги сценария × метода по проходам: три Res (источники «чужих»), подписи — base + " " + SRC. */
    static void results(Map<String, FeiThreshold.Res> res, String base, List<Pass> passes, int persons, XBaseEval.Gt tpl) {
        List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        List<List<FeiThreshold.Imp>> imps = List.of(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        for (Pass p : passes) {
            cfgs.add(p.cfg());
            imps.get(0).add(p.cv());
            imps.get(1).add(p.cal());
            imps.get(2).add(union(p.cv(), p.cal()));
        }
        for (int s = 0; s < 3; s++) {
            String l = base + " " + SRC[s];
            res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), cfgs, imps.get(s), persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        }
    }

    /** Сводка «способ × сценарий»: FAR контроля по снимкам с КП, по людям, |FAR − 1 %|, допуск, «держит» (КП ↑ ≤ 1 %), ошибка своих. */
    static void summary(StringBuilder t, Map<String, FeiThreshold.Res> res, List<String> bases, int b, Map<String, Integer> minCal) {
        t.append("\n=== Сводка: способ × сценарий (FAR 1 %; «держит» – верхняя 95 % граница Клоппера – Пирсона FAR контроля по снимкам ≤ 1 %) ===\n"
                + "сценарий и метод | способ | FAR контроля по снимкам; КП | |FAR − 1 %| | по людям (КП) | допуск (правило проекта) | держит | ошибка своих\n");
        for (String base : bases) {
            for (Way w : WAYS) {
                FeiThreshold.Res r = res.get(base + " " + SRC[w.src()]);
                int x = (int) r.far[w.m()][0][0];
                int n = (int) r.far[w.m()][0][1];
                double far = 100.0 * x / n;
                double up = 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95);
                String row = LbpNormReport.farRow(List.of(), r, w.m(), 0, b);
                String[] f = row.trim().split(" \\| ");
                String people = f.length >= 3 ? f[2] : "?";
                String adm = f.length >= 4 ? f[3] : "?";
                String deg = w.label().startsWith("К2") && minCal.getOrDefault(base, Integer.MAX_VALUE) < K2_MIN ? " (вырожден: К1 < " + K2_MIN + " на проход)" : "";
                t.append(String.format(Locale.ROOT, "%s | %s%s | %.2f %%; КП %d/%d ↑%.2f %% | %.2f | %s | %s | %s | %s%n", base, w.label(), deg, far, x, n, up,
                        Math.abs(far - 1), people, adm, up <= 1.0 ? "да" : "нет", FeiThreshold.errCell(r, w.m(), 0, 0, b)));
            }
        }
    }

    static List<List<String>> split(Set<String> people, long seed) {
        List<String> perm = new ArrayList<>(new TreeSet<>(people));
        Collections.shuffle(perm, new Random(seed + CAL_SEED_OFFSET));
        int h = perm.size() / 2;
        return List.of(new ArrayList<>(perm.subList(0, h)), new ArrayList<>(perm.subList(h, perm.size())));
    }

    // ---------------------------------------------------------------- feret-calib

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-calib: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(out);
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);
        Map<String, String> personOf = new HashMap<>();
        pool.forEach((p, ds) -> ds.forEach(d -> d.forEach(k -> personOf.put(k, p))));

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
        for (Mat m : fr.values()) m.release();

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-calib %s; снимков %d; seed %d; r0 %s; пул %d; №8 галерея fa/fb", dir,
                su.imgs.size(), tpl.seed, tpl.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_calib_eq"), fp);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (2а) калибровка порога межсессионно на FERET (XBaseEval feret-calib): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        r.append(XBaseFeret.EQ_NOTE);
        r.append("Способы (FAR 1 %): К0 – П2 по «чужим» галереи (как сейчас); К0' – П3; К1 – П2 по чужим К1 – всем снимкам калибровочных людей (не в\n"
                + "галерее, не в обучении проекции), дата которых не совпадает ни с одной датой галереи своих прохода; К2 – П3 по К1 (при К1 < " + K2_MIN
                + " на проход –\nвырожден: верхняя граница ≤ 1 % недостижима); К3 – П2 по объединению. Модель и пробы прежние; К1 – дополнительные пробы основной\n"
                + "модели. Деление пула – java.util.Random(faces.seed + " + CAL_SEED_OFFSET + "). Оговорка: пул проекции вдвое меньше, чем в №2 – L1-xs-F здесь\n"
                + "слабее; меряется калибровка, а не метод.\n");
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder lines = new StringBuilder();
        double[] fitS = {0};
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String> bases = new ArrayList<>();
        Map<String, Integer> minCal = new HashMap<>();
        StringBuilder vol = new StringBuilder("\n=== Объём: чужие К1 и пул по проходам ===\nсценарий | проход | своих | пул проекции | калибровочных людей | чужих К1 (снимков)\n");

        // ---- Сценарий 1: половины №2.
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons1 = new ArrayList<>(ps0);
        XBaseEval.Gt base1 = XBaseFeret.gt(tpl, persons1, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        List<XBaseXsF.Conf> confs = List.of(new XBaseXsF.Conf("межс.", 0, ro.own1, ro.gal1, ro.pro1), new XBaseXsF.Conf("внутри", 1, ro.own1in, ro.gal1in,
                ro.pro1in), new XBaseXsF.Conf("G1", 2, ro.own2, ro.g1, ro.pro2), new XBaseXsF.Conf("G2'", 5, ro.own2, ro.g2p, ro.pro2));
        for (XBaseXsF.Conf c : confs) {
            for (Meth mt : FERET_METHS) {
                List<Pass> passes = new ArrayList<>();
                int minK = Integer.MAX_VALUE;
                for (int pp = 0; pp < 2; pp++) {
                    Set<String> mine = halves.get(pp);
                    List<List<String>> bs = split(halves.get(1 - pp), tpl.seed);
                    Map<String, List<List<String>>> projPool = new TreeMap<>();
                    for (String p : bs.get(0)) projPool.put(p, pool.get(p));
                    List<String> own = c.own().stream().filter(mine::contains).toList();
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Map<String, List<String>> pro = new LinkedHashMap<>();
                    Set<String> galDates = new HashSet<>();
                    for (String p : own) {
                        gal.put(p, c.gal().get(p));
                        pro.put(p, c.pro().get(p));
                        for (String k : c.gal().get(p)) galDates.add(su.byKey.get(k).date());
                    }
                    List<String> calKeys = new ArrayList<>();
                    for (String p : bs.get(1)) for (List<String> d : pool.get(p)) for (String k : d) if (!galDates.contains(su.byKey.get(k).date())) calKeys.add(k);
                    minK = Math.min(minK, calKeys.size());
                    XBaseEval.Gt g = XBaseFeret.withOwn(base1, own);
                    List<List<String>> aug = new ArrayList<>(cohort);
                    if (mt.pool()) aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                    g.cohort = aug;
                    List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                    passes.add(pass(g, xs[mt.feat()], gal, pro, parts, c.s(), calKeys, personOf, cp, String.format(Locale.ROOT, "s1_%d_%s_p%d", c.s(),
                            mt.label(), pp), lines, fitS));
                    if (mt == FERET_METHS.get(0)) vol.append(String.format(Locale.ROOT, "1 %s | %d | %d | %d | %d | %d%n", c.name(), pp + 1, own.size(),
                            projPool.size(), bs.get(1).size(), calKeys.size()));
                }
                String b = "1 " + c.name() + " " + mt.label();
                bases.add(b);
                minCal.put(b, minK);
                results(res, b, passes, persons1.size(), tpl);
            }
        }

        // ---- Сценарий 2: смешанная галерея D = 2, строки (а) и (в2).
        int dd = 2;
        List<String> elig = new ArrayList<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= dd + 1) elig.add(e.getKey());
        Set<String> two = new TreeSet<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (!elig.contains(e.getKey()) && e.getValue().size() == 2) two.add(e.getKey());
        List<List<String>> bs2 = split(two, tpl.seed);
        Map<String, List<List<String>>> projPool2 = new TreeMap<>();
        for (String p : bs2.get(0)) projPool2.put(p, pool.get(p));
        List<String> permE = new ArrayList<>(elig);
        Collections.shuffle(permE, new Random(tpl.seed + XBaseSynth.SPLIT_SEED_OFFSET));
        int hE = (permE.size() + 1) / 2;
        List<Set<String>> halvesE = List.of(new TreeSet<>(permE.subList(0, hE)), new TreeSet<>(permE.subList(hE, permE.size())));
        TreeSet<String> ps2 = new TreeSet<>(elig);
        ps2.addAll(ro.ctrl);
        ps2.addAll(bs2.get(1));
        List<String> persons2 = new ArrayList<>(ps2);
        XBaseEval.Gt base2 = XBaseFeret.gt(tpl, persons2, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        List<String> own2 = new ArrayList<>(new TreeSet<>(elig));
        Map<String, List<String>> pro2 = new LinkedHashMap<>();
        Map<String, List<String>> oldGal = new LinkedHashMap<>();
        for (String p : own2) {
            List<List<String>> ds = pool.get(p);
            List<String> gl = new ArrayList<>();
            for (int j = 0; j < dd; j++) gl.addAll(ds.get(j));
            oldGal.put(p, XBaseFeret.sorted(gl));
            List<String> pr = new ArrayList<>();
            for (int j = dd; j < ds.size(); j++) pr.addAll(ds.get(j));
            pro2.put(p, XBaseFeret.sorted(pr));
        }
        Map<String, List<String>> synth2 = synth2D(su, tpl, pool, ro, own2, pro2, dd, xs);
        r.append(String.format(Locale.ROOT, "Сценарий 2: подходящих %d, пул проекции %d, калибровочных %d (люди ровно с 2 датами); синтез (в2) – как feret-synth2 "
                + "(seed + 9940 + 2).%n", elig.size(), projPool2.size(), bs2.get(1).size()));
        for (Meth mt : POOL_METHS) {
            for (String row : new String[] {"(а) реальные", "(в2) + синтез, этап 2"}) {
                List<Pass> passes = new ArrayList<>();
                List<Pass> newPasses = new ArrayList<>();
                int minK = Integer.MAX_VALUE;
                for (int pp = 0; pp < 2; pp++) {
                    Set<String> newbies = halvesE.get(pp);
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Set<String> galDates = new HashSet<>();
                    for (String p : own2) {
                        List<String> gl = newbies.contains(p) ? new ArrayList<>(pool.get(p).get(0)) : new ArrayList<>(oldGal.get(p));
                        if (newbies.contains(p) && row.startsWith("(в2)")) gl.addAll(synth2.get(p));
                        gal.put(p, gl);
                        for (String k : gl) galDates.add(su.byKey.get(k.contains("#") ? k.substring(0, k.indexOf('#')) : k).date());
                    }
                    List<String> calKeys = new ArrayList<>();
                    for (String p : bs2.get(1)) for (List<String> d : pool.get(p)) for (String k : d) if (!galDates.contains(su.byKey.get(k).date())) calKeys.add(k);
                    minK = Math.min(minK, calKeys.size());
                    XBaseEval.Gt g = XBaseFeret.withOwn(base2, own2);
                    List<List<String>> aug = new ArrayList<>(cohort);
                    if (mt.pool()) aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool2, Set.of(), Map.of()).extra());
                    g.cohort = aug;
                    List<List<String>> parts = XBaseOther.cvParts(own2, tpl.seed, XBaseEval.CV_S + 30 + dd);
                    Pass ps = pass(g, xs[mt.feat()], gal, pro2, parts, 30 + dd, calKeys, personOf, cp, String.format(Locale.ROOT, "s2_%s_%s_p%d", mt.label(),
                            row.substring(0, 4), pp), lines, fitS);
                    passes.add(ps);
                    // Новички прохода – часть проб своих (проверка: точки как в feret-synth2 у L1 без пула).
                    List<Integer> ni = new ArrayList<>();
                    int idx = 0;
                    for (Probe q : XBaseEval.probes(g, pro2)) {
                        if (!q.role().equals(FeiScores.OWN)) continue;
                        if (newbies.contains(q.person())) ni.add(idx);
                        idx++;
                    }
                    newPasses.add(new Pass(XBaseFeret.subset(ps.cfg(), ni), ps.cv(), ps.cal()));
                    if (mt == POOL_METHS.get(0) && row.startsWith("(а)")) vol.append(String.format(Locale.ROOT, "2 D = 2 | %d | %d | %d | %d | %d%n", pp + 1,
                            own2.size(), projPool2.size(), bs2.get(1).size(), calKeys.size()));
                }
                String b = "2 " + mt.label() + " " + row;
                bases.add(b);
                minCal.put(b, minK);
                results(res, b, passes, persons2.size(), tpl);
                String bn = "2 " + mt.label() + " новички " + row;
                bases.add(bn);
                minCal.put(bn, minK);
                results(res, bn, newPasses, persons2.size(), tpl);
            }
        }
        r.append(vol);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS[0], tpl.threads));
        summary(r, res, bases, tpl.b, minCal);
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons1.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, FeiThreshold.Res> r1 = new LinkedHashMap<>();
        Map<String, FeiThreshold.Res> r2 = new LinkedHashMap<>();
        res.forEach((k, v) -> (k.startsWith("1 ") ? r1 : r2).put(k, v));
        for (Object[] grp : new Object[][] {{"Сценарий 1 – половины №2", r1, persons1.size()}, {"Сценарий 2 – смешанная галерея D = 2", r2, persons2.size()}}) {
            @SuppressWarnings("unchecked")
            Map<String, FeiThreshold.Res> m = (Map<String, FeiThreshold.Res>) grp[1];
            int n = (Integer) grp[2];
            int[][] ww = tpl.b > 0 ? FeiBootstrap.weights(n, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : w;
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            for (Map.Entry<String, FeiThreshold.Res> e : m.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(n), ww));
            r.append("\n\n######## ").append(grp[0]).append(" ########\n");
            XBaseOther.tables(r, m, sep, tpl.b, List.of());
        }
        XBaseOther.writeScores(out.resolve("xbase_feret_calib_scores.tsv.gz"), "# FERET (2а) калибровка, XBaseCalib feret-calib, только локально; роль cal – чужие К1\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_calib.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /**
     * Синтез этапа 2 для D (как XBaseSynth.run2: диапазоны — люди вне подходящих и вне контроля без дат проб, при < 100
     * размеченных — выравнивание без исключения дат; java.util.Random(faces.seed + 9940 + D), основа — fa, fb даты 1).
     * Векторы синтеза добавляются в xs.
     */
    static Map<String, List<String>> synth2D(XBaseA0.FeretSetup su, XBaseEval.Gt tpl, Map<String, List<List<String>>> pool, XBaseFeret.Roles ro,
                                             List<String> own, Map<String, List<String>> pro, int dd, XBaseEval[] xs) throws IOException {
        Set<String> probeDates = new TreeSet<>();
        for (String p : own) for (String k : pro.get(p)) probeDates.add(su.byKey.get(k).date());
        Set<String> exclP = new HashSet<>(own);
        exclP.addAll(ro.ctrl);
        Map<String, double[]> cache = new HashMap<>();
        List<Double> noise = new ArrayList<>();
        List<Double> asym = new ArrayList<>();
        List<double[]> align = new ArrayList<>();
        for (Img im : su.imgs) {
            double[] row = su.det.get(im.key());
            if (row == null || su.rOf.get(im.key()) > tpl.rLimit) continue;
            if (exclP.contains(im.person()) || probeDates.contains(im.date())) continue;
            double[] m = XBaseA0Check.measures(su.src, im.key(), cache);
            asym.add(m[0]);
            noise.add(m[1]);
            double[][] man = FeretData.manual4(im);
            if (man != null) align.add(FaceSynth.alignDelta(FeretData.yunet4(row), man));
        }
        if (align.size() < XBaseSynth.RANGE_MIN) {
            align.clear();
            for (Img im : su.imgs) {
                double[] row = su.det.get(im.key());
                if (row == null || su.rOf.get(im.key()) > tpl.rLimit || exclP.contains(im.person())) continue;
                double[][] man = FeretData.manual4(im);
                if (man != null) align.add(FaceSynth.alignDelta(FeretData.yunet4(row), man));
            }
        }
        FaceSynth.Ranges ranges = new FaceSynth.Ranges(XBaseSynth.sortedArr(noise), XBaseSynth.sortedArr(asym), align);
        int extraN = 2 * dd - 2;
        Random rnd = new Random(tpl.seed + XBaseSynth.SYNTH2_SEED_OFFSET + dd);
        Map<String, List<String>> out = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String p : own) {
            List<String> real = pool.get(p).get(0);
            List<String> ks = new ArrayList<>();
            for (int i = 0; i < extraN; i++) {
                String b = real.get(i % real.size());
                Mat color = FeretData.decode(su.byKey.get(b).file());
                FaceSynth.Params prm = FaceSynth.sample(ranges, rnd);
                Mat a = FaceSynth.synth(color, su.det.get(b), prm, rnd);
                color.release();
                String key = b + "#t" + dd + "_" + i;
                sf.put(key, a);
                ks.add(key);
                su.rOf.put(key, su.rOf.get(b));
                su.imageNo.put(key, 3000 + 100 * dd + 10 * su.imageNo.get(b) + i);
            }
            out.put(p, ks);
        }
        XBaseEval h0 = new XBaseEval();
        h0.vectors(sf, EnumSet.of(V.L1), tpl.params);
        XBaseEval h1 = new XBaseEval();
        h1.vectors(sf, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : sf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(h0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(h0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(h1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(h1.mir.get(V.L1));
        return out;
    }

    // ---------------------------------------------------------------- gt-calib

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-calib: только равные галереи FERET (faces.feret.gallery = eq)");
        String gtDir = settings.loadFacesGtDir();
        if (gtDir == null) throw new IllegalStateException("gt-calib: задайте faces.gt.dir / FACES_GT_DIR (исходные снимки gt_db)");
        Path gtDb = Path.of(gtDir);
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> fs = split(pool.keySet(), g.seed);
        Map<String, List<List<String>>> projPool = new TreeMap<>();
        for (String p : fs.get(0)) projPool.put(p, pool.get(p));
        Map<String, String> calPerson = new HashMap<>();
        List<String> calKeys = new ArrayList<>();
        for (String p : fs.get(1)) for (List<String> d : pool.get(p)) for (String k : d) {
            calKeys.add(k);
            calPerson.put(k, p);
        }
        // Люди бутстрепа: GT и калибровочные FERET.
        List<String> persons = new ArrayList<>(g.persons);
        for (String p : fs.get(1)) {
            g.pIdx.put(p, persons.size());
            persons.add(p);
        }
        g.persons = persons;
        Map<String, Integer> imageNo = new HashMap<>();
        for (Map.Entry<String, Sample> e : g.samples.entrySet()) imageNo.put(e.getKey(), XBaseEval.image(e.getValue()));
        for (String k : calKeys) {
            imageNo.put(k, su.imageNo.get(k));
            g.rOf.put(k, su.rOf.get(k));
        }
        g.imageNo = imageNo;

        // Векторы: GT (none – g.x, tt – заново), пул FERET (проекция и калибровка).
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        Map<String, double[]> meas = new HashMap<>();
        for (Map.Entry<String, Mat> e : fr.entrySet()) if (g.samples.containsKey(e.getKey())) meas.put(e.getKey(), XBaseSynth.measure(e.getValue()));
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        xs[1].vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        Map<String, Mat> pf = new LinkedHashMap<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) pf.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval p0 = new XBaseEval();
        p0.vectors(pf, EnumSet.of(V.L1), g.params);
        XBaseEval p1 = new XBaseEval();
        p1.vectors(pf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : pf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(p0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(p0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(p1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(p1.mir.get(V.L1));

        // Синтез (в) – как gt-synth (тот же порядок случайных чисел; ключи #t).
        Map<String, Map<String, List<String>>> synthOf = gtSynth(g, su, dir, gtDb, meas, imageNo, xs);
        List<List<String>> projCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra();
        List<Map<String, List<String>>> split = XBaseEval.within(g);
        Object[][] modes = {{"межс.", g.galleryKeys, g.probeKeys}, {"внутри", split.get(0), split.get(1)}};
        List<List<String>> cvParts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-calib; seed %d; r0 %s; пул FERET %d; калибровка %d", g.seed, g.r0, pool.size(),
                fs.get(1).size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_calib"), fp);
        StringBuilder lines = new StringBuilder();
        double[] fitS = {0};
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String> bases = new ArrayList<>();
        List<List<String>> cohort0 = g.cohort;
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal0 = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                for (Meth mt : POOL_METHS) {
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    if (mt.pool()) aug.addAll(projCls);
                    for (String row : new String[] {"(а) реальные", "(в) + синтез"}) {
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        for (String p : g.own) {
                            List<String> l = new ArrayList<>(gal0.get(p));
                            if (row.startsWith("(в)")) l.addAll(synthOf.get(mode).get(p));
                            gal.put(p, l);
                        }
                        g.cohort = aug;
                        Pass ps = pass(g, xs[mt.feat()], gal, pro, cvParts, 0, calKeys, calPerson, cp, String.format(Locale.ROOT, "%s_%s_%s",
                                mode.equals("межс.") ? "x" : "i", mt.label(), row.substring(0, 3)), lines, fitS);
                        String b = "3 " + mode + " " + mt.label() + " " + row;
                        bases.add(b);
                        results(res, b, List.of(ps), persons.size(), g);
                    }
                }
            }
        } finally {
            g.cohort = cohort0;
        }
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, (2а) калибровка порога межсессионно на GT – межбазовая (XBaseEval gt-calib)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; исходные снимки – ").append(gtDb)
                .append("; пул проекции и чужие К1 – Color FERET (имён снимков FERET в отчёте нет; в файле оценок – только роль cal)\n");
        t.append(String.format(Locale.ROOT, "Пул FERET %d людей → проекция %d, калибровка %d (java.util.Random(faces.seed + %d)); чужие К1 – все снимки калибровочных "
                + "FERET (даты FERET с GT не совпадают): %d. Межбазовая калибровка: К1 – люди другой базы.%n", pool.size(), fs.get(0).size(), fs.get(1).size(),
                CAL_SEED_OFFSET, calKeys.size()));
        t.append("Способы – как в feret-calib (К0, К0', К1, К2, К3); строки (а) – как gt-xs-f, (в) – как gt-synth (синтез тот же), но пул проекции вдвое\n"
                + "меньше – L1-xs-F здесь слабее; меряется калибровка, а не метод. L1 (без пула) – как gt и gt-synth.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT и калибровочным FERET (%d): B = %d, seed = %d.%n", persons.size(), g.b,
                g.seed + FeiBootstrap.SEED_OFFSET));
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки), потоков %d.%n", fitS[0], g.threads));
        summary(t, res, bases, g.b, Map.of());
        int[][] w = g.b > 0 ? FeiBootstrap.weights(persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(persons.size()), w));
        XBaseOther.tables(t, res, sep, g.b, List.of());
        StringBuilder safe = new StringBuilder();
        Set<String> calSet = new HashSet<>(calPerson.values());
        for (String l : lines.toString().split("\n")) {
            if (l.startsWith("#")) {
                safe.append(l).append('\n');
                continue;
            }
            String[] f = l.split("\t");
            if (f.length > 4 && f[3].equals(CAL_ROLE)) continue;
            safe.append(l).append('\n');
        }
        XBaseOther.writeScores(out.resolve("xbase_gt_calib_scores.tsv.gz"), "# GT (2а) калибровка, XBaseCalib gt-calib; строки чужих К1 (FERET) не записываются\n# Код: "
                + "коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", safe.toString());
        FarMethods.write(out.resolve("xbase_gt_calib.txt"), t);
        System.out.print(t);
        cp.delete();
        if (calSet.isEmpty()) throw new IllegalStateException("gt-calib: нет калибровочных людей");
    }

    /** Синтез (в) gt-synth: тот же порядок, что XBaseSynth.gt (части seed + 9950, java.util.Random(seed + 9960)); векторы — в xs. */
    static Map<String, Map<String, List<String>>> gtSynth(XBaseEval.Gt g, XBaseA0.FeretSetup su, Path dir, Path gtDb, Map<String, double[]> meas,
                                                         Map<String, Integer> imageNo, XBaseEval[] xs) throws IOException {
        Map<String, double[]> rowsGt = XBaseSynth.gtRows(dir);
        Map<String, String> dateGt = XBaseSynth.gtDates(dir);
        List<double[]> align = new ArrayList<>();
        for (Img im : su.imgs) {
            double[][] man = FeretData.manual4(im);
            double[] row = su.det.get(im.key());
            if (man == null || row == null || su.rOf.get(im.key()) > su.tpl.rLimit) continue;
            align.add(FaceSynth.alignDelta(FeretData.yunet4(row), man));
        }
        List<List<String>> parts = XBaseOther.cvParts(g.own, g.seed, XBaseSynth.GT_PART_S);
        List<Map<String, List<String>>> split = XBaseEval.within(g);
        Object[][] modes = {{"межс.", g.galleryKeys, g.probeKeys}, {"внутри", split.get(0), split.get(1)}};
        Random rnd = new Random(g.seed + XBaseSynth.GT_SYNTH_SEED_OFFSET);
        Map<String, Map<String, List<String>>> out = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (Object[] md : modes) {
            String mode = (String) md[0];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
            Map<String, List<String>> syn = new LinkedHashMap<>();
            for (int q = 0; q < parts.size(); q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                Set<String> probeDates = new HashSet<>();
                for (String p : parts.get(q)) for (String k : pro.get(p)) probeDates.add(dateGt.get(p + "/" + XBaseEval.image(g.samples.get(k))));
                List<Double> noise = new ArrayList<>();
                List<Double> asym = new ArrayList<>();
                for (Map.Entry<String, double[]> e : meas.entrySet()) {
                    Sample s = g.samples.get(e.getKey());
                    if (part.contains(s.person())) continue;
                    String d = dateGt.get(s.person() + "/" + XBaseEval.image(s));
                    if (d != null && probeDates.contains(d)) continue;
                    asym.add(e.getValue()[0]);
                    noise.add(e.getValue()[1]);
                }
                FaceSynth.Ranges rg = new FaceSynth.Ranges(XBaseSynth.sortedArr(noise), XBaseSynth.sortedArr(asym), align);
                for (String p : parts.get(q)) {
                    List<String> real = gal.get(p);
                    List<String> ks = new ArrayList<>();
                    for (int i = 0; i < XBaseSynth.GT_EXTRA; i++) {
                        String b = real.get(i % real.size());
                        Sample s = g.samples.get(b);
                        int no = XBaseEval.image(s);
                        double[] row = rowsGt.get(s.person() + "/" + no);
                        Mat color = org.opencv.imgcodecs.Imgcodecs.imread(gtDb.resolve(s.person()).resolve(String.format(Locale.ROOT, "%02d.jpg", no)).toString());
                        if (color.empty()) throw new IOException("gt-calib: не читается исходный снимок " + s.person() + " №" + no);
                        FaceSynth.Params prm = FaceSynth.sample(rg, rnd);
                        Mat a = FaceSynth.synth(color, row, prm, rnd);
                        color.release();
                        String key = b + "#t" + (mode.equals("межс.") ? "x" : "i") + i;
                        sf.put(key, a);
                        ks.add(key);
                        g.rOf.put(key, g.rOf.get(b));
                        imageNo.put(key, 1000 + 10 * no + i + (mode.equals("межс.") ? 0 : 500));
                    }
                    syn.put(p, ks);
                }
            }
            out.put(mode, syn);
        }
        XBaseEval h0 = new XBaseEval();
        h0.vectors(sf, EnumSet.of(V.L1), g.params);
        XBaseEval h1 = new XBaseEval();
        h1.vectors(sf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : sf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(h0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(h0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(h1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(h1.mir.get(V.L1));
        return out;
    }
}
