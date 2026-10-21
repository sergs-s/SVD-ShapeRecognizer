package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseNewcomer.SD;
import svd.recognizer.faces.XBaseUpdate.Ctx;
import svd.recognizer.faces.XBaseUpdate.Proj;
import svd.recognizer.faces.XBaseUpdate.Step;
import svd.recognizer.faces.XBaseUpdate.Var;
import svd.recognizer.storage.SettingsStore;

/**
 * Все свои в одном статусе относительно W (PLAN.md п. 5.5, очередь (4), часть 3; задание Жорика №8 от 15.10.2026, план с
 * поправками №1 от 16.10.2026). Схемы проекции MLDA: (Wа) – W на всех людях галереи (их стартовые сессии) + когорта + пул; (Wб)
 * – W только на пуле и когорте, своих в обучении нет, центры – проекции снимков шаблона; (Wа′) – (Wа), переобученная на всех
 * текущих сессиях шаблонов перед шагами 2 и 3 (часть 1). Режимы: feret-wstatus – п. 2 (FERET eq межсессионно, роль «межс.»
 * feret-xs2, половины №2), п. 3 (новички (н1) и (н0) как в №7), п. 4 и 4а (часть 1: (а), (б), (д)); gt-wstatus – п. 2 на GT
 * (межсессионно, правило №8, N = 6, пул FERET). Отчёты FERET – только D:\data\Feret\reports\eq\.
 *
 * @author ssv
 */
final class XBaseWStatus {

    private XBaseWStatus() {
    }

    static final int RETRAIN_LAST = 3;

    /** Пробный режим (отладка): FACES_WSTATUS_SMOKE=1 – только L1 (в п. 4 – и (Wб)), r = 0; отчёт и контрольная точка *_smoke. */
    static final boolean SMOKE = "1".equals(System.getenv("FACES_WSTATUS_SMOKE"));

    static List<Proj> projs() {
        return SMOKE ? List.of(XBaseUpdate.PROJS.get(2)) : XBaseUpdate.PROJS;
    }

    static String[][] proj2() {
        return SMOKE ? new String[][] {PROJ2[2]} : PROJ2;
    }

    // ---------------------------------------------------------------- п. 2: протокол межсессионно, (Wа) / (Wб)

    /**
     * Строки оценок (как XBaseEval.fitLines для V.L1): основная модель и модели без части q, оба зеркала. wb – W только на
     * когорте g.cohort (пул и когорта; свои вне обучения): модели частей – та же W, центры по галерее без части. common (при
     * (Wа)) – «чужие» П2/П3 по основной модели (центры людей части исключены).
     */
    static String fitLines(XBaseEval.Gt g, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, int s,
                           boolean wb, boolean common) {
        List<String> own = g.own;
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        List<Callable<String>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> {
                LdaModel full = wb ? (LdaModel) XBaseXsF.fit(x, m, g.cohort, fullGallery) : (LdaModel) x.fit(V.L1, m, fullGallery, g.cohort);
                StringBuilder t = new StringBuilder(x.lines(method, s, 0, full, V.L1, own, mainProbes));
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    List<String> rest = own.stream().filter(p -> !part.contains(p)).toList();
                    List<List<String>> gk = rest.stream().map(galleryKeys::get).toList();
                    List<Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, g.img(k), k, -1));
                    GalleryScorer.ScoreModel mq;
                    if (wb || common) {
                        // Та же W, центры – по галерее без части (как LdaModel.of).
                        List<List<double[]>> cls = new ArrayList<>();
                        for (List<String> keys : gk) cls.add(XBaseXsF.vectors(x, keys, m));
                        mq = LdaModel.of(cls, gk.size(), full.mean(), full.projection(), full.info());
                    } else {
                        mq = x.fit(V.L1, m, gk, g.cohort);
                    }
                    t.append(x.lines(method, s, 0, mq, V.L1, rest, pr));
                }
                return t.toString();
            });
        }
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, g.threads).forEach(all::append);
        return all.toString();
    }

    /** Хабы у контроля по строкам: доля первого хаба, двух самых частых, анти-хабов среди своих. */
    static int[] hubs(XBaseEval.Gt g, List<Probe> probes, String lines) {
        Map<String, Row> rows = new HashMap<>();
        for (String l : lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        Map<String, Integer> cnt = new HashMap<>();
        int n = 0;
        for (Probe q : probes) {
            if (q.role().equals(FeiScores.OWN)) continue;
            Row r = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), g.rOf.get(q.key()) <= g.r0);
            cnt.merge(r.bestId(), 1, Integer::sum);
            n++;
        }
        List<Integer> v = new ArrayList<>(cnt.values());
        v.sort(Collections.reverseOrder());
        int anti = 0;
        for (String p : g.own) if (!cnt.containsKey(p)) anti++;
        return new int[] {n, v.isEmpty() ? 0 : v.get(0), v.size() > 1 ? v.get(0) + v.get(1) : v.isEmpty() ? 0 : v.get(0), anti, g.own.size()};
    }

    static String hubCell(int[] h) {
        return String.format(Locale.ROOT, "%.1f %% | %.1f %% | %d из %d", 100.0 * h[1] / h[0], 100.0 * h[2] / h[0], h[3], h[4]);
    }

    /** Строки блока, отсортированные (сверка с прежним файлом оценок). */
    static List<String> sortedRows(String lines) {
        List<String> out = new ArrayList<>();
        for (String l : lines.split("\n")) if (!l.isEmpty()) out.add(l);
        Collections.sort(out);
        return out;
    }

    /** Строки «чужих» (роль cv*) блока. */
    static List<String> cvRows(String lines) {
        return sortedRows(lines).stream().filter(l -> l.split("\t", -1)[3].startsWith(FeiMethods.CV_ROLE)).toList();
    }

    /** Основные пробы (свои и контроль) блока. */
    static String mainRows(String lines) {
        StringBuilder t = new StringBuilder();
        for (String l : lines.split("\n")) if (!l.isEmpty() && !l.split("\t", -1)[3].startsWith(FeiMethods.CV_ROLE)) t.append(l).append('\n');
        return t.toString();
    }

    /** Итоги строк схем одного протокола: Res, хабы. */
    static final class Acc {
        final Map<String, List<FeiDecompose.Cfg>> cfgs = new LinkedHashMap<>();
        final Map<String, List<FeiThreshold.Imp>> imps = new LinkedHashMap<>();
        final Map<String, int[]> hub = new LinkedHashMap<>();

        void add(String label, XBaseEval.Gt g, Map<String, List<String>> gal, List<Probe> probes, List<List<String>> parts, String lines) {
            XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, lines);
            cfgs.computeIfAbsent(label, k -> new ArrayList<>()).add(p.cfg.get(V.L1));
            imps.computeIfAbsent(label, k -> new ArrayList<>()).add(p.imp.get(V.L1));
            int[] h = hubs(g, probes, lines);
            int[] a = hub.computeIfAbsent(label, k -> new int[5]);
            for (int i = 0; i < 5; i++) a[i] += h[i];
        }

        String tables(String title, int persons, XBaseEval.Gt tpl, List<String[]> pairs) {
            StringBuilder t = new StringBuilder("\n\n######## " + title + " ########\n");
            t.append("\n=== Хабы у контрольных чужих (оба прохода / все конфигурации вместе) ===\nстрока | доля первого хаба | два самых частых | "
                    + "анти-хабов среди своих\n");
            for (Map.Entry<String, int[]> e : hub.entrySet()) t.append(e.getKey()).append(" | ").append(hubCell(e.getValue())).append('\n');
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            for (String l : cfgs.keySet()) res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), cfgs.get(l), imps.get(l), persons, tpl.b, tpl.seed
                    + FeiBootstrap.SEED_OFFSET));
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(persons), w));
            pairs.removeIf(p -> !res.containsKey(p[0]) || !res.containsKey(p[1]));
            XBaseOther.tables(t, res, sep, tpl.b, pairs);
            return t.toString();
        }
    }

    static final String[][] PROJ2 = {{"L1-xs-F", "0", "1"}, {"L1-tt-xs-F", "1", "1"}, {"L1", "0", "0"}};

    /** П. 2, FERET: роль «межс.» feret-xs2 (половины №2), (Wа) – проверка против файла feret-xs2 побайтно. */
    static String feret2(Setup s, XBaseOther.Checkpoint cp) throws IOException {
        XBaseFeret.Roles ro = XBaseFeret.equalize(s.su.ro, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        XBaseEval.Gt base = XBaseFeret.gt(s.tpl, new ArrayList<>(ps0), ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, String> xs2 = XBaseSubsp.blocks(s.out.resolve("xbase_feret_xs2_scores.tsv.gz"));
        StringBuilder check = new StringBuilder("\n=== П. 2, проверка: (Wа) = строки feret-xs2, роль «межс.» (побайтно); (Wб) – модели частей = общая W ===\n");
        Acc acc = new Acc();
        List<String[]> pairs = new ArrayList<>();
        for (String[] pj : proj2()) {
            int f = Integer.parseInt(pj[1]);
            boolean usePool = pj[2].equals("1");
            for (int pp = 0; pp < 2; pp++) {
                Set<String> mine = s.halves.get(pp);
                List<String> own = ro.own1.stream().filter(mine::contains).toList();
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    gal.put(p, ro.gal1.get(p));
                    pro.put(p, ro.pro1.get(p));
                }
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                if (usePool) {
                    Map<String, List<List<String>>> pp2 = new TreeMap<>();
                    for (String p : s.halves.get(1 - pp)) pp2.put(p, s.pool.get(p));
                    aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp2, Set.of(), Map.of()).extra());
                }
                g.cohort = aug;
                List<Probe> probes = XBaseEval.probes(g, pro);
                List<List<String>> parts = XBaseOther.cvParts(own, s.tpl.seed, XBaseEval.CV_S);
                for (String sch : usePool ? new String[] {"Wa", "WaC", "Wb"} : new String[] {"Wa", "WaC"}) {
                    String name = String.format(Locale.ROOT, "f2_%s_p%d_%s", pj[0], pp, sch);
                    String ls = cp.read(name);
                    if (ls == null) {
                        ls = fitLines(g, s.xs[f], gal, probes, parts, 0, sch.equals("Wb"), sch.equals("WaC"));
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "feret-wstatus: п. 2 %s проход %d %s готово%n", pj[0], pp + 1, sch);
                    }
                    if (sch.equals("Wa")) {
                        String bn = "проход " + (pp + 1) + ", конфигурация межс., строка " + pj[0];
                        String old = xs2.get(bn);
                        if (old == null) throw new IllegalStateException("feret-wstatus: нет блока «" + bn + "» в xbase_feret_xs2_scores.tsv.gz");
                        boolean same = sortedRows(old).equals(sortedRows(ls));
                        check.append(String.format(Locale.ROOT, "(Wа) %s проход %d: %s (%d строк)%n", pj[0], pp + 1, same ? "совпало" : "НЕ СОВПАЛО", sortedRows(ls).size()));
                        if (!same) throw new IllegalStateException("feret-wstatus: (Wа) не совпало с feret-xs2: " + bn);
                    }
                    if (sch.equals("Wb")) {
                        // Модели частей при (Wб) – та же W: «чужие» по общей W те же (по построению – центры без части).
                        check.append(String.format(Locale.ROOT, "(Wб) %s проход %d: модели частей = общая W по построению (одна W, центры без части)%n",
                                pj[0], pp + 1));
                    }
                    String lab = label(pj[0], sch);
                    String lines = sch.equals("WaC") ? mainRows(cp.read(String.format(Locale.ROOT, "f2_%s_p%d_Wa", pj[0], pp))) + cvOnly(ls) : ls;
                    acc.add(lab, g, gal, probes, parts, lines);
                }
            }
            if (usePool) pairs.add(new String[] {label(pj[0], "Wb"), label(pj[0], "Wa")});
            pairs.add(new String[] {label(pj[0], "WaC"), label(pj[0], "Wa")});
        }
        return check + acc.tables("П. 2, FERET eq межсессионно (роль «межс.» feret-xs2, половины №2)", base.persons.size(), s.tpl, pairs);
    }

    static String cvOnly(String lines) {
        StringBuilder t = new StringBuilder();
        for (String l : lines.split("\n")) if (!l.isEmpty() && l.split("\t", -1)[3].startsWith(FeiMethods.CV_ROLE)) t.append(l).append('\n');
        return t.toString();
    }

    static String label(String proj, String sch) {
        return proj + " " + switch (sch) {
            case "Wa" -> "(Wа)";
            case "WaC" -> "(Wа), П2/П3 по общей W";
            default -> "(Wб)";
        };
    }

    // ---------------------------------------------------------------- п. 2 на GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-wstatus: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString())).toList(),
                null);
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
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, g.seed + XBaseEval.EQ_SEED_OFFSET);
        List<Probe> probes = XBaseEval.probes(g, g.probeKeys);
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Map<String, String> old = XBaseSubsp.blocks(out.resolve("xbase_gt_subsp_scores.tsv.gz"));
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-wstatus; seed %d; r0 %s; пул FERET %d", g.seed, g.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_wstatus"), fp);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4), часть 3: все свои в одном статусе относительно W – GT межсессионно (XBaseEval gt-wstatus)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F – Color "
                + "FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте нет)\n");
        t.append("Протокол gt, межсессионно, галерея по правилу №8 (N = 6, seed faces.seed + 9800). (Wа) – W на своих + когорта + пул (как gt-subsp k = 0);\n"
                + "(Wб) – W только на когорте и пуле, центры своих – проекции снимков галереи; «П2/П3 по общей W» – «чужие» по основной модели (центры\n"
                + "людей части исключены). L1 – только (Wа).\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: (Wа) = строки gt-subsp «межс., k = 0» (побайтно, по отсортированным строкам) ===\n");
        Acc acc = new Acc();
        List<String[]> pairs = new ArrayList<>();
        List<List<String>> cohort0 = g.cohort;
        try {
            for (String[] pj : PROJ2) {
                int f = Integer.parseInt(pj[1]);
                boolean usePool = pj[2].equals("1");
                List<List<String>> aug = new ArrayList<>(cohort0);
                if (usePool) aug.addAll(poolCls);
                g.cohort = aug;
                for (String sch : usePool ? new String[] {"Wa", "WaC", "Wb"} : new String[] {"Wa", "WaC"}) {
                    String name = "gt_" + pj[0] + "_" + sch;
                    String ls = cp.read(name);
                    if (ls == null) {
                        ls = fitLines(g, xs[f], galX, probes, parts, 0, sch.equals("Wb"), sch.equals("WaC"));
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "gt-wstatus: %s %s готово%n", pj[0], sch);
                    }
                    if (sch.equals("Wa")) {
                        String o = old.get("межс., проекция " + pj[0] + ", k = 0");
                        if (o == null) {
                            check.append(pj[0]).append(": блока gt-subsp нет – проверка не делалась\n");
                        } else {
                            boolean same = sortedRows(o).equals(sortedRows(ls));
                            check.append(String.format(Locale.ROOT, "(Wа) %s: %s (%d строк)%n", pj[0], same ? "совпало" : "НЕ СОВПАЛО", sortedRows(ls).size()));
                            if (!same) throw new IllegalStateException("gt-wstatus: (Wа) не совпало с gt-subsp: " + pj[0]);
                        }
                    }
                    String lines = sch.equals("WaC") ? mainRows(cp.read("gt_" + pj[0] + "_Wa")) + cvOnly(ls) : ls;
                    acc.add(label(pj[0], sch), g, galX, probes, parts, lines);
                }
                if (usePool) pairs.add(new String[] {label(pj[0], "Wb"), label(pj[0], "Wa")});
                pairs.add(new String[] {label(pj[0], "WaC"), label(pj[0], "Wa")});
            }
        } finally {
            g.cohort = cohort0;
        }
        t.append(check);
        t.append(acc.tables("П. 2, GT межсессионно (N = 6, пул FERET)", g.persons.size(), g, pairs));
        FarMethods.write(out.resolve("xbase_gt_wstatus.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- FERET: общая подготовка

    static final class Setup {
        XBaseA0.FeretSetup su;
        XBaseEval.Gt tpl;
        Path out;
        Map<String, List<List<String>>> pool;
        List<Set<String>> halves;
        List<List<String>> cohort;
        List<XBaseUpdate.Ctrl> ctrl;
        XBaseEval[] xs;
        Map<String, Integer> pIdx;
        List<String> persons;
        Map<String, String> dateOf;
        Set<String> measured;
        int maxStep;
    }

    static Setup setup(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        Setup s = new Setup();
        s.su = XBaseA0.feretSetup(exportDir, selLines, settings);
        s.tpl = s.su.tpl;
        s.out = s.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles roAll = s.su.ro;
        s.pool = XBaseXsF.pool(roAll, s.su.byKey);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, s.tpl.seed);
        s.cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            s.cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        s.ctrl = new ArrayList<>();
        for (String p : roAll.ctrl) {
            List<String> ks = XBaseFeret.sorted(XBaseFeret.pairOf(roAll.ctrlKeys.get(p), s.su.byKey, false));
            s.ctrl.add(new XBaseUpdate.Ctrl(p, s.su.byKey.get(ks.get(0)).date(), ks));
        }
        Set<String> used = new TreeSet<>();
        for (List<List<String>> ds : s.pool.values()) for (List<String> d : ds) used.addAll(d);
        for (XBaseUpdate.Ctrl k : s.ctrl) used.addAll(k.keys());
        // Роль «межс.» п. 2 (галерея, пробы, контроль – как feret-xs2).
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.ctrlKeys)) m.values().forEach(used::addAll);
        s.dateOf = new HashMap<>();
        for (String k : used) s.dateOf.put(k, s.su.byKey.get(k).date());
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(s.su.dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        s.xs = new XBaseEval[] {new XBaseEval(), new XBaseEval()};
        s.xs[0].vectors(fr, EnumSet.of(V.L1), s.tpl.params);
        s.xs[1].vectors(fr, EnumSet.of(V.L1), s.tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        List<String> perm = new ArrayList<>(s.pool.keySet());
        Collections.shuffle(perm, new Random(s.tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        s.halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        for (XBaseUpdate.Ctrl k : s.ctrl) ps0.add(k.person());
        s.persons = new ArrayList<>(ps0);
        s.pIdx = new HashMap<>();
        for (int i = 0; i < s.persons.size(); i++) s.pIdx.put(s.persons.get(i), i);
        s.measured = new TreeSet<>();
        for (Map.Entry<String, List<List<String>>> e : s.pool.entrySet()) if (e.getValue().size() >= XBaseUpdate.MIN_DATES) s.measured.add(e.getKey());
        for (String p : s.measured) s.maxStep = Math.max(s.maxStep, s.pool.get(p).size() - XBaseUpdate.START);
        return s;
    }

    static List<List<String>> aug(Setup s, int pp, boolean usePool) {
        return XBaseUpdate.aug(s.cohort, s.halves.get(1 - pp), s.pool, usePool);
    }

    /** Контекст: схема W по train (человек → ключи; пусто – (Wб)). */
    static Ctx ctx(Setup s, int pp, Proj pj, Map<String, List<String>> train) {
        return XBaseNewcomer.ctx(s.xs[pj.f()], s.tpl, s.su.rOf, new ArrayList<>(s.halves.get(pp)), train, aug(s, pp, pj.pool()), s.pool, s.dateOf);
    }

    /** Классы обучения: (Wа) – все люди галереи по стартовым сессиям (новички – 1 дата); (Wб) – никого. */
    static Map<String, List<String>> train(List<String> gal, Map<String, List<List<String>>> pool, Set<String> nw, boolean wb) {
        Map<String, List<String>> t = new LinkedHashMap<>();
        if (wb) return t;
        for (String p : gal) {
            List<String> k = new ArrayList<>();
            int n = nw.contains(p) ? 1 : XBaseUpdate.START;
            for (int d = 0; d < n; d++) k.addAll(pool.get(p).get(d));
            t.put(p, k);
        }
        return t;
    }

    // ---------------------------------------------------------------- п. 3: новички

    static List<SD> relabel(List<SD> sds, Set<String> nw, Map<String, Integer> pIdx) {
        Set<Integer> ni = XBaseNewcomer.nwIdx(nw, pIdx);
        List<SD> out = new ArrayList<>();
        for (SD sd0 : sds) {
            SD sd = SD.decode(sd0.encode()).get(0);
            List<XBaseNewcomer.OwnA> own = new ArrayList<>();
            for (XBaseNewcomer.OwnA a : sd.own) own.add(new XBaseNewcomer.OwnA(a.p(), ni.contains(a.p()), a.r(), a.ok(), a.front(), a.best(), ni.contains(a.best()),
                    a.key()));
            sd.own = own;
            List<XBaseNewcomer.CtrlA> ct = new ArrayList<>();
            for (XBaseNewcomer.CtrlA a : sd.ctrl) ct.add(new XBaseNewcomer.CtrlA(a.p(), a.r(), a.best(), ni.contains(a.best()), a.same(), a.expNew(), a.expOld(), a.key()));
            sd.ctrl = ct;
            for (int i = 0; i < sd.galP.length; i++) sd.galNew[i] = ni.contains(sd.galP[i]);
            out.add(sd);
        }
        return out;
    }

    static String encode(List<SD> sds) {
        StringBuilder t = new StringBuilder();
        for (SD sd : sds) t.append(sd.encode());
        return t.toString();
    }

    /** П. 3: (н1) и (н0) при (Wа) и (Wб); (Wа) (н1) 30 %, r = 0, L1-xs-F – сверка с (н1-W) №7 (XBaseNewcomer.runConfig). */
    static String newcomers(Setup s, XBaseOther.Checkpoint cp, Map<String, List<SD>> res) throws IOException {
        StringBuilder check = new StringBuilder("\n=== П. 3, сверка: (Wа) (н1), 30 %, r = 0, L1-xs-F = (н1-W) №7 (пересчёт кода XBaseNewcomer.runConfig, кодировки шагов "
                + "побайтно) ===\n");
        Map<String, Set<String>> newOf = new LinkedHashMap<>();
        for (int r = 0; r < XBaseNewcomer.R; r++) {
            for (int pp = 0; pp < 2; pp++) {
                List<String> hp = new ArrayList<>(s.halves.get(pp));
                Collections.shuffle(hp, new Random(s.tpl.seed + XBaseNewcomer.PERM_SEED_OFFSET + r));
                for (int si = 0; si < XBaseNewcomer.SHARES.length; si++) {
                    newOf.put(r + "_" + si + "_" + pp, new TreeSet<>(hp.subList(0, (int) Math.round(XBaseNewcomer.SHARES[si] * hp.size()))));
                }
            }
        }
        for (Proj pj : projs()) {
            for (String sch : pj.pool() ? new String[] {"Wa", "Wb"} : new String[] {"Wa"}) {
                boolean wb = sch.equals("Wb");
                for (int pp = 0; pp < 2; pp++) {
                    List<String> gal = new ArrayList<>(s.halves.get(pp));
                    // (н0): все с 2 датами; при (Wа) – W на всех по 2 датам, при (Wб) – та же W, что у (н1).
                    String n0name = String.format(Locale.ROOT, "n0_%s_%s_p%d", pj.label(), sch, pp);
                    String n0 = cp.read(n0name);
                    Ctx cb = null;
                    if (n0 == null) {
                        Ctx c0 = ctx(s, pp, pj, train(gal, s.pool, Set.of(), wb));
                        if (wb) cb = c0;
                        n0 = encode(XBaseNewcomer.simulate(c0, XBaseNewcomer.plan(gal, s.pool, Set.of(), null, 0, s.dateOf), false, false, s.ctrl, s.pIdx,
                                XBaseNewcomer.STEPS));
                        cp.save(n0name, n0);
                        System.out.printf(Locale.ROOT, "feret-wstatus: п. 3 %s %s проход %d (н0) готово%n", pj.label(), sch, pp + 1);
                    }
                    List<SD> n0sd = SD.decode(n0);
                    for (int r = 0; r < (SMOKE ? 1 : XBaseNewcomer.R); r++) {
                        for (int si = 0; si < XBaseNewcomer.SHARES.length; si++) {
                            Set<String> nw = newOf.get(r + "_" + si + "_" + pp);
                            String name = String.format(Locale.ROOT, "n1_%s_%s_r%d_s%d_p%d", pj.label(), sch, r, si, pp);
                            String n1 = cp.read(name);
                            if (n1 == null) {
                                long ts = System.nanoTime();
                                Ctx c = wb ? (cb != null ? cb : (cb = ctx(s, pp, pj, Map.of()))) : ctx(s, pp, pj, train(gal, s.pool, nw, false));
                                n1 = encode(XBaseNewcomer.simulate(c, XBaseNewcomer.plan(gal, s.pool, nw, null, r, s.dateOf), false, false, s.ctrl, s.pIdx,
                                        XBaseNewcomer.STEPS));
                                cp.save(name, n1);
                                System.out.printf(Locale.ROOT, "feret-wstatus: п. 3 %s готово (%.0f с)%n", name, (System.nanoTime() - ts) / 1e9);
                            }
                            String k = pj.label() + "|" + sch + "|" + si;
                            res.computeIfAbsent(k + "|n1", z -> new ArrayList<>()).addAll(SD.decode(n1));
                            res.computeIfAbsent(k + "|n0", z -> new ArrayList<>()).addAll(relabel(n0sd, nw, s.pIdx));
                            if (sch.equals("Wa") && r == 0 && si == 1 && pj.label().equals("L1-xs-F")) {
                                String cname = "check_n1w_p" + pp;
                                String old = cp.read(cname);
                                if (old == null) {
                                    Map<XBaseNewcomer.Traj, List<SD>> m = XBaseNewcomer.runConfig(s.xs[pj.f()], s.tpl, s.su.rOf, s.halves.get(pp), s.halves.get(1 - pp),
                                            s.pool, pj.pool(), s.cohort, s.dateOf, nw, Map.of(), 0, s.ctrl, s.pIdx, true);
                                    old = encode(m.get(XBaseNewcomer.Traj.N1W));
                                    cp.save(cname, old);
                                }
                                boolean same = old.equals(n1);
                                check.append(String.format(Locale.ROOT, "проход %d: %s (шагов %d)%n", pp + 1, same ? "совпало" : "НЕ СОВПАЛО", SD.decode(n1).size()));
                                if (!same) throw new IllegalStateException("feret-wstatus: (Wа) (н1) не совпало с (н1-W) №7, проход " + (pp + 1));
                            }
                        }
                    }
                }
            }
        }
        return check.toString();
    }

    static String newcomerTables(Map<String, List<SD>> res, int persons, XBaseEval.Gt tpl) {
        StringBuilder t = new StringBuilder("\n\n######## П. 3: новички при (Wа) и (Wб) ########\n");
        t.append("Оговорка (поправка 3 №1 от 16.10): при (Wа) после шага 1 у новичков 1 сессия в W и 1 вне, у старожилов – 2 и 1; при (Wб) вне W все "
                + "сессии у всех.\n");
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseNewcomer.Res> rs = new LinkedHashMap<>();
        for (Proj pj : projs()) {
            for (String sch : pj.pool() ? new String[] {"Wa", "Wb"} : new String[] {"Wa"}) {
                for (int si = 0; si < XBaseNewcomer.SHARES.length; si++) {
                    for (String tr : new String[] {"n1", "n0"}) {
                        for (int j = 1; j <= XBaseNewcomer.STEPS; j++) {
                            List<SD> sds = XBaseNewcomer.steps(res.get(pj.label() + "|" + sch + "|" + si + "|" + tr), j);
                            rs.put(key(pj, sch, si, tr, j), XBaseNewcomer.evaluate(sds, 0, persons, w));
                        }
                    }
                }
            }
        }
        t.append("\n=== Ошибка своих по группам, FAR контроля, разрыв новичок − старожил (точка [95 %]) ===\n"
                + "проекция | схема | доля | строка | шаг | новичков: проб / людей | старожилов: проб / людей | ошибка новичков П3 | старожилов П3 | "
                + "новичков П2 | старожилов П2 | FAR П3 | FAR П2 | разд.: новичков | старожилов | разрыв П3 | разрыв разд.\n");
        for (Map.Entry<String, XBaseNewcomer.Res> e : rs.entrySet()) {
            XBaseNewcomer.Res r = e.getValue();
            List<String> row = new ArrayList<>(List.of(e.getKey().split("\\|")));
            row.add(r.nNew + " / " + r.pNew);
            row.add(r.nOld + " / " + r.pOld);
            for (int i = 0; i < 8; i++) row.add(XBaseNewcomer.cell(r.point[i], r.reps[i]));
            row.add(gap(r, 0, 1));
            row.add(gap(r, 6, 7));
            t.append(String.join(" | ", row)).append('\n');
        }
        t.append("\n=== Парные разности, п. п. ===\nпара | шаг | ошибка новичков П3 | П2 | разд. | старожилов П3 | П2 | разд. | FAR П3 | П2\n");
        for (Proj pj : projs()) {
            for (int si = 0; si < XBaseNewcomer.SHARES.length; si++) {
                for (int j = 1; j <= XBaseNewcomer.STEPS; j++) {
                    String sh = String.format(Locale.ROOT, "%.0f %%", 100 * XBaseNewcomer.SHARES[si]);
                    if (pj.pool()) {
                        t.append(XBaseNewcomer.diffRow(sh, pj.label() + " (н1): (Wб) − (Wа)", j, rs.get(key(pj, "Wb", si, "n1", j)), rs.get(key(pj, "Wa", si, "n1", j)),
                                false));
                    }
                    for (String sch : pj.pool() ? new String[] {"Wa", "Wb"} : new String[] {"Wa"}) {
                        t.append(XBaseNewcomer.diffRow(sh, pj.label() + " " + schLabel(sch) + ": (н1) − (н0)", j, rs.get(key(pj, sch, si, "n1", j)),
                                rs.get(key(pj, sch, si, "n0", j)), true));
                    }
                }
            }
        }
        t.append("\n=== Хабы у контрольных чужих по группам ===\nпроекция | схема | доля | строка | шаг | лучших на новичках / доля новичков в галерее | "
                + "анти-хабов: новичков (из) | старожилов (из)\n");
        for (Proj pj : projs()) {
            for (String sch : pj.pool() ? new String[] {"Wa", "Wb"} : new String[] {"Wa"}) {
                for (int si = 0; si < XBaseNewcomer.SHARES.length; si++) {
                    for (String tr : new String[] {"n1", "n0"}) {
                        for (int j = 1; j <= XBaseNewcomer.STEPS; j++) {
                            t.append(key(pj, sch, si, tr, j).replace("|", " | ")).append(" | ")
                                    .append(hubRow(XBaseNewcomer.steps(res.get(pj.label() + "|" + sch + "|" + si + "|" + tr), j))).append('\n');
                        }
                    }
                }
            }
        }
        return t.toString();
    }

    static String schLabel(String sch) {
        return sch.equals("Wa") ? "(Wа)" : "(Wб)";
    }

    static String key(Proj pj, String sch, int si, String tr, int j) {
        return String.format(Locale.ROOT, "%s|%s|%.0f %%|%s|%d", pj.label(), schLabel(sch), 100 * XBaseNewcomer.SHARES[si], tr.equals("n1") ? "(н1) как есть" :
                "(н0) без новичков", j);
    }

    /** Разрыв «новички − старожилы» мер a и b одной строки с интервалом по репликам. */
    static String gap(XBaseNewcomer.Res r, int a, int b) {
        double p = r.point[a] - r.point[b];
        if (Double.isNaN(p)) return "–";
        double[] d = new double[r.reps[a].length];
        int k = 0;
        for (int q = 0; q < d.length; q++) {
            double v = r.reps[a][q] - r.reps[b][q];
            if (!Double.isNaN(v)) d[k++] = v;
        }
        if (k == 0) return String.format(Locale.ROOT, "%+.1f", p);
        double[] s = FeiBootstrap.sorted(Arrays.copyOf(d, k));
        double lo = FeiBootstrap.quantile(s, 0.025);
        double hi = FeiBootstrap.quantile(s, 0.975);
        String f = "%+.1f";
        return String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", FeiBootstrap.nz(p, f), FeiBootstrap.nz(lo, f), FeiBootstrap.nz(hi, f), FeiBootstrap.sig(lo, hi) ? " *" : "");
    }

    static String hubRow(List<SD> sds) {
        int onNew = 0;
        int n = 0;
        int galNew = 0;
        int gal = 0;
        int antiNew = 0;
        int antiOld = 0;
        for (SD sd : sds) {
            Set<Integer> best = new HashSet<>();
            for (XBaseNewcomer.CtrlA a : sd.ctrl) {
                best.add(a.best());
                if (a.bestNew()) onNew++;
                n++;
            }
            for (int i = 0; i < sd.galP.length; i++) {
                gal++;
                if (sd.galNew[i]) galNew++;
                if (!best.contains(sd.galP[i])) {
                    if (sd.galNew[i]) antiNew++;
                    else antiOld++;
                }
            }
        }
        return String.format(Locale.ROOT, "%.1f %% / %.1f %% | %d (%d) | %d (%d)", 100.0 * onNew / n, 100.0 * galNew / gal, antiNew, galNew, antiOld, gal - galNew);
    }

    // ---------------------------------------------------------------- п. 4 и 4а: часть 1 при (Wа), (Wа′), (Wб)

    static final List<Var> VARS = List.of(Var.A, Var.B, Var.E);

    /** Шаг части 1 – в строку контрольной точки и обратно (нужные таблицам поля). */
    static String encode(Step s) {
        FeiDecompose.Cfg c = s.cfg;
        StringBuilder t = new StringBuilder("S\t" + c.j + "\t" + Double.toString(s.th2) + "\t" + Double.toString(s.th3) + "\n");
        for (int i = 0; i < c.ownR.length; i++) {
            t.append(String.join("\t", "O", "" + c.ownP[i], Double.toString(c.ownR[i]), b(c.ownOk[i]), b(c.noSecond[i]), b(c.trueInf[i]), b(c.poseFrontal[i]))).append('\n');
        }
        for (int i = 0; i < c.ctrlR.length; i++) t.append("C\t").append(c.ctrlP[i]).append('\t').append(Double.toString(c.ctrlR[i])).append('\t').append(s.ctrlBest.get(i))
                .append('\n');
        for (int i = 0; i < s.imp.r().length; i++) t.append("I\t").append(s.imp.p()[i]).append('\t').append(Double.toString(s.imp.r()[i])).append('\n');
        t.append("F\t").append(s.imp.frontal()).append('\n');
        for (String p : new TreeSet<>(s.updTo)) t.append("U\t").append(p).append('\n');
        t.append("N\t").append(s.galN).append('\n');
        return t.toString();
    }

    static String b(boolean v) {
        return v ? "1" : "0";
    }

    static List<Step> decodeSteps(String s) {
        List<Step> out = new ArrayList<>();
        List<String[]> own = new ArrayList<>();
        List<String[]> ctl = new ArrayList<>();
        List<double[]> imp = new ArrayList<>();
        Step cur = null;
        int frontal = 0;
        for (String l : (s + "S\tEND\n").split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (c[0].equals("S")) {
                if (cur != null) {
                    FeiDecompose.Cfg cfg = cur.cfg;
                    int n = own.size();
                    cfg.ownP = new int[n];
                    cfg.ownR = new double[n];
                    cfg.ownOk = new boolean[n];
                    cfg.noSecond = new boolean[n];
                    cfg.trueInf = new boolean[n];
                    cfg.poseFrontal = new boolean[n];
                    for (int i = 0; i < n; i++) {
                        String[] o = own.get(i);
                        cfg.ownP[i] = Integer.parseInt(o[1]);
                        cfg.ownR[i] = Double.parseDouble(o[2]);
                        cfg.ownOk[i] = o[3].equals("1");
                        cfg.noSecond[i] = o[4].equals("1");
                        cfg.trueInf[i] = o[5].equals("1");
                        cfg.poseFrontal[i] = o[6].equals("1");
                    }
                    cfg.thrP = new int[0];
                    cfg.thrR = new double[0];
                    cfg.ctrlP = ctl.stream().mapToInt(x -> Integer.parseInt(x[1])).toArray();
                    cfg.ctrlR = ctl.stream().mapToDouble(x -> Double.parseDouble(x[2])).toArray();
                    for (String[] x : ctl) cur.ctrlBest.add(x[3]);
                    cur.imp = new FeiThreshold.Imp(imp.stream().mapToDouble(x -> x[1]).toArray(), imp.stream().mapToInt(x -> (int) x[0]).toArray(), frontal);
                    out.add(cur);
                }
                if (c[1].equals("END")) break;
                cur = new Step();
                cur.cfg = new FeiDecompose.Cfg();
                cur.cfg.j = Integer.parseInt(c[1]);
                cur.th2 = Double.parseDouble(c[2]);
                cur.th3 = Double.parseDouble(c[3]);
                own = new ArrayList<>();
                ctl = new ArrayList<>();
                imp = new ArrayList<>();
            } else if (c[0].equals("O")) {
                own.add(c);
            } else if (c[0].equals("C")) {
                ctl.add(c);
            } else if (c[0].equals("I")) {
                imp.add(new double[] {Integer.parseInt(c[1]), Double.parseDouble(c[2])});
            } else if (c[0].equals("F")) {
                frontal = Integer.parseInt(c[1]);
            } else if (c[0].equals("U")) {
                cur.updTo.add(c[1]);
            } else if (c[0].equals("N")) {
                cur.galN = Integer.parseInt(c[1]);
            }
        }
        return out;
    }

    static String encodeSteps(List<Step> steps) {
        StringBuilder t = new StringBuilder();
        for (Step s : steps) t.append(encode(s));
        return t.toString();
    }

    /**
     * Переобучать ли W перед шагом step (j ≥ 2): не дальше last и только если на шаге есть пробы своих (у кого-то из галереи есть
     * дата START + step).
     */
    static boolean retrainAt(int step, int last, List<String> gal, Map<String, List<List<String>>> dates) {
        if (step < 2 || step > last) return false;
        for (String p : gal) if (dates.get(p).size() >= XBaseUpdate.START + step) return true;
        return false;
    }

    /**
     * (Wа′): перед шагами 2…last, на которых есть пробы своих, – W и модели частей заново на всех текущих сессиях шаблонов;
     * центры – по-прежнему по шаблонам. №8 – last = 3.
     */
    static XBaseUpdate.Retrain retrain(Setup s, int pp, Proj pj, int last) {
        return (step, tpl, c) -> {
            if (!retrainAt(step, last, c.gal, c.dates)) return c;
            Map<String, List<String>> tr = new LinkedHashMap<>();
            for (String p : c.gal) tr.put(p, tpl.get(p).keys());
            System.out.printf(Locale.ROOT, "feret-wstatus: (Wа′) %s проход %d – переобучение W перед шагом %d%n", pj.label(), pp + 1, step);
            return ctx(s, pp, pj, tr);
        };
    }

    /**
     * Параметры части 1: схемы, варианты, варианты таблицы доли обновлённых, до какого шага переобучать (Wа′), пары
     * «уменьшаемое − вычитаемое», строки файла оценок (null – не писать).
     */
    record P1(String[] schemes, List<Var> vars, List<Var> updVars, int retrainLast, List<Var[]> pairs, StringBuilder lines) {}

    /** №8 (п. 4 и 4а): (а), (б), (д) при (Wа), (Wа′) (шаги 2–3), (Wб). */
    static final P1 P1_N8 = new P1(new String[] {"Wa", "Wa2", "Wb"}, VARS, List.of(Var.B, Var.E), RETRAIN_LAST,
            List.of(new Var[] {Var.E, Var.A}, new Var[] {Var.B, Var.A}), null);

    /** П. 4 и 4а: (а), (б), (д) при (Wа) (проверка против №4), (Wа′), (Wб); порог приёма П3. */
    static String part1(Setup s, XBaseOther.Checkpoint cp) throws IOException {
        return part1(s, cp, P1_N8);
    }

    /** Часть 1 по параметрам p; (Wа) – проверка против файла оценок №4. */
    static String part1(Setup s, XBaseOther.Checkpoint cp, P1 p) throws IOException {
        Map<String, String> old = new HashMap<>();
        Path sf = s.out.resolve(XBaseSession.SCORES);
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(sf), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                String[] c = l.split("\t", -1);
                if (c.length < 12 || !c[2].equals("П3") || c[5].equals("threshold")) continue;
                old.put(c[0] + "\t" + c[1] + "\t" + c[3] + "\t" + c[4] + "\t" + c[5] + "\t" + c[6], c[8] + "\t" + c[9] + "\t" + c[10]);
            }
        }
        StringBuilder check = new StringBuilder("\n=== П. 4, проверка: (Wа) " + String.join(", ", p.vars().stream().map(v -> v.label.substring(0, 3)).toList())
                + " = строки feret-update №4 (порог приёма П3; ключ решающего снимка, лучший, d₁/d₂; свои и контроль, все шаги) ===\n");
        StringBuilder body = new StringBuilder();
        StringBuilder upd = new StringBuilder("\n=== П. 4: доля обновлённых после шага 1 среди лучших кандидатов контрольных чужих на шаге 2 против их доли в галерее "
                + "(порог приёма П3; бутстреп по контрольным людям; критерий – заранее, поправка 2 №1 от 16.10; избыток значим, если нижняя граница "
                + "интервала > 0) ===\n"
                + "проекция | схема | вариант | обновлённых в галерее (доля) | лучших на обновлённых [95 %] | избыток [95 %] | вывод по критерию\n");
        List<String> ctrlP = s.ctrl.stream().map(XBaseUpdate.Ctrl::person).toList();
        int[][] wc = FeiBootstrap.weights(ctrlP.size(), s.tpl.b, s.tpl.seed + FeiBootstrap.SEED_OFFSET);
        for (Proj pj : projs()) {
            if (!pj.pool() && !SMOKE) continue;
            for (String sch : p.schemes()) {
                Map<Var, List<List<Step>>> all = new LinkedHashMap<>();
                for (Var v : p.vars()) all.put(v, new ArrayList<>());
                for (int pp = 0; pp < 2; pp++) {
                    List<String> gal = new ArrayList<>(s.halves.get(pp));
                    Ctx c = null;
                    for (Var v : p.vars()) {
                        String name = String.format(Locale.ROOT, "p1_%s_%s_p%d_%s", pj.label(), sch, pp, v.name());
                        String enc = cp.read(name);
                        if (enc == null) {
                            long ts = System.nanoTime();
                            if (c == null) c = ctx(s, pp, pj, train(gal, s.pool, Set.of(), sch.equals("Wb")));
                            String tag = pj.label() + "\t" + (pp + 1) + "\t" + XBaseUpdate.Gate.P3.label + "\t" + v.label;
                            List<Step> steps = XBaseUpdate.simulate(c, v, XBaseUpdate.Gate.P3, s.ctrl, s.pIdx, s.measured, s.maxStep, tag,
                                    sch.equals("Wa2") && v != Var.A ? retrain(s, pp, pj, p.retrainLast()) : null);
                            if (p.lines() != null) {
                                StringBuilder ls = new StringBuilder();
                                for (Step st : steps) for (String l : st.lines.toString().split("\n")) if (!l.isEmpty()) ls.append(schName(sch)).append('\t').append(l)
                                        .append('\n');
                                cp.save(name + "_lines", ls.toString());
                            }
                            if (sch.equals("Wa")) {
                                int n = 0;
                                for (Step st : steps) {
                                    for (String l : st.lines.toString().split("\n")) {
                                        String[] q = l.split("\t", -1);
                                        if (q[5].equals("threshold")) continue;
                                        String k = q[0] + "\t" + q[1] + "\t" + q[3] + "\t" + q[4] + "\t" + q[5] + "\t" + q[6];
                                        String now = q[8] + "\t" + q[9] + "\t" + q[10];
                                        if (!now.equals(old.get(k))) throw new IllegalStateException("feret-wstatus: (Wа) не совпало с №4: " + k + " – было " + old.get(k)
                                                + ", сейчас " + now);
                                        n++;
                                    }
                                }
                                cp.save(name + "_check", Integer.toString(n));
                            }
                            enc = encodeSteps(steps);
                            cp.save(name, enc);
                            System.out.printf(Locale.ROOT, "feret-wstatus: п. 4 %s готово (%.0f с)%n", name, (System.nanoTime() - ts) / 1e9);
                        }
                        if (sch.equals("Wa")) check.append(String.format(Locale.ROOT, "%s проход %d %s: совпало %s строк%n", pj.label(), pp + 1, v.label,
                                cp.read(name + "_check")));
                        all.get(v).add(decodeSteps(enc));
                        if (p.lines() != null) p.lines().append(cp.read(name + "_lines"));
                    }
                }
                body.append(part1Tables(pj, sch, all, s, p));
                for (Var v : p.updVars()) upd.append(updShare(pj, sch, v, all.get(v), s, ctrlP, wc));
            }
        }
        return check + upd.toString() + body;
    }

    static String schName(String sch) {
        return switch (sch) {
            case "Wa" -> "(Wа)";
            case "Wa2" -> "(Wа′)";
            default -> "(Wб)";
        };
    }

    static String updShare(Proj pj, String sch, Var v, List<List<Step>> passes, Setup s, List<String> ctrlP, int[][] w) {
        int upd = 0;
        int gal = 0;
        Map<String, double[]> per = new TreeMap<>(); // контрольный → {на обновлённых, проб, сумма долей}
        for (List<Step> steps : passes) {
            Step s1 = steps.get(0);
            Step s2 = steps.get(1);
            upd += s1.updTo.size();
            gal += s1.galN;
            double share = (double) s1.updTo.size() / s1.galN;
            for (int i = 0; i < s2.ctrlBest.size(); i++) {
                double[] a = per.computeIfAbsent(ctrlP.get(i), z -> new double[3]);
                if (s1.updTo.contains(s2.ctrlBest.get(i))) a[0]++;
                a[1]++;
                a[2] += share;
            }
        }
        double[] p = new double[3];
        for (double[] a : per.values()) for (int i = 0; i < 3; i++) p[i] += a[i];
        double[] reps = new double[w.length];
        double[] dreps = new double[w.length];
        for (int q = 0; q < w.length; q++) {
            double[] z = new double[3];
            for (int i = 0; i < ctrlP.size(); i++) {
                double[] a = per.get(ctrlP.get(i));
                if (a == null) continue;
                for (int k = 0; k < 3; k++) z[k] += w[q][i] * a[k];
            }
            reps[q] = 100 * z[0] / z[1];
            dreps[q] = 100 * (z[0] - z[2]) / z[1];
        }
        double[] ci = FeiNorm.ci(reps);
        double[] cd = FeiNorm.ci(dreps);
        // Критерий – избыток (одностороннее: значим, если нижняя граница > 0; значимый недостаток – не избыток).
        boolean sig = FeiBootstrap.sig(cd[0], cd[1]);
        boolean excess = cd[0] > FeiBootstrap.SIG_EPS;
        String verdict = sch.equals("Wa") ? "– (опора: сессии вне W есть)" : excess ? "избыток значим – причина в свежести или общей сессии"
                : cd[1] < -FeiBootstrap.SIG_EPS ? "избытка нет (значимый недостаток) – причина «сессии вне W»" : "избыток незначим – причина «сессии вне W»";
        String f = "%+.1f";
        return String.format(Locale.ROOT, "%s | %s | %s | %d из %d (%.1f %%) | %.1f %% (%d/%d) [%.1f; %.1f] | %+.1f [%+.1f; %+.1f]%s | %s%n", pj.label(), schName(sch), v.label,
                upd, gal, 100.0 * upd / gal, 100 * p[0] / p[1], (int) p[0], (int) p[1], ci[0], ci[1], FeiBootstrap.nz(100 * (p[0] - p[2]) / p[1], f), FeiBootstrap.nz(cd[0], f),
                FeiBootstrap.nz(cd[1], f), sig ? " *" : "", verdict);
    }

    /** Общие таблицы части 1 схемы: шаг 2 и шаги ≥ 2; пары (д) − (а), (б) − (а). */
    static String part1Tables(Proj pj, String sch, Map<Var, List<List<Step>>> all, Setup s, P1 p) {
        StringBuilder t = new StringBuilder("\n\n######## П. 4: часть 1 при " + schName(sch) + ", " + pj.label() + ", порог приёма П3 ########\n");
        int persons = s.persons.size();
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        String p0 = pj.label() + " " + schName(sch) + " ";
        for (Var v : p.vars()) {
            for (int j = 2; j <= XBaseUpdate.TABLE_STEPS + 1; j++) {
                int from = j <= XBaseUpdate.TABLE_STEPS ? j : 2;
                int to = j <= XBaseUpdate.TABLE_STEPS ? j : s.maxStep;
                List<FeiDecompose.Cfg> cs = XBaseUpdate.cfgs(all.get(v), from, to);
                if (cs.isEmpty()) continue;
                String l = p0 + v.label + (j <= XBaseUpdate.TABLE_STEPS ? " шаг " + j : " шаги ≥ 2");
                res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), cs, XBaseUpdate.imps(all.get(v), from, to, false), persons, s.tpl.b, s.tpl.seed
                        + FeiBootstrap.SEED_OFFSET));
            }
        }
        List<String[]> pairs = new ArrayList<>();
        for (String st : new String[] {" шаг 2", " шаги ≥ 2"}) for (Var[] pr : p.pairs()) pairs.add(new String[] {p0 + pr[0].label + st, p0 + pr[1].label + st});
        pairs.removeIf(q -> !res.containsKey(q[0]) || !res.containsKey(q[1]));
        int[][] w = s.tpl.b > 0 ? FeiBootstrap.weights(persons, s.tpl.b, s.tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(persons), w));
        XBaseOther.tables(t, res, sep, s.tpl.b, pairs);
        return t.toString();
    }

    // ---------------------------------------------------------------- запуск feret-wstatus

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-wstatus: только равные галереи (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        Setup s = setup(export, exportDir, feiKeys, selLines, settings);
        System.out.printf(Locale.ROOT, "feret-wstatus: векторы готовы (%.0f с)%n", (System.nanoTime() - t0) / 1e9);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-wstatus %s; снимков %d; seed %d; r0 %s; пул %d", s.su.dir, s.su.imgs.size(),
                s.tpl.seed, s.tpl.r0, s.pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_wstatus_eq" + (SMOKE ? "_smoke" : "")), fp);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4), часть 3: все свои в одном статусе относительно W (XBaseEval feret-wstatus): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("""
                Схемы W (задание №8 от 15.10, поправки №1 от 16.10): (Wа) – W на всех людях галереи (их стартовые сессии) + когорта FEI + пул;
                (Wб) – W только на когорте и пуле, своих в обучении нет, центры – проекции снимков шаблона; модели частей при (Wб) – та же W
                (центры без части); (Wа′) – (Wа), переобученная (W и 5 моделей частей) на всех текущих сессиях шаблонов перед шагами 2 и 3
                части 1, дальше не меняется; у (а) шаблоны не меняются – (Wа′) = (Wа). L1 (без пула) – только (Wа).
                П. 2 – роль «межс.» feret-xs2 (половины №2), «П2/П3 по общей W» – «чужие» по основной модели (центры людей части исключены).
                П. 3 – новички как в №7 (доли 10 и 30 %, R = 3, seed faces.seed + 9970 + r, обновление (д), M = 3), только (н1) и (н0).
                П. 4 / 4а – часть 1 как в №4 (половины №2, старт 2 даты, проход – пара, контроль 200), варианты (а), (б), (д), порог приёма П3.
                """);
        if (s.tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» – интервал разности не содержит 0.%n", s.tpl.b, s.tpl.seed
                + FeiBootstrap.SEED_OFFSET));
        t.append(feret2(s, cp));
        Map<String, List<SD>> res = new HashMap<>();
        t.append(newcomers(s, cp, res));
        t.append(newcomerTables(res, s.persons.size(), s.tpl));
        t.append(part1(s, cp));
        t.append(String.format(Locale.ROOT, "%nВсего %.0f с; потоков %d.%n", (System.nanoTime() - t0) / 1e9, s.tpl.threads));
        FarMethods.write(s.out.resolve("xbase_feret_wstatus" + (SMOKE ? "_smoke" : "") + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- feret-wstatus4: часть 4 пункта (4)

    /** Строки отчёта, начинающиеся с подписи строки таблицы («подпись | …»), по порядку. */
    static List<String> rowsOf(List<String> report, String label) {
        return report.stream().filter(l -> l.startsWith(label + " | ")).toList();
    }

    /**
     * Часть 4 (задание Жорика №2 от 16.10.2026, план принят №3): (а), (д), (е) при (Wа) и (Wа′) – W и 5 моделей частей
     * переобучаются перед каждым шагом с пробами своих; L1-xs-F и L1-tt-xs-F; порог приёма П3. Проверки: (Wа) – строки №4
     * побайтно; (Wа′) (а), (д) шагов 2–3 – строки таблиц отчёта №8 побайтно. Отчёт и файл оценок – только D:\data\Feret\reports\eq\.
     */
    static void feret4(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-wstatus4: только равные галереи (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        Setup s = setup(export, exportDir, feiKeys, selLines, settings);
        System.out.printf(Locale.ROOT, "feret-wstatus4: векторы готовы (%.0f с)%n", (System.nanoTime() - t0) / 1e9);
        String sfx = SMOKE ? "_smoke" : "";
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-wstatus4 %s; снимков %d; seed %d; r0 %s; пул %d", s.su.dir, s.su.imgs.size(),
                s.tpl.seed, s.tpl.r0, s.pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_wstatus4_eq" + sfx), fp);
        StringBuilder lines = new StringBuilder();
        P1 p = new P1(new String[] {"Wa", "Wa2"}, List.of(Var.A, Var.E, Var.F), List.of(Var.E, Var.F), Integer.MAX_VALUE,
                List.of(new Var[] {Var.F, Var.A}, new Var[] {Var.E, Var.A}, new Var[] {Var.F, Var.E}), lines);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4), часть 4: польза подтверждённых обновлений при равных статусе в W и ширине шаблона (XBaseEval feret-wstatus4): "
                + "Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("""
                Схема (задание №2 от 16.10, план принят №3): часть 1 как в №4 (половины №2, старт 2 даты, проход – пара, контроль 200, порог
                приёма П3); варианты (а) статично, (д) оракул M = 3, (е) оракул M = 2 (у всех шаблон ровно из 2 сессий); схемы (Wа) – W и 5
                моделей частей обучены на старте (как №4) и (Wа′) – W и 5 моделей частей заново на всех текущих сессиях шаблонов перед
                каждым шагом, на котором есть пробы своих (шаги 2…9); у (а) шаблоны не меняются – (Wа′) = (Wа) без переобучения.
                Основной ответ – (е) − (а) и (д) − (а) при (Wа′) по каждому кандидату. (е) − (д) – справочно: варианты различаются и окном
                (2 / 3), и равенством ширины (в (д) у людей с 2 датами шаблон уже, чем у обновлённых), чистого ответа «окно 2 или 3»
                разность не даёт.
                """);
        if (s.tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» – интервал разности не содержит 0.%n", s.tpl.b, s.tpl.seed
                + FeiBootstrap.SEED_OFFSET));
        String body = part1(s, cp, p);
        // Сверка (Wа′) (а), (д), шаги 2 и 3, со строками таблиц отчёта №8.
        StringBuilder check = new StringBuilder("\n=== Проверка: (Wа′) (а), (д), шаги 2 и 3 = строки таблиц отчёта №8 xbase_feret_wstatus.txt (побайтно) ===\n");
        Path n8 = s.out.resolve("xbase_feret_wstatus.txt");
        if (!SMOKE) {
            List<String> old = Files.readAllLines(n8, StandardCharsets.UTF_8).stream().map(l -> l.replace("\r", "")).toList();
            List<String> now = List.of(body.replace("\r", "").split("\n"));
            for (Proj pj : projs()) {
                if (!pj.pool()) continue;
                for (Var v : List.of(Var.A, Var.E)) {
                    for (int j = 2; j <= 3; j++) {
                        String label = pj.label() + " " + schName("Wa2") + " " + v.label + " шаг " + j;
                        List<String> a = rowsOf(old, label);
                        List<String> b = rowsOf(now, label);
                        boolean same = !a.isEmpty() && a.equals(b);
                        check.append(String.format(Locale.ROOT, "%s: %s (%d строк)%n", label, same ? "совпало" : "НЕ СОВПАЛО", a.size()));
                        if (!same) throw new IllegalStateException("feret-wstatus4: не совпало с №8: " + label);
                    }
                }
            }
            check.append("Отчёт №8: ").append(FeiThreshold.sha(n8)).append('\n');
        } else {
            check.append("пробный режим – сверка не делалась\n");
        }
        t.append(check).append(body);
        t.append(String.format(Locale.ROOT, "%nВсего %.0f с; потоков %d.%n", (System.nanoTime() - t0) / 1e9, s.tpl.threads));
        Path sf = s.out.resolve("xbase_feret_wstatus4" + sfx + "_scores.tsv.gz");
        try (java.io.Writer w = new java.io.OutputStreamWriter(new java.util.zip.GZIPOutputStream(Files.newOutputStream(sf), 1 << 16), StandardCharsets.UTF_8)) {
            w.write("# FERET, часть 4 пункта (4) (XBaseWStatus feret-wstatus4), только локально\n# Код: коммит " + FarMethods.commit() + "\n");
            w.write("# Как файл feret-update, первым столбцом – схема W; строка threshold – пороги шага\n");
            w.write("scheme\tproj\tpass\tgate\tvariant\tstep\trole\tperson\tmeasured\tkey\tbest\tratio\tfrontal\n");
            w.write(lines.toString());
        }
        FarMethods.write(s.out.resolve("xbase_feret_wstatus4" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
