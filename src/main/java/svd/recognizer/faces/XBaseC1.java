package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * Сводка С1, п. (5) – досчёт (№8–№9 Жорика от 21.10.2026; решение Хозяина – вариант Б).
 * <p>
 * fei-xs: FEI, протокол п. 3а, 5 разбиений, по одной конфигурации на разбиение – проба №11 (анфас), галерея – остальные 7 снимков (как
 * конфигурация 4 FeiProtocol); роли (свои, пороговые, контроль, когорта) – как в fei_bootstrap. Строки: L1 без пула; L1-xs-F и L1-tt-xs-F –
 * пул FERET (173 человека, по правилу eq, люди – один класс), схема (Wа) – W на всех людях галереи; (9б) «только классика» для обоих
 * (Хаар + LBF: кадры по пяти точкам LBF, отбор и модель по r_LBF, R_LBF, r0_LBF, как (9б) в (4а)); SFace (минимум 2 снимка шаблона); метод
 * Прокаева (Б2-К, оба комплекта Д10) – из готовых файлов оценок fei-prokaev тех же конфигураций. Меры: П3 и П2 (FAR 1 %, фактический FAR),
 * разделимость при FAR контроля 1 %, EER, разложение (i)–(iv) – как в XBaseOther.tables; ещё порог по пороговым чужим (FAR 1 %, по снимкам, как
 * fei_bootstrap) и верный первый кандидат. Бутстреп – 200 человек FEI, B и seed как у faces-fei-scores.
 * <p>
 * own-xs: своя база, внутри сессии (справочно): 12 ротаций по моментам (OwnEvaluation.split: контроль – представитель момента r mod m,
 * галерея – кадры «+» и «+-» 5 моментов), свои 6 человек; контроль – односессионные люди GT (отбор |r| ≤ R, все снимки); когорта – FEI 20
 * (как GT); пул FERET; строки как у FEI без метода Прокаева. При 6 своих П3 вырожден (части по одному человеку) – справочно. Отдельно – отказы
 * детекторов (YuNet 0,7 экспорта; Хаар – haar.tsv своей базы: нет лица, нет двух глаз; ложная рамка – IoU с YuNet &lt; 0,3) по всем снимкам и по
 * пробам.
 *
 * @author ssv
 */
final class XBaseC1 {

    /** Конфигурация варианта Б: проба №11 (индекс 4 в FeiProtocol.NUMBERS). */
    static final int PROBE_NO = 11;

    private XBaseC1() {
    }

    /** Строка таблицы: подпись, векторы, вариант, пул в обучении, (9б). */
    record Row(String label, XBaseEval x, V v, boolean pool, boolean lbf) {}

    /** Источник (9б) и пороги LBF. */
    record Lbf(XBaseAbl.Src src, XBaseAbl.Thr th, Set<String> ok) {}

    /** Накопители по строкам: разбор XBaseEval (П3, разделимость, разложение) и конфигурации для порога по пороговым и верного кандидата. */
    static final class Run {
        final Map<String, XBaseAbl.Acc> acc = new LinkedHashMap<>();
        final Map<String, List<ProkaevOther.FCfg>> thr = new LinkedHashMap<>();
        final StringBuilder lines = new StringBuilder();
        /** Строки оценок до отказов (9б) по имени блока – для сверки. */
        final Map<String, String> raw = new HashMap<>();
        double fitS;
    }

    /** Конфигурация для порога по пороговым: выбор модели по позе (как parse), (9б) – по LBF. */
    static ProkaevOther.FCfg fcfg(String ls, V v, List<Probe> probes, Map<String, Double> rOf, double r0, Map<String, Integer> pIdx) {
        Map<String, FeiScores.Row> rows = new HashMap<>();
        for (String l : ls.lines().toList()) {
            FeiScores.Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        List<double[]> own = new ArrayList<>();
        List<double[]> thr = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        for (Probe p : probes) {
            FeiScores.Row r = XBaseEval.choose(rows, v, p.role(), p.person(), p.image(), rOf.get(p.key()) <= r0);
            double x = FeiScores.ratio(r, null);
            int pi = pIdx.get(p.person());
            if (r.own()) own.add(new double[] {pi, x, r.bestId().equals(r.trueId()) ? 1 : 0});
            else (r.role().equals(FeiScores.THR) ? thr : ctrl).add(new double[] {pi, x});
        }
        thr.sort((a, z) -> Double.compare(a[1], z[1]));
        ProkaevOther.FCfg c = new ProkaevOther.FCfg();
        c.ownP = own.stream().mapToInt(a -> (int) a[0]).toArray();
        c.ownR = own.stream().mapToDouble(a -> a[1]).toArray();
        c.ownOk = new boolean[own.size()];
        for (int i = 0; i < own.size(); i++) c.ownOk[i] = own.get(i)[2] == 1;
        c.thrP = thr.stream().mapToInt(a -> (int) a[0]).toArray();
        c.thrR = thr.stream().mapToDouble(a -> a[1]).toArray();
        c.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
        c.ctrlR = ctrl.stream().mapToDouble(a -> a[1]).toArray();
        return c;
    }

    /**
     * Счёт строки на одной конфигурации: строки оценок (контрольная точка), разбор для П3 и разделимости (пробы без пороговых), конфигурация
     * для порога по пороговым. probes – все пробы (свои, пороговые, контроль); gal – галерея; train – когорта (+ пул).
     */
    static void row(Row rw, XBaseEval.Gt g, Map<String, List<String>> gal, List<Probe> probes, List<List<String>> parts, List<List<String>> train, int s,
                    Lbf lbf, XBaseOther.Checkpoint cp, Run run, String name, String mode) throws IOException {
        Map<String, List<String>> galR = gal;
        List<List<String>> trainR = train;
        Map<String, Double> rOf = g.rOf;
        double r0 = g.r0;
        Set<String> rej = new HashSet<>();
        if (rw.lbf()) {
            // (9б): в галерее, когорте и пуле – только снимки с точками LBF; проба без точек или с |r_LBF| > R_LBF – отказ; модель – по r_LBF и r0_LBF.
            galR = new LinkedHashMap<>();
            for (Map.Entry<String, List<String>> e : gal.entrySet()) galR.put(e.getKey(), e.getValue().stream().filter(lbf.ok()::contains).toList());
            trainR = XBaseAbl.filter(train, lbf.ok());
            rOf = new HashMap<>(g.rOf);
            for (Probe q : probes) {
                LbfFrames.Lm lm = lbf.src().lbf.get(q.key());
                double r = lm == null ? 0 : Math.abs(lm.pose());
                rOf.put(q.key(), r);
                if (lm == null || r > lbf.th().rLbf()) rej.add(q.role() + "\t" + q.person() + "\t" + q.image());
            }
            for (List<String> ks : galR.values()) for (String k : ks) rOf.put(k, Math.abs(lbf.src().lbf.get(k).pose()));
            r0 = lbf.th().r0Lbf();
        }
        XBaseEval.Gt gf = XBaseAbl.copy(g, trainR, rOf, r0);
        String ls = cp.read(name);
        if (ls == null) {
            long t1 = System.nanoTime();
            ls = XBaseEval.fitLines(gf, rw.x(), EnumSet.of(rw.v()), galR, probes, parts, s);
            run.fitS += (System.nanoTime() - t1) / 1e9;
            cp.save(name, ls);
            System.out.printf(Locale.ROOT, "%s: %s, %s ready%n", mode, name, rw.label());
        }
        run.raw.put(name, ls);
        run.lines.append("# ").append(name).append(", ").append(rw.label()).append('\n').append(ls);
        if (!rej.isEmpty()) ls = XBaseMp2.refuse(ls, rej);
        List<Probe> parseProbes = probes.stream().filter(q -> !q.role().equals(FeiScores.THR)).toList();
        run.acc.computeIfAbsent(rw.label(), x -> new XBaseAbl.Acc(rw.v())).add(XBaseEval.parse(gf, EnumSet.of(rw.v()), galR, parseProbes, parts, ls));
        run.thr.computeIfAbsent(rw.label(), x -> new ArrayList<>()).add(fcfg(ls, rw.v(), probes, rOf, r0, g.pIdx));
    }

    /** Таблица «верный первый кандидат; ошибка своих по порогу пороговых чужих (FAR 1 %) и FAR контроля при нём» с бутстрепом по людям. */
    static void thrTable(StringBuilder r, Map<String, List<ProkaevOther.FCfg>> thr, int persons, int b, long seed, boolean hasThr) {
        int[][] w = b > 0 ? FeiBootstrap.weights(persons, b, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons);
        r.append(hasThr ? "\n=== Верный первый кандидат; порог по пороговым чужим (FAR 1 %, по снимкам, как fei_bootstrap) ===\n"
                : "\n=== Верный первый кандидат (без порога) ===\n");
        r.append(hasThr ? "строка | верный первый, % | ошибка своих при пороге по пороговым, % [2,5; 97,5] | FAR контроля, %\n" : "строка | верный первый, %\n");
        for (Map.Entry<String, List<ProkaevOther.FCfg>> e : thr.entrySet()) {
            long good = 0;
            long all = 0;
            for (ProkaevOther.FCfg c : e.getValue()) {
                all += c.ownOk.length;
                for (boolean x : c.ownOk) if (x) good++;
            }
            if (!hasThr) {
                r.append(String.format(Locale.ROOT, "%s | %.1f%n", e.getKey(), 100.0 * good / all));
                continue;
            }
            double[] p = ProkaevOther.feiErr(e.getValue(), unit, 0.01);
            double[] rep = new double[w.length];
            for (int q = 0; q < w.length; q++) rep[q] = ProkaevOther.feiErr(e.getValue(), w[q], 0.01)[0];
            r.append(String.format(Locale.ROOT, "%s | %.1f | %s | %.2f%n", e.getKey(), 100.0 * good / all, XBaseSepLtp.cell(p[0], rep), p[1]));
        }
    }

    /** Таблицы П3 / П2 / разделимость / EER / разложение (XBaseOther.tables) по накопленным строкам; пары – с L1 без пула и (9б) с основой. */
    static void tables(StringBuilder r, Run run, int persons, XBaseEval.Gt tpl) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons);
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, XBaseAbl.Acc> e : run.acc.entrySet()) {
            XBaseAbl.Acc a = e.getValue();
            res.put(e.getKey(), FeiThreshold.evaluate(e.getKey(), a.v.ratioId(), a.cfgs, a.imps, persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
            sep.put(e.getKey(), XBaseOther.sep(a.cfgs, unit, w));
        }
        List<String[]> pairs = new ArrayList<>();
        String[][] want = {{"L1-xs-F", "L1 без пула"}, {"L1-tt-xs-F", "L1 без пула"}, {"L1-xs-F (9б) Хаар + LBF", "L1-xs-F"},
            {"L1-tt-xs-F (9б) Хаар + LBF", "L1-tt-xs-F"}, {"SFace", "L1-xs-F"}};
        for (String[] p : want) if (res.containsKey(p[0]) && res.containsKey(p[1])) pairs.add(p);
        XBaseOther.tables(r, res, sep, tpl.b, pairs);
    }

    // ---------------------------------------------------------------- FEI

    static void fei(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("fei-xs: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new TreeSet<>(m.keySet())));
        List<FeiProtocol.Split> splits = FeiProtocol.splits(included, seed);
        int jn = -1;
        for (int i = 0; i < FeiProtocol.NUMBERS.length; i++) if (FeiProtocol.NUMBERS[i] == PROBE_NO) jn = i;
        List<String> persons = new ArrayList<>(new TreeSet<>(feiKeys.keySet()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        Map<String, Integer> numOf = new HashMap<>();
        feiKeys.forEach((p, m) -> m.forEach((num, k) -> numOf.put(k, num)));
        // Поза |r| по YuNet экспорта (как fei_selection) – выбор модели X / X+mirror при r0 FEI.
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "fei");
        List<Sample> feiS = new ArrayList<>();
        export.data().fei().values().forEach(feiS::addAll);
        Map<String, Double> rOf = new HashMap<>();
        for (Sample s : feiS) {
            double[] row = det.get(s.person() + "/" + s.file().getFileName().toString());
            if (row == null) throw new IllegalStateException("fei-xs: нет строки YuNet для " + s.file());
            rOf.put(s.file().toString(), Math.abs(FeiDataset.pose(row)));
        }
        // Пул FERET (как (4а): люди с ≥ 2 датами, снимки по правилу eq, человек – один класс).
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        IlluminationNorm.NormParams params = IlluminationNorm.NormParams.of(settings);
        // Кадры «а»: FEI – экспорт, пул – derived/a.
        Map<String, Mat> fr = new LinkedHashMap<>(XBaseEval.frames(export, feiS, null));
        for (String k : poolKeys) fr.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval[] xa = {XBaseAbl.vectors(XBaseAbl.Sch.BASE, 0, fr, params), XBaseAbl.vectors(XBaseAbl.Sch.BASE, 1, fr, params)};
        fr.values().forEach(Mat::release);
        // (9б): кадры по пяти точкам LBF (Хаар – лицо); кэши haar.tsv, lbf.tsv; FERET – кэш derived/a_lbf.
        XBaseAbl.Common cm = XBaseAbl.common(dir, feiKeys, selLines, settings);
        Set<String> feiKeySet = new TreeSet<>(rOf.keySet());
        XBaseAbl.Src fsrc = XBaseAbl.feiSrc(export, dir, feiKeySet, cm, settings);
        XBaseAbl.Src psrc = XBaseAbl.feretSrc(su, cm, poolKeys);
        XBaseAbl.Src src = new XBaseAbl.Src();
        src.putAll(fsrc);
        src.putAll(psrc);
        Map<String, Mat[]> f3 = new LinkedHashMap<>(XBaseAbl.frames3(fsrc, feiKeySet, cm.th));
        f3.putAll(XBaseAbl.frames3(psrc, poolKeys, cm.th));
        Map<String, Mat> frl = new LinkedHashMap<>();
        for (Map.Entry<String, Mat[]> e : f3.entrySet()) {
            frl.put(e.getKey(), e.getValue()[2] != null ? e.getValue()[2] : XBaseAbl.dummy());
            for (int q = 0; q < 2; q++) if (e.getValue()[q] != null) e.getValue()[q].release();
        }
        XBaseEval[] xl = {XBaseAbl.vectors(XBaseAbl.Sch.LBF, 0, frl, params), XBaseAbl.vectors(XBaseAbl.Sch.LBF, 1, frl, params)};
        frl.values().forEach(Mat::release);
        Lbf lbf = new Lbf(src, cm.th, XBaseAbl.lbfOk(src));
        // SFace: кадры alignCrop экспорта (sface/fei).
        Path model = Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
        XBaseEval xs = new XBaseEval();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (Sample s : feiS) {
            Mat c = export.frame(s, Variant.SFACE);
            if (c == null) throw new IllegalStateException("fei-xs: нет кадра SFace " + s.file());
            sf.put(s.file().toString(), c);
        }
        xs.sfaceVectors(sf, FaceRecognizerSF.create(model.toString(), ""));
        sf.values().forEach(Mat::release);
        List<Row> rows = List.of(new Row("L1 без пула", xa[0], V.L1, false, false), new Row("L1-xs-F", xa[0], V.L1, true, false),
                new Row("L1-tt-xs-F", xa[1], V.L1, true, false), new Row("L1-xs-F (9б) Хаар + LBF", xl[0], V.L1, true, true),
                new Row("L1-tt-xs-F (9б) Хаар + LBF", xl[1], V.L1, true, true), new Row("SFace", xs, V.SFACE, false, false));
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.seed = seed;
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.params = params;
        tpl.r0 = r0;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "fei-xs; seed %d; r0 %s; проба №%d; пул %d; R_LBF %s; r0_LBF %s; %s", seed, r0, PROBE_NO,
                pool.size(), cm.th.rLbf(), cm.th.r0Lbf(), cm.lbfSig));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_fei_xs"), fp);
        Run run = new Run();
        StringBuilder check = new StringBuilder();
        for (FeiProtocol.Split sp : splits) {
            Map<String, List<String>> gal = new LinkedHashMap<>();
            List<Probe> probes = new ArrayList<>();
            for (int i = 0; i < sp.gallery().size(); i++) {
                String p = sp.gallery().get(i);
                gal.put(p, FeiProtocol.trainNumbers(jn).stream().map(n -> feiKeys.get(p).get(n)).toList());
                probes.add(new Probe(FeiScores.OWN, p, PROBE_NO, feiKeys.get(p).get(PROBE_NO), i));
            }
            for (String role : new String[] {FeiScores.THR, FeiScores.CTRL}) {
                for (String p : role.equals(FeiScores.THR) ? sp.threshold() : sp.control()) {
                    for (Map.Entry<Integer, String> e : new TreeMap<>(feiKeys.get(p)).entrySet()) probes.add(new Probe(role, p, e.getKey(), e.getValue(), -1));
                }
            }
            check.append(String.format(Locale.ROOT, "разбиение %d: своих %d (проба №%d), пороговых %d человек, контрольных %d человек, когорта %d; проб %d%n", sp.index(),
                    sp.gallery().size(), PROBE_NO, sp.threshold().size(), sp.control().size(), sp.cohort().size(), probes.size()));
            List<List<String>> cohort = new ArrayList<>();
            for (String p : sp.cohort()) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
            List<List<String>> parts = XBaseOther.cvParts(sp.gallery(), seed, XBaseEval.CV_S + sp.index());
            XBaseEval.Gt g = XBaseAbl.copy(tpl, cohort, rOf, r0);
            g.own = sp.gallery();
            g.ctrl = sp.control();
            g.persons = persons;
            g.pIdx = pIdx;
            g.imageNo = numOf;
            for (Row rw : rows) {
                List<List<String>> train = new ArrayList<>(cohort);
                if (rw.pool()) train.addAll(poolCls);
                row(rw, g, gal, probes, parts, train, sp.index(), lbf, cp, run, String.format(Locale.ROOT, "s%d_%s", sp.index(), rw.label().replaceAll("[^A-Za-z0-9]", "")),
                        "fei-xs");
            }
        }
        // Сверка с fei_630f21a (FeiMethods, 40 конфигураций): роли конфигурации (своя проба, пороговые, контроль) и строки L1 без пула
        // (5-lbp-mlda и +mirror; столбец j – номер конфигурации – у нас 0) – побайтно, иначе остановка.
        Path ref = dir.resolve("reports").resolve("fei_630f21a").resolve("fei_scores.tsv.gz");
        Map<Integer, java.util.TreeSet<String>> refRoles = new HashMap<>();
        Map<Integer, java.util.TreeSet<String>> refL1 = new HashMap<>();
        try (java.io.BufferedReader in = new java.io.BufferedReader(new java.io.InputStreamReader(new java.util.zip.GZIPInputStream(java.nio.file.Files.newInputStream(ref),
                1 << 16), java.nio.charset.StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                String[] c = l.split("\t", -1);
                if (c.length < 4 || !(c[0].equals(V.L1.id) || c[0].equals(V.L1.id + FeiMethods.MIRROR)) || !c[2].equals(Integer.toString(jn))) continue;
                int sIdx = Integer.parseInt(c[1]);
                if (c[0].equals(V.L1.id)) refRoles.computeIfAbsent(sIdx, x -> new java.util.TreeSet<>()).add(c[3] + "\t" + c[4] + "\t" + c[5]);
                c[2] = "0";
                refL1.computeIfAbsent(sIdx, x -> new java.util.TreeSet<>()).add(String.join("\t", c));
            }
        }
        for (FeiProtocol.Split sp : splits) {
            java.util.TreeSet<String> mineRoles = new java.util.TreeSet<>();
            java.util.TreeSet<String> mineL1 = new java.util.TreeSet<>();
            for (String l : run.raw.get(String.format(Locale.ROOT, "s%d_L1", sp.index())).split("\n")) {
                String[] c = l.split("\t", -1);
                if (c.length < 4 || c[3].startsWith(FeiMethods.CV_ROLE)) continue;
                if (c[0].equals(V.L1.id)) mineRoles.add(c[3] + "\t" + c[4] + "\t" + c[5]);
                mineL1.add(l);
            }
            boolean rolesOk = mineRoles.equals(refRoles.get(sp.index()));
            boolean l1Ok = mineL1.equals(refL1.get(sp.index()));
            check.append(String.format(Locale.ROOT, "сверка с fei_630f21a, разбиение %d: роли (%d проб) – %s; строки L1 без пула (%d) – %s%n", sp.index(), mineRoles.size(),
                    rolesOk ? "совпали" : "НЕ СОВПАЛИ", mineL1.size(), l1Ok ? "совпали" : "НЕ СОВПАЛИ"));
            if (!rolesOk || !l1Ok) throw new IllegalStateException("fei-xs: сверка с fei_630f21a не прошла\n" + check);
        }
        // Метод Прокаева (оба комплекта Д10) – из готовых файлов оценок fei-prokaev, конфигурации с пробой №11.
        Map<String, List<ProkaevOther.FCfg>> pk = new LinkedHashMap<>();
        Map<String, List<FeiDecompose.Cfg>> pkSep = new LinkedHashMap<>();
        StringBuilder pkFiles = new StringBuilder();
        for (String[] t : new String[][] {{"_face0.700", "метод Прокаева (Б2-К), К-лицо"}, {"_plain", "метод Прокаева (Б2-К), К-обычный"}}) {
            Path f = out.resolve("xbase_fei_prokaev" + t[0] + "_scores.tsv.gz");
            pkFiles.append(f).append(" (").append(FeiThreshold.sha(f)).append(")\n");
            Map<String, String> bl = XBaseSubsp.blocks(f);
            for (FeiProtocol.Split sp : splits) {
                String key = "FEI, проход " + (sp.index() * FeiProtocol.NUMBERS.length + jn + 1) + ", " + ProkaevEval.MAIN;
                String ls = bl.get(key);
                if (ls == null) throw new IllegalStateException("fei-xs: нет блока «" + key + "» в " + f);
                ls = XBaseAbl.abs(ls);
                pkSep.computeIfAbsent(t[1], x -> new ArrayList<>()).add(ProkaevN.cfg(ls, pIdx));
                ProkaevOther.FCfg c = new ProkaevOther.FCfg();
                List<double[]> own = new ArrayList<>();
                List<double[]> thr = new ArrayList<>();
                List<double[]> ctrl = new ArrayList<>();
                for (String l : ls.split("\n")) {
                    if (l.isEmpty()) continue;
                    FeiScores.Row r = FeiScores.parse(l);
                    double x = FeiScores.ratio(r, null);
                    int pi = pIdx.get(r.person());
                    if (r.own()) own.add(new double[] {pi, x, r.bestId().equals(r.trueId()) ? 1 : 0});
                    else (r.role().equals(FeiScores.THR) ? thr : ctrl).add(new double[] {pi, x});
                }
                thr.sort((a, z) -> Double.compare(a[1], z[1]));
                c.ownP = own.stream().mapToInt(a -> (int) a[0]).toArray();
                c.ownR = own.stream().mapToDouble(a -> a[1]).toArray();
                c.ownOk = new boolean[own.size()];
                for (int i = 0; i < own.size(); i++) c.ownOk[i] = own.get(i)[2] == 1;
                c.thrP = thr.stream().mapToInt(a -> (int) a[0]).toArray();
                c.thrR = thr.stream().mapToDouble(a -> a[1]).toArray();
                c.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
                c.ctrlR = ctrl.stream().mapToDouble(a -> a[1]).toArray();
                pk.computeIfAbsent(t[1], x -> new ArrayList<>()).add(c);
            }
        }
        // Отчёт.
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, сводка С1 (5): FEI, вариант Б (XBaseEval fei-xs) – L1, L1-xs-F, L1-tt-xs-F, (9б), SFace, метод Прокаева\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/fei, sface/fei, fei_selection.tsv, "
                + "YuNet – detections.tsv); пул – Color FERET (derived/a, a_lbf; имён снимков нет); SFace – ").append(model).append(" (").append(FeiThreshold.sha(model))
                .append(")\n");
        r.append(String.format(Locale.ROOT, "Вариант Б (решение Хозяина 21.10.2026): 5 разбиений п. 3а, по одной конфигурации на разбиение – проба своего №%d (анфас), "
                + "галерея – остальные 7 снимков (№4–7, 12–14); пороговые, контроль, когорта – как в fei_bootstrap (все снимки людей). От полного протокола "
                + "(40 конфигураций, все 8 проб) отличается: проб своих в 8 раз меньше (500 вместо 4000), только анфас – интервалы шире; пороговые и контроль – "
                + "те же снимки, но в 5 конфигурациях, а не в 40.%n", PROBE_NO));
        r.append(String.format(Locale.ROOT, "L1-xs-F, L1-tt-xs-F: пул FERET %d человек (люди с ≥ 2 датами, снимки по правилу eq, человек – один класс), схема (Wа) – W "
                + "и модели частей обучаются на всех людях галереи, когорте и пуле; L1 без пула – только когорта. Модель X / X+mirror – по |r| ≤ r0 = %s. (9б): "
                + "Хаар (лицо) + LBF, кадр по пяти точкам LBF; в галерее, когорте и пуле – снимки с точками LBF; проба без точек или с |r_LBF| > R_LBF = %.3f – "
                + "отказ; модель – по r0_LBF = %.3f. SFace – шаблон по снимкам галереи (минимум 2), без обучения. П3 и П2 – «чужие» галереи (5 частей своих, "
                + "seed faces.seed + 8000 + 100 + s).%n", pool.size(), r0, cm.th.rLbf(), cm.th.r0Lbf()));
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям FEI: %d человек, B = %d, seed %d (правила после 1bc4311, 2ea947f, 9f8c54f).%n",
                persons.size(), tpl.b, seed + FeiBootstrap.SEED_OFFSET));
        r.append("\n=== Роли (как fei_bootstrap) ===\n").append(check);
        tables(r, run, persons.size(), tpl);
        thrTable(r, run.thr, persons.size(), tpl.b, seed, true);
        r.append("\n=== Метод Прокаева (Б2-К, k = 6, l = 3, гости 10), оба комплекта Д10 – по готовым файлам оценок fei-prokaev тех же 5 конфигураций ===\n");
        r.append(pkFiles);
        r.append("П3 и П2 у метода Прокаева на FEI не считались (нет частей «чужих» галереи); разделимость – порог по контрольным чужим каждой конфигурации.\n");
        r.append("строка | верный первый, % | разд. FAR 1 %, % (фактический FAR, %) | EER, % | ошибка своих при пороге по пороговым, % [2,5; 97,5] | FAR контроля, %\n");
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons.size(), tpl.b, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons.size());
        for (String label : pk.keySet()) {
            long good = 0;
            long all = 0;
            for (ProkaevOther.FCfg c : pk.get(label)) {
                all += c.ownOk.length;
                for (boolean x : c.ownOk) if (x) good++;
            }
            XBaseOther.Sep s = XBaseOther.sep(pkSep.get(label), unit, w);
            double[] p = ProkaevOther.feiErr(pk.get(label), unit, 0.01);
            double[] rep = new double[w.length];
            for (int q = 0; q < w.length; q++) rep[q] = ProkaevOther.feiErr(pk.get(label), w[q], 0.01)[0];
            r.append(String.format(Locale.ROOT, "%s | %.1f | %.1f (%.2f) | %.1f | %s | %.2f%n", label, 100.0 * good / all, s.err[0], s.far[0], s.eer[3],
                    XBaseSepLtp.cell(p[0], rep), p[1]));
        }
        r.append(String.format(Locale.ROOT, "%nОбучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", run.fitS, tpl.threads));
        XBaseOther.writeScores(out.resolve("xbase_fei_xs_scores.tsv.gz"), "# FEI, С1 (5) вариант Б (XBaseC1 fei-xs); блоки – разбиение, строка\n# Код: коммит "
                + FarMethods.commit() + "\n", run.lines.toString());
        r.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_fei_xs.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    // ---------------------------------------------------------------- своя база

    static void own(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("own-xs: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        // Контроль – односессионные люди GT (роли gt, отбор |r| ≤ R), когорта – FEI 20 (как GT).
        Map<String, double[]> gtDet = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rOld = new HashMap<>();
        gtDet.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        // Своя база: моменты, кадры экспорта, YuNet экспорта (порог 0,7).
        List<OwnDataset.Person> ownP = export.data().own().persons();
        List<List<OwnDataset.Moment>> moments = export.data().moments();
        Map<String, double[]> ownDet = XBaseMpPts.exportRows(dir, "own");
        Map<String, Sample> ownS = new LinkedHashMap<>();
        Map<String, String> pkOf = new HashMap<>();
        Map<String, Integer> numOf = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>(g0.rOf);
        int unprocessed = 0;
        for (int i = 0; i < ownP.size(); i++) {
            int n = 0;
            for (OwnDataset.Moment m : moments.get(i)) {
                for (OwnDataset.Frame f : m.frames()) {
                    String k = f.file().toString();
                    // Кадры с пометкой «брак» детектор не обрабатывал (строк в detections.tsv нет) – в протокол не входят.
                    if (export.found().get(k) == null) {
                        unprocessed++;
                        continue;
                    }
                    ownS.put(k, new Sample(GalleryEvaluation.Base.OWN, ownP.get(i).name(), f.file(), ' '));
                    String pk = ownP.get(i).name() + "/" + f.file().getFileName().toString();
                    pkOf.put(k, pk);
                    numOf.put(k, n++);
                    double[] row = ownDet.get(pk);
                    if (row != null) rOf.put(k, Math.abs(FeiDataset.pose(row)));
                }
            }
        }
        for (Sample s : g0.samples.values()) numOf.put(s.file().toString(), XBaseEval.image(s));
        // Кадры «а»: своя база и GT – экспорт; когорта FEI – экспорт; пул FERET – derived/a.
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        Set<String> cohortKeys = new HashSet<>();
        g0.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        IlluminationNorm.NormParams params = IlluminationNorm.NormParams.of(settings);
        List<Sample> ownFound = ownS.values().stream().filter(s -> export.found().get(s.file().toString())).toList();
        Map<String, Mat> fr = new LinkedHashMap<>(XBaseEval.frames(export, ownFound, null));
        fr.putAll(XBaseEval.frames(export, g0.samples.values().stream().sorted((u, z) -> u.file().toString().compareTo(z.file().toString())).toList(), null));
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (String k : poolKeys) fr.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval[] xa = {XBaseAbl.vectors(XBaseAbl.Sch.BASE, 0, fr, params), XBaseAbl.vectors(XBaseAbl.Sch.BASE, 1, fr, params)};
        fr.values().forEach(Mat::release);
        // (9б): своя база – вход 0,25 (как abl-haar и abl-lbf), GT – gt_db, когорта – FEI, пул – FERET.
        XBaseAbl.Common cm = XBaseAbl.common(dir, feiKeys, selLines, settings);
        HaarFrames.Set ob = cm.b.own();
        Map<String, LbfFrames.Lm> ol = cm.lbf.get(ob.name());
        Path ownDir = Paths.get(settings.loadFacesOwnDir());
        XBaseAbl.Src osrc = new XBaseAbl.Src();
        Map<String, Path> rawOf = new HashMap<>();
        OwnDataset raw = OwnDataset.load(ownDir);
        for (OwnDataset.Person p : raw.persons()) for (OwnDataset.Moment m : p.moments()) for (OwnDataset.Frame f : m.frames()) {
            String base = f.file().getFileName().toString();
            rawOf.put(p.name() + "/" + base.substring(0, base.lastIndexOf('.')), f.file());
        }
        for (Map.Entry<String, String> e : pkOf.entrySet()) {
            HaarFrames.Det d = ob.haar().get(e.getValue());
            if (d == null) throw new IllegalStateException("own-xs: нет Хаара своей базы для " + e.getValue());
            // Координаты Хаара и LBF – в масштабе исходного снимка; снимок подаётся уменьшенным до 0,25 (как при детекции) – координаты × 0,25.
            double sc = OwnEvaluation.INPUT_SCALE;
            osrc.haar.put(e.getKey(), new HaarFrames.Det(d.rects(), mul(d.face(), sc), mul(d.left(), sc), mul(d.right(), sc)));
            LbfFrames.Lm lm = ol.get(e.getValue());
            if (lm != null) {
                double[][] p = new double[lm.p().length][];
                for (int i = 0; i < p.length; i++) p[i] = mul(lm.p()[i], sc);
                lm = new LbfFrames.Lm(p);
            }
            osrc.lbf.put(e.getKey(), lm);
            Path rf = rawOf.get(e.getValue());
            if (rf == null) throw new IllegalStateException("own-xs: нет исходного снимка " + e.getValue());
            osrc.load.put(e.getKey(), y -> {
                Mat c = Imgcodecs.imread(rf.toString(), Imgcodecs.IMREAD_COLOR);
                Mat s = new Mat();
                Imgproc.resize(c, s, new Size(), OwnEvaluation.INPUT_SCALE, OwnEvaluation.INPUT_SCALE, Imgproc.INTER_AREA);
                c.release();
                return s;
            });
        }
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        XBaseAbl.Src gsrc = new XBaseAbl.Src();
        Map<String, LbfFrames.Lm> gl = cm.lbf.get(cm.b.gt().name());
        for (Sample s : g0.samples.values()) {
            String k = s.file().toString();
            String pk = s.person() + "/" + s.file().getFileName().toString();
            gsrc.haar.put(k, cm.b.gt().haar().get(pk));
            gsrc.lbf.put(k, gl.get(pk));
            gsrc.load.put(k, y -> Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR));
        }
        XBaseAbl.Src csrc = XBaseAbl.feiSrc(export, dir, cohortKeys, cm, settings);
        XBaseAbl.Src psrc = XBaseAbl.feretSrc(su, cm, poolKeys);
        XBaseAbl.Src src = new XBaseAbl.Src();
        for (XBaseAbl.Src s : List.of(osrc, gsrc, csrc, psrc)) src.putAll(s);
        Map<String, Mat[]> f3 = new LinkedHashMap<>(XBaseAbl.frames3(osrc, new TreeSet<>(pkOf.keySet()), cm.th));
        f3.putAll(XBaseAbl.frames3(gsrc, new TreeSet<>(gsrc.haar.keySet()), cm.th));
        f3.putAll(XBaseAbl.frames3(csrc, new TreeSet<>(cohortKeys), cm.th));
        f3.putAll(XBaseAbl.frames3(psrc, poolKeys, cm.th));
        Map<String, Mat> frl = new LinkedHashMap<>();
        for (Map.Entry<String, Mat[]> e : f3.entrySet()) {
            frl.put(e.getKey(), e.getValue()[2] != null ? e.getValue()[2] : XBaseAbl.dummy());
            for (int q = 0; q < 2; q++) if (e.getValue()[q] != null) e.getValue()[q].release();
        }
        XBaseEval[] xl = {XBaseAbl.vectors(XBaseAbl.Sch.LBF, 0, frl, params), XBaseAbl.vectors(XBaseAbl.Sch.LBF, 1, frl, params)};
        frl.values().forEach(Mat::release);
        Lbf lbf = new Lbf(src, cm.th, XBaseAbl.lbfOk(src));
        // SFace.
        Path model = Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
        XBaseEval xs = new XBaseEval();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (Sample s : ownFound) sf.put(s.file().toString(), export.frame(s, Variant.SFACE));
        for (Sample s : g0.samples.values()) sf.put(s.file().toString(), export.frame(s, Variant.SFACE));
        xs.sfaceVectors(sf, FaceRecognizerSF.create(model.toString(), ""));
        sf.values().forEach(Mat::release);
        List<Row> rows = List.of(new Row("L1 без пула", xa[0], V.L1, false, false), new Row("L1-xs-F", xa[0], V.L1, true, false),
                new Row("L1-tt-xs-F", xa[1], V.L1, true, false), new Row("L1-xs-F (9б) Хаар + LBF", xl[0], V.L1, true, true),
                new Row("L1-tt-xs-F (9б) Хаар + LBF", xl[1], V.L1, true, true), new Row("SFace", xs, V.SFACE, false, false));
        // Люди бутстрепа: свои 6 и контрольные GT.
        List<String> own = ownP.stream().map(OwnDataset.Person::name).toList();
        List<String> persons = new ArrayList<>(own);
        persons.addAll(g0.ctrl);
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        XBaseEval.Gt tpl = XBaseAbl.copy(g0, g0.cohort, rOf, r0);
        tpl.own = own;
        tpl.persons = persons;
        tpl.pIdx = pIdx;
        tpl.imageNo = numOf;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "own-xs; seed %d; r0 %s; R %s; пул %d; R_LBF %s; r0_LBF %s; %s; ротаций %s", g0.seed, r0, rLimit,
                pool.size(), cm.th.rLbf(), cm.th.r0Lbf(), cm.lbfSig, System.getenv("FACES_OWN_ROTATIONS")));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_own_xs"), fp);
        Run run = new Run();
        int[] detRej = new int[2];
        StringBuilder check = new StringBuilder();
        // Ротаций – FACES_OWN_ROTATIONS (по умолчанию все 12; решение Хозяина 22.10.2026 для С1 – первые 3).
        int rotations = System.getenv("FACES_OWN_ROTATIONS") == null ? OwnDetect.CONFIGS : Integer.parseInt(System.getenv("FACES_OWN_ROTATIONS"));
        if (rotations < 1 || rotations > OwnDetect.CONFIGS) throw new IllegalStateException("own-xs: FACES_OWN_ROTATIONS – от 1 до " + OwnDetect.CONFIGS);
        for (int r = 0; r < rotations; r++) {
            Map<String, List<String>> gal = new LinkedHashMap<>();
            List<Probe> probes = new ArrayList<>();
            int ownN = 0;
            for (int i = 0; i < own.size(); i++) {
                List<OwnDataset.Moment> ms = moments.get(i);
                OwnEvaluation.Split spl = OwnEvaluation.split(ms, r % ms.size(), false);
                List<String> ks = new ArrayList<>();
                for (int j : spl.train()) for (OwnDataset.Frame f : ms.get(j).frames()) {
                    String k = f.file().toString();
                    if (f.quality() >= OwnDataset.FAIR && Boolean.TRUE.equals(export.found().get(k))) ks.add(k);
                }
                gal.put(own.get(i), ks);
                OwnDataset.Frame rep = ms.get(spl.control()).representative();
                String k = rep.file().toString();
                if (Boolean.TRUE.equals(export.found().get(k))) {
                    probes.add(new Probe(FeiScores.OWN, own.get(i), numOf.get(k), k, i));
                    ownN++;
                } else if (r == 0) {
                    detRej[0]++;
                }
            }
            for (String p : g0.ctrl) for (List<String> ks : g0.byPerson.get(p).values()) for (String k : ks) probes.add(new Probe(FeiScores.CTRL, p, numOf.get(k), k, -1));
            check.append(String.format(Locale.ROOT, "ротация %d: свои – проб %d, снимков галереи %s; контрольных проб %d%n", r, ownN,
                    gal.values().stream().map(l -> Integer.toString(l.size())).toList(), probes.size() - ownN));
            List<List<String>> parts = XBaseOther.cvParts(own, g0.seed, XBaseEval.CV_S + r);
            XBaseEval.Gt g = XBaseAbl.copy(tpl, g0.cohort, rOf, r0);
            g.own = own;
            g.ctrl = g0.ctrl;
            g.persons = persons;
            g.pIdx = pIdx;
            g.imageNo = numOf;
            for (Row rw : rows) {
                List<List<String>> train = new ArrayList<>(g0.cohort);
                if (rw.pool()) train.addAll(poolCls);
                row(rw, g, gal, probes, parts, train, r, lbf, cp, run, String.format(Locale.ROOT, "r%d_%s", r, rw.label().replaceAll("[^A-Za-z0-9]", "")), "own-xs");
            }
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, сводка С1 (5): своя база внутри сессии, справочно (XBaseEval own-xs) – L1, L1-xs-F, L1-tt-xs-F, (9б), "
                + "SFace; отказы детекторов\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/own, sface/own, own_moments.tsv, a/gt, "
                + "detections.tsv); снимки своей базы – ").append(ownDir).append(" ((9б), вход 0,25); пул – Color FERET (имён снимков нет)\n");
        r.append("Оговорка: своя база – одна сессия, 6 человек; ошибка своих оптимистична (как во всех отчётах своей базы). При 6 своих П3 и П2 "
                + "вырождены (части «чужих» – по одному человеку) – справочно; главная мера – разделимость при FAR контроля 1 % и EER.\n");
        r.append(String.format(Locale.ROOT, "Протокол: ротаций %d из 12 (первые; решение Хозяина 22.10.2026 – 3) (контроль – представитель момента r mod m, галерея – кадры «+» и «+-» 5 моментов по кругу); "
                + "контроль – односессионные люди GT (%d человек, отбор |r| ≤ R = %.3f, все снимки); когорта – FEI 20; пул FERET %d человек; модель X / "
                + "X+mirror по |r| ≤ r0 = %s; (9б) – как в fei-xs (R_LBF = %.3f, r0_LBF = %.3f).%n", rotations, g0.ctrl.size(), rLimit, pool.size(), r0, cm.th.rLbf(), cm.th.r0Lbf()));
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: %d (свои 6 и контроль GT), B = %d, seed %d.%n", persons.size(), tpl.b,
                g0.seed + FeiBootstrap.SEED_OFFSET));
        r.append("\n=== Роли ===\n").append(check);
        r.append(OwnDetect.text(export, ownS, pkOf, ob, ownDet, moments, own));
        r.append(String.format(Locale.ROOT, "Кадров с пометкой «брак», не обработанных детектором (нет в detections.tsv) – %d; в протокол и счёт отказов не входят.%n", unprocessed));
        tables(r, run, persons.size(), tpl);
        thrTable(r, run.thr, persons.size(), tpl.b, g0.seed, false);
        r.append(String.format(Locale.ROOT, "%nОбучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", run.fitS, tpl.threads));
        XBaseOther.writeScores(out.resolve("xbase_own_xs_scores.tsv.gz"), "# Своя база, С1 (5) справочно (XBaseC1 own-xs); блоки – ротация, строка\n# Код: коммит "
                + FarMethods.commit() + "\n", run.lines.toString());
        r.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_own_xs.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    static double[] mul(double[] a, double s) {
        if (a == null) return null;
        double[] b = new double[a.length];
        for (int i = 0; i < a.length; i++) b[i] = a[i] * s;
        return b;
    }

    /** Отказы детекторов своей базы: YuNet (экспорт, порог 0,7) и Хаар (haar.tsv своей базы), ложные рамки Хаара – IoU с YuNet &lt; 0,3. */
    static final class OwnDetect {
        static final int CONFIGS = 12;

        static String text(FarExport.Loaded export, Map<String, Sample> ownS, Map<String, String> pkOf, HaarFrames.Set ob, Map<String, double[]> ownDet,
                           List<List<OwnDataset.Moment>> moments, List<String> own) {
            int[] all = count(ownS.keySet(), export, pkOf, ob, ownDet);
            Set<String> probeKeys = new TreeSet<>();
            for (List<OwnDataset.Moment> ms : moments) for (OwnDataset.Moment m : ms) if (m.representative() != null && pkOf.containsKey(m.representative().file().toString())) probeKeys.add(m.representative().file().toString());
            int[] pr = count(probeKeys, export, pkOf, ob, ownDet);
            StringBuilder t = new StringBuilder("\n=== Отказы детекторов своей базы (без распознавания) ===\n");
            t.append("набор | снимков | YuNet: отказ | Хаар: нет лица | Хаар: лицо без двух глаз | Хаар: ложная рамка (IoU с YuNet < 0,3) | Хаар или YuNet: отказ\n");
            t.append(String.format(Locale.ROOT, "все снимки (%d человек) | %d | %d | %d | %d | %d | %d%n", own.size(), all[0], all[1], all[2], all[3], all[4], all[5]));
            t.append(String.format(Locale.ROOT, "пробы (представители моментов) | %d | %d | %d | %d | %d | %d%n", pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]));
            t.append("Проба с отказом YuNet в распознавание не входит (в таблицах ниже – только найденные); у (9б) отказ Хаара или LBF – отказ пробы.\n");
            return t.toString();
        }

        /** {снимков, отказ YuNet, нет лица Хаара, лицо без двух глаз, ложная рамка, отказ хотя бы одного}. */
        static int[] count(Set<String> keys, FarExport.Loaded export, Map<String, String> pkOf, HaarFrames.Set ob, Map<String, double[]> ownDet) {
            int[] c = new int[6];
            for (String k : keys) {
                c[0]++;
                boolean y = Boolean.TRUE.equals(export.found().get(k));
                HaarFrames.Det d = ob.haar().get(pkOf.get(k));
                double[] row = ownDet.get(pkOf.get(k));
                if (!y) c[1]++;
                if (!d.found()) c[2]++;
                else if (!d.both()) c[3]++;
                if (d.found() && row != null && HaarFrames.iou(d.face(), row) < HaarFrames.IOU_FALSE) c[4]++;
                if (!y || !d.found()) c[5]++;
            }
            return c;
        }
    }
}
