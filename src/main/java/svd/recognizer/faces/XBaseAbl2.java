package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Повтор (4а) на новой базе (постановка Жорика №1 от 27.10.2026, план принят с поправкой №2, подтверждён Хозяином): база – (П2)
 * LBP + POEM · 1/√3 (L1-tt-xs-F 9×9, L1-xs-F 7×7), MLDA с пулом FERET, пол c = 1, 1 − cos, d₁/d₂, модель X / X+mirror по |r|, порог
 * (Т2). Строки (по одной отключается): (1) без пула в обучении W; (2) без корня гистограмм (множитель 1/√3 тот же); (3) у L1-tt-xs-F
 * без Тан – Триггс, справочно у L1-xs-F с Тан – Триггс (сетка кандидата); (4) d₁ (второе расстояние – постоянная XBaseAbl.ABS_C);
 * (5) r0 = +∞ (всегда X); (6) r0 = −1 (всегда X+mirror); (7) евклид на той же W; (8) без множителя 1/√3. Обучений на признак – 5
 * (база, (1), (2), (3), (8)); (4)–(7) – на модели базы. Выбор по пулу (схема pool-rule): «отключено − база» по критерию: больше +1 п. п.
 * – «нужен», меньше −1 п. п. – «мешает», иначе «лишний»; на своих – база и строки, «лишние» или «мешающие» хоть в одном из шести
 * случаев.
 *
 * @author ssv
 */
final class XBaseAbl2 {

    /** Наборы векторов: 0 – база (П2), 1 – без корня, 2 – без множителя, 3 – другая нормировка входа. */
    static final int NX = 4;
    /** Обучения: 0 – база, 1 – без пула, 2 – без корня, 3 – нормировка, 4 – без множителя; набор векторов обучения. */
    static final int NF = 5;
    static final int[] FIT_X = {0, 0, 1, 3, 2};
    static final int ROWS = 9;
    static final int BASE = 0;
    /** Обучение, из которого берётся строка. */
    static final int[] ROW_FIT = {0, 1, 2, 3, 0, 0, 0, 0, 4};
    static final String SELECT = "xbase_pool_abl2_select.tsv";
    static final double FERET_TO_GT = XBaseNap.FERET_TO_GT;
    static final int SMOKE2_BLOCKS = 2;
    static final String NEED = "нужен";
    static final String SPARE = "лишний";
    static final String HARM = "мешает";

    private XBaseAbl2() {
    }

    /** Пробный режим FACES_ABL2_SMOKE: 0 – нет; 1 – малые данные, весь путь; 2 – первая группа в полном размере – замер. */
    static int smoke() {
        String v = System.getenv("FACES_ABL2_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    static String rname(int f, int r) {
        return switch (r) {
            case 0 -> "база (П2)";
            case 1 -> "(1) без пула";
            case 2 -> "(2) без корня";
            case 3 -> f == 1 ? "(3) без Тан – Триггс" : "(3) с Тан – Триггс (справочно)";
            case 4 -> "(4) d₁";
            case 5 -> "(5) r0 = +∞ (всегда X)";
            case 6 -> "(6) r0 = −1 (всегда X+mirror)";
            case 7 -> "(7) евклид";
            case 8 -> "(8) без 1/√3";
            default -> throw new IllegalArgumentException("abl2: строка " + r);
        };
    }

    // ---------------------------------------------------------------- признаки

    /** Нормировка входа набора векторов x признака f: у набора 3 – другая (L1-tt-xs-F – без нормировки, L1-xs-F – Тан – Триггс). */
    static IlluminationNorm norm(int f, int x) {
        if (x != 3) return XBaseDates.NORMS[f];
        return f == 1 ? IlluminationNorm.NONE : IlluminationNorm.TAN_TRIGGS;
    }

    /**
     * Вектор набора x по кадру: сырые гистограммы LBP (сетка кандидата) и POEM (та же сетка) по входу нормировки norm(f, x); база и
     * набор 3 – корень обеих частей и PoemFeature.join (как XBasePoem.feature, вариант 2); набор 1 – PoemFeature.join без корня;
     * набор 2 – корень обеих частей подряд, без множителя.
     */
    static double[] feature(Mat gray8, int f, int x, IlluminationNorm.NormParams params) {
        IlluminationNorm nm = norm(f, x);
        double[] img = PoemFeature.input(gray8, nm, params);
        double[] lh = LbpFeature.histogram(gray8, nm, params, XBaseDates.PARAMS[f]);
        double[] ph = PoemFeature.histogram(img, gray8.cols(), gray8.rows(), XBasePoem.grid(f));
        return switch (x) {
            case 0, 3 -> PoemFeature.join(LbpFeature.sqrt(lh), LbpFeature.sqrt(ph));
            case 1 -> PoemFeature.join(lh, ph);
            case 2 -> {
                double[] a = LbpFeature.sqrt(lh);
                double[] b = LbpFeature.sqrt(ph);
                double[] v = new double[a.length + b.length];
                System.arraycopy(a, 0, v, 0, a.length);
                System.arraycopy(b, 0, v, a.length, b.length);
                yield v;
            }
            default -> throw new IllegalArgumentException("abl2: набор " + x);
        };
    }

    /** Набор векторов x признака f (база – XBasePoem.vectors, вариант 2, побайтно прежняя); отражения – FaceMirror. */
    static XBaseEval vectors(Map<String, Mat> fr, IlluminationNorm.NormParams params, int f, int x) {
        if (x == 0) return XBasePoem.vectors(fr, params, f, new int[] {XBaseNap.P2})[XBaseNap.P2];
        XBaseEval e = new XBaseEval();
        e.vec.put(V.L1, new HashMap<>());
        e.mir.put(V.L1, new HashMap<>());
        for (Map.Entry<String, Mat> en : fr.entrySet()) {
            Mat m = FaceMirror.mirror(en.getValue());
            e.vec.get(V.L1).put(en.getKey(), feature(en.getValue(), f, x, params));
            e.mir.get(V.L1).put(en.getKey(), feature(m, f, x, params));
            m.release();
        }
        return e;
    }

    // ---------------------------------------------------------------- строки

    /** Gt обучения fit: у (1) – когорта без классов пула (первые poolFrom). */
    static XBaseEval.Gt fitGt(XBaseEval.Gt g, int fit, int poolFrom) {
        if (fit != 1) return g;
        return XBaseAbl.copy(g, new ArrayList<>(g.cohort.subList(0, poolFrom)), g.rOf, g.r0);
    }

    /** Gt разбора строки r: у (5), (6) – другое r0. */
    static XBaseEval.Gt rowGt(XBaseEval.Gt g, int r) {
        if (r == 5) return XBaseAbl.copy(g, g.cohort, g.rOf, Double.POSITIVE_INFINITY);
        if (r == 6) return XBaseAbl.copy(g, g.cohort, g.rOf, -1);
        return g;
    }

    /** Строки d₁/d₂ строки r по итогам обучений блока. */
    static String rowLines(int r, XBaseLbp.One[] fits) {
        XBaseLbp.One o = fits[ROW_FIT[r]];
        return switch (r) {
            case 4 -> XBaseAbl.abs(o.rows()[1]);
            case 7 -> o.rows()[0];
            default -> o.rows()[1];
        };
    }

    /** Метка по разности критериев «отключено − база», п. п. */
    static String label(double diff) {
        if (diff > XBaseRule.NEAR) return NEED;
        if (diff < -XBaseRule.NEAR) return HARM;
        return SPARE;
    }

    /** Строки для своих по файлу выбора: база и строки с меткой «лишний» или «мешает» хоть в одном случае. */
    static int[] ownRows(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("abl2: нет файла выбора " + f + " (сначала pool-abl2)");
        TreeSet<Integer> rs = new TreeSet<>(List.of(BASE));
        for (String l : Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int fi = List.of(XBaseRule.FEATS).indexOf(p[1]);
            int r = Integer.parseInt(p[2]);
            if (fi < 0 || r <= BASE || r >= ROWS || !rname(fi, r).equals(p[3]) || !List.of(NEED, SPARE, HARM).contains(p[5]))
                throw new IllegalStateException("abl2: строка «" + l + "» не из сетки");
            if (!p[5].equals(NEED)) rs.add(r);
        }
        return rs.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Обучения, нужные строкам rs (база – всегда). */
    static int[] fitsFor(int[] rs) {
        TreeSet<Integer> fs = new TreeSet<>(List.of(0));
        for (int r : rs) fs.add(ROW_FIT[r]);
        return fs.stream().mapToInt(Integer::intValue).toArray();
    }

    static final String NOTE = "База – (П2) LBP + POEM, часть POEM после корня · 1/√3 (L1-xs-F 7×7, L1-tt-xs-F 9×9), MLDA с пулом FERET, пол c = 1, 1 − cos до "
            + "центра класса,\nd₁/d₂, модель X / X+mirror по |r| ≤ r0. Строки (по одной отключается, остальное – база): (1) без пула FERET в обучении W (галерея и "
            + "когорта FEI);\n(2) без корня гистограмм (сумма 1 на блок и ориентацию, множитель 1/√3 тот же); (3) L1-tt-xs-F – без Тан – Триггс (сетка 9×9), "
            + "L1-xs-F – с Тан – Триггс (7×7, справочно);\n(4) решение по d₁ (второе расстояние – постоянная 1e6); (5) r0 = +∞ – всегда X; (6) r0 = −1 – всегда "
            + "X+mirror; (7) евклид на той же W; (8) без множителя 1/√3.\nОбучений на признак – 5 (база, (1), (2), (3), (8)); (4)–(7) – на модели базы (тот же "
            + "вызов, другой разбор).\n";

    // ---------------------------------------------------------------- pool-abl2

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        int poolFrom = s.cohort.size();
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(sm == 1 ? all.subList(0, 30) : all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        if (sm == 1) {
            sets.put("FERET-1", new TreeSet<>(new ArrayList<>(s.halves.get(1)).subList(0, 30)));
            sets.put("FERET-2", new TreeSet<>(new ArrayList<>(s.halves.get(0)).subList(0, 30)));
        }
        List<String> setNames = new ArrayList<>(sets.keySet());
        Path poemScores = s.out.resolve("xbase_pool_poem_scores.tsv.gz");
        if (!Files.exists(poemScores)) throw new IllegalStateException("pool-abl2: для сверки базы нужен " + poemScores + " (pool-poem)");
        Map<String, String> blocks = XBaseSubsp.blocks(poemScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-abl2 %s; seed %d; r0 %s; пул %d; половины %d/%d; строк %d; пробный %d",
                s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), ROWS, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_abl2_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        StringBuilder head = new StringBuilder();
        head.append("Повтор (4а) на новой базе: поочерёдное отключение, выбор по пулу FERET (XBaseEval pool-abl2)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                .append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-poem: GT – весь пул; FERET-1 – половина B (%d); FERET-2 – половина A (%d);%n5 частей "
                + "по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула; обучение W – галерея своих + "
                + "когорта FEI 20 + люди 4 частей.%n", s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET, XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; «отключено − база»: больше +%.0f п. п. – «%s», "
                + "меньше −%.0f п. п. – «%s» (взаимное мешание), иначе – «%s».%n", XBaseRule.NEAR, NEED, XBaseRule.NEAR, HARM, SPARE));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки (П2) pool-poem, побайтно ===\n");
        // Блоки: [набор] → список {q, h, Gt, галерея, пробы, тег, One[NF]}.
        List<List<Object[]>> blocksBy = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            Set<String> members = sets.get(set);
            List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
            List<Object[]> blk = new ArrayList<>();
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                if (sm == 2 && (si > 0 || blk.size() >= SMOKE2_BLOCKS)) break;
                List<String> held = new ArrayList<>(parts.get(q));
                Set<String> train = new TreeSet<>(members);
                held.forEach(train::remove);
                Collections.shuffle(held, new Random(tpl.seed + XBaseRule.HALF_SEED_OFFSET + q));
                int h0 = (held.size() + 1) / 2;
                List<List<String>> hv = List.of(new ArrayList<>(new TreeSet<>(held.subList(0, h0))), new ArrayList<>(new TreeSet<>(held.subList(h0, held.size()))));
                Map<String, List<List<String>>> trainPool = new TreeMap<>();
                for (String p : train) trainPool.put(p, s.pool.get(p));
                List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), trainPool, Set.of(), Map.of()).extra();
                for (int h = 0; h < 2; h++) {
                    if (sm == 2 && blk.size() >= SMOKE2_BLOCKS) break;
                    List<String> own = hv.get(h);
                    List<String> ctrl = hv.get(1 - h);
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                    List<List<String>> aug = new ArrayList<>(s.cohort);
                    aug.addAll(extra);
                    g.cohort = aug;
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    List<Probe> probes = new ArrayList<>();
                    for (int i = 0; i < own.size(); i++) {
                        List<List<String>> ds = s.pool.get(own.get(i));
                        gal.put(own.get(i), ds.get(0));
                        for (int d = 1; d < ds.size(); d++) for (String k : ds.get(d)) probes.add(new Probe(FeiScores.OWN, own.get(i), g.img(k), k, i));
                    }
                    for (String p : ctrl) for (List<String> d : s.pool.get(p)) for (String k : d) probes.add(new Probe(FeiScores.CTRL, p, g.img(k), k, -1));
                    blk.add(new Object[] {q, h, g, gal, probes, String.format(Locale.ROOT, "%s_q%d_h%d", set, q, h), null});
                }
            }
            blocksBy.add(blk);
        }
        double fitS = 0;
        double[] smokeS = new double[2];
        // [f][набор][блок][обучение]
        XBaseLbp.One[][][][] got = new XBaseLbp.One[2][setNames.size()][][];
        for (int f = 0; f < 2; f++) {
            for (int si = 0; si < setNames.size(); si++) got[f][si] = new XBaseLbp.One[blocksBy.get(si).size()][NF];
            // Набор векторов – внешний цикл: в памяти один набор признака.
            for (int xi = 0; xi < NX; xi++) {
                XBaseEval x = null;
                for (int si = 0; si < setNames.size(); si++) {
                    List<Object[]> blk = blocksBy.get(si);
                    List<XBaseLbp.Job> jobs = new ArrayList<>();
                    List<int[]> todo = new ArrayList<>();
                    for (int bi = 0; bi < blk.size(); bi++) {
                        Object[] b = blk.get(bi);
                        for (int fit = 0; fit < NF; fit++) {
                            if (FIT_X[fit] != xi) continue;
                            String tag = b[5] + "_f" + f + "_fit" + fit;
                            got[f][si][bi][fit] = XBaseLbp.read(cp, tag);
                            if (got[f][si][bi][fit] != null) continue;
                            if (x == null) {
                                long t0 = System.nanoTime();
                                x = vectors(fr, tpl.params, f, xi);
                                System.out.printf(Locale.ROOT, "pool-abl2: f%d x%d vectors (%.0f s)%n", f, xi, (System.nanoTime() - t0) / 1e9);
                            }
                            @SuppressWarnings("unchecked")
                            Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                            @SuppressWarnings("unchecked")
                            List<Probe> probes = (List<Probe>) b[4];
                            jobs.add(XBaseLbp.job(fitGt((XBaseEval.Gt) b[2], fit, poolFrom), x, gal, probes, null, 2 * (int) b[0] + (int) b[1], null));
                            todo.add(new int[] {bi, fit});
                        }
                    }
                    if (jobs.isEmpty()) continue;
                    long t0 = System.nanoTime();
                    List<XBaseLbp.One> done = XBaseLbp.run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    if (sm == 2) smokeS[f] += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        int[] t = todo.get(i);
                        got[f][si][t[0]][t[1]] = done.get(i);
                        XBaseLbp.save(cp, blk.get(t[0])[5] + "_f" + f + "_fit" + t[1], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-abl2: %s f%d x%d ready, %d jobs (%.0f s)%n", setNames.get(si), f, xi, jobs.size(), sec);
                }
            }
        }
        for (Mat m : fr.values()) m.release();
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        StringBuilder scores = new StringBuilder();
        // [набор][f][строка] → конфигурации
        List<List<List<List<FeiDecompose.Cfg>>>> cfgs = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            List<List<List<FeiDecompose.Cfg>>> a = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                List<List<FeiDecompose.Cfg>> c = new ArrayList<>();
                for (int r = 0; r < ROWS; r++) c.add(new ArrayList<>());
                a.add(c);
                List<Object[]> blk = blocksBy.get(si);
                for (int bi = 0; bi < blk.size(); bi++) {
                    Object[] b = blk.get(bi);
                    XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    XBaseLbp.One[] fits = got[f][si][bi];
                    String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + b[0] + ", смена " + b[1] + ", ";
                    if (sm != 1) check.append(XBaseReg.check(bp + XBasePoem.vname(f, XBaseNap.P2) + ", косинус", fits[0].rows()[1], blocks, poemScores));
                    for (int r = 0; r < ROWS; r++) {
                        String l = rowLines(r, fits);
                        scores.append("# ").append(bp).append(rname(f, r)).append('\n').append(l);
                        c.get(r).add(XBaseEval.parse(rowGt(g, r), EnumSet.of(V.L1), gal, probes, noParts, l).cfg.get(V.L1));
                    }
                }
            }
            cfgs.add(a);
        }
        if (sm == 2) {
            double est = (smokeS[0] + smokeS[1]) * (10.0 / SMOKE2_BLOCKS) * (1 + FERET_TO_GT);
            StringBuilder t = new StringBuilder(head).append(check).append(String.format(Locale.ROOT, "Замер: набор GT, %d блока из 10 на признак, 5 обучений – L1-xs-F "
                    + "%.0f с, L1-tt-xs-F %.0f с (с векторами); потоков %d.%nОценка полного прогона: (сумма) · %d · (1 + %.3f) = %.0f с (%.1f ч); %.3f – доля наборов "
                    + "FERET-1 и FERET-2 к GT по pool-poem.%n", SMOKE2_BLOCKS, smokeS[0], smokeS[1], tpl.threads, 10 / SMOKE2_BLOCKS, FERET_TO_GT, est, est / 3600,
                    FERET_TO_GT));
            FarMethods.write(s.out.resolve("xbase_pool_abl2" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Повтор (4а), выбор по пулу (XBaseAbl2 pool-abl2): набор, признак, номер строки, строка, «отключено − база» (п. п.), "
                + "метка\n# Код: коммит " + FarMethods.commit() + "\n");
        // [строка][набор·2 + f] – разность и метка для сводки.
        double[][] diffs = new double[ROWS][setNames.size() * 2];
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("строка | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | "
                    + "контрольные | отключено − база, п. п. | метка\n");
            for (int f = 0; f < 2; f++) {
                double[] crit = new double[ROWS];
                XBaseRule.Sep[] sp = new XBaseRule.Sep[ROWS];
                for (int r = 0; r < ROWS; r++) {
                    sp[r] = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(r))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[r] = sp[r].crit();
                }
                for (int r = 0; r < ROWS; r++) {
                    double d = crit[r] - crit[BASE];
                    diffs[r][2 * si + f] = d;
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f | %s | %s%n", rname(f, r), XBaseRule.FEATS[f],
                            XBaseRule.fmt(sp[r].err[0]), XBaseRule.fmt(sp[r].err[1]), XBaseRule.fmt(sp[r].err[2]), XBaseRule.fmt(sp[r].err[3]), XBaseRule.fmt(crit[r]),
                            XBaseRule.fmt(sp[r].eer[3]), sp[r].medOwn, sp[r].medCtrl, r == BASE ? "–" : String.format(Locale.ROOT, "%+.1f", d), r == BASE ? "–"
                            : label(d)));
                    if (r != BASE) sel.append(String.format(Locale.ROOT, "%s\t%s\t%d\t%s\t%+.4f\t%s%n", set, XBaseRule.FEATS[f], r, rname(f, r), d, label(d)));
                }
            }
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder sum = new StringBuilder("\n\n=== Сводка «отключено − база» по критерию пула, п. п. (метка: н – нужен, л – лишний, м – мешает) ===\n"
                + "строка | GT L1-xs-F | GT L1-tt-xs-F | FERET-1 L1-xs-F | FERET-1 L1-tt-xs-F | FERET-2 L1-xs-F | FERET-2 L1-tt-xs-F | на своих\n");
        for (int r = 1; r < ROWS; r++) {
            StringBuilder row = new StringBuilder(rname(1, r).replace("без Тан – Триггс", "Тан – Триггс переключён"));
            boolean own = false;
            for (int c = 0; c < setNames.size() * 2; c++) {
                String lb = label(diffs[r][c]);
                own |= !lb.equals(NEED);
                row.append(String.format(Locale.ROOT, " | %+.1f %s", diffs[r][c], lb.equals(NEED) ? "н" : lb.equals(SPARE) ? "л" : "м"));
            }
            sum.append(row).append(" | ").append(own ? "да" : "нет").append('\n');
        }
        StringBuilder t = new StringBuilder(head).append(check)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads))
                .append(sum).append(body);
        if (sm == 0) Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_abl2" + sfx + "_scores.tsv.gz"), "# Повтор (4а), выбор по пулу FERET (XBaseAbl2 pool-abl2), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_abl2" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_abl2" + sfx + "_gt.txt"), new StringBuilder(head).append(sum).append(gtBody));
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- прогоны на своих (feret-abl2, gt-abl2)

    static final String OWN_NOTE = "На своих – база и строки, «лишние» или «мешающие» по пулу хоть в одном из шести случаев (файл " + SELECT + "); у каждого "
            + "обучения – основные модели X / X+mirror\nи 5 моделей частей (порог П2/П3); (4)–(7) – на моделях базы.\n";

    /** Вычисление обучений fs блока одной постановки: [обучение] → One (по контрольной точке или заново). */
    static XBaseLbp.One[] fitAll(XBaseOther.Checkpoint cp, String tag, int[] fs, XBaseEval.Gt g, Map<String, List<String>> gal, List<Probe> probes,
                                 List<List<String>> parts, int s, List<List<Probe>> extra, int poolFrom, XBaseEval[] xs, int threads) throws IOException {
        XBaseLbp.One[] got = new XBaseLbp.One[NF];
        List<XBaseLbp.Job> jobs = new ArrayList<>();
        List<Integer> todo = new ArrayList<>();
        for (int fit : fs) {
            got[fit] = XBaseLbp.read(cp, tag + "_fit" + fit);
            if (got[fit] != null) continue;
            jobs.add(XBaseLbp.job(fitGt(g, fit, poolFrom), xs[FIT_X[fit]], gal, probes, parts, s, extra));
            todo.add(fit);
        }
        if (!jobs.isEmpty()) {
            List<XBaseLbp.One> done = XBaseLbp.run(jobs, threads);
            for (int i = 0; i < todo.size(); i++) {
                got[todo.get(i)] = done.get(i);
                XBaseLbp.save(cp, tag + "_fit" + todo.get(i), done.get(i));
            }
        }
        return got;
    }

    /** Наборы векторов обучений fs признака f (прочие – null). */
    static XBaseEval[] vectorsFor(Map<String, Mat> fr, IlluminationNorm.NormParams params, int f, int[] fs, String mode) {
        XBaseEval[] xs = new XBaseEval[NX];
        for (int fit : fs) {
            int xi = FIT_X[fit];
            if (xs[xi] != null) continue;
            long t0 = System.nanoTime();
            xs[xi] = vectors(fr, params, f, xi);
            System.out.printf(Locale.ROOT, "%s: f%d x%d vectors (%.0f s)%n", mode, f, xi, (System.nanoTime() - t0) / 1e9);
        }
        return xs;
    }

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        int[] rs = ownRows(s.out.resolve(SELECT));
        if (rs.length == 1) {
            System.out.println("feret-abl2: по пулу все строки «нужен» – прогон на своих не нужен");
            return;
        }
        int[] fs = fitsFor(rs);
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        int poolFrom = s.cohort.size();
        Map<String, List<List<String>>> kept = new TreeMap<>();
        for (String p : ro0.own1) {
            Map<String, List<String>> bd = new TreeMap<>();
            List<String> al = new ArrayList<>(ro0.gal1.get(p));
            al.addAll(ro0.pro1.get(p));
            for (String k : al) bd.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ks = new ArrayList<>();
            for (List<String> l : bd.values()) ks.add(XBaseFeret.sorted(l));
            kept.put(p, ks);
        }
        Path poemScores = s.out.resolve("xbase_feret_poem_scores.tsv.gz");
        Path poemTxt = s.out.resolve("xbase_feret_poem.txt");
        for (Path f : List.of(poemScores, poemTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-abl2: нужен " + f + " (feret-poem)");
        Map<String, String> poemBlocks = XBaseSubsp.blocks(poemScores);
        List<String> poemReport = Files.readAllLines(poemTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-abl2 %s; seed %d; r0 %s; пул %d; половины %d/%d; строки %s; пробный %d",
                s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), Arrays.toString(rs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_abl2_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);

        StringBuilder t = new StringBuilder();
        t.append("Повтор (4а) на новой базе, FERET (XBaseEval feret-abl2): Color FERET – только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-poem); (ii) s = 3 – как feret-dates (люди с ≥ 4 датами, первые 3 "
                + "даты, пробы с 4-й,\nчасти – номер 71), справочно. Контрольные – те же 200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = строки (П2) feret-poem («межс.» и (ii) s3), побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder rowsNote = new StringBuilder("\n=== Строки на своих: ");
        for (int r : rs) rowsNote.append(r == rs[0] ? "" : "; ").append(rname(1, r).replace("без Тан – Триггс", "Тан – Триггс переключён"));
        rowsNote.append(" ===\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        List<List<XBaseKiss.Pass>> passesBy = new ArrayList<>();
        for (int post = 0; post < 2; post++) {
            List<XBaseKiss.Pass> passes = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (post == 0) {
                    own = XBaseVt.trimOwn(ro.own1.stream().filter(mine::contains).toList(), sm);
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else {
                    own = XBaseVt.trimOwn(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList(), sm);
                    for (String p : own) {
                        List<String> g = new ArrayList<>();
                        for (int j = 0; j < 3; j++) g.addAll(s.pool.get(p).get(j));
                        gal.put(p, XBaseFeret.sorted(g));
                        List<String> pr = new ArrayList<>();
                        for (int j = 3; j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                        pro.put(p, XBaseFeret.sorted(pr));
                    }
                }
                Map<String, List<List<String>>> projPool = new TreeMap<>();
                for (String p : s.halves.get(1 - ps)) if (sm != 1 || projPool.size() < XBaseDates.SMOKE_POOL) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + (post == 0 ? 0 : 71));
                List<Probe> probes = XBaseEval.probes(g, pro);
                passes.add(new XBaseKiss.Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
            }
            passesBy.add(passes);
        }
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            String vn = XBasePoem.vname(f, XBaseNap.P2);
            XBaseEval[] xs = null;
            for (int post = 0; post < 2; post++) {
                String pname = post == 0 ? "межс." : "(ii) s3";
                int conf = post == 0 ? 0 : 71;
                List<XBaseKiss.Pass> passes = passesBy.get(post);
                XBaseLbp.One[][] got = new XBaseLbp.One[2][];
                long t0 = System.nanoTime();
                for (int ps = 0; ps < 2; ps++) {
                    String tag = String.format(Locale.ROOT, "c%d_p%d_f%d", conf, ps, f);
                    boolean all = true;
                    for (int fit : fs) all &= cp.read(tag + "_fit" + fit + "_0") != null;
                    if (!all && xs == null) xs = vectorsFor(fr, tpl.params, f, fs, "feret-abl2");
                    XBaseKiss.Pass p = passes.get(ps);
                    got[ps] = fitAll(cp, tag, fs, p.g(), p.gal(), p.probes(), p.parts(), conf, p.extra(), poolFrom, xs, tpl.threads);
                }
                double sec = (System.nanoTime() - t0) / 1e9;
                fitS += sec;
                System.out.printf(Locale.ROOT, "feret-abl2: %s %s ready (%.0f s)%n", pname, feat, sec);
                XBaseThr.Unit[][] units = new XBaseThr.Unit[ROWS][2];
                for (int ps = 0; ps < 2; ps++) {
                    XBaseKiss.Pass p = passes.get(ps);
                    if (sm != 1) check.append(XBaseReg.check(pname + ", " + feat + ", проход " + (ps + 1) + ", " + vn + ", косинус", got[ps][0].rows()[1], poemBlocks,
                            poemScores));
                    for (int r : rs) {
                        String l = rowLines(r, got[ps]);
                        lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ").append(rname(f, r)).append('\n')
                                .append(l);
                        units[r][ps] = new XBaseThr.Unit(rowGt(p.g(), r), p.gal(), p.pro(), p.parts(), l, null, true);
                    }
                }
                if (sm >= 2) {
                    t.append(check).append(rowsNote).append(String.format(Locale.ROOT, "Замер: «межс.» %s, 2 прохода, обучений %d (с векторами) – %.0f с; групп всего "
                            + "4 (2 признака × 2 постановки).%nОценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по feret-poem.%n", feat,
                            fs.length, fitS, fitS, XBaseNap.FERET_FIRST, fitS / XBaseNap.FERET_FIRST, fitS / XBaseNap.FERET_FIRST / 3600, XBaseNap.FERET_FIRST));
                    FarMethods.write(s.out.resolve("xbase_feret_abl2" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = feat + " " + rname(f, BASE);
                for (int r : rs) {
                    String l = feat + " " + rname(f, r);
                    XBaseThr.Out o = XBaseThr.evaluate(l, List.of(units[r]), persons.size(), tpl);
                    if (r == BASE && sm != 1) {
                        for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(pname + " " + feat + " " + vn, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]),
                                XBaseNap.relabel(poemReport, feat + " " + vn, pname + " " + feat + " " + vn), tpl.b, poemTxt));
                    }
                    outs.put(l, o);
                    if (r != BASE) pairs.add(new String[] {l, bl});
                }
                XBaseLbp.tables(body, "FERET " + pname + ", " + feat + (post == 1 ? " (справочно; 25 человек)" : ""), outs, persons.size(), tpl, pairs, true);
            }
            xs = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(rowsNote);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_abl2" + sfx + "_scores.tsv.gz"), "# FERET, повтор (4а) (XBaseAbl2 feret-abl2), только локально\n# Код: "
                + "коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_abl2" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-abl2: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        int[] rs = ownRows(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        if (rs.length == 1) {
            System.out.println("gt-abl2: по пулу все строки «нужен» – прогон на своих не нужен");
            return;
        }
        int[] fs = fitsFor(rs);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new java.util.HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = XBaseVt.trimOwn(g.own, sm);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        int poolFrom = g.cohort.size();
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path poemScores = out.resolve("xbase_gt_poem_scores.tsv.gz");
        Path poemTxt = out.resolve("xbase_gt_poem.txt");
        if (!Files.exists(poemScores) || !Files.exists(poemTxt)) throw new IllegalStateException("gt-abl2: нужны " + poemScores + " и " + poemTxt + " (gt-poem)");
        Map<String, String> poemBlocks = XBaseSubsp.blocks(poemScores);
        List<String> poemReport = Files.readAllLines(poemTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-abl2; seed %d; r0 %s; пул FERET %d; строки %s; пробный %d", g.seed, g.r0, pool.size(),
                Arrays.toString(rs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_abl2" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("Повтор (4а) на новой базе, GT (XBaseEval gt-abl2)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-poem: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «строка − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = строки (П2) gt-poem (межс. и внутри), побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «строка − база») ===\n");
        double fitS = 0;
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            String vn = XBasePoem.vname(f, XBaseNap.P2);
            XBaseEval[] xs = null;
            for (int mi = 0; mi < modes.length; mi++) {
                String mode = (String) modes[mi][0];
                boolean xm = mi == 0;
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                List<Probe> probes = XBaseEval.probes(ga, pro);
                List<List<Probe>> extra = xm ? XBaseThr.extra(ga, pro, parts) : null;
                String tag = (xm ? "x" : "i") + "_f" + f;
                long t0 = System.nanoTime();
                boolean all = true;
                for (int fit : fs) all &= cp.read(tag + "_fit" + fit + "_0") != null;
                if (!all && xs == null) xs = vectorsFor(fr, g.params, f, fs, "gt-abl2");
                XBaseLbp.One[] got = fitAll(cp, tag, fs, ga, gal, probes, parts, 0, extra, poolFrom, xs, g.threads);
                double sec = (System.nanoTime() - t0) / 1e9;
                fitS += sec;
                System.out.printf(Locale.ROOT, "gt-abl2: %s %s ready (%.0f s)%n", mode, feat, sec);
                if (sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + vn + ", косинус", got[0].rows()[1], poemBlocks, poemScores));
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = mode + " " + feat + " " + rname(f, BASE);
                for (int r : rs) {
                    String l = rowLines(r, got);
                    lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(rname(f, r)).append('\n').append(l);
                    String lb = mode + " " + feat + " " + rname(f, r);
                    XBaseThr.Out o = XBaseThr.evaluate(lb, List.of(new XBaseThr.Unit(rowGt(ga, r), gal, pro, parts, l, null, xm)), g.persons.size(), g);
                    if (r == BASE && sm != 1) {
                        for (int w : xm ? new int[] {0, 2} : new int[] {0}) check.append(XBaseReg.checkThr(mode + " " + feat + " " + vn, XBaseThr.WAYS[w],
                                o.res.get(XBaseThr.WAYS[w]), poemReport, g.b, poemTxt));
                    }
                    outs.put(lb, o);
                    if (r != BASE) pairs.add(new String[] {lb, bl});
                }
                if (sm >= 2) {
                    t.append(check).append(String.format(Locale.ROOT, "Замер: межс. %s, обучений %d (с векторами) – %.0f с; групп всего 4 (2 признака × 2 режима).%n"
                            + "Оценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по gt-poem.%n", feat, fs.length, fitS, fitS,
                            XBaseNap.GT_FIRST, fitS / XBaseNap.GT_FIRST, fitS / XBaseNap.GT_FIRST / 3600, XBaseNap.GT_FIRST));
                    FarMethods.write(out.resolve("xbase_gt_abl2" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                if (!xm) {
                    for (String[] p : pairs) {
                        double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], g.persons.size(), g);
                        dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3
                                ? "выполнено" : "НЕ выполнено"));
                    }
                }
                XBaseLbp.tables(body, "GT " + mode + ", " + feat, outs, g.persons.size(), g, pairs, xm);
            }
            xs = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(dl);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_abl2" + sfx + "_scores.tsv.gz"), "# GT, повтор (4а) (XBaseAbl2 gt-abl2)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_abl2" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
