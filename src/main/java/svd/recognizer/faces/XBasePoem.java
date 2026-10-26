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
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * POEM (п. 2.4; постановка Жорика 25.10.2026, план принят с поправкой №6 от 26.10, подтверждён Хозяином): варианты признака при
 * прежнем конвейере (корень гистограмм, MLDA xs-F с пулом FERET, пол c = 1, 1 − cos по всем осям, d₁/d₂, зеркала и выбор модели
 * по |r|): база – LBP кандидата (L1-xs-F 7×7, L1-tt-xs-F 9×9 (8, 2)); (П1) POEM на сетке кандидата; (П2) LBP и POEM на сетке
 * кандидата подряд, часть POEM после корня умножена на 1/√3 (PoemFeature.JOIN_SCALE); (П3) POEM 10×10. Вход POEM – тот же, что у
 * кандидата (без нормировки / Tan – Triggs). Режим pool-poem – выбор по пулу FERET по схеме pool-lbp (база, если в пределах 1 п. п.
 * от лучшего, иначе среди вариантов в пределах 1 п. п. – меньшая размерность); база сверяется побайтно с pool-lbp. Прогоны на
 * своих – только если выбор хоть в одном случае не база.
 *
 * @author ssv
 */
final class XBasePoem {

    static final int NV = 4;
    static final int BASE = 0;
    static final int POEM_GRID3 = 10;
    static final String SELECT = "xbase_pool_poem_select.tsv";
    /**
     * Доля наборов FERET-1 и FERET-2 к набору GT во времени обучения (pool-vt 26.10: GT 2899 + 3836 с, FERET-1 и FERET-2 898 +
     * 1351 + 1060 + 1461 с) – для оценки полного прогона по замеру пробного режима 2 (только GT).
     */
    static final double FERET_TO_GT = (898.0 + 1351 + 1060 + 1461) / (2899 + 3836);
    /** Пробный режим 2: блоки (часть 0, смены 0 и 1) из 10 в группе. */
    static final int SMOKE2_BLOCKS = 2;

    private XBasePoem() {
    }

    /** Пробный режим FACES_POEM_SMOKE: 0 – нет; 1 – наборы по 30 человек, весь путь; 2 – набор GT, оба признака, 2 блока из 10 – замер. */
    static int smoke() {
        String v = System.getenv("FACES_POEM_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    static int grid(int f) {
        return XBaseDates.PARAMS[f].grid();
    }

    static String vname(int f, int v) {
        int g = grid(f);
        return switch (v) {
            case 0 -> String.format(Locale.ROOT, "база LBP %d×%d (8, 2)", g, g);
            case 1 -> String.format(Locale.ROOT, "(П1) POEM %d×%d", g, g);
            case 2 -> String.format(Locale.ROOT, "(П2) LBP + POEM %d×%d, POEM · 1/√3", g, g);
            case 3 -> String.format(Locale.ROOT, "(П3) POEM %d×%d", POEM_GRID3, POEM_GRID3);
            default -> throw new IllegalArgumentException("POEM: вариант " + v);
        };
    }

    static int[] dims(int f) {
        int lbp = XBaseDates.PARAMS[f].length();
        int poem = PoemFeature.length(grid(f));
        return new int[] {lbp, poem, lbp + poem, PoemFeature.length(POEM_GRID3)};
    }

    /** Признак варианта v (v ≥ 1) по кадру: корень гистограмм, у (П2) – сцепление с множителем. */
    static double[] feature(Mat gray8, int f, int v, IlluminationNorm.NormParams params) {
        IlluminationNorm norm = XBaseDates.NORMS[f];
        double[] img = PoemFeature.input(gray8, norm, params);
        int w = gray8.cols();
        int h = gray8.rows();
        return switch (v) {
            case 1 -> LbpFeature.sqrt(PoemFeature.histogram(img, w, h, grid(f)));
            case 2 -> PoemFeature.join(LbpFeature.sqrt(LbpFeature.histogram(gray8, norm, params, XBaseDates.PARAMS[f])),
                    LbpFeature.sqrt(PoemFeature.histogram(img, w, h, grid(f))));
            case 3 -> LbpFeature.sqrt(PoemFeature.histogram(img, w, h, POEM_GRID3));
            default -> throw new IllegalArgumentException("POEM: вариант " + v);
        };
    }

    /** Признаки признака f по вариантам: [0] – как XBaseKiss.vectors (база pool-lbp), [1…3] – POEM; отражения – FaceMirror. */
    static XBaseEval[] vectors(Map<String, Mat> fr, IlluminationNorm.NormParams params, int f) {
        return vectors(fr, params, f, new int[] {0, 1, 2, 3});
    }

    /** То же только для вариантов vs (база – всегда); у прочих – null. */
    static XBaseEval[] vectors(Map<String, Mat> fr, IlluminationNorm.NormParams params, int f, int[] vs) {
        XBaseEval[] xs = new XBaseEval[NV];
        xs[0] = new XBaseEval();
        xs[0].vectors(fr, params, XBaseDates.NORMS[f], XBaseDates.PARAMS[f]);
        List<Integer> need = new ArrayList<>();
        for (int v : vs) if (v > 0) need.add(v);
        for (int v : need) {
            xs[v] = new XBaseEval();
            xs[v].vec.put(V.L1, new HashMap<>());
            xs[v].mir.put(V.L1, new HashMap<>());
        }
        for (Map.Entry<String, Mat> e : fr.entrySet()) {
            Mat m = FaceMirror.mirror(e.getValue());
            for (int v : need) {
                xs[v].vec.get(V.L1).put(e.getKey(), feature(e.getValue(), f, v, params));
                xs[v].mir.get(V.L1).put(e.getKey(), feature(m, f, v, params));
            }
            m.release();
        }
        return xs;
    }

    /** tr S_b / tr S_w в пространстве W по выборке W (классы галереи и когорты, при mirror – и отражения; S_b с весом n_c). */
    static double ratio(XBaseEval x, boolean mirror, List<List<String>> gallery, List<List<String>> cohort, LdaModel lm) {
        List<List<String>> all = new ArrayList<>(gallery);
        all.addAll(cohort);
        int d = lm.projection().length;
        List<double[][]> ys = new ArrayList<>();
        double[] mu = new double[d];
        int n = 0;
        for (List<String> keys : all) {
            List<double[]> list = new ArrayList<>();
            for (String k : keys) list.add(lm.embed(x.vec.get(V.L1).get(k)));
            if (mirror) for (String k : keys) list.add(lm.embed(x.mir.get(V.L1).get(k)));
            double[][] a = list.toArray(new double[0][]);
            ys.add(a);
            for (double[] y : a) for (int i = 0; i < d; i++) mu[i] += y[i];
            n += a.length;
        }
        for (int i = 0; i < d; i++) mu[i] /= n;
        double trB = 0;
        double trW = 0;
        for (double[][] a : ys) {
            if (a.length == 0) continue;
            double[] mc = new double[d];
            for (double[] y : a) for (int i = 0; i < d; i++) mc[i] += y[i] / a.length;
            for (int i = 0; i < d; i++) trB += a.length * (mc[i] - mu[i]) * (mc[i] - mu[i]);
            for (double[] y : a) for (int i = 0; i < d; i++) trW += (y[i] - mc[i]) * (y[i] - mc[i]);
        }
        return trB / trW;
    }

    /** Задачи блока: на каждый вариант основные модели X и X+mirror (как XBaseLbp.job при parts == null) со сведениями. */
    static List<Callable<String[]>> job(XBaseEval.Gt gt, XBaseEval[] xs, Map<String, List<String>> gal, List<Probe> probes, int s) {
        List<String> own = gt.own;
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (int v = 0; v < NV; v++) {
            XBaseEval x = xs[v];
            for (boolean m : new boolean[] {false, true}) {
                String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
                tasks.add(() -> {
                    XBaseRule.Fit fit = XBaseRule.fit(x, m, fullGallery, cohort, null);
                    String[] r = XBaseReg.lines(x, method, s, fit.lm, own, probes);
                    return new String[] {r[0], r[1], fit.lm.info(), Integer.toString(fit.axes()),
                        String.format(Locale.ROOT, "%.4f", ratio(x, m, fullGallery, cohort, fit.lm))};
                });
            }
        }
        return tasks;
    }

    /** Блок: [вариант] → {евклид, косинус, сведения X и X+mirror, tr S_b / tr S_w X / X+mirror}. */
    static String[][] collect(String[][] r) {
        String[][] out = new String[NV][];
        for (int v = 0; v < NV; v++) {
            String[] a = r[2 * v];
            String[] b = r[2 * v + 1];
            out[v] = new String[] {a[0] + b[0], a[1] + b[1], a[3] + "\t" + a[2] + "\n" + b[3] + "\t" + b[2] + "\n", a[4] + " / " + b[4]};
        }
        return out;
    }

    static String[][] read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[][] out = new String[NV][4];
        for (int v = 0; v < NV; v++) {
            for (int r = 0; r < 4; r++) {
                out[v][r] = cp.read(tag + "_v" + v + "_" + r);
                if (out[v][r] == null) return null;
            }
        }
        return out;
    }

    static void save(XBaseOther.Checkpoint cp, String tag, String[][] o) throws IOException {
        for (int v = 0; v < NV; v++) for (int r = 0; r < 4; r++) cp.save(tag + "_v" + v + "_" + r, o[v][r]);
    }

    static List<String[][]> run(List<List<Callable<String[]>>> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (List<Callable<String[]>> j : jobs) flat.addAll(j);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<String[][]> out = new ArrayList<>();
            int i = 0;
            for (List<Callable<String[]>> j : jobs) {
                String[][] r = new String[j.size()][];
                for (int k = 0; k < j.size(); k++) r[k] = fs.get(i++).get();
                out.add(collect(r));
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBasePoem: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    /** Строка сведений: размерность | ранг оболочки X / X+mirror | заменено | осей | tr S_b / tr S_w X / X+mirror. */
    static String infoRow(String head, int f, int v, String[] blk) {
        String[] p = blk[2].split("\n");
        int[] a = XBaseLbp.hull(p[0]);
        int[] b = XBaseLbp.hull(p[1]);
        String ax = p[0].split("\t", 2)[0] + " / " + p[1].split("\t", 2)[0];
        return String.format(Locale.ROOT, "%s | %s | %d | %d / %d | %d / %d | %s | %s%n", head, vname(f, v), dims(f)[v], a[1], b[1], a[0], b[0], ax, blk[3]);
    }

    static final String INFO_HEAD = "\n=== Модели (основная модель блока): размерность признака | ранг оболочки X / X+mirror | заменено собственных значений S_w в "
            + "оболочке X / X+mirror | осей MLDA X / X+mirror | tr S_b / tr S_w в W X / X+mirror ===\n";

    static final String NOTE = "Варианты (вход POEM – тот же, что у кандидата: L1-xs-F – кадр без нормировки, L1-tt-xs-F – Tan – Triggs): база – LBP u2 (8, 2) на "
            + "сетке кандидата (7×7 – 2891, 9×9 – 4779);\n(П1) POEM на сетке кандидата (7×7 – 8673, 9×9 – 14 337); (П2) LBP и POEM на сетке кандидата подряд, часть "
            + "POEM после корня · 1/√3 (7×7 – 11 564, 9×9 – 19 116);\n(П3) POEM 10×10 (17 700). POEM: Собель 3×3, ориентации без знака, 3 интервала, ячейка 7×7, "
            + "8 соседей на окружности радиуса 5, τ = 1e-6 · max карты,\nравномерные коды (59), гистограмма каждого (блока, ориентации) на сумму 1, корень. На каждый "
            + "вариант обучаются заново основные модели X / X+mirror;\nMLDA с пулом FERET, пол λ̄ (c = 1), 1 − cos до центра класса по всем осям, d₁/d₂, модель "
            + "X / X+mirror по |r| пробы.\n";

    // ---------------------------------------------------------------- pool-poem

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(sm == 1 ? all.subList(0, 30) : all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        if (sm == 1) {
            sets.put("FERET-1", new TreeSet<>(new ArrayList<>(s.halves.get(1)).subList(0, 30)));
            sets.put("FERET-2", new TreeSet<>(new ArrayList<>(s.halves.get(0)).subList(0, 30)));
        }
        List<String> setNames = new ArrayList<>(sets.keySet());
        Path lbpScores = s.out.resolve("xbase_pool_lbp_scores.tsv.gz");
        if (!Files.exists(lbpScores)) throw new IllegalStateException("pool-poem: для сверки базы нужен " + lbpScores + " (pool-lbp)");
        Map<String, String> blocks = XBaseSubsp.blocks(lbpScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        List<String> names = new ArrayList<>();
        for (int f = 0; f < 2; f++) for (int v = 0; v < NV; v++) names.add(vname(f, v));
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-poem %s; seed %d; r0 %s; пул %d; половины %d/%d; варианты %s; пробный %d",
                s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", names), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_poem_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        StringBuilder head = new StringBuilder();
        head.append("П. 2.4, POEM: выбор варианта признака по пулу FERET (XBaseEval pool-poem)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-lbp: GT – весь пул; FERET-1 – половина B (%d); FERET-2 – половина A (%d);%n5 частей по "
                + "людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула; обучение W – галерея своих + когорта "
                + "FEI 20 + люди 4 частей.%n", s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET, XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база, если в пределах %.0f п. п. от лучшего,%nиначе "
                + "среди вариантов в пределах %.0f п. п. – с меньшей размерностью признака (при равной – база).%n", XBaseRule.NEAR, XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = pool-lbp (L1-xs-F – база 7×7, L1-tt-xs-F – вариант (2) 9×9), побайтно ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("набор, признак, часть, смена | вариант | размерность | ранг оболочки | заменено | осей | tr S_b / tr S_w\n");
        StringBuilder infoGt = new StringBuilder(info);
        List<List<List<List<FeiDecompose.Cfg>>>> cfgs = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            List<List<List<FeiDecompose.Cfg>>> a = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                List<List<FeiDecompose.Cfg>> b = new ArrayList<>();
                for (int v = 0; v < NV; v++) b.add(new ArrayList<>());
                a.add(b);
            }
            cfgs.add(a);
        }
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        double[] smokeS = new double[2];
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        // Признак – внешний цикл: векторы четырёх вариантов одного признака в памяти, затем освобождаются.
        for (int f = 0; f < 2; f++) {
            XBaseEval[] xv = null;
            for (int si = 0; si < setNames.size(); si++) {
                if (sm == 2 && si > 0) break;
                String set = setNames.get(si);
                Set<String> members = sets.get(set);
                List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
                List<Object[]> blk = new ArrayList<>();
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    if (sm == 2 && blk.size() >= SMOKE2_BLOCKS) break;
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
                        String tag = String.format(Locale.ROOT, "%s_q%d_h%d_f%d", set, q, h, f);
                        blk.add(new Object[] {q, h, g, gal, probes, tag, read(cp, tag)});
                    }
                }
                List<List<Callable<String[]>>> jobs = new ArrayList<>();
                List<Object[]> todo = new ArrayList<>();
                for (Object[] b : blk) {
                    if (b[6] != null) continue;
                    if (xv == null) {
                        long t0 = System.nanoTime();
                        xv = vectors(fr, tpl.params, f);
                        System.out.printf(Locale.ROOT, "pool-poem: f%d vectors (%.0f s)%n", f, (System.nanoTime() - t0) / 1e9);
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    jobs.add(job((XBaseEval.Gt) b[2], xv, gal, probes, 2 * (int) b[0] + (int) b[1]));
                    todo.add(b);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[][]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    if (sm == 2) smokeS[f] = sec;
                    for (int i = 0; i < todo.size(); i++) {
                        todo.get(i)[6] = done.get(i);
                        save(cp, (String) todo.get(i)[5], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-poem: %s f%d ready, %d blocks (%.0f s)%n", set, f, jobs.size(), sec);
                }
                for (Object[] b : blk) {
                    int q = (int) b[0];
                    int h = (int) b[1];
                    XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    String[][] ls = (String[][]) b[6];
                    String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                    if (sm != 1) check.append(XBaseReg.check(bp + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", ls[BASE][1], blocks, lbpScores));
                    for (int v = 0; v < NV; v++) {
                        String row = infoRow(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], q, h), f, v, ls[v]);
                        info.append(row);
                        if (set.equals("GT")) infoGt.append(row);
                        scores.append("# ").append(bp).append(vname(f, v)).append(", косинус\n").append(ls[v][1]);
                        cfgs.get(si).get(f).get(v).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls[v][1]).cfg.get(V.L1));
                    }
                }
            }
            xv = null;
        }
        for (Mat m : fr.values()) m.release();
        if (sm == 2) {
            double est = (smokeS[0] + smokeS[1]) * (10.0 / SMOKE2_BLOCKS) * (1 + FERET_TO_GT);
            StringBuilder t = new StringBuilder(head).append(check).append(info).append(String.format(Locale.ROOT, "Замер: набор GT, %d блока из 10 на признак, 4 варианта – "
                    + "L1-xs-F %.0f с, L1-tt-xs-F %.0f с; потоков %d.%nОценка полного прогона: (сумма) · %d · (1 + %.3f) = %.0f с (%.1f ч); %.3f – доля наборов FERET-1 "
                    + "и FERET-2 к GT по pool-vt.%n", SMOKE2_BLOCKS, smokeS[0], smokeS[1], tpl.threads, 10 / SMOKE2_BLOCKS, FERET_TO_GT, est, est / 3600, FERET_TO_GT));
            FarMethods.write(s.out.resolve("xbase_pool_poem" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBasePoem pool-poem): набор, признак, номер варианта, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("вариант | размерность | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным "
                    + "лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < 2; f++) {
                int[] dims = dims(f);
                double[] crit = new double[NV];
                for (int v = 0; v < NV; v++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(v))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[v] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", vname(f, v), dims[v], XBaseRule.FEATS[f],
                            XBaseRule.fmt(sp.err[0]), XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[v]),
                            XBaseRule.fmt(sp.eer[3]), sp.medOwn, sp.medCtrl));
                }
                int c = XBaseLbp.choose(crit, dims);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], vname(f, c),
                        crit[c], crit[BASE], Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(vname(f, c)).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body).append(info);
        if (sm == 0) Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_poem" + sfx + "_scores.tsv.gz"), "# POEM, выбор по пулу FERET (XBasePoem pool-poem), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_poem" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_poem" + sfx + "_gt.txt"), new StringBuilder(head).append(gtBody).append(infoGt));
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- прогоны на своих (feret-poem, gt-poem)

    /**
     * Доля первой группы (L1-tt-xs-F, «межс.») во времени всего прогона на своих по feret-vt / gt-vt 26.10 (FERET: 1575 из 1575 +
     * 911 + 1357 + 678 с; GT: 2078 из 2078 + 1819 + 1807 + 1582 с) – оценка полного прогона по замеру пробного режима 2.
     */
    static final double FERET_FIRST = 1575.0 / (1575 + 911 + 1357 + 678);
    static final double GT_FIRST = 2078.0 / (2078 + 1819 + 1807 + 1582);

    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBasePoem: нет файла выбора " + f + " (сначала pool-poem)");
        List<String> feats = List.of(XBaseRule.FEATS);
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int fi = feats.indexOf(p[1]);
            int v = Integer.parseInt(p[2]);
            if (fi < 0 || v < 0 || v >= NV || !vname(fi, v).equals(p[3])) throw new IllegalStateException("XBasePoem: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    static final String OWN_INFO_HEAD = "\n=== Модели (основная модель): ранг оболочки X / X+mirror | заменено собственных значений S_w в оболочке X / X+mirror | осей "
            + "MLDA X / X+mirror ===\n";

    static final String OWN_NOTE = "На своих считаются база и варианты, выбранные pool-poem хоть в одном случае (вход POEM – тот же, что у кандидата); на каждый "
            + "вариант обучаются\nосновные модели X / X+mirror и 5 моделей частей (порог П2/П3). MLDA с пулом FERET, пол c = 1, 1 − cos по всем осям, d₁/d₂, модель "
            + "X / X+mirror по |r| пробы.\n";

    static String infoRow(String head, int f, int v, XBaseLbp.One l) {
        int[] a = XBaseLbp.hull(l.info()[0]);
        int[] b = XBaseLbp.hull(l.info()[1]);
        return String.format(Locale.ROOT, "%s | %s | %d / %d | %d / %d | %d / %d%n", head, vname(f, v), a[1], b[1], a[0], b[0], l.axes()[0], l.axes()[1]);
    }

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        int[] vs = XBaseVt.ownVariants(select);
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        // Все отобранные снимки своих по датам – как XBaseDates и feret-vt.
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
        Path lbpScores = s.out.resolve("xbase_feret_lbp_scores.tsv.gz");
        Path lbpTxt = s.out.resolve("xbase_feret_lbp.txt");
        Path datesScores = s.out.resolve("xbase_feret_dates_scores.tsv.gz");
        Path datesTxt = s.out.resolve("xbase_feret_dates.txt");
        for (Path f : List.of(lbpScores, lbpTxt, datesScores, datesTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-poem: нужен " + f);
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        Map<String, String> datesBlocks = XBaseSubsp.blocks(datesScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, java.nio.charset.StandardCharsets.UTF_8);
        List<String> datesReport = Files.readAllLines(datesTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-poem %s; seed %d; r0 %s; пул %d; половины %d/%d; выбор %s; варианты %s; "
                + "пробный %d", s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), select, Arrays.toString(vs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_poem_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);

        StringBuilder t = new StringBuilder();
        t.append("П. 2.4, POEM, FERET (XBaseEval feret-poem): Color FERET – только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-lbp); (ii) s = 3 – как feret-dates (люди с ≥ 4 датами, первые 3 "
                + "даты, пробы с 4-й,\nчасти – номер 71), справочно. «Выбор по пулу» – вариант из pool-poem (проход 1 – FERET-1, проход 2 – FERET-2). Контрольные – те "
                + "же 200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = feret-lbp («межс.») и feret-dates ((ii) s3 (м)), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("постановка, признак, проход | вариант | ранг оболочки | заменено | осей\n");
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
        // Признак – внешний цикл: векторы вариантов одного признака в памяти, затем освобождаются.
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            XBaseEval[] xv = null;
            for (int post = 0; post < 2; post++) {
                String pname = post == 0 ? "межс." : "(ii) s3";
                int conf = post == 0 ? 0 : 71;
                List<XBaseKiss.Pass> passes = passesBy.get(post);
                // [вариант][проход]; недостающие в контрольной точке – одной пачкой задач.
                XBaseLbp.One[][] got = new XBaseLbp.One[NV][2];
                List<XBaseLbp.Job> jobs = new ArrayList<>();
                List<int[]> todo = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int v : vs) {
                    for (int ps = 0; ps < 2; ps++) {
                        String tag = String.format(Locale.ROOT, "c%d_p%d_f%d_v%d", conf, ps, f, v);
                        got[v][ps] = XBaseLbp.read(cp, tag);
                        if (got[v][ps] != null) continue;
                        if (xv == null) {
                            long tv = System.nanoTime();
                            xv = vectors(fr, tpl.params, f, vs);
                            System.out.printf(Locale.ROOT, "feret-poem: f%d vectors (%.0f s)%n", f, (System.nanoTime() - tv) / 1e9);
                        }
                        XBaseKiss.Pass p = passes.get(ps);
                        jobs.add(XBaseLbp.job(p.g(), xv[v], p.gal(), p.probes(), p.parts(), conf, p.extra()));
                        todo.add(new int[] {v, ps});
                    }
                }
                if (!jobs.isEmpty()) {
                    List<XBaseLbp.One> done = XBaseLbp.run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        int[] vp = todo.get(i);
                        got[vp[0]][vp[1]] = done.get(i);
                        XBaseLbp.save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d_v%d", conf, vp[1], f, vp[0]), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "feret-poem: %s %s ready, %d blocks (%.0f s)%n", pname, feat, jobs.size(), sec);
                }
                XBaseThr.Unit[][] units = new XBaseThr.Unit[NV][2];
                for (int v : vs) {
                    for (int ps = 0; ps < 2; ps++) {
                        XBaseKiss.Pass p = passes.get(ps);
                        XBaseLbp.One ls = got[v][ps];
                        if (v == BASE && sm != 1) {
                            if (post == 0) check.append(XBaseReg.check("проход " + (ps + 1) + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус",
                                    ls.rows()[1], lbpBlocks, lbpScores));
                            else check.append(XBaseReg.check("(i)=(ii) s3 (м) " + feat + ", проход " + (ps + 1), ls.rows()[1], datesBlocks, datesScores));
                        }
                        info.append(infoRow(pname + ", " + feat + ", проход " + (ps + 1), f, v, ls));
                        lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ").append(vname(f, v))
                                .append(", косинус\n").append(ls.rows()[1]);
                        units[v][ps] = new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), ls.rows()[1], null, true);
                    }
                }
                if (sm >= 2) {
                    t.append(check).append(info).append(String.format(Locale.ROOT, "Замер: «межс.» %s, 2 прохода × %d вариантов (с векторами) – %.0f с; групп всего 4 "
                            + "(2 признака × 2 постановки).%nОценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по feret-vt.%n", feat,
                            vs.length, fitS, fitS, FERET_FIRST, fitS / FERET_FIRST, fitS / FERET_FIRST / 3600, FERET_FIRST));
                    FarMethods.write(s.out.resolve("xbase_feret_poem" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                int[] chosen = {select.get("FERET-1").get(feat), select.get("FERET-2").get(feat)};
                if (post == 0) sel.append(String.format(Locale.ROOT, "%s: проход 1 – %s, проход 2 – %s%n", feat, vname(f, chosen[0]), vname(f, chosen[1])));
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = feat + " " + vname(f, BASE);
                for (int v : vs) {
                    String l = feat + " " + vname(f, v);
                    XBaseThr.Out o = XBaseThr.evaluate(l, List.of(units[v]), persons.size(), tpl);
                    if (v == BASE && sm != 1) {
                        String rl = post == 0 ? "межс. " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]] : "(i)=(ii) s3 (м) " + feat;
                        for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), post == 0 ? lbpReport
                                : datesReport, tpl.b, post == 0 ? lbpTxt : datesTxt));
                    }
                    outs.put(l, o);
                    if (v != BASE) pairs.add(new String[] {l, bl});
                }
                String ch = feat + " " + XBaseLbp.CHOICE;
                outs.put(ch, XBaseThr.evaluate(ch, List.of(units[chosen[0]][0], units[chosen[1]][1]), persons.size(), tpl));
                pairs.add(new String[] {ch, bl});
                XBaseLbp.tables(body, "FERET " + pname + ", " + feat + (post == 1 ? " (справочно; 25 человек)" : ""), outs, persons.size(), tpl, pairs, true);
            }
            xv = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_poem" + sfx + "_scores.tsv.gz"), "# FERET, POEM (XBasePoem feret-poem), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_poem" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-poem: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        int[] vs = XBaseVt.ownVariants(select);
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
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        Path lbpTxt = out.resolve("xbase_gt_lbp.txt");
        if (!Files.exists(lbpScores) || !Files.exists(lbpTxt)) throw new IllegalStateException("gt-poem: нужны " + lbpScores + " и " + lbpTxt + " (gt-lbp)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-poem; seed %d; r0 %s; пул FERET %d; выбор %s; варианты %s; пробный %d", g.seed, g.r0,
                pool.size(), select, Arrays.toString(vs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_poem" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("П. 2.4, POEM, GT (XBaseEval gt-poem)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-lbp: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «вариант − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = gt-lbp (межс. и внутри), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("режим, признак | вариант | ранг оболочки | заменено | осей\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        double fitS = 0;
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            XBaseEval[] xv = null;
            for (int mi = 0; mi < modes.length; mi++) {
                String mode = (String) modes[mi][0];
                boolean x = mi == 0;
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                List<Probe> probes = XBaseEval.probes(ga, pro);
                List<List<Probe>> extra = x ? XBaseThr.extra(ga, pro, parts) : null;
                XBaseLbp.One[] got = new XBaseLbp.One[NV];
                List<XBaseLbp.Job> jobs = new ArrayList<>();
                List<Integer> todo = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int v : vs) {
                    got[v] = XBaseLbp.read(cp, (x ? "x" : "i") + "_f" + f + "_v" + v);
                    if (got[v] != null) continue;
                    if (xv == null) {
                        long tv = System.nanoTime();
                        xv = vectors(fr, g.params, f, vs);
                        System.out.printf(Locale.ROOT, "gt-poem: f%d vectors (%.0f s)%n", f, (System.nanoTime() - tv) / 1e9);
                    }
                    jobs.add(XBaseLbp.job(ga, xv[v], gal, probes, parts, 0, extra));
                    todo.add(v);
                }
                if (!jobs.isEmpty()) {
                    List<XBaseLbp.One> done = XBaseLbp.run(jobs, g.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        got[todo.get(i)] = done.get(i);
                        XBaseLbp.save(cp, (x ? "x" : "i") + "_f" + f + "_v" + todo.get(i), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "gt-poem: %s %s ready, %d blocks (%.0f s)%n", mode, feat, jobs.size(), sec);
                }
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = mode + " " + feat + " " + vname(f, BASE);
                for (int v : vs) {
                    XBaseLbp.One ls = got[v];
                    if (v == BASE && sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", ls.rows()[1],
                            lbpBlocks, lbpScores));
                    info.append(infoRow(mode + ", " + feat, f, v, ls));
                    lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(vname(f, v)).append(", косинус\n").append(ls.rows()[1]);
                    String l = mode + " " + feat + " " + vname(f, v);
                    XBaseThr.Out o = XBaseThr.evaluate(l, List.of(new XBaseThr.Unit(ga, gal, pro, parts, ls.rows()[1], null, x)), g.persons.size(), g);
                    if (v == BASE && sm != 1) {
                        String rl = mode + " " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]];
                        for (int w : x ? new int[] {0, 2} : new int[] {0}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), lbpReport,
                                g.b, lbpTxt));
                    }
                    outs.put(l, o);
                    if (v != BASE) pairs.add(new String[] {l, bl});
                }
                if (sm >= 2) {
                    t.append(check).append(info).append(String.format(Locale.ROOT, "Замер: межс. %s, %d вариантов (с векторами) – %.0f с; групп всего 4 (2 признака × "
                            + "2 режима).%nОценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по gt-vt.%n", feat, vs.length, fitS, fitS,
                            GT_FIRST, fitS / GT_FIRST, fitS / GT_FIRST / 3600, GT_FIRST));
                    FarMethods.write(out.resolve("xbase_gt_poem" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                int c = select.get("GT").get(feat);
                String ch = mode + " " + feat + " " + XBaseLbp.CHOICE;
                outs.put(ch, XBaseThr.evaluate(ch, List.of(new XBaseThr.Unit(ga, gal, pro, parts, got[c].rows()[1], null, x)), g.persons.size(), g));
                pairs.add(new String[] {ch, bl});
                if (!x) {
                    for (String[] p : pairs) {
                        double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], g.persons.size(), g);
                        boolean isChoice = p[0].equals(ch);
                        dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0] + (isChoice ? " (" + vname(f, c)
                                + ")" : ""), d[0], d[1], isChoice && c == BASE ? "не применяется (выбор – база)" : d[1] <= 3 ? "выполнено" : "НЕ выполнено"));
                    }
                }
                XBaseLbp.tables(body, "GT " + mode + ", " + feat + " (выбор по пулу – " + vname(f, c) + ")", outs, g.persons.size(), g, pairs, x);
            }
            xv = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(dl).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_poem" + sfx + "_scores.tsv.gz"), "# GT, POEM (XBasePoem gt-poem)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_poem" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
