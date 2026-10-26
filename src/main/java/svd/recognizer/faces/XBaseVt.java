package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
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
 * Опыт «пул FERET + VidTIMIT в обучении W» (задание Жорика 26.10.2026, план принят, подтверждён Хозяином): варианты база, (В2) –
 * + 43 класса VidTIMIT по 2 кадра на сессию, (В4) – по 4 кадра на сессию (VidTimitData). Классы VidTIMIT добавляются к выборке
 * обучения W (когорта FEI + обучающая часть пула xs-F) в каждой части и смене, больше ничего не меняется: признаки L1-xs-F 7×7 и
 * L1-tt-xs-F 9×9 (8, 2), пол c = 1, 1 − cos по всем осям, d₁/d₂, зеркала (у X+mirror отражаются все кадры класса, как у FERET).
 * Режим pool-vt – выбор по пулу FERET по схеме pool-lbp (наборы GT, FERET-1, FERET-2; VidTIMIT – только в обучении); база
 * сверяется побайтно с pool-lbp. Прогоны на своих (feret-vt, gt-vt) – только если выбор хоть в одном случае не база.
 *
 * @author ssv
 */
final class XBaseVt {

    static final String[] VNAME = {"база", "(В2) + VidTIMIT, 2 кадра на сессию", "(В4) + VidTIMIT, 4 кадра на сессию"};
    static final int BASE = 0;
    static final int NV = VNAME.length;
    static final String SELECT = "xbase_pool_vt_select.tsv";

    private XBaseVt() {
    }

    /** Пробный режим FACES_VT_SMOKE: 0 – нет; 1 – наборы по 30 человек, весь путь; 2 – только первая группа (GT, L1-xs-F) – замер. */
    static int smoke() {
        String v = System.getenv("FACES_VT_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    /** Правило выбора: база, если в пределах NEAR п. п. от лучшего; иначе первый в порядке простоты (В2) < (В4) в пределах NEAR. */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        if (crit[BASE] <= best + XBaseRule.NEAR) return BASE;
        for (int v = 1; v < crit.length; v++) if (crit[v] <= best + XBaseRule.NEAR) return v;
        throw new IllegalStateException("XBaseVt: выбор не определён");
    }

    /** Выборка W: классы галереи и когорты (при mirror – и отражения), из них – VidTIMIT (ключи KEY_PREFIX). */
    static boolean isVt(List<String> keys) {
        return !keys.isEmpty() && keys.get(0).startsWith(VidTimitData.KEY_PREFIX);
    }

    /**
     * Сведения модели: классов, людей и кадров VidTIMIT (с отражениями), осей W, tr S_b / tr S_w в пространстве W по выборке W
     * (S_b – по центрам классов с весом n_c), средний квадрат отклонения от центра класса у кадров VidTIMIT и у прочих.
     */
    static String diag(XBaseEval x, boolean mirror, List<List<String>> gallery, List<List<String>> cohort, LdaModel lm) {
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
        double[] trW = new double[2];
        int[] cnt = new int[2];
        int vtP = 0;
        for (int c = 0; c < all.size(); c++) {
            double[][] a = ys.get(c);
            if (a.length == 0) continue;
            int g = isVt(all.get(c)) ? 1 : 0;
            if (g == 1) vtP++;
            double[] mc = new double[d];
            for (double[] y : a) for (int i = 0; i < d; i++) mc[i] += y[i] / a.length;
            for (int i = 0; i < d; i++) trB += a.length * (mc[i] - mu[i]) * (mc[i] - mu[i]);
            for (double[] y : a) for (int i = 0; i < d; i++) trW[g] += (y[i] - mc[i]) * (y[i] - mc[i]);
            cnt[g] += a.length;
        }
        return String.format(Locale.ROOT, "%d | %d / %d | %d | %.4f | %s / %.4f", all.size(), vtP, cnt[1], d, trB / (trW[0] + trW[1]),
                cnt[1] == 0 ? "-" : String.format(Locale.ROOT, "%.4f", trW[1] / cnt[1]), trW[0] / cnt[0]);
    }

    static final String DIAG_HEAD = "\n=== Сведения моделей (основная модель блока) X / X+mirror: классов выборки W | людей / кадров VidTIMIT (с отражениями) | осей W | "
            + "tr S_b / tr S_w в W | ‖y − μ_c‖² на кадр: VidTIMIT / прочие ===\n";

    /** Задачи блока: на каждый вариант основные модели X и X+mirror (как XBaseLbp.job при parts == null) со сведениями. */
    static List<Callable<String[]>> job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> probes, int s, List<List<List<String>>> vt) {
        List<String> own = gt.own;
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (int v = 0; v < NV; v++) {
            List<List<String>> cohort = new ArrayList<>(gt.cohort);
            if (v > 0) cohort.addAll(vt.get(v - 1));
            for (boolean m : new boolean[] {false, true}) {
                String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
                tasks.add(() -> {
                    XBaseRule.Fit f = XBaseRule.fit(x, m, fullGallery, cohort, null);
                    String[] r = XBaseReg.lines(x, method, s, f.lm, own, probes);
                    return new String[] {r[0], r[1], f.lm.info(), Integer.toString(f.axes()), diag(x, m, fullGallery, cohort, f.lm)};
                });
            }
        }
        return tasks;
    }

    /** Блок: [вариант] → {евклид, косинус, сведения X, X+mirror}; строки X, затем X+mirror (как XBaseLbp.run). */
    static String[][] collect(String[][] r) {
        String[][] out = new String[NV][];
        for (int v = 0; v < NV; v++) {
            String[] a = r[2 * v];
            String[] b = r[2 * v + 1];
            out[v] = new String[] {a[0] + b[0], a[1] + b[1], a[3] + "\t" + a[2] + "\n" + b[3] + "\t" + b[2] + "\n", a[4] + " || " + b[4]};
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
            throw new IllegalStateException("XBaseVt: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    // ---------------------------------------------------------------- pool-vt

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Path vdir = VidTimitData.dir(settings);
        VidTimitData.Loaded[] vt = {VidTimitData.load(vdir, VidTimitData.VARIANT[0]), VidTimitData.load(vdir, VidTimitData.VARIANT[1])};
        List<List<List<String>>> vtCls = List.of(vt[0].classes, vt[1].classes);
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
        if (!Files.exists(lbpScores)) throw new IllegalStateException("pool-vt: для сверки базы нужен " + lbpScores + " (pool-lbp)");
        Map<String, String> blocks = XBaseSubsp.blocks(lbpScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-vt %s; seed %d; r0 %s; пул %d; половины %d/%d; варианты %s; VidTIMIT %s; "
                + "пробный %d", s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", VNAME), vt[0].sha256, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_vt_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        for (VidTimitData.Loaded l : vt) for (Map.Entry<String, Mat> e : l.frames.entrySet()) fr.putIfAbsent(e.getKey(), e.getValue());
        XBaseEval[] xv = XBaseKiss.vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();
        for (VidTimitData.Loaded l : vt) for (Mat m : l.frames.values()) m.release();
        StringBuilder head = new StringBuilder();
        head.append("Опыт «пул FERET + VidTIMIT в обучении W»: выбор варианта по пулу FERET (XBaseEval pool-vt)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                .append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append("; VidTIMIT – ").append(vdir).append(" (только локально), отбор ")
                .append(VidTimitData.SELECTION).append(" SHA-256 ").append(vt[0].sha256).append('\n');
        head.append(String.format(Locale.ROOT, "VidTIMIT в выборке W: (В2) – людей %d, кадров %d; (В4) – людей %d, кадров %d (без отражений; у X+mirror – вдвое больше).%n",
                vt[0].persons, vt[0].images, vt[1].persons, vt[1].images));
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-lbp: GT – весь пул; FERET-1 – половина B (%d); FERET-2 – половина A (%d);%n5 частей по "
                + "людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула; обучение W – галерея своих + когорта "
                + "FEI 20 + люди 4 частей (+ VidTIMIT в (В2), (В4)).%n", s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET,
                XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база, если в пределах %.0f п. п. от лучшего, иначе "
                + "первый в порядке простоты (В2) < (В4) в пределах %.0f п. п.%n", XBaseRule.NEAR, XBaseRule.NEAR));
        head.append("Признаки: L1-xs-F – LBP 7×7 (8, 2) без нормировки; L1-tt-xs-F – Tan – Triggs, 9×9 (8, 2); корень гистограмм, MLDA, пол c = 1, 1 − cos по всем "
                + "осям, d₁/d₂, модель X / X+mirror по |r| пробы.\n");
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = pool-lbp (L1-xs-F – база 7×7, L1-tt-xs-F – вариант (2) 9×9), побайтно ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD).append("набор, признак, часть, смена | вариант | сведения X || X+mirror\n");
        StringBuilder dgGt = new StringBuilder(dg);
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
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        outer:
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            Set<String> members = sets.get(set);
            List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
            for (int f = 0; f < 2; f++) {
                List<Object[]> blk = new ArrayList<>();
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
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
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    jobs.add(job((XBaseEval.Gt) b[2], xv[f], gal, probes, 2 * (int) b[0] + (int) b[1], vtCls));
                    todo.add(b);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[][]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        todo.get(i)[6] = done.get(i);
                        save(cp, (String) todo.get(i)[5], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-vt: %s f%d ready, %d blocks (%.0f s)%n", set, f, jobs.size(), sec);
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
                        String row = String.format(Locale.ROOT, "%s %s часть %d смена %d | %s | %s%n", set, XBaseRule.FEATS[f], q, h, VNAME[v], ls[v][3]);
                        dg.append(row);
                        if (set.equals("GT")) dgGt.append(row);
                        scores.append("# ").append(bp).append(VNAME[v]).append(", косинус\n").append(ls[v][1]);
                        cfgs.get(si).get(f).get(v).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls[v][1]).cfg.get(V.L1));
                    }
                }
                if (sm >= 2) break outer;
            }
        }
        if (sm >= 2) {
            StringBuilder t = new StringBuilder(head).append(check).append(dg).append(String.format(Locale.ROOT, "Замер: первая группа (набор GT, L1-xs-F, 10 блоков × 3 "
                    + "варианта) – %.0f с; групп всего %d (3 набора × 2 признака); потоков %d.%n", fitS, setNames.size() * 2, tpl.threads));
            FarMethods.write(s.out.resolve("xbase_pool_vt" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseVt pool-vt): набор, признак, номер варианта, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < 2; f++) {
                double[] crit = new double[NV];
                for (int v = 0; v < NV; v++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(v))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[v] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", VNAME[v], XBaseRule.FEATS[f], XBaseRule.fmt(sp.err[0]),
                            XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[v]), XBaseRule.fmt(sp.eer[3]), sp.medOwn,
                            sp.medCtrl));
                }
                int c = choose(crit);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], VNAME[c], crit[c],
                        crit[BASE], Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(VNAME[c]).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body).append(dg);
        if (sm == 0) Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_vt" + sfx + "_scores.tsv.gz"), "# Пул FERET + VidTIMIT, выбор по пулу FERET (XBaseVt pool-vt), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_vt" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_vt" + sfx + "_gt.txt"), new StringBuilder(head).append(gtBody).append(dgGt));
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- прогоны на своих (feret-vt, gt-vt)

    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseVt: нет файла выбора " + f + " (сначала pool-vt)");
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int v = Integer.parseInt(p[2]);
            if (v < 0 || v >= NV || !VNAME[v].equals(p[3])) throw new IllegalStateException("XBaseVt: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    /** Варианты прогона на своих: база и все варианты, выбранные по пулу хоть в одном случае (по возрастанию номера). */
    static int[] ownVariants(Map<String, Map<String, Integer>> select) {
        TreeSet<Integer> vs = new TreeSet<>(List.of(BASE));
        for (Map<String, Integer> m : select.values()) vs.addAll(m.values());
        return vs.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Задачи блока варианта v – XBaseLbp.job (основные модели и 5 моделей частей) с классами VidTIMIT в выборке W. */
    static XBaseLbp.Job ownJob(XBaseEval.Gt g, XBaseEval x, Map<String, List<String>> gal, List<Probe> probes, List<List<String>> parts, int s,
                               List<List<Probe>> extra, int v, List<List<List<String>>> vtCls) {
        List<List<String>> c0 = g.cohort;
        List<List<String>> c = new ArrayList<>(c0);
        if (v > 0) c.addAll(vtCls.get(v - 1));
        g.cohort = c;
        try {
            return XBaseLbp.job(g, x, gal, probes, parts, s, extra);
        } finally {
            g.cohort = c0;
        }
    }

    static String infoRow(String head, int v, XBaseLbp.One l) {
        int[] a = XBaseLbp.hull(l.info()[0]);
        int[] b = XBaseLbp.hull(l.info()[1]);
        return String.format(Locale.ROOT, "%s | %s | %d / %d | %d / %d | %d / %d%n", head, VNAME[v], a[1], b[1], a[0], b[0], l.axes()[0], l.axes()[1]);
    }

    static final String OWN_INFO_HEAD = "\n=== Модели (основная модель): ранг оболочки X / X+mirror | заменено собственных значений S_w в оболочке X / X+mirror | осей "
            + "MLDA X / X+mirror ===\n";

    static final String OWN_NOTE = "На своих считаются база и варианты, выбранные pool-vt хоть в одном случае; на каждый вариант обучаются основные модели "
            + "X / X+mirror\nи 5 моделей частей (порог П2/П3), классы VidTIMIT – только в выборке обучения W. Признаки L1-xs-F 7×7 и L1-tt-xs-F 9×9 (8, 2), "
            + "пол c = 1,\n1 − cos по всем осям, d₁/d₂, модель X / X+mirror по |r| пробы.\n";

    /** Разность «a − z» разделимости при FAR 1 % (способ way) и верхняя граница 97,5 % по бутстрепу (как XBaseKiss.post). */
    static double[] delta(XBaseThr.Out a, XBaseThr.Out z, String way, int persons, XBaseEval.Gt tpl) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        XBaseRule.Sep sa = XBaseRule.sep(a.res.get(way).cfgs, FeiBootstrap.unit(persons), w);
        XBaseRule.Sep sz = XBaseRule.sep(z.res.get(way).cfgs, FeiBootstrap.unit(persons), w);
        double[] d = FeiNorm.diff(sa.errRep[0], sz.errRep[0]);
        return new double[] {sa.err[0] - sz.err[0], d.length == 0 ? Double.NaN : FeiBootstrap.quantile(FeiBootstrap.sorted(d), 0.975)};
    }

    static List<String> trimOwn(List<String> own, int sm) {
        return sm == 1 && own.size() > XBaseDates.SMOKE_N ? own.subList(0, XBaseDates.SMOKE_N) : own;
    }

    static List<List<List<String>>> vtClasses(SettingsStore settings, Map<String, Mat> fr, int[] vs, StringBuilder head) throws IOException {
        Path vdir = VidTimitData.dir(settings);
        List<List<List<String>>> out = new ArrayList<>();
        for (int i = 0; i < NV - 1; i++) {
            final int v = i + 1;
            if (Arrays.stream(vs).noneMatch(u -> u == v)) {
                out.add(List.of());
                continue;
            }
            VidTimitData.Loaded l = VidTimitData.load(vdir, VidTimitData.VARIANT[i]);
            for (Map.Entry<String, Mat> e : l.frames.entrySet()) {
                Mat old = fr.putIfAbsent(e.getKey(), e.getValue());
                if (old != null) e.getValue().release();
            }
            out.add(l.classes);
            head.append(String.format(Locale.ROOT, "VidTIMIT %s: %s, отбор %s SHA-256 %s; людей %d, кадров %d (без отражений).%n", VNAME[v], vdir, VidTimitData.SELECTION,
                    l.sha256, l.persons, l.images));
        }
        return out;
    }

    // ---------------------------------------------------------------- feret-vt

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        int[] vs = ownVariants(select);
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        // Все отобранные снимки своих по датам – как XBaseDates и feret-kiss.
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
        for (Path f : List.of(lbpScores, lbpTxt, datesScores, datesTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-vt: нужен " + f);
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        Map<String, String> datesBlocks = XBaseSubsp.blocks(datesScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, java.nio.charset.StandardCharsets.UTF_8);
        List<String> datesReport = Files.readAllLines(datesTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        StringBuilder vhead = new StringBuilder();
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);
        List<List<List<String>>> vtCls = vtClasses(settings, fr, vs, vhead);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-vt %s; seed %d; r0 %s; пул %d; половины %d/%d; выбор %s; варианты %s; %s"
                + "пробный %d", s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), select, Arrays.toString(vs), vhead, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_vt_eq" + sfx), fp);
        XBaseEval[] xv = XBaseKiss.vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();

        StringBuilder t = new StringBuilder();
        t.append("Опыт «пул FERET + VidTIMIT в обучении W», FERET (XBaseEval feret-vt): Color FERET и VidTIMIT – только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ "
                + "РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(vhead);
        t.append(XBaseFeret.EQ_NOTE);
        t.append(OWN_NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-lbp); (ii) s = 3 – как feret-dates (люди с ≥ 4 датами, первые 3 "
                + "даты, пробы с 4-й,\nчасти – номер 71), справочно. «Выбор по пулу» – вариант из pool-vt (проход 1 – FERET-1, проход 2 – FERET-2). Контрольные – те же "
                + "200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = feret-lbp («межс.») и feret-dates ((ii) s3 (м)), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("постановка, признак, проход | вариант | ранг оболочки | заменено | осей\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        for (int post = 0; post < 2; post++) {
            String pname = post == 0 ? "межс." : "(ii) s3";
            int conf = post == 0 ? 0 : 71;
            List<XBaseKiss.Pass> passes = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (post == 0) {
                    own = trimOwn(ro.own1.stream().filter(mine::contains).toList(), sm);
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else {
                    own = trimOwn(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList(), sm);
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
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + conf);
                List<Probe> probes = XBaseEval.probes(g, pro);
                passes.add(new XBaseKiss.Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
            }
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                // [вариант][проход]; недостающие в контрольной точке – одной пачкой задач.
                XBaseLbp.One[][] got = new XBaseLbp.One[NV][2];
                List<XBaseLbp.Job> jobs = new ArrayList<>();
                List<int[]> todo = new ArrayList<>();
                for (int v : vs) {
                    for (int ps = 0; ps < 2; ps++) {
                        String tag = String.format(Locale.ROOT, "c%d_p%d_f%d_v%d", conf, ps, f, v);
                        got[v][ps] = XBaseLbp.read(cp, tag);
                        if (got[v][ps] != null) continue;
                        XBaseKiss.Pass p = passes.get(ps);
                        jobs.add(ownJob(p.g(), xv[f], p.gal(), p.probes(), p.parts(), conf, p.extra(), v, vtCls));
                        todo.add(new int[] {v, ps});
                    }
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<XBaseLbp.One> done = XBaseLbp.run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        int[] vp = todo.get(i);
                        got[vp[0]][vp[1]] = done.get(i);
                        XBaseLbp.save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d_v%d", conf, vp[1], f, vp[0]), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "feret-vt: %s %s ready, %d blocks (%.0f s)%n", pname, feat, jobs.size(), sec);
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
                        info.append(infoRow(pname + ", " + feat + ", проход " + (ps + 1), v, ls));
                        lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ").append(VNAME[v])
                                .append(", косинус\n").append(ls.rows()[1]);
                        units[v][ps] = new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), ls.rows()[1], null, true);
                    }
                }
                if (sm >= 2) {
                    t.append(check).append(info).append(String.format(Locale.ROOT, "Замер: «межс.» %s, 2 прохода × %d вариантов – %.0f с; групп всего 4 (2 постановки × "
                            + "2 признака).%n", feat, vs.length, fitS));
                    FarMethods.write(s.out.resolve("xbase_feret_vt" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int[] chosen = {select.get("FERET-1").get(feat), select.get("FERET-2").get(feat)};
                if (post == 0) sel.append(String.format(Locale.ROOT, "%s: проход 1 – %s, проход 2 – %s%n", feat, VNAME[chosen[0]], VNAME[chosen[1]]));
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = feat + " " + VNAME[BASE];
                for (int v : vs) {
                    String l = feat + " " + VNAME[v];
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
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_vt" + sfx + "_scores.tsv.gz"), "# FERET, пул FERET + VidTIMIT (XBaseVt feret-vt), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_vt" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-vt

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-vt: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        int[] vs = ownVariants(select);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new java.util.HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        StringBuilder vhead = new StringBuilder();
        List<List<List<String>>> vtCls = vtClasses(settings, fr, vs, vhead);
        XBaseEval[] xv = XBaseKiss.vectors(fr, g.params);
        for (Mat m : fr.values()) m.release();
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = trimOwn(g.own, sm);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        Path lbpTxt = out.resolve("xbase_gt_lbp.txt");
        if (!Files.exists(lbpScores) || !Files.exists(lbpTxt)) throw new IllegalStateException("gt-vt: нужны " + lbpScores + " и " + lbpTxt + " (gt-lbp)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-vt; seed %d; r0 %s; пул FERET %d; выбор %s; варианты %s; %sпробный %d", g.seed, g.r0,
                pool.size(), select, Arrays.toString(vs), vhead, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_vt" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("Опыт «пул FERET + VidTIMIT в обучении W», GT (XBaseEval gt-vt)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек, и VidTIMIT (только обучение; имён снимков FERET и VidTIMIT в отчёте и оценках нет)\n");
        t.append(vhead.toString().replaceAll("D:\\S*VidTIMIT, ", ""));
        t.append("Протокол как gt-lbp: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «вариант − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(OWN_NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = gt-lbp (межс. и внутри), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("режим, признак | вариант | ранг оболочки | заменено | осей\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        double fitS = 0;
        for (int mi = 0; mi < modes.length; mi++) {
            String mode = (String) modes[mi][0];
            boolean x = mi == 0;
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
            List<Probe> probes = XBaseEval.probes(ga, pro);
            List<List<Probe>> extra = x ? XBaseThr.extra(ga, pro, parts) : null;
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                XBaseLbp.One[] got = new XBaseLbp.One[NV];
                List<XBaseLbp.Job> jobs = new ArrayList<>();
                List<Integer> todo = new ArrayList<>();
                for (int v : vs) {
                    got[v] = XBaseLbp.read(cp, (x ? "x" : "i") + "_f" + f + "_v" + v);
                    if (got[v] != null) continue;
                    jobs.add(ownJob(ga, xv[f], gal, probes, parts, 0, extra, v, vtCls));
                    todo.add(v);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<XBaseLbp.One> done = XBaseLbp.run(jobs, g.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        got[todo.get(i)] = done.get(i);
                        XBaseLbp.save(cp, (x ? "x" : "i") + "_f" + f + "_v" + todo.get(i), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "gt-vt: %s %s ready, %d blocks (%.0f s)%n", mode, feat, jobs.size(), sec);
                }
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = mode + " " + feat + " " + VNAME[BASE];
                for (int v : vs) {
                    XBaseLbp.One ls = got[v];
                    if (v == BASE && sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", ls.rows()[1],
                            lbpBlocks, lbpScores));
                    info.append(infoRow(mode + ", " + feat, v, ls));
                    lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(VNAME[v]).append(", косинус\n").append(ls.rows()[1]);
                    String l = mode + " " + feat + " " + VNAME[v];
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
                    t.append(check).append(info).append(String.format(Locale.ROOT, "Замер: межс. %s, %d вариантов – %.0f с; групп всего 4 (2 режима × 2 признака).%n",
                            feat, vs.length, fitS));
                    FarMethods.write(out.resolve("xbase_gt_vt" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int c = select.get("GT").get(feat);
                String ch = mode + " " + feat + " " + XBaseLbp.CHOICE;
                outs.put(ch, XBaseThr.evaluate(ch, List.of(new XBaseThr.Unit(ga, gal, pro, parts, got[c].rows()[1], null, x)), g.persons.size(), g));
                pairs.add(new String[] {ch, bl});
                if (!x) {
                    for (String[] p : pairs) {
                        double[] d = delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], g.persons.size(), g);
                        boolean isChoice = p[0].equals(ch);
                        dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0] + (isChoice ? " (" + VNAME[c]
                                + ")" : ""), d[0], d[1], isChoice && c == BASE ? "не применяется (выбор – база)" : d[1] <= 3 ? "выполнено" : "НЕ выполнено"));
                    }
                }
                XBaseLbp.tables(body, "GT " + mode + ", " + feat + " (выбор по пулу – " + VNAME[c] + ")", outs, g.persons.size(), g, pairs, x);
            }
        }
        t.append(check).append(dl).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_vt" + sfx + "_scores.tsv.gz"), "# GT, пул FERET + VidTIMIT (XBaseVt gt-vt)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_vt" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
