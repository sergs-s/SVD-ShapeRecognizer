package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Этап «к цели», проба 2 – PLDA вместо MLDA (решение (39), Жорик №17 от 04.11): признак кандидата (32) → PCA обучающей выборки до 500 осей → PLDA (PldaEm) → LLR пробы против набора снимков шаблона.
 * Обучающая выборка – как у MLDA в (32): галерея, когорта, пул; у X+mirror – с отражениями, отражения входят и в набор регистрации, как в центр (32). Оценка кандидата – −LLR(g); слияние по человеку:
 * ℓ(g) = 0,75 · (−LLR_tt − μ_tt) / σ_tt + 0,25 · (−LLR_xs − μ_xs) / σ_xs, приведение – как (32) (пул – контрольные блока, иначе (Т2)); опознание – наименьшее ℓ; отказ: (а) ℓ₁, (б) ℓ₁ − ℓ₂. Значения −LLR – в поле dg строк.
 * pool-plda-f (кадры «а», 6 конфигураций) – выбор, файл выбора; gt-plda-f, feret-plda-f (кадры Б) – выбранная против (32). Пробный режим FACES_PLDA_SMOKE=1 (пул, 2 блока, замер).
 *
 * @author ssv
 */
final class XBasePldaF {

    private XBasePldaF() {
    }

    static final int PCA_M = 500;
    static final int[][] DIMS = {{128, 64}, {64, 32}, {32, 16}};
    static final int MAX_ITER = 100;
    static final double REL_TOL = 1e-6;
    static final String SELECT = "xbase_pool_plda_f_select.tsv";
    static final String[] RULES = {"а", "б"};

    private static String sizeId(int[] d) {
        return "f" + d[0] + "_g" + d[1];
    }

    /** Модели задачи: [размер] → модель, строка сведений. */
    private record Fit(XBaseVec.Model[] models, String[] info) {}

    private static Fit fit(XBaseEval x, XBaseEsrc.Task t, XBaseVec.Blk b, XBaseVec.Model fin, List<int[]> sizes, String tag) {
        Map<String, double[]> v = x.vec.get(V.L1);
        Map<String, double[]> mir = x.mir.get(V.L1);
        long t0 = System.nanoTime();
        List<List<double[]>> cls = new ArrayList<>();
        for (List<List<String>> group : List.of(t.gallery(), b.g().cohort)) {
            for (List<String> keys : group) {
                List<double[]> l = new ArrayList<>();
                for (String k : keys) l.add(v.get(k));
                if (t.mirror()) for (String k : keys) l.add(mir.get(k));
                cls.add(l);
            }
        }
        LdaMath.Data d = LdaMath.Data.of(cls);
        LdaMath.Span span = MldaFast.span(d, 1e-10, Integer.MAX_VALUE);
        int rank = span.basis().length;
        int axes = Math.min(PCA_M, rank);
        // классы в координатах PCA (порядок снимков – как в Data.of)
        List<double[][]> train = new ArrayList<>();
        int pos = 0;
        int ng = t.gallery().size();
        for (List<double[]> l : cls) {
            double[][] xs = new double[l.size()][];
            for (int i = 0; i < xs.length; i++) xs[i] = Arrays.copyOf(span.y()[pos++], axes);
            train.add(xs);
        }
        double secPca = (System.nanoTime() - t0) / 1e9;
        double[][] basis = Arrays.copyOf(span.basis(), axes);
        double[] mean = d.mean();
        double[][] probes = new double[t.probes().size()][];
        for (int i = 0; i < probes.length; i++) {
            if (!fin.rows().get(i).key().equals(t.probes().get(i).key())) throw new IllegalStateException("plda-f: пробы разошлись: " + t.probes().get(i).key());
            probes[i] = project(basis, v.get(t.probes().get(i).key()), mean);
        }
        XBaseVec.Model[] out = new XBaseVec.Model[sizes.size()];
        String[] info = new String[sizes.size()];
        for (int s = 0; s < sizes.size(); s++) {
            long ts = System.nanoTime();
            PldaEm m = PldaEm.train(train, sizes.get(s)[0], sizes.get(s)[1], MAX_ITER, REL_TOL);
            double secEm = (System.nanoTime() - ts) / 1e9;
            long tc = System.nanoTime();
            double[][] bigU = new double[ng][];
            int[] nImg = new int[ng];
            for (int g = 0; g < ng; g++) {
                double[][] xs = train.get(g);
                double[] sum = new double[axes];
                for (double[] q : xs) for (int i = 0; i < axes; i++) sum[i] += q[i];
                bigU[g] = m.u(sum);
                nImg[g] = xs.length;
            }
            List<XBaseVec.Row> rows = new ArrayList<>();
            for (int i = 0; i < probes.length; i++) {
                double[] u = m.u(probes[i]);
                double[] dg = new double[ng];
                for (int g = 0; g < ng; g++) dg[g] = -m.llr(bigU[g], nImg[g], u);
                XBaseVec.Row o = fin.rows().get(i);
                rows.add(new XBaseVec.Row(o.role(), o.person(), o.image(), o.key(), o.trueIdx(), dg, new double[0], new double[0]));
            }
            double msProbe = (System.nanoTime() - tc) / 1e6 / Math.max(1, probes.length);
            out[s] = new XBaseVec.Model(t.name(), t.method(), fin.s(), t.own(), rows, new double[ng][]);
            info[s] = String.format(Locale.ROOT, "%s\t%s\t%s\t%d\t%d\t%d\t%.6f\t%.2f\t%.2f\t%.3f%n", tag, t.name(), t.mirror() ? "X+mirror" : "X", rank, axes, m.iterations, m.logLik, secPca, secEm, msProbe);
        }
        return new Fit(out, info);
    }

    private static double[] project(double[][] basis, double[] x, double[] mean) {
        double[] y = new double[basis.length];
        for (int j = 0; j < y.length; j++) {
            double[] b = basis[j];
            double s = 0;
            for (int i = 0; i < b.length; i++) s += b[i] * (x[i] - mean[i]);
            y[j] = s;
        }
        return y;
    }

    private static Path vfile(Path root, int[] size, XBaseVec.Blk b, XBaseFinal.Cfg c) {
        return root.resolve(sizeId(size)).resolve("vec").resolve(b.name() + "_" + c.id() + ".tsv.gz");
    }

    private static Path ifile(Path root, int[] size, XBaseVec.Blk b, XBaseFinal.Cfg c) {
        return root.resolve(sizeId(size)).resolve("vec").resolve(b.name() + "_" + c.id() + "_info.tsv");
    }

    private static XBaseFinal.Run runOf(XBaseVec.Setup u, Path root, int[] size) {
        return new XBaseFinal.Run(u, Map.of(), Map.of(), root.resolve(sizeId(size)), false, 0);
    }

    /** μ, σ −LLR у «чужих» по всем людям галереи (как XBaseVec.fuseStats). */
    private static double[] stats(List<XBaseVec.Model> ms, boolean pool) {
        double s = 0;
        double ss = 0;
        long n = 0;
        for (XBaseVec.Model m : ms) {
            for (XBaseVec.Row r : m.rows()) {
                boolean imp = r.trueIdx() < 0 && (pool ? r.role().equals(FeiScores.CTRL) : r.role().startsWith(FeiMethods.CV_ROLE) || r.role().startsWith("od"));
                if (!imp) continue;
                for (double d : r.dg()) {
                    s += d;
                    ss += d * d;
                    n++;
                }
            }
        }
        if (n < 2) throw new IllegalStateException("plda-f: мало «чужих»");
        double m = s / n;
        return new double[] {m, Math.sqrt(Math.max(0, (ss - n * m * m) / (n - 1)))};
    }

    /** Конфигурация оценки блока по правилу отказа rule (0 – (а), 1 – (б)). */
    static FeiDecompose.Cfg score(XBaseVec.Blk b, boolean pool, List<XBaseVec.Model> mt, List<XBaseVec.Model> mx, int rule) {
        double[] st = stats(mt, pool);
        double[] sx = stats(mx, pool);
        XBaseVec.Model tX = XBaseVec.find(mt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(mt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(mx, "main", true);
        List<double[]> own = new ArrayList<>();
        List<Boolean> ok = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        Map<String, Integer> row = new HashMap<>();
        for (int i = 0; i < tX.rows().size(); i++) row.put(tX.rows().get(i).key(), i);
        List<Probe> pr = b.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList();
        for (Probe p : pr) {
            int i = row.get(p.key());
            XBaseVec.Row rt = b.g().rOf.get(p.key()) <= b.g().r0 ? tX.rows().get(i) : tM.rows().get(i);
            XBaseVec.Row rx = xM.rows().get(i);
            int ng = rt.dg().length;
            int bi = -1;
            double l1 = Double.POSITIVE_INFINITY;
            double l2 = Double.POSITIVE_INFINITY;
            for (int g = 0; g < ng; g++) {
                double l = XBaseFeat2.W_FUSE * (rt.dg()[g] - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (rx.dg()[g] - sx[0]) / sx[1];
                if (l < l1) {
                    l2 = l1;
                    l1 = l;
                    bi = g;
                } else if (l < l2) {
                    l2 = l;
                }
            }
            double s = rule == 0 ? l1 : l1 - l2;
            int pIdx = b.g().pIdx.get(p.person());
            if (p.role().equals(FeiScores.OWN)) {
                own.add(new double[] {pIdx, s});
                ok.add(bi == rt.trueIdx());
            } else {
                ctrl.add(new double[] {pIdx, s});
            }
        }
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        int n = own.size();
        c.ownP = new int[n];
        c.ownR = new double[n];
        c.ownOk = new boolean[n];
        c.noSecond = new boolean[n];
        c.trueInf = new boolean[n];
        c.poseFrontal = new boolean[n];
        for (int i = 0; i < n; i++) {
            c.ownP[i] = (int) own.get(i)[0];
            c.ownR[i] = own.get(i)[1];
            c.ownOk[i] = ok.get(i);
        }
        c.thrP = new int[0];
        c.thrR = new double[0];
        c.ctrlP = ctrl.stream().mapToInt(d -> (int) d[0]).toArray();
        c.ctrlR = ctrl.stream().mapToDouble(d -> d[1]).toArray();
        return c;
    }

    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        boolean pool = base.equals("pool");
        if (pool == FrameB.on()) throw new IllegalStateException("plda-f: пул – кадры «а», GT и FERET – FACES_FRAMES=b");
        boolean smoke = pool && "1".equals(System.getenv("FACES_PLDA_SMOKE"));
        long t0 = System.nanoTime();
        XBaseFinal.Run r = pool ? null : XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = pool ? XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0) : r.u();
        Map<String, FrameK2.F> k2 = pool ? null : r.k2();
        List<String> groups = base.equals("gt") ? List.of("межс.", "внутри") : base.equals("feret") ? List.of("межс.") : u.groups;
        List<XBaseVec.Blk> bs0 = u.blocks.stream().filter(b -> groups.contains(b.group())).toList();
        List<XBaseVec.Blk> bs = smoke ? bs0.subList(0, 2) : bs0;
        Path eqDir = u.su.dir.resolve("reports").resolve("eq");
        Path selFile = eqDir.resolve(SELECT);
        List<int[]> sizes;
        int ruleSel = -1;
        if (pool) {
            sizes = Arrays.asList(DIMS);
        } else {
            String[] sel = Files.readAllLines(selFile).stream().filter(l -> l.startsWith("выбор\t")).findFirst().orElseThrow(() -> new IllegalStateException("plda-f: нет выбора " + selFile)).split("\t");
            sizes = Arrays.stream(DIMS).filter(d -> sizeId(d).equals(sel[1])).toList();
            ruleSel = sel[2].equals(RULES[0]) ? 0 : 1;
        }
        Path root = pool ? eqDir.resolve("plda_f") : r.dir().resolveSibling("plda_f");
        Files.createDirectories(root);
        Set<String> need = new HashSet<>();
        for (XBaseVec.Blk b : bs) {
            b.gal().values().forEach(need::addAll);
            b.g().cohort.forEach(need::addAll);
            for (Probe q : b.probes()) if (!q.role().equals(XBaseFinal.CTRL_X)) need.add(q.key());
            if (b.extra() != null) for (List<Probe> l : b.extra()) for (Probe q : l) need.add(q.key());
        }
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        if (!fr.keySet().containsAll(need)) throw new IllegalStateException("plda-f: не все кадры загружены");
        if (pool) k2 = FrameK2.load(fr.keySet(), u.su, u.su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        int threads = u.tpl.threads;
        String code = FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBasePldaF.java") + " " + FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/PldaEm.java");
        Map<String, Double> sec = new LinkedHashMap<>();
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            long tf = System.nanoTime();
            XBaseEval x = XBasePart4.vectorsB(fr, k2, XBaseFinal.param(c, u.tpl.params), c.f(), threads);
            sec.merge("признаки", (System.nanoTime() - tf) / 1e9, Double::sum);
            int done = 0;
            for (XBaseVec.Blk b : bs) {
                boolean have = true;
                for (int[] s : sizes) have &= Files.exists(vfile(root, s, b, c));
                if (have) {
                    for (int[] s : sizes) if (!XBaseVec.readHead(vfile(root, s, b, c)).contains(code)) throw new IllegalStateException("plda-f: другой код: " + vfile(root, s, b, c));
                    done++;
                    continue;
                }
                long ts0 = System.nanoTime();
                List<XBaseVec.Model> fins = pool ? XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, c.f(), 8) : XBaseFinal.models(r, b, c, false);
                List<XBaseEsrc.Task> ts = XBaseEsrc.tasks(b, pool);
                String tag = b.name() + "\t" + c.id();
                List<Callable<Fit>> jobs = new ArrayList<>();
                for (XBaseEsrc.Task t : ts) jobs.add(() -> fit(x, t, b, XBaseVec.find(fins, t.name(), t.mirror()), sizes, tag));
                java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(Math.min(8, threads));
                List<Fit> fits = new ArrayList<>();
                try {
                    for (java.util.concurrent.Future<Fit> fu : ex.invokeAll(jobs)) fits.add(fu.get());
                } finally {
                    ex.shutdown();
                }
                for (int s = 0; s < sizes.size(); s++) {
                    List<XBaseVec.Model> ms = new ArrayList<>();
                    StringBuilder inf = new StringBuilder();
                    for (Fit f : fits) {
                        ms.add(f.models()[s]);
                        inf.append(f.info()[s]);
                    }
                    Path vf = vfile(root, sizes.get(s), b, c);
                    Files.createDirectories(vf.getParent());
                    java.io.Writer w = XBaseVec.writer(vf, "# −LLR PLDA (XBasePldaF): " + sizeId(sizes.get(s)) + ", " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; код " + code + "\n");
                    XBaseVec.write(w, b.name() + " " + sizeId(sizes.get(s)) + " " + c.name(), ms);
                    XBaseVec.finish(w, vf);
                    Files.writeString(ifile(root, sizes.get(s), b, c), inf.toString());
                }
                sec.merge("обучение и оценка", (System.nanoTime() - ts0) / 1e9, Double::sum);
                System.out.printf(Locale.ROOT, "plda-f %s: %s блок %s, %d из %d (%.0f с)%n", base, c.id(), b.name(), ++done, bs.size(), (System.nanoTime() - t0) / 1e9);
            }
            x.vec.clear();
            x.mir.clear();
        }
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : k2.values()) k.frame().release();
        StringBuilder t = new StringBuilder("Этап «к цели», проба 2 – PLDA вместо MLDA – " + base + (smoke ? " – пробный режим" : "") + " (XBaseEval " + base + "-plda-f; решение (39), Жорик №17)\nКод: коммит " + FarMethods.commit()
                + String.format(Locale.ROOT, "%nPCA обучающей выборки до %d осей; EM – начало по статье (SVD разброса, seed не используется), остановка: относительное изменение правдоподобия < %.0e или %d итераций; X+mirror – отражения и в обучении, и в наборе регистрации.%n", PCA_M, REL_TOL, MAX_ITER));
        t.append(infoSummary(root, sizes, bs, !pool));
        if (pool && !smoke) t.append(selectPool(u, root, bs, selFile));
        if (!pool) t.append(check(r, root, bs, groups, u, sizes.get(0), ruleSel));
        t.append("\n=== Время ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f с%n", k, v)));
        t.append(String.format(Locale.ROOT, "%nВсего: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        if (smoke) {
            double est = sec.getOrDefault("обучение и оценка", 0.0) * u.blocks.size() / bs.size() + sec.getOrDefault("признаки", 0.0);
            t.append(String.format(Locale.ROOT, "Оценка полного прогона пула (блоков %d): %.1f ч%n", u.blocks.size(), est / 3600));
        }
        FarMethods.write(root.resolve("xbase_" + base + "_plda_f" + (smoke ? "_smoke" : "") + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    private static String infoSummary(Path root, List<int[]> sizes, List<XBaseVec.Blk> bs, boolean perModel) throws IOException {
        StringBuilder t = new StringBuilder("\n=== PCA и EM по моделям ===\nразмер | моделей | ранг | осей | итераций мин / медиана / макс | дошло до предела | log p мин / макс | с PCA | с EM | мс оценки на пробу (один поток)\n");
        StringBuilder all = new StringBuilder("\nблок | признак | задача | модель | ранг | осей | итераций | log p | с PCA | с EM | мс на пробу\n");
        for (int[] s : sizes) {
            List<String[]> rows = new ArrayList<>();
            for (XBaseVec.Blk b : bs) {
                for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
                    for (String l : Files.readAllLines(ifile(root, s, b, c))) {
                        if (l.isBlank()) continue;
                        String[] p = l.split("\t");
                        rows.add(p);
                        if (perModel) all.append(sizeId(s)).append(' ').append(String.join(" | ", p)).append('\n');
                    }
                }
            }
            int[] iters = rows.stream().mapToInt(p -> Integer.parseInt(p[6])).sorted().toArray();
            int[] ranks = rows.stream().mapToInt(p -> Integer.parseInt(p[4])).sorted().toArray();
            int[] axes = rows.stream().mapToInt(p -> Integer.parseInt(p[5])).sorted().toArray();
            double[] ll = rows.stream().mapToDouble(p -> Double.parseDouble(p[7])).sorted().toArray();
            long atMax = Arrays.stream(iters).filter(i -> i >= MAX_ITER).count();
            t.append(String.format(Locale.ROOT, "%s | %d | %d–%d | %d–%d | %d / %d / %d | %d | %.1f / %.1f | %.1f | %.1f | %.2f%n", sizeId(s), rows.size(), ranks[0], ranks[ranks.length - 1], axes[0], axes[axes.length - 1], iters[0], iters[iters.length / 2],
                    iters[iters.length - 1], atMax, ll[0], ll[ll.length - 1], rows.stream().mapToDouble(p -> Double.parseDouble(p[8])).average().orElse(0), rows.stream().mapToDouble(p -> Double.parseDouble(p[9])).average().orElse(0),
                    rows.stream().mapToDouble(p -> Double.parseDouble(p[10])).average().orElse(0)));
        }
        return perModel ? t.append(all).toString() : t.toString();
    }

    private static String selectPool(XBaseVec.Setup u, Path root, List<XBaseVec.Blk> bs, Path selFile) throws Exception {
        List<String> sets = u.groups;
        int n = u.persons;
        int[][] w = FeiBootstrap.weights(n, XBasePart4.NOISE_B, u.tpl.seed + XBasePart4.NOISE_SEED_OFFSET);
        int[] unit = FeiBootstrap.unit(n);
        Map<String, FeiDecompose.Cfg[]> rows = new LinkedHashMap<>();
        FeiDecompose.Cfg[] baseC = new FeiDecompose.Cfg[sets.size()];
        int[] flat = new int[1];
        for (int s = 0; s < sets.size(); s++) {
            List<FeiDecompose.Cfg> acc = new ArrayList<>();
            for (XBaseVec.Blk b : bs) {
                if (b.group().equals(sets.get(s))) acc.add(XBaseEsrcLam.cfg(b, true, XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, 1, 8), XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, 0, 8), false, flat));
            }
            baseC[s] = XBaseRule.merge(acc);
        }
        rows.put("(32)", baseC);
        for (int[] d : DIMS) {
            XBaseFinal.Run rs = runOf(u, root, d);
            for (int rule = 0; rule < 2; rule++) {
                FeiDecompose.Cfg[] cs = new FeiDecompose.Cfg[sets.size()];
                for (int s = 0; s < sets.size(); s++) {
                    List<FeiDecompose.Cfg> acc = new ArrayList<>();
                    for (XBaseVec.Blk b : bs) {
                        if (b.group().equals(sets.get(s))) acc.add(score(b, true, XBaseFinal.models(rs, b, XBaseFinal.TT1, false), XBaseFinal.models(rs, b, XBaseFinal.XS, false), rule));
                    }
                    cs[s] = XBaseRule.merge(acc);
                }
                rows.put(sizeId(d) + "\t" + RULES[rule], cs);
            }
        }
        Map<String, double[]> pts = new LinkedHashMap<>();
        Map<String, double[][]> reps = new LinkedHashMap<>();
        ForkJoinPool fj = new ForkJoinPool(u.tpl.threads);
        for (Map.Entry<String, FeiDecompose.Cfg[]> e : rows.entrySet()) {
            double[] pt = new double[sets.size()];
            double[][] rp = new double[sets.size()][XBasePart4.NOISE_B];
            for (int s = 0; s < sets.size(); s++) {
                FeiDecompose.Cfg c = e.getValue()[s];
                pt[s] = XBasePart4.critW(c, unit);
                double[] o = rp[s];
                fj.submit(() -> IntStream.range(0, XBasePart4.NOISE_B).parallel().forEach(q -> o[q] = XBasePart4.critW(c, w[q]))).get();
            }
            pts.put(e.getKey(), pt);
            reps.put(e.getKey(), rp);
        }
        fj.shutdown();
        StringBuilder t = new StringBuilder(String.format(Locale.ROOT, "%n=== Выбор по пулу (кадры «а»): критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; бутстреп по людям B = %d, seed %d; база – векторы v8 pool-feat2 ===%nвыбирается наименьшее среднее по трём наборам; «*» – парная разность с (32) значима (95 %% без нуля)%nконфигурация",
                XBasePart4.NOISE_B, u.tpl.seed + XBasePart4.NOISE_SEED_OFFSET));
        for (String s : sets) t.append(" | ").append(s);
        t.append(" | среднее | разность с (32) по наборам [95 %]\n");
        double[] p0 = pts.get("(32)");
        String best = null;
        double bestV = Double.POSITIVE_INFINITY;
        boolean anyBetter = false;
        for (Map.Entry<String, double[]> e : pts.entrySet()) {
            double m = Arrays.stream(e.getValue()).average().orElseThrow();
            t.append(e.getKey().replace('\t', ' '));
            for (double v : e.getValue()) t.append(String.format(Locale.ROOT, " | %.2f", v));
            t.append(String.format(Locale.ROOT, " | %.2f |", m));
            if (!e.getKey().equals("(32)")) {
                for (int s = 0; s < sets.size(); s++) {
                    double[] sorted = FeiBootstrap.sorted(FeiNorm.diff(reps.get(e.getKey())[s], reps.get("(32)")[s]));
                    double lo = FeiBootstrap.quantile(sorted, 0.025);
                    double hi = FeiBootstrap.quantile(sorted, 0.975);
                    anyBetter |= hi < 0;
                    t.append(String.format(Locale.ROOT, " %s %+.2f [%+.2f; %+.2f]%s;", sets.get(s), e.getValue()[s] - p0[s], lo, hi, lo > 0 || hi < 0 ? " *" : ""));
                }
                if (m < bestV) {
                    bestV = m;
                    best = e.getKey();
                }
            }
            t.append('\n');
        }
        t.append(String.format(Locale.ROOT, "%nВыбрана: %s (среднее %.2f против %.2f у (32)).%n%s%n", best.replace('\t', ' '), bestV, Arrays.stream(p0).average().orElseThrow(),
                anyBetter ? "Хотя бы одна конфигурация значимо лучше базы хотя бы на одном наборе." : "Ни одна не лучше базы значимо ни на одном наборе; на GT всё равно проверяется (№17, п. 5)."));
        Files.writeString(selFile, "выбор\t" + best + "\n");
        return t.toString();
    }

    private static String check(XBaseFinal.Run r, Path root, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u, int[] size, int rule) throws IOException {
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        XBaseFinal.Run rs = runOf(u, root, size);
        StringBuilder t = new StringBuilder("\n=== Выбранная по пулу: " + sizeId(size) + ", правило (" + RULES[rule] + ") − (32); одна строка, основной интервал 95 % ===\n");
        for (String grp : groups) {
            List<XBaseVec.Blk> gb = bs.stream().filter(b -> b.group().equals(grp)).toList();
            List<FeiDecompose.Cfg> csB = new ArrayList<>();
            List<FeiDecompose.Cfg> csS = new ArrayList<>();
            for (XBaseVec.Blk b : gb) {
                csB.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                csS.add(score(b, false, XBaseFinal.models(rs, b, XBaseFinal.TT1, false), XBaseFinal.models(rs, b, XBaseFinal.XS, false), rule));
            }
            XBaseFinalReport.M mB = XBaseFinalReport.measure(csB, u.persons, w);
            XBaseFinalReport.M mS = XBaseFinalReport.measure(csS, u.persons, w);
            t.append("\n").append(grp).append(": строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.level("(32) слияние", mB)).append(XBaseEsrc.level("PLDA, слияние", mS));
            t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.row("PLDA − (32)", mS, mB));
        }
        return t.toString();
    }
}
