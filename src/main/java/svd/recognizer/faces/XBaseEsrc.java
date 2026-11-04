package svd.recognizer.faces;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
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
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Этап «к цели», проба 1 – ESRC (решение (39); Жорик №12, №13 от 04.11): gt-esrc (межс., внутри), feret-esrc (межс.). Кадры Б, блоки, модели и части –
 * как итоговый прогон. y = Aα + Dβ + z: A – центры галереи (один столбец на человека), D – словарь: снимок пула FERET (оригинал) − центр снимков
 * человека; столбцы и y единичной длины; ‖y − Aα − Dβ‖₂ ≤ ε = 0,05 (EsrcSolver); невязка r_g = ‖y − A_g α_g − Dβ‖₂ идёт вместо 1 − cos, дальше –
 * прежний код: приведения (Т2), слияние, s = 2ℓ₁ − ℓ₂. Две заранее объявленные строки: W – пространство MLDA той же модели; PCA – 500 первых
 * компонент пула без W (у FERET межс. – вся оболочка половины пула). Вложения W – контрольная точка; при каждой загрузке 1 − cos по вложениям
 * сверяется с векторами итогового прогона (≤ 1e-12), иначе остановка. Пробный режим FACES_ESRC_SMOKE=1: GT межс., только основные модели, только
 * сведения о решении, включая долю проб с r₁ = r₂.
 *
 * @author ssv
 */
final class XBaseEsrc {

    private XBaseEsrc() {
    }

    static final double EPS = 0.05;
    static final int PCA_M = 500;
    static final double TIE = 1e-12;

    /** Вложения одной модели: центры, атомы (сырые разности), пробы – в том же порядке, что строки модели итогового прогона. */
    record Emb(double[][] centers, double[][] atoms, List<String> keys, double[][] probes) implements Serializable {
        private static final long serialVersionUID = 1L;
    }

    /** Задача: модель блока. */
    record Task(boolean mirror, String name, List<String> own, List<List<String>> gallery, List<Probe> probes) {
        String file(XBaseVec.Blk b, XBaseFinal.Cfg c) {
            return b.name() + "_" + c.id() + "_" + (mirror ? "m" : "x") + "_" + name.replace(' ', '_') + ".ser.gz";
        }

        String method() {
            return V.L1.id + (mirror ? FeiMethods.MIRROR : "");
        }
    }

    static List<Task> tasks(XBaseVec.Blk b, boolean mainOnly) {
        XBaseEval.Gt gt = b.g();
        List<String> own = gt.own;
        List<Probe> main = b.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList();
        List<List<String>> full = own.stream().map(b.gal()::get).toList();
        List<Task> out = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            out.add(new Task(m, "main", own, full, main));
            if (mainOnly) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(b.parts().get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(b.gal()::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : b.gal().get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (b.extra() != null) pr.addAll(b.extra().get(q));
                out.add(new Task(m, "part " + q, g, gk, pr));
            }
        }
        return out;
    }

    /** Классы пула FERET в обучении W блока: g.cohort без первых 20 (когорта FEI). */
    static List<List<String>> poolOf(XBaseVec.Blk b) {
        List<List<String>> c = b.g().cohort;
        if (c.size() <= XBaseEval.COHORT) throw new IllegalStateException("esrc: нет пула в обучении");
        return c.subList(XBaseEval.COHORT, c.size());
    }

    private static double[] diff(double[] a, double[] m) {
        double[] r = new double[a.length];
        for (int i = 0; i < r.length; i++) r[i] = a[i] - m[i];
        return r;
    }

    static double[][] atoms(List<List<String>> pool, java.util.function.Function<String, double[]> f) {
        List<double[]> out = new ArrayList<>();
        for (List<String> cls : pool) {
            List<double[]> ys = cls.stream().map(f).toList();
            double[] m = new double[ys.get(0).length];
            for (double[] y : ys) for (int i = 0; i < m.length; i++) m[i] += y[i];
            for (int i = 0; i < m.length; i++) m[i] /= ys.size();
            for (double[] y : ys) out.add(diff(y, m));
        }
        return out.toArray(new double[0][]);
    }

    private static Emb embW(XBaseEval x, Task t, List<List<String>> cohort, List<List<String>> pool) {
        LdaModel lm = XBaseRule.fit(x, t.mirror(), t.gallery(), cohort, null).lm;
        Map<String, double[]> v = x.vec.get(V.L1);
        double[][] ys = new double[t.probes().size()][];
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < ys.length; i++) {
            keys.add(t.probes().get(i).key());
            ys[i] = lm.embed(v.get(keys.get(i)));
        }
        return new Emb(lm.centers(), atoms(pool, k -> lm.embed(v.get(k))), keys, ys);
    }

    /** PCA пула: среднее и строки базиса. */
    record Basis(double[] mean, double[][] rows, int rank) {
        double[] project(double[] x) {
            double[] y = new double[rows.length];
            for (int j = 0; j < y.length; j++) {
                double[] b = rows[j];
                double s = 0;
                for (int i = 0; i < b.length; i++) s += b[i] * (x[i] - mean[i]);
                y[j] = s;
            }
            return y;
        }
    }

    private static Basis pca(XBaseEval x, List<List<String>> pool) {
        List<double[]> xs = new ArrayList<>();
        for (List<String> cls : pool) for (String k : cls) xs.add(x.vec.get(V.L1).get(k));
        XBaseRule.Pca p = XBaseRule.Pca.of(xs);
        int m = Math.min(PCA_M, p.basis().length);
        return new Basis(p.mean(), Arrays.copyOf(p.basis(), m), p.basis().length);
    }

    private static Emb embP(XBaseEval x, Task t, Basis b, List<List<String>> pool, Map<String, double[]> cache) {
        Map<String, double[]> v = x.vec.get(V.L1);
        Map<String, double[]> w = x.mir.get(V.L1);
        double[][] c = new double[t.gallery().size()][];
        for (int g = 0; g < c.length; g++) {
            List<String> keys = t.gallery().get(g);
            double[] m = new double[b.mean().length];
            int n = 0;
            for (String k : keys) {
                double[] a = v.get(k);
                for (int i = 0; i < m.length; i++) m[i] += a[i];
                n++;
                if (t.mirror()) {
                    double[] z = w.get(k);
                    for (int i = 0; i < m.length; i++) m[i] += z[i];
                    n++;
                }
            }
            for (int i = 0; i < m.length; i++) m[i] /= n;
            c[g] = b.project(m);
        }
        double[][] ys = new double[t.probes().size()][];
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < ys.length; i++) {
            String k = t.probes().get(i).key();
            keys.add(k);
            ys[i] = cache.computeIfAbsent(k, q -> b.project(v.get(q)));
        }
        return new Emb(c, atoms(pool, k -> cache.computeIfAbsent(k, q -> b.project(v.get(q)))), keys, ys);
    }

    /** Исходное признаковое пространство (Жорик №15): центр людей галереи как в модели, пробы и атомы – сами признаки; без вычитания общего среднего. */
    static Emb embRaw(XBaseEval x, Task t, double[][] atoms) {
        Map<String, double[]> v = x.vec.get(V.L1);
        Map<String, double[]> w = x.mir.get(V.L1);
        int n0 = v.values().iterator().next().length;
        double[][] c = new double[t.gallery().size()][];
        for (int g = 0; g < c.length; g++) {
            double[] m = new double[n0];
            int n = 0;
            for (String k : t.gallery().get(g)) {
                double[] a = v.get(k);
                for (int i = 0; i < n0; i++) m[i] += a[i];
                n++;
                if (t.mirror()) {
                    double[] z = w.get(k);
                    for (int i = 0; i < n0; i++) m[i] += z[i];
                    n++;
                }
            }
            for (int i = 0; i < n0; i++) m[i] /= n;
            c[g] = m;
        }
        double[][] ys = new double[t.probes().size()][];
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < ys.length; i++) {
            keys.add(t.probes().get(i).key());
            ys[i] = v.get(keys.get(i));
        }
        return new Emb(c, atoms, keys, ys);
    }

    static double[] unit(double[] a) {
        double s = 0;
        for (double v : a) s += v * v;
        s = Math.sqrt(s);
        if (s < 1e-12) return null;
        double[] r = new double[a.length];
        for (int i = 0; i < r.length; i++) r[i] = a[i] / s;
        return r;
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /**
     * SCI (Wright и др., TPAMI 2009): (k · max_g ‖α_g‖₁ / ‖α‖₁ − 1) / (k − 1), только по коэффициентам галереи; у нас один столбец на человека,
     * k – число людей галереи; ‖α‖₁ = 0 – 0 (худшая оценка).
     */
    static double sci(double[] alpha, int k) {
        double sum = 0;
        double max = 0;
        for (double a : alpha) {
            double v = Math.abs(a);
            sum += v;
            max = Math.max(max, v);
        }
        if (sum == 0) return 0;
        return (k * max / sum - 1) / (k - 1);
    }

    /** Сведения о решении основной модели. */
    static final class Stat {
        long[] n = new long[2];
        long[] ties = new long[2];
        long[] noA = new long[2];
        double[] atoms = new double[2];
        double[] cols = new double[2];
        double[] steps = new double[2];
        double nanos;
        long solves;
        long capped;
        long reached;
        double rEndSum;
        double rEndMax;
        int d;
        int p;
    }

    /** Решение всех проб модели: невязки по людям галереи; final – строки модели итогового прогона (те же пробы). */
    static XBaseVec.Model solve(Emb e, Task t, XBaseVec.Model fin, ForkJoinPool pool, Stat st) throws Exception {
        int ng = e.centers().length;
        List<double[]> colsL = new ArrayList<>();
        int[] colOfG = new int[ng];
        for (int g = 0; g < ng; g++) {
            double[] u = e.centers()[g] == null ? null : unit(e.centers()[g]);
            colOfG[g] = u == null ? -1 : colsL.size();
            if (u != null) colsL.add(u);
        }
        int nA = colsL.size();
        for (double[] a : e.atoms()) {
            double[] u = unit(a);
            if (u != null) colsL.add(u);
        }
        double[][] phi = colsL.toArray(new double[0][]);
        int p = phi.length;
        double[][] gram = new double[p][p];
        pool.submit(() -> IntStream.range(0, p).parallel().forEach(i -> {
            for (int j = i; j < p; j++) gram[i][j] = dot(phi[i], phi[j]);
        })).get();
        for (int i = 0; i < p; i++) for (int j = 0; j < i; j++) gram[i][j] = gram[j][i];
        int d = phi[0].length;
        int maxSteps = 20 * Math.max(1, Math.min(p, d));
        int nr = e.probes().length;
        XBaseVec.Row[] rows = new XBaseVec.Row[nr];
        double[][] info = new double[nr][6];
        pool.submit(() -> IntStream.range(0, nr).parallel().forEach(i -> {
            long t0 = System.nanoTime();
            double[] y = unit(e.probes()[i]);
            double[] c = new double[p];
            for (int j = 0; j < p; j++) c[j] = dot(phi[j], y);
            EsrcSolver.Result r = EsrcSolver.solve(gram, c, 1.0, EPS, maxSteps);
            double[] x = r.x();
            // база = y − Dβ; r_g = ‖база − столбец g · α_g‖
            double[] bs = y.clone();
            int nzAtoms = 0;
            int nzCols = 0;
            for (int j = 0; j < p; j++) {
                if (x[j] == 0) continue;
                if (j < nA) {
                    nzCols++;
                } else {
                    nzAtoms++;
                    double[] col = phi[j];
                    for (int q = 0; q < d; q++) bs[q] -= x[j] * col[q];
                }
            }
            double[] dg = new double[ng];
            double bb = dot(bs, bs);
            for (int g = 0; g < ng; g++) {
                int j = colOfG[g];
                if (j < 0) {
                    dg[g] = Double.POSITIVE_INFINITY;
                } else if (x[j] == 0) {
                    dg[g] = Math.sqrt(bb);
                } else {
                    double s = 0;
                    double[] col = phi[j];
                    for (int q = 0; q < d; q++) {
                        double v = bs[q] - x[j] * col[q];
                        s += v * v;
                    }
                    dg[g] = Math.sqrt(s);
                }
            }
            XBaseVec.Row o = fin.rows().get(i);
            if (!o.key().equals(e.keys().get(i))) throw new IllegalStateException("esrc: пробы разошлись: " + o.key());
            // Поле dci: SCI и ‖Aα‖² (‖y‖ = 1) – оценка отказа по концентрации коэффициентов галереи
            double[] alpha = new double[nA];
            double[] aa = new double[d];
            for (int j = 0; j < nA; j++) {
                alpha[j] = x[j];
                if (x[j] == 0) continue;
                double[] col = phi[j];
                for (int q = 0; q < d; q++) aa[q] += x[j] * col[q];
            }
            rows[i] = new XBaseVec.Row(o.role(), o.person(), o.image(), o.key(), o.trueIdx(), dg, new double[] {sci(alpha, ng), dot(aa, aa)}, new double[0]);
            info[i] = new double[] {nzAtoms, nzCols, r.steps(), r.capped() ? 1 : 0, System.nanoTime() - t0, Math.sqrt(Math.max(r.rr(), 0))};
        })).get();
        if (st != null) {
            st.d = d;
            st.p = p;
            for (int i = 0; i < nr; i++) {
                st.nanos += info[i][4];
                st.solves++;
                st.capped += (long) info[i][3];
                if (info[i][5] <= EPS + 1e-9) st.reached++;
                st.rEndSum += info[i][5];
                st.rEndMax = Math.max(st.rEndMax, info[i][5]);
                XBaseVec.Row o = rows[i];
                boolean own = o.role().equals(FeiScores.OWN);
                if (!own && !o.role().equals(FeiScores.CTRL)) continue;
                int a = own ? 0 : 1;
                st.n[a]++;
                st.atoms[a] += info[i][0];
                st.cols[a] += info[i][1];
                st.steps[a] += info[i][2];
                if (info[i][1] == 0) st.noA[a]++;
                double[] v = o.dg().clone();
                Arrays.sort(v);
                if (v.length > 1 && v[1] - v[0] <= TIE) st.ties[a]++;
            }
        }
        return new XBaseVec.Model(t.name(), t.method(), fin.s(), t.own(), Arrays.asList(rows), new double[ng][]);
    }

    private static String statLine(String lab, Stat s) {
        return String.format(Locale.ROOT, "%s | %d × %d | %.2f %% (%d/%d) | %.2f %% (%d/%d) | %.2f %% / %.2f %% | %.1f / %.1f | %.2f / %.2f | %.1f / %.1f | %.1f | %d | %.1f %% | %.4f / %.4f%n", lab, s.d, s.p,
                100.0 * s.ties[0] / Math.max(1, s.n[0]), s.ties[0], s.n[0], 100.0 * s.ties[1] / Math.max(1, s.n[1]), s.ties[1], s.n[1], 100.0 * s.noA[0] / Math.max(1, s.n[0]),
                100.0 * s.noA[1] / Math.max(1, s.n[1]), s.atoms[0] / Math.max(1, s.n[0]), s.atoms[1] / Math.max(1, s.n[1]), s.cols[0] / Math.max(1, s.n[0]), s.cols[1] / Math.max(1, s.n[1]),
                s.steps[0] / Math.max(1, s.n[0]), s.steps[1] / Math.max(1, s.n[1]), s.nanos / Math.max(1, s.solves) / 1e6, s.capped, 100.0 * s.reached / Math.max(1, s.solves), s.rEndSum / Math.max(1, s.solves), s.rEndMax);
    }

    static final String STAT_HEAD = "пространство, блок, признак, модель | d × p | r₁ = r₂, свои | r₁ = r₂, контрольные | без столбцов A, свои / контр. | атомов D, свои / контр. | столбцов A, свои / контр. "
            + "| шагов, свои / контр. | мс на решение (один поток) | оборвано по пределу | ε достигнуто | невязка в конце, среднее / макс.\n";

    private static void writeEmb(Path f, Emb e) throws IOException {
        Path tmp = f.resolveSibling(f.getFileName() + ".tmp");
        try (ObjectOutputStream o = new ObjectOutputStream(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16))) {
            o.writeObject(e);
        }
        Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    private static Emb readEmb(Path f) throws IOException {
        try (ObjectInputStream o = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(f), 1 << 16))) {
            return (Emb) o.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException(e);
        }
    }

    /** Сверка: 1 − cos по вложениям = векторы итогового прогона. */
    private static void check(Emb e, XBaseVec.Model fin, String lab) {
        checkDiff(e, fin, lab);
    }

    static double checkDiff(Emb e, XBaseVec.Model fin, String lab) {
        if (fin.rows().size() != e.probes().length) throw new IllegalStateException("esrc: число проб ≠ итоговый прогон: " + lab);
        double worst = 0;
        for (int i = 0; i < e.probes().length; i++) {
            XBaseVec.Row o = fin.rows().get(i);
            if (!o.key().equals(e.keys().get(i))) throw new IllegalStateException("esrc: порядок проб ≠ итоговый прогон: " + lab);
            for (int g = 0; g < e.centers().length; g++) {
                if (e.centers()[g] == null) continue;
                worst = Math.max(worst, Math.abs(XBaseVec.cos(e.probes()[i], e.centers()[g]) - o.dg()[g]));
            }
        }
        if (worst > 1e-12) throw new IllegalStateException("esrc: вложения ≠ векторы итогового прогона, " + lab + ": " + worst);
        return worst;
    }

    private static String treeOfSrc() {
        try {
            return FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBaseEsrc.java") + " " + FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/EsrcSolver.java");
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    static void run(String base, boolean sci, boolean raw, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("esrc: FACES_FRAMES=b");
        double[] worstDg = {0};
        boolean smoke = "1".equals(System.getenv("FACES_ESRC_SMOKE"));
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        List<String> groups = base.equals("gt") ? (smoke ? List.of("межс.") : List.of("межс.", "внутри")) : List.of("межс.");
        List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> groups.contains(b.group())).toList();
        Set<String> need = new HashSet<>();
        for (XBaseVec.Blk b : bs) {
            b.gal().values().forEach(need::addAll);
            b.pro().values().forEach(need::addAll);
            b.g().cohort.forEach(need::addAll);
            for (Probe q : b.probes()) if (!q.role().equals(XBaseFinal.CTRL_X)) need.add(q.key());
            if (b.extra() != null) for (List<Probe> l : b.extra()) for (Probe q : l) need.add(q.key());
        }
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) {
            if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        }
        if (!fr.keySet().containsAll(need)) throw new IllegalStateException("esrc: не все кадры загружены");
        Path root = r.dir().resolveSibling("esrc");
        Path embDir = root.resolve("emb");
        Files.createDirectories(embDir);
        int threads = u.tpl.threads;
        int fitThreads = Math.min(8, threads);
        ForkJoinPool pool = new ForkJoinPool(threads);
        String code = treeOfSrc();
        StringBuilder stats = new StringBuilder(STAT_HEAD);
        StringBuilder vol = new StringBuilder();
        Map<String, Double> sec = new LinkedHashMap<>();
        List<XBaseFinal.Cfg> cfgs = List.of(XBaseFinal.TT1, XBaseFinal.XS);
        for (XBaseFinal.Cfg c : cfgs) {
            long tf = System.nanoTime();
            XBaseEval x = XBasePart4.vectorsB(fr, r.k2(), XBaseFinal.param(c, u.tpl.params), c.f(), threads);
            sec.merge("признаки", (System.nanoTime() - tf) / 1e9, Double::sum);
            System.out.printf(Locale.ROOT, "esrc %s: признаки %s (%.0f с)%n", base, c.id(), (System.nanoTime() - tf) / 1e9);
            for (XBaseVec.Blk b : bs) {
                List<Task> ts = tasks(b, smoke);
                if (smoke && c == XBaseFinal.XS) ts = ts.stream().filter(Task::mirror).toList();
                List<XBaseVec.Model> fins = XBaseFinal.models(r, b, c, false);
                List<List<String>> poolCls = poolOf(b);
                // Вложения W – контрольная точка
                List<Callable<Void>> jobs = new ArrayList<>();
                for (Task t : ts) {
                    Path f = embDir.resolve(t.file(b, c));
                    if (raw || Files.exists(f)) continue;
                    jobs.add(() -> {
                        writeEmb(f, embW(x, t, b.g().cohort, poolCls));
                        System.out.printf(Locale.ROOT, "esrc %s: W %s готово%n", base, f.getFileName());
                        return null;
                    });
                }
                long tw = System.nanoTime();
                java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(fitThreads);
                try {
                    for (java.util.concurrent.Future<Void> fu : ex.invokeAll(jobs)) fu.get();
                } finally {
                    ex.shutdown();
                }
                sec.merge("W", (System.nanoTime() - tw) / 1e9, Double::sum);
                Basis bp = raw ? null : pca(x, poolCls);
                double[][] rawAtoms = raw ? atoms(poolCls, x.vec.get(V.L1)::get) : null;
                vol.append(raw ? String.format(Locale.ROOT, "%s %s: классов пула %d, снимков %d, исходное пространство%n", b.name(), c.id(), poolCls.size(), poolCls.stream().mapToInt(List::size).sum())
                        : String.format(Locale.ROOT, "%s %s: классов пула %d, снимков %d, ранг PCA %d, взято осей %d%n", b.name(), c.id(), poolCls.size(),
                        poolCls.stream().mapToInt(List::size).sum(), bp.rank(), bp.rows().length));
                Map<String, double[]> pc = new java.util.concurrent.ConcurrentHashMap<>();
                for (String space : raw ? List.of("RAW") : List.of("W", "PCA")) {
                    Path sd = (raw ? root.resolve("raw") : sci ? root.resolve("sci") : root).resolve(space);
                    Files.createDirectories(sd.resolve("vec"));
                    XBaseFinal.Run rs = new XBaseFinal.Run(u, r.k2(), Map.of(), sd, false, 0);
                    Path vf = XBaseFinal.vfile(rs, b, c);
                    if (!smoke && Files.exists(vf)) {
                        if (!XBaseVec.readHead(vf).contains(code)) throw new IllegalStateException("esrc: " + vf + " – другой код решателя; удалите");
                        Path sf = sd.resolve("vec").resolve(b.name() + "_" + c.id() + "_stat.txt");
                        if (Files.exists(sf)) stats.append(Files.readString(sf));
                        continue;
                    }
                    long ts0 = System.nanoTime();
                    List<XBaseVec.Model> ms = new ArrayList<>();
                    StringBuilder sb = new StringBuilder();
                    for (Task t : ts) {
                        XBaseVec.Model fin = XBaseVec.find(fins, t.name(), t.mirror());
                        Emb e;
                        if (raw) {
                            e = embRaw(x, t, rawAtoms);
                        } else if (space.equals("W")) {
                            e = readEmb(embDir.resolve(t.file(b, c)));
                            check(e, fin, t.file(b, c));
                        } else {
                            e = embP(x, t, bp, poolCls, pc);
                        }
                        Stat st = new Stat();
                        XBaseVec.Model sm = solve(e, t, fin, pool, st);
                        if (sci && !raw) {
                            // те же решения: невязки = файл прогона 72d1799
                            XBaseFinal.Run ro = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve(space), false, 0);
                            XBaseVec.Model om = XBaseVec.find(XBaseFinal.models(ro, b, c, false), t.name(), t.mirror());
                            for (int i = 0; i < om.rows().size(); i++) {
                                double[] a = om.rows().get(i).dg();
                                double[] z = sm.rows().get(i).dg();
                                for (int g = 0; g < a.length; g++) if (!Double.isInfinite(a[g])) worstDg[0] = Math.max(worstDg[0], Math.abs(a[g] - z[g]));
                            }
                        }
                        ms.add(sm);
                        if (t.name().equals("main")) sb.append(statLine(space + ", " + b.name() + ", " + c.id() + ", " + (t.mirror() ? "X+mirror" : "X"), st));
                    }
                    sec.merge("решатель " + space, (System.nanoTime() - ts0) / 1e9, Double::sum);
                    stats.append(sb);
                    System.out.print(sb);
                    if (!smoke) {
                        java.io.Writer w = XBaseVec.writer(vf, "# Невязки ESRC (XBaseEsrc " + base + "): " + space + ", " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; решатель " + code + "\n");
                        XBaseVec.write(w, b.name() + " " + space + " " + c.name(), ms);
                        XBaseVec.finish(w, vf);
                        Files.writeString(sd.resolve("vec").resolve(b.name() + "_" + c.id() + "_stat.txt"), sb.toString());
                    }
                }
            }
            x.vec.clear();
            x.mir.clear();
        }
        pool.shutdown();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        StringBuilder t = new StringBuilder("Этап «к цели», проба 1 – ESRC – " + base + (smoke ? " – пробный режим" : "") + " (XBaseEval " + base + "-esrc; решение (39), Жорик №13)\nКод: коммит " + FarMethods.commit()
                + "; ε = " + EPS + "; бутстреп по людям B = " + u.tpl.b + ", seed " + (u.tpl.seed + FeiBootstrap.SEED_OFFSET) + "; людей " + u.persons + "\n");
        t.append("\n=== Объём словаря и PCA ===\n").append(vol);
        t.append("\n=== Сведения о решении, основная модель ===\n").append(stats);
        if (raw) {
            if (!smoke) t.append(reportRaw(base, r, root, bs, groups, u));
        } else if (sci) {
            if (worstDg[0] > 1e-12) throw new IllegalStateException("esrc: невязки ≠ прогон 72d1799: " + worstDg[0]);
            t.append(String.format(Locale.ROOT, "%nСверка: те же решения – наибольшее |Δr_g| с файлами прогона 72d1799 = %.2e%n", worstDg[0]));
            t.append(reportSci(base, r, root, bs, groups, u));
        } else if (!smoke) {
            t.append(report(base, r, root, bs, groups, u));
        }
        t.append("\n=== Время ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f с%n", k, v)));
        t.append(String.format(Locale.ROOT, "%nВсего: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(root.resolve("xbase_" + base + "_esrc" + (smoke ? "_smoke" : "") + (raw ? "_raw" : sci ? "_sci" : "") + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /**
     * Диагностика без выбора (gt-esrc-check, GT межс., W; вложения – из контрольной точки): 1) первый шаг пути без словаря = лучший по косинусу; 2) условия
     * оптимальности и невязка решателя на настоящих данных; 3) энергия пробы: ‖Dβ‖², ‖Aα‖², зазор r₁ / r₂; верный лучший кандидат; 4) строка «без словаря» (A – тот же решатель, D пуст).
     */
    static void check(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("esrc: FACES_FRAMES=b");
        XBaseFinal.Run r = XBaseFinal.setup("gt", false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        XBaseVec.Blk b = u.blocks.stream().filter(x -> x.group().equals("межс.")).findFirst().orElseThrow();
        Path root = r.dir().resolveSibling("esrc");
        Path nod = root.resolve("check_nodict");
        Files.createDirectories(nod.resolve("vec"));
        XBaseFinal.Run rn = new XBaseFinal.Run(u, r.k2(), Map.of(), nod, false, 0);
        ForkJoinPool pool = new ForkJoinPool(u.tpl.threads);
        StringBuilder t = new StringBuilder("Этап «к цели», проба 1 – ESRC – диагностика, НЕ ДЛЯ ВЫБОРА (XBaseEval gt-esrc-check; GT межс., W)\nКод: коммит " + FarMethods.commit() + "; ε = " + EPS + "\n\n");
        t.append("признак, модель | первый шаг без D = лучший по косинусу | макс. |невязка − ε| | макс. нарушение условий оптимальности | ‖Dβ‖², свои / контр. | ‖Aα‖², свои / контр. | 1 − r₁/r₂, медиана, свои / контр. | верный лучший: (32), ESRC, без D\n");
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            List<XBaseVec.Model> fins = XBaseFinal.models(r, b, c, false);
            List<XBaseVec.Model> ms = new ArrayList<>();
            for (Task task : tasks(b, false)) {
                XBaseVec.Model fin = XBaseVec.find(fins, task.name(), task.mirror());
                Emb e = readEmb(root.resolve("emb").resolve(task.file(b, c)));
                check(e, fin, task.file(b, c));
                XBaseVec.Model nm = solve(new Emb(e.centers(), new double[0][], e.keys(), e.probes()), task, fin, pool, null);
                ms.add(nm);
                if (!task.name().equals("main")) continue;
                int ng = e.centers().length;
                double[][] a = new double[ng][];
                for (int g = 0; g < ng; g++) a[g] = unit(e.centers()[g]);
                double[][] ga = new double[ng][ng];
                for (int i = 0; i < ng; i++) for (int j = 0; j < ng; j++) ga[i][j] = dot(a[i], a[j]);
                List<double[]> cl = new ArrayList<>(Arrays.asList(a));
                for (double[] at : e.atoms()) {
                    double[] un = unit(at);
                    if (un != null) cl.add(un);
                }
                double[][] phi = cl.toArray(new double[0][]);
                int p = phi.length;
                double[][] gram = new double[p][p];
                pool.submit(() -> IntStream.range(0, p).parallel().forEach(i -> {
                    for (int j = 0; j < p; j++) gram[i][j] = dot(phi[i], phi[j]);
                })).get();
                int same = 0;
                double worstR = 0;
                double worstK = 0;
                double[] eD = new double[2];
                double[] eA = new double[2];
                int[] cnt = new int[2];
                List<List<Double>> gaps = List.of(new ArrayList<>(), new ArrayList<>());
                int own = 0;
                int okB = 0;
                int okE = 0;
                int okN = 0;
                int nProbes = e.probes().length;
                for (int i = 0; i < nProbes; i++) {
                    double[] y = unit(e.probes()[i]);
                    double[] c0 = new double[ng];
                    for (int g = 0; g < ng; g++) c0[g] = dot(a[g], y);
                    EsrcSolver.Result s1 = EsrcSolver.solve(ga, c0, 1.0, 0, 1);
                    int b1 = -1;
                    for (int g = 0; g < ng; g++) if (s1.x()[g] != 0) b1 = g;
                    XBaseVec.Row fr = fin.rows().get(i);
                    if (b1 == argmin(fr.dg())) same++;
                    double[] cc = new double[p];
                    for (int j = 0; j < p; j++) cc[j] = dot(phi[j], y);
                    EsrcSolver.Result s = EsrcSolver.solve(gram, cc, 1.0, EPS, 20 * Math.min(p, y.length));
                    double[] res = y.clone();
                    double[] da = new double[y.length];
                    double[] dd = new double[y.length];
                    for (int j = 0; j < p; j++) {
                        if (s.x()[j] == 0) continue;
                        double[] target = j < ng ? da : dd;
                        for (int q = 0; q < y.length; q++) {
                            res[q] -= s.x()[j] * phi[j][q];
                            target[q] += s.x()[j] * phi[j][q];
                        }
                    }
                    worstR = Math.max(worstR, Math.abs(Math.sqrt(dot(res, res)) - EPS));
                    double lam = 0;
                    for (int j = 0; j < p; j++) lam = Math.max(lam, Math.abs(dot(phi[j], res)));
                    for (int j = 0; j < p; j++) if (s.x()[j] != 0) worstK = Math.max(worstK, Math.abs(Math.abs(dot(phi[j], res)) - lam));
                    if (!fr.role().equals(FeiScores.OWN) && !fr.role().equals(FeiScores.CTRL)) continue;
                    int k = fr.role().equals(FeiScores.OWN) ? 0 : 1;
                    cnt[k]++;
                    eD[k] += dot(dd, dd);
                    eA[k] += dot(da, da);
                    // невязки ESRC – прежней формулой
                    double[] bs = y.clone();
                    for (int q = 0; q < y.length; q++) bs[q] -= dd[q];
                    double[] rg = new double[ng];
                    for (int g = 0; g < ng; g++) {
                        double sum = 0;
                        for (int q = 0; q < y.length; q++) {
                            double v = bs[q] - s.x()[g] * a[g][q];
                            sum += v * v;
                        }
                        rg[g] = Math.sqrt(sum);
                    }
                    double[] so = rg.clone();
                    Arrays.sort(so);
                    gaps.get(k).add(1 - so[0] / so[1]);
                    if (k == 0) {
                        own++;
                        if (argmin(fr.dg()) == fr.trueIdx()) okB++;
                        if (argmin(rg) == fr.trueIdx()) okE++;
                        if (argmin(nm.rows().get(i).dg()) == fr.trueIdx()) okN++;
                    }
                }
                t.append(String.format(Locale.ROOT, "%s, %s | %d/%d | %.2e | %.2e | %.3f / %.3f | %.3f / %.3f | %.4f / %.4f | %d, %d, %d из %d%n", c.id(), task.mirror() ? "X+mirror" : "X", same, nProbes, worstR, worstK,
                        eD[0] / cnt[0], eD[1] / cnt[1], eA[0] / cnt[0], eA[1] / cnt[1], median(gaps.get(0)), median(gaps.get(1)), okB, okE, okN, own));
            }
            java.io.Writer w = XBaseVec.writer(XBaseFinal.vfile(rn, b, c), "# Без словаря (XBaseEsrc gt-esrc-check), " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "\n");
            XBaseVec.write(w, b.name() + " без D " + c.name(), ms);
            XBaseVec.finish(w, XBaseFinal.vfile(rn, b, c));
        }
        pool.shutdown();
        int[][] wt = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        XBaseFinalReport.M mB = XBaseFinalReport.measure(List.of(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0))), u.persons, wt);
        XBaseFinalReport.M mN = XBaseFinalReport.measure(List.of(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(rn, b, XBaseFinal.TT1, b.g().rOf, b.g().r0))), u.persons, wt);
        t.append("\nстрока | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(level("(32) слияние", mB)).append(level("A без словаря, W, слияние", mN));
        t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(row("без словаря − (32)", mN, mB));
        FarMethods.write(root.resolve("xbase_gt_esrc_check.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    private static int argmin(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) if (v[i] < v[b]) b = i;
        return b;
    }

    private static double median(List<Double> l) {
        double[] v = l.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return v.length == 0 ? Double.NaN : v[v.length / 2];
    }

    static String row(String lab, XBaseFinalReport.M a, XBaseFinalReport.M z) {
        StringBuilder b = new StringBuilder(lab);
        double[][] ds = {FeiNorm.diff(a.paucRep, z.paucRep), FeiNorm.diff(a.sep.eerRep, z.sep.eerRep), FeiNorm.diff(a.sep.errRep[2], z.sep.errRep[2]), FeiNorm.diff(a.sep.errRep[0], z.sep.errRep[0])};
        double[] ps = {a.pauc - z.pauc, a.sep.eer[3] - z.sep.eer[3], a.sep.err[2] - z.sep.err[2], a.sep.err[0] - z.sep.err[0]};
        for (int k = 0; k < ds.length; k++) {
            double[] ok = Arrays.stream(ds[k]).filter(v -> !Double.isNaN(v)).toArray();
            double[] s = FeiBootstrap.sorted(ok);
            double l95 = FeiBootstrap.quantile(s, 0.025);
            double h95 = FeiBootstrap.quantile(s, 0.975);
            double l97 = FeiBootstrap.quantile(s, 0.0125);
            double h97 = FeiBootstrap.quantile(s, 0.9875);
            b.append(String.format(Locale.ROOT, " | %+.2f [%+.2f; %+.2f] {%+.2f; %+.2f}%s", ps[k], l95, h95, l97, h97, l97 > 0 || h97 < 0 ? " **" : l95 > 0 || h95 < 0 ? " *" : ""));
        }
        return b.append('\n').toString();
    }

    static String level(String lab, XBaseFinalReport.M m) {
        return lab + String.format(Locale.ROOT, " | %s | %s | %s | %s%n", XBaseSepLtp.cell(m.pauc, m.paucRep), XBaseSepLtp.cell(m.sep.eer[3], m.sep.eerRep), XBaseSepLtp.cell(m.sep.err[2], m.sep.errRep[2]),
                XBaseSepLtp.cell(m.sep.err[0], m.sep.errRep[0]));
    }

    /** Поле строки: 0 – −SCI, 1 – −‖Aα‖². */
    private static double field(XBaseVec.Row q, int f) {
        return -q.dci()[f];
    }

    /** μ, σ поля у «чужих» (Т2) всех моделей признака – роли cv q, od q (как XBaseVec.fuseStats). */
    private static double[] fieldStats(List<XBaseVec.Model> ms, int f) {
        double s = 0;
        double ss = 0;
        long n = 0;
        for (XBaseVec.Model m : ms) {
            for (XBaseVec.Row q : m.rows()) {
                if (q.trueIdx() >= 0 || !(q.role().startsWith(FeiMethods.CV_ROLE) || q.role().startsWith("od"))) continue;
                double v = field(q, f);
                s += v;
                ss += v * v;
                n++;
            }
        }
        if (n < 2) throw new IllegalStateException("esrc: мало «чужих» (Т2)");
        double m = s / n;
        double v = Math.sqrt(Math.max(0, (ss - n * m * m) / (n - 1)));
        if (v == 0) throw new IllegalStateException("esrc: разброс оценки у «чужих» (Т2) равен нулю");
        return new double[] {m, v};
    }

    /**
     * Конфигурация с оценкой вместо d₁/d₂: опознание (ownOk) – прежнее, по невязкам строки lines. kind: 0 – слияние 0,75 · tt + 0,25 · xs после приведения по (Т2); 1 – L1-tt-xs-F
     * (модель по позе); 2 – L1-xs-F (X+mirror).
     */
    private static FeiDecompose.Cfg scored(XBaseVec.Blk b, XBaseFinal.Run rs, int f, int kind) throws IOException {
        List<XBaseVec.Model> mt = XBaseFinal.models(rs, b, XBaseFinal.TT1, false);
        List<XBaseVec.Model> mx = XBaseFinal.models(rs, b, XBaseFinal.XS, false);
        double[] st = fieldStats(mt, f);
        double[] sx = fieldStats(mx, f);
        XBaseVec.Model tX = XBaseVec.find(mt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(mt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(mx, "main", true);
        Map<String, Double> score = new HashMap<>();
        for (int i = 0; i < tX.rows().size(); i++) {
            XBaseVec.Row q0 = tX.rows().get(i);
            XBaseVec.Row rt = b.g().rOf.get(q0.key()) <= b.g().r0 ? q0 : tM.rows().get(i);
            XBaseVec.Row rx = xM.rows().get(i);
            if (!rx.key().equals(q0.key())) throw new IllegalStateException("esrc: пробы кандидатов разошлись");
            double v = kind == 1 ? field(rt, f) : kind == 2 ? field(rx, f)
                    : XBaseFeat2.W_FUSE * (field(rt, f) - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (field(rx, f) - sx[0]) / sx[1];
            score.put(q0.key(), v);
        }
        FeiDecompose.Cfg c = kind == 0 ? XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(rs, b, XBaseFinal.TT1, b.g().rOf, b.g().r0))
                : kind == 1 ? XBaseGoalPrep.cfg(b, b.g(), XBaseVec.normLines(mt, null, new int[2]))
                : XBaseGoalPrep.cfg(b, XBaseAbl3.mirrorGt(b.g(), true), XBaseVec.normLines(mx, null, new int[2]));
        int io = 0;
        int ic = 0;
        for (Probe p : XBaseEval.probes(b.g(), b.pro())) {
            if (p.role().equals(FeiScores.OWN)) c.ownR[io++] = score.get(p.key());
            else c.ctrlR[ic++] = score.get(p.key());
        }
        if (io != c.ownR.length || ic != c.ctrlR.length) throw new IllegalStateException("esrc: число проб разошлось");
        return c;
    }

    private static String quart(List<Double> l) {
        double[] v = l.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return String.format(Locale.ROOT, "%.3f / %.3f / %.3f", v[v.length / 4], v[v.length / 2], v[3 * v.length / 4]);
    }

    private static String reportSci(String base, XBaseFinal.Run r, Path root, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u) throws IOException {
        Map<String, XBaseFinal.Run> runs = new LinkedHashMap<>();
        Path sr = root.resolve("sci");
        runs.put("ESRC-SCI (W)", new XBaseFinal.Run(u, r.k2(), Map.of(), sr.resolve("W"), false, 0));
        runs.put(base.equals("feret") ? "ESRC-SCI (PCA, вся оболочка половины пула)" : "ESRC-SCI (PCA 500)", new XBaseFinal.Run(u, r.k2(), Map.of(), sr.resolve("PCA"), false, 0));
        return scoreTables(r, bs, groups, u, runs, false);
    }

    /** Жорик №15: исходное пространство, одна строка (95 %), d₁/d₂ по невязкам – справочно. */
    private static String reportRaw(String base, XBaseFinal.Run r, Path root, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u) throws IOException {
        Map<String, XBaseFinal.Run> runs = new LinkedHashMap<>();
        runs.put("ESRC-SCI (исходное пространство)", new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("raw").resolve("RAW"), false, 0));
        return "\nОдна строка: поправки на множественность нет, основной интервал – 95 % (квадратные скобки); столбцы A и y без вычитания общего среднего.\n" + scoreTables(r, bs, groups, u, runs, true);
    }

    private static String scoreTables(XBaseFinal.Run r, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u, Map<String, XBaseFinal.Run> runs, boolean withRatio) throws IOException {
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        StringBuilder t = new StringBuilder("\n=== ESRC-SCI − (32): оценка −SCI, опознание – по наименьшей невязке; запись: разность [95 %] {97,5 % – Бонферрони, k = 2}; «*» – 95 % без нуля, «**» – 97,5 % без нуля; у pAUC больше – лучше, у прочих меньше – лучше ===\n");
        StringBuilder diag = new StringBuilder("\n=== Диагностика, не для выбора: квартили по пробам (25 / 50 / 75 %); модель L1-tt-xs-F – по позе, L1-xs-F – X+mirror ===\nгруппа, пространство, признак, величина | свои | контрольные\n");
        StringBuilder eer = new StringBuilder("\n=== Диагностика, не для выбора: одна ‖Aα‖² как оценка (опознание по невязке) ===\nгруппа, строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n");
        for (String grp : groups) {
            List<XBaseVec.Blk> gb = bs.stream().filter(b -> b.group().equals(grp)).toList();
            List<FeiDecompose.Cfg> csB = new ArrayList<>();
            for (XBaseVec.Blk b : gb) csB.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
            XBaseFinalReport.M mB = XBaseFinalReport.measure(csB, u.persons, w);
            t.append("\n").append(grp).append(": строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(level("(32) слияние", mB));
            StringBuilder dif = new StringBuilder("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n");
            for (Map.Entry<String, XBaseFinal.Run> e : runs.entrySet()) {
                String[] names = {"слияние", "L1-tt-xs-F", "L1-xs-F"};
                for (int kind = 0; kind < 3; kind++) {
                    List<FeiDecompose.Cfg> cs = new ArrayList<>();
                    for (XBaseVec.Blk b : gb) cs.add(scored(b, e.getValue(), 0, kind));
                    XBaseFinalReport.M m = XBaseFinalReport.measure(cs, u.persons, w);
                    t.append(level(e.getKey() + " " + names[kind], m));
                    if (kind == 0) dif.append(row(e.getKey() + " − (32), слияние", m, mB));
                }
                if (withRatio) {
                    List<FeiDecompose.Cfg> cr = new ArrayList<>();
                    for (XBaseVec.Blk b : gb) cr.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(e.getValue(), b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                    XBaseFinalReport.M mr = XBaseFinalReport.measure(cr, u.persons, w);
                    t.append(level("справочно: d₁/d₂ по невязкам, слияние", mr));
                    dif.append(row("справочно: d₁/d₂ по невязкам − (32), слияние", mr, mB));
                }
                List<FeiDecompose.Cfg> ca = new ArrayList<>();
                for (XBaseVec.Blk b : gb) ca.add(scored(b, e.getValue(), 1, 0));
                eer.append(level(grp + ", " + e.getKey() + ", слияние по ‖Aα‖²", XBaseFinalReport.measure(ca, u.persons, w)));
                String[] fn = {"SCI", "‖Aα‖²"};
                for (int f = 0; f < 2; f++) {
                    for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
                        List<Double> own = new ArrayList<>();
                        List<Double> ctl = new ArrayList<>();
                        for (XBaseVec.Blk b : gb) {
                            List<XBaseVec.Model> ms = XBaseFinal.models(e.getValue(), b, c, false);
                            XBaseVec.Model mX = XBaseVec.find(ms, "main", false);
                            XBaseVec.Model mM = XBaseVec.find(ms, "main", true);
                            for (int i = 0; i < mM.rows().size(); i++) {
                                XBaseVec.Row q = c == XBaseFinal.TT1 && b.g().rOf.get(mX.rows().get(i).key()) <= b.g().r0 ? mX.rows().get(i) : mM.rows().get(i);
                                if (q.role().equals(FeiScores.OWN)) own.add(q.dci()[f]);
                                else if (q.role().equals(FeiScores.CTRL)) ctl.add(q.dci()[f]);
                            }
                        }
                        diag.append(String.format(Locale.ROOT, "%s, %s, %s, %s | %s | %s%n", grp, e.getKey(), c.id(), fn[f], quart(own), quart(ctl)));
                    }
                }
            }
            t.append(dif);
        }
        return t.append(diag).append(eer).toString();
    }

    private static String report(String base, XBaseFinal.Run r, Path root, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u) throws IOException {
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        StringBuilder t = new StringBuilder("\n=== ESRC − (32): запись: разность [95 %] {97,5 % – Бонферрони, k = 2}; «*» – 95 % без нуля, «**» – 97,5 % без нуля; у pAUC больше – лучше, у прочих меньше – лучше ===\n");
        Map<String, XBaseFinal.Run> runs = new LinkedHashMap<>();
        runs.put("ESRC (W)", new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("W"), false, 0));
        runs.put(base.equals("feret") ? "ESRC (PCA, вся оболочка половины пула)" : "ESRC (PCA 500)", new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("PCA"), false, 0));
        for (String grp : groups) {
            List<XBaseVec.Blk> gb = bs.stream().filter(b -> b.group().equals(grp)).toList();
            List<FeiDecompose.Cfg> csB = new ArrayList<>();
            List<FeiDecompose.Cfg> csT = new ArrayList<>();
            for (XBaseVec.Blk b : gb) {
                csB.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                csT.add(XBaseGoalPrep.cfg(b, b.g(), XBaseVec.normLines(XBaseFinal.models(r, b, XBaseFinal.TT1, false), null, new int[2])));
            }
            XBaseFinalReport.M mB = XBaseFinalReport.measure(csB, u.persons, w);
            XBaseFinalReport.M mT = XBaseFinalReport.measure(csT, u.persons, w);
            t.append("\n").append(grp).append(": строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n");
            t.append(level("(32) слияние", mB)).append(level("(32) L1-tt-xs-F", mT));
            StringBuilder dif = new StringBuilder("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n");
            for (Map.Entry<String, XBaseFinal.Run> e : runs.entrySet()) {
                List<FeiDecompose.Cfg> cF = new ArrayList<>();
                List<FeiDecompose.Cfg> cT = new ArrayList<>();
                for (XBaseVec.Blk b : gb) {
                    cF.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(e.getValue(), b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                    cT.add(XBaseGoalPrep.cfg(b, b.g(), XBaseVec.normLines(XBaseFinal.models(e.getValue(), b, XBaseFinal.TT1, false), null, new int[2])));
                }
                XBaseFinalReport.M f = XBaseFinalReport.measure(cF, u.persons, w);
                XBaseFinalReport.M tt = XBaseFinalReport.measure(cT, u.persons, w);
                t.append(level(e.getKey() + " слияние", f)).append(level(e.getKey() + " L1-tt-xs-F", tt));
                dif.append(row(e.getKey() + " − (32), слияние", f, mB)).append(row(e.getKey() + " − (32), L1-tt-xs-F", tt, mT));
            }
            t.append(dif);
        }
        return t.toString();
    }
}
