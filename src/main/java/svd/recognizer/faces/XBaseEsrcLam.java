package svd.recognizer.faces;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * ESRC, последняя попытка (Жорик №16 от 04.11, PLAN.md, абзац «ESRC, последняя попытка»): ½‖y − Bx‖² + λ‖x‖₁, B = [A, D], столбцы и y единичной длины, λ = q · ‖Bᵀy‖∞,
 * q = 0,3; 0,1; 0,03; 0,01 – точки одного пути. Вычитание среднего пула (обучающая часть): без / с (D от него не зависит); пространства – исходное и W. 16 конфигураций; отказ – SCI по α,
 * опознание – по наименьшей невязке, слияние как в (32). pool-esrc-lam – выбор по пулу на кадрах «а» (база – векторы v8 pool-feat2, сверка вложений), файл выбора; gt-esrc-lam,
 * feret-esrc-lam (кадры Б) – только выбранная конфигурация против (32). Выбор: наименьшее среднее критерия по трём наборам (правило объявлено до прогона, TASKS.md).
 * Пробный режим FACES_ESRC_SMOKE=1 (pool): два блока, замер времени и оценка полного прогона.
 *
 * @author ssv
 */
final class XBaseEsrcLam {

    private XBaseEsrcLam() {
    }

    static final double[] QS = {0.3, 0.1, 0.03, 0.01};
    static final String[] Q_ID = {"q30", "q10", "q03", "q01"};
    static final String SELECT = "xbase_pool_esrc_lam_select.tsv";

    /** Конфигурация: пространство, вычитание среднего, номер q. */
    record Conf(String space, boolean mean, int q) {
        String id() {
            return space + (mean ? "_m1_" : "_m0_") + Q_ID[q];
        }

        String label() {
            return (space.equals("RAW") ? "исходное" : "W") + ", " + (mean ? "со средним пула" : "без вычитания") + ", q = " + QS[q];
        }
    }

    static List<Conf> all() {
        List<Conf> l = new ArrayList<>();
        for (String s : List.of("RAW", "W")) for (boolean m : new boolean[] {false, true}) for (int q = 0; q < QS.length; q++) l.add(new Conf(s, m, q));
        return l;
    }

    /** Накопитель сведений конфигурации. */
    static final class St {
        double atoms;
        double cols;
        long n;
        long ties;
        long capped;
        double nanos;
        long paths;
    }

    private static void writeObj(Path f, Object e) throws IOException {
        Path tmp = f.resolveSibling(f.getFileName() + ".tmp");
        try (ObjectOutputStream o = new ObjectOutputStream(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16))) {
            o.writeObject(e);
        }
        Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    private static Object readObj(Path f) throws IOException {
        try (ObjectInputStream o = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(f), 1 << 16))) {
            return o.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException(e);
        }
    }

    /** Среднее признаков людей обучающей части пула. */
    private static double[] poolMean(XBaseEval x, List<List<String>> pool) {
        Map<String, double[]> v = x.vec.get(V.L1);
        double[] m = null;
        int n = 0;
        for (List<String> cls : pool) {
            for (String k : cls) {
                double[] a = v.get(k);
                if (m == null) m = new double[a.length];
                for (int i = 0; i < a.length; i++) m[i] += a[i];
                n++;
            }
        }
        for (int i = 0; i < m.length; i++) m[i] /= n;
        return m;
    }

    /** Решения одного пути для всех проб модели: [q] → модель; для обеих конфигураций среднего вызывается отдельно. */
    private static XBaseVec.Model[] solve(XBaseEsrc.Emb e, double[] mu, XBaseEsrc.Task t, XBaseVec.Model fin, ForkJoinPool pool, St[] st) throws Exception {
        int ng = e.centers().length;
        int d = (mu != null ? mu : e.probes()[0]).length;
        List<double[]> cl = new ArrayList<>();
        int[] colOfG = new int[ng];
        for (int g = 0; g < ng; g++) {
            double[] u = e.centers()[g] == null ? null : XBaseEsrc.unit(sub(e.centers()[g], mu));
            colOfG[g] = u == null ? -1 : cl.size();
            if (u != null) cl.add(u);
        }
        int nA = cl.size();
        for (double[] a : e.atoms()) {
            double[] u = XBaseEsrc.unit(a);
            if (u != null) cl.add(u);
        }
        double[][] phi = cl.toArray(new double[0][]);
        int p = phi.length;
        double[][] gram = new double[p][p];
        pool.submit(() -> IntStream.range(0, p).parallel().forEach(i -> {
            for (int j = i; j < p; j++) gram[i][j] = XBaseEsrc.dot(phi[i], phi[j]);
        })).get();
        for (int i = 0; i < p; i++) for (int j = 0; j < i; j++) gram[i][j] = gram[j][i];
        int maxSteps = 20 * Math.max(1, Math.min(p, d));
        int nr = e.probes().length;
        XBaseVec.Row[][] rows = new XBaseVec.Row[QS.length][nr];
        double[][][] info = new double[QS.length][nr][4];
        long[] nanos = new long[nr];
        boolean[] capped = new boolean[nr];
        pool.submit(() -> IntStream.range(0, nr).parallel().forEach(i -> {
            long t0 = System.nanoTime();
            double[] y = XBaseEsrc.unit(sub(e.probes()[i], mu));
            double[] c = new double[p];
            for (int j = 0; j < p; j++) c[j] = XBaseEsrc.dot(phi[j], y);
            EsrcSolver.Path r = EsrcSolver.path(gram, c, 1.0, QS, maxSteps);
            capped[i] = r.capped();
            nanos[i] = System.nanoTime() - t0;
            XBaseVec.Row o = fin.rows().get(i);
            if (!o.key().equals(e.keys().get(i))) throw new IllegalStateException("esrc-lam: пробы разошлись: " + o.key());
            for (int q = 0; q < QS.length; q++) {
                double[] x = r.xs()[q];
                double[] bs = y.clone();
                double[] aa = new double[d];
                int nzD = 0;
                int nzA = 0;
                double[] alpha = new double[nA];
                for (int j = 0; j < p; j++) {
                    if (x[j] == 0) continue;
                    double[] col = phi[j];
                    if (j < nA) {
                        nzA++;
                        alpha[j] = x[j];
                        for (int s = 0; s < d; s++) aa[s] += x[j] * col[s];
                    } else {
                        nzD++;
                        for (int s = 0; s < d; s++) bs[s] -= x[j] * col[s];
                    }
                }
                double[] dg = new double[ng];
                double base = Math.sqrt(XBaseEsrc.dot(bs, bs));
                for (int g = 0; g < ng; g++) {
                    int j = colOfG[g];
                    if (j < 0) {
                        dg[g] = Double.POSITIVE_INFINITY;
                    } else if (x[j] == 0) {
                        dg[g] = base;
                    } else {
                        double sum = 0;
                        double[] col = phi[j];
                        for (int s = 0; s < d; s++) {
                            double v = bs[s] - x[j] * col[s];
                            sum += v * v;
                        }
                        dg[g] = Math.sqrt(sum);
                    }
                }
                double[] so = dg.clone();
                Arrays.sort(so);
                rows[q][i] = new XBaseVec.Row(o.role(), o.person(), o.image(), o.key(), o.trueIdx(), dg, new double[] {XBaseEsrc.sci(alpha, ng), XBaseEsrc.dot(aa, aa)}, new double[0]);
                info[q][i] = new double[] {nzD, nzA, so.length > 1 && so[1] - so[0] <= XBaseEsrc.TIE ? 1 : 0, 0};
            }
        })).get();
        XBaseVec.Model[] out = new XBaseVec.Model[QS.length];
        for (int q = 0; q < QS.length; q++) {
            out[q] = new XBaseVec.Model(t.name(), t.method(), fin.s(), t.own(), Arrays.asList(rows[q]), new double[ng][]);
            if (st == null) continue;
            for (int i = 0; i < nr; i++) {
                if (fin.rows().get(i).trueIdx() < 0 && !fin.rows().get(i).role().equals(FeiScores.CTRL)) continue;
                st[q].atoms += info[q][i][0];
                st[q].cols += info[q][i][1];
                st[q].ties += (long) info[q][i][2];
                st[q].n++;
                if (capped[i]) st[q].capped++;
                st[q].nanos += nanos[i];
                st[q].paths++;
            }
        }
        return out;
    }

    private static double[] sub(double[] a, double[] mu) {
        if (mu == null) return a;
        double[] r = new double[a.length];
        for (int i = 0; i < r.length; i++) r[i] = a[i] - mu[i];
        return r;
    }

    private static double[] embMean(LdaModel lm, double[] m) {
        return lm.embed(m);
    }

    /** Вложения W и среднее пула в W: контрольная точка; сверка с векторами базы – при каждой загрузке. */
    private static void fits(XBaseEval x, List<XBaseEsrc.Task> ts, XBaseVec.Blk b, XBaseFinal.Cfg c, Path embDir, int threads) throws Exception {
        List<List<String>> poolCls = XBaseEsrc.poolOf(b);
        double[] mRaw = poolMean(x, poolCls);
        List<Callable<Void>> jobs = new ArrayList<>();
        for (XBaseEsrc.Task t : ts) {
            Path f = embDir.resolve(t.file(b, c));
            Path fm = embDir.resolve(t.file(b, c) + ".mu");
            if (Files.exists(f) && Files.exists(fm)) continue;
            jobs.add(() -> {
                LdaModel lm = XBaseRule.fit(x, t.mirror(), t.gallery(), b.g().cohort, null).lm;
                Map<String, double[]> v = x.vec.get(V.L1);
                double[][] ys = new double[t.probes().size()][];
                List<String> keys = new ArrayList<>();
                for (int i = 0; i < ys.length; i++) {
                    keys.add(t.probes().get(i).key());
                    ys[i] = lm.embed(v.get(keys.get(i)));
                }
                writeObj(f, new XBaseEsrc.Emb(lm.centers(), XBaseEsrc.atoms(poolCls, k -> lm.embed(v.get(k))), keys, ys));
                writeObj(fm, embMean(lm, mRaw));
                return null;
            });
        }
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            for (java.util.concurrent.Future<Void> fu : ex.invokeAll(jobs)) fu.get();
        } finally {
            ex.shutdown();
        }
    }

    /** Строка сведений (файл рядом с векторами – для продолжения после обрыва). */
    private static String statText(St s) {
        return String.format(Locale.ROOT, "%.0f\t%.0f\t%d\t%d\t%d\t%.0f\t%d%n", s.atoms, s.cols, s.n, s.ties, s.capped, s.nanos, s.paths);
    }

    private static void addStat(St to, String text) {
        for (String l : text.split("\n")) {
            if (l.isBlank()) continue;
            String[] c = l.trim().split("\t");
            to.atoms += Double.parseDouble(c[0]);
            to.cols += Double.parseDouble(c[1]);
            to.n += Long.parseLong(c[2]);
            to.ties += Long.parseLong(c[3]);
            to.capped += Long.parseLong(c[4]);
            to.nanos += Double.parseDouble(c[5]);
            to.paths += Long.parseLong(c[6]);
        }
    }

    private static String statCells(St s) {
        double n = Math.max(1, s.n);
        return String.format(Locale.ROOT, "%.1f | %.1f | %.2f %% | %.2f %% | %.0f", s.atoms / n, s.cols / n, 100.0 * s.ties / n, 100.0 * s.capped / n, s.nanos / Math.max(1, s.paths) / 1e6);
    }

    private static List<XBaseVec.Model> baseModels(XBaseVec.Setup u, XBaseFinal.Run r, XBaseVec.Blk b, XBaseFinal.Cfg c) throws IOException {
        return r != null ? XBaseFinal.models(r, b, c, false) : XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, c.f(), 8);
    }

    /** Считает выбранные конфигурации блока, пишет векторы и сведения. */
    private static void compute(XBaseVec.Setup u, XBaseFinal.Run r, XBaseEval x, XBaseVec.Blk b, XBaseFinal.Cfg c, List<XBaseEsrc.Task> ts, List<Conf> confs, Path root, Path embDir, ForkJoinPool pool, String code, double[] worst)
            throws Exception {
        List<XBaseVec.Model> fins = baseModels(u, r, b, c);
        List<List<String>> poolCls = XBaseEsrc.poolOf(b);
        double[][] rawAtoms = null;
        double[] rawMean = poolMean(x, poolCls);
        for (String space : List.of("RAW", "W")) {
            for (boolean mean : new boolean[] {false, true}) {
                List<Conf> mine = confs.stream().filter(k -> k.space().equals(space) && k.mean() == mean).toList();
                if (mine.isEmpty()) continue;
                boolean done = true;
                for (Conf k : mine) done &= Files.exists(vfile(root, k, b, c));
                if (done) {
                    for (Conf k : mine) if (!XBaseVec.readHead(vfile(root, k, b, c)).contains(code)) throw new IllegalStateException("esrc-lam: другой код решателя: " + vfile(root, k, b, c));
                    continue;
                }
                List<List<XBaseVec.Model>> ms = new ArrayList<>();
                for (int q = 0; q < QS.length; q++) ms.add(new ArrayList<>());
                St[] st = new St[QS.length];
                for (int q = 0; q < QS.length; q++) st[q] = new St();
                for (XBaseEsrc.Task t : ts) {
                    XBaseVec.Model fin = XBaseVec.find(fins, t.name(), t.mirror());
                    XBaseEsrc.Emb e;
                    double[] mu;
                    if (space.equals("W")) {
                        e = (XBaseEsrc.Emb) readObj(embDir.resolve(t.file(b, c)));
                        worst[0] = Math.max(worst[0], XBaseEsrc.checkDiff(e, fin, t.file(b, c)));
                        mu = mean ? (double[]) readObj(embDir.resolve(t.file(b, c) + ".mu")) : null;
                    } else {
                        if (rawAtoms == null) rawAtoms = XBaseEsrc.atoms(poolCls, x.vec.get(V.L1)::get);
                        e = XBaseEsrc.embRaw(x, t, rawAtoms);
                        mu = mean ? rawMean : null;
                    }
                    XBaseVec.Model[] o = solve(e, mu, t, fin, pool, st);
                    for (int q = 0; q < QS.length; q++) ms.get(q).add(o[q]);
                }
                for (Conf k : mine) {
                    Path vf = vfile(root, k, b, c);
                    Files.createDirectories(vf.getParent());
                    java.io.Writer w = XBaseVec.writer(vf, "# ESRC λ (XBaseEsrcLam): " + k.label() + ", " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; решатель " + code + "\n");
                    XBaseVec.write(w, b.name() + " " + k.id() + " " + c.name(), ms.get(k.q()));
                    XBaseVec.finish(w, vf);
                    Files.writeString(vf.resolveSibling(b.name() + "_" + c.id() + "_stat.txt"), statText(st[k.q()]));
                }
            }
        }
    }

    private static Path vfile(Path root, Conf k, XBaseVec.Blk b, XBaseFinal.Cfg c) {
        return root.resolve(k.id()).resolve("vec").resolve(b.name() + "_" + c.id() + ".tsv.gz");
    }

    private static XBaseFinal.Run runOf(XBaseVec.Setup u, Map<String, FrameK2.F> k2, Path root, Conf k) {
        return new XBaseFinal.Run(u, k2, Map.of(), root.resolve(k.id()), false, 0);
    }

    /** μ, σ −SCI у «чужих»: пул – контрольные блока, иначе – роли cv q, od q; σ = 0 (все одинаковы) – 1, число случаев считается. */
    private static double[] stats(List<XBaseVec.Model> ms, boolean pool, int[] flat) {
        double s = 0;
        double ss = 0;
        long n = 0;
        for (XBaseVec.Model m : ms) {
            for (XBaseVec.Row q : m.rows()) {
                boolean imp = q.trueIdx() < 0 && (pool ? q.role().equals(FeiScores.CTRL) : q.role().startsWith(FeiMethods.CV_ROLE) || q.role().startsWith("od"));
                if (!imp) continue;
                s -= q.dci()[0];
                ss += q.dci()[0] * q.dci()[0];
                n++;
            }
        }
        if (n < 2) throw new IllegalStateException("esrc-lam: мало «чужих»");
        double m = s / n;
        double v = Math.sqrt(Math.max(0, (ss - n * m * m) / (n - 1)));
        if (v == 0) {
            flat[0]++;
            v = 1;
        }
        return new double[] {m, v};
    }

    /** Конфигурация оценки блока: score – слияние по SCI, ratio – d₁/d₂ по невязкам (строки слияния). */
    private static FeiDecompose.Cfg cfg(XBaseVec.Blk b, boolean pool, List<XBaseVec.Model> mt, List<XBaseVec.Model> mx, boolean sci, int[] flat) {
        List<Probe> pr = pool ? b.probes() : b.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList();
        String lines = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, pool), XBaseVec.fuseStats(mx, pool), b.g().rOf, b.g().r0, new int[2]);
        FeiDecompose.Cfg c = XBaseEval.parse(b.g(), java.util.EnumSet.of(V.L1), b.gal(), pr, XBaseFeat2.noParts(), lines).cfg.get(V.L1);
        if (!sci) return c;
        double[] st = stats(mt, pool, flat);
        double[] sx = stats(mx, pool, flat);
        XBaseVec.Model tX = XBaseVec.find(mt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(mt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(mx, "main", true);
        Map<String, Double> score = new HashMap<>();
        for (int i = 0; i < tX.rows().size(); i++) {
            XBaseVec.Row q0 = tX.rows().get(i);
            XBaseVec.Row rt = b.g().rOf.get(q0.key()) <= b.g().r0 ? q0 : tM.rows().get(i);
            XBaseVec.Row rx = xM.rows().get(i);
            score.put(q0.key(), XBaseFeat2.W_FUSE * (-rt.dci()[0] - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (-rx.dci()[0] - sx[0]) / sx[1]);
        }
        int io = 0;
        int ic = 0;
        for (Probe p : pr) {
            if (p.role().equals(FeiScores.OWN)) c.ownR[io++] = score.get(p.key());
            else c.ctrlR[ic++] = score.get(p.key());
        }
        return c;
    }

    private static List<XBaseEsrc.Task> tasksFor(XBaseVec.Blk b, boolean pool) {
        if (!pool) return XBaseEsrc.tasks(b, false);
        return XBaseEsrc.tasks(b, true);
    }

    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        boolean pool = base.equals("pool");
        if (pool == FrameB.on()) throw new IllegalStateException("esrc-lam: пул – кадры «а» (как база выборов), GT и FERET – FACES_FRAMES=b");
        boolean smoke = pool && "1".equals(System.getenv("FACES_ESRC_SMOKE"));
        long t0 = System.nanoTime();
        XBaseFinal.Run r = pool ? null : XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = pool ? XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0) : r.u();
        Map<String, FrameK2.F> k2 = pool ? null : r.k2();
        List<String> groups = base.equals("gt") ? List.of("межс.", "внутри") : base.equals("feret") ? List.of("межс.") : u.groups;
        List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> groups.contains(b.group())).toList();
        if (smoke) bs = bs.subList(0, 2);
        Path eqDir = u.su.dir.resolve("reports").resolve("eq");
        Path selFile = eqDir.resolve(SELECT);
        List<Conf> confs;
        if (pool) {
            confs = all();
        } else {
            String id = Files.readAllLines(selFile).stream().filter(l -> l.startsWith("выбор\t")).findFirst().orElseThrow(() -> new IllegalStateException("esrc-lam: нет выбора в " + selFile)).split("\t")[1];
            confs = all().stream().filter(k -> k.id().equals(id)).toList();
            if (confs.size() != 1) throw new IllegalStateException("esrc-lam: неизвестная конфигурация " + id);
        }
        Path root = pool ? eqDir.resolve("esrc_lam") : r.dir().resolveSibling("esrc").resolve("lam");
        Path embDir = root.resolve("emb");
        Files.createDirectories(embDir);
        Set<String> need = new HashSet<>();
        for (XBaseVec.Blk b : bs) {
            b.gal().values().forEach(need::addAll);
            b.g().cohort.forEach(need::addAll);
            for (Probe q : b.probes()) if (!q.role().equals(XBaseFinal.CTRL_X)) need.add(q.key());
            if (b.extra() != null) for (List<Probe> l : b.extra()) for (Probe q : l) need.add(q.key());
        }
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        if (!fr.keySet().containsAll(need)) throw new IllegalStateException("esrc-lam: не все кадры загружены");
        if (pool) k2 = FrameK2.load(fr.keySet(), u.su, u.su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        int threads = u.tpl.threads;
        ForkJoinPool fj = new ForkJoinPool(threads);
        String code = FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBaseEsrcLam.java") + " " + FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/EsrcSolver.java");
        boolean needW = confs.stream().anyMatch(k -> k.space().equals("W"));
        double[] worst = {0};
        Map<String, Double> sec = new LinkedHashMap<>();
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            long tf = System.nanoTime();
            XBaseEval x = XBasePart4.vectorsB(fr, k2, XBaseFinal.param(c, u.tpl.params), c.f(), threads);
            sec.merge("признаки", (System.nanoTime() - tf) / 1e9, Double::sum);
            int done = 0;
            for (XBaseVec.Blk b : bs) {
                List<XBaseEsrc.Task> ts = tasksFor(b, pool);
                if (needW) {
                    long tw = System.nanoTime();
                    fits(x, ts, b, c, embDir, Math.min(8, threads));
                    sec.merge("W", (System.nanoTime() - tw) / 1e9, Double::sum);
                }
                long ts0 = System.nanoTime();
                compute(u, r, x, b, c, ts, confs, root, embDir, fj, code, worst);
                sec.merge("решатель", (System.nanoTime() - ts0) / 1e9, Double::sum);
                System.out.printf(Locale.ROOT, "esrc-lam %s: %s блок %s, %d из %d (%.0f с)%n", base, c.id(), b.name(), ++done, bs.size(), (System.nanoTime() - t0) / 1e9);
            }
            x.vec.clear();
            x.mir.clear();
        }
        fj.shutdown();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : k2.values()) k.frame().release();
        StringBuilder t = new StringBuilder("ESRC, последняя попытка – " + base + (smoke ? " – пробный режим" : "") + " (XBaseEval " + base + "-esrc-lam; Жорик №16)\nКод: коммит " + FarMethods.commit() + "; λ = q · ‖Bᵀy‖∞\n");
        t.append(String.format(Locale.ROOT, "Сверка вложений W с векторами базы: наибольшее расхождение 1 − cos = %.2e%n", worst[0]));
        t.append("\n=== Сведения о решении, все модели блоков, свои и контрольные ===\nконфигурация | атомов D | столбцов A | r₁ = r₂ | путь оборван | мс на путь (один поток)\n");
        Map<String, St> sts = new LinkedHashMap<>();
        for (Conf k : confs) {
            St s = new St();
            for (XBaseVec.Blk b : bs) {
                for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) addStat(s, Files.readString(vfile(root, k, b, c).resolveSibling(b.name() + "_" + c.id() + "_stat.txt")));
            }
            sts.put(k.id(), s);
            t.append(k.label()).append(" | ").append(statCells(s)).append('\n');
        }
        if (pool && !smoke) t.append(selectPool(u, root, bs, confs, selFile));
        if (!pool) t.append(check(base, r, root, bs, groups, u, confs.get(0)));
        t.append("\n=== Время ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f с%n", k, v)));
        double total = (System.nanoTime() - t0) / 1e9;
        t.append(String.format(Locale.ROOT, "%nВсего: %.0f с%n", total));
        if (smoke) {
            double est = (sec.getOrDefault("W", 0.0) + sec.getOrDefault("решатель", 0.0)) * u.blocks.size() / bs.size() + sec.getOrDefault("признаки", 0.0);
            t.append(String.format(Locale.ROOT, "Оценка полного прогона пула (блоков %d): %.1f ч%n", u.blocks.size(), est / 3600));
        }
        FarMethods.write(root.resolve("xbase_" + base + "_esrc_lam" + (smoke ? "_smoke" : "") + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Выбор по пулу: критерий XBasePart4.critW по наборам, бутстреп как pool-noise. */
    private static String selectPool(XBaseVec.Setup u, Path root, List<XBaseVec.Blk> bs, List<Conf> confs, Path selFile) throws Exception {
        List<String> sets = u.groups;
        int n = u.persons;
        int[][] w = FeiBootstrap.weights(n, XBasePart4.NOISE_B, u.tpl.seed + XBasePart4.NOISE_SEED_OFFSET);
        int[] unit = FeiBootstrap.unit(n);
        int[] flat = new int[1];
        Map<String, double[]> pts = new LinkedHashMap<>();
        Map<String, double[][]> reps = new LinkedHashMap<>();
        ForkJoinPool fj = new ForkJoinPool(u.tpl.threads);
        Map<String, FeiDecompose.Cfg[]> rows = new LinkedHashMap<>();
        FeiDecompose.Cfg[] baseC = new FeiDecompose.Cfg[sets.size()];
        for (int s = 0; s < sets.size(); s++) {
            String set = sets.get(s);
            List<FeiDecompose.Cfg> acc = new ArrayList<>();
            for (XBaseVec.Blk b : bs) {
                if (!b.group().equals(set)) continue;
                acc.add(cfg(b, true, XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, 1, 8), XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, 0, 8), false, flat));
            }
            baseC[s] = XBaseRule.merge(acc);
        }
        rows.put("(32)", baseC);
        for (Conf k : confs) {
            FeiDecompose.Cfg[] cs = new FeiDecompose.Cfg[sets.size()];
            XBaseFinal.Run rs = new XBaseFinal.Run(u, Map.of(), Map.of(), root.resolve(k.id()), false, 0);
            for (int s = 0; s < sets.size(); s++) {
                String set = sets.get(s);
                List<FeiDecompose.Cfg> acc = new ArrayList<>();
                for (XBaseVec.Blk b : bs) {
                    if (!b.group().equals(set)) continue;
                    acc.add(cfg(b, true, XBaseFinal.models(rs, b, XBaseFinal.TT1, false), XBaseFinal.models(rs, b, XBaseFinal.XS, false), true, flat));
                }
                cs[s] = XBaseRule.merge(acc);
            }
            rows.put(k.id(), cs);
        }
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
        StringBuilder t = new StringBuilder(String.format(Locale.ROOT, "%n=== Выбор по пулу (кадры «а», как прежние выборы): критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; бутстреп по людям B = %d, seed %d; база – векторы v8 pool-feat2 ===%n",
                XBasePart4.NOISE_B, u.tpl.seed + XBasePart4.NOISE_SEED_OFFSET));
        t.append("Правило, объявлено до прогона: выбирается наименьшее среднее критерия по трём наборам; «*» – парная разность с (32) значима (95 % без нуля).\nконфигурация");
        for (String s : sets) t.append(" | ").append(s);
        t.append(" | среднее | разность с (32) по наборам [95 %]\n");
        String best = null;
        double bestV = Double.POSITIVE_INFINITY;
        double[] p0 = pts.get("(32)");
        for (Map.Entry<String, double[]> e : pts.entrySet()) {
            double m = Arrays.stream(e.getValue()).average().orElseThrow();
            String lab = e.getKey().equals("(32)") ? "(32)" : confs.stream().filter(k -> k.id().equals(e.getKey())).findFirst().orElseThrow().label();
            t.append(lab);
            for (double v : e.getValue()) t.append(String.format(Locale.ROOT, " | %.2f", v));
            t.append(String.format(Locale.ROOT, " | %.2f |", m));
            for (int s = 0; s < sets.size() && !e.getKey().equals("(32)"); s++) {
                double[] sorted = FeiBootstrap.sorted(FeiNorm.diff(reps.get(e.getKey())[s], reps.get("(32)")[s]));
                double lo = FeiBootstrap.quantile(sorted, 0.025);
                double hi = FeiBootstrap.quantile(sorted, 0.975);
                t.append(String.format(Locale.ROOT, " %s %+.2f [%+.2f; %+.2f]%s;", sets.get(s), e.getValue()[s] - p0[s], lo, hi, lo > 0 || hi < 0 ? " *" : ""));
            }
            t.append('\n');
            if (!e.getKey().equals("(32)") && m < bestV) {
                bestV = m;
                best = e.getKey();
            }
        }
        boolean betterAny = false;
        StringBuilder sig = new StringBuilder();
        for (int s = 0; s < sets.size(); s++) {
            double[] sorted = FeiBootstrap.sorted(FeiNorm.diff(reps.get(best)[s], reps.get("(32)")[s]));
            boolean b = FeiBootstrap.quantile(sorted, 0.975) < 0;
            betterAny |= b;
            sig.append(String.format(Locale.ROOT, "%s %+.2f [%+.2f; %+.2f]%s; ", sets.get(s), pts.get(best)[s] - p0[s], FeiBootstrap.quantile(sorted, 0.025), FeiBootstrap.quantile(sorted, 0.975), b ? " значимо лучше" : ""));
        }
        String bb = best;
        t.append(String.format(Locale.ROOT, "%nВыбрана: %s (среднее %.2f против %.2f у (32)). Против (32): %s%n%s%nСлучаев нулевого разброса −SCI у «чужих» блока (σ = 1): %d%n", confs.stream().filter(k -> k.id().equals(bb)).findFirst().orElseThrow().label(),
                bestV, Arrays.stream(p0).average().orElseThrow(), sig, betterAny ? "Лучше базы значимо хотя бы на одном наборе." : "Ни на одном наборе не лучше базы значимо (как pool-noise); на GT всё равно проверяется (№16, п. 4).", flat[0]));
        Files.writeString(selFile, "выбор\t" + best + "\n");
        return t.toString();
    }

    private static String check(String base, XBaseFinal.Run r, Path root, List<XBaseVec.Blk> bs, List<String> groups, XBaseVec.Setup u, Conf k) throws IOException {
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        XBaseFinal.Run rs = runOf(u, r.k2(), root, k);
        int[] flat = new int[1];
        StringBuilder t = new StringBuilder("\n=== Выбранная по пулу: " + k.label() + " − (32); одна строка, основной интервал 95 % ===\n");
        for (String grp : groups) {
            List<XBaseVec.Blk> gb = bs.stream().filter(b -> b.group().equals(grp)).toList();
            List<FeiDecompose.Cfg> csB = new ArrayList<>();
            List<FeiDecompose.Cfg> csS = new ArrayList<>();
            List<FeiDecompose.Cfg> csR = new ArrayList<>();
            for (XBaseVec.Blk b : gb) {
                csB.add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                List<XBaseVec.Model> mt = XBaseFinal.models(rs, b, XBaseFinal.TT1, false);
                List<XBaseVec.Model> mx = XBaseFinal.models(rs, b, XBaseFinal.XS, false);
                csS.add(cfg(b, false, mt, mx, true, flat));
                csR.add(cfg(b, false, mt, mx, false, flat));
            }
            XBaseFinalReport.M mB = XBaseFinalReport.measure(csB, u.persons, w);
            XBaseFinalReport.M mS = XBaseFinalReport.measure(csS, u.persons, w);
            XBaseFinalReport.M mR = XBaseFinalReport.measure(csR, u.persons, w);
            t.append("\n").append(grp).append(": строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.level("(32) слияние", mB)).append(XBaseEsrc.level("ESRC-SCI, слияние", mS)).append(XBaseEsrc.level("справочно: d₁/d₂ по невязкам, слияние", mR));
            t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.row("ESRC-SCI − (32)", mS, mB)).append(XBaseEsrc.row("справочно: d₁/d₂ − (32)", mR, mB));
        }
        t.append(String.format(Locale.ROOT, "%nСлучаев нулевого разброса −SCI у «чужих» блока (σ = 1): %d%n", flat[0]));
        return t.toString();
    }
}
