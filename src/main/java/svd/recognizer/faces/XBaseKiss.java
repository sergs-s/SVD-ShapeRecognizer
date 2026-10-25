package svd.recognizer.faces;

import java.io.IOException;
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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * KISSME (Köstinger и др., CVPR 2012) – п. 3 порядка ревизии (№12 Жорика от 25.10.2026 с поправками к плану, план подтверждён
 * Хозяином). Пары S – снимки одного класса обучения с разных дат (разные даты есть у людей пула FERET и, в (ii) s = 3, у галереи),
 * D – случайные пары снимков разных классов, |D| = 10·|S| (Random(faces.seed + 10400 + номер модели в блоке)); выборка снимков –
 * та же, что у W (галерея своих, когорта FEI, пул FERET; у X+mirror – с отражениями). Σ_S – с полом λ̄ = tr/n (c = 1), Σ_D – без
 * пола (ранг Σ_D меньше размерности – остановка), M = Σ_S⁻¹ − Σ_D⁻¹, отрицательные собственные значения M – в 0. Оценка –
 * (x − μ_c)ᵀ M (x − μ_c) до центра класса, решение – d₁/d₂, модель X / X+mirror по |r|. Варианты: (К2) – в пространстве W текущей
 * модели MLDA; (К1) – вместо MLDA, на PCA той же выборки, m = 50, 100, 200, 300. База – косинус (решение (26)), c = 1, L1-xs-F 7×7,
 * L1-tt-xs-F 9×9 (решение (27)). Режимы XBaseEval pool-kiss (выбор по пулу, схема pool-rule), feret-kiss, gt-kiss.
 *
 * @author ssv
 */
final class XBaseKiss {

    static final int[] MS = {50, 100, 200, 300};
    /** Строки: 0 – база; 1 – (К2); 2–5 – (К1) по MS; NO + v – вариант v без пола у Σ_S. */
    static final String[] VNAME = {"база (косинус)", "(К2) KISSME в W", "(К1) m = 50", "(К1) m = 100", "(К1) m = 200", "(К1) m = 300"};
    static final int BASE = 0;
    static final int NV = VNAME.length;
    static final int NO = NV - 1;
    static final int ROWS = 2 * NV - 1;
    static final long SEED = 10400;
    static final int D_PER_S = 10;
    static final String SELECT = "xbase_pool_kiss_select.tsv";
    static final double RANK_TOL = 1e-10;

    private XBaseKiss() {
    }

    static String rowName(int r) {
        return r < NV ? VNAME[r] : VNAME[r - NO] + ", без пола у Σ_S";
    }

    /** Пробный режим: 1 – малые данные, весь путь (12 своих, пул 15, без сверок); 2 – первая группа блоков в полном размере (замер времени). */
    static int smoke() {
        String v = System.getenv("FACES_KISS_SMOKE");
        return v == null ? 0 : Integer.parseInt(v.trim());
    }

    static List<String> trim(List<String> own) {
        return smoke() == 1 && own.size() > XBaseDates.SMOKE_N ? own.subList(0, XBaseDates.SMOKE_N) : own;
    }

    // ---------------------------------------------------------------- KISSME

    /** Сведения модели KISSME: |S|, |D|, размерность, ранги Σ_S и Σ_D, поднято до пола у Σ_S, обнулено у M. */
    record Diag(int s, int d, int dim, int rankS, int rankD, int raised, int zeroed) {
        String row() {
            return String.format(Locale.ROOT, "%d\t%d\t%d\t%d\t%d\t%d\t%d", s, d, dim, rankS, rankD, raised, zeroed);
        }
    }

    /** Пары обучения: векторы (с отражениями), класс и дата каждого; S и D – индексы пар. */
    record Pairs(List<double[]> vec, int[][] s, int[][] d) {}

    /**
     * Пары выборки classes: S – одного класса с разных дат (дата – dateOf, ключи без даты в S не входят), D – разных классов,
     * |D| = D_PER_S·|S| (с повторами, Random(seed)); все пары D, если их не больше.
     */
    static Pairs pairs(XBaseEval x, boolean mirror, List<List<String>> classes, Map<String, String> dateOf, long seed) {
        List<double[]> vec = new ArrayList<>();
        List<Integer> cls = new ArrayList<>();
        List<String> date = new ArrayList<>();
        for (int c = 0; c < classes.size(); c++) {
            for (String k : classes.get(c)) {
                vec.add(x.vec.get(V.L1).get(k));
                cls.add(c);
                date.add(dateOf.get(k));
                if (mirror) {
                    vec.add(x.mir.get(V.L1).get(k));
                    cls.add(c);
                    date.add(dateOf.get(k));
                }
            }
        }
        int n = vec.size();
        List<int[]> s = new ArrayList<>();
        int from = 0;
        while (from < n) {
            int to = from;
            while (to < n && cls.get(to).equals(cls.get(from))) to++;
            for (int i = from; i < to; i++) for (int j = i + 1; j < to; j++) {
                if (date.get(i) != null && date.get(j) != null && !date.get(i).equals(date.get(j))) s.add(new int[] {i, j});
            }
            from = to;
        }
        if (s.isEmpty()) throw new IllegalStateException("XBaseKiss: нет межсессионных пар S в выборке обучения");
        long within = 0;
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int c : cls) cnt.merge(c, 1, Integer::sum);
        for (int v : cnt.values()) within += (long) v * (v - 1) / 2;
        long total = (long) n * (n - 1) / 2 - within;
        long want = (long) D_PER_S * s.size();
        List<int[]> d = new ArrayList<>();
        if (total <= want) {
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (!cls.get(i).equals(cls.get(j))) d.add(new int[] {i, j});
        } else {
            Random rnd = new Random(seed);
            while (d.size() < want) {
                int i = rnd.nextInt(n);
                int j = rnd.nextInt(n);
                if (cls.get(i).equals(cls.get(j))) continue;
                d.add(new int[] {i, j});
            }
        }
        return new Pairs(vec, s.toArray(new int[0][]), d.toArray(new int[0][]));
    }

    /** Σ = среднее (z_i − z_j)(z_i − z_j)ᵀ по парам. */
    static double[][] cov(double[][] z, int[][] pr) {
        int m = z[0].length;
        double[][] c = new double[m][m];
        double[] t = new double[m];
        for (int[] p : pr) {
            double[] a = z[p[0]];
            double[] b = z[p[1]];
            for (int i = 0; i < m; i++) t[i] = a[i] - b[i];
            for (int i = 0; i < m; i++) {
                double ti = t[i];
                double[] ci = c[i];
                for (int j = 0; j <= i; j++) ci[j] += ti * t[j];
            }
        }
        for (int i = 0; i < m; i++) for (int j = 0; j <= i; j++) {
            c[i][j] /= pr.length;
            c[j][i] = c[i][j];
        }
        return c;
    }

    static double[][] lead(double[][] c, int m) {
        double[][] o = new double[m][];
        for (int i = 0; i < m; i++) o[i] = Arrays.copyOf(c[i], m);
        return o;
    }

    /** Модель KISSME: L (строки √μ_k u_kᵀ, μ_k > 0) – d = ‖L(z − μ_c)‖²; сведения. */
    record Kiss(double[][] l, Diag diag) {}

    /**
     * M = Σ_S⁻¹ − Σ_D⁻¹ (пол λ̄ = tr/n у Σ_S при floor; без пола – псевдообратная по рангу), отрицательные собственные значения M – в 0.
     * Ранг Σ_D меньше размерности – исключение.
     */
    static Kiss kiss(double[][] cs, double[][] cd, int ns, int nd, boolean floor) {
        int m = cs.length;
        LdaMath.Eigen es = LdaMath.eigen(cs);
        LdaMath.Eigen ed = LdaMath.eigen(cd);
        double maxS = Arrays.stream(es.values()).max().orElse(0);
        double maxD = Arrays.stream(ed.values()).max().orElse(0);
        int rankS = 0;
        int rankD = 0;
        for (double v : es.values()) if (v > RANK_TOL * maxS) rankS++;
        for (double v : ed.values()) if (v > RANK_TOL * maxD) rankD++;
        if (rankD < m) throw new IllegalStateException("XBaseKiss: ранг Σ_D " + rankD + " меньше размерности " + m + " (|D| = " + nd + ")");
        double tr = 0;
        for (double v : es.values()) tr += v;
        double bar = tr / m;
        int raised = 0;
        double[] invS = new double[m];
        for (int k = 0; k < m; k++) {
            double v = es.values()[k];
            if (floor) {
                if (v < bar) {
                    v = bar;
                    raised++;
                }
                invS[k] = 1 / v;
            } else {
                invS[k] = v > RANK_TOL * maxS ? 1 / v : 0;
            }
        }
        double[][] mm = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                double a = 0;
                for (int k = 0; k < m; k++) {
                    a += es.vectors()[i][k] * invS[k] * es.vectors()[j][k];
                    a -= ed.vectors()[i][k] / ed.values()[k] * ed.vectors()[j][k];
                }
                mm[i][j] = a;
                mm[j][i] = a;
            }
        }
        EigenDecomposition e = new EigenDecomposition(new Array2DRowRealMatrix(mm, false));
        double[] mu = e.getRealEigenvalues();
        int zeroed = 0;
        List<double[]> rows = new ArrayList<>();
        for (int k = 0; k < m; k++) {
            if (mu[k] <= 0) {
                zeroed++;
                continue;
            }
            double sq = Math.sqrt(mu[k]);
            double[] u = e.getEigenvector(k).toArray();
            for (int i = 0; i < m; i++) u[i] *= sq;
            rows.add(u);
        }
        return new Kiss(rows.toArray(new double[0][]), new Diag(ns, nd, m, rankS, rankD, raised, zeroed));
    }

    static double dist(double[][] l, double[] z, double[] c) {
        double s = 0;
        for (double[] r : l) {
            double a = 0;
            for (int i = 0; i < r.length; i++) a += r[i] * (z[i] - c[i]);
            s += a * a;
        }
        return s;
    }

    /** Центры классов gallery (первые g классов выборки) в координатах z. */
    static double[][] centers(double[][] z, List<List<String>> classes, int g, boolean mirror) {
        double[][] c = new double[g][];
        int at = 0;
        for (int q = 0; q < g; q++) {
            int n = classes.get(q).size() * (mirror ? 2 : 1);
            if (n == 0) continue;
            double[] m = new double[z[0].length];
            for (int i = at; i < at + n; i++) for (int j = 0; j < m.length; j++) m[j] += z[i][j];
            for (int j = 0; j < m.length; j++) m[j] /= n;
            c[q] = m;
            at += n;
        }
        return c;
    }

    static String lines(String method, int s, List<String> gallery, List<Probe> probes, double[][] l, double[][] c, java.util.function.Function<String, double[]> proj) {
        StringBuilder out = new StringBuilder();
        for (Probe p : probes) {
            double[] z = proj.apply(p.key());
            double[] sc = new double[c.length];
            for (int g = 0; g < c.length; g++) sc[g] = c[g] == null ? Double.POSITIVE_INFINITY : dist(l, z, c[g]);
            out.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc, false));
        }
        return out.toString();
    }

    /**
     * Все строки одной модели: [0] база – XBaseReg.lines (косинус); [1] (К2); [2–5] (К1); [NO + v] без пола у Σ_S; последний элемент – сведения
     * (строка на вариант с полом). gallery – классы галереи (в порядке own), rest – когорта и пул.
     */
    static String[] model(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> rest, List<Probe> probes,
                          Map<String, String> dateOf, long seed, String tag) {
        XBaseRule.Fit f = XBaseRule.fit(x, mirror, gallery, rest, null);
        String[] out = new String[ROWS + 1];
        out[0] = XBaseReg.lines(x, method, s, f.lm, own, probes)[1];
        List<List<String>> classes = new ArrayList<>(gallery);
        classes.addAll(rest);
        Pairs pr = pairs(x, mirror, classes, dateOf, seed);
        StringBuilder diag = new StringBuilder();
        // (К2): в пространстве W.
        double[][] y = new double[pr.vec().size()][];
        for (int i = 0; i < y.length; i++) y[i] = f.lm.embed(pr.vec().get(i));
        double[][] csW = cov(y, pr.s());
        double[][] cdW = cov(y, pr.d());
        double[][] cW = centers(y, gallery, gallery.size(), mirror);
        for (boolean fl : new boolean[] {true, false}) {
            Kiss k = kiss(csW, cdW, pr.s().length, pr.d().length, fl);
            out[fl ? 1 : NO + 1] = lines(method, s, own, probes, k.l(), cW, key -> f.lm.embed(x.vec.get(V.L1).get(key)));
            if (fl) diag.append(tag).append('\t').append(VNAME[1]).append('\t').append(k.diag().row()).append('\n');
        }
        // (К1): PCA выборки (m = 300; меньшие m – первые компоненты и левые верхние блоки ковариаций).
        LdaMath.Data dt = LdaMath.Data.of(List.of(pr.vec()));
        int mMax = MS[MS.length - 1];
        LdaMath.Span sp = LdaMath.span(dt, RANK_TOL, mMax);
        int r = sp.basis().length;
        double[] mean = dt.mean();
        double[][] basis = sp.basis();
        java.util.function.Function<double[], double[]> pca = v -> {
            double[] z = new double[r];
            for (int j = 0; j < r; j++) {
                double a = 0;
                double[] b = basis[j];
                for (int i = 0; i < v.length; i++) a += b[i] * (v[i] - mean[i]);
                z[j] = a;
            }
            return z;
        };
        double[][] z = new double[pr.vec().size()][];
        for (int i = 0; i < z.length; i++) z[i] = pca.apply(pr.vec().get(i));
        double[][] csP = cov(z, pr.s());
        double[][] cdP = cov(z, pr.d());
        double[][] cP = centers(z, gallery, gallery.size(), mirror);
        Map<String, double[]> zp = new HashMap<>();
        for (Probe p : probes) zp.computeIfAbsent(p.key(), key -> pca.apply(x.vec.get(V.L1).get(key)));
        for (int mi = 0; mi < MS.length; mi++) {
            int m = Math.min(MS[mi], r);
            double[][] cs = lead(csP, m);
            double[][] cd = lead(cdP, m);
            double[][] cm = new double[cP.length][];
            for (int g = 0; g < cP.length; g++) cm[g] = cP[g] == null ? null : Arrays.copyOf(cP[g], m);
            java.util.function.Function<String, double[]> pm = key -> Arrays.copyOf(zp.get(key), m);
            for (boolean fl : new boolean[] {true, false}) {
                Kiss k = kiss(cs, cd, pr.s().length, pr.d().length, fl);
                out[(fl ? 0 : NO) + 2 + mi] = lines(method, s, own, probes, k.l(), cm, pm);
                if (fl) diag.append(tag).append('\t').append(VNAME[2 + mi]).append('\t').append(k.diag().row()).append('\n');
            }
        }
        out[ROWS] = diag.toString();
        return out;
    }

    /** Задачи блока – как XBaseLbp.job (тот же порядок моделей и проб); строки каждой модели – model(). */
    static List<Callable<String[]>> job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts, int s,
                                        List<List<Probe>> extra, Map<String, String> dateOf, long seed) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<String[]>> tasks = new ArrayList<>();
        int mno = 0;
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            long s0 = seed + SEED + mno++;
            String t0 = (m ? "X+mirror" : "X") + " основная";
            tasks.add(() -> model(x, m, method, s, own, fullGallery, cohort, main, dateOf, s0, t0));
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                long sq = seed + SEED + mno++;
                String tq = (m ? "X+mirror" : "X") + " без части " + q;
                tasks.add(() -> model(x, m, method, s, g, gk, cohort, pr, dateOf, sq, tq));
            }
        }
        return tasks;
    }

    /** Блоки – все задачи разом; [блок][строка ROWS + сведения]. */
    static List<String[]> run(List<List<Callable<String[]>>> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (List<Callable<String[]>> j : jobs) flat.addAll(j);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<String[]> out = new ArrayList<>();
            int i = 0;
            for (List<Callable<String[]>> j : jobs) {
                StringBuilder[] b = new StringBuilder[ROWS + 1];
                for (int r = 0; r <= ROWS; r++) b[r] = new StringBuilder();
                for (int k = 0; k < j.size(); k++) {
                    String[] r = fs.get(i++).get();
                    for (int q = 0; q <= ROWS; q++) b[q].append(r[q]);
                }
                String[] o = new String[ROWS + 1];
                for (int r = 0; r <= ROWS; r++) o[r] = b[r].toString();
                out.add(o);
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseKiss: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    static String[] read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[] o = new String[ROWS + 1];
        for (int r = 0; r <= ROWS; r++) {
            o[r] = cp.read(tag + "_" + r);
            if (o[r] == null) return null;
        }
        return o;
    }

    static void save(XBaseOther.Checkpoint cp, String tag, String[] o) throws IOException {
        for (int r = 0; r <= ROWS; r++) cp.save(tag + "_" + r, o[r]);
    }

    static final String DIAG_HEAD = "\n=== Сведения KISSME (варианты с полом; модели блока: X / X+mirror, основная и без части q) ===\n"
            + "блок | модель | вариант | |S| | |D| | размерность | ранг Σ_S | ранг Σ_D | поднято до пола у Σ_S | обнулено у M\n";

    static String diag(String block, String d) {
        StringBuilder t = new StringBuilder();
        for (String l : d.lines().toList()) t.append(block).append(" | ").append(l.replace("\t", " | ")).append('\n');
        return t.toString();
    }

    /** Даты снимков FERET (ключ → дата), для пар S. */
    static Map<String, String> dates(Map<String, Img> byKey) {
        Map<String, String> m = new HashMap<>();
        byKey.forEach((k, im) -> m.put(k, im.date()));
        return m;
    }

    static final String NOTE = "KISSME: пары S – снимки одного класса выборки обучения W с разных дат (разные даты – у людей пула FERET; в (ii) s = 3 – и у галереи), D – "
            + "случайные пары снимков\nразных классов, |D| = 10·|S| (Random(faces.seed + 10400 + номер модели в блоке)); выборка – та же, что у W (галерея своих, когорта "
            + "FEI 20, пул FERET; у X+mirror – с\nотражениями); Σ_S – пол λ̄ = tr/n (c = 1), Σ_D – без пола; M = Σ_S⁻¹ − Σ_D⁻¹, отрицательные собственные значения – "
            + "в 0; оценка (x − μ_c)ᵀ M (x − μ_c)\nдо центра класса, решение – d₁/d₂, модель X / X+mirror по |r|. (К2) – в пространстве W той же модели MLDA (база и "
            + "(К2) – одна модель); (К1) – вместо\nMLDA, на PCA той же выборки, m = 50, 100, 200, 300 (меньшие m – первые компоненты). База – 1 − cos (решение (26)), "
            + "L1-xs-F 7×7, L1-tt-xs-F 9×9 (решение (27)).\n";

    // ---------------------------------------------------------------- pool-kiss

    /** Правило выбора (№12): база, если в пределах NEAR от лучшего; иначе первый в порядке простоты ((К2) < (К1) m = 50 < … < 300) в пределах NEAR. */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        if (crit[BASE] <= best + XBaseRule.NEAR) return BASE;
        for (int v = 1; v < crit.length; v++) if (crit[v] <= best + XBaseRule.NEAR) return v;
        throw new IllegalStateException("XBaseKiss: выбор не определён");
    }

    static XBaseEval[] vectors(Map<String, Mat> fr, IlluminationNorm.NormParams params) {
        XBaseEval[] xs = new XBaseEval[2];
        for (int f = 0; f < 2; f++) {
            xs[f] = new XBaseEval();
            xs[f].vectors(fr, params, XBaseDates.NORMS[f], XBaseDates.PARAMS[f]);
        }
        return xs;
    }

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
        Map<String, String> dateOf = dates(s.su.byKey);
        Path lbpScores = s.out.resolve("xbase_pool_lbp_scores.tsv.gz");
        if (!Files.exists(lbpScores)) throw new IllegalStateException("pool-kiss: для сверки базы нужен " + lbpScores + " (pool-lbp)");
        Map<String, String> blocks = XBaseSubsp.blocks(lbpScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-kiss %s; seed %d; r0 %s; пул %d; половины %d/%d; варианты %s; пробный %d",
                s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", VNAME), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_kiss_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        XBaseEval[] xv = vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();
        StringBuilder head = new StringBuilder();
        head.append("П. 3 ревизии: KISSME, выбор варианта по пулу FERET (XBaseEval pool-kiss)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-lbp: GT – весь пул; FERET-1 – половина B (%d; проход 1 feret-kiss); FERET-2 – половина A "
                + "(%d; проход 2);%n5 частей по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула; "
                + "обучение – галерея своих + когорта FEI 20 + люди 4 частей.%n", s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET,
                XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база, если в пределах %.0f п. п. от лучшего, иначе "
                + "первый в порядке простоты%n((К2) < (К1) m = 50 < 100 < 200 < 300) в пределах %.0f п. п. Seed пар D – faces.seed + 10400 + номер модели в блоке (0 – X, 1 – "
                + "X+mirror).%n", XBaseRule.NEAR, XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = pool-lbp (L1-xs-F – база 7×7, L1-tt-xs-F – вариант (2) 9×9), побайтно ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        // [набор][признак][вариант] → конфигурации по частям и сменам.
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
            for (int f = 1; f >= 0; f--) {
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
                    jobs.add(job((XBaseEval.Gt) b[2], xv[f], gal, probes, null, 2 * (int) b[0] + (int) b[1], null, dateOf, tpl.seed));
                    todo.add(b);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        todo.get(i)[6] = done.get(i);
                        save(cp, (String) todo.get(i)[5], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-kiss: %s f%d ready, %d blocks (%.0f s)%n", set, f, jobs.size(), sec);
                }
                for (Object[] b : blk) {
                    int q = (int) b[0];
                    int h = (int) b[1];
                    XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    String[] ls = (String[]) b[6];
                    String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                    if (sm != 1) check.append(XBaseReg.check(bp + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", ls[0], blocks, lbpScores));
                    dg.append(diag(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], q, h), ls[ROWS]));
                    for (int v = 0; v < NV; v++) {
                        scores.append("# ").append(bp).append(VNAME[v]).append('\n').append(ls[v]);
                        cfgs.get(si).get(f).get(v).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls[v]).cfg.get(V.L1));
                    }
                }
                if (sm == 2) break outer;
            }
        }
        if (sm == 2) {
            StringBuilder t = new StringBuilder(head).append(check).append(dg).append(String.format(Locale.ROOT, "Замер: первая группа (набор GT, L1-tt-xs-F, 10 блоков) – "
                    + "%.0f с; групп всего %d (3 набора × 2 признака).%n", fitS, setNames.size() * 2));
            FarMethods.write(s.out.resolve("xbase_pool_kiss" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseKiss pool-kiss): набор, признак, номер варианта, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
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
        XBaseOther.writeScores(s.out.resolve("xbase_pool_kiss" + sfx + "_scores.tsv.gz"), "# KISSME, выбор по пулу FERET (XBaseKiss pool-kiss), только локально\n# Код: "
                + "коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_kiss" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_kiss" + sfx + "_gt.txt"), new StringBuilder(head).append(gtBody));
        System.out.print(t);
        cp.delete();
    }

    /** Выбор по пулу: набор → признак → номер варианта; в пробном режиме 1 – (К2) везде. */
    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (smoke() == 1) {
            Map<String, Map<String, Integer>> o = new LinkedHashMap<>();
            for (String set : List.of("GT", "FERET-1", "FERET-2")) for (String ft : XBaseRule.FEATS) o.computeIfAbsent(set, k -> new LinkedHashMap<>()).put(ft, 1);
            return o;
        }
        if (!Files.exists(f)) throw new IllegalStateException("XBaseKiss: нет файла выбора " + f + " (сначала pool-kiss)");
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int v = Integer.parseInt(p[2]);
            if (v < 0 || v >= NV || !VNAME[v].equals(p[3])) throw new IllegalStateException("XBaseKiss: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    // ---------------------------------------------------------------- на своих

    /** Проход: Gt, галерея, пробы своих, части, все пробы, пробы (Т1). */
    record Pass(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, List<Probe> probes,
                List<List<Probe>> extra) {}

    /**
     * Строки отчёта постановки: база, все варианты, «выбор по пулу» (вариант – свой на проход) и «выбор без пола у Σ_S» (если выбор – не база);
     * пары «вариант − база». rows[проход] – строки модели (ROWS + 1).
     */
    static void post(StringBuilder t, String title, String feat, List<Pass> passes, List<String[]> rows, int[] chosen, int persons, XBaseEval.Gt tpl, boolean t2,
                     double[] delta) {
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        String bl = feat + " " + VNAME[BASE];
        List<Integer> show = new ArrayList<>();
        for (int v = 0; v < NV; v++) show.add(v);
        for (int v : show) outs.put(feat + " " + VNAME[v], XBaseThr.evaluate(feat + " " + VNAME[v], units(passes, rows, new int[] {v, v}, t2), persons, tpl));
        String ch = feat + " выбор по пулу";
        outs.put(ch, XBaseThr.evaluate(ch, units(passes, rows, chosen, t2), persons, tpl));
        boolean nf = chosen[0] != BASE && chosen[chosen.length - 1] != BASE;
        String chNo = feat + " выбор по пулу, без пола у Σ_S";
        if (nf) outs.put(chNo, XBaseThr.evaluate(chNo, units(passes, rows, Arrays.stream(chosen).map(c -> c + NO).toArray(), t2), persons, tpl));
        for (String k : outs.keySet()) if (!k.equals(bl)) pairs.add(new String[] {k, bl});
        if (delta != null) {
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            String way = t2 ? XBaseThr.WAYS[2] : XBaseThr.WAYS[0];
            XBaseRule.Sep a = XBaseRule.sep(outs.get(ch).res.get(way).cfgs, FeiBootstrap.unit(persons), w);
            XBaseRule.Sep z = XBaseRule.sep(outs.get(bl).res.get(way).cfgs, FeiBootstrap.unit(persons), w);
            double[] d = FeiNorm.diff(a.errRep[0], z.errRep[0]);
            delta[0] = a.err[0] - z.err[0];
            delta[1] = d.length == 0 ? Double.NaN : FeiBootstrap.quantile(FeiBootstrap.sorted(d), 0.975);
        }
        XBaseLbp.tables(t, title, outs, persons, tpl, pairs, t2);
    }

    static List<XBaseThr.Unit> units(List<Pass> passes, List<String[]> rows, int[] v, boolean t2) {
        List<XBaseThr.Unit> us = new ArrayList<>();
        for (int i = 0; i < passes.size(); i++) {
            Pass p = passes.get(i);
            us.add(new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), rows.get(i)[v[i]], null, t2));
        }
        return us;
    }

    // ---------------------------------------------------------------- feret-kiss

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        Map<String, Img> byKey = s.su.byKey;
        Map<String, String> dateOf = dates(byKey);
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        // Все отобранные снимки своих по датам – как XBaseDates.
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
        for (Path f : List.of(lbpScores, lbpTxt, datesScores, datesTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-kiss: нужен " + f);
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        Map<String, String> datesBlocks = XBaseSubsp.blocks(datesScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        List<String> datesReport = Files.readAllLines(datesTxt, StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-kiss %s; seed %d; r0 %s; пул %d; половины %d/%d; выбор %s; пробный %d", s.su.dir,
                tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), select, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_kiss_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);
        XBaseEval[] xv = vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();

        StringBuilder t = new StringBuilder();
        t.append("П. 3 ревизии: KISSME, FERET (XBaseEval feret-kiss): Color FERET — только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-lbp); (ii) s = 3 – как feret-dates (люди с ≥ 4 датами, первые 3 "
                + "даты, пробы с 4-й,\nчасти – номер 71), справочно. «Выбор по пулу» – вариант из pool-kiss (проход 1 – FERET-1, проход 2 – FERET-2); «без пола у Σ_S» – "
                + "тот же вариант без пола\n(справочно, если выбор – не база). Контрольные – те же 200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = feret-lbp («межс.») и feret-dates ((ii) s3 (м)), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        for (int post = 0; post < 2; post++) {
            String pname = post == 0 ? "межс." : "(ii) s3";
            int conf = post == 0 ? 0 : 71;
            List<Pass> passes = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (post == 0) {
                    own = trim(ro.own1.stream().filter(mine::contains).toList());
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else {
                    own = trim(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList());
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
                passes.add(new Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
            }
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                List<String[]> rows = new ArrayList<>();
                List<List<Callable<String[]>>> jobs = new ArrayList<>();
                List<Integer> todo = new ArrayList<>();
                for (int ps = 0; ps < 2; ps++) {
                    String tag = String.format(Locale.ROOT, "c%d_p%d_f%d", conf, ps, f);
                    String[] r = read(cp, tag);
                    rows.add(r);
                    if (r != null) continue;
                    Pass p = passes.get(ps);
                    jobs.add(job(p.g(), xv[f], p.gal(), p.probes(), p.parts(), conf, p.extra(), dateOf, tpl.seed));
                    todo.add(ps);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        rows.set(todo.get(i), done.get(i));
                        save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d", conf, todo.get(i), f), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "feret-kiss: %s %s ready, %d blocks (%.0f s)%n", pname, feat, jobs.size(), sec);
                }
                for (int ps = 0; ps < 2; ps++) {
                    String[] r = rows.get(ps);
                    if (sm != 1) {
                        if (post == 0) check.append(XBaseReg.check("проход " + (ps + 1) + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", r[0],
                                lbpBlocks, lbpScores));
                        else check.append(XBaseReg.check("(i)=(ii) s3 (м) " + feat + ", проход " + (ps + 1), r[0], datesBlocks, datesScores));
                    }
                    dg.append(diag(pname + " " + feat + " проход " + (ps + 1), r[ROWS]));
                    for (int v = 0; v < ROWS; v++) lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ")
                            .append(rowName(v)).append('\n').append(r[v]);
                }
                if (sm == 2) {
                    t.append(check).append(dg).append(String.format(Locale.ROOT, "Замер: «межс.» L1-tt-xs-F, 2 прохода – %.0f с; блоков всего 8 (2 постановки × 2 признака "
                            + "× 2 прохода).%n", fitS));
                    FarMethods.write(s.out.resolve("xbase_feret_kiss" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int[] chosen = {select.get("FERET-1").get(feat), select.get("FERET-2").get(feat)};
                if (post == 0) sel.append(String.format(Locale.ROOT, "%s: проход 1 – %s, проход 2 – %s%n", feat, VNAME[chosen[0]], VNAME[chosen[1]]));
                if (sm != 1) {
                    XBaseThr.Out o = XBaseThr.evaluate("база", units(passes, rows, new int[] {0, 0}, true), persons.size(), tpl);
                    String rl = post == 0 ? "межс. " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]] : "(i)=(ii) s3 (м) " + feat;
                    for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), post == 0 ? lbpReport : datesReport,
                            tpl.b, post == 0 ? lbpTxt : datesTxt));
                }
                post(body, "FERET " + pname + ", " + feat + (post == 1 ? " (справочно; 25 человек)" : ""), feat, passes, rows, chosen, persons.size(), tpl, true, null);
            }
        }
        t.append(check).append(sel);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body).append(dg);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_kiss" + sfx + "_scores.tsv.gz"), "# FERET, KISSME (XBaseKiss feret-kiss), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_kiss" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-kiss

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-kiss: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Map<String, String> dateOf = dates(su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval[] xv = vectors(fr, g.params);
        for (Mat m : fr.values()) m.release();
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = trim(g.own);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        Path lbpTxt = out.resolve("xbase_gt_lbp.txt");
        if (!Files.exists(lbpScores) || !Files.exists(lbpTxt)) throw new IllegalStateException("gt-kiss: нужны " + lbpScores + " и " + lbpTxt + " (gt-lbp)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-kiss; seed %d; r0 %s; пул FERET %d; выбор %s; пробный %d", g.seed, g.r0, pool.size(),
                select, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_kiss" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("П. 3 ревизии: KISSME, GT (XBaseEval gt-kiss)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-lbp: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «выбор по пулу − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = gt-lbp (межс. и внутри), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «выбор по пулу − база») ===\n");
        double fitS = 0;
        for (int mi = 0; mi < modes.length; mi++) {
            String mode = (String) modes[mi][0];
            boolean x = mi == 0;
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
            List<Probe> probes = XBaseEval.probes(ga, pro);
            Pass pass = new Pass(ga, gal, pro, parts, probes, x ? XBaseThr.extra(ga, pro, parts) : null);
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                String tag = (x ? "x" : "i") + "_f" + f;
                String[] r = read(cp, tag);
                if (r == null) {
                    long t0 = System.nanoTime();
                    r = run(List.of(job(ga, xv[f], gal, probes, parts, 0, pass.extra(), dateOf, g.seed)), g.threads).get(0);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    save(cp, tag, r);
                    System.out.printf(Locale.ROOT, "gt-kiss: %s %s ready (%.0f s)%n", mode, feat, sec);
                }
                if (sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", r[0], lbpBlocks, lbpScores));
                dg.append(diag(mode + " " + feat, r[ROWS]));
                for (int v = 0; v < ROWS; v++) lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(rowName(v)).append('\n').append(r[v]);
                if (sm == 2) {
                    t.append(check).append(dg).append(String.format(Locale.ROOT, "Замер: межс. L1-tt-xs-F – %.0f с; блоков всего 4 (2 режима × 2 признака).%n", fitS));
                    FarMethods.write(out.resolve("xbase_gt_kiss" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int c = select.get("GT").get(feat);
                if (sm != 1) {
                    XBaseThr.Out o = XBaseThr.evaluate("база", List.of(new XBaseThr.Unit(ga, gal, pro, parts, r[0], null, x)), g.persons.size(), g);
                    String rl = mode + " " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]];
                    for (int w : x ? new int[] {0, 2} : new int[] {0}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), lbpReport, g.b,
                            lbpTxt));
                }
                double[] delta = x ? null : new double[2];
                post(body, "GT " + mode + ", " + feat + " (выбор по пулу – " + VNAME[c] + ")", feat, List.of(pass), Collections.singletonList(r), new int[] {c}, g.persons.size(), g, x, delta);
                if (delta != null) dl.append(String.format(Locale.ROOT, "%s: выбор – %s; разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", feat, VNAME[c],
                        delta[0], delta[1], c == BASE ? "не применяется (выбор – база)" : delta[1] <= 3 ? "выполнено" : "НЕ выполнено"));
            }
        }
        t.append(check).append(dl);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body).append(dg);
        XBaseOther.writeScores(out.resolve("xbase_gt_kiss" + sfx + "_scores.tsv.gz"), "# GT, KISSME (XBaseKiss gt-kiss)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_kiss" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
