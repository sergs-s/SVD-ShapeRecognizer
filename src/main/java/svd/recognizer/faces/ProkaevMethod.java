package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * «Метод Прокаева» (Д-№1 от 19.10.2026, ред. 4; PLAN.md п. 5.5, (9) (а2г), решения Д2–Д7; статья автора – Гидроакустика, 2023,
 * вып. 54 (2), с. 43–52, и его ответ 17.10, Д5). Не зависит от протоколов – годится для отчётов и для ПО.
 * <p>
 * Обучающая матрица A0 (m × n): строки – снимки (серый, без нормировки), гости – в ней же, одним классом {@link #GUEST}. SVD
 * A = L S R без центрирования – через матрицу Грама G = A·Aᵀ (double): собственные векторы – столбцы L, s_j = √λ_j; правый вектор
 * R_j = Aᵀ·L_j / s_j явно не строится: z·R_j = Σ_i (z·a_i) L_ij / s_j (нужны только скалярные произведения пробы со строками базиса).
 * Точка снимка и проба – проекции (для строки базиса проекция точно равна строке L_k):
 * <ul>
 *   <li>К (Б2-К, основная) – w_j = z·R_j / s_j, доля класса – сумма p_j его точек (формула 7);</li>
 *   <li>Н (Б2-н) – w_j = z·R_j (= строка L_k S_k), формула 7;</li>
 *   <li>С (Б2-с) – как К, доля класса – наибольшая p_j его точек, нормированная на сумму таких по классам (формула 8).</li>
 * </ul>
 * l ближайших точек (евклидово расстояние по первым k координатам); p_j = (1/d_j) / Σ(1/d_i); точки на нулевом расстоянии – поровну
 * между ними, прочие – 0. Правило автора: гость (отказ), если доля гостей > 0 и доля каждого своего < 0,95; иначе – свой с наибольшей
 * долей. Непрерывная оценка: 1 − доля лучшего своего, ничьи – по d₁ (расстояние до ближайшей точки этого человека).
 *
 * @author ssv
 */
final class ProkaevMethod {

    static final String GUEST = "*гости*";
    static final double RULE = 0.95;

    enum Variant {
        K, N, S
    }

    /** Сингулярные числа (по убыванию), k – число осей. */
    final double[] s;
    /** Столбцы L (по осям) на строках базиса. */
    final double[][] l;
    /** Точки обучения: варианты К/С и Н, [строка][ось]. */
    final double[][] ptsK;
    final double[][] ptsN;
    final String[] labels;

    private ProkaevMethod(double[] s, double[][] l, double[][] ptsK, double[][] ptsN, String[] labels) {
        this.s = s;
        this.l = l;
        this.ptsK = ptsK;
        this.ptsN = ptsN;
        this.labels = labels;
    }

    /** Модель по готовым точкам (для проверок решающего правила): точки – одинаковые для всех вариантов. */
    static ProkaevMethod ofPoints(double[][] pts, String[] labels) {
        return new ProkaevMethod(new double[0], new double[0][], pts, pts, labels);
    }

    /**
     * Обучение по матрице Грама строк базиса g (m × m) с метками basisLabels; extraDots – скалярные произведения прочих точек
     * обучения (гости при «базисе только по галерее») со строками базиса, extraLabels – их метки. Точки: сначала базис, затем прочие.
     */
    static ProkaevMethod fitGram(double[][] g, String[] basisLabels, double[][] extraDots, String[] extraLabels, int kMax) {
        int m = g.length;
        EigenDecomposition ed = new EigenDecomposition(new Array2DRowRealMatrix(g, true));
        double[] lam = ed.getRealEigenvalues();
        Integer[] ord = new Integer[m];
        for (int i = 0; i < m; i++) ord[i] = i;
        Arrays.sort(ord, Comparator.comparingDouble(i -> -lam[i]));
        double top = lam[ord[0]];
        int k = 0;
        while (k < Math.min(kMax, m) && lam[ord[k]] > 1e-12 * top) k++;
        double[] s = new double[k];
        double[][] lCol = new double[k][];
        RealMatrix v = ed.getV();
        for (int j = 0; j < k; j++) {
            s[j] = Math.sqrt(lam[ord[j]]);
            lCol[j] = v.getColumn(ord[j]);
        }
        int e = extraDots == null ? 0 : extraDots.length;
        String[] lab = new String[m + e];
        System.arraycopy(basisLabels, 0, lab, 0, m);
        if (e > 0) System.arraycopy(extraLabels, 0, lab, m, e);
        ProkaevMethod pm = new ProkaevMethod(s, lCol, new double[m + e][], new double[m + e][], lab);
        for (int i = 0; i < m; i++) {
            double[] pk = new double[k];
            double[] pn = new double[k];
            for (int j = 0; j < k; j++) {
                pk[j] = lCol[j][i];
                pn[j] = pk[j] * s[j];
            }
            pm.ptsK[i] = pk;
            pm.ptsN[i] = pn;
        }
        for (int i = 0; i < e; i++) {
            pm.ptsK[m + i] = pm.projectDots(extraDots[i], Variant.K);
            pm.ptsN[m + i] = pm.projectDots(extraDots[i], Variant.N);
        }
        return pm;
    }

    /** Обучение по строкам a (m × n) – для проверок и малых наборов: в базис – строки с basis[i] (null – все), прочие – точками. */
    static ProkaevMethod fit(float[][] a, String[] labels, boolean[] basis, int kMax) {
        int[] rows = IntStream.range(0, a.length).filter(i -> basis == null || basis[i]).toArray();
        int[] extra = IntStream.range(0, a.length).filter(i -> basis != null && !basis[i]).toArray();
        double[][] g = new double[rows.length][rows.length];
        for (int i = 0; i < rows.length; i++) for (int j = 0; j < rows.length; j++) g[i][j] = dot(a[rows[i]], a[rows[j]]);
        double[][] ed = new double[extra.length][rows.length];
        for (int i = 0; i < extra.length; i++) for (int j = 0; j < rows.length; j++) ed[i][j] = dot(a[extra[i]], a[rows[j]]);
        String[] bl = Arrays.stream(rows).mapToObj(i -> labels[i]).toArray(String[]::new);
        String[] el = Arrays.stream(extra).mapToObj(i -> labels[i]).toArray(String[]::new);
        ProkaevMethod pm = fitGram(g, bl, ed, el, kMax);
        // Точки – в исходном порядке строк.
        double[][] pk = new double[a.length][];
        double[][] pn = new double[a.length][];
        for (int i = 0; i < rows.length; i++) {
            pk[rows[i]] = pm.ptsK[i];
            pn[rows[i]] = pm.ptsN[i];
        }
        for (int i = 0; i < extra.length; i++) {
            pk[extra[i]] = pm.ptsK[rows.length + i];
            pn[extra[i]] = pm.ptsN[rows.length + i];
        }
        ProkaevMethod out = new ProkaevMethod(pm.s, pm.l, pk, pn, labels.clone());
        out.basisRows = Arrays.stream(rows).mapToObj(i -> a[i]).toArray(float[][]::new);
        return out;
    }

    /** Строки базиса (только для fit по строкам; проекция пробы по строкам). */
    float[][] basisRows;

    static double dot(float[] x, float[] y) {
        double d = 0;
        for (int c = 0; c < x.length; c++) d += (double) x[c] * y[c];
        return d;
    }

    int axes() {
        return s.length;
    }

    /** Проекция по скалярным произведениям пробы со строками базиса: z·R_j = Σ_i dots_i L_ij / s_j; К и С – ещё / s_j. */
    double[] projectDots(double[] dots, Variant v) {
        double[] w = new double[s.length];
        for (int j = 0; j < s.length; j++) {
            double d = 0;
            for (int i = 0; i < dots.length; i++) d += dots[i] * l[j][i];
            d /= s[j];
            w[j] = v == Variant.N ? d : d / s[j];
        }
        return w;
    }

    /** Проекция пробы z (только после fit по строкам). */
    double[] project(float[] z, Variant v) {
        double[] dots = new double[basisRows.length];
        for (int i = 0; i < dots.length; i++) dots[i] = dot(z, basisRows[i]);
        return projectDots(dots, v);
    }

    /** Итог пробы: доли классов, лучший свой, доля гостей, d₁, решение по правилу автора, справочно d₁/d₂ нашего правила. */
    record Result(Map<String, Double> share, String best, double bestShare, double guestShare, double d1, boolean guest, double nn1, String nn1Class,
                  double nn2, Map<String, Double> near) {

        /** Решение по правилу автора: null – гость (отказ), иначе свой. */
        String decided() {
            return guest ? null : best;
        }

        /** Непрерывная оценка (меньше – ближе): 1 − доля лучшего своего. */
        double score() {
            return 1 - bestShare;
        }
    }

    /**
     * Решение по проекции w (варианта v) на первых k осях, l ближайших точек. Лучший свой – наибольшая доля, при равных – меньшее
     * расстояние до ближайшей точки; если среди l ближайших своих нет – свой с ближайшей точкой, доля 0.
     */
    Result decide(double[] w, Variant v, int k, int l) {
        double[][] pts = v == Variant.N ? ptsN : ptsK;
        int m = pts.length;
        double[] d = new double[m];
        for (int i = 0; i < m; i++) {
            double q = 0;
            for (int j = 0; j < k; j++) {
                double t = w[j] - pts[i][j];
                q += t * t;
            }
            d[i] = Math.sqrt(q);
        }
        Integer[] ord = new Integer[m];
        for (int i = 0; i < m; i++) ord[i] = i;
        Arrays.sort(ord, Comparator.comparingDouble(i -> d[i]));
        int ll = Math.min(l, m);
        double[] p = new double[ll];
        int zeros = 0;
        for (int t = 0; t < ll; t++) if (d[ord[t]] == 0) zeros++;
        if (zeros > 0) {
            for (int t = 0; t < ll; t++) p[t] = d[ord[t]] == 0 ? 1.0 / zeros : 0;
        } else {
            double sum = 0;
            for (int t = 0; t < ll; t++) sum += 1 / d[ord[t]];
            for (int t = 0; t < ll; t++) p[t] = (1 / d[ord[t]]) / sum;
        }
        Map<String, Double> share = new LinkedHashMap<>();
        if (v == Variant.S) {
            Map<String, Double> mx = new LinkedHashMap<>();
            for (int t = 0; t < ll; t++) mx.merge(labels[ord[t]], p[t], Math::max);
            double sum = mx.values().stream().mapToDouble(Double::doubleValue).sum();
            mx.forEach((c, x) -> share.put(c, sum > 0 ? x / sum : 0));
        } else {
            for (int t = 0; t < ll; t++) share.merge(labels[ord[t]], p[t], Double::sum);
        }
        // Ближайшая точка каждого класса.
        Map<String, Double> near = new LinkedHashMap<>();
        for (int t = 0; t < m; t++) near.putIfAbsent(labels[ord[t]], d[ord[t]]);
        String best = null;
        double bs = -1;
        double bd = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Double> e : near.entrySet()) {
            if (e.getKey().equals(GUEST)) continue;
            double sh = share.getOrDefault(e.getKey(), 0.0);
            if (sh > bs || (sh == bs && e.getValue() < bd)) {
                best = e.getKey();
                bs = sh;
                bd = e.getValue();
            }
        }
        double gs = share.getOrDefault(GUEST, 0.0);
        boolean allLow = true;
        for (Map.Entry<String, Double> e : share.entrySet()) if (!e.getKey().equals(GUEST) && e.getValue() >= RULE) allLow = false;
        boolean guest = gs > 0 && allLow;
        // Справочно: d₁ – ближайшая точка своего, d₂ – ближайшая точка любого другого класса (и гостей).
        String c1 = null;
        double nn1 = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Double> e : near.entrySet()) {
            if (!e.getKey().equals(GUEST) && e.getValue() < nn1) {
                nn1 = e.getValue();
                c1 = e.getKey();
            }
        }
        double nn2 = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Double> e : near.entrySet()) if (!e.getKey().equals(c1)) nn2 = Math.min(nn2, e.getValue());
        return new Result(share, best, bs, gs, bd, guest, nn1, c1, nn2, near);
    }

    /** Классы своих по порядку первого появления в метках. */
    List<String> classes() {
        List<String> out = new ArrayList<>();
        for (String c : labels) if (!c.equals(GUEST) && !out.contains(c)) out.add(c);
        return out;
    }
}
