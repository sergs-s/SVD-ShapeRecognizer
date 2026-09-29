package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * Линейная алгебра LDA (шаг 5, этап 2): линейная оболочка обучающих векторов через матрицу Грама,
 * разбросы S_w и S_b, критерий Фишера через отбеливание S_w (с заменой спектра MLDA или без неё).
 *
 * @author ssv
 */
final class LdaMath {

    private LdaMath() {
    }

    /** Разложение симметричной матрицы: собственные значения по убыванию, векторы — столбцы. */
    record Eigen(double[] values, double[][] vectors) {}

    /**
     * Разложение симметричной неотрицательно определённой матрицы (матрица Грама, S_w, WᵀS_bW). Если QL-алгоритм
     * Commons Math не сходится (сильно вырожденный спектр — S_w в полном пространстве), — через SVD: для такой
     * матрицы сингулярные числа и левые векторы совпадают с собственными.
     */
    static Eigen eigen(double[][] sym) {
        EigenDecomposition ed;
        try {
            ed = new EigenDecomposition(new Array2DRowRealMatrix(sym, false));
        } catch (MaxCountExceededException e) {
            SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(sym, false));
            double[] values = svd.getSingularValues();
            double[][] u = svd.getU().getData();
            int n = sym.length;
            double[][] vectors = new double[n][n];
            for (int i = 0; i < n; i++) System.arraycopy(u[i], 0, vectors[i], 0, n);
            return new Eigen(values.clone(), vectors);
        }
        double[] vals = ed.getRealEigenvalues();
        Integer[] idx = new Integer[vals.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(vals[b], vals[a]));
        int n = sym.length;
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            values[j] = vals[idx[j]];
            double[] v = ed.getEigenvector(idx[j]).toArray();
            for (int i = 0; i < n; i++) vectors[i][j] = v[i];
        }
        return new Eigen(values, vectors);
    }

    /** Обучающие данные: векторы, метки классов, число классов, среднее. */
    record Data(double[][] x, int[] labels, int classes, double[] mean) {
        static Data of(List<List<double[]>> byClass) {
            List<double[]> xs = new ArrayList<>();
            List<Integer> ls = new ArrayList<>();
            int c = 0;
            for (List<double[]> list : byClass) {
                if (list.isEmpty()) continue;
                for (double[] v : list) {
                    xs.add(v);
                    ls.add(c);
                }
                c++;
            }
            double[][] x = xs.toArray(new double[0][]);
            double[] mean = new double[x[0].length];
            for (double[] v : x) for (int i = 0; i < mean.length; i++) mean[i] += v[i];
            for (int i = 0; i < mean.length; i++) mean[i] /= x.length;
            return new Data(x, ls.stream().mapToInt(Integer::intValue).toArray(), c, mean);
        }
    }

    /**
     * Ортонормированный базис линейной оболочки центрированных данных: строки basis (r × n), координаты
     * данных в нём y (N × r) = UΣ, сингулярные числа σ (по убыванию). Ранг r — σᵢ > relTol·σ₁, не больше maxRank и
     * min(N − 1, n).
     */
    record Span(double[][] basis, double[][] y, double[] sigma) {}

    static Span span(Data d, double relTol, int maxRank) {
        int nn = d.x().length;
        int n = d.mean().length;
        double[][] c = new double[nn][n];
        for (int k = 0; k < nn; k++) for (int i = 0; i < n; i++) c[k][i] = d.x()[k][i] - d.mean()[i];
        double[][] g = new double[nn][nn];
        IntStream.range(0, nn).parallel().forEach(a -> {
            for (int b = 0; b <= a; b++) {
                double s = 0;
                for (int i = 0; i < n; i++) s += c[a][i] * c[b][i];
                g[a][b] = s;
            }
        });
        for (int a = 0; a < nn; a++) for (int b = a + 1; b < nn; b++) g[a][b] = g[b][a];
        Eigen e = eigen(g);
        double s1 = Math.sqrt(Math.max(e.values()[0], 0));
        // Ранг центрированных данных не больше min(N − 1, n): сверх него — только шум округления.
        maxRank = Math.min(maxRank, Math.min(nn - 1, n));
        int r = 0;
        while (r < nn && r < maxRank && Math.sqrt(Math.max(e.values()[r], 0)) > relTol * s1) r++;
        double[] sigma = new double[r];
        for (int j = 0; j < r; j++) sigma[j] = Math.sqrt(e.values()[j]);
        double[][] basis = new double[r][];
        IntStream.range(0, r).parallel().forEach(j -> {
            double[] b = new double[n];
            for (int k = 0; k < nn; k++) {
                double w = e.vectors()[k][j] / sigma[j];
                for (int i = 0; i < n; i++) b[i] += w * c[k][i];
            }
            basis[j] = b;
        });
        double[][] y = new double[nn][r];
        for (int k = 0; k < nn; k++) for (int j = 0; j < r; j++) y[k][j] = e.vectors()[k][j] * sigma[j];
        return new Span(basis, y, sigma);
    }

    /** Средние классов в координатах y (данные центрированы общим средним). */
    static double[][] classMeans(double[][] y, int[] labels, int classes) {
        int r = y[0].length;
        double[][] m = new double[classes][r];
        int[] cnt = new int[classes];
        for (int k = 0; k < y.length; k++) {
            cnt[labels[k]]++;
            for (int j = 0; j < r; j++) m[labels[k]][j] += y[k][j];
        }
        for (int c = 0; c < classes; c++) for (int j = 0; j < r; j++) m[c][j] /= cnt[c];
        return m;
    }

    /** S_w = Σ_c Σ_{x∈c} (y − μ_c)(y − μ_c)ᵀ. */
    static double[][] scatterWithin(double[][] y, int[] labels, double[][] means) {
        int r = y[0].length;
        double[][] s = new double[r][r];
        IntStream.range(0, r).parallel().forEach(a -> {
            for (int k = 0; k < y.length; k++) {
                double da = y[k][a] - means[labels[k]][a];
                if (da == 0) continue;
                double[] row = s[a];
                double[] yk = y[k];
                double[] mk = means[labels[k]];
                for (int b = 0; b < r; b++) row[b] += da * (yk[b] - mk[b]);
            }
        });
        symmetrize(s);
        return s;
    }

    /** S_b = Σ_c N_c (μ_c − μ)(μ_c − μ)ᵀ, μ = 0 (данные центрированы). */
    static double[][] scatterBetween(double[][] means, int[] labels) {
        int r = means[0].length;
        int[] cnt = new int[means.length];
        for (int l : labels) cnt[l]++;
        double[][] s = new double[r][r];
        for (int c = 0; c < means.length; c++) {
            for (int a = 0; a < r; a++) {
                double w = cnt[c] * means[c][a];
                for (int b = 0; b < r; b++) s[a][b] += w * means[c][b];
            }
        }
        symmetrize(s);
        return s;
    }

    private static void symmetrize(double[][] s) {
        for (int a = 0; a < s.length; a++) {
            for (int b = a + 1; b < s.length; b++) {
                double v = (s[a][b] + s[b][a]) / 2;
                s[a][b] = v;
                s[b][a] = v;
            }
        }
    }

    /** Оси Фишера A (r × dims) с AᵀS_wA = I; число заменённых собственных значений S_w (MLDA). */
    record Axes(double[][] a, double[] swValues, int replaced, double floor) {}

    /**
     * Критерий Фишера: S_w = VΛVᵀ; при floor > 0 (MLDA) λⱼ → max(λⱼ, floor); W = VΛ^−½; разложение WᵀS_bW;
     * первые dims собственных векторов E; A = WE. S_b = Σ_c N_c μ_c μ_cᵀ = MMᵀ (M — столбцы √N_c·μ_c), поэтому
     * WᵀS_bW = BBᵀ, B = WᵀM (r × C): разлагается малая матрица BᵀB (C × C), e = Bv/√λ.
     *
     * @param means  средние классов (данные центрированы общим средним)
     * @param labels метки обучающих векторов (для N_c)
     */
    static Axes fisherAxes(double[][] sw, double[][] means, int[] labels, int dims, double floor) {
        int r = sw.length;
        int c = means.length;
        int[] cnt = new int[c];
        for (int l : labels) cnt[l]++;
        Eigen es = eigen(sw);
        double[] lam = es.values().clone();
        int replaced = 0;
        if (floor > 0) {
            for (int j = 0; j < r; j++) {
                if (lam[j] < floor) {
                    lam[j] = floor;
                    replaced++;
                }
            }
        }
        double[][] w = new double[r][r];
        for (int i = 0; i < r; i++) for (int j = 0; j < r; j++) w[i][j] = es.vectors()[i][j] / Math.sqrt(lam[j]);
        double[][] m = new double[r][c];
        for (int k = 0; k < c; k++) for (int i = 0; i < r; i++) m[i][k] = Math.sqrt(cnt[k]) * means[k][i];
        double[][] b = multiply(transpose(w), m);
        double[][] k2 = multiply(transpose(b), b);
        symmetrize(k2);
        Eigen ek = eigen(k2);
        int d = Math.min(dims, Math.min(r, c));
        double[][] e = new double[r][d];
        for (int j = 0; j < d; j++) {
            double s = Math.sqrt(Math.max(ek.values()[j], Double.MIN_NORMAL));
            for (int i = 0; i < r; i++) {
                double v = 0;
                for (int t = 0; t < c; t++) v += b[i][t] * ek.vectors()[t][j];
                e[i][j] = v / s;
            }
        }
        return new Axes(multiply(w, e), es.values(), replaced, floor);
    }

    /** То же через полную матрицу S_b (r × r) — для проверки. */
    static Axes fisherAxesFull(double[][] sw, double[][] sb, int dims, double floor) {
        int r = sw.length;
        Eigen es = eigen(sw);
        double[] lam = es.values().clone();
        int replaced = 0;
        if (floor > 0) {
            for (int j = 0; j < r; j++) {
                if (lam[j] < floor) {
                    lam[j] = floor;
                    replaced++;
                }
            }
        }
        double[][] w = new double[r][r];
        for (int i = 0; i < r; i++) for (int j = 0; j < r; j++) w[i][j] = es.vectors()[i][j] / Math.sqrt(lam[j]);
        double[][] m = multiply(transpose(w), multiply(sb, w));
        symmetrize(m);
        Eigen em = eigen(m);
        int d = Math.min(dims, r);
        double[][] e = new double[r][d];
        for (int i = 0; i < r; i++) for (int j = 0; j < d; j++) e[i][j] = em.vectors()[i][j];
        return new Axes(multiply(w, e), es.values(), replaced, floor);
    }

    /** Проекция P (dims × n) = Aᵀ · basis. */
    static double[][] projection(double[][] a, double[][] basis) {
        int d = a[0].length;
        int r = basis.length;
        int n = basis[0].length;
        double[][] p = new double[d][n];
        IntStream.range(0, d).parallel().forEach(j -> {
            for (int k = 0; k < r; k++) {
                double w = a[k][j];
                if (w == 0) continue;
                for (int i = 0; i < n; i++) p[j][i] += w * basis[k][i];
            }
        });
        return p;
    }

    /** y = P (x − μ). */
    static double[] project(double[][] p, double[] x, double[] mean) {
        double[] y = new double[p.length];
        for (int j = 0; j < p.length; j++) {
            double s = 0;
            double[] row = p[j];
            for (int i = 0; i < x.length; i++) s += row[i] * (x[i] - mean[i]);
            y[j] = s;
        }
        return y;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int k = b.length;
        int m = b[0].length;
        double[][] c = new double[n][m];
        IntStream.range(0, n).parallel().forEach(i -> {
            for (int t = 0; t < k; t++) {
                double v = a[i][t];
                if (v == 0) continue;
                for (int j = 0; j < m; j++) c[i][j] += v * b[t][j];
            }
        });
        return c;
    }

    static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) t[j][i] = a[i][j];
        return t;
    }

    /**
     * Наибольший синус главного угла между пространствами строк P₁ и P₂ (одинаковой размерности):
     * ‖(I − Q₁Q₁ᵀ)Q₂‖₂, Q — ортонормированные базисы строк.
     */
    static double maxPrincipalSin(double[][] p1, double[][] p2) {
        double[][] q1 = orthonormalRows(p1);
        double[][] q2 = orthonormalRows(p2);
        double[][] c = multiply(q2, transpose(q1));
        double[][] res = new double[q2.length][q2[0].length];
        for (int i = 0; i < q2.length; i++) {
            for (int j = 0; j < q2[0].length; j++) {
                double s = q2[i][j];
                for (int k = 0; k < q1.length; k++) s -= c[i][k] * q1[k][j];
                res[i][j] = s;
            }
        }
        return new SingularValueDecomposition(new Array2DRowRealMatrix(res, false)).getNorm();
    }

    /** Ортонормированный базис пространства строк (через SVD). */
    static double[][] orthonormalRows(double[][] p) {
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(transpose(p), false));
        double[][] u = svd.getU().getData();
        int d = p.length;
        double[][] q = new double[d][u.length];
        for (int j = 0; j < d; j++) for (int i = 0; i < u.length; i++) q[j][i] = u[i][j];
        return q;
    }
}
