package svd.recognizer.faces;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;
import org.bytedeco.openblas.global.openblas;

/**
 * MLDA для (Н) дуэли (Д-№1, разрешение Хозяина 30.10 на вариант (в)): те же шаги, что MldaScorer.fit (LdaMath.span,
 * classMeans, scatterWithin, fisherAxes, projection), но матрица Грама и базис оболочки – через BLAS (dsyrk, dgemm), а
 * разложения симметричных матриц (Грама n × n и S_w r × r) – через LAPACK dsyevd вместо Commons Math. Результат совпадает
 * с MldaScorer до округления (не побайтно): знаки собственных векторов и порядок суммирования другие, расстояния 1 − cos
 * от знаков не зависят. Основной код MLDA не меняется; класс используется только режимами DuelThr для (Н).
 *
 * @author ssv
 */
final class MldaFast {

    private MldaFast() {
    }

    /** OpenBLAS из нескольких потоков Java одновременно виснет (проверено 31.10: два параллельных dsyevd) – все вызовы BLAS и LAPACK по одному. */
    private static final Object BLAS = new Object();

    /** Как new MldaScorer(c).fit(classes, gallery) (relTol 1e-10). */
    static LdaModel fit(List<List<double[]>> classes, int gallery, double c) {
        return fit(classes, gallery, c, 1e-10);
    }

    static LdaModel fit(List<List<double[]>> classes, int gallery, double c, double relTol) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int nFull = d.mean().length;
        LdaMath.Span span = span(d, relTol, Integer.MAX_VALUE);
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(span.y(), d.labels(), means);
        double trace = 0;
        for (int i = 0; i < sw.length; i++) trace += sw[i][i];
        double floor = c * (trace / nFull);
        LdaMath.Axes axes = fisherAxes(sw, means, d.labels(), d.classes() - 1, floor);
        String info = String.format(Locale.ROOT, "MldaFast (LAPACK dsyevd): N %d, C %d, n %d, ранг оболочки r %d, осей %d; λ̄ = tr(S_w)/n = %.6g; "
                + "заменено %d из %d", d.x().length, d.classes(), nFull, span.basis().length, axes.a()[0].length, floor, axes.replaced(),
                span.basis().length);
        if (c != 1.0) info = String.format(Locale.ROOT, "пол λ̄·c, c = %s; ", c) + info;
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.projection(axes.a(), span.basis()), info);
    }

    /** Разложение симметричной матрицы (LAPACK dsyevd): собственные значения по убыванию, векторы – столбцы. */
    static LdaMath.Eigen eigen(double[][] sym) {
        int n = sym.length;
        double[] a = new double[n * n];
        for (int i = 0; i < n; i++) System.arraycopy(sym[i], 0, a, i * n, n);
        double[] w = new double[n];
        int info;
        synchronized (BLAS) {
            info = openblas.LAPACKE_dsyevd(openblas.LAPACK_ROW_MAJOR, (byte) 'V', (byte) 'U', n, a, n, w);
        }
        if (info != 0) throw new IllegalStateException("MldaFast: LAPACKE_dsyevd, info = " + info + " (n = " + n + ")");
        // dsyevd: по возрастанию, столбец j – собственный вектор; переворачиваем в порядок по убыванию.
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            int s = n - 1 - j;
            values[j] = w[s];
            for (int i = 0; i < n; i++) vectors[i][j] = a[i * n + s];
        }
        return new LdaMath.Eigen(values, vectors);
    }

    /** Как LdaMath.span: матрица Грама – dsyrk, базис – dgemm, разложение – eigen (LAPACK). */
    static LdaMath.Span span(LdaMath.Data d, double relTol, int maxRank) {
        int nn = d.x().length;
        int n = d.mean().length;
        double[] c = new double[nn * n];
        IntStream.range(0, nn).parallel().forEach(k -> {
            double[] x = d.x()[k];
            for (int i = 0; i < n; i++) c[k * n + i] = x[i] - d.mean()[i];
        });
        double[] gu = new double[nn * nn];
        synchronized (BLAS) {
            openblas.cblas_dsyrk(openblas.CblasRowMajor, openblas.CblasUpper, openblas.CblasNoTrans, nn, n, 1.0, c, n, 0.0, gu, nn);
        }
        double[][] g = new double[nn][nn];
        for (int a = 0; a < nn; a++) {
            for (int b = a; b < nn; b++) {
                g[a][b] = gu[a * nn + b];
                g[b][a] = gu[a * nn + b];
            }
        }
        LdaMath.Eigen e = eigen(g);
        double s1 = Math.sqrt(Math.max(e.values()[0], 0));
        maxRank = Math.min(maxRank, Math.min(nn - 1, n));
        int r = 0;
        while (r < nn && r < maxRank && Math.sqrt(Math.max(e.values()[r], 0)) > relTol * s1) r++;
        double[] sigma = new double[r];
        for (int j = 0; j < r; j++) sigma[j] = Math.sqrt(e.values()[j]);
        // basis (r × n) = (V_r Σ⁻¹)ᵀ · C
        double[] vt = new double[r * nn];
        for (int j = 0; j < r; j++) for (int k = 0; k < nn; k++) vt[j * nn + k] = e.vectors()[k][j] / sigma[j];
        double[] bf = new double[r * n];
        synchronized (BLAS) {
            openblas.cblas_dgemm(openblas.CblasRowMajor, openblas.CblasNoTrans, openblas.CblasNoTrans, r, n, nn, 1.0, vt, nn, c, n, 0.0, bf, n);
        }
        double[][] basis = new double[r][];
        for (int j = 0; j < r; j++) basis[j] = Arrays.copyOfRange(bf, j * n, (j + 1) * n);
        double[][] y = new double[nn][r];
        for (int k = 0; k < nn; k++) for (int j = 0; j < r; j++) y[k][j] = e.vectors()[k][j] * sigma[j];
        return new LdaMath.Span(basis, y, sigma);
    }

    /** Как LdaMath.fisherAxes, разложения – eigen (LAPACK). */
    static LdaMath.Axes fisherAxes(double[][] sw, double[][] means, int[] labels, int dims, double floor) {
        int r = sw.length;
        int c = means.length;
        int[] cnt = new int[c];
        for (int l : labels) cnt[l]++;
        LdaMath.Eigen es = eigen(sw);
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
        double[][] b = LdaMath.multiply(LdaMath.transpose(w), m);
        double[][] k2 = LdaMath.multiply(LdaMath.transpose(b), b);
        for (int i = 0; i < c; i++) {
            for (int j = i + 1; j < c; j++) {
                double v = (k2[i][j] + k2[j][i]) / 2;
                k2[i][j] = v;
                k2[j][i] = v;
            }
        }
        LdaMath.Eigen ek = eigen(k2);
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
        return new LdaMath.Axes(LdaMath.multiply(w, e), es.values(), replaced, floor);
    }
}
