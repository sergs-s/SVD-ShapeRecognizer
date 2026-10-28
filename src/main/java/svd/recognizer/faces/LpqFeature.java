package svd.recognizer.faces;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * Гистограммы LPQ (локальное квантование фазы; Ojansivu, Heikkilä, ICISP 2008; для лиц – Ahonen, Rahtu, Ojansivu, Heikkilä,
 * «Recognition of blurred faces using local phase quantization», ICPR 2008; пачка 2, задание Жорика №2 от 28.10.2026). Параметры
 * – по статье Ahonen 2008: окно M×M, M = 7, a = 1/M, ρ = 0,9. Кратковременное преобразование Фурье F(u, x) = Σ_y f(x − y)
 * e^(−j2π uᵀy), y ∈ [−3, 3]², в четырёх частотах u1 = [a, 0], u2 = [0, a], u3 = [a, a], u4 = [a, −a] (первая координата – по
 * столбцам кадра); G = [Re F1…F4, Im F1…F4] = W f_x. Декорреляция: модель Маркова первого порядка, ковариация отсчётов окна
 * σ_ij = ρ^‖y_i − y_j‖, D = W C Wᵀ = U Σ Vᵀ (SVD), E = Vᵀ G; знак столбца V – так, чтобы наибольший по модулю элемент был
 * положителен (SVD определяет его с точностью до знака; смена знака лишь переставляет коды одинаково у всех кадров). Код –
 * Σ_j [E_j ≥ 0] · 2^(j−1), 256 меток. Коды – у пикселей с окном целиком в кадре (край (M − 1)/2 = 3; в статье край не оговорён,
 * как у LbpFeature); сетка блоков по кадру кодов – как у LBP (границы ⌊i·cw/G⌋), гистограмма блока нормирована на сумму 1,
 * блоки по строкам; длина G²·256.
 *
 * @author ssv
 */
final class LpqFeature {

    static final int M = 7;
    static final int R = M / 2;
    static final double A = 1.0 / M;
    static final double RHO = 0.9;
    static final int BITS = 8;
    static final int BINS = 1 << BITS;
    /** Частоты [u_x, u_y]. */
    static final double[][] FREQ = {{A, 0}, {0, A}, {A, A}, {A, -A}};
    /** W (8 × M²): строки Re w_u1…u4, Im w_u1…u4; столбец k – смещение y = (k % M − R, k / M − R). */
    static final double[][] W = w();
    /** Vᵀ (8 × 8) декорреляции. */
    static final double[][] VT = decorrelation();

    private LpqFeature() {
    }

    static int length(int grid) {
        return grid * grid * BINS;
    }

    private static double[][] w() {
        double[][] w = new double[2 * FREQ.length][M * M];
        for (int k = 0; k < M * M; k++) {
            int yx = k % M - R;
            int yy = k / M - R;
            for (int f = 0; f < FREQ.length; f++) {
                double ph = -2 * Math.PI * (FREQ[f][0] * yx + FREQ[f][1] * yy);
                w[f][k] = Math.cos(ph);
                w[FREQ.length + f][k] = Math.sin(ph);
            }
        }
        return w;
    }

    /** Ковариация отсчётов окна C (M² × M²): ρ^‖y_i − y_j‖. */
    static double[][] covariance() {
        int n = M * M;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dx = i % M - j % M;
                double dy = i / M - j / M;
                c[i][j] = Math.pow(RHO, Math.sqrt(dx * dx + dy * dy));
            }
        }
        return c;
    }

    private static double[][] decorrelation() {
        RealMatrix wm = new Array2DRowRealMatrix(W);
        RealMatrix d = wm.multiply(new Array2DRowRealMatrix(covariance())).multiply(wm.transpose());
        RealMatrix v = new SingularValueDecomposition(d).getV();
        double[][] vt = new double[v.getColumnDimension()][v.getRowDimension()];
        for (int c = 0; c < vt.length; c++) {
            int big = 0;
            for (int r = 0; r < v.getRowDimension(); r++) if (Math.abs(v.getEntry(r, c)) > Math.abs(v.getEntry(big, c))) big = r;
            double sg = v.getEntry(big, c) < 0 ? -1 : 1;
            for (int r = 0; r < v.getRowDimension(); r++) vt[c][r] = sg * v.getEntry(r, c);
        }
        return vt;
    }

    /** G = W f_x пикселя (x, y): f_x[k] = f(x − y_k). */
    static double[] g(double[] img, int w, int x, int y) {
        double[] g = new double[W.length];
        for (int k = 0; k < M * M; k++) {
            double f = img[(y - (k / M - R)) * w + x - (k % M - R)];
            for (int j = 0; j < g.length; j++) g[j] += W[j][k] * f;
        }
        return g;
    }

    /** Коды пикселей с окном целиком в кадре: [(h − 2R)·(w − 2R)] по строкам; decorr = false – без декорреляции (для проверки). */
    static int[] codes(double[] img, int w, int h, boolean decorr) {
        int cw = w - 2 * R;
        int ch = h - 2 * R;
        if (cw <= 0 || ch <= 0) throw new IllegalArgumentException("LPQ: кадр " + w + "×" + h + " меньше окна");
        int[] out = new int[cw * ch];
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double[] g = g(img, w, x + R, y + R);
                int code = 0;
                for (int j = 0; j < BITS; j++) {
                    double e = 0;
                    if (decorr) for (int i = 0; i < BITS; i++) e += VT[j][i] * g[i];
                    else e = g[j];
                    if (e >= 0) code |= 1 << j;
                }
                out[y * cw + x] = code;
            }
        }
        return out;
    }

    static int[] codes(double[] img, int w, int h) {
        return codes(img, w, h, true);
    }

    /** Гистограммы меток codes (кадр кодов cw × ch) по сетке grid×grid, каждая на сумму 1; по строкам блоков. */
    static double[] blocks(int[] codes, int cw, int ch, int grid, int bins) {
        double[] v = new double[grid * grid * bins];
        for (int by = 0; by < grid; by++) {
            int y0 = by * ch / grid;
            int y1 = (by + 1) * ch / grid;
            for (int bx = 0; bx < grid; bx++) {
                int x0 = bx * cw / grid;
                int x1 = (bx + 1) * cw / grid;
                int off = (by * grid + bx) * bins;
                for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) v[off + codes[y * cw + x]]++;
                double n = (double) (y1 - y0) * (x1 - x0);
                for (int k = 0; k < bins; k++) v[off + k] /= n;
            }
        }
        return v;
    }

    /** Вектор гистограмм (length(grid)) по кадру img (по строкам, ширина w). */
    static double[] histogram(double[] img, int w, int h, int grid) {
        return blocks(codes(img, w, h), w - 2 * R, h - 2 * R, grid, BINS);
    }
}
