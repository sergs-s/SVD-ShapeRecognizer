package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.Random;
import org.junit.Test;

/**
 * LPQ (LpqFeature, пачка 2): а) длина и сумма 1 по блокам; б) G = W f_x – сверка с независимой реализацией (сепарабельные
 * одномерные свёртки по строкам и столбцам, как в формуле (1) Ahonen 2008); в) известный ответ – наклонная плоскость по
 * столбцам; г) декорреляция: Vᵀ D V диагональна, Vᵀ ортогональна; д) детерминированность.
 *
 * @author ssv
 */
public class LpqFeatureTest {

    static final int W = 92;
    static final int H = 112;

    static double[] random(long seed) {
        Random r = new Random(seed);
        double[] img = new double[W * H];
        for (int i = 0; i < img.length; i++) img[i] = r.nextInt(256);
        return img;
    }

    @Test
    public void lengthAndSums() {
        assertEquals(12544, LpqFeature.length(7));
        assertEquals(20736, LpqFeature.length(9));
        for (int g : new int[] {7, 9}) {
            double[] h = LpqFeature.histogram(random(5), W, H, g);
            assertEquals(LpqFeature.length(g), h.length);
            for (int b = 0; b < g * g; b++) {
                double s = 0;
                for (int k = 0; k < LpqFeature.BINS; k++) s += h[b * LpqFeature.BINS + k];
                assertEquals(1.0, s, 1e-12);
            }
        }
        assertEquals((W - 6) * (H - 6), LpqFeature.codes(random(1), W, H).length);
    }

    /** F(u, x) = Σ_y f(x − y) e^(−j2π uᵀy) сепарабельно: сначала по столбцам (u_x), затем по строкам (u_y). */
    static double[] naiveG(double[] img, int w, int x, int y) {
        double[] g = new double[8];
        int r = LpqFeature.R;
        for (int f = 0; f < 4; f++) {
            double ux = LpqFeature.FREQ[f][0];
            double uy = LpqFeature.FREQ[f][1];
            double re = 0;
            double im = 0;
            for (int dy = -r; dy <= r; dy++) {
                double rowRe = 0;
                double rowIm = 0;
                for (int dx = -r; dx <= r; dx++) {
                    double v = img[(y - dy) * w + x - dx];
                    rowRe += v * Math.cos(-2 * Math.PI * ux * dx);
                    rowIm += v * Math.sin(-2 * Math.PI * ux * dx);
                }
                double c = Math.cos(-2 * Math.PI * uy * dy);
                double s = Math.sin(-2 * Math.PI * uy * dy);
                re += rowRe * c - rowIm * s;
                im += rowRe * s + rowIm * c;
            }
            g[f] = re;
            g[4 + f] = im;
        }
        return g;
    }

    @Test
    public void naive() {
        double[] img = random(11);
        for (int[] p : new int[][] {{3, 3}, {45, 60}, {W - 4, H - 4}, {10, 100}}) {
            assertArrayEquals(naiveG(img, W, p[0], p[1]), LpqFeature.g(img, W, p[0], p[1]), 1e-9);
        }
    }

    @Test
    public void ramp() {
        // f = столбец: по u1 = [a, 0] F = Σ_y (x − y_x) e^(−jθy_x) · 7 = 7j Σ y sin(θy), Re = 0; по u2 = [0, a] F = 0 (Σ e^(−jθy) = 0).
        double[] img = new double[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) img[y * W + x] = x;
        double s = 0;
        for (int k = -3; k <= 3; k++) s += k * Math.sin(2 * Math.PI * k / 7);
        double[] g = LpqFeature.g(img, W, 40, 50);
        assertEquals(0, g[0], 1e-9);
        assertEquals(7 * s, g[4], 1e-9);
        assertEquals(0, g[1], 1e-9);
        assertEquals(0, g[5], 1e-9);
    }

    @Test
    public void decorrelation() {
        double[][] w = LpqFeature.W;
        double[][] c = LpqFeature.covariance();
        double[][] d = new double[8][8];
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) for (int a = 0; a < 49; a++) for (int b = 0; b < 49; b++) d[i][j] += w[i][a] * c[a][b] * w[j][b];
        double[][] vt = LpqFeature.VT;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double e = 0;
                double o = 0;
                for (int a = 0; a < 8; a++) {
                    o += vt[i][a] * vt[j][a];
                    for (int b = 0; b < 8; b++) e += vt[i][a] * d[a][b] * vt[j][b];
                }
                assertEquals(i == j ? 1 : 0, o, 1e-9);
                if (i != j) assertEquals(0, e, 1e-6);
            }
        }
    }

    @Test
    public void deterministic() {
        assertArrayEquals(LpqFeature.histogram(random(7), W, H, 9), LpqFeature.histogram(random(7), W, H, 9), 0);
    }
}
