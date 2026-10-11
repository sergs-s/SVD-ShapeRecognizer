package svd.recognizer.faces;

import java.util.Arrays;
import org.opencv.core.Mat;

/**
 * Гистограммы LTP (local ternary patterns; Tan, Triggs, IEEE TIP, 2010, разд. III.B, формула (4), рис. 8; п. 4.2, «LTP –
 * устойчивость LBP к шуму»). Сосед u при центре c: +1 при u ≥ c + t, −1 при u ≤ c − t, иначе 0; тернарный код делится на
 * верхний двоичный (бит 1 — сосед ≥ c + t) и нижний (бит 1 — сосед ≤ c − t), каждый — отдельный канал LBP. Соседи, порядок
 * битов, метки u2 (59), пиксели с полной окрестностью и сетка 7×7 — как у LbpFeature; гистограмма блока нормирована на
 * сумму 1 в каждой половине отдельно; вектор — блоки верхней половины по строкам блоков, затем блоки нижней:
 * 2·7·7·59 = 5782. t — на шкале входа (решение Хозяина 11.10.2026: t = 0,1 на выходе Tan – Triggs, в (−τ; τ)).
 *
 * @author ssv
 */
final class LtpFeature {

    static final int LENGTH = 2 * LbpFeature.LENGTH;
    /** Смещения соседей (x, y) — как у LbpFeature (округлены до 1e-9). */
    private static final double[][] OFFSETS = offsets();

    private LtpFeature() {
    }

    private static double[][] offsets() {
        double[][] o = new double[LbpFeature.NEIGHBORS][2];
        for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
            double a = 2 * Math.PI * p / LbpFeature.NEIGHBORS;
            o[p][0] = Math.round(LbpFeature.RADIUS * Math.cos(a) * 1e9) / 1e9;
            o[p][1] = Math.round(-LbpFeature.RADIUS * Math.sin(a) * 1e9) / 1e9;
        }
        return o;
    }

    /** Коды пикселей с полной окрестностью: {верхние, нижние}, каждый [(h − 4)·(w − 4)] по строкам. */
    static int[][] codes(double[] img, int w, int h, double t) {
        int r = (int) LbpFeature.RADIUS;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        int[] up = new int[cw * ch];
        int[] lo = new int[cw * ch];
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double c = img[(y + r) * w + x + r];
                int cu = 0;
                int cl = 0;
                for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
                    double u = LbpFeature.sample(img, w, x + r + OFFSETS[p][0], y + r + OFFSETS[p][1]);
                    if (u >= c + t) cu |= 1 << p;
                    if (u <= c - t) cl |= 1 << p;
                }
                up[y * cw + x] = cu;
                lo[y * cw + x] = cl;
            }
        }
        return new int[][] {up, lo};
    }

    /** Вектор гистограмм (5782): [верхняя половина | нижняя], блоки 7×7, каждая гистограмма блока — на сумму 1. */
    static double[] histogram(double[] img, int w, int h, double t) {
        int r = (int) LbpFeature.RADIUS;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        int[][] codes = codes(img, w, h, t);
        double[] v = new double[LENGTH];
        for (int half = 0; half < 2; half++) {
            int[] cs = codes[half];
            for (int by = 0; by < LbpFeature.GRID; by++) {
                int y0 = by * ch / LbpFeature.GRID;
                int y1 = (by + 1) * ch / LbpFeature.GRID;
                for (int bx = 0; bx < LbpFeature.GRID; bx++) {
                    int x0 = bx * cw / LbpFeature.GRID;
                    int x1 = (bx + 1) * cw / LbpFeature.GRID;
                    int off = half * LbpFeature.LENGTH + (by * LbpFeature.GRID + bx) * LbpFeature.BINS;
                    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) v[off + LbpFeature.LABEL[cs[y * cw + x]]]++;
                    double n = (double) (y1 - y0) * (x1 - x0);
                    for (int k = 0; k < LbpFeature.BINS; k++) v[off + k] /= n;
                }
            }
        }
        return v;
    }

    /** Вектор гистограмм кадра (8 бит, один канал) после Tan – Triggs (вещественный выход, без квантования). */
    static double[] histogramTt(Mat gray8, IlluminationNorm.NormParams p, double t) {
        return histogram(IlluminationNorm.TAN_TRIGGS.vector(gray8, p), gray8.cols(), gray8.rows(), t);
    }

    /** Блоки маски в каждой половине (как LbpFeature.select), сцеплены: [верх маски | низ маски]. */
    static double[] select(double[] v, LbpFeature.Mask mask) {
        if (v.length != LENGTH) throw new IllegalArgumentException("LTP: длина вектора " + v.length + " вместо " + LENGTH);
        double[] a = LbpFeature.select(Arrays.copyOfRange(v, 0, LbpFeature.LENGTH), mask);
        double[] b = LbpFeature.select(Arrays.copyOfRange(v, LbpFeature.LENGTH, LENGTH), mask);
        double[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * |сосед − центр| всех пикселей с полной окрестностью и всех 8 соседей (билинейно, как в codes) — проверка шкалы t:
     * [(h − 4)·(w − 4)·8].
     */
    static double[] absDiffs(double[] img, int w, int h) {
        int r = (int) LbpFeature.RADIUS;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        double[] out = new double[cw * ch * LbpFeature.NEIGHBORS];
        int i = 0;
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double c = img[(y + r) * w + x + r];
                for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
                    out[i++] = Math.abs(LbpFeature.sample(img, w, x + r + OFFSETS[p][0], y + r + OFFSETS[p][1]) - c);
                }
            }
        }
        return out;
    }
}
