package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.junit.Test;

/**
 * POEM (PoemFeature, постановка 2.4): а) ровный кадр – все коды 0; б) длина и сумма 1 по каждому (блоку, ориентации); в)
 * интервалы ориентации и края; г) порог τ; д) сверка с независимой прямой реализацией на случайных кадрах; е) детерминированность;
 * ж) сцепление (П2) и множитель 1/√3.
 *
 * @author ssv
 */
public class PoemFeatureTest {

    static final int W = 92;
    static final int H = 112;

    static double[] random(long seed) {
        Random r = new Random(seed);
        double[] img = new double[W * H];
        for (int i = 0; i < img.length; i++) img[i] = r.nextInt(256);
        return img;
    }

    @Test
    public void flat() {
        double[] img = new double[W * H];
        java.util.Arrays.fill(img, 117);
        int[][] c = PoemFeature.codes(img, W, H);
        assertEquals(PoemFeature.ORIENT, c.length);
        for (int[] co : c) {
            assertEquals((W - 18) * (H - 18), co.length);
            for (int v : co) assertEquals(0, v);
        }
        double[] h = PoemFeature.histogram(img, W, H, 7);
        for (int seg = 0; seg < h.length / LbpFeature.BINS; seg++) assertEquals(1.0, h[seg * LbpFeature.BINS + LbpFeature.LABEL[0]], 0);
    }

    @Test
    public void lengthAndSums() {
        assertEquals(8673, PoemFeature.length(7));
        assertEquals(14337, PoemFeature.length(9));
        assertEquals(17700, PoemFeature.length(10));
        assertEquals(9, PoemFeature.BORDER);
        for (int g : new int[] {7, 9, 10}) {
            double[] h = PoemFeature.histogram(random(3), W, H, g);
            assertEquals(PoemFeature.length(g), h.length);
            for (int seg = 0; seg < h.length / LbpFeature.BINS; seg++) {
                double s = 0;
                for (int k = 0; k < LbpFeature.BINS; k++) s += h[seg * LbpFeature.BINS + k];
                assertEquals(1.0, s, 1e-12);
            }
        }
    }

    @Test
    public void orientation() {
        assertEquals(0, PoemFeature.bin(1, 0));
        assertEquals(0, PoemFeature.bin(-1, 0));
        assertEquals(0, PoemFeature.bin(1, 1));
        assertEquals(1, PoemFeature.bin(0, 1));
        assertEquals(1, PoemFeature.bin(0, -1));
        assertEquals(2, PoemFeature.bin(-1, 1));
        assertEquals(2, PoemFeature.bin(1, -1));
        // Вертикальная граница: gx ≠ 0, gy = 0 – вся величина в интервале 0; горизонтальная – в интервале 1 (θ = π/2).
        double[] v = new double[W * H];
        double[] hz = new double[W * H];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                v[y * W + x] = x < W / 2 ? 0 : 255;
                hz[y * W + x] = y < H / 2 ? 0 : 255;
            }
        }
        double[][] mv = PoemFeature.magnitudes(v, W, H);
        double[][] mh = PoemFeature.magnitudes(hz, W, H);
        double[] sv = new double[3];
        double[] sh = new double[3];
        for (int o = 0; o < 3; o++) {
            for (double a : mv[o]) sv[o] += a;
            for (double a : mh[o]) sh[o] += a;
        }
        assertEquals(4 * 255.0 * 2 * (H - 2), sv[0], 1e-9);
        assertEquals(0, sv[1], 0);
        assertEquals(0, sv[2], 0);
        assertEquals(0, sh[0], 0);
        assertEquals(4 * 255.0 * 2 * (W - 2), sh[1], 1e-9);
        assertEquals(0, sh[2], 0);
    }

    @Test
    public void tau() {
        int n = 11;
        double[] a = new double[n * n];
        java.util.Arrays.fill(a, 10);
        a[5 * n + 10] = 10.5; // сосед p = 0 (вправо на 5) центра (5, 5)
        assertEquals(0, PoemFeature.codes(a, n, n, 1.0)[0]);
        assertEquals(1, PoemFeature.codes(a, n, n, 0.4)[0]);
        a[5 * n + 10] = 10.4; // ровно τ – не больше τ
        assertEquals(0, PoemFeature.codes(a, n, n, 10.4 - 10)[0]);
        a[0 * n + 5] = 11; // сосед p = 2 (вверх на 5)
        assertEquals(4, PoemFeature.codes(a, n, n, 0.5)[0]);
    }

    /** Прямая реализация: ядра Собеля, сумма ячейки по 49 пикселям, билинейная интерполяция по 4 точкам всегда. */
    static double[] naive(double[] img, int w, int h, int grid) {
        int[][] kx = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] ky = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        double[][] mag = new double[3][w * h];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double gx = 0;
                double gy = 0;
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        gx += kx[j + 1][i + 1] * img[(y + j) * w + x + i];
                        gy += ky[j + 1][i + 1] * img[(y + j) * w + x + i];
                    }
                }
                double t = Math.atan2(gy, gx);
                if (t < 0) t += Math.PI;
                if (t >= Math.PI) t -= Math.PI;
                int o = Math.min(2, (int) Math.floor(3 * t / Math.PI));
                mag[o][y * w + x] = Math.hypot(gx, gy);
            }
        }
        // Ячейка с центром (x, y) – пиксели x − 3…x + 3 при 1 ≤ x − 3 и x + 3 ≤ w − 2.
        double[][] acc = new double[3][w * h];
        double[] max = new double[3];
        for (int o = 0; o < 3; o++) {
            for (int y = 4; y <= h - 5; y++) {
                for (int x = 4; x <= w - 5; x++) {
                    double s = 0;
                    for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) s += mag[o][(y + j) * w + x + i];
                    acc[o][y * w + x] = s;
                    max[o] = Math.max(max[o], s);
                }
            }
        }
        int cw = w - 18;
        int ch = h - 18;
        double[] v = new double[grid * grid * 3 * 59];
        for (int by = 0; by < grid; by++) {
            for (int bx = 0; bx < grid; bx++) {
                int y0 = by * ch / grid;
                int y1 = (by + 1) * ch / grid;
                int x0 = bx * cw / grid;
                int x1 = (bx + 1) * cw / grid;
                for (int o = 0; o < 3; o++) {
                    int off = ((by * grid + bx) * 3 + o) * 59;
                    for (int y = y0; y < y1; y++) {
                        for (int x = x0; x < x1; x++) {
                            int cx = x + 9;
                            int cy = y + 9;
                            double c = acc[o][cy * w + cx];
                            int code = 0;
                            for (int p = 0; p < 8; p++) {
                                double a = 2 * Math.PI * p / 8;
                                double px = cx + Math.round(5 * Math.cos(a) * 1e9) / 1e9;
                                double py = cy + Math.round(-5 * Math.sin(a) * 1e9) / 1e9;
                                int ix = (int) Math.floor(px);
                                int iy = (int) Math.floor(py);
                                double fx = px - ix;
                                double fy = py - iy;
                                double s = (1 - fx) * (1 - fy) * acc[o][iy * w + ix] + fx * (1 - fy) * acc[o][iy * w + Math.min(ix + 1, w - 1)]
                                        + (1 - fx) * fy * acc[o][Math.min(iy + 1, h - 1) * w + ix] + fx * fy * acc[o][Math.min(iy + 1, h - 1) * w + Math.min(ix + 1, w - 1)];
                                if (s - c > 1e-6 * max[o]) code |= 1 << p;
                            }
                            v[off + LbpFeature.LABEL[code]]++;
                        }
                    }
                    for (int k = 0; k < 59; k++) v[off + k] /= (double) (y1 - y0) * (x1 - x0);
                }
            }
        }
        return v;
    }

    @Test
    public void againstNaive() {
        for (long seed = 1; seed <= 4; seed++) {
            double[] img = random(seed);
            // Гладкий кадр с плоскими участками – проверка τ на равных соседях.
            if (seed == 4) for (int i = 0; i < img.length; i++) img[i] = (i % W) < 40 ? 50 : img[i];
            for (int g : new int[] {7, 9, 10}) assertArrayEquals("seed " + seed + ", сетка " + g, naive(img, W, H, g), PoemFeature.histogram(img, W, H, g), 0);
        }
    }

    @Test
    public void variants() {
        assertArrayEquals(new int[] {2891, 8673, 11564, 17700}, XBasePoem.dims(0));
        assertArrayEquals(new int[] {4779, 14337, 19116, 17700}, XBasePoem.dims(1));
        assertEquals("(П2) LBP + POEM 9×9, POEM · 1/√3", XBasePoem.vname(1, 2));
        assertEquals("база LBP 7×7 (8, 2)", XBasePoem.vname(0, 0));
        assertEquals(0.708, XBasePoem.FERET_TO_GT, 0.001);
        // Правило выбора pool-lbp: база в пределах 1 п. п.; иначе меньшая размерность в пределах 1 п. п.
        int[] d = XBasePoem.dims(1);
        assertEquals(0, XBaseLbp.choose(new double[] {14.0, 13.5, 13.2, 13.9}, d));
        assertEquals(1, XBaseLbp.choose(new double[] {14.0, 12.5, 12.0, 12.9}, d));
        assertEquals(3, XBaseLbp.choose(new double[] {14.0, 13.5, 12.0, 12.9}, d));
    }

    @Test
    public void deterministic() {
        double[] img = random(11);
        assertArrayEquals(PoemFeature.histogram(img, W, H, 9), PoemFeature.histogram(img.clone(), W, H, 9), 0);
    }

    @Test
    public void join() {
        double[] img = random(5);
        double[] l = LbpFeature.sqrt(LbpFeature.histogram(img, W, H, new LbpFeature.Params(8, new double[] {2}, 9)));
        double[] p = LbpFeature.sqrt(PoemFeature.histogram(img, W, H, 9));
        double[] j = PoemFeature.join(l, p);
        assertEquals(4779 + 14337, j.length);
        assertEquals(1 / Math.sqrt(3), PoemFeature.JOIN_SCALE, 1e-15);
        for (int i = 0; i < l.length; i++) assertEquals(l[i], j[i], 0);
        for (int i = 0; i < p.length; i++) assertEquals(p[i] * PoemFeature.JOIN_SCALE, j[l.length + i], 0);
        // Квадрат нормы на блок: LBP – 1 (одна гистограмма с суммой 1 под корнем), POEM – 3 · 1/3 = 1.
        for (int b = 0; b < 81; b++) {
            double nl = 0;
            double np = 0;
            for (int k = 0; k < 59; k++) nl += j[b * 59 + k] * j[b * 59 + k];
            for (int k = 0; k < 3 * 59; k++) np += j[l.length + b * 177 + k] * j[l.length + b * 177 + k];
            assertEquals(1.0, nl, 1e-12);
            assertEquals(1.0, np, 1e-12);
        }
        assertTrue(PoemFeature.JOIN_SCALE < 1);
    }
}
