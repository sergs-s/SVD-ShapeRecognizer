package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Многомасштабный LBP (MsLbpFeature, пачка 2) и Тан – Триггс на кадре k = 2: а) длина 23 600, размеры масштабов, множитель
 * сцепления; б) ровный кадр – у каждой ячейки одна метка (код 255); в) квадрат нормы каждой ячейки 1; г) сверка окна масштаба 0
 * с прямой реализацией (повтор края, LBP (8, 2)) для точек у края кадра; д) точки FrameK2.mirrorPts – отражение шаблона
 * переходит в шаблон; е) Тан – Триггс на кадре k = 2 с σ × 2, уменьшенный INTER_AREA до 92×112, близок к Тан – Триггсу на
 * кадре 92×112 (уменьшенном INTER_AREA из того же гладкого кадра): корреляция ≥ 0,95 (порог – решение Васи; замер на гладком случайном кадре – 0,9996).
 *
 * @author ssv
 */
public class MsLbpFeatureTest {

    @BeforeClass
    public static void opencv() {
        svd.recognizer.OpenCvLoader.load();
    }

    static double[] random(long seed, int n) {
        Random r = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = r.nextInt(256);
        return v;
    }

    @Test
    public void sizes() {
        assertEquals(23600, MsLbpFeature.LENGTH);
        int[] w = {184, 130, 92, 65, 46};
        int[] h = {224, 158, 112, 79, 56};
        for (int s = 0; s < 5; s++) {
            assertEquals(w[s], MsLbpFeature.size(184, s));
            assertEquals(h[s], MsLbpFeature.size(224, s));
        }
        assertEquals(Math.sqrt(98.0 / 400), MsLbpFeature.joinScale(7), 1e-15);
        assertEquals(Math.sqrt(162.0 / 400), MsLbpFeature.joinScale(9), 1e-15);
    }

    static double[][] pts() {
        return new double[][] {{60, 90}, {124, 90}, {92, 130}, {1, 2}, {183, 223}};
    }

    @Test
    public void flatAndNorms() {
        double[] flat = new double[184 * 224];
        // Нули: билинейная интерполяция точна (у ненулевого ровного кадра округление даёт сосед чуть меньше центра).
        double[] v = MsLbpFeature.feature(flat, 184, 224, pts());
        int lab = LbpFeature.LABEL[255];
        for (int c = 0; c < v.length / LbpFeature.BINS; c++) for (int k = 0; k < LbpFeature.BINS; k++) assertEquals(k == lab ? 1 : 0, v[c * LbpFeature.BINS + k], 0);
        double[] r = MsLbpFeature.feature(random(4, 184 * 224), 184, 224, pts());
        assertEquals(MsLbpFeature.LENGTH, r.length);
        for (int c = 0; c < r.length / LbpFeature.BINS; c++) {
            double s = 0;
            for (int k = 0; k < LbpFeature.BINS; k++) s += r[c * LbpFeature.BINS + k] * r[c * LbpFeature.BINS + k];
            assertEquals(1.0, s, 1e-12);
        }
    }

    /** Прямая реализация масштаба 0: окно 40×40 вокруг round(p), коды LBP (8, 2) с повтором края, ячейки 10×10, корень. */
    static double[] naive(double[] img, int w, int h, double[] p) {
        int cx = (int) Math.round(p[0]);
        int cy = (int) Math.round(p[1]);
        double[] v = new double[16 * 59];
        for (int y = 0; y < 40; y++) {
            for (int x = 0; x < 40; x++) {
                int px = cx - 20 + x;
                int py = cy - 20 + y;
                double c = at(img, w, h, px, py);
                int code = 0;
                for (int q = 0; q < 8; q++) {
                    double a = 2 * Math.PI * q / 8;
                    double ox = Math.round(2 * Math.cos(a) * 1e9) / 1e9;
                    double oy = Math.round(-2 * Math.sin(a) * 1e9) / 1e9;
                    double sx = px + ox;
                    double sy = py + oy;
                    int x0 = (int) Math.floor(sx);
                    int y0 = (int) Math.floor(sy);
                    double fx = sx - x0;
                    double fy = sy - y0;
                    double s = (1 - fx) * (1 - fy) * at(img, w, h, x0, y0) + fx * (1 - fy) * at(img, w, h, x0 + 1, y0) + (1 - fx) * fy * at(img, w, h, x0, y0 + 1)
                            + fx * fy * at(img, w, h, x0 + 1, y0 + 1);
                    if (s >= c) code |= 1 << q;
                }
                v[((y / 10) * 4 + x / 10) * 59 + LbpFeature.LABEL[code]]++;
            }
        }
        for (int i = 0; i < v.length; i++) v[i] = Math.sqrt(v[i] / 100);
        return v;
    }

    static double at(double[] img, int w, int h, int x, int y) {
        return img[Math.max(0, Math.min(h - 1, y)) * w + Math.max(0, Math.min(w - 1, x))];
    }

    @Test
    public void againstNaive() {
        double[] img = random(8, 184 * 224);
        double[][] p = pts();
        double[] v = MsLbpFeature.feature(img, 184, 224, p);
        int per = 5 * 16 * 59;
        for (int i = 0; i < p.length; i++) {
            double[] n = naive(img, 184, 224, p[i]);
            double[] got = java.util.Arrays.copyOfRange(v, i * per, i * per + 16 * 59);
            assertArrayEquals(n, got, 1e-12);
        }
    }

    @Test
    public void mirrorPoints() {
        double[][] t = FrameK2.template();
        double[][] m = FrameK2.mirrorPts(FaceMirror.mirroredTemplate(t, FrameK2.W));
        // Отражённый шаблон в порядке шаблона, отражённый ещё раз с перестановкой и поправкой, ложится на шаблон по наименьшим квадратам.
        double err = 0;
        for (int i = 0; i < 5; i++) err += Math.hypot(m[i][0] - t[i][0], m[i][1] - t[i][1]);
        assertTrue("сумма расхождений " + err, err / 5 < 2.5);
    }

    static double corr(double[] a, double[] b) {
        double ma = 0, mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i];
            mb += b[i];
        }
        ma /= a.length;
        mb /= b.length;
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(saa * sbb);
    }

    @Test
    public void tanTriggsK2() {
        // Гладкий «кадр» 184×224: шум, размытый гауссианом σ = 3, приведённый к 0…255.
        Mat n = new Mat(224, 184, CvType.CV_64F);
        n.put(0, 0, random(21, 184 * 224));
        Mat b = new Mat();
        Imgproc.GaussianBlur(n, b, new Size(0, 0), 3);
        Mat c8 = new Mat();
        org.opencv.core.Core.normalize(b, c8, 0, 255, org.opencv.core.Core.NORM_MINMAX, CvType.CV_8U);
        Mat a8 = new Mat();
        Imgproc.resize(c8, a8, new Size(92, 112), 0, 0, Imgproc.INTER_AREA);
        IlluminationNorm.NormParams p = new IlluminationNorm.NormParams(2, 8, 0.2, 1, 2, 0.1, 10);
        double[] ta = PoemFeature.input(a8, IlluminationNorm.TAN_TRIGGS, p);
        double[] tc = PoemFeature.input(c8, IlluminationNorm.TAN_TRIGGS, XBaseFeat2.scaledTt(p, 2));
        Mat m = new Mat(224, 184, CvType.CV_64F);
        m.put(0, 0, tc);
        Mat d = new Mat();
        Imgproc.resize(m, d, new Size(92, 112), 0, 0, Imgproc.INTER_AREA);
        double[] td = new double[92 * 112];
        d.get(0, 0, td);
        double r = corr(ta, td);
        System.out.printf("Тан – Триггс k = 2 против 92×112: корреляция %.4f%n", r);
        assertTrue("корреляция " + r, r >= 0.95);
    }
}
