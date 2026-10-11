package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.List;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Качество кадров GT (PLAN.md п. 4.2): σ шума по Immerkær, дисперсия лапласиана и она же на дисперсию яркости, межглазье,
 * квартили и доли ошибок.
 *
 * @author ssv
 */
public class XBaseQualityTest {

    @BeforeClass
    public static void load() {
        nu.pattern.OpenCV.loadLocally();
    }

    static Mat mat64(double[] v, int w, int h) {
        Mat m = new Mat(h, w, CvType.CV_64F);
        m.put(0, 0, v);
        return m;
    }

    @Test
    public void immerkaerNoise() {
        int w = 200;
        int h = 200;
        double[] flat = new double[w * h];
        java.util.Arrays.fill(flat, 100);
        assertEquals(0, XBaseQuality.noiseSigma(mat64(flat, w, h)), 1e-12);
        double[] grad = new double[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) grad[y * w + x] = 0.3 * x + 0.7 * y;
        assertEquals(0, XBaseQuality.noiseSigma(mat64(grad, w, h)), 1e-9);
        Random rnd = new Random(1);
        double[] noise = new double[w * h];
        for (int i = 0; i < noise.length; i++) noise[i] = 100 + 5 * rnd.nextGaussian();
        assertEquals(5, XBaseQuality.noiseSigma(mat64(noise, w, h)), 0.25);
    }

    @Test
    public void laplacianMeasures() {
        // Кадр 8 бит: одна яркая точка 100 на нуле — лапласиан: −400 в точке, 100 у 4 соседей.
        int w = 10;
        int h = 10;
        byte[] b = new byte[w * h];
        b[5 * w + 5] = 100;
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        m.put(0, 0, b);
        double[] r = XBaseQuality.frameMeasures(m);
        double n = w * h;
        double mean = (-400 + 4 * 100) / n;
        double lapVar = (400.0 * 400 + 4 * 100 * 100) / n - mean * mean;
        double mb = 100 / n;
        double var = 100.0 * 100 / n - mb * mb;
        assertEquals(lapVar, r[0], 1e-9);
        assertEquals(lapVar / var, r[1], 1e-9);
        assertEquals(mb, r[3], 1e-12);
        assertEquals(Math.sqrt(var), r[4], 1e-12);
        m.release();
    }

    @Test
    public void eyeDistanceAndQuartiles() {
        double[] row = new double[15];
        row[4] = 10;
        row[5] = 20;
        row[6] = 13;
        row[7] = 24;
        assertEquals(5, XBaseQuality.eyeDistance(row), 1e-12);
        double[] q = {1, 2, 3};
        assertArrayEquals(new int[] {0, 0, 1, 2, 3}, new int[] {XBaseQuality.quartile(0.5, q), XBaseQuality.quartile(1, q), XBaseQuality.quartile(1.5, q),
            XBaseQuality.quartile(3, q), XBaseQuality.quartile(3.5, q)});
        // Пробы: {человек, r, верный кандидат}; порог 0,5. В Q1: верно и принят; верно и отказ (ii); неверный кандидат.
        List<double[]> probe = List.of(new double[] {0, 0.4, 1}, new double[] {1, 0.6, 1}, new double[] {2, 0.3, 0}, new double[] {0, 0.1, 1});
        double[] diff = {0, 0.5, 1, 4};
        double[] s = XBaseQuality.shares(probe, diff, q, 0, new int[] {1, 1, 1}, 0.5);
        assertEquals(100 * 2 / 3.0, s[0], 1e-12);
        assertEquals(100 / 3.0, s[1], 1e-12);
        double[] s3 = XBaseQuality.shares(probe, diff, q, 3, new int[] {1, 1, 1}, 0.5);
        assertEquals(0, s3[0], 0);
        assertTrue(Double.isNaN(XBaseQuality.shares(probe, diff, q, 1, new int[] {1, 1, 1}, 0.5)[0]));
    }
}
