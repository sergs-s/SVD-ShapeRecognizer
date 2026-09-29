package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Проверка нормализаций освещения: размер вектора, отсутствие NaN на постоянном и случайном кадре, диапазон.
 *
 * @author ssv
 */
public class IlluminationNormTest {

    private static final IlluminationNorm.NormParams P = new IlluminationNorm.NormParams(2.0, 8, 0.2, 1.0, 2.0, 0.1, 10.0);

    @BeforeClass
    public static void load() {
        nu.pattern.OpenCV.loadLocally();
    }

    private static void check(Mat frame) {
        for (IlluminationNorm n : IlluminationNorm.values()) {
            double[] v = n.vector(frame, P);
            assertEquals(92 * 112, v.length);
            for (double x : v) {
                assertTrue(n + ": " + x, Double.isFinite(x));
                if (n == IlluminationNorm.TAN_TRIGGS) assertTrue(Math.abs(x) <= P.tau());
                else assertTrue(x >= 0 && x <= 1);
            }
        }
    }

    @Test
    public void constantFrame() {
        Mat m = new Mat(112, 92, CvType.CV_8UC1, new Scalar(120));
        check(m);
        // Постоянный кадр: DoG нулевой — Tan – Triggs даёт нули.
        for (double x : IlluminationNorm.TAN_TRIGGS.vector(m, P)) assertEquals(0.0, x, 1e-9);
    }

    @Test
    public void randomFrame() {
        Mat m = new Mat(112, 92, CvType.CV_8UC1);
        Core.randu(m, 0, 256);
        check(m);
    }

    @Test
    public void contrastIsScaleInvariant() {
        double[] a = {1, -2, 3, -4, 0.5};
        double[] b = {10, -20, 30, -40, 5};
        double[] ra = IlluminationNorm.tanTriggsContrast(a.clone(), 0.1, 10);
        double[] rb = IlluminationNorm.tanTriggsContrast(b.clone(), 0.1, 10);
        for (int i = 0; i < a.length; i++) assertEquals(ra[i], rb[i], 1e-9);
    }
}
