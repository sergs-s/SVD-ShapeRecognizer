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
        svd.recognizer.OpenCvLoader.load();
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

    /** Поправка ε (№3, 13.10.2026): ровный кадр 255 — DoG обнуляется целиком, выход ровно 0. */
    @Test
    public void flatFrameGivesZero() {
        Mat m = new Mat(112, 92, CvType.CV_8UC1, new Scalar(255));
        for (double x : IlluminationNorm.tanTriggs(m, P, true)) assertEquals(0.0, x, 0.0);
    }

    /**
     * Поправка ε: шум последних разрядов в ровном участке (DoG ≈ 1e-16 вместо 0) без поправки сдвигает нормировку контраста,
     * с поправкой — нет (результат тот же, что при точных нулях). Сдвиг проявляется, когда часть значений упирается в τ.
     */
    @Test
    public void negligibleDogDoesNotShiftContrast() {
        java.util.Random r = new java.util.Random(7);
        double[] exact = new double[1000];
        for (int i = 0; i < 600; i++) exact[i] = r.nextGaussian() * 0.05;
        for (int i = 0; i < 60; i++) exact[i] *= 40; // сильные края: после выравнивания упираются в τ (как в реальных кадрах)
        double[] noisy = exact.clone();
        for (int i = 600; i < 1000; i++) noisy[i] = (r.nextBoolean() ? 1 : -1) * 1e-16 * (1 + r.nextInt(5));
        double[] ref = IlluminationNorm.tanTriggsContrast(exact.clone(), 0.1, 10);
        double[] raw = IlluminationNorm.tanTriggsContrast(noisy.clone(), 0.1, 10);
        double shift = 0;
        for (int i = 0; i < ref.length; i++) shift = Math.max(shift, Math.abs(raw[i] - ref[i]));
        assertTrue("без поправки шум должен сдвигать выход: " + shift, shift > 1e-6);
        double[] fixed = noisy.clone();
        assertEquals(400, IlluminationNorm.zeroNegligible(fixed, IlluminationNorm.DOG_EPS * Math.pow(255, 0.2)));
        double[] out = IlluminationNorm.tanTriggsContrast(fixed, 0.1, 10);
        for (int i = 0; i < out.length; i++) assertEquals(ref[i], out[i], 0.0);
    }

    /** Поправка ε: на кадре без ровных участков результат с поправкой и без совпадает точно. */
    @Test
    public void ordinaryFrameUnchanged() {
        Mat m = new Mat(112, 92, CvType.CV_8UC1);
        Core.randu(m, 20, 230);
        org.junit.Assert.assertArrayEquals(IlluminationNorm.tanTriggs(m, P, false), IlluminationNorm.tanTriggs(m, P, true), 0.0);
    }
}
