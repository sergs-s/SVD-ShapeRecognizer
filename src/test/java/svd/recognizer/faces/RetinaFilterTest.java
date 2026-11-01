package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Ретинальный фильтр Vu (RetinaFilter): постоянный кадр – нули; линейный кадр – конечные значения в [−Th, Th], не нули,
 * отражение входа даёт отражённый выход; умножение яркостей на число выход не меняет.
 *
 * @author ssv
 */
public class RetinaFilterTest {

    private static final int W = 92;
    private static final int H = 112;

    @BeforeClass
    public static void load() {
        svd.recognizer.OpenCvLoader.load();
    }

    private static double[] ramp(double a, double b) {
        double[] img = new double[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) img[y * W + x] = a + b * x + 0.5 * y;
        return img;
    }

    @Test
    public void constantFrame() {
        double[] img = new double[W * H];
        java.util.Arrays.fill(img, 120);
        for (double v : RetinaFilter.filter(img, W, H, RetinaFilter.AUTHOR)) assertEquals(0.0, v, 0.0);
        double[] zero = new double[W * H];
        for (double v : RetinaFilter.filter(zero, W, H, RetinaFilter.AUTHOR)) assertEquals(0.0, v, 0.0);
    }

    @Test
    public void linearFrame() {
        double[] out = RetinaFilter.filter(ramp(10, 2), W, H, RetinaFilter.AUTHOR);
        double s = 0;
        for (double v : out) {
            assertTrue(Double.isFinite(v));
            assertTrue(Math.abs(v) <= RetinaFilter.AUTHOR.th());
            s += v * v;
        }
        assertTrue(s > 0);
    }

    @Test
    public void mirrorEquivariant() {
        double[] img = ramp(10, 2);
        for (int i = 0; i < img.length; i += 7) img[i] += 30 * Math.sin(i);
        double[] m = new double[img.length];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) m[y * W + x] = img[y * W + W - 1 - x];
        double[] a = RetinaFilter.filter(img, W, H, RetinaFilter.AUTHOR);
        double[] b = RetinaFilter.filter(m, W, H, RetinaFilter.AUTHOR);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) assertEquals(a[y * W + x], b[y * W + W - 1 - x], 1e-9);
    }

    @Test
    public void scaleInvariant() {
        double[] img = ramp(10, 2);
        for (int i = 0; i < img.length; i += 5) img[i] += 40 * Math.abs(Math.cos(i));
        double[] k = img.clone();
        for (int i = 0; i < k.length; i++) k[i] *= 2.5;
        double[] a = RetinaFilter.filter(img, W, H, RetinaFilter.AUTHOR);
        double[] b = RetinaFilter.filter(k, W, H, RetinaFilter.AUTHOR);
        for (int i = 0; i < a.length; i++) assertEquals(a[i], b[i], 1e-9);
    }
}
