package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;

/**
 * Ревизия 1.4 (LbpFeature.Params, XBaseLbp): а) при DEFAULT – побайтно прежняя гистограмма; б) длины вариантов; в) вариант (4)
 * – подряд (3) и база; г) правило выбора.
 *
 * @author ssv
 */
public class XBaseLbpTest {

    static double[] image(Random rnd, int w, int h, boolean integer) {
        double[] img = new double[w * h];
        for (int i = 0; i < img.length; i++) img[i] = integer ? rnd.nextInt(256) : rnd.nextGaussian();
        return img;
    }

    @Test
    public void defaultIsOld() {
        Random rnd = new Random(14);
        for (int t = 0; t < 20; t++) {
            double[] img = image(rnd, 92, 112, t % 2 == 0);
            double[] a = LbpFeature.histogram(img, 92, 112);
            double[] b = LbpFeature.histogram(img, 92, 112, LbpFeature.Params.DEFAULT);
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
        }
    }

    @Test
    public void lengths() {
        int[] want = {2891, 1475, 4779, 2891, 5782};
        assertArrayEquals(want, XBaseLbp.dims());
        double[] img = image(new Random(3), 92, 112, true);
        for (int v = 0; v < XBaseLbp.VARS.length; v++) {
            double[] h = LbpFeature.histogram(img, 92, 112, XBaseLbp.VARS[v]);
            assertEquals(want[v], h.length);
            // Каждый блок нормирован на сумму 1.
            for (int b = 0; b < h.length / LbpFeature.BINS; b++) {
                double s = 0;
                for (int k = 0; k < LbpFeature.BINS; k++) s += h[b * LbpFeature.BINS + k];
                assertEquals(1.0, s, 1e-12);
            }
        }
    }

    @Test
    public void twoScalesConcatenated() {
        Random rnd = new Random(5);
        for (int t = 0; t < 6; t++) {
            double[] img = image(rnd, 92, 112, t % 2 == 0);
            double[] two = LbpFeature.histogram(img, 92, 112, XBaseLbp.VARS[4]);
            double[] r1 = LbpFeature.histogram(img, 92, 112, XBaseLbp.VARS[3]);
            double[] r2 = LbpFeature.histogram(img, 92, 112, XBaseLbp.VARS[0]);
            assertArrayEquals(r1, Arrays.copyOfRange(two, 0, r1.length), 0);
            assertArrayEquals(r2, Arrays.copyOfRange(two, r1.length, two.length), 0);
        }
    }

    @Test
    public void choice() {
        int[] d = XBaseLbp.dims();
        // База в пределах 1 п. п. от лучшего – база.
        assertEquals(0, XBaseLbp.choose(new double[] {20.9, 21, 20.0, 25, 22}, d));
        // Иначе – меньшая размерность среди близких: (1) 1475 раньше (4) 5782.
        assertEquals(1, XBaseLbp.choose(new double[] {22, 20.5, 25, 25, 20.0}, d));
        // (3) и (2) близки – (3) (2891 < 4779).
        assertEquals(3, XBaseLbp.choose(new double[] {22, 25, 20.0, 20.8, 25}, d));
        // Единственный близкий – он.
        assertEquals(4, XBaseLbp.choose(new double[] {22, 25, 25, 25, 20.0}, d));
    }
}
