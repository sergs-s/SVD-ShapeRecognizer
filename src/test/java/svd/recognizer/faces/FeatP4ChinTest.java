package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import svd.recognizer.OpenCvLoader;

public class FeatP4ChinTest {

    @BeforeClass
    public static void load() {
        OpenCvLoader.load();
    }

    static double[] img(int w, int h, long seed) {
        Random r = new Random(seed);
        double[] v = new double[w * h];
        for (int i = 0; i < v.length; i++) v[i] = r.nextInt(256);
        return v;
    }

    @Test
    public void fullMaskGivesSameHistograms() {
        int w = 92, h = 132;
        double[] im = img(w, h, 7);
        boolean[] all = new boolean[w * h];
        Arrays.fill(all, true);
        for (LbpFeature.Params lp : XBaseDates.PARAMS) {
            double[] a = LbpFeature.histogram(im, w, h, lp);
            double[] b = FeatP4.lbpMasked(im, w, h, lp, all);
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
        }
        for (int grid : new int[] {7, 9}) {
            double[] a = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS);
            double[] b = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, all);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
        }
    }

    @Test
    public void heightAndEllipse() {
        assertEquals(132, XBaseChin.height());
        boolean[] m = FeatP4.ellipse(92, 132);
        assertTrue(m[66 * 92 + 46]);
        assertTrue(!m[0] && !m[131 * 92 + 91]);
        int out = 0;
        for (boolean b : m) if (!b) out++;
        assertEquals(2596, out);
        // Симметрия при отражении по ширине.
        for (int y = 0; y < 132; y++) for (int x = 0; x < 92; x++) assertEquals(m[y * 92 + x], m[y * 92 + 91 - x]);
    }

    @Test
    public void maskedCellsSumToOneOrZero() {
        int w = 92, h = 132;
        double[] v = FeatP4.lbpMasked(img(w, h, 3), w, h, XBaseDates.PARAMS[1], FeatP4.ellipse(w, h));
        for (int c = 0; c < v.length / LbpFeature.BINS; c++) {
            double s = 0;
            for (int k = 0; k < LbpFeature.BINS; k++) s += v[c * LbpFeature.BINS + k];
            assertTrue(Math.abs(s - 1) < 1e-12 || s == 0);
        }
        assertArrayEquals(new double[0], new double[0], 0);
    }
}
