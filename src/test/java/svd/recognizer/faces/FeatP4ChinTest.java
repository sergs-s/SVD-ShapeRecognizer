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

    /** (Чро): маска «всё внутри» при сетке (Чр) – побайтно как (Чр) без маски; с эллипсом – доли блока в сумме 1 или 0, длина как у (Чр). */
    @Test
    public void croMask() {
        int w = 92, h = 132, hb = 112;
        double[] im = img(w, h, 13);
        boolean[] all = new boolean[w * h];
        Arrays.fill(all, true);
        boolean[] el = FeatP4.ellipse(w, h);
        for (LbpFeature.Params lp : XBaseDates.PARAMS) {
            double[] a = FeatP4.lbpMasked(im, w, h, lp, null, hb);
            double[] b = FeatP4.lbpMasked(im, w, h, lp, all, hb);
            double[] c = FeatP4.lbpMasked(im, w, h, lp, el, hb);
            assertEquals(a.length, c.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
            for (int k = 0; k < c.length / LbpFeature.BINS; k++) {
                double s = 0;
                for (int j = 0; j < LbpFeature.BINS; j++) s += c[k * LbpFeature.BINS + j];
                assertTrue(Math.abs(s - 1) < 1e-12 || s == 0);
            }
        }
        for (int grid : new int[] {7, 9}) {
            double[] a = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, null, hb);
            double[] b = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, all, hb);
            double[] c = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, el, hb);
            assertEquals(a.length, c.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
        }
        String s = XBaseChin.maskStats();
        assertTrue(s, s.contains("LBP 9×9 + нижний ряд (90 блоков)") && s.contains("POEM 7×7 + нижний ряд (56 блоков)"));
    }

    /** (Чр): границы верхних рядов – как у кадра базы 92×112, нижний ряд – 20 px (LBP 110–130, POEM 103–123 в пикселях кадра). */
    @Test
    public void crRowEdges() {
        for (int grid : new int[] {7, 9}) {
            for (int[] a : new int[][] {{128, 108}, {114, 94}}) {
                for (int by = 0; by <= grid; by++) assertEquals(FeatP4.rowEdge(by, a[1], grid, 0), FeatP4.rowEdge(by, a[0], grid, a[1]));
                assertEquals(a[0], FeatP4.rowEdge(grid + 1, a[0], grid, a[1]));
                assertEquals(20, a[0] - a[1]);
            }
        }
    }

    /**
     * (Чр): верхние grid рядов признака кадра 92×132 побайтно равны признаку базы по верхним 92×112 (у LBP коды верхних рядов видят только
     * строки 0–111; у POEM – так же, порог τ одинаков, если наибольшая сумма ячейки – в верхней части), нижний ряд – тот же формат.
     */
    @Test
    public void crTopRowsEqualBase() {
        int w = 92, h = 132, hb = 112;
        double[] im = img(w, h, 11);
        for (int y = hb; y < h; y++) for (int x = 0; x < w; x++) im[y * w + x] = im[y * w + x] / 64;
        for (int y = 40; y < 60; y++) for (int x = 30; x < 60; x++) im[y * w + x] = (x + y) % 2 == 0 ? 4000 : 0;
        double[] top = Arrays.copyOf(im, w * hb);
        for (LbpFeature.Params lp : XBaseDates.PARAMS) {
            double[] a = LbpFeature.histogram(top, w, hb, lp);
            double[] b = FeatP4.lbpMasked(im, w, h, lp, null, hb);
            int g = lp.grid();
            assertEquals(g * (g + 1) * LbpFeature.BINS, b.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
            for (int c = g * g; c < g * (g + 1); c++) {
                double s = 0;
                for (int k = 0; k < LbpFeature.BINS; k++) s += b[c * LbpFeature.BINS + k];
                assertEquals(1, s, 1e-12);
            }
        }
        for (int grid : new int[] {7, 9}) {
            double[] a = FeatP4.poem(top, w, hb, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS);
            double[] b = FeatP4.poem(im, w, h, grid, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, null, hb);
            assertEquals(grid * (grid + 1) * PoemFeature.ORIENT * LbpFeature.BINS, b.length);
            for (int i = 0; i < a.length; i++) assertEquals(Double.doubleToLongBits(a[i]), Double.doubleToLongBits(b[i]));
        }
    }
}
