package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import org.junit.Assume;
import org.junit.Test;

/**
 * BSIF (BsifFeature, пачка 2): а) известный ответ – фильтр-«дельта» в центре: бит = знак пикселя; б) сверка с прямой реализацией
 * bsif.m (filter2 'valid', бит i – фильтр b − i + 1) на случайных фильтрах; в) длина и сумма 1 по блокам; г) фильтры авторов
 * (если есть FACES_BSIF_DIR или D:\data\models\bsif): 7×7 и 9×9 по 8 бит читаются, конечны и ненулевые.
 *
 * @author ssv
 */
public class BsifFeatureTest {

    static final int W = 92;
    static final int H = 112;

    static double[] random(long seed, int n) {
        Random r = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = r.nextGaussian();
        return v;
    }

    @Test
    public void delta() {
        double[][][] f = new double[1][3][3];
        f[0][1][1] = 1;
        BsifFeature b = new BsifFeature(f);
        double[] img = {-1, 2, -3, 4, 5, -6, 7, -8, 9, 1, 1, -1};
        int[] c = b.codes(img, 4, 3);
        // Окно целиком в кадре – центры (1, 1) и (2, 1): значения −6 и 7.
        assertArrayEquals(new int[] {0, 1}, c);
    }

    /** bsif.m: ci = filter2(tmp, img, 'valid'), tmp – фильтр b − i + 1 (1-нумерация), бит i − 1. */
    static int[] naive(double[][][] f, double[] img, int w, int h) {
        int n = f[0].length;
        int bits = f.length;
        int cw = w - n + 1;
        int ch = h - n + 1;
        int[] code = new int[cw * ch];
        for (int i = 1; i <= bits; i++) {
            double[][] t = f[bits - i];
            for (int y = 0; y < ch; y++) for (int x = 0; x < cw; x++) {
                double s = 0;
                for (int u = 0; u < n; u++) for (int v = 0; v < n; v++) s += t[u][v] * img[(y + u) * w + x + v];
                if (s > 0) code[y * cw + x] += 1 << (i - 1);
            }
        }
        return code;
    }

    @Test
    public void againstNaive() {
        Random r = new Random(3);
        double[][][] f = new double[8][5][5];
        for (double[][] a : f) for (double[] row : a) for (int i = 0; i < row.length; i++) row[i] = r.nextGaussian();
        double[] img = random(9, W * H);
        BsifFeature b = new BsifFeature(f);
        assertArrayEquals(naive(f, img, W, H), b.codes(img, W, H));
        assertEquals(7 * 7 * 256, b.length(7));
        double[] h = b.histogram(img, W, H, 7);
        for (int k = 0; k < 49; k++) {
            double s = 0;
            for (int j = 0; j < 256; j++) s += h[k * 256 + j];
            assertEquals(1.0, s, 1e-12);
        }
    }

    @Test
    public void authorsFilters() throws Exception {
        String e = System.getenv("FACES_BSIF_DIR");
        Path dir = Paths.get(e == null || e.isBlank() ? "D:/data/models/bsif" : e);
        Assume.assumeTrue(Files.exists(BsifFeature.file(dir, 7, 8)));
        for (int n : new int[] {7, 9}) {
            BsifFeature b = BsifFeature.load(dir, n, 8);
            assertEquals(8, b.filters.length);
            double norm = 0;
            for (double[][] a : b.filters) {
                assertEquals(n, a.length);
                for (double[] row : a) {
                    assertEquals(n, row.length);
                    for (double v : row) {
                        assertTrue(Double.isFinite(v));
                        norm += v * v;
                    }
                }
            }
            assertTrue(norm > 0);
        }
    }
}
