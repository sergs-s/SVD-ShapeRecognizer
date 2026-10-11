package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;

/**
 * LTP (PLAN.md п. 4.2, «LTP – устойчивость LBP к шуму»): коды на ручном кадре (граница u = c + t — +1), при t = 0 верхняя
 * половина — коды LBP, нижняя — коды LBP отрицательного кадра; гистограммы половин в блоках на сумму 1; маска -in по
 * половинам; |сосед − центр|; разбор faces.ltp.t; доля и квартили проверки шкалы.
 *
 * @author ssv
 */
public class LtpFeatureTest {

    /** Кадр 5×5 (один пиксель кодов — центр (2, 2) = 50): справа 56, сверху 55, слева 44, снизу 50, прочие 50. */
    static double[] hand() {
        double[] img = new double[25];
        Arrays.fill(img, 50);
        img[2 * 5 + 4] = 56;
        img[2] = 55;
        img[2 * 5] = 44;
        return img;
    }

    @Test
    public void handCodes() {
        int[][] c = LtpFeature.codes(hand(), 5, 5, 5);
        assertEquals(1, c[0].length);
        // Верх: справа (бит 0, 56 ≥ 55) и сверху (бит 2, 55 ≥ 55 — граница даёт +1); диагонали — 50 (в зоне).
        assertEquals(1 | 4, c[0][0]);
        // Низ: слева (бит 4, 44 ≤ 45).
        assertEquals(16, c[1][0]);
        // При t = 7 сверху и справа — в зоне, слева — тоже (|44 − 50| < 7).
        int[][] c7 = LtpFeature.codes(hand(), 5, 5, 7);
        assertEquals(0, c7[0][0]);
        assertEquals(0, c7[1][0]);
    }

    @Test
    public void zeroThresholdIsLbp() {
        Random rnd = new Random(1);
        int w = 20;
        int h = 17;
        double[] img = new double[w * h];
        double[] neg = new double[w * h];
        for (int i = 0; i < img.length; i++) {
            img[i] = rnd.nextInt(256);
            neg[i] = -img[i];
        }
        int[][] c = LtpFeature.codes(img, w, h, 0);
        assertArrayEquals(LbpFeature.codes(img, w, h), c[0]);
        assertArrayEquals(LbpFeature.codes(neg, w, h), c[1]);
    }

    @Test
    public void halvesSumToOnePerBlock() {
        Random rnd = new Random(2);
        double[] img = new double[92 * 112];
        for (int i = 0; i < img.length; i++) img[i] = rnd.nextGaussian();
        double[] v = LtpFeature.histogram(img, 92, 112, 0.1);
        assertEquals(LtpFeature.LENGTH, v.length);
        assertEquals(5782, v.length);
        for (int b = 0; b < 2 * LbpFeature.GRID * LbpFeature.GRID; b++) {
            double s = 0;
            for (int k = 0; k < LbpFeature.BINS; k++) s += v[b * LbpFeature.BINS + k];
            assertEquals(1, s, 1e-12);
        }
        // Верхняя половина при t = 0 — гистограмма LBP.
        double[] v0 = LtpFeature.histogram(img, 92, 112, 0);
        assertArrayEquals(LbpFeature.histogram(img, 92, 112), Arrays.copyOfRange(v0, 0, LbpFeature.LENGTH), 0);
    }

    @Test
    public void innerMaskPerHalf() {
        double[] v = new double[LtpFeature.LENGTH];
        for (int i = 0; i < v.length; i++) v[i] = i;
        double[] s = LtpFeature.select(v, LbpFeature.Mask.INNER);
        assertEquals(2 * 25 * LbpFeature.BINS, s.length);
        double[] up = LbpFeature.select(Arrays.copyOfRange(v, 0, LbpFeature.LENGTH), LbpFeature.Mask.INNER);
        double[] lo = LbpFeature.select(Arrays.copyOfRange(v, LbpFeature.LENGTH, LtpFeature.LENGTH), LbpFeature.Mask.INNER);
        assertArrayEquals(up, Arrays.copyOfRange(s, 0, up.length), 0);
        assertArrayEquals(lo, Arrays.copyOfRange(s, up.length, s.length), 0);
    }

    @Test
    public void absDiffsOfHandFrame() {
        double[] d = LtpFeature.absDiffs(hand(), 5, 5);
        assertArrayEquals(new double[] {6, 0, 5, 0, 6, 0, 0, 0}, d, 1e-12);
    }

    @Test
    public void thresholdParsing() {
        assertEquals(0.1, XBaseLtp.threshold("0.1"), 0);
        assertEquals(0.1, XBaseLtp.threshold(" 0.1 "), 0);
        for (String bad : new String[] {null, "0", "-0.1", "abc", "Infinity", "NaN"}) {
            try {
                XBaseLtp.threshold(bad);
                fail("принято «" + bad + "»");
            } catch (IllegalStateException expected) {
                // ожидаемо
            }
        }
    }

    @Test
    public void scaleShareAndQuartiles() {
        double[] s = XBaseLtp.scale(new double[] {0.05, 0.1, 0.2, 0.3}, 0.1);
        assertEquals(25, s[0], 1e-12);
        assertEquals(0.05, s[1], 0);
        assertEquals(0.1, s[2], 0);
        assertEquals(0.2, s[3], 0);
        assertEquals(4, s[4], 0);
    }
}
