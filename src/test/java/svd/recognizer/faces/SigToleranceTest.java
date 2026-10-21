package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import org.junit.Test;

/**
 * №5 Жорика от 21.10.2026: «интервал не содержит 0» – с допуском FeiBootstrap.SIG_EPS (конец, равный 0 с точностью до ошибки
 * округления, звёздочку не даёт); «−0,0» печатается как 0.
 *
 * @author ssv
 */
public class SigToleranceTest {

    @Test
    public void roundingNoiseIsNotSignificant() {
        assertFalse(FeiBootstrap.sig(-1e-15, 5));
        assertFalse(FeiBootstrap.sig(-8.4, -1e-15));
        assertFalse(FeiBootstrap.sig(1e-15, 3));
        assertTrue(FeiBootstrap.sig(0.001, 3));
        assertTrue(FeiBootstrap.sig(-3, -0.001));
    }

    @Test
    public void diffWithZeroEnd() {
        double[] reps = new double[40];
        Arrays.fill(reps, -1e-15);
        assertEquals("+0.0 [+0.0; +0.0]", FeiBootstrap.diff(-1e-15, reps, 1));
        Arrays.fill(reps, -0.5);
        assertEquals("-0.5 [-0.5; -0.5] *", FeiBootstrap.diff(-0.5, reps, 1));
    }

    @Test
    public void negativeZeroPrintsAsZero() {
        assertEquals(0.0, FeiBootstrap.nz(-1e-15, "%+.1f"), 0.0);
        assertEquals(-0.04, FeiBootstrap.nz(-0.04, "%+.2f"), 0.0);
        assertEquals("+0.0", String.format(java.util.Locale.ROOT, "%+.1f", FeiBootstrap.nz(-0.0, "%+.1f")));
        assertEquals(-0.06, FeiBootstrap.nz(-0.06, "%+.1f"), 0.0);
    }
}
