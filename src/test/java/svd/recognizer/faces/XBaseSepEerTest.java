package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

/**
 * Точка EER (XBaseSep.eer; решение Хозяина 20.10.2026, №8 Жорика): разности сравниваются в целых, при равенстве – первая точка по θ.
 *
 * @author ssv
 */
public class XBaseSepEerTest {

    /**
     * 2 своих (верные), 24 контрольных. θ₁ = 0,13: промахов 2, принято 13 – |100 − 54,1(6)| = 45,8(3); θ₂ = 0,5: промахов 1,
     * принято 23 – |50 − 95,8(3)| = 45,8(3). В double разность у θ₂ меньше в последнем разряде (45,833333333333336 против
     * 45,83333333333333); выбирается первая по θ.
     */
    @Test
    public void tieTakesFirstTheta() {
        double g1 = Math.abs(100.0 * 2 / 2 - 100.0 * 13 / 24);
        double g2 = Math.abs(100.0 * 1 / 2 - 100.0 * 23 / 24);
        assertTrue("пример должен давать разный последний разряд", g2 < g1);
        XBaseSep.Data d = new XBaseSep.Data();
        d.ownP = new int[] {0, 1};
        d.ownR = new double[] {0.5, 0.9};
        d.ownOk = new boolean[] {true, true};
        int nc = 24;
        d.ctrlP = new int[nc];
        d.ctrlR = new double[nc];
        for (int i = 0; i < nc; i++) {
            d.ctrlP[i] = 2 + i;
            d.ctrlR[i] = i < 13 ? 0.10 + 0.0025 * i : i < 23 ? 0.5 : 0.95;
        }
        double[] e = XBaseSep.eer(d, FeiBootstrap.unit(2 + nc));
        assertEquals(0.10 + 0.0025 * 12, e[0], 0.0);
        assertEquals(100.0, e[1], 1e-12);
        assertEquals(100.0 * 13 / 24, e[2], 1e-12);
    }
}
