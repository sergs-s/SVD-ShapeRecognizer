package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

/**
 * Разделимость при равном фактическом FAR контроля (PLAN.md п. 4.2, решения 11.10.2026 по LTP, п. А): порог по контрольным
 * чужим — наибольший θ при FAR ≤ α (совпадающие значения не делятся), ошибка своих при нём, веса бутстрепа.
 *
 * @author ssv
 */
public class XBaseSepLtpTest {

    /** Контроль: 10 снимков людей 0–4 (r не по порядку, два равных 0,3); свои — 4 попытки. */
    static XBaseSep.Data data() {
        XBaseSep.Data d = new XBaseSep.Data();
        d.ctrlR = new double[] {0.9, 0.3, 0.8, 0.3, 0.7, 0.95, 0.6, 0.85, 0.5, 0.99};
        d.ctrlP = new int[] {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
        d.ownR = new double[] {0.2, 0.4, 0.35, 0.1};
        d.ownOk = new boolean[] {true, true, true, false};
        d.ownP = new int[] {5, 5, 6, 6};
        return d;
    }

    @Test
    public void thresholdBelowTies() {
        XBaseSep.Data d = data();
        int[] unit = FeiBootstrap.unit(7);
        // α = 10 %: допускается ⌊1⌋ = 1 снимок, но 0,3 встречается дважды → θ ниже 0,3, принятых 0.
        double th = XBaseSepLtp.ctrlThreshold(d, unit, 0.10);
        assertTrue(th < 0.3 && th > 0.29);
        assertEquals(0, XBaseSep.ctrlFar(d, unit, th), 0);
        // Свои при θ < 0,3: принят только 0,2 (0,1 — неверный кандидат) → ошибка 3/4.
        assertEquals(75, XBaseSep.ownErr(d, unit, th), 1e-12);
        // α = 30 %: допускается 3 снимка → θ ниже 0,6 (0,3; 0,3; 0,5 приняты).
        double th3 = XBaseSepLtp.ctrlThreshold(d, unit, 0.30);
        assertTrue(th3 < 0.6 && th3 >= 0.5);
        assertEquals(30, XBaseSep.ctrlFar(d, unit, th3), 1e-12);
        assertEquals(25, XBaseSep.ownErr(d, unit, th3), 1e-12);
    }

    @Test
    public void weightsCountPeople() {
        XBaseSep.Data d = data();
        // Человек 1 (r 0,3 и 0,6) с весом 0, остальные 1: 8 снимков, α = 25 % → 2 снимка (0,3 у человека 3 и 0,5) → θ < 0,7.
        int[] w = {1, 0, 1, 1, 1, 1, 1};
        double th = XBaseSepLtp.ctrlThreshold(d, w, 0.25);
        assertTrue(th < 0.7 && th >= 0.5);
        assertEquals(25, XBaseSep.ctrlFar(d, w, th), 1e-12);
    }
}
