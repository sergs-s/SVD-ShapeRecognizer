package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.junit.Test;

/**
 * Пустые реплики бутстрепа (решение Хозяина 20.10.2026, №6 Жорика): мера пустой реплики – NaN (EER, порог), процентили – по
 * непустым репликам, пометка «мало людей» при непустых < 0,95·B; без пустых реплик – как прежде.
 *
 * @author ssv
 */
public class EmptyReplicateTest {

    /** Строка с 2 своими (люди 0, 1) и 5 контрольными людьми (2–6). */
    static XBaseSep.Data data() {
        XBaseSep.Data d = new XBaseSep.Data();
        d.ownP = new int[] {0, 0, 1, 1};
        d.ownR = new double[] {0.2, 0.5, 0.3, 0.9};
        d.ownOk = new boolean[] {true, true, true, false};
        d.ctrlP = new int[] {2, 3, 4, 5, 6, 2, 3};
        d.ctrlR = new double[] {0.4, 0.6, 0.7, 0.8, 0.95, 0.85, 0.65};
        return d;
    }

    @Test
    public void eerEmptyIsNaN() {
        XBaseSep.Data d = data();
        int[] noOwn = {0, 0, 1, 1, 1, 1, 1};
        assertTrue(Double.isNaN(XBaseSep.eer(d, noOwn)[3]));
        int[] noCtrl = {1, 1, 0, 0, 0, 0, 0};
        assertTrue(Double.isNaN(XBaseSep.eer(d, noCtrl)[3]));
        // Все веса 1 – не NaN и не заготовка 50.
        double e = XBaseSep.eer(d, FeiBootstrap.unit(7))[3];
        assertFalse(Double.isNaN(e));
    }

    @Test
    public void bootstrapRowWithTwoOwn() {
        XBaseSep.Data d = data();
        int[][] w = FeiBootstrap.weights(7, 2000, 7042);
        int empty = 0;
        for (int[] wq : w) if (wq[0] + wq[1] == 0 || wq[2] + wq[3] + wq[4] + wq[5] + wq[6] == 0) empty++;
        assertTrue("при 2 своих из 7 людей пустые реплики должны быть", empty > 0);
        XBaseSepLtp.Res r = XBaseSepLtp.evaluate(d, FeiBootstrap.unit(7), w);
        int nan = 0;
        for (int q = 0; q < w.length; q++) {
            boolean e = w[q][0] + w[q][1] == 0 || w[q][2] + w[q][3] + w[q][4] + w[q][5] + w[q][6] == 0;
            if (e) assertTrue(Double.isNaN(r.eerRep[q]));
            if (Double.isNaN(r.eerRep[q])) nan++;
            assertFalse("реплика EER не должна давать заготовку 50 % у пустой реплики", e && r.eerRep[q] == 50);
        }
        assertEquals(empty, nan);
        // Процентиль – по непустым.
        double[] s = FeiBootstrap.sorted(r.eerRep);
        double[] ok = java.util.Arrays.stream(r.eerRep).filter(x -> !Double.isNaN(x)).sorted().toArray();
        assertEquals(ok[(int) Math.ceil(0.975 * ok.length - 1e-9) - 1], FeiBootstrap.quantile(s, 0.975), 0.0);
        String cell = XBaseSep.ci(r.eer[3], r.eerRep);
        assertEquals(empty > 0.05 * w.length, cell.contains("мало людей"));
    }

    @Test
    public void noEmptyAsBefore() {
        Random rnd = new Random(3);
        double[] v = new double[2000];
        for (int i = 0; i < v.length; i++) v[i] = rnd.nextGaussian();
        double[] s = FeiBootstrap.sorted(v);
        for (double p : new double[] {0.025, 0.5, 0.95, 0.975}) {
            int i = (int) Math.ceil(p * s.length - 1e-9) - 1;
            assertEquals(s[i], FeiBootstrap.quantile(s, p), 0.0);
        }
        assertEquals("", FeiBootstrap.note(v));
        assertEquals("", FeiBootstrap.note(new double[0]));
    }

    @Test
    public void thresholdEmptyIsNaN() {
        double[] r = {0.1, 0.2, 0.3};
        int[] p = {0, 1, 1};
        assertTrue(Double.isNaN(FeiBootstrap.threshold(r, p, new int[] {0, 0}, 0.05)));
        assertTrue(Double.isNaN(FeiThreshold.margin(r, p, new int[] {0, 0}, 0.05).theta()));
        assertEquals(Double.POSITIVE_INFINITY, FeiBootstrap.threshold(new double[0], new int[0], new int[] {1}, 0.05), 0.0);
    }

    @Test
    public void noteMarksFewPeople() {
        double[] v = new double[100];
        for (int i = 0; i < 100; i++) v[i] = i < 94 ? i : Double.NaN;
        assertTrue(FeiBootstrap.note(v).contains("непустых реплик 94 из 100"));
        v[94] = 1;
        assertEquals("", FeiBootstrap.note(v));
    }
}
