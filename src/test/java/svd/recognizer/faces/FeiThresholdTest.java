package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * Калибровка порога (п. 4.2): граница П1 / П3, порог с запасом, критерий выбора, части галереи П2.
 *
 * @author ssv
 */
public class FeiThresholdTest {

    @Test
    public void boundMatchesClopperPearson() {
        // Целые x: граница не выше far ⇔ точная граница Клоппера – Пирсона не выше far.
        int[][] xn = {{0, 351}, {1, 351}, {0, 200}, {3, 800}, {12, 351}, {20, 351}};
        for (int[] c : xn) {
            double ub = FaceEvaluation.binomialUpperBound(c[0], c[1], 0.95);
            for (double far : new double[] {0.01, 0.05}) {
                if (Math.abs(ub - far) < 1e-6) continue;
                assertEquals(c[0] + "/" + c[1] + " при " + far, ub <= far, FeiThreshold.boundOk(c[0], c[1], far, 0.95));
            }
        }
        // Ноль при n = 351: 1 − 0,05^(1/351) ≈ 0,85 % ≤ 1 %; при n = 200 — 1,49 % > 1 %.
        assertTrue(FeiThreshold.boundOk(0, 351, 0.01, 0.95));
        assertFalse(FeiThreshold.boundOk(0, 200, 0.01, 0.95));
    }

    @Test
    public void deffGrouping() {
        // 10 людей по 10 снимков, принято 5 снимков у одного человека — сильная группировка.
        double n = 100;
        double k = 10;
        double snn = 10 * 100;
        double grouped = FeiThreshold.deff(5, n, k, 25, 5 * 10, snn);
        // Те же 5 принятых — по одному у пяти людей.
        double spread = FeiThreshold.deff(5, n, k, 5, 5 * 10, snn);
        assertTrue(grouped > 4);
        assertEquals(1, spread, 0);
        assertEquals(1, FeiThreshold.deff(0, n, k, 0, 0, snn), 0);
        assertTrue(Double.isInfinite(FeiThreshold.deff(1, n, 1, 1, 10, snn)));
    }

    @Test
    public void marginThreshold() {
        // 400 чужих по одному снимку: при FAR 5 % допустимо несколько принятых, при 1 % — ноль (граница при x = 1 — 1,2 %).
        int n = 400;
        double[] r = new double[n];
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i + 1;
            p[i] = i;
        }
        int[] w = FeiBootstrap.unit(n);
        FeiThreshold.Th t1 = FeiThreshold.margin(r, p, w, 0.01);
        assertFalse(t1.zeroFail());
        assertEquals(Math.nextDown(1.0), t1.theta(), 0);
        FeiThreshold.Th t5 = FeiThreshold.margin(r, p, w, 0.05);
        int accepted = 0;
        for (double x : r) if (x <= t5.theta()) accepted++;
        assertTrue(accepted >= 8 && accepted < 20);
        // Порог с запасом не выше точечного.
        assertTrue(t5.theta() <= FeiBootstrap.threshold(r, p, w, 0.05));
        // Мало чужих: ноль недопустим — порог ниже наименьшей оценки.
        FeiThreshold.Th z = FeiThreshold.margin(new double[] {1, 2, 3}, new int[] {0, 1, 2}, FeiBootstrap.unit(3), 0.01);
        assertTrue(z.zeroFail());
        assertEquals(Math.nextDown(1.0), z.theta(), 0);
        // Равные оценки не разделяются: оба снимка с оценкой 1 принимаются вместе или не принимаются.
        FeiThreshold.Th tie = FeiThreshold.margin(new double[] {1, 1, 2}, new int[] {0, 1, 2}, FeiBootstrap.unit(3), 0.99);
        int tieAccepted = 0;
        for (double x : new double[] {1, 1, 2}) if (x <= tie.theta()) tieAccepted++;
        assertTrue(tieAccepted != 1);
    }

    @Test
    public void marginWeights() {
        // Человек с весом 0 не влияет: его снимки ниже всех, порог тот же, что без него.
        double[] r = new double[401];
        int[] p = new int[401];
        r[0] = 0.5;
        p[0] = 400;
        for (int i = 0; i < 400; i++) {
            r[i + 1] = i + 1;
            p[i + 1] = i;
        }
        int[] w = FeiBootstrap.unit(401);
        w[400] = 0;
        assertEquals(Math.nextDown(1.0), FeiThreshold.margin(r, p, w, 0.01).theta(), 0);
    }

    @Test
    public void chooseRule() {
        boolean[] all = {true, true, true, true};
        // Наименьшая ошибка — П2; П0 незначимо отличается — выбирается П0.
        double[][] reps = {{5, 6, 7}, {9, 9, 9}, {4, 5, 8}, {8, 8, 8}};
        assertEquals(0, FeiThreshold.choose(new double[] {6, 9, 5, 8}, reps, all));
        // П0 значимо хуже П2 — П2.
        double[][] reps2 = {{9, 9, 9}, {9, 9, 9}, {4, 5, 6}, {8, 8, 8}};
        assertEquals(2, FeiThreshold.choose(new double[] {9, 9, 5, 8}, reps2, all));
        // П0 и П2 недопустимы — лучший из допустимых П3 (П1 значимо хуже).
        assertEquals(3, FeiThreshold.choose(new double[] {6, 9, 5, 8}, reps2, new boolean[] {false, true, false, true}));
        assertEquals(-1, FeiThreshold.choose(new double[] {1, 2, 3, 4}, reps, new boolean[4]));
    }

    @Test
    public void tolerance() {
        // Допуск: номинал + 10 % от него — 1,1 % при 1 %, 5,5 % при 5 %.
        assertTrue(FeiThreshold.withinTolerance(110, 10000, 0.01));
        assertFalse(FeiThreshold.withinTolerance(111, 10000, 0.01));
        assertTrue(FeiThreshold.withinTolerance(154, 14032, 0.01));
        assertTrue(FeiThreshold.withinTolerance(550, 10000, 0.05));
        assertFalse(FeiThreshold.withinTolerance(551, 10000, 0.05));
    }

    @Test
    public void cvParts() {
        List<String> g = new ArrayList<>();
        for (int i = 0; i < 100; i++) g.add("p" + i);
        List<List<String>> a = FeiMethods.cvParts(g, 42, 3);
        assertEquals(a, FeiMethods.cvParts(g, 42, 3));
        assertTrue(!a.equals(FeiMethods.cvParts(g, 42, 4)));
        assertEquals(FeiMethods.CV_PARTS, a.size());
        Set<String> all = new HashSet<>();
        for (List<String> q : a) {
            assertEquals(20, q.size());
            all.addAll(q);
        }
        assertEquals(new HashSet<>(g), all);
    }
}
