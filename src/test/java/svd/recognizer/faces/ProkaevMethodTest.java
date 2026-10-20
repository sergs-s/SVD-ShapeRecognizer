package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.junit.Test;

/**
 * Метод Прокаева (Д-№1): решающее правило на маленьком примере, посчитанном вручную (формулы 7 и 8, правило гостя, d = 0, ничьи);
 * SVD через матрицу Грама против Commons Math.
 *
 * @author ssv
 */
public class ProkaevMethodTest {

    private static final double EPS = 1e-12;

    /**
     * Точки на прямой: A – 0 и 1, B – 3, гости – 5. Проба w = 2: ближайшие 3 – A(1) d = 1, B(3) d = 1, A(0) d = 2.
     * 1/d = 1, 1, 0,5; сумма 2,5; p = 0,4, 0,4, 0,2. Формула 7: A = 0,6, B = 0,4. Формула 8: A max 0,4, B 0,4 → 0,5 и 0,5.
     */
    @Test
    public void formulas7and8() {
        ProkaevMethod pm = ProkaevMethod.ofPoints(new double[][] {{0}, {1}, {3}, {5}}, new String[] {"A", "A", "B", ProkaevMethod.GUEST});
        ProkaevMethod.Result r = pm.decide(new double[] {2}, ProkaevMethod.Variant.K, 1, 3);
        assertEquals(0.6, r.share().get("A"), EPS);
        assertEquals(0.4, r.share().get("B"), EPS);
        assertEquals(0.0, r.guestShare(), EPS);
        assertEquals("A", r.best());
        assertEquals(0.4, r.score(), EPS);
        assertEquals(1.0, r.d1(), EPS);
        assertFalse(r.guest());
        ProkaevMethod.Result s = pm.decide(new double[] {2}, ProkaevMethod.Variant.S, 1, 3);
        assertEquals(0.5, s.share().get("A"), EPS);
        assertEquals(0.5, s.share().get("B"), EPS);
        // Ничья 0,5 / 0,5: оба на расстоянии 1 – первый по порядку близости точек (A(1) раньше B(3) при равных d – порядок сортировки устойчив).
        assertEquals(0.5, s.bestShare(), EPS);
    }

    /**
     * Правило гостя. Проба w = 4: ближайшие – B(3) d = 1, гость(5) d = 1, A(1) d = 3; 1/d = 1, 1, 1/3; сумма 7/3; p = 3/7, 3/7, 1/7.
     * Доля гостей 3/7 > 0, доля каждого своего < 0,95 – гость (отказ). Проба w = 3: B на нуле – 100 % B, гость – 0: свой B.
     */
    @Test
    public void guestRule() {
        ProkaevMethod pm = ProkaevMethod.ofPoints(new double[][] {{0}, {1}, {3}, {5}}, new String[] {"A", "A", "B", ProkaevMethod.GUEST});
        ProkaevMethod.Result r = pm.decide(new double[] {4}, ProkaevMethod.Variant.K, 1, 3);
        assertEquals(3.0 / 7, r.guestShare(), EPS);
        assertEquals(3.0 / 7, r.share().get("B"), EPS);
        assertEquals(1.0 / 7, r.share().get("A"), EPS);
        assertTrue(r.guest());
        assertNull(r.decided());
        assertEquals("B", r.best());
        ProkaevMethod.Result z = pm.decide(new double[] {3}, ProkaevMethod.Variant.K, 1, 3);
        assertEquals(1.0, z.share().get("B"), EPS);
        assertEquals(0.0, z.share().getOrDefault("A", 0.0), EPS);
        assertFalse(z.guest());
        assertEquals("B", z.decided());
        assertEquals(0.0, z.score(), EPS);
    }

    /** d = 0 у двух точек разных классов: поровну, по 0,5; доля 0,5 < 0,95 и гостей нет – свой (правило требует доли гостей > 0). */
    @Test
    public void twoZeros() {
        ProkaevMethod pm = ProkaevMethod.ofPoints(new double[][] {{0}, {0}, {2}}, new String[] {"A", "B", ProkaevMethod.GUEST});
        ProkaevMethod.Result r = pm.decide(new double[] {0}, ProkaevMethod.Variant.K, 1, 3);
        assertEquals(0.5, r.share().get("A"), EPS);
        assertEquals(0.5, r.share().get("B"), EPS);
        assertEquals(0.0, r.guestShare(), EPS);
        assertFalse(r.guest());
    }

    /** Среди l ближайших своих нет: лучший – свой с ближайшей точкой, доля 0, отказ. */
    @Test
    public void onlyGuestsNear() {
        ProkaevMethod pm = ProkaevMethod.ofPoints(new double[][] {{10}, {12}, {0}, {1}, {2}}, new String[] {"A", "B", ProkaevMethod.GUEST,
            ProkaevMethod.GUEST, ProkaevMethod.GUEST});
        ProkaevMethod.Result r = pm.decide(new double[] {1}, ProkaevMethod.Variant.K, 1, 3);
        assertEquals(1.0, r.guestShare(), EPS);
        assertEquals("A", r.best());
        assertEquals(0.0, r.bestShare(), EPS);
        assertEquals(9.0, r.d1(), EPS);
        assertTrue(r.guest());
        // Справочно d₁/d₂: ближайший свой A (9), ближайший другой класс – гость (0).
        assertEquals(9.0, r.nn1(), EPS);
        assertEquals(0.0, r.nn2(), EPS);
    }

    /** SVD через Грама: сингулярные числа и проекции строк базиса совпадают с Commons Math (с точностью до знака оси). */
    @Test
    public void gramSvd() {
        Random rnd = new Random(1);
        int m = 6;
        int n = 15;
        float[][] a = new float[m][n];
        double[][] ad = new double[m][n];
        for (int i = 0; i < m; i++) for (int c = 0; c < n; c++) {
            a[i][c] = rnd.nextInt(256);
            ad[i][c] = a[i][c];
        }
        String[] lab = {"A", "A", "B", "B", "C", ProkaevMethod.GUEST};
        ProkaevMethod pm = ProkaevMethod.fit(a, lab, null, 4);
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(ad));
        double[] sv = svd.getSingularValues();
        for (int j = 0; j < 4; j++) assertEquals(sv[j], pm.s[j], 1e-6 * sv[0]);
        double[][] u = svd.getU().getData();
        for (int j = 0; j < 4; j++) {
            double sign = Math.signum(u[0][j] * pm.ptsK[0][j]);
            for (int i = 0; i < m; i++) {
                assertEquals(u[i][j], sign * pm.ptsK[i][j], 1e-8);
                assertEquals(u[i][j] * sv[j], sign * pm.ptsN[i][j], 1e-6 * sv[0]);
                // Проекция строки базиса равна её точке.
                assertEquals(pm.ptsK[i][j], pm.project(a[i], ProkaevMethod.Variant.K)[j], 1e-8);
                assertEquals(pm.ptsN[i][j], pm.project(a[i], ProkaevMethod.Variant.N)[j], 1e-6 * sv[0]);
            }
        }
        // Базис без гостя: точка гостя – его проекция.
        boolean[] basis = {true, true, true, true, true, false};
        ProkaevMethod pg = ProkaevMethod.fit(a, lab, basis, 4);
        double[] w = pg.project(a[5], ProkaevMethod.Variant.K);
        for (int j = 0; j < 4; j++) assertEquals(w[j], pg.ptsK[5][j], 1e-12);
    }
}
