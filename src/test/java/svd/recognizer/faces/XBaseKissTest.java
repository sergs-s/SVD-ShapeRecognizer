package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import svd.recognizer.faces.XBaseEval.V;

/**
 * KISSME (XBaseKiss): правило выбора, пары S и D, M = Σ_S⁻¹ − Σ_D⁻¹ с полом и обнулением отрицательной части.
 *
 * @author ssv
 */
public class XBaseKissTest {

    @Test
    public void choose() {
        // База в пределах 1 п. п. от лучшего – база.
        assertEquals(0, XBaseKiss.choose(new double[] {20.5, 20.0, 19.8, 30, 30, 30}));
        // Иначе – первый в порядке простоты в пределах 1 п. п.
        assertEquals(1, XBaseKiss.choose(new double[] {25, 19.9, 19.0, 30, 30, 30}));
        assertEquals(3, XBaseKiss.choose(new double[] {25, 22, 21, 19.5, 19.0, 19.2}));
    }

    @Test
    public void pairs() {
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new HashMap<>();
        Map<String, double[]> m = new HashMap<>();
        for (String k : List.of("a1", "a2", "a3", "b1", "c1")) {
            v.put(k, new double[] {k.charAt(0), k.charAt(1)});
            m.put(k, new double[] {-k.charAt(0), k.charAt(1)});
        }
        x.vec.put(V.L1, v);
        x.mir.put(V.L1, m);
        Map<String, String> date = Map.of("a1", "d1", "a2", "d1", "a3", "d2");
        List<List<String>> cls = List.of(List.of("a1", "a2", "a3"), List.of("b1"), List.of("c1"));
        XBaseKiss.Pairs p = XBaseKiss.pairs(x, false, cls, date, 1);
        // S: a1–a3, a2–a3; D – все 7 пар разных классов (их меньше 10·|S|).
        assertEquals(2, p.s().length);
        assertEquals(7, p.d().length);
        XBaseKiss.Pairs pm = XBaseKiss.pairs(x, true, cls, date, 1);
        // С отражениями: 6 векторов человека a, даты d1 (4) и d2 (2) – 8 пар S.
        assertEquals(8, pm.s().length);
        assertThrows(IllegalStateException.class, () -> XBaseKiss.pairs(x, false, List.of(List.of("b1"), List.of("c1")), date, 1));
    }

    /** Диагональный случай: Σ_S = diag(1, 4), Σ_D = diag(2, 2); пол λ̄ = 2,5 поднимает 1; M = diag(1/2,5 − 1/2, 1/4 − 1/2) → (−0,1; −0,25) → 0. */
    @Test
    public void kiss() {
        double[][] cs = {{1, 0}, {0, 4}};
        double[][] cd = {{2, 0}, {0, 2}};
        XBaseKiss.Kiss k = XBaseKiss.kiss(cs, cd, 10, 100, true);
        assertEquals(1, k.diag().raised());
        assertEquals(2, k.diag().zeroed());
        assertEquals(0, k.l().length);
        // Без пола: M = diag(1 − 0,5, 0,25 − 0,5) → первая ось 0,5, вторая – 0.
        XBaseKiss.Kiss n = XBaseKiss.kiss(cs, cd, 10, 100, false);
        assertEquals(1, n.diag().zeroed());
        assertEquals(0.5, XBaseKiss.dist(n.l(), new double[] {1, 0}, new double[] {0, 0}), 1e-12);
        assertEquals(0.0, XBaseKiss.dist(n.l(), new double[] {0, 1}, new double[] {0, 0}), 1e-12);
        // Ранг Σ_D меньше размерности – остановка.
        assertThrows(IllegalStateException.class, () -> XBaseKiss.kiss(cs, new double[][] {{1, 0}, {0, 0}}, 10, 100, true));
    }
}
