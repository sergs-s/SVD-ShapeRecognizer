package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
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

    // ---------------------------------------------------------------- синтетика (диагностика п. 3, ответ Жорика 25.10)

    static double[][] spd(Random rnd, int d, double scale, double[] diag) {
        double[][] a = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) a[i][j] = rnd.nextGaussian() * scale;
        double[][] s = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
            double v = 0;
            for (int k = 0; k < d; k++) v += a[i][k] * a[j][k];
            s[i][j] = v + (i == j ? diag[i] : 0);
        }
        return s;
    }

    static double[][] inv(double[][] m) {
        return new org.apache.commons.math3.linear.LUDecomposition(new org.apache.commons.math3.linear.Array2DRowRealMatrix(m)).getSolver().getInverse().getData();
    }

    static double[][] ltl(double[][] l, int d) {
        double[][] m = new double[d][d];
        for (double[] r : l) for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) m[i][j] += r[i] * r[j];
        return m;
    }

    static double frob(double[][] a) {
        double s = 0;
        for (double[] r : a) for (double v : r) s += v * v;
        return Math.sqrt(s);
    }

    static double[][] minus(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a.length; j++) c[i][j] = a[i][j] - b[i][j];
        return c;
    }

    /** Множитель Холецкого: случайный вектор N(0, Σ) = C·z. */
    static double[][] chol(double[][] s) {
        return new org.apache.commons.math3.linear.CholeskyDecomposition(new org.apache.commons.math3.linear.Array2DRowRealMatrix(s)).getL().getData();
    }

    static double[] draw(Random rnd, double[][] c, double[] mean) {
        int d = c.length;
        double[] z = new double[d];
        for (int i = 0; i < d; i++) z[i] = rnd.nextGaussian();
        double[] x = mean == null ? new double[d] : mean.clone();
        for (int i = 0; i < d; i++) for (int k = 0; k <= i; k++) x[i] += c[i][k] * z[k];
        return x;
    }

    /** (i) Точные несогласованные (повёрнутые) Σ_S и Σ_D, d = 20: без пола LᵀL = Σ_S⁻¹ − Σ_D⁻¹ – одна ли система координат у обеих обратных. */
    @Test
    public void exactBasis() {
        Random rnd = new Random(7);
        int d = 20;
        double[] one = new double[d];
        java.util.Arrays.fill(one, 1);
        double[][] cs = spd(rnd, d, 1, one);
        double[][] extra = spd(rnd, d, 1, one);
        double[][] cd = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) cd[i][j] = cs[i][j] + extra[i][j];
        double[][] want = minus(inv(cs), inv(cd));
        XBaseKiss.Kiss k = XBaseKiss.kiss(cs, cd, 100, 1000, false);
        assertEquals(0, k.diag().zeroed());
        double err = frob(minus(ltl(k.l(), d), want)) / frob(want);
        assertTrue("относительная ошибка " + err, err < 1e-9);
    }

    /** Классы: центр ~ N(0, B), снимки двух дат ~ центр + N(0, W); n снимков на дату. */
    static List<List<String>> classes(Random rnd, int cls, int n, double[][] cb, double[][] cw, Map<String, double[]> v, Map<String, String> date, String pre) {
        List<List<String>> out = new ArrayList<>();
        for (int c = 0; c < cls; c++) {
            double[] mu = draw(rnd, cb, null);
            List<String> keys = new ArrayList<>();
            for (int t = 0; t < 2; t++) for (int i = 0; i < n; i++) {
                String k = pre + c + "_" + t + "_" + i;
                v.put(k, draw(rnd, cw, mu));
                date.put(k, "d" + t);
                keys.add(k);
            }
            out.add(keys);
        }
        return out;
    }

    /** (ii) Оценка по парам: Σ_S → 2W, Σ_D → 2(W + B), M без пола – (2W)⁻¹ − (2(W + B))⁻¹ с относительной ошибкой ≤ 10 %. */
    @Test
    public void estimate() {
        Random rnd = new Random(11);
        int d = 20;
        double[] dw = new double[d];
        double[] db = new double[d];
        for (int i = 0; i < d; i++) {
            dw[i] = i < 5 ? 9 : 0.5;
            db[i] = i < 5 ? 0.5 : 4;
        }
        double[][] w = spd(rnd, d, 0.3, dw);
        double[][] b = spd(rnd, d, 0.3, db);
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new HashMap<>();
        Map<String, String> date = new HashMap<>();
        List<List<String>> cls = classes(rnd, 1000, 3, chol(b), chol(w), v, date, "c");
        x.vec.put(V.L1, v);
        XBaseKiss.Pairs p = XBaseKiss.pairs(x, false, cls, date, 5);
        assertEquals(9000, p.s().length);
        assertEquals(90000, p.d().length);
        double[][] z = p.vec().toArray(new double[0][]);
        double[][] cs = XBaseKiss.cov(z, p.s());
        double[][] cd = XBaseKiss.cov(z, p.d());
        double[][] w2 = new double[d][d];
        double[][] wb2 = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
            w2[i][j] = 2 * w[i][j];
            wb2[i][j] = 2 * (w[i][j] + b[i][j]);
        }
        double[][] want = minus(inv(w2), inv(wb2));
        XBaseKiss.Kiss k = XBaseKiss.kiss(cs, cd, p.s().length, p.d().length, false);
        double err = frob(minus(ltl(k.l(), d), want)) / frob(want);
        assertTrue("относительная ошибка M " + err, err < 0.10);
    }

    /** (iii) Опознавание на новых классах (галерея – дата 0, пробы – дата 1): KISSME без пола лучше евклида. */
    @Test
    public void identify() {
        Random rnd = new Random(13);
        int d = 20;
        double[] dw = new double[d];
        double[] db = new double[d];
        for (int i = 0; i < d; i++) {
            dw[i] = i < 5 ? 9 : 0.5;
            db[i] = i < 5 ? 0.5 : 4;
        }
        double[][] cw = chol(spd(rnd, d, 0.3, dw));
        double[][] cb = chol(spd(rnd, d, 0.3, db));
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new HashMap<>();
        Map<String, String> date = new HashMap<>();
        List<List<String>> train = classes(rnd, 1000, 3, cb, cw, v, date, "t");
        List<List<String>> test = classes(rnd, 200, 3, cb, cw, v, date, "q");
        x.vec.put(V.L1, v);
        XBaseKiss.Pairs p = XBaseKiss.pairs(x, false, train, date, 5);
        double[][] z = p.vec().toArray(new double[0][]);
        XBaseKiss.Kiss k = XBaseKiss.kiss(XBaseKiss.cov(z, p.s()), XBaseKiss.cov(z, p.d()), p.s().length, p.d().length, false);
        double[][] eye = new double[d][d];
        for (int i = 0; i < d; i++) eye[i][i] = 1;
        double[][] c = new double[test.size()][d];
        for (int q = 0; q < test.size(); q++) {
            for (int i = 0; i < 3; i++) for (int j = 0; j < d; j++) c[q][j] += v.get(test.get(q).get(i))[j] / 3;
        }
        int okK = 0;
        int okE = 0;
        for (int q = 0; q < test.size(); q++) {
            for (int i = 3; i < 6; i++) {
                double[] y = v.get(test.get(q).get(i));
                int bk = 0;
                int be = 0;
                for (int g = 1; g < c.length; g++) {
                    if (XBaseKiss.dist(k.l(), y, c[g]) < XBaseKiss.dist(k.l(), y, c[bk])) bk = g;
                    if (XBaseKiss.dist(eye, y, c[g]) < XBaseKiss.dist(eye, y, c[be])) be = g;
                }
                if (bk == q) okK++;
                if (be == q) okE++;
            }
        }
        assertTrue("KISSME " + okK + ", евклид " + okE, okK > okE);
    }
}
