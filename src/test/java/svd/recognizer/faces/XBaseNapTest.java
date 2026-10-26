package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Test;
import svd.recognizer.faces.XBaseEval.Probe;

/**
 * NAP / WCCN (XBaseNap, п. 2.3): а) S_n – по формуле; б) P – проектор (P² = P, Pᵀ = P, ранг n − k), после NAP векторы ортогональны
 * U_k; в) WCCN (α = 0) на синтетике с известной S_n даёт единичную (с точностью до масштаба m = tr S_n / n) внутриклассовую
 * ковариацию; г) α = 1 – база побайтно; д) выбор по простоте и файл выбора.
 *
 * @author ssv
 */
public class XBaseNapTest {

    static final int D = 12;

    /** Классы с внутриклассовой ковариацией A Aᵀ (A – случайная), разные средние. */
    static List<List<double[]>> classes(long seed, int c, int n) {
        Random r = new Random(seed);
        double[][] a = new double[D][D];
        for (double[] row : a) for (int j = 0; j < D; j++) row[j] = r.nextGaussian() * (1 + j);
        List<List<double[]>> out = new ArrayList<>();
        for (int k = 0; k < c; k++) {
            double[] mu = new double[D];
            for (int i = 0; i < D; i++) mu[i] = 10 * r.nextGaussian();
            List<double[]> l = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                double[] e = new double[D];
                for (int j = 0; j < D; j++) e[j] = r.nextGaussian();
                double[] v = mu.clone();
                for (int p = 0; p < D; p++) for (int q = 0; q < D; q++) v[p] += a[p][q] * e[q];
                l.add(v);
            }
            out.add(l);
        }
        return out;
    }

    static double[][] mul(double[][] a, double[][] b) {
        return XBasePlda.mul(a, b);
    }

    @Test
    public void snFormula() {
        List<List<double[]>> cl = List.of(List.of(new double[] {1, 0}, new double[] {3, 0}), List.of(new double[] {0, 2}, new double[] {0, 4}, new double[] {0, 6}));
        XBaseNap.Sn s = XBaseNap.sn(cl);
        // Класс 1: отклонения ±1 по x; класс 2: −2, 0, +2 по y; N − C = 3.
        assertEquals(5, s.n());
        assertEquals(2, s.c());
        assertEquals(2.0 / 3, s.s()[0][0], 1e-15);
        assertEquals(8.0 / 3, s.s()[1][1], 1e-15);
        assertEquals(0, s.s()[0][1], 1e-15);
        try {
            XBaseNap.sn(List.of(List.of(new double[] {1, 2})));
            fail("класс из одного вектора");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("класс помехи"));
        }
    }

    @Test
    public void napProjector() {
        XBaseNap.Sn s = XBaseNap.sn(classes(1, 20, 6));
        LdaMath.Eigen e = LdaMath.eigen(s.s());
        for (int k : new int[] {1, 5, 10}) {
            double[][] p = XBaseNap.nap(e, k);
            double[][] pp = mul(p, p);
            double tr = 0;
            for (int i = 0; i < D; i++) {
                tr += p[i][i];
                for (int j = 0; j < D; j++) {
                    assertEquals(p[i][j], pp[i][j], 1e-12);
                    assertEquals(p[i][j], p[j][i], 0);
                }
            }
            assertEquals(D - k, tr, 1e-10);
            Random r = new Random(k);
            double[] v = new double[D];
            for (int i = 0; i < D; i++) v[i] = r.nextGaussian();
            double[] pv = XBasePlda.mul(p, v);
            for (int q = 0; q < k; q++) {
                double dot = 0;
                for (int i = 0; i < D; i++) dot += e.vectors()[i][q] * pv[i];
                assertEquals(0, dot, 1e-12);
            }
        }
    }

    @Test
    public void wccnWhitens() {
        List<List<double[]>> cl = classes(2, 30, 8);
        XBaseNap.Sn s = XBaseNap.sn(cl);
        LdaMath.Eigen e = LdaMath.eigen(s.s());
        double m = 0;
        for (double v : e.values()) m += v;
        m /= D;
        double[][] t = XBaseNap.wccn(e, 0);
        List<List<double[]>> tc = new ArrayList<>();
        for (List<double[]> l : cl) {
            List<double[]> z = new ArrayList<>();
            for (double[] v : l) z.add(XBasePlda.mul(t, v));
            tc.add(z);
        }
        double[][] w = XBaseNap.sn(tc).s();
        for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) assertEquals(i == j ? m : 0, w[i][j], 1e-9 * m);
        // α = 0,5: собственные значения T S_n T = m · λ / ((λ + m) / 2) – между 0 и 2m.
        double[][] t5 = XBaseNap.wccn(e, 0.5);
        double[][] w5 = mul(mul(t5, s.s()), t5);
        double[] g = XBaseNap.scaled(e, 0.5, null);
        for (int i = 0; i < D; i++) {
            assertEquals((e.values()[i] / 2 + m / 2) / m, g[i], 1e-12);
            assertTrue(w5[i][i] > 0 && w5[i][i] < 2 * m);
        }
    }

    @Test
    public void alphaOneIsBase() {
        XBaseNap.Sn s = XBaseNap.sn(classes(3, 10, 5));
        LdaMath.Eigen e = LdaMath.eigen(s.s());
        assertNull(XBaseNap.wccn(e, 1));
        for (double v : XBaseNap.scaled(e, 1, null)) assertEquals(1, v, 1e-12);
        Random r = new Random(4);
        double[][] c = new double[3][];
        for (int g = 0; g < 2; g++) {
            c[g] = new double[D];
            for (int i = 0; i < D; i++) c[g][i] = r.nextGaussian();
        }
        List<String> gallery = List.of("a", "b", "z");
        List<Probe> probes = new ArrayList<>();
        Map<String, double[]> y = new java.util.HashMap<>();
        for (int i = 0; i < 5; i++) {
            double[] v = new double[D];
            for (int j = 0; j < D; j++) v[j] = r.nextGaussian();
            y.put("k" + i, v);
            probes.add(new Probe(FeiScores.OWN, i % 2 == 0 ? "a" : "b", i, "k" + i, i % 2));
        }
        double[][] id = new double[D][D];
        for (int i = 0; i < D; i++) id[i][i] = 1;
        String base = XBaseNap.lines("L1", 0, c, gallery, probes, y, null);
        assertEquals(base, XBaseNap.lines("L1", 0, c, gallery, probes, y, id));
        // База – те же числа, что XBaseRule.cos (путь XBaseReg.lines).
        StringBuilder ref = new StringBuilder();
        for (Probe p : probes) {
            double[] sc = {XBaseRule.cos(y.get(p.key()), c[0], D, null), XBaseRule.cos(y.get(p.key()), c[1], D, null), Double.POSITIVE_INFINITY};
            ref.append(FeiScores.line("L1", 0, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc, false));
        }
        assertEquals(ref.toString(), base);
    }

    @Test
    public void chooseAndSelect() throws Exception {
        assertEquals(0, XBaseNap.choose(new double[] {10.5, 10, 9.6, 12, 12, 12, 12}));
        assertEquals(2, XBaseNap.choose(new double[] {12, 11.2, 10.5, 9.8, 12, 12, 12}));
        assertEquals(6, XBaseNap.choose(new double[] {12, 12, 12, 12, 12, 12, 9}));
        assertEquals("(N) k = 20", XBaseNap.rname(3));
        assertEquals("(W) α = 0,5", XBaseNap.rname(5));
        assertEquals("(W) α = 0", XBaseNap.rname(6));
        Path f = Files.createTempFile("nap_select", ".tsv");
        try {
            Files.writeString(f, "# x\nGT\tL1-xs-F\t0\tбаза (П2)\nGT\tL1-tt-xs-F\t5\t(W) α = 0,5\nFERET-1\tL1-xs-F\t2\t(N) k = 10\n");
            Map<String, Map<String, Integer>> sel = XBaseNap.readSelect(f);
            assertEquals(5, (int) sel.get("GT").get("L1-tt-xs-F"));
            assertEquals(List.of(0, 2, 5), java.util.Arrays.stream(XBaseNap.ownRows(sel)).boxed().toList());
            Files.writeString(f, "GT\tL1-xs-F\t1\t(N) k = 10\n");
            try {
                XBaseNap.readSelect(f);
                fail("строка не из сетки");
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("не из сетки"));
            }
        } finally {
            Files.deleteIfExists(f);
        }
        assertEquals(0.495, XBaseNap.FERET_TO_GT, 0.001);
    }
}
