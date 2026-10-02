package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Test;

/**
 * Бутстреп по людям (этап 5б): порог по весам при весах 1 — порог FeiScores, при целых весах — порог на размноженных
 * снимках; точка (веса 1) совпадает с разбором FeiScores; реплики воспроизводимы по seed; разность одинаковых строк — 0.
 *
 * @author ssv
 */
public class FeiBootstrapTest {

    @Test
    public void unitWeightsGiveNeymanPearsonThreshold() {
        Random rnd = new Random(1);
        for (int trial = 0; trial < 500; trial++) {
            int n = 1 + rnd.nextInt(60);
            double[] v = new double[n];
            int[] person = new int[n];
            int persons = 1 + rnd.nextInt(10);
            for (int i = 0; i < n; i++) {
                v[i] = rnd.nextInt(15) / 3.0;
                person[i] = rnd.nextInt(persons);
            }
            double[] s = v.clone();
            Arrays.sort(s);
            int[] w = new int[persons];
            Arrays.fill(w, 1);
            for (double far : FeiScores.FARS) {
                assertEquals(FaceEvaluation.neymanPearsonThreshold(v, far), FeiBootstrap.threshold(s, person, w, far), 0);
            }
            assertEquals(FaceEvaluation.neymanPearsonThreshold(v, 0.2), FeiBootstrap.threshold(s, person, w, 0.2), 0);
        }
    }

    @Test
    public void weightsEqualDuplication() {
        Random rnd = new Random(2);
        for (int trial = 0; trial < 500; trial++) {
            int persons = 1 + rnd.nextInt(8);
            int n = 1 + rnd.nextInt(40);
            double[] v = new double[n];
            int[] person = new int[n];
            for (int i = 0; i < n; i++) {
                v[i] = rnd.nextInt(20);
                person[i] = rnd.nextInt(persons);
            }
            int[] w = new int[persons];
            for (int p = 0; p < persons; p++) w[p] = rnd.nextInt(4);
            List<Double> dup = new ArrayList<>();
            for (int i = 0; i < n; i++) for (int k = 0; k < w[person[i]]; k++) dup.add(v[i]);
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
            double[] s = new double[n];
            int[] ps = new int[n];
            for (int i = 0; i < n; i++) {
                s[i] = v[idx[i]];
                ps[i] = person[idx[i]];
            }
            for (double far : new double[] {0.01, 0.05, 0.25}) {
                double expected = dup.isEmpty() ? Double.POSITIVE_INFINITY
                        : FaceEvaluation.neymanPearsonThreshold(dup.stream().mapToDouble(Double::doubleValue).toArray(), far);
                assertEquals(expected, FeiBootstrap.threshold(s, ps, w, far), 0);
            }
        }
    }

    /** Два разбиения × конфигурации j = 0 (поворот) и 4 (анфас); свои 10, пороговые 8 (1–4 снимка), контроль 8. */
    private static List<FeiScores.Row> rows(String method, long seed) {
        Random rnd = new Random(seed);
        List<FeiScores.Row> rows = new ArrayList<>();
        for (int s = 0; s < 2; s++) {
            for (int j : new int[] {0, 4}) {
                for (int p = 0; p < 10; p++) {
                    String id = "o" + s + p;
                    double v = rnd.nextDouble();
                    double t = rnd.nextInt(10) == 0 ? Double.POSITIVE_INFINITY : v;
                    rows.add(new FeiScores.Row(method, s, j, FeiScores.OWN, id, j, id, rnd.nextInt(5) == 0 ? "z" : id, v, "x", v + 1, t, -1));
                }
                for (String role : List.of(FeiScores.THR, FeiScores.CTRL)) {
                    for (int p = 0; p < 8; p++) {
                        int k = 1 + (p * 7 + s) % 4;
                        for (int q = 0; q < k; q++) {
                            double v = rnd.nextDouble() * 1.5;
                            rows.add(new FeiScores.Row(method, s, j, role, role + s + p, q, FeiScores.NONE, "o00", v, "o01", v + 1, Double.NaN, -1));
                        }
                    }
                }
            }
        }
        return rows;
    }

    private static FeiScores.Analysis analysis() {
        FeiScores.Analysis a = new FeiScores.Analysis();
        a.boot = new FeiBootstrap();
        for (FeiScores.Row r : rows("2c-mlda", 5)) a.add(r);
        for (FeiScores.Row r : rows("2c-mlda+mirror", 5)) a.add(r);
        a.finish();
        return a;
    }

    @Test
    public void pointMatchesFeiScores() {
        FeiScores.Analysis a = analysis();
        FeiBootstrap.Result r = a.boot.run(50, 7);
        assertEquals(List.of("2c-mlda", "2c-mlda-ratio", "2c-mlda+mirror", "2c-mlda-ratio+mirror"), new ArrayList<>(r.point.keySet()));
        for (String id : r.point.keySet()) {
            FeiScores.Stat s = a.stats.get(id);
            double[] p = r.point.get(id);
            assertEquals(100.0 * s.errors(FeiScores.T1) / s.att, p[FeiBootstrap.ERR1], 0);
            assertEquals(100.0 * s.errors(FeiScores.T5) / s.att, p[FeiBootstrap.ERR5], 0);
            assertEquals(100.0 * s.errors(FeiScores.T5, 1) / s.attBy[1], p[FeiBootstrap.ERR5_FRONTAL], 0);
            assertEquals(100.0 * s.errors(FeiScores.T5, 0) / s.attBy[0], p[FeiBootstrap.ERR5_TURN], 0);
            int[] f1 = sum(s.far(s.farCtrl, FeiScores.T1));
            int[] f5 = sum(s.far(s.farCtrl, FeiScores.T5));
            assertEquals(100.0 * f1[0] / f1[1], p[FeiBootstrap.FAR1], 0);
            assertEquals(100.0 * f5[0] / f5[1], p[FeiBootstrap.FAR5], 0);
            assertEquals(100.0 * f1[2] / f1[3], p[FeiBootstrap.FARP1], 0);
            assertArrayEquals(new int[] {f1[0], f1[1]}, r.cp.get(id)[0]);
            assertArrayEquals(new int[] {f5[0], f5[1]}, r.cp.get(id)[1]);
        }
    }

    private static int[] sum(int[][] cfg) {
        int[] t = new int[4];
        for (int[] c : cfg) for (int k = 0; k < 4; k++) t[k] += c[k];
        return t;
    }

    @Test
    public void reproducibleAndPaired() {
        FeiBootstrap.Result r1 = analysis().boot.run(200, 11);
        FeiBootstrap.Result r2 = analysis().boot.run(200, 11);
        for (String id : r1.reps.keySet()) {
            for (int k = 0; k < FeiBootstrap.METRICS; k++) assertArrayEquals(r1.reps.get(id)[k], r2.reps.get(id)[k], 0);
        }
        // Строки с одинаковыми оценками: реплики попарно равны — разность ровно 0 во всех репликах.
        for (int k = 0; k < FeiBootstrap.METRICS; k++) {
            assertArrayEquals(r1.reps.get("2c-mlda")[k], r1.reps.get("2c-mlda+mirror")[k], 0);
        }
        List<String[]> pairs = FeiBootstrap.pairs(r1.point.keySet(), r1.point);
        assertEquals(2, pairs.size());
        assertArrayEquals(new String[] {"2c-mlda+mirror", "2c-mlda"}, pairs.get(0));
        assertArrayEquals(new String[] {"2c-mlda-ratio+mirror", "2c-mlda-ratio"}, pairs.get(1));
        String text = FeiBootstrap.report(r1, List.of("# шапка"), "test").toString();
        assertTrue(text.contains("2c-mlda+mirror − 2c-mlda | +0.0 [+0.0; +0.0] |"));
        // Реплики меняются (иначе интервал вырожден).
        double[] e = r1.reps.get("2c-mlda")[FeiBootstrap.ERR5];
        assertTrue(Arrays.stream(e).distinct().count() > 5);
    }

    @Test
    public void weightsResampleWholePersons() {
        int[][] sizes = {{10, 8, 8}, {10, 8, 8}};
        int[][][][] w = FeiBootstrap.weights(sizes, 100, 3);
        for (int[][][] b : w) {
            for (int s = 0; s < sizes.length; s++) {
                for (int k = 0; k < 3; k++) assertEquals(sizes[s][k], Arrays.stream(b[s][k]).sum());
            }
        }
        assertArrayEquals(w[5][1][2], FeiBootstrap.weights(sizes, 100, 3)[5][1][2]);
    }

    @Test
    public void quantiles() {
        double[] v = new double[2000];
        for (int i = 0; i < v.length; i++) v[i] = i + 1;
        assertEquals(50, FeiBootstrap.quantile(v, 0.025), 0);
        assertEquals(1950, FeiBootstrap.quantile(v, 0.975), 0);
        assertEquals(1900, FeiBootstrap.quantile(v, 0.95), 0);
    }
}
