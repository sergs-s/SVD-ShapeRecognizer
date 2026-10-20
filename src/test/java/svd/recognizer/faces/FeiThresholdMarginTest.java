package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Test;

/**
 * Порог с запасом FeiThreshold.margin (П1, П3) в реплике бутстрепа (исправление 20.10.2026, №4 Жорика): люди с весом 0 не влияют
 * на θ – результат тот же, что без их оценок; при всех весах 1 – как прежняя реализация.
 *
 * @author ssv
 */
public class FeiThresholdMarginTest {

    /** Прежняя реализация (до 20.10.2026): θ по всему списку оценок, включая людей с весом 0. */
    static FeiThreshold.Th old(double[] r, int[] p, int[] w, double far) {
        Map<Integer, Integer> ni = new HashMap<>();
        for (int q : p) ni.merge(q, 1, Integer::sum);
        double k = 0;
        double n = 0;
        double snn = 0;
        for (Map.Entry<Integer, Integer> e : ni.entrySet()) {
            int wt = w[e.getKey()];
            k += wt;
            n += (double) wt * e.getValue();
            snn += (double) wt * e.getValue() * e.getValue();
        }
        if (r.length == 0 || n == 0) return new FeiThreshold.Th(Double.POSITIVE_INFINITY, false);
        if (!FeiThreshold.marginOk(0, n, k, 0, 0, snn, far)) return new FeiThreshold.Th(Math.nextDown(r[0]), true);
        Map<Integer, Integer> xi = new HashMap<>();
        double x = 0;
        double sxx = 0;
        double sxn = 0;
        double theta = Math.nextDown(r[0]);
        for (int i = 0; i < r.length; i++) {
            int q = p[i];
            int wt = w[q];
            int xq = xi.getOrDefault(q, 0);
            x += wt;
            sxx += (double) wt * (2 * xq + 1);
            sxn += (double) wt * ni.get(q);
            xi.put(q, xq + 1);
            if (i + 1 < r.length && r[i + 1] == r[i]) continue;
            if (!FeiThreshold.marginOk(x, n, k, sxx, sxn, snn, far)) break;
            theta = i + 1 < r.length ? Math.nextDown(r[i + 1]) : Double.POSITIVE_INFINITY;
        }
        return new FeiThreshold.Th(theta, false);
    }

    /** Случайные оценки (по возрастанию, с повторами) n снимков people людей. */
    static Object[] data(Random rnd, int n, int people) {
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = new double[] {Math.round(rnd.nextDouble() * 200) / 200.0, rnd.nextInt(people)};
        Arrays.sort(a, (x, y) -> Double.compare(x[0], y[0]));
        double[] r = new double[n];
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = a[i][0];
            p[i] = (int) a[i][1];
        }
        return new Object[] {r, p};
    }

    @Test
    public void unitWeightsAsBefore() {
        Random rnd = new Random(1);
        for (int t = 0; t < 500; t++) {
            Object[] d = data(rnd, 50 + rnd.nextInt(400), 5 + rnd.nextInt(60));
            double[] r = (double[]) d[0];
            int[] p = (int[]) d[1];
            int[] w = new int[100];
            Arrays.fill(w, 1);
            for (double far : new double[] {0.01, 0.05, 0.2}) {
                FeiThreshold.Th a = FeiThreshold.margin(r, p, w, far);
                FeiThreshold.Th b = old(r, p, w, far);
                assertEquals(b.theta(), a.theta(), 0.0);
                assertEquals(b.zeroFail(), a.zeroFail());
            }
        }
    }

    @Test
    public void zeroWeightIgnored() {
        Random rnd = new Random(2);
        for (int t = 0; t < 500; t++) {
            int people = 5 + rnd.nextInt(60);
            Object[] d = data(rnd, 50 + rnd.nextInt(400), people);
            double[] r = (double[]) d[0];
            int[] p = (int[]) d[1];
            int[] w = new int[people];
            for (int q = 0; q < people; q++) w[q] = rnd.nextInt(4);
            w[p[rnd.nextInt(p.length)]] = Math.max(1, w[p[0]]);
            List<Double> rr = new ArrayList<>();
            List<Integer> pp = new ArrayList<>();
            for (int i = 0; i < r.length; i++) if (w[p[i]] > 0) {
                rr.add(r[i]);
                pp.add(p[i]);
            }
            double[] r2 = rr.stream().mapToDouble(Double::doubleValue).toArray();
            int[] p2 = pp.stream().mapToInt(Integer::intValue).toArray();
            for (double far : new double[] {0.01, 0.05, 0.2}) {
                FeiThreshold.Th a = FeiThreshold.margin(r, p, w, far);
                FeiThreshold.Th b = FeiThreshold.margin(r2, p2, w, far);
                assertEquals(b.theta(), a.theta(), 0.0);
                assertEquals(b.zeroFail(), a.zeroFail());
            }
        }
    }

    /** Ручной пример: чужой веса 0 с оценкой между оценками вытянутых не сдвигает θ. */
    @Test
    public void handExample() {
        // Люди 0 и 1 (вес 1) и 2 (вес 0): оценки 0,1 (0), 0,2 (2), 0,3 (1), 0,4 (0), 0,5 (2), 0,6 (1).
        double[] r = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        int[] p = {0, 2, 1, 0, 2, 1};
        int[] w = {1, 1, 0};
        double[] r2 = {0.1, 0.3, 0.4, 0.6};
        int[] p2 = {0, 1, 0, 1};
        for (double far : new double[] {0.01, 0.3, 0.9}) {
            assertEquals(FeiThreshold.margin(r2, p2, w, far).theta(), FeiThreshold.margin(r, p, w, far).theta(), 0.0);
        }
    }
}
