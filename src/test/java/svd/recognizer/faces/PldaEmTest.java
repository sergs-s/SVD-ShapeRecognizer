package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

public class PldaEmTest {

    private static double[][] randomMat(Random r, int a, int b, double s) {
        double[][] m = new double[a][b];
        for (double[] row : m) for (int j = 0; j < b; j++) row[j] = s * r.nextGaussian();
        return m;
    }

    private static double[] sample(Random r, double[][] f, double[] h, double[][] g, double noise) {
        int d = f.length;
        double[] x = new double[d];
        double[] w = new double[g[0].length];
        for (int j = 0; j < w.length; j++) w[j] = r.nextGaussian();
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h.length; j++) x[i] += f[i][j] * h[j];
            for (int j = 0; j < w.length; j++) x[i] += g[i][j] * w[j];
            x[i] += noise * r.nextGaussian();
        }
        return x;
    }

    private static List<double[][]> data(Random r, double[][] f, double[][] g, double noise, int classes, int per) {
        List<double[][]> out = new ArrayList<>();
        int df = f[0].length;
        double[] mean = new double[f.length];
        for (int c = 0; c < classes; c++) {
            double[] h = new double[df];
            for (int j = 0; j < df; j++) h[j] = r.nextGaussian();
            double[][] xs = new double[per][];
            for (int k = 0; k < per; k++) {
                xs[k] = sample(r, f, h, g, noise);
                for (int i = 0; i < mean.length; i++) mean[i] += xs[k][i] / (classes * per);
            }
            out.add(xs);
        }
        for (double[][] c : out) for (double[] x : c) for (int i = 0; i < mean.length; i++) x[i] -= mean[i];
        return out;
    }

    /** Наибольший синус угла: насколько столбцы true выходят за оболочку est. */
    private static double outOfSpan(double[][] truth, double[][] est) {
        int d = truth.length;
        int k = est[0].length;
        double[][] q = new double[k][d];
        int rank = 0;
        for (int j = 0; j < k; j++) {
            double[] v = new double[d];
            for (int i = 0; i < d; i++) v[i] = est[i][j];
            for (int t = 0; t < rank; t++) {
                double s = 0;
                for (int i = 0; i < d; i++) s += q[t][i] * v[i];
                for (int i = 0; i < d; i++) v[i] -= s * q[t][i];
            }
            double n = 0;
            for (double a : v) n += a * a;
            n = Math.sqrt(n);
            if (n < 1e-12) continue;
            for (int i = 0; i < d; i++) q[rank][i] = v[i] / n;
            rank++;
        }
        double worst = 0;
        for (int j = 0; j < truth[0].length; j++) {
            double[] v = new double[d];
            double n0 = 0;
            for (int i = 0; i < d; i++) {
                v[i] = truth[i][j];
                n0 += v[i] * v[i];
            }
            for (int t = 0; t < rank; t++) {
                double s = 0;
                for (int i = 0; i < d; i++) s += q[t][i] * v[i];
                for (int i = 0; i < d; i++) v[i] -= s * q[t][i];
            }
            double n = 0;
            for (double a : v) n += a * a;
            worst = Math.max(worst, Math.sqrt(n / n0));
        }
        return worst;
    }

    /** log N(x; 0, cov) через разложение Холецкого. */
    private static double logNormal(double[] x, double[][] cov) {
        double[][] l = PldaEm.chol(cov);
        int n = x.length;
        double[] z = new double[n];
        double ld = 0;
        for (int i = 0; i < n; i++) {
            double s = x[i];
            for (int k = 0; k < i; k++) s -= l[i][k] * z[k];
            z[i] = s / l[i][i];
            ld += 2 * Math.log(l[i][i]);
        }
        double q = 0;
        for (double v : z) q += v * v;
        return -0.5 * (n * Math.log(2 * Math.PI) + ld + q);
    }

    /** Совместный закон n снимков одной личности: блоки FFᵀ, диагональ FFᵀ + GGᵀ + Σ. */
    private static double bruteSet(double[][] xs, double[][] f, double[][] g, double[] sigma) {
        int d = f.length;
        int n = xs.length;
        double[][] cov = new double[n * d][n * d];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        double v = 0;
                        for (int k = 0; k < f[0].length; k++) v += f[i][k] * f[j][k];
                        if (a == b) {
                            for (int k = 0; k < g[0].length; k++) v += g[i][k] * g[j][k];
                            if (i == j) v += sigma[i];
                        }
                        cov[a * d + i][b * d + j] = v;
                    }
                }
            }
        }
        double[] x = new double[n * d];
        for (int a = 0; a < n; a++) System.arraycopy(xs[a], 0, x, a * d, d);
        return logNormal(x, cov);
    }

    @Test
    public void closedFormMatchesBruteForceLikelihoodAndLlr() {
        Random r = new Random(5);
        double[][] f = randomMat(r, 6, 2, 1.0);
        double[][] g = randomMat(r, 6, 3, 0.7);
        List<double[][]> train = data(r, f, g, 0.5, 40, 4);
        PldaEm m = PldaEm.train(train, 2, 3, 3, 0);
        double[][] set = train.get(0);
        assertEquals(bruteSet(set, m.f, m.g, m.sigma), m.classLogLik(set), 1e-8);
        double[] probe = train.get(1)[0];
        double[][] all = new double[set.length + 1][];
        System.arraycopy(set, 0, all, 0, set.length);
        all[set.length] = probe;
        double brute = bruteSet(all, m.f, m.g, m.sigma) - bruteSet(set, m.f, m.g, m.sigma) - bruteSet(new double[][] {probe}, m.f, m.g, m.sigma);
        double[] sum = new double[6];
        for (double[] x : set) for (int i = 0; i < 6; i++) sum[i] += x[i];
        assertEquals(brute, m.llr(m.u(sum), set.length, m.u(probe)), 1e-8);
    }

    @Test
    public void likelihoodDoesNotDecrease() {
        Random r = new Random(9);
        double[][] f = randomMat(r, 15, 3, 1.0);
        double[][] g = randomMat(r, 15, 2, 0.8);
        PldaEm m = PldaEm.train(data(r, f, g, 0.4, 60, 5), 3, 2, 30, 0);
        for (int i = 1; i < m.history.length; i++) assertTrue("итерация " + i, m.history[i] >= m.history[i - 1] - 1e-7 * Math.abs(m.history[i - 1]));
    }

    @Test
    public void recoversSubspacesAndLlrSign() {
        Random r = new Random(13);
        int d = 25;
        double[][] f = randomMat(r, d, 3, 1.0);
        double[][] g = randomMat(r, d, 2, 0.8);
        PldaEm m = PldaEm.train(data(r, f, g, 0.3, 400, 8), 3, 2, 200, 1e-7);
        // F и G по отдельности определены с точностью до поворота; оболочка F ∪ G восстанавливается, F – по большей части
        double[][] both = new double[d][5];
        double[][] tr = new double[d][5];
        for (int i = 0; i < d; i++) {
            System.arraycopy(m.f[i], 0, both[i], 0, 3);
            System.arraycopy(m.g[i], 0, both[i], 3, 2);
            System.arraycopy(f[i], 0, tr[i], 0, 3);
            System.arraycopy(g[i], 0, tr[i], 3, 2);
        }
        assertTrue("оболочка F и G: " + outOfSpan(tr, both), outOfSpan(tr, both) < 0.1);
        assertTrue("оболочка G: " + outOfSpan(g, m.g), outOfSpan(g, m.g) < 0.35);
        assertTrue("оболочка F: " + outOfSpan(f, m.f), outOfSpan(f, m.f) < 0.35);
        // Знак LLR: своя проба выше нуля и выше чужой
        List<double[][]> test = data(r, f, g, 0.3, 60, 4);
        int okSame = 0;
        int okDiff = 0;
        for (int c = 0; c < test.size(); c++) {
            double[] sum = new double[d];
            for (int k = 0; k < 3; k++) for (int i = 0; i < d; i++) sum[i] += test.get(c)[k][i];
            double[] bigU = m.u(sum);
            if (m.llr(bigU, 3, m.u(test.get(c)[3])) > 0) okSame++;
            if (m.llr(bigU, 3, m.u(test.get((c + 1) % test.size())[3])) < 0) okDiff++;
        }
        assertTrue("своя проба: " + okSame, okSame >= 57);
        assertTrue("чужая проба: " + okDiff, okDiff >= 57);
    }
}
