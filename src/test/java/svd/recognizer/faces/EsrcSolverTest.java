package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.junit.Test;

public class EsrcSolverTest {

    private static double[][] columns(Random rnd, int d, int p) {
        double[][] phi = new double[p][d];
        for (double[] col : phi) {
            double n = 0;
            for (int i = 0; i < d; i++) {
                col[i] = rnd.nextGaussian();
                n += col[i] * col[i];
            }
            n = Math.sqrt(n);
            for (int i = 0; i < d; i++) col[i] /= n;
        }
        return phi;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static double[][] gram(double[][] phi) {
        double[][] g = new double[phi.length][phi.length];
        for (int i = 0; i < phi.length; i++) for (int j = 0; j < phi.length; j++) g[i][j] = dot(phi[i], phi[j]);
        return g;
    }

    private static double[] atPhi(double[][] phi, double[] y) {
        double[] c = new double[phi.length];
        for (int j = 0; j < phi.length; j++) c[j] = dot(phi[j], y);
        return c;
    }

    private static double[] residual(double[][] phi, double[] x, double[] y) {
        double[] r = y.clone();
        for (int j = 0; j < phi.length; j++) if (x[j] != 0) for (int i = 0; i < r.length; i++) r[i] -= x[j] * phi[j][i];
        return r;
    }

    /** Условия оптимальности LASSO: на активных столбцах Φᵀr = λ·sign(x), на прочих |Φᵀr| ≤ λ. */
    private static void kkt(double[][] phi, double[] x, double[] r) {
        double lambda = 0;
        for (int j = 0; j < phi.length; j++) lambda = Math.max(lambda, Math.abs(dot(phi[j], r)));
        for (int j = 0; j < phi.length; j++) {
            if (x[j] != 0) assertEquals("активный столбец " + j, lambda * Math.signum(x[j]), dot(phi[j], r), 1e-7);
        }
    }

    private static double objective(double[][] phi, double[] x, double[] y, double lambda) {
        double[] r = residual(phi, x, y);
        double l1 = 0;
        for (double v : x) l1 += Math.abs(v);
        return 0.5 * dot(r, r) + lambda * l1;
    }

    private static double[] ista(double[][] phi, double[] y, double lambda, int iters) {
        double[][] g = gram(phi);
        double[] c0 = atPhi(phi, y);
        double l = 0;
        for (int i = 0; i < g.length; i++) l += g[i][i];
        double[] x = new double[phi.length];
        for (int it = 0; it < iters; it++) {
            double[] nx = new double[x.length];
            for (int j = 0; j < x.length; j++) {
                double gr = -c0[j];
                for (int t = 0; t < x.length; t++) gr += g[j][t] * x[t];
                double v = x[j] - gr / l;
                nx[j] = Math.signum(v) * Math.max(Math.abs(v) - lambda / l, 0);
            }
            x = nx;
        }
        return x;
    }

    @Test
    public void reachesEpsilonAndSatisfiesKkt() {
        for (int seed = 0; seed < 30; seed++) {
            Random rnd = new Random(seed);
            boolean over = seed % 2 == 0;
            int d = over ? 40 : 120;
            int p = over ? 120 : 60;
            double[][] phi = columns(rnd, d, p);
            double[] y = columns(rnd, d, 1)[0];
            double eps = over ? 0.05 : 0.5;
            EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, y), 1, eps, 5000);
            assertFalse(s.capped());
            double[] r = residual(phi, s.x(), y);
            assertEquals("невязка, сид " + seed, s.rr(), dot(r, r), 1e-9);
            if (over) assertEquals(eps * eps, dot(r, r), 1e-9);
            kkt(phi, s.x(), r);
        }
    }

    @Test
    public void matchesIsta() {
        Random rnd = new Random(7);
        int d = 30;
        double[][] phi = columns(rnd, d, 50);
        double[] y = columns(rnd, d, 1)[0];
        EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, y), 1, 0.4, 5000);
        double[] r = residual(phi, s.x(), y);
        double lambda = 0;
        for (double[] col : phi) lambda = Math.max(lambda, Math.abs(dot(col, r)));
        double[] x = ista(phi, y, lambda, 40000);
        assertEquals(objective(phi, x, y, lambda), objective(phi, s.x(), y, lambda), 1e-6);
    }

    @Test
    public void exactColumnIsFewSteps() {
        Random rnd = new Random(3);
        double[][] phi = columns(rnd, 200, 20);
        EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, phi[4]), 1, 0.05, 100);
        assertTrue(s.x()[4] > 0.9);
        int nz = 0;
        for (double v : s.x()) if (v != 0) nz++;
        assertTrue("активных " + nz, nz <= 3);
    }

    @Test
    public void duplicateColumnDoesNotBreak() {
        Random rnd = new Random(11);
        double[][] phi = columns(rnd, 15, 30);
        phi[7] = phi[3].clone();
        double[] y = columns(rnd, 15, 1)[0];
        EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, y), 1, 0.05, 5000);
        assertFalse(s.capped());
        double[] r = residual(phi, s.x(), y);
        assertEquals(s.rr(), dot(r, r), 1e-8);
    }

    @Test
    public void bigEpsilonIsZero() {
        Random rnd = new Random(1);
        double[][] phi = columns(rnd, 10, 5);
        EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, phi[0]), 1, 1.5, 10);
        assertEquals(0, s.steps());
    }

    /** Как словарь ESRC: разности от центра дают зависимые столбцы; размерность больше числа столбцов, ε недостижимо. */
    @Test
    public void dependentColumnsUnreachableEpsilon() {
        Random rnd = new Random(21);
        int d = 200;
        double[][] raw = columns(rnd, d, 30);
        double[][] phi = new double[30][d];
        for (int c = 0; c < 10; c++) {
            double[] m = new double[d];
            for (int i = 0; i < 3; i++) for (int q = 0; q < d; q++) m[q] += raw[3 * c + i][q] / 3;
            for (int i = 0; i < 3; i++) {
                double n = 0;
                for (int q = 0; q < d; q++) {
                    phi[3 * c + i][q] = raw[3 * c + i][q] - m[q];
                    n += phi[3 * c + i][q] * phi[3 * c + i][q];
                }
                n = Math.sqrt(n);
                for (int q = 0; q < d; q++) phi[3 * c + i][q] /= n;
            }
        }
        double[] y = columns(rnd, d, 1)[0];
        EsrcSolver.Result s = EsrcSolver.solve(gram(phi), atPhi(phi, y), 1, 0.05, 2000);
        assertFalse(s.capped());
        double[] r = residual(phi, s.x(), y);
        assertEquals(s.rr(), dot(r, r), 1e-8);
        assertTrue(s.rr() > 0.05 * 0.05);
        for (double[] col : phi) assertEquals(0, dot(col, r), 1e-6);
    }

    /** Точки одного пути: при λ = q · λ_max – условия оптимальности с этим λ, при q ближе к 1 активных меньше. */
    @Test
    public void pathPointsAreOptimalAtTheirLambda() {
        double[] qs = {0.3, 0.1, 0.03, 0.01};
        for (int seed = 0; seed < 20; seed++) {
            Random rnd = new Random(100 + seed);
            int d = seed % 2 == 0 ? 40 : 200;
            int p = seed % 2 == 0 ? 120 : 60;
            double[][] phi = columns(rnd, d, p);
            double[] y = columns(rnd, d, 1)[0];
            double[] c0 = atPhi(phi, y);
            double lmax = 0;
            for (double v : c0) lmax = Math.max(lmax, Math.abs(v));
            EsrcSolver.Path r = EsrcSolver.path(gram(phi), c0, 1, qs, 5000);
            assertFalse(r.capped());
            int prev = 0;
            for (int i = 0; i < qs.length; i++) {
                double[] x = r.xs()[i];
                double[] res = residual(phi, x, y);
                double lam = qs[i] * lmax;
                int nz = 0;
                for (int j = 0; j < p; j++) {
                    double cj = dot(phi[j], res);
                    if (x[j] != 0) {
                        nz++;
                        assertEquals("q " + qs[i] + ", активный " + j, lam * Math.signum(x[j]), cj, 1e-9);
                    } else {
                        assertTrue("q " + qs[i] + ", вне активных " + j, Math.abs(cj) <= lam + 1e-9);
                    }
                }
                assertTrue(nz >= prev - 3);
                prev = nz;
            }
        }
    }

    /** Совпадение с ISTA при том же λ. */
    @Test
    public void pathMatchesIsta() {
        Random rnd = new Random(77);
        double[][] phi = columns(rnd, 30, 50);
        double[] y = columns(rnd, 30, 1)[0];
        double[] c0 = atPhi(phi, y);
        double lmax = 0;
        for (double v : c0) lmax = Math.max(lmax, Math.abs(v));
        EsrcSolver.Path r = EsrcSolver.path(gram(phi), c0, 1, new double[] {0.3}, 5000);
        double lam = 0.3 * lmax;
        double[] x = ista(phi, y, lam, 40000);
        assertEquals(objective(phi, x, y, lam), objective(phi, r.xs()[0], y, lam), 1e-6);
    }

    @Test
    public void sciFormula() {
        assertEquals(1.0, XBaseEsrc.sci(new double[] {0, 0.3, 0, 0}, 4), 1e-12);
        assertEquals(0.0, XBaseEsrc.sci(new double[] {0.5, -0.5, 0.5, 0.5}, 4), 1e-12);
        assertEquals(0.0, XBaseEsrc.sci(new double[4], 4), 0);
        assertEquals((4 * 0.6 - 1) / 3.0, XBaseEsrc.sci(new double[] {0.6, -0.2, 0.2, 0}, 4), 1e-12);
    }
}
