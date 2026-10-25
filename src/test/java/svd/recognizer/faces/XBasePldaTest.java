package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

/**
 * PLDA (XBasePlda): LLR в замкнутой форме против совместной гауссовой плотности, оценка S_b и S_w моментами, опознавание, формы для
 * разбора (п. 2.2, ответы Жорика 25.10).
 *
 * @author ssv
 */
public class XBasePldaTest {

    static final int D = 8;

    static double logpdf(double[] x, double[] mean, double[][] cov) {
        int n = x.length;
        CholeskyDecomposition ch = new CholeskyDecomposition(new Array2DRowRealMatrix(cov));
        RealMatrix l = ch.getL();
        double[] r = new double[n];
        for (int i = 0; i < n; i++) r[i] = x[i] - mean[i];
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            double s = r[i];
            for (int k = 0; k < i; k++) s -= l.getEntry(i, k) * z[k];
            z[i] = s / l.getEntry(i, i);
        }
        double q = 0;
        double ld = 0;
        for (int i = 0; i < n; i++) {
            q += z[i] * z[i];
            ld += 2 * Math.log(l.getEntry(i, i));
        }
        return -0.5 * (q + ld + n * Math.log(2 * Math.PI));
    }

    /** log p(x₁ … x_k) под одним классом: блоки S_b + S_w на диагонали, S_b вне. */
    static double joint(List<double[]> xs, double[] m, double[][] sb, double[][] sw) {
        int k = xs.size();
        double[] x = new double[k * D];
        double[] mu = new double[k * D];
        double[][] c = new double[k * D][k * D];
        for (int a = 0; a < k; a++) {
            System.arraycopy(xs.get(a), 0, x, a * D, D);
            System.arraycopy(m, 0, mu, a * D, D);
            for (int b = 0; b < k; b++) for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) c[a * D + i][b * D + j] = sb[i][j] + (a == b ? sw[i][j] : 0);
        }
        return logpdf(x, mu, c);
    }

    static double[] ones(double v) {
        double[] o = new double[D];
        java.util.Arrays.fill(o, v);
        return o;
    }

    /** (i) LLR кода = log p(y, X | один класс) − log p(y) − log p(X), n = 1 и n = 3. */
    @Test
    public void llrMatchesJoint() {
        Random rnd = new Random(21);
        double[][] sb = XBaseKissTest.spd(rnd, D, 0.7, ones(0.5));
        double[][] sw = XBaseKissTest.spd(rnd, D, 0.5, ones(0.3));
        double[] m = new double[D];
        for (int i = 0; i < D; i++) m[i] = rnd.nextGaussian();
        XBasePlda.Scorer sc = new XBasePlda.Scorer(new XBasePlda.Model(m, sb, sw, 0, new double[2], new double[2]));
        for (int n : new int[] {1, 3}) {
            for (int rep = 0; rep < 5; rep++) {
                List<double[]> xs = new ArrayList<>();
                double[] s = new double[D];
                for (int a = 0; a < n; a++) {
                    double[] v = new double[D];
                    for (int i = 0; i < D; i++) v[i] = m[i] + 2 * rnd.nextGaussian();
                    xs.add(v);
                    for (int i = 0; i < D; i++) s[i] += v[i];
                }
                double[] y = new double[D];
                for (int i = 0; i < D; i++) y[i] = m[i] + 2 * rnd.nextGaussian();
                List<double[]> xy = new ArrayList<>(xs);
                xy.add(y);
                double want = joint(xy, m, sb, sw) - joint(List.of(y), m, sb, sw) - joint(xs, m, sb, sw);
                double got = sc.llr(y, s, n);
                assertEquals("n = " + n, want, got, 1e-8 * Math.max(1, Math.abs(want)));
            }
        }
    }

    static List<List<double[]>> classes(Random rnd, int cls, double[][] cb, double[][] cw, double[] m) {
        List<List<double[]>> out = new ArrayList<>();
        for (int c = 0; c < cls; c++) {
            double[] mu = XBaseKissTest.draw(rnd, cb, m);
            int n = 4 + rnd.nextInt(3);
            List<double[]> l = new ArrayList<>();
            for (int i = 0; i < n; i++) l.add(XBaseKissTest.draw(rnd, cw, mu));
            out.add(l);
        }
        return out;
    }

    /**
     * (ii) Моменты по 5000 классам (4–6 снимков; на 1000 классах выборочная ошибка ковариации средних в размерности 8 – около 11 %, порог
     * 10 % там не отделяет смещение от разброса): S_w и поправленная S_b сходятся к заданным (≤ 10 % по Фробениусу), а без поправки
     * (S_b + (1/C) Σ S_w / n_c) ошибка больше.
     */
    @Test
    public void moments() {
        Random rnd = new Random(23);
        double[][] sb = XBaseKissTest.spd(rnd, D, 0.7, ones(1.0));
        double[][] sw = XBaseKissTest.spd(rnd, D, 0.5, ones(0.5));
        double[] m = new double[D];
        List<List<double[]>> cls = classes(rnd, 5000, XBaseKissTest.chol(sb), XBaseKissTest.chol(sw), m);
        XBasePlda.Model md = XBasePlda.fit(cls);
        double eb = XBaseKissTest.frob(XBaseKissTest.minus(md.sb(), sb)) / XBaseKissTest.frob(sb);
        double ew = XBaseKissTest.frob(XBaseKissTest.minus(md.sw(), sw)) / XBaseKissTest.frob(sw);
        double invN = 0;
        for (List<double[]> c : cls) invN += 1.0 / c.size();
        invN /= cls.size();
        double[][] raw = new double[D][D];
        for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) raw[i][j] = md.sb()[i][j] + md.sw()[i][j] * invN;
        double er = XBaseKissTest.frob(XBaseKissTest.minus(raw, sb)) / XBaseKissTest.frob(sb);
        assertTrue("S_b " + eb, eb < 0.10);
        assertTrue("S_w " + ew, ew < 0.10);
        assertTrue("без поправки " + er + ", с поправкой " + eb, er > eb);
        assertEquals(0, md.raised());
    }

    /** (iii) Опознавание на 200 новых классах (шаблон – 3 снимка, пробы – 3 других): PLDA (Р1) не хуже евклида до центра. */
    @Test
    public void identify() {
        Random rnd = new Random(25);
        double[] dw = new double[D];
        double[] db = new double[D];
        for (int i = 0; i < D; i++) {
            dw[i] = i < 3 ? 9 : 0.5;
            db[i] = i < 3 ? 0.5 : 4;
        }
        double[][] cw = XBaseKissTest.chol(XBaseKissTest.spd(rnd, D, 0.3, dw));
        double[][] cb = XBaseKissTest.chol(XBaseKissTest.spd(rnd, D, 0.3, db));
        double[] m = new double[D];
        XBasePlda.Scorer sc = new XBasePlda.Scorer(XBasePlda.fit(classes(rnd, 1000, cb, cw, m)));
        int g = 200;
        double[][] sum = new double[g][D];
        double[][] tpl = new double[g][];
        List<List<double[]>> probes = new ArrayList<>();
        for (int c = 0; c < g; c++) {
            double[] mu = XBaseKissTest.draw(rnd, cb, m);
            for (int i = 0; i < 3; i++) {
                double[] v = XBaseKissTest.draw(rnd, cw, mu);
                for (int j = 0; j < D; j++) sum[c][j] += v[j];
            }
            tpl[c] = sc.template(sum[c], 3);
            List<double[]> pr = new ArrayList<>();
            for (int i = 0; i < 3; i++) pr.add(XBaseKissTest.draw(rnd, cw, mu));
            probes.add(pr);
        }
        double[][] p3 = sc.forN(3)[1];
        int okP = 0;
        int okE = 0;
        for (int c = 0; c < g; c++) {
            for (double[] y : probes.get(c)) {
                double[] py = XBasePlda.mul(p3, y);
                double prior = sc.prior(y);
                int bp = 0;
                int be = 0;
                double bestP = Double.NEGATIVE_INFINITY;
                double bestE = Double.POSITIVE_INFINITY;
                for (int q = 0; q < g; q++) {
                    double l = sc.llr(py, tpl[q], 3, prior);
                    if (l > bestP) {
                        bestP = l;
                        bp = q;
                    }
                    double e = 0;
                    for (int j = 0; j < D; j++) e += (y[j] - sum[q][j] / 3) * (y[j] - sum[q][j] / 3);
                    if (e < bestE) {
                        bestE = e;
                        be = q;
                    }
                }
                if (bp == c) okP++;
                if (be == c) okE++;
            }
        }
        assertTrue("PLDA " + okP + ", евклид " + okE, okP >= okE);
    }

    /** (iv) Формы для разбора сохраняют порядок: (-а) – по LLR₁, (-р) – по LLR₁ − LLR₂; счёт ограничений ±700. */
    @Test
    public void evalForms() {
        int[] cl = new int[1];
        double[] a = XBasePlda.evalA(new double[] {5, -3, 12}, cl);
        assertEquals(1 + 700 - 12, a[2], 0);
        assertEquals(0, cl[0]);
        XBasePlda.evalA(new double[] {800, 1}, cl);
        assertEquals(1, cl[0]);
        double[] a1 = XBasePlda.evalA(new double[] {30, 0}, new int[1]);
        double[] a2 = XBasePlda.evalA(new double[] {40, 0}, new int[1]);
        assertTrue(a2[0] < a1[0]);
        int[] cr = new int[1];
        double[] r = XBasePlda.evalR(new double[] {5, -3, 12}, cr);
        assertEquals(1, r[2], 0);
        assertEquals(1 + 7, r[0], 0);
        assertEquals(1 + 15, r[1], 0);
        XBasePlda.evalR(new double[] {900, 0}, cr);
        assertEquals(1, cr[0]);
        double[] r1 = XBasePlda.evalR(new double[] {10, 2}, new int[1]);
        double[] r2 = XBasePlda.evalR(new double[] {10, 4}, new int[1]);
        assertTrue(r1[0] / r1[1] < r2[0] / r2[1]);
    }
}
