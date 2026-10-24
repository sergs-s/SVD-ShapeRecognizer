package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

/**
 * Ревизия 1.2, 2.1 (XBaseRule): на малом примере строка (0) – побайтно прежняя оценка LdaModel.scores; собственные значения S_b по
 * осям убывают; усечение, косинус, веса √λ и WPCA – против прямого счёта.
 *
 * @author ssv
 */
public class XBaseRuleTest {

    static List<List<double[]>> classes(Random rnd, int c, int per, int n) {
        List<List<double[]>> out = new ArrayList<>();
        for (int k = 0; k < c; k++) {
            double[] mu = new double[n];
            for (int i = 0; i < n; i++) mu[i] = rnd.nextGaussian() * 2;
            List<double[]> list = new ArrayList<>();
            for (int j = 0; j < per; j++) {
                double[] v = new double[n];
                for (int i = 0; i < n; i++) v[i] = mu[i] + rnd.nextGaussian();
                list.add(v);
            }
            out.add(list);
        }
        return out;
    }

    static XBaseRule.Fit fit(List<List<double[]>> cls, int gallery, XBaseRule.Pca pca) {
        LdaModel lm = (LdaModel) new MldaScorer().fit(cls, gallery);
        double[][] wc = new double[gallery][];
        for (int c = 0; c < gallery; c++) {
            double[] m = new double[pca.basis().length];
            for (double[] v : cls.get(c)) {
                double[] z = pca.whiten(v);
                for (int j = 0; j < m.length; j++) m[j] += z[j];
            }
            for (int j = 0; j < m.length; j++) m[j] /= cls.get(c).size();
            wc[c] = m;
        }
        return new XBaseRule.Fit(lm, XBaseRule.lambdaB(lm, cls), pca, wc);
    }

    static int idx(String label) {
        for (int v = 0; v < XBaseRule.RVS.size(); v++) if (XBaseRule.RVS.get(v).label().equals(label)) return v;
        throw new IllegalArgumentException(label);
    }

    @Test
    public void rowsAgainstDirect() {
        Random rnd = new Random(7);
        int n = 60;
        List<List<double[]>> cls = classes(rnd, 30, 4, n);
        List<double[]> poolX = new ArrayList<>();
        for (List<double[]> l : classes(rnd, 25, 5, n)) poolX.addAll(l);
        XBaseRule.Pca pca = XBaseRule.Pca.of(poolX);
        XBaseRule.Fit f = fit(cls, 10, pca);
        LdaModel lm = f.lm;
        int d = f.axes();
        assertEquals(29, d);
        for (int j = 1; j < d; j++) assertTrue("λ_b по убыванию: " + j, f.lamB[j] <= f.lamB[j - 1] * (1 + 1e-9));
        assertTrue(f.lamB[d - 1] > 0);
        for (int t = 0; t < 20; t++) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) x[i] = rnd.nextGaussian() * 2;
            double[][] s = f.scores(x);
            double[] old = lm.scores(x);
            for (int c = 0; c < old.length; c++) assertEquals("(0) побайтно", Double.doubleToLongBits(old[c]), Double.doubleToLongBits(s[0][c]));
            double[] y = lm.embed(x);
            double[] z = pca.whiten(x);
            for (int c = 0; c < old.length; c++) {
                double[] mu = lm.centers()[c];
                // (а) k = 20: евклид по первым 20 осям.
                double e = 0;
                for (int j = 0; j < 20; j++) e += (y[j] - mu[j]) * (y[j] - mu[j]);
                assertEquals(Math.sqrt(e), s[idx("(а) k = 20")][c], 1e-12);
                // k = 50, 100 > осей – все оси, как база.
                assertEquals(old[c], s[idx("(а) k = 100")][c], 1e-12);
                // (б): 1 − cos по всем осям и по первым 20.
                assertEquals(1 - dot(y, mu, d) / Math.sqrt(dot(y, y, d) * dot(mu, mu, d)), s[idx("(б) косинус, все оси")][c], 1e-12);
                assertEquals(1 - dot(y, mu, 20) / Math.sqrt(dot(y, y, 20) * dot(mu, mu, 20)), s[idx("(б) косинус, k = 20")][c], 1e-12);
                // (в): координаты, умноженные на √λ.
                double[] yw = new double[d];
                double[] mw = new double[d];
                for (int j = 0; j < d; j++) {
                    yw[j] = y[j] * Math.sqrt(f.lamB[j]);
                    mw[j] = mu[j] * Math.sqrt(f.lamB[j]);
                }
                double ew = 0;
                for (int j = 0; j < d; j++) ew += (yw[j] - mw[j]) * (yw[j] - mw[j]);
                assertEquals(Math.sqrt(ew), s[idx("(в) √λ, евклид")][c], 1e-9 * Math.max(1, Math.sqrt(ew)));
                assertEquals(1 - dot(yw, mw, d) / Math.sqrt(dot(yw, yw, d) * dot(mw, mw, d)), s[idx("(в) √λ, косинус")][c], 1e-12);
                // (г): m = 100 больше ранга пула – все компоненты.
                double[] wc = f.wCenters[c];
                int m = Math.min(100, z.length);
                assertEquals(1 - dot(z, wc, m) / Math.sqrt(dot(z, z, m) * dot(wc, wc, m)), s[idx("(г) WPCA m = 100")][c], 1e-12);
            }
        }
    }

    @Test
    public void whitening() {
        Random rnd = new Random(11);
        List<double[]> xs = new ArrayList<>();
        for (List<double[]> l : classes(rnd, 20, 6, 40)) xs.addAll(l);
        XBaseRule.Pca pca = XBaseRule.Pca.of(xs);
        int r = pca.basis().length;
        assertEquals(40, r);
        double[][] g = new double[r][r];
        for (double[] x : xs) {
            double[] z = pca.whiten(x);
            for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) g[a][b] += z[a] * z[b];
        }
        for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) assertEquals(a == b ? 1 : 0, g[a][b], 1e-8);
    }

    @Test
    public void choice() {
        // База в пределах 1 п. п. – база; иначе первая по порядку в пределах 1 п. п. от лучшего.
        double[] c = new double[XBaseRule.RVS.size()];
        java.util.Arrays.fill(c, 30);
        c[0] = 20.5;
        c[idx("(г) WPCA m = 200")] = 19.8;
        assertEquals(0, XBaseRule.choose(c));
        c[0] = 25;
        c[idx("(в) √λ, косинус")] = 20.7;
        assertEquals(idx("(в) √λ, косинус"), XBaseRule.choose(c));
        c[idx("(а) k = 50")] = 20.5;
        assertEquals(idx("(а) k = 50"), XBaseRule.choose(c));
        c[idx("(б) косинус, все оси")] = 20.6;
        assertEquals(idx("(б) косинус, все оси"), XBaseRule.choose(c));
    }

    static double dot(double[] a, double[] b, int k) {
        double s = 0;
        for (int j = 0; j < k; j++) s += a[j] * b[j];
        return s;
    }
}
