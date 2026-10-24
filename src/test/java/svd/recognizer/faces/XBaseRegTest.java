package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.junit.Test;

/**
 * Ревизия 1.3 (MldaScorer(c), XBaseReg): при c = 1 – побайтно прежняя модель; число заменённых собственных значений не
 * убывает с c; расчёт в оболочке при c ≠ 1 совпадает с прямым расчётом в полном пространстве с тем же полом; правило выбора c.
 *
 * @author ssv
 */
public class XBaseRegTest {

    static List<List<double[]>> classes(long seed, int c, int per, int n) {
        return XBaseRuleTest.classes(new Random(seed), c, per, n);
    }

    static int replaced(String info) {
        String s = info.substring(info.indexOf("заменено собственных значений в оболочке ") + "заменено собственных значений в оболочке ".length());
        return Integer.parseInt(s.substring(0, s.indexOf(' ')));
    }

    @Test
    public void defaultIsCOne() {
        List<List<double[]>> cls = classes(5, 12, 4, 60);
        LdaModel a = (LdaModel) new MldaScorer().fit(cls, 12);
        LdaModel b = (LdaModel) new MldaScorer(1.0).fit(cls, 12);
        assertEquals(a.info(), b.info());
        Random rnd = new Random(9);
        for (int t = 0; t < 10; t++) {
            double[] x = new double[60];
            for (int i = 0; i < 60; i++) x[i] = rnd.nextGaussian() * 2;
            double[] u = a.scores(x);
            double[] v = b.scores(x);
            for (int k = 0; k < u.length; k++) assertEquals(Double.doubleToLongBits(u[k]), Double.doubleToLongBits(v[k]));
        }
    }

    @Test
    public void replacedMonotone() {
        List<List<double[]>> cls = classes(6, 10, 6, 80);
        int prev = -1;
        for (double c : XBaseReg.CS) {
            String info = new MldaScorer(c).fit(cls, 10) instanceof LdaModel m ? m.info() : "";
            int r = replaced(info);
            assertTrue("c = " + c + ": " + r + " < " + prev, r >= prev);
            prev = r;
        }
    }

    /** Прямой расчёт в полном пространстве с полом c · tr(S_w)/n (как MldaCheck.full). */
    static LdaModel full(List<List<double[]>> classes, int gallery, double c) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int n = d.mean().length;
        double[][] y = new double[d.x().length][n];
        for (int k = 0; k < y.length; k++) for (int i = 0; i < n; i++) y[k][i] = d.x()[k][i] - d.mean()[i];
        double[][] means = LdaMath.classMeans(y, d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(y, d.labels(), means);
        double trace = 0;
        for (int i = 0; i < n; i++) trace += sw[i][i];
        LdaMath.Axes axes = LdaMath.fisherAxesFull(sw, LdaMath.scatterBetween(means, d.labels()), d.classes() - 1, c * (trace / n));
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.transpose(axes.a()), String.format(Locale.ROOT, "c %s", c));
    }

    @Test
    public void spanEqualsFullSpace() {
        // N = 3·6 = 18 < n = 40: оболочка – настоящее подпространство.
        List<List<double[]>> classes = classes(2, 3, 6, 40);
        for (double c : XBaseReg.CS) {
            LdaModel span = (LdaModel) new MldaScorer(c).fit(classes, 3);
            LdaModel full = full(classes, 3, c);
            assertTrue("c = " + c, LdaMath.maxPrincipalSin(span.projection(), full.projection()) < 1e-8);
            Random rnd = new Random(3);
            for (int k = 0; k < 10; k++) {
                double[] x = new double[40];
                for (int i = 0; i < 40; i++) x[i] = 3 * rnd.nextGaussian();
                assertArrayEquals(full.scores(x), span.scores(x), 1e-8);
            }
        }
    }

    @Test
    public void choice() {
        // c = 1 в пределах 1 п. п. от лучшего – c = 1.
        assertEquals(1, XBaseReg.choose(new double[] {20.0, 20.9, 25, 30}));
        // Иначе – наименьшее |ln c| среди близких: 3 (ln 3 ≈ 1,10) ближе к 1, чем 0,3 (|ln 0,3| ≈ 1,20).
        assertEquals(2, XBaseReg.choose(new double[] {20.0, 22, 20.5, 25}));
        assertEquals(0, XBaseReg.choose(new double[] {20.0, 22, 21.5, 25}));
        assertEquals(3, XBaseReg.choose(new double[] {30, 30, 30, 20}));
    }
}
