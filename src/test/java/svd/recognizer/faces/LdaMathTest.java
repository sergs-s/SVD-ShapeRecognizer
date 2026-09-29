package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

/**
 * Проверка LDA (Fisherfaces, MLDA в оболочке против полного пространства) на синтетике (без OpenCV и базы).
 *
 * @author ssv
 */
public class LdaMathTest {

    /** classes классов по perClass векторов размерности n: центр класса + шум. */
    private static List<List<double[]>> synthetic(int classes, int perClass, int n, double spread, long seed) {
        Random rnd = new Random(seed);
        List<List<double[]>> out = new ArrayList<>();
        for (int c = 0; c < classes; c++) {
            double[] center = new double[n];
            for (int i = 0; i < n; i++) center[i] = 3 * rnd.nextGaussian();
            List<double[]> list = new ArrayList<>();
            for (int k = 0; k < perClass; k++) {
                double[] x = new double[n];
                for (int i = 0; i < n; i++) x[i] = center[i] + spread * rnd.nextGaussian();
                list.add(x);
            }
            out.add(list);
        }
        return out;
    }

    @Test
    public void fisherSeparatesClassesAndWhitensWithin() {
        List<List<double[]>> classes = synthetic(3, 10, 8, 0.3, 1);
        LdaModel m = (LdaModel) new FisherScorer().fit(classes, 3);
        assertEquals(2, m.projection().length);
        for (int c = 0; c < 3; c++) {
            for (double[] x : classes.get(c)) {
                double[] s = m.scores(x);
                for (int o = 0; o < 3; o++) if (o != c) assertTrue(s[c] < s[o]);
            }
        }
        // AᵀS_wA = I в пространстве PCA: разброс внутри классов по каждой оси LDA — единичный.
        LdaMath.Data d = LdaMath.Data.of(classes);
        LdaMath.Span span = LdaMath.span(d, 1e-12, d.x().length - d.classes());
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(span.y(), d.labels(), means);
        LdaMath.Axes axes = LdaMath.fisherAxes(sw, means, d.labels(), 2, 0);
        double[][] w = LdaMath.multiply(LdaMath.transpose(axes.a()), LdaMath.multiply(sw, axes.a()));
        assertArrayEquals(new double[] {1, 0}, w[0], 1e-9);
        assertArrayEquals(new double[] {0, 1}, w[1], 1e-9);
    }

    @Test
    public void pcaByEnergyTakesSmallestKReachingShareAndAtMostNMinusC() {
        // N = 4·6 = 24 < n = 60; N − C = 20.
        List<List<double[]>> classes = synthetic(4, 6, 60, 1.0, 5);
        LdaMath.Data d = LdaMath.Data.of(classes);
        double[][] c = new double[d.x().length][];
        for (int k = 0; k < c.length; k++) {
            c[k] = d.x()[k].clone();
            for (int i = 0; i < c[k].length; i++) c[k][i] -= d.mean()[i];
        }
        double[] sigma = new org.apache.commons.math3.linear.SingularValueDecomposition(
                new org.apache.commons.math3.linear.Array2DRowRealMatrix(c, false)).getSingularValues();
        double total = 0;
        for (double v : sigma) total += v * v;
        for (double share : new double[] {0.5, 0.8, 0.95, 1.0}) {
            int expected = 0;
            double acc = 0;
            while (acc < share * total && expected < sigma.length) acc += sigma[expected] * sigma[expected++];
            expected = Math.min(expected, d.x().length - d.classes());
            String info = FisherScorer.byEnergy(share).fit(classes, 4).info();
            assertTrue(info, info.contains("PCA " + expected + " (N − C = 20"));
        }
    }

    @Test
    public void smallBetweenScatterDecompositionEqualsFull() {
        List<List<double[]>> classes = synthetic(4, 5, 6, 0.5, 4);
        LdaMath.Data d = LdaMath.Data.of(classes);
        LdaMath.Span span = LdaMath.span(d, 1e-12, d.x().length - d.classes());
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(span.y(), d.labels(), means);
        LdaMath.Axes fast = LdaMath.fisherAxes(sw, means, d.labels(), 3, 0);
        LdaMath.Axes full = LdaMath.fisherAxesFull(sw, LdaMath.scatterBetween(means, d.labels()), 3, 0);
        assertTrue(LdaMath.maxPrincipalSin(LdaMath.transpose(fast.a()), LdaMath.transpose(full.a())) < 1e-9);
    }

    @Test
    public void mldaInSpanEqualsFullSpace() {
        // N = 3·6 = 18 < n = 40: оболочка — настоящее подпространство.
        List<List<double[]>> classes = synthetic(3, 6, 40, 1.0, 2);
        LdaModel span = (LdaModel) new MldaScorer().fit(classes, 3);
        LdaModel full = MldaCheck.full(classes, 3);
        assertTrue(LdaMath.maxPrincipalSin(span.projection(), full.projection()) < 1e-8);
        Random rnd = new Random(3);
        for (int k = 0; k < 20; k++) {
            double[] x = new double[40];
            for (int i = 0; i < 40; i++) x[i] = 3 * rnd.nextGaussian();
            assertArrayEquals(full.scores(x), span.scores(x), 1e-8);
        }
    }
}
