package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;

/**
 * Объединение методов и блочные области (этап 4): нормировка частей по когорте, сумма, правило согласия; вырез области.
 *
 * @author ssv
 */
public class FusionScorerTest {

    private static final double INF = Double.POSITIVE_INFINITY;

    @Test
    public void sumOfNormalizedScores() {
        double[][] raw = {{1, 13}, {4, 2}};
        double[][] mu = {{3, 12}, {2, 2}};
        double[][] sigma = {{2, 2}, {1, 4}};
        // z части 1: (−1, 0,5); части 2: (2, 0) → сумма (1, 0,5).
        assertArrayEquals(new double[] {1, 0.5}, FusionScorer.combine(raw, mu, sigma, FusionScorer.Mode.SUM), 1e-12);
    }

    @Test
    public void agreementRule() {
        double[][] mu = {{0, 0, 0}, {0, 0, 0}};
        double[][] sigma = {{1, 1, 1}, {2, 2, 2}};
        // argmin обеих частей — человек 1: итог — сумма z, прочие — MAX_VALUE.
        double[][] agree = {{3, 1, 2}, {4, 0, 6}};
        assertArrayEquals(new double[] {Double.MAX_VALUE, 1, Double.MAX_VALUE},
                FusionScorer.combine(agree, mu, sigma, FusionScorer.Mode.AGREE), 0);
        // argmin по сырым оценкам: 1 и 0 — несогласие, отказ (+∞ у всех), хотя сумма z определена.
        double[][] disagree = {{3, 1, 2}, {0, 1, 6}};
        assertArrayEquals(new double[] {INF, INF, INF}, FusionScorer.combine(disagree, mu, sigma, FusionScorer.Mode.AGREE), 0);
    }

    @Test
    public void fitNormalizesEachPartOnCohort() {
        // Части: оценка = вектор и оценка = 2·вектор; когорта {0, 0}, {2, 4}: μ = (1, 2) и (2, 4), σ = (√2, 2√2) и (2√2, 4√2).
        GalleryScorer id = scorer(1);
        GalleryScorer twice = scorer(2);
        List<double[]> cohort = List.of(new double[] {0, 0}, new double[] {2, 4});
        GalleryScorer.ScoreModel m = new FusionScorer(List.of(id, twice), cohort, "тест", FusionScorer.Mode.SUM).fit(List.of(), 2);
        double s = Math.sqrt(2);
        // z обеих частей одинаковы (масштаб не влияет): ((3 − 1)/√2, (2 − 2)/(2√2)) → сумма вдвое.
        assertArrayEquals(new double[] {2 * 2 / s, 0}, m.scores(new double[] {3, 2}), 1e-12);
    }

    @Test
    public void regionCrop() {
        double[] v = new double[RegionScorer.FRAME_W * RegionScorer.FRAME_H];
        for (int i = 0; i < v.length; i++) v[i] = i;
        double[] c = new RegionScorer.Region("t", "тест", 2, 3, 2, 2).crop(v);
        assertArrayEquals(new double[] {3 * 92 + 2, 3 * 92 + 3, 4 * 92 + 2, 4 * 92 + 3}, c, 0);
        assertEquals(76 * 26, RegionScorer.EYES.crop(v).length);
    }

    private static GalleryScorer scorer(double k) {
        return new GalleryScorer() {
            @Override
            public String label() {
                return "×" + k;
            }

            @Override
            public ScoreModel fit(List<List<double[]>> classes, int gallery) {
                return x -> new double[] {k * x[0], k * x[1]};
            }
        };
    }
}
