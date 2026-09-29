package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;

/**
 * Z-norm (этап 3): μᵢ и σᵢ (N − 1) по когорте, zᵢ = (sᵢ − μᵢ)/σᵢ, человек без модели — +∞.
 *
 * @author ssv
 */
public class ZNormScorerTest {

    @Test
    public void zNormStats() {
        double[][] s = {{1, 10, Double.POSITIVE_INFINITY}, {3, 14, Double.POSITIVE_INFINITY}, {5, 12, Double.POSITIVE_INFINITY}};
        double[] mu = new double[3];
        double[] sigma = new double[3];
        ZNormScorer.stats(s, mu, sigma);
        assertEquals(3, mu[0], 1e-12);
        assertEquals(2, sigma[0], 1e-12);
        assertEquals(12, mu[1], 1e-12);
        assertEquals(2, sigma[1], 1e-12);
        assertArrayEquals(new double[] {-1, 0.5, Double.POSITIVE_INFINITY}, ZNormScorer.normalize(new double[] {1, 13, 7}, mu, sigma), 1e-12);
    }

    @Test
    public void scoresAreNormalizedOnCohortOfEachModel() {
        // Базовая оценка — сам вектор (оценка по «людям» = координаты); когорта: {0, 0}, {2, 4}.
        GalleryScorer identity = new GalleryScorer() {
            @Override
            public String label() {
                return "id";
            }

            @Override
            public ScoreModel fit(List<List<double[]>> classes, int gallery) {
                return x -> x.clone();
            }
        };
        ZNormScorer z = new ZNormScorer(identity, List.of(new double[] {0, 0}, new double[] {2, 4}), "тест");
        GalleryScorer.ScoreModel m = z.fit(List.of(), 2);
        // μ = {1, 2}, σ = {√2, 2√2}.
        assertArrayEquals(new double[] {1 / Math.sqrt(2), 0}, m.scores(new double[] {2, 2}), 1e-12);
    }
}
