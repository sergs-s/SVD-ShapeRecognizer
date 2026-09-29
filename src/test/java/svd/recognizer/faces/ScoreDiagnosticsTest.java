package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;

/**
 * Проверка AUC, отношения ε₁/ε₂ по людям и базовой линии «ближайший вектор» (без OpenCV и базы).
 *
 * @author ssv
 */
public class ScoreDiagnosticsTest {

    @Test
    public void aucSeparatedReversedAndTied() {
        assertEquals(1.0, ScoreDiagnostics.auc(new double[] {1, 2}, new double[] {3, 4, 5}), 0.0);
        assertEquals(0.0, ScoreDiagnostics.auc(new double[] {6, 7}, new double[] {3, 4, 5}), 0.0);
        assertEquals(0.5, ScoreDiagnostics.auc(new double[] {2, 2}, new double[] {2, 2, 2}), 0.0);
        // Свой 2: чужие 1 (меньше), 2 (ничья), 3 (больше) → (0 + 0.5 + 1) / 3.
        assertEquals(0.5, ScoreDiagnostics.auc(new double[] {2}, new double[] {1, 2, 3}), 1e-12);
        assertEquals(Double.NaN, ScoreDiagnostics.auc(new double[0], new double[] {1}), 0.0);
    }

    @Test
    public void ratioOfArgminEqualsEps1OverEps2() {
        double[] s = SubspaceScorer.ratio(new double[] {4, 2, 8, Double.POSITIVE_INFINITY});
        // argmin — 1: 2/4; остальные — εᵢ / ε_argmin.
        assertArrayEquals(new double[] {2, 0.5, 4, Double.POSITIVE_INFINITY}, s, 1e-12);
    }

    @Test
    public void nearestVectorAndMean() {
        List<List<double[]>> classes = List.of(
                List.of(new double[] {0, 0}, new double[] {2, 0}),
                List.of(new double[] {10, 0}));
        double[] x = {0, 0};
        assertArrayEquals(new double[] {0, 10}, new NearestVectorScorer(false).fit(classes, 2).scores(x), 1e-12);
        assertArrayEquals(new double[] {1, 10}, new NearestVectorScorer(true).fit(classes, 2).scores(x), 1e-12);
        // Посторонние (третий класс) — не кандидаты.
        List<List<double[]>> withOutsider = List.of(classes.get(0), classes.get(1), List.of(new double[] {0, 0}));
        assertEquals(2, new NearestVectorScorer(false).fit(withOutsider, 2).scores(x).length);
    }
}
