package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import java.util.List;
import org.junit.Test;

/**
 * Проверка выбора пары порогов двумерного правила (без OpenCV и базы).
 *
 * @author ssv
 */
public class TwoThresholdEvaluationTest {

    @Test
    public void theta2CountsAllImpostorAttemptsInDenominator() {
        // 10 чужих: 5 с ε₁ = 1 (проходят θ₁ = 2), 5 с ε₁ = 5 (отсекаются θ₁).
        double[] e1 = {1, 1, 1, 1, 1, 5, 5, 5, 5, 5};
        double[] ratio = {0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1};
        // α = 0.2: можно принять ⌊0.2·10⌋ = 2 → θ₂ чуть меньше третьего из прошедших (0.7).
        double theta2 = TwoThresholdEvaluation.theta2(e1, ratio, 2.0, 0.2);
        assertTrue(theta2 >= 0.6 && theta2 < 0.7);
        // Без ограничения по θ₁ проходят все: θ₂ чуть меньше третьего наименьшего (0.1).
        assertTrue(TwoThresholdEvaluation.theta2(e1, ratio, Double.POSITIVE_INFINITY, 0.2) < 0.1);
        // θ₁ отсекает всех, кроме двух допустимых → θ₂ = ∞.
        assertEquals(Double.POSITIVE_INFINITY, TwoThresholdEvaluation.theta2(e1, ratio, 0.5, 0.2), 0.0);
    }

    @Test
    public void quantileIsNearestRank() {
        double[] values = new double[100];
        for (int i = 0; i < 100; i++) {
            values[i] = i + 1;
        }
        assertEquals(90.0, TwoThresholdEvaluation.quantile(values, 0.90), 0.0);
        assertEquals(99.0, TwoThresholdEvaluation.quantile(values, 0.99), 0.0);
    }

    @Test
    public void tiesAreResolvedInFavourOfBase() {
        TwoThresholdEvaluation.Pair base = new TwoThresholdEvaluation.Pair(Double.NaN,
                Double.POSITIVE_INFINITY, 0.8, 0.10, 0.0);
        TwoThresholdEvaluation.Pair same = new TwoThresholdEvaluation.Pair(0.95, 10, 0.9, 0.10, 0.3);
        TwoThresholdEvaluation.Pair better = new TwoThresholdEvaluation.Pair(0.99, 12, 0.85, 0.08, 0.2);
        assertSame(base, TwoThresholdEvaluation.select(List.of(base, same)));
        assertSame(better, TwoThresholdEvaluation.select(List.of(base, same, better)));
    }
}
