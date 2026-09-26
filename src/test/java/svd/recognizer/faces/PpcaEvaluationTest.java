package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import svd.recognizer.faces.FaceEvaluation.Role;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceTrainer;

/**
 * Проверка PPCA-скоринга на синтетике (без OpenCV и базы).
 *
 * @author ssv
 */
public class PpcaEvaluationTest {

    private final SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());

    private static final double[][] VECTORS = {
        {1, 2, 0, 1, 3, 0},
        {2, 0, 1, 0, 1, 1},
        {0, 1, 2, 2, 0, 1},
        {3, 1, 1, 0, 2, 2}
    };

    @Test
    public void eigenvaluesFromProjectionsEqualSigmaSquaredOverNMinusOne() {
        SubspaceModel model = trainer.train(VECTORS, 3);
        double[] lambda = PpcaEvaluation.eigenvalues(VECTORS, model.getMeanVector(), model.getBasisMatrix(), 3);
        double[] mean = model.getMeanVector();
        double[][] centered = new double[6][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 6; j++) {
                centered[j][i] = VECTORS[i][j] - mean[j];
            }
        }
        double[] sigma = new CommonsMathSvdEngine().decompose(centered).getSingularValues();
        for (int i = 0; i < 3; i++) {
            assertEquals(sigma[i] * sigma[i] / 3, lambda[i], 1e-9);
        }
    }

    @Test
    public void looNoiseVarianceIsZeroForCollinearVectors() {
        double[][] line = {{0, 0, 0}, {1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
        // Все снимки на одной прямой: подпространство без одного снимка (k′ = 2) её содержит.
        assertEquals(0.0, PpcaEvaluation.looNoiseVariance(trainer, line, 3, 3 + 10), 1e-18);
    }

    @Test
    public void shrinkInterpolatesTowardMean() {
        double[] lambda = {4, 2, 0};
        assertArrayEquals(lambda, PpcaEvaluation.shrink(lambda, 0), 0.0);
        assertArrayEquals(new double[] {2, 2, 2}, PpcaEvaluation.shrink(lambda, 1), 1e-12);
        assertArrayEquals(new double[] {3, 2, 1}, PpcaEvaluation.shrink(lambda, 0.5), 1e-12);
    }

    @Test
    public void scoreInsideSubspaceIsWeightedInnerTerm() {
        // Снимок в подпространстве: ε = 0, d_w = w·Σ y²/λ̃.
        PpcaEvaluation.ProbeData p = new PpcaEvaluation.ProbeData(Role.KNOWN_TEST, 0,
                new double[] {0.0}, new double[][] {{2, 1}});
        PpcaEvaluation.Gallery g = new PpcaEvaluation.Gallery(0, new int[] {0},
                new double[][] {{4, 1}}, new double[] {0.5}, 0.5, 100, 2, List.of(p));
        double d = PpcaEvaluation.modelScore(g, p, 0, PpcaEvaluation.Variant.D_OWN, 10, new double[] {4, 1});
        assertEquals(10 * (4.0 / 4 + 1.0 / 1), d, 1e-12);
        // Вне подпространства без проекций: d_w = ε²/s² при любом w.
        PpcaEvaluation.ProbeData q = new PpcaEvaluation.ProbeData(Role.KNOWN_TEST, 0,
                new double[] {3.0}, new double[][] {{0, 0}});
        assertEquals(9 / 0.5, PpcaEvaluation.modelScore(g, q, 0, PpcaEvaluation.Variant.D_GLOBAL, 100,
                new double[] {4, 1}), 1e-12);
    }

    @Test
    public void curveValidationImageIsOutsideTrainingAndControl() {
        int[] order = new EvaluationProtocol(42L, 5).imageOrder(3);
        for (int r = 0; r < TrainingSizeCurve.ROTATIONS; r++) {
            Set<Integer> used = new HashSet<>();
            for (int image : TrainingSizeCurve.trainImages(order, r, PpcaEvaluation.CURVE_N)) {
                used.add(image);
            }
            used.add(TrainingSizeCurve.testImage(order, r));
            assertEquals(PpcaEvaluation.CURVE_N + 1, used.size());
            assertFalse(used.contains(PpcaEvaluation.validationImage(order, r)));
        }
    }
}
