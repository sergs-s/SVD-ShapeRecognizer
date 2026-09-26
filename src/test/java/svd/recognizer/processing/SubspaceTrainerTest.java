package svd.recognizer.processing;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import org.junit.Test;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;

/**
 * Проверка обобщённого ядра подпространств (без OpenCV).
 *
 * @author ssv
 */
public class SubspaceTrainerTest {

    private final SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());

    @Test
    public void vectorizesRectangularMatrixRowMajor() {
        double[][] matrix = {
            {1, 2, 3},
            {4, 5, 6}
        };
        assertArrayEquals(new double[] {1, 2, 3, 4, 5, 6}, ImageVectorizer.toVector(matrix), 0.0);
    }

    @Test
    public void basisIsLimitedByNMinusOne() {
        // 3 вектора длины 5: ранг центрированной матрицы ≤ 2, поэтому k = 10 → 2.
        double[][] vectors = {
            {1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 1, 0, 0}
        };
        SubspaceModel model = trainer.train(vectors, 10);
        assertEquals(2, model.getK());
        assertEquals(5, model.getMeanVector().length);
        // Обучающие векторы лежат в аффинной оболочке — ошибка реконструкции нулевая.
        for (double[] v : vectors) {
            assertEquals(0.0, new SubspaceRecognizer(1.0).reconstructionError(v, model), 1e-9);
        }
    }

    @Test
    public void energyKPicksSmallestSufficientK() {
        double[] sigma = {3, 2, 1, 0}; // энергии 9, 4, 1 из 14 (σ₄ вне ранга N − 1 = 3)
        assertEquals(1, SubspaceTrainer.energyK(sigma, 3, 0.60)); // 9/14 ≈ 0.64
        assertEquals(2, SubspaceTrainer.energyK(sigma, 3, 0.90)); // 13/14 ≈ 0.93
        assertEquals(3, SubspaceTrainer.energyK(sigma, 3, 0.95));
        assertEquals(1, SubspaceTrainer.energyK(new double[] {0, 0}, 1, 0.9));
    }

    @Test(expected = IllegalArgumentException.class)
    public void singleVectorIsRejected() {
        trainer.train(new double[][] {{1, 2, 3}}, 1);
    }
}
