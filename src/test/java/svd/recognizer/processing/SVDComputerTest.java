package svd.recognizer.processing;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import svd.recognizer.math.CommonsMathSvdEngine;

/**
 * Базовый тест вычисления SVD-признаков.
 *
 * Проверяется:
 * 1. Возможность вычислить требуемое число признаков
 * 2. Нормализация по первому сингулярному значению
 * 3. Убывание сингулярных значений после нормализации
 *
 * @author ssv
 */
public class SVDComputerTest {
    @Test
    public void testComputeFeatures() {
        double[][] matrix = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };
        SVDComputer computer = new SVDComputer(new CommonsMathSvdEngine());
        double[] features = computer.computeFeatures(matrix, 3);
        assertEquals(3, features.length);
        assertEquals(1.0, features[0], 1e-9);
        assertTrue(features[0] >= features[1]);
        assertTrue(features[1] >= features[2]);
    }
}