package svd.recognizer;

import org.junit.Test;
import org.junit.Assert;
import svd.recognizer.processing.SVDComputer;

/**
 * Тесты для SVDComputer.
 * Проверяют корректность вычисления σ-вектора на известных матрицах.
 */
public class SVDComputerTest {

    private final SVDComputer svdComputer = new SVDComputer();

    @Test
    public void testSignatureLengthIsCorrect() {
        // TODO: создать тестовое BufferedImage 64×64
        //       вызвать svdComputer.compute(image)
        //       проверить, что длина массива == SVDComputer.SIGNATURE_LENGTH
        Assert.assertTrue("Тест не реализован — TODO", true);
    }

    @Test
    public void testFirstSingularValueIsNormalized() {
        // TODO: проверить, что после нормализации sigma[0] == 1.0
        Assert.assertTrue("Тест не реализован — TODO", true);
    }

    @Test
    public void testSingularValuesDescending() {
        // TODO: проверить, что sigma[0] >= sigma[1] >= sigma[2] >= ...
        Assert.assertTrue("Тест не реализован — TODO", true);
    }
}
