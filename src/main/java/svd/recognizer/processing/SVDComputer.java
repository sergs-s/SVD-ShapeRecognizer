package svd.recognizer.processing;

import java.awt.image.BufferedImage;

/**
 * Вычисление SVD-сигнатуры изображения.
 *
 * Принимает нормализованное изображение 64×64 (из ImagePreprocessor),
 * строит матрицу яркостей, выполняет сингулярное разложение A = U·Σ·Vᵀ
 * и возвращает первые N сингулярных чисел в виде вектора признаков.
 *
 * Используется: Apache Commons Math — SingularValueDecomposition
 */
public class SVDComputer {

    /** Количество сингулярных чисел в сигнатуре */
    public static final int SIGNATURE_LENGTH = 20;

    /**
     * Вычислить σ-вектор для изображения.
     *
     * @param image  нормализованное изображение 64×64
     * @return       массив double[SIGNATURE_LENGTH] — сингулярные числа по убыванию
     */
    public double[] compute(BufferedImage image) {
        // TODO:
        //   1. Извлечь пиксели в double[][] matrix (64×64)
        //   2. Создать RealMatrix: MatrixUtils.createRealMatrix(matrix)
        //   3. SVD: new SingularValueDecomposition(realMatrix)
        //   4. Взять первые SIGNATURE_LENGTH значений из getSingularValues()
        //   5. Нормализовать: делить на sigma[0] (инвариантность к масштабу)
        return new double[SIGNATURE_LENGTH];
    }

    /** Вспомогательный: BufferedImage → double[][] (яркость пикселей 0.0–1.0) */
    private double[][] imageToMatrix(BufferedImage image) {
        // TODO: (pixel & 0xFF) / 255.0
        return new double[64][64];
    }
}
