package svd.recognizer.processing;

/**
 * Утилитный класс для векторизации изображений.
 *
 * Разворачивает матрицу 64×64 в вектор длины 4096 построчной укладкой (row-major).
 *
 * @author ssv
 */
public final class ImageVectorizer {

    public static final int VECTOR_LENGTH = 64 * 64; // 4096
    private static final int IMAGE_SIZE = 64;

    private ImageVectorizer() {}

    /**
     * Преобразует матрицу яркостей в вектор построчной укладкой.
     *
     * @param matrix матрица яркостей 0..1 размера 64x64
     * @return вектор длины 4096, построчная укладка
     * @throws IllegalArgumentException если матрица null или имеет некорректный размер
     */
    public static double[] toVector(double[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Матрица не может быть null");
        }

        int rows = matrix.length;
        if (rows != IMAGE_SIZE) {
            throw new IllegalArgumentException(
                    "Матрица должна иметь " + IMAGE_SIZE + " строк, получено: " + rows
            );
        }

        int cols = matrix[0].length;
        if (cols != IMAGE_SIZE) {
            throw new IllegalArgumentException(
                    "Матрица должна иметь " + IMAGE_SIZE + " столбцов, получено: " + cols
            );
        }

        // Проверка, что все строки имеют одинаковую длину
        for (int i = 1; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new IllegalArgumentException(
                        "Неравномерная матрица: строка 0 имеет длину " + cols +
                                ", строка " + i + " имеет длину " + matrix[i].length
                );
            }
        }

        double[] vector = new double[VECTOR_LENGTH];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(matrix[r], 0, vector, r * cols, cols);
        }
        return vector;
    }
}