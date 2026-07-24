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
        validateMatrix(matrix);

        int rows = matrix.length;
        int cols = matrix[0].length;

        double[] vector = new double[VECTOR_LENGTH];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(matrix[r], 0, vector, r * cols, cols);
        }
        return vector;
    }

    /**
     * Проверяет, что матрица не null и имеет корректный размер 64x64.
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если матрица null или имеет некорректный размер
     */
    private static void validateMatrix(double[][] matrix) {
        validateNotNull(matrix);
        validateRowCount(matrix);
        validateColumnCount(matrix);
        validateUniformRows(matrix);
    }

    /**
     * Проверяет, что матрица не null.
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если матрица null
     */
    private static void validateNotNull(double[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Матрица не может быть null");
        }
    }

    /**
     * Проверяет, что матрица имеет правильное количество строк (64).
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если количество строк не равно 64
     */
    private static void validateRowCount(double[][] matrix) {
        int rows = matrix.length;
        if (rows != IMAGE_SIZE) {
            throw new IllegalArgumentException(
                    "Матрица должна иметь " + IMAGE_SIZE + " строк, получено: " + rows
            );
        }
    }

    /**
     * Проверяет, что матрица имеет правильное количество столбцов (64).
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если количество столбцов не равно 64
     */
    private static void validateColumnCount(double[][] matrix) {
        if (matrix.length == 0) {
            return;
        }
        int cols = matrix[0].length;
        if (cols != IMAGE_SIZE) {
            throw new IllegalArgumentException(
                    "Матрица должна иметь " + IMAGE_SIZE + " столбцов, получено: " + cols
            );
        }
    }

    /**
     * Проверяет, что все строки матрицы имеют одинаковую длину.
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если строки имеют разную длину
     */
    private static void validateUniformRows(double[][] matrix) {
        int rows = matrix.length;
        if (rows == 0) {
            return;
        }

        int expectedCols = matrix[0].length;
        for (int i = 1; i < rows; i++) {
            if (matrix[i].length != expectedCols) {
                throw new IllegalArgumentException(
                        "Неравномерная матрица: строка 0 имеет длину " + expectedCols +
                                ", строка " + i + " имеет длину " + matrix[i].length
                );
            }
        }
    }
}