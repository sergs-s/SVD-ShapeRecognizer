package svd.recognizer.processing;

/**
 * Утилитный класс для векторизации изображений.
 *
 * Разворачивает прямоугольную матрицу rows×cols в вектор длины rows·cols
 * построчной укладкой (row-major). Для фигур это 64×64 → 4096, для лиц ORL
 * 112×92 → 10 304.
 *
 * @author ssv
 */
public final class ImageVectorizer {

    private ImageVectorizer() {}

    /**
     * Преобразует матрицу яркостей в вектор построчной укладкой.
     *
     * @param matrix прямоугольная матрица яркостей (все строки одной длины)
     * @return вектор длины rows·cols, построчная укладка
     * @throws IllegalArgumentException если матрица null, пустая или строки разной длины
     */
    public static double[] toVector(double[][] matrix) {
        validateMatrix(matrix);

        int rows = matrix.length;
        int cols = matrix[0].length;

        double[] vector = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(matrix[r], 0, vector, r * cols, cols);
        }
        return vector;
    }

    /**
     * Проверяет, что матрица не null, не пустая и прямоугольная.
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если матрица некорректна
     */
    private static void validateMatrix(double[][] matrix) {
        validateNotNull(matrix);
        validateNotEmpty(matrix);
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
     * Проверяет, что в матрице есть хотя бы одна строка и один столбец.
     *
     * @param matrix матрица для проверки
     * @throws IllegalArgumentException если матрица пустая
     */
    private static void validateNotEmpty(double[][] matrix) {
        if (matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) {
            throw new IllegalArgumentException("Матрица не может быть пустой");
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
        int expectedCols = matrix[0].length;
        for (int i = 1; i < rows; i++) {
            if (matrix[i] == null || matrix[i].length != expectedCols) {
                throw new IllegalArgumentException(
                        "Неравномерная матрица: строка 0 имеет длину " + expectedCols +
                                ", строка " + i + " имеет длину " +
                                (matrix[i] == null ? 0 : matrix[i].length)
                );
            }
        }
    }
}
