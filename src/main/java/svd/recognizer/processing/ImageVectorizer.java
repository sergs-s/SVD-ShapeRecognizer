package svd.recognizer.processing;

/**
 * Утилитный класс для векторизации изображений.
 *
 * Разворачивает матрицу 64×64 в вектор длины 4096 построчной укладкой (row-major):
 * вектор = [строка0, строка1, ..., строка63]
 *
 * @author ssv
 */
public final class ImageVectorizer {

    private ImageVectorizer() {}

    /**
     * Преобразует матрицу яркостей в вектор построчной укладкой.
     *
     * @param matrix матрица яркостей 0..1 размера 64x64
     * @return вектор длины 4096, где первые 64 элемента — первая строка матрицы,
     *         следующие 64 — вторая строка, и т.д.
     * @throws IllegalArgumentException если матрица null или имеет некорректный размер
     */
    public static double[] toVector(double[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Матрица не может быть null");
        }

        int rows = matrix.length;
        if (rows == 0) {
            throw new IllegalArgumentException("Матрица не может быть пустой");
        }

        int cols = matrix[0].length;
        if (cols == 0) {
            throw new IllegalArgumentException("Матрица не может иметь нулевую ширину");
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

        double[] vector = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(matrix[r], 0, vector, r * cols, cols);
        }
        return vector;
    }

    /**
     * Преобразует вектор обратно в матрицу.
     *
     * @param vector вектор длины 4096
     * @param rows количество строк в матрице
     * @param cols количество столбцов в матрице
     * @return матрица размера rows x cols
     * @throws IllegalArgumentException если размер вектора не соответствует rows * cols
     */
    public static double[][] toMatrix(double[] vector, int rows, int cols) {
        if (vector == null) {
            throw new IllegalArgumentException("Вектор не может быть null");
        }

        if (vector.length != rows * cols) {
            throw new IllegalArgumentException(
                    "Размер вектора " + vector.length +
                            " не соответствует размеру матрицы " + rows + "x" + cols +
                            " = " + (rows * cols)
            );
        }

        double[][] matrix = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(vector, r * cols, matrix[r], 0, cols);
        }
        return matrix;
    }
}