package svd.recognizer.math;

/**
 * Интерфейс вычислительного движка сингулярного разложения.
 *
 * Это архитектурный шов между прикладной частью приложения и конкретной
 * математической библиотекой. Благодаря интерфейсу можно заменить backend
 * Apache Commons Math на EJML без переписывания GUI, preprocessing,
 * хранилища эталонов и классификатора.
 *
 * @author ssv
 */
public interface SvdEngine {
    /**
     * Выполнить сингулярное разложение матрицы.
     *
     * @param matrix входная матрица
     * @return полный результат SVD
     */
    SvdResult decompose(double[][] matrix);

    /**
     * Получить первые k сингулярных значений.
     *
     * @param matrix входная матрица
     * @param k количество сингулярных чисел
     * @return массив из первых k сингулярных значений
     */
    default double[] computeTopSingularValues(double[][] matrix, int k) {
        return decompose(matrix).getTopSingularValues(k);
    }
}