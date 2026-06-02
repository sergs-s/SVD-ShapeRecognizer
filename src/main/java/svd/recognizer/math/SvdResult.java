package svd.recognizer.math;

import java.util.Arrays;

/**
 * Контейнер результата сингулярного разложения матрицы.
 *
 * Содержит:
 * 1. Сингулярные значения в невозрастающем порядке
 * 2. Матрицу левых сингулярных векторов U
 * 3. Матрицу правых сингулярных векторов V
 *
 * Такое представление позволяет сейчас использовать только σ-вектор
 * для фигур, а позже перейти к задачам лиц, eigenfaces и проекциям
 * в подпространство без изменения прикладной архитектуры.
 *
 * @author ssv
 */
public class SvdResult {
    private final double[] singularValues;
    private final double[][] u;
    private final double[][] v;

    public SvdResult(double[] singularValues, double[][] u, double[][] v) {
        this.singularValues = singularValues;
        this.u = u;
        this.v = v;
    }

    public double[] getSingularValues() {
        return singularValues;
    }

    public double[][] getU() {
        return u;
    }

    public double[][] getV() {
        return v;
    }

    public int getRank(double eps) {
        int rank = 0;
        for (double s : singularValues) {
            if (s > eps) {
                rank++;
            }
        }
        return rank;
    }

    public double[] getTopSingularValues(int k) {
        int n = Math.min(k, singularValues.length);
        return Arrays.copyOf(singularValues, n);
    }
}