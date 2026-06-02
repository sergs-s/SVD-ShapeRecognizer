package svd.recognizer.processing;

import svd.recognizer.math.SvdEngine;

/**
 * Вычисление SVD-сигнатуры изображения через абстрактный интерфейс SvdEngine.
 *
 * Принцип работы:
 * 1. На вход поступает матрица яркостей нормализованного изображения 64x64
 * 2. Математический backend выполняет сингулярное разложение
 * 3. Из результата извлекаются первые k сингулярных чисел
 * 4. Признаковый вектор нормализуется по первому сингулярному значению
 *
 * Благодаря этой схеме SVDComputer не знает, какая именно библиотека
 * находится под ним: Apache Commons Math, EJML или любая другая.
 *
 * @author ssv
 */
public class SVDComputer {
    public static final int DEFAULT_FEATURE_COUNT = 20;

    private final SvdEngine svdEngine;

    public SVDComputer(SvdEngine svdEngine) {
        this.svdEngine = svdEngine;
    }

    public double[] computeFeatures(double[][] imageMatrix) {
        return computeFeatures(imageMatrix, DEFAULT_FEATURE_COUNT);
    }

    public double[] computeFeatures(double[][] imageMatrix, int featureCount) {
        double[] singularValues = svdEngine.computeTopSingularValues(imageMatrix, featureCount);
        if (singularValues.length == 0) {
            return singularValues;
        }
        double first = singularValues[0] == 0.0 ? 1.0 : singularValues[0];
        double[] normalized = new double[singularValues.length];
        for (int i = 0; i < singularValues.length; i++) {
            normalized[i] = singularValues[i] / first;
        }
        return normalized;
    }
}