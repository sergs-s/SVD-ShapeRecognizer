package svd.recognizer.processing;

import svd.recognizer.math.SvdEngine;

/**
 * Вычисление SVD-сигнатуры изображения (σ-вектора) через абстрактный
 * интерфейс {@link SvdEngine}.
 *
 * Принцип работы:
 * 1. На вход поступает матрица яркостей нормализованного изображения 64×64
 *    (значения 0..1, где фигура — единицы, фон — нули).
 * 2. Математический backend выполняет сингулярное разложение и возвращает
 *    первые k сингулярных значений (старшие, несущие основную энергию формы).
 * 3. Вектор нормализуется по первому сингулярному значению.
 *
 * Зачем нормализация по σ₁: она убирает зависимость подписи от абсолютного
 * масштаба «энергии» изображения (общего количества и яркости пикселей) и
 * оставляет только относительную форму спектра σ. Благодаря этому две фигуры
 * одного класса, нарисованные с разной толщиной линии, дают близкие σ-векторы —
 * сравниваются пропорции сингулярных значений, а не их абсолютные величины.
 *
 * Благодаря интерфейсу SvdEngine этот класс не зависит от конкретной
 * библиотеки линейной алгебры под ним (Apache Commons Math, EJML и т.п.).
 *
 * @author ssv
 */
public class SVDComputer {

    /** Длина σ-вектора по умолчанию: сколько старших сингулярных значений берём. */
    public static final int DEFAULT_FEATURE_COUNT = 20;

    private final SvdEngine svdEngine;

    public SVDComputer(SvdEngine svdEngine) {
        this.svdEngine = svdEngine;
    }

    /**
     * Признаковый σ-вектор изображения с числом компонент по умолчанию.
     *
     * @param imageMatrix матрица яркостей 64×64 (0..1)
     * @return нормализованный σ-вектор
     */
    public double[] computeFeatures(double[][] imageMatrix) {
        return computeFeatures(imageMatrix, DEFAULT_FEATURE_COUNT);
    }

    /**
     * Признаковый σ-вектор изображения заданной длины.
     *
     * Алгоритм: SVD матрицы → первые featureCount сингулярных значений →
     * деление всех значений на первое (σ₁). Если σ₁ = 0 (пустое изображение),
     * делитель заменяется на 1.0, чтобы избежать деления на ноль.
     *
     * @param imageMatrix матрица яркостей 64×64 (0..1)
     * @param featureCount желаемая длина σ-вектора
     * @return нормализованный по σ₁ массив сингулярных значений
     */
    public double[] computeFeatures(double[][] imageMatrix, int featureCount) {
        double[] singularValues = svdEngine.computeTopSingularValues(imageMatrix, featureCount);
        if (singularValues.length == 0) {
            return singularValues;
        }
        // Нормировка по σ₁
        double first = singularValues[0] == 0.0 ? 1.0 : singularValues[0];
        double[] normalized = new double[singularValues.length];
        for (int i = 0; i < singularValues.length; i++) {
            normalized[i] = singularValues[i] / first;
        }
        return normalized;
    }

    /**
     * Возвращает движок SVD, используемый для вычислений.
     * Необходим для создания SubspaceTrainer.
     *
     * @return экземпляр SvdEngine
     */
    public SvdEngine getSvdEngine() {
        return svdEngine;
    }
}