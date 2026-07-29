package svd.recognizer.processing;

import svd.recognizer.model.RecognitionMode;
import svd.recognizer.model.RecognitionResult;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.model.TemplateStore;

import java.util.EnumMap;
import java.util.Map;

/**
 * Классификатор фигур по методу собственных подпространств (eigenshapes).
 *
 * Распознавание по схеме «Путь B»: тестовая фигура прогоняется через ВСЕ три
 * ветки обработки (как круг, как треугольник, как прямоугольник), и для каждой
 * гипотезы вычисляется 4096-вектор именно той ветки. Затем:
 *   1. для каждой гипотезы вычисляется ошибка реконструкции в подпространстве
 *      соответствующего класса;
 *   2. выбирается класс с минимальной ошибкой реконструкции;
 *   3. если минимальная ошибка не превышает единый порог θ — фигура
 *      распознана, иначе — «не распознано».
 *
 * @author ssv
 */
public class SubspaceRecognizer {

    public static final double DEFAULT_THRESHOLD = 13.0;
    private static final int VECTOR_LENGTH = 4096;

    private double theta = DEFAULT_THRESHOLD;

    public SubspaceRecognizer() {}

    /**
     * Устанавливает единый порог отвержения θ.
     *
     * @param theta порог отвержения (рекомендуемое значение 13.0)
     */
    public void setThreshold(double theta) {
        this.theta = theta;
    }

    /**
     * @return текущий порог отвержения
     */
    public double getThreshold() {
        return theta;
    }

    /**
     * Вычисляет ошибку реконструкции вектора в подпространстве класса.
     *
     * @param x     входной вектор (длина 4096)
     * @param model обученная модель подпространства класса
     * @return евклидова норма остатка проекции (reconstruction error)
     * @throws IllegalArgumentException если размеры не совпадают
     */
    public double reconstructionError(double[] x, SubspaceModel model) {
        if (x == null || model == null) {
            throw new IllegalArgumentException("Вектор и модель не могут быть null");
        }

        double[] mean = model.getMeanVector();
        double[][] basis = model.getBasisMatrix();
        int k = model.getK();
        int dim = x.length;

        // Проверка размеров
        if (dim != mean.length) {
            throw new IllegalArgumentException(
                    "Размер вектора (" + dim + ") не совпадает с размером среднего (" + mean.length + ")"
            );
        }
        if (basis.length != dim) {
            throw new IllegalArgumentException(
                    "Количество строк базиса (" + basis.length + ") не совпадает с размерностью (" + dim + ")"
            );
        }
        if (basis[0].length < k) {
            throw new IllegalArgumentException(
                    "Базис имеет " + basis[0].length + " столбцов, ожидается как минимум " + k
            );
        }

        // Центрирование вектора
        double[] centered = new double[dim];
        for (int i = 0; i < dim; i++) {
            centered[i] = x[i] - mean[i];
        }

        // Проекция на подпространство
        // coords = Bᵀ * centered
        double[] coords = new double[k];
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int i = 0; i < dim; i++) {
                sum += basis[i][j] * centered[i];
            }
            coords[j] = sum;
        }

        // Восстановление и вычисление ошибки
        // reconstructed = B * coords
        // error = ||centered - reconstructed||
        double sumSq = 0.0;
        for (int i = 0; i < dim; i++) {
            double reconstructed = 0.0;
            for (int j = 0; j < k; j++) {
                reconstructed += basis[i][j] * coords[j];
            }
            double residual = centered[i] - reconstructed;
            sumSq += residual * residual;
        }

        return Math.sqrt(sumSq);
    }

    /**
     * Распознавание по подпространствам (Путь B).
     *
     * @param hypothesisVectors 4096-векторы трёх гипотез (длина 4096)
     * @param stores            хранилища классов; все должны быть обучены (isTrained()==true)
     * @param theta             единый порог отвержения
     * @return результат распознавания
     * @throws IllegalStateException если хотя бы один класс не обучен
     */
    public RecognitionResult recognize(
            Map<ShapeClass, double[]> hypothesisVectors,
            Map<ShapeClass, TemplateStore> stores,
            double theta) {

        // Проверка: все классы должны быть обучены
        for (ShapeClass sc : ShapeClass.values()) {
            TemplateStore store = stores.get(sc);
            if (store == null) {
                throw new IllegalStateException("Хранилище для класса " + sc + " не найдено");
            }
            if (!store.isTrained()) {
                throw new IllegalStateException(
                        "Класс " + sc + " не обучен — выполните «Обучение» в интерфейсе"
                );
            }
        }

        ShapeClass bestClass = null;
        double bestError = Double.MAX_VALUE;
        Map<ShapeClass, Double> classScores = new EnumMap<>(ShapeClass.class);

        // Вычисляем ошибку реконструкции для каждой гипотезы
        for (ShapeClass sc : ShapeClass.values()) {
            double[] x = hypothesisVectors.get(sc);
            if (x == null) {
                throw new IllegalArgumentException(
                        "Нет вектора для гипотезы класса " + sc
                );
            }
            if (x.length != VECTOR_LENGTH) {
                throw new IllegalArgumentException(
                        "Вектор для класса " + sc + " имеет длину " + x.length +
                                ", ожидается " + VECTOR_LENGTH
                );
            }

            SubspaceModel model = stores.get(sc).getSubspaceModel();
            double error = reconstructionError(x, model);
            classScores.put(sc, error);

            if (error < bestError) {
                bestError = error;
                bestClass = sc;
            }
        }

        // Принятие решения: проверка порога
        if (bestError <= theta) {
            return new RecognitionResult(bestClass, bestError, theta, true,
                    RecognitionMode.SUBSPACE, classScores);
        }
        return new RecognitionResult(null, bestError, theta, false,
                RecognitionMode.SUBSPACE, classScores);
    }

    /**
     * Распознавание по подпространствам с порогом по умолчанию.
     * @param hypothesisVectors
     * @param stores
     * @return 
     */
    public RecognitionResult recognize(
            Map<ShapeClass, double[]> hypothesisVectors,
            Map<ShapeClass, TemplateStore> stores) {
        return recognize(hypothesisVectors, stores, theta);
    }
}