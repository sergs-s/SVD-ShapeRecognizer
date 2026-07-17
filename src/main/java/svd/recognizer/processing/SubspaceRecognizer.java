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

    public SubspaceRecognizer() {
        // Используем значения по умолчанию
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
        validateInputs(x, model);

        double[] mean = model.getMeanVector();
        double[][] basis = model.getBasisMatrix();
        int k = model.getK();
        int dim = x.length;

        validateDimensions(x, mean, basis, k);

        // Шаг 1: Центрирование вектора
        double[] centered = centerVector(x, mean);

        // Шаг 2: Проекция на подпространство (координаты в базисе)
        double[] coords = projectToSubspace(centered, basis, k);

        // Шаг 3: Восстановление и вычисление ошибки
        return computeReconstructionError(centered, basis, coords, k);
    }

    /**
     * Проверяет входные параметры на null.
     */
    private void validateInputs(double[] x, SubspaceModel model) {
        if (x == null || model == null) {
            throw new IllegalArgumentException("Вектор и модель не могут быть null");
        }
    }

    /**
     * Проверяет соответствие размерностей вектора, среднего и базиса.
     */
    private void validateDimensions(double[] x, double[] mean, double[][] basis, int k) {
        int dim = x.length;

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
    }

    /**
     * Центрирует вектор: вычитает среднее значение.
     *
     * @param x    исходный вектор
     * @param mean средний вектор класса
     * @return центрированный вектор
     */
    private double[] centerVector(double[] x, double[] mean) {
        int dim = x.length;
        double[] centered = new double[dim];
        for (int i = 0; i < dim; i++) {
            centered[i] = x[i] - mean[i];
        }
        return centered;
    }

    /**
     * Проецирует центрированный вектор на подпространство.
     *
     * @param centered центрированный вектор
     * @param basis    матрица базиса подпространства
     * @param k        размерность подпространства
     * @return координаты вектора в базисе подпространства
     */
    private double[] projectToSubspace(double[] centered, double[][] basis, int k) {
        int dim = centered.length;
        double[] coords = new double[k];
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int i = 0; i < dim; i++) {
                sum += basis[i][j] * centered[i];
            }
            coords[j] = sum;
        }
        return coords;
    }

    /**
     * Вычисляет ошибку реконструкции: ||centered - B * coords||.
     *
     * @param centered центрированный вектор
     * @param basis    матрица базиса подпространства
     * @param coords   координаты в базисе
     * @param k        размерность подпространства
     * @return евклидова норма остатка проекции
     */
    private double computeReconstructionError(double[] centered, double[][] basis,
                                              double[] coords, int k) {
        int dim = centered.length;
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

        validateStoresTrained(stores);
        validateHypothesisVectors(hypothesisVectors);

        Map<ShapeClass, Double> classScores = computeAllErrors(hypothesisVectors, stores);
        ShapeClass bestClass = findBestClass(classScores);
        double bestError = classScores.get(bestClass);

        // Принятие решения: проверка порога
        if (bestError <= theta) {
            return new RecognitionResult(bestClass, bestError, theta, true,
                    RecognitionMode.SUBSPACE, classScores);
        }
        return new RecognitionResult(null, bestError, theta, false,
                RecognitionMode.SUBSPACE, classScores);
    }

    /**
     * Проверяет, что все классы обучены.
     */
    private void validateStoresTrained(Map<ShapeClass, TemplateStore> stores) {
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
    }

    /**
     * Проверяет, что все векторы гипотез присутствуют и имеют правильную длину.
     */
    private void validateHypothesisVectors(Map<ShapeClass, double[]> hypothesisVectors) {
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
        }
    }

    /**
     * Вычисляет ошибки реконструкции для всех гипотез.
     *
     * @param hypothesisVectors векторы гипотез для каждого класса
     * @param stores            хранилища классов
     * @return карта "класс → ошибка реконструкции"
     */
    private Map<ShapeClass, Double> computeAllErrors(
            Map<ShapeClass, double[]> hypothesisVectors,
            Map<ShapeClass, TemplateStore> stores) {

        Map<ShapeClass, Double> classScores = new EnumMap<>(ShapeClass.class);

        for (ShapeClass sc : ShapeClass.values()) {
            double[] x = hypothesisVectors.get(sc);
            SubspaceModel model = stores.get(sc).getSubspaceModel();
            double error = reconstructionError(x, model);
            classScores.put(sc, error);
        }

        return classScores;
    }

    /**
     * Находит класс с минимальной ошибкой реконструкции.
     *
     * @param classScores карта "класс → ошибка реконструкции"
     * @return класс с минимальной ошибкой
     */
    private ShapeClass findBestClass(Map<ShapeClass, Double> classScores) {
        ShapeClass bestClass = null;
        double bestError = Double.MAX_VALUE;

        for (Map.Entry<ShapeClass, Double> entry : classScores.entrySet()) {
            double error = entry.getValue();
            if (error < bestError) {
                bestError = error;
                bestClass = entry.getKey();
            }
        }

        return bestClass;
    }

    /**
     * Распознавание по подпространствам с порогом по умолчанию.
     */
    public RecognitionResult recognize(
            Map<ShapeClass, double[]> hypothesisVectors,
            Map<ShapeClass, TemplateStore> stores) {
        return recognize(hypothesisVectors, stores, DEFAULT_THRESHOLD);
    }
}