package svd.recognizer.processing;

import java.util.EnumMap;
import java.util.Map;
import svd.recognizer.model.RecognitionMode;
import svd.recognizer.model.RecognitionResult;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

/**
 * Классификатор фигур по SVD-сигнатурам (σ-векторам) и по подпространствам.
 *
 * Распознавание по схеме «Путь B»: тестовая фигура прогоняется через ВСЕ три
 * ветки обработки (как круг, как треугольник, как прямоугольник), и для каждой
 * гипотезы вычисляется признак (σ-вектор или 4096-вектор) именно той ветки.
 *
 * Режимы работы:
 * 1. SIGMA_VECTOR: сравнение σ-векторов с усреднённым эталоном класса
 * 2. SUBSPACE: ошибка реконструкции в подпространстве класса
 *
 * @author ssv
 */
public class ShapeRecognizer {

    public static final double DEFAULT_THRESHOLD = 0.35;
    public static final double DEFAULT_AUTO_MULTIPLIER = 2.0;

    public static final double DEFAULT_SUBSPACE_THRESHOLD = 13.0;
    public static final int DEFAULT_SUBSPACE_K = 4;
    public static final int VECTOR_LENGTH = 4096;

    private final Map<ShapeClass, Double> thresholds = new EnumMap<>(ShapeClass.class);

    public ShapeRecognizer() {
        for (ShapeClass sc : ShapeClass.values()) {
            thresholds.put(sc, DEFAULT_THRESHOLD);
        }
    }

    /**
     * Устанавливает порог распознавания для одного класса. Порог — максимально
     * допустимое расстояние до эталона, при котором фигура ещё считается
     * принадлежащей классу.
     *
     * @param shapeClass класс фигуры
     * @param threshold  максимально допустимое расстояние до эталона класса
     */
    public void setThreshold(ShapeClass shapeClass, double threshold) {
        thresholds.put(shapeClass, threshold);
    }

    /**
     * @param shapeClass класс фигуры
     * @return текущий порог распознавания класса (или значение по умолчанию)
     */
    public double getThreshold(ShapeClass shapeClass) {
        Double t = thresholds.get(shapeClass);
        return t != null ? t : DEFAULT_THRESHOLD;
    }

    /**
     * Распознавание по Пути B (σ-векторы).
     *
     * @param hypothesisSignatures σ-вектор тестовой фигуры для каждой гипотезы
     *                             класса (фигура, обработанная веткой этого класса)
     * @param stores               эталоны (усреднённые σ-векторы) по классам
     * @return результат с найденным классом, его расстоянием и порогом
     */
    public RecognitionResult recognize(Map<ShapeClass, double[]> hypothesisSignatures,
                                       Map<ShapeClass, TemplateStore> stores) {

        ShapeClass bestClass = null;
        double bestDistance = Double.MAX_VALUE;
        double bestThreshold = 0.0;

        ShapeClass nearestClass = null;
        double nearestDistance = Double.MAX_VALUE;
        double nearestThreshold = 0.0;

        for (ShapeClass sc : ShapeClass.values()) {
            TemplateStore store = stores.get(sc);
            double[] signature = hypothesisSignatures.get(sc);
            if (store == null || signature == null) {
                continue;
            }
            double[] average = store.getAverageSingularValues();
            if (average == null) {
                continue;
            }
            double distance = euclideanDistance(signature, average);
            double threshold = getThreshold(sc);

            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestClass = sc;
                nearestThreshold = threshold;
            }
            if (distance <= threshold && distance < bestDistance) {
                bestDistance = distance;
                bestClass = sc;
                bestThreshold = threshold;
            }
        }

        if (bestClass != null) {
            return new RecognitionResult(bestClass, bestDistance, bestThreshold, true,
                    RecognitionMode.SIGMA_VECTOR, null);
        }
        return new RecognitionResult(null, nearestDistance, nearestThreshold, false,
                RecognitionMode.SIGMA_VECTOR, null);
    }

    /**
     * Рассчитывает авто-порог класса: среднее расстояние эталонов до их
     * усреднённого σ-вектора, умноженное на коэффициент. Кнопки ×1.0/×1.5/×2.0
     * задают коэффициент; у каждого класса свой масштаб расстояний.
     *
     * @param store      хранилище эталонов класса
     * @param multiplier коэффициент строгости (например, 2.0)
     * @return порог расстояния для класса (или значение по умолчанию, если эталонов нет)
     */
    public double calculateAutoThreshold(TemplateStore store, double multiplier) {
        if (store == null || store.getAverageSingularValues() == null
                || store.getTemplates().isEmpty()) {
            return DEFAULT_THRESHOLD;
        }
        double sum = 0.0;
        int count = 0;
        for (Template template : store.getTemplates()) {
            sum += euclideanDistance(template.getSingularValues(),
                    store.getAverageSingularValues());
            count++;
        }
        if (count == 0) {
            return DEFAULT_THRESHOLD;
        }
        double mean = sum / count;
        return mean > 0.0 ? mean * multiplier : DEFAULT_THRESHOLD;
    }

    /**
     * Евклидово расстояние между двумя σ-векторами — мера непохожести форм.
     * Чем меньше расстояние, тем ближе подпись фигуры к эталонной. Если длины
     * различаются, сравнение идёт по общей (меньшей) части.
     *
     * @param a первый σ-вектор
     * @param b второй σ-вектор
     * @return sqrt(Σ (aᵢ − bᵢ)²)
     */
    private double euclideanDistance(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }


    /**
     * Вычисляет ошибку реконструкции вектора в подпространстве класса.
     *
     * Формула: ε = ||x' - B·Bᵀ·x'||, где x' = x - meanVector
     *
     * Алгоритм:
     * 1. Центрировать вектор: centered = x - mean
     * 2. Спроецировать на подпространство: coords = Bᵀ * centered
     * 3. Восстановить: reconstructed = B * coords
     * 4. Ошибка = ||centered - reconstructed||
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

        // Шаг 1: Центрирование вектора
        double[] centered = new double[dim];
        for (int i = 0; i < dim; i++) {
            centered[i] = x[i] - mean[i];
        }

        // Шаг 2: Проекция на подпространство (координаты в базисе)
        // coords = Bᵀ * centered
        double[] coords = new double[k];
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int i = 0; i < dim; i++) {
                sum += basis[i][j] * centered[i];
            }
            coords[j] = sum;
        }

        // Шаг 3: Восстановление и вычисление ошибки
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
     * Сохраняет «Путь B»: на входе — уже готовые 4096-векторы трёх гипотез
     * (фигура, обработанная как круг / треугольник / прямоугольник).
     *
     * @param hypothesisVectors 4096-векторы трёх гипотез (длина 4096)
     * @param stores            хранилища классов; все должны быть обучены (isTrained()==true)
     * @param theta             единый порог отвержения
     * @return результат распознавания (mode = SUBSPACE)
     * @throws IllegalStateException если хотя бы один класс не обучен
     */
    public RecognitionResult recognizeBySubspaces(
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
     */
    public RecognitionResult recognizeBySubspaces(
            Map<ShapeClass, double[]> hypothesisVectors,
            Map<ShapeClass, TemplateStore> stores) {
        return recognizeBySubspaces(hypothesisVectors, stores, DEFAULT_SUBSPACE_THRESHOLD);
    }
}