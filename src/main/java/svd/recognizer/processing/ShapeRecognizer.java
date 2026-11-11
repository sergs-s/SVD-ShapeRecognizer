package svd.recognizer.processing;

import java.util.EnumMap;
import java.util.Map;
import svd.recognizer.model.RecognitionMode;
import svd.recognizer.model.RecognitionResult;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

/**
 * Классификатор фигур по SVD-сигнатурам (σ-векторам), режим SIGMA_VECTOR.
 *
 * Распознавание по схеме «Путь B»: тестовая фигура прогоняется через ВСЕ три
 * ветки обработки (как круг, как треугольник, как прямоугольник), и для каждой
 * гипотезы вычисляется σ-вектор именно той ветки; он сравнивается с
 * усреднённым эталоном класса.
 *
 * Режим подпространств (SUBSPACE) реализован в {@link SubspaceRecognizer}.
 *
 * @author ssv
 */
public class ShapeRecognizer {

    public static final double DEFAULT_THRESHOLD = 0.35;
    public static final double DEFAULT_AUTO_MULTIPLIER = 2.0;

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
}
