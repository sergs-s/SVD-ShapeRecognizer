package svd.recognizer.processing;

import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;
import java.util.Map;

/**
 * Классификатор фигур по σ-вектору.
 *
 * Алгоритм:
 *   1. Получить σ-вектор тестового изображения
 *   2. Для каждого класса вычислить евклидово расстояние до усреднённого эталона
 *   3. Найти класс с минимальным расстоянием
 *   4. Если минимальное расстояние > порога — вернуть null ("Фигура не распознана")
 *   5. Иначе вернуть найденный ShapeClass
 */
public class ShapeRecognizer {

    /** Порог по умолчанию (пересчитывается после загрузки эталонов) */
    public static final double DEFAULT_THRESHOLD = 5.0;

    private double threshold;

    public ShapeRecognizer() {
        this.threshold = DEFAULT_THRESHOLD;
    }

    public ShapeRecognizer(double threshold) {
        this.threshold = threshold;
    }

    /**
     * Распознать фигуру.
     *
     * @param testSignature   σ-вектор тестового изображения
     * @param stores          карта: ShapeClass → TemplateStore с усреднёнными эталонами
     * @return                распознанный ShapeClass или null, если фигура не распознана
     */
    public ShapeClass recognize(double[] testSignature, Map<ShapeClass, TemplateStore> stores) {
        // TODO:
        //   double minDist = Double.MAX_VALUE;
        //   ShapeClass best = null;
        //   for (ShapeClass sc : stores.keySet()) {
        //       double dist = euclideanDistance(testSignature, stores.get(sc).getAverageSingularValues());
        //       if (dist < minDist) { minDist = dist; best = sc; }
        //   }
        //   return (minDist <= threshold) ? best : null;
        return null;
    }

    /**
     * Автоматически вычислить порог как среднее внутриклассовое расстояние × коэффициент.
     * Вызывается после загрузки всех эталонов.
     *
     * @param stores  карта эталонов
     * @return        рекомендуемый порог
     */
    public double calculateAutoThreshold(Map<ShapeClass, TemplateStore> stores) {
        // TODO: для каждого класса посчитать среднее расстояние от каждого
        //       эталона до усреднённого, взять максимум × 1.5
        return DEFAULT_THRESHOLD;
    }

    /** Евклидово расстояние между двумя σ-векторами */
    private double euclideanDistance(double[] a, double[] b) {
        // TODO: sqrt( sum( (a[i]-b[i])^2 ) )
        return 0.0;
    }

    public double getThreshold()          { return threshold; }
    public void setThreshold(double thr)  { this.threshold = thr; }
}
