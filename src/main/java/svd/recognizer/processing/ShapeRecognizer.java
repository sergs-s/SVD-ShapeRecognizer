package svd.recognizer.processing;

import java.util.Map;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

/**
 * Классификатор фигур по SVD-сигнатурам.
 *
 * Алгоритм распознавания:
 * 1. Получить σ-вектор тестового изображения
 * 2. Для каждого класса взять усреднённый эталонный σ-вектор
 * 3. Вычислить евклидово расстояние до каждого эталона класса
 * 4. Найти класс с минимальным расстоянием
 * 5. Сравнить минимум с порогом, задаваемым пользователем через JSpinner
 * 6. Если расстояние превышает порог, вернуть состояние "Фигура не распознана"
 *
 * Дополнительно реализован автоматический расчёт рекомендуемого порога на основе
 * среднего внутриклассового расстояния, умноженного на коэффициент 1.5.
 *
 * @author ssv
 */
public class ShapeRecognizer {
    public static final double DEFAULT_THRESHOLD = 0.35;

    private double threshold = DEFAULT_THRESHOLD;

    public ShapeRecognizer() {
    }

    public ShapeRecognizer(double threshold) {
        this.threshold = threshold;
    }

    public RecognitionResult recognize(double[] testSignature, Map<ShapeClass, TemplateStore> stores) {
        double minDistance = Double.MAX_VALUE;
        ShapeClass bestClass = null;

        for (Map.Entry<ShapeClass, TemplateStore> entry : stores.entrySet()) {
            double[] average = entry.getValue().getAverageSingularValues();
            if (average == null) {
                continue;
            }
            double distance = euclideanDistance(testSignature, average);
            if (distance < minDistance) {
                minDistance = distance;
                bestClass = entry.getKey();
            }
        }

        boolean recognized = bestClass != null && minDistance <= threshold;
        return new RecognitionResult(recognized ? bestClass : null, minDistance, threshold, recognized);
    }

    public double calculateAutoThreshold(Map<ShapeClass, TemplateStore> stores) {
        double maxMeanDistance = 0.0;
        for (TemplateStore store : stores.values()) {
            if (store.getAverageSingularValues() == null || store.getTemplates().isEmpty()) {
                continue;
            }
            double sum = 0.0;
            int count = 0;
            for (Template template : store.getTemplates()) {
                sum += euclideanDistance(template.getSingularValues(), store.getAverageSingularValues());
                count++;
            }
            if (count > 0) {
                maxMeanDistance = Math.max(maxMeanDistance, sum / count);
            }
        }
        return maxMeanDistance > 0.0 ? maxMeanDistance * 1.5 : DEFAULT_THRESHOLD;
    }

    private double euclideanDistance(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * Результат распознавания для GUI и служебных сообщений.
     *
     * Содержит найденный класс, фактическое расстояние, использованный порог
     * и признак успешного распознавания.
     */
    public static class RecognitionResult {
        private final ShapeClass shapeClass;
        private final double distance;
        private final double threshold;
        private final boolean recognized;

        public RecognitionResult(ShapeClass shapeClass, double distance, double threshold, boolean recognized) {
            this.shapeClass = shapeClass;
            this.distance = distance;
            this.threshold = threshold;
            this.recognized = recognized;
        }

        public ShapeClass getShapeClass() {
            return shapeClass;
        }

        public double getDistance() {
            return distance;
        }

        public double getThreshold() {
            return threshold;
        }

        public boolean isRecognized() {
            return recognized;
        }
    }
}