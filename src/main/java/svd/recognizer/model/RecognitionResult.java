package svd.recognizer.model;

import java.io.Serializable;
import java.util.Collections;
import java.util.Map;

/**
 * Результат распознавания для GUI и служебных сообщений.
 *
 * Поддерживает оба режима: SIGMA_VECTOR и SUBSPACE.
 *
 * @author ssv
 */
public class RecognitionResult implements Serializable {

    private static final long serialVersionUID = 1L;

    private final ShapeClass shapeClass;
    private final double score;
    private final double threshold;
    private final boolean recognized;
    private final RecognitionMode mode;
    private final Map<ShapeClass, Double> classScores;

    /**
     * Конструктор для σ-режима (сохраняет обратную совместимость).
     */
    public RecognitionResult(ShapeClass shapeClass, double score,
                             double threshold, boolean recognized) {
        this(shapeClass, score, threshold, recognized,
                RecognitionMode.SIGMA_VECTOR, null);
    }

    /**
     * Полный конструктор для обоих режимов.
     */
    public RecognitionResult(ShapeClass shapeClass, double score,
                             double threshold, boolean recognized, RecognitionMode mode,
                             Map<ShapeClass, Double> classScores) {
        this.shapeClass = shapeClass;
        this.score = score;
        this.threshold = threshold;
        this.recognized = recognized;
        this.mode = mode;
        // Защита от изменений извне
        this.classScores = classScores != null
                ? Collections.unmodifiableMap(classScores)
                : null;
    }

    public ShapeClass getShapeClass() {
        return shapeClass;
    }

    /**
     * В σ-режиме: расстояние до среднего σ-вектора.
     * В subspace-режиме: ошибка реконструкции.
     */
    public double getScore() {
        return score;
    }

    /**
     * Алиас для обратной совместимости со старым кодом.
     */
    public double getDistance() {
        return score;
    }

    public double getThreshold() {
        return threshold;
    }

    public boolean isRecognized() {
        return recognized;
    }

    public RecognitionMode getMode() {
        return mode;
    }

    public Map<ShapeClass, Double> getClassScores() {
        return classScores;
    }

    /**
     * @return название режима для UI
     */
    public String getModeDisplayName() {
        return mode == RecognitionMode.SUBSPACE ? "Subspace" : "Sigma-vector";
    }

    /**
     * @return название метрики для UI
     */
    public String getMetricName() {
        return mode == RecognitionMode.SUBSPACE ? "Ошибка реконструкции" : "Расстояние";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("RecognitionResult{");
        sb.append("mode=").append(mode);
        sb.append(", recognized=").append(recognized);
        sb.append(", score=").append(String.format("%.4f", score));
        sb.append(", threshold=").append(String.format("%.4f", threshold));
        if (shapeClass != null) {
            sb.append(", class=").append(shapeClass.getDisplayName());
        } else {
            sb.append(", class=null");
        }
        if (mode == RecognitionMode.SUBSPACE && classScores != null) {
            sb.append(", scores=").append(classScores);
        }
        sb.append("}");
        return sb.toString();
    }
}