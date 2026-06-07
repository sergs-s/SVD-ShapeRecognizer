package svd.recognizer.core;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

/**
 * Вспомогательный извлекатель контурных признаков.
 *
 * Этот класс не является основным трактом распознавания, потому что рабочее
 * решение в проекте основано на SVD-сигнатурах. Однако он оставлен как
 * экспериментальный инструмент для сравнения контурных и спектральных признаков.
 *
 * Алгоритм:
 * 1. Найти внешние контуры на бинарном изображении
 * 2. Выбрать контур с максимальной площадью
 * 3. Равномерно дискретизировать contour featureSize точками
 * 4. Построить вектор координат
 * 5. Нормировать его по евклидовой норме
 *
 * @author ssv
 */
public class ContourFeatureExtractor {
    private final int featureSize;

    public ContourFeatureExtractor(int featureSize) {
        this.featureSize = featureSize;
    }

    /**
     * Извлекает контурный признаковый вектор из бинарного изображения: находит
     * внешние контуры, берёт крупнейший и строит из него вектор фиксированной
     * длины.
     *
     * @param binaryMat бинарное изображение фигуры
     * @return нормированный вектор признаков длиной featureSize·2 или null, если контуров нет
     */
    public double[] extract(Mat binaryMat) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binaryMat.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        if (contours.isEmpty()) {
            return null;
        }
        MatOfPoint largest = getLargestContour(contours);
        return buildFeatureVector(largest);
    }

    /**
     * Возвращает контур с наибольшей площадью (основная фигура, а не шум).
     *
     * @param contours список найденных контуров
     * @return контур максимальной площади
     */
    private MatOfPoint getLargestContour(List<MatOfPoint> contours) {
        MatOfPoint largest = contours.get(0);
        double maxArea = Imgproc.contourArea(largest);
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                largest = contour;
                maxArea = area;
            }
        }
        return largest;
    }

    /**
     * Строит вектор признаков, равномерно выбирая featureSize точек вдоль
     * контура (индекс i отображается в позицию i/featureSize · total) и
     * раскладывая их координаты (x, y) подряд, после чего нормирует вектор.
     *
     * @param contour контур фигуры
     * @return нормированный вектор координат длиной featureSize·2
     */
    private double[] buildFeatureVector(MatOfPoint contour) {
        Point[] points = contour.toArray();
        double[] featureVector = new double[featureSize * 2];
        int total = points.length;
        for (int i = 0; i < featureSize; i++) {
            int index = (int) ((double) i / featureSize * total);
            featureVector[2 * i] = points[index].x;
            featureVector[2 * i + 1] = points[index].y;
        }
        return normalize(featureVector);
    }

    /**
     * Нормировка вектора по евклидовой норме (деление на длину), чтобы признак
     * не зависел от масштаба фигуры.
     *
     * @param vector исходный вектор
     * @return вектор единичной длины (или исходный, если он нулевой)
     */
    private double[] normalize(double[] vector) {
        double sum = 0.0;
        for (double v : vector) {
            sum += v * v;
        }
        double norm = Math.sqrt(sum);
        if (norm == 0.0) {
            return vector;
        }
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }
}