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