package svd.recognizer.core;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author ssv
 */
public class ContourFeatureExtractor {

    private final int featureSize;

    /**
     * @param featureSize number of feature points sampled from contour
     */
    public ContourFeatureExtractor(int featureSize) {
        this.featureSize = featureSize;
    }

    /**
     * Extracts a normalized feature vector from a binary image.
     *
     * @param binaryMat preprocessed binary Mat
     * @return double[] feature vector, or null if no contour found
     */
    public double[] extract(org.opencv.core.Mat binaryMat) {
        List<org.opencv.core.MatOfPoint> contours = new ArrayList<>();
        org.opencv.core.Mat hierarchy = new org.opencv.core.Mat();
        org.opencv.imgproc.Imgproc.findContours(binaryMat.clone(), contours, hierarchy,
                org.opencv.imgproc.Imgproc.RETR_EXTERNAL,
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE);

        if (contours.isEmpty()) return null;

        org.opencv.core.MatOfPoint largest = getLargestContour(contours);
        return buildFeatureVector(largest);
    }

    private org.opencv.core.MatOfPoint getLargestContour(List<org.opencv.core.MatOfPoint> contours) {
        org.opencv.core.MatOfPoint largest = contours.get(0);
        double maxArea = org.opencv.imgproc.Imgproc.contourArea(largest);
        for (org.opencv.core.MatOfPoint c : contours) {
            double area = org.opencv.imgproc.Imgproc.contourArea(c);
            if (area > maxArea) {
                maxArea = area;
                largest = c;
            }
        }
        return largest;
    }

    private double[] buildFeatureVector(org.opencv.core.MatOfPoint contour) {
        org.opencv.core.Point[] points = contour.toArray();
        double[] features = new double[featureSize * 2];
        int total = points.length;
        for (int i = 0; i < featureSize; i++) {
            int idx = (int) ((double) i / featureSize * total);
            features[2 * i] = points[idx].x;
            features[2 * i + 1] = points[idx].y;
        }
        return normalize(features);
    }

    private double[] normalize(double[] v) {
        double sum = 0;
        for (double d : v) sum += d * d;
        double norm = Math.sqrt(sum);
        if (norm == 0) return v;
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) result[i] = v[i] / norm;
        return result;
    }
}
