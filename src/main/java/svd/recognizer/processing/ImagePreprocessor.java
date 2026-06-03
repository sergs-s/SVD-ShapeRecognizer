package svd.recognizer.processing;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {

    public static final int OUTPUT_SIZE = 64;
    // Увеличен с 12 до 24: вершины треугольника не вылетают за край холста при warpAffine
    private static final int BBOX_PADDING = 24;

    private static final boolean DEBUG_SAVE = true;
    private static final String DEBUG_DIR = "debug-preprocess";

    private static final Point[] CANONICAL = {
        new Point(32, 4),
        new Point(4, 60),
        new Point(60, 60)
    };

    public static class PreprocessResult {

        private final BufferedImage image;
        private final double[][] matrix;

        public PreprocessResult(BufferedImage image, double[][] matrix) {
            this.image = image;
            this.matrix = matrix;
        }

        public BufferedImage getImage() {
            return image;
        }

        public double[][] getMatrix() {
            return matrix;
        }
    }

    public PreprocessResult preprocess(File imageFile) throws Exception {
        Mat source = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (source.empty()) {
            throw new IllegalArgumentException(
                    "Не удалось загрузить изображение: " + imageFile.getAbsolutePath()
            );
        }

        String debugName = sanitizeFileName(imageFile.getName());

        Mat gray = toGrayscale(source);
        saveDebugMat(debugName, "01_source_gray.png", gray);

        Mat binary = binarize(gray);
        saveDebugMat(debugName, "02_binary.png", binary);

        Mat cleaned = morphClean(binary);
        saveDebugMat(debugName, "03_cleaned.png", cleaned);

        Mat cropped = extractROI(cleaned);
        saveDebugMat(debugName, "04_cropped.png", cropped);

        Mat canonicalMask = buildCanonicalMask();
        saveDebugMat(debugName, "05_canonical_mask.png", canonicalMask);

        Mat aligned = alignToCanonical(cropped, debugName);
        saveDebugMat(debugName, "06_best_aligned.png", aligned);

        BufferedImage image = matToBufferedImage(aligned);
        double[][] matrix = imageToMatrix(image);

        return new PreprocessResult(image, matrix);
    }

    private Mat toGrayscale(Mat source) {
        Mat gray = new Mat();
        if (source.channels() == 1) {
            gray = source.clone();
        } else {
            Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
        }
        return gray;
    }

    private Mat binarize(Mat gray) {
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        return binary;
    }

    private Mat morphClean(Mat binary) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(binary, cleaned, Imgproc.MORPH_OPEN, kernel);
        return cleaned;
    }

    private Mat extractROI(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return binary;
        }

        MatOfPoint largest = contours.stream()
                .max(Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(contours.get(0));

        Rect rect = Imgproc.boundingRect(largest);

        int x = Math.max(0, rect.x - BBOX_PADDING);
        int y = Math.max(0, rect.y - BBOX_PADDING);
        int w = Math.min(binary.cols() - x, rect.width + 2 * BBOX_PADDING);
        int h = Math.min(binary.rows() - y, rect.height + 2 * BBOX_PADDING);

        Mat roi = new Mat(binary, new Rect(x, y, w, h)).clone();

        Mat kernelDenoise = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
        Mat denoised = new Mat();
        Imgproc.morphologyEx(roi, denoised, Imgproc.MORPH_OPEN, kernelDenoise);

        return denoised;
    }

    private Mat buildCanonicalMask() {
        Mat mask = Mat.zeros(new Size(OUTPUT_SIZE, OUTPUT_SIZE), CvType.CV_8UC1);
        MatOfPoint pts = new MatOfPoint(CANONICAL);
        List<MatOfPoint> list = new ArrayList<>();
        list.add(pts);
        Imgproc.drawContours(mask, list, 0, new Scalar(255), 1);
        return mask;
    }

    private Mat alignToCanonical(Mat binary, String debugName) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return resizeToOutput(binary);
        }

        MatOfPoint largest = contours.stream()
                .max(Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(contours.get(0));

        saveContourPreview(binary, largest, debugName, "05a_largest_contour.png");

        Point[] srcPts = approxTriangle(largest, debugName);

        if (srcPts == null) {
            System.out.println("approxPolyDP не дал 3 точек, fallback → minEnclosingTriangle");
            Mat triangleMat = new Mat();
            Imgproc.minEnclosingTriangle(new MatOfPoint2f(largest.toArray()), triangleMat);
            saveTrianglePreview(binary, triangleMat, debugName, "05b_min_enclosing_triangle.png");
            srcPts = readTrianglePoints(triangleMat);
        }

        if (srcPts == null) {
            return resizeToOutput(binary);
        }

        Point[][] perms = permutations(srcPts);

        Mat best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;

        for (int i = 0; i < perms.length; i++) {
            MatOfPoint2f src = new MatOfPoint2f(perms[i]);
            MatOfPoint2f dst = new MatOfPoint2f(CANONICAL);
            Mat transform = Imgproc.getAffineTransform(src, dst);

            Mat warped = new Mat();
            Imgproc.warpAffine(binary, warped, transform,
                    new Size(OUTPUT_SIZE, OUTPUT_SIZE),
                    Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));

            Imgproc.threshold(warped, warped, 64, 255, Imgproc.THRESH_BINARY);
            warped = removeSmallComponents(warped, 6);

            saveDebugMat(debugName, String.format("perm_%d.png", i), warped);

            double score = scoreTriangleAlignment(warped);
            System.out.println("perm " + i + " score = " + score);

            if (score > bestScore) {
                bestScore = score;
                best = warped.clone();
                bestIndex = i;
            }
        }

        System.out.println("best permutation index = " + bestIndex + ", score = " + bestScore);

        if (best == null) {
            return resizeToOutput(binary);
        }

        return best;
    }

    private Point[] approxTriangle(MatOfPoint contour, String debugName) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);

        // Шаг 1: грубые значения 10%..1%
        for (int pct = 10; pct >= 1; pct--) {
            double epsilon = (pct / 100.0) * perimeter;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

            if (approx.rows() == 3) {
                Point[] pts = approx.toArray();
                System.out.println("approxPolyDP: 3 точки при epsilon=" + String.format("%.1f", epsilon)
                        + " (" + pct + "% периметра)");
                saveApproxPreview(contour, pts, debugName, "05b_approx_triangle.png");
                return pts;
            }
        }

        // Шаг 2: мелкие значения 0.9%..0.5% — для сложных контуров (перевёрнутые, вогнутые)
        for (int tenths = 9; tenths >= 5; tenths--) {
            double epsilon = (tenths / 1000.0) * perimeter;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

            if (approx.rows() == 3) {
                Point[] pts = approx.toArray();
                System.out.println("approxPolyDP: 3 точки при epsilon=" + String.format("%.1f", epsilon)
                        + " (0." + tenths + "% периметра)");
                saveApproxPreview(contour, pts, debugName, "05b_approx_triangle.png");
                return pts;
            }
        }

        System.out.println("approxPolyDP: не удалось получить 3 точки (периметр=" + String.format("%.1f", perimeter) + ")");
        return null;
    }

    private Point[] readTrianglePoints(Mat triangle) {
        if (triangle == null || triangle.empty()) return null;

        if (triangle.rows() == 3 && triangle.cols() == 1) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] v = triangle.get(i, 0);
                if (v == null || v.length < 2) return null;
                pts[i] = new Point(v[0], v[1]);
            }
            return pts;
        }

        if (triangle.rows() == 1 && triangle.cols() == 3) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] v = triangle.get(0, i);
                if (v == null || v.length < 2) return null;
                pts[i] = new Point(v[0], v[1]);
            }
            return pts;
        }

        List<Point> pts = new ArrayList<>();
        for (int r = 0; r < triangle.rows() && pts.size() < 3; r++) {
            for (int c = 0; c < triangle.cols() && pts.size() < 3; c++) {
                double[] v = triangle.get(r, c);
                if (v != null && v.length >= 2) {
                    pts.add(new Point(v[0], v[1]));
                }
            }
        }
        return pts.size() == 3 ? pts.toArray(new Point[0]) : null;
    }

    private double scoreTriangleAlignment(Mat warped) {
        if (warped.empty()) return Double.NEGATIVE_INFINITY;

        int sz = OUTPUT_SIZE;
        double score = 0;

        for (int row = 0; row < warped.rows(); row++) {
            for (int col = 0; col < warped.cols(); col++) {
                double pixel = warped.get(row, col)[0];
                if (pixel < 128) continue;

                double expectedHalfWidth = (row / (double) sz) * (sz / 2.0);
                double centerCol = sz / 2.0;
                double leftBound = centerCol - expectedHalfWidth;
                double rightBound = centerCol + expectedHalfWidth;

                if (col >= leftBound && col <= rightBound) {
                    score += 1.0;
                } else {
                    score -= 0.5;
                }
            }
        }

        return score;
    }

    private Point[][] permutations(Point[] pts) {
        if (pts == null || pts.length != 3) {
            throw new IllegalArgumentException("Нужно ровно 3 точки");
        }
        return new Point[][]{
            {pts[0], pts[1], pts[2]},
            {pts[0], pts[2], pts[1]},
            {pts[1], pts[0], pts[2]},
            {pts[1], pts[2], pts[0]},
            {pts[2], pts[0], pts[1]},
            {pts[2], pts[1], pts[0]}
        };
    }

    private Mat removeSmallComponents(Mat binary, int minArea) {
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();

        int n = Imgproc.connectedComponentsWithStats(
                binary, labels, stats, centroids, 8, CvType.CV_32S);

        Mat cleaned = Mat.zeros(binary.size(), CvType.CV_8UC1);

        for (int label = 1; label < n; label++) {
            int area = (int) stats.get(label, Imgproc.CC_STAT_AREA)[0];
            if (area >= minArea) {
                Mat mask = new Mat();
                Core.compare(labels, new Scalar(label), mask, Core.CMP_EQ);
                cleaned.setTo(new Scalar(255), mask);
            }
        }

        return cleaned;
    }

    private Mat resizeToOutput(Mat src) {
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(OUTPUT_SIZE, OUTPUT_SIZE), 0, 0, Imgproc.INTER_NEAREST);
        return resized;
    }

    // ─────────────────────────────── debug helpers ───────────────────────────────

    private void saveDebugMat(String debugName, String fileName, Mat mat) {
        if (!DEBUG_SAVE) return;
        try {
            Path dir = Paths.get(DEBUG_DIR, debugName);
            Files.createDirectories(dir);
            Mat out = mat;
            if (mat.type() == CvType.CV_64F) {
                out = new Mat();
                mat.convertTo(out, CvType.CV_8U, 255.0);
            }
            boolean ok = Imgcodecs.imwrite(dir.resolve(fileName).toString(), out);
            if (!ok) {
                System.err.println("Не удалось сохранить: " + dir.resolve(fileName));
            }
        } catch (IOException e) {
            System.err.println("saveDebugMat: " + e.getMessage());
        }
    }

    private void saveContourPreview(Mat binary, MatOfPoint contour, String debugName, String filename) {
        Mat preview = Mat.zeros(binary.size(), CvType.CV_8UC3);
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);
        List<MatOfPoint> list = new ArrayList<>();
        list.add(contour);
        Imgproc.drawContours(preview, list, 0, new Scalar(0, 255, 0), 1);

        Rect rect = Imgproc.boundingRect(contour);
        int x = Math.max(0, rect.x - BBOX_PADDING);
        int y = Math.max(0, rect.y - BBOX_PADDING);
        int w = Math.min(binary.cols() - x, rect.width + 2 * BBOX_PADDING);
        int h = Math.min(binary.rows() - y, rect.height + 2 * BBOX_PADDING);
        Imgproc.rectangle(preview, new Point(x, y), new Point(x + w, y + h),
                new Scalar(255, 0, 0), 1);

        saveDebugMat(debugName, filename, preview);
    }

    private void saveApproxPreview(MatOfPoint contour, Point[] approx, String debugName, String filename) {
        Rect bound = Imgproc.boundingRect(contour);
        int margin = 20;
        int W = bound.x + bound.width + margin;
        int H = bound.y + bound.height + margin;
        Mat canvas = Mat.zeros(H, W, CvType.CV_8UC3);

        List<MatOfPoint> list = new ArrayList<>();
        list.add(contour);
        Imgproc.drawContours(canvas, list, 0, new Scalar(255, 255, 255), 1);

        MatOfPoint approxPoly = new MatOfPoint(approx);
        List<MatOfPoint> approxList = new ArrayList<>();
        approxList.add(approxPoly);
        Imgproc.drawContours(canvas, approxList, 0, new Scalar(0, 255, 255), 1);

        String[] labels = {"P1", "P2", "P3"};
        Scalar[] colors = {new Scalar(0, 255, 0), new Scalar(0, 0, 255), new Scalar(255, 0, 0)};
        for (int i = 0; i < approx.length; i++) {
            Imgproc.circle(canvas, approx[i], 4, colors[i], -1);
            Imgproc.putText(canvas, labels[i], new Point(approx[i].x + 5, approx[i].y - 5),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1);
        }

        saveDebugMat(debugName, filename, canvas);
    }

    private void saveTrianglePreview(Mat binary, Mat triangleMat, String debugName, String filename) {
        Mat preview = Mat.zeros(binary.size(), CvType.CV_8UC3);
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);

        Point[] pts = readTrianglePoints(triangleMat);
        if (pts == null || pts.length < 3) {
            saveDebugMat(debugName, filename, preview);
            return;
        }

        Scalar yellow = new Scalar(0, 255, 255);
        Imgproc.line(preview, pts[0], pts[1], yellow, 1);
        Imgproc.line(preview, pts[1], pts[2], yellow, 1);
        Imgproc.line(preview, pts[2], pts[0], yellow, 1);

        String[] labels = {"P1", "P2", "P3"};
        Scalar[] colors = {new Scalar(0, 255, 0), new Scalar(0, 0, 255), new Scalar(255, 0, 0)};
        for (int i = 0; i < pts.length; i++) {
            Imgproc.circle(preview, pts[i], 5, colors[i], -1);
            Imgproc.putText(preview, labels[i], new Point(pts[i].x + 6, pts[i].y - 6),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
        }

        saveDebugMat(debugName, filename, preview);
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        Mat display = mat;
        if (mat.channels() == 1) {
            display = new Mat();
            Imgproc.cvtColor(mat, display, Imgproc.COLOR_GRAY2BGR);
        }
        int width = display.cols();
        int height = display.rows();
        byte[] data = new byte[width * height * (int) display.elemSize()];
        display.get(0, 0, data);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        byte[] imgData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, imgData, 0, data.length);
        return image;
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._\\-]", "_");
    }

    private double[][] imageToMatrix(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        double[][] matrix = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                matrix[y][x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }
        return matrix;
    }
}
