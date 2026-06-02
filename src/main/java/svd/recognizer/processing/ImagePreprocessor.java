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
    private static final int BBOX_PADDING = 12;

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
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        Mat equalized = new Mat();
        clahe.apply(gray, equalized);

        Mat blurred = new Mat();
        Imgproc.GaussianBlur(equalized, blurred, new Size(9, 9), 0.0);

        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(
                blurred,
                binary,
                255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV,
                31,
                10
        );

        return binary;
    }

    private Mat morphClean(Mat binary) {
        Mat kernelDilate = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat kernelClose = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5));
        Mat kernelOpen = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(3, 3));

        Mat dilated = new Mat();
        Mat closed  = new Mat();
        Mat opened  = new Mat();

        Imgproc.dilate(binary, dilated, kernelDilate);
        Imgproc.morphologyEx(dilated, closed, Imgproc.MORPH_CLOSE, kernelClose);
        Imgproc.morphologyEx(closed,  opened, Imgproc.MORPH_OPEN,  kernelOpen);

        return opened;
    }

    private Mat extractROI(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(
                binary.clone(),
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        if (contours.isEmpty()) {
            return binary.clone();
        }

        MatOfPoint largest = contours.get(0);
        double maxArea = Imgproc.contourArea(largest);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                largest = contour;
                maxArea = area;
            }
        }

        Rect rect = Imgproc.boundingRect(largest);

        int x = Math.max(0, rect.x - BBOX_PADDING);
        int y = Math.max(0, rect.y - BBOX_PADDING);
        int w = Math.min(binary.cols() - x, rect.width  + 2 * BBOX_PADDING);
        int h = Math.min(binary.rows() - y, rect.height + 2 * BBOX_PADDING);

        return new Mat(binary, new Rect(x, y, w, h)).clone();
    }

    private Mat resizeToOutput(Mat source) {
        Mat resized = new Mat();
        Imgproc.resize(
                source,
                resized,
                new Size(OUTPUT_SIZE, OUTPUT_SIZE),
                0, 0,
                Imgproc.INTER_AREA
        );
        return resized;
    }

    private Mat alignToCanonical(Mat binary, String debugName) {
        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.findContours(
                binary.clone(),
                contours,
                new Mat(),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        if (contours.isEmpty()) {
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "fallback_no_contours.png", fallback);
            return fallback;
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
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "fallback_bad_triangle.png", fallback);
            return fallback;
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
            Imgproc.warpAffine(
                    binary,
                    warped,
                    transform,
                    new Size(OUTPUT_SIZE, OUTPUT_SIZE),
                    Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT,
                    new Scalar(0)
            );

            // Восстановить бинарность после интерполяции warpAffine:
            // INTER_LINEAR создаёт промежуточные серые значения на границах.
            // Threshold возвращает чистые 0/255.
            Imgproc.threshold(warped, warped, 64, 255, Imgproc.THRESH_BINARY);

            saveDebugMat(debugName, String.format("perm_%d.png", i), warped);

            double score = scoreTriangleAlignment(warped);
            System.out.println("perm " + i + " score = " + score);

            if (score > bestScore) {
                bestScore = score;
                best = warped;
                bestIndex = i;
            }
        }

        System.out.println("best permutation index = " + bestIndex + ", score = " + bestScore);

        if (best == null) {
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "fallback_no_best.png", fallback);
            return fallback;
        }

        return best;
    }

    private Point[] approxTriangle(MatOfPoint contour, String debugName) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);

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

        System.out.println("approxPolyDP: не удалось получить 3 точки (периметр=" + String.format("%.1f", perimeter) + ")");
        return null;
    }

    private Point[] readTrianglePoints(Mat triangle) {
        if (triangle == null || triangle.empty()) {
            return null;
        }

        if (triangle.rows() == 3 && triangle.cols() == 1) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] v = triangle.get(i, 0);
                if (v == null || v.length < 2) return null;
                pts[i] = new Point(v[0], v[1]);
            }
            return pts;
        }

        if (triangle.rows() == 3 && triangle.cols() == 2 && triangle.channels() == 1) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] x = triangle.get(i, 0);
                double[] y = triangle.get(i, 1);
                if (x == null || y == null) return null;
                pts[i] = new Point(x[0], y[0]);
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

        return null;
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

    private double scoreTriangleAlignment(Mat img) {
        Mat mask = buildCanonicalMask();

        double inside  = 0.0;
        double outside = 0.0;

        for (int y = 0; y < img.rows(); y++) {
            for (int x = 0; x < img.cols(); x++) {
                double[] pv = img.get(y, x);
                double[] mv = mask.get(y, x);

                double value     = (pv == null) ? 0.0 : pv[0];
                double maskValue = (mv == null) ? 0.0 : mv[0];

                if (maskValue > 0) {
                    inside  += value;
                } else {
                    outside += value;
                }
            }
        }

        return inside - outside;
    }

    private Mat buildCanonicalMask() {
        Mat mask = Mat.zeros(OUTPUT_SIZE, OUTPUT_SIZE, CvType.CV_8UC1);
        MatOfPoint poly = new MatOfPoint(
                new Point(32, 4),
                new Point(4, 60),
                new Point(60, 60)
        );
        Imgproc.fillConvexPoly(mask, poly, new Scalar(255));
        return mask;
    }

    // ─── Debug helpers ────────────────────────────────────────────────────────

    private void saveDebugMat(String folderName, String fileName, Mat mat) {
        if (!DEBUG_SAVE || mat == null || mat.empty()) return;

        try {
            Path dir = Paths.get(DEBUG_DIR, folderName);
            Files.createDirectories(dir);

            Mat out;
            if (mat.type() == CvType.CV_8UC1 || mat.type() == CvType.CV_8UC3) {
                out = mat;
            } else {
                out = new Mat();
                mat.convertTo(out, CvType.CV_8UC1);
            }

            boolean ok = Imgcodecs.imwrite(dir.resolve(fileName).toString(), out);
            if (!ok) {
                System.err.println("Не удалось сохранить debug image: " + dir.resolve(fileName));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void saveContourPreview(Mat binary, MatOfPoint contour, String folderName, String fileName) {
        if (!DEBUG_SAVE || binary == null || binary.empty() || contour == null) return;

        Mat preview = new Mat();
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);

        List<MatOfPoint> one = new ArrayList<>();
        one.add(contour);
        Imgproc.drawContours(preview, one, 0, new Scalar(0, 255, 0), 1);

        Rect rect = Imgproc.boundingRect(contour);
        Imgproc.rectangle(preview,
                new Point(rect.x, rect.y),
                new Point(rect.x + rect.width, rect.y + rect.height),
                new Scalar(255, 0, 0), 1);

        saveDebugMat(folderName, fileName, preview);
    }

    private void saveApproxPreview(MatOfPoint contour, Point[] pts, String folderName, String fileName) {
        if (!DEBUG_SAVE || contour == null || pts == null || pts.length != 3) return;

        Rect rect = Imgproc.boundingRect(contour);
        Mat preview = Mat.zeros(
                rect.y + rect.height + BBOX_PADDING,
                rect.x + rect.width  + BBOX_PADDING,
                CvType.CV_8UC3
        );

        List<MatOfPoint> one = new ArrayList<>();
        one.add(contour);
        Imgproc.drawContours(preview, one, 0, new Scalar(80, 80, 80), 1);

        Imgproc.line(preview, pts[0], pts[1], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[1], pts[2], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[2], pts[0], new Scalar(0, 255, 255), 1);

        Scalar[] colors = {new Scalar(0, 0, 255), new Scalar(0, 255, 0), new Scalar(255, 0, 0)};
        String[] labels = {"P0", "P1", "P2"};
        for (int i = 0; i < 3; i++) {
            Imgproc.circle(preview, pts[i], 4, colors[i], -1);
            Imgproc.putText(preview, labels[i], pts[i],
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1);
        }

        saveDebugMat(folderName, fileName, preview);
    }

    private void saveTrianglePreview(Mat binary, Mat triangle, String folderName, String fileName) {
        if (!DEBUG_SAVE || binary == null || binary.empty()) return;

        Point[] pts = readTrianglePoints(triangle);
        if (pts == null) return;

        Mat preview = new Mat();
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);

        Imgproc.line(preview, pts[0], pts[1], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[1], pts[2], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[2], pts[0], new Scalar(0, 255, 255), 1);

        Scalar[] colors = {new Scalar(0, 0, 255), new Scalar(0, 255, 0), new Scalar(255, 0, 0)};
        String[] labels = {"P0", "P1", "P2"};
        for (int i = 0; i < 3; i++) {
            Imgproc.circle(preview, pts[i], 3, colors[i], -1);
            Imgproc.putText(preview, labels[i], pts[i],
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1);
        }

        saveDebugMat(folderName, fileName, preview);
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private BufferedImage matToBufferedImage(Mat source) {
        Mat normalized;
        if (source.type() != CvType.CV_8UC1) {
            normalized = new Mat();
            source.convertTo(normalized, CvType.CV_8UC1);
        } else {
            normalized = source;
        }

        BufferedImage image = new BufferedImage(
                normalized.cols(), normalized.rows(), BufferedImage.TYPE_BYTE_GRAY
        );

        byte[] src = new byte[(int) (normalized.total() * normalized.channels())];
        normalized.get(0, 0, src);

        byte[] dst = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(src, 0, dst, 0, src.length);

        return image;
    }

    private double[][] imageToMatrix(BufferedImage image) {
        double[][] matrix = new double[image.getHeight()][image.getWidth()];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                matrix[y][x] = (image.getRGB(x, y) & 0xFF) / 255.0;
            }
        }

        return matrix;
    }
}
