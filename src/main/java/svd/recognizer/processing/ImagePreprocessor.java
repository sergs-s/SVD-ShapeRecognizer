package svd.recognizer.processing;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ImagePreprocessor {

    private static final int OUTPUT_SIZE = 64;
    private static final int BINARY_THRESHOLD = 128;

    // Канонические вершины треугольника: верхний центр, нижний левый, нижний правый.
    private static final Point[] CANONICAL = {
        new Point(OUTPUT_SIZE / 2.0, 2),
        new Point(2, OUTPUT_SIZE - 2),
        new Point(OUTPUT_SIZE - 2, OUTPUT_SIZE - 2)
    };

    // Padding вокруг bounding box контура при кропе.
    // Увеличен с 12 до 24, чтобы вершины треугольника не вылетали за край холста при warpAffine.
    private static final int BBOX_PADDING = 24;

    private final String debugOutputDir;

    public ImagePreprocessor(String debugOutputDir) {
        this.debugOutputDir = debugOutputDir;
    }

    /**
     * Препроцессинг изображения фигуры:
     * 1. Grayscale
     * 2. Бинаризация (Otsu)
     * 3. Морфологическая очистка (denoise)
     * 4. Кроп по bounding box наибольшего контура + padding
     * 5. Выравнивание по каноническому треугольнику (affine)
     * 6. Resize до OUTPUT_SIZE×OUTPUT_SIZE
     */
    public Mat preprocess(Mat src, String debugName) {
        // 1. Grayscale
        Mat gray = new Mat();
        if (src.channels() == 3) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = src.clone();
        }
        saveDebugMat(debugName, "01_source_gray.png", gray);

        // 2. Бинаризация Otsu
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        saveDebugMat(debugName, "02_binary.png", binary);

        // 3. Денойз — убрать мелкие артефакты
        Mat kernel3 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(binary, cleaned, Imgproc.MORPH_OPEN, kernel3);
        saveDebugMat(debugName, "03_cleaned.png", cleaned);

        // 4. Кроп по bounding box наибольшего контура
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cleaned.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            Mat fallback = resizeToOutput(cleaned);
            saveDebugMat(debugName, "fallback_no_contours_crop.png", fallback);
            return fallback;
        }

        MatOfPoint largest = null;
        double maxArea = -1;
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
        int w = Math.min(cleaned.cols() - x, rect.width + 2 * BBOX_PADDING);
        int h = Math.min(cleaned.rows() - y, rect.height + 2 * BBOX_PADDING);

        Mat roi = new Mat(cleaned, new Rect(x, y, w, h)).clone();

        // Удалить точечный шум из кропнутой области (ядро 7x7 убирает объекты <7px,
        // линии треугольника толще и сохраняются)
        Mat kernelDenoise = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, new Size(7, 7));
        Mat denoised = new Mat();
        Imgproc.morphologyEx(roi, denoised, Imgproc.MORPH_OPEN, kernelDenoise);
        saveDebugMat(debugName, "04_cropped.png", denoised);

        // 5. Affine-выравнивание по каноническому треугольнику
        return alignToCanonical(denoised, debugName);
    }

    private Mat alignToCanonical(Mat binary, String debugName) {
        // Найти контуры в кропнутом изображении
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
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "fallback_no_contours.png", fallback);
            return fallback;
        }

        MatOfPoint largest = contours.stream()
                .max(Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(contours.get(0));

        saveContourPreview(binary, largest, debugName, "05a_largest_contour.png");

        // Построить канонический треугольник для сравнения
        Mat canonicalMask = Mat.zeros(new Size(OUTPUT_SIZE, OUTPUT_SIZE), CvType.CV_8UC1);
        MatOfPoint canonicalPts = new MatOfPoint(CANONICAL);
        List<MatOfPoint> canonicalList = new ArrayList<>();
        canonicalList.add(canonicalPts);
        Imgproc.drawContours(canonicalMask, canonicalList, 0, new Scalar(255), 1);
        saveDebugMat(debugName, "05_canonical_mask.png", canonicalMask);

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
                    Imgproc.INTER_NEAREST,
                    Core.BORDER_CONSTANT,
                    new Scalar(0)
            );

            // Восстановить бинарность после интерполяции warpAffine:
            // INTER_NEAREST не даёт серых, но threshold оставляем как страховку.
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
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "fallback_no_best.png", fallback);
            return fallback;
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
                double x = triangle.get(i, 0)[0];
                double y = triangle.get(i, 1)[0];
                pts[i] = new Point(x, y);
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

        // Универсальный fallback: перебрать все ячейки
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

    /**
     * Оценивает, насколько хорошо двоичное изображение соответствует
     * каноническому треугольнику вершиной вверх.
     *
     * Метрика: количество белых пикселей в верхней зоне (треугольник должен иметь
     * вершину вверху) минус количество в нижней зоне вне основания.
     *
     * Точнее: делим OUTPUT_SIZE×OUTPUT_SIZE на три горизонтальных полосы
     * и взвешиваем попадание белых пикселей в ожидаемые позиции.
     */
    private double scoreTriangleAlignment(Mat warped) {
        if (warped.empty()) return Double.NEGATIVE_INFINITY;

        int sz = OUTPUT_SIZE;
        double score = 0;

        for (int row = 0; row < warped.rows(); row++) {
            for (int col = 0; col < warped.cols(); col++) {
                double pixel = warped.get(row, col)[0];
                if (pixel < 128) continue; // фон

                // Ожидаемая ширина треугольника на данной строке (вершина вверху, основание внизу)
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
                binary,
                labels,
                stats,
                centroids,
                8,
                CvType.CV_32S
        );

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

    private void saveDebugMat(String debugName, String filename, Mat mat) {
        if (debugOutputDir == null || debugName == null) return;
        try {
            File dir = new File(debugOutputDir, debugName);
            dir.mkdirs();
            File out = new File(dir, filename);
            // OpenCV не умеет сохранять через стандартные средства без highgui;
            // конвертируем Mat → BufferedImage → PNG
            BufferedImage img = matToBufferedImage(mat);
            ImageIO.write(img, "png", out);
        } catch (IOException e) {
            System.err.println("Не удалось сохранить дебаг-изображение: " + e.getMessage());
        }
    }

    private void saveContourPreview(Mat binary, MatOfPoint contour, String debugName, String filename) {
        Mat preview = Mat.zeros(binary.size(), CvType.CV_8UC3);
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);
        List<MatOfPoint> list = new ArrayList<>();
        list.add(contour);
        Imgproc.drawContours(preview, list, 0, new Scalar(0, 255, 0), 1);

        // Нарисовать bounding box
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
        // Создаём пустой холст подходящего размера
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

        // Пометить вершины
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

        // Нарисовать стороны треугольника
        Scalar yellow = new Scalar(0, 255, 255);
        Imgproc.line(preview, pts[0], pts[1], yellow, 1);
        Imgproc.line(preview, pts[1], pts[2], yellow, 1);
        Imgproc.line(preview, pts[2], pts[0], yellow, 1);

        // Пометить вершины
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
        int type;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();
        byte[] data = new byte[width * height * channels];
        mat.get(0, 0, data);

        BufferedImage img = new BufferedImage(width, height, type);
        img.getRaster().setDataElements(0, 0, width, height,
                type == BufferedImage.TYPE_3BYTE_BGR
                        ? bgrToRgb(data)
                        : data);
        return img;
    }

    private byte[] bgrToRgb(byte[] bgr) {
        byte[] rgb = new byte[bgr.length];
        for (int i = 0; i < bgr.length; i += 3) {
            rgb[i]     = bgr[i + 2]; // R
            rgb[i + 1] = bgr[i + 1]; // G
            rgb[i + 2] = bgr[i];     // B
        }
        return rgb;
    }
}
