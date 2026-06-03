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
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {

    public static final int OUTPUT_SIZE = 64;
    private static final int BBOX_PADDING = 24;

    private static final boolean DEBUG_SAVE = true;
    private static final String DEBUG_DIR = "debug-preprocess";

    private static final Point[] CANONICAL = {
        new Point(32, 4),
        new Point(4, 60),
        new Point(60, 60)
    };

    private static final int NORMALIZED_SHAPE_SIZE = 180;
    private static final int NORMALIZED_PADDING = 12;

    // WARP_SCALE=4 (вместо 3): даёт запас 256×256 для скошенных и перевёрнутых треугольников
    private static final int WARP_SCALE = 4;
    private static final int WARP_SIZE  = OUTPUT_SIZE * WARP_SCALE; // 256

    private static final Point[] CANONICAL_BIG = {
        new Point(CANONICAL[0].x * WARP_SCALE, CANONICAL[0].y * WARP_SCALE), // (128, 16)
        new Point(CANONICAL[1].x * WARP_SCALE, CANONICAL[1].y * WARP_SCALE), // (16, 240)
        new Point(CANONICAL[2].x * WARP_SCALE, CANONICAL[2].y * WARP_SCALE)  // (240, 240)
    };

    /**
     * Пороги для оценки вогнутости контура.
     * Если maxDepthNorm >= MAX_DEPTH_NORM_THRESHOLD или
     *    sumDepthNorm >= SUM_DEPTH_NORM_THRESHOLD,
     * контур считается «слишком вогнутым» и approxPolyDP используется только для дебага,
     * а для warp берётся minEnclosingTriangle.
     * Подбирайте на своих примерах.
     */
    private static final double MAX_DEPTH_NORM_THRESHOLD = 0.04;
    private static final double SUM_DEPTH_NORM_THRESHOLD = 0.10;

    // -------------------------------------------------------------------------
    // Публичный record — результат оценки вогнутости
    // -------------------------------------------------------------------------
    public record ConcavityScore(double maxDepthNorm, double sumDepthNorm, int defectCount) {
        public boolean isTooHighFor(double maxThresh, double sumThresh) {
            return maxDepthNorm >= maxThresh || sumDepthNorm >= sumThresh;
        }
    }

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

        Mat normalized = normalizeScale(cropped);
        saveDebugMat(debugName, "04b_normalized.png", normalized);

        Mat canonicalMask = buildCanonicalMask();
        saveDebugMat(debugName, "05_canonical_mask.png", canonicalMask);

        Mat aligned = alignToCanonical(normalized, debugName);
        saveDebugMat(debugName, "06_best_aligned.png", aligned);

        BufferedImage image = matToBufferedImage(aligned);
        double[][] matrix = imageToMatrix(image);

        return new PreprocessResult(image, matrix);
    }

    // =========================================================================
    // Оценка вогнутости контура через convexityDefects
    // =========================================================================

    /**
     * Вычисляет нормированные метрики вогнутости контура.
     *
     * <p>Использует {@code Imgproc.convexityDefects}: каждый дефект описывается
     * как (startIdx, endIdx, farIdx, depth*256). Глубина нормируется на периметр
     * выпуклой оболочки, чтобы быть независимой от масштаба фигуры.
     *
     * @param contour входной контур (MatOfPoint)
     * @return ConcavityScore с maxDepthNorm, sumDepthNorm, defectCount
     */
    public ConcavityScore computeConcavityScore(MatOfPoint contour) {
        // Строим выпуклую оболочку (индексы)
        MatOfInt hullIdx = new MatOfInt();
        Imgproc.convexHull(contour, hullIdx);

        // convexityDefects требует, чтобы hull был отсортирован против часовой стрелки
        // (OpenCV обычно даёт CW — убедимся в правильном порядке через флаг clockwise=false)
        MatOfInt hullIdxCCW = new MatOfInt();
        Imgproc.convexHull(contour, hullIdxCCW, false);

        // Периметр hull для нормировки
        int[] idxArr = hullIdxCCW.toArray();
        Point[] allPts = contour.toArray();
        double hullPerimeter = 0;
        for (int i = 0; i < idxArr.length; i++) {
            Point p1 = allPts[idxArr[i]];
            Point p2 = allPts[idxArr[(i + 1) % idxArr.length]];
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            hullPerimeter += Math.sqrt(dx * dx + dy * dy);
        }

        if (hullPerimeter < 1.0) {
            return new ConcavityScore(0, 0, 0);
        }

        MatOfInt4 defects = new MatOfInt4();
        try {
            Imgproc.convexityDefects(contour, hullIdxCCW, defects);
        } catch (Exception e) {
            // Контур слишком маленький или вырожденный
            System.out.println("convexityDefects: " + e.getMessage());
            return new ConcavityScore(0, 0, 0);
        }

        if (defects.empty()) {
            return new ConcavityScore(0, 0, 0);
        }

        int[] d = defects.toArray();
        int defectCount = d.length / 4;
        double maxDepth = 0;
        double sumDepth = 0;

        for (int i = 0; i < defectCount; i++) {
            // depth хранится как depth * 256 (fixed-point)
            double depth = d[i * 4 + 3] / 256.0;
            if (depth > maxDepth) maxDepth = depth;
            sumDepth += depth;
        }

        double maxDepthNorm = maxDepth / hullPerimeter;
        double sumDepthNorm = sumDepth / hullPerimeter;

        System.out.printf(
            "ConcavityScore: defects=%d  maxDepthNorm=%.4f  sumDepthNorm=%.4f%n",
            defectCount, maxDepthNorm, sumDepthNorm
        );

        return new ConcavityScore(maxDepthNorm, sumDepthNorm, defectCount);
    }

    // =========================================================================
    // Приватные методы обработки (без изменений, кроме approxTriangle)
    // =========================================================================

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

        return new Mat(binary, new Rect(x, y, w, h)).clone();
    }

    /**
     * Любой треугольник любого размера приводим к единому масштабу:
     * находим bbox содержимого, масштабируем так, чтобы max(w,h) = NORMALIZED_SHAPE_SIZE,
     * добавляем паддинг.
     * morphClose залатывает дыры на гранях без внесения лишних пикселей.
     */
    private Mat normalizeScale(Mat binary) {
        Mat closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Mat closed = new Mat();
        Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, closeKernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(closed.clone(), contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return binary;
        }

        Rect bbox = contours.stream()
                .map(Imgproc::boundingRect)
                .reduce((a, b) -> {
                    int x  = Math.min(a.x, b.x);
                    int y  = Math.min(a.y, b.y);
                    int x2 = Math.max(a.x + a.width,  b.x + b.width);
                    int y2 = Math.max(a.y + a.height, b.y + b.height);
                    return new Rect(x, y, x2 - x, y2 - y);
                })
                .orElse(new Rect(0, 0, binary.cols(), binary.rows()));

        int x = Math.max(0, bbox.x);
        int y = Math.max(0, bbox.y);
        int w = Math.min(binary.cols() - x, bbox.width);
        int h = Math.min(binary.rows() - y, bbox.height);

        Mat cropped = new Mat(binary, new Rect(x, y, w, h)).clone();

        double scale = (double) NORMALIZED_SHAPE_SIZE / Math.max(cropped.cols(), cropped.rows());
        int newW = Math.max(1, (int) Math.round(cropped.cols() * scale));
        int newH = Math.max(1, (int) Math.round(cropped.rows() * scale));

        Mat resized = new Mat();
        Imgproc.resize(cropped, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_AREA);
        Imgproc.threshold(resized, resized, 64, 255, Imgproc.THRESH_BINARY);

        Mat padded = new Mat();
        Core.copyMakeBorder(resized, padded,
                NORMALIZED_PADDING, NORMALIZED_PADDING,
                NORMALIZED_PADDING, NORMALIZED_PADDING,
                Core.BORDER_CONSTANT, new Scalar(0));
        return padded;
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
            MatOfPoint2f dst = new MatOfPoint2f(CANONICAL_BIG);
            Mat transform = Imgproc.getAffineTransform(src, dst);

            Mat warpedBig = new Mat();
            Imgproc.warpAffine(binary, warpedBig, transform,
                    new Size(WARP_SIZE, WARP_SIZE),
                    Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));

            Imgproc.threshold(warpedBig, warpedBig, 64, 255, Imgproc.THRESH_BINARY);
            warpedBig = removeSmallComponents(warpedBig, 6 * WARP_SCALE * WARP_SCALE);

            saveDebugMat(debugName, String.format("perm_%d.png", i), warpedBig);

            double score = scoreTriangleAlignment(warpedBig);
            System.out.println("perm " + i + " score = " + score);

            if (score > bestScore) {
                bestScore = score;
                best = warpedBig.clone();
                bestIndex = i;
            }
        }

        System.out.println("best permutation index = " + bestIndex + ", score = " + bestScore);

        if (best == null) {
            return resizeToOutput(binary);
        }

        return cropToBboxAndResize(best);
    }

    /**
     * Геометрический скор пермутации:
     * Правильная ориентация — вершина в верхней трети холста,
     * основание в нижней трети холста.
     * Бонус +1000 за каждое выполненное условие.
     */
    private double scoreTriangleAlignment(Mat warpedBig) {
        if (warpedBig.empty()) return Double.NEGATIVE_INFINITY;

        int H = warpedBig.rows();
        int W = warpedBig.cols();

        int topRow = H;
        for (int r = 0; r < H; r++) {
            if (Core.countNonZero(warpedBig.row(r)) > 0) {
                topRow = r;
                break;
            }
        }
        boolean apexOk = topRow < H / 3;

        int maxRowPixels = 0;
        int bottomRow = 0;
        for (int r = H * 2 / 3; r < H; r++) {
            int cnt = Core.countNonZero(warpedBig.row(r));
            if (cnt > maxRowPixels) {
                maxRowPixels = cnt;
                bottomRow = r;
            }
        }
        boolean baseOk = bottomRow > H * 2 / 3 && maxRowPixels > W / 4;

        double total = Core.countNonZero(warpedBig);
        if (total == 0) return Double.NEGATIVE_INFINITY;

        double bonus = (apexOk ? 1000.0 : 0.0) + (baseOk ? 1000.0 : 0.0);
        return bonus + total;
    }

    private Mat cropToBboxAndResize(Mat big) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(big.clone(), contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return resizeToOutput(big);
        }

        Rect bbox = contours.stream()
                .map(Imgproc::boundingRect)
                .reduce((a, b) -> {
                    int x  = Math.min(a.x, b.x);
                    int y  = Math.min(a.y, b.y);
                    int x2 = Math.max(a.x + a.width,  b.x + b.width);
                    int y2 = Math.max(a.y + a.height, b.y + b.height);
                    return new Rect(x, y, x2 - x, y2 - y);
                })
                .orElse(new Rect(0, 0, big.cols(), big.rows()));

        int pad = 4;
        int x = Math.max(0, bbox.x - pad);
        int y = Math.max(0, bbox.y - pad);
        int w = Math.min(big.cols() - x, bbox.width  + 2 * pad);
        int h = Math.min(big.rows() - y, bbox.height + 2 * pad);

        Mat cropped = new Mat(big, new Rect(x, y, w, h));
        Mat result = new Mat();
        Imgproc.resize(cropped, result, new Size(OUTPUT_SIZE, OUTPUT_SIZE), 0, 0, Imgproc.INTER_AREA);
        Imgproc.threshold(result, result, 64, 255, Imgproc.THRESH_BINARY);
        return centerOnMass(result);
    }

    /**
     * Сдвигает бинарное изображение так, чтобы центр масс контура оказался
     * в геометрическом центре холста (cols/2, rows/2).
     *
     * <p>Использует нулевой (m00) и первые (m10, m01) моменты для вычисления
     * centroid, затем применяет целочисленный сдвиг через warpAffine.
     * Инвариантен к форме — одинаково корректен для треугольников, прямоугольников
     * и кругов. При пустом изображении возвращает оригинал без изменений.
     *
     * @param binary бинарное изображение CV_8UC1 (0/255)
     * @return центрированное изображение того же размера
     */
    private Mat centerOnMass(Mat binary) {
        org.opencv.core.Moments m = Imgproc.moments(binary, true);
        if (m.get_m00() < 1.0) return binary;

        double cx = m.get_m10() / m.get_m00();
        double cy = m.get_m01() / m.get_m00();

        double tx = binary.cols() / 2.0 - cx;
        double ty = binary.rows() / 2.0 - cy;

        Mat T = Mat.eye(2, 3, CvType.CV_64F);
        T.put(0, 2, tx);
        T.put(1, 2, ty);

        Mat centered = new Mat();
        Imgproc.warpAffine(binary, centered, T,
                binary.size(), Imgproc.INTER_NEAREST,
                Core.BORDER_CONSTANT, new Scalar(0));
        Imgproc.threshold(centered, centered, 64, 255, Imgproc.THRESH_BINARY);
        return centered;
    }

    /**
     * Аппроксимирует контур треугольником через approxPolyDP.
     *
     * <p><b>Новая логика (concavity-aware):</b><br>
     * Перед запуском approxPolyDP вычисляется {@link ConcavityScore}.
     * Если вогнутость превышает пороги ({@code MAX_DEPTH_NORM_THRESHOLD} /
     * {@code SUM_DEPTH_NORM_THRESHOLD}), контур считается слишком ломаным:
     * approxPolyDP выполняется для дебага, но возвращается {@code null},
     * что переключает {@code alignToCanonical} на minEnclosingTriangle.
     * При низкой вогнутости работает прежняя логика.
     *
     * @param contour   контур фигуры
     * @param debugName имя папки для дебаг-сохранения
     * @return массив из 3 точек или {@code null} для fallback
     */
    private Point[] approxTriangle(MatOfPoint contour, String debugName) {
        // --- Оценка вогнутости ---
        ConcavityScore cs = computeConcavityScore(contour);
        boolean tooConcave = cs.isTooHighFor(MAX_DEPTH_NORM_THRESHOLD, SUM_DEPTH_NORM_THRESHOLD);

        if (tooConcave) {
            System.out.printf(
                "Контур слишком вогнутый (maxDepthNorm=%.4f, sumDepthNorm=%.4f, defects=%d) → fallback%n",
                cs.maxDepthNorm(), cs.sumDepthNorm(), cs.defectCount()
            );
        }

        // --- Строим hull для approxPolyDP (как раньше) ---
        MatOfInt hullIdx = new MatOfInt();
        Imgproc.convexHull(contour, hullIdx);
        int[] idx = hullIdx.toArray();
        Point[] allPts = contour.toArray();
        Point[] hullPts = new Point[idx.length];
        for (int i = 0; i < idx.length; i++) {
            hullPts[i] = allPts[idx[i]];
        }
        MatOfPoint2f contour2f = new MatOfPoint2f(hullPts);
        double perimeter = Imgproc.arcLength(contour2f, true);

        Point[] approxPts = null;

        outer:
        for (int pct = 10; pct >= 1; pct--) {
            double epsilon = (pct / 100.0) * perimeter;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);
            if (approx.rows() == 3) {
                approxPts = approx.toArray();
                System.out.println("approxPolyDP (hull): 3 точки при epsilon=" + String.format("%.1f", epsilon)
                        + " (" + pct + "% периметра)");
                break outer;
            }
        }

        if (approxPts == null) {
            for (int tenths = 9; tenths >= 5; tenths--) {
                double epsilon = (tenths / 1000.0) * perimeter;
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(contour2f, approx, epsilon, true);
                if (approx.rows() == 3) {
                    approxPts = approx.toArray();
                    System.out.println("approxPolyDP (hull): 3 точки при epsilon=" + String.format("%.1f", epsilon)
                            + " (0." + tenths + "% периметра)");
                    break;
                }
            }
        }

        if (approxPts == null) {
            System.out.println("approxPolyDP (hull): не удалось получить 3 точки (периметр=" + String.format("%.1f", perimeter) + ")");
            return null;
        }

        // Всегда сохраняем approx для дебага
        saveApproxPreview(contour, approxPts, debugName, "05b_approx_triangle.png");

        // Если вогнутость высокая — approx идёт только в дебаг, возвращаем null → fallback
        if (tooConcave) {
            System.out.println("approxPolyDP сохранён для дебага, но для warp используется fallback (высокая вогнутость)");
            return null;
        }

        return approxPts;
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
        return name.replaceAll("[^a-zA-Z0-9._\\\\-]", "_");
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
