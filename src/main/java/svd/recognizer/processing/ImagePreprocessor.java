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

    /**
     * Включает/выключает сохранение промежуточных изображений. Когда закончишь
     * отладку, можно поставить false.
     */
    private static final boolean DEBUG_SAVE = true;

    /**
     * Корневая папка для отладочных изображений. Внутри неё для каждого
     * входного файла создаётся отдельная подпапка.
     */
    private static final String DEBUG_DIR = "debug-preprocess";

    /**
     * Канонический треугольник, к которому приводим все входные треугольники.
     * Формат 64x64: - верхняя вершина - нижняя левая - нижняя правая
     */
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
        saveDebugMat(debugName, "02_binary_otsu.png", binary);

        Mat cleaned = morphClean(binary);
        saveDebugMat(debugName, "03_cleaned.png", cleaned);

        Mat cropped = extractROI(cleaned);
        saveDebugMat(debugName, "04_cropped.png", cropped);

        Mat canonicalMask = buildCanonicalMask();
        saveDebugMat(debugName, "05_canonical_mask.png", canonicalMask);

        Mat aligned = alignToCanonical(cropped, debugName);
        saveDebugMat(debugName, "07_best_aligned.png", aligned);

        BufferedImage image = matToBufferedImage(aligned);
        double[][] matrix = imageToMatrix(image);

        saveDebugMat(debugName, "08_final.png", aligned);

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

    /**
     * Бинаризация для фото на телефон при плохом освещении.
     *
     * Шаги: 1. CLAHE — выравнивает локальный контраст, вытягивает слабые линии.
     * 2. GaussianBlur 9x9 — убирает шум, неизбежный при плохом освещении. 3.
     * adaptiveThreshold — бинаризует локально, не зависит от общей яркости.
     *
     * Почему не Отсу: Отсу ищет один глобальный порог. При тени в углу или
     * неравномерном освещении этот порог "улетает" в пользу одной области, и в
     * другой области линия либо теряется, либо фон становится белым.
     */
    private Mat binarize(Mat gray) {
        // Шаг 1: CLAHE — адаптивное выравнивание гистограммы
        // clipLimit = 2.0 — умеренно, чтобы не усиливать шум
        // tileGridSize 8x8 — достаточно мелко для неравномерного освещения
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        Mat equalized = new Mat();
        clahe.apply(gray, equalized);

        // Шаг 2: Размытие для подавления шума
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(equalized, blurred, new Size(9, 9), 0.0);

        // Шаг 3: Адаптивная бинаризация
        // ADAPTIVE_THRESH_GAUSSIAN_C — взвешенное среднее по соседям (лучше для рукописи)
        // THRESH_BINARY_INV — линия белая, фон чёрный
        // blockSize 31 — размер блока, подбирается под масштаб линии на фото
        // C = 10 — вычитаемая константа, убирает мелкий шум в фоне
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
        // Ядро 5x5 для CLOSE — рваные линии от плохого освещения требуют
        // более агрессивного закрытия разрывов.
        Mat kernelClose = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5));

        // Ядро 3x3 для OPEN — не переусердствуем с удалением шума,
        // чтобы не "съесть" тонкие части линии.
        Mat kernelOpen = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(3, 3));

        Mat closed = new Mat();
        Mat opened = new Mat();

        Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, kernelClose);
        Imgproc.morphologyEx(closed, opened, Imgproc.MORPH_OPEN, kernelOpen);

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
        int w = Math.min(binary.cols() - x, rect.width + 2 * BBOX_PADDING);
        int h = Math.min(binary.rows() - y, rect.height + 2 * BBOX_PADDING);

        return new Mat(binary, new Rect(x, y, w, h)).clone();
    }

    private Mat resizeToOutput(Mat source) {
        Mat resized = new Mat();
        Imgproc.resize(
                source,
                resized,
                new Size(OUTPUT_SIZE, OUTPUT_SIZE),
                0,
                0,
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
            saveDebugMat(debugName, "06_fallback_no_contours.png", fallback);
            return fallback;
        }

        MatOfPoint largest = contours.stream()
                .max(Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(contours.get(0));

        saveContourPreview(binary, largest, debugName, "05a_largest_contour.png");

        Mat triangle = new Mat();
        Imgproc.minEnclosingTriangle(new MatOfPoint2f(largest.toArray()), triangle);

        System.out.println("triangle rows=" + triangle.rows()
                + ", cols=" + triangle.cols()
                + ", channels=" + triangle.channels()
                + ", type=" + triangle.type());

        saveTrianglePreview(binary, triangle, debugName, "05b_min_enclosing_triangle.png");

        Point[] srcPts = readTrianglePoints(triangle);
        if (srcPts == null) {
            Mat fallback = resizeToOutput(binary);
            saveDebugMat(debugName, "06_fallback_bad_triangle.png", fallback);
            return fallback;
        }

        Point[][] perms = permutations(srcPts);

        Mat best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;

        for (int i = 0; i < perms.length; i++) {
            Point[] perm = perms[i];

            MatOfPoint2f src = new MatOfPoint2f(perm);
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

            saveDebugMat(debugName, String.format("06_perm_%d.png", i), warped);

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
            saveDebugMat(debugName, "06_fallback_no_best.png", fallback);
            return fallback;
        }

        return best;
    }

    /**
     * Пытается прочитать 3 вершины треугольника из Mat, который вернул
     * minEnclosingTriangle.
     *
     * В зависимости от сборки OpenCV Java формат может отличаться. Здесь
     * покрываем самые типичные варианты: - rows=3, cols=1, channels=2 - rows=3,
     * cols=2, channels=1 - rows=1, cols=3, channels=2
     */
    private Point[] readTrianglePoints(Mat triangle) {
        if (triangle == null || triangle.empty()) {
            return null;
        }

        // Случай 1: 3x1, 2-channel
        if (triangle.rows() == 3 && triangle.cols() == 1) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] v = triangle.get(i, 0);
                if (v == null || v.length < 2) {
                    return null;
                }
                pts[i] = new Point(v[0], v[1]);
            }
            return pts;
        }

        // Случай 2: 3x2, 1-channel
        if (triangle.rows() == 3 && triangle.cols() == 2 && triangle.channels() == 1) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] x = triangle.get(i, 0);
                double[] y = triangle.get(i, 1);
                if (x == null || y == null || x.length < 1 || y.length < 1) {
                    return null;
                }
                pts[i] = new Point(x[0], y[0]);
            }
            return pts;
        }

        // Случай 3: 1x3, 2-channel
        if (triangle.rows() == 1 && triangle.cols() == 3) {
            Point[] pts = new Point[3];
            for (int i = 0; i < 3; i++) {
                double[] v = triangle.get(0, i);
                if (v == null || v.length < 2) {
                    return null;
                }
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

        double inside = 0.0;
        double outside = 0.0;

        for (int y = 0; y < img.rows(); y++) {
            for (int x = 0; x < img.cols(); x++) {
                double[] pv = img.get(y, x);
                double[] mv = mask.get(y, x);

                double value = (pv == null) ? 0.0 : pv[0];
                double maskValue = (mv == null) ? 0.0 : mv[0];

                if (maskValue > 0) {
                    inside += value;
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

    private void saveDebugMat(String folderName, String fileName, Mat mat) {
        if (!DEBUG_SAVE || mat == null || mat.empty()) {
            return;
        }

        try {
            Path dir = Paths.get(DEBUG_DIR, folderName);
            Files.createDirectories(dir);

            String fullPath = dir.resolve(fileName).toString();

            Mat out = new Mat();
            if (mat.type() == CvType.CV_8UC1 || mat.type() == CvType.CV_8UC3) {
                out = mat;
            } else {
                mat.convertTo(out, CvType.CV_8UC1);
            }

            boolean ok = Imgcodecs.imwrite(fullPath, out);
            if (!ok) {
                System.err.println("Не удалось сохранить debug image: " + fullPath);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void saveContourPreview(Mat binary, MatOfPoint contour, String folderName, String fileName) {
        if (!DEBUG_SAVE || binary == null || binary.empty() || contour == null) {
            return;
        }

        Mat preview = new Mat();
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);

        List<MatOfPoint> one = new ArrayList<>();
        one.add(contour);

        Imgproc.drawContours(preview, one, 0, new Scalar(0, 255, 0), 1);

        Rect rect = Imgproc.boundingRect(contour);
        Imgproc.rectangle(
                preview,
                new Point(rect.x, rect.y),
                new Point(rect.x + rect.width, rect.y + rect.height),
                new Scalar(255, 0, 0),
                1
        );

        saveDebugMat(folderName, fileName, preview);
    }

    private void saveTrianglePreview(Mat binary, Mat triangle, String folderName, String fileName) {
        if (!DEBUG_SAVE || binary == null || binary.empty()) {
            return;
        }

        Point[] pts = readTrianglePoints(triangle);
        if (pts == null) {
            return;
        }

        Mat preview = new Mat();
        Imgproc.cvtColor(binary, preview, Imgproc.COLOR_GRAY2BGR);

        Imgproc.line(preview, pts[0], pts[1], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[1], pts[2], new Scalar(0, 255, 255), 1);
        Imgproc.line(preview, pts[2], pts[0], new Scalar(0, 255, 255), 1);

        Imgproc.circle(preview, pts[0], 3, new Scalar(0, 0, 255), -1);
        Imgproc.circle(preview, pts[1], 3, new Scalar(0, 255, 0), -1);
        Imgproc.circle(preview, pts[2], 3, new Scalar(255, 0, 0), -1);

        Imgproc.putText(preview, "P0", pts[0], Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(0, 0, 255), 1);
        Imgproc.putText(preview, "P1", pts[1], Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(0, 255, 0), 1);
        Imgproc.putText(preview, "P2", pts[2], Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(255, 0, 0), 1);

        saveDebugMat(folderName, fileName, preview);
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private BufferedImage matToBufferedImage(Mat source) {
        Mat normalized = new Mat();

        if (source.type() != CvType.CV_8UC1) {
            source.convertTo(normalized, CvType.CV_8UC1);
        } else {
            normalized = source;
        }

        BufferedImage image = new BufferedImage(
                normalized.cols(),
                normalized.rows(),
                BufferedImage.TYPE_BYTE_GRAY
        );

        byte[] sourceData = new byte[(int) (normalized.total() * normalized.channels())];
        normalized.get(0, 0, sourceData);

        byte[] targetData
                = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        System.arraycopy(sourceData, 0, targetData, 0, sourceData.length);

        return image;
    }

    private double[][] imageToMatrix(BufferedImage image) {
        double[][] matrix = new double[image.getHeight()][image.getWidth()];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int value = image.getRGB(x, y) & 0xFF;
                matrix[y][x] = value / 255.0;
            }
        }

        return matrix;
    }
}
