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
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor {

    public static final int OUTPUT_SIZE = 64;
    private static final int BBOX_PADDING = 24;
    private static final int NORMALIZED_SHAPE_SIZE = 52; // фигура занимает ~52px из 64
    private static final int CANVAS_SIZE = OUTPUT_SIZE;  // итоговый холст 64×64

    private static final boolean DEBUG_SAVE = true;
    private static final String DEBUG_DIR = "debug-preprocess";

    public static class PreprocessResult {
        private final BufferedImage image;
        private final double[][] matrix;

        public PreprocessResult(BufferedImage image, double[][] matrix) {
            this.image = image;
            this.matrix = matrix;
        }

        public BufferedImage getImage() { return image; }
        public double[][] getMatrix()   { return matrix; }
    }

    public PreprocessResult preprocess(File imageFile) throws Exception {
        Mat source = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (source.empty()) {
            throw new IllegalArgumentException(
                "Не удалось загрузить изображение: " + imageFile.getAbsolutePath());
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

        Mat aligned = alignTriangle(cropped, debugName);
        saveDebugMat(debugName, "05_aligned.png", aligned);

        BufferedImage image = matToBufferedImage(aligned);
        double[][] matrix = imageToMatrix(image);

        return new PreprocessResult(image, matrix);
    }

    // =========================================================================
    // Выравнивание треугольника: поворот основания вниз, вершина вверх
    // =========================================================================

    private Mat alignTriangle(Mat binary, String debugName) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return renderOnCanvas(binary);
        }

        MatOfPoint largest = contours.stream()
            .max(Comparator.comparingDouble(Imgproc::contourArea))
            .orElse(contours.get(0));

        // Получаем 3 вершины
        Point[] vertices = getTriangleVertices(largest, debugName);
        if (vertices == null) {
            System.out.println("Не удалось получить 3 вершины, fallback to renderOnCanvas");
            return renderOnCanvas(binary);
        }

        // Находим основание — самую длинную сторону
        int baseIdx = longestSideIndex(vertices);
        Point baseA = vertices[baseIdx];
        Point baseB = vertices[(baseIdx + 1) % 3];
        Point apex  = vertices[(baseIdx + 2) % 3];

        // Угол наклона основания
        double angle = Math.toDegrees(Math.atan2(baseB.y - baseA.y, baseB.x - baseA.x));

        // Поворачиваем изображение так, чтобы основание было горизонтально
        Point center = new Point(binary.cols() / 2.0, binary.rows() / 2.0);
        Mat rot = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        // Вычисляем новый размер холста после поворота, чтобы ничего не обрезалось
        int diagonal = (int) Math.ceil(
            Math.sqrt(binary.cols() * binary.cols() + binary.rows() * binary.rows()));
        Size rotSize = new Size(diagonal, diagonal);
        Point rotCenter = new Point(diagonal / 2.0, diagonal / 2.0);
        rot = Imgproc.getRotationMatrix2D(
            new Point(binary.cols() / 2.0, binary.rows() / 2.0), angle, 1.0);
        // Поправка смещения для нового центра
        rot.put(0, 2, rot.get(0, 2)[0] + (diagonal - binary.cols()) / 2.0);
        rot.put(1, 2, rot.get(1, 2)[0] + (diagonal - binary.rows()) / 2.0);

        Mat rotated = new Mat();
        Imgproc.warpAffine(binary, rotated, rot, rotSize,
            Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
        Imgproc.threshold(rotated, rotated, 64, 255, Imgproc.THRESH_BINARY);
        saveDebugMat(debugName, "05a_rotated.png", rotated);

        // Повернуть apex тоже, чтобы понять — он сверху или снизу
        double[] apexRot = applyAffine(rot, apex);
        // Центр повёрнутого изображения
        double midY = diagonal / 2.0;
        // Если вершина ниже центра — отразить по Y
        if (apexRot[1] > midY) {
            Mat flipped = new Mat();
            Core.flip(rotated, flipped, 0); // 0 = по горизонтальной оси (flip Y)
            rotated = flipped;
            saveDebugMat(debugName, "05b_flipped.png", rotated);
        }

        return renderOnCanvas(rotated);
    }

    /**
     * Применяет матрицу аффинного преобразования 2×3 к точке.
     */
    private double[] applyAffine(Mat m, Point p) {
        double x = m.get(0, 0)[0] * p.x + m.get(0, 1)[0] * p.y + m.get(0, 2)[0];
        double y = m.get(1, 0)[0] * p.x + m.get(1, 1)[0] * p.y + m.get(1, 2)[0];
        return new double[]{x, y};
    }

    /**
     * Получает 3 вершины треугольника через approxPolyDP, fallback — minEnclosingTriangle.
     */
    private Point[] getTriangleVertices(MatOfPoint contour, String debugName) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);

        // Пробуем approxPolyDP с убывающим epsilon
        for (int pct = 10; pct >= 1; pct--) {
            double epsilon = (pct / 100.0) * perimeter;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);
            if (approx.rows() == 3) {
                Point[] pts = approx.toArray();
                saveApproxPreview(contour, pts, debugName, "05b_approx_triangle.png");
                System.out.println("approxPolyDP: 3 вершины при epsilon=" +
                    String.format("%.1f", epsilon) + " (" + pct + "% периметра)");
                return pts;
            }
        }

        // fallback: minEnclosingTriangle
        System.out.println("approxPolyDP не дал 3 точек → minEnclosingTriangle");
        Mat triMat = new Mat();
        Imgproc.minEnclosingTriangle(new MatOfPoint2f(contour.toArray()), triMat);
        saveTrianglePreview(contour, triMat, debugName, "05b_min_enclosing_triangle.png");
        return readTrianglePoints(triMat);
    }

    /**
     * Возвращает индекс вершины, начинающей самую длинную сторону.
     * Сторона i соединяет vertices[i] и vertices[(i+1)%3].
     */
    private int longestSideIndex(Point[] v) {
        double best = -1;
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            Point a = v[i], b = v[(i + 1) % 3];
            double dx = b.x - a.x, dy = b.y - a.y;
            double len = dx * dx + dy * dy;
            if (len > best) { best = len; idx = i; }
        }
        return idx;
    }

    /**
     * Вписывает бинарное изображение в центр холста OUTPUT_SIZE×OUTPUT_SIZE
     * с сохранением пропорций.
     */
    private Mat renderOnCanvas(Mat binary) {
        // Crop по bbox контента
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Rect bbox;
        if (contours.isEmpty()) {
            bbox = new Rect(0, 0, binary.cols(), binary.rows());
        } else {
            bbox = contours.stream()
                .map(Imgproc::boundingRect)
                .reduce((a, b) -> {
                    int x  = Math.min(a.x, b.x);
                    int y  = Math.min(a.y, b.y);
                    int x2 = Math.max(a.x + a.width,  b.x + b.width);
                    int y2 = Math.max(a.y + a.height, b.y + b.height);
                    return new Rect(x, y, x2 - x, y2 - y);
                }).orElse(new Rect(0, 0, binary.cols(), binary.rows()));
        }

        int x = Math.max(0, bbox.x);
        int y = Math.max(0, bbox.y);
        int w = Math.min(binary.cols() - x, bbox.width);
        int h = Math.min(binary.rows() - y, bbox.height);
        if (w <= 0 || h <= 0) return resizeToOutput(binary);

        Mat cropped = new Mat(binary, new Rect(x, y, w, h));

        // Масштаб с сохранением пропорций
        double scale = (double) NORMALIZED_SHAPE_SIZE / Math.max(w, h);
        int newW = Math.max(1, (int) Math.round(w * scale));
        int newH = Math.max(1, (int) Math.round(h * scale));

        Mat resized = new Mat();
        Imgproc.resize(cropped, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_AREA);
        Imgproc.threshold(resized, resized, 64, 255, Imgproc.THRESH_BINARY);

        // Центрируем на холсте CANVAS_SIZE×CANVAS_SIZE
        Mat canvas = Mat.zeros(new Size(CANVAS_SIZE, CANVAS_SIZE), CvType.CV_8UC1);
        int offX = (CANVAS_SIZE - newW) / 2;
        int offY = (CANVAS_SIZE - newH) / 2;
        resized.copyTo(canvas.submat(offY, offY + newH, offX, offX + newW));
        return canvas;
    }

    // =========================================================================
    // Базовые методы обработки
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
        Imgproc.threshold(gray, binary, 0, 255,
            Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
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
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) return binary;

        MatOfPoint largest = contours.stream()
            .max(Comparator.comparingDouble(Imgproc::contourArea))
            .orElse(contours.get(0));

        Rect rect = Imgproc.boundingRect(largest);
        int x = Math.max(0, rect.x - BBOX_PADDING);
        int y = Math.max(0, rect.y - BBOX_PADDING);
        int ww = Math.min(binary.cols() - x, rect.width  + 2 * BBOX_PADDING);
        int hh = Math.min(binary.rows() - y, rect.height + 2 * BBOX_PADDING);
        return new Mat(binary, new Rect(x, y, ww, hh)).clone();
    }

    private Mat resizeToOutput(Mat src) {
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(OUTPUT_SIZE, OUTPUT_SIZE),
            0, 0, Imgproc.INTER_NEAREST);
        return resized;
    }

    // =========================================================================
    // Вспомогательные методы для получения вершин
    // =========================================================================

    private Point[] readTrianglePoints(Mat triangle) {
        if (triangle == null || triangle.empty()) return null;
        List<Point> pts = new ArrayList<>();
        for (int r = 0; r < triangle.rows() && pts.size() < 3; r++) {
            for (int c = 0; c < triangle.cols() && pts.size() < 3; c++) {
                double[] v = triangle.get(r, c);
                if (v != null && v.length >= 2) pts.add(new Point(v[0], v[1]));
            }
        }
        return pts.size() == 3 ? pts.toArray(new Point[0]) : null;
    }

    // =========================================================================
    // Debug-вывод
    // =========================================================================

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
            Imgcodecs.imwrite(dir.resolve(fileName).toString(), out);
        } catch (IOException e) {
            System.err.println("saveDebugMat: " + e.getMessage());
        }
    }

    private void saveApproxPreview(MatOfPoint contour, Point[] approx,
                                    String debugName, String filename) {
        Rect bound = Imgproc.boundingRect(contour);
        int margin = 20;
        int W = bound.x + bound.width  + margin;
        int H = bound.y + bound.height + margin;
        Mat canvas = Mat.zeros(H, W, CvType.CV_8UC3);

        List<MatOfPoint> list = new ArrayList<>();
        list.add(contour);
        Imgproc.drawContours(canvas, list, 0, new Scalar(255, 255, 255), 1);

        MatOfPoint poly = new MatOfPoint(approx);
        List<MatOfPoint> polyList = new ArrayList<>();
        polyList.add(poly);
        Imgproc.drawContours(canvas, polyList, 0, new Scalar(0, 255, 255), 1);

        String[] labels = {"P1", "P2", "P3"};
        Scalar[] colors = {new Scalar(0,255,0), new Scalar(0,0,255), new Scalar(255,0,0)};
        for (int i = 0; i < approx.length; i++) {
            Imgproc.circle(canvas, approx[i], 4, colors[i], -1);
            Imgproc.putText(canvas, labels[i],
                new Point(approx[i].x + 5, approx[i].y - 5),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1);
        }
        saveDebugMat(debugName, filename, canvas);
    }

    private void saveTrianglePreview(MatOfPoint contour, Mat triangleMat,
                                      String debugName, String filename) {
        Rect bound = Imgproc.boundingRect(contour);
        int margin = 20;
        int W = bound.x + bound.width  + margin;
        int H = bound.y + bound.height + margin;
        Mat canvas = Mat.zeros(H, W, CvType.CV_8UC3);

        List<MatOfPoint> list = new ArrayList<>();
        list.add(contour);
        Imgproc.drawContours(canvas, list, 0, new Scalar(255, 255, 255), 1);

        Point[] pts = readTrianglePoints(triangleMat);
        if (pts != null && pts.length == 3) {
            Scalar yellow = new Scalar(0, 255, 255);
            Imgproc.line(canvas, pts[0], pts[1], yellow, 1);
            Imgproc.line(canvas, pts[1], pts[2], yellow, 1);
            Imgproc.line(canvas, pts[2], pts[0], yellow, 1);
            String[] labels = {"P1", "P2", "P3"};
            Scalar[] colors = {new Scalar(0,255,0), new Scalar(0,0,255), new Scalar(255,0,0)};
            for (int i = 0; i < pts.length; i++) {
                Imgproc.circle(canvas, pts[i], 5, colors[i], -1);
                Imgproc.putText(canvas, labels[i],
                    new Point(pts[i].x + 6, pts[i].y - 6),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
            }
        }
        saveDebugMat(debugName, filename, canvas);
    }

    // =========================================================================
    // Конвертация
    // =========================================================================

    private BufferedImage matToBufferedImage(Mat mat) {
        Mat display = mat;
        if (mat.channels() == 1) {
            display = new Mat();
            Imgproc.cvtColor(mat, display, Imgproc.COLOR_GRAY2BGR);
        }
        int width  = display.cols();
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
                int g = (rgb >> 8)  & 0xFF;
                int b = rgb & 0xFF;
                matrix[y][x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }
        return matrix;
    }
}
