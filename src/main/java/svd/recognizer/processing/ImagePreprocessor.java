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
import svd.recognizer.model.ShapeClass;

public class ImagePreprocessor {

    public static final int OUTPUT_SIZE = 64;
    private static final int BBOX_PADDING = 24;
    private static final int NORMALIZED_SHAPE_SIZE = 52;
    private static final int CANVAS_SIZE = OUTPUT_SIZE;

    private static final double NOISE_THRESHOLD = 0.10;
    private static final double DEGENERATE_ASPECT_RATIO = 3.0;
    private static final int RECT_CLOSE_KERNEL = 7;
    // Прямоугольник вращаем по длинной грани только если он заметно вытянут.
    // Для почти-квадрата (ниже порога) длинной грани фактически нет — не вращаем.
    private static final double RECT_MIN_ASPECT_FOR_ROTATION = 1.25;

    // Доли диагонали bbox фигуры для эскалации ядра MORPH_CLOSE (смыкание разрывов).
    // Перебираются по возрастанию: берётся первое ядро, собравшее фигуру.
    private static final double[] SHAPE_CLOSE_FRACTIONS = {0.03, 0.05, 0.08, 0.12};
    // Порог покрытия: доля белых пикселей внутри маски главного контура, при
    // которой фигура считается собранной. Ниже — развал на несвязные куски (брак).
    private static final double COVERAGE_OK_PERCENT = 85.0;
    // Удаление мелкого шума: компоненты меньше этой доли площади bbox фигуры
    // считаются шумом и удаляются (но не меньше MIN_SPECKLE_AREA пикселей).
    private static final double SPECKLE_AREA_FRACTION = 0.0008;
    private static final int MIN_SPECKLE_AREA = 20;
    // Множитель ядра max-pooling при уменьшении: тонкая линия должна пережить
    // даунскейл и не порвать фигуру. Подобрано по реальным данным (1.0 рвало).
    private static final double MAXPOOL_DILATE_MULT = 1.5;

    private static final boolean DEBUG_SAVE = true;
    private static final String DEBUG_DIR = "debug-preprocess";

    public static class PreprocessResult {
        private final BufferedImage image;
        private final double[][] matrix;
        private final boolean lowQuality;
        private final String qualityReason;

        public PreprocessResult(BufferedImage image, double[][] matrix,
                                boolean lowQuality, String qualityReason) {
            this.image = image;
            this.matrix = matrix;
            this.lowQuality = lowQuality;
            this.qualityReason = qualityReason;
        }

        public BufferedImage getImage()      { return image; }
        public double[][] getMatrix()        { return matrix; }
        public boolean isLowQuality()        { return lowQuality; }
        public String getQualityReason()     { return qualityReason; }
    }

    public PreprocessResult preprocess(File imageFile) throws Exception {
        return preprocess(imageFile, null);
    }

    public PreprocessResult preprocess(File imageFile, ShapeClass shapeClass) throws Exception {
        Mat source = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (source.empty()) {
            throw new IllegalArgumentException(
                "Не удалось загрузить изображение: " + imageFile.getAbsolutePath());
        }

        String debugName = sanitizeFileName(imageFile.getName());

        Mat gray = toGrayscale(source);
        saveDebugMat(debugName, "01_source_gray.png", gray);

        boolean[] lowQualityFlag = {false};
        String[]  qualityReason  = {""};

        Mat binary = binarize(gray, lowQualityFlag, qualityReason);
        saveDebugMat(debugName, "02_binary.png", binary);

        Mat cleaned = isolateMainShape(binary, debugName, lowQualityFlag, qualityReason);
        saveDebugMat(debugName, "03_cleaned.png", cleaned);

        Mat cropped = extractROI(cleaned);
        saveDebugMat(debugName, "04_cropped.png", cropped);

        Mat aligned;
        if (shapeClass == ShapeClass.RECTANGLE) {
            aligned = alignRectangle(cropped, gray, debugName, lowQualityFlag, qualityReason);
        } else {
            aligned = alignTriangle(cropped, debugName, lowQualityFlag, qualityReason,
                    shapeClass == ShapeClass.TRIANGLE);
        }
        saveDebugMat(debugName, "05_aligned.png", aligned);

        BufferedImage image = matToBufferedImage(aligned);
        double[][] matrix = imageToMatrix(image);

        return new PreprocessResult(image, matrix, lowQualityFlag[0], qualityReason[0]);
    }

    private Mat binarize(Mat gray, boolean[] lowQualityFlag, String[] qualityReason) {
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 0, 255,
            Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        double whitePct = (double) Core.countNonZero(binary) / (binary.rows() * binary.cols());
        if (whitePct > NOISE_THRESHOLD) {
            System.out.println("binarize: Otsu whitePct=" +
                String.format("%.1f%%", whitePct * 100) + " > 10% \u2192 fallback adaptive gaussian");
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);
            Imgproc.adaptiveThreshold(blurred, binary, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 51, 10);

            double whitePctAfter = (double) Core.countNonZero(binary) / (binary.rows() * binary.cols());
            if (whitePctAfter > NOISE_THRESHOLD) {
                lowQualityFlag[0] = true;
                qualityReason[0]  = String.format(
                    "высокий уровень шума после бинаризации (%.0f%% белых пикселей)",
                    whitePctAfter * 100);
                System.out.println("binarize: adaptive gaussian whitePct=" +
                    String.format("%.1f%%", whitePctAfter * 100) + " — low quality flag set");
            }
        }

        return binary;
    }

    private Mat rebinarizeAdaptive(Mat gray, String debugName) {
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(blurred, binary, 255,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV, 51, 10);
        System.out.println("rebinarizeAdaptive: применён адаптивный гауссиан 51/10");
        saveDebugMat(debugName, "04b_rebinarized.png", binary);
        return binary;
    }

    private Mat alignTriangle(Mat binary, String debugName,
                               boolean[] lowQualityFlag, String[] qualityReason,
                               boolean validateVertices) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return renderOnCanvas(binary);
        }

        MatOfPoint largest = contours.stream()
            .max(Comparator.comparingDouble(Imgproc::contourArea))
            .orElse(contours.get(0));

        Point[] vertices = getTriangleVertices(largest, debugName);
        if (vertices == null) {
            if (validateVertices && !lowQualityFlag[0]) {
                lowQualityFlag[0] = true;
                qualityReason[0]  = "не удалось распознать треугольник: не найдены 3 вершины";
            }
            System.out.println("alignTriangle: не удалось получить 3 вершины"
                + (validateVertices ? " \u2192 low quality" : ""));
            return renderOnCanvas(binary);
        }

        int baseIdx = longestSideIndex(vertices);
        Point baseA = vertices[baseIdx];
        Point baseB = vertices[(baseIdx + 1) % 3];
        Point apex  = vertices[(baseIdx + 2) % 3];

        double angle = Math.toDegrees(Math.atan2(baseB.y - baseA.y, baseB.x - baseA.x));

        int diagonal = (int) Math.ceil(
            Math.sqrt(binary.cols() * binary.cols() + binary.rows() * binary.rows()));
        Size rotSize = new Size(diagonal, diagonal);
        Mat rot = Imgproc.getRotationMatrix2D(
            new Point(binary.cols() / 2.0, binary.rows() / 2.0), angle, 1.0);
        rot.put(0, 2, rot.get(0, 2)[0] + (diagonal - binary.cols()) / 2.0);
        rot.put(1, 2, rot.get(1, 2)[0] + (diagonal - binary.rows()) / 2.0);

        Mat rotated = new Mat();
        Imgproc.warpAffine(binary, rotated, rot, rotSize,
            Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
        Imgproc.threshold(rotated, rotated, 40, 255, Imgproc.THRESH_BINARY);
        saveDebugMat(debugName, "05a_rotated.png", rotated);

        double[] apexRot = applyAffine(rot, apex);
        double midY = diagonal / 2.0;
        if (apexRot[1] > midY) {
            Mat flipped = new Mat();
            Core.flip(rotated, flipped, 0);
            rotated = flipped;
            saveDebugMat(debugName, "05b_flipped.png", rotated);
        }

        return renderOnCanvas(rotated);
    }

    private Mat alignRectangle(Mat binary, Mat originalGray, String debugName,
                                boolean[] lowQualityFlag, String[] qualityReason) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return renderOnCanvas(binary);
        }

        MatOfPoint largest = contours.stream()
            .max(Comparator.comparingDouble(Imgproc::contourArea))
            .orElse(contours.get(0));

        if (isDegenerate(largest)) {
            System.out.println("alignRectangle: контур вырожден (aspect ratio > "
                + DEGENERATE_ASPECT_RATIO + ") \u2192 fallback adaptive gaussian + morphClose");

            Mat rebinarized = rebinarizeAdaptive(originalGray, debugName);
            Mat rebinarizedCleaned = morphClean(rebinarized);
            Mat rebinarizedClosed = morphClose(rebinarizedCleaned, RECT_CLOSE_KERNEL);
            saveDebugMat(debugName, "04c_rebinarized_closed.png", rebinarizedClosed);

            Mat rebinarizedCropped = extractROI(rebinarizedClosed);
            saveDebugMat(debugName, "04d_rebinarized_cropped.png", rebinarizedCropped);

            List<MatOfPoint> contours2 = new ArrayList<>();
            Imgproc.findContours(rebinarizedCropped.clone(), contours2, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            if (!contours2.isEmpty()) {
                MatOfPoint largest2 = contours2.stream()
                    .max(Comparator.comparingDouble(Imgproc::contourArea))
                    .orElse(contours2.get(0));
                if (!isDegenerate(largest2)) {
                    largest = largest2;
                    binary  = rebinarizedCropped;
                    System.out.println("alignRectangle: после adaptive + morphClose контур восстановлен");
                } else {
                    if (!lowQualityFlag[0]) {
                        lowQualityFlag[0] = true;
                        qualityReason[0]  = "контур прямоугольника вырожден даже после adaptive + morphClose";
                    }
                    System.out.println("alignRectangle: contour still degenerate after adaptive + morphClose \u2192 low quality");
                    return renderOnCanvas(rebinarizedCropped);
                }
            }
        }

        // ---- Выравнивание прямоугольника по самой длинной грани ----
        // minAreaRect выбирает ось нестабильно для почти-квадратов, поэтому угол
        // поворота берём от самой длинной прямой грани (HoughLinesP) — устойчивый
        // ориентир для вытянутого прямоугольника. Для почти-квадрата (aspect < 1.25)
        // и при отсутствии надёжных линий поворот не выполняется (фигура симметрична,
        // наклон мал и некритичен), чтобы не вносить случайных ориентаций в SVD-базис.
        Rect bbox = Imgproc.boundingRect(largest);
        double maxSide = Math.max(bbox.width, bbox.height);
        double minSide = Math.max(1, Math.min(bbox.width, bbox.height));
        double aspect = maxSide / minSide;

        if (aspect < RECT_MIN_ASPECT_FOR_ROTATION) {
            System.out.println("alignRectangle: почти квадрат (aspect="
                + String.format("%.2f", aspect) + ") \u2192 поворот не выполняется");
            return renderOnCanvas(binary);
        }

        Double edgeAngle = longestEdgeAngle(binary, minSide);
        if (edgeAngle == null) {
            System.out.println("alignRectangle: длинная грань не найдена \u2192 поворот не выполняется");
            return renderOnCanvas(binary);
        }

        double angle = normalizeEdgeAngle(edgeAngle);
        System.out.println("alignRectangle: угол длинной грани="
            + String.format("%.1f", edgeAngle) + " \u2192 нормализованный="
            + String.format("%.1f", angle));

        int diagonal = (int) Math.ceil(
            Math.sqrt(binary.cols() * binary.cols() + binary.rows() * binary.rows()));
        Size rotSize = new Size(diagonal, diagonal);
        Mat rot = Imgproc.getRotationMatrix2D(
            new Point(binary.cols() / 2.0, binary.rows() / 2.0), angle, 1.0);
        rot.put(0, 2, rot.get(0, 2)[0] + (diagonal - binary.cols()) / 2.0);
        rot.put(1, 2, rot.get(1, 2)[0] + (diagonal - binary.rows()) / 2.0);

        Mat rotated = new Mat();
        Imgproc.warpAffine(binary, rotated, rot, rotSize,
            Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
        Imgproc.threshold(rotated, rotated, 40, 255, Imgproc.THRESH_BINARY);
        saveDebugMat(debugName, "05a_rotated_rect.png", rotated);

        // Привести к единой (ландшафтной) ориентации: если после выравнивания
        // фигура осталась портретной (высота > ширины), довернуть на 90°.
        // Иначе вытянутые прямоугольники одного класса разъезжаются на две
        // ориентации (горизонтальную и вертикальную) и SVD-подпространство
        // получается грязным. Длинная сторона всегда кладётся горизонтально.
        rotated = forceLandscape(rotated);
        saveDebugMat(debugName, "05c_landscape.png", rotated);

        return renderOnCanvas(rotated);
    }

    /**
     * Если фигура портретная (высота её bbox больше ширины), доворачивает её
     * на 90°, чтобы длинная сторона легла горизонтально. Для уже ландшафтной
     * фигуры возвращает без изменений.
     */
    private Mat forceLandscape(Mat bin) {
        Mat nz = new Mat();
        Core.findNonZero(bin, nz);
        if (nz.empty()) return bin;
        Rect bb = Imgproc.boundingRect(nz);
        if (bb.height <= bb.width) return bin;

        int diagonal = (int) Math.ceil(
            Math.sqrt(bin.cols() * (double) bin.cols() + bin.rows() * (double) bin.rows()));
        Mat rot = Imgproc.getRotationMatrix2D(
            new Point(bin.cols() / 2.0, bin.rows() / 2.0), 90, 1.0);
        rot.put(0, 2, rot.get(0, 2)[0] + (diagonal - bin.cols()) / 2.0);
        rot.put(1, 2, rot.get(1, 2)[0] + (diagonal - bin.rows()) / 2.0);
        Mat out = new Mat();
        Imgproc.warpAffine(bin, out, rot, new Size(diagonal, diagonal),
            Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
        Imgproc.threshold(out, out, 40, 255, Imgproc.THRESH_BINARY);
        return out;
    }

    /**
     * Угол (в градусах) самой длинной прямой грани фигуры, найденной HoughLinesP.
     * Возвращает null, если надёжных линий не найдено.
     *
     * minLineLength привязан к меньшей стороне bbox, чтобы ловить настоящие грани,
     * а не короткие шумовые отрезки.
     */
    private Double longestEdgeAngle(Mat binary, double minSide) {
        Mat lines = new Mat();
        int minLineLength = (int) Math.max(10, minSide * 0.5);
        int maxLineGap    = (int) Math.max(5, minSide * 0.25);
        Imgproc.HoughLinesP(binary, lines, 1, Math.PI / 180.0,
            50, minLineLength, maxLineGap);

        if (lines.empty()) {
            return null;
        }

        double bestLen = -1;
        double bestAngle = 0;
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            if (l == null || l.length < 4) continue;
            double dx = l[2] - l[0];
            double dy = l[3] - l[1];
            double len = Math.sqrt(dx * dx + dy * dy);
            if (len > bestLen) {
                bestLen = len;
                bestAngle = Math.toDegrees(Math.atan2(dy, dx));
            }
        }
        return bestLen > 0 ? bestAngle : null;
    }

    /**
     * Приводит угол грани к минимальному повороту в диапазоне [-45, 45].
     * Грань горизонтальна с точностью до 90°, поэтому достаточно привести
     * угол по модулю 90 к ближайшему к нулю значению.
     */
    private double normalizeEdgeAngle(double angle) {
        while (angle >  45.0) angle -= 90.0;
        while (angle < -45.0) angle += 90.0;
        return angle;
    }

    private Point[] getTriangleVertices(MatOfPoint contour, String debugName) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);

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

        System.out.println("approxPolyDP не дал 3 точек \u2192 minEnclosingTriangle");
        Mat triMat = new Mat();
        Imgproc.minEnclosingTriangle(new MatOfPoint2f(contour.toArray()), triMat);
        saveTrianglePreview(contour, triMat, debugName, "05b_min_enclosing_triangle.png");
        return readTrianglePoints(triMat);
    }

    private boolean isDegenerate(MatOfPoint contour) {
        Rect bbox = Imgproc.boundingRect(contour);
        double maxSide = Math.max(bbox.width, bbox.height);
        double minSide = Math.min(bbox.width, bbox.height);
        if (minSide == 0) return true;
        double ratio = maxSide / minSide;
        System.out.println("isDegenerate: bbox=" + bbox.width + "x" + bbox.height
            + " ratio=" + String.format("%.2f", ratio));
        return ratio > DEGENERATE_ASPECT_RATIO;
    }

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

    private double[] applyAffine(Mat m, Point p) {
        double x = m.get(0, 0)[0] * p.x + m.get(0, 1)[0] * p.y + m.get(0, 2)[0];
        double y = m.get(1, 0)[0] * p.x + m.get(1, 1)[0] * p.y + m.get(1, 2)[0];
        return new double[]{x, y};
    }

    private Mat renderOnCanvas(Mat binary) {
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

        double scale = (double) NORMALIZED_SHAPE_SIZE / Math.max(w, h);
        int newW = Math.max(1, (int) Math.round(w * scale));
        int newH = Math.max(1, (int) Math.round(h * scale));

        int reduction = (int) Math.ceil(1.0 / scale);
        if (reduction < 1) reduction = 1;

        Mat thick = cropped;
        if (reduction > 1) {
            // Ядро max-pooling = 1.5 * коэффициент сжатия. Ровно reduction
            // оказалось мало: самые тонкие участки рукописной линии всё же
            // выпадали при уменьшении и фигура рвалась на 64x64 (triangle2,
            // square5, circle2/4). 1.5x гарантирует выживание тонкой линии,
            // утолщая её всего на ~1px на финальном холсте.
            int k = (int) Math.round(reduction * MAXPOOL_DILATE_MULT);
            if (k < 3) k = 3;
            if (k % 2 == 0) k++;
            Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, new Size(k, k));
            thick = new Mat();
            Imgproc.dilate(cropped, thick, kernel);
        }

        Mat resized = new Mat();
        Imgproc.resize(thick, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_NEAREST);
        Imgproc.threshold(resized, resized, 64, 255, Imgproc.THRESH_BINARY);

        Mat canvas = Mat.zeros(new Size(CANVAS_SIZE, CANVAS_SIZE), CvType.CV_8UC1);
        int offX = (CANVAS_SIZE - newW) / 2;
        int offY = (CANVAS_SIZE - newH) / 2;
        resized.copyTo(canvas.submat(offY, offY + newH, offX, offX + newW));
        return canvas;
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
     * Удаляет мелкие изолированные связные компоненты (внутренний и точечный
     * шум), сохраняя крупные структуры. Порог площади привязан к размеру bbox
     * фигуры, поэтому масштабируется под любой размер изображения.
     *
     * Подобрано по реальным данным: убирает крап внутри рукописных фигур
     * (тени, грязь), не трогая контур. Переносится на лица — удалит шум сенсора,
     * сохранив черты лица как крупные компоненты.
     */
    private Mat despeckle(Mat binary, Rect whiteBox) {
        double bboxArea = (double) whiteBox.width * whiteBox.height;
        int minArea = (int) Math.max(MIN_SPECKLE_AREA, bboxArea * SPECKLE_AREA_FRACTION);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, 8, CvType.CV_32S);

        Mat keepMask = Mat.zeros(binary.size(), CvType.CV_8UC1);
        int removed = 0;
        for (int i = 1; i < n; i++) {
            int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
            if (area >= minArea) {
                // оставить компоненту: добавить её пиксели в маску
                Mat comp = new Mat();
                Core.compare(labels, new Scalar(i), comp, Core.CMP_EQ);
                Core.bitwise_or(keepMask, comp, keepMask);
            } else {
                removed++;
            }
        }

        Mat result = new Mat();
        Core.bitwise_and(binary, keepMask, result);
        System.out.println("despeckle: minArea=" + minArea
            + " removed " + removed + " small components, white "
            + Core.countNonZero(binary) + " \u2192 " + Core.countNonZero(result));
        return result;
    }

    /**
     * Изолирует основную фигуру и убирает посторонний шум, СОХРАНЯЯ контур
     * (без заливки в результат).
     *
     * Заменяет прежний morphClean(MORPH_OPEN): open делал эрозию, рвавшую тонкие
     * рукописные линии. Здесь наоборот — смыкаем разрывы и отсекаем внешний шум.
     *
     * Алгоритм с эскалацией ядра смыкания:
     *   Для долей диагонали f="3,5,8,12%":
     *     1. MORPH_CLOSE ядром k=f*диагональ — смыкает разрывы (углы, грани).
     *     2. Берём самый большой контур, строим его ЗАЛИТУЮ маску (в памяти).
     *     3. coverage = доля исходных белых пикселей, попавших в маску.
     *        Если coverage >= COVERAGE_OK_PERCENT — фигура собрана: возвращаем
     *        binary AND mask (контур сохранён, внешний шум убран).
     *   Если ни одно ядро не собрало фигуру (coverage остаётся низким — значит
     *   фигура развалилась на несвязные куски), помечаем lowQuality и возвращаем
     *   binary как есть, без отсечения.
     *
     * coverage честно различает смыкаемую фигуру (~100%) и реальный развал на
     * куски (главный контур охватывает лишь часть → coverage заметно < 100%).
     * Дырку в грани coverage не штрафует — её и не нужно: MORPH_CLOSE её закроет.
     */
    private Mat isolateMainShape(Mat binary, String debugName,
                                  boolean[] lowQualityFlag, String[] qualityReason) {
        Mat nz = new Mat();
        Core.findNonZero(binary, nz);
        if (nz.empty()) {
            return binary.clone();
        }
        Rect whiteBox = Imgproc.boundingRect(nz);

        // (despeckle перенесён ПОСЛЕ смыкания: иначе он выкидывает оторванные
        //  куски фигуры — например часть основания square5 — как "шум" до того,
        //  как MORPH_CLOSE успеет присоединить их к контуру.)
        double diag = Math.sqrt(
            whiteBox.width * (double) whiteBox.width
            + whiteBox.height * (double) whiteBox.height);

        double total = Core.countNonZero(binary);
        double bestCoverage = 0;
        Mat bestResult = null;

        for (double frac : SHAPE_CLOSE_FRACTIONS) {
            int k = (int) Math.max(3, Math.round(diag * frac));
            if (k % 2 == 0) k++;

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(k, k));
            Mat closed = new Mat();
            Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, kernel);

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(closed.clone(), contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            if (contours.isEmpty()) continue;

            MatOfPoint main = contours.stream()
                .max(Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(contours.get(0));

            Mat mask = Mat.zeros(binary.size(), CvType.CV_8UC1);
            List<MatOfPoint> one = new ArrayList<>();
            one.add(main);
            Imgproc.drawContours(mask, one, 0, new Scalar(255), -1);

            Mat result = new Mat();
            Core.bitwise_and(binary, mask, result);

            // despeckle ПОСЛЕ смыкания: фрагменты фигуры уже присоединены к
            // главному контуру и не будут отброшены; убираем только настоящий
            // мелкий шум (точки внутри/снаружи).
            Mat cleanedResult = despeckle(result, whiteBox);

            double coverage = total > 0 ? 100.0 * Core.countNonZero(cleanedResult) / total : 0;
            System.out.println("isolateMainShape: k=" + k
                + " coverage=" + String.format("%.0f%%", coverage));

            if (coverage > bestCoverage) {
                bestCoverage = coverage;
                bestResult = cleanedResult;
            }
            if (coverage >= COVERAGE_OK_PERCENT) {
                saveDebugMat(debugName, "02b_despeckled.png", cleanedResult);
                return cleanedResult;
            }
        }

        // Ни одно ядро не собрало фигуру целиком — вероятно, развал на куски.
        if (!lowQualityFlag[0]) {
            lowQualityFlag[0] = true;
            qualityReason[0]  = String.format(
                "не удалось собрать фигуру: контур разорван (покрытие %.0f%%)", bestCoverage);
        }
        System.out.println("isolateMainShape: фигуру не удалось собрать, max coverage="
            + String.format("%.0f%%", bestCoverage) + " \u2192 low quality");
        return bestResult != null ? bestResult : binary.clone();
    }

    private Mat morphClean(Mat binary) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(binary, cleaned, Imgproc.MORPH_OPEN, kernel);
        return cleaned;
    }

    private Mat morphClose(Mat binary, int ksize) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
        Mat closed = new Mat();
        Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, kernel);
        return closed;
    }

    private Mat extractROI(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary.clone(), contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) return binary;

        // Bounding box по ВСЕМ контурам, а не по одному крупнейшему.
        // Рукописная фигура с разрывами распадается на несколько контуров
        // (например, оторванная нижняя грань прямоугольника). Если брать bbox
        // только самого большого контура, остальные куски обрезаются (баг square4).
        Rect rect = contours.stream()
            .map(Imgproc::boundingRect)
            .reduce((a, b) -> {
                int x1 = Math.min(a.x, b.x);
                int y1 = Math.min(a.y, b.y);
                int x2 = Math.max(a.x + a.width,  b.x + b.width);
                int y2 = Math.max(a.y + a.height, b.y + b.height);
                return new Rect(x1, y1, x2 - x1, y2 - y1);
            })
            .orElseGet(() -> Imgproc.boundingRect(contours.get(0)));

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

        String[] labels = {"P1", "P2", "P3", "P4"};
        Scalar[] colors = {new Scalar(0,255,0), new Scalar(0,0,255),
                           new Scalar(255,0,0), new Scalar(255,255,0)};
        for (int i = 0; i < approx.length; i++) {
            Imgproc.circle(canvas, approx[i], 4, colors[i % colors.length], -1);
            Imgproc.putText(canvas, labels[i % labels.length],
                new Point(approx[i].x + 5, approx[i].y - 5),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % colors.length], 1);
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
        return name.replaceAll("[^a-zA-Z0-9.\\\\-]", "_");
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