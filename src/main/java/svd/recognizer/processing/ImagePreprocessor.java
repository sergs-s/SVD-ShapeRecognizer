package svd.recognizer.processing;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Конвейер предобработки изображения перед подачей в SVD.
 *
 * Шаги конвейера:
 * 1. Загрузка JPEG с диска
 * 2. Перевод в оттенки серого
 * 3. Бинаризация методом Оцу (адаптивный порог, не зависит от цвета ручки)
 * 4. Морфологические операции (убрать шум, соединить разрывы линии)
 * 5. Поиск контура и обрезка по bounding box с запасом
 * 6. Помещение ROI на увеличенный холст перед поворотом
 * 7. Грубое выравнивание угла через PCA-подобную оценку ориентации
 * 8. Поворот через аффинную матрицу OpenCV 2x3, применяемую функцией warpAffine
 * 9. Масштабирование до 64×64 пикселей
 * 10. Возврат нормализованного изображения и матрицы яркостей
 *
 * Важные замечания:
 * - перед поворотом ROI помещается на увеличенный холст с коэффициентом CANVAS_SCALE = 1.6,
 *   чтобы warpAffine не срезал углы;
 * - PCA используется только как грубое выравнивание оси;
 * - для симметричных фигур возможна неоднозначность по 180 градусов;
 * - для круга ориентация почти неинформативна.
 *
 * Требует инициализации OpenCV: nu.pattern.OpenCV.loadLocally()
 *
 * @author ssv
 */
public class ImagePreprocessor {
    public static final int OUTPUT_SIZE = 64;
    private static final int BBOX_PADDING = 12;
    private static final double CANVAS_SCALE = 1.6;

    /**
     * Контейнер результата предобработки.
     *
     * Хранит:
     * 1. Нормализованное изображение 64x64 для GUI и усреднения эталонов
     * 2. Матрицу яркостей double[][] для вычисления SVD-признаков
     */
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
            throw new IllegalArgumentException("Не удалось загрузить изображение: " + imageFile.getAbsolutePath());
        }

        Mat gray = toGrayscale(source);
        Mat binary = binarizeOtsu(gray);
        Mat cleaned = morphClean(binary);
        Mat cropped = extractROI(cleaned);
        Mat aligned = normalizeAngle(cropped);
        Mat resized = resize64(aligned);
        BufferedImage image = matToBufferedImage(resized);
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

    private Mat binarizeOtsu(Mat gray) {
        Mat blurred = new Mat();
        Mat binary = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0.0);
        Imgproc.threshold(blurred, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        return binary;
    }

    private Mat morphClean(Mat binary) {
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat closed = new Mat();
        Mat opened = new Mat();
        Imgproc.morphologyEx(binary, closed, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(closed, opened, Imgproc.MORPH_OPEN, kernel);
        return opened;
    }

    private Mat extractROI(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
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

    private Mat normalizeAngle(Mat cropped) {
        double angle = estimateOrientationAngle(cropped);
        int canvasWidth = Math.max(1, (int) Math.round(cropped.cols() * CANVAS_SCALE));
        int canvasHeight = Math.max(1, (int) Math.round(cropped.rows() * CANVAS_SCALE));
        Mat canvas = Mat.zeros(canvasHeight, canvasWidth, cropped.type());
        int offsetX = (canvasWidth - cropped.cols()) / 2;
        int offsetY = (canvasHeight - cropped.rows()) / 2;
        cropped.copyTo(canvas.submat(new Rect(offsetX, offsetY, cropped.cols(), cropped.rows())));

        Point center = new Point(canvas.cols() / 2.0, canvas.rows() / 2.0);
        Mat rotation = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
        Mat rotated = new Mat();
        Imgproc.warpAffine(
                canvas,
                rotated,
                rotation,
                new Size(canvas.cols(), canvas.rows()),
                Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );
        return extractROI(rotated);
    }

    private double estimateOrientationAngle(Mat binary) {
        List<Point> points = new ArrayList<>();
        for (int y = 0; y < binary.rows(); y++) {
            for (int x = 0; x < binary.cols(); x++) {
                double[] pixel = binary.get(y, x);
                if (pixel != null && pixel[0] > 0) {
                    points.add(new Point(x, y));
                }
            }
        }
        if (points.size() < 5) {
            return 0.0;
        }
        MatOfPoint2f pointSet = new MatOfPoint2f();
        pointSet.fromList(points);
        RotatedRect box = Imgproc.minAreaRect(pointSet);
        double angle = box.angle;
        if (box.size.width < box.size.height) {
            angle += 90.0;
        }
        return angle;
    }

    private Mat resize64(Mat source) {
        Mat resized = new Mat();
        Imgproc.resize(source, resized, new Size(OUTPUT_SIZE, OUTPUT_SIZE), 0, 0, Imgproc.INTER_AREA);
        return resized;
    }

    private BufferedImage matToBufferedImage(Mat source) {
        Mat normalized = new Mat();
        if (source.type() != CvType.CV_8UC1) {
            source.convertTo(normalized, CvType.CV_8UC1);
        } else {
            normalized = source;
        }
        BufferedImage image = new BufferedImage(normalized.cols(), normalized.rows(), BufferedImage.TYPE_BYTE_GRAY);
        byte[] sourceData = new byte[(int) (normalized.total() * normalized.channels())];
        normalized.get(0, 0, sourceData);
        byte[] targetData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
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