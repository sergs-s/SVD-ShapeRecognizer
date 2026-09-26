package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;

/**
 * Детекция лица YuNet и выравнивание по 5 точкам (шаг 5, подготовка).
 *
 * Модели opencv_zoo (ONNX, вне git, путь — SettingsStore.loadFacesModelsDir()):
 * YuNet face_detection_yunet_2023mar.onnx (MIT) и SFace
 * face_recognition_sface_2021dec.onnx (Apache 2.0).
 *
 * Способы выравнивания:
 * <ol>
 *   <li>{@code ALIGN_CROP} — FaceRecognizerSF.alignCrop: кадр 112×112 (вход SFace).
 *       Режим заполнения задан внутри OpenCV: warpAffine с BORDER_CONSTANT = 0;</li>
 *   <li>{@code OWN_AFFINE} — estimateAffinePartial2D (подобие) 5 точек YuNet на шаблон
 *       ArcFace, приведённый к кадру faces.frame.width × faces.frame.height (по умолчанию
 *       92×112, как ORL: сдвиг на −10 по x), warpAffine с BORDER_REPLICATE.</li>
 * </ol>
 * Поля перед детекцией — copyMakeBorder с BORDER_REPLICATE. Для каждого кадра
 * считается доля пикселей, взятых вне исходного снимка (поля или заполнение).
 *
 * Экземпляр не потокобезопасен (FaceDetectorYN хранит размер входа).
 *
 * @author ssv
 */
public final class FaceAlignment {

    public static final String YUNET_FILE = "face_detection_yunet_2023mar.onnx";
    public static final String SFACE_FILE = "face_recognition_sface_2021dec.onnx";
    static final float SCORE_THRESHOLD = 0.9f;
    static final float NMS_THRESHOLD = 0.3f;
    static final int TOP_K = 5000;

    /** Шаблон 5 точек ArcFace для кадра 112×112 (левый глаз на снимке, правый, нос, углы рта). */
    static final double[][] ARCFACE_112 = {
        {38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}
    };
    /** Сторона кадра шаблона ArcFace. */
    static final int ARCFACE_SIZE = 112;

    /** Способ выравнивания. */
    public enum Method {
        ALIGN_CROP("align_crop"), OWN_AFFINE("own_affine");

        public final String label;

        Method(String label) {
            this.label = label;
        }
    }

    /**
     * Результат для одного снимка: признаки по способам или неудача детекции.
     * landmarks — 5 точек в координатах исходного снимка (x, y), roll — наклон
     * линии глаз в градусах.
     */
    public record Result(boolean detected, double score, double[][] landmarks, double roll,
                         double[] alignCropVector, double alignCropOutside,
                         double[] ownAffineVector, double ownAffineOutside,
                         double[] sfaceFeature, Mat alignCropGray, Mat ownAffineGray, Mat original) {}

    private final FaceDetectorYN detector;
    private final FaceRecognizerSF recognizer;
    private final int frameWidth;
    private final int frameHeight;
    private final double[][] ownTemplate;

    /**
     * @param modelsDir   папка моделей opencv_zoo
     * @param frameWidth  ширина кадра своего аффинного выравнивания (SettingsStore, по умолчанию 92)
     * @param frameHeight высота кадра своего аффинного выравнивания (SettingsStore, по умолчанию 112)
     */
    public FaceAlignment(Path modelsDir, int frameWidth, int frameHeight) {
        if (frameWidth <= 0 || frameHeight <= 0) {
            throw new IllegalArgumentException("Размер кадра должен быть положительным: " + frameWidth + "×" + frameHeight);
        }
        this.detector = FaceDetectorYN.create(modelsDir.resolve(YUNET_FILE).toString(), "",
                new Size(320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K);
        this.recognizer = FaceRecognizerSF.create(modelsDir.resolve(SFACE_FILE).toString(), "");
        this.frameWidth = frameWidth;
        this.frameHeight = frameHeight;
        this.ownTemplate = ownTemplate(frameWidth, frameHeight);
    }

    /** Читает снимок в оттенках серого через imdecode (пути с не-ASCII символами). */
    static Mat readGray(Path file) throws IOException {
        Mat gray = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(file)), Imgcodecs.IMREAD_GRAYSCALE);
        if (gray.empty()) {
            throw new IOException("Не удалось декодировать " + file);
        }
        return gray;
    }

    /** Поле (в пикселях) по ширине и высоте при доле padding от размера снимка. */
    static int[] paddingPixels(Mat gray, double padding) {
        return new int[] {(int) Math.round(padding * gray.cols()), (int) Math.round(padding * gray.rows())};
    }

    /**
     * Детекция и выравнивание одного снимка.
     *
     * @param gray    исходный снимок, 8 бит, 1 канал
     * @param padding доля поля с каждой стороны (0 — без полей)
     */
    public Result process(Mat gray, double padding) {
        int[] pad = paddingPixels(gray, padding);
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
        Mat padded = new Mat();
        Core.copyMakeBorder(bgr, padded, pad[1], pad[1], pad[0], pad[0], Core.BORDER_REPLICATE);
        // Маска исходного снимка в координатах кадра с полями: 255 внутри снимка, 0 в полях.
        Mat mask = Mat.zeros(padded.size(), CvType.CV_8UC3);
        mask.submat(pad[1], pad[1] + gray.rows(), pad[0], pad[0] + gray.cols()).setTo(new Scalar(255, 255, 255));

        detector.setInputSize(padded.size());
        Mat faces = new Mat();
        detector.detect(padded, faces);
        if (faces.rows() == 0) {
            return new Result(false, 0, null, Double.NaN, null, Double.NaN, null, Double.NaN, null, null, null, gray);
        }
        int best = 0;
        for (int i = 1; i < faces.rows(); i++) {
            if (faces.get(i, 14)[0] > faces.get(best, 14)[0]) {
                best = i;
            }
        }
        Mat face = faces.row(best);
        double score = face.get(0, 14)[0];
        double[][] landmarks = new double[5][2];
        for (int j = 0; j < 5; j++) {
            landmarks[j][0] = face.get(0, 4 + 2 * j)[0] - pad[0];
            landmarks[j][1] = face.get(0, 5 + 2 * j)[0] - pad[1];
        }
        double roll = Math.toDegrees(Math.atan2(landmarks[1][1] - landmarks[0][1], landmarks[1][0] - landmarks[0][0]));

        // Способ 1: alignCrop (кадр 112×112) по кадру с полями; маска — тем же преобразованием.
        Mat crop = new Mat();
        recognizer.alignCrop(padded, face, crop);
        Mat cropMask = new Mat();
        recognizer.alignCrop(mask, face, cropMask);
        Mat cropGray = new Mat();
        Imgproc.cvtColor(crop, cropGray, Imgproc.COLOR_BGR2GRAY);
        Mat feature = new Mat();
        recognizer.feature(crop, feature);
        Mat feature64 = new Mat();
        feature.convertTo(feature64, CvType.CV_64F);
        double[] sface = new double[(int) feature64.total()];
        feature64.get(0, 0, sface);

        // Способ 2: собственное аффинное выравнивание по исходному снимку (без полей).
        Mat transform = similarity(landmarks, ownTemplate);
        Mat own = new Mat();
        Imgproc.warpAffine(gray, own, transform, new Size(frameWidth, frameHeight),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        Mat ownMask = new Mat();
        Mat ones = new Mat(gray.size(), CvType.CV_8UC1, new Scalar(255));
        Imgproc.warpAffine(ones, ownMask, transform, new Size(frameWidth, frameHeight),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));

        return new Result(true, score, landmarks, roll,
                toVector(cropGray), outsideFraction(cropMask), toVector(own), outsideFraction(ownMask),
                sface, cropGray, own, gray);
    }

    /**
     * Шаблон 5 точек для кадра width×height: ArcFace (112×112), масштабированный
     * по высоте кадра (s = height/112) и центрированный по ширине. При 92×112 —
     * ArcFace, сдвинутый на −10 по x.
     */
    static double[][] ownTemplate(int width, int height) {
        double s = height / (double) ARCFACE_SIZE;
        double shiftX = (width - ARCFACE_SIZE * s) / 2;
        double[][] t = new double[5][2];
        for (int i = 0; i < 5; i++) {
            t[i][0] = ARCFACE_112[i][0] * s + shiftX;
            t[i][1] = ARCFACE_112[i][1] * s;
        }
        return t;
    }

    /** Преобразование подобия (4 степени свободы) из точек from в точки to. */
    static Mat similarity(double[][] from, double[][] to) {
        MatOfPoint2f src = new MatOfPoint2f(toPoints(from));
        MatOfPoint2f dst = new MatOfPoint2f(toPoints(to));
        Mat inliers = new Mat();
        return Calib3d.estimateAffinePartial2D(src, dst, inliers, Calib3d.LMEDS);
    }

    private static Point[] toPoints(double[][] xy) {
        Point[] points = new Point[xy.length];
        for (int i = 0; i < xy.length; i++) {
            points[i] = new Point(xy[i][0], xy[i][1]);
        }
        return points;
    }

    /** Вектор яркостей 0..1 построчно (как ImageVectorizer). */
    static double[] toVector(Mat gray8) {
        byte[] pixels = new byte[(int) gray8.total()];
        gray8.get(0, 0, pixels);
        double[] v = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            v[i] = (pixels[i] & 0xFF) / 255.0;
        }
        return v;
    }

    /** Доля пикселей маски, пришедших не из исходного снимка (значение < 128). */
    static double outsideFraction(Mat mask) {
        Mat single = mask;
        if (mask.channels() > 1) {
            single = new Mat();
            Imgproc.cvtColor(mask, single, Imgproc.COLOR_BGR2GRAY);
        }
        byte[] pixels = new byte[(int) single.total()];
        single.get(0, 0, pixels);
        int outside = 0;
        for (byte b : pixels) {
            if ((b & 0xFF) < 128) outside++;
        }
        return outside / (double) pixels.length;
    }
}
