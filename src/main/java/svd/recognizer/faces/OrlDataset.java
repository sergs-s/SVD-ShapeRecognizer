package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.processing.ImageVectorizer;

/**
 * База лиц ORL (AT&amp;T Laboratories Cambridge, «The Database of Faces»):
 * 40 человек по 10 снимков 92×112, оттенки серого, PGM, папки s1…s40,
 * файлы 1.pgm…10.pgm.
 *
 * Снимки декодируются OpenCV. Файл читается средствами Java и передаётся в
 * Imgcodecs.imdecode: Imgcodecs.imread на Windows не открывает пути с
 * не-ASCII символами (например, кириллицей). Путь к базе —
 * SettingsStore.loadFacesDatasetDir().
 *
 * Яркость приводится к 0..1 делением на 255 (как у эталонов фигур); других
 * нормализаций нет. Эквализация гистограммы — отдельный вариант для
 * сравнения, включается флагом.
 *
 * @author ssv
 */
public final class OrlDataset {

    public static final int PERSONS = 40;
    public static final int IMAGES_PER_PERSON = 10;
    /** Родной размер снимков ORL (размер кадра задаётся отдельно, SettingsStore). */
    public static final int WIDTH = 92;
    public static final int HEIGHT = 112;

    /** vectors[person][image] — вектор длины width·height кадра (построчно). */
    private final double[][][] vectors;

    private OrlDataset(double[][][] vectors) {
        this.vectors = vectors;
    }

    /**
     * Загружает всю базу. OpenCV должен быть уже инициализирован
     * (nu.pattern.OpenCV.loadLocally()).
     *
     * @param dir           корневая папка базы (с подпапками s1…s40)
     * @param equalizeHist  применять ли эквализацию гистограммы к каждому снимку
     * @param width         ширина кадра (SettingsStore.loadFacesFrameWidth(); при 92 — без масштабирования)
     * @param height        высота кадра (SettingsStore.loadFacesFrameHeight(); при 112 — без масштабирования)
     * @return загруженная база
     * @throws IOException              если файл не читается
     * @throws IllegalArgumentException если файл не декодируется, исходный размер не 92×112
     *                                  или размер кадра не положительный
     */
    public static OrlDataset load(Path dir, boolean equalizeHist, int width, int height) throws IOException {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Размер кадра должен быть положительным: " + width + "×" + height);
        }
        double[][][] vectors = new double[PERSONS][IMAGES_PER_PERSON][];
        for (int p = 0; p < PERSONS; p++) {
            for (int i = 0; i < IMAGES_PER_PERSON; i++) {
                Path file = dir.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm");
                vectors[p][i] = readVector(file, equalizeHist, width, height);
            }
        }
        return new OrlDataset(vectors);
    }

    private static double[] readVector(Path file, boolean equalizeHist, int width, int height) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException("Нет файла базы ORL: " + file);
        }
        Mat gray = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(file)),
                Imgcodecs.IMREAD_GRAYSCALE);
        if (gray.empty()) {
            throw new IllegalArgumentException("Не удалось декодировать " + file);
        }
        if (gray.cols() != WIDTH || gray.rows() != HEIGHT) {
            throw new IllegalArgumentException("Размер " + file + ": " + gray.cols() + "×"
                    + gray.rows() + ", ожидается " + WIDTH + "×" + HEIGHT);
        }
        if (width != WIDTH || height != HEIGHT) {
            Mat resized = new Mat();
            Imgproc.resize(gray, resized, new Size(width, height), 0, 0, Imgproc.INTER_AREA);
            gray = resized;
        }
        if (equalizeHist) {
            Imgproc.equalizeHist(gray, gray);
        }
        byte[] pixels = new byte[width * height];
        gray.get(0, 0, pixels);
        double[][] matrix = new double[height][width];
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                matrix[r][c] = (pixels[r * width + c] & 0xFF) / 255.0;
            }
        }
        return ImageVectorizer.toVector(matrix);
    }

    /**
     * @param person номер человека 0…39 (папка s{person+1})
     * @param image  номер снимка 0…9 (файл {image+1}.pgm)
     * @return вектор снимка (не копия — не изменять)
     */
    public double[] vector(int person, int image) {
        return vectors[person][image];
    }
}
