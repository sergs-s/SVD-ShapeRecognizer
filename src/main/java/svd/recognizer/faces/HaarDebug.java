package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import svd.recognizer.storage.SettingsStore;

/**
 * Отладка способа (3) — каскады Хаара для лица и глаз и аффинное выравнивание
 * по двум глазам — на ORL (шаг 5, задание (в); одна команда:
 * {@code mvn compile exec:java@faces-haar}). Только проверка цепочки глазами,
 * без таблицы метрик.
 *
 * Лицо — haarcascade_frontalface_default (scaleFactor {@value #SCALE_FACTOR},
 * minNeighbors {@value #MIN_NEIGHBORS}, наименьший размер лица — доля
 * {@value #MIN_FACE_FRACTION} от меньшей стороны входа; для ORL 92×112 —
 * 46 пикселей), при нескольких — наибольшее. Глаза — в полосе 20–60 % высоты
 * рамки лица, размер {@value #MIN_EYE}–{@value #MAX_EYE} ширины рамки; двумя
 * каскадами: haarcascade_eye и haarcascade_eye_tree_eyeglasses. Глаз «левый на
 * снимке» — кандидат левее середины рамки, ближайший к ожидаемой точке
 * (0,3; 0,38 рамки), правый — симметрично. Центр глаза — центр прямоугольника.
 * Выравнивание — подобие по двум глазам на точки глаз шаблона кадра 92×112.
 *
 * Лист (вне git): reports/faces/haar/orl_sheet.png — рамка лица (синяя), глаза
 * Хаара (красные — haarcascade_eye, фиолетовые — eyeglasses), глаза YuNet для
 * сравнения (зелёные), справа кадр выравнивания по haarcascade_eye.
 *
 * @author ssv
 */
public final class HaarDebug {

    public static final String FACE_FILE = "haarcascade_frontalface_default.xml";
    public static final String EYE_FILE = "haarcascade_eye.xml";
    public static final String EYE_GLASSES_FILE = "haarcascade_eye_tree_eyeglasses.xml";
    static final double SCALE_FACTOR = 1.1;
    static final int MIN_NEIGHBORS = 3;
    static final double MIN_FACE_FRACTION = 0.5;
    static final double MIN_EYE = 0.12;
    static final double MAX_EYE = 0.40;
    static final int EYE_FACE_WIDTH = 200;
    static final int ZOOM = 2;

    /** Результат: рамка лица (или null), глаза левый и правый на снимке (или null). */
    record Found(Rect face, double[] left, double[] right) {
        boolean eyes() {
            return left != null && right != null;
        }
    }

    private HaarDebug() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        CascadeClassifier face = load(modelsDir.resolve(FACE_FILE));
        CascadeClassifier eye = load(modelsDir.resolve(EYE_FILE));
        CascadeClassifier glasses = load(modelsDir.resolve(EYE_GLASSES_FILE));
        FaceAlignment alignment = new FaceAlignment(modelsDir, OrlDataset.WIDTH, OrlDataset.HEIGHT, settings.loadFacesDetectorScore());
        double[][] template = FaceAlignment.ownTemplate(OrlDataset.WIDTH, OrlDataset.HEIGHT);

        int minFace = (int) Math.round(MIN_FACE_FRACTION * Math.min(OrlDataset.WIDTH, OrlDataset.HEIGHT));
        int n = 0;
        int faces = 0;
        int eyesPlain = 0;
        int eyesGlasses = 0;
        List<Double> dist = new ArrayList<>();
        List<Object[]> shown = new ArrayList<>();
        long time = 0;
        for (int p = 20; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                Mat gray = FaceAlignment.readGray(datasetDir.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm"));
                long t0 = System.nanoTime();
                Found a = detect(gray, face, eye, minFace);
                time += System.nanoTime() - t0;
                Found b = a.face() == null ? a : eyes(gray, a.face(), glasses);
                FaceAlignment.Result y = alignment.process(gray, 0.0);
                n++;
                if (a.face() != null) faces++;
                if (a.eyes()) eyesPlain++;
                if (b.eyes()) eyesGlasses++;
                if (a.eyes() && y.detected()) {
                    double[][] lm = y.landmarks();
                    dist.add(Math.max(Math.hypot(a.left()[0] - lm[0][0], a.left()[1] - lm[0][1]),
                            Math.hypot(a.right()[0] - lm[1][0], a.right()[1] - lm[1][1])));
                }
                if (i == 0 || i == 5) {
                    Mat aligned = a.eyes() ? align(gray, a.left(), a.right(), template, OrlDataset.WIDTH, OrlDataset.HEIGHT) : null;
                    shown.add(new Object[] {p, i, gray, a, b, y, aligned});
                }
            }
        }
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "haar");
        Files.createDirectories(outDir);
        sheet(outDir.resolve("orl_sheet.png"), shown);
        double[] d = dist.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        StringBuilder text = new StringBuilder();
        text.append("Отладка каскадов Хаара (способ 3) на ORL — только проверка цепочки, без таблицы метрик.\n");
        text.append(String.format(Locale.ROOT, "Лицо: %s, scaleFactor %.2f, minNeighbors %d, наименьший размер лица %d px "
                + "(%.2f от меньшей стороны входа 92×112).%n", FACE_FILE, SCALE_FACTOR, MIN_NEIGHBORS, minFace, MIN_FACE_FRACTION));
        text.append(String.format(Locale.ROOT, "Глаза: полоса 20–60 %% высоты рамки, размер %.2f–%.2f ширины рамки.%n", MIN_EYE, MAX_EYE));
        text.append(String.format(Locale.ROOT, "Люди 21–40, снимков %d: лицо найдено %d; оба глаза: %s — %d, %s — %d.%n",
                n, faces, EYE_FILE, eyesPlain, EYE_GLASSES_FILE, eyesGlasses));
        text.append(String.format(Locale.ROOT, "Время (лицо + глаза %s): %.1f мс на снимок.%n", EYE_FILE, time / 1e6 / n));
        if (d.length > 0) {
            text.append(String.format(Locale.ROOT, "Для отладки (не метрика): max расстояния глаз Хаара (%s) до точек YuNet, "
                    + "пиксели ORL: медиана %.1f, 90-й перцентиль %.1f, максимум %.1f (n = %d).%n",
                    EYE_FILE, d[d.length / 2], d[(int) (0.9 * (d.length - 1))], d[d.length - 1], d.length));
        }
        text.append("Лист: reports/faces/haar/orl_sheet.png (снимки 1 и 6 людей 21–40).\n");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("orl_debug.txt"), StandardCharsets.UTF_8))) {
            out.print(text);
        }
        System.out.print(text);
    }

    static CascadeClassifier load(Path file) throws IOException {
        CascadeClassifier c = new CascadeClassifier();
        if (!c.load(file.toString())) {
            throw new IOException("Не удалось загрузить каскад " + file);
        }
        return c;
    }

    /** Лицо (наибольшее) и глаза каскадом eye; gray — 8 бит; minFace — наименьший размер лица, px входа. */
    static Found detect(Mat gray, CascadeClassifier face, CascadeClassifier eye, int minFace) {
        MatOfRect rects = new MatOfRect();
        face.detectMultiScale(gray, rects, SCALE_FACTOR, MIN_NEIGHBORS, 0, new Size(minFace, minFace), new Size());
        Rect best = null;
        for (Rect r : rects.toArray()) {
            if (best == null || r.area() > best.area()) best = r;
        }
        if (best == null) {
            return new Found(null, null, null);
        }
        return eyes(gray, best, eye);
    }

    /** Глаза в полосе 20–60 % высоты рамки лица. */
    static Found eyes(Mat gray, Rect face, CascadeClassifier eye) {
        int y0 = face.y + (int) Math.round(0.2 * face.height);
        int y1 = Math.min(gray.rows(), face.y + (int) Math.round(0.6 * face.height));
        Rect band = new Rect(face.x, y0, Math.min(face.width, gray.cols() - face.x), y1 - y0);
        // Полоса увеличивается до ширины лица EYE_FACE_WIDTH: окно каскада глаз 20×20 больше глаза на малых лицах.
        double f = Math.max(1.0, EYE_FACE_WIDTH / (double) face.width);
        Mat roi = new Mat();
        Imgproc.resize(gray.submat(band), roi, new Size(Math.round(band.width * f), Math.round(band.height * f)), 0, 0,
                Imgproc.INTER_LINEAR);
        MatOfRect found = new MatOfRect();
        int min = Math.max(4, (int) Math.round(MIN_EYE * face.width * f));
        int max = (int) Math.round(MAX_EYE * face.width * f);
        eye.detectMultiScale(roi, found, SCALE_FACTOR, MIN_NEIGHBORS, 0, new Size(min, min), new Size(max, max));
        double cx = face.x + face.width / 2.0;
        double[] expL = {face.x + 0.3 * face.width, face.y + 0.38 * face.height};
        double[] expR = {face.x + 0.7 * face.width, face.y + 0.38 * face.height};
        double[] left = null;
        double[] right = null;
        double dl = Double.MAX_VALUE;
        double dr = Double.MAX_VALUE;
        for (Rect r : found.toArray()) {
            double[] c = {band.x + (r.x + r.width / 2.0) / f, band.y + (r.y + r.height / 2.0) / f};
            if (c[0] < cx) {
                double d = Math.hypot(c[0] - expL[0], c[1] - expL[1]);
                if (d < dl) {
                    dl = d;
                    left = c;
                }
            } else {
                double d = Math.hypot(c[0] - expR[0], c[1] - expR[1]);
                if (d < dr) {
                    dr = d;
                    right = c;
                }
            }
        }
        return new Found(face, left, right);
    }

    /** Подобие по двум глазам (левый и правый на снимке) на точки глаз шаблона; BORDER_REPLICATE. */
    static Mat align(Mat gray, double[] left, double[] right, double[][] template, int width, int height) {
        double sx = right[0] - left[0];
        double sy = right[1] - left[1];
        double tx = template[1][0] - template[0][0];
        double ty = template[1][1] - template[0][1];
        double scale = Math.hypot(tx, ty) / Math.hypot(sx, sy);
        double angle = Math.atan2(ty, tx) - Math.atan2(sy, sx);
        double a = scale * Math.cos(angle);
        double b = scale * Math.sin(angle);
        double ox = template[0][0] - (a * left[0] - b * left[1]);
        double oy = template[0][1] - (b * left[0] + a * left[1]);
        Mat m = new Mat(2, 3, CvType.CV_64F);
        m.put(0, 0, a, -b, ox, b, a, oy);
        Mat out = new Mat();
        Imgproc.warpAffine(gray, out, m, new Size(width, height), Imgproc.INTER_LINEAR, org.opencv.core.Core.BORDER_REPLICATE);
        return out;
    }

    private static void sheet(Path file, List<Object[]> list) {
        int cols = 4;
        int tw = OrlDataset.WIDTH * ZOOM;
        int th = OrlDataset.HEIGHT * ZOOM;
        int tile = 2 * tw;
        int rows = (list.size() + cols - 1) / cols;
        Mat sheet = new Mat(rows * (th + 16), cols * tile, CvType.CV_8UC3, new Scalar(255, 255, 255));
        for (int t = 0; t < list.size(); t++) {
            Object[] o = list.get(t);
            int p = (int) o[0];
            int i = (int) o[1];
            Mat gray = (Mat) o[2];
            Found a = (Found) o[3];
            Found b = (Found) o[4];
            FaceAlignment.Result y = (FaceAlignment.Result) o[5];
            Mat aligned = (Mat) o[6];
            Mat big = new Mat();
            Imgproc.resize(gray, big, new Size(tw, th), 0, 0, Imgproc.INTER_NEAREST);
            Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2BGR);
            if (a.face() != null) {
                Rect f = a.face();
                Imgproc.rectangle(big, new Point(f.x * ZOOM, f.y * ZOOM), new Point((f.x + f.width) * ZOOM, (f.y + f.height) * ZOOM),
                        new Scalar(255, 0, 0), 1);
            }
            if (y.detected()) {
                for (int j = 0; j < 2; j++) {
                    Imgproc.circle(big, new Point(y.landmarks()[j][0] * ZOOM, y.landmarks()[j][1] * ZOOM), 3, new Scalar(0, 200, 0), -1);
                }
            }
            for (double[] e : new double[][] {b.left(), b.right()}) {
                if (e != null) Imgproc.drawMarker(big, new Point(e[0] * ZOOM, e[1] * ZOOM), new Scalar(255, 0, 255), Imgproc.MARKER_TILTED_CROSS, 12, 2);
            }
            for (double[] e : new double[][] {a.left(), a.right()}) {
                if (e != null) Imgproc.drawMarker(big, new Point(e[0] * ZOOM, e[1] * ZOOM), new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 12, 2);
            }
            int x0 = (t % cols) * tile;
            int y0 = (t / cols) * (th + 16);
            big.copyTo(sheet.submat(new Rect(x0, y0, tw, th)));
            if (aligned != null) {
                Mat al = new Mat();
                Imgproc.resize(aligned, al, new Size(tw, th), 0, 0, Imgproc.INTER_NEAREST);
                Imgproc.cvtColor(al, al, Imgproc.COLOR_GRAY2BGR);
                al.copyTo(sheet.submat(new Rect(x0 + tw, y0, tw, th)));
            }
            String caption = "s" + (p + 1) + "/" + (i + 1) + (a.face() == null ? " no face" : a.eyes() ? "" : " eyes?")
                    + (b.eyes() ? "" : " glasses:no");
            Imgproc.putText(sheet, caption, new Point(x0 + 2, y0 + th + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }
}
