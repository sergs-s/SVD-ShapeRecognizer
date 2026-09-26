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
import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import svd.recognizer.storage.SettingsStore;

/**
 * Разбор отказов YuNet на ORL (подготовка к шагу 5; одна команда:
 * {@code mvn compile exec:java@faces-yunet}). Основной протокол не меняется.
 *
 * Эталон — рабочая настройка (порог 0,9, без полей). Варианты: порог 0,6 / 0,7 /
 * 0,8; масштаб 2× (INTER_CUBIC, setInputSize под новый размер, координаты
 * делятся на 2); повтор с полями 25 %, затем 50 % только при неудаче
 * (координаты сдвигаются на поле). По каждому варианту: найдено ли лицо на
 * снимках, где эталон не нашёл; лишние (больше одной рамки) и ложные рамки
 * на всех 400 снимках. Критерий ложной рамки рассчитан на ORL, где лицо
 * занимает почти весь кадр: центр вне средней половины кадра или ширина меньше
 * 30 % ширины кадра. На свои фото не переносится.
 *
 * Листы для проверки глазами: снимки в родном разрешении, увеличенные ×2 без
 * сглаживания, с рамками (зелёная — лучшая, красная — остальные) и 5 точками.
 *
 * @author ssv
 */
public final class DetectorDiagnostics {

    static final double FALSE_CENTER_MARGIN = 0.25;
    static final double FALSE_MIN_WIDTH = 0.30;
    static final int ZOOM = 2;
    static final int SHEET_COLS = 4;
    static final int LABEL = 16;

    /** Вариант детекции. */
    enum Variant {
        REFERENCE("reference", "порог 0,9, без полей (рабочая настройка)", 0.9f, 1, false),
        T08("t08", "порог 0,8", 0.8f, 1, false),
        T07("t07", "порог 0,7", 0.7f, 1, false),
        T06("t06", "порог 0,6", 0.6f, 1, false),
        SCALE2("scale2", "масштаб 2×, порог 0,9", 0.9f, 2, false),
        PAD_RETRY("pad_retry", "порог 0,9, при неудаче поля 25 %, затем 50 %", 0.9f, 1, true),
        PROBE("probe", "порог 0,1 (только для оценки лучшего кандидата)", 0.1f, 1, false);

        final String id;
        final String label;
        final float threshold;
        final int scale;
        final boolean padRetry;

        Variant(String id, String label, float threshold, int scale, boolean padRetry) {
            this.id = id;
            this.label = label;
            this.threshold = threshold;
            this.scale = scale;
            this.padRetry = padRetry;
        }
    }

    /** Рамка: x, y, w, h, 5 точек, оценка — в координатах исходного снимка. */
    record Box(double x, double y, double w, double h, double[][] landmarks, double score) {}

    private DetectorDiagnostics() {}

    public static void main(String[] args) throws IOException {
        OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "yunet");
        Files.createDirectories(outDir);

        Mat[][] images = new Mat[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                images[p][i] = FaceAlignment.readGray(datasetDir.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm"));
            }
        }
        List<Box>[][][] boxes = detectAll(modelsDir, images);

        List<int[]> failed = new ArrayList<>();
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                if (boxes[Variant.REFERENCE.ordinal()][p][i].isEmpty()) failed.add(new int[] {p, i});
            }
        }

        StringBuilder text = new StringBuilder();
        text.append("Разбор отказов YuNet на ORL (подготовка к шагу 5). Основной протокол не меняется.\n");
        text.append("Модель: ").append(modelsDir.resolve(FaceAlignment.YUNET_FILE)).append("; NMS 0,3.\n");
        text.append(String.format(Locale.ROOT,
                "Ложная рамка — центр вне средней половины кадра или ширина < %.0f %% ширины кадра. Критерий рассчитан на ORL%n"
                        + "(лицо на весь кадр) и на свои фото не переносится. Лишние — снимки с более чем одной рамкой.%n",
                100 * FALSE_MIN_WIDTH));
        text.append(String.format(Locale.ROOT, "Снимков без лица у рабочей настройки: %d — %s.%n%n",
                failed.size(), names(failed)));

        text.append("Лучший кандидат на этих снимках при пороге 0,1:\n");
        for (int[] f : failed) {
            List<Box> b = boxes[Variant.PROBE.ordinal()][f[0]][f[1]];
            text.append(String.format(Locale.ROOT, "  s%d/%d: %s%n", f[0] + 1, f[1] + 1,
                    b.isEmpty() ? "кандидатов нет" : String.format(Locale.ROOT, "оценка %.3f%s", best(b).score(),
                            isFalse(best(b), images[f[0]][f[1]]) ? " (по критерию — ложная рамка)" : "")));
        }
        text.append('\n');
        text.append("вариант                                        найдено на отказах   лишние (снимков / рамок)   ложные рамки (снимков / рамок)   всего снимков с лицом\n");
        List<int[]> flagged = new ArrayList<>();
        for (Variant v : Variant.values()) {
            if (v == Variant.PROBE) {
                continue;
            }
            int found = 0;
            StringBuilder foundList = new StringBuilder();
            for (int[] f : failed) {
                List<Box> b = boxes[v.ordinal()][f[0]][f[1]];
                if (!b.isEmpty()) {
                    found++;
                    foundList.append(String.format(Locale.ROOT, " s%d/%d(%.2f)", f[0] + 1, f[1] + 1, best(b).score()));
                }
            }
            int extraImages = 0;
            int extraBoxes = 0;
            int falseImages = 0;
            int falseBoxes = 0;
            int withFace = 0;
            for (int p = 0; p < OrlDataset.PERSONS; p++) {
                for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                    List<Box> b = boxes[v.ordinal()][p][i];
                    if (!b.isEmpty()) withFace++;
                    if (b.size() > 1) {
                        extraImages++;
                        extraBoxes += b.size() - 1;
                    }
                    int f = 0;
                    for (Box box : b) {
                        if (isFalse(box, images[p][i])) f++;
                    }
                    if (f > 0) {
                        falseImages++;
                        falseBoxes += f;
                    }
                    if ((f > 0 || b.size() > 1) && !contains(flagged, p, i)) {
                        flagged.add(new int[] {p, i});
                    }
                }
            }
            text.append(String.format(Locale.ROOT, "%-46s %2d/%d%-10s %3d / %-3d                  %3d / %-3d                        %d/400%n",
                    v.label, found, failed.size(), "", extraImages, extraBoxes, falseImages, falseBoxes, withFace));
            if (found > 0) {
                text.append("    найдены:").append(foundList).append('\n');
            }
            sheet(outDir.resolve("sheet_" + v.id + ".png"), failed, images, boxes[v.ordinal()], v.id);
        }
        text.append(String.format(Locale.ROOT, "%nСнимков с лишними или ложными рамками хотя бы в одном варианте: %d%s.%n",
                flagged.size(), flagged.isEmpty() ? "" : " — " + names(flagged)));
        if (!flagged.isEmpty()) {
            // Лист помеченных — по варианту с наименьшим порогом из рабочих (0,6).
            sheet(outDir.resolve("sheet_flagged_t06.png"), flagged.subList(0, Math.min(20, flagged.size())),
                    images, boxes[Variant.T06.ordinal()], "flagged, t06");
        }
        text.append("Листы для проверки глазами: reports/faces/yunet/sheet_<вариант>.png (отказы рабочей настройки,\n");
        text.append("родное разрешение ×2 без сглаживания), sheet_flagged_t06.png — снимки с лишними или ложными рамками.\n");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.getParent().resolve("yunet.txt"), StandardCharsets.UTF_8))) {
            out.print(text);
        }
        System.out.println("Готово: " + outDir.getParent().resolve("yunet.txt"));
    }

    @SuppressWarnings("unchecked")
    static List<Box>[][][] detectAll(Path modelsDir, Mat[][] images) {
        List<Box>[][][] all = new List[Variant.values().length][OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
        for (Variant v : Variant.values()) {
            FaceDetectorYN detector = FaceDetectorYN.create(modelsDir.resolve(FaceAlignment.YUNET_FILE).toString(), "",
                    new Size(320, 320), v.threshold, FaceAlignment.NMS_THRESHOLD, FaceAlignment.TOP_K);
            for (int p = 0; p < OrlDataset.PERSONS; p++) {
                for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                    List<Box> b = detect(detector, images[p][i], v.scale, 0.0);
                    if (b.isEmpty() && v.padRetry) {
                        b = detect(detector, images[p][i], 1, 0.25);
                        if (b.isEmpty()) {
                            b = detect(detector, images[p][i], 1, 0.50);
                        }
                    }
                    all[v.ordinal()][p][i] = b;
                }
            }
        }
        return all;
    }

    /** Детекция с масштабом и полями; рамки и точки — в координатах исходного снимка. */
    static List<Box> detect(FaceDetectorYN detector, Mat gray, int scale, double padding) {
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
        Mat input = bgr;
        if (scale != 1) {
            input = new Mat();
            Imgproc.resize(bgr, input, new Size(bgr.cols() * scale, bgr.rows() * scale), 0, 0, Imgproc.INTER_CUBIC);
        }
        int[] pad = FaceAlignment.paddingPixels(input, padding);
        if (pad[0] > 0 || pad[1] > 0) {
            Mat padded = new Mat();
            Core.copyMakeBorder(input, padded, pad[1], pad[1], pad[0], pad[0], Core.BORDER_REPLICATE);
            input = padded;
        }
        detector.setInputSize(input.size());
        Mat faces = new Mat();
        detector.detect(input, faces);
        List<Box> result = new ArrayList<>();
        for (int r = 0; r < faces.rows(); r++) {
            double[] lm = new double[10];
            for (int j = 0; j < 10; j++) {
                double offset = j % 2 == 0 ? pad[0] : pad[1];
                lm[j] = (faces.get(r, 4 + j)[0] - offset) / scale;
            }
            double[][] landmarks = new double[5][2];
            for (int j = 0; j < 5; j++) {
                landmarks[j][0] = lm[2 * j];
                landmarks[j][1] = lm[2 * j + 1];
            }
            result.add(new Box((faces.get(r, 0)[0] - pad[0]) / scale, (faces.get(r, 1)[0] - pad[1]) / scale,
                    faces.get(r, 2)[0] / scale, faces.get(r, 3)[0] / scale, landmarks, faces.get(r, 14)[0]));
        }
        return result;
    }

    static Box best(List<Box> boxes) {
        Box best = boxes.get(0);
        for (Box b : boxes) {
            if (b.score() > best.score()) best = b;
        }
        return best;
    }

    /** Ложная рамка (критерий для ORL): центр вне средней половины кадра или ширина < 30 % ширины кадра. */
    static boolean isFalse(Box b, Mat image) {
        double cx = b.x() + b.w() / 2;
        double cy = b.y() + b.h() / 2;
        double w = image.cols();
        double h = image.rows();
        boolean centerOut = cx < FALSE_CENTER_MARGIN * w || cx > (1 - FALSE_CENTER_MARGIN) * w
                || cy < FALSE_CENTER_MARGIN * h || cy > (1 - FALSE_CENTER_MARGIN) * h;
        return centerOut || b.w() < FALSE_MIN_WIDTH * w;
    }

    private static boolean contains(List<int[]> list, int p, int i) {
        for (int[] a : list) {
            if (a[0] == p && a[1] == i) return true;
        }
        return false;
    }

    private static String names(List<int[]> list) {
        List<String> n = new ArrayList<>();
        for (int[] a : list) n.add("s" + (a[0] + 1) + "/" + (a[1] + 1));
        return String.join(", ", n);
    }

    /**
     * Лист: снимки ×2 без сглаживания, рамки (зелёная — лучшая, красная — остальные) и 5 точек.
     * Подписи — латиницей: шрифты Hershey в OpenCV не рисуют кириллицу.
     */
    static void sheet(Path file, List<int[]> list, Mat[][] images, List<Box>[][] boxes, String title) {
        int tileW = OrlDataset.WIDTH * ZOOM;
        int tileH = OrlDataset.HEIGHT * ZOOM;
        int rows = (list.size() + SHEET_COLS - 1) / SHEET_COLS;
        Mat sheet = new Mat(LABEL + rows * (tileH + LABEL), SHEET_COLS * tileW, CvType.CV_8UC3, new Scalar(255, 255, 255));
        Imgproc.putText(sheet, title, new Point(4, 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
        for (int t = 0; t < list.size(); t++) {
            int p = list.get(t)[0];
            int i = list.get(t)[1];
            Mat bgr = new Mat();
            Imgproc.cvtColor(images[p][i], bgr, Imgproc.COLOR_GRAY2BGR);
            Mat big = new Mat();
            Imgproc.resize(bgr, big, new Size(tileW, tileH), 0, 0, Imgproc.INTER_NEAREST);
            List<Box> b = boxes[p][i];
            Box top = b.isEmpty() ? null : best(b);
            for (Box box : b) {
                Scalar color = box == top ? new Scalar(0, 200, 0) : new Scalar(0, 0, 255);
                Imgproc.rectangle(big, new Point(box.x() * ZOOM, box.y() * ZOOM),
                        new Point((box.x() + box.w()) * ZOOM, (box.y() + box.h()) * ZOOM), color, 1);
                for (double[] lm : box.landmarks()) {
                    Imgproc.circle(big, new Point(lm[0] * ZOOM, lm[1] * ZOOM), 2, color, -1);
                }
            }
            int x0 = (t % SHEET_COLS) * tileW;
            int y0 = LABEL + (t / SHEET_COLS) * (tileH + LABEL);
            big.copyTo(sheet.submat(new Rect(x0, y0, tileW, tileH)));
            String caption = "s" + (p + 1) + "/" + (i + 1) + (top == null ? " none"
                    : String.format(Locale.ROOT, " %.2f", top.score()) + (b.size() > 1 ? " +" + (b.size() - 1) : ""));
            Imgproc.putText(sheet, caption, new Point(x0 + 2, y0 + tileH + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9,
                    top == null ? new Scalar(0, 0, 255) : new Scalar(0, 0, 0), 1);
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }
}
