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
import java.util.Map;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import svd.recognizer.faces.DetectorDiagnostics.Box;
import svd.recognizer.storage.SettingsStore;

/**
 * Порог YuNet на своей базе (шаг 5, задание (б); одна команда:
 * {@code mvn compile exec:java@faces-own-yunet}).
 *
 * Снимки (после поворота по EXIF, 3000×4000) уменьшаются до
 * {@value #INPUT_WIDTH}×{@value #INPUT_HEIGHT} (INTER_AREA, масштаб 0,25) и
 * подаются в YuNet цветными. Сетка порогов 0,5…0,9. По каждому порогу: найдено
 * лицо (хотя бы одна рамка) на всех кадрах и на первых кадрах моментов;
 * лишние рамки (больше одной на кадре: в кадре один человек, поэтому каждая
 * лишняя рамка — ложная). Верность лучшей рамки проверяется глазами по листам;
 * если есть ручная разметка глаз (eyes.csv), дополнительно считается, сколько
 * лучших рамок содержат обе размеченные точки.
 *
 * Листы (вне git): reports/faces/own/yunet/ — слева весь кадр со всеми рамками
 * при пороге 0,5 (зелёная — лучшая, красная — остальные), справа лучшая рамка
 * крупно с 5 точками; подписи латиницей.
 *
 * @author ssv
 */
public final class OwnDetectorThreshold {

    static final int INPUT_WIDTH = 750;
    static final int INPUT_HEIGHT = 1000;
    static final float[] THRESHOLDS = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    /** Порог для справки: оценка лучшего кандидата на кадрах без лица. */
    static final float PROBE_THRESHOLD = 0.05f;

    private static final int WHOLE_W = 150;
    private static final int WHOLE_H = 200;
    private static final int FACE = 200;
    private static final int COLS = 4;
    private static final int PER_SHEET = 24;
    private static final int LABEL = 16;

    private OwnDetectorThreshold() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        OwnDataset dataset = OwnDataset.load(Paths.get(settings.loadFacesOwnDir()));
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "own");
        Path sheetDir = outDir.resolve("yunet");
        Files.createDirectories(sheetDir);

        List<OwnDataset.Frame> frames = new ArrayList<>();
        List<Boolean> first = new ArrayList<>();
        for (OwnDataset.Person p : dataset.persons()) {
            for (OwnDataset.Moment m : p.moments()) {
                for (OwnDataset.Frame f : m.frames()) {
                    frames.add(f);
                    first.add(f == m.representative());
                }
            }
        }
        int n = frames.size();
        Mat[] inputs = new Mat[n];
        double scale = 0;
        for (int i = 0; i < n; i++) {
            Mat full = read(frames.get(i).file());
            scale = INPUT_WIDTH / (double) full.cols();
            inputs[i] = new Mat();
            Imgproc.resize(full, inputs[i], new Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, Imgproc.INTER_AREA);
        }

        // boxes[t][i] — рамки при пороге THRESHOLDS[t]; probe[i] — при PROBE_THRESHOLD (справка).
        List<Box>[][] boxes = new List[THRESHOLDS.length][n];
        for (int t = 0; t < THRESHOLDS.length; t++) {
            FaceDetectorYN detector = create(modelsDir, THRESHOLDS[t]);
            for (int i = 0; i < n; i++) {
                boxes[t][i] = detect(detector, inputs[i], scale);
            }
        }
        FaceDetectorYN probeDetector = create(modelsDir, PROBE_THRESHOLD);
        List<Box>[] probe = new List[n];
        for (int i = 0; i < n; i++) {
            probe[i] = detect(probeDetector, inputs[i], scale);
        }

        Map<String, EyeLabeler.Label> eyes = EyeLabeler.readCsv(dataset.root().resolve(EyeLabeler.CSV_NAME));
        StringBuilder text = new StringBuilder();
        text.append("Порог YuNet на своей базе (шаг 5, задание (б))\n");
        text.append("База: ").append(dataset.root()).append(" — ").append(dataset.persons().size()).append(" человек, ")
                .append(n).append(" кадров, ").append(first.stream().filter(b -> b).count())
                .append(" представителей пригодных моментов (лучшая пометка качества в имени, при равенстве — более ранний по EXIF).\n");
        text.append("Одна сессия съёмки: результаты на этой базе оптимистичны.\n");
        text.append(String.format(Locale.ROOT, "Вход YuNet: %d×%d, цветной, INTER_AREA из 3000×4000 после поворота по EXIF "
                + "(масштаб %.2f); NMS %.1f, topK %d; при нескольких рамках берётся рамка с наибольшей оценкой.%n",
                INPUT_WIDTH, INPUT_HEIGHT, scale, FaceAlignment.NMS_THRESHOLD, FaceAlignment.TOP_K));
        text.append("Лишняя рамка — любая сверх одной на кадре (в кадре один человек, т. е. лишняя = ложная).\n");
        text.append("Верность лучшей рамки — проверка глазами по листам reports/faces/own/yunet/.\n\n");
        text.append("порог  найдено (все кадры)  найдено (представители)  кадров с лишними  лишних рамок  "
                + "лучшая рамка содержит обе точки глаз (ручная разметка)\n");
        for (int t = 0; t < THRESHOLDS.length; t++) {
            int found = 0;
            int foundFirst = 0;
            int framesExtra = 0;
            int extra = 0;
            int eyesIn = 0;
            int eyesTotal = 0;
            for (int i = 0; i < n; i++) {
                List<Box> b = boxes[t][i];
                if (!b.isEmpty()) {
                    found++;
                    if (first.get(i)) foundFirst++;
                }
                if (b.size() > 1) {
                    framesExtra++;
                    extra += b.size() - 1;
                }
                EyeLabeler.Label l = eyes.get(dataset.relative(frames.get(i).file()));
                if (l != null && !Double.isNaN(l.lx())) {
                    eyesTotal++;
                    if (!b.isEmpty() && inside(DetectorDiagnostics.best(b), l)) eyesIn++;
                }
            }
            text.append(String.format(Locale.ROOT, "%.1f    %3d/%d              %2d/%d                  %3d               %3d           %s%n",
                    THRESHOLDS[t], found, n, foundFirst, first.stream().filter(x -> x).count(), framesExtra, extra,
                    eyesTotal == 0 ? "разметки нет" : eyesIn + "/" + eyesTotal));
        }
        text.append("\nКадры без лица при пороге 0,5 (оценка лучшего кандидата при пороге ")
                .append(String.format(Locale.ROOT, "%.2f", PROBE_THRESHOLD)).append("):\n");
        boolean any = false;
        for (int i = 0; i < n; i++) {
            if (boxes[0][i].isEmpty()) {
                any = true;
                text.append("  ").append(dataset.relative(frames.get(i).file())).append(first.get(i) ? " (представитель момента)" : "")
                        .append(": ").append(probe[i].isEmpty() ? "кандидатов нет"
                                : String.format(Locale.ROOT, "%.3f", DetectorDiagnostics.best(probe[i]).score()))
                        .append('\n');
            }
        }
        if (!any) text.append("  нет\n");
        text.append("\nОценка лучшей рамки по кадрам (порог 0,5), по людям: минимум / медиана\n");
        int start = 0;
        for (OwnDataset.Person p : dataset.persons()) {
            List<Double> s = new ArrayList<>();
            int count = p.moments().stream().mapToInt(m -> m.frames().size()).sum();
            for (int i = start; i < start + count; i++) {
                if (!boxes[0][i].isEmpty()) s.add(DetectorDiagnostics.best(boxes[0][i]).score());
            }
            s.sort(null);
            text.append(String.format(Locale.ROOT, "  %-7s %2d кадров, с лицом %2d: %s%n", p.name(), count, s.size(),
                    s.isEmpty() ? "—" : String.format(Locale.ROOT, "%.3f / %.3f", s.get(0), s.get(s.size() / 2))));
            start += count;
        }

        Path report = outDir.resolve("yunet_own.txt");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(report, StandardCharsets.UTF_8))) {
            out.print(text);
        }
        System.out.print(text);
        for (int s = 0; s * PER_SHEET < n; s++) {
            sheet(sheetDir.resolve(String.format(Locale.ROOT, "sheet_%d.png", s + 1)), dataset, frames,
                    boxes[0], s * PER_SHEET, Math.min(n, (s + 1) * PER_SHEET));
        }
        System.out.println("Отчёт: " + report + "; листы: " + sheetDir);
    }

    static Mat read(Path file) throws IOException {
        Mat m = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(file)), Imgcodecs.IMREAD_COLOR);
        if (m.empty()) {
            throw new IOException("Не удалось декодировать " + file);
        }
        return m;
    }

    static FaceDetectorYN create(Path modelsDir, float threshold) {
        return FaceDetectorYN.create(modelsDir.resolve(FaceAlignment.YUNET_FILE).toString(), "",
                new Size(320, 320), threshold, FaceAlignment.NMS_THRESHOLD, FaceAlignment.TOP_K);
    }

    /** Детекция на уменьшенном цветном кадре; рамки и точки — в координатах полного снимка. */
    static List<Box> detect(FaceDetectorYN detector, Mat input, double scale) {
        detector.setInputSize(input.size());
        Mat faces = new Mat();
        detector.detect(input, faces);
        List<Box> result = new ArrayList<>();
        for (int r = 0; r < faces.rows(); r++) {
            double[][] landmarks = new double[5][2];
            for (int j = 0; j < 5; j++) {
                landmarks[j][0] = faces.get(r, 4 + 2 * j)[0] / scale;
                landmarks[j][1] = faces.get(r, 5 + 2 * j)[0] / scale;
            }
            result.add(new Box(faces.get(r, 0)[0] / scale, faces.get(r, 1)[0] / scale, faces.get(r, 2)[0] / scale,
                    faces.get(r, 3)[0] / scale, landmarks, faces.get(r, 14)[0]));
        }
        return result;
    }

    private static boolean inside(Box b, EyeLabeler.Label l) {
        return in(b, l.lx(), l.ly()) && in(b, l.rx(), l.ry());
    }

    private static boolean in(Box b, double x, double y) {
        return x >= b.x() && x <= b.x() + b.w() && y >= b.y() && y <= b.y() + b.h();
    }

    /** Лист: весь кадр с рамками (зелёная — лучшая, красная — остальные) и лучшая рамка крупно с точками. */
    private static void sheet(Path file, OwnDataset dataset, List<OwnDataset.Frame> frames, List<Box>[] boxes,
                              int from, int to) throws IOException {
        int tileW = WHOLE_W + FACE;
        int tileH = FACE;
        int n = to - from;
        int rows = (n + COLS - 1) / COLS;
        Mat sheet = new Mat(rows * (tileH + LABEL), COLS * tileW, CvType.CV_8UC3, new Scalar(255, 255, 255));
        for (int k = 0; k < n; k++) {
            int i = from + k;
            Mat full = read(frames.get(i).file());
            List<Box> b = boxes[i];
            Box top = b.isEmpty() ? null : DetectorDiagnostics.best(b);
            double s = WHOLE_W / (double) full.cols();
            Mat whole = new Mat();
            Imgproc.resize(full, whole, new Size(WHOLE_W, WHOLE_H), 0, 0, Imgproc.INTER_AREA);
            for (Box box : b) {
                Imgproc.rectangle(whole, new Point(box.x() * s, box.y() * s),
                        new Point((box.x() + box.w()) * s, (box.y() + box.h()) * s),
                        box == top ? new Scalar(0, 200, 0) : new Scalar(0, 0, 255), 1);
            }
            Mat face = new Mat(FACE, FACE, CvType.CV_8UC3, new Scalar(200, 200, 200));
            if (top != null) {
                double side = 1.3 * Math.max(top.w(), top.h());
                double cx = top.x() + top.w() / 2;
                double cy = top.y() + top.h() / 2;
                int x0 = (int) Math.max(0, Math.round(cx - side / 2));
                int y0 = (int) Math.max(0, Math.round(cy - side / 2));
                int w = (int) Math.min(full.cols() - x0, Math.round(side));
                int h = (int) Math.min(full.rows() - y0, Math.round(side));
                double fs = FACE / side;
                Mat part = new Mat();
                Imgproc.resize(new Mat(full, new Rect(x0, y0, w, h)), part,
                        new Size(Math.min(FACE, Math.round(w * fs)), Math.min(FACE, Math.round(h * fs))), 0, 0, Imgproc.INTER_AREA);
                part.copyTo(face.submat(0, part.rows(), 0, part.cols()));
                Imgproc.rectangle(face, new Point((top.x() - x0) * fs, (top.y() - y0) * fs),
                        new Point((top.x() + top.w() - x0) * fs, (top.y() + top.h() - y0) * fs), new Scalar(0, 200, 0), 1);
                Scalar[] colors = {new Scalar(0, 0, 255), new Scalar(0, 200, 0), new Scalar(255, 0, 0),
                    new Scalar(0, 200, 255), new Scalar(255, 0, 255)};
                for (int j = 0; j < 5; j++) {
                    Imgproc.circle(face, new Point((top.landmarks()[j][0] - x0) * fs, (top.landmarks()[j][1] - y0) * fs),
                            2, colors[j], -1);
                }
            }
            int x0 = (k % COLS) * tileW;
            int y0 = (k / COLS) * (tileH + LABEL);
            whole.copyTo(sheet.submat(new Rect(x0, y0, WHOLE_W, WHOLE_H)));
            face.copyTo(sheet.submat(new Rect(x0 + WHOLE_W, y0, FACE, FACE)));
            Imgproc.rectangle(sheet, new Point(x0, y0), new Point(x0 + tileW - 1, y0 + tileH - 1), new Scalar(128, 128, 128), 1);
            String name = frames.get(i).file().getFileName().toString().replace(".jpg", "");
            String caption = dataset.relative(frames.get(i).file()).split("/")[0] + " " + name.substring(name.indexOf('_') + 1)
                    + (top == null ? " none" : String.format(Locale.ROOT, " %.2f", top.score()) + (b.size() > 1 ? " +" + (b.size() - 1) : ""));
            Imgproc.putText(sheet, caption, new Point(x0 + 2, y0 + tileH + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9,
                    top == null ? new Scalar(0, 0, 255) : new Scalar(0, 0, 0), 1);
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }
}
