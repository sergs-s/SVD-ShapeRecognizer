package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
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
 * Отладка способа (4) — поиск лица и глаз по DFFS (eigenfaces, eigeneyes) —
 * на ORL (шаг 5, задание (в); одна команда: {@code mvn compile exec:java@faces-dffs}).
 * Только проверка цепочки глазами, без таблицы метрик.
 *
 * Обучение — люди ORL 1–20 (все 10 снимков): кадры своего аффинного
 * выравнивания 92×112 по точкам YuNet (порог faces.detector.score);
 * пространство лиц — по кадрам целиком, пространства глаз (левый и правый на
 * снимке) — по фрагментам {@value #EYE_W}×{@value #EYE_H} вокруг точек глаз
 * шаблона. Точки YuNet здесь — только для нарезки обучающих образцов.
 * Проверка — люди 21–40: снимок с полями {@value #PAD} пикселей (BORDER_REPLICATE),
 * пирамида масштабов, окно лица с наименьшим DFFS (центр окна — внутри снимка, не в полях); затем глаза — наименьший
 * DFFS глаз в окрестности ±{@value #EYE_SEARCH} пикселей точки шаблона.
 *
 * Лист (вне git): reports/faces/dffs/orl_sheet.png — окно лица (синее),
 * глаза DFFS (красные), глаза YuNet для сравнения (зелёные).
 *
 * @author ssv
 */
public final class DffsDebug {

    static final double ETA = 0.95;
    static final int K_MAX_FACE = 50;
    static final int K_MAX_EYE = 20;
    static final int EYE_W = 20;
    static final int EYE_H = 14;
    static final int EYE_SEARCH = 6;
    static final double PAIR_DX = 0.15;
    static final double PAIR_DY = 0.10;
    /** Окно на лице: центр окна в пределах этой доли ширины рамки YuNet от её центра. */
    static final double FACE_HIT = 0.25;
    static final int PAD = 40;
    static final double[] SCALES = {0.70, 0.77, 0.85, 0.93, 1.02, 1.12, 1.24, 1.36, 1.50};
    static final int ZOOM = 2;

    /** Результат поиска: окно лица и глаза в координатах исходного снимка; value — DFFS окна. */
    record Found(double x, double y, double w, double h, double[] left, double[] right, double[] leftSep, double[] rightSep,
                 double value, double scale) {}

    private DffsDebug() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        FaceAlignment alignment = new FaceAlignment(modelsDir, OrlDataset.WIDTH, OrlDataset.HEIGHT, settings.loadFacesDetectorScore());
        double[][] template = FaceAlignment.ownTemplate(OrlDataset.WIDTH, OrlDataset.HEIGHT);

        Mat[][] raw = new Mat[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
        FaceAlignment.Result[][] res = new FaceAlignment.Result[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                raw[p][i] = FaceAlignment.readGray(datasetDir.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm"));
                res[p][i] = alignment.process(raw[p][i], 0.0);
            }
        }
        List<Mat> faces = new ArrayList<>();
        List<Mat> leftEyes = new ArrayList<>();
        List<Mat> rightEyes = new ArrayList<>();
        for (int p = 0; p < 20; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                if (!res[p][i].detected()) continue;
                Mat own = res[p][i].ownAffineGray();
                faces.add(own);
                leftEyes.add(eyePatch(own, template[0][0], template[0][1]));
                rightEyes.add(eyePatch(own, template[1][0], template[1][1]));
            }
        }
        FaceSpace faceSpace = FaceSpace.train(faces, ETA, K_MAX_FACE);
        FaceSpace leftSpace = FaceSpace.train(leftEyes, ETA, K_MAX_EYE);
        FaceSpace rightSpace = FaceSpace.train(rightEyes, ETA, K_MAX_EYE);

        List<int[]> shown = new ArrayList<>();
        List<Found> found = new ArrayList<>();
        List<Double> errJoint = new ArrayList<>();
        List<Double> errSep = new ArrayList<>();
        List<String> missesShown = new ArrayList<>();
        FaceDetectorYN yunet = OwnDetectorThreshold.create(modelsDir, settings.loadFacesDetectorScore());
        long time = 0;
        int tested = 0;
        int withBox = 0;
        int hits = 0;
        int shownHits = 0;
        for (int p = 20; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                long t0 = System.nanoTime();
                Found f = search(raw[p][i], faceSpace, leftSpace, rightSpace, template, PAD, SCALES);
                time += System.nanoTime() - t0;
                tested++;
                Mat bgr = new Mat();
                Imgproc.cvtColor(raw[p][i], bgr, Imgproc.COLOR_GRAY2BGR);
                List<DetectorDiagnostics.Box> boxes = OwnDetectorThreshold.detect(yunet, bgr, 1.0);
                boolean shownImage = i == 0 || i == 5;
                if (!boxes.isEmpty() && res[p][i].detected()) {
                    withBox++;
                    DetectorDiagnostics.Box b = DetectorDiagnostics.best(boxes);
                    boolean hit = Math.hypot(f.x() + f.w() / 2 - (b.x() + b.w() / 2), f.y() + f.h() / 2 - (b.y() + b.h() / 2))
                            <= FACE_HIT * b.w();
                    if (hit) {
                        hits++;
                        double[][] lm = res[p][i].landmarks();
                        errJoint.add(eyeError(f.left(), f.right(), lm));
                        errSep.add(eyeError(f.leftSep(), f.rightSep(), lm));
                        if (shownImage) shownHits++;
                    } else if (shownImage) {
                        missesShown.add("s" + (p + 1) + "/" + (i + 1));
                    }
                }
                if (shownImage) {
                    shown.add(new int[] {p, i});
                    found.add(f);
                }
            }
        }
        double msPerImage = time / 1e6 / tested;

        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "dffs");
        Files.createDirectories(outDir);
        sheet(outDir.resolve("orl_sheet.png"), shown, found, raw, res);
        StringBuilder text = new StringBuilder();
        text.append("Отладка DFFS (способ 4) на ORL — только проверка цепочки, без таблицы метрик.\n");
        text.append(String.format(Locale.ROOT, "Обучение: люди 1–20, кадров %d; пространство лиц k = %d (энергия %.3f, η = %.2f, kMax %d);%n",
                faces.size(), faceSpace.k(), faceSpace.energy, ETA, K_MAX_FACE));
        text.append(String.format(Locale.ROOT, "глаза %d×%d: левый на снимке k = %d (%.3f), правый k = %d (%.3f), kMax %d.%n",
                EYE_W, EYE_H, leftSpace.k(), leftSpace.energy, rightSpace.k(), rightSpace.energy, K_MAX_EYE));
        text.append("Проверка: люди 21–40, 200 снимков; поля " + PAD + " px, масштабы " + Arrays.toString(SCALES)
                + "; глаза — окрестность ±" + EYE_SEARCH + " px.\n");
        text.append(String.format(Locale.ROOT, "Время поиска: %.0f мс на снимок 92×112.%n", msPerImage));
        text.append(String.format(Locale.ROOT, "Окно на лице (центр окна в пределах %.2f ширины рамки YuNet от её центра, "
                + "порог YuNet faces.detector.score): %d/%d снимков с рамкой YuNet; на листе %d/%d, мимо: %s.%n",
                FACE_HIT, hits, withBox, shownHits, shown.size(), missesShown.isEmpty() ? "нет" : String.join(", ", missesShown)));
        text.append(String.format(Locale.ROOT, "Глаза при окне на лице, ошибка max(|Δлев|, |Δправ|) / межзрачковое YuNet "
                + "(для отладки, эталон — YuNet, не ручная разметка):%n  каждый глаз отдельно: %s;%n"
                + "  пара совместно (Δx в [%.2f; %.2f]·T, |Δy| ≤ %.2f·T, T — межзрачковое шаблона, %.2f ширины окна): %s.%n",
                quantiles(errSep), 1 - PAIR_DX, 1 + PAIR_DX, PAIR_DY,
                (template[1][0] - template[0][0]) / OrlDataset.WIDTH, quantiles(errJoint)));
        text.append("Лист: reports/faces/dffs/orl_sheet.png (снимки 1 и 6 людей 21–40; синее — окно, красные — глаза DFFS, "
                + "зелёные — YuNet).\n");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("orl_debug.txt"), StandardCharsets.UTF_8))) {
            out.print(text);
        }
        System.out.print(text);
    }

    /** max(|Δлев|, |Δправ|) / межзрачковое по точкам YuNet (lm[0] — левый на снимке, lm[1] — правый). */
    static double eyeError(double[] left, double[] right, double[][] lm) {
        double ipd = Math.hypot(lm[1][0] - lm[0][0], lm[1][1] - lm[0][1]);
        return Math.max(Math.hypot(left[0] - lm[0][0], left[1] - lm[0][1]),
                Math.hypot(right[0] - lm[1][0], right[1] - lm[1][1])) / ipd;
    }

    /** Медиана, 90-й перцентиль, доли ≤ 0,10 и ≤ 0,25. */
    static String quantiles(List<Double> values) {
        if (values.isEmpty()) return "нет данных";
        double[] d = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        long le10 = values.stream().filter(v -> v <= 0.10).count();
        long le25 = values.stream().filter(v -> v <= 0.25).count();
        return String.format(Locale.ROOT, "медиана %.2f, 90-й перцентиль %.2f; ≤ 0,10 — %d/%d, ≤ 0,25 — %d/%d",
                d[d.length / 2], d[(int) (0.9 * (d.length - 1))], le10, d.length, le25, d.length);
    }

    static Mat eyePatch(Mat frame, double cx, double cy) {
        int x = (int) Math.round(cx - EYE_W / 2.0);
        int y = (int) Math.round(cy - EYE_H / 2.0);
        return frame.submat(new Rect(x, y, EYE_W, EYE_H)).clone();
    }

    /**
     * Поиск лица (наименьший DFFS окна по пирамиде) и глаз в найденном окне.
     * gray — 8 бит; pad — поля (BORDER_REPLICATE); координаты результата — в исходном снимке.
     */
    static Found search(Mat gray, FaceSpace face, FaceSpace left, FaceSpace right, double[][] template, int pad,
                        double[] scales) {
        return search(gray, face, left, right, template, pad, scales, 0);
    }

    /** То же с отбором окон по контрасту (стандартное отклонение яркости окна не ниже minStd). */
    static Found search(Mat gray, FaceSpace face, FaceSpace left, FaceSpace right, double[][] template, int pad,
                        double[] scales, double minStd) {
        Mat padded = new Mat();
        Core.copyMakeBorder(gray, padded, pad, pad, pad, pad, Core.BORDER_REPLICATE);
        double best = Double.POSITIVE_INFINITY;
        int bx = 0;
        int by = 0;
        double bs = 1;
        Mat bestImg = null;
        for (double s : scales) {
            Mat img = new Mat();
            Imgproc.resize(padded, img, new Size(Math.round(padded.cols() * s), Math.round(padded.rows() * s)), 0, 0,
                    s < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
            if (img.cols() < face.width + 2 * EYE_SEARCH || img.rows() < face.height + 2 * EYE_SEARCH) continue;
            Mat map = face.map(img, minStd);
            // Центр окна — внутри исходного снимка (не в полях): иначе ровные поля дают ложные минимумы.
            int x0 = (int) Math.max(0, Math.ceil(pad * s - face.width / 2.0));
            int y0 = (int) Math.max(0, Math.ceil(pad * s - face.height / 2.0));
            int x1 = (int) Math.min(map.cols(), Math.floor((pad + gray.cols()) * s - face.width / 2.0) + 1);
            int y1 = (int) Math.min(map.rows(), Math.floor((pad + gray.rows()) * s - face.height / 2.0) + 1);
            if (x1 <= x0 || y1 <= y0) continue;
            Core.MinMaxLocResult mm = Core.minMaxLoc(map.submat(y0, y1, x0, x1));
            mm.minLoc = new Point(mm.minLoc.x + x0, mm.minLoc.y + y0);
            if (mm.minVal < best) {
                best = mm.minVal;
                bx = (int) mm.minLoc.x;
                by = (int) mm.minLoc.y;
                bs = s;
                bestImg = img;
            }
        }
        List<double[]> lg = eyeGrid(bestImg, left, bx + template[0][0], by + template[0][1]);
        List<double[]> rg = eyeGrid(bestImg, right, bx + template[1][0], by + template[1][1]);
        double[] ls = min(lg);
        double[] rs = min(rg);
        double[][] pair = eyePair(lg, rg, template);
        double[] l = pair == null ? ls : pair[0];
        double[] r = pair == null ? rs : pair[1];
        return new Found(bx / bs - pad, by / bs - pad, face.width / bs, face.height / bs,
                new double[] {l[0] / bs - pad, l[1] / bs - pad}, new double[] {r[0] / bs - pad, r[1] / bs - pad},
                new double[] {ls[0] / bs - pad, ls[1] / bs - pad}, new double[] {rs[0] / bs - pad, rs[1] / bs - pad}, best, bs);
    }

    /** Сетка DFFS глаза в окрестности (cx, cy) ± EYE_SEARCH: {x центра, y центра, DFFS} по всем положениям. */
    static List<double[]> eyeGrid(Mat img, FaceSpace space, double cx, double cy) {
        List<double[]> grid = new ArrayList<>();
        for (int dy = -EYE_SEARCH; dy <= EYE_SEARCH; dy++) {
            for (int dx = -EYE_SEARCH; dx <= EYE_SEARCH; dx++) {
                int x = (int) Math.round(cx + dx - EYE_W / 2.0);
                int y = (int) Math.round(cy + dy - EYE_H / 2.0);
                if (x < 0 || y < 0 || x + EYE_W > img.cols() || y + EYE_H > img.rows()) continue;
                grid.add(new double[] {x + EYE_W / 2.0, y + EYE_H / 2.0, space.dffs(img.submat(new Rect(x, y, EYE_W, EYE_H)))});
            }
        }
        return grid;
    }

    /**
     * Пара глаз совместно: наименьшая сумма DFFS левого и правого при ограничениях
     * по шаблону ORL — расстояние по x в [1 − PAIR_DX; 1 + PAIR_DX] межзрачкового
     * расстояния шаблона T (T = 0,38 ширины окна), |Δy| ≤ PAIR_DY·T. Нет пары — null.
     */
    static double[][] eyePair(List<double[]> left, List<double[]> right, double[][] template) {
        double t = template[1][0] - template[0][0];
        double best = Double.POSITIVE_INFINITY;
        double[][] pair = null;
        for (double[] l : left) {
            for (double[] r : right) {
                double dx = r[0] - l[0];
                if (dx < (1 - PAIR_DX) * t || dx > (1 + PAIR_DX) * t || Math.abs(r[1] - l[1]) > PAIR_DY * t) continue;
                if (l[2] + r[2] < best) {
                    best = l[2] + r[2];
                    pair = new double[][] {l, r};
                }
            }
        }
        return pair;
    }

    private static double[] min(List<double[]> grid) {
        double[] m = grid.get(0);
        for (double[] g : grid) if (g[2] < m[2]) m = g;
        return m;
    }

    private static void sheet(Path file, List<int[]> list, List<Found> found, Mat[][] raw, FaceAlignment.Result[][] res) {
        int cols = 8;
        int tw = OrlDataset.WIDTH * ZOOM;
        int th = OrlDataset.HEIGHT * ZOOM;
        int rows = (list.size() + cols - 1) / cols;
        Mat sheet = new Mat(rows * (th + 16), cols * tw, CvType.CV_8UC3, new Scalar(255, 255, 255));
        for (int t = 0; t < list.size(); t++) {
            int p = list.get(t)[0];
            int i = list.get(t)[1];
            Found f = found.get(t);
            Mat bgr = new Mat();
            Imgproc.cvtColor(raw[p][i], bgr, Imgproc.COLOR_GRAY2BGR);
            Mat big = new Mat();
            Imgproc.resize(bgr, big, new Size(tw, th), 0, 0, Imgproc.INTER_NEAREST);
            Imgproc.rectangle(big, new Point(f.x() * ZOOM, f.y() * ZOOM),
                    new Point((f.x() + f.w()) * ZOOM, (f.y() + f.h()) * ZOOM), new Scalar(255, 0, 0), 1);
            if (res[p][i].detected()) {
                for (int j = 0; j < 2; j++) {
                    double[] lm = res[p][i].landmarks()[j];
                    Imgproc.circle(big, new Point(lm[0] * ZOOM, lm[1] * ZOOM), 3, new Scalar(0, 200, 0), -1);
                }
            }
            Imgproc.drawMarker(big, new Point(f.left()[0] * ZOOM, f.left()[1] * ZOOM), new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 12, 2);
            Imgproc.drawMarker(big, new Point(f.right()[0] * ZOOM, f.right()[1] * ZOOM), new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 12, 2);
            int x0 = (t % cols) * tw;
            int y0 = (t / cols) * (th + 16);
            big.copyTo(sheet.submat(new Rect(x0, y0, tw, th)));
            Imgproc.putText(sheet, String.format(Locale.ROOT, "s%d/%d d=%.2f s=%.2f", p + 1, i + 1, f.value(), f.scale()),
                    new Point(x0 + 2, y0 + th + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }
}
