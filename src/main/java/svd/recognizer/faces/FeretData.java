package svd.recognizer.faces;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;

/**
 * Color FERET (PLAN.md п. 5.5, «Протокол FERET»; Release Agreement v1 — база и всё производное только локально в каталоге
 * базы): снимки анфас fa и fb из dvd1|dvd2/data/images (имя ID_ГГММДД_поза[_флаг].ppm.bz2), ручные координаты глаз из
 * data/ground_truths/name_value (только для диагностики), распаковка bz2 в памяти (файлов .ppm не пишется), YuNet в два
 * прохода и выравнивание — как у GT (кадр «а» 92×112 серый — подобие по 5 точкам на FaceAlignment.ownTemplate(92, 112);
 * alignCrop 112×112 BGR — вход SFace). Производные — в &lt;база&gt;/derived/: detections.tsv (ключ, найдено, 15 чисел строки
 * YuNet; пишется по снимку — повтор продолжает), a/&lt;ключ&gt;.png, sface/&lt;ключ&gt;.png.
 *
 * @author ssv
 */
final class FeretData {

    /** Порог уверенности YuNet — как у баз faces-far и экспорта (0,7). */
    static final float SCORE = 0.7f;
    static final String DERIVED = "derived";
    static final String DETECTIONS = "detections.tsv";

    private FeretData() {
    }

    /**
     * Снимок: ключ (имя без .ppm), человек, дата ГГММДД, поза, флаг («-» — нет; a — очки, b / c — другая причёска без очков /
     * в очках), файл, ручные координаты глаз {x, y} (left — левый глаз человека, на снимке справа; null — нет в разметке).
     */
    record Img(String key, String person, String date, String pose, String flag, Path file, double[] leftEye, double[] rightEye, double[] nose,
               double[] mouth) {
        boolean frontal() {
            return pose.equals("fa") || pose.equals("fb");
        }
    }

    /** Разбор имени ID_ГГММДД_поза[_флаг] (без расширения). */
    static Img parse(String key, Path file, double[] left, double[] right) {
        return parse(key, file, left, right, null, null);
    }

    /** То же с ручными координатами носа и центра рта (null — нет). */
    static Img parse(String key, Path file, double[] left, double[] right, double[] nose, double[] mouth) {
        String[] p = key.split("_");
        if (p.length < 3 || p.length > 4 || p[0].length() != 5 || p[1].length() != 6) throw new IllegalArgumentException("FERET: имя " + key);
        return new Img(key, p[0], p[1], p[2], p.length == 4 ? p[3] : "-", file, left, right, nose, mouth);
    }

    /** Дни между датами ГГММДД (годы 19ГГ). */
    static long days(String from, String to) {
        return ChronoUnit.DAYS.between(date(from), date(to));
    }

    static LocalDate date(String yymmdd) {
        return LocalDate.of(1900 + Integer.parseInt(yymmdd.substring(0, 2)), Integer.parseInt(yymmdd.substring(2, 4)), Integer.parseInt(yymmdd.substring(4, 6)));
    }

    /** «348 350» → {348, 350}. */
    static double[] coords(String v) {
        String[] p = v.trim().split("\\s+");
        return new double[] {Double.parseDouble(p[0]), Double.parseDouble(p[1])};
    }

    /** Снимки анфас (fa, fb) обоих дисков по возрастанию ключа, с ручными координатами глаз. */
    static List<Img> frontal(Path dir) throws IOException {
        List<Img> out = new ArrayList<>();
        for (String dvd : new String[] {"dvd1", "dvd2"}) {
            Path images = dir.resolve(dvd).resolve("data").resolve("images");
            Path truth = dir.resolve(dvd).resolve("data").resolve("ground_truths").resolve("name_value");
            try (Stream<Path> s = Files.walk(images)) {
                for (Path f : s.filter(p -> p.getFileName().toString().endsWith(".ppm.bz2")).sorted().toList()) {
                    String key = f.getFileName().toString().replace(".ppm.bz2", "");
                    Img probe = parse(key, f, null, null);
                    if (!probe.frontal()) continue;
                    double[] left = null;
                    double[] right = null;
                    double[] nose = null;
                    double[] mouth = null;
                    Path gt = truth.resolve(probe.person()).resolve(key + ".txt");
                    if (Files.exists(gt)) {
                        for (String l : Files.readAllLines(gt, StandardCharsets.UTF_8)) {
                            if (l.startsWith("left_eye_coordinates=")) left = coords(l.substring(l.indexOf('=') + 1));
                            if (l.startsWith("right_eye_coordinates=")) right = coords(l.substring(l.indexOf('=') + 1));
                            if (l.startsWith("nose_coordinates=")) nose = coords(l.substring(l.indexOf('=') + 1));
                            if (l.startsWith("mouth_coordinates=")) mouth = coords(l.substring(l.indexOf('=') + 1));
                        }
                    }
                    out.add(parse(key, f, left, right, nose, mouth));
                }
            }
        }
        out.sort((a, b) -> a.key().compareTo(b.key()));
        return out;
    }

    /** Снимок BGR из .ppm.bz2: распаковка в памяти (Commons Compress), декодирование OpenCV. */
    static Mat decode(Path bz2) throws IOException {
        byte[] ppm;
        try (InputStream in = new BZip2CompressorInputStream(new BufferedInputStream(Files.newInputStream(bz2), 1 << 16))) {
            ppm = in.readAllBytes();
        }
        Mat color = Imgcodecs.imdecode(new MatOfByte(ppm), Imgcodecs.IMREAD_COLOR);
        if (color.empty()) throw new IOException("FERET: не декодируется " + bz2);
        return color;
    }

    /** Кадр «а» 92×112 серый по строке YuNet (как GalleryEvaluation.frame, вариант A, масштаб 1). */
    static Mat frameA(Mat color, double[] row) {
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = row[4 + 2 * j];
            lm[j][1] = row[5 + 2 * j];
        }
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat a = new Mat();
        Imgproc.warpAffine(gray, a, FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(92, 112)), new Size(92, 112), Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE);
        gray.release();
        return a;
    }

    /** alignCrop 112×112 BGR по строке YuNet (вход SFace). */
    static Mat frameSface(Mat color, double[] row, FaceRecognizerSF sface) {
        Mat r = new Mat(1, 15, CvType.CV_32F);
        for (int j = 0; j < 15; j++) r.put(0, j, row[j]);
        Mat crop = new Mat();
        sface.alignCrop(color, r, crop);
        r.release();
        return crop;
    }

    /**
     * Детекции и кадры всех снимков: прочитанные из derived/detections.tsv пропускаются (кадры уже записаны), прочие —
     * распаковка, YuNet в два прохода, кадры «а» и alignCrop (через временный файл и переименование), затем строка в
     * detections.tsv. Ключ → строка YuNet (null — лицо не найдено).
     */
    static Map<String, double[]> detect(List<Img> imgs, Path dir, Path modelsDir, FaceRecognizerSF sface) throws IOException {
        Path derived = dir.resolve(DERIVED);
        Files.createDirectories(derived.resolve("a"));
        Files.createDirectories(derived.resolve("sface"));
        Path file = derived.resolve(DETECTIONS);
        Map<String, double[]> done = new HashMap<>();
        if (Files.exists(file)) {
            for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                String[] c = l.split("\t", -1);
                if (c.length != 17) continue; // оборванная последняя строка
                double[] row = null;
                if (c[1].equals("1")) {
                    row = new double[15];
                    for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[2 + i]);
                }
                done.put(c[0], row);
            }
        }
        FaceDetectorYN detector = OwnDetectorThreshold.create(modelsDir, SCORE);
        int n = 0;
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            for (Img im : imgs) {
                if (done.containsKey(im.key())) continue;
                Mat color = decode(im.file());
                double[] first = YunetTwoPass.firstPass(detector, color, 1.0);
                double[] row = null;
                if (first != null) {
                    double[] second = YunetTwoPass.secondPass(detector, color, new double[] {first[0], first[1], first[2], first[3]});
                    row = second != null ? second : first;
                    Mat a = frameA(color, row);
                    Mat s = frameSface(color, row, sface);
                    writePng(derived.resolve("a").resolve(im.key() + ".png"), a);
                    writePng(derived.resolve("sface").resolve(im.key() + ".png"), s);
                    a.release();
                    s.release();
                }
                color.release();
                StringBuilder l = new StringBuilder(im.key()).append('\t').append(row == null ? "0" : "1");
                for (int i = 0; i < 15; i++) l.append('\t').append(row == null ? "" : Double.toString(row[i]));
                w.write(l.append('\n').toString());
                w.flush();
                done.put(im.key(), row);
                if (++n % 200 == 0) System.out.printf(Locale.ROOT, "FERET: детекция %d новых (всего %d/%d)%n", n, done.size(), imgs.size());
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Img im : imgs) out.put(im.key(), done.get(im.key()));
        return out;
    }

    // ---------------------------------------------------------------- диагностика выравнивания (feret-eyes)

    static final String MANUAL = "a_manual";
    static final String Y4 = "a_y4";

    static double[] mid(double[] a, double[] b) {
        return new double[] {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2};
    }

    /**
     * Шаблон 4 точек кадра «а» 92×112: глаз на снимке слева (правый глаз человека), глаз справа, нос, центр рта —
     * середина уголков рта FaceAlignment.ownTemplate(92, 112).
     */
    static double[][] template4() {
        double[][] t = FaceAlignment.ownTemplate(92, 112);
        return new double[][] {t[0], t[1], t[2], mid(t[3], t[4])};
    }

    /** Ручные 4 точки в порядке template4: right_eye (правый глаз человека, на снимке слева), left_eye, нос, рот; null — нет разметки. */
    static double[][] manual4(Img im) {
        if (im.leftEye() == null || im.rightEye() == null || im.nose() == null || im.mouth() == null) return null;
        return new double[][] {im.rightEye(), im.leftEye(), im.nose(), im.mouth()};
    }

    /** Те же 4 точки YuNet: точки 0, 1, 2 и середина уголков рта (3, 4). */
    static double[][] yunet4(double[] row) {
        return new double[][] {{row[4], row[5]}, {row[6], row[7]}, {row[8], row[9]}, mid(new double[] {row[10], row[11]}, new double[] {row[12], row[13]})};
    }

    /** Кадр 92×112 серый подобием (FaceAlignment.similarity, LMEDS) по 4 точкам на template4. */
    static Mat frame4(Mat color, double[][] pts) {
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat a = new Mat();
        Imgproc.warpAffine(gray, a, FaceAlignment.similarity(pts, template4()), new Size(92, 112), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        gray.release();
        return a;
    }

    /**
     * Кадры Р (ручные 4 точки) и Ю4 (те же точки YuNet) снимков с разметкой и найденным лицом: derived/a_manual, derived/a_y4
     * (уже записанные пропускаются; Ю4 пишется вторым — его наличие означает, что оба кадра готовы).
     */
    static void frames4(List<Img> imgs, Map<String, double[]> det, Path dir) throws IOException {
        Path derived = dir.resolve(DERIVED);
        Files.createDirectories(derived.resolve(MANUAL));
        Files.createDirectories(derived.resolve(Y4));
        int n = 0;
        for (Img im : imgs) {
            double[][] m = manual4(im);
            double[] row = det.get(im.key());
            if (m == null || row == null || Files.exists(derived.resolve(Y4).resolve(im.key() + ".png"))) continue;
            Mat color = decode(im.file());
            Mat a = frame4(color, m);
            Mat y = frame4(color, yunet4(row));
            writePng(derived.resolve(MANUAL).resolve(im.key() + ".png"), a);
            writePng(derived.resolve(Y4).resolve(im.key() + ".png"), y);
            a.release();
            y.release();
            color.release();
            if (++n % 200 == 0) System.out.printf(Locale.ROOT, "FERET: кадры по 4 точкам — %d новых%n", n);
        }
    }

    static void writePng(Path f, Mat m) throws IOException {
        Path tmp = f.resolveSibling(f.getFileName() + ".tmp.png");
        if (!Imgcodecs.imwrite(tmp.toString(), m)) throw new IOException("Не записан " + tmp);
        Files.move(tmp, f, StandardCopyOption.REPLACE_EXISTING);
    }

    /** Кадр из derived (вариант "sface" — BGR, прочие — серые: a, a_manual, a_y4). */
    static Mat frame(Path dir, String variant, String key) throws IOException {
        Path f = dir.resolve(DERIVED).resolve(variant).resolve(key + ".png");
        Mat m = Imgcodecs.imread(f.toString(), variant.equals("sface") ? Imgcodecs.IMREAD_COLOR : Imgcodecs.IMREAD_GRAYSCALE);
        if (m.empty()) throw new IOException("Нет кадра " + f);
        return m;
    }

    /**
     * Расхождение точек глаз YuNet с ручными, в долях ручного межглазья: {левый, правый, среднее} или null (нет разметки или
     * детекции). Точки YuNet: 0 — правый глаз человека (на снимке слева), 1 — левый.
     */
    static double[] eyeError(Img im, double[] row) {
        if (row == null || im.leftEye() == null || im.rightEye() == null) return null;
        double iod = Math.hypot(im.leftEye()[0] - im.rightEye()[0], im.leftEye()[1] - im.rightEye()[1]);
        double l = Math.hypot(row[6] - im.leftEye()[0], row[7] - im.leftEye()[1]) / iod;
        double r = Math.hypot(row[4] - im.rightEye()[0], row[5] - im.rightEye()[1]) / iod;
        return new double[] {l, r, (l + r) / 2, iod};
    }

    /**
     * То же для явных точек: re — правый глаз человека (на снимке слева), le — левый; null — нет разметки или точки.
     */
    static double[] eyeError(Img im, double[] re, double[] le) {
        if (re == null || le == null) return null;
        double[] row = new double[15];
        row[4] = re[0];
        row[5] = re[1];
        row[6] = le[0];
        row[7] = le[1];
        return eyeError(im, row);
    }
}
