package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
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
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Base;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.storage.SettingsStore;

/**
 * (4а), отключение (9): YuNet → каскады Хаара (режим «только классика»; PLAN.md п. 5.5, план Васи от 17.10.2026 в TASKS.md,
 * уточнения №2 Жорика). Лицо – haarcascade_frontalface_default (параметры {@link HaarDebug}, наибольшая рамка, наименьший
 * размер лица – {@value #MIN_FACE_FRACTION} меньшей стороны входа для всех баз); глаза – haarcascade_eye_tree_eyeglasses в
 * полосе 20–60 % высоты рамки ({@link HaarDebug#eyes}). Всё – на входе конвейера: своя база – снимок, уменьшенный в
 * {@link OwnEvaluation#INPUT_SCALE} (как вход YuNet и кадр «а»), прочие – как есть; координаты в кэше – в полном снимке.
 * <p>
 * Кадр 92×112 – подобие по двум глазам на точки глаз шаблона кадра «а». Глаз не найден: (i) – отказ; (ii) – оба глаза по
 * геометрии рамки ({@value #GEOM_X}; {@value #GEOM_Y}), поза тогда формально 0. Поза r_H = (x середины глаз − x центра рамки)
 * / межглазье по x.
 * <p>
 * Режим abl-haar (XBaseEval) – проверка без распознавания: FERET fa/fb, GT, FEI (все исходные, №11–14, когорта), своя база;
 * кэши рамок и глаз – derived/haar.tsv (FERET), haar.tsv рядом с gt_db, в каталогах FEI и своей базы.
 *
 * @author ssv
 */
final class HaarFrames {

    static final double MIN_FACE_FRACTION = HaarDebug.OWN_MIN_FACE_FRACTION;
    static final double GEOM_X = 0.3;
    static final double GEOM_Y = 0.38;
    /** «Ложная рамка» (только диагностика): IoU с рамкой YuNet меньше этого. */
    static final double IOU_FALSE = 0.3;
    static final String CACHE = "haar.tsv";
    static final String SIGNATURE = String.format(Locale.ROOT, "haar v1|face %s|eyes %s|scaleFactor %.2f|minNeighbors %d|minFace %.2f|"
            + "band 0.2-0.6|eye %.2f-%.2f|eyeFaceWidth %d|own input %.2f", HaarDebug.FACE_FILE, HaarDebug.EYE_GLASSES_FILE, HaarDebug.SCALE_FACTOR,
            HaarDebug.MIN_NEIGHBORS, MIN_FACE_FRACTION, HaarDebug.MIN_EYE, HaarDebug.MAX_EYE, HaarDebug.EYE_FACE_WIDTH, OwnEvaluation.INPUT_SCALE);
    static final int W = 92;
    static final int H = 112;

    private HaarFrames() {
    }

    /** Итог Хаара на снимке (полные координаты): число рамок, наибольшая {x, y, w, h} (или null), глаз слева и справа на снимке. */
    record Det(int rects, double[] face, double[] left, double[] right) {

        boolean found() {
            return face != null;
        }

        int eyes() {
            return (left != null ? 1 : 0) + (right != null ? 1 : 0);
        }

        boolean both() {
            return left != null && right != null;
        }

        /** Глаза для кадра: найденные оба; иначе при geom – по геометрии рамки; иначе null. */
        double[][] eyesFor(boolean geom) {
            if (both()) return new double[][] {left, right};
            if (!geom || face == null) return null;
            return new double[][] {{face[0] + GEOM_X * face[2], face[1] + GEOM_Y * face[3]},
                {face[0] + (1 - GEOM_X) * face[2], face[1] + GEOM_Y * face[3]}};
        }

        /** r_H по найденным глазам; NaN – глаз не два. */
        double pose() {
            if (!both()) return Double.NaN;
            return ((left[0] + right[0]) / 2 - (face[0] + face[2] / 2)) / Math.abs(right[0] - left[0]);
        }
    }

    interface Loader {
        Mat color(String key) throws IOException;
    }

    /** Вход конвейера: серый, для своей базы – уменьшенный (INTER_AREA). */
    static Mat input(Mat color, double scale) {
        Mat g = EyeRefine.gray(color);
        if (scale == 1.0) return g;
        Mat small = new Mat();
        Imgproc.resize(g, small, new Size(Math.round(g.cols() * scale), Math.round(g.rows() * scale)), 0, 0, Imgproc.INTER_AREA);
        g.release();
        return small;
    }

    static Det detect(Mat in, double scale, CascadeClassifier face, CascadeClassifier eye) {
        int minFace = (int) Math.round(MIN_FACE_FRACTION * Math.min(in.cols(), in.rows()));
        MatOfRect rects = new MatOfRect();
        face.detectMultiScale(in, rects, HaarDebug.SCALE_FACTOR, HaarDebug.MIN_NEIGHBORS, 0, new Size(minFace, minFace), new Size());
        Rect[] all = rects.toArray();
        Rect best = null;
        for (Rect r : all) if (best == null || r.area() > best.area()) best = r;
        if (best == null) return new Det(0, null, null, null);
        HaarDebug.Found f = HaarDebug.eyes(in, best, eye);
        return new Det(all.length, new double[] {best.x / scale, best.y / scale, best.width / scale, best.height / scale}, back(f.left(), scale),
                back(f.right(), scale));
    }

    private static double[] back(double[] p, double scale) {
        return p == null ? null : new double[] {p[0] / scale, p[1] / scale};
    }

    /** Кадр 92×112 по глазам Хаара (geom – вариант (ii)); null – отказ. */
    static Mat frame(Mat in, double scale, Det d, boolean geom) {
        double[][] e = d.eyesFor(geom);
        if (e == null) return null;
        return HaarDebug.align(in, new double[] {e[0][0] * scale, e[0][1] * scale}, new double[] {e[1][0] * scale, e[1][1] * scale},
                FaceAlignment.ownTemplate(W, H), W, H);
    }

    // ---------------------------------------------------------------- кэш

    /** Кэш: готовые строки читаются, недостающие ключи считаются и дописываются; итог – в порядке keys. */
    static Map<String, Det> cached(Path file, List<String> keys, Loader loader, double scale, CascadeClassifier face, CascadeClassifier eye,
                                   String label) throws IOException {
        Map<String, Det> done = new HashMap<>();
        boolean fresh = !Files.exists(file);
        if (!fresh) {
            List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            if (lines.isEmpty() || !lines.get(0).equals("# " + SIGNATURE))
                throw new IllegalStateException(file + ": другая подпись кэша Хаара – удалите файл и повторите (ожидалось «# " + SIGNATURE + "»)");
            for (String l : lines.subList(1, lines.size())) {
                String[] c = l.split("\t", -1);
                if (c.length != 10) continue;
                double[] f = c[2].isEmpty() ? null : new double[] {Double.parseDouble(c[2]), Double.parseDouble(c[3]), Double.parseDouble(c[4]),
                    Double.parseDouble(c[5])};
                double[] le = c[6].isEmpty() ? null : new double[] {Double.parseDouble(c[6]), Double.parseDouble(c[7])};
                double[] re = c[8].isEmpty() ? null : new double[] {Double.parseDouble(c[8]), Double.parseDouble(c[9])};
                done.put(c[0], new Det(Integer.parseInt(c[1]), f, le, re));
            }
        }
        int n = 0;
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (fresh) w.write("# " + SIGNATURE + "\n");
            for (String k : keys) {
                if (done.containsKey(k)) continue;
                Mat color = loader.color(k);
                Mat in = input(color, scale);
                color.release();
                Det d = detect(in, scale, face, eye);
                in.release();
                StringBuilder l = new StringBuilder(k).append('\t').append(d.rects());
                for (int i = 0; i < 4; i++) l.append('\t').append(d.face() == null ? "" : Double.toString(d.face()[i]));
                l.append("\t\t");
                for (double[] p : new double[][] {d.left(), d.right()})
                    l.append('\t').append(p == null ? "" : Double.toString(p[0])).append('\t').append(p == null ? "" : Double.toString(p[1]));
                w.write(l.append('\n').toString());
                w.flush();
                done.put(k, d);
                if (++n % 500 == 0) System.out.printf(Locale.ROOT, "abl-haar: %s – %d новых%n", label, n);
            }
        }
        Map<String, Det> out = new LinkedHashMap<>();
        for (String k : keys) out.put(k, done.get(k));
        return out;
    }

    static double iou(double[] a, double[] b) {
        double x0 = Math.max(a[0], b[0]);
        double y0 = Math.max(a[1], b[1]);
        double x1 = Math.min(a[0] + a[2], b[0] + b[2]);
        double y1 = Math.min(a[1] + a[3], b[1] + b[3]);
        double i = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
        return i / (a[2] * a[3] + b[2] * b[3] - i);
    }

    // ---------------------------------------------------------------- abl-haar

    /** База для отчёта: снимки, итоги Хаара, строки YuNet (null – не найдено), группы, загрузчик и масштаб входа. */
    record Set(String name, Map<String, Det> haar, Map<String, double[]> yunet, Map<String, List<String>> groups, Loader loader, double scale) {}

    static void ablHaar(Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        double rLimit = XBaseEval.poseLimit(selLines);
        Path models = Paths.get(settings.loadFacesModelsDir());
        CascadeClassifier face = HaarDebug.load(models.resolve(HaarDebug.FACE_FILE));
        CascadeClassifier eye = HaarDebug.load(models.resolve(HaarDebug.EYE_GLASSES_FILE));

        Bases b = bases(exportDir, feiKeys, rLimit, settings, face, eye);
        Set own = b.own();
        Set gt = b.gt();
        Set fei = b.fei();
        Set feret = b.feret();
        Map<String, Det> ownH = own.haar();
        Map<String, Det> feiH = fei.haar();
        List<String> ownProto = b.ownProto();
        List<String> fei1114 = b.fei1114();
        Path gtDir = b.gtDir();
        Path feiDir = b.feiDir();
        Path ownDir = b.ownDir();
        Path feretDir = b.feretDir();
        Map<String, Img> frImg = b.frImg();

        // Пороги позы.
        double bigR = percentileAbs(ownProto, ownH);
        double r0 = percentileAbs(fei1114, feiH);
        int ownBoth = (int) ownProto.stream().filter(k -> ownH.get(k).both()).count();
        int feiBoth = (int) fei1114.stream().filter(k -> feiH.get(k).both()).count();
        String r0Base = settings.loadFacesFeiPoseR0();

        String head = String.format(Locale.ROOT, "Код: коммит %s%nКаскады: лицо %s, глаза %s; scaleFactor %.2f, minNeighbors %d; наименьшее лицо – %.2f меньшей "
                + "стороны входа (все базы); глаза – полоса 20–60 %% высоты рамки, размер %.2f–%.2f ширины рамки; при нескольких рамках – наибольшая.%n"
                + "Вход: своя база – снимок, уменьшенный в %.2f (как вход YuNet и кадр «а»), прочие – как есть.%n"
                + "Поза r_H = (x середины глаз − x центра рамки) / межглазье по x (только при двух найденных глазах).%n"
                + "R_H = 95-й процентиль |r_H| по снимкам протокола своей базы с двумя глазами (%d из %d) = %.4f (у YuNet R = %.4f).%n"
                + "r0_H = 95-й процентиль |r_H| по №11–14 FEI с двумя глазами (%d из %d) = %.4f (у YuNet r0 = %s).%n"
                + "«Ложная рамка» – IoU с рамкой YuNet < %.1f (только диагностика, на снимках, где лицо нашли оба).%n"
                + "Отказ (i) – нет лица или глаз меньше двух; отказ (ii) – нет лица (глаза по геометрии рамки %.2f; %.2f).%n",
                FarMethods.commit(), HaarDebug.FACE_FILE, HaarDebug.EYE_GLASSES_FILE, HaarDebug.SCALE_FACTOR, HaarDebug.MIN_NEIGHBORS, MIN_FACE_FRACTION,
                HaarDebug.MIN_EYE, HaarDebug.MAX_EYE, OwnEvaluation.INPUT_SCALE, ownBoth, ownProto.size(), bigR, rLimit, feiBoth,
                fei1114.size(), r0, r0Base == null ? "не задан" : r0Base, IOU_FALSE, GEOM_X, GEOM_Y);

        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, (4а), отключение (9) YuNet → каскады Хаара: проверка без распознавания (XBaseEval abl-haar)\n");
        r.append(head).append("Данные: экспорт ").append(exportDir).append("; снимки – ").append(gtDir).append(", ").append(feiDir).append(", ")
                .append(ownDir).append("; кэши Хаара – ").append(CACHE).append(" в их каталогах (рядом с gt_db).\n");
        r.append("FERET – отдельным отчётом, только в каталоге FERET.\n");
        for (Set s : List.of(gt, fei, own)) report(r, s, bigR, r0, rLimit, null);
        Path sheets = out.resolve("abl_haar");
        Files.createDirectories(sheets);
        for (Set s : List.of(gt, fei, own)) {
            Path f = sheets.resolve(s.name().equals("своя база") ? "own_sheet.png" : s.name().toLowerCase(Locale.ROOT) + "_sheet.png");
            sheet(f, s, seed);
            r.append("Лист: ").append(f).append('\n');
        }
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_abl_haar.txt"), r);
        System.out.print(r);

        StringBuilder q = new StringBuilder("PLAN.md п. 5.5, (4а), отключение (9) YuNet → каскады Хаара, FERET: проверка без распознавания (XBaseEval abl-haar); "
                + "Color FERET – только локально\n");
        q.append(head).append("Данные: ").append(feretDir).append(" (YuNet – derived/detections.tsv; Хаар – derived/").append(CACHE).append(")\n");
        report(q, feret, bigR, r0, rLimit, frImg);
        Path fOut = feretDir.resolve("reports").resolve("eq");
        Files.createDirectories(fOut.resolve("abl_haar"));
        sheet(fOut.resolve("abl_haar").resolve("feret_sheet.png"), feret, seed);
        q.append("Лист: ").append(fOut.resolve("abl_haar").resolve("feret_sheet.png")).append('\n');
        FarMethods.write(fOut.resolve("xbase_feret_abl_haar.txt"), q);
        System.out.print(q);
    }

    static double percentileAbs(List<String> keys, Map<String, Det> h) {
        double[] v = keys.stream().map(h::get).filter(Det::both).mapToDouble(d -> Math.abs(d.pose())).sorted().toArray();
        return FeiDataset.percentile(v, FeiDataset.POSE_PERCENTILE);
    }

    /** YuNet FEI (все исходные) из кэша faces-far, как mp-boxes: «человек-NN» → строка. */
    static Map<String, double[]> feiYunet(Path feiDir, Map<String, List<Sample>> orig, SettingsStore settings) throws IOException {
        List<Sample> samples = new ArrayList<>();
        orig.values().forEach(samples::addAll);
        for (String p : orig.keySet()) {
            for (int k = 0; k < 2; k++) {
                Path f = FeiDataset.reference(feiDir, p, "ab".charAt(k));
                if (f != null) samples.add(new Sample(Base.FEI, p, f, ' '));
            }
        }
        Path cache = Paths.get(System.getProperty("user.dir"), "reports", "faces", "far").resolve(FeiDataset.CACHE_NAME);
        Map<String, GalleryEvaluation.Det> dets = GalleryEvaluation.loadCache(cache, GalleryEvaluation.signature(samples, 0.7f, settings));
        if (dets == null) throw new IllegalStateException("abl-haar: кэш детекций FEI " + cache + " не подходит (подпись)");
        Map<String, double[]> out = new TreeMap<>();
        for (List<Sample> l : orig.values())
            for (Sample s : l) out.put(s.person() + "-" + String.format(Locale.ROOT, "%02d", FeiDataset.number(s)), dets.get(s.file().toString()).row());
        return out;
    }

    static String pct(int a, int n) {
        return String.format(Locale.ROOT, "%d (%.1f %%)", a, n == 0 ? 0.0 : 100.0 * a / n);
    }

    static void report(StringBuilder r, Set s, double bigR, double r0, double rLimit, Map<String, Img> feret) {
        r.append("\n=== ").append(s.name()).append(": обнаружение ===\n");
        r.append("набор | снимков | YuNet нашёл | Хаар: лицо не найдено | рамок > 1 | ложная рамка | глаз 0 | глаз 1 | глаз 2 | отказ (i) | отказ (ii) | "
                + "лицо YuNet, но не Хаар | Хаар, но не YuNet\n");
        for (Map.Entry<String, List<String>> g : s.groups().entrySet()) {
            List<String> ks = g.getValue();
            int n = ks.size();
            int y = 0, nf = 0, multi = 0, fal = 0, both = 0, e0 = 0, e1 = 0, e2 = 0, yOnly = 0, hOnly = 0;
            for (String k : ks) {
                Det d = s.haar().get(k);
                double[] row = s.yunet().get(k);
                if (row != null) y++;
                if (!d.found()) {
                    nf++;
                    if (row != null) yOnly++;
                    continue;
                }
                if (row == null) hOnly++;
                if (d.rects() > 1) multi++;
                if (row != null) {
                    both++;
                    if (iou(d.face(), row) < IOU_FALSE) fal++;
                }
                if (d.eyes() == 0) e0++;
                else if (d.eyes() == 1) e1++;
                else e2++;
            }
            r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %d | %d из %d | %d | %d | %d | %s | %s | %d | %d%n", g.getKey(), n, pct(y, n), pct(nf, n),
                    multi, fal, both, e0, e1, e2, pct(n - e2, n), pct(nf, n), yOnly, hOnly));
        }

        r.append("\n=== ").append(s.name()).append(": поза r_H (снимки с двумя глазами) и отбор ===\n");
        r.append(String.format(Locale.ROOT, "набор | с двумя глазами | |r_H|: 50 %% | 75 %% | 95 %% | макс | |r_H| ≤ R_H = %.4f | |r_H| ≤ r0_H = %.4f (модель X) | "
                + "YuNet: |r| ≤ R = %.4f (из найденных) | Хаар (i) и отбор R_H ∩ отбор YuNet%n", bigR, r0, rLimit));
        for (Map.Entry<String, List<String>> g : s.groups().entrySet()) {
            List<String> ks = g.getValue().stream().filter(k -> s.haar().get(k).both()).toList();
            double[] v = FeiBootstrap.sorted(ks.stream().mapToDouble(k -> Math.abs(s.haar().get(k).pose())).toArray());
            int pass = (int) ks.stream().filter(k -> Math.abs(s.haar().get(k).pose()) <= bigR).count();
            int x = (int) ks.stream().filter(k -> Math.abs(s.haar().get(k).pose()) <= r0).count();
            List<String> yk = g.getValue().stream().filter(k -> s.yunet().get(k) != null).toList();
            int yPass = (int) yk.stream().filter(k -> Math.abs(FeiDataset.pose(s.yunet().get(k))) <= rLimit).count();
            int inter = (int) ks.stream().filter(k -> Math.abs(s.haar().get(k).pose()) <= bigR && s.yunet().get(k) != null
                    && Math.abs(FeiDataset.pose(s.yunet().get(k))) <= rLimit).count();
            r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s | %s | %s | %s | %d%n", g.getKey(), ks.size(), q(v, 0.5), q(v, 0.75), q(v, 0.95),
                    v.length == 0 ? "–" : String.format(Locale.ROOT, "%.3f", v[v.length - 1]), pct(pass, ks.size()), pct(x, ks.size()), pct(yPass, yk.size()),
                    inter));
        }

        r.append("\n=== ").append(s.name()).append(": глаза Хаара против Ю5 (справочно; среднее по двум глазам / межглазье YuNet; снимки, где оба нашли "
                + "лицо и Хаар – два глаза) ===\n");
        r.append("набор | снимков | медиана | 95 % | > 10 % | > 25 %\n");
        for (Map.Entry<String, List<String>> g : s.groups().entrySet()) {
            List<String> ks = g.getValue().stream().filter(k -> s.haar().get(k).both() && s.yunet().get(k) != null).toList();
            double[] v = FeiBootstrap.sorted(ks.stream().mapToDouble(k -> {
                Det d = s.haar().get(k);
                double[] row = s.yunet().get(k);
                double iod = Math.hypot(row[6] - row[4], row[7] - row[5]);
                return (Math.hypot(d.left()[0] - row[4], d.left()[1] - row[5]) + Math.hypot(d.right()[0] - row[6], d.right()[1] - row[7])) / 2 / iod;
            }).toArray());
            r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s%n", g.getKey(), v.length, q(v, 0.5), q(v, 0.95), share(v, 0.10), share(v, 0.25)));
        }

        if (feret != null) {
            r.append("\n=== FERET: глаза против ручной разметки (среднее по двум глазам / ручное межглазье; размеченные снимки, где YuNet нашёл лицо и Хаар – "
                    + "два глаза) ===\n");
            r.append("кандидат | снимков | медиана | 95 % | > 5 % | > 10 % | > 25 %\n");
            List<double[]> hy = new ArrayList<>();
            for (String k : s.groups().get("fa, fb (все)")) {
                Img im = feret.get(k);
                Det d = s.haar().get(k);
                double[] row = s.yunet().get(k);
                if (im.leftEye() == null || im.rightEye() == null || row == null || !d.both()) continue;
                hy.add(new double[] {FeretData.eyeError(im, d.left(), d.right())[2], FeretData.eyeError(im, row)[2]});
            }
            for (int c = 0; c < 2; c++) {
                final int cc = c;
                double[] v = FeiBootstrap.sorted(hy.stream().mapToDouble(a -> a[cc]).toArray());
                r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s | %s%n", c == 0 ? "Хаар" : "Ю5", v.length, q(v, 0.5), q(v, 0.95), share(v, 0.05),
                        share(v, 0.10), share(v, 0.25)));
            }
        }
    }

    static String q(double[] sorted, double p) {
        return sorted.length == 0 ? "–" : String.format(Locale.ROOT, "%.3f", FeiBootstrap.quantile(sorted, p));
    }

    static String share(double[] v, double t) {
        if (v.length == 0) return "–";
        int c = 0;
        for (double x : v) if (x > t) c++;
        return String.format(Locale.ROOT, "%.1f %%", 100.0 * c / v.length);
    }

    // ---------------------------------------------------------------- лист

    /**
     * Лист проверки: до 12 отказов (i) и 12 случайных удач (seed); плитка – окрестность лица (рамка Хаара синяя, глаза Хаара
     * красные, глаза YuNet зелёные) и кадр (ii) ×2.
     */
    static void sheet(Path file, Set s, long seed) throws IOException {
        List<String> fails = new ArrayList<>();
        List<String> ok = new ArrayList<>();
        for (String k : s.haar().keySet()) (s.haar().get(k).both() ? ok : fails).add(k);
        Random rnd = new Random(seed + 1);
        Collections.shuffle(fails, rnd);
        Collections.shuffle(ok, rnd);
        List<String> list = new ArrayList<>(fails.subList(0, Math.min(12, fails.size())));
        list.addAll(ok.subList(0, Math.min(24 - list.size(), ok.size())));
        int side = 224;
        int fw = 2 * W;
        int tile = side + fw;
        int cols = 4;
        int rows = (list.size() + cols - 1) / cols;
        Mat sheet = new Mat(Math.max(1, rows) * (side + 16), cols * tile, CvType.CV_8UC3, new Scalar(255, 255, 255));
        for (int t = 0; t < list.size(); t++) {
            String k = list.get(t);
            Det d = s.haar().get(k);
            double[] row = s.yunet().get(k);
            Mat color = s.loader().color(k);
            Mat in = input(color, s.scale());
            color.release();
            double sc = s.scale();
            double[] box = d.found() ? d.face() : row != null ? row : new double[] {0, 0, in.cols() / sc, in.rows() / sc};
            double cx = (box[0] + box[2] / 2) * sc;
            double cy = (box[1] + box[3] / 2) * sc;
            double half = 0.8 * Math.max(box[2], box[3]) * sc;
            int x0 = (int) Math.max(0, Math.round(cx - half));
            int y0 = (int) Math.max(0, Math.round(cy - half));
            int x1 = (int) Math.min(in.cols(), Math.round(cx + half));
            int y1 = (int) Math.min(in.rows(), Math.round(cy + half));
            double k2 = side / (double) Math.max(x1 - x0, y1 - y0);
            Mat crop = new Mat();
            Imgproc.resize(in.submat(new Rect(x0, y0, x1 - x0, y1 - y0)), crop, new Size(Math.round((x1 - x0) * k2), Math.round((y1 - y0) * k2)), 0, 0,
                    Imgproc.INTER_AREA);
            Imgproc.cvtColor(crop, crop, Imgproc.COLOR_GRAY2BGR);
            java.util.function.Function<double[], Point> map = p -> new Point((p[0] * sc - x0) * k2, (p[1] * sc - y0) * k2);
            if (d.found()) Imgproc.rectangle(crop, map.apply(d.face()), map.apply(new double[] {d.face()[0] + d.face()[2], d.face()[1] + d.face()[3]}),
                    new Scalar(255, 0, 0), 1);
            if (row != null) for (int j = 0; j < 2; j++) Imgproc.circle(crop, map.apply(new double[] {row[4 + 2 * j], row[5 + 2 * j]}), 3, new Scalar(0, 200, 0), -1);
            for (double[] e : new double[][] {d.left(), d.right()})
                if (e != null) Imgproc.drawMarker(crop, map.apply(e), new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 12, 2);
            int tx = (t % cols) * tile;
            int ty = (t / cols) * (side + 16);
            crop.copyTo(sheet.submat(new Rect(tx, ty, crop.cols(), crop.rows())));
            Mat f = frame(in, sc, d, true);
            if (f != null) {
                Mat big = new Mat();
                Imgproc.resize(f, big, new Size(fw, 2 * H), 0, 0, Imgproc.INTER_NEAREST);
                Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2BGR);
                big.copyTo(sheet.submat(new Rect(tx + side, ty, fw, 2 * H)));
            }
            String cap = k.replaceAll("[^\\x20-\\x7E]", "?") + (d.found() ? " eyes " + d.eyes() : " no face")
                    + (d.both() ? String.format(Locale.ROOT, " r %.2f", d.pose()) : "");
            Imgproc.putText(sheet, cap, new Point(tx + 2, ty + side + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
            in.release();
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }

    /** Базы abl-haar: своя база, GT, FEI, FERET; снимки протокола своей базы, №11–14 FEI, каталоги. */
    record Bases(Set own, Set gt, Set fei, Set feret, Map<String, Img> frImg, List<String> ownProto, List<String> fei1114, Path gtDir, Path feiDir,
                 Path ownDir, Path feretDir) {}

    /** Снимки, строки YuNet и итоги Хаара (кэши haar.tsv) по базам. */
    static Bases bases(Path exportDir, Map<String, Map<Integer, String>> feiKeys, double rLimit, SettingsStore settings, CascadeClassifier face,
                       CascadeClassifier eye) throws IOException {
        long seed = settings.loadFacesSeed();
        Path models = Paths.get(settings.loadFacesModelsDir());
        // Своя база: все снимки (кэш) и снимки протокола (экспорт) – для R_H.
        Path ownDir = Paths.get(settings.loadFacesOwnDir());
        Map<String, Path> ownFiles = new TreeMap<>();
        for (OwnDataset.Person p : OwnDataset.load(ownDir).persons())
            for (OwnDataset.Moment m : p.moments()) for (OwnDataset.Frame f : m.frames()) ownFiles.put(p.name() + "/" + FarExport.stem(f.file()), f.file());
        Map<String, double[]> ownY = XBaseMpPts.exportRows(exportDir, "own");
        List<String> ownKeys = new ArrayList<>(ownFiles.keySet());
        Map<String, Det> ownH = cached(ownDir.resolve(CACHE), ownKeys, k -> Imgcodecs.imread(ownFiles.get(k).toString(), Imgcodecs.IMREAD_COLOR),
                OwnEvaluation.INPUT_SCALE, face, eye, "своя база");
        Map<String, List<String>> ownG = new LinkedHashMap<>();
        ownG.put("все снимки", ownKeys);
        ownG.put("снимки протокола", new ArrayList<>(ownY.keySet()));
        for (String k : ownY.keySet()) if (!ownFiles.containsKey(k)) throw new IllegalStateException("abl-haar: нет снимка своей базы " + k);

        // FEI: все исходные; YuNet – кэш faces-far.
        Path feiDir = Paths.get(settings.loadFacesFeiDir());
        Map<String, List<Sample>> orig = FeiDataset.originals(feiDir);
        Map<String, Path> feiFiles = new TreeMap<>();
        for (List<Sample> l : orig.values()) for (Sample s : l) feiFiles.put(s.person() + "-" + String.format(Locale.ROOT, "%02d", FeiDataset.number(s)), s.file());
        Map<String, double[]> feiY = feiYunet(feiDir, orig, settings);
        List<String> feiKeysAll = new ArrayList<>(feiFiles.keySet());
        Map<String, Det> feiH = cached(feiDir.resolve(CACHE), feiKeysAll, k -> Imgcodecs.imread(feiFiles.get(k).toString(), Imgcodecs.IMREAD_COLOR), 1.0,
                face, eye, "FEI");
        List<String> fei1114 = feiKeysAll.stream().filter(k -> Integer.parseInt(k.substring(4)) >= 11).toList();
        List<String> cohort = new ArrayList<>();
        for (String p : XBaseEval.cohortPersons(feiKeys, seed))
            for (int num : feiKeys.get(p).keySet()) cohort.add(p + "-" + String.format(Locale.ROOT, "%02d", num));
        Map<String, List<String>> feiG = new LinkedHashMap<>();
        feiG.put("все исходные", feiKeysAll);
        feiG.put("№11–14", fei1114);
        feiG.put("когорта (20 человек, отобранные)", cohort);

        // GT: все снимки gt_db; отобранные по YuNet.
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> gtY = XBaseMpPts.exportRows(exportDir, "gt");
        List<String> gtKeys = new ArrayList<>(gtY.keySet());
        Map<String, Det> gtH = cached(gtDir.getParent().resolve(CACHE), gtKeys, k -> Imgcodecs.imread(gtDir.resolve(k + ".jpg").toString(),
                Imgcodecs.IMREAD_COLOR), 1.0, face, eye, "GT");
        Map<String, List<String>> gtG = new LinkedHashMap<>();
        gtG.put("все", gtKeys);
        gtG.put("отобранные (YuNet, |r| ≤ R)", gtKeys.stream().filter(k -> gtY.get(k) != null && Math.abs(FeiDataset.pose(gtY.get(k))) <= rLimit).toList());

        // FERET fa/fb.
        Path feretDir = Paths.get(settings.loadFacesFeretDir());
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        List<Img> fr = FeretData.frontal(feretDir);
        Map<String, double[]> frY = FeretData.detect(fr, feretDir, models, sface);
        Map<String, Img> frImg = new LinkedHashMap<>();
        for (Img im : fr) frImg.put(im.key(), im);
        List<String> frKeys = new ArrayList<>(frImg.keySet());
        Map<String, Det> frH = cached(feretDir.resolve(FeretData.DERIVED).resolve(CACHE), frKeys, k -> FeretData.decode(frImg.get(k).file()), 1.0, face,
                eye, "FERET");
        Map<String, List<String>> frG = new LinkedHashMap<>();
        frG.put("fa, fb (все)", frKeys);
        frG.put("fa", frKeys.stream().filter(k -> frImg.get(k).pose().equals("fa")).toList());
        frG.put("fb", frKeys.stream().filter(k -> frImg.get(k).pose().equals("fb")).toList());
        frG.put("в очках (a, c)", frKeys.stream().filter(k -> frImg.get(k).flag().equals("a") || frImg.get(k).flag().equals("c")).toList());

        Set own = new Set("своя база", ownH, ownY, ownG, k -> Imgcodecs.imread(ownFiles.get(k).toString(), Imgcodecs.IMREAD_COLOR), OwnEvaluation.INPUT_SCALE);
        Set gt = new Set("GT", gtH, gtY, gtG, k -> Imgcodecs.imread(gtDir.resolve(k + ".jpg").toString(), Imgcodecs.IMREAD_COLOR), 1.0);
        Set fei = new Set("FEI", feiH, feiY, feiG, k -> Imgcodecs.imread(feiFiles.get(k).toString(), Imgcodecs.IMREAD_COLOR), 1.0);
        Set feret = new Set("FERET", frH, frY, frG, k -> FeretData.decode(frImg.get(k).file()), 1.0);
        return new Bases(own, gt, fei, feret, frImg, ownG.get("снимки протокола"), fei1114, gtDir, feiDir, ownDir, feretDir);
    }
}
