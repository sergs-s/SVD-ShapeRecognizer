package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.storage.SettingsStore;

/**
 * Пункт (8) MediaPipe Face Landmarker, этап 1 (PLAN.md п. 5.5; план принят Жориком 16.10.2026, №7): точки и поза МП без
 * распознавания. Точки МП – кэш src/main/python/mp_points.py (Python отдельным процессом, без сети): FERET –
 * derived/mp_points.tsv, GT – mp_points.tsv рядом с каталогом gt_db. Здесь – только чтение кэша и счёт.
 * <p>
 * feret-mppts: (1) доля найденных лиц МП и YuNet на снимках fa/fb и на отобранных (YuNet, |r| ≤ R); (2) расхождение точек с
 * ручной разметкой на снимках feret-eyepts (2409) теми же мерами: глаза – МП (центры радужек 468, 473), МП (середины углов
 * глаз), Ю5, 0б (строки Ю5 и 0б сверяются побайтно с reports/xbase_feret_eyepts.txt, иначе остановка); нос и рот (центр рта –
 * середина углов) – МП и Ю5; (3) парные разности «МП − Ю5», «МП − 0б» с бутстрепом по людям и критерий перехода к этапу 2;
 * (4) yaw МП (из матрицы преобразования лица) и r YuNet против номинального угла позы из recordings.xml на всех позах FERET
 * (YuNet для поз кроме fa/fb – отдельный кэш derived/detections_allpose.tsv); (5) справочно – eyeBlink; (6) сверка
 * раскодирования OpenCV 5.0 (Python) и нашей сборки 4.9 по PNG из derived/mp_decode. Отчёт – только
 * reports/xbase_feret_mppts.txt в каталоге FERET.
 * <p>
 * gt-mppts: (1) обнаружение; расхождение МП с Ю5 (справочно, ручной разметки нет); yaw МП против r; eyeBlink; сверка
 * раскодирования (mp_decode рядом с mp_points.tsv). Отчёт – reports/faces/xbase/xbase_gt_mppts.txt.
 *
 * @author ssv
 */
final class XBaseMpPts {

    static final String FILE = "mp_points.tsv";
    static final String ALLPOSE = "detections_allpose.tsv";
    static final String DECODE = "mp_decode";
    /** Половины людей для перевода yaw / r в градусы: seed faces.seed + 9990. */
    static final long SPLIT_SEED_OFFSET = 9990;

    private XBaseMpPts() {
    }

    /** Строка кэша МП: найдено ли лицо, число лиц, точки по имени {x, y}, yaw, pitch, roll, eyeBlinkRight, eyeBlinkLeft. */
    record Mp(boolean found, int faces, Map<String, double[]> p, double yaw, double pitch, double roll, double blinkR, double blinkL) {

        double[] mid(String a, String b) {
            return FeretData.mid(p.get(a), p.get(b));
        }
    }

    /** Кэш МП: ключ → строка; шапка (строки «#») – в head. */
    static Map<String, Mp> read(Path file, List<String> head) throws IOException {
        if (!Files.exists(file)) throw new IllegalStateException("Нет кэша точек МП " + file + " – запустите src/main/python/mp_points.py");
        Map<String, Mp> out = new LinkedHashMap<>();
        List<String> cols = null;
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#")) {
                head.add(l);
                continue;
            }
            String[] c = l.split("\t", -1);
            if (cols == null) {
                cols = List.of(c);
                continue;
            }
            if (c.length != cols.size()) throw new IllegalStateException(file + ": неполная строка " + c[0]);
            boolean found = c[cols.indexOf("found")].equals("1");
            Map<String, double[]> p = new HashMap<>();
            double[] v = new double[5];
            if (found) {
                for (String nm : new String[] {"iris_re", "iris_le", "eye_re_out", "eye_re_in", "eye_le_in", "eye_le_out", "nose", "mouth_r", "mouth_l"})
                    p.put(nm, new double[] {Double.parseDouble(c[cols.indexOf(nm + "_x")]), Double.parseDouble(c[cols.indexOf(nm + "_y")])});
                String[] vn = {"yaw", "pitch", "roll", "eyeBlinkRight", "eyeBlinkLeft"};
                for (int i = 0; i < 5; i++) v[i] = Double.parseDouble(c[cols.indexOf(vn[i])]);
            }
            out.put(c[0], new Mp(found, Integer.parseInt(c[cols.indexOf("faces")]), p, v[0], v[1], v[2], v[3], v[4]));
        }
        return out;
    }

    // ---------------------------------------------------------------- статистика

    /** Ранги с усреднением равных (с 1). */
    static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
        double[] r = new double[v.length];
        for (int i = 0; i < v.length;) {
            int j = i;
            while (j + 1 < v.length && v[idx[j + 1]] == v[idx[i]]) j++;
            for (int k = i; k <= j; k++) r[idx[k]] = (i + j) / 2.0 + 1;
            i = j + 1;
        }
        return r;
    }

    static double pearson(double[] x, double[] y) {
        int n = x.length;
        double mx = Arrays.stream(x).sum() / n;
        double my = Arrays.stream(y).sum() / n;
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < n; i++) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
            syy += (y[i] - my) * (y[i] - my);
        }
        return sxy / Math.sqrt(sxx * syy);
    }

    /** Ранговая корреляция Спирмена (Пирсон по рангам с усреднением равных). */
    static double spearman(double[] x, double[] y) {
        return pearson(ranks(x), ranks(y));
    }

    /** AUC: вероятность, что значение из pos больше, чем из neg (равные – половина). */
    static double auc(double[] neg, double[] pos) {
        double s = 0;
        for (double p : pos) for (double q : neg) s += p > q ? 1 : p == q ? 0.5 : 0;
        return s / ((double) pos.length * neg.length);
    }

    /** Наименьшие квадраты y = a + b·x по индексам idx: {a, b}. */
    static double[] fit(double[] x, double[] y, List<Integer> idx) {
        double mx = 0;
        double my = 0;
        for (int i : idx) {
            mx += x[i];
            my += y[i];
        }
        mx /= idx.size();
        my /= idx.size();
        double sxy = 0;
        double sxx = 0;
        for (int i : idx) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
        }
        double b = sxy / sxx;
        return new double[] {my - b * mx, b};
    }

    /** Взвешенное среднее v по весам людей w[owner[i]]. */
    static double wmean(double[] v, int[] owner, int[] w) {
        double s = 0;
        long n = 0;
        for (int i = 0; i < v.length; i++) {
            s += v[i] * w[owner[i]];
            n += w[owner[i]];
        }
        return s / n;
    }

    /** Строка таблицы расхождений в формате feret-eyepts. */
    static String errLine(String group, String name, double[][] err, List<Integer> idx) {
        double[] mean = FeiBootstrap.sorted(idx.stream().mapToDouble(i -> err[i][2]).toArray());
        double[] eyes = FeiBootstrap.sorted(idx.stream().flatMapToDouble(i -> Arrays.stream(new double[] {err[i][0], err[i][1]})).toArray());
        return String.format(Locale.ROOT, "%s | %s | %d | %.4f | %.4f | %.4f | %.1f %% | %.1f %% | %.4f | %.1f %% | %.1f %%%n", group, name, idx.size(),
                FeiBootstrap.quantile(mean, 0.5), FeiBootstrap.quantile(mean, 0.95), mean[mean.length - 1], 100 * XBaseEyePts.share(mean, 0.05),
                100 * XBaseEyePts.share(mean, 0.10), FeiBootstrap.quantile(eyes, 0.5), 100 * XBaseEyePts.share(eyes, 0.05), 100 * XBaseEyePts.share(eyes, 0.10));
    }

    /** Строка одной точки (нос, рот): медиана, 95 %, макс, доли > 5 % и > 10 %. */
    static String oneLine(String point, String name, double[] e) {
        double[] s = FeiBootstrap.sorted(e);
        return String.format(Locale.ROOT, "%s | %s | %d | %.4f | %.4f | %.4f | %.1f %% | %.1f %%%n", point, name, s.length, FeiBootstrap.quantile(s, 0.5),
                FeiBootstrap.quantile(s, 0.95), s[s.length - 1], 100 * XBaseEyePts.share(s, 0.05), 100 * XBaseEyePts.share(s, 0.10));
    }

    /**
     * Парная разность «c − y» по снимкам (медиана, доли > 5 % и > 10 %) с бутстрепом по людям; {строка, верхняя граница
     * разности медиан}.
     */
    static Object[] pairDiff(String name, double[] c, double[] y, int[] owner, int[][] wp) {
        int n = c.length;
        int[] unit = new int[n];
        Arrays.fill(unit, 1);
        double[] point = {XBaseEyePts.weightedQuantile(c, unit, 0.5) - XBaseEyePts.weightedQuantile(y, unit, 0.5),
                XBaseEyePts.weightedShare(c, unit, 0.05) - XBaseEyePts.weightedShare(y, unit, 0.05),
                XBaseEyePts.weightedShare(c, unit, 0.10) - XBaseEyePts.weightedShare(y, unit, 0.10)};
        double[][] rep = new double[3][wp.length];
        for (int q = 0; q < wp.length; q++) {
            int[] wi = new int[n];
            for (int i = 0; i < n; i++) wi[i] = wp[q][owner[i]];
            rep[0][q] = XBaseEyePts.weightedQuantile(c, wi, 0.5) - XBaseEyePts.weightedQuantile(y, wi, 0.5);
            rep[1][q] = XBaseEyePts.weightedShare(c, wi, 0.05) - XBaseEyePts.weightedShare(y, wi, 0.05);
            rep[2][q] = XBaseEyePts.weightedShare(c, wi, 0.10) - XBaseEyePts.weightedShare(y, wi, 0.10);
        }
        double hi = wp.length == 0 ? point[0] : FeiBootstrap.quantile(FeiBootstrap.sorted(rep[0]), 0.975);
        String line = String.join(" | ", name, XBaseEyePts.diff(point[0], rep[0], 1, "%+.4f"), XBaseEyePts.diff(point[1], rep[1], 100, "%+.1f"),
                XBaseEyePts.diff(point[2], rep[2], 100, "%+.1f")) + "\n";
        return new Object[] {line, hi};
    }

    static double dist(double[] a, double[] b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    static String quantiles(String name, double[] v) {
        double[] s = FeiBootstrap.sorted(v);
        return String.format(Locale.ROOT, "%s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f %%%n", name, s.length, FeiBootstrap.quantile(s, 0.05),
                FeiBootstrap.quantile(s, 0.25), FeiBootstrap.quantile(s, 0.5), FeiBootstrap.quantile(s, 0.75), FeiBootstrap.quantile(s, 0.95),
                FeiBootstrap.quantile(s, 0.99), 100 * XBaseEyePts.share(s, 0.5));
    }

    /** eyeBlink по снимкам с найденным лицом МП. */
    static void blink(StringBuilder r, String title, List<Mp> list) {
        r.append("\n=== ").append(title).append(" ===\n");
        r.append("величина | снимков | 5 % | 25 % | 50 % | 75 % | 95 % | 99 % | доля > 0,5\n");
        r.append(quantiles("eyeBlinkRight", list.stream().mapToDouble(Mp::blinkR).toArray()));
        r.append(quantiles("eyeBlinkLeft", list.stream().mapToDouble(Mp::blinkL).toArray()));
        r.append(quantiles("наибольший из двух", list.stream().mapToDouble(m -> Math.max(m.blinkR(), m.blinkL())).toArray()));
    }

    /**
     * Сверка раскодирования: PNG из каталога dump (OpenCV 5.0 в Python) против раскодирования нашей сборкой; строка отчёта.
     */
    static String decodeCheck(Path dump, Map<String, Path> files, boolean feret) throws IOException {
        if (!Files.isDirectory(dump)) return "Сверка раскодирования: нет каталога " + dump + " – не проверено.\n";
        List<Path> pngs;
        try (Stream<Path> s = Files.list(dump)) {
            pngs = s.filter(p -> p.toString().endsWith(".png")).sorted().toList();
        }
        int same = 0;
        double maxDiff = 0;
        long diffPx = 0;
        long totalPx = 0;
        for (Path png : pngs) {
            String name = png.getFileName().toString().replace(".png", "");
            String key = feret ? name : name.replace('_', '/');
            Path src = files.get(key);
            if (src == null) throw new IllegalStateException("Сверка раскодирования: нет снимка для " + png);
            Mat ours = feret ? FeretData.decode(src) : Imgcodecs.imread(src.toString(), Imgcodecs.IMREAD_COLOR);
            Mat py = Imgcodecs.imread(png.toString(), Imgcodecs.IMREAD_COLOR);
            if (ours.size().width != py.size().width || ours.size().height != py.size().height)
                throw new IllegalStateException("Сверка раскодирования: разный размер " + key);
            Mat d = new Mat();
            Core.absdiff(ours, py, d);
            Mat d1 = d.reshape(1);
            double mx = Core.minMaxLoc(d1).maxVal;
            int nz = Core.countNonZero(d1);
            diffPx += nz;
            totalPx += d1.total();
            if (mx == 0) same++;
            maxDiff = Math.max(maxDiff, mx);
            ours.release();
            py.release();
            d.release();
        }
        return String.format(Locale.ROOT, "Сверка раскодирования (Python OpenCV 5.0 против Java OpenCV 4.9, %d снимков): совпали побайтно %d; наибольшее "
                + "расхождение %.0f уровня яркости, различающихся значений %d из %d.%n", pngs.size(), same, maxDiff, diffPx, totalPx);
    }

    // ---------------------------------------------------------------- FERET

    /** Номинальный yaw позы по recordings.xml: ключ снимка → угол, градусы; поза – из имени. */
    static Map<String, Double> nominalYaw(Path dir) throws IOException {
        Map<String, Double> out = new HashMap<>();
        Pattern url = Pattern.compile("relative=\"[^\"]*/([^/\"]+)\\.ppm\\.bz2\"");
        Pattern pose = Pattern.compile("<Pose name=\"[a-z]+\" yaw=\"([-0-9.]+)\"");
        String key = null;
        for (String l : Files.readAllLines(dir.resolve("dvd1/data/ground_truths/xml/recordings.xml"), StandardCharsets.UTF_8)) {
            Matcher m = url.matcher(l);
            if (m.find()) key = m.group(1);
            Matcher p = pose.matcher(l);
            if (p.find() && key != null) {
                out.put(key, Double.parseDouble(p.group(1)));
                key = null;
            }
        }
        return out;
    }

    /** Все снимки FERET (все позы) по ключу. */
    static Map<String, Path> allFiles(Path dir) throws IOException {
        Map<String, Path> out = new TreeMap<>();
        for (String dvd : new String[] {"dvd1", "dvd2"}) {
            try (Stream<Path> s = Files.walk(dir.resolve(dvd).resolve("data").resolve("images"))) {
                s.filter(p -> p.getFileName().toString().endsWith(".ppm.bz2")).forEach(p -> out.put(p.getFileName().toString().replace(".ppm.bz2", ""), p));
            }
        }
        return out;
    }

    /**
     * YuNet для снимков, которых нет в detections.tsv (позы кроме fa/fb): тот же детектор (0,7, два прохода), без кадров;
     * кэш derived/detections_allpose.tsv (дозапись, готовые пропускаются). Ключ → строка (null – не найдено).
     */
    static Map<String, double[]> detectAllPose(Map<String, Path> files, Map<String, double[]> frontal, Path dir, Path models) throws IOException {
        Path file = dir.resolve(FeretData.DERIVED).resolve(ALLPOSE);
        Map<String, double[]> done = new HashMap<>(frontal);
        if (Files.exists(file)) {
            for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                String[] c = l.split("\t", -1);
                if (c.length != 17) continue;
                double[] row = null;
                if (c[1].equals("1")) {
                    row = new double[15];
                    for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[2 + i]);
                }
                done.put(c[0], row);
            }
        }
        FaceDetectorYN detector = OwnDetectorThreshold.create(models, FeretData.SCORE);
        int n = 0;
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            for (Map.Entry<String, Path> e : files.entrySet()) {
                if (done.containsKey(e.getKey())) continue;
                Mat color = FeretData.decode(e.getValue());
                double[] first = YunetTwoPass.firstPass(detector, color, 1.0);
                double[] row = null;
                if (first != null) {
                    double[] second = YunetTwoPass.secondPass(detector, color, new double[] {first[0], first[1], first[2], first[3]});
                    row = second != null ? second : first;
                }
                color.release();
                StringBuilder l = new StringBuilder(e.getKey()).append('\t').append(row == null ? "0" : "1");
                for (int i = 0; i < 15; i++) l.append('\t').append(row == null ? "" : Double.toString(row[i]));
                w.write(l.append('\n').toString());
                w.flush();
                done.put(e.getKey(), row);
                if (++n % 500 == 0) System.out.printf(Locale.ROOT, "feret-mppts: YuNet по всем позам – %d новых%n", n);
            }
        }
        return done;
    }

    static void feret(List<String> selLines, SettingsStore settings) throws IOException {
        long t0 = System.nanoTime();
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-mppts: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        long seed = settings.loadFacesSeed();
        int b = settings.loadFacesFeiBootstrap();
        double rLimit = XBaseEval.poseLimit(selLines);
        Path models = Paths.get(settings.loadFacesModelsDir());
        List<String> head = new ArrayList<>();
        Map<String, Mp> mp = read(dir.resolve(FeretData.DERIVED).resolve(FILE), head);
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        List<Img> all = FeretData.frontal(dir);
        Map<String, double[]> det = FeretData.detect(all, dir, models, sface);
        Map<String, Path> files = allFiles(dir);
        for (String k : files.keySet()) if (!mp.containsKey(k)) throw new IllegalStateException("feret-mppts: нет точек МП для " + k + " – дозапустите mp_points.py");

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (8) MediaPipe Face Landmarker, этап 1, FERET (XBaseEval feret-mppts): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (детекции YuNet — кэш derived/detections.tsv и derived/").append(ALLPOSE).append("; точки МП — derived/")
                .append(FILE).append(")\n");
        for (String h : head) r.append("Кэш МП: ").append(h.substring(1).trim()).append('\n');
        r.append("Точки МП: глаза – центры радужек 468 (на снимке слева, правый глаз человека) и 473; «МП, углы» – середины углов глаз 33–133 и "
                + "362–263; нос – 1; рот – середина углов 61 и 291 (в ручной разметке FERET – одна точка рта). Сторона индексов проверена глазами "
                + "на листе 24 снимков (reports/mp/mp_sides_sheet.png).\n");
        r.append(decodeCheck(dir.resolve(FeretData.DERIVED).resolve(DECODE), files, true));

        // (1) Обнаружение.
        r.append("\n=== Обнаружение лица: YuNet (0,7, два прохода) и МП (до 2 лиц, берётся наибольшее) ===\n");
        r.append("набор | снимков | YuNet | МП | оба | только YuNet | только МП | ни один | МП: лиц > 1\n");
        List<String> fab = all.stream().map(Img::key).toList();
        List<String> sel = new ArrayList<>();
        for (Img im : all) {
            double[] row = det.get(im.key());
            if (row != null && Math.abs(FeiDataset.pose(row)) <= rLimit) sel.add(im.key());
        }
        Map<String, double[]> yAll = detectAllPose(files, det, dir, models);
        r.append(detLine("fa, fb (все)", fab, k -> det.get(k) != null, mp));
        r.append(detLine("отобранные (YuNet, |r| ≤ R)", sel, k -> true, mp));
        r.append(detLine("все позы", new ArrayList<>(files.keySet()), k -> yAll.get(k) != null, mp));

        // (2) Расхождение с ручной разметкой – снимки feret-eyepts.
        List<Img> imgs = new ArrayList<>();
        for (Img im : all) {
            double[] row = det.get(im.key());
            if (im.leftEye() == null || im.rightEye() == null || row == null) continue;
            if (Math.abs(FeiDataset.pose(row)) > rLimit) continue;
            imgs.add(im);
        }
        int n = imgs.size();
        CascadeClassifier tree = new CascadeClassifier(models.resolve(XBaseEyePts.EYE_TREE).toString());
        CascadeClassifier eye = new CascadeClassifier(models.resolve(XBaseEyePts.EYE).toString());
        if (tree.empty() || eye.empty()) throw new IllegalStateException("feret-mppts: каскады глаз не загружены из " + models);
        double[][] eY = new double[n][];
        double[][] e0b = new double[n][];
        double[][] eMp = new double[n][];
        double[][] eMpC = new double[n][];
        for (int i = 0; i < n; i++) {
            Img im = imgs.get(i);
            double[] row = det.get(im.key());
            Mat color = FeretData.decode(im.file());
            Mat g = EyeRefine.gray(color);
            color.release();
            double[] re = {row[4], row[5]};
            double[] le = {row[6], row[7]};
            double iod = Math.hypot(le[0] - re[0], le[1] - re[1]);
            double[] r2 = XBaseEyePts.refine(2, g, re, iod, tree, eye);
            double[] l2 = XBaseEyePts.refine(2, g, le, iod, tree, eye);
            g.release();
            double[] e = FeretData.eyeError(im, re, le);
            eY[i] = new double[] {e[1], e[0], e[2]};
            e = FeretData.eyeError(im, r2 == null ? re : r2, l2 == null ? le : l2);
            e0b[i] = new double[] {e[1], e[0], e[2]};
            Mp m = mp.get(im.key());
            if (m.found()) {
                e = FeretData.eyeError(im, m.p().get("iris_re"), m.p().get("iris_le"));
                eMp[i] = new double[] {e[1], e[0], e[2]};
                e = FeretData.eyeError(im, m.mid("eye_re_out", "eye_re_in"), m.mid("eye_le_in", "eye_le_out"));
                eMpC[i] = new double[] {e[1], e[0], e[2]};
            }
            if ((i + 1) % 500 == 0) System.out.printf(Locale.ROOT, "feret-mppts: 0б %d / %d%n", i + 1, n);
        }
        List<String> persons = new ArrayList<>(new TreeSet<>(imgs.stream().map(Img::person).toList()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        int[][] wp = b > 0 ? FeiBootstrap.weights(persons.size(), b, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        List<Integer> mpIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) if (eMp[i] != null) mpIdx.add(i);

        r.append(String.format(Locale.ROOT, "%nСнимки сравнения с ручной разметкой — как feret-eyepts: %d (размечены, лицо YuNet найдено, |r| ≤ R = %.4f), людей %d; "
                + "лицо МП найдено на %d.%n", n, rLimit, persons.size(), mpIdx.size()));
        if (b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", b,
                seed + FeiBootstrap.SEED_OFFSET));
        Map<String, Predicate<Img>> groups = new LinkedHashMap<>();
        groups.put("все", im -> true);
        groups.put("без очков", im -> !(im.flag().equals("a") || im.flag().equals("c")));
        groups.put("в очках (a, c)", im -> im.flag().equals("a") || im.flag().equals("c"));
        groups.put("fa", im -> im.pose().equals("fa"));
        groups.put("fb", im -> im.pose().equals("fb"));
        String eyepts = Files.readString(dir.resolve("reports").resolve("xbase_feret_eyepts.txt"), StandardCharsets.UTF_8).replace("\r", "");
        int checked = 0;
        r.append("\n=== Глаза: расхождение с ручной разметкой (доли ручного межглазья): среднее по двум глазам на снимок; глаза — пулом ===\n");
        r.append("группа | кандидат | снимков | среднее: медиана | 95 % | макс | > 5 % | > 10 % | глаза: медиана | > 5 % | > 10 %\n");
        boolean allMp = mpIdx.size() == n;
        for (Map.Entry<String, Predicate<Img>> g : groups.entrySet()) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) if (g.getValue().test(imgs.get(i))) idx.add(i);
            if (idx.isEmpty()) continue;
            String ly = errLine(g.getKey(), "YuNet", eY, idx);
            String l0 = errLine(g.getKey(), "0б", e0b, idx);
            for (String l : new String[] {ly, l0}) {
                if (!eyepts.contains(l.replace("\r", ""))) throw new IllegalStateException("feret-mppts: строка не совпала с xbase_feret_eyepts.txt: " + l);
                checked++;
            }
            List<Integer> idxMp = idx.stream().filter(i -> eMp[i] != null).toList();
            r.append(ly.replace(" | YuNet | ", " | Ю5 | ")).append(l0);
            if (!allMp) {
                r.append(errLine(g.getKey(), "Ю5 (где МП нашёл)", eY, idxMp));
                r.append(errLine(g.getKey(), "0б (где МП нашёл)", e0b, idxMp));
            }
            r.append(errLine(g.getKey(), "МП, радужки", eMp, idxMp));
            r.append(errLine(g.getKey(), "МП, углы", eMpC, idxMp));
        }
        r.append(String.format(Locale.ROOT, "Сверка: строки Ю5 (YuNet) и 0б — %d строк — побайтно совпали с reports/xbase_feret_eyepts.txt.%n", checked));

        // Нос и рот.
        List<Integer> nmIdx = new ArrayList<>();
        for (int i : mpIdx) if (imgs.get(i).nose() != null && imgs.get(i).mouth() != null) nmIdx.add(i);
        double[][] nm = new double[4][nmIdx.size()];
        for (int j = 0; j < nmIdx.size(); j++) {
            Img im = imgs.get(nmIdx.get(j));
            double[] row = det.get(im.key());
            double[][] y4 = FeretData.yunet4(row);
            double iod = dist(im.leftEye(), im.rightEye());
            Mp m = mp.get(im.key());
            nm[0][j] = dist(y4[2], im.nose()) / iod;
            nm[1][j] = dist(m.p().get("nose"), im.nose()) / iod;
            nm[2][j] = dist(y4[3], im.mouth()) / iod;
            nm[3][j] = dist(m.mid("mouth_r", "mouth_l"), im.mouth()) / iod;
        }
        r.append("\n=== Нос и рот: расхождение с ручной разметкой (доли ручного межглазья; рот — середина углов против ручной точки рта) ===\n");
        r.append("точка | кандидат | снимков | медиана | 95 % | макс | > 5 % | > 10 %\n");
        r.append(oneLine("нос", "Ю5", nm[0])).append(oneLine("нос", "МП", nm[1])).append(oneLine("рот", "Ю5", nm[2])).append(oneLine("рот", "МП", nm[3]));

        // (3) Парные разности и критерий.
        int[] owner = mpIdx.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray();
        double[] mpE = mpIdx.stream().mapToDouble(i -> eMp[i][2]).toArray();
        double[] mpC = mpIdx.stream().mapToDouble(i -> eMpC[i][2]).toArray();
        double[] yE = mpIdx.stream().mapToDouble(i -> eY[i][2]).toArray();
        double[] bE = mpIdx.stream().mapToDouble(i -> e0b[i][2]).toArray();
        r.append("\n=== Парные разности (снимки, где МП нашёл лицо; глаза — среднее по двум на снимок), бутстреп по людям ===\n");
        r.append("разность | медиана, доли межглазья | доля > 5 %, п. п. | доля > 10 %, п. п.\n");
        Object[] dY = pairDiff("глаза: МП, радужки − Ю5", mpE, yE, owner, wp);
        Object[] dB = pairDiff("глаза: МП, радужки − 0б", mpE, bE, owner, wp);
        r.append((String) dY[0]).append((String) dB[0]);
        r.append((String) pairDiff("глаза: МП, углы − Ю5", mpC, yE, owner, wp)[0]);
        r.append((String) pairDiff("глаза: МП, углы − 0б", mpC, bE, owner, wp)[0]);
        int[] ownerNm = nmIdx.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray();
        r.append((String) pairDiff("нос: МП − Ю5", nm[1], nm[0], ownerNm, wp)[0]);
        r.append((String) pairDiff("рот: МП − Ю5", nm[3], nm[2], ownerNm, wp)[0]);
        boolean eyesPass = (double) dY[1] < 0 && (double) dB[1] < 0;
        r.append(String.format(Locale.ROOT, "Критерий (а) перехода к этапу 2 — глаза МП значимо ближе к ручной разметке, чем Ю5 и 0б (верхняя граница "
                + "разности медиан < 0 для обоих): %s (верхние границы %+.4f и %+.4f).%n", eyesPass ? "выполнен" : "не выполнен", (double) dY[1], (double) dB[1]));

        // (4) Yaw.
        Map<String, Double> nominal = nominalYaw(dir);
        yaw(r, files.keySet(), nominal, yAll, mp, seed, b);

        // (5) eyeBlink.
        blink(r, "Справочно: eyeBlink МП на отобранных снимках fa/fb (без вывода о пользе)",
                sel.stream().map(mp::get).filter(Mp::found).toList());

        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        Path out = dir.resolve("reports");
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_feret_mppts.txt"), r);
        System.out.print(r);
    }

    static String detLine(String name, List<String> keys, Predicate<String> yunet, Map<String, Mp> mp) {
        int y = 0;
        int m = 0;
        int both = 0;
        int multi = 0;
        for (String k : keys) {
            boolean a = yunet.test(k);
            boolean c = mp.get(k).found();
            if (a) y++;
            if (c) m++;
            if (a && c) both++;
            if (mp.get(k).faces() > 1) multi++;
        }
        int n = keys.size();
        return String.format(Locale.ROOT, "%s | %d | %d (%.1f %%) | %d (%.1f %%) | %d | %d | %d | %d | %d%n", name, n, y, 100.0 * y / n, m, 100.0 * m / n, both,
                y - both, m - both, n - y - m + both, multi);
    }

    /** Yaw МП и r YuNet против номинального угла позы FERET (снимки, где оба нашли лицо). */
    static void yaw(StringBuilder r, java.util.Set<String> keys, Map<String, Double> nominal, Map<String, double[]> yAll, Map<String, Mp> mp, long seed,
                    int b) {
        List<String> ks = new ArrayList<>();
        for (String k : keys) if (nominal.containsKey(k) && yAll.get(k) != null && mp.get(k).found()) ks.add(k);
        int n = ks.size();
        double[] nom = new double[n];
        double[] ym = new double[n];
        double[] rr = new double[n];
        String[] person = new String[n];
        for (int i = 0; i < n; i++) {
            String k = ks.get(i);
            nom[i] = nominal.get(k);
            ym[i] = mp.get(k).yaw();
            rr[i] = FeiDataset.pose(yAll.get(k));
            person[i] = k.substring(0, k.indexOf('_'));
        }
        r.append("\n=== Поворот: yaw МП (матрица преобразования лица, градусы) и r YuNet против номинального угла позы (recordings.xml) ===\n");
        r.append(String.format(Locale.ROOT, "Снимки всех поз, где оба нашли лицо: %d из %d с номинальным углом.%n", n, nominal.size()));
        r.append("поза | номинал | снимков | YuNet нашёл | МП нашёл | оба | yaw МП: медиана [25 %; 75 %] | r: медиана [25 %; 75 %]\n");
        Map<String, List<Integer>> byPose = new TreeMap<>();
        for (int i = 0; i < n; i++) byPose.computeIfAbsent(ks.get(i).split("_")[2], x -> new ArrayList<>()).add(i);
        Map<String, int[]> found = new TreeMap<>();
        for (String k : nominal.keySet()) {
            int[] c = found.computeIfAbsent(k.split("_")[2], x -> new int[3]);
            c[0]++;
            if (yAll.get(k) != null) c[1]++;
            if (mp.containsKey(k) && mp.get(k).found()) c[2]++;
        }
        for (Map.Entry<String, int[]> e : found.entrySet()) {
            List<Integer> idx = byPose.getOrDefault(e.getKey(), List.of());
            double nv = nominal.entrySet().stream().filter(x -> x.getKey().split("_")[2].equals(e.getKey())).findFirst().get().getValue();
            String q = "–";
            String qr = "–";
            if (!idx.isEmpty()) {
                double[] a = FeiBootstrap.sorted(idx.stream().mapToDouble(i -> ym[i]).toArray());
                double[] c = FeiBootstrap.sorted(idx.stream().mapToDouble(i -> rr[i]).toArray());
                q = String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]", FeiBootstrap.quantile(a, 0.5), FeiBootstrap.quantile(a, 0.25), FeiBootstrap.quantile(a, 0.75));
                qr = String.format(Locale.ROOT, "%+.3f [%+.3f; %+.3f]", FeiBootstrap.quantile(c, 0.5), FeiBootstrap.quantile(c, 0.25), FeiBootstrap.quantile(c, 0.75));
            }
            r.append(String.format(Locale.ROOT, "%s | %+.1f | %d | %.1f %% | %.1f %% | %d | %s | %s%n", e.getKey(), nv, e.getValue()[0], 100.0 * e.getValue()[1] / e.getValue()[0],
                    100.0 * e.getValue()[2] / e.getValue()[0], idx.size(), q, qr));
        }

        // Сводные меры на двух наборах: |номинал| ≤ 22,5° (диапазон отбора) и все позы.
        List<String> pp = new ArrayList<>(new TreeSet<>(Arrays.asList(person)));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < pp.size(); i++) pIdx.put(pp.get(i), i);
        List<String> shuffled = new ArrayList<>(pp);
        Collections.shuffle(shuffled, new Random(seed + SPLIT_SEED_OFFSET));
        java.util.Set<String> halfA = new java.util.HashSet<>(shuffled.subList(0, shuffled.size() / 2));
        int[][] wp = b > 0 ? FeiBootstrap.weights(pp.size(), b, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        r.append("\nМеры: Спирмен с номиналом (со знаком); AUC «анфас (0°) против ±15°» и «анфас против ±22,5°» по модулю; ошибка (MAE, градусы) после "
                + "линейного перевода в градусы — подбор на одной половине людей, проверка на другой (seed ").append(seed + SPLIT_SEED_OFFSET)
                .append("), обе половины; разность MAE «МП − r» — бутстреп по людям.\n");
        r.append("набор | снимков | Спирмен МП | Спирмен r | AUC 0/15 МП | AUC 0/15 r | AUC 0/22,5 МП | AUC 0/22,5 r | MAE МП | MAE r | MAE МП − r\n");
        for (double lim : new double[] {22.5, 90}) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) if (Math.abs(nom[i]) <= lim) idx.add(i);
            double[] nn = idx.stream().mapToDouble(i -> nom[i]).toArray();
            double[] a = idx.stream().mapToDouble(i -> ym[i]).toArray();
            double[] c = idx.stream().mapToDouble(i -> rr[i]).toArray();
            int[] owner = idx.stream().mapToInt(i -> pIdx.get(person[i])).toArray();
            double[] errM = outOfSample(a, nn, idx, person, halfA);
            double[] errR = outOfSample(c, nn, idx, person, halfA);
            double[] rep = new double[wp.length];
            for (int q = 0; q < wp.length; q++) rep[q] = wmean(errM, owner, wp[q]) - wmean(errR, owner, wp[q]);
            int[] unit = new int[pp.size()];
            Arrays.fill(unit, 1);
            r.append(String.format(Locale.ROOT, "%s | %d | %+.3f | %+.3f | %s | %s | %s | %s | %.2f | %.2f | %s%n", lim == 90 ? "все позы" : "|номинал| ≤ 22,5°",
                    idx.size(), spearman(nn, a), spearman(nn, c), aucAbs(nn, a, 15), aucAbs(nn, c, 15), aucAbs(nn, a, 22.5), aucAbs(nn, c, 22.5),
                    wmean(errM, owner, unit), wmean(errR, owner, unit), XBaseEyePts.diff(wmean(errM, owner, unit) - wmean(errR, owner, unit), rep, 1, "%+.2f")));
        }
    }

    /** AUC по модулю: снимки с номиналом 0 против |номинал| = deg. */
    static String aucAbs(double[] nom, double[] v, double deg) {
        List<Double> neg = new ArrayList<>();
        List<Double> pos = new ArrayList<>();
        for (int i = 0; i < nom.length; i++) {
            if (nom[i] == 0) neg.add(Math.abs(v[i]));
            else if (Math.abs(nom[i]) == deg) pos.add(Math.abs(v[i]));
        }
        if (neg.isEmpty() || pos.isEmpty()) return "–";
        return String.format(Locale.ROOT, "%.3f", auc(neg.stream().mapToDouble(x -> x).toArray(), pos.stream().mapToDouble(x -> x).toArray()));
    }

    /** |номинал − (a + b·x)| для каждого снимка; a, b подобраны на другой половине людей. */
    static double[] outOfSample(double[] x, double[] nom, List<Integer> idx, String[] person, java.util.Set<String> halfA) {
        List<Integer> inA = new ArrayList<>();
        List<Integer> inB = new ArrayList<>();
        for (int j = 0; j < idx.size(); j++) (halfA.contains(person[idx.get(j)]) ? inA : inB).add(j);
        double[] fa = fit(x, nom, inA);
        double[] fb = fit(x, nom, inB);
        double[] e = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            double[] f = halfA.contains(person[idx.get(j)]) ? fb : fa;
            e[j] = Math.abs(nom[j] - (f[0] + f[1] * x[j]));
        }
        return e;
    }

    // ---------------------------------------------------------------- GT

    static void gt(Path exportDir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        String raw = settings.loadFacesGtDir();
        if (raw == null) throw new IllegalStateException("XBaseEval gt-mppts: задайте faces.gt.dir / FACES_GT_DIR");
        Path gtDir = Paths.get(raw);
        double rLimit = XBaseEval.poseLimit(selLines);
        List<String> head = new ArrayList<>();
        Map<String, Mp> mp = read(gtDir.getParent().resolve(FILE), head);
        Map<String, double[]> det = new TreeMap<>();
        for (String l : Files.readAllLines(exportDir.resolve("detections.tsv"), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (!c[0].equals("gt")) continue;
            double[] row = null;
            if (c[4].equals("1")) {
                row = new double[15];
                for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[5 + i]);
            }
            det.put(c[1] + "/" + c[2], row);
        }
        Map<String, Path> files = new TreeMap<>();
        for (String k : det.keySet()) files.put(k, gtDir.resolve(k + ".jpg"));
        for (String k : det.keySet()) if (!mp.containsKey(k)) throw new IllegalStateException("gt-mppts: нет точек МП для " + k);

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (8) MediaPipe Face Landmarker, этап 1, GT (XBaseEval gt-mppts)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(gtDir).append(" (снимки); YuNet — ").append(exportDir.resolve("detections.tsv")).append("; точки МП — ")
                .append(gtDir.getParent().resolve(FILE)).append('\n');
        for (String h : head) r.append("Кэш МП: ").append(h.substring(1).trim()).append('\n');
        r.append("Ручной разметки на GT нет: расхождение МП с Ю5 и yaw МП против r — справочно.\n");
        r.append(decodeCheck(gtDir.getParent().resolve(DECODE), files, false));
        List<String> all = new ArrayList<>(det.keySet());
        List<String> sel = all.stream().filter(k -> det.get(k) != null && Math.abs(FeiDataset.pose(det.get(k))) <= rLimit).toList();
        r.append("\n=== Обнаружение лица: YuNet (экспорт, 0,7) и МП (до 2 лиц, берётся наибольшее) ===\n");
        r.append("набор | снимков | YuNet | МП | оба | только YuNet | только МП | ни один | МП: лиц > 1\n");
        r.append(detLine("все", all, k -> det.get(k) != null, mp));
        r.append(detLine(String.format(Locale.ROOT, "отобранные (YuNet, |r| ≤ R = %.4f)", rLimit), sel, k -> true, mp));

        List<String> both = sel.stream().filter(k -> mp.get(k).found()).toList();
        double[][] d = new double[4][both.size()];
        for (int j = 0; j < both.size(); j++) {
            double[] row = det.get(both.get(j));
            Mp m = mp.get(both.get(j));
            double iod = Math.hypot(row[6] - row[4], row[7] - row[5]);
            d[0][j] = dist(m.p().get("iris_re"), new double[] {row[4], row[5]}) / iod;
            d[1][j] = dist(m.p().get("iris_le"), new double[] {row[6], row[7]}) / iod;
            d[2][j] = dist(m.p().get("nose"), new double[] {row[8], row[9]}) / iod;
            d[3][j] = dist(m.mid("mouth_r", "mouth_l"), FeretData.mid(new double[] {row[10], row[11]}, new double[] {row[12], row[13]})) / iod;
        }
        r.append("\n=== Справочно: расстояние точек МП от Ю5 на отобранных (доли межглазья YuNet) ===\n");
        r.append("точка | кандидат | снимков | медиана | 95 % | макс | > 5 % | > 10 %\n");
        r.append(oneLine("глаз слева на снимке", "МП − Ю5", d[0])).append(oneLine("глаз справа", "МП − Ю5", d[1])).append(oneLine("нос", "МП − Ю5", d[2]))
                .append(oneLine("рот (середина углов)", "МП − Ю5", d[3]));

        r.append("\n=== Поворот: yaw МП против r YuNet (номинального угла нет) ===\n");
        r.append("набор | снимков | Спирмен (со знаком) | Спирмен по модулю\n");
        for (Map.Entry<String, List<String>> e : Map.of("все", all, "отобранные", sel).entrySet().stream().sorted(Map.Entry.comparingByKey()).toList()) {
            List<String> ks = e.getValue().stream().filter(k -> det.get(k) != null && mp.get(k).found()).toList();
            double[] a = ks.stream().mapToDouble(k -> mp.get(k).yaw()).toArray();
            double[] c = ks.stream().mapToDouble(k -> FeiDataset.pose(det.get(k))).toArray();
            r.append(String.format(Locale.ROOT, "%s | %d | %+.3f | %+.3f%n", e.getKey(), ks.size(), spearman(a, c),
                    spearman(Arrays.stream(a).map(Math::abs).toArray(), Arrays.stream(c).map(Math::abs).toArray())));
        }
        blink(r, "Справочно: eyeBlink МП на отобранных снимках GT (без вывода о пользе)", sel.stream().map(mp::get).filter(Mp::found).toList());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_gt_mppts.txt"), r);
        System.out.print(r);
    }
}
