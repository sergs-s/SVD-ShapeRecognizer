package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.core.Mat;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.storage.SettingsStore;

/**
 * Уточнение точек глаз на FERET, этап 1 (PLAN.md п. 5.5, (а2б); план подтверждён Хозяином 12.10.2026): расхождение точек
 * глаз кандидатов с ручной разметкой в долях ручного межглазья, без распознавания. Снимки — как feret-eyes (размеченные, лицо
 * найдено, |r| ≤ R). Кандидаты (EyeRefine): YuNet (как есть), 0а — градиенты, 0б — каскад haarcascade_eye_tree_eyeglasses,
 * 0б' — каскад haarcascade_eye (справочно), 0в — каскад, затем градиенты (справочно); каскад не нашёл глаз — точка YuNet
 * (число таких случаев — в отчёт). Межглазье для областей — по точкам YuNet. Бутстреп по людям (разность медиан «кандидат −
 * YuNet»). Порог перехода к этапу 2 (заранее): медиана ≤ 0,8 медианы YuNet и доля &gt; 10 % не выше, чем у YuNet. Отчёт —
 * только reports/xbase_feret_eyepts.txt, точки — derived/eyepts.tsv в каталоге базы.
 *
 * @author ssv
 */
final class XBaseEyePts {

    static final String[] NAMES = {"YuNet", "0а", "0б", "0б'", "0в"};
    static final String EYE_TREE = "haarcascade_eye_tree_eyeglasses.xml";
    static final String EYE = "haarcascade_eye.xml";
    static final double GATE_MEDIAN = 0.8;

    private XBaseEyePts() {
    }

    /** Квантиль q по значениям v с целыми весами w (нижний: первое значение, где накопленный вес ≥ q·сумма). */
    static double weightedQuantile(double[] v, int[] w, double q) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
        long total = 0;
        for (int x : w) total += x;
        if (total == 0) return Double.NaN;
        double need = q * total;
        long cum = 0;
        for (int i : idx) {
            cum += w[i];
            if (cum >= need && w[i] > 0) return v[i];
        }
        return v[idx[idx.length - 1]];
    }

    /** Взвешенная доля значений больше t. */
    static double weightedShare(double[] v, int[] w, double t) {
        long n = 0;
        long x = 0;
        for (int i = 0; i < v.length; i++) {
            n += w[i];
            if (v[i] > t) x += w[i];
        }
        return n == 0 ? Double.NaN : (double) x / n;
    }

    static void run(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-eyepts: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        long seed = settings.loadFacesSeed();
        int b = settings.loadFacesFeiBootstrap();
        double rLimit = XBaseEval.poseLimit(selLines);
        Path models = Paths.get(settings.loadFacesModelsDir());
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        List<Img> all = FeretData.frontal(dir);
        Map<String, double[]> det = FeretData.detect(all, dir, models, sface);
        CascadeClassifier tree = new CascadeClassifier(models.resolve(EYE_TREE).toString());
        CascadeClassifier eye = new CascadeClassifier(models.resolve(EYE).toString());
        if (tree.empty() || eye.empty()) throw new IllegalStateException("feret-eyepts: каскады глаз не загружены из " + models);

        List<Img> imgs = new ArrayList<>();
        for (Img im : all) {
            double[] row = det.get(im.key());
            if (im.leftEye() == null || im.rightEye() == null || row == null) continue;
            if (Math.abs(FeiDataset.pose(row)) > rLimit) continue;
            imgs.add(im);
        }
        int n = imgs.size();
        int m = NAMES.length;
        // [метод][снимок]: {ошибка правого (на снимке слева), левого, среднее}; точки — для файла.
        double[][][] err = new double[m][n][];
        double[][][] pts = new double[m][n][];
        int[] fallback = new int[m];
        long[] ns = new long[m];
        long decodeNs = 0;
        for (int i = 0; i < n; i++) {
            Img im = imgs.get(i);
            double[] row = det.get(im.key());
            long t0 = System.nanoTime();
            Mat color = FeretData.decode(im.file());
            Mat g = EyeRefine.gray(color);
            color.release();
            decodeNs += System.nanoTime() - t0;
            double[] re = {row[4], row[5]};
            double[] le = {row[6], row[7]};
            double iod = Math.hypot(le[0] - re[0], le[1] - re[1]);
            for (int k = 0; k < m; k++) {
                long t1 = System.nanoTime();
                double[] r2 = refine(k, g, re, iod, tree, eye);
                double[] l2 = refine(k, g, le, iod, tree, eye);
                ns[k] += System.nanoTime() - t1;
                if (r2 == null) {
                    r2 = re;
                    fallback[k]++;
                }
                if (l2 == null) {
                    l2 = le;
                    fallback[k]++;
                }
                pts[k][i] = new double[] {r2[0], r2[1], l2[0], l2[1]};
                double[] e = FeretData.eyeError(im, r2, l2);
                err[k][i] = new double[] {e[1], e[0], e[2]};
            }
            g.release();
            if ((i + 1) % 200 == 0) System.out.printf(Locale.ROOT, "feret-eyepts: %d / %d%n", i + 1, n);
        }

        // Точки — для этапа 2.
        Path derived = dir.resolve(FeretData.DERIVED);
        Path tmp = derived.resolve("eyepts.tsv.tmp");
        try (Writer w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
            StringBuilder h = new StringBuilder("key");
            for (String nm : NAMES) h.append('\t').append(nm).append("_re_x\t").append(nm).append("_re_y\t").append(nm).append("_le_x\t").append(nm).append("_le_y");
            w.write(h.append('\n').toString());
            for (int i = 0; i < n; i++) {
                StringBuilder l = new StringBuilder(imgs.get(i).key());
                for (int k = 0; k < m; k++) for (double v : pts[k][i]) l.append('\t').append(String.format(Locale.ROOT, "%.3f", v));
                w.write(l.append('\n').toString());
            }
        }
        Files.move(tmp, derived.resolve("eyepts.tsv"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);

        // Люди бутстрепа.
        List<String> persons = new ArrayList<>(new TreeSet<>(imgs.stream().map(Img::person).toList()));
        Map<String, Integer> pIdx = new LinkedHashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        int[] owner = imgs.stream().mapToInt(im -> pIdx.get(im.person())).toArray();
        int[][] wp = b > 0 ? FeiBootstrap.weights(persons.size(), b, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а2б) уточнение точек глаз, этап 1, FERET (XBaseEval feret-eyepts): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb с ручной разметкой глаз; детекции YuNet — кэш derived/detections.tsv)\n");
        r.append(String.format(Locale.ROOT, "Снимков: %d (размечены, лицо найдено, |r| ≤ R = %.4f), людей %d.%n", n, rLimit, persons.size()));
        r.append(String.format(Locale.ROOT, "Каскады: %s (%s); %s (%s).%n", EYE_TREE, FeiThreshold.sha(models.resolve(EYE_TREE)), EYE,
                FeiThreshold.sha(models.resolve(EYE))));
        r.append(String.format(Locale.ROOT, "Параметры (заданы заранее): 0а — квадрат %.1f межглазья YuNet, ширина %d px, Гаусс σ = 1, порог градиента среднее + "
                + "%.1f·СКО, край %d px; 0б, 0б' — квадрат %.1f межглазья, scaleFactor 1,1, minNeighbors 3, размер от 1/4 до 1 стороны;%n"
                + "0в — 0б, затем 0а в найденном прямоугольнике (не найден — 0а). Расхождение — в долях ручного межглазья (FeretData.eyeError).%n",
                EyeRefine.GRAD_SIDE, EyeRefine.GRAD_WIDTH, EyeRefine.GRAD_THRESHOLD, EyeRefine.BORDER, EyeRefine.CASCADE_SIDE));
        if (b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", b,
                seed + FeiBootstrap.SEED_OFFSET));

        Map<String, Predicate<Img>> groups = new LinkedHashMap<>();
        groups.put("все", im -> true);
        groups.put("без очков", im -> !(im.flag().equals("a") || im.flag().equals("c")));
        groups.put("в очках (a, c)", im -> im.flag().equals("a") || im.flag().equals("c"));
        groups.put("fa", im -> im.pose().equals("fa"));
        groups.put("fb", im -> im.pose().equals("fb"));
        r.append("\n=== Расхождение с ручной разметкой (доли ручного межглазья): среднее по двум глазам на снимок; глаза — пулом ===\n");
        r.append("группа | кандидат | снимков | среднее: медиана | 95 % | макс | > 5 % | > 10 % | глаза: медиана | > 5 % | > 10 %\n");
        for (Map.Entry<String, Predicate<Img>> e : groups.entrySet()) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) if (e.getValue().test(imgs.get(i))) idx.add(i);
            if (idx.isEmpty()) continue;
            for (int k = 0; k < m; k++) {
                int kk = k;
                double[] mean = FeiBootstrap.sorted(idx.stream().mapToDouble(i -> err[kk][i][2]).toArray());
                double[] eyes = FeiBootstrap.sorted(idx.stream().flatMapToDouble(i -> Arrays.stream(new double[] {err[kk][i][0], err[kk][i][1]})).toArray());
                r.append(String.format(Locale.ROOT, "%s | %s | %d | %.4f | %.4f | %.4f | %.1f %% | %.1f %% | %.4f | %.1f %% | %.1f %%%n", e.getKey(), NAMES[k],
                        idx.size(), FeiBootstrap.quantile(mean, 0.5), FeiBootstrap.quantile(mean, 0.95), mean[mean.length - 1], 100 * share(mean, 0.05),
                        100 * share(mean, 0.10), FeiBootstrap.quantile(eyes, 0.5), 100 * share(eyes, 0.05), 100 * share(eyes, 0.10)));
            }
        }

        r.append("\n=== Разность «кандидат − YuNet» (все снимки; среднее по двум глазам на снимок), бутстреп по людям ===\n");
        r.append("кандидат | медиана, доли межглазья | доля > 5 %, п. п. | доля > 10 %, п. п. | откат на YuNet (глаз из ").append(2 * n)
                .append(") | переход к этапу 2\n");
        int[] unit = new int[n];
        Arrays.fill(unit, 1);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = err[0][i][2];
        double yMed = weightedQuantile(y, unit, 0.5);
        double y10 = weightedShare(y, unit, 0.10);
        for (int k = 1; k < m; k++) {
            double[] c = new double[n];
            for (int i = 0; i < n; i++) c[i] = err[k][i][2];
            double[] point = {weightedQuantile(c, unit, 0.5) - yMed, weightedShare(c, unit, 0.05) - weightedShare(y, unit, 0.05),
                    weightedShare(c, unit, 0.10) - y10};
            double[][] rep = new double[3][wp.length];
            for (int q = 0; q < wp.length; q++) {
                int[] wi = new int[n];
                for (int i = 0; i < n; i++) wi[i] = wp[q][owner[i]];
                rep[0][q] = weightedQuantile(c, wi, 0.5) - weightedQuantile(y, wi, 0.5);
                rep[1][q] = weightedShare(c, wi, 0.05) - weightedShare(y, wi, 0.05);
                rep[2][q] = weightedShare(c, wi, 0.10) - weightedShare(y, wi, 0.10);
            }
            boolean pass = weightedQuantile(c, unit, 0.5) <= GATE_MEDIAN * yMed && weightedShare(c, unit, 0.10) <= y10;
            r.append(String.join(" | ", NAMES[k], diff(point[0], rep[0], 1, "%+.4f"), diff(point[1], rep[1], 100, "%+.1f"), diff(point[2], rep[2], 100, "%+.1f"),
                    Integer.toString(fallback[k]), pass ? "да" : "нет")).append('\n');
        }
        r.append(String.format(Locale.ROOT, "Порог перехода (заранее): медиана ≤ %.1f × медиана YuNet = %.4f и доля > 10 %% ≤ доли YuNet = %.1f %%.%n", GATE_MEDIAN,
                GATE_MEDIAN * yMed, 100 * y10));

        r.append("\n=== Время на снимок (оба глаза), мс ===\n");
        r.append(String.format(Locale.ROOT, "распаковка ppm.bz2 и перевод в серый: %.1f%n", decodeNs / 1e6 / n));
        for (int k = 1; k < m; k++) r.append(String.format(Locale.ROOT, "%s: %.2f%n", NAMES[k], ns[k] / 1e6 / n));
        r.append("Точки всех кандидатов — derived/eyepts.tsv (re — правый глаз человека, на снимке слева).\n");
        Path out = dir.resolve("reports");
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_feret_eyepts.txt"), r);
        System.out.print(r);
    }

    /** Точка кандидата k (0 — YuNet); null — каскад не нашёл. */
    static double[] refine(int k, Mat g, double[] p, double iod, CascadeClassifier tree, CascadeClassifier eye) {
        return switch (k) {
            case 0 -> p;
            case 1 -> EyeRefine.gradient(g, p, iod);
            case 2 -> EyeRefine.cascade(g, p, iod, tree);
            case 3 -> EyeRefine.cascade(g, p, iod, eye);
            default -> EyeRefine.combined(g, p, iod, tree);
        };
    }

    static double share(double[] v, double t) {
        return Arrays.stream(v).filter(x -> x > t).count() / (double) v.length;
    }

    /** «точка [2,5; 97,5]» со звёздочкой, если интервал не содержит 0. */
    static String diff(double point, double[] rep, double mult, String fmt) {
        if (rep.length == 0) return String.format(Locale.ROOT, fmt, mult * point);
        double[] s = FeiBootstrap.sorted(rep);
        double lo = FeiBootstrap.quantile(s, 0.025);
        double hi = FeiBootstrap.quantile(s, 0.975);
        return String.format(Locale.ROOT, fmt + " [" + fmt + "; " + fmt + "]%s", mult * point, mult * lo, mult * hi, lo > 0 || hi < 0 ? " *" : "");
    }
}
