package svd.recognizer.faces;

import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.Face;
import org.opencv.face.Facemark;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.HaarFrames.Det;
import svd.recognizer.storage.SettingsStore;

/**
 * (4а), строка (9б) режима «только классика» (решение Хозяина 19.10.2026): лицо – каскад Хаара ({@link HaarFrames}), точки –
 * Facemark LBF (68 точек, модуль face OpenCV; модель {@value #MODEL_FILE} в faces.models.dir). Пять точек для выравнивания
 * подобием (порядок YuNet): центр глаза слева на снимке – среднее точек 36–41, справа – 42–47, кончик носа – 30, углы рта –
 * 48 (слева на снимке) и 54 (индексы с 0, разметка iBUG 68). Поза r – как в базе ({@link FeiDataset#pose}) по этим точкам.
 * Отказ – только при пропуске лица Хааром.
 * <p>
 * Режим abl-lbf (XBaseEval) – проверка без распознавания: точность глаз и носа LBF против ручной разметки FERET рядом с Ю5, 0б и
 * Хааром, поза r_LBF, R_LBF (своя база) и r0_LBF (№11–14 FEI), листы. Кэши точек – lbf.tsv рядом с haar.tsv.
 *
 * @author ssv
 */
final class LbfFrames {

    static final String MODEL_FILE = "lbfmodel.yaml";
    static final String CACHE = "lbf.tsv";
    static final int POINTS = 68;

    private LbfFrames() {
    }

    /** 68 точек в полном снимке; five() – строка в формате YuNet (индексы 4–13 – пять точек), для pose и кадра. */
    record Lm(double[][] p) {

        double[] five() {
            double[] row = new double[15];
            double[][] f = {mean(36, 41), mean(42, 47), p[30], p[48], p[54]};
            for (int j = 0; j < 5; j++) {
                row[4 + 2 * j] = f[j][0];
                row[5 + 2 * j] = f[j][1];
            }
            return row;
        }

        double[] mean(int a, int b) {
            double x = 0;
            double y = 0;
            for (int i = a; i <= b; i++) {
                x += p[i][0];
                y += p[i][1];
            }
            return new double[] {x / (b - a + 1), y / (b - a + 1)};
        }

        double pose() {
            return FeiDataset.pose(five());
        }
    }

    static Facemark load(Path models) throws IOException {
        Path f = models.resolve(MODEL_FILE);
        if (!Files.exists(f)) throw new IllegalStateException("Нет модели LBF " + f);
        Facemark fm = Face.createFacemarkLBF();
        fm.loadModel(f.toString());
        return fm;
    }

    static String sha256(Path f) throws IOException {
        try (InputStream in = Files.newInputStream(f)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0;) md.update(buf, 0, n);
            StringBuilder s = new StringBuilder();
            for (byte x : md.digest()) s.append(String.format(Locale.ROOT, "%02x", x));
            return s.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Точки LBF по рамке Хаара на входе in (масштаб scale от полного снимка); null – нет лица или fit не удался. */
    static Lm fit(Facemark fm, Mat in, double scale, Det d) {
        if (!d.found()) return null;
        double[] f = d.face();
        MatOfRect rect = new MatOfRect(new Rect((int) Math.round(f[0] * scale), (int) Math.round(f[1] * scale), (int) Math.round(f[2] * scale),
                (int) Math.round(f[3] * scale)));
        List<MatOfPoint2f> out = new ArrayList<>();
        if (!fm.fit(in, rect, out) || out.isEmpty()) return null;
        Point[] pts = out.get(0).toArray();
        if (pts.length != POINTS) throw new IllegalStateException("LBF: точек " + pts.length + " вместо " + POINTS);
        double[][] p = new double[POINTS][];
        for (int i = 0; i < POINTS; i++) p[i] = new double[] {pts[i].x / scale, pts[i].y / scale};
        return new Lm(p);
    }

    /** Кадр 92×112: подобие по пяти точкам LBF на шаблон кадра «а» (как кадр «а» по точкам YuNet). */
    static Mat frame(Mat in, double scale, double[] row) {
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = row[4 + 2 * j] * scale;
            lm[j][1] = row[5 + 2 * j] * scale;
        }
        Mat a = new Mat();
        Imgproc.warpAffine(in, a, FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(HaarFrames.W, HaarFrames.H)), new Size(HaarFrames.W, HaarFrames.H),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        return a;
    }

    /** Кэш точек: «# подпись», затем ключ, найдено, 136 чисел; недостающие ключи считаются и дописываются. */
    static Map<String, Lm> cached(Path file, String signature, HaarFrames.Set s, Facemark fm, String label) throws IOException {
        Map<String, Lm> done = new HashMap<>();
        boolean fresh = !Files.exists(file);
        if (!fresh) {
            List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            if (lines.isEmpty() || !lines.get(0).equals("# " + signature))
                throw new IllegalStateException(file + ": другая подпись кэша LBF – удалите файл и повторите");
            for (String l : lines.subList(1, lines.size())) {
                String[] c = l.split("\t", -1);
                if (c.length != 2 + 2 * POINTS) continue;
                Lm lm = null;
                if (c[1].equals("1")) {
                    double[][] p = new double[POINTS][];
                    for (int i = 0; i < POINTS; i++) p[i] = new double[] {Double.parseDouble(c[2 + 2 * i]), Double.parseDouble(c[3 + 2 * i])};
                    lm = new Lm(p);
                }
                done.put(c[0], lm);
            }
        }
        int n = 0;
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (fresh) w.write("# " + signature + "\n");
            for (String k : s.haar().keySet()) {
                if (done.containsKey(k)) continue;
                Det d = s.haar().get(k);
                Lm lm = null;
                if (d.found()) {
                    Mat color = s.loader().color(k);
                    Mat in = HaarFrames.input(color, s.scale());
                    color.release();
                    lm = fit(fm, in, s.scale(), d);
                    in.release();
                }
                StringBuilder l = new StringBuilder(k).append('\t').append(lm == null ? "0" : "1");
                for (int i = 0; i < POINTS; i++)
                    l.append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][0])).append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][1]));
                w.write(l.append('\n').toString());
                w.flush();
                done.put(k, lm);
                if (++n % 500 == 0) System.out.printf(Locale.ROOT, "abl-lbf: %s – %d новых%n", label, n);
            }
        }
        Map<String, Lm> out = new LinkedHashMap<>();
        for (String k : s.haar().keySet()) out.put(k, done.get(k));
        return out;
    }

    static Path cacheFile(HaarFrames.Bases b, String name) {
        return switch (name) {
            case "FERET" -> b.feretDir().resolve(FeretData.DERIVED).resolve(CACHE);
            case "GT" -> b.gtDir().getParent().resolve(CACHE);
            case "FEI" -> b.feiDir().resolve(CACHE);
            default -> b.ownDir().resolve(CACHE);
        };
    }

    // ---------------------------------------------------------------- abl-lbf

    static void ablLbf(Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        int bs = settings.loadFacesFeiBootstrap();
        double rLimit = XBaseEval.poseLimit(selLines);
        Path models = Paths.get(settings.loadFacesModelsDir());
        CascadeClassifier face = HaarDebug.load(models.resolve(HaarDebug.FACE_FILE));
        CascadeClassifier eye = HaarDebug.load(models.resolve(HaarDebug.EYE_GLASSES_FILE));
        HaarFrames.Bases b = HaarFrames.bases(exportDir, feiKeys, rLimit, settings, face, eye);
        Path modelFile = models.resolve(MODEL_FILE);
        String sha = sha256(modelFile);
        long size = Files.size(modelFile);
        String signature = String.format(Locale.ROOT, "lbf v1|%s %d bytes sha256 %s|%s", MODEL_FILE, size, sha, HaarFrames.SIGNATURE);
        Facemark fm = load(models);
        Map<String, Map<String, Lm>> lbf = new LinkedHashMap<>();
        for (HaarFrames.Set s : List.of(b.own(), b.fei(), b.gt(), b.feret())) lbf.put(s.name(), cached(cacheFile(b, s.name()), signature, s, fm, s.name()));

        // R_LBF и r0_LBF – правилом базы; R_LBF' – только снимки своей базы, где рамка Хаара совпадает с YuNet (справочно).
        Map<String, Lm> ownL = lbf.get(b.own().name());
        Map<String, Lm> feiL = lbf.get(b.fei().name());
        double[] ownAbs = b.ownProto().stream().filter(k -> ownL.get(k) != null).mapToDouble(k -> Math.abs(ownL.get(k).pose())).sorted().toArray();
        double[] ownAbsTrue = b.ownProto().stream().filter(k -> ownL.get(k) != null && b.own().yunet().get(k) != null
                && HaarFrames.iou(b.own().haar().get(k).face(), b.own().yunet().get(k)) >= HaarFrames.IOU_FALSE).mapToDouble(k -> Math.abs(ownL.get(k).pose()))
                .sorted().toArray();
        double[] feiAbs = b.fei1114().stream().filter(k -> feiL.get(k) != null).mapToDouble(k -> Math.abs(feiL.get(k).pose())).sorted().toArray();
        double bigR = FeiDataset.percentile(ownAbs, FeiDataset.POSE_PERCENTILE);
        double bigRTrue = FeiDataset.percentile(ownAbsTrue, FeiDataset.POSE_PERCENTILE);
        double r0 = FeiDataset.percentile(feiAbs, FeiDataset.POSE_PERCENTILE);
        String r0Base = settings.loadFacesFeiPoseR0();

        String head = String.format(Locale.ROOT, "Код: коммит %s%nМодель LBF: %s, %d байт, SHA-256 %s (Facemark LBF, модуль face OpenCV); лицо – Хаар (кэш "
                + "haar.tsv, параметры abl-haar; наибольшая рамка).%nПять точек (порядок YuNet, индексы iBUG 68 с 0): глаз слева на снимке – среднее 36–41, "
                + "справа – 42–47, нос – 30, рот – 48 и 54.%nПоза r_LBF – как r базы по этим точкам. Отказ – только при пропуске лица.%n"
                + "R_LBF = 95-й процентиль |r_LBF| по снимкам протокола своей базы с точками (%d из %d) = %.4f (у YuNet R = %.4f); справочно R_LBF' – "
                + "только где рамка Хаара совпала с YuNet (IoU ≥ %.1f; %d снимков) = %.4f.%n"
                + "r0_LBF = 95-й процентиль |r_LBF| по №11–14 FEI с точками (%d из %d) = %.4f (у YuNet r0 = %s).%n",
                FarMethods.commit(), MODEL_FILE, size, sha, ownAbs.length, b.ownProto().size(), bigR, rLimit, HaarFrames.IOU_FALSE, ownAbsTrue.length, bigRTrue,
                feiAbs.length, b.fei1114().size(), r0, r0Base == null ? "не задан" : r0Base);

        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, (4а), строка (9б) Хаар + Facemark LBF: проверка без распознавания (XBaseEval abl-lbf)\n");
        r.append(head).append("FERET – отдельным отчётом, только в каталоге FERET.\n");
        for (HaarFrames.Set s : List.of(b.gt(), b.fei(), b.own())) report(r, s, lbf.get(s.name()), bigR, r0, rLimit);
        Path sheets = out.resolve("abl_lbf");
        Files.createDirectories(sheets);
        sheet(sheets.resolve("gt_sheet.png"), b.gt(), lbf.get(b.gt().name()), seed);
        r.append("Лист: ").append(sheets.resolve("gt_sheet.png")).append('\n');
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_abl_lbf.txt"), r);
        System.out.print(r);

        StringBuilder q = new StringBuilder("PLAN.md п. 5.5, (4а), строка (9б) Хаар + Facemark LBF, FERET: проверка без распознавания (XBaseEval abl-lbf); "
                + "Color FERET – только локально\n");
        q.append(head);
        HaarFrames.Set feret = b.feret();
        Map<String, Lm> frL = lbf.get(feret.name());
        report(q, feret, frL, bigR, r0, rLimit);
        manual(q, b, frL, models, rLimit, seed, bs);
        Path fOut = b.feretDir().resolve("reports").resolve("eq");
        Files.createDirectories(fOut.resolve("abl_lbf"));
        sheet(fOut.resolve("abl_lbf").resolve("feret_sheet.png"), feret, frL, seed);
        q.append("Лист: ").append(fOut.resolve("abl_lbf").resolve("feret_sheet.png")).append('\n');
        q.append(String.format(Locale.ROOT, "%nВремя (всего abl-lbf): %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(fOut.resolve("xbase_feret_abl_lbf.txt"), q);
        System.out.print(q);
    }

    static void report(StringBuilder r, HaarFrames.Set s, Map<String, Lm> l, double bigR, double r0, double rLimit) {
        r.append("\n=== ").append(s.name()).append(": обнаружение и поза ===\n");
        r.append(String.format(Locale.ROOT, "набор | снимков | YuNet нашёл | нет лица (отказ) | fit не удался | с точками | |r_LBF|: 50 %% | 75 %% | 95 %% | макс | "
                + "|r_LBF| ≤ R_LBF = %.4f | |r_LBF| ≤ r0_LBF = %.4f (модель X) | YuNet: |r| ≤ R = %.4f (из найденных) | Спирмен r_LBF и r YuNet%n", bigR, r0, rLimit));
        for (Map.Entry<String, List<String>> g : s.groups().entrySet()) {
            List<String> ks = g.getValue();
            int n = ks.size();
            int y = (int) ks.stream().filter(k -> s.yunet().get(k) != null).count();
            int nf = (int) ks.stream().filter(k -> !s.haar().get(k).found()).count();
            int ff = (int) ks.stream().filter(k -> s.haar().get(k).found() && l.get(k) == null).count();
            List<String> with = ks.stream().filter(k -> l.get(k) != null).toList();
            double[] v = FeiBootstrap.sorted(with.stream().mapToDouble(k -> Math.abs(l.get(k).pose())).toArray());
            int pass = (int) with.stream().filter(k -> Math.abs(l.get(k).pose()) <= bigR).count();
            int x = (int) with.stream().filter(k -> Math.abs(l.get(k).pose()) <= r0).count();
            int yPass = (int) ks.stream().filter(k -> s.yunet().get(k) != null && Math.abs(FeiDataset.pose(s.yunet().get(k))) <= rLimit).count();
            List<String> both = with.stream().filter(k -> s.yunet().get(k) != null).toList();
            String sp = both.size() < 3 ? "–" : String.format(Locale.ROOT, "%+.3f (%d)", XBaseMpPts.spearman(both.stream().mapToDouble(k -> l.get(k).pose()).toArray(),
                    both.stream().mapToDouble(k -> FeiDataset.pose(s.yunet().get(k))).toArray()), both.size());
            r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s%n", g.getKey(), n, HaarFrames.pct(y, n),
                    HaarFrames.pct(nf, n), ff, with.size(), HaarFrames.q(v, 0.5), HaarFrames.q(v, 0.75), HaarFrames.q(v, 0.95),
                    v.length == 0 ? "–" : String.format(Locale.ROOT, "%.3f", v[v.length - 1]), HaarFrames.pct(pass, with.size()), HaarFrames.pct(x, with.size()),
                    HaarFrames.pct(yPass, y), sp));
        }
        r.append("\n=== ").append(s.name()).append(": точки LBF против Ю5 (справочно; расстояние / межглазье YuNet; снимки, где есть обе) ===\n");
        r.append("набор | снимков | глаза (среднее по двум): медиана | 95 % | > 10 % | > 25 % | нос: медиана | 95 % | > 10 % | > 25 %\n");
        for (Map.Entry<String, List<String>> g : s.groups().entrySet()) {
            List<String> ks = g.getValue().stream().filter(k -> l.get(k) != null && s.yunet().get(k) != null).toList();
            double[] e = new double[ks.size()];
            double[] nn = new double[ks.size()];
            for (int i = 0; i < ks.size(); i++) {
                double[] y = s.yunet().get(ks.get(i));
                double[] f = l.get(ks.get(i)).five();
                double iod = Math.hypot(y[6] - y[4], y[7] - y[5]);
                e[i] = (Math.hypot(f[4] - y[4], f[5] - y[5]) + Math.hypot(f[6] - y[6], f[7] - y[7])) / 2 / iod;
                nn[i] = Math.hypot(f[8] - y[8], f[9] - y[9]) / iod;
            }
            e = FeiBootstrap.sorted(e);
            nn = FeiBootstrap.sorted(nn);
            r.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s | %s | %s | %s | %s%n", g.getKey(), ks.size(), HaarFrames.q(e, 0.5), HaarFrames.q(e, 0.95),
                    HaarFrames.share(e, 0.10), HaarFrames.share(e, 0.25), HaarFrames.q(nn, 0.5), HaarFrames.q(nn, 0.95), HaarFrames.share(nn, 0.10),
                    HaarFrames.share(nn, 0.25)));
        }
    }

    /**
     * FERET: глаза и нос против ручной разметки на снимках feret-eyepts (размечены, YuNet нашёл, |r| ≤ R); Ю5 и 0б – как
     * feret-eyepts (строки «все» сверяются побайтно с reports/xbase_feret_eyepts.txt); LBF и Хаар – на своих подмножествах,
     * рядом Ю5 и 0б на тех же снимках; парные разности с бутстрепом по людям.
     */
    static void manual(StringBuilder r, HaarFrames.Bases b, Map<String, Lm> frL, Path models, double rLimit, long seed, int bs) throws IOException {
        HaarFrames.Set s = b.feret();
        List<Img> imgs = new ArrayList<>();
        for (Img im : b.frImg().values()) {
            double[] row = s.yunet().get(im.key());
            if (im.leftEye() == null || im.rightEye() == null || row == null || Math.abs(FeiDataset.pose(row)) > rLimit) continue;
            imgs.add(im);
        }
        int n = imgs.size();
        CascadeClassifier tree = new CascadeClassifier(models.resolve(XBaseEyePts.EYE_TREE).toString());
        CascadeClassifier eye = new CascadeClassifier(models.resolve(XBaseEyePts.EYE).toString());
        double[][] eY = new double[n][];
        double[][] e0b = new double[n][];
        double[][] eH = new double[n][];
        double[][] eL = new double[n][];
        double[][] nose = new double[n][];
        for (int i = 0; i < n; i++) {
            Img im = imgs.get(i);
            double[] row = s.yunet().get(im.key());
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
            Det d = s.haar().get(im.key());
            if (d.both()) {
                e = FeretData.eyeError(im, d.left(), d.right());
                eH[i] = new double[] {e[1], e[0], e[2]};
            }
            Lm lm = frL.get(im.key());
            if (lm != null) {
                double[] f = lm.five();
                e = FeretData.eyeError(im, new double[] {f[4], f[5]}, new double[] {f[6], f[7]});
                eL[i] = new double[] {e[1], e[0], e[2]};
                if (im.nose() != null) {
                    double miod = XBaseMpPts.dist(im.leftEye(), im.rightEye());
                    nose[i] = new double[] {XBaseMpPts.dist(new double[] {row[8], row[9]}, im.nose()) / miod, XBaseMpPts.dist(new double[] {f[8], f[9]}, im.nose()) / miod};
                }
            }
            if ((i + 1) % 500 == 0) System.out.printf(Locale.ROOT, "abl-lbf: 0б %d / %d%n", i + 1, n);
        }
        List<String> persons = new ArrayList<>(new TreeSet<>(imgs.stream().map(Img::person).toList()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        int[][] wp = bs > 0 ? FeiBootstrap.weights(persons.size(), bs, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];

        r.append(String.format(Locale.ROOT, "%n=== FERET: глаза против ручной разметки (доли ручного межглазья) – снимки feret-eyepts: %d (размечены, YuNet нашёл, "
                + "|r| ≤ R = %.4f), людей %d ===%n", n, rLimit, persons.size()));
        if (bs > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» – интервал разности не содержит 0.%n", bs,
                seed + FeiBootstrap.SEED_OFFSET));
        String eyepts = Files.readString(b.feretDir().resolve("reports").resolve("xbase_feret_eyepts.txt"), StandardCharsets.UTF_8).replace("\r", "");
        Map<String, Predicate<Img>> groups = new LinkedHashMap<>();
        groups.put("все", im -> true);
        groups.put("без очков", im -> !(im.flag().equals("a") || im.flag().equals("c")));
        groups.put("в очках (a, c)", im -> im.flag().equals("a") || im.flag().equals("c"));
        r.append("группа | кандидат | снимков | среднее: медиана | 95 % | макс | > 5 % | > 10 % | глаза: медиана | > 5 % | > 10 %\n");
        int checked = 0;
        for (Map.Entry<String, Predicate<Img>> g : groups.entrySet()) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) if (g.getValue().test(imgs.get(i))) idx.add(i);
            String ly = XBaseMpPts.errLine(g.getKey(), "YuNet", eY, idx);
            String l0 = XBaseMpPts.errLine(g.getKey(), "0б", e0b, idx);
            for (String l : new String[] {ly, l0}) {
                if (!eyepts.contains(l.replace("\r", ""))) throw new IllegalStateException("abl-lbf: строка не совпала с xbase_feret_eyepts.txt: " + l);
                checked++;
            }
            List<Integer> idxL = idx.stream().filter(i -> eL[i] != null).toList();
            List<Integer> idxH = idx.stream().filter(i -> eH[i] != null).toList();
            r.append(ly.replace(" | YuNet | ", " | Ю5 | ")).append(l0);
            r.append(XBaseMpPts.errLine(g.getKey(), "LBF", eL, idxL));
            r.append(XBaseMpPts.errLine(g.getKey(), "Ю5 (где LBF)", eY, idxL));
            r.append(XBaseMpPts.errLine(g.getKey(), "0б (где LBF)", e0b, idxL));
            r.append(XBaseMpPts.errLine(g.getKey(), "Хаар (два глаза)", eH, idxH));
            r.append(XBaseMpPts.errLine(g.getKey(), "LBF (где Хаар – два глаза)", eL, idxH.stream().filter(i -> eL[i] != null).toList()));
        }
        r.append(String.format(Locale.ROOT, "Сверка: строки Ю5 и 0б – %d строк – побайтно совпали с reports/xbase_feret_eyepts.txt.%n", checked));

        List<Integer> nIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) if (nose[i] != null) nIdx.add(i);
        double[] nY = nIdx.stream().mapToDouble(i -> nose[i][0]).toArray();
        double[] nL = nIdx.stream().mapToDouble(i -> nose[i][1]).toArray();
        r.append("\n=== FERET: нос против ручной разметки (доли ручного межглазья; снимки, где есть LBF и ручная точка носа) ===\n");
        r.append("точка | кандидат | снимков | медиана | 95 % | макс | > 5 % | > 10 %\n");
        r.append(XBaseMpPts.oneLine("нос", "Ю5", nY)).append(XBaseMpPts.oneLine("нос", "LBF", nL));

        r.append("\n=== FERET: парные разности (глаза – среднее по двум на снимок), бутстреп по людям ===\n");
        r.append("разность | медиана, доли межглазья | доля > 5 %, п. п. | доля > 10 %, п. п.\n");
        List<Integer> li = new ArrayList<>();
        for (int i = 0; i < n; i++) if (eL[i] != null) li.add(i);
        int[] owner = li.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray();
        double[] lE = li.stream().mapToDouble(i -> eL[i][2]).toArray();
        r.append((String) XBaseMpPts.pairDiff("глаза: LBF − Ю5", lE, li.stream().mapToDouble(i -> eY[i][2]).toArray(), owner, wp)[0]);
        r.append((String) XBaseMpPts.pairDiff("глаза: LBF − 0б", lE, li.stream().mapToDouble(i -> e0b[i][2]).toArray(), owner, wp)[0]);
        List<Integer> lh = li.stream().filter(i -> eH[i] != null).toList();
        r.append((String) XBaseMpPts.pairDiff("глаза: LBF − Хаар (где два глаза)", lh.stream().mapToDouble(i -> eL[i][2]).toArray(),
                lh.stream().mapToDouble(i -> eH[i][2]).toArray(), lh.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray(), wp)[0]);
        r.append((String) XBaseMpPts.pairDiff("нос: LBF − Ю5", nL, nY, nIdx.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray(), wp)[0]);
    }

    /** Лист: 24 случайных снимка с точками LBF (seed + 2); плитка – окрестность лица (68 точек жёлтые, 5 точек красные, глаза YuNet зелёные), кадр LBF и кадр «а» YuNet ×2. */
    static void sheet(Path file, HaarFrames.Set s, Map<String, Lm> l, long seed) throws IOException {
        List<String> ks = new ArrayList<>(s.haar().keySet().stream().filter(k -> l.get(k) != null && s.yunet().get(k) != null).toList());
        Collections.shuffle(ks, new Random(seed + 2));
        List<String> list = ks.subList(0, Math.min(24, ks.size()));
        int side = 224;
        int fw = 2 * HaarFrames.W;
        int tile = side + 2 * fw;
        int cols = 3;
        int rows = (list.size() + cols - 1) / cols;
        Mat sheet = new Mat(Math.max(1, rows) * (side + 16), cols * tile, CvType.CV_8UC3, new Scalar(255, 255, 255));
        for (int t = 0; t < list.size(); t++) {
            String k = list.get(t);
            Lm lm = l.get(k);
            double[] row = s.yunet().get(k);
            double[] box = s.haar().get(k).face();
            Mat color = s.loader().color(k);
            Mat in = HaarFrames.input(color, s.scale());
            color.release();
            double sc = s.scale();
            double cx = (box[0] + box[2] / 2) * sc;
            double cy = (box[1] + box[3] / 2) * sc;
            double half = 0.65 * Math.max(box[2], box[3]) * sc;
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
            for (double[] p : lm.p()) Imgproc.circle(crop, map.apply(p), 1, new Scalar(0, 220, 220), -1);
            double[] f = lm.five();
            for (int j = 0; j < 5; j++) Imgproc.drawMarker(crop, map.apply(new double[] {f[4 + 2 * j], f[5 + 2 * j]}), new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 9, 1);
            for (int j = 0; j < 2; j++) Imgproc.circle(crop, map.apply(new double[] {row[4 + 2 * j], row[5 + 2 * j]}), 2, new Scalar(0, 200, 0), -1);
            int tx = (t % cols) * tile;
            int ty = (t / cols) * (side + 16);
            crop.copyTo(sheet.submat(new Rect(tx, ty, crop.cols(), crop.rows())));
            Mat[] frames = {frame(in, sc, f), frame(in, sc, row)};
            for (int j = 0; j < 2; j++) {
                Mat big = new Mat();
                Imgproc.resize(frames[j], big, new Size(fw, 2 * HaarFrames.H), 0, 0, Imgproc.INTER_NEAREST);
                Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2BGR);
                big.copyTo(sheet.submat(new Rect(tx + side + j * fw, ty, fw, 2 * HaarFrames.H)));
            }
            String cap = k.replaceAll("[^\\x20-\\x7E]", "?") + String.format(Locale.ROOT, " r %.2f / YuNet %.2f  [LBF | a]", lm.pose(), FeiDataset.pose(row));
            Imgproc.putText(sheet, cap, new Point(tx + 2, ty + side + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
            in.release();
        }
        Imgcodecs.imwrite(file.toString(), sheet);
    }
}
