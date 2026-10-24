package svd.recognizer.faces;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseMp2.FeretSets;
import svd.recognizer.faces.XBaseMp2.Thr;
import svd.recognizer.storage.SettingsStore;

/**
 * Дешёвые исправления конвейера по ревизии (Зануда, пп. 1.11–1.13; Жорик №7, №8 от 23.10.2026) – tools/verify, ветка cloud-verify.
 * Код проекта не меняется, методы проекта вызываются как есть.
 * <p>
 * Варианты: А – поза r′ (нос в осях, повёрнутых на −roll по линии глаз: r′ = ((нос − середина глаз)·e) / |e|², e – вектор между
 * глазами), отбор |r′| ≤ R′, модель X / X+mirror по |r′| ≤ r0′; Б – кадр со сглаживанием: warpAffine в кадр k·(92×112), k =
 * ceil(1 / масштаб подобия), матрица k·M со сдвигом (k − 1)/2 (центры блоков k×k совпадают с центрами пикселей кадра), затем
 * resize INTER_AREA до 92×112; В – подобие наименьшими квадратами по 5 точкам (FaceMirror.similarityLsq) вместо LMEDS
 * (FaceAlignment.similarity). Когорта FEI – всегда прежние кадры экспорта (исходных снимков FEI в облаке нет).
 * <p>
 * Режимы: stats &lt;выход FERET&gt; &lt;выход прочих&gt; – без распознавания: по каждому снимку r, r′, число инлайеров LMEDS, масштабы
 * LMEDS и наименьших квадратов, k, σ шума (Immerkær, XBaseQuality.frameMeasures) кадра базы и Б, сверка кадра базы из сырого снимка
 * с кадром экспорта; feret &lt;вариант&gt; R′ r0′ &lt;каталог прежних оценок&gt; &lt;выход&gt; [nocohort] – роли feret-xs2 (половины №2, два
 * прохода), только межс.; gt &lt;вариант&gt; R′ r0′ &lt;файл оценок gt-subsp&gt; &lt;выход&gt; [nocohort] – gt-subsp k = 0, межс. N = 6 и
 * внутри N = 3, пул FERET. Вариант: base | A | B | C | ABC. Ключи – как XBaseEval (FACES_EXPORT_DIR, FACES_FERET_DIR,
 * FACES_GT_DIR, FACES_MODELS_DIR, FACES_FEI_POSE_R0).
 */
public final class VerifyRev {

    private VerifyRev() {
    }

    static final int W = 92;
    static final int H = 112;

    /** r′: проекция смещения носа от середины глаз на линию глаз, в долях межглазья (= x носа в осях, повёрнутых на −roll). */
    static double rPrime(double[] row) {
        double ex = row[6] - row[4];
        double ey = row[7] - row[5];
        if (ex < 0) {
            ex = -ex;
            ey = -ey;
        }
        double mx = (row[4] + row[6]) / 2;
        double my = (row[5] + row[7]) / 2;
        return ((row[8] - mx) * ex + (row[9] - my) * ey) / (ex * ex + ey * ey);
    }

    static double[][] five(double[] row) {
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = row[4 + 2 * j];
            lm[j][1] = row[5 + 2 * j];
        }
        return lm;
    }

    static Mat toMat(double[][] m) {
        Mat out = new Mat(2, 3, CvType.CV_64F);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) out.put(i, j, m[i][j]);
        return out;
    }

    /** Подобие: LMEDS (как FaceAlignment.similarity) или наименьшие квадраты (FaceMirror.similarityLsq). */
    static Mat sim(double[] row, boolean lsq) {
        double[][] t = FaceAlignment.ownTemplate(W, H);
        return lsq ? toMat(FaceMirror.similarityLsq(five(row), t)) : FaceAlignment.similarity(five(row), t);
    }

    static double scale(Mat m) {
        double a = m.get(0, 0)[0];
        double b = m.get(1, 0)[0];
        return Math.sqrt(a * a + b * b);
    }

    static int kOf(Mat m) {
        return Math.max(1, (int) Math.ceil(1.0 / scale(m) - 1e-9));
    }

    /** Кадр 92×112 серый: smooth – вариант Б, lsq – вариант В; без обоих – как FeretData.frameA. */
    static Mat frame(Mat color, double[] row, boolean smooth, boolean lsq) {
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat m = sim(row, lsq);
        Mat a = new Mat();
        if (!smooth) {
            Imgproc.warpAffine(gray, a, m, new Size(W, H), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        } else {
            int k = kOf(m);
            Mat mk = new Mat(2, 3, CvType.CV_64F);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) mk.put(i, j, k * m.get(i, j)[0] + (j == 2 ? (k - 1) / 2.0 : 0));
            }
            Mat big = new Mat();
            Imgproc.warpAffine(gray, big, mk, new Size(W * k, H * k), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
            Imgproc.resize(big, a, new Size(W, H), 0, 0, Imgproc.INTER_AREA);
            big.release();
            mk.release();
        }
        gray.release();
        m.release();
        return a;
    }

    static int inliers(double[] row) {
        double[][] f = five(row);
        double[][] t = FaceAlignment.ownTemplate(W, H);
        Point[] a = new Point[5];
        Point[] b = new Point[5];
        for (int i = 0; i < 5; i++) {
            a[i] = new Point(f[i][0], f[i][1]);
            b[i] = new Point(t[i][0], t[i][1]);
        }
        Mat in = new Mat();
        Mat m = Calib3d.estimateAffinePartial2D(new MatOfPoint2f(a), new MatOfPoint2f(b), in, Calib3d.LMEDS);
        int n = Core.countNonZero(in);
        m.release();
        in.release();
        return n;
    }

    static boolean same(Mat x, Mat y) {
        Mat d = new Mat();
        Core.absdiff(x, y, d);
        boolean s = Core.countNonZero(d) == 0;
        d.release();
        return s;
    }

    /** Строка stats по снимку (color == null – без кадров). */
    static String statLine(String base, String key, double[] row, Mat color, Mat stored) {
        Mat ml = sim(row, false);
        Mat mq = sim(row, true);
        StringBuilder b = new StringBuilder(base).append('\t').append(key);
        b.append('\t').append(FeiDataset.pose(row)).append('\t').append(rPrime(row)).append('\t').append(inliers(row));
        b.append('\t').append(scale(ml)).append('\t').append(scale(mq)).append('\t').append(kOf(ml)).append('\t').append(kOf(mq));
        if (color != null) {
            Mat fa = frame(color, row, false, false);
            Mat fb = frame(color, row, true, false);
            Mat fc = frame(color, row, false, true);
            Mat fbc = frame(color, row, true, true);
            b.append('\t').append(stored == null ? "-" : same(fa, stored) ? "1" : "0");
            for (Mat f : new Mat[] {fa, fb, fc, fbc}) {
                b.append('\t').append(XBaseQuality.frameMeasures(f)[2]);
                f.release();
            }
        } else {
            b.append("\t-\t-\t-\t-\t-");
        }
        ml.release();
        mq.release();
        return b.append('\n').toString();
    }

    static void stats(SettingsStore settings, Path outFeret, Path outOther) throws IOException {
        Path dir = Paths.get(settings.loadFacesExportDir());
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        String head = "base\tkey\tr\tr_prime\tlmeds_inliers\tscale_lmeds\tscale_lsq\tk_lmeds\tk_lsq\tframe_same\tsigma_base\tsigma_B\tsigma_C\tsigma_BC\n";
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        try (BufferedWriter w = Files.newBufferedWriter(outFeret, StandardCharsets.UTF_8)) {
            w.write(head);
            int n = 0;
            for (String k : new TreeSet<>(su.det.keySet())) {
                double[] row = su.det.get(k);
                if (row == null) continue;
                Mat color = FeretData.decode(su.byKey.get(k).file());
                Mat st = FeretData.frame(su.dir, "a", k);
                w.write(statLine("feret", k, row, color, st));
                color.release();
                st.release();
                if (++n % 500 == 0) System.out.println("stats feret " + n);
            }
        }
        FarExport.Loaded export = FarExport.read(dir);
        Map<String, Sample> gtSample = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) for (Sample s : l) gtSample.put(s.person() + "/" + s.file().getFileName().toString(), s);
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        try (BufferedWriter w = Files.newBufferedWriter(outOther, StandardCharsets.UTF_8)) {
            w.write(head);
            for (String base : new String[] {"gt", "own", "muct", "fei", "orl"}) {
                for (Map.Entry<String, double[]> e : XBaseMpPts.exportRows(dir, base).entrySet()) {
                    if (e.getValue() == null) continue;
                    if (base.equals("gt")) {
                        Mat color = Imgcodecs.imread(gtDir.resolve(e.getKey() + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                        if (color.empty()) throw new IOException("нет снимка GT " + e.getKey());
                        Sample s = gtSample.get(e.getKey());
                        Map<String, Mat> st = XBaseEval.frames(export, List.of(s), null);
                        Mat stored = st.get(s.file().toString());
                        w.write(statLine(base, e.getKey(), e.getValue(), color, stored));
                        color.release();
                        stored.release();
                    } else {
                        w.write(statLine(base, e.getKey(), e.getValue(), null, null));
                    }
                }
                System.out.println("stats " + base);
            }
        }
        System.out.println("DONE");
    }

    /** Вариант: поза A, сглаживание B, наименьшие квадраты C. */
    record Var(String name, boolean a, boolean b, boolean c) {
        static Var of(String s) {
            return new Var(s, s.contains("A"), s.contains("B"), s.contains("C"));
        }

        boolean frames() {
            return b || c;
        }
    }

    /** Отбор и модель по варианту на FERET: su.rOf, su.tpl.rLimit, su.tpl.r0 – r′ и R′, r0′ (A) или прежние. */
    static void feretPose(XBaseA0.FeretSetup su, Var v, double rp, double r0p) {
        if (!v.a()) return;
        su.rOf.clear();
        for (Map.Entry<String, double[]> e : su.det.entrySet()) if (e.getValue() != null) su.rOf.put(e.getKey(), Math.abs(rPrime(e.getValue())));
        su.tpl.rLimit = rp;
        su.tpl.r0 = r0p;
    }

    /** VerifyMp2Ly.sets (без МП) с позой варианта: rB – |r| или |r′|. */
    static FeretSets feretSets(XBaseA0.FeretSetup su, Var v) throws IOException {
        FeretSets fs = VerifyMp2Ly.sets(su, new Thr(Double.POSITIVE_INFINITY, 0, 0, Double.NaN, 0, 0), true);
        if (v.a()) {
            // sets() считает rB по FeiDataset.pose; для А – заново с r′ и теми же ролями по отбору r′
            Map<String, Map<String, List<FeretData.Img>>> kept = new TreeMap<>();
            fs.rB.clear();
            fs.i.clear();
            for (FeretData.Img im : su.imgs) {
                double[] row = su.det.get(im.key());
                if (row == null) continue;
                double r = Math.abs(rPrime(row));
                fs.rB.put(im.key(), r);
                if (r <= su.tpl.rLimit) {
                    fs.i.add(im.key());
                    kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
                }
            }
            fs.roI = XBaseFeret.roles(XBaseFeret.dates(su.imgs), kept, su.tpl.seed);
            fs.ro = XBaseFeret.equalize(fs.roI, su.byKey);
            fs.pool = XBaseXsF.pool(fs.roI, su.byKey);
        }
        return fs;
    }

    static Mat feretFrame(XBaseA0.FeretSetup su, String k, Var v) throws IOException {
        if (!v.frames()) return FeretData.frame(su.dir, "a", k);
        Mat color = FeretData.decode(su.byKey.get(k).file());
        Mat f = frame(color, su.det.get(k), v.b(), v.c());
        color.release();
        return f;
    }

    static void writeBlock(Path bf, String ls) throws IOException {
        Path tmp = bf.resolveSibling(bf.getFileName() + ".tmp");
        Files.writeString(tmp, ls, StandardCharsets.UTF_8);
        Files.move(tmp, bf, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    /** Поза каждого ключа FERET в счёте: человек, номер снимка, r варианта (для выбора модели в разборе). */
    static String feretPoseLines(XBaseA0.FeretSetup su, Set<String> keys) {
        StringBuilder b = new StringBuilder();
        for (String k : keys) {
            double[] row = su.det.get(k);
            b.append(su.byKey.get(k).person()).append('\t').append(su.imageNo.get(k)).append('\t').append(FeiDataset.pose(row)).append('\t')
                    .append(rPrime(row)).append('\n');
        }
        return b.toString();
    }

    static void feret(SettingsStore settings, Var v, double rp, double r0p, Path old, Path out, boolean noCohort) throws IOException {
        Files.createDirectories(out.resolve("blocks"));
        Path dir = Paths.get(settings.loadFacesExportDir());
        FarExport.Loaded export = FarExport.read(dir);
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        long t0 = System.nanoTime();
        // половины пула – по пулу базы (у варианта А пул может отличаться: люди вне пула базы не входят, их число – в отчёт)
        XBaseA0.FeretSetup su0 = XBaseA0.feretSetup(dir, selLines, settings);
        FeretSets fs0 = feretSets(su0, Var.of("base"));
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        feretPose(su, v, rp, r0p);
        FeretSets fs = feretSets(su, v);
        XBaseEval.Gt tpl = su.tpl;
        List<String> perm = new ArrayList<>(fs0.pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        Set<String> outside = new TreeSet<>(fs.pool.keySet());
        outside.removeAll(fs0.pool.keySet());
        XBaseFeret.Roles ro = fs.ro;
        List<Map<String, List<List<String>>>> passPool = new ArrayList<>();
        for (int ps = 0; ps < 2; ps++) {
            Map<String, List<List<String>>> m = new TreeMap<>();
            for (String p : halves.get(1 - ps)) if (fs.pool.containsKey(p)) m.put(p, fs.pool.get(p));
            passPool.add(m);
        }
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        if (!noCohort) {
            for (String p : cohortP) {
                List<String> ks = new ArrayList<>(feiKeys.get(p).values());
                cohort.add(ks);
                cohortKeys.addAll(ks);
            }
        }
        TreeSet<String> ps0 = new TreeSet<>(fs.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseXsF.Conf c = XBaseMp2.confs(ro).get(0);
        if (!c.name().equals("межс.")) throw new IllegalStateException("ожидалась межс.");
        Set<String> used = new TreeSet<>();
        for (String p : c.own()) {
            if (!halves.get(0).contains(p) && !halves.get(1).contains(p)) continue;
            used.addAll(c.gal().get(p));
            used.addAll(c.pro().get(p));
        }
        ro.ctrlKeys.values().forEach(used::addAll);
        fs.pool.values().forEach(ds -> ds.forEach(used::addAll));
        StringBuilder r = new StringBuilder();
        r.append(String.format(Locale.ROOT, "VerifyRev feret, вариант %s%s: R = %.6f, r0 = %s; отобрано %d; своих (межс.) %d, контрольных %d, пул %d (вне пула базы %d);"
                + " снимков FERET в счёте %d.%n", v.name(), noCohort ? ", без когорты" : "", tpl.rLimit, tpl.r0, fs.i.size(), c.own().size(), ro.ctrl.size(),
                fs.pool.size(), outside.size(), used.size()));
        Map<String, Mat> fr = new LinkedHashMap<>();
        int n = 0;
        for (String k : used) {
            fr.put(k, feretFrame(su, k, v));
            if (++n % 500 == 0) System.out.printf(Locale.ROOT, "VerifyRev: кадры FERET %d / %d%n", n, used.size());
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        if (!cohortKeys.isEmpty()) fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);
        XBaseEval.Gt baseI = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        Map<String, String> xs2 = XBaseSubsp.blocks(old.resolve("xbase_feret_xs2_scores.tsv.gz"));
        StringBuilder lines = new StringBuilder();
        StringBuilder check = new StringBuilder();
        for (int f = 0; f < XBaseXsF.FEATS.length; f++) {
            String rowLabel = XBaseXsF.label(f, XBaseXsF.Kind.XS);
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = halves.get(ps);
                List<String> own = c.own().stream().filter(mine::contains).toList();
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    gal.put(p, c.gal().get(p));
                    pro.put(p, c.pro().get(p));
                }
                List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), passPool.get(ps), Set.of(), Map.of()).extra();
                List<List<String>> aug = new ArrayList<>(cohort);
                aug.addAll(extra);
                XBaseEval.Gt gI = XBaseFeret.withOwn(baseI, own);
                gI.cohort = aug;
                List<Probe> probes = XBaseEval.probes(gI, pro);
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                Path bf = out.resolve("blocks").resolve(String.format(Locale.ROOT, "feret_%s%s_%d_%d.tsv", v.name(), noCohort ? "_nc" : "", f, ps));
                String ls;
                if (Files.exists(bf)) {
                    ls = Files.readString(bf, StandardCharsets.UTF_8);
                } else {
                    long t1 = System.nanoTime();
                    ls = XBaseEval.fitLines(gI, xs[f], EnumSet.of(V.L1), gal, probes, parts, c.s());
                    writeBlock(bf, ls);
                    System.out.printf(Locale.ROOT, "VerifyRev: feret %s %s, проход %d (%.0f с)%n", v.name(), rowLabel, ps + 1, (System.nanoTime() - t1) / 1e9);
                }
                lines.append("# проход ").append(ps + 1).append(", конфигурация межс., строка ").append(rowLabel).append('\n').append(ls);
                if (v.name().equals("base") && !noCohort) {
                    String o = xs2.get("проход " + (ps + 1) + ", конфигурация межс., строка " + rowLabel);
                    boolean ok = o != null && XBaseSubsp.l1Rows(o).equals(XBaseSubsp.l1Rows(ls));
                    check.append(String.format(Locale.ROOT, "Сверка с feret-xs2: %s, проход %d – %s (%d строк)%n", rowLabel, ps + 1, ok ? "побайтно" : "НЕ СОВПАЛО",
                            XBaseSubsp.l1Rows(ls).size()));
                }
            }
        }
        r.append(check);
        String tag = "verify_rev_feret_" + v.name() + (noCohort ? "_nc" : "");
        XBaseOther.writeScores(out.resolve(tag + "_scores.tsv.gz"), "# FERET, VerifyRev " + v.name() + (noCohort ? " без когорты" : "")
                + "; блоки – проход, конфигурация, строка\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        Files.writeString(out.resolve(tag + "_pose.tsv"), feretPoseLines(su, used), StandardCharsets.UTF_8);
        r.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        Files.writeString(out.resolve(tag + ".txt"), r.toString(), StandardCharsets.UTF_8);
        System.out.print(r);
        System.out.println("DONE");
    }

    static void gt(SettingsStore settings, Var v, double rp, double r0p, Path oldSubsp, Path out, boolean noCohort) throws IOException {
        Files.createDirectories(out.resolve("blocks"));
        Path dir = Paths.get(settings.loadFacesExportDir());
        FarExport.Loaded export = FarExport.read(dir);
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        feretPose(su, v, rp, r0p);
        FeretSets fs = feretSets(su, v);
        double rLimit = v.a() ? rp : XBaseEval.poseLimit(selLines);
        double r0 = v.a() ? r0p : FeiPose.r0(settings.loadFacesFeiPoseR0(), "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rB = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rB.put(pk, Math.abs(v.a() ? rPrime(row) : FeiDataset.pose(row)));
        });
        Predicate<String> inI = pk -> rB.containsKey(pk) && rB.get(pk) <= rLimit;
        XBaseEval.Gt b = XBaseMp2.gtRoles(export, dir, feiKeys, settings, inI, rB, r0);
        b.rLimit = rLimit;
        long eqSeed = b.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(b.own, b.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(b.own, b.galleryAll, b.seed);
        Map<String, List<String>> galIn = XBaseEval.equalize(b.own, split.get(0), eqSeed);
        Map<String, Sample> sampleOf = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) for (Sample s : l) sampleOf.put(s.file().toString(), s);
        Object[][] modes = {{"межс.", galX, b.probeKeys}, {"внутри", galIn, split.get(1)}};
        List<List<String>> cohort0 = noCohort ? new ArrayList<>() : new ArrayList<>(b.cohort);
        Set<String> cohortKeys = new HashSet<>();
        cohort0.forEach(cohortKeys::addAll);
        Map<String, List<List<String>>> pool = fs.pool;
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        StringBuilder r = new StringBuilder();
        r.append(String.format(Locale.ROOT, "VerifyRev gt, вариант %s%s: R = %.6f, r0 = %s; снимков GT %d, отобрано %d; своих %d, контрольных %d; пул FERET %d людей.%n",
                v.name(), noCohort ? ", без когорты" : "", rLimit, r0, det.size(), rB.keySet().stream().filter(inI).count(), b.own.size(), b.ctrl.size(), pool.size()));
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : b.samples.keySet()) {
            Sample s = b.samples.get(k);
            String pk = s.person() + "/" + s.file().getFileName().toString();
            if (v.frames()) {
                Mat color = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                if (color.empty()) throw new IOException("нет снимка GT " + pk);
                fr.put(k, frame(color, det.get(pk), v.b(), v.c()));
                color.release();
            } else {
                fr.putAll(XBaseEval.frames(export, List.of(s), null));
            }
        }
        if (!cohortKeys.isEmpty()) {
            List<Sample> fei = new ArrayList<>();
            export.data().fei().values().forEach(fei::addAll);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        }
        for (String k : poolKeys) fr.put(k, feretFrame(su, k, v));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), b.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), b.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);
        Map<String, String> old = XBaseSubsp.blocks(oldSubsp);
        List<List<String>> parts = FeiMethods.cvParts(b.own, b.seed, XBaseEval.CV_S);
        String[][] projs = {{"L1-xs-F", "0"}, {"L1-tt-xs-F", "1"}};
        StringBuilder lines = new StringBuilder();
        StringBuilder check = new StringBuilder();
        List<List<String>> aug = new ArrayList<>(cohort0);
        aug.addAll(poolCls);
        b.cohort = aug;
        for (Object[] md : modes) {
            String mname = (String) md[0];
            // без когорты – только проверка «Б − база» на GT межс., L1-xs-F (Жорик №7)
            if (noCohort && !mname.equals("межс.")) continue;
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
            List<Probe> probes = XBaseEval.probes(b, pro);
            for (String[] pj : projs) {
                int f = Integer.parseInt(pj[1]);
                if (noCohort && f != 0) continue;
                Path bf = out.resolve("blocks").resolve(String.format(Locale.ROOT, "gt_%s%s_%s_%d.tsv", v.name(), noCohort ? "_nc" : "", mname.equals("межс.") ? "x" : "i", f));
                String ls;
                if (Files.exists(bf)) {
                    ls = Files.readString(bf, StandardCharsets.UTF_8);
                } else {
                    long t1 = System.nanoTime();
                    ls = XBaseEval.fitLines(b, xs[f], EnumSet.of(V.L1), gal, probes, parts, 0);
                    writeBlock(bf, ls);
                    System.out.printf(Locale.ROOT, "VerifyRev: gt %s %s %s (%.0f с)%n", v.name(), mname, pj[0], (System.nanoTime() - t1) / 1e9);
                }
                lines.append("# ").append(mname).append(", проекция ").append(pj[0]).append('\n').append(ls);
                if (v.name().equals("base") && !noCohort) {
                    String o = old.get(mname + ", проекция " + pj[0] + ", k = 0");
                    boolean ok = o != null && XBaseSubsp.l1Rows(o).equals(XBaseSubsp.l1Rows(ls));
                    check.append(String.format(Locale.ROOT, "Сверка с gt-subsp k = 0: %s %s – %s (%d строк)%n", mname, pj[0], ok ? "побайтно" : "НЕ СОВПАЛО",
                            XBaseSubsp.l1Rows(ls).size()));
                }
            }
        }
        r.append(check);
        StringBuilder pose = new StringBuilder();
        det.forEach((pk, row) -> {
            if (row != null) pose.append(pk.substring(0, pk.indexOf('/'))).append('\t').append(pk.substring(pk.indexOf('/') + 1)).append('\t')
                    .append(FeiDataset.pose(row)).append('\t').append(rPrime(row)).append('\n');
        });
        String tag = "verify_rev_gt_" + v.name() + (noCohort ? "_nc" : "");
        XBaseOther.writeScores(out.resolve(tag + "_scores.tsv.gz"), "# GT, VerifyRev " + v.name() + (noCohort ? " без когорты" : "")
                + "; блоки – постановка, проекция\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        Files.writeString(out.resolve(tag + "_pose.tsv"), pose.toString(), StandardCharsets.UTF_8);
        r.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        Files.writeString(out.resolve(tag + ".txt"), r.toString(), StandardCharsets.UTF_8);
        System.out.print(r);
        System.out.println("DONE");
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        SettingsStore settings = new SettingsStore();
        switch (a[0]) {
            case "stats" -> stats(settings, Paths.get(a[1]), Paths.get(a[2]));
            case "feret" -> feret(settings, Var.of(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3]), Paths.get(a[4]), Paths.get(a[5]),
                    a.length > 6 && a[6].equals("nocohort"));
            case "gt" -> gt(settings, Var.of(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3]), Paths.get(a[4]), Paths.get(a[5]),
                    a.length > 6 && a[6].equals("nocohort"));
            default -> throw new IllegalArgumentException(a[0]);
        }
    }
}
