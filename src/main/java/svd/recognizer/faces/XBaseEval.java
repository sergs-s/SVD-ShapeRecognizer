package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * PLAN.md п. 5.5, сверка на других базах, шаг 2: L1, L1-in и A (опора) на базе, не FEI, с умолчаниями FEI без подбора
 * (кадр «а», r0 = 0,12, зеркала, LBP u2 (8, 2), 7×7, маска L1-in, MLDA, d₁/d₂, порог П2 / П3 по галерее). Сейчас — Georgia
 * Tech (решения Хозяина 04.10.2026):
 * <ul>
 *   <li>отбор снимков по позе — как FEI: |r| ≤ R (R — из fei_selection.tsv);</li>
 *   <li>свои — люди с ≥ 2 датами съёмки (EXIF, reports/gt_sessions.tsv экспорта); галерея — дата с наибольшим числом
 *       снимков (при равенстве — более ранняя), пробы — все снимки остальных дат;</li>
 *   <li>контрольные чужие — люди с одной датой, все их снимки;</li>
 *   <li>когорта MLDA — встроенная: фиксированные 20 «полных» людей FEI (перестановка seed + COHORT_SEED_OFFSET), все
 *       их отобранные снимки;</li>
 *   <li>порог: П2 и П3 (FeiThreshold) по «чужим» галереи — 5 частей своих (FeiMethods.cvParts, s = CV_S), модель без
 *       части, «чужие» — только снимки галереи людей части;</li>
 *   <li>бутстреп по людям базы (когорта FEI не перевыбирается), B = faces.fei.bootstrap, seed faces.seed + 7000.</li>
 * </ul>
 * Режим fei-check: тот же путь построения модели и оценок для L1 на конфигурации 0 FEI — строки оценок должны совпасть с
 * файлом оценок faces-fei (fei_630f21a).
 *
 * Запуск: {@code java … XBaseEval gt} или {@code fei-check <файл оценок FEI>} при заданных faces.export.dir,
 * faces.fei.pose.r0; отчёт reports/faces/xbase/xbase_gt.txt, оценки — xbase_gt_scores.tsv.gz.
 *
 * @author ssv
 */
public final class XBaseEval {

    static final String DIR = "xbase";
    /** Seed встроенной когорты FEI: faces.seed + COHORT_SEED_OFFSET. */
    static final long COHORT_SEED_OFFSET = 9000;
    static final int COHORT = 20;
    /** Части П2 на GT: FeiMethods.cvParts(своих, seed, CV_S) — seed faces.seed + 8000 + 100. */
    static final int CV_S = 100;
    static final String SESSIONS = "reports/gt_sessions.tsv";
    static final String OWN = FeiScores.OWN;
    static final String CTRL = FeiScores.CTRL;

    /** Варианты: метод (имя строки файла оценок), метка, вход. */
    enum V {
        A("2c-mlda", "A"), L1(FeiMethods.LBP_MLDA, "L1"), L1IN(FeiMethods.LBP_MLDA_IN, "L1-in"),
        /** Диагностика GT (gt-diag): L0 — χ² до ближайшего снимка (как 5-lbp-chi FEI), L0-mean — χ² до средней гистограммы. */
        L0(FeiMethods.LBP_CHI, "L0"), L0MEAN(FeiMethods.LBP_CHI + "-mean", "L0-mean"),
        /** Диагностика GT: SFace — справочный предел (как 0-sface FarMethods), без зеркал и выбора по позе. */
        SFACE("0-sface", "SFace"),
        /**
         * Проверка выравнивания (gt-crop): L1 на кадре alignCrop 112×112 (sface экспорта, в сером) и на его центральной части
         * 92×112 (столбцы 10–101 — геометрия кадра «а»).
         */
        L1CROP(FeiMethods.LBP_MLDA + "-crop", "L1-crop"), L1CROP92(FeiMethods.LBP_MLDA + "-crop92", "L1-crop-92"),
        /**
         * LTP (gt-ltp, PLAN.md п. 4.2, «LTP – устойчивость LBP к шуму»): корень гистограмм LTP (LtpFeature) по выходу Tan –
         * Triggs, всё остальное — как у L1; LTP-in-tt — внутренние блоки 1–5 в каждой половине.
         */
        LTP("5-ltp-mlda", "LTP-tt"), LTPIN("5-ltp-mlda-in", "LTP-in-tt");

        final String id;
        final String label;

        V(String id, String label) {
            this.id = id;
            this.label = label;
        }

        String ratioId() {
            return id + "-ratio+pose";
        }
    }

    /** Векторы снимков (ключ — Sample.file().toString()): [вариант] исходные и отражённые. */
    final Map<V, Map<String, double[]>> vec = new HashMap<>();
    final Map<V, Map<String, double[]>> mir = new HashMap<>();
    /** Наименьшее число снимков шаблона SFace (SFaceScorer): 2 — как прежде; FERET — 1. */
    int sfaceMin = 2;

    /** Векторы вариантов vs по кадрам (OpenCV — только здесь, в главном потоке); LBP — без нормировки освещения. */
    void vectors(Map<String, Mat> frames, Set<V> vs, IlluminationNorm.NormParams params) {
        vectors(frames, vs, params, IlluminationNorm.NONE);
    }

    /**
     * То же с нормировкой освещения перед LBP lbpNorm (faces.lbp.norm; у отражённых кадров — после отражения и повторного
     * выравнивания); A — всегда CLAHE по пикселям, от lbpNorm не зависит.
     */
    void vectors(Map<String, Mat> frames, Set<V> vs, IlluminationNorm.NormParams params, IlluminationNorm lbpNorm) {
        for (V v : vs) {
            vec.put(v, new HashMap<>());
            mir.put(v, new HashMap<>());
        }
        for (Map.Entry<String, Mat> e : frames.entrySet()) {
            Mat m = FaceMirror.mirror(e.getValue());
            if (vs.contains(V.A)) {
                vec.get(V.A).put(e.getKey(), IlluminationNorm.CLAHE.vector(e.getValue(), params));
                mir.get(V.A).put(e.getKey(), IlluminationNorm.CLAHE.vector(m, params));
            }
            if (vs.contains(V.L1) || vs.contains(V.L1IN)) {
                double[] s = LbpFeature.sqrt(LbpFeature.histogram(e.getValue(), lbpNorm, params));
                double[] sm = LbpFeature.sqrt(LbpFeature.histogram(m, lbpNorm, params));
                if (vs.contains(V.L1)) {
                    vec.get(V.L1).put(e.getKey(), s);
                    mir.get(V.L1).put(e.getKey(), sm);
                }
                if (vs.contains(V.L1IN)) {
                    vec.get(V.L1IN).put(e.getKey(), LbpFeature.select(s, LbpFeature.Mask.INNER));
                    mir.get(V.L1IN).put(e.getKey(), LbpFeature.select(sm, LbpFeature.Mask.INNER));
                }
            }
            if (vs.contains(V.L0) || vs.contains(V.L0MEAN)) {
                // L0 и L0-mean: гистограммы LBP без корня (как 5-lbp-chi FEI).
                double[] h = LbpFeature.histogram(e.getValue(), lbpNorm, params);
                double[] hm = LbpFeature.histogram(m, lbpNorm, params);
                for (V v : new V[] {V.L0, V.L0MEAN}) {
                    if (!vs.contains(v)) continue;
                    vec.get(v).put(e.getKey(), h);
                    mir.get(v).put(e.getKey(), hm);
                }
            }
            m.release();
        }
    }

    /** Столбец начала центральной части 92×112 кадра alignCrop 112×112 (кадр «а» — шаблон ArcFace со сдвигом −10 px). */
    static final int CROP92_X0 = (FaceAlignment.ARCFACE_SIZE - 92) / 2;

    /** Центральная часть 92×112 серого кадра alignCrop 112×112 (столбцы 10–101), копия. */
    static Mat crop92(Mat gray112) {
        Mat sub = gray112.submat(0, gray112.rows(), CROP92_X0, CROP92_X0 + 92);
        Mat out = sub.clone();
        sub.release();
        return out;
    }

    /**
     * Векторы L1-crop и L1-crop-92 (как L1: корень гистограмм LBP) по кадрам alignCrop 112×112 BGR: перевод в серый
     * (COLOR_BGR2GRAY); отражения — способ а (FaceMirror) с шаблоном ArcFace 112×112 для L1-crop и ownTemplate(92, 112)
     * для L1-crop-92 (отражается уже вырезанная часть).
     */
    void cropVectors(Map<String, Mat> sfaceFrames) {
        for (V v : new V[] {V.L1CROP, V.L1CROP92}) {
            vec.put(v, new HashMap<>());
            mir.put(v, new HashMap<>());
        }
        double[][] t92 = FaceAlignment.ownTemplate(92, 112);
        for (Map.Entry<String, Mat> e : sfaceFrames.entrySet()) {
            Mat g = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(e.getValue(), g, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);
            Mat gm = FaceMirror.mirror(g, FaceAlignment.ARCFACE_112);
            vec.get(V.L1CROP).put(e.getKey(), LbpFeature.sqrt(LbpFeature.histogram(g)));
            mir.get(V.L1CROP).put(e.getKey(), LbpFeature.sqrt(LbpFeature.histogram(gm)));
            Mat c = crop92(g);
            Mat cm = FaceMirror.mirror(c, t92);
            vec.get(V.L1CROP92).put(e.getKey(), LbpFeature.sqrt(LbpFeature.histogram(c)));
            mir.get(V.L1CROP92).put(e.getKey(), LbpFeature.sqrt(LbpFeature.histogram(cm)));
            for (Mat m : new Mat[] {g, gm, c, cm}) m.release();
        }
    }

    /**
     * Векторы LTP-tt и LTP-in-tt (gt-ltp): корень гистограмм LTP с порогом t по выходу Tan – Triggs (params) кадра «а»;
     * отражённые кадры — FaceMirror (способ а), Tan – Triggs после отражения и повторного выравнивания.
     */
    void ltpVectors(Map<String, Mat> frames, IlluminationNorm.NormParams params, double t) {
        for (V v : new V[] {V.LTP, V.LTPIN}) {
            vec.put(v, new HashMap<>());
            mir.put(v, new HashMap<>());
        }
        for (Map.Entry<String, Mat> e : frames.entrySet()) {
            Mat m = FaceMirror.mirror(e.getValue());
            double[] s = LbpFeature.sqrt(LtpFeature.histogramTt(e.getValue(), params, t));
            double[] sm = LbpFeature.sqrt(LtpFeature.histogramTt(m, params, t));
            vec.get(V.LTP).put(e.getKey(), s);
            mir.get(V.LTP).put(e.getKey(), sm);
            vec.get(V.LTPIN).put(e.getKey(), LtpFeature.select(s, LbpFeature.Mask.INNER));
            mir.get(V.LTPIN).put(e.getKey(), LtpFeature.select(sm, LbpFeature.Mask.INNER));
            m.release();
        }
    }

    /** Признаки SFace (вариант SFACE) по кадрам alignCrop 112×112 (ключ — Sample.file().toString()); отражений нет. */
    void sfaceVectors(Map<String, Mat> sfaceFrames, org.opencv.objdetect.FaceRecognizerSF sface) {
        Map<String, double[]> v = new HashMap<>();
        for (Map.Entry<String, Mat> e : sfaceFrames.entrySet()) v.put(e.getKey(), GalleryEvaluation.vector(e.getValue(), Variant.SFACE, sface));
        vec.put(V.SFACE, v);
        mir.put(V.SFACE, new HashMap<>());
    }

    /**
     * Модель MLDA варианта v: классы — свои (ключи галереи; при mirror — затем их отражения), затем когорта (все ключи;
     * при mirror — затем отражения); как FeiMethods.classes / mirrorClasses (+mirror, Mirror.ALL).
     */
    GalleryScorer.ScoreModel fit(V v, boolean mirror, List<List<String>> gallery, List<List<String>> cohort) {
        if (v == V.SFACE && mirror) throw new IllegalArgumentException("SFace: модели +mirror нет");
        List<List<double[]>> cls = new ArrayList<>();
        // L0, L0-mean, SFace — без обучения: когорта не нужна (оценщик берёт только классы галереи).
        boolean mlda = v == V.A || v == V.L1 || v == V.L1IN || v == V.L1CROP || v == V.L1CROP92 || v == V.LTP || v == V.LTPIN;
        for (List<List<String>> group : mlda ? List.of(gallery, cohort) : List.of(gallery)) {
            for (List<String> keys : group) {
                List<double[]> list = new ArrayList<>();
                for (String k : keys) list.add(vec.get(v).get(k));
                if (mirror) for (String k : keys) list.add(mir.get(v).get(k));
                cls.add(list);
            }
        }
        GalleryScorer scorer = switch (v) {
            case L0 -> new ChiSquareScorer(false);
            case L0MEAN -> new ChiSquareScorer(true);
            case SFACE -> new SFaceScorer(sfaceMin);
            default -> new MldaScorer();
        };
        return scorer.fit(cls, gallery.size());
    }

    /** Попытка: роль, человек, снимок, ключ, индекс истинного в галерее (−1 — чужой). */
    record Probe(String role, String person, int image, String key, int trueIdx) {}

    /** Строки оценок проб модели (FeiScores.line). */
    String lines(String method, int s, int j, GalleryScorer.ScoreModel m, V v, List<String> gallery, List<Probe> probes) {
        StringBuilder t = new StringBuilder();
        for (Probe p : probes) {
            t.append(FeiScores.line(method, s, j, p.role(), p.person(), p.image(), p.trueIdx(), gallery, m.scores(vec.get(v).get(p.key())), false));
        }
        return t.toString();
    }

    /** Задачи выполняются в threads потоках (только чистая Java), результаты — в порядке задач. */
    static List<String> run(List<Callable<String>> tasks, int threads) {
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        List<String> out = new ArrayList<>();
        try {
            for (Future<String> f : ex.invokeAll(tasks)) out.add(f.get());
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseEval: " + e, e);
        } finally {
            ex.shutdownNow();
        }
        return out;
    }

    // ---------------------------------------------------------------- данные

    /** r по строке detections.tsv: «база/человек/имя» → r. */
    static Map<String, Double> poses(Path detections) throws IOException {
        Map<String, Double> out = new HashMap<>();
        for (String l : Files.readAllLines(detections, StandardCharsets.UTF_8)) {
            if (l.startsWith("base\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            if (!c[4].equals("1")) continue;
            double[] row = new double[15];
            for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[5 + i]);
            out.put(c[0] + "/" + c[1] + "/" + c[2], FeiDataset.pose(row));
        }
        return out;
    }

    /** Порог позы R отбора FEI (строка «# R» fei_selection.tsv). */
    static double poseLimit(List<String> selLines) {
        for (String l : selLines) if (l.startsWith("# R\t")) return Double.parseDouble(l.substring(4).trim());
        throw new IllegalStateException("В fei_selection.tsv нет строки # R");
    }

    /** Сессии GT: «человек/снимок» → номер даты (0 — без даты). */
    static Map<String, Integer> sessions(Path file) throws IOException {
        Map<String, Integer> out = new HashMap<>();
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            out.put(c[0] + "/" + c[1], c[3].equals("-") ? 0 : Integer.parseInt(c[3]));
        }
        return out;
    }

    /**
     * Дата галереи своего: наибольшее число снимков, при равенстве — более ранняя (меньший номер); снимки без даты не
     * учитываются.
     */
    static int galleryDate(Map<Integer, List<String>> byDate) {
        int best = -1;
        for (Map.Entry<Integer, List<String>> e : byDate.entrySet()) {
            if (e.getKey() == 0) continue;
            if (best < 0 || e.getValue().size() > byDate.get(best).size()) best = e.getKey();
        }
        return best;
    }

    /** Встроенная когорта: 20 «полных» людей FEI (перестановка seed + COHORT_SEED_OFFSET), по возрастанию. */
    static List<String> cohortPersons(Map<String, Map<Integer, String>> feiKeys, long seed) {
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new HashSet<>(m.keySet())));
        List<String> full = new ArrayList<>(FeiProtocol.fullPersons(included));
        java.util.Collections.shuffle(full, new java.util.Random(seed + COHORT_SEED_OFFSET));
        List<String> out = new ArrayList<>(full.subList(0, COHORT));
        java.util.Collections.sort(out);
        return out;
    }

    // ---------------------------------------------------------------- запуск

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        String[] a = args.length == 1 ? args[0].trim().split("\\s+") : args;
        if (a.length < 1) throw new IllegalArgumentException("XBaseEval: gt | fei-check <файл оценок FEI>");
        SettingsStore settings = new SettingsStore();
        String exportDir = settings.loadFacesExportDir();
        if (exportDir == null) throw new IllegalStateException("XBaseEval: не задан faces.export.dir / FACES_EXPORT_DIR");
        Path dir = Paths.get(exportDir);
        FarExport.Loaded export = FarExport.read(dir);
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", DIR);
        Files.createDirectories(out);
        long t0 = System.nanoTime();
        if (a[0].equals("fei-check")) {
            String text = feiCheck(export, feiKeys, settings, Paths.get(a[1]));
            FarMethods.write(out.resolve("xbase_fei_check.txt"), text);
            System.out.print(text);
        } else if (a[0].equals("gt")) {
            gt(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-in")) {
            gtIn(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-diag")) {
            gtDiag(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-crop")) {
            gtCrop(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-xs")) {
            XBaseXs.run(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-quality")) {
            XBaseQuality.run(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-ltp")) {
            XBaseLtp.run(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("feret-eyes")) {
            XBaseFeretEyes.run(export, dir, feiKeys, selLines, settings);
        } else if (a[0].equals("feret-excl")) {
            XBaseExcl.feret(export, dir, feiKeys, selLines, settings);
        } else if (a[0].equals("feret-eyes2")) {
            XBaseFeretEyes.run(export, dir, feiKeys, selLines, settings, XBaseFeretEyes.EYES2);
        } else if (a[0].equals("feret-eyepts")) {
            XBaseEyePts.run(dir, selLines, settings);
        } else if (a[0].equals("feret-hubs")) {
            XBaseHubs.run(export, dir, feiKeys, selLines, settings);
        } else if (a[0].equals("gt-excl")) {
            XBaseExcl.gt(export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("feret")) {
            XBaseFeret.run(export, dir, feiKeys, selLines, settings);
        } else if (a[0].equals("muct") || a[0].equals("orl")) {
            XBaseOther.run(a[0], export, dir, feiKeys, selLines, settings, out);
        } else if (a[0].equals("gt-sep-ltp")) {
            XBaseSepLtp.run(dir, settings, out);
        } else if (a[0].equals("gt-sep")) {
            XBaseSep.run(dir, selLines, settings, out);
        } else {
            throw new IllegalArgumentException("XBaseEval: неизвестный режим " + a[0]);
        }
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "XBaseEval %s: %.0f с", a[0],
                (System.nanoTime() - t0) / 1e9));
    }

    static Map<String, Mat> frames(FarExport.Loaded export, Iterable<Sample> samples, Set<String> keys) throws IOException {
        Map<String, Mat> out = new LinkedHashMap<>();
        for (Sample s : samples) {
            String k = s.file().toString();
            if (keys != null && !keys.contains(k)) continue;
            Mat f = export.frame(s, Variant.A);
            if (f == null) throw new IllegalStateException("Нет кадра снимка: " + s.file());
            out.put(k, f);
        }
        return out;
    }

    /** Контроль пути: L1 на конфигурации 0 FEI — строки оценок = строки файла оценок faces-fei (5-lbp-mlda и +mirror). */
    static String feiCheck(FarExport.Loaded export, Map<String, Map<Integer, String>> feiKeys, SettingsStore settings, Path scores)
            throws IOException {
        long seed = settings.loadFacesSeed();
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new HashSet<>(m.keySet())));
        FeiProtocol.Split sp = FeiProtocol.splits(included, seed).get(0);
        int j = 0;
        List<Sample> all = new ArrayList<>();
        export.data().fei().values().forEach(all::addAll);
        XBaseEval x = new XBaseEval();
        Map<String, Mat> fr = frames(export, all, null);
        x.vectors(fr, Set.of(V.L1), IlluminationNorm.NormParams.of(settings));
        fr.values().forEach(Mat::release);
        List<List<String>> gallery = new ArrayList<>();
        for (String p : sp.gallery()) gallery.add(FeiProtocol.trainNumbers(j).stream().map(n -> feiKeys.get(p).get(n)).toList());
        List<List<String>> cohort = new ArrayList<>();
        for (String p : sp.cohort()) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
        List<Probe> probes = new ArrayList<>();
        int n = FeiProtocol.NUMBERS[j];
        for (int i = 0; i < sp.gallery().size(); i++) {
            String p = sp.gallery().get(i);
            probes.add(new Probe(OWN, p, n, feiKeys.get(p).get(n), i));
        }
        for (String role : new String[] {FeiScores.THR, CTRL}) {
            for (String p : role.equals(FeiScores.THR) ? sp.threshold() : sp.control()) {
                for (Map.Entry<Integer, String> e : feiKeys.get(p).entrySet()) probes.add(new Probe(role, p, e.getKey(), e.getValue(), -1));
            }
        }
        List<Callable<String>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> x.lines(method, sp.index(), j, x.fit(V.L1, m, gallery, cohort), V.L1, sp.gallery(), probes));
        }
        List<String> got = run(tasks, 2);
        StringBuilder t = new StringBuilder("XBaseEval fei-check: L1 (5-lbp-mlda и +mirror), конфигурация 0 FEI, путь XBaseEval против файла оценок\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл оценок: ").append(scores).append('\n');
        boolean ok = true;
        for (int i = 0; i < 2; i++) {
            String method = V.L1.id + (i == 1 ? FeiMethods.MIRROR : "");
            List<String> ref = new ArrayList<>();
            try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(scores), 1 << 16),
                    StandardCharsets.UTF_8), 1 << 16)) {
                String l;
                while ((l = in.readLine()) != null) if (l.startsWith(method + "\t" + sp.index() + "\t" + j + "\t")) ref.add(l);
            }
            List<String> mine = got.get(i).lines().toList();
            boolean same = mine.equals(ref);
            ok &= same;
            t.append(String.format(Locale.ROOT, "%s: строк %d, в файле оценок %d — %s%n", method, mine.size(), ref.size(), same ? "совпадают" : "РАЗЛИЧАЮТСЯ"));
        }
        t.append(ok ? "Итог: совпадают.\n" : "Итог: РАЗЛИЧАЮТСЯ.\n");
        if (!ok) throw new IllegalStateException(t.toString());
        return t.toString();
    }

    // ---------------------------------------------------------------- Georgia Tech

    /** Подготовленные данные GT: отбор по позе, роли, межсессионные галерея и пробы, когорта, векторы. */
    static final class Gt {
        long seed;
        double r0;
        int b;
        int threads;
        double rLimit;
        Path dir;
        Map<String, Map<Integer, List<String>>> byPerson = new TreeMap<>();
        Map<String, Sample> samples = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>();
        int total;
        List<String> dropped = new ArrayList<>();
        List<String> own = new ArrayList<>();
        List<String> ctrl = new ArrayList<>();
        List<String> persons;
        Map<String, Integer> pIdx = new HashMap<>();
        Map<String, List<String>> galleryKeys = new LinkedHashMap<>();
        Map<String, Integer> galleryDateOf = new LinkedHashMap<>();
        Map<String, List<String>> probeKeys = new LinkedHashMap<>();
        List<String> cohortP;
        List<List<String>> cohort = new ArrayList<>();
        XBaseEval x;
        /** Нормировки перед LBP кроме none (faces.lbp.norm, в порядке ключа) и их векторы L1, L1-in. */
        IlluminationNorm.NormParams params;
        final Map<IlluminationNorm, XBaseEval> xn = new LinkedHashMap<>();
        /** gt-diag: модель SFace (путь и SHA-256). */
        String sfaceModel;
        /** Номер снимка по ключу (MUCT, ORL — XBaseOther); null — цифры имени файла (GT). */
        Map<String, Integer> imageNo;

        int img(String key) {
            return imageNo != null ? imageNo.get(key) : image(samples.get(key));
        }
    }

    /** Итоги одного протокола: строки оценок, итоги вариантов и сведения для отчёта. */
    static final class Eval {
        final List<FeiThreshold.Res> results = new ArrayList<>();
        String lines;
        List<Probe> probes;
        List<List<String>> parts;
        int ctrlImages;
        int cvImages;
        final int[] cases = new int[FeiScores.CASES];
        double fitS;
    }

    static Gt prepare(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        return prepare(export, dir, feiKeys, selLines, settings, false, false);
    }

    /** То же; diag — ещё векторы L0, L0-mean и SFace (gt-diag); crop — векторы L1-crop и L1-crop-92 (gt-crop). */
    static Gt prepare(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                      boolean diag, boolean crop) throws IOException {
        Gt g = new Gt();
        g.dir = dir;
        g.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        g.r0 = FeiPose.r0(rawR0, "+pose");
        g.b = settings.loadFacesFeiBootstrap();
        g.threads = settings.loadFacesFeiCvThreads();
        g.rLimit = poseLimit(selLines);
        Map<String, Double> pose = poses(dir.resolve("detections.tsv"));
        Map<String, Integer> sess = sessions(dir.resolve(SESSIONS));

        // Снимки GT после отбора по позе: человек → дата → ключи (по номеру снимка).
        for (Map.Entry<String, List<Sample>> e : new TreeMap<>(export.data().gt()).entrySet()) {
            List<Sample> list = new ArrayList<>(e.getValue());
            list.sort((x, y) -> x.file().getFileName().toString().compareTo(y.file().getFileName().toString()));
            for (Sample s : list) {
                g.total++;
                String name = s.file().getFileName().toString();
                Double r = pose.get("gt/" + s.person() + "/" + name);
                if (r == null) throw new IllegalStateException("Нет позы снимка GT " + s.file());
                String k = s.file().toString();
                if (Math.abs(r) > g.rLimit) {
                    g.dropped.add(s.person() + "/" + name);
                    continue;
                }
                Integer d = sess.get(s.person() + "/" + name);
                if (d == null) throw new IllegalStateException("Нет сессии снимка GT " + s.file());
                g.byPerson.computeIfAbsent(s.person(), x -> new TreeMap<>()).computeIfAbsent(d, x -> new ArrayList<>()).add(k);
                g.samples.put(k, s);
                g.rOf.put(k, Math.abs(r));
            }
        }
        // Свои — ≥ 2 датами по всем снимкам (до отбора), как в инвентаризации.
        Map<String, Set<Integer>> datesAll = new TreeMap<>();
        sess.forEach((pk, d) -> {
            if (d > 0) datesAll.computeIfAbsent(pk.substring(0, pk.indexOf('/')), x -> new HashSet<>()).add(d);
        });
        for (String p : g.byPerson.keySet()) (datesAll.getOrDefault(p, Set.of()).size() >= 2 ? g.own : g.ctrl).add(p);
        g.persons = new ArrayList<>(g.byPerson.keySet());
        for (int i = 0; i < g.persons.size(); i++) g.pIdx.put(g.persons.get(i), i);

        for (String p : g.own) {
            Map<Integer, List<String>> bd = g.byPerson.get(p);
            int gd = galleryDate(bd);
            g.galleryDateOf.put(p, gd);
            g.galleryKeys.put(p, bd.get(gd));
            List<String> pr = new ArrayList<>();
            bd.forEach((d, ks) -> {
                if (d != gd) pr.addAll(ks);
            });
            pr.sort(null);
            if (pr.isEmpty()) throw new IllegalStateException("У своего " + p + " после отбора нет проб");
            g.probeKeys.put(p, pr);
        }
        g.cohortP = cohortPersons(feiKeys, g.seed);
        Set<String> cohortKeySet = new HashSet<>();
        for (String p : g.cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            g.cohort.add(ks);
            cohortKeySet.addAll(ks);
        }

        // Векторы: GT после отбора и когорта FEI.
        g.x = new XBaseEval();
        Map<String, Mat> fr = frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(frames(export, fei, cohortKeySet));
        g.params = IlluminationNorm.NormParams.of(settings);
        g.x.vectors(fr, BASE_VARIANTS, g.params);
        // Нормировки перед LBP (faces.lbp.norm): none — всегда (прежние отчёты), прочие — только L1 и L1-in.
        for (IlluminationNorm n : IlluminationNorm.lbpNorms(settings.loadFacesLbpNorm())) {
            if (n == IlluminationNorm.NONE) continue;
            XBaseEval xx = new XBaseEval();
            xx.vectors(fr, NORM_VARIANTS, g.params, n);
            g.xn.put(n, xx);
        }
        if (diag) {
            // L0 и L0-mean — в тот же набор векторов (A и L1 — общие с прежними отчётами); когорта им не нужна.
            Map<String, Mat> gtFr = new LinkedHashMap<>(fr);
            gtFr.keySet().retainAll(g.samples.keySet());
            XBaseEval h = new XBaseEval();
            h.vectors(gtFr, java.util.EnumSet.of(V.L0, V.L0MEAN), g.params);
            g.x.vec.putAll(h.vec);
            g.x.mir.putAll(h.mir);
            // SFace: кадры alignCrop 112×112 экспорта (sface/gt), модель faces.models.dir.
            Path model = Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
            g.sfaceModel = model + " (" + FeiThreshold.sha(model) + ")";
            org.opencv.objdetect.FaceRecognizerSF sface = org.opencv.objdetect.FaceRecognizerSF.create(model.toString(), "");
            Map<String, Mat> sf = new LinkedHashMap<>();
            for (Sample s : g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString())).toList()) {
                Mat c = export.frame(s, Variant.SFACE);
                if (c == null) throw new IllegalStateException("Нет кадра SFace снимка: " + s.file());
                sf.put(s.file().toString(), c);
            }
            g.x.sfaceVectors(sf, sface);
            sf.values().forEach(Mat::release);
        }
        if (crop) {
            // L1-crop и L1-crop-92: кадры alignCrop 112×112 экспорта (sface/gt и sface/fei когорты).
            Map<String, Mat> sf = new LinkedHashMap<>();
            for (Sample s : g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString())).toList()) {
                Mat c = export.frame(s, Variant.SFACE);
                if (c == null) throw new IllegalStateException("Нет кадра alignCrop снимка: " + s.file());
                sf.put(s.file().toString(), c);
            }
            for (Sample s : fei) {
                if (!cohortKeySet.contains(s.file().toString())) continue;
                Mat c = export.frame(s, Variant.SFACE);
                if (c == null) throw new IllegalStateException("Нет кадра alignCrop снимка когорты: " + s.file());
                sf.put(s.file().toString(), c);
            }
            g.x.cropVectors(sf);
            sf.values().forEach(Mat::release);
        }
        fr.values().forEach(Mat::release);
        return g;
    }

    /** Варианты отчёта по нормировкам перед LBP (A от нормировки не зависит и не нужна). */
    static final Set<V> NORM_VARIANTS = java.util.EnumSet.of(V.L1, V.L1IN);
    /** Варианты прежних отчётов gt и gt-in. */
    static final Set<V> BASE_VARIANTS = java.util.EnumSet.of(V.A, V.L1, V.L1IN);
    /** Варианты диагностики gt-diag (PLAN.md п. 4.2, «Диагностика L0 и SFace»): A и L1 справочно, L0, L0-mean, SFace. */
    static final Set<V> DIAG_VARIANTS = java.util.EnumSet.of(V.A, V.L1, V.L0, V.L0MEAN, V.SFACE);
    /** Варианты проверки выравнивания gt-crop (PLAN.md п. 4.2, «Проверка выравнивания и разделимости на GT»). */
    static final Set<V> CROP_VARIANTS = java.util.EnumSet.of(V.L1, V.L1CROP, V.L1CROP92);

    /** Протокол с галереей galleryKeys и пробами своих probeKeys (контроль, когорта, части П2 — общие). */
    static Eval evaluate(Gt gt, Map<String, List<String>> galleryKeys, Map<String, List<String>> probeKeys) {
        return evaluate(gt, gt.x, BASE_VARIANTS, galleryKeys, probeKeys);
    }

    /** То же для векторов x и вариантов vs (по порядку V). */
    static Eval evaluate(Gt gt, XBaseEval x, Set<V> vs, Map<String, List<String>> galleryKeys, Map<String, List<String>> probeKeys) {
        Eval ev = new Eval();
        List<Probe> mainProbes = probes(gt, probeKeys);
        List<List<String>> parts = FeiMethods.cvParts(gt.own, gt.seed, CV_S);
        long t1 = System.nanoTime();
        ev.lines = fitLines(gt, x, vs, galleryKeys, mainProbes, parts, 0);
        ev.fitS = (System.nanoTime() - t1) / 1e9;
        ev.probes = mainProbes;
        ev.parts = parts;
        Parsed p = parse(gt, vs, galleryKeys, mainProbes, parts, ev.lines);
        for (V v : vs) {
            ev.results.add(FeiThreshold.evaluate(v.label, v.ratioId(), List.of(p.cfg.get(v)), List.of(p.imp.get(v)), gt.persons.size(), gt.b,
                    gt.seed + FeiBootstrap.SEED_OFFSET));
        }
        ev.ctrlImages = p.ctrlImages;
        ev.cvImages = p.cvImages;
        System.arraycopy(p.cases, 0, ev.cases, 0, ev.cases.length);
        return ev;
    }

    /** Пробы основного протокола: свои (пробы probeKeys) и контрольные чужие (все снимки gt.ctrl); свои первыми, по ключу. */
    static List<Probe> probes(Gt gt, Map<String, List<String>> probeKeys) {
        List<Probe> mainProbes = new ArrayList<>();
        for (int i = 0; i < gt.own.size(); i++) {
            String p = gt.own.get(i);
            for (String k : probeKeys.get(p)) mainProbes.add(new Probe(OWN, p, gt.img(k), k, i));
        }
        for (String p : gt.ctrl) for (List<String> ks : gt.byPerson.get(p).values()) for (String k : ks) mainProbes.add(new Probe(CTRL, p, gt.img(k), k, -1));
        mainProbes.sort((u, w) -> u.role().equals(w.role()) ? u.key().compareTo(w.key()) : u.role().equals(OWN) ? -1 : 1);
        return mainProbes;
    }

    /**
     * Строки оценок вариантов vs: модель на всей галерее (пробы mainProbes) и модели без части q (пробы — снимки галереи людей
     * части, роль cv&lt;q&gt;); s — номер конфигурации в строках файла оценок.
     */
    static String fitLines(Gt gt, XBaseEval x, Set<V> vs, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts,
                           int s) {
        List<String> own = gt.own;
        List<List<String>> cohort = gt.cohort;
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        List<Callable<String>> tasks = new ArrayList<>();
        for (V v : vs) {
            for (boolean m : v == V.SFACE ? new boolean[] {false} : new boolean[] {false, true}) {
                String method = v.id + (m ? FeiMethods.MIRROR : "");
                tasks.add(() -> x.lines(method, s, 0, x.fit(v, m, fullGallery, cohort), v, own, mainProbes));
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                    List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                    List<Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                    tasks.add(() -> x.lines(method, s, 0, x.fit(v, m, gk, cohort), v, g, pr));
                }
            }
        }
        StringBuilder all = new StringBuilder();
        run(tasks, gt.threads).forEach(all::append);
        return all.toString();
    }

    /** Разбор строк одной конфигурации: [вариант] данные своих и контроля, «чужие» П2/П3; объём; случаи d₁/d₂ (L1). */
    static final class Parsed {
        final Map<V, FeiDecompose.Cfg> cfg = new HashMap<>();
        final Map<V, FeiThreshold.Imp> imp = new HashMap<>();
        int ctrlImages;
        int cvImages;
        final int[] cases = new int[FeiScores.CASES];
    }

    /** Разбор строк оценок: модель по позе пробы, конфигурация и «чужие» П2 / П3. */
    static Parsed parse(Gt gt, Set<V> vs, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, String lines) {
        Parsed out = new Parsed();
        Map<String, Row> rows = new HashMap<>();
        for (String l : lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        for (V v : vs) {
            FeiDecompose.Cfg c = new FeiDecompose.Cfg();
            List<double[]> ctrlList = new ArrayList<>();
            List<double[]> cvList = new ArrayList<>();
            int cvFront = 0;
            List<Boolean> front = new ArrayList<>();
            List<Row> ownRows = new ArrayList<>();
            for (Probe p : mainProbes) {
                Row r = choose(rows, v, p.role(), p.person(), p.image(), gt.rOf.get(p.key()) <= gt.r0);
                if (p.role().equals(OWN)) {
                    ownRows.add(r);
                    front.add(gt.rOf.get(p.key()) <= gt.r0);
                } else {
                    ctrlList.add(new double[] {gt.pIdx.get(p.person()), FeiScores.ratio(r, null)});
                }
            }
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                for (String p : parts.get(q)) {
                    for (String k : galleryKeys.get(p)) {
                        boolean f = gt.rOf.get(k) <= gt.r0;
                        if (f) cvFront++;
                        Row r = choose(rows, v, FeiMethods.CV_ROLE + q, p, gt.img(k), f);
                        cvList.add(new double[] {gt.pIdx.get(p), FeiScores.ratio(r, v == V.L1 ? out.cases : null)});
                    }
                }
            }
            int n = ownRows.size();
            c.ownP = new int[n];
            c.ownR = new double[n];
            c.ownOk = new boolean[n];
            c.noSecond = new boolean[n];
            c.trueInf = new boolean[n];
            c.poseFrontal = new boolean[n];
            for (int i = 0; i < n; i++) {
                Row r = ownRows.get(i);
                int[] cs = new int[FeiScores.CASES];
                c.ownP[i] = gt.pIdx.get(r.person());
                c.ownR[i] = FeiScores.ratio(r, cs);
                c.noSecond[i] = cs[FeiScores.CASE_NO_SECOND] > 0;
                c.trueInf[i] = Double.isInfinite(r.trueScore());
                c.ownOk[i] = r.bestId().equals(r.trueId());
                c.poseFrontal[i] = front.get(i);
            }
            c.thrP = new int[0];
            c.thrR = new double[0];
            c.ctrlP = ctrlList.stream().mapToInt(d -> (int) d[0]).toArray();
            c.ctrlR = ctrlList.stream().mapToDouble(d -> d[1]).toArray();
            out.ctrlImages = c.ctrlR.length;
            out.cvImages = cvList.size();
            out.cfg.put(v, c);
            out.imp.put(v, FeiThreshold.Imp.of(cvList, cvFront));
        }
        return out;
    }

    /** Файл оценок протокола (шапка — описание ролей, коммиты). */
    static void writeScores(Path file, String what, Gt g, Eval ev) throws IOException {
        String header = "# Сверка на других базах (XBaseEval, PLAN.md п. 5.5): Georgia Tech" + what + "; роли own — свой, ctrl — контрольный чужой, "
                + "cv<q> — снимок галереи человека части q против модели без части q\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт "
                + g.dir + ", коммит данных " + FarMethods.dataCommit(g.dir) + "\n" + FeiScores.COLUMNS + "\n";
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(file), 1 << 16), StandardCharsets.UTF_8)) {
            w.write(header);
            w.write(ev.lines);
        }
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                   Path out) throws IOException {
        Gt g = prepare(export, dir, feiKeys, selLines, settings);
        Eval ev = evaluate(g, g.galleryKeys, g.probeKeys);
        writeScores(out.resolve("xbase_gt_scores.tsv.gz"), "", g, ev);
        StringBuilder t = report(ev.results, g.own, g.ctrl, g.galleryKeys, g.galleryDateOf, g.probeKeys, g.byPerson, ev.parts, g.cohortP, g.cohort,
                g.rOf, g.r0, g.rLimit, g.total, g.dropped, ev.ctrlImages, ev.cvImages, ev.cases, g.b, g.seed, g.persons.size(), dir, ev.fitS,
                g.threads, ev.probes);
        FarMethods.write(out.resolve("xbase_gt.txt"), t);
        System.out.print(t);
        normReport(g, ev, g.galleryKeys, g.probeKeys, "gt", "", "межсессионно", LbpNormReport.Kind.CROSS, out);
    }

    /**
     * Отчёт «Нормировка освещения перед LBP» (PLAN.md, п. 4.2), если в faces.lbp.norm есть что-то кроме none: L1 и L1-in
     * каждой нормировки на том же протоколе (галерея gal, пробы pro; контроль, когорта, части П2 — общие), оценки — отдельный
     * файл на нормировку xbase_&lt;mode&gt;_&lt;id&gt;_scores.tsv.gz, отчёт xbase_&lt;mode&gt;_norm.txt; none — итоги none этого
     * же запуска (прежние отчёты не меняются); те же веса бутстрепа — разности парные.
     */
    static void normReport(Gt g, Eval none, Map<String, List<String>> gal, Map<String, List<String>> pro, String mode, String what, String title,
                           LbpNormReport.Kind kind, Path out) throws IOException {
        if (g.xn.isEmpty()) return;
        Map<IlluminationNorm, List<FeiThreshold.Res>> res = new LinkedHashMap<>();
        res.put(IlluminationNorm.NONE, List.of(none.results.get(V.L1.ordinal()), none.results.get(V.L1IN.ordinal())));
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: нормировка освещения перед LBP — Georgia Tech, ").append(title).append(" (XBaseEval ").append(mode.replace('_', '-'))
                .append(", faces.lbp.norm)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(g.dir).append(", коммит данных ").append(FarMethods.dataCommit(g.dir)).append('\n');
        t.append("Нормировки (решение Хозяина 09.10.2026; метод и параметры выбраны заранее, по результатам не подбираются; одна для всех кадров:\n"
                + "галерея, пробы, «чужие» П2/П3, контроль, когорта FEI, отражённые кадры — после отражения и повторного выравнивания):\n");
        t.append("  ").append(FeiMethods.lbpNormText(IlluminationNorm.NONE, g.params)).append(" (опорная; итоги — как в xbase_").append(mode).append(".txt)\n");
        for (IlluminationNorm n : g.xn.keySet()) t.append("  ").append(FeiMethods.lbpNormText(n, g.params)).append('\n');
        t.append("Варианты: L1 = 5-lbp-mlda-ratio+pose, L1-in = 5-lbp-mlda-in-ratio+pose; прочее — как в xbase_").append(mode).append(".txt (A не нужна).\n");
        int ownN = pro.values().stream().mapToInt(List::size).sum();
        t.append(String.format(Locale.ROOT, "Объём: пробы своих %d (%d человек), контроль %d снимков %d людей, «чужие» П2/П3 %d снимков %d людей.%n", ownN,
                g.own.size(), none.ctrlImages, g.ctrl.size(), none.cvImages, g.own.size()));
        if (g.b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d; одни веса для всех нормировок, строк и "
                    + "способов;\nпорог — заново в каждой реплике.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        for (Map.Entry<IlluminationNorm, XBaseEval> e : g.xn.entrySet()) {
            Eval ev = evaluate(g, e.getValue(), NORM_VARIANTS, gal, pro);
            String file = "xbase_" + mode + "_" + e.getKey().id + "_scores.tsv.gz";
            writeScores(out.resolve(file), what + " — нормировка перед LBP " + e.getKey().id, g, ev);
            res.put(e.getKey(), ev.results);
            t.append(String.format(Locale.ROOT, "%s: оценки — %s; обучение и оценки %.0f с; правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6): ", e.getKey().id, file,
                    ev.fitS));
            for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(ev.cases[k]);
            t.append('\n');
        }
        LbpNormReport.tables(t, res, g.b, kind);
        FarMethods.write(out.resolve("xbase_" + mode + "_norm.txt"), t);
        System.out.print(t);
    }

    // ---------------------------------------------------------------- GT: контроль «внутри сессии»

    /** Seed деления снимков даты галереи пополам: faces.seed + WITHIN_SEED_OFFSET. */
    static final long WITHIN_SEED_OFFSET = 9100;
    /** Люди GT для листов кадров: свои с этими номерами по порядку. */
    static final int[] SHEET_OWN = {0, 5, 10, 15, 20, 25};

    /**
     * Деление «внутри сессии» (решение Хозяина 04.10.2026): у каждого своего снимки даты галереи (то же правило) по порядку
     * людей перемешиваются одним java.util.Random(seed + WITHIN_SEED_OFFSET); первые ⌈n/2⌉ — галерея, остальные — пробы
     * (обе части по возрастанию ключа). Возвращает {галерея, пробы}.
     */
    static List<Map<String, List<String>>> withinSplit(List<String> own, Map<String, List<String>> dateKeys, long seed) {
        java.util.Random rnd = new java.util.Random(seed + WITHIN_SEED_OFFSET);
        Map<String, List<String>> gal = new LinkedHashMap<>();
        Map<String, List<String>> pro = new LinkedHashMap<>();
        for (String p : own) {
            List<String> ks = new ArrayList<>(dateKeys.get(p));
            java.util.Collections.shuffle(ks, rnd);
            int h = (ks.size() + 1) / 2;
            List<String> g = new ArrayList<>(ks.subList(0, h));
            List<String> pr = new ArrayList<>(ks.subList(h, ks.size()));
            g.sort(null);
            pr.sort(null);
            if (pr.isEmpty()) throw new IllegalStateException("Внутри сессии: у " + p + " нет проб");
            gal.put(p, g);
            pro.put(p, pr);
        }
        return List.of(gal, pro);
    }

    /** Доля верного кандидата (группы (i) + (ii)) варианта при FAR 1 %, всего: {точка %, реплики}. */
    static Object[] correct(FeiThreshold.Res r, int m) {
        double[] c = r.counts[m][0][0];
        double n = Arrays.stream(c).sum();
        double pt = 100.0 * (c[FeiDecompose.G_OK] + c[FeiDecompose.G_REJ]) / n;
        double[] reps = null;
        if (r.shareRep != null) {
            reps = new double[r.shareRep[m][0][0][0].length];
            for (int q = 0; q < reps.length; q++) reps[q] = r.shareRep[m][0][0][FeiDecompose.G_OK][q] + r.shareRep[m][0][0][FeiDecompose.G_REJ][q];
        }
        return new Object[] {pt, reps, (int) (c[FeiDecompose.G_OK] + c[FeiDecompose.G_REJ]), (int) n};
    }

    static void gtIn(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        Gt g = prepare(export, dir, feiKeys, selLines, settings);
        Eval cross = evaluate(g, g.galleryKeys, g.probeKeys);
        List<Map<String, List<String>>> split = withinSplit(g.own, g.galleryKeys, g.seed);
        Map<String, List<String>> gal = split.get(0);
        Map<String, List<String>> pro = split.get(1);
        Eval in = evaluate(g, gal, pro);
        writeScores(out.resolve("xbase_gt_in_scores.tsv.gz"), " — контроль внутри сессии", g, in);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, сверка, шаг 2: Georgia Tech — контроль «внутри сессии» (XBaseEval gt-in)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append('\n');
        t.append(String.format(Locale.ROOT, "Протокол (решение Хозяина 04.10.2026): те же %d своих, дата галереи — по прежнему правилу (наибольшее число "
                + "снимков, при\nравенстве — более ранняя); снимки этой даты делятся случайно пополам: java.util.Random(seed %d + %d = %d), люди по порядку,\n"
                + "первые ⌈n/2⌉ после перемешивания — галерея, остальные — пробы. Остальное без изменений: отбор по позе (осталось %d из %d), "
                + "когорта FEI\n(%d человек), «чужие» П2/П3 — снимки галереи (теперь — половины даты), 5 частей, контроль — %d односессионных, "
                + "варианты L1, L1-in, A,\nбутстреп по людям GT, B = %d, seed = %d (те же веса, что у межсессионного — разности парные).%n", g.own.size(),
                g.seed, WITHIN_SEED_OFFSET, g.seed + WITHIN_SEED_OFFSET, g.total - g.dropped.size(), g.total, g.cohortP.size(), g.ctrl.size(), g.b,
                g.seed + FeiBootstrap.SEED_OFFSET));
        t.append("Оговорка: галерея внутри сессии вдвое меньше межсессионной (половина снимков той же даты), «чужих» П2/П3 тоже меньше —\n"
                + "размеры ниже. Межсессионный протокол пересчитан в этом же запуске (тот же код, что xbase_gt.txt).\n");
        int[] gs = gal.values().stream().mapToInt(List::size).sorted().toArray();
        int[] ps = pro.values().stream().mapToInt(List::size).sorted().toArray();
        int[] cgs = g.galleryKeys.values().stream().mapToInt(List::size).sorted().toArray();
        int[] cps = g.probeKeys.values().stream().mapToInt(List::size).sorted().toArray();
        t.append(String.format(Locale.ROOT, "Внутри сессии: галерея %d снимков (на человека %d–%d), пробы своих %d (%d–%d); «чужие» П2/П3 — %d снимков "
                + "%d людей; контроль — %d снимков %d людей.%n", Arrays.stream(gs).sum(), gs[0], gs[gs.length - 1], Arrays.stream(ps).sum(), ps[0],
                ps[ps.length - 1], in.cvImages, g.own.size(), in.ctrlImages, g.ctrl.size()));
        t.append(String.format(Locale.ROOT, "Межсессионно: галерея %d снимков (%d–%d), пробы своих %d (%d–%d); «чужие» П2/П3 — %d снимков %d людей; "
                + "контроль — %d снимков %d людей.%n", Arrays.stream(cgs).sum(), cgs[0], cgs[cgs.length - 1], Arrays.stream(cps).sum(), cps[0],
                cps[cps.length - 1], cross.cvImages, g.own.size(), cross.ctrlImages, g.ctrl.size()));
        t.append("Правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6), внутри сессии: ");
        for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(in.cases[k]);
        t.append('\n');

        t.append("\n=== Галереи и пробы своих (внутри сессии) ===\n");
        t.append("человек | дата галереи | снимков даты | в галерее | проб\n");
        for (String p : g.own) {
            t.append(String.join(" | ", p, Integer.toString(g.galleryDateOf.get(p)), Integer.toString(g.galleryKeys.get(p).size()),
                    Integer.toString(gal.get(p).size()), Integer.toString(pro.get(p).size()))).append('\n');
        }

        t.append("\n=== Сводка: ошибка своих, % (межсессионно / внутри сессии) ===\n");
        t.append("вариант | способ | FAR 1 %: межсессионно | внутри сессии | FAR 5 %: межсессионно | внутри сессии | FAR контроля 1 %: межс. / внутри | "
                + "5 %: межс. / внутри\n");
        for (int v = 0; v < BASE_VARIANTS.size(); v++) {
            FeiThreshold.Res c = cross.results.get(v);
            FeiThreshold.Res w = in.results.get(v);
            for (int m = 2; m < FeiThreshold.M; m++) {
                t.append(String.join(" | ", c.label, FeiThreshold.METHODS[m], FeiThreshold.errCell(c, m, 0, 0, g.b), FeiThreshold.errCell(w, m, 0, 0, g.b),
                        FeiThreshold.errCell(c, m, 1, 0, g.b), FeiThreshold.errCell(w, m, 1, 0, g.b),
                        String.format(Locale.ROOT, "%.2f / %.2f %%", FeiThreshold.farPct(c, m, 0), FeiThreshold.farPct(w, m, 0)),
                        String.format(Locale.ROOT, "%.2f / %.2f %%", FeiThreshold.farPct(c, m, 1), FeiThreshold.farPct(w, m, 1)))).append('\n');
            }
        }

        t.append("\n=== Доля верного кандидата (группы (i) + (ii), не зависит от порога), всего ===\n");
        t.append("вариант | межсессионно | внутри сессии | разность межс. − внутри, п. п.\n");
        for (int v = 0; v < BASE_VARIANTS.size(); v++) {
            Object[] c = correct(cross.results.get(v), 2);
            Object[] w = correct(in.results.get(v), 2);
            String d = g.b > 0 ? FeiBootstrap.diff((double) c[0] - (double) w[0], FeiNorm.diff((double[]) c[1], (double[]) w[1]), 1) : "";
            t.append(String.join(" | ", cross.results.get(v).label, FeiNorm.cell((int) c[2], (int) c[3], (double[]) c[1]),
                    FeiNorm.cell((int) w[2], (int) w[3], (double[]) w[1]), d)).append('\n');
        }

        t.append("\n--- Внутри сессии: таблицы как в xbase_gt.txt ---\n");
        tables(t, in.results, g.b);

        if (g.b > 0) {
            t.append("\n=== Разности «межсессионно − внутри сессии», п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("вариант | способ | ошибка FAR 1 % всего | анфас | поворот | FAR 5 % всего | анфас | поворот | FAR контроля 1 % | FAR контроля 5 %\n");
            for (V v : new V[] {V.L1, V.A}) {
                FeiThreshold.Res c = cross.results.get(v.ordinal());
                FeiThreshold.Res w = in.results.get(v.ordinal());
                for (int m = 2; m < FeiThreshold.M; m++) {
                    List<String> row = new ArrayList<>(List.of(v.label, FeiThreshold.METHODS[m]));
                    for (int k : KS) {
                        for (int sp = 0; sp < FeiThreshold.SP; sp++) {
                            row.add(FeiBootstrap.diff(FeiThreshold.err(c, m, k, sp) - FeiThreshold.err(w, m, k, sp),
                                    FeiNorm.diff(c.errRep[m][k][sp], w.errRep[m][k][sp]), 1));
                        }
                    }
                    for (int k : KS) {
                        row.add(FeiBootstrap.diff(FeiThreshold.farPct(c, m, k) - FeiThreshold.farPct(w, m, k), FeiNorm.diff(c.farRep[m][k], w.farRep[m][k]), 2));
                    }
                    t.append(String.join(" | ", row)).append('\n');
                }
            }
        }
        FarMethods.write(out.resolve("xbase_gt_in.txt"), t);
        System.out.print(t);
        normReport(g, in, gal, pro, "gt_in", " — контроль внутри сессии", "внутри сессии", LbpNormReport.Kind.WITHIN, out);
        frameSheet(export, g, out.resolve("gt_frames.png"));
    }

    /**
     * Диагностика межсессионного провала на GT (PLAN.md, п. 4.2, «Диагностика L0 и SFace»): L0 (χ² до ближайшего снимка,
     * как 5-lbp-chi FEI), L0-mean (χ² до средней гистограммы), SFace (как 0-sface FarMethods), L1 и A справочно — межсессионно
     * и внутри сессии (то же деление, что gt-in), те же веса бутстрепа; отчёт xbase_gt_diag.txt, оценки
     * xbase_gt_diag_scores.tsv.gz и xbase_gt_in_diag_scores.tsv.gz. Прежние отчёты не пишутся и не меняются.
     */
    static void gtDiag(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out) throws IOException {
        Gt g = prepare(export, dir, feiKeys, selLines, settings, true, false);
        Eval cross = evaluate(g, g.x, DIAG_VARIANTS, g.galleryKeys, g.probeKeys);
        List<Map<String, List<String>>> split = withinSplit(g.own, g.galleryKeys, g.seed);
        Eval in = evaluate(g, g.x, DIAG_VARIANTS, split.get(0), split.get(1));
        writeScores(out.resolve("xbase_gt_diag_scores.tsv.gz"), " — диагностика L0 и SFace, межсессионно", g, cross);
        writeScores(out.resolve("xbase_gt_in_diag_scores.tsv.gz"), " — диагностика L0 и SFace, внутри сессии", g, in);
        Map<String, FeiThreshold.Res> cm = new LinkedHashMap<>();
        Map<String, FeiThreshold.Res> wm = new LinkedHashMap<>();
        for (FeiThreshold.Res r : cross.results) cm.put(r.label, r);
        for (FeiThreshold.Res r : in.results) wm.put(r.label, r);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: диагностика межсессионного провала на GT — L0, L0-mean и SFace (XBaseEval gt-diag)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir))
                .append(" (a/gt — кадр а 92×112; sface/gt — alignCrop 112×112; когорта L1 и A — a/fei)\n");
        t.append("Методы (нормировка none; протокол GT как в xbase_gt.txt и xbase_gt_in.txt; выбор модели по позе пробы r0 = ").append(g.r0)
                .append(";\nоценка d₁/d₂ — FeiScores.ratio; порог П2 / П3 по «чужим» галереи, 5 частей):\n"
                        + "  L0 = 5-lbp-chi-ratio+pose — гистограммы LBP u2 (8, 2), 7×7, на сумму 1, без корня; χ² до ближайшего снимка галереи\n"
                        + "       человека (как 5-lbp-chi FEI); без обучения и когорты; +mirror — к снимкам галереи добавлены их отражения;\n"
                        + "  L0-mean = 5-lbp-chi-mean-ratio+pose — то же, χ² до средней гистограммы снимков галереи человека (при +mirror —\n"
                        + "       вместе с отражениями);\n"
                        + "  SFace = 0-sface-ratio+pose — справочный предел, в рабочее ПО не идёт: признак FaceRecognizerSF по кадру sface/gt, шаблон\n"
                        + "       — нормированное среднее признаков галереи человека, оценка 1 − cos (как 0-sface FarMethods); зеркал и выбора по\n"
                        + "       позе нет (одна модель);\n"
                        + "  L1, A — как в xbase_gt.txt (справочно; строки оценок — те же, что в прежних файлах оценок).\n");
        t.append("Модель SFace: ").append(g.sfaceModel).append('\n');
        t.append(String.format(Locale.ROOT, "Межсессионно: пробы своих %d, «чужие» П2/П3 %d снимков; внутри сессии: пробы своих %d, «чужие» П2/П3 %d; "
                + "контроль — %d снимков %d людей; свои — %d человек.%n", cross.probes.stream().filter(p -> p.role().equals(OWN)).count(), cross.cvImages,
                in.probes.stream().filter(p -> p.role().equals(OWN)).count(), in.cvImages, cross.ctrlImages, g.ctrl.size(), g.own.size()));
        if (g.b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d; одни веса для всех методов и обоих "
                    + "режимов.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: межсессионно %.0f с, внутри сессии %.0f с, потоков %d.%n", cross.fitS, in.fitS, g.threads));
        t.append("Оговорка: 30 своих — различимы только крупные эффекты; внутри сессии снимки почти дубли (оценка «внутри» оптимистична).\n");
        LbpNormReport.diag(t, cm, wm, List.of("L0-mean", "L0", "SFace", "L1", "A"), "L1", List.of("L0-mean", "L0", "SFace"), g.b);
        FarMethods.write(out.resolve("xbase_gt_diag.txt"), t);
        System.out.print(t);
    }

    /**
     * Проверка выравнивания на GT (PLAN.md, п. 4.2, «Проверка выравнивания и разделимости на GT»): L1-crop (L1 на кадре
     * alignCrop 112×112 в сером), L1-crop-92 (его центральная часть 92×112, столбцы 10–101) и L1 — межсессионно и внутри
     * сессии; разложение: L1-crop-92 − L1 (выравнивание и путь получения серого), L1-crop − L1-crop-92 (ширина кадра).
     * Отчёт xbase_gt_crop.txt, оценки xbase_gt_crop_scores.tsv.gz и xbase_gt_in_crop_scores.tsv.gz.
     */
    static void gtCrop(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out) throws IOException {
        Gt g = prepare(export, dir, feiKeys, selLines, settings, false, true);
        Eval cross = evaluate(g, g.x, CROP_VARIANTS, g.galleryKeys, g.probeKeys);
        List<Map<String, List<String>>> split = withinSplit(g.own, g.galleryKeys, g.seed);
        Eval in = evaluate(g, g.x, CROP_VARIANTS, split.get(0), split.get(1));
        writeScores(out.resolve("xbase_gt_crop_scores.tsv.gz"), " — проверка выравнивания (L1-crop), межсессионно", g, cross);
        writeScores(out.resolve("xbase_gt_in_crop_scores.tsv.gz"), " — проверка выравнивания (L1-crop), внутри сессии", g, in);
        Map<String, FeiThreshold.Res> cm = new LinkedHashMap<>();
        Map<String, FeiThreshold.Res> wm = new LinkedHashMap<>();
        for (FeiThreshold.Res r : cross.results) cm.put(r.label, r);
        for (FeiThreshold.Res r : in.results) wm.put(r.label, r);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: проверка выравнивания на GT — L1 на кадрах alignCrop (XBaseEval gt-crop)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir))
                .append(" (a/gt — кадр а 92×112; sface/gt и sface/fei — alignCrop 112×112 BGR)\n");
        t.append("Методы (нормировка none; протокол GT как в xbase_gt.txt и xbase_gt_in.txt; всё, кроме входа, — как у L1: LBP u2 (8, 2), 7×7,\n"
                + "гистограммы на сумму 1, корень, MLDA, когорта — те же 20 людей FEI (на кадрах того же вида), d₁/d₂ — FeiScores.ratio,\n"
                + "выбор X / X+mirror по |r| пробы, r0 = ").append(g.r0).append(" (r — из detections.tsv), П2 / П3 по «чужим» галереи):\n"
                + "  L1-crop = 5-lbp-mlda-crop-ratio+pose — кадр alignCrop 112×112 (sface экспорта) → COLOR_BGR2GRAY; зеркала — способ а\n"
                + "       (FaceMirror) с шаблоном ArcFace 112×112;\n"
                + "  L1-crop-92 = 5-lbp-mlda-crop92-ratio+pose — центральная часть того же серого кадра 92×112 (столбцы 10–101; кадр «а» —\n"
                + "       тот же шаблон ArcFace со сдвигом −10 px); зеркала — способ а с шаблоном ownTemplate(92, 112);\n"
                + "  L1 — как в xbase_gt.txt (кадр «а»: своё подобие по 5 точкам YuNet, серый до warpAffine).\n"
                + "Разложение: L1-crop-92 − L1 — выравнивание (подобие alignCrop OpenCV против своего) и путь получения серого; L1-crop −\n"
                + "L1-crop-92 — ширина кадра (по 10 px с каждой стороны).\n");
        t.append(String.format(Locale.ROOT, "Межсессионно: пробы своих %d, «чужие» П2/П3 %d снимков; внутри сессии: пробы своих %d, «чужие» П2/П3 %d; "
                + "контроль — %d снимков %d людей; свои — %d человек.%n", cross.probes.stream().filter(p -> p.role().equals(OWN)).count(), cross.cvImages,
                in.probes.stream().filter(p -> p.role().equals(OWN)).count(), in.cvImages, cross.ctrlImages, g.ctrl.size(), g.own.size()));
        if (g.b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d; одни веса для всех методов и обоих "
                    + "режимов.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: межсессионно %.0f с, внутри сессии %.0f с, потоков %d.%n", cross.fitS, in.fitS, g.threads));
        t.append("Правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6), межсессионно: ");
        for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(cross.cases[k]);
        t.append('\n');
        t.append("Оговорка: 30 своих — различимы только крупные эффекты; внутри сессии снимки почти дубли (оценка «внутри» оптимистична).\n");
        LbpNormReport.diag(t, cm, wm, List.of("L1-crop", "L1-crop-92", "L1"), "L1", List.of("L1-crop-92", "L1-crop"), g.b);
        if (g.b > 0) LbpNormReport.dropLevel(t, cm, wm, "L1-crop-92", List.of("L1-crop"));
        FarMethods.write(out.resolve("xbase_gt_crop.txt"), t);
        System.out.print(t);
    }

    /**
     * Лист выровненных кадров 92×112 (вне git): 6 своих GT (SHEET_OWN), у каждого по 3 снимка галереи и 3 пробы
     * межсессионного протокола; увеличение ×2, подписи латиницей (человек, g / p, номер, |r|).
     */
    static void frameSheet(FarExport.Loaded export, Gt g, Path file) throws IOException {
        int sc = 2;
        int tw = 92 * sc;
        int th = 112 * sc;
        int cap = 16;
        List<String> people = new ArrayList<>();
        for (int i : SHEET_OWN) people.add(g.own.get(i));
        Mat sheet = new Mat(people.size() * (th + cap), 6 * tw, org.opencv.core.CvType.CV_8UC3, new org.opencv.core.Scalar(255, 255, 255));
        for (int row = 0; row < people.size(); row++) {
            String p = people.get(row);
            List<String> ks = new ArrayList<>();
            List<String> tags = new ArrayList<>();
            for (String k : g.galleryKeys.get(p).subList(0, Math.min(3, g.galleryKeys.get(p).size()))) {
                ks.add(k);
                tags.add("g");
            }
            for (String k : g.probeKeys.get(p).subList(0, Math.min(3, g.probeKeys.get(p).size()))) {
                ks.add(k);
                tags.add("p");
            }
            for (int c = 0; c < ks.size(); c++) {
                Mat f = export.frame(g.samples.get(ks.get(c)), Variant.A);
                Mat big = new Mat();
                org.opencv.imgproc.Imgproc.resize(f, big, new org.opencv.core.Size(tw, th), 0, 0, org.opencv.imgproc.Imgproc.INTER_NEAREST);
                Mat bgr = new Mat();
                org.opencv.imgproc.Imgproc.cvtColor(big, bgr, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR);
                int x0 = c * tw;
                int y0 = row * (th + cap);
                bgr.copyTo(sheet.submat(y0, y0 + th, x0, x0 + tw));
                String text = String.format(Locale.ROOT, "%s %s %02d r=%.2f", p, tags.get(c), image(g.samples.get(ks.get(c))), g.rOf.get(ks.get(c)));
                org.opencv.imgproc.Imgproc.putText(sheet, text, new org.opencv.core.Point(x0 + 3, y0 + th + 12),
                        org.opencv.imgproc.Imgproc.FONT_HERSHEY_PLAIN, 0.9, new org.opencv.core.Scalar(0, 0, 0), 1);
                f.release();
            }
        }
        org.opencv.imgcodecs.Imgcodecs.imwrite(file.toString(), sheet);
        System.out.println("Лист кадров: " + file);
    }

    /** Номер снимка из имени файла GT («01» → 1). */
    static int image(Sample s) {
        return Integer.parseInt(s.file().getFileName().toString().replaceAll("\\D", ""));
    }

    /** Строка модели по позе: анфас (|r| ≤ r0) — X, иначе X+mirror. */
    static Row choose(Map<String, Row> rows, V v, String role, String person, int image, boolean frontal) {
        String method = v.id + (frontal || v == V.SFACE ? "" : FeiMethods.MIRROR);
        Row r = rows.get(method + "\t" + role + "\t" + person + "\t" + image);
        if (r == null) throw new IllegalStateException("Нет строки оценок " + method + " " + role + " " + person + " №" + image);
        return r;
    }

    static final int[] KS = {0, 1};
    static final String[] FAR_NAME = {"FAR 1 %", "FAR 5 %"};

    static StringBuilder report(List<FeiThreshold.Res> res, List<String> own, List<String> ctrl, Map<String, List<String>> galleryKeys,
                                Map<String, Integer> galleryDateOf, Map<String, List<String>> probeKeys, Map<String, Map<Integer, List<String>>> byPerson,
                                List<List<String>> parts, List<String> cohortP, List<List<String>> cohort, Map<String, Double> rOf, double r0, double rLimit,
                                int total, List<String> dropped, int ctrlImages, int cvImages, int[] cases, int b, long seed, int persons, Path dir,
                                double fitS, int threads, List<Probe> probes) {
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, сверка на других базах, шаг 2: Georgia Tech (XBaseEval gt)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir))
                .append(" (a/gt — кадр а 92×112; detections.tsv; reports/gt_sessions.tsv; когорта — a/fei)\n");
        t.append("Умолчания FEI без подбора: кадр а; L1 = корень гистограмм LBP u2 (8, 2), 7×7 + MLDA; L1-in — внутренние блоки 1–5;\n"
                + "A = CLAHE + MLDA (2c-mlda); зеркала (FaceMirror, способ а) — модель X+mirror; выбор модели по позе пробы r0 = ").append(r0)
                .append(";\nоценка d₁/d₂ (FeiScores.ratio); порог Нейман – Пирсон по «чужим» галереи: П2 — точечно, П3 — с запасом 95 % "
                        + "(FeiThreshold.margin).\n");
        t.append(String.format(Locale.ROOT, "Отбор по позе — как FEI: |r| ≤ R = %.4f (fei_selection.tsv): осталось %d из %d; исключены: %s.%n", rLimit,
                total - dropped.size(), total, String.join(", ", dropped)));
        t.append("Роли (решения Хозяина 04.10.2026): свои — люди с ≥ 2 датами съёмки (EXIF, gt_sessions.tsv); галерея — дата с наибольшим\n"
                + "числом снимков (при равенстве — более ранняя), пробы своих — все снимки остальных дат; контрольные чужие — люди с одной\n"
                + "датой, все снимки; когорта MLDA — встроенная: 20 «полных» людей FEI (seed faces.seed + 9000), все их отобранные снимки.\n");
        t.append(String.format(Locale.ROOT, "Свои: %d человек; контрольные чужие: %d человек, %d снимков; когорта FEI: %d человек, %d снимков (%s).%n",
                own.size(), ctrl.size(), ctrlImages, cohortP.size(), cohort.stream().mapToInt(List::size).sum(), String.join(", ", cohortP)));
        t.append(String.format(Locale.ROOT, "«Чужие» П2/П3: снимки галереи своих, 5 частей по %d человек (FeiMethods.cvParts, seed faces.seed + 8000 + "
                + "%d); модель без части — на остальных %d своих и той же когорте; всего %d снимков %d людей.%n", parts.get(0).size(), CV_S,
                own.size() - parts.get(0).size(), cvImages, own.size()));
        for (int q = 0; q < parts.size(); q++) t.append("  часть ").append(q).append(": ").append(String.join(", ", parts.get(q))).append('\n');
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком (когорта FEI не перевыбирается): B = %d, seed = %d, людей %d; "
                    + "одни веса для\nвсех строк; порог — заново своим способом в каждой реплике; [2,5; 97,5] — процентили; «*» — интервал разности "
                    + "не содержит 0.%n", b, seed + FeiBootstrap.SEED_OFFSET, persons));
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с, потоков %d.%n", fitS, threads));
        t.append("Правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6): ");
        for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(cases[k]);
        t.append('\n');

        long ownF = probes.stream().filter(p -> p.role().equals(OWN) && rOf.get(p.key()) <= r0).count();
        long ownN = probes.stream().filter(p -> p.role().equals(OWN)).count();
        long ctrlF = probes.stream().filter(p -> p.role().equals(CTRL) && rOf.get(p.key()) <= r0).count();
        long galF = galleryKeys.values().stream().flatMap(List::stream).filter(k -> rOf.get(k) <= r0).count();
        long galN = galleryKeys.values().stream().mapToLong(List::size).sum();
        t.append(String.format(Locale.ROOT, "Доля анфаса (|r| ≤ r0): пробы своих %s; галерея (и «чужие» П2/П3) %s; контрольные чужие %s.%n",
                FeiThreshold.share(ownF, ownN), FeiThreshold.share(galF, galN), FeiThreshold.share(ctrlF, ctrlImages)));

        t.append("\n=== Галереи и пробы своих ===\n");
        t.append("человек | даты (номер: снимков после отбора) | дата галереи | снимков в галерее | проб\n");
        for (String p : own) {
            StringBuilder ds = new StringBuilder();
            byPerson.get(p).forEach((d, ks) -> ds.append(ds.length() > 0 ? ", " : "").append(d == 0 ? "без даты" : d).append(": ").append(ks.size()));
            t.append(String.join(" | ", p, ds, Integer.toString(galleryDateOf.get(p)), Integer.toString(galleryKeys.get(p).size()),
                    Integer.toString(probeKeys.get(p).size()))).append('\n');
        }
        int[] gs = galleryKeys.values().stream().mapToInt(List::size).sorted().toArray();
        int[] ps = probeKeys.values().stream().mapToInt(List::size).sorted().toArray();
        t.append(String.format(Locale.ROOT, "Галерея: снимков %d (на человека %d–%d, медиана %d); проб своих %d (на человека %d–%d, медиана %d).%n",
                Arrays.stream(gs).sum(), gs[0], gs[gs.length - 1], gs[gs.length / 2], Arrays.stream(ps).sum(), ps[0], ps[ps.length - 1], ps[ps.length / 2]));
        tables(t, res, b);
        return t;
    }

    /** Таблицы: ошибка своих и FAR контроля, разложение (i)–(iv), парные разности L1 − L1-in и L1 − A (способы П2, П3). */
    static void tables(StringBuilder t, List<FeiThreshold.Res> res, int b) {
        t.append("\n=== Ошибка своих и FAR контроля ===\n");
        t.append("вариант | способ | ошибка своих FAR 1 % | FAR 5 % | анфас FAR 1 % | поворот FAR 1 % | FAR контроля 1 %: точка ↑95; КП | 5 % | "
                + "θ 1 % / 5 % | «ноль недопустим» 1 % / 5 % (реплик)\n");
        for (FeiThreshold.Res r : res) {
            for (int m = 2; m < FeiThreshold.M; m++) {
                List<String> c = new ArrayList<>(List.of(r.label + " = " + r.id, FeiThreshold.METHODS[m]));
                for (int[] ks : new int[][] {{0, 0}, {1, 0}, {0, 1}, {0, 2}}) c.add(FeiThreshold.errCell(r, m, ks[0], ks[1], b));
                for (int k : KS) c.add(FeiThreshold.farCell(r, m, k, b));
                c.add(String.format(Locale.ROOT, "%.6f / %.6f", r.theta[m][0][0], r.theta[m][1][0]));
                c.add(r.zeroFail[m][0] + " / " + r.zeroFail[m][1] + (b > 0 ? " (" + r.zeroFailRep[m][0] + " / " + r.zeroFailRep[m][1] + ")" : ""));
                t.append(String.join(" | ", c)).append('\n');
            }
        }

        t.append("\n=== Разложение ошибки своих ===\n");
        for (int k : KS) {
            t.append("-- ").append(FAR_NAME[k]).append('\n');
            t.append("вариант | способ | деление | (i) | (ii) | (iii) | (iv) | (v-а)\n");
            for (FeiThreshold.Res r : res) {
                for (int m = 2; m < FeiThreshold.M; m++) {
                    for (int sp = 0; sp < FeiThreshold.SP; sp++) {
                        double[] cnt = r.counts[m][k][sp];
                        double n = Arrays.stream(cnt).sum();
                        List<String> c = new ArrayList<>(List.of(r.label, FeiThreshold.METHODS[m], FeiDecompose.SPLIT_NAMES[sp]));
                        for (int g = 0; g <= FeiDecompose.G_NO_SECOND; g++) c.add(FeiNorm.cell(cnt[g], n, b > 0 ? r.shareRep[m][k][sp][g] : null));
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }

        if (b > 0) {
            t.append("\n=== Парные разности, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("пара | способ | ошибка FAR 1 % всего | анфас | поворот | FAR 5 % всего | анфас | поворот | FAR контроля 1 % | FAR контроля 5 %\n");
            FeiThreshold.Res a = res.get(V.A.ordinal());
            FeiThreshold.Res l1 = res.get(V.L1.ordinal());
            FeiThreshold.Res in = res.get(V.L1IN.ordinal());
            for (FeiThreshold.Res[] pr : new FeiThreshold.Res[][] {{l1, in}, {l1, a}}) {
                for (int m = 2; m < FeiThreshold.M; m++) {
                    List<String> c = new ArrayList<>(List.of(pr[0].label + " − " + pr[1].label, FeiThreshold.METHODS[m]));
                    for (int k : KS) {
                        for (int sp = 0; sp < FeiThreshold.SP; sp++) {
                            c.add(FeiBootstrap.diff(FeiThreshold.err(pr[0], m, k, sp) - FeiThreshold.err(pr[1], m, k, sp),
                                    FeiNorm.diff(pr[0].errRep[m][k][sp], pr[1].errRep[m][k][sp]), 1));
                        }
                    }
                    for (int k : KS) {
                        c.add(FeiBootstrap.diff(FeiThreshold.farPct(pr[0], m, k) - FeiThreshold.farPct(pr[1], m, k),
                                FeiNorm.diff(pr[0].farRep[m][k], pr[1].farRep[m][k]), 2));
                    }
                    t.append(String.join(" | ", c)).append('\n');
                }
            }
        }
    }
}
