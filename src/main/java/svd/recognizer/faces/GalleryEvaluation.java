package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
import java.util.stream.IntStream;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * FAR и FRR на расширенной галерее (шаг 5, задача FAR; одна команда:
 * {@code mvn compile exec:exec@faces-far}, отдельная JVM с -Xmx6g). Отчёт — reports/faces/far/far_eval.txt
 * (вне git), время — far_time.txt, кэш детекций (только строки YuNet обоих проходов, без кадров) —
 * reports/faces/far/far_detections.cache; пересчёт — {@code -Dfar.args=fresh}. Векторы строятся на оценке
 * по одному варианту и после него освобождаются.
 *
 * Галерея — 56 человек: 6 своих (пригодные моменты, протокол own_eval: 5 моментов
 * обучение, кадры «+» и «+-», контроль — представитель момента) и 50 Georgia Tech
 * (сессий в данных нет: 5 случайных снимков — обучение, 10 — контроль, 3 разбиения
 * по seed). 12 конфигураций = 12 ротаций своих; разбиение Georgia Tech — r mod 3,
 * контроль Georgia Tech считается только в конфигурациях 0–2 (без дублей).
 * Чужие — MUCT, 276 человек, по людям на две половины по seed: валидация (порог
 * Неймана – Пирсона) и контроль (FAR); вариант — только камера a (фронтальная).
 * ORL — отдельная строка чужих на контроле.
 *
 * Детекция — двухпроходный YuNet (2′, YunetTwoPass — общий код с own_eval),
 * порог faces.own.detector.score (0,7) для всех баз. Вход первого прохода: своя
 * база — ×0,25, остальные — родное разрешение. Второй проход — область лица с
 * полями 50 % из полного разрешения, лицо около 300 px. Точки — второго прохода.
 * Кадр SVD: а) 92×112, пиксели — вход первого прохода (как own_eval); б) 92×112,
 * пиксели из полного разрешения (INTER_AREA, при увеличении — INTER_LINEAR);
 * в) 184×224 оттуда же. SFace — alignCrop 112×112 по строке второго прохода.
 *
 * @author ssv
 */
public final class GalleryEvaluation {

    static final double OWN_INPUT_SCALE = OwnEvaluation.INPUT_SCALE;
    static final double PASS2_MARGIN = OwnEvaluation.PASS2_MARGIN;
    static final double PASS2_FACE = OwnEvaluation.PASS2_FACE;
    static final int CONFIGS = OwnEvaluation.ROTATIONS;
    static final int GT_SPLITS = 3;
    static final int GT_TRAIN = 5;
    static final int K = 4;
    static final double[] ALPHAS = {0.05, 0.02, 0.01, 0.005, 0.0};
    static final String CACHE_NAME = "far_detections.cache";

    /** База снимка. */
    enum Base { OWN, GT, MUCT, ORL }

    /** Снимок: база, человек, файл, камера MUCT (иначе ' '). */
    record Sample(Base base, String person, Path file, char camera) {}

    /** Признаки снимка: найдено лицо; векторы SVD (варианты а, б, в) и SFace. */
    record Feat(boolean found, double[] a, double[] b, double[] c, double[] sface) {
        static final Feat NONE = new Feat(false, null, null, null, null);
    }

    /** Способ: вариант кадра SVD или SFace. */
    enum Variant {
        A("SVD а) 92×112, пиксели как в own_eval"), B("SVD б) 92×112, пиксели из полного разрешения"),
        C("SVD в) 184×224, пиксели из полного разрешения"), SFACE("SFace (1′), alignCrop 112×112");

        final String label;

        Variant(String label) {
            this.label = label;
        }

        double[] of(Feat f) {
            return !f.found() ? null : switch (this) {
                case A -> f.a();
                case B -> f.b();
                case C -> f.c();
                case SFACE -> f.sface();
            };
        }
    }

    /** Скоринг: ε₁/ε₂, ε (SVD) или 1 − cos₁ (SFace). */
    enum Score { RATIO("ε₁/ε₂"), EPS("ε"), COS("1 − cos₁");

        final String label;

        Score(String label) {
            this.label = label;
        }
    }

    private GalleryEvaluation() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        long start = System.nanoTime();
        SettingsStore settings = new SettingsStore();
        long seed = settings.loadFacesSeed();
        float score = settings.loadFacesOwnDetectorScore();
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        // При заданном faces.export.dir все четыре базы — из экспорта (FarExport): сырые снимки не читаются.
        String exportDir = settings.loadFacesExportDir();
        FarExport.Loaded export = exportDir == null ? null : FarExport.read(Paths.get(exportDir));
        Data data = export == null ? rawData(settings) : export.data();
        OwnDataset own = data.own();
        List<List<OwnDataset.Moment>> moments = data.moments();
        Map<String, List<Sample>> gt = data.gt();
        Map<String, List<Sample>> muct = data.muct();
        Map<String, List<Sample>> orl = data.orl();
        List<Sample> samples = data.samples();

        // Детекция (кэш) — только по сырым снимкам; в экспорте отказы детектора уже записаны.
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "far");
        Files.createDirectories(outDir);
        boolean fresh = args.length > 0 && args[0].equals("fresh");
        Map<String, Det> dets = null;
        long detTime = 0;
        if (export == null) {
            dets = detections(samples, score, settings, modelsDir, outDir, fresh);
            detTime = lastDetTime;
        }
        FaceRecognizerSF sface = FaceRecognizerSF.create(modelsDir.resolve(FaceAlignment.SFACE_FILE).toString(), "");

        // Разбиения.
        Splits splits = splits(gt, muct, seed);
        if (export != null) export.checkSplits(splits);
        List<String> gtPersons = splits.gtPersons();
        List<List<List<Integer>>> gtTrain = splits.gtTrain();
        List<String> muctVal = splits.muctVal();
        List<String> muctCtrl = splits.muctCtrl();

        StringBuilder text = new StringBuilder();
        header(text, own, moments, gt, muct, orl, muctVal, muctCtrl, null, samples, score, seed);
        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        int n = samples.size();
        StringBuilder timeText = new StringBuilder();
        if (export == null) {
            timeText.append(String.format(Locale.ROOT, "Детекция (два прохода YuNet), на снимок: %.0f мс (снимков %d; при запуске "
                    + "из кэша — время исходного прогона).%n", detTime / 1e6 / n, n));
        } else {
            timeText.append(String.format(Locale.ROOT, "Базы — из экспорта %s (снимков %d; детекция не выполнялась, векторы — "
                    + "из PNG).%n", exportDir, n));
        }
        // По одному варианту: векторы строятся из детекций (или из PNG экспорта) и освобождаются после оценки.
        for (Variant v : Variant.values()) {
            long t0 = System.nanoTime();
            Map<String, Feat> feats = export == null ? vectors(samples, dets, v, sface) : export.vectors(samples, v, sface);
            long tv = System.nanoTime() - t0;
            System.out.printf(Locale.ROOT, "%s: векторы %.0f с%n", v.label, tv / 1e9);
            long te = 0;
            for (Score sc : v == Variant.SFACE ? new Score[] {Score.COS} : new Score[] {Score.RATIO, Score.EPS}) {
                long t1 = System.nanoTime();
                evaluate(text, v, sc, moments, gt, gtPersons, gtTrain, muct, muctVal, muctCtrl, orl, feats, trainer);
                te += System.nanoTime() - t1;
            }
            DIST.clear();
            timeText.append(String.format(Locale.ROOT, "%s: построение вектора на снимок %.1f мс; оценка (12 конфигураций, "
                    + "все скоринги) %.0f с.%n", v.label, tv / 1e6 / n, te / 1e9));
        }
        text.append("\n=== Время ===\n");
        text.append("Время построения векторов и оценки меняется от запуска к запуску и вынесено в far_time.txt "
                + "(не входит в побайтное сравнение отчёта).\n");
        Path report = outDir.resolve("far_eval.txt");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(report, StandardCharsets.UTF_8))) {
            out.print(text);
        }
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("far_time.txt"), StandardCharsets.UTF_8))) {
            out.print(timeText);
        }
        System.out.print(timeText);
        System.out.printf(Locale.ROOT, "Всего %.0f с%n", (System.nanoTime() - start) / 1e9);
    }

    // ---------------------------------------------------------------- базы

    /** Базы оценки: своя база и её пригодные моменты, GT, MUCT, ORL (человек → снимки) и все снимки по порядку. */
    record Data(OwnDataset own, List<List<OwnDataset.Moment>> moments, Map<String, List<Sample>> gt,
                Map<String, List<Sample>> muct, Map<String, List<Sample>> orl, List<Sample> samples) {}

    /** Базы из сырых снимков (пути — SettingsStore). */
    static Data rawData(SettingsStore settings) throws IOException {
        OwnDataset own = OwnDataset.load(Paths.get(settings.loadFacesOwnDir()));
        Map<String, List<Sample>> gt = listGt(Paths.get(settings.loadFacesGtDir()));
        Map<String, List<Sample>> muct = listMuct(Paths.get(settings.loadFacesMuctDir()));
        Map<String, List<Sample>> orl = new TreeMap<>();
        Path orlDir = Paths.get(settings.loadFacesDatasetDir());
        for (int p = 1; p <= OrlDataset.PERSONS; p++) {
            List<Sample> list = new ArrayList<>();
            for (int i = 1; i <= OrlDataset.IMAGES_PER_PERSON; i++) {
                list.add(new Sample(Base.ORL, String.format(Locale.ROOT, "s%02d", p), orlDir.resolve("s" + p).resolve(i + ".pgm"), ' '));
            }
            orl.put(String.format(Locale.ROOT, "s%02d", p), list);
        }
        return data(own, gt, muct, orl);
    }

    /** Снимки всех баз: своя (пригодные моменты — представитель и кадры «+», «+-»), GT, MUCT, ORL. */
    static Data data(OwnDataset own, Map<String, List<Sample>> gt, Map<String, List<Sample>> muct, Map<String, List<Sample>> orl) {
        List<List<OwnDataset.Moment>> moments = new ArrayList<>();
        for (OwnDataset.Person p : own.persons()) moments.add(p.usableMoments());
        List<Sample> samples = new ArrayList<>();
        for (int p = 0; p < own.persons().size(); p++) {
            Map<Path, Sample> m = new LinkedHashMap<>();
            for (OwnDataset.Moment mo : moments.get(p)) {
                m.put(mo.representative().file(), new Sample(Base.OWN, own.persons().get(p).name(), mo.representative().file(), ' '));
                for (OwnDataset.Frame f : mo.frames()) {
                    if (f.quality() >= OwnDataset.FAIR) m.put(f.file(), new Sample(Base.OWN, own.persons().get(p).name(), f.file(), ' '));
                }
            }
            samples.addAll(m.values());
        }
        gt.values().forEach(samples::addAll);
        muct.values().forEach(samples::addAll);
        orl.values().forEach(samples::addAll);
        return new Data(own, moments, gt, muct, orl, samples);
    }

    /** Разбиения: GT — обучающие индексы по разбиениям и людям; MUCT — валидация и контроль по людям. */
    record Splits(List<String> gtPersons, List<List<List<Integer>>> gtTrain, List<String> muctVal, List<String> muctCtrl) {}

    /** GT: разбиение s — seed + s, 5 обучающих на человека; MUCT: люди перемешаны по seed, пополам. */
    static Splits splits(Map<String, List<Sample>> gt, Map<String, List<Sample>> muct, long seed) {
        List<String> gtPersons = new ArrayList<>(gt.keySet());
        List<List<List<Integer>>> gtTrain = new ArrayList<>();
        for (int s = 0; s < GT_SPLITS; s++) {
            Random rnd = new Random(seed + s);
            List<List<Integer>> perPerson = new ArrayList<>();
            for (String p : gtPersons) {
                List<Integer> idx = new ArrayList<>();
                for (int i = 0; i < gt.get(p).size(); i++) idx.add(i);
                Collections.shuffle(idx, rnd);
                perPerson.add(idx.subList(0, GT_TRAIN));
            }
            gtTrain.add(perPerson);
        }
        List<String> muctPersons = new ArrayList<>(muct.keySet());
        Collections.shuffle(muctPersons, new Random(seed));
        List<String> muctVal = muctPersons.subList(0, muctPersons.size() / 2);
        List<String> muctCtrl = muctPersons.subList(muctPersons.size() / 2, muctPersons.size());
        return new Splits(gtPersons, gtTrain, muctVal, muctCtrl);
    }

    static Map<String, List<Sample>> listGt(Path dir) throws IOException {
        return toSamples(ExternalDatasets.georgiaTech(dir), Base.GT);
    }

    static Map<String, List<Sample>> listMuct(Path dir) throws IOException {
        return toSamples(ExternalDatasets.muct(dir), Base.MUCT);
    }

    private static Map<String, List<Sample>> toSamples(Map<String, List<ExternalDatasets.Image>> base, Base b) {
        Map<String, List<Sample>> map = new TreeMap<>();
        base.forEach((p, list) -> map.put(p, list.stream().map(x -> new Sample(b, p, x.file(), x.camera())).toList()));
        return map;
    }

    // ---------------------------------------------------------------- детекция и признаки

    /** Детекция снимка: строки YuNet первого и второго прохода в координатах полного снимка (null — лица нет). */
    record Det(double[] first, double[] second) implements Serializable {
        /** Строка для выравнивания: второго прохода, если он нашёл лицо, иначе первого; null — отказ детектора. */
        double[] row() {
            return second != null ? second : first;
        }
    }

    /** Снимок (цветной полный) и вход первого прохода: {полный, вход}; своя база — ×0,25, ORL серый → BGR. */
    static Mat[] load(Sample s) throws IOException {
        Mat color;
        if (s.base() == Base.ORL) {
            Mat g = FaceAlignment.readGray(s.file());
            color = new Mat();
            Imgproc.cvtColor(g, color, Imgproc.COLOR_GRAY2BGR);
            g.release();
        } else {
            color = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(s.file())), Imgcodecs.IMREAD_COLOR);
        }
        Mat input = color;
        if (scale(s) != 1.0) {
            input = new Mat();
            Imgproc.resize(color, input, new Size(Math.round(color.cols() * scale(s)), Math.round(color.rows() * scale(s))), 0, 0,
                    Imgproc.INTER_AREA);
        }
        return new Mat[] {color, input};
    }

    static double scale(Sample s) {
        return s.base() == Base.OWN ? OWN_INPUT_SCALE : 1.0;
    }

    static void release(Mat[] m) {
        for (Mat x : m) x.release();
    }

    /** Двухпроходный YuNet (YunetTwoPass): первый проход на входе, второй — по области лица из полного разрешения. */
    static Det detect(Sample s, FaceDetectorYN detector) throws IOException {
        Mat[] m = load(s);
        try {
            double[] first = YunetTwoPass.firstPass(detector, m[1], scale(s));
            if (first == null) return new Det(null, null);
            return new Det(first, YunetTwoPass.secondPass(detector, m[0], new double[] {first[0], first[1], first[2], first[3]}));
        } finally {
            release(m);
        }
    }

    /** Время детекции последнего вызова detections (из кэша — время исходного прогона). */
    static long lastDetTime;

    /** Детекции всех снимков: из кэша reports/faces/far/far_detections.cache или заново (fresh) с записью кэша. */
    static Map<String, Det> detections(List<Sample> samples, float score, SettingsStore settings, Path modelsDir, Path outDir,
                                       boolean fresh) throws IOException {
        Path cacheFile = outDir.resolve(CACHE_NAME);
        String signature = signature(samples, score, settings);
        Map<String, Det> dets = fresh ? null : loadCache(cacheFile, signature);
        if (dets == null) {
            dets = new LinkedHashMap<>();
            FaceDetectorYN detector = OwnDetectorThreshold.create(modelsDir, score);
            long t0 = System.nanoTime();
            int done = 0;
            for (Sample s : samples) {
                dets.put(s.file().toString(), detect(s, detector));
                if (++done % 250 == 0) System.out.println("Детекция: " + done + "/" + samples.size());
            }
            lastDetTime = System.nanoTime() - t0;
            saveCache(cacheFile, signature, dets, lastDetTime);
        } else {
            lastDetTime = cachedDetTime;
        }
        return dets;
    }

    /** Векторы одного варианта для всех снимков (по детекциям); остальные варианты не строятся. */
    static Map<String, Feat> vectors(List<Sample> samples, Map<String, Det> dets, Variant v, FaceRecognizerSF sface) throws IOException {
        Map<String, Feat> feats = new LinkedHashMap<>();
        int done = 0;
        for (Sample s : samples) {
            String key = s.file().toString();
            double[] row = dets.get(key).row();
            if (row == null) {
                feats.put(key, Feat.NONE);
            } else {
                double[] x = vector(s, row, v, sface);
                feats.put(key, new Feat(true, v == Variant.A ? x : null, v == Variant.B ? x : null, v == Variant.C ? x : null,
                        v == Variant.SFACE ? x : null));
            }
            if (++done % 1000 == 0) System.out.println(v.label + ": векторы " + done + "/" + samples.size());
        }
        return feats;
    }

    /** Вектор снимка для варианта v по строке детекции row (координаты полного снимка). */
    static double[] vector(Sample s, double[] row, Variant v, FaceRecognizerSF sface) throws IOException {
        Mat[] m = load(s);
        try {
            return vector(frame(s, m, row, v, sface), v, sface);
        } finally {
            release(m);
        }
    }

    /**
     * Кадр снимка для варианта v (m — {полный, вход} из load): а, б, в — серый 8 бит (92×112, 92×112, 184×224),
     * SFace — alignCrop 112×112 BGR. Эти кадры FarExport пишет в PNG без потерь.
     */
    static Mat frame(Sample s, Mat[] m, double[] row, Variant v, FaceRecognizerSF sface) {
        double scale = scale(s);
        double[][] lmFull = new double[5][2];
        double[][] lmSmall = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lmFull[j][0] = row[4 + 2 * j];
            lmFull[j][1] = row[5 + 2 * j];
            lmSmall[j][0] = lmFull[j][0] * scale;
            lmSmall[j][1] = lmFull[j][1] * scale;
        }
        switch (v) {
            case A -> {
                Mat inputGray = new Mat();
                Imgproc.cvtColor(m[1], inputGray, Imgproc.COLOR_BGR2GRAY);
                Mat a = new Mat();
                Imgproc.warpAffine(inputGray, a, FaceAlignment.similarity(lmSmall, FaceAlignment.ownTemplate(92, 112)),
                        new Size(92, 112), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
                inputGray.release();
                return a;
            }
            case B, C -> {
                int w = v == Variant.B ? 92 : 184;
                int h = v == Variant.B ? 112 : 224;
                Mat fullGray = new Mat();
                Imgproc.cvtColor(m[0], fullGray, Imgproc.COLOR_BGR2GRAY);
                Mat out = fromFull(fullGray, lmFull, FaceAlignment.ownTemplate(w, h), w, h);
                fullGray.release();
                return out;
            }
            default -> {
                Mat smallRow = new Mat(1, 15, CvType.CV_32F);
                for (int j = 0; j < 15; j++) smallRow.put(0, j, j == 14 ? row[j] : row[j] * scale);
                Mat crop = new Mat();
                sface.alignCrop(m[1], smallRow, crop);
                return crop;
            }
        }
    }

    /** Вектор по кадру варианта v: а, б, в — пиксели /255; SFace — признак feature(), нормированный. */
    static double[] vector(Mat frame, Variant v, FaceRecognizerSF sface) {
        if (v != Variant.SFACE) return FaceAlignment.toVector(frame);
        Mat feature = new Mat();
        sface.feature(frame, feature);
        Mat f64 = new Mat();
        feature.convertTo(f64, CvType.CV_64F);
        double[] x = new double[(int) f64.total()];
        f64.get(0, 0, x);
        OwnEvaluation.normalize(x);
        return x;
    }

    /** Кадр из полного разрешения: снимок масштабируется до масштаба подобия (INTER_AREA / INTER_LINEAR), затем warpAffine. */
    static Mat fromFull(Mat fullGray, double[][] lmFull, double[][] template, int w, int h) {
        Mat m = FaceAlignment.similarity(lmFull, template);
        double s = Math.hypot(m.get(0, 0)[0], m.get(1, 0)[0]);
        Mat scaled = new Mat();
        Imgproc.resize(fullGray, scaled, new Size(Math.max(1, Math.round(fullGray.cols() * s)), Math.max(1, Math.round(fullGray.rows() * s))),
                0, 0, s < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        double sx = scaled.cols() / (double) fullGray.cols();
        double sy = scaled.rows() / (double) fullGray.rows();
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = lmFull[j][0] * sx;
            lm[j][1] = lmFull[j][1] * sy;
        }
        Mat out = new Mat();
        Imgproc.warpAffine(scaled, out, FaceAlignment.similarity(lm, template), new Size(w, h), Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE);
        return out;
    }

    // ---------------------------------------------------------------- кэш

    private static long cachedDetTime;

    static String signature(List<Sample> samples, float score, SettingsStore settings) throws IOException {
        StringBuilder s = new StringBuilder("v2|").append(score).append('|').append(OWN_INPUT_SCALE).append('|')
                .append(PASS2_MARGIN).append('|').append(PASS2_FACE).append('|');
        for (Sample x : samples) {
            s.append(x.file()).append(':').append(Files.size(x.file())).append(':').append(Files.getLastModifiedTime(x.file()).toMillis())
                    .append(';');
        }
        return s.toString();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Det> loadCache(Path file, String signature) {
        if (!Files.exists(file)) return null;
        try (java.io.ObjectInputStream in = new java.io.ObjectInputStream(Files.newInputStream(file))) {
            if (!signature.equals(in.readObject())) return null;
            Map<String, Det> map = (Map<String, Det>) in.readObject();
            cachedDetTime = in.readLong();
            System.out.println("Детекции — из кэша " + file);
            return map;
        } catch (IOException | ClassNotFoundException | ClassCastException e) {
            return null;
        }
    }

    static void saveCache(Path file, String signature, Map<String, Det> dets, long detTime) throws IOException {
        try (java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(Files.newOutputStream(file))) {
            out.writeObject(signature);
            out.writeObject(new LinkedHashMap<>(dets));
            out.writeLong(detTime);
        }
    }

    // ---------------------------------------------------------------- оценка

    /** Модель человека: подпространство (среднее, базис, k — копии из SubspaceModel один раз) или шаблон SFace. */
    record Model(double[] mean, double[][] basis, int k, double[] template) {
        static Model of(SubspaceModel m) {
            return new Model(m.getMeanVector(), m.getBasisMatrix(), m.getK(), null);
        }
    }

    static Model train(SubspaceTrainer trainer, List<double[]> vectors, boolean sface) {
        if (vectors.size() < 2) return null;
        double[][] x = vectors.toArray(new double[0][]);
        if (sface) {
            double[] t = new double[x[0].length];
            for (double[] v : x) for (int i = 0; i < t.length; i++) t[i] += v[i];
            OwnEvaluation.normalize(t);
            return new Model(null, null, 0, t);
        }
        return Model.of(trainer.train(x, Math.min(K, x.length - 1)));
    }

    /** Расстояния до галереи по конфигурациям (вариант|конфигурация → файл → {ε₁, ε₂, индекс}); общие для ε₁/ε₂ и ε. */
    private static final Map<String, Map<String, double[]>> DIST = new HashMap<>();

    /** Расстояния всех найденных снимков до галереи конфигурации r (параллельно; результат от порядка не зависит). */
    static Map<String, double[]> distances(Variant v, int r, Model[] gallery, Map<String, Feat> feats) {
        return DIST.computeIfAbsent(v + "|" + r, key -> {
            List<String> files = feats.entrySet().stream().filter(e -> v.of(e.getValue()) != null).map(Map.Entry::getKey).toList();
            double[][] out = new double[files.size()][];
            IntStream.range(0, files.size()).parallel().forEach(i -> out[i] = nearest(gallery, v.of(feats.get(files.get(i)))));
            Map<String, double[]> m = new HashMap<>();
            for (int i = 0; i < files.size(); i++) m.put(files.get(i), out[i]);
            return m;
        });
    }

    /** {оценка, предсказанный индекс галереи} по расстояниям {ε₁, ε₂, индекс}. */
    static double[] pick(double[] d, Score sc) {
        return new double[] {sc == Score.RATIO ? d[0] / d[1] : d[0], d[2]};
    }


    /** {лучшее расстояние, второе, индекс лучшего}. */
    static double[] nearest(Model[] gallery, double[] x) {
        double best = Double.MAX_VALUE;
        double second = Double.MAX_VALUE;
        int predicted = -1;
        for (int p = 0; p < gallery.length; p++) {
            if (gallery[p] == null) continue;
            double d;
            if (gallery[p].template() != null) {
                double dot = 0;
                for (int i = 0; i < x.length; i++) dot += x[i] * gallery[p].template()[i];
                d = 1 - dot;
            } else {
                Model m = gallery[p];
                d = SubspaceRecognizer.reconstructionError(x, m.mean(), m.basis(), m.k());
            }
            if (d < best) {
                second = best;
                best = d;
                predicted = p;
            } else if (d < second) {
                second = d;
            }
        }
        return new double[] {best, second, predicted};
    }

    /** Попытки чужих одной конфигурации: оценки по людям (NaN — отказ детектора). */
    record Impostors(Map<String, double[]> byPerson) {
        double[] detected() {
            return byPerson.values().stream().flatMapToDouble(Arrays::stream).filter(v -> !Double.isNaN(v)).toArray();
        }

        int refused() {
            return (int) byPerson.values().stream().flatMapToDouble(Arrays::stream).filter(Double::isNaN).count();
        }
    }

    static Impostors impostors(Map<String, List<Sample>> base, List<String> persons, Map<String, Feat> feats, Variant v,
                               Map<String, double[]> dist, Score sc, boolean cameraA) {
        Map<String, double[]> map = new LinkedHashMap<>();
        for (String p : persons) {
            List<Sample> list = base.get(p).stream().filter(s -> !cameraA || s.camera() == 'a').toList();
            double[] vals = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                double[] x = v.of(feats.get(list.get(i).file().toString()));
                vals[i] = x == null ? Double.NaN : pick(dist.get(list.get(i).file().toString()), sc)[0];
            }
            map.put(p, vals);
        }
        return new Impostors(map);
    }

    /** FAR одной конфигурации при пороге θ: {принято попыток, попыток с детекцией, людей принято, людей с детекцией}. */
    static int[] far(Impostors imp, double theta) {
        int acc = 0;
        int n = 0;
        int accP = 0;
        int nP = 0;
        for (double[] vals : imp.byPerson().values()) {
            boolean any = false;
            boolean detected = false;
            for (double v : vals) {
                if (Double.isNaN(v)) continue;
                detected = true;
                n++;
                if (v <= theta) {
                    acc++;
                    any = true;
                }
            }
            if (detected) nP++;
            if (any) accP++;
        }
        return new int[] {acc, n, accP, nP};
    }

    private static void evaluate(StringBuilder text, Variant v, Score sc, List<List<OwnDataset.Moment>> moments,
                                 Map<String, List<Sample>> gt, List<String> gtPersons, List<List<List<Integer>>> gtTrain,
                                 Map<String, List<Sample>> muct, List<String> muctVal, List<String> muctCtrl,
                                 Map<String, List<Sample>> orl, Map<String, Feat> feats, SubspaceTrainer trainer) {
        boolean sface = v == Variant.SFACE;
        int nOwn = moments.size();
        int nGal = nOwn + gtPersons.size();
        int na = ALPHAS.length;
        int[] ownRej = new int[na];
        int[] ownDetRej = new int[1];
        int ownAtt = 0;
        int[] gtRej = new int[na];
        int gtDetRej = 0;
        int gtAtt = 0;
        int ownCorrect = 0;
        int gtCorrect = 0;
        // FAR по конфигурациям: [α][конфигурация] -> {x, n, xP, nP}; строки: MUCT все камеры, MUCT камера a, ORL.
        int[][][][] farCfg = new int[3][na][CONFIGS][];
        double[][] thetas = new double[na][CONFIGS];
        List<Double> marginOwn = new ArrayList<>();
        List<Double> marginGt = new ArrayList<>();
        int[] refused = new int[4];
        long evalStart = System.nanoTime();
        for (int r = 0; r < CONFIGS; r++) {
            System.out.printf(Locale.ROOT, "%s, %s: конфигурация %d/%d, %.0f с%n", v.label, sc.label, r + 1, CONFIGS,
                    (System.nanoTime() - evalStart) / 1e9);
            Model[] gallery = new Model[nGal];
            OwnEvaluation.Split[] splits = new OwnEvaluation.Split[nOwn];
            for (int p = 0; p < nOwn; p++) {
                List<OwnDataset.Moment> ms = moments.get(p);
                splits[p] = OwnEvaluation.split(ms, r % ms.size(), false);
                List<double[]> train = new ArrayList<>();
                for (int j : splits[p].train()) {
                    for (OwnDataset.Frame f : ms.get(j).frames()) {
                        if (f.quality() < OwnDataset.FAIR) continue;
                        double[] x = v.of(feats.get(f.file().toString()));
                        if (x != null) train.add(x);
                    }
                }
                gallery[p] = train(trainer, train, sface);
            }
            int split = r % GT_SPLITS;
            for (int g = 0; g < gtPersons.size(); g++) {
                List<Sample> imgs = gt.get(gtPersons.get(g));
                List<double[]> train = new ArrayList<>();
                for (int i : gtTrain.get(split).get(g)) {
                    double[] x = v.of(feats.get(imgs.get(i).file().toString()));
                    if (x != null) train.add(x);
                }
                gallery[nOwn + g] = train(trainer, train, sface);
            }
            Map<String, double[]> dist = distances(v, r, gallery, feats);
            Impostors val = impostors(muct, muctVal, feats, v, dist, sc, false);
            Impostors valA = impostors(muct, muctVal, feats, v, dist, sc, true);
            Impostors ctrl = impostors(muct, muctCtrl, feats, v, dist, sc, false);
            Impostors ctrlA = impostors(muct, muctCtrl, feats, v, dist, sc, true);
            Impostors orlCtrl = impostors(orl, new ArrayList<>(orl.keySet()), feats, v, dist, sc, false);
            if (r == 0) {
                refused[0] = val.refused();
                refused[1] = ctrl.refused();
                refused[2] = orlCtrl.refused();
            }
            double[] valScores = val.detected();
            double[] valAScores = valA.detected();
            double minImp = Arrays.stream(ctrl.detected()).min().orElse(Double.NaN);
            // Свои: контроль — представитель момента r.
            double maxOwn = Double.NEGATIVE_INFINITY;
            List<Double> ownScores = new ArrayList<>();
            for (int p = 0; p < nOwn; p++) {
                if (r >= moments.get(p).size()) continue;
                ownAtt++;
                double[] x = v.of(feats.get(moments.get(p).get(r).representative().file().toString()));
                if (x == null) {
                    ownDetRej[0]++;
                    ownScores.add(Double.POSITIVE_INFINITY);
                    continue;
                }
                double[] s = pick(dist.get(moments.get(p).get(r).representative().file().toString()), sc);
                if ((int) s[1] == p) ownCorrect++;
                ownScores.add(gallery[p] == null ? Double.POSITIVE_INFINITY : s[0]);
                maxOwn = Math.max(maxOwn, s[0]);
            }
            List<Double> gtScores = new ArrayList<>();
            double maxGt = Double.NEGATIVE_INFINITY;
            if (r < GT_SPLITS) {
                for (int g = 0; g < gtPersons.size(); g++) {
                    List<Sample> imgs = gt.get(gtPersons.get(g));
                    List<Integer> train = gtTrain.get(split).get(g);
                    for (int i = 0; i < imgs.size(); i++) {
                        if (train.contains(i)) continue;
                        gtAtt++;
                        double[] x = v.of(feats.get(imgs.get(i).file().toString()));
                        if (x == null) {
                            gtDetRej++;
                            gtScores.add(Double.POSITIVE_INFINITY);
                            continue;
                        }
                        double[] s = pick(dist.get(imgs.get(i).file().toString()), sc);
                        if ((int) s[1] == nOwn + g) gtCorrect++;
                        gtScores.add(gallery[nOwn + g] == null ? Double.POSITIVE_INFINITY : s[0]);
                        maxGt = Math.max(maxGt, s[0]);
                    }
                }
                marginGt.add(minImp - maxGt);
            }
            marginOwn.add(minImp - maxOwn);
            for (int a = 0; a < na; a++) {
                double theta = FaceEvaluation.neymanPearsonThreshold(valScores, ALPHAS[a]);
                double thetaA = FaceEvaluation.neymanPearsonThreshold(valAScores, ALPHAS[a]);
                thetas[a][r] = theta;
                for (double s : ownScores) if (s > theta) ownRej[a]++;
                for (double s : gtScores) if (s > theta) gtRej[a]++;
                farCfg[0][a][r] = far(ctrl, theta);
                farCfg[1][a][r] = far(ctrlA, thetaA);
                farCfg[2][a][r] = far(orlCtrl, theta);
            }
        }
        text.append(String.format(Locale.ROOT, "%n=== %s, скоринг %s ===%n", v.label, sc.label));
        text.append(String.format(Locale.ROOT, "argmin (галерея из %d): свои %d/%d, Georgia Tech %d/%d; отказы детектора на контроле: "
                + "свои %d/%d, Georgia Tech %d/%d; чужие: MUCT валидация %d, MUCT контроль %d, ORL %d попыток.%n",
                nGal, ownCorrect, ownAtt, gtCorrect, gtAtt, ownDetRej[0], ownAtt, gtDetRej, gtAtt, refused[0], refused[1], refused[2]));
        text.append(String.format(Locale.ROOT, "Запас на контроле (наименьшая оценка чужого MUCT минус наибольшая своего): свои — "
                + "мин %s / медиана %s (по 12 конфигурациям); Georgia Tech — мин %s / медиана %s (конфигурации 0–2).%n",
                q(marginOwn, 0), q(marginOwn, 0.5), q(marginGt, 0), q(marginGt, 0.5)));
        String[] rows = {"MUCT, все камеры", "MUCT, камера a (порог по валидации камеры a)", "ORL (порог MUCT все камеры)"};
        for (int a = 0; a < na; a++) {
            double[] th = thetas[a].clone();
            Arrays.sort(th);
            int gtDet = gtAtt - gtDetRej;
            int ownDet = ownAtt - ownDetRej[0];
            text.append(String.format(Locale.ROOT, "  α = %.3f (θ медиана %.6f): FRR свои %d/%d = %s %% (без отказов детектора %d/%d), "
                    + "Georgia Tech %d/%d = %s %% (без отказов %d/%d), общий %d/%d = %s %%%n",
                    ALPHAS[a], th[th.length / 2], ownRej[a], ownAtt, pct(ownRej[a], ownAtt), ownRej[a] - ownDetRej[0], ownDet,
                    gtRej[a], gtAtt, pct(gtRej[a], gtAtt), gtRej[a] - gtDetRej, gtDet,
                    ownRej[a] + gtRej[a], ownAtt + gtAtt, pct(ownRej[a] + gtRej[a], ownAtt + gtAtt)));
            for (int k = 0; k < 3; k++) {
                text.append("    FAR ").append(rows[k]).append(": ").append(farText(farCfg[k][a])).append('\n');
            }
        }
    }

    /** FAR по конфигурациям: худшая и медианная по попыткам и по людям, с ↑95 внутри конфигурации; сумма — справочно. */
    static String farText(int[][] cfg) {
        Integer[] byAtt = new Integer[cfg.length];
        Integer[] byPer = new Integer[cfg.length];
        for (int i = 0; i < cfg.length; i++) {
            byAtt[i] = i;
            byPer[i] = i;
        }
        Arrays.sort(byAtt, (x, y) -> Double.compare(cfg[x][0] / (double) cfg[x][1], cfg[y][0] / (double) cfg[y][1]));
        Arrays.sort(byPer, (x, y) -> Double.compare(cfg[x][2] / (double) cfg[x][3], cfg[y][2] / (double) cfg[y][3]));
        int[] w = cfg[byAtt[cfg.length - 1]];
        int[] m = cfg[byAtt[cfg.length / 2]];
        int[] wp = cfg[byPer[cfg.length - 1]];
        int[] mp = cfg[byPer[cfg.length / 2]];
        int sumX = 0;
        int sumN = 0;
        for (int[] c : cfg) {
            sumX += c[0];
            sumN += c[1];
        }
        return String.format(Locale.ROOT, "попытки — худш. %d/%d (↑95 %s %%), медиана %d/%d (↑95 %s %%); люди — худш. %d/%d "
                + "(↑95 %s %%), медиана %d/%d (↑95 %s %%); сумма по 12 конфигурациям справочно %d/%d",
                w[0], w[1], ub(w[0], w[1]), m[0], m[1], ub(m[0], m[1]), wp[2], wp[3], ub(wp[2], wp[3]), mp[2], mp[3], ub(mp[2], mp[3]),
                sumX, sumN);
    }

    private static String ub(int x, int n) {
        return String.format(Locale.ROOT, "%.2f", 100 * FaceEvaluation.binomialUpperBound(x, n, FaceEvaluation.CONFIDENCE));
    }

    private static String pct(int x, int n) {
        return n == 0 ? "—" : String.format(Locale.ROOT, "%.1f", 100.0 * x / n);
    }

    private static String q(List<Double> v, double p) {
        if (v.isEmpty()) return "—";
        double[] d = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return String.format(Locale.ROOT, "%+.4f", d[(int) Math.round(p * (d.length - 1))]);
    }

    // ---------------------------------------------------------------- шапка

    private static void header(StringBuilder text, OwnDataset own, List<List<OwnDataset.Moment>> moments,
                               Map<String, List<Sample>> gt, Map<String, List<Sample>> muct, Map<String, List<Sample>> orl,
                               List<String> muctVal, List<String> muctCtrl, Map<String, Feat> feats, List<Sample> samples,
                               float score, long seed) {
        text.append("FAR и FRR на расширенной галерее (шаг 5, задача FAR)\n");
        text.append("Оговорки:\n");
        text.append("  - чужие MUCT сняты в лаборатории (вебкамеры, 480×640), свои — телефоном (4000×3000): чужие отличаются\n"
                + "    и по виду снимка, FAR может быть занижен;\n");
        text.append("  - своя база — одна сессия (FRR своих оптимистичен); обучение своих — 5 моментов (уровень N = 5 по ORL);\n");
        text.append("  - Georgia Tech: пометки сессий в данных нет, обучение и контроль — из одних сессий (FRR оптимистичен);\n");
        text.append("  - для SFace возможное пересечение его обучающего набора с MUCT, Georgia Tech и ORL не проверено;\n");
        text.append("  - ORL — серые 92×112, лицо около 70 px: кадр SVD — увеличение (в варианте в) — вдвое больше), SFace на\n"
                + "    сером входе; FAR по ORL — справочно;\n");
        text.append("  - FAR по попыткам: граница Клоппера – Пирсона внутри конфигурации (оптимистична: снимки одного человека\n"
                + "    коррелируют); сумма по 12 конфигурациям — только справочно (одни и те же чужие).\n");
        int ownFrames = (int) samples.stream().filter(s -> s.base() == Base.OWN).count();
        StringBuilder mom = new StringBuilder();
        for (int p = 0; p < moments.size(); p++) mom.append(p == 0 ? "" : ", ").append(own.persons().get(p).name()).append(' ').append(moments.get(p).size());
        text.append(String.format(Locale.ROOT, "Галерея: свои 6 (пригодных моментов: %s; кадров %d), Georgia Tech %d человек × 15 "
                + "(обучение %d случайных, контроль остальные; %d разбиения, seed %d + номер).%n",
                mom, ownFrames, gt.size(), GT_TRAIN, GT_SPLITS, seed));
        text.append(String.format(Locale.ROOT, "Конфигураций %d: ротация своих r, разбиение Georgia Tech r mod %d; контроль Georgia Tech — "
                + "только в конфигурациях 0–2.%n", CONFIGS, GT_SPLITS));
        int valN = muctVal.stream().mapToInt(p -> muct.get(p).size()).sum();
        int ctrlN = muctCtrl.stream().mapToInt(p -> muct.get(p).size()).sum();
        int valA = muctVal.stream().mapToInt(p -> (int) muct.get(p).stream().filter(s -> s.camera() == 'a').count()).sum();
        int ctrlA = muctCtrl.stream().mapToInt(p -> (int) muct.get(p).stream().filter(s -> s.camera() == 'a').count()).sum();
        text.append(String.format(Locale.ROOT, "Чужие: MUCT %d человек по людям (seed %d): валидация %d человек, %d снимков (камера a %d); "
                + "контроль %d человек, %d снимков (камера a %d). ORL %d × %d — только контроль.%n",
                muct.size(), seed, muctVal.size(), valN, valA, muctCtrl.size(), ctrlN, ctrlA, orl.size(), OrlDataset.IMAGES_PER_PERSON));
        text.append(String.format(Locale.ROOT, "Детекция: двухпроходный YuNet (2′), порог %.1f для всех баз; вход первого прохода — своя "
                + "база ×%.2f, остальные — родное разрешение; второй проход — поля %.0f %%, лицо около %.0f px; точки — второго прохода.%n",
                score, OWN_INPUT_SCALE, 100 * PASS2_MARGIN, PASS2_FACE));
        text.append("Порог — Нейман – Пирсон по чужим валидации MUCT (с детекцией) в каждой конфигурации; отказы детектора у чужих\n"
                + "исключены из знаменателя FAR (число — в строке argmin), у своих и Georgia Tech — отказ в FRR.\n");
        text.append(String.format(Locale.ROOT, "SVD: k = %d, ε₁/ε₂ (основной) и ε; SFace: шаблон — нормированное среднее, 1 − cos₁.%n", K));
    }
}
