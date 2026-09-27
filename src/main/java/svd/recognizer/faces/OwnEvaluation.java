package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * Сравнение способов детекции и выравнивания на своей базе (шаг 5, задание (г);
 * одна команда: {@code mvn compile exec:java@faces-own-eval}). Отчёт —
 * reports/faces/own/own_eval.txt (вне git).
 *
 * Способы: (1) YuNet + alignCrop — только вход SFace; (2) YuNet + своё аффинное
 * (5 точек); (3) Хаар (лицо и глаза) + подобие по двум глазам; (4) DFFS
 * (пространство лиц и глаз — все 40 человек ORL, свои кадры аффинного
 * выравнивания) + подобие по двум глазам. Для SVD — серые кадры
 * faces.frame.width × faces.frame.height из снимка, уменьшенного до масштаба
 * {@value #INPUT_SCALE} (лицо около 110 px); для SFace — цветной alignCrop 112×112.
 *
 * Протокол — TASKS.md, «Своя база» и план (г): единица учёта — пригодный момент;
 * ротации r = 0…11; у человека с m моментами контроль — момент r (если r &lt; m),
 * обучение — 5 моментов по кругу после (r mod m), только кадры «+» и «+-»;
 * валидация — остальные; при r ≥ m человек не контролируется. Строгий вариант:
 * моменты в пределах ±1 с (EXIF) от контрольного — в валидацию, обучение — до 5
 * следующих не-соседних. Контроль и валидация — представитель момента.
 * SVD: k = 4 (основной), k = n − 1, k по доле энергии η = 0,95; скоринг ε₁/ε₂.
 * SFace: шаблон — нормированное среднее признаков, оценка 1 − cos₁.
 * Закрытая галерея из 6 — точность argmin. Исключение одного человека: галерея
 * из 5, шестой — чужой; порог для каждой пары (чужой, ротация) — α = 0 (чуть
 * ниже наименьшей оценки чужого на его валидационных моментах этой ротации);
 * запас — наименьшая оценка чужого минус наибольшая оценка своих на валидации.
 *
 * @author ssv
 */
public final class OwnEvaluation {

    static final double INPUT_SCALE = 0.25;
    static final String CACHE_NAME = "own_detections.cache";
    static final int ROTATIONS = 12;
    static final int TRAIN_MOMENTS = 5;
    static final double NEIGHBOR_SECONDS = 1.0;
    static final double ETA = 0.95;
    static final int K_MAIN = 4;
    static final double[] DFFS_SCALES = {0.15, 0.165, 0.18, 0.20, 0.22, 0.24, 0.27, 0.29, 0.32, 0.35};

    enum Method {
        YUNET_OWN("(2) YuNet + своё аффинное, SVD"), HAAR("(3) Хаар + по глазам, SVD"),
        DFFS("(4) DFFS + по глазам, SVD"), SFACE("(1) YuNet + alignCrop, SFace");

        final String label;

        Method(String label) {
            this.label = label;
        }
    }

    /** Детекция одного кадра одним способом: найдено, рамка и глаза (полный снимок), вектор признака. */
    record Det(boolean found, double[] box, double[] left, double[] right, double[] nose, double[] vector)
            implements java.io.Serializable {
        static final Det NONE = new Det(false, null, null, null, null, null);

        boolean eyes() {
            return left != null && right != null;
        }
    }

    /** Разбиение одного человека в одной конфигурации (c = r mod m). */
    record Split(int control, List<Integer> train, List<Integer> validation) {}

    private OwnEvaluation() {
    }

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        OwnDataset dataset = OwnDataset.load(Paths.get(settings.loadFacesOwnDir()));
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        int frameW = settings.loadFacesFrameWidth();
        int frameH = settings.loadFacesFrameHeight();
        float score = settings.loadFacesOwnDetectorScore();
        double[][] template = FaceAlignment.ownTemplate(frameW, frameH);

        // DFFS: пространства лиц и глаз по всем 40 людям ORL (кадры своего аффинного выравнивания).
        FaceSpace[] spaces = trainDffs(settings, modelsDir);

        List<OwnDataset.Person> persons = dataset.persons();
        int n = persons.size();
        List<List<OwnDataset.Moment>> moments = new ArrayList<>();
        for (OwnDataset.Person p : persons) moments.add(p.usableMoments());

        // Кадры для обработки: представители и кадры «+», «+-» пригодных моментов.
        Map<Path, OwnDataset.Frame> needed = new LinkedHashMap<>();
        for (List<OwnDataset.Moment> ms : moments) {
            for (OwnDataset.Moment m : ms) {
                needed.put(m.representative().file(), m.representative());
                for (OwnDataset.Frame f : m.frames()) {
                    if (f.quality() >= OwnDataset.FAIR) needed.put(f.file(), f);
                }
            }
        }
        FaceDetectorYN yunet = OwnDetectorThreshold.create(modelsDir, score);
        FaceRecognizerSF sface = FaceRecognizerSF.create(modelsDir.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        CascadeClassifier haarFace = HaarDebug.load(modelsDir.resolve(HaarDebug.FACE_FILE));
        CascadeClassifier haarEye = HaarDebug.load(modelsDir.resolve(HaarDebug.EYE_FILE));
        Map<Method, Map<Path, Det>> det = new HashMap<>();
        for (Method m : Method.values()) det.put(m, new HashMap<>());
        long[] time = new long[Method.values().length];
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "own");
        Files.createDirectories(outDir);
        Path cacheFile = outDir.resolve(CACHE_NAME);
        String signature = signature(needed, score, frameW, frameH);
        boolean fresh = args.length > 0 && args[0].equals("fresh");
        boolean cached = !fresh && loadCache(cacheFile, signature, det, time);
        if (!cached) {
        int done = 0;
        for (OwnDataset.Frame f : needed.values()) {
            Mat color = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(f.file())), Imgcodecs.IMREAD_COLOR);
            Mat gray = new Mat();
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
            Size small = new Size(Math.round(color.cols() * INPUT_SCALE), Math.round(color.rows() * INPUT_SCALE));
            Mat smallColor = new Mat();
            Imgproc.resize(color, smallColor, small, 0, 0, Imgproc.INTER_AREA);
            Mat smallGray = new Mat();
            Imgproc.cvtColor(smallColor, smallGray, Imgproc.COLOR_BGR2GRAY);

            long t0 = System.nanoTime();
            Det[] y = yunet(yunet, sface, smallColor, smallGray, template, frameW, frameH);
            time[Method.YUNET_OWN.ordinal()] += System.nanoTime() - t0;
            det.get(Method.YUNET_OWN).put(f.file(), y[0]);
            det.get(Method.SFACE).put(f.file(), y[1]);

            t0 = System.nanoTime();
            HaarDebug.Found h = HaarDebug.detectOwn(gray, INPUT_SCALE, haarFace, haarEye);
            det.get(Method.HAAR).put(f.file(), byEyes(h.face() == null ? null
                    : new double[] {h.face().x, h.face().y, h.face().width, h.face().height}, h.left(), h.right(),
                    smallGray, template, frameW, frameH));
            time[Method.HAAR.ordinal()] += System.nanoTime() - t0;

            t0 = System.nanoTime();
            DffsDebug.Found d = DffsDebug.search(gray, spaces[0], spaces[1], spaces[2], template, DffsDebug.PAD, DFFS_SCALES);
            det.get(Method.DFFS).put(f.file(), byEyes(new double[] {d.x(), d.y(), d.w(), d.h()}, d.left(), d.right(),
                    smallGray, template, frameW, frameH));
            time[Method.DFFS.ordinal()] += System.nanoTime() - t0;
            done++;
            if (done % 10 == 0) System.out.println("Обработано кадров: " + done + "/" + needed.size());
        }
            saveCache(cacheFile, signature, det, time);
        }

        Map<String, EyeLabeler.Label> eyes = EyeLabeler.readCsv(dataset.root().resolve(EyeLabeler.CSV_NAME));
        StringBuilder text = new StringBuilder();
        header(text, dataset, moments, needed.size(), score, frameW, frameH, spaces);
        detectionTable(text, dataset, moments, needed, det, eyes, time);
        Map<Path, Integer> yaw = yawGroups(dataset, moments, det.get(Method.YUNET_OWN), eyes);
        int[] groupCount = new int[3];
        for (int g : yaw.values()) groupCount[g]++;
        text.append(String.format(Locale.ROOT, "Поворот головы (представители, r = (x_нос − x_глаз_л)/(x_глаз_п − x_глаз_л), нос — YuNet,%n"
                + "глаза — ручная разметка ok, иначе YuNet; фронтальные |r − 0,5| ≤ %.1f): фронтальных %d, повёрнутых %d, "
                + "не определено %d.%n", FRONTAL_TOLERANCE, groupCount[FRONTAL], groupCount[TURNED], groupCount[UNKNOWN]));
        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        for (boolean strict : new boolean[] {false, true}) {
            trainingCounts(text, moments, det, strict);
        }
        recognitionTable(text, moments, det, yaw, trainer);

        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("own_eval.txt"), StandardCharsets.UTF_8))) {
            out.print(text);
        }
        System.out.print(text);
    }

    // ---------------------------------------------------------------- кэш детекций

    /**
     * Подпись кэша: параметры детекции и список кадров (размер и время изменения
     * файла). Изменения кода детекции подпись не видит — после них запускать с
     * аргументом fresh (-Dexec.args=fresh) или удалить кэш.
     */
    static String signature(Map<Path, OwnDataset.Frame> needed, float score, int frameW, int frameH) throws IOException {
        StringBuilder s = new StringBuilder();
        s.append(score).append('|').append(frameW).append('x').append(frameH).append('|').append(INPUT_SCALE).append('|')
                .append(Arrays.toString(DFFS_SCALES)).append('|').append(HaarDebug.OWN_MIN_FACE_FRACTION).append('|')
                .append(HaarDebug.EYE_FACE_WIDTH).append('|').append(DffsDebug.EYE_SEARCH).append('|')
                .append(DffsDebug.PAIR_DX).append('|').append(DffsDebug.PAIR_DY).append('|');
        for (Path p : needed.keySet()) {
            s.append(p).append(':').append(Files.size(p)).append(':').append(Files.getLastModifiedTime(p).toMillis()).append(';');
        }
        return s.toString();
    }

    @SuppressWarnings("unchecked")
    static boolean loadCache(Path file, String signature, Map<Method, Map<Path, Det>> det, long[] time) {
        if (!Files.exists(file)) return false;
        try (java.io.ObjectInputStream in = new java.io.ObjectInputStream(Files.newInputStream(file))) {
            if (!signature.equals(in.readObject())) return false;
            Map<String, Det[]> map = (Map<String, Det[]>) in.readObject();
            long[] t = (long[]) in.readObject();
            for (Map.Entry<String, Det[]> e : map.entrySet()) {
                for (Method m : Method.values()) det.get(m).put(Paths.get(e.getKey()), e.getValue()[m.ordinal()]);
            }
            System.arraycopy(t, 0, time, 0, time.length);
            System.out.println("Детекции — из кэша " + file);
            return true;
        } catch (IOException | ClassNotFoundException | ClassCastException e) {
            return false;
        }
    }

    static void saveCache(Path file, String signature, Map<Method, Map<Path, Det>> det, long[] time) throws IOException {
        Map<String, Det[]> map = new LinkedHashMap<>();
        for (Path p : det.get(Method.YUNET_OWN).keySet()) {
            Det[] d = new Det[Method.values().length];
            for (Method m : Method.values()) d[m.ordinal()] = det.get(m).get(p);
            map.put(p.toString(), d);
        }
        try (java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(Files.newOutputStream(file))) {
            out.writeObject(signature);
            out.writeObject(map);
            out.writeObject(time);
        }
    }

    // ---------------------------------------------------------------- детекция

    static FaceSpace[] trainDffs(SettingsStore settings, Path modelsDir) throws IOException {
        Path orl = Paths.get(settings.loadFacesDatasetDir());
        FaceAlignment alignment = new FaceAlignment(modelsDir, OrlDataset.WIDTH, OrlDataset.HEIGHT, settings.loadFacesDetectorScore());
        double[][] t = FaceAlignment.ownTemplate(OrlDataset.WIDTH, OrlDataset.HEIGHT);
        List<Mat> faces = new ArrayList<>();
        List<Mat> left = new ArrayList<>();
        List<Mat> right = new ArrayList<>();
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                FaceAlignment.Result r = alignment.process(FaceAlignment.readGray(orl.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm")), 0.0);
                if (!r.detected()) continue;
                faces.add(r.ownAffineGray());
                left.add(DffsDebug.eyePatch(r.ownAffineGray(), t[0][0], t[0][1]));
                right.add(DffsDebug.eyePatch(r.ownAffineGray(), t[1][0], t[1][1]));
            }
        }
        return new FaceSpace[] {FaceSpace.train(faces, DffsDebug.ETA, DffsDebug.K_MAX_FACE),
            FaceSpace.train(left, DffsDebug.ETA, DffsDebug.K_MAX_EYE), FaceSpace.train(right, DffsDebug.ETA, DffsDebug.K_MAX_EYE)};
    }

    /** YuNet на уменьшенном кадре: {своё аффинное (SVD), alignCrop + SFace}. */
    static Det[] yunet(FaceDetectorYN detector, FaceRecognizerSF sface, Mat smallColor, Mat smallGray, double[][] template,
                       int frameW, int frameH) {
        detector.setInputSize(smallColor.size());
        Mat faces = new Mat();
        detector.detect(smallColor, faces);
        if (faces.rows() == 0) return new Det[] {Det.NONE, Det.NONE};
        int best = 0;
        for (int i = 1; i < faces.rows(); i++) {
            if (faces.get(i, 14)[0] > faces.get(best, 14)[0]) best = i;
        }
        Mat row = faces.row(best);
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = row.get(0, 4 + 2 * j)[0];
            lm[j][1] = row.get(0, 5 + 2 * j)[0];
        }
        double[] box = {row.get(0, 0)[0] / INPUT_SCALE, row.get(0, 1)[0] / INPUT_SCALE, row.get(0, 2)[0] / INPUT_SCALE,
            row.get(0, 3)[0] / INPUT_SCALE};
        double[] l = {lm[0][0] / INPUT_SCALE, lm[0][1] / INPUT_SCALE};
        double[] r = {lm[1][0] / INPUT_SCALE, lm[1][1] / INPUT_SCALE};
        Mat own = new Mat();
        Imgproc.warpAffine(smallGray, own, FaceAlignment.similarity(lm, template), new Size(frameW, frameH),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        Mat crop = new Mat();
        sface.alignCrop(smallColor, row, crop);
        Mat feature = new Mat();
        sface.feature(crop, feature);
        Mat f64 = new Mat();
        feature.convertTo(f64, CvType.CV_64F);
        double[] v = new double[(int) f64.total()];
        f64.get(0, 0, v);
        normalize(v);
        double[] nose = {lm[2][0] / INPUT_SCALE, lm[2][1] / INPUT_SCALE};
        return new Det[] {new Det(true, box, l, r, nose, FaceAlignment.toVector(own)), new Det(true, box, l, r, nose, v)};
    }

    /** Выравнивание по двум глазам (координаты полного снимка) на уменьшенном кадре. */
    static Det byEyes(double[] box, double[] left, double[] right, Mat smallGray, double[][] template, int frameW, int frameH) {
        if (box == null) return Det.NONE;
        if (left == null || right == null) return new Det(true, box, left, right, null, null);
        Mat a = HaarDebug.align(smallGray, new double[] {left[0] * INPUT_SCALE, left[1] * INPUT_SCALE},
                new double[] {right[0] * INPUT_SCALE, right[1] * INPUT_SCALE}, template, frameW, frameH);
        return new Det(true, box, left, right, null, FaceAlignment.toVector(a));
    }

    static void normalize(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        s = Math.sqrt(s);
        for (int i = 0; i < v.length; i++) v[i] /= s;
    }

    // ---------------------------------------------------------------- разбиение

    /** Разбиение человека с моментами ms в конфигурации c; strict — соседи ±1 с контрольного в валидацию. */
    static Split split(List<OwnDataset.Moment> ms, int c, boolean strict) {
        int m = ms.size();
        List<Integer> train = new ArrayList<>();
        List<Integer> val = new ArrayList<>();
        for (int k = 1; k < m; k++) {
            int j = (c + k) % m;
            boolean neighbor = strict && neighbors(ms.get(j), ms.get(c));
            if (!neighbor && train.size() < TRAIN_MOMENTS) train.add(j);
            else val.add(j);
        }
        return new Split(c, train, val);
    }

    static boolean neighbors(OwnDataset.Moment a, OwnDataset.Moment b) {
        for (OwnDataset.Frame x : a.frames()) {
            for (OwnDataset.Frame y : b.frames()) {
                if (Math.abs(x.time() - y.time()) <= NEIGHBOR_SECONDS) return true;
            }
        }
        return false;
    }

    /** Обучающие векторы: кадры «+» и «+-» обучающих моментов с успешной детекцией. */
    static List<double[]> trainVectors(List<OwnDataset.Moment> ms, Split s, Map<Path, Det> det) {
        List<double[]> list = new ArrayList<>();
        for (int j : s.train()) {
            for (OwnDataset.Frame f : ms.get(j).frames()) {
                if (f.quality() < OwnDataset.FAIR) continue;
                Det d = det.get(f.file());
                if (d != null && d.vector() != null) list.add(d.vector());
            }
        }
        return list;
    }

    // ---------------------------------------------------------------- модели и оценки

    /** Модель человека: для SVD — подпространство, для SFace — шаблон (нормированное среднее). */
    record Model(SubspaceModel subspace, double[] template, int k) {}

    enum KVariant {
        K4("k = 4"), N1("k = n − 1"), ETA("k по η = 0,95");

        final String label;

        KVariant(String label) {
            this.label = label;
        }
    }

    static Model train(SubspaceTrainer trainer, List<double[]> vectors, boolean sface, KVariant kv) {
        if (vectors.size() < 2) return null;
        double[][] x = vectors.toArray(new double[0][]);
        if (sface) {
            double[] t = new double[x[0].length];
            for (double[] v : x) for (int i = 0; i < t.length; i++) t[i] += v[i];
            normalize(t);
            return new Model(null, t, 0);
        }
        SubspaceModel m = switch (kv) {
            case K4 -> trainer.train(x, Math.min(K_MAIN, x.length - 1));
            case N1 -> trainer.train(x, x.length - 1);
            case ETA -> trainer.trainByEnergy(x, ETA);
        };
        return new Model(m, null, m.getK());
    }

    static double distance(Model m, double[] x) {
        if (m.template() != null) {
            double dot = 0;
            for (int i = 0; i < x.length; i++) dot += x[i] * m.template()[i];
            return 1 - dot;
        }
        return SubspaceRecognizer.reconstructionError(x, m.subspace().getMeanVector(), m.subspace().getBasisMatrix(), m.k());
    }

    /** {оценка, предсказанный человек}: SVD — ε₁/ε₂, SFace — 1 − cos₁; галерея — люди с моделями. */
    static double[] score(Model[] gallery, double[] x, boolean sface, int exclude) {
        double best = Double.MAX_VALUE;
        double second = Double.MAX_VALUE;
        int predicted = -1;
        for (int p = 0; p < gallery.length; p++) {
            if (p == exclude || gallery[p] == null) continue;
            double d = distance(gallery[p], x);
            if (d < best) {
                second = best;
                best = d;
                predicted = p;
            } else if (d < second) {
                second = d;
            }
        }
        return new double[] {sface ? best : best / second, predicted};
    }

    // ---------------------------------------------------------------- отчёт

    private static void header(StringBuilder text, OwnDataset dataset, List<List<OwnDataset.Moment>> moments, int frames,
                               float score, int frameW, int frameH, FaceSpace[] spaces) {
        text.append("Сравнение способов детекции и выравнивания на своей базе (шаг 5, задание (г))\n");
        text.append("База: ").append(dataset.root()).append("; кадров обработано ").append(frames)
                .append(" (представители пригодных моментов и кадры «+», «+-»).\n");
        text.append("Оговорки:\n");
        text.append("  - одна сессия съёмки (все снимки за 2 минуты): FRR оптимистичен; главное условие достоверности —\n"
                + "    повторная съёмка в другой день; моменты почти непрерывны (у Mol серий нет), строгий вариант (±1 с) —\n"
                + "    рядом с основным;\n");
        text.append("  - чужих 6 (исключение одного человека): FAR и порог статистически не обоснованы (при 0/6 верхняя\n"
                + "    95 % граница по людям — 39 %); в каждой галерее 5–11 попыток одного чужого на валидации, поэтому\n"
                + "    рабочая точка — α = 0; чужой один и тот же на валидации и контроле (утечка на уровне человека);\n");
        text.append("  - обучение — 5 моментов (уровень N = 5 по ORL, не N ≥ 8): ограничение базы;\n");
        text.append(String.format(Locale.ROOT, "  - порог YuNet %.1f (faces.own.detector.score) выбран на этой же базе (между 0,5 и 0,7 отсекаются\n"
                + "    только кадры без видимого лица и лишние рамки);%n", score));
        text.append("  - представитель момента и пригодность — по пометке качества Хозяина в именах (до результатов алгоритмов).\n");
        StringBuilder unusable = new StringBuilder();
        StringBuilder poor = new StringBuilder();
        StringBuilder counts = new StringBuilder();
        for (int p = 0; p < dataset.persons().size(); p++) {
            OwnDataset.Person person = dataset.persons().get(p);
            List<String> u = new ArrayList<>();
            for (OwnDataset.Moment m : person.moments()) if (!m.usable()) u.add(hms(m.second()));
            if (!u.isEmpty()) unusable.append(' ').append(person.name()).append(": ").append(String.join(", ", u)).append(';');
            for (OwnDataset.Moment m : moments.get(p)) {
                if (m.representative().quality() == OwnDataset.POOR) {
                    poor.append(' ').append(person.name()).append(' ').append(hms(m.second())).append(';');
                }
            }
            counts.append(' ').append(person.name()).append(' ').append(moments.get(p).size());
        }
        text.append("Пригодных моментов:").append(counts).append(".\n");
        text.append("Непригодные моменты:").append(unusable.length() == 0 ? " нет." : unusable).append('\n');
        text.append("Моменты с лучшим кадром «+--»:").append(poor.length() == 0 ? " нет." : poor).append('\n');
        text.append(String.format(Locale.ROOT, "Вход YuNet и Хаара (лицо) — снимок ×%.2f (750×1000); кадр SVD %d×%d из того же уменьшенного снимка;%n"
                + "Хаар: лицо — наименьшее %.2f меньшей стороны входа, глаза — в рамке лица в полном разрешении;%n"
                + "DFFS: ORL 40 человек, лица k = %d, глаза k = %d/%d, поиск по всему кадру, масштабы %s от полного снимка.%n",
                INPUT_SCALE, frameW, frameH, HaarDebug.OWN_MIN_FACE_FRACTION, spaces[0].k(), spaces[1].k(), spaces[2].k(),
                Arrays.toString(DFFS_SCALES)));
    }


    static final int FRONTAL = 0;
    static final int TURNED = 1;
    static final int UNKNOWN = 2;
    static final String[] GROUP_NAMES = {"фронтальные", "повёрнутые", "не определено"};
    static final double FRONTAL_TOLERANCE = 0.1;

    /**
     * Группа поворота представителя: r = (x_нос − x_глаз_л) / (x_глаз_п − x_глаз_л); нос — YuNet, глаза —
     * ручная разметка (где flag ok), иначе YuNet; |r − 0,5| ≤ 0,1 — фронтальный. Без YuNet — не определено.
     */
    static Map<Path, Integer> yawGroups(OwnDataset dataset, List<List<OwnDataset.Moment>> moments, Map<Path, Det> yunet,
                                        Map<String, EyeLabeler.Label> eyes) {
        Map<Path, Integer> groups = new HashMap<>();
        for (List<OwnDataset.Moment> ms : moments) {
            for (OwnDataset.Moment m : ms) {
                Path file = m.representative().file();
                Det d = yunet.get(file);
                if (d == null || !d.found()) {
                    groups.put(file, UNKNOWN);
                    continue;
                }
                EyeLabeler.Label l = eyes.get(dataset.relative(file));
                double xl = l != null && l.flag().equals("ok") ? l.lx() : d.left()[0];
                double xr = l != null && l.flag().equals("ok") ? l.rx() : d.right()[0];
                double r = (d.nose()[0] - xl) / (xr - xl);
                groups.put(file, Math.abs(r - 0.5) <= FRONTAL_TOLERANCE ? FRONTAL : TURNED);
            }
        }
        return groups;
    }
    private static String hms(int second) {
        return String.format(Locale.ROOT, "%02d%02d%02d", second / 3600, second / 60 % 60, second % 60);
    }

    private static void detectionTable(StringBuilder text, OwnDataset dataset, List<List<OwnDataset.Moment>> moments,
                                       Map<Path, OwnDataset.Frame> needed, Map<Method, Map<Path, Det>> det,
                                       Map<String, EyeLabeler.Label> eyes, long[] time) {
        text.append("\n=== Детекция и глаза (ошибка глаз — max(|Δлев|, |Δправ|) / межзрачковое ручной разметки; на профилях\n"
                + "межзрачковое мало и нормированная ошибка раздувается; skipped — только в доле найденных) ===\n");
        List<OwnDataset.Frame> reps = new ArrayList<>();
        for (List<OwnDataset.Moment> ms : moments) for (OwnDataset.Moment m : ms) reps.add(m.representative());
        for (Method m : new Method[] {Method.YUNET_OWN, Method.HAAR, Method.DFFS}) {
            Map<Path, Det> d = det.get(m);
            int foundRep = 0;
            int foundAll = 0;
            for (OwnDataset.Frame f : reps) if (d.get(f.file()).found()) foundRep++;
            for (OwnDataset.Frame f : needed.values()) if (d.get(f.file()).found()) foundAll++;
            List<Double> ok = new ArrayList<>();
            List<Double> doubtful = new ArrayList<>();
            int labeled = 0;
            int boxHit = 0;
            int noEyes = 0;
            for (OwnDataset.Frame f : reps) {
                EyeLabeler.Label l = eyes.get(dataset.relative(f.file()));
                if (l == null || Double.isNaN(l.lx())) continue;
                labeled++;
                Det x = d.get(f.file());
                if (x.found() && inBox(x.box(), l.lx(), l.ly()) && inBox(x.box(), l.rx(), l.ry())) boxHit++;
                if (!x.eyes()) {
                    noEyes++;
                    continue;
                }
                double ipd = Math.hypot(l.rx() - l.lx(), l.ry() - l.ly());
                double e = Math.max(Math.hypot(x.left()[0] - l.lx(), x.left()[1] - l.ly()),
                        Math.hypot(x.right()[0] - l.rx(), x.right()[1] - l.ry())) / ipd;
                (l.flag().equals("ok") ? ok : doubtful).add(e);
            }
            String name = m == Method.YUNET_OWN ? "(1), (2) YuNet" : m.label.substring(0, m.label.indexOf(','));
            text.append(String.format(Locale.ROOT, "%s: лицо найдено — представители %d/%d, все кадры %d/%d; время %.0f мс на кадр%n",
                    name, foundRep, reps.size(), foundAll, needed.size(), time[m.ordinal()] / 1e6 / needed.size()));
            text.append(String.format(Locale.ROOT, "  размечено %d: рамка содержит обе точки глаз %d/%d; глаза не найдены %d;%n"
                    + "  глаза ok: %s;%n  глаза doubtful: %s%n", labeled, boxHit, labeled, noEyes,
                    DffsDebug.quantiles(ok), DffsDebug.quantiles(doubtful)));
        }
    }

    private static boolean inBox(double[] b, double x, double y) {
        return b != null && x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3];
    }

    private static void trainingCounts(StringBuilder text, List<List<OwnDataset.Moment>> moments,
                                       Map<Method, Map<Path, Det>> det, boolean strict) {
        text.append(strict ? "\nОбучающие кадры (строгий вариант ±1 с): " : "\n=== Обучающие кадры, вошедшие в модель (мин–макс по конфигурациям) ===\n"
                + "Основной вариант: ");
        StringBuilder moms = new StringBuilder();
        for (int p = 0; p < moments.size(); p++) {
            int lo = Integer.MAX_VALUE;
            int hi = 0;
            for (int c = 0; c < moments.get(p).size(); c++) {
                int t = split(moments.get(p), c, strict).train().size();
                lo = Math.min(lo, t);
                hi = Math.max(hi, t);
            }
            moms.append(lo == hi ? lo : lo + "–" + hi).append(p + 1 < moments.size() ? "/" : "");
        }
        text.append("обучающих моментов по людям ").append(moms).append('\n');
        for (Method m : Method.values()) {
            StringBuilder line = new StringBuilder("  ").append(m.label).append(":");
            for (int p = 0; p < moments.size(); p++) {
                int lo = Integer.MAX_VALUE;
                int hi = 0;
                for (int c = 0; c < moments.get(p).size(); c++) {
                    int t = trainVectors(moments.get(p), split(moments.get(p), c, strict), det.get(m)).size();
                    lo = Math.min(lo, t);
                    hi = Math.max(hi, t);
                }
                line.append(' ').append(lo == hi ? String.valueOf(lo) : lo + "–" + hi);
            }
            text.append(line).append('\n');
        }
    }

    /** Итоги одного сочетания (способ, k, вариант разбиения). */
    static final class Stats {
        int argminCorrect;
        int argminAttempts;
        int ownAttempts;
        int ownRejects;
        int ownDetectorRejects;
        int impAttempts;
        int impAccepted;
        boolean[] impPersonAccepted;
        List<Double> margins = new ArrayList<>();
        int kMin = Integer.MAX_VALUE;
        int kMax = 0;
        int missingModels;
        int[] argminAttemptsG = new int[3];
        int[] argminCorrectG = new int[3];
        int[] ownAttemptsG = new int[3];
        int[] ownRejectsG = new int[3];
    }

    static Stats evaluate(List<List<OwnDataset.Moment>> moments, Map<Path, Det> det, Map<Path, Integer> yaw, SubspaceTrainer trainer,
                          boolean sface, KVariant kv, boolean strict) {
        int n = moments.size();
        Stats s = new Stats();
        s.impPersonAccepted = new boolean[n];
        Map<String, Model> cache = new HashMap<>();
        for (int r = 0; r < ROTATIONS; r++) {
            Model[] gallery = new Model[n];
            Split[] splits = new Split[n];
            for (int p = 0; p < n; p++) {
                int c = r % moments.get(p).size();
                splits[p] = split(moments.get(p), c, strict);
                String key = p + "/" + c;
                if (!cache.containsKey(key)) {
                    Model m = train(trainer, trainVectors(moments.get(p), splits[p], det), sface, kv);
                    cache.put(key, m);
                    if (m == null) s.missingModels++;
                    else if (!sface) {
                        s.kMin = Math.min(s.kMin, m.k());
                        s.kMax = Math.max(s.kMax, m.k());
                    }
                }
                gallery[p] = cache.get(key);
            }
            // Закрытая галерея из 6: точность argmin на контроле.
            for (int p = 0; p < n; p++) {
                if (r >= moments.get(p).size()) continue;
                s.argminAttempts++;
                int g = yaw.getOrDefault(moments.get(p).get(r).representative().file(), UNKNOWN);
                s.argminAttemptsG[g]++;
                double[] x = vec(det, moments.get(p).get(r));
                if (x != null && (int) score(gallery, x, sface, -1)[1] == p) {
                    s.argminCorrect++;
                    s.argminCorrectG[g]++;
                }
            }
            // Исключение одного человека q: галерея из 5, порог α = 0 по валидации чужого в этой ротации.
            for (int q = 0; q < n; q++) {
                List<OwnDataset.Moment> qm = moments.get(q);
                boolean qTested = r < qm.size();
                double minImp = Double.POSITIVE_INFINITY;
                for (int j = 0; j < qm.size(); j++) {
                    if (qTested && j == r) continue;
                    double[] x = vec(det, qm.get(j));
                    if (x != null) minImp = Math.min(minImp, score(gallery, x, sface, q)[0]);
                }
                double theta = Math.nextDown(minImp);
                double maxOwn = Double.NEGATIVE_INFINITY;
                for (int p = 0; p < n; p++) {
                    if (p == q) continue;
                    for (int j : gallery[p] == null ? List.<Integer>of() : splits[p].validation()) {
                        double[] x = vec(det, moments.get(p).get(j));
                        if (x != null) maxOwn = Math.max(maxOwn, score(gallery, x, sface, q)[0]);
                    }
                    if (r >= moments.get(p).size()) continue;
                    s.ownAttempts++;
                    int g = yaw.getOrDefault(moments.get(p).get(r).representative().file(), UNKNOWN);
                    s.ownAttemptsG[g]++;
                    double[] x = vec(det, moments.get(p).get(r));
                    if (x == null) {
                        s.ownDetectorRejects++;
                        s.ownRejects++;
                        s.ownRejectsG[g]++;
                    } else if (gallery[p] == null || score(gallery, x, sface, q)[0] > theta) {
                        s.ownRejects++;
                        s.ownRejectsG[g]++;
                    }
                }
                if (maxOwn > Double.NEGATIVE_INFINITY && minImp < Double.POSITIVE_INFINITY) s.margins.add(minImp - maxOwn);
                if (qTested) {
                    s.impAttempts++;
                    double[] x = vec(det, qm.get(r));
                    if (x != null && score(gallery, x, sface, q)[0] <= theta) {
                        s.impAccepted++;
                        s.impPersonAccepted[q] = true;
                    }
                }
            }
        }
        return s;
    }

    private static double[] vec(Map<Path, Det> det, OwnDataset.Moment m) {
        Det d = det.get(m.representative().file());
        return d == null ? null : d.vector();
    }

    private static void recognitionTable(StringBuilder text, List<List<OwnDataset.Moment>> moments,
                                         Map<Method, Map<Path, Det>> det, Map<Path, Integer> yaw, SubspaceTrainer trainer) {
        text.append("\n=== Распознавание: основной вариант | строгий (±1 с) ===\n");
        text.append("argmin — закрытая галерея из 6, контроль (один представитель на момент); FRR — исключение одного человека,\n"
                + "α = 0, отказ детектора на пригодном кадре — отказ (рядом — FRR без отказов детектора); FAR — контрольные\n"
                + "попытки чужого (по попыткам и по людям, ↑95 — верхняя 95 % граница Клоппера – Пирсона); запас — наименьшая\n"
                + "оценка чужого минус наибольшая оценка своих на валидации, по парам (чужой, ротация): минимум / медиана,\n"
                + "доля пар с зазором. Скоринг: SVD — ε₁/ε₂, SFace — 1 − cos₁.\n");
        for (Method m : Method.values()) {
            boolean sface = m == Method.SFACE;
            for (KVariant kv : sface ? new KVariant[] {KVariant.K4} : KVariant.values()) {
                StringBuilder line = new StringBuilder();
                line.append(m.label).append(sface ? "" : ", " + kv.label).append('\n');
                for (boolean strict : new boolean[] {false, true}) {
                    Stats s = evaluate(moments, det.get(m), yaw, trainer, sface, kv, strict);
                    int impPersons = 0;
                    for (boolean b : s.impPersonAccepted) if (b) impPersons++;
                    double[] mg = s.margins.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                    long positive = s.margins.stream().filter(v -> v > 0).count();
                    int detected = s.ownAttempts - s.ownDetectorRejects;
                    line.append(String.format(Locale.ROOT,
                            "  %-9s argmin %d/%d (%s %%); FRR %d/%d = %s %% (без отказов детектора %d/%d = %s %%); "
                                    + "FAR %d/%d, люди %d/%d (↑95 %s %%); запас %s / %s, с зазором %d/%d%s%s%n",
                            strict ? "строгий:" : "основной:", s.argminCorrect, s.argminAttempts,
                            pct(s.argminCorrect / (double) s.argminAttempts), s.ownRejects, s.ownAttempts,
                            pct(s.ownRejects / (double) s.ownAttempts), s.ownRejects - s.ownDetectorRejects, detected,
                            pct((s.ownRejects - s.ownDetectorRejects) / (double) detected), s.impAccepted, s.impAttempts,
                            impPersons, moments.size(),
                            pct(FaceEvaluation.binomialUpperBound(impPersons, moments.size(), FaceEvaluation.CONFIDENCE)),
                            mg.length == 0 ? "—" : String.format(Locale.ROOT, "%+.4f", mg[0]),
                            mg.length == 0 ? "—" : String.format(Locale.ROOT, "%+.4f", mg[mg.length / 2]), positive, mg.length,
                            sface || kv != KVariant.ETA ? "" : "; k " + s.kMin + "–" + s.kMax,
                            s.missingModels == 0 ? "" : "; без модели (меньше 2 обучающих кадров; попытка своего — отказ) " + s.missingModels));
                    line.append("             по повороту:");
                    for (int g = 0; g < 3; g++) {
                        if (g == UNKNOWN && s.argminAttemptsG[g] == 0 && s.ownAttemptsG[g] == 0) continue;
                        line.append(String.format(Locale.ROOT, " %s argmin %d/%d, FRR %d/%d%s", GROUP_NAMES[g],
                                s.argminCorrectG[g], s.argminAttemptsG[g], s.ownRejectsG[g], s.ownAttemptsG[g], g == FRONTAL ? ";" : ""));
                    }
                    line.append('\n');
                }
                text.append(line);
            }
        }
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.ROOT, "%.1f", 100 * v);
    }
}
