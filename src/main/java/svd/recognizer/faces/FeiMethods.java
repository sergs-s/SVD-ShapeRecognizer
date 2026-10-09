package svd.recognizer.faces;

import java.io.IOException;
import java.io.UncheckedIOException;
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
import java.util.stream.IntStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Impostors;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * Этап 5, протокол «только FEI» (решение Хозяина 30.09.2026; одна команда: {@code mvn compile exec:exec@faces-fei},
 * отдельная JVM с -Xmx6g). Данные — экспорт SVD-faces-data (faces.export.dir): кадры а 92×112 (a/fei) и
 * fei_selection.tsv; сырые снимки не нужны. Разбиение людей — {@link FeiProtocol}: 5 разбиений × 8 конфигураций, свои
 * 100 (8 снимков №4–7, 11–14; на контроле один, на обучении 7), чужие 100: когорта 20, пороговые 40, контроль 40.
 *
 * Порог — Нейман – Пирсон по доле людей пороговых чужих в каждой конфигурации (α = 0,05 и 0). Ошибка своих — FRR +
 * принят под чужим именем. Контрольные 40 чужих — только FAR; в обучении, пороге, когорте и выборе метода не участвуют.
 * Методы и их параметры — как в этапе 4 (FarMethods), заново не подбираются; новый — 4-wpca ({@link WpcaScorer}).
 *
 * Отчёты (вне git): reports/faces/fei/fei_methods.txt, fei_time.txt (время; не входит в сравнение); оценки каждой
 * попытки — fei_scores.tsv.gz ({@link FeiScores}; этап 5б, разбор — faces-fei-scores).
 *
 * @author ssv
 */
public final class FeiMethods {

    static final double[] ALPHAS = {0.05, 0.0};
    static final int A05 = 0;
    static final int A0 = 1;
    static final int CONFIGS = FeiProtocol.CONFIGS;
    /** Значения c для δ = c·λ_k (4-wpca). */
    static final double[] WPCA_C = {0, 0.1, 1};

    /** Базовый метод: оценщик обучается на своих (и, если cohortInTraining, на когорте). */
    record Base(String id, String about, IlluminationNorm norm, GalleryScorer scorer, boolean cohortInTraining, Mirror mirror, Feature feature) {}

    /** Вход метода: пиксели после нормализации освещения norm; гистограммы LBP (LbpFeature); их квадратный корень. */
    enum Feature { PIXELS, LBP, LBP_SQRT, LBP_SQRT_IN, LBP_SQRT_OUT }

    /** Строки LBP (п. 4.2, ключ faces.fei.lbp): L0 — χ², L1 — корень гистограмм + MLDA. */
    static final String LBP_CHI = "5-lbp-chi";
    static final String LBP_MLDA = "5-lbp-mlda";
    /** L1 по внутренним блокам (LbpFeature.Mask.INNER) и по краевому кольцу (RING); faces.fei.lbp = mask. */
    static final String LBP_MLDA_IN = "5-lbp-mlda-in";
    static final String LBP_MLDA_OUT = "5-lbp-mlda-out";

    /**
     * Режимы ключа faces.fei.lbp: строки LBP (true), маски внутренние / краевые (mask), только внутренние, без краевого
     * кольца (in), только строки LBP-MLDA (only).
     */
    enum Lbp { ON, MASK, ONLY, IN }

    /**
     * Режимы из значения ключа faces.fei.lbp: список через запятую из true, mask, in, only, без учёта регистра; null, пусто
     * или false — нет режимов; прочее — ошибка. mask, in и only включают строки LBP и без true.
     */
    static Set<Lbp> lbpModes(String raw) {
        Set<Lbp> m = java.util.EnumSet.noneOf(Lbp.class);
        if (raw == null || raw.isBlank()) return m;
        for (String v : raw.split(",")) {
            switch (v.trim().toLowerCase(Locale.ROOT)) {
                case "true" -> m.add(Lbp.ON);
                case "mask" -> m.add(Lbp.MASK);
                case "only" -> m.add(Lbp.ONLY);
                case "in" -> m.add(Lbp.IN);
                case "false" -> { }
                default -> throw new IllegalStateException("faces.fei.lbp: неизвестное значение «" + v.trim() + "» (true, mask, in, only)");
            }
        }
        return m;
    }

    /** Зеркальные копии в обучении: нет; все обучающие снимки (+mirror); только с |r| > r0 (+mirror-turn). */
    enum Mirror { NONE, ALL, TURN }

    /** Суффикс строк с зеркальными копиями в обучении (этап 5б, ключ faces.fei.mirror). */
    static final String MIRROR = "+mirror";
    /** Суффикс строк с отражением только повёрнутых обучающих снимков (|r| > r0; faces.fei.mirror = turn). */
    static final String MIRROR_TURN = "+mirror-turn";

    /**
     * Режимы из значения ключа faces.fei.mirror: список через запятую из true (ALL) и turn (TURN), без учёта регистра;
     * null, пусто или false — нет режимов; прочее — ошибка.
     */
    static Set<Mirror> mirrorModes(String raw) {
        Set<Mirror> m = java.util.EnumSet.noneOf(Mirror.class);
        if (raw == null || raw.isBlank()) return m;
        for (String v : raw.split(",")) {
            switch (v.trim().toLowerCase(Locale.ROOT)) {
                case "true" -> m.add(Mirror.ALL);
                case "turn" -> m.add(Mirror.TURN);
                case "false" -> { }
                default -> throw new IllegalStateException("faces.fei.mirror: неизвестное значение «" + v.trim() + "» (true, turn, true,turn)");
            }
        }
        return m;
    }

    /**
     * Ключи обучающих снимков человека, к которым добавляются отражения: ALL — все, TURN — с |r| > r0 (граница — без
     * отражения); порядок — порядок numbers.
     */
    static List<String> mirrorKeys(String person, Map<Integer, String> byNumber, Iterable<Integer> numbers, Mirror mode, FeiPose pose,
                                   double r0) {
        List<String> out = new ArrayList<>();
        for (int n : numbers) {
            if (mode == Mirror.ALL || mode == Mirror.TURN && pose.turned(person, n, r0)) out.add(byNumber.get(n));
            else if (mode != Mirror.TURN) throw new IllegalArgumentException("Режим без отражений: " + mode);
        }
        return out;
    }

    /** Производный метод: оценки из оценок других методов той же конфигурации. */
    record Derived(String id, String about, List<String> parts, Kind kind) {}

    enum Kind { RATIO, ZNORM, SUM, AGREE }

    // ---------------------------------------------------------------- данные

    final SettingsStore settings;
    final long seed;
    /** Человек → номер снимка → ключ снимка (только прошедшие отбор). */
    final Map<String, Map<Integer, String>> keys;
    final Map<String, Set<Integer>> included;
    final List<FeiProtocol.Split> splits;
    final FeiDataset.Info info;
    final IlluminationNorm.NormParams params;
    final Map<IlluminationNorm, Map<String, double[]>> vectors = new HashMap<>();
    /** Режим +mirror (faces.fei.mirror = true): к обучению добавляются отражённые кадры (FaceMirror, способ а). */
    final boolean mirror;
    /** Режим +mirror-turn (faces.fei.mirror = turn): отражаются только обучающие снимки с |r| > r0. */
    final boolean mirrorTurn;
    /** Поза отобранных снимков (fei_selection.tsv) и порог r0 — только при mirrorTurn. */
    final FeiPose pose;
    final double r0;
    /** Векторы CLAHE отражённых кадров (только при mirror или mirrorTurn). */
    final Map<String, double[]> mirrorVectors = new HashMap<>();
    /** Строки LBP (faces.fei.lbp) и векторы LBP: гистограммы и их корень, исходных и отражённых кадров. */
    final boolean lbp;
    final Map<String, double[]> lbpVectors = new HashMap<>();
    final Map<String, double[]> lbpSqrtVectors = new HashMap<>();
    final Map<String, double[]> lbpMirror = new HashMap<>();
    final Map<String, double[]> lbpSqrtMirror = new HashMap<>();
    /** faces.fei.lbp = mask или in: корень гистограмм по маскам INNER / RING, исходных и отражённых кадров (куски векторов L1). */
    final boolean lbpMask;
    /** faces.fei.lbp = mask: строки краевого кольца (5-lbp-mlda-out); при одном in — нет. */
    final boolean lbpRing;
    /** Нормировка освещения кадра перед LBP (faces.lbp.norm; п. 4.2, «Нормировка освещения перед LBP»), по умолчанию NONE. */
    final IlluminationNorm lbpNorm;
    /** faces.fei.lbp = only: считаются только строки LBP-MLDA (прочие методы и производные строки пропускаются). */
    final boolean lbpOnly;
    final Map<LbpFeature.Mask, Map<String, double[]>> lbpMasked = new java.util.EnumMap<>(LbpFeature.Mask.class);
    final Map<LbpFeature.Mask, Map<String, double[]>> lbpMaskedMirror = new java.util.EnumMap<>(LbpFeature.Mask.class);

    final List<Base> bases = new ArrayList<>();
    final List<Derived> derived = new ArrayList<>();
    /** Все строки в порядке отчёта; подпись строки. */
    final Map<String, String> about = new LinkedHashMap<>();
    final Map<String, Result> results = new LinkedHashMap<>();
    final StringBuilder timeText = new StringBuilder();
    /** Контрольная точка (оценки конфигураций и состояние после каждой конфигурации); null — не пишется. */
    FeiCheckpoint checkpoint;
    FeiCheckpoint.State state;

    FeiMethods(SettingsStore settings, Map<String, Map<Integer, String>> keys, FeiPose pose, Map<String, Mat> frames, FeiDataset.Info info) {
        this.settings = settings;
        this.seed = settings.loadFacesSeed();
        this.keys = keys;
        this.info = info;
        this.params = IlluminationNorm.NormParams.of(settings);
        included = new TreeMap<>();
        keys.forEach((p, m) -> included.put(p, new HashSet<>(m.keySet())));
        splits = FeiProtocol.splits(included, seed);
        for (IlluminationNorm n : new IlluminationNorm[] {IlluminationNorm.CLAHE, IlluminationNorm.TAN_TRIGGS}) {
            List<String> k = new ArrayList<>(frames.keySet());
            double[][] out = new double[k.size()][];
            IntStream.range(0, k.size()).parallel().forEach(i -> out[i] = n.vector(frames.get(k.get(i)), params));
            Map<String, double[]> map = new HashMap<>();
            for (int i = 0; i < k.size(); i++) map.put(k.get(i), out[i]);
            vectors.put(n, map);
        }
        Set<Mirror> modes = mirrorModes(settings.loadFacesFeiMirror());
        mirror = modes.contains(Mirror.ALL);
        mirrorTurn = modes.contains(Mirror.TURN);
        this.pose = mirrorTurn ? pose : null;
        r0 = mirrorTurn ? FeiPose.r0(settings.loadFacesFeiPoseR0(), "faces.fei.mirror = turn") : Double.NaN;
        if (mirror || mirrorTurn) {
            List<String> k = new ArrayList<>(frames.keySet());
            double[][] out = new double[k.size()][];
            IntStream.range(0, k.size()).parallel().forEach(i -> {
                Mat m = FaceMirror.mirror(frames.get(k.get(i)));
                out[i] = IlluminationNorm.CLAHE.vector(m, params);
                m.release();
            });
            for (int i = 0; i < k.size(); i++) mirrorVectors.put(k.get(i), out[i]);
        }
        Set<Lbp> lbpModes = lbpModes(settings.loadFacesFeiLbp());
        lbp = !lbpModes.isEmpty();
        lbpMask = lbpModes.contains(Lbp.MASK) || lbpModes.contains(Lbp.IN);
        lbpRing = lbpModes.contains(Lbp.MASK);
        lbpOnly = lbpModes.contains(Lbp.ONLY);
        lbpNorm = IlluminationNorm.lbpNorm(settings.loadFacesLbpNorm());
        if (lbpNorm != IlluminationNorm.NONE && !lbp) throw new IllegalStateException("faces.lbp.norm = " + lbpNorm.id + " без строк LBP (faces.fei.lbp)");
        if (lbp) {
            // Гистограммы LBP кадра «а» после нормировки lbpNorm (NONE — без нормализации освещения); отражённые — LBP
            // отражённого и заново выровненного кадра после той же нормировки.
            List<String> k = new ArrayList<>(frames.keySet());
            boolean withMirror = mirror || mirrorTurn;
            double[][] out = new double[k.size()][];
            double[][] mout = new double[k.size()][];
            IntStream.range(0, k.size()).parallel().forEach(i -> {
                out[i] = LbpFeature.histogram(frames.get(k.get(i)), lbpNorm, params);
                if (withMirror) {
                    Mat m = FaceMirror.mirror(frames.get(k.get(i)));
                    mout[i] = LbpFeature.histogram(m, lbpNorm, params);
                    m.release();
                }
            });
            for (int i = 0; i < k.size(); i++) {
                lbpVectors.put(k.get(i), out[i]);
                lbpSqrtVectors.put(k.get(i), LbpFeature.sqrt(out[i]));
                if (withMirror) {
                    lbpMirror.put(k.get(i), mout[i]);
                    lbpSqrtMirror.put(k.get(i), LbpFeature.sqrt(mout[i]));
                }
            }
            if (lbpMask) {
                for (LbpFeature.Mask m : LbpFeature.Mask.values()) {
                    Map<String, double[]> v = new HashMap<>();
                    lbpSqrtVectors.forEach((key, x) -> v.put(key, LbpFeature.select(x, m)));
                    lbpMasked.put(m, v);
                    Map<String, double[]> mv = new HashMap<>();
                    lbpSqrtMirror.forEach((key, x) -> mv.put(key, LbpFeature.select(x, m)));
                    lbpMaskedMirror.put(m, mv);
                }
            }
        }
        register();
        vectorIds = vectorIds(settings.loadFacesFeiVectors());
        String cvRaw = settings.loadFacesFeiCv();
        cvOnly = cvRaw != null && Arrays.stream(cvRaw.split(",")).anyMatch(v -> v.trim().equalsIgnoreCase(CV_ONLY));
        cvIds = cvIds(cvRaw);
        cvThreads = settings.loadFacesFeiCvThreads();
        if (cvOnly && cvIds.isEmpty()) throw new IllegalStateException("faces.fei.cv: only без методов");
        if (cvOnly && !vectorIds.isEmpty()) throw new IllegalStateException("faces.fei.cv = only: полные векторы (faces.fei.vectors) не пишутся");
    }

    // ---------------------------------------------------------------- перекрёстная оценка по галерее (П2)

    /** Частей галереи (п. 4.2, «Калибровка порога», П2; фиксировано заранее). */
    static final int CV_PARTS = 5;
    /** Seed разбиения галереи разбиения s на части: faces.seed + CV_SEED_OFFSET + s. */
    static final long CV_SEED_OFFSET = 8000;
    static final String CV_FILE = "fei_cv_scores.tsv.gz";
    /** Роль строки файла перекрёстной оценки: cv0…cv4 — номер части. */
    static final String CV_ROLE = "cv";
    static final String CV_ONLY = "only";

    /** Методы перекрёстной оценки (faces.fei.cv): базовые и их +mirror / +mirror-turn; пусто — выключено. */
    final List<String> cvIds;
    /** faces.fei.cv содержит only: основные строки не считаются (только перекрёстная оценка). */
    final boolean cvOnly;
    final int cvThreads;

    /**
     * Методы перекрёстной оценки: перечисленные базовые методы (оценщик — MLDA: вызовы OpenCV в потоках исключены) и их
     * +mirror / +mirror-turn, если они есть; only пропускается; прочее — ошибка.
     */
    List<String> cvIds(String raw) {
        List<String> out = new ArrayList<>();
        if (raw == null || raw.isBlank()) return out;
        Map<String, Base> byId = new HashMap<>();
        for (Base b : bases) byId.put(b.id(), b);
        for (String m : raw.split(",")) {
            String id = m.trim();
            if (id.equalsIgnoreCase(CV_ONLY)) continue;
            Base b = byId.get(id);
            if (b == null || isMirror(id)) throw new IllegalStateException("faces.fei.cv: " + id + " — не базовый метод faces-fei");
            if (!(b.scorer() instanceof MldaScorer)) throw new IllegalStateException("faces.fei.cv: " + id + " — не MLDA");
            out.add(id);
            for (String suffix : new String[] {MIRROR, MIRROR_TURN}) if (byId.containsKey(id + suffix)) out.add(id + suffix);
        }
        return out;
    }

    /**
     * Части галереи разбиения s: перестановка своих (java.util.Random, seed + CV_SEED_OFFSET + s), подряд по
     * gallery.size() / CV_PARTS человек.
     */
    static List<List<String>> cvParts(List<String> gallery, long seed, int s) {
        if (gallery.size() % CV_PARTS != 0) throw new IllegalArgumentException("Своих " + gallery.size() + " не делится на " + CV_PARTS);
        List<String> perm = new ArrayList<>(gallery);
        java.util.Collections.shuffle(perm, new java.util.Random(seed + CV_SEED_OFFSET + s));
        int size = gallery.size() / CV_PARTS;
        List<List<String>> out = new ArrayList<>();
        for (int q = 0; q < CV_PARTS; q++) out.add(List.copyOf(perm.subList(q * size, (q + 1) * size)));
        return out;
    }

    /**
     * Обучающие классы метода b по галерее gallery (подмножество своих разбиения) и когорте cohort (пусто — без неё) —
     * как classes / mirrorClasses основной строки.
     */
    List<List<double[]>> cvClasses(List<String> gallery, List<String> cohort, int j, Base b) {
        boolean mirrored = b.mirror() != Mirror.NONE;
        Map<String, double[]> vec = mirrored ? input(b.feature(), IlluminationNorm.CLAHE) : input(b.feature(), b.norm());
        Map<String, double[]> mir = !mirrored ? null : switch (b.feature()) {
            case PIXELS -> mirrorVectors;
            case LBP -> lbpMirror;
            case LBP_SQRT -> lbpSqrtMirror;
            case LBP_SQRT_IN -> lbpMaskedMirror.get(LbpFeature.Mask.INNER);
            case LBP_SQRT_OUT -> lbpMaskedMirror.get(LbpFeature.Mask.RING);
        };
        List<List<double[]>> out = new ArrayList<>();
        for (String p : gallery) {
            List<double[]> list = new ArrayList<>();
            for (int n : FeiProtocol.trainNumbers(j)) list.add(vec.get(key(p, n)));
            if (mirrored) for (String k : mirrorKeys(p, keys.get(p), FeiProtocol.trainNumbers(j), b.mirror(), pose, r0)) list.add(mir.get(k));
            out.add(list);
        }
        for (String p : cohort) {
            List<double[]> list = new ArrayList<>(keysOf(List.of(p)).stream().map(vec::get).toList());
            if (mirrored) for (String k : mirrorKeys(p, keys.get(p), keys.get(p).keySet(), b.mirror(), pose, r0)) list.add(mir.get(k));
            out.add(list);
        }
        return out;
    }

    /** Контроль: классы cvClasses по всей галерее — те же векторы (те же объекты) и порядок, что у основной строки; иначе ошибка. */
    void checkCvClasses(FeiProtocol.Split sp, int j, Base b) {
        List<List<double[]>> main = b.mirror() != Mirror.NONE ? mirrorClasses(sp, j, b.cohortInTraining(), b.mirror(), b.feature())
                : classes(sp, j, input(b.feature(), b.norm()), b.cohortInTraining());
        List<List<double[]>> cv = cvClasses(sp.gallery(), b.cohortInTraining() ? sp.cohort() : List.of(), j, b);
        boolean same = main.size() == cv.size();
        for (int c = 0; same && c < main.size(); c++) {
            same = main.get(c).size() == cv.get(c).size();
            for (int i = 0; same && i < main.get(c).size(); i++) same = main.get(c).get(i) == cv.get(c).get(i);
        }
        if (!same) throw new IllegalStateException("Перекрёстная оценка: классы без части ≠ классам основной строки " + b.id());
    }

    /**
     * Часть q: модель метода b обучена без людей части (свои — остальные 80, когорта — как у основной строки); «чужие» — все 8
     * снимков (NUMBERS) людей части, оценки по 80 своим; строки файла оценок с ролью cv&lt;q&gt;.
     */
    String cvPart(FeiProtocol.Split sp, int j, Base b, List<String> part, int q) {
        Set<String> out = new HashSet<>(part);
        List<String> gallery = sp.gallery().stream().filter(p -> !out.contains(p)).toList();
        GalleryScorer.ScoreModel m = b.scorer().fit(cvClasses(gallery, b.cohortInTraining() ? sp.cohort() : List.of(), j, b), gallery.size());
        Map<String, double[]> vec = input(b.feature(), b.norm());
        StringBuilder t = new StringBuilder();
        for (String p : sp.gallery()) {
            if (!out.contains(p)) continue;
            for (int n : FeiProtocol.NUMBERS) {
                t.append(FeiScores.line(b.id(), sp.index(), j, CV_ROLE + q, p, n, -1, gallery, m.scores(vec.get(key(p, n))), false));
            }
        }
        return t.toString();
    }

    /** Перекрёстная оценка конфигурации: методы cvIds × части 0…4 (части — в cvThreads потоках), порядок фиксирован. */
    String cvText(FeiProtocol.Split sp, int j) {
        List<List<String>> parts = cvParts(sp.gallery(), seed, sp.index());
        List<Base> bs = new ArrayList<>();
        for (String id : cvIds) for (Base b : bases) if (b.id().equals(id)) bs.add(b);
        for (Base b : bs) checkCvClasses(sp, j, b);
        List<java.util.concurrent.Callable<String>> tasks = new ArrayList<>();
        for (Base b : bs) for (int q = 0; q < CV_PARTS; q++) {
            int qq = q;
            tasks.add(() -> cvPart(sp, j, b, parts.get(qq), qq));
        }
        StringBuilder t = new StringBuilder();
        if (cvThreads == 1) {
            for (java.util.concurrent.Callable<String> c : tasks) {
                try {
                    t.append(c.call());
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            }
            return t.toString();
        }
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(cvThreads);
        try {
            for (java.util.concurrent.Future<String> f : ex.invokeAll(tasks)) t.append(f.get());
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("Перекрёстная оценка: " + e, e);
        } finally {
            ex.shutdownNow();
        }
        return t.toString();
    }

    /** Шапка файла перекрёстной оценки: части галереи по разбиениям. */
    String cvHeader(String code, String data, int configs) {
        StringBuilder t = new StringBuilder();
        t.append("# Перекрёстная оценка по галерее faces-fei (FeiMethods, faces.fei.cv; PLAN.md, п. 4.2, «Калибровка порога», П2)\n");
        t.append("# Код: коммит ").append(code).append('\n');
        t.append("# Данные: ").append(data).append('\n');
        t.append(String.format(Locale.ROOT, "# Конфигураций: %d из %d; части: %d, seed %d + %d + s (java.util.Random, перестановка своих разбиения)%n",
                configs, FeiProtocol.CONFIGS, CV_PARTS, seed, CV_SEED_OFFSET));
        t.append("# Роль cv<q>: снимок человека части q (все 8 снимков №4–7, 11–14) против модели, обученной без части q (свои — остальные, "
                + "когорта — как у основной строки); argmin по оставшимся своим; меньше — ближе\n");
        t.append("# Методы (").append(cvIds.size()).append("): ").append(String.join(", ", cvIds)).append('\n');
        if (lbpNorm != IlluminationNorm.NONE) t.append("# ").append(lbpNormText()).append('\n');
        for (FeiProtocol.Split sp : splits) {
            List<List<String>> parts = cvParts(sp.gallery(), seed, sp.index());
            for (int q = 0; q < CV_PARTS; q++) t.append("# Часть s=").append(sp.index()).append(" q=").append(q).append(": ")
                    .append(String.join(",", parts.get(q))).append('\n');
        }
        t.append(FeiScores.COLUMNS).append('\n');
        return t.toString();
    }

    /** Строки, для которых пишутся полные векторы расстояний (faces.fei.vectors); пусто — не пишутся. */
    final List<String> vectorIds;

    /**
     * Методы полных векторов: перечисленные базовые методы и их строки +mirror / +mirror-turn, если эти режимы включены;
     * не базовый метод или нет такого — ошибка.
     */
    List<String> vectorIds(String raw) {
        List<String> out = new ArrayList<>();
        if (raw == null || raw.isBlank()) return out;
        Set<String> baseIds = new HashSet<>();
        for (Base b : bases) baseIds.add(b.id());
        for (String m : raw.split(",")) {
            String id = m.trim();
            if (!baseIds.contains(id)) throw new IllegalStateException("faces.fei.vectors: " + id + " — не базовый метод faces-fei");
            out.add(id);
            for (String suffix : new String[] {MIRROR, MIRROR_TURN}) if (baseIds.contains(id + suffix)) out.add(id + suffix);
        }
        return out;
    }

    /** Полные векторы конфигурации (строки FeiVectors) по методам vectorIds — те же попытки и порядок, что в scoresText. */
    String vectorsText(FeiProtocol.Split sp, int j, Map<String, Map<String, double[]>> sc) {
        List<String> gallery = sp.gallery();
        StringBuilder b = new StringBuilder();
        for (String id : vectorIds) {
            Map<String, double[]> s = sc.get(id);
            int n = FeiProtocol.NUMBERS[j];
            for (String person : gallery) b.append(FeiVectors.line(id, sp.index(), j, FeiScores.OWN, person, n, person, s.get(key(person, n))));
            for (String role : new String[] {FeiScores.THR, FeiScores.CTRL}) {
                for (String person : role.equals(FeiScores.THR) ? sp.threshold() : sp.control()) {
                    for (Map.Entry<Integer, String> e : keys.get(person).entrySet()) {
                        b.append(FeiVectors.line(id, sp.index(), j, role, person, e.getKey(), FeiScores.NONE, s.get(e.getValue())));
                    }
                }
            }
        }
        return b.toString();
    }

    /** Методы этапа 4 (параметры те же) и 4-wpca; порядок — порядок таблицы. */
    private void register() {
        if (lbpOnly) {
            // faces.fei.lbp = only: только строки LBP-MLDA (L1 и, при mask, по маскам) и их +mirror / +mirror-turn.
            lbpRows("", Mirror.NONE, "");
            if (mirror) lbpRows(MIRROR, Mirror.ALL, "; обучение + отражённые кадры (LBP отражённого кадра)");
            if (mirrorTurn) lbpRows(MIRROR_TURN, Mirror.TURN, "; обучение + отражённые кадры с |r| > r0 = " + r0);
            return;
        }
        IlluminationNorm cl = IlluminationNorm.CLAHE;
        base("1-clahe-eps", "CLAHE + подпространство на человека (k = 4), ε", cl, new SubspaceScorer(false), false);
        derived("1-clahe-ratio", "CLAHE + подпространство, ε₁/ε₂", Kind.RATIO, "1-clahe-eps");
        base("1-tt-eps", "Tan – Triggs + подпространство, ε", IlluminationNorm.TAN_TRIGGS, new SubspaceScorer(false), false);
        base("2a95-fisher", "CLAHE + Fisherfaces (PCA по энергии 95 % + LDA)", cl, FisherScorer.byEnergy(FarMethods.PCA_ENERGY), false);
        base("2b-fisher-bg", "то же, обучение — свои + когорта 20", cl, FisherScorer.byEnergy(FarMethods.PCA_ENERGY), true);
        base("2c-mlda", "MLDA (Thomaz, Kitani, Gillies, 2006), обучение — свои + когорта 20", cl, new MldaScorer(), true);
        derived("3-znorm-svd", "Z-norm (когорта 20) над 1-clahe-eps", Kind.ZNORM, "1-clahe-eps");
        derived("3-znorm-lda", "Z-norm (когорта 20) над 2a95-fisher", Kind.ZNORM, "2a95-fisher");
        for (RegionScorer.Region g : FarMethods.REGIONS) {
            base("4-" + g.id() + "-fisher", "область " + g.name() + ", Fisherfaces как 2a95", cl,
                    new RegionScorer(FisherScorer.byEnergy(FarMethods.PCA_ENERGY), g), false);
            base("4-" + g.id() + "-eps", "область " + g.name() + ", подпространство (k = 4), ε", cl, new RegionScorer(new SubspaceScorer(false), g),
                    false);
        }
        derived("4-blk-fisher", "блочный: Fisherfaces глаз, носа, рта и всего лица, сумма нормированных", Kind.SUM,
                "4-eyes-fisher", "4-nose-fisher", "4-mouth-fisher", "2a95-fisher");
        derived("4-blk-subspace", "блочный: подпространство (ε) на тех же областях", Kind.SUM,
                "4-eyes-eps", "4-nose-eps", "4-mouth-eps", "1-clahe-eps");
        derived("4-blk-both", "блочный: все 8 оценок", Kind.SUM, "4-eyes-fisher", "4-nose-fisher", "4-mouth-fisher", "2a95-fisher",
                "4-eyes-eps", "4-nose-eps", "4-mouth-eps", "1-clahe-eps");
        Map<String, String> participants = new LinkedHashMap<>();
        participants.put("ratio", "1-clahe-ratio");
        participants.put("eps", "1-clahe-eps");
        participants.put("fisher", "2a95-fisher");
        participants.put("zsvd", "3-znorm-svd");
        participants.put("zlda", "3-znorm-lda");
        for (Kind mode : new Kind[] {Kind.SUM, Kind.AGREE}) {
            for (String[] c : FarMethods.COMBINATIONS) {
                String[] ids = new String[c.length];
                for (int i = 0; i < c.length; i++) ids[i] = participants.get(c[i]);
                String id = "4-" + (mode == Kind.SUM ? "sum" : "agr") + "-" + (c.length == 5 ? "все пять" : String.join("+", c));
                derived(id, (mode == Kind.SUM ? "сумма нормированных: " : "согласие argmin, итог — сумма: ") + String.join(" + ", ids), mode, ids);
            }
        }
        for (WpcaScorer.Mode m : WpcaScorer.Mode.values()) {
            for (double c : WPCA_C) {
                String id = wpcaId(m == WpcaScorer.Mode.COS ? "cos" : "eps", c);
                base(id, "CLAHE + отбеливание PCA (свои + когорта 20, энергия 95 %), δ = " + WpcaScorer.fmt(c) + "·λ_k, "
                        + (m == WpcaScorer.Mode.COS ? "1 − cos до среднего человека" : "подпространство человека, ε"), cl, new WpcaScorer(m, c), true);
                if (m == WpcaScorer.Mode.COS) {
                    derived(wpcaId("cos-z", c), "то же, 1 − cos с Z-norm (когорта 20)", Kind.ZNORM, id);
                }
            }
        }
        // Условная строка (добавляется в таблицу, если лучшая строка wpca обходит 3-znorm-lda): считается для всех 9 wpca.
        for (String w : wpcaIds()) derived(blkWpcaId(w), "сумма нормированных: 4-blk-both + " + w, Kind.SUM, "4-blk-both", w);
        // +mirror (этап 5б): те же методы, к обучению добавлены отражённые кадры своих (и когорты, если она в обучении);
        // пробы и когорта в нормировке — без отражения. Строки — в конце таблицы, прежние не меняются.
        if (mirror) mirrorRows(MIRROR, Mirror.ALL, "; обучение + отражённые кадры");
        // +mirror-turn: отражаются только обучающие снимки с |r| > r0 (анфас — без отражения).
        if (mirrorTurn) mirrorRows(MIRROR_TURN, Mirror.TURN, "; обучение + отражённые кадры с |r| > r0 = " + r0);
        // LBP (п. 4.2, faces.fei.lbp): строки — после всех прежних, прежние не меняются.
        if (lbp) {
            lbpRows("", Mirror.NONE, "");
            if (mirror) lbpRows(MIRROR, Mirror.ALL, "; обучение + отражённые кадры (LBP отражённого кадра)");
            if (mirrorTurn) lbpRows(MIRROR_TURN, Mirror.TURN, "; обучение + отражённые кадры с |r| > r0 = " + r0);
        }
    }

    private void mirrorRows(String suffix, Mirror mode, String note) {
        IlluminationNorm cl = IlluminationNorm.CLAHE;
        base("1-clahe-eps" + suffix, about.get("1-clahe-eps") + note, cl, new SubspaceScorer(false), false, mode);
        derived("1-clahe-ratio" + suffix, about.get("1-clahe-ratio") + note, Kind.RATIO, "1-clahe-eps" + suffix);
        base("2a95-fisher" + suffix, about.get("2a95-fisher") + note, cl, FisherScorer.byEnergy(FarMethods.PCA_ENERGY), false, mode);
        base("2c-mlda" + suffix, about.get("2c-mlda") + note + " (и когорты)", cl, new MldaScorer(), true, mode);
        String w0 = wpcaId("cos", 0);
        base(w0 + suffix, about.get(w0) + note + " (и когорты; своя PCA)", cl, new WpcaScorer(WpcaScorer.Mode.COS, 0), true, mode);
    }

    /** Строка режима +mirror. */
    static boolean isMirror(String id) {
        return id.endsWith(MIRROR) || id.endsWith(MIRROR_TURN);
    }

    static String wpcaId(String kind, double c) {
        return "4-wpca-" + kind + "-" + WpcaScorer.fmt(c);
    }

    static String blkWpcaId(String wpca) {
        return "4-sum-blk-both+" + wpca.substring(2);
    }

    List<String> wpcaIds() {
        return about.keySet().stream().filter(id -> id.startsWith("4-wpca-") && !isMirror(id)).toList();
    }

    private void base(String id, String text, IlluminationNorm norm, GalleryScorer scorer, boolean cohort) {
        base(id, text, norm, scorer, cohort, Mirror.NONE);
    }

    /**
     * Строки LBP: L0 (χ², ближайший снимок, без когорты) и L1 (корень гистограмм + MLDA, свои + когорта 20); при mask —
     * L1 по внутренним блокам и по краевому кольцу; при only — без L0.
     */
    private void lbpRows(String suffix, Mirror mode, String note) {
        if (!lbpOnly) {
            base(LBP_CHI + suffix, "LBP u2 (8, 2), сетка 7×7, кадр а без нормализации освещения; χ², ближайший снимок человека" + note,
                    IlluminationNorm.NONE, new ChiSquareScorer(), false, mode, Feature.LBP);
        }
        base(LBP_MLDA + suffix, "корень гистограмм LBP (как " + LBP_CHI + ") + MLDA, обучение — свои + когорта 20" + note, IlluminationNorm.NONE,
                new MldaScorer(), true, mode, Feature.LBP_SQRT);
        if (lbpMask) {
            base(LBP_MLDA_IN + suffix, "как " + LBP_MLDA + ", только внутренние блоки (строки и столбцы 1–5 сетки 7×7, 25 блоков, вектор 1475)"
                    + note, IlluminationNorm.NONE, new MldaScorer(), true, mode, Feature.LBP_SQRT_IN);
        }
        if (lbpRing) {
            base(LBP_MLDA_OUT + suffix, "как " + LBP_MLDA + ", только краевое кольцо (24 блока, вектор 1416)" + note, IlluminationNorm.NONE,
                    new MldaScorer(), true, mode, Feature.LBP_SQRT_OUT);
        }
    }

    /** Строка о нормировке перед LBP для отчётов и шапок (lbpNorm ≠ NONE). */
    String lbpNormText() {
        return lbpNormText(lbpNorm, params);
    }

    static String lbpNormText(IlluminationNorm n, IlluminationNorm.NormParams p) {
        return switch (n) {
            case NONE -> "Нормировка освещения перед LBP (faces.lbp.norm): none — без нормировки";
            case CLAHE -> String.format(Locale.ROOT, "Нормировка освещения перед LBP (faces.lbp.norm): clahe — CLAHE clip %.2f, сетка %d×%d, LBP по "
                    + "8-битному результату", p.claheClip(), p.claheTile(), p.claheTile());
            case TAN_TRIGGS -> String.format(Locale.ROOT, "Нормировка освещения перед LBP (faces.lbp.norm): tt — Tan – Triggs γ %.2f, σ₀ %.2f, σ₁ %.2f, "
                    + "α %.2f, τ %.2f, без маски, LBP по вещественному результату без квантования", p.gamma(), p.sigma0(), p.sigma1(), p.alpha(), p.tau());
        };
    }

    private void base(String id, String text, IlluminationNorm norm, GalleryScorer scorer, boolean cohort, Mirror mirrored) {
        base(id, text, norm, scorer, cohort, mirrored, Feature.PIXELS);
    }

    private void base(String id, String text, IlluminationNorm norm, GalleryScorer scorer, boolean cohort, Mirror mirrored, Feature feature) {
        bases.add(new Base(id, text, norm, scorer, cohort, mirrored, feature));
        about.put(id, text);
    }

    private void derived(String id, String text, Kind kind, String... parts) {
        derived.add(new Derived(id, text, List.of(parts), kind));
        about.put(id, text);
    }

    // ---------------------------------------------------------------- запуск

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        long start = System.nanoTime();
        SettingsStore settings = new SettingsStore();
        String exportDir = settings.loadFacesExportDir();
        if (exportDir == null) throw new IllegalStateException("faces-fei: не задан faces.export.dir / FACES_EXPORT_DIR");
        Path dir = Paths.get(exportDir);
        FarExport.Loaded export = FarExport.read(dir);
        if (export.data().fei().isEmpty() || export.feiInfo() == null) throw new IllegalStateException("В экспорте нет FEI");
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> keys = selection(selLines, export.data().fei());
        Map<String, Mat> frames = new HashMap<>();
        for (List<Sample> list : export.data().fei().values()) {
            for (Sample s : list) {
                Mat f = export.frame(s, Variant.A);
                if (f == null) throw new IllegalStateException("Нет кадра отобранного снимка FEI: " + s.file());
                frames.put(s.file().toString(), f);
            }
        }
        int limit = args.length > 0 && !args[0].isBlank() ? Integer.parseInt(args[0].trim()) : CONFIGS;
        FeiMethods fm = new FeiMethods(settings, keys, FeiPose.parse(selLines), frames, export.feiInfo());
        Path outDir0 = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        if (fm.mirror || fm.mirrorTurn) {
            Files.createDirectories(outDir0);
            mirrorCheck(keys, frames, outDir0.resolve("fei_mirror_check.png"));
        }
        frames.values().forEach(Mat::release);
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(outDir);
        int n = Math.min(limit, CONFIGS);
        // Контрольная точка: отпечаток — код, данные, ключи, seed, число конфигураций.
        String keysText = String.format(Locale.ROOT, "ключи: faces.fei.mirror %s, faces.fei.pose.r0 %s, seed %d, конфигураций %d",
                settings.loadFacesFeiMirror(), fm.mirrorTurn ? Double.toString(fm.r0) : "-", fm.seed, n);
        // Без faces.fei.vectors отпечаток — прежний.
        if (!fm.vectorIds.isEmpty()) keysText += ", faces.fei.vectors " + String.join(",", fm.vectorIds);
        // При faces.fei.lbp = true — прежний текст; mask / only — значение ключа.
        if (fm.lbp) keysText += ", faces.fei.lbp " + (fm.lbpMask || fm.lbpOnly ? settings.loadFacesFeiLbp().trim().toLowerCase(Locale.ROOT) : "true");
        // Без faces.fei.cv отпечаток — прежний; число потоков на результат не влияет и в отпечаток не входит.
        if (!fm.cvIds.isEmpty()) keysText += ", faces.fei.cv " + String.join(",", fm.cvIds) + (fm.cvOnly ? ",only" : "");
        // При faces.lbp.norm = none отпечаток — прежний.
        if (fm.lbpNorm != IlluminationNorm.NONE) keysText += ", faces.lbp.norm " + fm.lbpNorm.id;
        String fingerprint = FeiCheckpoint.fingerprint(dir, keysText);
        fm.checkpoint = new FeiCheckpoint(outDir.resolve(FeiCheckpoint.DIR));
        fm.checkpoint.vectors = !fm.vectorIds.isEmpty();
        fm.checkpoint.cv = !fm.cvIds.isEmpty();
        FeiCheckpoint.State st = fm.checkpoint.load(fingerprint);
        double loadS = (System.nanoTime() - start) / 1e9;
        if (st == null) {
            st = new FeiCheckpoint.State(fingerprint, FarMethods.commit(), "экспорт " + exportDir + ", коммит данных " + FarMethods.dataCommit(dir));
            fm.timeText.append(String.format(Locale.ROOT, "Загрузка и векторы: %.0f с, снимков FEI %d.%n", loadS, frames.size()));
        } else {
            fm.results.putAll(st.results);
            fm.timeText.append(st.timeText);
            fm.timeText.append(String.format(Locale.ROOT, "Продолжено с конфигурации %d (контрольная точка %s); загрузка и векторы: %.0f с.%n",
                    st.done, fm.checkpoint.dir, loadS));
            System.out.print(fm.timeText.substring(fm.timeText.lastIndexOf("Продолжено")));
        }
        fm.state = st;
        fm.run(st.done, limit);
        if (!fm.cvIds.isEmpty()) {
            fm.checkpoint.assemble(outDir.resolve(CV_FILE), fm.cvHeader(st.code, st.source, n), n, fm.checkpoint::cvPart);
        }
        if (fm.cvOnly) {
            // Только перекрёстная оценка: файла оценок, векторов и fei_methods.txt нет.
            fm.timeText.append(String.format(Locale.ROOT, "Всего %.0f с%s.%n", (System.nanoTime() - start) / 1e9,
                    fm.timeText.indexOf("Продолжено") >= 0 ? " (последний запуск)" : ""));
            FarMethods.write(outDir.resolve("fei_time.txt"), fm.timeText);
            fm.checkpoint.delete();
            System.out.print(fm.timeText);
            return;
        }
        List<String> extra = new ArrayList<>();
        if (fm.mirrorTurn) extra.add(FeiScores.R0_HEADER + fm.r0);
        if (fm.lbpNorm != IlluminationNorm.NONE) extra.add("# " + fm.lbpNormText());
        String header = FeiScores.header(st.code, st.source, n, new ArrayList<>(fm.about.keySet()), extra);
        fm.checkpoint.assemble(outDir.resolve(FeiScores.FILE), header, n);
        if (!fm.vectorIds.isEmpty()) {
            String vecKeys = String.format(Locale.ROOT, "faces.fei.mirror = %s, faces.fei.vectors = %s, seed %d", settings.loadFacesFeiMirror(),
                    settings.loadFacesFeiVectors(), fm.seed);
            FeiCheckpoint.State s0 = st;
            fm.checkpoint.assembleVectors(outDir, fm.vectorIds, m -> FeiVectors.header(s0.code, s0.source, vecKeys, n, m, fm.splits), n);
        }
        FarMethods.write(outDir.resolve("fei_methods.txt"), fm.report(st.code, st.source, limit));
        fm.timeText.append(String.format(Locale.ROOT, "Всего %.0f с%s.%n", (System.nanoTime() - start) / 1e9,
                fm.timeText.indexOf("Продолжено") >= 0 ? " (последний запуск)" : ""));
        FarMethods.write(outDir.resolve("fei_time.txt"), fm.timeText);
        fm.checkpoint.delete();
        System.out.print(fm.timeText);
    }

    /** Номера снимков в мозаике проверки отражения: повороты в обе стороны и анфас. */
    static final int[] MIRROR_CHECK_NUMBERS = {4, 5, 6, 7, 11};

    /**
     * Мозаика для проверки глазами (+mirror): у первых двух людей, у которых есть все снимки MIRROR_CHECK_NUMBERS, — пары
     * «исходный кадр а | отражённый с поправкой», увеличение ×3, красные точки — шаблон кадра (глаза, нос, углы рта).
     */
    static void mirrorCheck(Map<String, Map<Integer, String>> keys, Map<String, Mat> frames, Path file) throws IOException {
        List<Mat> rows = new ArrayList<>();
        int persons = 0;
        for (Map.Entry<String, Map<Integer, String>> e : keys.entrySet()) {
            if (persons == 2) break;
            if (!Arrays.stream(MIRROR_CHECK_NUMBERS).allMatch(n -> e.getValue().containsKey(n))) continue;
            persons++;
            for (int n : MIRROR_CHECK_NUMBERS) {
                Mat src = frames.get(e.getValue().get(n));
                Mat mir = FaceMirror.mirror(src);
                List<Mat> pair = new ArrayList<>();
                for (Mat m : new Mat[] {src, mir}) pair.add(checkTile(m, "p" + e.getKey() + " n" + n + (m == src ? " src" : " mirror")));
                Mat row = new Mat();
                org.opencv.core.Core.hconcat(pair, row);
                rows.add(row);
                mir.release();
            }
        }
        Mat sheet = new Mat();
        org.opencv.core.Core.vconcat(rows, sheet);
        org.opencv.core.MatOfByte buf = new org.opencv.core.MatOfByte();
        org.opencv.imgcodecs.Imgcodecs.imencode(".png", sheet, buf);
        Files.write(file, buf.toArray());
    }

    private static Mat checkTile(Mat gray, String label) {
        int k = 3;
        Mat big = new Mat();
        org.opencv.imgproc.Imgproc.resize(gray, big, new org.opencv.core.Size(gray.cols() * k, gray.rows() * k), 0, 0,
                org.opencv.imgproc.Imgproc.INTER_NEAREST);
        Mat bgr = new Mat();
        org.opencv.imgproc.Imgproc.cvtColor(big, bgr, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR);
        for (double[] p : FaceAlignment.ownTemplate(gray.cols(), gray.rows())) {
            org.opencv.imgproc.Imgproc.circle(bgr, new org.opencv.core.Point(p[0] * k + (k - 1) / 2.0, p[1] * k + (k - 1) / 2.0), 2,
                    new org.opencv.core.Scalar(0, 0, 255), -1);
        }
        org.opencv.imgproc.Imgproc.putText(bgr, label, new org.opencv.core.Point(3, 12), org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX, 0.4,
                new org.opencv.core.Scalar(0, 255, 0), 1);
        big.release();
        return bgr;
    }

    /**
     * Ключи отобранных снимков: человек → номер → ключ. Отобранные — included = 1 в fei_selection.tsv; они же — ровно
     * снимки FEI экспорта (иначе ошибка).
     */
    static Map<String, Map<Integer, String>> selection(List<String> lines, Map<String, List<Sample>> fei) {
        Map<String, Map<Integer, String>> byFile = new TreeMap<>();
        for (List<Sample> list : fei.values()) {
            for (Sample s : list) {
                String name = s.file().getFileName().toString();
                int number = Integer.parseInt(name.substring(name.indexOf('-') + 1));
                byFile.computeIfAbsent(s.person(), k -> new TreeMap<>()).put(number, s.file().toString());
            }
        }
        Map<String, Set<Integer>> sel = new TreeMap<>();
        for (String l : lines) {
            if (l.startsWith("#") || l.startsWith("person\t")) continue;
            String[] c = l.split("\t", -1);
            if (c[6].equals("1")) sel.computeIfAbsent(c[0], k -> new HashSet<>()).add(Integer.parseInt(c[1]));
        }
        Map<String, Set<Integer>> have = new TreeMap<>();
        byFile.forEach((p, m) -> have.put(p, new HashSet<>(m.keySet())));
        if (!sel.equals(have)) throw new IllegalStateException("Снимки FEI экспорта не совпадают с included = 1 в fei_selection.tsv");
        return byFile;
    }

    // ---------------------------------------------------------------- оценка

    /** Ключ снимка человека p с номером n. */
    String key(String p, int n) {
        return keys.get(p).get(n);
    }

    /** Все отобранные снимки людей. */
    List<String> keysOf(List<String> persons) {
        List<String> out = new ArrayList<>();
        for (String p : persons) out.addAll(keys.get(p).values());
        return out;
    }

    /** Обучающие классы конфигурации: свои (7 снимков), затем (withCohort) люди когорты со всеми снимками. */
    List<List<double[]>> classes(FeiProtocol.Split sp, int j, IlluminationNorm norm, boolean withCohort) {
        return classes(sp, j, vectors.get(norm), withCohort);
    }

    /** То же по векторам снимков входа метода. */
    List<List<double[]>> classes(FeiProtocol.Split sp, int j, Map<String, double[]> vec, boolean withCohort) {
        List<List<double[]>> out = new ArrayList<>();
        for (String p : sp.gallery()) {
            List<double[]> list = new ArrayList<>();
            for (int n : FeiProtocol.trainNumbers(j)) list.add(vec.get(key(p, n)));
            out.add(list);
        }
        if (withCohort) {
            for (String p : sp.cohort()) out.add(keysOf(List.of(p)).stream().map(vec::get).toList());
        }
        return out;
    }

    /**
     * Обучающие классы +mirror (только CLAHE): в каждом классе — те же снимки, затем отражения (ALL: свои 7 → 14, у когорты
     * в обучении — все её снимки и их отражения; TURN: отражения только снимков с |r| > r0, FeiMethods.mirrorKeys).
     */
    List<List<double[]>> mirrorClasses(FeiProtocol.Split sp, int j, boolean withCohort, Mirror mode) {
        return mirrorClasses(sp, j, withCohort, mode, Feature.PIXELS);
    }

    /** То же для входа feature: пиксели — CLAHE и mirrorVectors; LBP — гистограммы (или их корень) исходных и отражённых кадров. */
    List<List<double[]>> mirrorClasses(FeiProtocol.Split sp, int j, boolean withCohort, Mirror mode, Feature feature) {
        Map<String, double[]> mirrored = switch (feature) {
            case PIXELS -> mirrorVectors;
            case LBP -> lbpMirror;
            case LBP_SQRT -> lbpSqrtMirror;
            case LBP_SQRT_IN -> lbpMaskedMirror.get(LbpFeature.Mask.INNER);
            case LBP_SQRT_OUT -> lbpMaskedMirror.get(LbpFeature.Mask.RING);
        };
        List<List<double[]>> base = classes(sp, j, input(feature, IlluminationNorm.CLAHE), withCohort);
        List<List<String>> ks = new ArrayList<>();
        for (String p : sp.gallery()) ks.add(mirrorKeys(p, keys.get(p), FeiProtocol.trainNumbers(j), mode, pose, r0));
        if (withCohort) for (String p : sp.cohort()) ks.add(mirrorKeys(p, keys.get(p), keys.get(p).keySet(), mode, pose, r0));
        List<List<double[]>> out = new ArrayList<>();
        for (int c = 0; c < base.size(); c++) {
            List<double[]> list = new ArrayList<>(base.get(c));
            for (String k : ks.get(c)) list.add(mirrored.get(k));
            out.add(list);
        }
        return out;
    }

    /** Векторы снимков входа метода: пиксели — по нормализации norm, LBP — гистограммы или их корень. */
    Map<String, double[]> input(Feature feature, IlluminationNorm norm) {
        return switch (feature) {
            case PIXELS -> vectors.get(norm);
            case LBP -> lbpVectors;
            case LBP_SQRT -> lbpSqrtVectors;
            case LBP_SQRT_IN -> lbpMasked.get(LbpFeature.Mask.INNER);
            case LBP_SQRT_OUT -> lbpMasked.get(LbpFeature.Mask.RING);
        };
    }

    /** Все конфигурации (limit — первые limit, для отладки). */
    /** Конфигурации с from (продолжение после контрольной точки; итоги до from — уже в results) до limit. */
    void run(int from, int limit) {
        if (results.isEmpty()) for (String id : about.keySet()) results.put(id, new Result(id));
        for (int c = from; c < Math.min(limit, CONFIGS); c++) {
            FeiProtocol.Split sp = splits.get(c / FeiProtocol.NUMBERS.length);
            int j = c % FeiProtocol.NUMBERS.length;
            long t0 = System.nanoTime();
            // faces.fei.cv = only: основные строки не считаются (в отчётах их нет).
            Map<String, Map<String, double[]>> sc = cvOnly ? null : scoreConfig(sp, j);
            if (!cvOnly) for (String id : about.keySet()) results.get(id).add(this, sp, j, sc.get(id));
            String text = cvOnly ? "" : scoresText(sp, j, sc);
            String cv = null;
            if (!cvIds.isEmpty()) {
                long t1 = System.nanoTime();
                cv = cvText(sp, j);
                timeText.append(String.format(Locale.ROOT, "  перекрёстная оценка (%d потоков): %.1f с%n", cvThreads, (System.nanoTime() - t1) / 1e9));
            }
            timeText.append(String.format(Locale.ROOT, "Конфигурация %d (разбиение %d, контроль №%d): %.0f с.%n", c, sp.index(),
                    FeiProtocol.NUMBERS[j], (System.nanoTime() - t0) / 1e9));
            if (checkpoint != null) {
                try {
                    if (!vectorIds.isEmpty()) checkpoint.saveVectors(c, vectorsText(sp, j, sc));
                    if (cv != null) checkpoint.saveCv(c, cv);
                    checkpoint.savePart(c, text);
                    state.done = c + 1;
                    state.results = new LinkedHashMap<>(results);
                    state.timeText = timeText.toString();
                    checkpoint.save(state);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            System.out.print(timeText.substring(timeText.lastIndexOf("Конфигурация")));
        }
    }

    /** Ключи проб конфигурации: контроль своих, пороговые, контроль чужих, когорта. */
    List<String> probes(FeiProtocol.Split sp, int j) {
        List<String> out = new ArrayList<>();
        for (String p : sp.gallery()) out.add(key(p, FeiProtocol.NUMBERS[j]));
        out.addAll(keysOf(sp.threshold()));
        out.addAll(keysOf(sp.control()));
        out.addAll(keysOf(sp.cohort()));
        return out;
    }

    /** Оценки всех методов в конфигурации (sp, j) по пробам: метод → ключ → оценки по 100 своим; сведения — в Result. */
    Map<String, Map<String, double[]>> scoreConfig(FeiProtocol.Split sp, int j) {
        List<String> probes = probes(sp, j);
        int g = FeiProtocol.GALLERY;
        Map<String, Map<String, double[]>> sc = new HashMap<>();
        WpcaScorer.Pca pca = null;
        for (Base b : bases) {
            long t0 = System.nanoTime();
            boolean mirrored = b.mirror() != Mirror.NONE;
            List<List<double[]>> cls = mirrored ? mirrorClasses(sp, j, b.cohortInTraining(), b.mirror(), b.feature())
                    : classes(sp, j, input(b.feature(), b.norm()), b.cohortInTraining());
            GalleryScorer.ScoreModel model;
            if (b.scorer() instanceof WpcaScorer && mirrored) {
                // +mirror: своя PCA на отражённых классах.
                WpcaScorer w = (WpcaScorer) b.scorer();
                model = WpcaScorer.model(WpcaScorer.Pca.of(cls), cls, g, w.mode(), w.c());
            } else if (b.scorer() instanceof WpcaScorer) {
                // Одна PCA на конфигурацию для всех строк wpca.
                if (pca == null) pca = WpcaScorer.Pca.of(cls);
                WpcaScorer w = (WpcaScorer) b.scorer();
                model = WpcaScorer.model(pca, cls, g, w.mode(), w.c());
            } else {
                model = b.scorer().fit(cls, g);
            }
            Map<String, double[]> vec = input(b.feature(), b.norm());
            double[][] out = new double[probes.size()][];
            GalleryScorer.ScoreModel m = model;
            IntStream.range(0, probes.size()).parallel().forEach(i -> out[i] = m.scores(vec.get(probes.get(i))));
            Map<String, double[]> map = new HashMap<>();
            for (int i = 0; i < probes.size(); i++) map.put(probes.get(i), out[i]);
            sc.put(b.id(), map);
            results.get(b.id()).info.add(model.info());
            timeText.append(String.format(Locale.ROOT, "  %s: %.1f с%n", b.id(), (System.nanoTime() - t0) / 1e9));
        }
        List<String> cohort = keysOf(sp.cohort());
        for (Derived d : derived) {
            Map<String, double[]> map = new HashMap<>();
            switch (d.kind()) {
                case RATIO -> sc.get(d.parts().get(0)).forEach((k, v) -> map.put(k, SubspaceScorer.ratio(v)));
                case ZNORM -> {
                    Map<String, double[]> s = sc.get(d.parts().get(0));
                    double[] mu = new double[g];
                    double[] sigma = new double[g];
                    ZNormScorer.stats(cohort.stream().map(s::get).toArray(double[][]::new), mu, sigma);
                    s.forEach((k, v) -> map.put(k, ZNormScorer.normalize(v, mu, sigma)));
                    results.get(d.id()).info.add(String.format(Locale.ROOT, "Z-norm: когорта %d снимков, σ медиана %.4g", cohort.size(), median(sigma)));
                }
                case SUM, AGREE -> {
                    int n = d.parts().size();
                    List<Map<String, double[]>> s = new ArrayList<>();
                    for (String p : d.parts()) s.add(sc.get(p));
                    double[][] mu = new double[n][g];
                    double[][] sigma = new double[n][g];
                    for (int q = 0; q < n; q++) ZNormScorer.stats(cohort.stream().map(s.get(q)::get).toArray(double[][]::new), mu[q], sigma[q]);
                    FusionScorer.Mode mode = d.kind() == Kind.SUM ? FusionScorer.Mode.SUM : FusionScorer.Mode.AGREE;
                    for (String k : probes) {
                        double[][] raw = new double[n][];
                        for (int q = 0; q < n; q++) raw[q] = s.get(q).get(k);
                        map.put(k, FusionScorer.combine(raw, mu, sigma, mode));
                    }
                    results.get(d.id()).info.add(String.format(Locale.ROOT, "нормировка: когорта %d снимков", cohort.size()));
                }
            }
            sc.put(d.id(), map);
        }
        return sc;
    }

    /** Оценки попыток конфигурации (строки файла оценок): по методам (порядок отчёта) — свои, пороговые, контрольные чужие. */
    String scoresText(FeiProtocol.Split sp, int j, Map<String, Map<String, double[]>> sc) {
        Set<String> agree = new HashSet<>();
        for (Derived d : derived) if (d.kind() == Kind.AGREE) agree.add(d.id());
        List<String> gallery = sp.gallery();
        StringBuilder b = new StringBuilder();
        for (String id : about.keySet()) {
            Map<String, double[]> s = sc.get(id);
            boolean agr = agree.contains(id);
            int n = FeiProtocol.NUMBERS[j];
            for (int p = 0; p < gallery.size(); p++) {
                String person = gallery.get(p);
                b.append(FeiScores.line(id, sp.index(), j, FeiScores.OWN, person, n, p, gallery, s.get(key(person, n)), agr));
            }
            for (String role : new String[] {FeiScores.THR, FeiScores.CTRL}) {
                for (String person : role.equals(FeiScores.THR) ? sp.threshold() : sp.control()) {
                    for (Map.Entry<Integer, String> e : keys.get(person).entrySet()) {
                        b.append(FeiScores.line(id, sp.index(), j, role, person, e.getKey(), -1, gallery, s.get(e.getValue()), agr));
                    }
                }
            }
        }
        return b.toString();
    }

    static double median(double[] v) {
        double[] s = Arrays.stream(v).filter(Double::isFinite).sorted().toArray();
        return s.length == 0 ? Double.NaN : s[s.length / 2];
    }

    /** Попытки чужих: оценка (минимум по галерее) по людям. */
    Impostors impostors(List<String> persons, Map<String, double[]> scores) {
        Map<String, double[]> map = new LinkedHashMap<>();
        for (String p : persons) {
            List<String> k = keysOf(List.of(p));
            double[] v = new double[k.size()];
            for (int i = 0; i < v.length; i++) v[i] = FarMethods.best(scores.get(k.get(i)))[0];
            map.put(p, v);
        }
        return new Impostors(map);
    }

    // ---------------------------------------------------------------- метрики

    /** Метрики метода, накопленные по конфигурациям. */
    static final class Result implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        final String id;
        final List<String> info = new ArrayList<>();
        int configs;
        int att;
        int correct;
        /** [анфас 1 / поворот 0][α]: отказы (FRR) и приняты под чужим именем; попытки — attBy. */
        final int[][] rej = new int[2][ALPHAS.length];
        final int[][] mis = new int[2][ALPHAS.length];
        final int[] attBy = new int[2];
        /** [α][конфигурация] → {x, n, xP, nP}: контроль 40 и пороговые 40. */
        final int[][][] farCtrl = new int[ALPHAS.length][CONFIGS][];
        final int[][][] farThr = new int[ALPHAS.length][CONFIGS][];
        final double[][] theta = new double[ALPHAS.length][CONFIGS];

        Result(String id) {
            this.id = id;
        }

        int errors(int a) {
            return rej[0][a] + rej[1][a] + mis[0][a] + mis[1][a];
        }

        int errors(int a, int frontal) {
            return rej[frontal][a] + mis[frontal][a];
        }

        int frr(int a) {
            return rej[0][a] + rej[1][a];
        }

        int misaccepted(int a) {
            return mis[0][a] + mis[1][a];
        }

        void add(FeiMethods fm, FeiProtocol.Split sp, int j, Map<String, double[]> s) {
            int c = configs++;
            Impostors thr = fm.impostors(sp.threshold(), s);
            Impostors ctrl = fm.impostors(sp.control(), s);
            int fr = FeiProtocol.frontal(j) ? 1 : 0;
            double[] own = new double[sp.gallery().size()];
            boolean[] ok = new boolean[own.length];
            for (int p = 0; p < own.length; p++) {
                double[] v = s.get(fm.key(sp.gallery().get(p), FeiProtocol.NUMBERS[j]));
                double[] b = FarMethods.best(v);
                ok[p] = (int) b[1] == p;
                own[p] = Double.isInfinite(v[p]) ? Double.POSITIVE_INFINITY : b[0];
                if (ok[p]) correct++;
            }
            att += own.length;
            attBy[fr] += own.length;
            for (int a = 0; a < ALPHAS.length; a++) {
                double th = FarMethods.personThreshold(thr, ALPHAS[a]);
                theta[a][c] = th;
                for (int p = 0; p < own.length; p++) {
                    if (own[p] > th) rej[fr][a]++;
                    else if (!ok[p]) mis[fr][a]++;
                }
                farCtrl[a][c] = GalleryEvaluation.far(ctrl, th);
                farThr[a][c] = GalleryEvaluation.far(thr, th);
            }
        }

        /** Конфигурации с результатом. */
        int[][] far(int[][][] f, int a) {
            return Arrays.copyOf(f[a], configs);
        }
    }

    // ---------------------------------------------------------------- выбор и отчёт

    /** Лучший: наименьшая ошибка своих при α = 0,05, при равенстве — при α = 0; FAR контроля в выборе не участвует. */
    String best(List<String> ids) {
        String best = null;
        for (String id : ids) {
            Result r = results.get(id);
            if (best == null) {
                best = id;
                continue;
            }
            Result b = results.get(best);
            if (r.errors(A05) < b.errors(A05) || r.errors(A05) == b.errors(A05) && r.errors(A0) < b.errors(A0)) best = id;
        }
        return best;
    }

    /** Строки таблицы: все, кроме условных 4-sum-blk-both+wpca; условная — лучшей wpca, если та обходит 3-znorm-lda. */
    List<String> tableIds() {
        List<String> ids = new ArrayList<>(about.keySet().stream().filter(id -> !id.startsWith("4-sum-blk-both+") && !isMirror(id)).toList());
        if (!wpcaIds().isEmpty() && wpcaBeats()) ids.add(blkWpcaId(best(wpcaIds())));
        // +mirror — в конце таблицы; в выборе лучшего по правилу и условной строке не участвуют.
        ids.addAll(about.keySet().stream().filter(FeiMethods::isMirror).toList());
        return ids;
    }

    boolean wpcaBeats() {
        return best(List.of(best(wpcaIds()), "3-znorm-lda")).startsWith("4-wpca-");
    }

    StringBuilder report(String code, String source, int limit) {
        StringBuilder t = new StringBuilder();
        int n = Math.min(limit, CONFIGS);
        Result any = results.values().iterator().next();
        t.append("Этап 5: протокол «только FEI» (FeiMethods, faces-fei)\n");
        t.append("Код: коммит ").append(code).append('\n');
        t.append("Данные: ").append(source).append(" (a/fei — кадр а 92×112, fei_selection.tsv).\n");
        if (n < CONFIGS) t.append("ОТЛАДОЧНЫЙ ПРОГОН: конфигураций ").append(n).append(" из ").append(CONFIGS).append(".\n");
        int images = keys.values().stream().mapToInt(Map::size).sum();
        t.append(String.format(Locale.ROOT, "FEI: %d человек, отобранных снимков %d (included = 1: |r| ≤ R = %.4f или №11/№12); «полных» (прошли все 8 "
                + "снимков №4, 5, 6, 7, 11, 12, 13, 14) — %d.%n", keys.size(), images, info.r(), FeiProtocol.fullPersons(included).size()));
        t.append(String.format(Locale.ROOT, "Разбиение людей (seed %d + s, s = 0…%d): свои %d — случайно из полных, у каждого ровно 8 снимков №4–7, "
                + "11–14; чужие %d (остальные полные + неполные) со всеми отобранными снимками: когорта %d (посторонние в обучении 2b, 2c, "
                + "4-wpca; когорта Z-norm и нормировки объединений), пороговые %d, контроль %d. Роли не пересекаются.%n", seed,
                FeiProtocol.SPLITS - 1, FeiProtocol.GALLERY, FeiProtocol.COHORT + FeiProtocol.THRESHOLD + FeiProtocol.CONTROL,
                FeiProtocol.COHORT, FeiProtocol.THRESHOLD, FeiProtocol.CONTROL));
        t.append(String.format(Locale.ROOT, "Конфигурации: %d разбиений × 8 = %d; в конфигурации j у каждого своего на контроле снимок j-й из (4, 5, 6, "
                + "7, 11, 12, 13, 14), на обучении остальные 7. Попыток своих %d (анфас №11–14 — %d, поворот №4–7 — %d).%n",
                FeiProtocol.SPLITS, CONFIGS, any.att, any.attBy[1], any.attBy[0]));
        t.append("Порог: Нейман – Пирсон по доле людей пороговых 40 чужих (чужой принят, если принят хотя бы один его снимок), в каждой\n"
                + "конфигурации; α = 0,05 — допускается ⌊0,05·40⌋ = 2 человека, α = 0 — ни одного. Ошибка своих = FRR + принят под\n"
                + "чужим именем (argmin — другой свой, оценка ≤ θ). Контрольные 40 чужих — только FAR: в обучении, пороге, когорте,\n"
                + "нормировке и выборе метода не участвуют. FAR контроля по людям — худшая и медианная из конфигураций (↑95 — верхняя\n"
                + "граница Клоппера – Пирсона), по попыткам — сумма по конфигурациям (справочно: одни и те же чужие).\n");
        t.append("Методы и параметры — как в этапе 4 (FarMethods, a47ea5d), заново не подбирались: предобработка CLAHE (clip ")
                .append(String.format(Locale.ROOT, "%.2f, сетка %d×%d), у 1-tt-eps — Tan – Triggs;%n", params.claheClip(), params.claheTile(),
                        params.claheTile()));
        t.append("подпространство k = 4 (≤ N − 1); Fisherfaces — PCA по энергии 95 % (2a95-fisher — это «2a-fisher» этапа 4, не PCA до N − C\n"
                + "этапа 2); области блоков — глаза 8, 38, 76×26, нос 30, 52, 32×30, рот 20, 80, 52×24; нормировка частей объединений —\n"
                + "по когорте 20 (как 69 посторонних в этапе 4); комбинации — те же, «все пять» = ratio + eps + fisher + zsvd + zlda.\n");
        t.append("Новое — 4-wpca: CLAHE; PCA на своих (7 × 100) + когорте 20, k по энергии 95 %; yᵢ = uᵢᵀ(x − μ)/√(λᵢ + δ), δ = c·λ_k,\n"
                + "c ∈ {0; 0,1; 1}; cos — эталон — среднее y по обучающим снимкам человека, оценка 1 − cos(y, m); cos-z — то же с\n"
                + "Z-norm по когорте 20 (когорта участвует и в PCA); eps — подпространство человека в отбеленном пространстве\n"
                + "(k по энергии 95 %, k ≤ N − 1 = 6), ε. Перед LDA отбеливание не делается (LDA к нему инвариантна).\n");
        t.append("Проверка разбиения (FeiProtocol.check, каждое разбиение): роли свои / когорта / пороговые / контроль не пересекаются,\n"
                + "размеры 100 / 20 / 40 / 40, у каждого своего есть все 8 снимков; порог — только по пороговым, нормировки — только по\n"
                + "когорте (FeiMethods.scoreConfig). Прошла.\n");
        if (mirror) {
            t.append("Режим +mirror (faces.fei.mirror, этап 5б): строки с суффиксом +mirror — те же методы, к обучению добавлены отражённые\n"
                    + "по горизонтали кадры своих (7 → 14), у 2c-mlda и 4-wpca — и когорты 20; отражённый кадр выровнен заново подобием\n"
                    + "«отражённый шаблон → шаблон» (FaceMirror, способ а; шаблон кадра несимметричен — без поправки сдвиг 0,8–1,3 px).\n"
                    + "Пробы (контроль своих, пороговые, контрольные чужие) и когорта в нормировке — без отражения. Строки +mirror — в конце\n"
                    + "таблиц, в выборе лучшего по правилу не участвуют; прежние строки не меняются.\n");
        }
        if (mirrorTurn) {
            t.append("Режим +mirror-turn (faces.fei.mirror = turn, этап 5б): то же, но отражаются только обучающие снимки с |r| > r0 = "
                    + r0 + "\n(faces.fei.pose.r0; r — смещение носа от середины глаз в долях межглазья, точки YuNet, fei_selection.tsv;\n"
                    + "|r| = r0 — без отражения), у своих и у когорты в обучении (2c-mlda, 4-wpca); строки — в конце таблиц, в выборе\n"
                    + "лучшего не участвуют.\n");
        }
        if (lbpRing) {
            t.append("Режим mask (faces.fei.lbp, п. 4.2, «LBP: внутренние и краевые блоки»): " + LBP_MLDA_IN + " и " + LBP_MLDA_OUT + " — как "
                    + LBP_MLDA + ", вход — корень гистограмм только\nвнутренних блоков (строки и столбцы 1–5 сетки 7×7) или только краевого "
                    + "кольца (строка или столбец 0 или 6); маски заданы заранее.\n");
        } else if (lbpMask) {
            t.append("Режим in (faces.fei.lbp): " + LBP_MLDA_IN + " — как " + LBP_MLDA + ", вход — корень гистограмм только внутренних блоков\n"
                    + "(строки и столбцы 1–5 сетки 7×7); строк краевого кольца (" + LBP_MLDA_OUT + ") нет.\n");
        }
        if (lbpNorm != IlluminationNorm.NONE) t.append(lbpNormText()).append('\n');
        if (lbpOnly) {
            t.append("Режим only (faces.fei.lbp): считаются только строки LBP-MLDA; прочих методов в таблицах нет, лучший по правилу —\n"
                    + "среди этих строк.\n");
        }

        t.append("\n=== Сводка (основная таблица FEI) ===\n");
        t.append("метод | суть | ошибки своих α=0,05 (FRR + под чужим) | α=0 | анфас / поворот α=0,05 | анфас / поворот α=0 | "
                + "FAR контроля люди α=0,05: худш. (↑95) / мед. | FAR контроля попытки α=0,05, сумма | FAR контроля люди α=0: худш. / мед. | "
                + "argmin своих\n");
        List<String> ids = tableIds();
        for (String id : ids) t.append(row(results.get(id))).append('\n');
        String best = best(ids.stream().filter(id -> !isMirror(id)).toList());
        t.append(String.format(Locale.ROOT, "%nЛучший по правилу (наименьшая ошибка своих при α = 0,05, при равенстве — α = 0; FAR контроля не "
                + "участвует): %s — %s.%n", best, pctOf(results.get(best).errors(A05), results.get(best).att)));
        if (!lbpOnly) {
            String bw = best(wpcaIds());
            t.append(String.format(Locale.ROOT, "Лучшая строка wpca: %s (%s) против 3-znorm-lda (%s): %s.%n", bw, pctOf(results.get(bw).errors(A05),
                    results.get(bw).att), pctOf(results.get("3-znorm-lda").errors(A05), results.get("3-znorm-lda").att),
                    wpcaBeats() ? "обходит — добавлена строка " + blkWpcaId(bw) : "не обходит — строка 4-sum-blk-both+wpca не добавляется"));
        }

        t.append("\n=== FAR на пороговых (набор порога, справочно) ===\n");
        t.append("метод | θ α=0,05 мед. | люди α=0,05: худш. / мед. | попытки α=0,05 сумма | θ α=0 мед.\n");
        for (String id : ids) {
            Result r = results.get(id);
            t.append(id).append(" | ").append(String.format(Locale.ROOT, "%.6f", median(Arrays.copyOf(r.theta[A05], r.configs)))).append(" | ")
                    .append(personsNoUb(r.far(r.farThr, A05))).append(" | ").append(attemptsSum(r.far(r.farThr, A05))).append(" | ")
                    .append(String.format(Locale.ROOT, "%.6f", median(Arrays.copyOf(r.theta[A0], r.configs)))).append('\n');
        }

        t.append("\n=== Разбиения людей ===\n");
        for (FeiProtocol.Split sp : splits) {
            t.append("Разбиение ").append(sp.index()).append(":\n");
            t.append("  свои: ").append(String.join(", ", sp.gallery())).append('\n');
            t.append("  когорта: ").append(String.join(", ", sp.cohort())).append('\n');
            t.append("  пороговые: ").append(String.join(", ", sp.threshold())).append('\n');
            t.append("  контроль: ").append(String.join(", ", sp.control())).append('\n');
        }

        t.append("\n=== Подробности ===\n");
        for (String id : about.keySet()) {
            Result r = results.get(id);
            t.append(String.format(Locale.ROOT, "%n--- %s — %s ---%n", id, about.get(id)));
            t.append(String.format(Locale.ROOT, "argmin своих %d/%d; попыток анфас %d, поворот %d.%n", r.correct, r.att, r.attBy[1], r.attBy[0]));
            for (int a = 0; a < ALPHAS.length; a++) {
                t.append(String.format(Locale.ROOT, "  α = %.2f: FRR %d, под чужим %d, всего %d/%d = %s %%; анфас FRR %d + под чужим %d, поворот "
                        + "FRR %d + под чужим %d; FAR контроля %s; FAR пороговых %s%n", ALPHAS[a], r.frr(a), r.misaccepted(a), r.errors(a), r.att,
                        GalleryEvaluation.pct(r.errors(a), r.att), r.rej[1][a], r.mis[1][a], r.rej[0][a], r.mis[0][a],
                        farText(r.far(r.farCtrl, a)), farText(r.far(r.farThr, a))));
            }
            if (!r.info.isEmpty() && !r.info.get(0).isEmpty()) {
                t.append("  Модель по конфигурациям:\n");
                for (int i = 0; i < r.info.size(); i++) t.append(String.format(Locale.ROOT, "    %2d: %s%n", i, r.info.get(i)));
            }
        }
        return t;
    }

    String row(Result r) {
        return String.join(" | ", r.id, about.get(r.id),
                String.format(Locale.ROOT, "%d + %d = %s", r.frr(A05), r.misaccepted(A05), pctOf(r.errors(A05), r.att)),
                pctOf(r.errors(A0), r.att),
                pctOf(r.errors(A05, 1), r.attBy[1]) + " / " + pctOf(r.errors(A05, 0), r.attBy[0]),
                pctOf(r.errors(A0, 1), r.attBy[1]) + " / " + pctOf(r.errors(A0, 0), r.attBy[0]),
                FarMethods.persons(r.far(r.farCtrl, A05)), attemptsSum(r.far(r.farCtrl, A05)), personsNoUb(r.far(r.farCtrl, A0)),
                r.correct + "/" + r.att);
    }

    static String pctOf(int x, int n) {
        return x + "/" + n + " = " + GalleryEvaluation.pct(x, n) + " %";
    }

    /** По людям: худшая / медианная конфигурация без границы. */
    static String personsNoUb(int[][] cfg) {
        Integer[] idx = new Integer[cfg.length];
        for (int i = 0; i < cfg.length; i++) idx[i] = i;
        Arrays.sort(idx, (x, y) -> Double.compare(cfg[x][2] / (double) cfg[x][3], cfg[y][2] / (double) cfg[y][3]));
        int[] w = cfg[idx[cfg.length - 1]];
        int[] m = cfg[idx[cfg.length / 2]];
        return w[2] + "/" + w[3] + " / " + m[2] + "/" + m[3];
    }

    /** По попыткам: сумма по конфигурациям. */
    static String attemptsSum(int[][] cfg) {
        int x = 0;
        int n = 0;
        for (int[] c : cfg) {
            x += c[0];
            n += c[1];
        }
        return x + "/" + n + " = " + String.format(Locale.ROOT, "%.2f", 100.0 * x / n) + " %";
    }

    /** FAR подробно: люди худш. (↑95) / мед., попытки худш. (↑95), сумма. */
    static String farText(int[][] cfg) {
        return "люди " + FarMethods.persons(cfg) + ", попытки худш. " + FarMethods.attempts(cfg) + ", сумма " + attemptsSum(cfg);
    }
}
