package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
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
import java.util.function.Supplier;
import org.opencv.core.Mat;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.GalleryEvaluation.Base;
import svd.recognizer.faces.GalleryEvaluation.Data;
import svd.recognizer.faces.GalleryEvaluation.Det;
import svd.recognizer.faces.GalleryEvaluation.Impostors;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Splits;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * Шаг 5, SVD: улучшение отсечения чужих (одна команда: {@code mvn compile exec:exec@faces-far-methods},
 * отдельная JVM с -Xmx6g). Протокол — как в GalleryEvaluation (отчёт 32dc1fa): галерея 6 своих + 50 Georgia
 * Tech, 12 конфигураций, чужие MUCT 138 валидация / 138 контроль, ORL — строка контроля; вход 2′, кадр а
 * 92×112. Базы — из экспорта (faces.export.dir) или из сырых снимков с кэшем детекций faces-far.
 *
 * Конфигурации (faces.far.methods, по умолчанию все): этап 1а — базовая линия «простой SVD по векторам»
 * (ближайший снимок или средний вектор человека, евклидово расстояние); этап 1 — подпространство на человека
 * (ε₁/ε₂ и ε) без нормализации освещения и с ней. Валидационные 138 MUCT делятся по людям (seed + 1000)
 * на 69 посторонних (обучение LDA) и 69 пороговых. Контрольные 138 MUCT не участвуют ни в обучении, ни в
 * пороге, ни в выборе лучшего.
 *
 * Отчёты (вне git): reports/faces/far/far_methods.txt (сводка и подробности), far_methods_auc.txt
 * (диагностика по людям галереи), far_methods_time.txt (время; не входит в побайтное сравнение).
 *
 * @author ssv
 */
public final class FarMethods {

    static final int CONFIGS = GalleryEvaluation.CONFIGS;
    static final int GT_SPLITS = GalleryEvaluation.GT_SPLITS;
    static final double[] ALPHAS = GalleryEvaluation.ALPHAS;
    /** Индекс α = 0,05 и α = 0 в ALPHAS. */
    static final int A05 = 0;
    static final int A0 = ALPHAS.length - 1;
    /** Сдвиг seed для деления валидации MUCT на посторонних и пороговых (не совпадает с делением MUCT). */
    static final long OUTSIDER_SEED_SHIFT = 1000;

    /** Набор чужих для порога Неймана – Пирсона. */
    enum Threshold {
        VAL138("валидация MUCT, 138 человек"), VAL69("пороговые 69 человек валидации MUCT");

        final String label;

        Threshold(String label) {
            this.label = label;
        }
    }

    /** Конфигурация: идентификатор, этап, нормализация, оценщик, набор порога, обучение с посторонними. */
    record Method(String id, String stage, IlluminationNorm norm, GalleryScorer scorer, Threshold threshold, boolean outsiders) {
        String label() {
            return String.format(Locale.ROOT, "%s — %s; %s; порог — %s%s", id, scorer.label(),
                    norm == null ? "вход SFace alignCrop 112×112" : norm.label, threshold.label,
                    outsiders ? "; обучение — галерея + 69 посторонних" : "");
        }
    }

    // ---------------------------------------------------------------- данные

    final SettingsStore settings;
    final Data data;
    final Splits splits;
    final List<String> outsiders;
    final List<String> thresholdSet;
    /** Чужие FEI (контроль) — нет в экспорте: null. */
    final Map<String, List<Sample>> fei;
    final IlluminationNorm.NormParams params;
    /** Кадры а (серый 8 бит 92×112) по ключу снимка; отказ детектора — нет ключа. */
    final Map<String, Mat> frames;
    /** Векторы по способу нормализации кадра а; ключ null — признаки SFace. */
    private final Map<IlluminationNorm, Map<String, double[]>> vectors = new HashMap<>();
    private final Map<String, Result> results = new LinkedHashMap<>();
    /** Оценки конфигураций (для пересчёта порога по другому набору чужих). */
    private final Map<String, Scores> scores = new HashMap<>();
    private final Map<String, Supplier<Method>> registry = new LinkedHashMap<>();
    final StringBuilder timeText = new StringBuilder();
    /** Источник баз для шапки отчёта (экспорт — с коммитом репозитория данных). */
    String dataSource = "";
    /** Заметки о выборе лучшего (в отчёт). */
    final List<String> choices = new ArrayList<>();

    FarMethods(SettingsStore settings, Data data, Splits splits, Map<String, Mat> frames, Map<String, List<Sample>> fei) {
        this.settings = settings;
        this.data = data;
        this.splits = splits;
        this.frames = frames;
        this.fei = fei;
        this.params = IlluminationNorm.NormParams.of(settings);
        List<String> val = new ArrayList<>(splits.muctVal());
        Collections.shuffle(val, new Random(settings.loadFacesSeed() + OUTSIDER_SEED_SHIFT));
        outsiders = List.copyOf(val.subList(0, val.size() / 2));
        thresholdSet = List.copyOf(val.subList(val.size() / 2, val.size()));
        // Посторонние и пороговые — только валидация MUCT: не пересекаются друг с другом и с контролем MUCT.
        if (!java.util.Collections.disjoint(outsiders, thresholdSet) || !java.util.Collections.disjoint(outsiders, splits.muctCtrl())
                || !java.util.Collections.disjoint(thresholdSet, splits.muctCtrl())) {
            throw new IllegalStateException("Посторонние / пороговые пересекаются между собой или с контролем MUCT");
        }
        register();
    }

    /** Реестр конфигураций в порядке этапов; зависимые (лучший предыдущего этапа) создаются при запросе. */
    private void register() {
        registry.put("0-sface", () -> new Method("0-sface", "справочно", null, new SFaceScorer(), Threshold.VAL138, false));
        registry.put("1a-nn", () -> new Method("1a-nn", "1а", IlluminationNorm.NONE, new NearestVectorScorer(false), Threshold.VAL138, false));
        registry.put("1a-mean", () -> new Method("1a-mean", "1а", IlluminationNorm.NONE, new NearestVectorScorer(true), Threshold.VAL138, false));
        for (IlluminationNorm n : IlluminationNorm.values()) {
            for (boolean ratio : new boolean[] {true, false}) {
                String id = "1-" + n.id + (ratio ? "-ratio" : "-eps");
                registry.put(id, () -> new Method(id, "1", n, new SubspaceScorer(ratio), Threshold.VAL138, false));
            }
        }
        // Этап 2: предобработка — лучшая из этапа 1; порог — 69 пороговых; рядом — лучший SVD этапа 1 с тем же порогом.
        registry.put("2a-fisher", () -> {
            rethreshold(bestSvd(), Threshold.VAL69);
            return new Method("2a-fisher", "2", bestNorm(), new FisherScorer(), Threshold.VAL69, false);
        });
        registry.put("2b-fisher-bg", () -> new Method("2b-fisher-bg", "2", bestNorm(), new FisherScorer(), Threshold.VAL69, true));
        registry.put("2c-mlda", () -> new Method("2c-mlda", "2", bestNorm(), new MldaScorer(), Threshold.VAL69, true));
    }

    /** Проверка эквивалентности MLDA (не конфигурация оценки). */
    static final String CHECK = "2c-check";
    /** Итог проверки эквивалентности MLDA (не выполнялась — null). */
    MldaCheck.Outcome check;

    /** MLDA идёт в сравнение (выбор лучшего, Z-norm) только после прошедшей проверки эквивалентности. */
    boolean mldaVerified() {
        return check != null && check.passed();
    }

    /** Строка итога проверки эквивалентности MLDA для отчёта. */
    String checkLine() {
        if (check == null) {
            return "Проверка эквивалентности MLDA: НЕ выполнялась — результаты MLDA (2c-mlda) в сравнение не идут.";
        }
        String line = String.format(Locale.ROOT, "Проверка эквивалентности MLDA: %s — наибольший синус главного угла %.3e, наибольшее "
                + "относительное расхождение расстояний %.3e, допуск %.0e.", check.passed() ? "прошла" : "НЕ прошла",
                check.maxSin(), check.maxRel(), MldaCheck.TOLERANCE);
        return check.passed() ? line : line + " Результаты MLDA (2c-mlda) в сравнение не идут, пока не разберёмся.";
    }
    private String bestSvd;

    /** Лучший SVD этапа 1 (один выбор на прогон). */
    String bestSvd() {
        if (bestSvd == null) bestSvd = best("SVD этапа 1: предобработка этапа 2 и основа Z-norm", stage1Ids());
        return bestSvd;
    }

    IlluminationNorm bestNorm() {
        return result(bestSvd()).method.norm();
    }

    /** Те же оценки конфигурации id, порог — по другому набору чужих (результат «id@набор»). */
    Result rethreshold(String id, Threshold th) {
        Result base = result(id);
        if (base.threshold == th) return base;
        String key = id + "@" + th;
        Result r = results.get(key);
        if (r == null) {
            r = metrics(base.method, scores.get(id), th);
            results.put(key, r);
        }
        return r;
    }

    /** Идентификаторы конфигураций SVD этапа 1. */
    List<String> stage1Ids() {
        return registry.keySet().stream().filter(id -> id.startsWith("1-")).toList();
    }

    // ---------------------------------------------------------------- запуск

    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        long start = System.nanoTime();
        SettingsStore settings = new SettingsStore();
        long seed = settings.loadFacesSeed();
        String exportDir = settings.loadFacesExportDir();
        FarExport.Loaded export = exportDir == null ? null : FarExport.read(Paths.get(exportDir));
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "far");
        Files.createDirectories(outDir);
        // Сырые снимки — с детекциями faces-far (и отбором FEI при заданном faces.fei.dir); экспорт — FEI, если она в нём есть.
        GalleryEvaluation.Raw raw = export == null ? GalleryEvaluation.raw(settings, settings.loadFacesOwnDetectorScore(),
                Paths.get(settings.loadFacesModelsDir()), outDir, args.length > 0 && args[0].equals("fresh")) : null;
        Data data = export == null ? raw.data() : export.data();
        Splits splits = GalleryEvaluation.splits(data.gt(), data.muct(), seed);
        if (export != null) export.checkSplits(splits);

        // Кадры а: из PNG экспорта или из сырых снимков по детекциям.
        long t0 = System.nanoTime();
        Map<String, Mat> frames = new HashMap<>();
        Map<String, Det> dets = export == null ? raw.dets() : null;
        // SFace (справочная строка 0-sface): вход alignCrop 112×112 — признаки один раз при загрузке.
        FaceRecognizerSF sface = FaceRecognizerSF.create(Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE)
                .toString(), "");
        Map<String, double[]> sfaceVectors = new HashMap<>();
        for (Sample s : data.samples()) {
            String key = s.file().toString();
            Mat f;
            Mat c;
            if (export != null) {
                f = export.frame(s, Variant.A);
                c = export.frame(s, Variant.SFACE);
            } else {
                double[] row = dets.get(key).row();
                if (row == null) {
                    f = null;
                    c = null;
                } else {
                    Mat[] m = GalleryEvaluation.load(s);
                    f = GalleryEvaluation.frame(s, m, row, Variant.A, null);
                    c = GalleryEvaluation.frame(s, m, row, Variant.SFACE, sface);
                    GalleryEvaluation.release(m);
                }
            }
            if (f != null) frames.put(key, f);
            if (c != null) {
                sfaceVectors.put(key, GalleryEvaluation.vector(c, Variant.SFACE, sface));
                c.release();
            }
        }
        FarMethods fm = new FarMethods(settings, data, splits, frames, data.fei().isEmpty() ? null : data.fei());
        fm.vectors.put(null, sfaceVectors);
        fm.dataSource = export == null ? "сырые снимки" : "экспорт " + exportDir + ", коммит данных " + dataCommit(Paths.get(exportDir));
        fm.timeText.append(String.format(Locale.ROOT, "Кадры а (%s): %.0f с, снимков %d, с лицом %d.%n",
                export == null ? "сырые снимки" : "экспорт " + exportDir, (System.nanoTime() - t0) / 1e9, data.samples().size(),
                frames.size()));

        List<String> ids = settings.loadFacesFarMethods();
        if (ids.isEmpty()) {
            ids = new ArrayList<>(fm.registry.keySet());
            ids.add(ids.indexOf("2c-mlda") + 1, CHECK);
        }
        for (String id : ids) {
            if (id.equals(CHECK)) {
                long t1 = System.nanoTime();
                try {
                    fm.check = MldaCheck.run(fm, fm.bestNorm());
                } catch (RuntimeException e) {
                    fm.check = new MldaCheck.Outcome("Проверка прервана ошибкой: " + e + "\n", false, Double.NaN, Double.NaN);
                }
                fm.timeText.append(String.format(Locale.ROOT, "%s: %.0f с.%n", CHECK, (System.nanoTime() - t1) / 1e9));
            } else {
                fm.result(id);
            }
        }

        write(outDir.resolve("far_methods.txt"), fm.report(fm.dataSource));
        write(outDir.resolve("far_methods_auc.txt"), fm.aucReport());
        fm.timeText.append(String.format(Locale.ROOT, "Всего %.0f с.%n", (System.nanoTime() - start) / 1e9));
        write(outDir.resolve("far_methods_time.txt"), fm.timeText);
        System.out.print(fm.timeText);
    }

    static void write(Path file, CharSequence text) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.print(text);
        }
    }

    /** Результат конфигурации (считается один раз). */
    Result result(String id) {
        Result r = results.get(id);
        if (r != null) return r;
        Supplier<Method> m = registry.get(id);
        if (m == null) throw new IllegalArgumentException("Неизвестная конфигурация faces.far.methods: " + id + "; есть " + registry.keySet());
        Method method = m.get();
        long t0 = System.nanoTime();
        Scores sc = score(method);
        scores.put(id, sc);
        r = metrics(method, sc, method.threshold());
        results.put(id, r);
        timeText.append(String.format(Locale.ROOT, "%s: %.0f с.%n", id, (System.nanoTime() - t0) / 1e9));
        return r;
    }

    // ---------------------------------------------------------------- векторы

    /** Векторы всех снимков с лицом для способа нормализации (параллельно; от порядка не зависит). */
    Map<String, double[]> vectors(IlluminationNorm norm) {
        return vectors.computeIfAbsent(norm, n -> {
            List<String> keys = new ArrayList<>(frames.keySet());
            double[][] out = new double[keys.size()][];
            java.util.stream.IntStream.range(0, keys.size()).parallel().forEach(i -> out[i] = n.vector(frames.get(keys.get(i)), params));
            Map<String, double[]> map = new HashMap<>();
            for (int i = 0; i < keys.size(); i++) map.put(keys.get(i), out[i]);
            return map;
        });
    }

    // ---------------------------------------------------------------- оценки

    /** Оценки по людям галереи для всех снимков с лицом, по конфигурациям; сведения моделей. */
    record Scores(List<Map<String, double[]>> byConfig, List<String> info) {}

    /** Обучающие векторы классов конфигурации r: 6 своих, 50 GT, (outsiders) 69 посторонних. */
    List<List<double[]>> classes(int r, Map<String, double[]> vec, boolean withOutsiders) {
        List<List<double[]>> classes = new ArrayList<>();
        List<List<OwnDataset.Moment>> moments = data.moments();
        for (List<OwnDataset.Moment> ms : moments) {
            OwnEvaluation.Split split = OwnEvaluation.split(ms, r % ms.size(), false);
            List<double[]> train = new ArrayList<>();
            for (int j : split.train()) {
                for (OwnDataset.Frame f : ms.get(j).frames()) {
                    if (f.quality() < OwnDataset.FAIR) continue;
                    double[] x = vec.get(f.file().toString());
                    if (x != null) train.add(x);
                }
            }
            classes.add(train);
        }
        int split = r % GT_SPLITS;
        for (int g = 0; g < splits.gtPersons().size(); g++) {
            List<Sample> imgs = data.gt().get(splits.gtPersons().get(g));
            List<double[]> train = new ArrayList<>();
            for (int i : splits.gtTrain().get(split).get(g)) {
                double[] x = vec.get(imgs.get(i).file().toString());
                if (x != null) train.add(x);
            }
            classes.add(train);
        }
        if (withOutsiders) {
            for (String p : outsiders) {
                List<double[]> list = new ArrayList<>();
                for (Sample s : data.muct().get(p)) {
                    double[] x = vec.get(s.file().toString());
                    if (x != null) list.add(x);
                }
                classes.add(list);
            }
        }
        return classes;
    }

    int gallerySize() {
        return data.moments().size() + splits.gtPersons().size();
    }

    Scores score(Method m) {
        Map<String, double[]> vec = vectors(m.norm());
        List<String> keys = new ArrayList<>(vec.keySet());
        List<Map<String, double[]>> byConfig = new ArrayList<>();
        List<String> info = new ArrayList<>();
        for (int r = 0; r < CONFIGS; r++) {
            System.out.printf(Locale.ROOT, "%s: конфигурация %d/%d%n", m.id(), r + 1, CONFIGS);
            GalleryScorer.ScoreModel model = m.scorer().fit(classes(r, vec, m.outsiders()), gallerySize());
            info.add(model.info());
            double[][] out = new double[keys.size()][];
            java.util.stream.IntStream.range(0, keys.size()).parallel().forEach(i -> out[i] = model.scores(vec.get(keys.get(i))));
            Map<String, double[]> map = new HashMap<>();
            for (int i = 0; i < keys.size(); i++) map.put(keys.get(i), out[i]);
            byConfig.add(map);
        }
        return new Scores(byConfig, info);
    }

    /** {минимальная оценка, argmin} по оценкам людей галереи. */
    static double[] best(double[] s) {
        int b = 0;
        for (int p = 1; p < s.length; p++) if (s[p] < s[b]) b = p;
        return new double[] {s[b], b};
    }

    /** Попытки чужих: оценка (минимум по галерее) по людям; отказ детектора — NaN. */
    static Impostors impostors(Map<String, List<Sample>> base, List<String> persons, Map<String, double[]> scores, boolean cameraA) {
        Map<String, double[]> map = new LinkedHashMap<>();
        for (String p : persons) {
            List<Sample> list = base.get(p).stream().filter(s -> !cameraA || s.camera() == 'a').toList();
            double[] vals = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                double[] s = scores.get(list.get(i).file().toString());
                vals[i] = s == null ? Double.NaN : best(s)[0];
            }
            map.put(p, vals);
        }
        return new Impostors(map);
    }

    // ---------------------------------------------------------------- метрики

    /** Метрики конфигурации на контроле (как GalleryEvaluation.evaluate) и данные диагностики. */
    static final class Result {
        Method method;
        Threshold threshold;
        List<String> info;
        int ownCorrect, ownAtt, ownDetRej, gtCorrect, gtAtt, gtDetRej;
        int[] ownRej = new int[ALPHAS.length];
        int[] gtRej = new int[ALPHAS.length];
        /** [строка: MUCT, MUCT камера a, ORL, FEI][α][конфигурация] → {x, n, xP, nP}. */
        int[][][][] far = new int[4][ALPHAS.length][CONFIGS][];
        double[][] thetas = new double[ALPHAS.length][CONFIGS];
        List<Double> marginOwn = new ArrayList<>();
        List<Double> marginGt = new ArrayList<>();
        int[] refused = new int[4];
        /** Диагностика по людям галереи: оценки своих проб и чужих валидации MUCT (138) против этого человека. */
        List<List<Double>> genuine = new ArrayList<>();
        List<List<Double>> impostor = new ArrayList<>();

        int frr(int a) {
            return ownRej[a] + gtRej[a];
        }

        int att() {
            return ownAtt + gtAtt;
        }
    }

    Result metrics(Method m, Scores sc, Threshold th) {
        Result res = new Result();
        res.method = m;
        res.threshold = th;
        res.info = sc.info();
        int nOwn = data.moments().size();
        int nGal = gallerySize();
        for (int i = 0; i < nGal; i++) {
            res.genuine.add(new ArrayList<>());
            res.impostor.add(new ArrayList<>());
        }
        List<String> thPersons = th == Threshold.VAL138 ? splits.muctVal() : thresholdSet;
        for (int r = 0; r < CONFIGS; r++) {
            Map<String, double[]> s = sc.byConfig().get(r);
            Impostors val = impostors(data.muct(), thPersons, s, false);
            Impostors valA = impostors(data.muct(), thPersons, s, true);
            Impostors ctrl = impostors(data.muct(), splits.muctCtrl(), s, false);
            Impostors ctrlA = impostors(data.muct(), splits.muctCtrl(), s, true);
            Impostors orl = impostors(data.orl(), new ArrayList<>(data.orl().keySet()), s, false);
            Impostors feiCtrl = fei == null ? null : impostors(fei, new ArrayList<>(fei.keySet()), s, false);
            if (r == 0) {
                res.refused[0] = val.refused();
                res.refused[1] = ctrl.refused();
                res.refused[2] = orl.refused();
                res.refused[3] = feiCtrl == null ? 0 : feiCtrl.refused();
            }
            double minImp = Arrays.stream(ctrl.detected()).min().orElse(Double.NaN);
            // Диагностика: чужие — валидация MUCT (138) против каждого человека галереи.
            List<double[]> valScores = new ArrayList<>();
            for (String p : splits.muctVal()) {
                for (Sample x : data.muct().get(p)) {
                    double[] v = s.get(x.file().toString());
                    if (v != null) valScores.add(v);
                }
            }
            // Свои: контроль — представитель момента r (решение — как в GalleryEvaluation: оценка argmin).
            double maxOwn = Double.NEGATIVE_INFINITY;
            List<Double> ownScores = new ArrayList<>();
            for (int p = 0; p < nOwn; p++) {
                List<OwnDataset.Moment> ms = data.moments().get(p);
                if (r >= ms.size()) continue;
                res.ownAtt++;
                double[] v = s.get(ms.get(r).representative().file().toString());
                if (v == null) {
                    res.ownDetRej++;
                    ownScores.add(Double.POSITIVE_INFINITY);
                    continue;
                }
                double[] b = best(v);
                if ((int) b[1] == p) res.ownCorrect++;
                ownScores.add(Double.isInfinite(v[p]) ? Double.POSITIVE_INFINITY : b[0]);
                maxOwn = Math.max(maxOwn, b[0]);
                diag(res, p, v, valScores);
            }
            List<Double> gtScores = new ArrayList<>();
            double maxGt = Double.NEGATIVE_INFINITY;
            if (r < GT_SPLITS) {
                for (int g = 0; g < splits.gtPersons().size(); g++) {
                    List<Sample> imgs = data.gt().get(splits.gtPersons().get(g));
                    List<Integer> train = splits.gtTrain().get(r).get(g);
                    boolean any = false;
                    for (int i = 0; i < imgs.size(); i++) {
                        if (train.contains(i)) continue;
                        res.gtAtt++;
                        double[] v = s.get(imgs.get(i).file().toString());
                        if (v == null) {
                            res.gtDetRej++;
                            gtScores.add(Double.POSITIVE_INFINITY);
                            continue;
                        }
                        double[] b = best(v);
                        if ((int) b[1] == nOwn + g) res.gtCorrect++;
                        gtScores.add(Double.isInfinite(v[nOwn + g]) ? Double.POSITIVE_INFINITY : b[0]);
                        maxGt = Math.max(maxGt, b[0]);
                        res.genuine.get(nOwn + g).add(v[nOwn + g]);
                        any = true;
                    }
                    if (any) for (double[] v : valScores) res.impostor.get(nOwn + g).add(v[nOwn + g]);
                }
                res.marginGt.add(minImp - maxGt);
            }
            res.marginOwn.add(minImp - maxOwn);
            double[] valDet = val.detected();
            double[] valADet = valA.detected();
            for (int a = 0; a < ALPHAS.length; a++) {
                double theta = FaceEvaluation.neymanPearsonThreshold(valDet, ALPHAS[a]);
                double thetaA = FaceEvaluation.neymanPearsonThreshold(valADet, ALPHAS[a]);
                res.thetas[a][r] = theta;
                for (double v : ownScores) if (v > theta) res.ownRej[a]++;
                for (double v : gtScores) if (v > theta) res.gtRej[a]++;
                res.far[0][a][r] = GalleryEvaluation.far(ctrl, theta);
                res.far[1][a][r] = GalleryEvaluation.far(ctrlA, thetaA);
                res.far[2][a][r] = GalleryEvaluation.far(orl, theta);
                res.far[3][a][r] = feiCtrl == null ? null : GalleryEvaluation.far(feiCtrl, theta);
            }
        }
        return res;
    }

    private static void diag(Result res, int p, double[] v, List<double[]> valScores) {
        res.genuine.get(p).add(v[p]);
        for (double[] w : valScores) res.impostor.get(p).add(w[p]);
    }

    // ---------------------------------------------------------------- выбор лучшего

    /**
     * Лучшая конфигурация: наименьший общий FRR (свои + Georgia Tech) на контрольных пробах при пороге с
     * валидации, α = 0,05; при равенстве — α = 0. FAR на контроле (MUCT, ORL, FEI) в выборе не участвует.
     */
    String best(String what, List<String> candidates) {
        String best = null;
        StringBuilder note = new StringBuilder("Выбор лучшего (" + what + "): по общему FRR своих и Georgia Tech на контроле "
                + "при пороге с валидации, α = 0,05, при равенстве — α = 0; FAR на контроле в выборе не участвует. Кандидаты:");
        for (String id : candidates) {
            Result r = result(id);
            note.append(String.format(Locale.ROOT, " %s — %d/%d, %d/%d;", id, r.frr(A05), r.att(), r.frr(A0), r.att()));
            if (best == null || better(r, result(best))) best = id;
        }
        note.append(" выбран ").append(best).append('.');
        choices.add(note.toString());
        return best;
    }

    private static boolean better(Result a, Result b) {
        if (a.frr(A05) != b.frr(A05)) return a.frr(A05) < b.frr(A05);
        return a.frr(A0) < b.frr(A0);
    }

    // ---------------------------------------------------------------- отчёт

    StringBuilder report(String source) {
        StringBuilder t = new StringBuilder();
        t.append("Шаг 5, SVD: улучшение отсечения чужих (FarMethods)\n");
        t.append("Код: коммит ").append(commit()).append('\n');
        t.append("Протокол — как в отчёте 32dc1fa (far_eval.txt): галерея 56 (6 своих + 50 Georgia Tech), 12 конфигураций, вход 2′,\n"
                + "кадр а) 92×112 (пиксели как в own_eval; варианты а/б/в в 32dc1fa не различались). Базы: " + source + ".\n");
        t.append(String.format(Locale.ROOT, "Чужие: MUCT валидация %d человек (порог), контроль %d человек (FAR); валидация делится по людям "
                + "(seed %d + %d) на %d посторонних (обучение LDA) и %d пороговых (порог этапов 2–3, когорта Z-norm). ORL — контроль.%n",
                splits.muctVal().size(), splits.muctCtrl().size(), settings.loadFacesSeed(), OUTSIDER_SEED_SHIFT, outsiders.size(),
                thresholdSet.size()));
        t.append(fei == null ? "FEI: нет в экспорте (строка добавится прогоном после подключения).\n"
                : String.format(Locale.ROOT, "FEI: %d человек — только контроль (порог MUCT).%n", fei.size()));
        t.append("Контрольные 138 MUCT (и FEI) не участвуют в обучении, пороге, когорте и выборе лучшего.\n");
        t.append("Обучение и порог по методам (разбиения своих — own_moments.tsv, GT и MUCT — splits.tsv экспорта):\n"
                + "  - все методы: галерея — свои 6 (обучающие моменты ротации r, кадры «+» и «+-») и Georgia Tech 50 × 5 (разбиение\n"
                + "    r mod 3); контроль своих и GT в обучение не входит;\n"
                + "  - 0-sface (справочно), 1а, 1: обучение — только галерея; порог — валидация MUCT 138 человек;\n"
                + "  - 2a-fisher: обучение — только галерея; порог — 69 пороговых;\n"
                + "  - 2b-fisher-bg, 2c-mlda: обучение — галерея + 69 посторонних MUCT (все снимки с лицом, все камеры); порог — 69\n"
                + "    пороговых;\n"
                + "  - нигде не участвуют: контроль MUCT (138 человек), ORL (40), FEI.\n");
        t.append("  69 посторонних: ").append(String.join(", ", outsiders)).append('\n');
        t.append("  69 пороговых: ").append(String.join(", ", thresholdSet)).append('\n');
        t.append("Оговорки 32dc1fa в силе: своя база — одна сессия, у Georgia Tech нет сессий (FRR оптимистичен); чужие сняты в\n"
                + "лаборатории, свои — телефоном; FAR по попыткам — граница внутри конфигурации (оптимистична).\n");
        t.append(String.format(Locale.ROOT, "Нормализация освещения: CLAHE clip %.2f, сетка %d×%d; Tan – Triggs γ %.2f, σ₀ %.2f, σ₁ %.2f, α %.2f, τ %.2f.%n",
                params.claheClip(), params.claheTile(), params.claheTile(), params.gamma(), params.sigma0(), params.sigma1(),
                params.alpha(), params.tau()));
        t.append("Решение: argmin по галерее; принят, если оценка argmin ≤ θ (θ — Нейман – Пирсон по чужим набора порога, в каждой\n"
                + "конфигурации). FRR — свои и Georgia Tech вместе; FAR по людям — худшая и медианная конфигурация, ↑95 —\n"
                + "верхняя граница Клоппера – Пирсона худшей.\n");
        for (String c : choices) t.append(c).append('\n');

        t.append("\n=== Сводка ===\n");
        if (results.containsKey("2c-mlda") || check != null) t.append(checkLine()).append('\n');
        t.append("конфигурация | argmin свои | argmin GT | FRR α=0,05 | FRR α=0 | FAR MUCT люди α=0,05: худш. (↑95) / мед. | "
                + "то же α=0 | ORL α=0,05 | FEI α=0,05 | запас свои / GT (мед.)\n");
        for (Result r : results.values()) t.append(summaryRow(r)).append('\n');
        if (check != null) t.append("\n=== ").append(CHECK).append(" ===\n").append(checkLine()).append('\n').append(check.text());

        for (Result r : results.values()) details(t, r);
        return t;
    }

    /** Коммит, на котором сделан прогон (git rev-parse HEAD; отметка, если отслеживаемые файлы изменены). */
    static String commit() {
        try {
            String head = git("rev-parse", "HEAD");
            String dirty = git("status", "--porcelain", "--untracked-files=no");
            return head + (dirty.isEmpty() ? "" : " (есть незакоммиченные изменения отслеживаемых файлов)");
        } catch (IOException | InterruptedException e) {
            return "не определён (" + e.getMessage() + ")";
        }
    }

    /** Коммит репозитория данных (каталог экспорта). */
    static String dataCommit(Path dir) {
        try {
            return git("-C", dir.toString(), "rev-parse", "HEAD");
        } catch (IOException | InterruptedException e) {
            return "не определён (" + e.getMessage() + ")";
        }
    }

    private static String git(String... args) throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>(List.of("git"));
        cmd.addAll(List.of(args));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        if (p.waitFor() != 0) throw new IOException(out);
        return out;
    }

    String summaryRow(Result r) {
        return String.format(Locale.ROOT, "%s%s | %d/%d | %d/%d | %d/%d = %s %% | %d/%d = %s %% | %s | %s | %s | %s | %s / %s",
                r.method.id() + (r.method.scorer() instanceof MldaScorer && !mldaVerified() ? " [не в сравнении: проверка MLDA]" : ""),
                r.threshold != r.method.threshold() ? " (порог: " + r.threshold.label + ")" : "",
                r.ownCorrect, r.ownAtt, r.gtCorrect, r.gtAtt,
                r.frr(A05), r.att(), GalleryEvaluation.pct(r.frr(A05), r.att()),
                r.frr(A0), r.att(), GalleryEvaluation.pct(r.frr(A0), r.att()),
                persons(r.far[0][A05]), persons(r.far[0][A0]), persons(r.far[2][A05]),
                r.far[3][A05][0] == null ? "—" : persons(r.far[3][A05]),
                GalleryEvaluation.q(r.marginOwn, 0.5), GalleryEvaluation.q(r.marginGt, 0.5));
    }

    /** FAR по людям: худшая конфигурация (↑95) / медианная. */
    static String persons(int[][] cfg) {
        Integer[] idx = new Integer[cfg.length];
        for (int i = 0; i < cfg.length; i++) idx[i] = i;
        Arrays.sort(idx, (x, y) -> Double.compare(cfg[x][2] / (double) cfg[x][3], cfg[y][2] / (double) cfg[y][3]));
        int[] w = cfg[idx[cfg.length - 1]];
        int[] m = cfg[idx[cfg.length / 2]];
        return String.format(Locale.ROOT, "%d/%d (%s %%) / %d/%d", w[2], w[3], GalleryEvaluation.ub(w[2], w[3]), m[2], m[3]);
    }

    void details(StringBuilder t, Result r) {
        t.append(String.format(Locale.ROOT, "%n=== %s ===%n", r.method.label()));
        if (r.threshold != r.method.threshold()) t.append("Порог пересчитан: ").append(r.threshold.label).append('\n');
        if (!r.info.get(0).isEmpty()) {
            t.append("Модель по конфигурациям:\n");
            for (int i = 0; i < r.info.size(); i++) t.append(String.format(Locale.ROOT, "  %2d: %s%n", i, r.info.get(i)));
        }
        t.append(String.format(Locale.ROOT, "argmin (галерея из %d): свои %d/%d, Georgia Tech %d/%d; отказы детектора на контроле: свои %d/%d, "
                + "Georgia Tech %d/%d; чужие: MUCT валидация %d, MUCT контроль %d, ORL %d попыток%s.%n",
                gallerySize(), r.ownCorrect, r.ownAtt, r.gtCorrect, r.gtAtt, r.ownDetRej, r.ownAtt, r.gtDetRej, r.gtAtt,
                r.refused[0], r.refused[1], r.refused[2], fei == null ? "" : ", FEI " + r.refused[3]));
        t.append(String.format(Locale.ROOT, "Запас на контроле (наименьшая оценка чужого MUCT минус наибольшая своего): свои — мин %s / медиана %s "
                + "(по 12 конфигурациям); Georgia Tech — мин %s / медиана %s (конфигурации 0–2).%n",
                GalleryEvaluation.q(r.marginOwn, 0), GalleryEvaluation.q(r.marginOwn, 0.5),
                GalleryEvaluation.q(r.marginGt, 0), GalleryEvaluation.q(r.marginGt, 0.5)));
        String[] rows = {"MUCT, все камеры", "MUCT, камера a (порог по набору порога, камера a)", "ORL (порог MUCT все камеры)",
                "FEI (порог MUCT все камеры)"};
        for (int a = 0; a < ALPHAS.length; a++) {
            double[] th = r.thetas[a].clone();
            Arrays.sort(th);
            t.append(String.format(Locale.ROOT, "  α = %.3f (θ медиана %.6f): FRR свои %d/%d = %s %% (без отказов детектора %d/%d), "
                    + "Georgia Tech %d/%d = %s %% (без отказов %d/%d), общий %d/%d = %s %%%n",
                    ALPHAS[a], th[th.length / 2], r.ownRej[a], r.ownAtt, GalleryEvaluation.pct(r.ownRej[a], r.ownAtt),
                    r.ownRej[a] - r.ownDetRej, r.ownAtt - r.ownDetRej, r.gtRej[a], r.gtAtt, GalleryEvaluation.pct(r.gtRej[a], r.gtAtt),
                    r.gtRej[a] - r.gtDetRej, r.gtAtt - r.gtDetRej, r.frr(a), r.att(), GalleryEvaluation.pct(r.frr(a), r.att())));
            for (int k = 0; k < 4; k++) {
                if (k == 3 && fei == null) continue;
                t.append("    FAR ").append(rows[k]).append(": ").append(GalleryEvaluation.farText(r.far[k][a])).append('\n');
            }
        }
    }

    // ---------------------------------------------------------------- диагностика

    StringBuilder aucReport() {
        StringBuilder t = new StringBuilder();
        t.append("Диагностика по людям галереи (шаг 5, FarMethods): оценка пробы против человека i (не только argmin).\n");
        t.append("Свои — контрольные пробы человека i по всем конфигурациям, где они есть (свои — 12 ротаций, Georgia Tech —\n"
                + "конфигурации 0–2); чужие — все снимки валидации MUCT (138 человек) против человека i в тех же конфигурациях.\n");
        t.append("AUC = P(оценка чужого > оценки своего) (Манн – Уитни, ничьи — 0,5). «Хвост»: доля своих выше 5 % квантиля\n"
                + "чужих и выше минимума чужих. Если AUC близка к 1, а своих выше минимума чужих много — мешают хвосты чужих;\n"
                + "если AUC далека от 1 — распределения перекрываются целиком.\n");
        List<String> names = new ArrayList<>();
        for (OwnDataset.Person p : data.own().persons()) names.add("своя " + p.name());
        for (String g : splits.gtPersons()) names.add("GT " + g);
        for (Result r : results.values()) {
            t.append(String.format(Locale.ROOT, "%n=== %s ===%n", r.method.label()));
            t.append(ScoreDiagnostics.perPerson(names, r.genuine, r.impostor));
        }
        return t;
    }
}
