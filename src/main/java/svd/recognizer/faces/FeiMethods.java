package svd.recognizer.faces;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
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
import java.util.stream.IntStream;
import java.util.zip.GZIPOutputStream;
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
    record Base(String id, String about, IlluminationNorm norm, GalleryScorer scorer, boolean cohortInTraining) {}

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

    final List<Base> bases = new ArrayList<>();
    final List<Derived> derived = new ArrayList<>();
    /** Все строки в порядке отчёта; подпись строки. */
    final Map<String, String> about = new LinkedHashMap<>();
    final Map<String, Result> results = new LinkedHashMap<>();
    final StringBuilder timeText = new StringBuilder();
    /** Файл оценок попыток (этап 5б); null — не пишется. */
    Writer scoresOut;

    FeiMethods(SettingsStore settings, Map<String, Map<Integer, String>> keys, Map<String, Mat> frames, FeiDataset.Info info) {
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
        register();
    }

    /** Методы этапа 4 (параметры те же) и 4-wpca; порядок — порядок таблицы. */
    private void register() {
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
    }

    static String wpcaId(String kind, double c) {
        return "4-wpca-" + kind + "-" + WpcaScorer.fmt(c);
    }

    static String blkWpcaId(String wpca) {
        return "4-sum-blk-both+" + wpca.substring(2);
    }

    List<String> wpcaIds() {
        return about.keySet().stream().filter(id -> id.startsWith("4-wpca-")).toList();
    }

    private void base(String id, String text, IlluminationNorm norm, GalleryScorer scorer, boolean cohort) {
        bases.add(new Base(id, text, norm, scorer, cohort));
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
        Map<String, Map<Integer, String>> keys = selection(Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8),
                export.data().fei());
        Map<String, Mat> frames = new HashMap<>();
        for (List<Sample> list : export.data().fei().values()) {
            for (Sample s : list) {
                Mat f = export.frame(s, Variant.A);
                if (f == null) throw new IllegalStateException("Нет кадра отобранного снимка FEI: " + s.file());
                frames.put(s.file().toString(), f);
            }
        }
        int limit = args.length > 0 && !args[0].isBlank() ? Integer.parseInt(args[0].trim()) : CONFIGS;
        FeiMethods fm = new FeiMethods(settings, keys, frames, export.feiInfo());
        frames.values().forEach(Mat::release);
        fm.timeText.append(String.format(Locale.ROOT, "Загрузка и векторы: %.0f с, снимков FEI %d.%n", (System.nanoTime() - start) / 1e9,
                frames.size()));
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(outDir);
        String source = "экспорт " + exportDir + ", коммит данных " + FarMethods.dataCommit(dir);
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(outDir.resolve(FeiScores.FILE)), 1 << 16),
                StandardCharsets.UTF_8)) {
            w.write(FeiScores.header(FarMethods.commit(), source, Math.min(limit, CONFIGS), new ArrayList<>(fm.about.keySet())));
            fm.scoresOut = w;
            fm.run(limit);
            fm.scoresOut = null;
        }
        FarMethods.write(outDir.resolve("fei_methods.txt"), fm.report(source, limit));
        fm.timeText.append(String.format(Locale.ROOT, "Всего %.0f с.%n", (System.nanoTime() - start) / 1e9));
        FarMethods.write(outDir.resolve("fei_time.txt"), fm.timeText);
        System.out.print(fm.timeText);
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
        Map<String, double[]> vec = vectors.get(norm);
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

    /** Все конфигурации (limit — первые limit, для отладки). */
    void run(int limit) {
        for (String id : about.keySet()) results.put(id, new Result(id));
        for (int c = 0; c < Math.min(limit, CONFIGS); c++) {
            FeiProtocol.Split sp = splits.get(c / FeiProtocol.NUMBERS.length);
            int j = c % FeiProtocol.NUMBERS.length;
            long t0 = System.nanoTime();
            Map<String, Map<String, double[]>> sc = scoreConfig(sp, j);
            for (String id : about.keySet()) results.get(id).add(this, sp, j, sc.get(id));
            writeScores(sp, j, sc);
            timeText.append(String.format(Locale.ROOT, "Конфигурация %d (разбиение %d, контроль №%d): %.0f с.%n", c, sp.index(),
                    FeiProtocol.NUMBERS[j], (System.nanoTime() - t0) / 1e9));
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
            List<List<double[]>> cls = classes(sp, j, b.norm(), b.cohortInTraining());
            GalleryScorer.ScoreModel model;
            if (b.scorer() instanceof WpcaScorer) {
                // Одна PCA на конфигурацию для всех строк wpca.
                if (pca == null) pca = WpcaScorer.Pca.of(cls);
                WpcaScorer w = (WpcaScorer) b.scorer();
                model = WpcaScorer.model(pca, cls, g, w.mode(), w.c());
            } else {
                model = b.scorer().fit(cls, g);
            }
            Map<String, double[]> vec = vectors.get(b.norm());
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

    /** Оценки попыток конфигурации в файл: по методам (порядок отчёта) — свои, пороговые, контрольные чужие. */
    void writeScores(FeiProtocol.Split sp, int j, Map<String, Map<String, double[]>> sc) {
        if (scoresOut == null) return;
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
        try {
            scoresOut.write(b.toString());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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
    static final class Result {
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
        List<String> ids = new ArrayList<>(about.keySet().stream().filter(id -> !id.startsWith("4-sum-blk-both+")).toList());
        if (wpcaBeats()) ids.add(blkWpcaId(best(wpcaIds())));
        return ids;
    }

    boolean wpcaBeats() {
        return best(List.of(best(wpcaIds()), "3-znorm-lda")).startsWith("4-wpca-");
    }

    StringBuilder report(String source, int limit) {
        StringBuilder t = new StringBuilder();
        int n = Math.min(limit, CONFIGS);
        Result any = results.values().iterator().next();
        t.append("Этап 5: протокол «только FEI» (FeiMethods, faces-fei)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
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

        t.append("\n=== Сводка (основная таблица FEI) ===\n");
        t.append("метод | суть | ошибки своих α=0,05 (FRR + под чужим) | α=0 | анфас / поворот α=0,05 | анфас / поворот α=0 | "
                + "FAR контроля люди α=0,05: худш. (↑95) / мед. | FAR контроля попытки α=0,05, сумма | FAR контроля люди α=0: худш. / мед. | "
                + "argmin своих\n");
        List<String> ids = tableIds();
        for (String id : ids) t.append(row(results.get(id))).append('\n');
        String best = best(ids);
        t.append(String.format(Locale.ROOT, "%nЛучший по правилу (наименьшая ошибка своих при α = 0,05, при равенстве — α = 0; FAR контроля не "
                + "участвует): %s — %s.%n", best, pctOf(results.get(best).errors(A05), results.get(best).att)));
        String bw = best(wpcaIds());
        t.append(String.format(Locale.ROOT, "Лучшая строка wpca: %s (%s) против 3-znorm-lda (%s): %s.%n", bw, pctOf(results.get(bw).errors(A05),
                results.get(bw).att), pctOf(results.get("3-znorm-lda").errors(A05), results.get("3-znorm-lda").att),
                wpcaBeats() ? "обходит — добавлена строка " + blkWpcaId(bw) : "не обходит — строка 4-sum-blk-both+wpca не добавляется"));

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
