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
        A("2c-mlda", "A"), L1(FeiMethods.LBP_MLDA, "L1"), L1IN(FeiMethods.LBP_MLDA_IN, "L1-in");

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

    /** Векторы вариантов vs по кадрам (OpenCV — только здесь, в главном потоке). */
    void vectors(Map<String, Mat> frames, Set<V> vs, IlluminationNorm.NormParams params) {
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
                double[] s = LbpFeature.sqrt(LbpFeature.histogram(e.getValue()));
                double[] sm = LbpFeature.sqrt(LbpFeature.histogram(m));
                if (vs.contains(V.L1)) {
                    vec.get(V.L1).put(e.getKey(), s);
                    mir.get(V.L1).put(e.getKey(), sm);
                }
                if (vs.contains(V.L1IN)) {
                    vec.get(V.L1IN).put(e.getKey(), LbpFeature.select(s, LbpFeature.Mask.INNER));
                    mir.get(V.L1IN).put(e.getKey(), LbpFeature.select(sm, LbpFeature.Mask.INNER));
                }
            }
            m.release();
        }
    }

    /**
     * Модель MLDA варианта v: классы — свои (ключи галереи; при mirror — затем их отражения), затем когорта (все ключи;
     * при mirror — затем отражения); как FeiMethods.classes / mirrorClasses (+mirror, Mirror.ALL).
     */
    GalleryScorer.ScoreModel fit(V v, boolean mirror, List<List<String>> gallery, List<List<String>> cohort) {
        List<List<double[]>> cls = new ArrayList<>();
        for (List<List<String>> group : List.of(gallery, cohort)) {
            for (List<String> keys : group) {
                List<double[]> list = new ArrayList<>();
                for (String k : keys) list.add(vec.get(v).get(k));
                if (mirror) for (String k : keys) list.add(mir.get(v).get(k));
                cls.add(list);
            }
        }
        return new MldaScorer().fit(cls, gallery.size());
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

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                   Path out) throws IOException {
        long seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        int b = settings.loadFacesFeiBootstrap();
        int threads = settings.loadFacesFeiCvThreads();
        double rLimit = poseLimit(selLines);
        Map<String, Double> pose = poses(dir.resolve("detections.tsv"));
        Map<String, Integer> sess = sessions(dir.resolve(SESSIONS));

        // Снимки GT после отбора по позе: человек → дата → ключи (по номеру снимка).
        Map<String, Map<Integer, List<String>>> byPerson = new TreeMap<>();
        Map<String, Sample> samples = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>();
        int total = 0;
        List<String> dropped = new ArrayList<>();
        for (Map.Entry<String, List<Sample>> e : new TreeMap<>(export.data().gt()).entrySet()) {
            List<Sample> list = new ArrayList<>(e.getValue());
            list.sort((x, y) -> x.file().getFileName().toString().compareTo(y.file().getFileName().toString()));
            for (Sample s : list) {
                total++;
                String name = s.file().getFileName().toString();
                Double r = pose.get("gt/" + s.person() + "/" + name);
                if (r == null) throw new IllegalStateException("Нет позы снимка GT " + s.file());
                String k = s.file().toString();
                if (Math.abs(r) > rLimit) {
                    dropped.add(s.person() + "/" + name);
                    continue;
                }
                Integer d = sess.get(s.person() + "/" + name);
                if (d == null) throw new IllegalStateException("Нет сессии снимка GT " + s.file());
                byPerson.computeIfAbsent(s.person(), x -> new TreeMap<>()).computeIfAbsent(d, x -> new ArrayList<>()).add(k);
                samples.put(k, s);
                rOf.put(k, Math.abs(r));
            }
        }
        // Свои — ≥ 2 датами по всем снимкам (до отбора), как в инвентаризации.
        Map<String, Set<Integer>> datesAll = new TreeMap<>();
        sess.forEach((pk, d) -> {
            if (d > 0) datesAll.computeIfAbsent(pk.substring(0, pk.indexOf('/')), x -> new HashSet<>()).add(d);
        });
        List<String> own = new ArrayList<>();
        List<String> ctrl = new ArrayList<>();
        for (String p : byPerson.keySet()) (datesAll.getOrDefault(p, Set.of()).size() >= 2 ? own : ctrl).add(p);
        List<String> persons = new ArrayList<>(byPerson.keySet());
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);

        Map<String, List<String>> galleryKeys = new LinkedHashMap<>();
        Map<String, Integer> galleryDateOf = new LinkedHashMap<>();
        Map<String, List<String>> probeKeys = new LinkedHashMap<>();
        for (String p : own) {
            Map<Integer, List<String>> bd = byPerson.get(p);
            int g = galleryDate(bd);
            galleryDateOf.put(p, g);
            galleryKeys.put(p, bd.get(g));
            List<String> pr = new ArrayList<>();
            bd.forEach((d, ks) -> {
                if (d != g) pr.addAll(ks);
            });
            pr.sort(null);
            if (pr.isEmpty()) throw new IllegalStateException("У своего " + p + " после отбора нет проб");
            probeKeys.put(p, pr);
        }
        List<String> cohortP = cohortPersons(feiKeys, seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeySet = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeySet.addAll(ks);
        }

        // Векторы: GT после отбора и когорта FEI.
        XBaseEval x = new XBaseEval();
        Map<String, Mat> fr = frames(export, samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(frames(export, fei, cohortKeySet));
        x.vectors(fr, Set.of(V.values()), IlluminationNorm.NormParams.of(settings));
        fr.values().forEach(Mat::release);

        // Пробы и роли.
        List<Probe> mainProbes = new ArrayList<>();
        for (int i = 0; i < own.size(); i++) {
            String p = own.get(i);
            for (String k : probeKeys.get(p)) mainProbes.add(new Probe(OWN, p, image(samples.get(k)), k, i));
        }
        for (String p : ctrl) for (List<String> ks : byPerson.get(p).values()) for (String k : ks) mainProbes.add(new Probe(CTRL, p, image(samples.get(k)), k, -1));
        mainProbes.sort((u, w) -> u.role().equals(w.role()) ? u.key().compareTo(w.key()) : u.role().equals(OWN) ? -1 : 1);
        List<List<String>> parts = FeiMethods.cvParts(own, seed, CV_S);
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();

        List<Callable<String>> tasks = new ArrayList<>();
        for (V v : V.values()) {
            for (boolean m : new boolean[] {false, true}) {
                String method = v.id + (m ? FeiMethods.MIRROR : "");
                tasks.add(() -> x.lines(method, 0, 0, x.fit(v, m, fullGallery, cohort), v, own, mainProbes));
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                    List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                    List<Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, image(samples.get(k)), k, -1));
                    tasks.add(() -> x.lines(method, 0, 0, x.fit(v, m, gk, cohort), v, g, pr));
                }
            }
        }
        long t1 = System.nanoTime();
        List<String> res = run(tasks, threads);
        double fitS = (System.nanoTime() - t1) / 1e9;
        StringBuilder all = new StringBuilder();
        res.forEach(all::append);

        // Файл оценок.
        String header = "# Сверка на других базах (XBaseEval, PLAN.md п. 5.5): Georgia Tech; роли own — свой, ctrl — контрольный чужой, cv<q> — "
                + "снимок галереи человека части q против модели без части q\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir
                + ", коммит данных " + FarMethods.dataCommit(dir) + "\n" + FeiScores.COLUMNS + "\n";
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(out.resolve("xbase_gt_scores.tsv.gz")), 1 << 16),
                StandardCharsets.UTF_8)) {
            w.write(header);
            w.write(all.toString());
        }

        // Разбор: модель по позе пробы, конфигурация и «чужие» П2 / П3.
        Map<String, Row> rows = new HashMap<>();
        for (String l : all.toString().lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        List<FeiThreshold.Res> results = new ArrayList<>();
        int[] cases = new int[FeiScores.CASES];
        int ctrlImages = 0;
        int cvImages = 0;
        for (V v : V.values()) {
            FeiDecompose.Cfg c = new FeiDecompose.Cfg();
            List<double[]> ownList = new ArrayList<>();
            List<double[]> ctrlList = new ArrayList<>();
            List<double[]> cvList = new ArrayList<>();
            int cvFront = 0;
            List<Boolean> front = new ArrayList<>();
            List<Row> ownRows = new ArrayList<>();
            for (Probe p : mainProbes) {
                Row r = choose(rows, v, p.role(), p.person(), p.image(), rOf.get(p.key()) <= r0);
                if (p.role().equals(OWN)) {
                    ownRows.add(r);
                    front.add(rOf.get(p.key()) <= r0);
                } else {
                    ctrlList.add(new double[] {pIdx.get(p.person()), FeiScores.ratio(r, null)});
                }
            }
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                for (String p : parts.get(q)) {
                    for (String k : galleryKeys.get(p)) {
                        boolean f = rOf.get(k) <= r0;
                        if (f) cvFront++;
                        Row r = choose(rows, v, FeiMethods.CV_ROLE + q, p, image(samples.get(k)), f);
                        cvList.add(new double[] {pIdx.get(p), FeiScores.ratio(r, v == V.L1 ? cases : null)});
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
                c.ownP[i] = pIdx.get(r.person());
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
            ctrlImages = c.ctrlR.length;
            cvImages = cvList.size();
            FeiThreshold.Imp imp = FeiThreshold.Imp.of(cvList, cvFront);
            results.add(FeiThreshold.evaluate(v.label, v.ratioId(), List.of(c), List.of(imp), persons.size(), b,
                    seed + FeiBootstrap.SEED_OFFSET));
        }
        StringBuilder t = report(results, own, ctrl, galleryKeys, galleryDateOf, probeKeys, byPerson, parts, cohortP, cohort, rOf, r0, rLimit,
                total, dropped, ctrlImages, cvImages, cases, b, seed, persons.size(), dir, fitS, threads, mainProbes);
        FarMethods.write(out.resolve("xbase_gt.txt"), t);
        System.out.print(t);
    }

    /** Номер снимка из имени файла GT («01» → 1). */
    static int image(Sample s) {
        return Integer.parseInt(s.file().getFileName().toString().replaceAll("\\D", ""));
    }

    /** Строка модели по позе: анфас (|r| ≤ r0) — X, иначе X+mirror. */
    static Row choose(Map<String, Row> rows, V v, String role, String person, int image, boolean frontal) {
        String method = v.id + (frontal ? "" : FeiMethods.MIRROR);
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
        return t;
    }
}
