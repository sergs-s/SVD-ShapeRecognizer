package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Сверка на других базах (PLAN.md п. 5.5; план подтверждён Хозяином 11.10.2026): MUCT и ORL без нормировки (none), варианты
 * L1, L1-in, A с умолчаниями FEI (кадр «а», r0, зеркала, MLDA, d₁/d₂, порог П2 / П3 по «чужим» галереи) — тот же путь, что
 * XBaseEval на GT (XBaseEval.fitLines / parse). Несколько конфигураций (ротаций) объединяются, как конфигурации FEI.
 * <ul>
 *   <li>отбор по позе — |r| ≤ R (fei_selection.tsv); люди с менее чем 2 отобранными снимками исключаются;</li>
 *   <li>MUCT: когорта MLDA — своя, 20 человек MUCT (перестановка seed + 9400, близнецы 247 и 248 не берутся), все их снимки;
 *       остальные люди — 5 частей (seed + 9410, близнецы — одной единицей); в ротации j часть j — контроль, остальные — свои;
 *       снимки своего — пополам случайно (seed + 9420; первые ⌈n/2⌉ — галерея), одинаково во всех ротациях; 5 конфигураций;
 *       справочно — L1 с когортой FEI 20 (как на GT);</li>
 *   <li>ORL: когорта — встроенная FEI 20; 5 частей по 8 человек (seed + 9500), снимки пополам (seed + 9510); конфигурация
 *       2j + k: часть j — контроль, k = 0 — галерея первая половина, k = 1 — вторая (подротация); 10 конфигураций;</li>
 *   <li>«чужие» П2/П3 конфигурации s — 5 частей своих (cvParts: как FeiMethods.cvParts, seed 100 + s); бутстреп по людям базы
 *       (когорта не перевыбирается), B = faces.fei.bootstrap, seed faces.seed + 7000.</li>
 * </ul>
 * Рядом с П2 и П3 — разделимость при равном фактическом FAR контроля (порог по контрольным чужим своей конфигурации, 1 и
 * 5 %; точка равных ошибок — по объединённым попыткам). Контрольная точка — после каждой конфигурации
 * (reports/faces/xbase/checkpoint_&lt;база&gt;/; повтор команды продолжает; отпечаток не совпал — ошибка). Отчёт
 * xbase_&lt;база&gt;.txt, оценки xbase_&lt;база&gt;_scores.tsv.gz (MUCT — ещё xbase_muct_fei_scores.tsv.gz).
 *
 * @author ssv
 */
final class XBaseOther {

    static final long MUCT_COHORT_OFFSET = 9400;
    static final long MUCT_PART_OFFSET = 9410;
    static final long MUCT_HALF_OFFSET = 9420;
    static final long ORL_PART_OFFSET = 9500;
    static final long ORL_HALF_OFFSET = 9510;
    static final List<String> TWINS = List.of("247", "248");
    static final int PARTS = 5;
    static final String[] FAR_NAME = {"FAR 1 %", "FAR 5 %"};
    static final double[] SEP_FARS = XBaseSepLtp.FARS;

    private XBaseOther() {
    }

    // ---------------------------------------------------------------- деления

    /** Когорта MUCT: люди без близнецов, перестановка Random(seed + 9400), первые 20, по возрастанию. */
    static List<String> muctCohort(List<String> persons, long seed) {
        List<String> c = new ArrayList<>(persons.stream().filter(p -> !TWINS.contains(p)).sorted().toList());
        Collections.shuffle(c, new Random(seed + MUCT_COHORT_OFFSET));
        List<String> out = new ArrayList<>(c.subList(0, XBaseEval.COHORT));
        Collections.sort(out);
        return out;
    }

    /**
     * Части по людям: единицы (человек; оба близнеца — одна единица, если они есть в списке) по возрастанию, перестановка
     * Random(seed + offset), единица i — в часть i mod 5; каждая часть по возрастанию.
     */
    static List<List<String>> parts(List<String> persons, long seed, long offset) {
        List<List<String>> units = new ArrayList<>();
        boolean twins = persons.containsAll(TWINS);
        for (String p : persons.stream().sorted().toList()) {
            if (twins && TWINS.contains(p)) {
                if (p.equals(TWINS.get(0))) units.add(TWINS);
            } else {
                units.add(List.of(p));
            }
        }
        Collections.shuffle(units, new Random(seed + offset));
        List<List<String>> out = new ArrayList<>();
        for (int i = 0; i < PARTS; i++) out.add(new ArrayList<>());
        for (int i = 0; i < units.size(); i++) out.get(i % PARTS).addAll(units.get(i));
        out.forEach(Collections::sort);
        return out;
    }

    /**
     * Снимки пополам: люди по возрастанию, у каждого ключи перемешиваются одним Random(seed + offset); первые ⌈n/2⌉ —
     * половина 0, остальные — половина 1 (обе по возрастанию). Возвращает {половина 0, половина 1}.
     */
    static List<Map<String, List<String>>> halves(Map<String, List<String>> keys, long seed, long offset) {
        Random rnd = new Random(seed + offset);
        Map<String, List<String>> a = new LinkedHashMap<>();
        Map<String, List<String>> b = new LinkedHashMap<>();
        for (String p : new TreeMap<>(keys).keySet()) {
            List<String> ks = new ArrayList<>(keys.get(p));
            Collections.shuffle(ks, rnd);
            int h = (ks.size() + 1) / 2;
            List<String> x = new ArrayList<>(ks.subList(0, h));
            List<String> y = new ArrayList<>(ks.subList(h, ks.size()));
            Collections.sort(x);
            Collections.sort(y);
            a.put(p, x);
            b.put(p, y);
        }
        return List.of(a, b);
    }

    /**
     * Части «чужих» П2/П3: как FeiMethods.cvParts (перестановка Random(seed + CV_SEED_OFFSET + s), подряд), но число своих
     * может не делиться на 5 — первые n mod 5 частей на человека больше; при делимом n — то же, что cvParts.
     */
    static List<List<String>> cvParts(List<String> own, long seed, int s) {
        List<String> perm = new ArrayList<>(own);
        Collections.shuffle(perm, new Random(seed + FeiMethods.CV_SEED_OFFSET + s));
        int n = FeiMethods.CV_PARTS;
        List<List<String>> out = new ArrayList<>();
        int from = 0;
        for (int q = 0; q < n; q++) {
            int size = own.size() / n + (q < own.size() % n ? 1 : 0);
            out.add(List.copyOf(perm.subList(from, from + size)));
            from += size;
        }
        return out;
    }

    // ---------------------------------------------------------------- данные базы

    /** Отобранные снимки базы: человек → ключи (по имени файла), номер снимка, |r|, исключённые. */
    static final class Base {
        String code;
        String title;
        Map<String, List<String>> keys = new TreeMap<>();
        Map<String, Sample> samples = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>();
        int total;
        List<String> dropped = new ArrayList<>();
        List<String> fewPersons = new ArrayList<>();
    }

    static Base base(String code, String title, Map<String, List<Sample>> data, Map<String, Double> pose, double rLimit) {
        Base b = new Base();
        b.code = code;
        b.title = title;
        for (Map.Entry<String, List<Sample>> e : new TreeMap<>(data).entrySet()) {
            List<Sample> list = new ArrayList<>(e.getValue());
            list.sort((x, y) -> x.file().getFileName().toString().compareTo(y.file().getFileName().toString()));
            List<String> ks = new ArrayList<>();
            for (int i = 0; i < list.size(); i++) {
                Sample s = list.get(i);
                b.total++;
                String name = s.file().getFileName().toString();
                Double r = pose.get(code + "/" + s.person() + "/" + name);
                if (r == null) throw new IllegalStateException("Нет позы снимка " + s.file());
                if (Math.abs(r) > rLimit) {
                    b.dropped.add(s.person() + "/" + name);
                    continue;
                }
                String k = s.file().toString();
                ks.add(k);
                b.samples.put(k, s);
                b.imageNo.put(k, code.equals("orl") ? XBaseEval.image(s) : i + 1);
                b.rOf.put(k, Math.abs(r));
            }
            if (ks.size() < 2) {
                b.fewPersons.add(e.getKey() + " (" + ks.size() + ")");
                ks.forEach(b.samples::remove);
                continue;
            }
            b.keys.put(e.getKey(), ks);
        }
        return b;
    }

    /** Конфигурация: номер, свои, контроль, галерея, пробы. */
    record Conf(int s, List<String> own, List<String> ctrl, Map<String, List<String>> gal, Map<String, List<String>> pro) {}

    /** Gt одной конфигурации (общие поля — из шаблона t). */
    static XBaseEval.Gt gt(XBaseEval.Gt t, Base b, Conf c, List<List<String>> cohort) {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.seed = t.seed;
        g.r0 = t.r0;
        g.b = t.b;
        g.threads = t.threads;
        g.rLimit = t.rLimit;
        g.dir = t.dir;
        g.samples = b.samples;
        g.rOf = b.rOf;
        g.imageNo = b.imageNo;
        g.persons = t.persons;
        g.pIdx = t.pIdx;
        g.own = c.own();
        g.ctrl = c.ctrl();
        for (String p : c.ctrl()) g.byPerson.put(p, new TreeMap<>(Map.of(0, b.keys.get(p))));
        g.cohort = cohort;
        return g;
    }

    // ---------------------------------------------------------------- контрольная точка

    static final class Checkpoint {
        final Path dir;

        Checkpoint(Path dir, String fingerprint) throws IOException {
            this.dir = dir;
            Path fp = dir.resolve("fingerprint.txt");
            if (Files.exists(fp)) {
                String old = Files.readString(fp, StandardCharsets.UTF_8);
                if (!old.equals(fingerprint)) {
                    throw new IllegalStateException("Контрольная точка " + dir + " от другого прогона (отпечаток не совпал): верните как было или удалите "
                            + "каталог.\nБыло:\n" + old + "\nСейчас:\n" + fingerprint);
                }
            } else {
                Files.createDirectories(dir);
                write(fp, fingerprint);
            }
        }

        Path file(String name) {
            return dir.resolve(name + ".tsv.gz");
        }

        String read(String name) throws IOException {
            Path f = file(name);
            if (!Files.exists(f)) return null;
            try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(f), 1 << 16), StandardCharsets.UTF_8))) {
                StringBuilder t = new StringBuilder();
                char[] buf = new char[1 << 16];
                int n;
                while ((n = in.read(buf)) > 0) t.append(buf, 0, n);
                return t.toString();
            }
        }

        void save(String name, String lines) throws IOException {
            Path tmp = dir.resolve(name + ".tmp");
            try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16), StandardCharsets.UTF_8)) {
                w.write(lines);
            }
            try {
                Files.move(tmp, file(name), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                Files.move(tmp, file(name), StandardCopyOption.REPLACE_EXISTING);
            }
        }

        void delete() throws IOException {
            try (Stream<Path> s = Files.list(dir)) {
                for (Path p : s.toList()) Files.delete(p);
            }
            Files.delete(dir);
        }

        private static void write(Path f, String text) throws IOException {
            Path tmp = f.resolveSibling(f.getFileName() + ".tmp");
            Files.writeString(tmp, text, StandardCharsets.UTF_8);
            Files.move(tmp, f, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // ---------------------------------------------------------------- прогон

    /** Итоги набора вариантов по всем конфигурациям: строки оценок, [вариант] конфигурации и «чужие». */
    static final class Run {
        final StringBuilder lines = new StringBuilder();
        final Map<V, List<FeiDecompose.Cfg>> cfg = new LinkedHashMap<>();
        final Map<V, List<FeiThreshold.Imp>> imp = new LinkedHashMap<>();
        final List<int[]> volumes = new ArrayList<>();
        final int[] cases = new int[FeiScores.CASES];
        double fitS;
        int resumed;
    }

    static Run run(XBaseEval.Gt t, Base b, XBaseEval x, Set<V> vs, List<Conf> confs, List<List<String>> cohort, Checkpoint cp, String tag)
            throws IOException {
        Run r = new Run();
        for (V v : vs) {
            r.cfg.put(v, new ArrayList<>());
            r.imp.put(v, new ArrayList<>());
        }
        for (Conf c : confs) {
            XBaseEval.Gt g = gt(t, b, c, cohort);
            List<XBaseEval.Probe> probes = XBaseEval.probes(g, c.pro());
            List<List<String>> parts = cvParts(g.own, g.seed, XBaseEval.CV_S + c.s());
            String name = String.format(Locale.ROOT, "conf_%02d_%s", c.s(), tag);
            String lines = cp.read(name);
            if (lines == null) {
                long t1 = System.nanoTime();
                lines = XBaseEval.fitLines(g, x, vs, c.gal(), probes, parts, c.s());
                r.fitS += (System.nanoTime() - t1) / 1e9;
                cp.save(name, lines);
                System.out.printf(Locale.ROOT, "%s %s: конфигурация %d готова (%.0f с всего)%n", b.code, tag, c.s(), r.fitS);
            } else {
                r.resumed++;
                System.out.printf(Locale.ROOT, "%s %s: конфигурация %d — из контрольной точки%n", b.code, tag, c.s());
            }
            r.lines.append(lines);
            XBaseEval.Parsed p = XBaseEval.parse(g, vs, c.gal(), probes, parts, lines);
            for (V v : vs) {
                r.cfg.get(v).add(p.cfg.get(v));
                r.imp.get(v).add(p.imp.get(v));
            }
            for (int k = 0; k < r.cases.length; k++) r.cases[k] += p.cases[k];
            int ownN = probes.stream().mapToInt(q -> q.role().equals(FeiScores.OWN) ? 1 : 0).sum();
            r.volumes.add(new int[] {c.s(), c.own().size(), c.ctrl().size(), ownN, p.ctrlImages, p.cvImages});
        }
        return r;
    }

    static Map<String, FeiThreshold.Res> results(XBaseEval.Gt t, Run r, Map<V, String> labels) {
        Map<String, FeiThreshold.Res> m = new LinkedHashMap<>();
        for (Map.Entry<V, String> e : labels.entrySet()) {
            V v = e.getKey();
            m.put(e.getValue(), FeiThreshold.evaluate(e.getValue(), v.ratioId(), r.cfg.get(v), r.imp.get(v), t.persons.size(), t.b, t.seed + FeiBootstrap.SEED_OFFSET));
        }
        return m;
    }

    static void writeScores(Path file, String header, String lines) throws IOException {
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(file), 1 << 16), StandardCharsets.UTF_8)) {
            w.write(header);
            w.write(FeiScores.COLUMNS + "\n");
            w.write(lines);
        }
    }

    static void run(String mode, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        boolean muct = mode.equals("muct");
        XBaseEval.Gt t = new XBaseEval.Gt();
        t.dir = dir;
        t.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval " + mode + ": задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        t.r0 = FeiPose.r0(rawR0, "+pose");
        t.b = settings.loadFacesFeiBootstrap();
        t.threads = settings.loadFacesFeiCvThreads();
        t.rLimit = XBaseEval.poseLimit(selLines);
        Map<String, Double> pose = XBaseEval.poses(dir.resolve("detections.tsv"));
        Base b = muct ? base("muct", "MUCT", export.data().muct(), pose, t.rLimit) : base("orl", "ORL", export.data().orl(), pose, t.rLimit);

        // Когорты: MUCT — своя 20 (и справочно FEI 20), ORL — FEI 20.
        List<String> feiCohortP = XBaseEval.cohortPersons(feiKeys, t.seed);
        List<List<String>> feiCohort = new ArrayList<>();
        Set<String> feiCohortKeys = new HashSet<>();
        for (String p : feiCohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            feiCohort.add(ks);
            feiCohortKeys.addAll(ks);
        }
        List<String> all = new ArrayList<>(b.keys.keySet());
        List<String> cohortP = muct ? muctCohort(all, t.seed) : feiCohortP;
        List<List<String>> cohort = muct ? cohortP.stream().map(b.keys::get).toList() : feiCohort;
        List<String> rest = all.stream().filter(p -> !muct || !cohortP.contains(p)).toList();
        t.persons = rest;
        for (int i = 0; i < rest.size(); i++) t.pIdx.put(rest.get(i), i);
        List<List<String>> parts = parts(rest, t.seed, muct ? MUCT_PART_OFFSET : ORL_PART_OFFSET);
        Map<String, List<String>> restKeys = new TreeMap<>();
        for (String p : rest) restKeys.put(p, b.keys.get(p));
        List<Map<String, List<String>>> half = halves(restKeys, t.seed, muct ? MUCT_HALF_OFFSET : ORL_HALF_OFFSET);
        List<Conf> confs = new ArrayList<>();
        for (int j = 0; j < PARTS; j++) {
            Set<String> ctrlSet = new HashSet<>(parts.get(j));
            List<String> own = rest.stream().filter(p -> !ctrlSet.contains(p)).toList();
            for (int k = 0; k < (muct ? 1 : 2); k++) {
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    gal.put(p, half.get(k).get(p));
                    pro.put(p, half.get(1 - k).get(p));
                }
                confs.add(new Conf(muct ? j : 2 * j + k, own, parts.get(j), gal, pro));
            }
        }

        // Векторы: отобранные снимки базы и когорта FEI.
        XBaseEval x = new XBaseEval();
        Map<String, Mat> fr = XBaseEval.frames(export, b.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, feiCohortKeys));
        t.params = IlluminationNorm.NormParams.of(settings);
        x.vectors(fr, XBaseEval.BASE_VARIANTS, t.params);
        fr.values().forEach(Mat::release);

        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "режим %s; данные %s; seed %d; r0 %s; конфигураций %d", mode,
                FarMethods.dataCommit(dir), t.seed, t.r0, confs.size()));
        Checkpoint cp = new Checkpoint(out.resolve("checkpoint_" + mode), fp);
        Run main = run(t, b, x, XBaseEval.BASE_VARIANTS, confs, cohort, cp, "main");
        Run ref = muct ? run(t, b, x, EnumSet.of(V.L1), confs, feiCohort, cp, "fei") : null;

        String head = "# Сверка на других базах (XBaseOther, PLAN.md п. 5.5): " + b.title + "; s — конфигурация; роли own — свой, ctrl — контрольный чужой, "
                + "cv<q> — снимок галереи человека части q против модели без части q\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir
                + ", коммит данных " + FarMethods.dataCommit(dir) + "\n";
        writeScores(out.resolve("xbase_" + mode + "_scores.tsv.gz"), head + "# Когорта: " + (muct ? "своя 20 MUCT" : "FEI 20") + "\n", main.lines.toString());
        if (ref != null) writeScores(out.resolve("xbase_muct_fei_scores.tsv.gz"), head + "# Когорта: FEI 20 (справочно, только L1)\n", ref.lines.toString());

        Map<V, String> labels = new LinkedHashMap<>();
        labels.put(V.L1, "L1");
        labels.put(V.L1IN, "L1-in");
        labels.put(V.A, "A");
        Map<String, FeiThreshold.Res> res = results(t, main, labels);
        if (ref != null) res.putAll(results(t, ref, Map.of(V.L1, "L1 (когорта FEI)")));

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, сверка на других базах: ").append(b.title).append(" (XBaseEval ").append(mode).append("), нормировка none\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/").append(mode)
                .append(" — кадр а 92×112; detections.tsv; когорта FEI — a/fei)\n");
        r.append(muct ? "Оговорка: MUCT — одна сессия съёмки; результаты внутрисессионные (оптимистичны, как FEI и GT внутри сессии).\n"
                : "Оговорка: ORL — часть людей снята в разное время без разметки сессий; результаты смешанные (внутри- и межсессионные).\n");
        r.append("Умолчания FEI без подбора: кадр а; L1 = корень гистограмм LBP u2 (8, 2), 7×7 + MLDA; L1-in — внутренние блоки 1–5;\n"
                + "A = CLAHE + MLDA (2c-mlda); зеркала (FaceMirror, способ а) — модель X+mirror; выбор модели по позе пробы r0 = ").append(t.r0)
                .append(";\nоценка d₁/d₂ (FeiScores.ratio); порог Нейман – Пирсон по «чужим» галереи конфигурации: П2 — точечно, П3 — с запасом 95 %.\n");
        r.append(String.format(Locale.ROOT, "Отбор по позе — как FEI: |r| ≤ R = %.4f: осталось %d из %d снимков; исключено снимков %d; людей с < 2 снимками "
                + "(исключены): %s.%n", t.rLimit, b.samples.size(), b.total, b.dropped.size(), b.fewPersons.isEmpty() ? "нет" : String.join(", ", b.fewPersons)));
        if (muct) {
            r.append(String.format(Locale.ROOT, "Когорта MLDA — своя: 20 человек MUCT (Random(seed + %d), близнецы 247 и 248 не берутся), %d снимков: %s.%n",
                    MUCT_COHORT_OFFSET, cohort.stream().mapToInt(List::size).sum(), String.join(", ", cohortP)));
            r.append(String.format(Locale.ROOT, "Справочно — L1 с когортой FEI 20 (seed faces.seed + 9000, как на GT), %d снимков.%n", feiCohortKeys.size()));
        } else {
            r.append(String.format(Locale.ROOT, "Когорта MLDA — встроенная: 20 «полных» людей FEI (seed faces.seed + 9000), %d снимков.%n", feiCohortKeys.size()));
        }
        r.append(String.format(Locale.ROOT, "Люди базы вне когорты: %d; 5 частей по людям (Random(seed + %d)%s): размеры %s; снимки пополам — Random(seed + %d), "
                + "первые ⌈n/2⌉ — половина 0.%n", rest.size(), muct ? MUCT_PART_OFFSET : ORL_PART_OFFSET, muct ? ", близнецы — одной единицей" : "",
                parts.stream().map(q -> Integer.toString(q.size())).toList(), muct ? MUCT_HALF_OFFSET : ORL_HALF_OFFSET));
        r.append(muct ? "Конфигурация j (0–4): часть j — контрольные чужие (все снимки), остальные — свои: галерея — половина 0, пробы — половина 1.\n"
                : "Конфигурация 2j + k: часть j — контрольные чужие (все снимки), остальные — свои; k = 0 — галерея половина 0, пробы половина 1;\n"
                        + "k = 1 — наоборот (подротация).\n");
        r.append("«Чужие» П2/П3 конфигурации s — снимки галереи своих, 5 частей (перестановка как FeiMethods.cvParts, seed + 8000 + 100 + s; первые n mod 5\n"
                + "частей на человека больше), модель без части.\n");
        for (int q = 0; q < PARTS; q++) r.append("  часть ").append(q).append(": ").append(String.join(", ", parts.get(q))).append('\n');
        r.append("конфигурация | своих | контрольных людей | проб своих | снимков контроля | «чужих» П2/П3\n");
        for (int[] v : main.volumes) r.append(String.format(Locale.ROOT, "%d | %d | %d | %d | %d | %d%n", v[0], v[1], v[2], v[3], v[4], v[5]));
        if (t.b > 0) {
            r.append(String.format(Locale.ROOT, "Бутстреп по людям %s вне когорты: B = %d, seed = %d, людей %d; одни веса для всех строк; порог — заново в каждой "
                    + "реплике;%n[2,5; 97,5] — процентили; «*» — интервал разности не содержит 0.%n", b.title, t.b, t.seed + FeiBootstrap.SEED_OFFSET, rest.size()));
        }
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с%s; из контрольной точки — конфигураций %d; потоков %d.%n", main.fitS + (ref != null ? ref.fitS : 0),
                ref != null ? String.format(Locale.ROOT, " (основные %.0f, когорта FEI %.0f)", main.fitS, ref.fitS) : "", main.resumed + (ref != null ? ref.resumed : 0),
                t.threads));
        r.append("Правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6), все конфигурации: ");
        for (int k = 0; k < FeiScores.CASES; k++) r.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(main.cases[k]);
        r.append('\n');

        int[][] w = t.b > 0 ? FeiBootstrap.weights(rest.size(), t.b, t.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(rest.size());
        Map<String, Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), sep(e.getValue().cfgs, unit, w));
        tables(r, res, sep, t.b, muct);
        FarMethods.write(out.resolve("xbase_" + mode + ".txt"), r);
        System.out.print(r);
        cp.delete();
    }

    // ---------------------------------------------------------------- разделимость

    /** Разделимость строки: [k] ошибка своих при фактическом FAR контроля SEP_FARS[k] (точка, достигнутый FAR, реплики), EER. */
    static final class Sep {
        double[] err = new double[SEP_FARS.length];
        double[] far = new double[SEP_FARS.length];
        double[][] errRep;
        double[] eer;
        double[] eerRep;
        double[][] q;
    }

    static XBaseSep.Data data(FeiDecompose.Cfg c) {
        XBaseSep.Data d = new XBaseSep.Data();
        d.ownP = c.ownP;
        d.ownR = c.ownR;
        d.ownOk = c.ownOk;
        d.ctrlP = c.ctrlP;
        d.ctrlR = c.ctrlR;
        return d;
    }

    /** Попытки всех конфигураций одной таблицей (для точки равных ошибок и квартилей). */
    static XBaseSep.Data pooled(List<FeiDecompose.Cfg> cs) {
        XBaseSep.Data d = new XBaseSep.Data();
        d.ownP = cs.stream().flatMapToInt(c -> Arrays.stream(c.ownP)).toArray();
        d.ownR = cs.stream().flatMapToDouble(c -> Arrays.stream(c.ownR)).toArray();
        int n = d.ownR.length;
        d.ownOk = new boolean[n];
        int i = 0;
        for (FeiDecompose.Cfg c : cs) for (boolean ok : c.ownOk) d.ownOk[i++] = ok;
        d.ctrlP = cs.stream().flatMapToInt(c -> Arrays.stream(c.ctrlP)).toArray();
        d.ctrlR = cs.stream().flatMapToDouble(c -> Arrays.stream(c.ctrlR)).toArray();
        return d;
    }

    /**
     * Ошибка своих при фактическом FAR контроля far: в каждой конфигурации — порог по её контрольным чужим
     * (XBaseSepLtp.ctrlThreshold), ошибки и число попыток своих суммируются по конфигурациям (веса людей w). {ошибка %, FAR %}.
     */
    static double[] errAtFar(List<FeiDecompose.Cfg> cs, int[] w, double far) {
        double e = 0;
        double n = 0;
        double fx = 0;
        double fn = 0;
        for (FeiDecompose.Cfg c : cs) {
            XBaseSep.Data d = data(c);
            double th = XBaseSepLtp.ctrlThreshold(d, w, far);
            for (int i = 0; i < d.ownR.length; i++) {
                int wt = w[d.ownP[i]];
                n += wt;
                if (!(d.ownOk[i] && d.ownR[i] <= th)) e += wt;
            }
            for (int i = 0; i < d.ctrlR.length; i++) {
                int wt = w[d.ctrlP[i]];
                fn += wt;
                if (d.ctrlR[i] <= th) fx += wt;
            }
        }
        return new double[] {100 * e / n, 100 * fx / fn};
    }

    static Sep sep(List<FeiDecompose.Cfg> cs, int[] unit, int[][] w) {
        Sep s = new Sep();
        s.errRep = new double[SEP_FARS.length][w.length];
        for (int k = 0; k < SEP_FARS.length; k++) {
            double[] p = errAtFar(cs, unit, SEP_FARS[k]);
            s.err[k] = p[0];
            s.far[k] = p[1];
            for (int q = 0; q < w.length; q++) s.errRep[k][q] = errAtFar(cs, w[q], SEP_FARS[k])[0];
        }
        XBaseSep.Data d = pooled(cs);
        s.eer = XBaseSep.eer(d, unit);
        s.eerRep = new double[w.length];
        for (int q = 0; q < w.length; q++) s.eerRep[q] = XBaseSep.eer(d, w[q])[3];
        s.q = new double[][] {XBaseSep.quartiles(d.ownR), XBaseSep.quartiles(d.ctrlR)};
        return s;
    }

    // ---------------------------------------------------------------- таблицы

    static void tables(StringBuilder t, Map<String, FeiThreshold.Res> res, Map<String, Sep> sep, int b, boolean muct) {
        t.append("\n=== Сводка: ошибка своих при П2, П3 и при равном фактическом FAR контроля (разделимость) ===\n");
        t.append("П2 / П3 — порог по «чужим» галереи (в скобках — FAR контроля по снимкам, точка); «разд.» — порог по контрольным чужим своей\n"
                + "конфигурации (фактический FAR контроля по снимкам — в скобках, по всем конфигурациям); EER — по объединённым попыткам.\n");
        t.append("строка | FAR 1 %: П2 | П3 | разд. | FAR 5 %: П2 | П3 | разд. | EER\n");
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) {
            FeiThreshold.Res r = e.getValue();
            Sep s = sep.get(e.getKey());
            List<String> row = new ArrayList<>(List.of(e.getKey()));
            for (int k = 0; k < 2; k++) {
                for (int m : LbpNormReport.MS) {
                    row.add(FeiThreshold.errCell(r, m, k, 0, b) + String.format(Locale.ROOT, " (%.2f %%)", FeiThreshold.farPct(r, m, k)));
                }
                row.add(XBaseSepLtp.cell(s.err[k], s.errRep[k]) + String.format(Locale.ROOT, " (%.2f %%)", s.far[k]));
            }
            row.add(XBaseSepLtp.cell(s.eer[3], s.eerRep));
            t.append(String.join(" | ", row)).append('\n');
        }

        t.append("\n=== Ошибка своих, разложение, верный кандидат ===\n");
        t.append("строка | способ | порог | ").append(LbpNormReport.ERR_HEAD).append('\n');
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) {
            for (int m : LbpNormReport.MS) {
                for (int k = 0; k < FeiThreshold.KS; k++) t.append(LbpNormReport.errRow(List.of(e.getKey(), FeiThreshold.METHODS[m], FAR_NAME[k]), e.getValue(), m, k, b));
            }
        }

        t.append("\n=== FAR контроля при П2 и П3 (данные к вопросу «П2 или П3») ===\n");
        t.append(LbpNormReport.FAR_NOTE);
        t.append("строка | способ | порог | ").append(LbpNormReport.FAR_HEAD).append('\n');
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) {
            for (int m : LbpNormReport.MS) {
                for (int k = 0; k < FeiThreshold.KS; k++) t.append(LbpNormReport.farRow(List.of(e.getKey(), FeiThreshold.METHODS[m], FAR_NAME[k]), e.getValue(), m, k, b));
            }
        }

        t.append("\n=== Разделимость: точка равных ошибок (по объединённым попыткам) и квартили d₁/d₂ (25 / 50 / 75 %) ===\n");
        t.append("строка | θ (d₁/d₂) | ошибка своих | FAR контроля | EER | d₁/d₂ своих | d₁/d₂ контрольных чужих\n");
        for (Map.Entry<String, Sep> e : sep.entrySet()) {
            Sep s = e.getValue();
            t.append(String.format(Locale.ROOT, "%s | %.6f | %.1f %% | %.2f %% | %s | %.4f / %.4f / %.4f | %.4f / %.4f / %.4f%n", e.getKey(), s.eer[0], s.eer[1], s.eer[2],
                    XBaseSepLtp.cell(s.eer[3], s.eerRep), s.q[0][0], s.q[0][1], s.q[0][2], s.q[1][0], s.q[1][1], s.q[1][2]));
        }

        if (b > 0) {
            List<String[]> pairs = new ArrayList<>(List.of(new String[] {"L1", "L1-in"}, new String[] {"L1", "A"}));
            if (muct) pairs.add(new String[] {"L1", "L1 (когорта FEI)"});
            t.append("\n=== Парные разности, п. п. (те же реплики; «*» — интервал [2,5; 97,5] не содержит 0) ===\n");
            t.append("пара | способ | ").append(LbpNormReport.DIFF_HEAD).append('\n');
            for (String[] p : pairs) {
                for (int m : LbpNormReport.MS) t.append(LbpNormReport.diffRow(List.of(p[0] + " − " + p[1], FeiThreshold.METHODS[m]), res.get(p[0]), res.get(p[1]), m));
            }
            t.append("\n=== Парные разности разделимости, п. п. (те же реплики) ===\n");
            t.append("пара | ошибка своих при фактическом FAR контроля 1 % | 5 % | EER\n");
            for (String[] p : pairs) {
                Sep a = sep.get(p[0]);
                Sep z = sep.get(p[1]);
                List<String> row = new ArrayList<>(List.of(p[0] + " − " + p[1]));
                for (int k = 0; k < SEP_FARS.length; k++) row.add(XBaseSepLtp.diff(a.err[k] - z.err[k], a.errRep[k], z.errRep[k]));
                row.add(XBaseSepLtp.diff(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                t.append(String.join(" | ", row)).append('\n');
            }
        }
    }
}
