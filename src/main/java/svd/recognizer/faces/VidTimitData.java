package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.IntPredicate;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.FaceDetectorYN;
import svd.recognizer.storage.SettingsStore;

/**
 * Кадры VidTIMIT для пула обучения W (опыт «пул FERET + VidTIMIT», задание Жорика 26.10.2026, план принят): фразы – в сессии 1
 * две последние по имени, в сессиях 2 и 3 – обе; повороты головы (head*) не берутся. Вариант (В2) – кадр ⌈n/2⌉ фразы, (В4) – кадры
 * ⌈n/3⌉ и ⌈2n/3⌉ (кадры фразы – по имени файла, n – их число). Кадр годен, если YuNet (0,7, два прохода, как FERET) нашёл лицо и |r|
 * ≤ R (R – строка «# R» fei_selection.tsv); негодный цел – ближайший годный (при равенстве – более ранний), в (В4) два кадра фразы
 * различны; не нашлось – строка описи «нет кадра», человек остаётся. Кадр «а» 92×112 – FeretData.frameA. Всё производное – только
 * в каталоге базы (derived\), вне git, не в SVD-faces-data, не в облако. Человек – один класс по всем трём сессиям (как xs-F).
 *
 * @author ssv
 */
final class VidTimitData {

    static final String DERIVED = "derived";
    static final String DETECTIONS = "vt_detections.tsv";
    static final String SELECTION = "vt_selection.tsv";
    static final String SUMMARY = "vt_selection.txt";
    static final String INVENTORY = "vidtimit_inventory.tsv";
    /** Префикс ключей кадров VidTIMIT в наборах кадров и классах («vt/человек/фраза/кадр»). */
    static final String KEY_PREFIX = "vt/";
    static final String[] VARIANT = {"В2", "В4"};
    static final String OK = "цель";
    static final String NEAR = "замена";
    static final String NONE = "нет кадра";

    private VidTimitData() {
    }

    /** Фраза: человек, сессия (1–3), имя каталога. */
    record Phrase(String person, int session, String name) {}

    /** Строка описи: вариант, фраза, n, цель, взятый кадр (−1 – нет), r взятого (NaN – нет), причина. */
    record Pick(String variant, Phrase phrase, int n, int target, int taken, double r, String reason) {
        String key(List<String> names) {
            return KEY_PREFIX + phrase.person() + "/" + phrase.name() + "/" + names.get(taken - 1);
        }
    }

    /** Цели фразы из n кадров (1-based): (В2) – ⌈n/2⌉; (В4) – ⌈n/3⌉, ⌈2n/3⌉. */
    static int[] targets(int n, boolean four) {
        if (n < 1) throw new IllegalArgumentException("VidTIMIT: пустая фраза");
        return four ? new int[] {(n + 2) / 3, (2 * n + 2) / 3} : new int[] {(n + 1) / 2};
    }

    /**
     * Ближайший к цели годный кадр 1..n, не из taken: расстояние 0, 1, 2, …, при равном – более ранний; −1 – годного нет.
     */
    static int nearest(int n, int target, IntPredicate ok, Set<Integer> taken) {
        for (int d = 0; d < n; d++) {
            for (int i : d == 0 ? new int[] {target} : new int[] {target - d, target + d}) {
                if (i < 1 || i > n || taken.contains(i)) continue;
                if (ok.test(i)) return i;
            }
        }
        return -1;
    }

    /** Фразы по описи: сессия 1 – две последние по имени, сессии 2 и 3 – все; без head*. Порядок – человек, сессия, имя. */
    static List<Phrase> phrases(List<String> inventory) {
        Map<String, Map<Integer, TreeSet<String>>> by = new TreeMap<>();
        for (String l : inventory) {
            if (l.startsWith("person\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            int ses = Integer.parseInt(c[1]);
            if (ses < 1 || ses > 3 || c[2].startsWith("head") || c[2].startsWith("#")) continue;
            by.computeIfAbsent(c[0], k -> new TreeMap<>()).computeIfAbsent(ses, k -> new TreeSet<>()).add(c[2]);
        }
        List<Phrase> out = new ArrayList<>();
        for (Map.Entry<String, Map<Integer, TreeSet<String>>> e : by.entrySet()) {
            for (int ses = 1; ses <= 3; ses++) {
                TreeSet<String> ns = e.getValue().get(ses);
                if (ns == null) throw new IllegalStateException("VidTIMIT: у " + e.getKey() + " нет фраз сессии " + ses);
                List<String> list = new ArrayList<>(ns);
                if (ses == 1) list = list.subList(Math.max(0, list.size() - 2), list.size());
                for (String n : list) out.add(new Phrase(e.getKey(), ses, n));
            }
        }
        return out;
    }

    /** Отбор по фразам: frames – имена кадров фразы по возрастанию, ok – годен ли кадр i (1-based). */
    static List<Pick> select(String variant, List<Phrase> phrases, Map<Phrase, List<String>> frames, Map<Phrase, IntPredicate> ok,
                             java.util.function.ToDoubleBiFunction<Phrase, Integer> rOf) {
        boolean four = variant.equals(VARIANT[1]);
        List<Pick> out = new ArrayList<>();
        for (Phrase p : phrases) {
            int n = frames.get(p).size();
            Set<Integer> taken = new TreeSet<>();
            for (int t : targets(n, four)) {
                int i = nearest(n, t, ok.get(p), taken);
                if (i > 0) taken.add(i);
                double r = i > 0 ? rOf.applyAsDouble(p, i) : Double.NaN;
                out.add(new Pick(variant, p, n, t, i, r, i < 0 ? NONE : i == t ? OK : NEAR));
            }
        }
        return out;
    }

    /** Классы W: человек → ключи взятых кадров варианта (сессия, фраза, кадр по порядку описи). */
    static List<List<String>> classes(List<Pick> picks, Map<Phrase, List<String>> frames) {
        Map<String, List<String>> by = new TreeMap<>();
        for (Pick k : picks) {
            List<String> c = by.computeIfAbsent(k.phrase().person(), x -> new ArrayList<>());
            if (k.taken() > 0) c.add(k.key(frames.get(k.phrase())));
        }
        return new ArrayList<>(by.values());
    }

    /** Строки описи отбора (без коммита – файл не меняется от коммитов кода). */
    static String table(double rLimit, List<Pick> picks) {
        StringBuilder t = new StringBuilder();
        t.append(String.format(Locale.ROOT, "# Отбор кадров VidTIMIT (VidTimitData vt-frames): R = %s; YuNet %s, два прохода; кадр «а» 92×112%n", rLimit,
                FeretData.SCORE));
        t.append("variant\tperson\tsession\tphrase\tn\ttarget\ttaken\tr\treason\n");
        for (Pick k : picks) {
            t.append(String.format(Locale.ROOT, "%s\t%s\t%d\t%s\t%d\t%d\t%d\t%s\t%s%n", k.variant(), k.phrase().person(), k.phrase().session(), k.phrase().name(),
                    k.n(), k.target(), k.taken(), Double.isNaN(k.r()) ? "-" : String.format(Locale.ROOT, "%.4f", k.r()), k.reason()));
        }
        return t.toString();
    }

    // ---------------------------------------------------------------- данные на диске

    static Path dir(SettingsStore settings) {
        String d = settings.loadFacesVidTimitDir();
        if (d == null) throw new IllegalStateException("VidTIMIT: не задан faces.vidtimit.dir / FACES_VIDTIMIT_DIR");
        return Paths.get(d);
    }

    static List<String> frameNames(Path dir, Phrase p) throws IOException {
        try (Stream<Path> s = Files.list(dir.resolve(p.person()).resolve("video").resolve(p.name()))) {
            return s.filter(Files::isRegularFile).map(f -> f.getFileName().toString()).sorted().toList();
        }
    }

    static String detKey(Phrase p, String frame) {
        return p.person() + "/" + p.name() + "/" + frame;
    }

    static Path png(Path dir, String key) {
        return dir.resolve(DERIVED).resolve("a").resolve(key.substring(KEY_PREFIX.length()).replace('/', '_') + ".png");
    }

    /** Кэш детекций: ключ → строка YuNet (null – лица нет); оборванная последняя строка пропускается. */
    static Map<String, double[]> readDetections(Path file) throws IOException {
        Map<String, double[]> done = new HashMap<>();
        if (!Files.exists(file)) return done;
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (c.length != 17) continue;
            double[] row = null;
            if (c[1].equals("1")) {
                row = new double[15];
                for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[2 + i]);
            }
            done.put(c[0], row);
        }
        return done;
    }

    /** Данные для отбора: имена кадров фраз; при detector != null – недостающие детекции считаются и дописываются в кэш. */
    static final class Dets {
        final Map<Phrase, List<String>> frames = new LinkedHashMap<>();
        final Map<String, double[]> rows;
        final Path dir;
        final double rLimit;
        final FaceDetectorYN detector;
        final Writer out;
        int computed;

        Dets(Path dir, List<Phrase> phrases, double rLimit, FaceDetectorYN detector, Writer out) throws IOException {
            this.dir = dir;
            this.rLimit = rLimit;
            this.detector = detector;
            this.out = out;
            this.rows = readDetections(dir.resolve(DERIVED).resolve(DETECTIONS));
            for (Phrase p : phrases) frames.put(p, frameNames(dir, p));
        }

        double[] row(Phrase p, int i) {
            String k = detKey(p, frames.get(p).get(i - 1));
            if (rows.containsKey(k)) return rows.get(k);
            if (detector == null) throw new IllegalStateException("VidTIMIT: нет детекции " + k + " в кэше");
            try {
                Mat color = Imgcodecs.imread(dir.resolve(p.person()).resolve("video").resolve(p.name()).resolve(frames.get(p).get(i - 1)).toString(),
                        Imgcodecs.IMREAD_COLOR);
                double[] row = null;
                if (!color.empty()) {
                    double[] first = YunetTwoPass.firstPass(detector, color, 1.0);
                    if (first != null) {
                        double[] second = YunetTwoPass.secondPass(detector, color, new double[] {first[0], first[1], first[2], first[3]});
                        row = second != null ? second : first;
                        Mat a = FeretData.frameA(color, row);
                        FeretData.writePng(png(dir, KEY_PREFIX + k), a);
                        a.release();
                    }
                }
                color.release();
                StringBuilder l = new StringBuilder(k).append('\t').append(row == null ? "0" : "1");
                for (int j = 0; j < 15; j++) l.append('\t').append(row == null ? "" : Double.toString(row[j]));
                out.write(l.append('\n').toString());
                out.flush();
                rows.put(k, row);
                computed++;
                return row;
            } catch (IOException e) {
                throw new IllegalStateException("VidTIMIT: " + e, e);
            }
        }

        boolean ok(Phrase p, int i) {
            double[] r = row(p, i);
            return r != null && Math.abs(FeiDataset.pose(r)) <= rLimit;
        }

        double r(Phrase p, int i) {
            String k = detKey(p, frames.get(p).get(i - 1));
            double[] row = rows.get(k);
            return row == null ? Double.NaN : FeiDataset.pose(row);
        }

        List<Pick> select(String variant, List<Phrase> phrases) {
            Map<Phrase, IntPredicate> ok = new HashMap<>();
            for (Phrase p : phrases) ok.put(p, i -> ok(p, i));
            return VidTimitData.select(variant, phrases, frames, ok, this::r);
        }
    }

    /** Отбор обоих вариантов по кэшу детекций (detector == null – только кэш). */
    static List<Pick> selectAll(Dets d, List<Phrase> phrases) {
        List<Pick> all = new ArrayList<>();
        for (String v : VARIANT) all.addAll(d.select(v, phrases));
        return all;
    }

    // ---------------------------------------------------------------- vt-frames

    static void frames(List<String> selLines, SettingsStore settings) throws IOException {
        Path dir = dir(settings);
        double rLimit = XBaseEval.poseLimit(selLines);
        List<Phrase> phrases = phrases(Files.readAllLines(dir.resolve(INVENTORY), StandardCharsets.UTF_8));
        Files.createDirectories(dir.resolve(DERIVED).resolve("a"));
        Path detFile = dir.resolve(DERIVED).resolve(DETECTIONS);
        FaceDetectorYN det = OwnDetectorThreshold.create(Paths.get(settings.loadFacesModelsDir()), FeretData.SCORE);
        long t0 = System.nanoTime();
        List<Pick> picks;
        int computed;
        try (Writer w = Files.newBufferedWriter(detFile, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            Dets d = new Dets(dir, phrases, rLimit, det, w);
            picks = selectAll(d, phrases);
            computed = d.computed;
        }
        double sec = (System.nanoTime() - t0) / 1e9;
        String table = table(rLimit, picks);
        // Повтор отбора только по кэшу – побайтно тот же.
        String again = table(rLimit, selectAll(new Dets(dir, phrases, rLimit, null, null), phrases));
        if (!again.equals(table)) throw new IllegalStateException("VidTIMIT: повтор отбора по кэшу не совпал");
        Path sel = dir.resolve(DERIVED).resolve(SELECTION);
        Path tmp = sel.resolveSibling(SELECTION + ".tmp");
        Files.writeString(tmp, table, StandardCharsets.UTF_8);
        Files.move(tmp, sel, StandardCopyOption.REPLACE_EXISTING);
        Dets d0 = new Dets(dir, phrases, rLimit, null, null);
        for (Pick k : picks) if (k.taken() > 0 && !Files.exists(png(dir, k.key(d0.frames.get(k.phrase()))))) {
            throw new IllegalStateException("VidTIMIT: нет кадра " + png(dir, k.key(d0.frames.get(k.phrase()))));
        }
        StringBuilder t = new StringBuilder("Отбор кадров VidTIMIT для пула W (XBaseEval vt-frames)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(dir).append(" (опись ").append(INVENTORY).append("); R = ").append(rLimit).append(" (fei_selection.tsv)\n");
        t.append("Фразы: сессия 1 – две последние по имени, сессии 2 и 3 – обе; повороты головы не берутся. (В2) – кадр ⌈n/2⌉, (В4) – ⌈n/3⌉ и ⌈2n/3⌉; "
                + "негодный – ближайший годный (|r| ≤ R), при равенстве – более ранний.\n");
        t.append(String.format(Locale.ROOT, "Фраз %d, людей %d; детекций посчитано в этом запуске %d (%.0f с), в кэше всего %d.%n", phrases.size(),
                phrases.stream().map(Phrase::person).distinct().count(), computed, sec, readDetections(detFile).size()));
        t.append("Файл отбора: ").append(sel).append(", SHA-256 ").append(DuelSet.sha256(sel)).append(" (повтор по кэшу – побайтно тот же)\n\n");
        t.append("вариант | сессия | кадров нужно | взято: цель | замена | нет кадра | людей с кадрами\n");
        for (String v : VARIANT) {
            for (int ses = 1; ses <= 3; ses++) {
                int need = 0;
                int[] c = new int[3];
                Set<String> ps = new TreeSet<>();
                for (Pick k : picks) {
                    if (!k.variant().equals(v) || k.phrase().session() != ses) continue;
                    need++;
                    c[k.reason().equals(OK) ? 0 : k.reason().equals(NEAR) ? 1 : 2]++;
                    if (k.taken() > 0) ps.add(k.phrase().person());
                }
                t.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d | %d | %d%n", v, ses, need, c[0], c[1], c[2], ps.size()));
            }
        }
        t.append("\nЗамены и отказы:\n");
        int z = 0;
        for (Pick k : picks) {
            if (k.reason().equals(OK)) continue;
            z++;
            t.append(String.format(Locale.ROOT, "%s %s сессия %d %s: n %d, цель %d, взят %d, %s%n", k.variant(), k.phrase().person(), k.phrase().session(),
                    k.phrase().name(), k.n(), k.target(), k.taken(), k.reason()));
        }
        if (z == 0) t.append("нет\n");
        double[] rs = picks.stream().filter(k -> k.taken() > 0).mapToDouble(k -> Math.abs(k.r())).sorted().toArray();
        t.append(String.format(Locale.ROOT, "|r| взятых: медиана %.4f, наибольшее %.4f%n", rs[rs.length / 2], rs[rs.length - 1]));
        FarMethods.write(dir.resolve(DERIVED).resolve(SUMMARY), t);
        System.out.print(t);
    }

    /** Отбор варианта из файла описи (для pool-vt): классы по людям и кадры; ключ – KEY_PREFIX + человек/фраза/кадр. */
    static final class Loaded {
        final List<List<String>> classes;
        final Map<String, Mat> frames;
        final String sha256;
        final int persons;
        final int images;

        Loaded(List<List<String>> classes, Map<String, Mat> frames, String sha256) {
            this.classes = classes;
            this.frames = frames;
            this.sha256 = sha256;
            this.persons = (int) classes.stream().filter(c -> !c.isEmpty()).count();
            this.images = classes.stream().mapToInt(List::size).sum();
        }
    }

    /** Классы варианта по файлу описи; кадры – PNG из derived\a (серые 92×112). */
    static Loaded load(Path dir, String variant) throws IOException {
        Path sel = dir.resolve(DERIVED).resolve(SELECTION);
        if (!Files.exists(sel)) throw new IllegalStateException("VidTIMIT: нет " + sel + " (сначала vt-frames)");
        Map<String, List<String>> by = new TreeMap<>();
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String l : Files.readAllLines(sel, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("variant\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            if (!c[0].equals(variant)) continue;
            List<String> cls = by.computeIfAbsent(c[1], k -> new ArrayList<>());
            int taken = Integer.parseInt(c[6]);
            if (taken < 1) continue;
            List<String> names = frameNames(dir, new Phrase(c[1], Integer.parseInt(c[2]), c[3]));
            String key = KEY_PREFIX + c[1] + "/" + c[3] + "/" + names.get(taken - 1);
            cls.add(key);
            Mat m = Imgcodecs.imread(png(dir, key).toString(), Imgcodecs.IMREAD_GRAYSCALE);
            if (m.empty()) throw new IOException("VidTIMIT: нет кадра " + png(dir, key));
            fr.put(key, m);
        }
        if (by.isEmpty()) throw new IllegalStateException("VidTIMIT: в " + sel + " нет варианта " + variant);
        return new Loaded(new ArrayList<>(by.values()), fr, DuelSet.sha256(sel));
    }
}
