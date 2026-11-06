package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import svd.recognizer.storage.SettingsStore;

/**
 * Итоговая проверка на ChokePoint (постановка Жорика 06.11, редакция 2). Этап 1 – cp-inventory: опись
 * проходов только по файлам разметки (номер человека в кадре; точки глаз не читаются), проверка имён
 * файлов кадров без их открытия, деление 25 общих людей пополам, сверка узлов таблиц порога для N = 12, 13.
 * Всё производное – только в каталоге базы (derived\, reports\).
 *
 * @author ssv
 */
final class XBaseChoke {

    private XBaseChoke() {
    }

    static final long SPLIT_SEED_OFFSET = 12000;
    static final String[] DIRS = {"P1E", "P1L", "P2E", "P2L"};
    /** Камера анфаса (README базы, «Sequence with the most frontal view»): направление → камера для S1–S4. */
    static final Map<String, int[]> FRONTAL = Map.of(
            "P1E", new int[] {1, 2, 3, 1},
            "P1L", new int[] {1, 2, 3, 1},
            "P2E", new int[] {3, 2, 1, 2},
            "P2L", new int[] {1, 2, 3, 2});
    static final List<String> ONLY_P2 = List.of("0002", "0028", "0029", "0030");
    static final int[] GALLERY_N = {12, 13};
    /** Узлы порога для N галереи, записанные в постановке (разд. 4); расхождение с кодом – остановка. */
    static final double T37 = -9.42;
    static final double T_OLD = -9.55;
    static final double T_SFACE = 0.46;

    private static final Pattern FRAME = Pattern.compile("<frame number=\"(\\d+)\"");
    private static final Pattern PERSON = Pattern.compile("<person id=\"(\\d+)\"");

    /** Запись разметки: человек в кадре; eyes – есть ли в записи обе точки глаз (значения не читаются). */
    record Rec(int frame, String person, boolean eyes) {}

    /** Один человек в одной последовательности-камере: номера кадров по возрастанию. */
    record Key(String dir, int s, String part, String person) implements Comparable<Key> {
        @Override
        public int compareTo(Key o) {
            int c = dir.compareTo(o.dir);
            if (c == 0) c = Integer.compare(s, o.s);
            if (c == 0) c = part.compareTo(o.part);
            return c != 0 ? c : person.compareTo(o.person);
        }
    }

    /** Разбор разметки: только номера кадров и людей; координаты глаз не извлекаются. */
    static List<Rec> parse(List<String> xml) {
        List<Rec> out = new ArrayList<>();
        int frame = -1;
        String person = null;
        boolean left = false;
        boolean right = false;
        for (String l : xml) {
            Matcher f = FRAME.matcher(l);
            if (f.find()) frame = Integer.parseInt(f.group(1));
            Matcher p = PERSON.matcher(l);
            if (p.find()) {
                if (frame < 0) throw new IllegalStateException("cp-inventory: <person> вне <frame>: " + l);
                if (person != null) throw new IllegalStateException("cp-inventory: вложенный <person>: " + l);
                person = p.group(1);
                left = false;
                right = false;
            }
            if (l.contains("<leftEye ")) left = true;
            if (l.contains("<rightEye ")) right = true;
            if (l.contains("</person>")) {
                if (person == null) throw new IllegalStateException("cp-inventory: лишний </person>: " + l);
                out.add(new Rec(frame, person, left && right));
                person = null;
            }
        }
        if (person != null) throw new IllegalStateException("cp-inventory: незакрытый <person>");
        return out;
    }

    /** Деление постановки: перестановка каждой группы одним генератором, первая половина – 11 из 21 и 2 из 4. */
    static List<List<String>> split(List<String> g28, List<String> g45, long seed) {
        if (g28.size() != 21 || g45.size() != 4) throw new IllegalStateException("cp-inventory: группы " + g28.size() + " и " + g45.size() + " вместо 21 и 4");
        List<String> a = new ArrayList<>(new TreeSet<>(g28));
        List<String> b = new ArrayList<>(new TreeSet<>(g45));
        Random rnd = new Random(seed);
        Collections.shuffle(a, rnd);
        Collections.shuffle(b, rnd);
        List<String> h1 = new ArrayList<>(a.subList(0, 11));
        h1.addAll(b.subList(0, 2));
        List<String> h2 = new ArrayList<>(a.subList(11, 21));
        h2.addAll(b.subList(2, 4));
        Collections.sort(h1);
        Collections.sort(h2);
        return List.of(h1, h2);
    }

    /** Имя файла разметки: P1E_S1_C1 или P2E_S1_C3.1. */
    static String seqName(String dir, int s, int c, String part) {
        return dir + "_S" + s + "_C" + c + (part.isEmpty() ? "" : "." + part);
    }

    private static String sha(String text) throws IOException {
        Path f = Files.createTempFile("cp", ".tsv");
        try {
            Files.write(f, text.getBytes(StandardCharsets.UTF_8));
            return FrameB.sha256(f);
        } finally {
            Files.delete(f);
        }
    }

    private static int median(List<Integer> v) {
        List<Integer> s = new ArrayList<>(v);
        Collections.sort(s);
        return s.isEmpty() ? 0 : s.get(s.size() / 2);
    }

    static void inventory(SettingsStore settings) throws IOException {
        String d = System.getenv("FACES_CHOKEPOINT_DIR");
        String t37 = System.getenv("FACES_P4_TABLE");
        String tOld = System.getenv("FACES_CP_TABLE_OLD");
        if (d == null || t37 == null || tOld == null) throw new IllegalStateException("cp-inventory: задайте FACES_CHOKEPOINT_DIR, FACES_P4_TABLE, FACES_CP_TABLE_OLD");
        Path root = Paths.get(d);
        Path gt = root.resolve("groundtruth");
        Path frames = root.resolve("frames");
        Path derived = root.resolve("derived");
        Path reports = root.resolve("reports");
        Files.createDirectories(derived);
        Files.createDirectories(reports);

        // Кадры прохода на камере анфаса и число кадров человека на всех камерах этой последовательности.
        Map<Key, List<Integer>> front = new TreeMap<>();
        Map<Key, Integer> anyCam = new TreeMap<>();
        Map<Key, Integer> noEyes = new TreeMap<>();
        Map<Key, Integer> missing = new TreeMap<>();
        int xmlFiles = 0;
        long records = 0;
        for (String dir : DIRS) {
            boolean p2 = dir.startsWith("P2");
            for (String part : p2 ? List.of("1", "2") : List.of("")) {
                for (int s = 1; s <= 4; s++) {
                    int fc = FRONTAL.get(dir)[s - 1];
                    for (int c = 1; c <= 3; c++) {
                        String seq = seqName(dir, s, c, part);
                        List<Rec> rs = parse(Files.readAllLines(gt.resolve(seq + ".xml"), StandardCharsets.UTF_8));
                        xmlFiles++;
                        records += rs.size();
                        for (Rec r : rs) {
                            Key k = new Key(dir, s, part, r.person());
                            anyCam.merge(k, 1, Integer::sum);
                            if (c != fc) continue;
                            front.computeIfAbsent(k, x -> new ArrayList<>()).add(r.frame());
                            if (!r.eyes()) noEyes.merge(k, 1, Integer::sum);
                            // Только наличие файла кадра; файл не открывается.
                            if (!Files.exists(frames.resolve(seq).resolve(String.format(Locale.ROOT, "%08d.jpg", r.frame())))) missing.merge(k, 1, Integer::sum);
                        }
                    }
                }
            }
        }

        // Люди и части портала 2.
        TreeSet<String> p1 = new TreeSet<>();
        Map<String, TreeSet<String>> partsOf = new TreeMap<>();
        for (Key k : anyCam.keySet()) {
            if (k.dir().startsWith("P1")) p1.add(k.person());
            else partsOf.computeIfAbsent(k.person(), x -> new TreeSet<>()).add(k.part());
        }
        List<String> g28 = new ArrayList<>();
        List<String> g45 = new ArrayList<>();
        for (String p : p1) {
            TreeSet<String> ps = partsOf.get(p);
            if (ps == null || ps.size() != 1) throw new IllegalStateException("cp-inventory: человек " + p + " портала 1 в частях портала 2: " + ps);
            (ps.first().equals("1") ? g28 : g45).add(p);
        }
        TreeSet<String> only2 = new TreeSet<>(partsOf.keySet());
        only2.removeAll(p1);
        if (!new ArrayList<>(only2).equals(ONLY_P2)) throw new IllegalStateException("cp-inventory: только в портале 2 – " + only2 + " вместо " + ONLY_P2);
        long seed = settings.loadFacesSeed() + SPLIT_SEED_OFFSET;
        List<List<String>> halves = split(g28, g45, seed);
        if (halves.get(0).size() != 13 || halves.get(1).size() != 12) throw new IllegalStateException("cp-inventory: половины не 13 и 12");

        // Опись проходов.
        StringBuilder pt = new StringBuilder("dir\ts\tpart\tdate\tperson\tfrontal_cam\tframes_frontal\tframes_all_cams\tfirst\tlast\tno_eyes\tmissing_files\n");
        Map<String, Integer> passesPer = new TreeMap<>();
        Map<String, List<Integer>> frontPer = new TreeMap<>();
        List<String> empty = new ArrayList<>();
        int gaps = 0;
        for (Map.Entry<Key, Integer> e : anyCam.entrySet()) {
            Key k = e.getKey();
            List<Integer> fr = front.getOrDefault(k, List.of());
            String date = k.dir().startsWith("P1") ? "2010-04-16" : k.part().equals("1") ? "2010-05-14" : "2010-05-31";
            pt.append(String.join("\t", k.dir(), "S" + k.s(), k.part().isEmpty() ? "-" : k.part(), date, k.person(), "C" + FRONTAL.get(k.dir())[k.s() - 1], Integer.toString(fr.size()), e.getValue().toString(),
                    fr.isEmpty() ? "-" : Integer.toString(Collections.min(fr)), fr.isEmpty() ? "-" : Integer.toString(Collections.max(fr)),
                    Integer.toString(noEyes.getOrDefault(k, 0)), Integer.toString(missing.getOrDefault(k, 0)))).append('\n');
            passesPer.merge(k.dir() + " " + k.person(), 1, Integer::sum);
            frontPer.computeIfAbsent(k.dir(), x -> new ArrayList<>()).add(fr.size());
            if (fr.isEmpty()) empty.add(k.dir() + "_S" + k.s() + (k.part().isEmpty() ? "" : "." + k.part()) + " " + k.person());
            List<Integer> sorted = new ArrayList<>(fr);
            Collections.sort(sorted);
            for (int i = 1; i < sorted.size(); i++) if (sorted.get(i) - sorted.get(i - 1) > 30) gaps++;
        }

        // Деление.
        StringBuilder st = new StringBuilder("person\tgroup\thalf\tpass1\tpass2\n");
        for (String p : p1) {
            int h = halves.get(0).contains(p) ? 1 : 2;
            st.append(p).append('\t').append(g45.contains(p) ? "45" : "28").append('\t').append(h).append('\t').append(h == 1 ? "gallery" : "ctrl").append('\t').append(h == 1 ? "ctrl" : "gallery").append('\n');
        }
        for (String p : only2) st.append(p).append("\tP2\t-\tctrl\tctrl\n");

        // Узлы порогов.
        Map<Integer, XBasePart4.TRow> tb37 = XBasePart4.readTable(Paths.get(t37));
        Map<Integer, XBasePart4.TRow> tbOld = XBasePart4.readTable(Paths.get(tOld));
        StringBuilder tt = new StringBuilder("N\tt37\tt37_node\tt_old\tt_old_node\tt_sface\n");
        for (int n : GALLERY_N) {
            double[] a = XBaseFinalReport.tOf(tb37, n);
            double[] b = XBaseFinalReport.tOf(tbOld, n);
            double c = XBaseFinalReport.sfaceT(n);
            if (a[0] != T37 || b[0] != T_OLD || c != T_SFACE) throw new IllegalStateException("cp-inventory: N = " + n + " – пороги " + a[0] + ", " + b[0] + ", " + c + " не совпали с постановкой");
            tt.append(String.format(Locale.ROOT, "%d\t%.2f\t%d\t%.2f\t%d\t%.2f%n", n, a[0], (int) a[1], b[0], (int) b[1], c).replace("\r", ""));
        }

        Files.write(derived.resolve("cp_passes.tsv"), pt.toString().getBytes(StandardCharsets.UTF_8));
        Files.write(derived.resolve("cp_split.tsv"), st.toString().getBytes(StandardCharsets.UTF_8));
        Files.write(derived.resolve("cp_thresholds.tsv"), tt.toString().getBytes(StandardCharsets.UTF_8));

        // Сводка.
        StringBuilder r = new StringBuilder("ChokePoint, этап 1 (cp-inventory): опись по разметке, деление, узлы порогов. Кадры не открывались, точки глаз не читались.\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append("\n\n");
        r.append("Файлов разметки ").append(xmlFiles).append(", записей человек-кадр ").append(records).append('\n');
        r.append("Люди: портал 1 – ").append(p1.size()).append(", только портал 2 – ").append(only2).append("; группа 28 дней – ").append(g28.size()).append(", группа 45 дней – ").append(g45).append('\n');
        r.append("Части портала 2 у 0002 – ").append(partsOf.get("0002")).append(", 0028 – ").append(partsOf.get("0028")).append(", 0029 – ").append(partsOf.get("0029")).append(", 0030 – ").append(partsOf.get("0030")).append('\n');
        for (String dir : DIRS) {
            List<Integer> v = frontPer.getOrDefault(dir, List.of());
            r.append(String.format(Locale.ROOT, "%s: проходов %d, кадров на камере анфаса: мин %d, медиана %d, макс %d, всего %d%n", dir, v.size(), v.isEmpty() ? 0 : Collections.min(v), median(v), v.isEmpty() ? 0 : Collections.max(v), v.stream().mapToInt(Integer::intValue).sum()));
        }
        r.append("Проходов без единого кадра на камере анфаса: ").append(empty.size()).append(' ').append(empty).append('\n');
        r.append("Записей не с двумя точками глаз на камере анфаса (годность не меняют): ").append(noEyes).append('\n');
        r.append("Файлов кадров камеры анфаса нет на диске: ").append(missing.values().stream().mapToInt(Integer::intValue).sum()).append('\n');
        r.append("Разрывов > 30 кадров внутри прохода на камере анфаса: ").append(gaps).append('\n');
        r.append("\nПроходы по людям (P1E, P1L, P2E, P2L):\n");
        TreeSet<String> all = new TreeSet<>(p1);
        all.addAll(only2);
        int[][] p2Count = new int[3][2];
        for (String p : all) {
            int[] c = new int[4];
            for (int i = 0; i < 4; i++) c[i] = passesPer.getOrDefault(DIRS[i] + " " + p, 0);
            r.append(p).append(' ').append(Arrays.toString(c)).append('\n');
            int cnt = c[2] + c[3];
            if (halves.get(0).contains(p)) p2Count[0][0] += cnt;
            else if (halves.get(1).contains(p)) p2Count[1][0] += cnt;
            else p2Count[2][0] += cnt;
        }
        r.append(String.format(Locale.ROOT, "%nДеление, seed %d: половина 1 (%d) %s; половина 2 (%d) %s%n", seed, halves.get(0).size(), halves.get(0), halves.get(1).size(), halves.get(1)));
        r.append(String.format(Locale.ROOT, "Попыток портала 2: половина 1 – %d, половина 2 – %d, только портал 2 – %d%n", p2Count[0][0], p2Count[1][0], p2Count[2][0]));
        r.append(String.format(Locale.ROOT, "Проход 1: свои – %d попыток (13 чел.), контрольные чужие – %d (16 чел.). Проход 2: свои – %d (12 чел.), контрольные чужие – %d (17 чел.)%n",
                p2Count[0][0], p2Count[1][0] + p2Count[2][0], p2Count[1][0], p2Count[0][0] + p2Count[2][0]));
        r.append("\nУзлы порогов (по коду: XBaseFinalReport.tOf, sfaceT):\n").append(tt);
        r.append("Таблица (37) – ").append(t37).append(", SHA-256 ").append(FrameB.sha256(Paths.get(t37))).append('\n');
        r.append("Прежняя таблица – ").append(tOld).append(", SHA-256 ").append(FrameB.sha256(Paths.get(tOld))).append('\n');
        r.append("\nSHA-256 описи cp_passes.tsv ").append(sha(pt.toString())).append('\n');
        r.append("SHA-256 деления cp_split.tsv ").append(sha(st.toString())).append('\n');
        r.append("SHA-256 узлов cp_thresholds.tsv ").append(sha(tt.toString())).append('\n');
        Files.write(reports.resolve("cp_inventory.txt"), r.toString().getBytes(StandardCharsets.UTF_8));
        System.out.print(r);
    }
}
