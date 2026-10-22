package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№2 (дуэль; PLAN.md п. 5.5, (1а), Д1–Д13): набор фото GT для коллеги – режим duel-set.
 * <p>
 * Снимки GT после отбора по позе |r| ≤ R (fei_selection.tsv), даты – gt_sessions.tsv экспорта, рамки – YuNet 0,7 экспорта (detections.tsv).
 * Свои – 10 человек с ≥ 2 датами, у которых в дне галереи (XBaseEval.galleryDate) ≥ 7 снимков, а в «другой дате» (наибольшее число снимков
 * среди остальных дат, при равенстве – более ранняя) ≥ 3; выбор и снимки – seed faces.seed + 10100: галерея – 5 снимков дня галереи, пробы –
 * 2 снимка того же дня вне галереи и 3 снимка другой даты. Чужие – 10 человек вне своих, по 2 снимка из всех отобранных (seed + 10101).
 * Гости – 10 человек вне своих и чужих, по одному снимку (seed + 10103). Пробы перемешаны (seed + 10102), имена Q001…Q070.
 * <p>
 * Кадры – {@link ProkaevFrames#colorFrame} (то же окно, что у кадра метода Прокаева: К-лицо H = h_рамки / 0,70 и К-обычный), 400 × 600,
 * цвет исходный, PNG. Набор для передачи – каталог FACES_DUEL_DIR (вне git): К-лицо\ и К-обычный\ (P01…P10\P01_1…P01_5.png, пробы\Q001…Q070.png,
 * гости\G01…G10.png), таблицы для заполнения (CSV UTF-8 с BOM), README.txt. Ключ key.tsv, листы проверки и опись – reports/faces/duel/.
 *
 * @author ssv
 */
final class DuelSet {

    static final long OWN_SEED = 10100;
    static final long STRANGER_SEED = 10101;
    static final long PERM_SEED = 10102;
    static final long GUEST_SEED = 10103;
    static final int OWN = 10;
    static final int GALLERY = 5;
    static final int SAME = 2;
    static final int OTHER = 3;
    static final int STRANGERS = 10;
    static final int STRANGER_IMAGES = 2;
    static final int GUESTS = 10;

    static final String GALLERY_ROLE = "галерея";
    static final String SAME_ROLE = "свой-тот-же-день";
    static final String OTHER_ROLE = "свой-другой-день";
    static final String STRANGER_ROLE = "чужой";
    static final String GUEST_ROLE = "гость";

    /** Комплекты: каталог, подпись, К-обычный. */
    static final String[] SETS = {"К-лицо", "К-обычный"};
    static final String[] SET_TAGS = {"face", "plain"};
    static final String PROBES_DIR = "пробы";
    static final String GUESTS_DIR = "гости";
    static final String KEY = "key.tsv";

    private DuelSet() {
    }

    /** Строка ключа: код (P01_1, Q001, G01), роль, человек GT, номер даты (0 – без даты), дата EXIF, ключ снимка «человек/снимок». */
    record Item(String code, String role, String person, int date, String exif, String pk) {

        /** Путь кадра внутри комплекта. */
        String rel() {
            if (code.startsWith("P")) return code.substring(0, 3) + "/" + code + ".png";
            if (code.startsWith("Q")) return PROBES_DIR + "/" + code + ".png";
            return GUESTS_DIR + "/" + code + ".png";
        }
    }

    /** Состав набора и сведения для отчёта. */
    static final class Composition {
        final List<Item> items = new ArrayList<>();
        int eligible;
        int multiDate;
        int persons;
        final List<String> skipped = new ArrayList<>();
    }

    /** Дата EXIF снимка по gt_sessions.tsv: «человек/снимок» → строка даты. */
    static Map<String, String> exif(Path file) throws IOException {
        Map<String, String> out = new HashMap<>();
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            out.put(c[0] + "/" + c[1], c[2]);
        }
        return out;
    }

    /** «Другая дата»: наибольшее число снимков среди дат кроме gd (и кроме 0), при равенстве – более ранняя; −1 – нет. */
    static int otherDate(Map<Integer, List<String>> byDate, int gd) {
        int best = -1;
        for (Map.Entry<Integer, List<String>> e : byDate.entrySet()) {
            if (e.getKey() == 0 || e.getKey() == gd) continue;
            if (best < 0 || e.getValue().size() > byDate.get(best).size()) best = e.getKey();
        }
        return best;
    }

    static Composition compose(Path exportDir, List<String> selLines, long seed) throws IOException {
        double rLimit = XBaseEval.poseLimit(selLines);
        Map<String, double[]> det = XBaseMpPts.exportRows(exportDir, "gt");
        Map<String, Integer> sess = XBaseEval.sessions(exportDir.resolve(XBaseEval.SESSIONS));
        Map<String, String> ex = exif(exportDir.resolve(XBaseEval.SESSIONS));
        // Человек → дата → ключи (после отбора по позе), по возрастанию.
        Map<String, Map<Integer, List<String>>> by = new TreeMap<>();
        for (Map.Entry<String, double[]> e : det.entrySet()) {
            if (e.getValue() == null || Math.abs(FeiDataset.pose(e.getValue())) > rLimit) continue;
            String pk = e.getKey();
            Integer d = sess.get(pk);
            if (d == null) throw new IllegalStateException("duel-set: нет даты снимка GT " + pk);
            by.computeIfAbsent(pk.substring(0, pk.indexOf('/')), x -> new TreeMap<>()).computeIfAbsent(d, x -> new ArrayList<>()).add(pk);
        }
        by.values().forEach(m -> m.values().forEach(Collections::sort));
        Composition c = new Composition();
        c.persons = by.size();
        List<String> eligible = new ArrayList<>();
        for (Map.Entry<String, Map<Integer, List<String>>> e : by.entrySet()) {
            Map<Integer, List<String>> bd = e.getValue();
            long dates = bd.keySet().stream().filter(d -> d > 0).count();
            if (dates < 2) continue;
            c.multiDate++;
            int gd = XBaseEval.galleryDate(bd);
            int od = otherDate(bd, gd);
            if (bd.get(gd).size() >= GALLERY + SAME && od > 0 && bd.get(od).size() >= OTHER) eligible.add(e.getKey());
        }
        c.eligible = eligible.size();
        if (eligible.size() < OWN) throw new IllegalStateException("duel-set: подходящих своих " + eligible.size() + " < " + OWN);
        // Свои: перестановка подходящих и снимки – одним генератором seed + 10100, по порядку P01…P10.
        Random rnd = new Random(seed + OWN_SEED);
        Collections.shuffle(eligible, rnd);
        List<String> own = eligible.subList(0, OWN);
        List<Item> probes = new ArrayList<>();
        for (int i = 0; i < OWN; i++) {
            String p = own.get(i);
            Map<Integer, List<String>> bd = by.get(p);
            int gd = XBaseEval.galleryDate(bd);
            int od = otherDate(bd, gd);
            List<String> g = new ArrayList<>(bd.get(gd));
            Collections.shuffle(g, rnd);
            List<String> o = new ArrayList<>(bd.get(od));
            Collections.shuffle(o, rnd);
            String code = String.format(Locale.ROOT, "P%02d", i + 1);
            for (int j = 0; j < GALLERY; j++) c.items.add(new Item(code + "_" + (j + 1), GALLERY_ROLE, p, gd, ex.get(g.get(j)), g.get(j)));
            for (int j = 0; j < SAME; j++) probes.add(new Item(null, SAME_ROLE, p, gd, ex.get(g.get(GALLERY + j)), g.get(GALLERY + j)));
            for (int j = 0; j < OTHER; j++) probes.add(new Item(null, OTHER_ROLE, p, od, ex.get(o.get(j)), o.get(j)));
        }
        // Чужие: люди вне своих (по возрастанию, перестановка seed + 10101), у кого ≥ 2 отобранных снимков; по 2 снимка.
        List<String> rest = new ArrayList<>(by.keySet());
        rest.removeAll(own);
        Random rs = new Random(seed + STRANGER_SEED);
        Collections.shuffle(rest, rs);
        List<String> strangers = new ArrayList<>();
        for (String p : rest) {
            if (strangers.size() == STRANGERS) break;
            List<String> all = new ArrayList<>();
            by.get(p).values().forEach(all::addAll);
            Collections.sort(all);
            if (all.size() < STRANGER_IMAGES) {
                c.skipped.add(p + " (чужой: снимков " + all.size() + ")");
                continue;
            }
            Collections.shuffle(all, rs);
            strangers.add(p);
            for (int j = 0; j < STRANGER_IMAGES; j++) probes.add(new Item(null, STRANGER_ROLE, p, sess.get(all.get(j)), ex.get(all.get(j)), all.get(j)));
        }
        if (strangers.size() < STRANGERS) throw new IllegalStateException("duel-set: чужих " + strangers.size());
        // Гости: люди вне своих и чужих (по возрастанию, перестановка seed + 10103), по одному снимку.
        List<String> gp = new ArrayList<>(by.keySet());
        gp.removeAll(own);
        gp.removeAll(strangers);
        Random rg = new Random(seed + GUEST_SEED);
        Collections.shuffle(gp, rg);
        if (gp.size() < GUESTS) throw new IllegalStateException("duel-set: гостей " + gp.size());
        for (int i = 0; i < GUESTS; i++) {
            String p = gp.get(i);
            List<String> all = new ArrayList<>();
            by.get(p).values().forEach(all::addAll);
            Collections.sort(all);
            String k = all.get(rg.nextInt(all.size()));
            c.items.add(new Item(String.format(Locale.ROOT, "G%02d", i + 1), GUEST_ROLE, p, sess.get(k), ex.get(k), k));
        }
        // Пробы: перемешивание seed + 10102, имена Q001…Q070.
        Collections.shuffle(probes, new Random(seed + PERM_SEED));
        for (int i = 0; i < probes.size(); i++) {
            Item it = probes.get(i);
            c.items.add(new Item(String.format(Locale.ROOT, "Q%03d", i + 1), it.role(), it.person(), it.date(), it.exif(), it.pk()));
        }
        c.items.sort((a, b) -> a.code().compareTo(b.code()));
        return c;
    }

    // ---------------------------------------------------------------- запись

    static String sha256(Path f) throws IOException {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(f)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Ключ: шапка (seed, коммиты) и строки «код, роль, человек, дата, EXIF, исходный файл». */
    static String key(Composition c, long seed, Path exportDir) {
        StringBuilder t = new StringBuilder("# Д-№2 (дуэль): ключ набора GT для коллеги – только у нас, в набор для передачи не кладётся\n");
        t.append("# Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("# Данные: экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        t.append(String.format(Locale.ROOT, "# seed: свои %d, чужие %d, перемешивание %d, гости %d (faces.seed + 10100…10103)%n", seed + OWN_SEED,
                seed + STRANGER_SEED, seed + PERM_SEED, seed + GUEST_SEED));
        t.append("code\trole\tperson\tsession\texif_datetime\tsource\n");
        for (Item it : c.items) t.append(it.code()).append('\t').append(it.role()).append('\t').append(it.person()).append('\t').append(it.date()).append('\t')
                .append(it.exif()).append('\t').append(it.pk()).append(".jpg\n");
        return t.toString();
    }

    /** Чтение ключа (для duel-eval). */
    static List<Item> readKey(Path file) throws IOException {
        List<Item> out = new ArrayList<>();
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("code\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            out.add(new Item(c[0], c[1], c[2], Integer.parseInt(c[3]), c[4], c[5].substring(0, c[5].length() - 4)));
        }
        return out;
    }

    static final String BOM = "﻿";

    /** Таблица для заполнения коллегой (CSV UTF-8 с BOM, разделитель «;»): шапка настроек, затем строки Q001…Q070. */
    static String table(String set) {
        StringBuilder t = new StringBuilder(BOM);
        t.append("Настройки программы (заполнить);\n");
        t.append("k (число осей);\n");
        t.append("l (число ближайших точек);\n");
        t.append("кадр -> вектор (серый / иначе - как);\n");
        t.append("гости (наши G01-G10 / свои - сколько людей и снимков);\n");
        t.append("комплект;").append(set).append('\n');
        t.append('\n');
        t.append("Q;кандидат 1;% 1;кандидат 2;% 2;решение (свой / гость);примечание\n");
        for (int i = 1; i <= 70; i++) t.append(String.format(Locale.ROOT, "Q%03d;;;;;;%n", i));
        return t.toString();
    }

    static String readme() {
        return BOM + """
                Набор снимков лиц для сравнения методов распознавания
                ======================================================

                Источник снимков: Georgia Tech Face Database (http://www.anefian.com/research/face_reco.htm), снимки 640 x 480.

                Два комплекта одних и тех же снимков, кадры 400 x 600 (ширина x высота), PNG без потерь, цвет как в исходнике:
                  К-лицо\\     - вырезка 2:3 вокруг лица (рамка детектора лица, лицо около 70 % высоты кадра), без поворота;
                  К-обычный\\  - полоса 320 x 480 на всю высоту снимка по центру лица, увеличенная до 400 x 600 (лицо не увеличивается).
                В обоих комплектах одно правило для всех кадров, без ручной правки.

                В каждом комплекте:
                  P01\\ ... P10\\  - галерея: 10 человек по 5 снимков (P01_1.png ... P01_5.png);
                  пробы\\         - 70 проб Q001.png ... Q070.png (свои люди из галереи и посторонние, перемешаны);
                  гости\\         - G01.png ... G10.png: 10 посторонних людей по одному снимку (вне галереи и проб) - класс гостей.

                Просьба:
                  1) прогнать программу на каждом комплекте с нашими гостями G01 ... G10 в классе гостей;
                     настройки программы по ходу не менять (если настройку пришлось менять - все 70 проб заново, отдельная таблица);
                  2) заполнить таблицу комплекта (таблица_К-лицо.csv, таблица_К-обычный.csv): для каждой пробы - первый и второй
                     кандидат с процентами и решение (свой из галереи P01 ... P10 или гость), в шапке - настройки программы;
                  3) по желанию - второй прогон с вашими гостями (отдельная таблица, в шапке - сколько людей и снимков).
                """;
    }

    /** Плитка листа: цветной кадр, уменьшенный вдвое, подпись латиницей. */
    static Mat tile(Mat frame, String caption) {
        Mat t = new Mat();
        Imgproc.resize(frame, t, new Size(ProkaevFrames.W / 4.0, ProkaevFrames.H / 4.0), 0, 0, Imgproc.INTER_AREA);
        Mat big = new Mat(t.rows() + 14, t.cols(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        t.copyTo(big.submat(new Rect(0, 0, t.cols(), t.rows())));
        Imgproc.putText(big, caption, new Point(2, t.rows() + 11), Imgproc.FONT_HERSHEY_PLAIN, 0.8, new Scalar(0, 0, 0), 1);
        t.release();
        return big;
    }

    /** Лист: по 10 плиток в ряд (ряд из меньшего числа – дополняется белым). */
    static void sheet(Path f, List<Mat> tiles) throws IOException {
        List<Mat> rows = new ArrayList<>();
        for (int i = 0; i < tiles.size(); i += 10) {
            List<Mat> r = new ArrayList<>(tiles.subList(i, Math.min(tiles.size(), i + 10)));
            while (r.size() < 10) r.add(new Mat(tiles.get(0).size(), CvType.CV_8UC3, new Scalar(255, 255, 255)));
            Mat m = new Mat();
            org.opencv.core.Core.hconcat(r, m);
            rows.add(m);
        }
        Mat all = new Mat();
        org.opencv.core.Core.vconcat(rows, all);
        Files.createDirectories(f.getParent());
        FeretData.writePng(f, all);
    }

    /** PNG через imencode и Java NIO (imwrite OpenCV не пишет по пути с кириллицей): временный файл и переименование. */
    static void writePng(Path f, Mat m) throws IOException {
        MatOfByte b = new MatOfByte();
        if (!Imgcodecs.imencode(".png", m, b)) throw new IOException("Не закодирован " + f);
        Path tmp = f.resolveSibling(f.getFileName() + ".tmp");
        Files.write(tmp, b.toArray());
        b.release();
        Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    /** Чтение PNG по пути с кириллицей (imdecode). */
    static Mat readColor(Path f) throws IOException {
        MatOfByte b = new MatOfByte(Files.readAllBytes(f));
        Mat m = Imgcodecs.imdecode(b, Imgcodecs.IMREAD_COLOR);
        b.release();
        if (m.empty()) throw new IOException("Не читается " + f);
        return m;
    }

    /** Режим duel-set. */
    static void run(Path exportDir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        String rawDuel = System.getenv("FACES_DUEL_DIR");
        if (rawDuel == null) throw new IllegalStateException("duel-set: задайте FACES_DUEL_DIR (каталог набора для передачи)");
        Path duel = Paths.get(rawDuel);
        if (Files.exists(duel)) {
            try (Stream<Path> s = Files.list(duel)) {
                if (s.findAny().isPresent()) throw new IllegalStateException("duel-set: каталог " + duel + " не пуст – удалите его или задайте другой");
            }
        }
        long seed = settings.loadFacesSeed();
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Composition c = compose(exportDir, selLines, seed);
        Map<String, double[]> det = XBaseMpPts.exportRows(exportDir, "gt");
        Path rep = out.getParent().resolve("duel");
        Files.createDirectories(rep);
        FarMethods.write(rep.resolve(KEY), key(c, seed, exportDir));
        int[][] clipped = new int[SETS.length][1];
        List<List<Mat>> tiles = new ArrayList<>();
        for (int s = 0; s < SETS.length; s++) tiles.add(new ArrayList<>());
        for (Item it : c.items) {
            Mat color = Imgcodecs.imread(gtDir.resolve(it.pk() + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            if (color.empty()) throw new IOException("duel-set: не читается " + it.pk());
            for (int s = 0; s < SETS.length; s++) {
                Mat f = ProkaevFrames.colorFrame(color, det.get(it.pk()), s == 1, clipped[s]);
                Path p = duel.resolve(SETS[s]).resolve(it.rel());
                Files.createDirectories(p.getParent());
                writePng(p, f);
                tiles.get(s).add(tile(f, it.code()));
                f.release();
            }
            color.release();
        }
        for (int s = 0; s < SETS.length; s++) {
            Files.writeString(duel.resolve("таблица_" + SETS[s] + ".csv"), table(SETS[s]), StandardCharsets.UTF_8);
            sheet(rep.resolve("sheet_" + SET_TAGS[s] + ".png"), tiles.get(s));
            tiles.get(s).forEach(Mat::release);
        }
        Files.writeString(duel.resolve("README.txt"), readme(), StandardCharsets.UTF_8);
        // Опись набора: SHA-256 всех файлов (отчёт duel-eval берёт её же).
        StringBuilder inv = new StringBuilder();
        List<Path> files;
        try (Stream<Path> st = Files.walk(duel)) {
            files = st.filter(Files::isRegularFile).sorted().toList();
        }
        for (Path f : files) inv.append(sha256(f)).append("  ").append(duel.relativize(f).toString().replace('\\', '/')).append('\n');
        FarMethods.write(rep.resolve("set_sha256.txt"), inv);
        Map<String, Integer> roles = new LinkedHashMap<>();
        c.items.forEach(it -> roles.merge(it.role(), 1, Integer::sum));
        StringBuilder t = new StringBuilder(String.format(Locale.ROOT, "duel-set: людей GT после отбора %d, с ≥ 2 датами %d, подходящих своих %d; роли %s; "
                + "не влезло: К-лицо %d, К-обычный %d; пропущено %s; файлов набора %d; %s; %.0f с%n", c.persons, c.multiDate, c.eligible, roles, clipped[0][0],
                clipped[1][0], c.skipped, files.size(), duel, (System.nanoTime() - t0) / 1e9));
        FarMethods.write(rep.resolve("duel_set.txt"), t);
        System.out.print(t);
    }
}
