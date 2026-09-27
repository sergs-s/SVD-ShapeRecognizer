package svd.recognizer.faces;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Своя база лиц (путь — SettingsStore.loadFacesOwnDir()): папки людей с JPEG,
 * имена вида ГГГГММДД_ЧЧММСС[(n)][пометка].jpg. Единица учёта — момент
 * съёмки (секунда в имени файла).
 *
 * Пометка качества (Хозяин, по виду лица, до результатов алгоритмов): «+» —
 * хороший, «+-» — приемлемый, «+--» — плохой, без пометки — мусор (лица не
 * видно). Кадры момента упорядочены по времени съёмки из EXIF
 * (DateTimeOriginal + SubSecTimeOriginal); пометка на порядок не влияет.
 * Представитель момента — кадр с лучшей пометкой, при равенстве — более
 * ранний. Момент без помеченных кадров — непригодный.
 *
 * Люди — в порядке имён папок, моменты — по времени.
 *
 * @author ssv
 */
public final class OwnDataset {

    static final Pattern NAME = Pattern.compile(
            "(\\d{8})_(\\d{2})(\\d{2})(\\d{2})(?:\\((\\d+)\\))?(\\+-{0,2})?\\.jpe?g", Pattern.CASE_INSENSITIVE);

    /** Оценки пометок: 3 — «+», 2 — «+-», 1 — «+--», 0 — без пометки. */
    public static final int GOOD = 3;
    public static final int FAIR = 2;
    public static final int POOR = 1;
    public static final int JUNK = 0;

    /**
     * Кадр: путь, момент (секунда суток по имени), номер в имени (−1 — без
     * суффикса), оценка пометки, время съёмки по EXIF (секунды суток).
     */
    public record Frame(Path file, int second, int index, int quality, double time) {}

    /** Момент съёмки одного человека: секунда по имени и кадры по времени съёмки. */
    public record Moment(int second, List<Frame> frames) {
        /** Первый снятый кадр момента. */
        public Frame first() {
            return frames.get(0);
        }

        /** Есть хотя бы один помеченный кадр. */
        public boolean usable() {
            return frames.stream().anyMatch(f -> f.quality() > JUNK);
        }

        /** Представитель: лучшая пометка, при равенстве — более ранний; null для непригодного. */
        public Frame representative() {
            Frame best = null;
            for (Frame f : frames) {
                if (f.quality() > JUNK && (best == null || f.quality() > best.quality())) best = f;
            }
            return best;
        }
    }

    /** Человек: имя папки и моменты по времени. */
    public record Person(String name, List<Moment> moments) {
        public List<Moment> usableMoments() {
            return moments.stream().filter(Moment::usable).toList();
        }
    }

    private final Path root;
    private final List<Person> persons;

    private OwnDataset(Path root, List<Person> persons) {
        this.root = root;
        this.persons = persons;
    }

    /** Читает список людей, моментов и кадров и время съёмки из EXIF (сами снимки не декодируются). */
    public static OwnDataset load(Path root) throws IOException {
        List<Path> dirs;
        try (Stream<Path> s = Files.list(root)) {
            dirs = s.filter(Files::isDirectory).sorted(Comparator.comparing(p -> p.getFileName().toString())).toList();
        }
        List<Person> persons = new ArrayList<>();
        for (Path dir : dirs) {
            Map<Integer, List<Frame>> bySecond = new TreeMap<>();
            try (Stream<Path> s = Files.list(dir)) {
                for (Path f : s.sorted().toList()) {
                    Matcher m = NAME.matcher(f.getFileName().toString());
                    if (!m.matches()) {
                        continue;
                    }
                    int second = Integer.parseInt(m.group(2)) * 3600 + Integer.parseInt(m.group(3)) * 60
                            + Integer.parseInt(m.group(4));
                    int index = m.group(5) == null ? -1 : Integer.parseInt(m.group(5));
                    int quality = m.group(6) == null ? JUNK : GOOD - (m.group(6).length() - 1);
                    bySecond.computeIfAbsent(second, k -> new ArrayList<>())
                            .add(new Frame(f, second, index, quality, captureTime(f)));
                }
            }
            if (bySecond.isEmpty()) {
                continue;
            }
            List<Moment> moments = new ArrayList<>();
            for (Map.Entry<Integer, List<Frame>> e : bySecond.entrySet()) {
                List<Frame> frames = new ArrayList<>(e.getValue());
                frames.sort(Comparator.comparingDouble(Frame::time));
                moments.add(new Moment(e.getKey(), List.copyOf(frames)));
            }
            persons.add(new Person(dir.getFileName().toString(), List.copyOf(moments)));
        }
        return new OwnDataset(root, List.copyOf(persons));
    }

    public Path root() {
        return root;
    }

    public List<Person> persons() {
        return persons;
    }

    /** Путь кадра относительно корня базы, через «/» (ключ в CSV). */
    public String relative(Path file) {
        return root.relativize(file).toString().replace('\\', '/');
    }

    /** Представители пригодных моментов: люди по порядку, моменты по времени. */
    public List<Frame> representatives() {
        List<Frame> list = new ArrayList<>();
        for (Person p : persons) {
            for (Moment m : p.usableMoments()) {
                list.add(m.representative());
            }
        }
        return list;
    }

    // ---------------------------------------------------------------- EXIF

    /**
     * Время съёмки (секунды суток) из EXIF: DateTimeOriginal (0x9003) и
     * SubSecTimeOriginal (0x9291). Без EXIF — IOException.
     */
    static double captureTime(Path file) throws IOException {
        byte[] head;
        try (InputStream in = Files.newInputStream(file)) {
            head = in.readNBytes(128 * 1024);
        }
        int pos = 2;
        while (pos + 4 < head.length && (head[pos] & 0xFF) == 0xFF) {
            int marker = head[pos + 1] & 0xFF;
            int len = ((head[pos + 2] & 0xFF) << 8) | (head[pos + 3] & 0xFF);
            if (marker == 0xE1 && new String(head, pos + 4, 4, StandardCharsets.US_ASCII).equals("Exif")) {
                double t = parseTiff(head, pos + 10);
                if (!Double.isNaN(t)) return t;
            }
            pos += 2 + len;
        }
        throw new IOException("Нет времени съёмки в EXIF: " + file);
    }

    private static double parseTiff(byte[] b, int base) {
        boolean le = b[base] == 'I';
        int ifd0 = base + u32(b, base + 4, le);
        int exif = -1;
        int n = u16(b, ifd0, le);
        for (int i = 0; i < n; i++) {
            int e = ifd0 + 2 + 12 * i;
            if (u16(b, e, le) == 0x8769) exif = base + u32(b, e + 8, le);
        }
        if (exif < 0) return Double.NaN;
        String dt = null;
        String sub = "0";
        n = u16(b, exif, le);
        for (int i = 0; i < n; i++) {
            int e = exif + 2 + 12 * i;
            int tag = u16(b, e, le);
            int count = u32(b, e + 4, le);
            int at = count <= 4 ? e + 8 : base + u32(b, e + 8, le);
            if (tag == 0x9003) dt = ascii(b, at, count);
            if (tag == 0x9291) sub = ascii(b, at, count);
        }
        if (dt == null || dt.length() < 19) return Double.NaN;
        double t = Integer.parseInt(dt.substring(11, 13)) * 3600 + Integer.parseInt(dt.substring(14, 16)) * 60
                + Integer.parseInt(dt.substring(17, 19));
        return sub.isEmpty() ? t : t + Double.parseDouble("0." + sub);
    }

    private static String ascii(byte[] b, int at, int count) {
        return new String(b, at, count, StandardCharsets.US_ASCII).replace("\0", "").trim();
    }

    private static int u16(byte[] b, int i, boolean le) {
        return le ? (b[i] & 0xFF) | ((b[i + 1] & 0xFF) << 8) : ((b[i] & 0xFF) << 8) | (b[i + 1] & 0xFF);
    }

    private static int u32(byte[] b, int i, boolean le) {
        return le ? (b[i] & 0xFF) | ((b[i + 1] & 0xFF) << 8) | ((b[i + 2] & 0xFF) << 16) | ((b[i + 3] & 0xFF) << 24)
                : ((b[i] & 0xFF) << 24) | ((b[i + 1] & 0xFF) << 16) | ((b[i + 2] & 0xFF) << 8) | (b[i + 3] & 0xFF);
    }
}
