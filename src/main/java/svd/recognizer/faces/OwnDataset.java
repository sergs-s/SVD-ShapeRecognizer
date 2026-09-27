package svd.recognizer.faces;

import java.io.IOException;
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
 * имена вида ГГГГММДД_ЧЧММСС.jpg и ГГГГММДД_ЧЧММСС(n).jpg. Единица учёта —
 * момент съёмки (секунда в имени файла). Кадры момента упорядочены так: без
 * суффикса, затем (0), (1), … — первый кадр момента («первый по имени») —
 * файл без суффикса.
 *
 * Люди — в порядке имён папок, моменты — по времени.
 *
 * @author ssv
 */
public final class OwnDataset {

    private static final Pattern NAME = Pattern.compile("(\\d{8})_(\\d{2})(\\d{2})(\\d{2})(?:\\((\\d+)\\))?\\.jpe?g",
            Pattern.CASE_INSENSITIVE);

    /** Кадр: путь, момент (секунда суток), номер в моменте (−1 — без суффикса). */
    public record Frame(Path file, int second, int index) {}

    /** Момент съёмки одного человека: секунда суток и кадры в порядке съёмки. */
    public record Moment(int second, List<Frame> frames) {
        /** Первый кадр момента (без суффикса). */
        public Frame first() {
            return frames.get(0);
        }
    }

    /** Человек: имя папки и моменты по времени. */
    public record Person(String name, List<Moment> moments) {}

    private final Path root;
    private final List<Person> persons;

    private OwnDataset(Path root, List<Person> persons) {
        this.root = root;
        this.persons = persons;
    }

    /** Читает список людей, моментов и кадров (сами снимки не декодируются). */
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
                    bySecond.computeIfAbsent(second, k -> new ArrayList<>()).add(new Frame(f, second, index));
                }
            }
            if (bySecond.isEmpty()) {
                continue;
            }
            List<Moment> moments = new ArrayList<>();
            for (Map.Entry<Integer, List<Frame>> e : bySecond.entrySet()) {
                List<Frame> frames = new ArrayList<>(e.getValue());
                frames.sort(Comparator.comparingInt(Frame::index));
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

    /** Первые кадры всех моментов: люди по порядку, моменты по времени. */
    public List<Frame> firstFrames() {
        List<Frame> list = new ArrayList<>();
        for (Person p : persons) {
            for (Moment m : p.moments()) {
                list.add(m.first());
            }
        }
        return list;
    }
}
