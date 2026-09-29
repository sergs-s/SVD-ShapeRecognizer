package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * Открытые базы лиц (вне git, пути — faces.gt.dir, faces.muct.dir; источники —
 * SOURCE.txt рядом с каждой базой).
 * <ul>
 *   <li>Georgia Tech Face Database: 50 человек × 15 JPEG, человек = папка (s01…s50);
 *       сессий в данных нет.</li>
 *   <li>MUCT: имя i&lt;NNN&gt;&lt;освещение q–z&gt;&lt;камера a–e&gt;-….jpg; человек — NNN,
 *       камера a — фронтальная.</li>
 * </ul>
 *
 * @author ssv
 */
public final class ExternalDatasets {

    /** Снимок открытой базы: человек, файл, камера и освещение MUCT (у Georgia Tech — ' '). */
    public record Image(String person, Path file, char camera, char light) {}

    private ExternalDatasets() {
    }

    /** Georgia Tech: человек → снимки (по имени файла). */
    public static Map<String, List<Image>> georgiaTech(Path dir) throws IOException {
        Map<String, List<Image>> map = new TreeMap<>();
        try (Stream<Path> s = Files.list(dir)) {
            for (Path p : s.filter(Files::isDirectory).sorted().toList()) {
                String person = p.getFileName().toString();
                try (Stream<Path> f = Files.list(p)) {
                    map.put(person, f.filter(x -> x.toString().toLowerCase(Locale.ROOT).endsWith(".jpg")).sorted()
                            .map(x -> new Image(person, x, ' ', ' ')).toList());
                }
            }
        }
        return map;
    }

    /** MUCT: человек → снимки (по имени файла). */
    public static Map<String, List<Image>> muct(Path dir) throws IOException {
        Map<String, List<Image>> map = new TreeMap<>();
        try (Stream<Path> s = Files.list(dir)) {
            for (Path f : s.sorted().toList()) {
                String n = f.getFileName().toString();
                if (!n.matches("i\\d{3}[a-z][a-e]-.*\\.jpg")) continue;
                String person = n.substring(1, 4);
                map.computeIfAbsent(person, k -> new ArrayList<>()).add(new Image(person, f, n.charAt(5), n.charAt(4)));
            }
        }
        return map;
    }
}
