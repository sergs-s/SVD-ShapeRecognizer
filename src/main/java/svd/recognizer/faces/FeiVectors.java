package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Полные векторы расстояний faces-fei (п. 4.2, «Нормировка оценки»; ключ faces.fei.vectors): для каждой попытки метода
 * (свой, пороговый, контрольный чужой) — оценки до всех 100 людей галереи, тот же массив, по которому FeiScores.line
 * выбирает лучшего и второго. Отдельный файл на метод: fei_vectors_&lt;метод&gt;.tsv.gz (TSV + gzip). Шапка (#): код, данные,
 * ключи, метод, по разбиению — порядок людей галереи (столбцы d_1…d_100); столбцы: method, s, j, role, person, image,
 * true_id, d_1…d_100. Числа — Double.toString (точно восстанавливаются).
 *
 * @author ssv
 */
final class FeiVectors {

    static final String PREFIX = "fei_vectors_";
    static final String GALLERY = "# Галерея s=";
    static final String COLUMNS = "method\ts\tj\trole\tperson\timage\ttrue_id\td_1…d_n";

    private FeiVectors() {
    }

    static String file(String method) {
        return PREFIX + method + ".tsv.gz";
    }

    /** Строка попытки: vector — оценки по людям галереи в её порядке. */
    static String line(String method, int s, int j, String role, String person, int image, String trueId, double[] vector) {
        StringBuilder b = new StringBuilder(vector.length * 20 + 64);
        b.append(method).append('\t').append(s).append('\t').append(j).append('\t').append(role).append('\t').append(person).append('\t')
                .append(image).append('\t').append(trueId);
        for (double v : vector) b.append('\t').append(Double.toString(v));
        return b.append('\n').toString();
    }

    /** Шапка файла метода. */
    static String header(String code, String data, String keys, int configs, String method, List<FeiProtocol.Split> splits) {
        StringBuilder t = new StringBuilder();
        t.append("# Полные векторы расстояний faces-fei (FeiMethods, faces.fei.vectors), протокол FEI (PLAN.md, п. 3а)\n");
        t.append("# Код: коммит ").append(code).append('\n');
        t.append("# Данные: ").append(data).append('\n');
        t.append("# Ключи: ").append(keys).append('\n');
        t.append(String.format(Locale.ROOT, "# Конфигураций: %d из %d; метод %s; оценка: меньше — ближе; у чужих true_id «-»\n", configs, FeiProtocol.CONFIGS,
                method));
        for (FeiProtocol.Split sp : splits) t.append(GALLERY).append(sp.index()).append(": ").append(String.join(",", sp.gallery())).append('\n');
        t.append(COLUMNS).append('\n');
        return t.toString();
    }

    /** Строка файла векторов. */
    record Row(String method, int s, int j, String role, String person, int image, String trueId, double[] d) {}

    /** Прочитанный файл: шапка, галереи по разбиениям, строки, SHA-256 сжатого файла и его размер. */
    record Loaded(List<String> header, Map<Integer, List<String>> galleries, List<Row> rows, String sha256, long bytes) {}

    static Row parse(String line) {
        String[] c = line.split("\t", -1);
        if (c.length < 8) throw new IllegalArgumentException("Строка файла векторов: " + c.length + " столбцов: " + line);
        double[] d = new double[c.length - 7];
        for (int i = 0; i < d.length; i++) d[i] = Double.parseDouble(c[7 + i]);
        return new Row(c[0], Integer.parseInt(c[1]), Integer.parseInt(c[2]), c[3], c[4], Integer.parseInt(c[5]), c[6], d);
    }

    static Loaded read(Path file) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        List<String> header = new ArrayList<>();
        Map<Integer, List<String>> galleries = new LinkedHashMap<>();
        List<Row> rows = new ArrayList<>();
        try (DigestInputStream raw = new DigestInputStream(Files.newInputStream(file), md);
             BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(raw, 1 << 16), StandardCharsets.UTF_8), 1 << 16)) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#")) {
                    header.add(l);
                    if (l.startsWith(GALLERY)) {
                        int colon = l.indexOf(": ");
                        galleries.put(Integer.parseInt(l.substring(GALLERY.length(), colon)), List.of(l.substring(colon + 2).split(",")));
                    }
                    continue;
                }
                if (l.isEmpty() || l.equals(COLUMNS)) continue;
                Row r = parse(l);
                List<String> g = galleries.get(r.s());
                if (g == null || g.size() != r.d().length) {
                    throw new IllegalStateException("Файл векторов " + file + ": нет галереи разбиения " + r.s() + " или длина вектора " + r.d().length
                            + " не равна её размеру");
                }
                rows.add(r);
            }
            raw.transferTo(java.io.OutputStream.nullOutputStream());
        }
        return new Loaded(header, galleries, rows, HexFormat.of().formatHex(md.digest()), Files.size(file));
    }

    /** Лучший и второй (первые минимумы, как FeiScores.line): {индекс лучшего, индекс второго}. */
    static int[] bestTwo(double[] v) {
        int b = 0;
        for (int p = 1; p < v.length; p++) if (v[p] < v[b]) b = p;
        int c = -1;
        for (int p = 0; p < v.length; p++) if (p != b && (c < 0 || v[p] < v[c])) c = p;
        return new int[] {b, c};
    }

    /** Ранг человека t в векторе (1 — лучший; равные — по индексу, как первый минимум). */
    static int rank(double[] v, int t) {
        int r = 1;
        for (int p = 0; p < v.length; p++) if (v[p] < v[t] || (v[p] == v[t] && p < t)) r++;
        return r;
    }
}
