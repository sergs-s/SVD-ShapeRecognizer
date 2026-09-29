package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Base;
import svd.recognizer.faces.GalleryEvaluation.Data;
import svd.recognizer.faces.GalleryEvaluation.Det;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * FEI Face Database (путь — faces.fei.dir / FACES_FEI_DIR; C. E. Thomaz, G. A. Giraldi, Image and Vision Computing
 * 28(6), 902–913, 2010; только исследовательские цели): второй независимый контрольный набор чужих в faces-far —
 * не участвует ни в обучении, ни в подборе порога.
 *
 * В протокол — только исходные снимки originalimages (200 человек × 14, имя {@code <человек>-<NN>.jpg}) через
 * конвейер 2′; справочные наборы в протокол не входят. Отбор:
 * <ul>
 *   <li>«a» и «b» — номер исходного снимка, наиболее похожего (корреляция кадров а) 92×112) на справочный
 *       frontalimages_manuallyaligned {@code <человек>a.jpg} / {@code b.jpg}; включаются всегда (отказ детектора на
 *       них учитывается как отказ);</li>
 *   <li>остальные — близкие к фронтальным: |r| ≤ R, r = (x носа − середина глаз по x) / межглазье по x по точкам
 *       YuNet второго прохода, R — 95-й процентиль |r| по снимкам своей базы в протоколе (с найденным лицом).</li>
 * </ul>
 *
 * @author ssv
 */
public final class FeiDataset {

    static final String ORIGINALS = "originalimages";
    static final String REFERENCE = "frontalimages_manuallyaligned";
    static final String CACHE_NAME = "fei_detections.cache";
    static final double POSE_PERCENTILE = 0.95;
    static final int IMAGES_PER_PERSON = 14;
    private static final Pattern NAME = Pattern.compile("(\\d+)-(\\d{2})\\.jpg", Pattern.CASE_INSENSITIVE);

    /** Сведения об отборе для шапки отчёта (в экспорте — строки «#» fei_selection.tsv). */
    record Info(double r, double ownMax, int ownN, String ab) {}

    /** Итог отбора: отобранные снимки по людям, их детекции, сведения, таблица fei_selection.tsv, отчёт fei_selection.txt. */
    record Selection(Map<String, List<Sample>> selected, Map<String, Det> dets, Info info, String table, String report) {}

    private FeiDataset() {
    }

    /** Исходные снимки: человек (три цифры) → снимки по номеру 01…14. */
    static Map<String, List<Sample>> originals(Path dir) throws IOException {
        Map<String, Map<Integer, Sample>> map = new TreeMap<>();
        try (Stream<Path> s = Files.walk(dir.resolve(ORIGINALS))) {
            for (Path f : s.filter(Files::isRegularFile).toList()) {
                Matcher m = NAME.matcher(f.getFileName().toString());
                if (!m.matches()) continue;
                String person = String.format(Locale.ROOT, "%03d", Integer.parseInt(m.group(1)));
                map.computeIfAbsent(person, k -> new TreeMap<>()).put(Integer.parseInt(m.group(2)), new Sample(Base.FEI, person, f, ' '));
            }
        }
        Map<String, List<Sample>> out = new TreeMap<>();
        map.forEach((p, m) -> out.put(p, List.copyOf(m.values())));
        return out;
    }

    /** Номер исходного снимка по имени (NN из «человек-NN.jpg»). */
    static int number(Sample s) {
        Matcher m = NAME.matcher(s.file().getFileName().toString());
        if (!m.matches()) throw new IllegalArgumentException("Не снимок FEI: " + s.file());
        return Integer.parseInt(m.group(2));
    }

    /** Справочный снимок человека (a или b) из frontalimages_manuallyaligned (любая часть); null — нет. */
    static Path reference(Path dir, String person, char ab) throws IOException {
        String name = Integer.parseInt(person) + String.valueOf(ab) + ".jpg";
        try (Stream<Path> s = Files.walk(dir.resolve(REFERENCE))) {
            return s.filter(p -> p.getFileName().toString().equalsIgnoreCase(name)).findFirst().orElse(null);
        }
    }

    /** Поворот головы по точкам YuNet: (x носа − середина глаз по x) / межглазье по x. */
    static double pose(double[] row) {
        return (row[8] - (row[4] + row[6]) / 2) / Math.abs(row[6] - row[4]);
    }

    /** Процентиль p по возрастающему массиву: элемент с индексом round(p·(n − 1)). */
    static double percentile(double[] sorted, double p) {
        return sorted[(int) Math.round(p * (sorted.length - 1))];
    }

    /**
     * Отбор FEI: детекция всех исходных и справочных снимков (кэш fei_detections.cache), сопоставление «a»/«b»,
     * порог позы R по своей базе (data, dets — снимки и детекции остальных баз).
     */
    static Selection select(Path dir, Data data, Map<String, Det> dets, float score, SettingsStore settings, Path modelsDir,
                            Path outDir, boolean fresh) throws IOException {
        Map<String, List<Sample>> orig = originals(dir);
        Map<String, Sample[]> refs = new LinkedHashMap<>();
        List<Sample> all = new ArrayList<>();
        orig.values().forEach(all::addAll);
        for (String p : orig.keySet()) {
            Sample[] r = new Sample[2];
            for (int k = 0; k < 2; k++) {
                Path f = reference(dir, p, "ab".charAt(k));
                if (f != null) {
                    r[k] = new Sample(Base.FEI, p, f, ' ');
                    all.add(r[k]);
                }
            }
            refs.put(p, r);
        }
        Map<String, Det> feiDets = GalleryEvaluation.detections(all, score, settings, modelsDir, outDir, fresh, CACHE_NAME);

        // Порог позы по своей базе.
        double[] own = data.samples().stream().filter(s -> s.base() == Base.OWN).map(s -> dets.get(s.file().toString()).row())
                .filter(r -> r != null).mapToDouble(r -> Math.abs(pose(r))).sorted().toArray();
        double bigR = percentile(own, POSE_PERCENTILE);
        double ownMax = own[own.length - 1];

        // Сопоставление «a»/«b»: наибольшая корреляция кадров а) со справочным снимком.
        Map<String, int[]> match = new LinkedHashMap<>();
        List<Double> margins = new ArrayList<>();
        int[][] hist = new int[2][IMAGES_PER_PERSON + 1];
        int[] unmatched = new int[2];
        for (String p : orig.keySet()) {
            List<Sample> list = orig.get(p);
            double[][] x = new double[list.size()][];
            for (int i = 0; i < list.size(); i++) x[i] = frameA(list.get(i), feiDets);
            int[] num = {-1, -1};
            for (int k = 0; k < 2; k++) {
                Sample ref = refs.get(p)[k];
                double[] y = ref == null ? null : frameA(ref, feiDets);
                if (y == null) {
                    unmatched[k]++;
                    continue;
                }
                double best = -2;
                double second = -2;
                for (int i = 0; i < list.size(); i++) {
                    if (x[i] == null) continue;
                    double c = correlation(x[i], y);
                    if (c > best) {
                        second = best;
                        best = c;
                        num[k] = number(list.get(i));
                    } else if (c > second) {
                        second = c;
                    }
                }
                if (num[k] < 0) {
                    unmatched[k]++;
                } else {
                    hist[k][num[k]]++;
                    margins.add(best - second);
                }
            }
            match.put(p, num);
        }
        int[] mode = {argmax(hist[0]), argmax(hist[1])};

        // Отбор и таблица.
        Map<String, List<Sample>> selected = new TreeMap<>();
        Map<String, Det> selDets = new LinkedHashMap<>();
        StringBuilder rows = new StringBuilder();
        int[] foundByNum = new int[IMAGES_PER_PERSON + 1];
        int[] passR = new int[IMAGES_PER_PERSON + 1];
        int[] passMax = new int[IMAGES_PER_PERSON + 1];
        List<List<Double>> absByNum = new ArrayList<>();
        for (int i = 0; i <= IMAGES_PER_PERSON; i++) absByNum.add(new ArrayList<>());
        int selTotal = 0;
        int selMaxTotal = 0;
        int refusedAll = 0;
        int refusedSel = 0;
        List<Integer> perPerson = new ArrayList<>();
        for (String p : orig.keySet()) {
            int a = match.get(p)[0] >= 0 ? match.get(p)[0] : mode[0];
            int b = match.get(p)[1] >= 0 ? match.get(p)[1] : mode[1];
            List<Sample> chosen = new ArrayList<>();
            for (Sample s : orig.get(p)) {
                int n = number(s);
                double[] row = feiDets.get(s.file().toString()).row();
                double r = row == null ? Double.NaN : pose(row);
                String ref = n == a ? "a" : n == b ? "b" : "-";
                String reason;
                boolean in;
                if (row == null) refusedAll++;
                else {
                    foundByNum[n]++;
                    absByNum.get(n).add(Math.abs(r));
                    if (Math.abs(r) <= bigR) passR[n]++;
                    if (Math.abs(r) <= ownMax) passMax[n]++;
                }
                if (!ref.equals("-")) {
                    in = true;
                    reason = "справочный «" + ref + "»";
                } else if (row == null) {
                    in = false;
                    reason = "исключён: отказ YuNet";
                } else if (Math.abs(r) <= bigR) {
                    in = true;
                    reason = "поза |r| ≤ R";
                } else {
                    in = false;
                    reason = "исключён: поза |r| > R";
                }
                if (!ref.equals("-") || (row != null && Math.abs(r) <= ownMax)) selMaxTotal++;
                if (in) {
                    chosen.add(s);
                    selDets.put(s.file().toString(), feiDets.get(s.file().toString()));
                    if (row == null) refusedSel++;
                }
                rows.append(p).append('\t').append(n).append('\t').append(FarExport.stem(s.file())).append('\t').append(row == null ? 0 : 1)
                        .append('\t').append(row == null ? "" : String.format(Locale.ROOT, "%.4f", r)).append('\t').append(ref).append('\t')
                        .append(in ? 1 : 0).append('\t').append(reason).append('\n');
            }
            selected.put(p, List.copyOf(chosen));
            selTotal += chosen.size();
            perPerson.add(chosen.size());
        }

        String ab = String.format(Locale.ROOT, "«a» — исходный №%02d (у %d из %d), «b» — №%02d (у %d из %d)", mode[0], hist[0][mode[0]],
                orig.size(), mode[1], hist[1][mode[1]], orig.size());
        Info info = new Info(bigR, ownMax, own.length, ab);
        StringBuilder table = new StringBuilder();
        table.append("# R\t").append(bigR).append('\n').append("# own_max\t").append(ownMax).append('\n').append("# own_n\t").append(own.length)
                .append('\n').append("# ab\t").append(ab).append('\n');
        table.append("person\tnumber\tname\tfound\tr\treference\tincluded\treason\n").append(rows);

        // Отчёт об отборе.
        int[] pp = perPerson.stream().mapToInt(Integer::intValue).sorted().toArray();
        double[] mg = margins.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        StringBuilder rep = new StringBuilder("Отбор снимков FEI (faces-far, чужие FEI)\n\n");
        rep.append(String.format(Locale.ROOT, "Исходных снимков %d (людей %d), справочных frontalimages_manuallyaligned %d.%n", all.size()
                - refs.values().stream().mapToLong(r -> Arrays.stream(r).filter(x -> x != null).count()).sum(), orig.size(),
                refs.values().stream().mapToLong(r -> Arrays.stream(r).filter(x -> x != null).count()).sum()));
        rep.append("\nСопоставление справочных «a»/«b» с исходными (наибольшая корреляция кадров а) 92×112):\n");
        for (int k = 0; k < 2; k++) {
            rep.append("  «").append("ab".charAt(k)).append("»:");
            for (int n = 1; n <= IMAGES_PER_PERSON; n++) if (hist[k][n] > 0) rep.append(String.format(Locale.ROOT, " №%02d — %d;", n, hist[k][n]));
            rep.append(" не сопоставлено (нет справочного, отказ YuNet на справочном или на всех исходных) — ").append(unmatched[k]).append('\n');
        }
        if (mg.length > 0) {
            rep.append(String.format(Locale.ROOT, "  Зазор корреляции (лучший − второй): мин %.4f, медиана %.4f.%n", mg[0], mg[mg.length / 2]));
        }
        rep.append(String.format(Locale.ROOT, "%nПоза r = (x носа − середина глаз по x) / межглазье по x (точки YuNet второго прохода).%n"
                + "Своя база (снимки протокола с найденным лицом, %d): 95-й процентиль |r| = R = %.4f, максимум %.4f.%n", own.length, bigR, ownMax));
        rep.append("\nFEI по номеру снимка: найдено лицо, медиана |r|, прошло бы при R (95-й процентиль) и при максимуме своей базы:\n");
        for (int n = 1; n <= IMAGES_PER_PERSON; n++) {
            double[] v = absByNum.get(n).stream().mapToDouble(Double::doubleValue).sorted().toArray();
            rep.append(String.format(Locale.ROOT, "  №%02d: найдено %3d, медиана |r| %s, |r| ≤ R — %3d, |r| ≤ максимум — %3d%n", n, foundByNum[n],
                    v.length == 0 ? "—" : String.format(Locale.ROOT, "%.3f", v[v.length / 2]), passR[n], passMax[n]));
        }
        rep.append(String.format(Locale.ROOT, "%nОтобрано в протокол (R = 95-й процентиль; «a», «b» всегда): %d снимков, на человека мин %d / медиана %d / "
                + "макс %d.%nСправочно при R = максимум своей базы: %d снимков.%n", selTotal, pp[0], pp[pp.length / 2], pp[pp.length - 1], selMaxTotal));
        rep.append(String.format(Locale.ROOT, "Отказы YuNet: среди всех исходных %d из %d; среди отобранных %d из %d.%n", refusedAll,
                orig.values().stream().mapToInt(List::size).sum(), refusedSel, selTotal));
        return new Selection(selected, selDets, info, table.toString(), rep.toString());
    }

    /** Кадр а) 92×112 снимка как вектор (null — отказ детектора). */
    private static double[] frameA(Sample s, Map<String, Det> dets) throws IOException {
        double[] row = dets.get(s.file().toString()).row();
        if (row == null) return null;
        Mat[] m = GalleryEvaluation.load(s);
        try {
            Mat f = GalleryEvaluation.frame(s, m, row, Variant.A, null);
            double[] x = FaceAlignment.toVector(f);
            f.release();
            return x;
        } finally {
            GalleryEvaluation.release(m);
        }
    }

    private static double correlation(double[] x, double[] y) {
        double mx = Arrays.stream(x).average().orElse(0);
        double my = Arrays.stream(y).average().orElse(0);
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < x.length; i++) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
            syy += (y[i] - my) * (y[i] - my);
        }
        return sxy / Math.sqrt(sxx * syy);
    }

    private static int argmax(int[] h) {
        int best = 0;
        for (int i = 1; i < h.length; i++) if (h[i] > h[best]) best = i;
        return best;
    }

    /** Сведения об отборе из строк «#» fei_selection.tsv (экспорт). */
    static Info readInfo(List<String> lines) {
        Map<String, String> kv = new LinkedHashMap<>();
        for (String l : lines) {
            if (!l.startsWith("# ")) break;
            String[] c = l.substring(2).split("\t", 2);
            kv.put(c[0], c[1]);
        }
        return new Info(Double.parseDouble(kv.get("R")), Double.parseDouble(kv.get("own_max")), Integer.parseInt(kv.get("own_n")), kv.get("ab"));
    }
}
