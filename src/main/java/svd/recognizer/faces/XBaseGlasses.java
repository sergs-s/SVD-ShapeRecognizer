package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.storage.SettingsStore;

/**
 * О-8, «мешает ли смена очков» (Жорик №4 от 05.11): feret-glasses, gt-glasses – по готовым моделям итогового прогона, без распознавания; проба своих – «совпадение», если её состояние очков есть хотя бы на
 * одном снимке галереи человека, иначе «смена»; ошибка своих при общем пороге разделимости (порог по контрольным чужим проходом, как в итоговом отчёте) и доля верного лучшего кандидата по группам, разность
 * «смена − совпадение», бутстреп по людям. FERET – поле glasses в ground_truths/name_value; GT – файл gt_glasses.tsv, заполненный Хозяином. gt-glasses-sheets – листы проверки и шаблон разметки.
 *
 * @author ssv
 */
final class XBaseGlasses {

    private XBaseGlasses() {
    }

    static final String UNKNOWN = "?";
    static final String GT_FILE = "gt_glasses.tsv";
    static final int MATCH = 0;
    static final int CHANGE = 1;
    static final int EXCLUDED = -1;

    /** Группа пробы: «?» у пробы или нет известных снимков галереи – исключена. */
    static int group(String probe, List<String> gallery) {
        if (probe == null || probe.equals(UNKNOWN)) return EXCLUDED;
        Set<String> known = new HashSet<>(gallery);
        known.remove(UNKNOWN);
        known.remove(null);
        if (known.isEmpty()) return EXCLUDED;
        return known.contains(probe) ? MATCH : CHANGE;
    }

    /** Разметка GT: человек, файл, дата, очки; значение – 0, 1 или ?; пустое или иное – ошибка с номером строки. Ключ – как у кадров (gt, человек, имя без расширения). */
    static Map<String, String> readGt(Path f) throws IOException {
        Map<String, String> m = new LinkedHashMap<>();
        List<String> ls = Files.readAllLines(f, StandardCharsets.UTF_8);
        for (int i = 1; i < ls.size(); i++) {
            String l = ls.get(i);
            int no = i + 1;
            String[] c = l.split("\t", -1);
            if (c.length != 4) throw new IllegalStateException(f + ", строка " + no + ": ожидается 4 столбца, а не " + c.length);
            String v = c[3].trim();
            if (v.isEmpty()) throw new IllegalStateException(f + ", строка " + no + ": пустое значение очков (" + c[0] + " " + c[1] + ")");
            if (!v.equals("0") && !v.equals("1") && !v.equals(UNKNOWN)) throw new IllegalStateException(f + ", строка " + no + ": значение «" + v + "» – допустимы 0, 1, ?");
            String stem = c[1].contains(".") ? c[1].substring(0, c[1].lastIndexOf('.')) : c[1];
            m.put(Paths.get("gt", c[0], stem).toString(), v);
        }
        return m;
    }

    private static String feretState(Path dir, String key) throws IOException {
        String person = key.substring(0, key.indexOf('_'));
        for (String dvd : List.of("dvd1", "dvd2")) {
            Path f = dir.resolve(dvd).resolve("data").resolve("ground_truths").resolve("name_value").resolve(person).resolve(key + ".txt");
            if (!Files.exists(f)) continue;
            for (String l : Files.readAllLines(f, StandardCharsets.ISO_8859_1)) {
                if (l.startsWith("glasses=")) return l.substring(8).trim().equalsIgnoreCase("yes") ? "1" : "0";
            }
            throw new IllegalStateException("glasses: нет поля в " + f);
        }
        throw new IllegalStateException("glasses: нет разметки снимка " + key);
    }

    /** Итог строки: группы проб своих по конфигурациям. */
    private record Line(String name, List<FeiDecompose.Cfg> cfgs, List<int[]> groups, List<int[]> days) {}

    /** Слой промежутка: 0 – все пробы, 1 – ≤ 30 дней, 2 – > 30 дней; неизвестный промежуток (−1) входит только в слой 0. */
    static boolean inLayer(int days, int layer) {
        return layer == 0 || (days >= 0 && (layer == 1 ? days <= 30 : days > 30));
    }

    private static double[] ownErr(Line l, int[] w, double far, int grp) {
        return ownErr(l, w, far, grp, 0);
    }

    private static double[] ownErr(Line l, int[] w, double far, int grp, int layer) {
        double e = 0;
        double n = 0;
        double ok = 0;
        for (int c = 0; c < l.cfgs().size(); c++) {
            FeiDecompose.Cfg k = l.cfgs().get(c);
            XBaseSep.Data d = XBaseOther.data(k);
            double th = XBaseSepLtp.ctrlThreshold(d, w, far);
            if (Double.isNaN(th)) return new double[] {Double.NaN, Double.NaN};
            int[] g = l.groups().get(c);
            int[] dd = l.days().get(c);
            for (int i = 0; i < k.ownR.length; i++) {
                if (g[i] != grp || !inLayer(dd[i], layer)) continue;
                int wt = w[k.ownP[i]];
                n += wt;
                if (k.ownOk[i]) ok += wt;
                if (!(k.ownOk[i] && k.ownR[i] <= th)) e += wt;
            }
        }
        return n == 0 ? new double[] {Double.NaN, Double.NaN} : new double[] {100 * e / n, 100 * ok / n};
    }

    private static String interval(double point, double[] reps) {
        double[] ok = Arrays.stream(reps).filter(v -> !Double.isNaN(v)).toArray();
        double[] s = FeiBootstrap.sorted(ok);
        double lo = FeiBootstrap.quantile(s, 0.025);
        double hi = FeiBootstrap.quantile(s, 0.975);
        return String.format(Locale.ROOT, "%+.2f [%+.2f; %+.2f]%s%s", point, lo, hi, lo > 0 || hi < 0 ? " *" : "", ok.length < reps.length ? " (без " + (reps.length - ok.length) + " реплик)" : "");
    }

    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("glasses: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        boolean feret = base.equals("feret");
        Path gtFile = feret ? null : Paths.get(settings.loadFacesGtDir()).getParent().resolve(GT_FILE);
        Map<String, String> gtLabels = feret ? null : readGt(gtFile);
        List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> b.group().equals("межс.")).toList();
        Map<String, String> cache = new HashMap<>();
        List<int[]> groups = new ArrayList<>();
        Set<String> pMatch = new HashSet<>();
        Set<String> pChange = new HashSet<>();
        int[] counts = new int[3];
        Map<String, String> gtDates = feret ? null : exifDates(dir);
        List<int[]> daysAll = new ArrayList<>();
        Map<String, String> galType = new HashMap<>();
        // Слой → группа → люди; группа → промежутки
        List<List<Set<String>>> layerPeople = new ArrayList<>();
        for (int a = 0; a < 3; a++) layerPeople.add(List.of(new HashSet<>(), new HashSet<>()));
        List<List<Integer>> gaps = List.of(new ArrayList<>(), new ArrayList<>());
        int[] noDate = new int[2];
        for (XBaseVec.Blk b : bs) {
            List<Probe> own = XBaseEval.probes(b.g(), b.pro()).stream().filter(q -> q.role().equals(FeiScores.OWN)).toList();
            for (String p : b.g().own) {
                List<String> st = new ArrayList<>();
                for (String k : b.gal().get(p)) st.add(state(feret, u.su.dir, gtLabels, cache, k));
                galType.put(p, galleryType(st));
            }
            int[] g = new int[own.size()];
            int[] dd = new int[own.size()];
            for (int i = 0; i < g.length; i++) {
                Probe q = own.get(i);
                List<String> gal = new ArrayList<>();
                List<java.time.LocalDate> galDates = new ArrayList<>();
                for (String k : b.gal().get(q.person())) {
                    gal.add(state(feret, u.su.dir, gtLabels, cache, k));
                    java.time.LocalDate d = date(feret, k, gtDates);
                    if (d != null) galDates.add(d);
                }
                g[i] = group(state(feret, u.su.dir, gtLabels, cache, q.key()), gal);
                dd[i] = gap(date(feret, q.key(), gtDates), galDates);
                counts[g[i] == EXCLUDED ? 2 : g[i]]++;
                if (g[i] == MATCH) pMatch.add(q.person());
                if (g[i] == CHANGE) pChange.add(q.person());
                if (g[i] != EXCLUDED) {
                    if (dd[i] < 0) {
                        noDate[g[i]]++;
                    } else {
                        gaps.get(g[i]).add(dd[i]);
                    }
                    for (int a = 0; a < 3; a++) if (inLayer(dd[i], a)) layerPeople.get(a).get(g[i]).add(q.person());
                }
            }
            groups.add(g);
            daysAll.add(dd);
        }
        Map<String, List<FeiDecompose.Cfg>> rows = new LinkedHashMap<>();
        rows.put("слияние (32)", new ArrayList<>());
        rows.put("L1-tt-xs-F", new ArrayList<>());
        rows.put("L1-xs-F", new ArrayList<>());
        for (XBaseVec.Blk b : bs) {
            rows.get("слияние (32)").add(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
            rows.get("L1-tt-xs-F").add(XBaseGoalPrep.cfg(b, b.g(), XBaseVec.normLines(XBaseFinal.models(r, b, XBaseFinal.TT1, false), null, new int[2])));
            rows.get("L1-xs-F").add(XBaseGoalPrep.cfg(b, XBaseAbl3.mirrorGt(b.g(), true), XBaseVec.normLines(XBaseFinal.models(r, b, XBaseFinal.XS, false), null, new int[2])));
        }
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        int[] unit = FeiBootstrap.unit(u.persons);
        StringBuilder t = new StringBuilder("О-8: мешает ли смена очков – " + base + " межс. (XBaseEval " + base + "-glasses; Жорик №4 от 05.11)\nКод: коммит " + FarMethods.commit() + "; модели – " + r.dir().resolve("vec")
                + "\nРазметка: " + (feret ? "поле glasses, ground_truths/name_value" : gtFile + ", SHA-256 " + FrameB.sha256(gtFile)) + String.format(Locale.ROOT, "; бутстреп по людям B = %d, seed %d%n", u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET));
        t.append(String.format(Locale.ROOT, "%nГруппы проб своих: совпадение – %d проб, %d людей; смена – %d проб, %d людей; исключено (нет состояния) – %d проб. Человек может быть в обеих группах.%n", counts[MATCH], pMatch.size(), counts[CHANGE], pChange.size(), counts[2]));
        t.append("\nСостав галерей по очкам (людей: только без очков | только в очках | смешанная | нет известных):\n");
        t.append("все люди галереи | ").append(typeCells(galType, galType.keySet())).append('\n');
        t.append("люди с пробами «совпадение» | ").append(typeCells(galType, pMatch)).append('\n');
        t.append("люди с пробами «смена» | ").append(typeCells(galType, pChange)).append('\n');
        t.append("\nПромежуток «проба − галерея», дней (ближайшая дата галереи): группа | проб | 25 / 50 / 75 % | доля > 30 дней | нет даты\n");
        String[] gName = {"совпадение", "смена"};
        for (int g = 0; g < 2; g++) t.append(gapRow(gName[g], gaps.get(g), noDate[g]));
        t.append("\nГруппа | людей в слое ≤ 30 дней | людей в слое > 30 дней\n");
        for (int g = 0; g < 2; g++) t.append(String.format(Locale.ROOT, "%s | %d | %d%n", gName[g], layerPeople.get(1).get(g).size(), layerPeople.get(2).get(g).size()));
        t.append("\nстрока | FAR (факт., %) | ошибка своих: совпадение | смена | смена − совпадение [95 %] | проверка: все вместе (при нуле исключённых = итоговый прогон)\n");
        StringBuilder rank = new StringBuilder("\nстрока | верный лучший: совпадение | смена | смена − совпадение [95 %]\n");
        StringBuilder layers = new StringBuilder("\nОшибка своих при FAR 1 % (порог общий, по всей строке) внутри слоя промежутка; считается, если в слое у обеих групп не меньше 5 людей:\nстрока | слой | совпадение | смена | смена − совпадение [95 %]\n");
        for (Map.Entry<String, List<FeiDecompose.Cfg>> e : rows.entrySet()) {
            Line l = new Line(e.getKey(), e.getValue(), groups, daysAll);
            for (int a = 1; a <= 2; a++) {
                String nm = a == 1 ? "≤ 30 дней" : "> 30 дней";
                int n0 = layerPeople.get(a).get(MATCH).size();
                int n1 = layerPeople.get(a).get(CHANGE).size();
                if (n0 < 5 || n1 < 5) {
                    layers.append(String.format(Locale.ROOT, "%s | %s | не считается: людей %d и %d%n", e.getKey(), nm, n0, n1));
                    continue;
                }
                double m0 = ownErr(l, unit, 0.01, MATCH, a)[0];
                double c0 = ownErr(l, unit, 0.01, CHANGE, a)[0];
                double[] rl = new double[w.length];
                for (int q = 0; q < w.length; q++) rl[q] = ownErr(l, w[q], 0.01, CHANGE, a)[0] - ownErr(l, w[q], 0.01, MATCH, a)[0];
                layers.append(String.format(Locale.ROOT, "%s | %s | %.1f %% | %.1f %% | %s%n", e.getKey(), nm, m0, c0, interval(c0 - m0, rl)));
            }
            for (double far : new double[] {0.01, 0.05}) {
                double[] m = ownErr(l, unit, far, MATCH);
                double[] c = ownErr(l, unit, far, CHANGE);
                double[] rep = new double[w.length];
                for (int q = 0; q < w.length; q++) rep[q] = ownErr(l, w[q], far, CHANGE)[0] - ownErr(l, w[q], far, MATCH)[0];
                double[] allFar = XBaseOther.errAtFar(e.getValue(), unit, far);
                t.append(String.format(Locale.ROOT, "%s | %.2f | %.1f %% | %.1f %% | %s | %.1f %%%n", e.getKey(), allFar[1], m[0], c[0], interval(c[0] - m[0], rep), allFar[0]));
            }
            double[] m = ownErr(l, unit, 0.01, MATCH);
            double[] c = ownErr(l, unit, 0.01, CHANGE);
            double[] rep = new double[w.length];
            for (int q = 0; q < w.length; q++) rep[q] = ownErr(l, w[q], 0.01, CHANGE)[1] - ownErr(l, w[q], 0.01, MATCH)[1];
            rank.append(String.format(Locale.ROOT, "%s | %.1f %% | %.1f %% | %s%n", e.getKey(), m[1], c[1], interval(c[1] - m[1], rep)));
        }
        t.append(rank).append(layers);
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Path rf = (feret ? r.dir().getParent() : out).resolve("xbase_" + base + "_glasses.txt");
        FarMethods.write(rf, t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Тип галереи по состояниям снимков: без очков, в очках, смешанная, неизвестно. */
    static String galleryType(List<String> states) {
        Set<String> s = new HashSet<>(states);
        s.remove(UNKNOWN);
        if (s.isEmpty()) return "нет";
        if (s.size() == 2) return "смеш";
        return s.contains("1") ? "очки" : "без";
    }

    private static String typeCells(Map<String, String> type, Set<String> people) {
        int[] c = new int[4];
        for (String p : people) {
            String t = type.get(p);
            c[t.equals("без") ? 0 : t.equals("очки") ? 1 : t.equals("смеш") ? 2 : 3]++;
        }
        return c[0] + " | " + c[1] + " | " + c[2] + " | " + c[3];
    }

    /** Промежуток в днях до ближайшей даты галереи; нет даты – −1. */
    static int gap(java.time.LocalDate probe, List<java.time.LocalDate> gallery) {
        if (probe == null || gallery.isEmpty()) return -1;
        long best = Long.MAX_VALUE;
        for (java.time.LocalDate d : gallery) best = Math.min(best, Math.abs(java.time.temporal.ChronoUnit.DAYS.between(d, probe)));
        return (int) best;
    }

    /** Дата снимка: FERET – из имени (ГГММДД, годы 19xx), GT – дата EXIF. */
    static java.time.LocalDate date(boolean feret, String key, Map<String, String> gtDates) {
        if (feret) {
            String[] c = key.split("_");
            return java.time.LocalDate.of(1900 + Integer.parseInt(c[1].substring(0, 2)), Integer.parseInt(c[1].substring(2, 4)), Integer.parseInt(c[1].substring(4, 6)));
        }
        Path p = Paths.get(key);
        String d = gtDates.get(p.getName(1) + "/" + p.getName(2));
        if (d == null) throw new IllegalStateException("glasses: нет строки описи дат: " + key);
        return d.equals("no-date") ? null : java.time.LocalDate.parse(d);
    }

    private static Map<String, String> exifDates(Path exportDir) throws IOException {
        Map<String, String> dates = new HashMap<>();
        for (String l : Files.readAllLines(exportDir.resolve("reports").resolve("gt_sessions.tsv"), StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person")) continue;
            String[] c = l.split("\t");
            dates.put(c[0] + "/" + c[1], c[2].length() < 10 ? "no-date" : c[2].substring(0, 10).replace(':', '-'));
        }
        return dates;
    }

    private static String gapRow(String name, List<Integer> v, int noDate) {
        if (v.isEmpty()) return name + " | 0 | – | – | " + noDate + "\n";
        int[] s = v.stream().mapToInt(Integer::intValue).sorted().toArray();
        long over = java.util.Arrays.stream(s).filter(x -> x > 30).count();
        return String.format(Locale.ROOT, "%s | %d | %d / %d / %d | %.1f %% | %d%n", name, s.length, s[s.length / 4], s[s.length / 2], s[3 * s.length / 4], 100.0 * over / s.length, noDate);
    }

    private static String state(boolean feret, Path dir, Map<String, String> gt, Map<String, String> cache, String key) throws IOException {
        if (!feret) {
            String v = gt.get(key);
            if (v == null) throw new IllegalStateException("glasses: нет снимка в разметке GT: " + key);
            return v;
        }
        String c = cache.get(key);
        if (c == null) {
            c = feretState(dir, key);
            cache.put(key, c);
        }
        return c;
    }

    /** Листы проверки (один PNG на человека) и шаблон разметки; заполненный шаблон не перезаписывается. */
    static void sheets(Path exportDir, SettingsStore settings) throws IOException {
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Path root = gtDir.getParent();
        Path sheetsDir = root.resolve("glasses");
        Files.createDirectories(sheetsDir);
        Path tsv = root.resolve(GT_FILE);
        if (Files.exists(tsv)) throw new IllegalStateException("glasses: " + tsv + " уже есть – не перезаписываю");
        Map<String, double[]> rows = FrameK2.exportRows(exportDir);
        Map<String, String> dates = new HashMap<>();
        for (String l : Files.readAllLines(exportDir.resolve("reports").resolve("gt_sessions.tsv"), StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person")) continue;
            String[] c = l.split("\t");
            // В описи есть снимок без EXIF (s37, 15: NONE)
            dates.put(c[0] + "/" + c[1], c[2].length() < 10 ? "no-date" : c[2].substring(0, 10).replace(':', '-'));
        }
        StringBuilder sb = new StringBuilder("человек\tфайл\tдата_exif\tочки\n");
        int noDet = 0;
        int n = 0;
        TreeMap<String, Path> persons = new TreeMap<>();
        try (java.util.stream.Stream<Path> s = Files.list(gtDir)) {
            s.filter(Files::isDirectory).forEach(p -> persons.put(p.getFileName().toString(), p));
        }
        int cell = 300;
        int cap = 34;
        for (Map.Entry<String, Path> e : persons.entrySet()) {
            Mat sheet = new Mat(3 * (cell + cap), 5 * cell, org.opencv.core.CvType.CV_8UC3, new Scalar(40, 40, 40));
            for (int i = 1; i <= 15; i++) {
                String stem = String.format(Locale.ROOT, "%02d", i);
                Path img = e.getValue().resolve(stem + ".jpg");
                Mat m = Imgcodecs.imread(img.toString());
                if (m.empty()) throw new IllegalStateException("glasses: не прочитан " + img);
                String date = dates.get(e.getKey() + "/" + stem);
                if (date == null) throw new IllegalStateException("glasses: нет даты EXIF " + img);
                double[] row = rows.get(Paths.get("gt", e.getKey(), stem).toString());
                Rect crop;
                if (row == null) {
                    noDet++;
                    crop = new Rect(0, 0, m.cols(), m.rows());
                } else {
                    double side = 1.6 * Math.max(row[2], row[3]);
                    double cx = row[0] + row[2] / 2;
                    double cy = row[1] + row[3] / 2;
                    int x0 = (int) Math.max(0, Math.round(cx - side / 2));
                    int y0 = (int) Math.max(0, Math.round(cy - side / 2));
                    int x1 = (int) Math.min(m.cols(), Math.round(cx + side / 2));
                    int y1 = (int) Math.min(m.rows(), Math.round(cy + side / 2));
                    crop = new Rect(x0, y0, x1 - x0, y1 - y0);
                }
                Mat sub = new Mat(m, crop);
                double scale = Math.min((double) cell / sub.cols(), (double) cell / sub.rows());
                Mat res = new Mat();
                Imgproc.resize(sub, res, new Size(Math.round(sub.cols() * scale), Math.round(sub.rows() * scale)), 0, 0, Imgproc.INTER_CUBIC);
                int col = (i - 1) % 5;
                int rowNo = (i - 1) / 5;
                int ox = col * cell + (cell - res.cols()) / 2;
                int oy = rowNo * (cell + cap) + (cell - res.rows()) / 2;
                res.copyTo(new Mat(sheet, new Rect(ox, oy, res.cols(), res.rows())));
                Imgproc.putText(sheet, stem + "  " + date + (row == null ? "  no det" : ""), new Point(col * cell + 6, rowNo * (cell + cap) + cell + 24), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
                res.release();
                sub.release();
                m.release();
                sb.append(e.getKey()).append('\t').append(stem).append(".jpg\t").append(date).append("\t\n");
                n++;
            }
            if (!Imgcodecs.imwrite(sheetsDir.resolve(e.getKey() + ".png").toString(), sheet)) throw new IllegalStateException("glasses: не записан лист " + e.getKey());
            sheet.release();
        }
        Files.writeString(tsv, sb.toString(), StandardCharsets.UTF_8);
        System.out.printf(Locale.ROOT, "Листов %d, снимков %d, без рамки YuNet %d; %s; %s%n", persons.size(), n, noDet, sheetsDir, tsv);
        System.out.println("DONE");
    }
}
