package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import svd.recognizer.storage.SettingsStore;

/**
 * BioID V1.2, справочная проверка «другая серия» (постановка Жорика 07.11, SHA-256 0ef9a14e…). bioid-inventory – опись ролей без
 * распознавания: люди – группы (а) (bioid_groups.tsv), отрезки, отрезок регистрации, 6 снимков галереи (с заменой по разд. 2), свои и
 * «только чужие», деление пополам (seed faces.seed + 13000), попытки по проходам, узел порога. Детекция YuNet нужна только для отбора
 * |r| ≤ R и замены в галерее. Всё производное – только D:\data\BioID\ (derived\, reports\).
 *
 * @author ssv
 */
final class XBaseBioId {

    private XBaseBioId() {
    }

    static final int N = 1521;
    static final String KEY = "bioid:";
    static final String CACHE = "bioid_detections.tsv";
    static final String GROUPS_SHA = "82137ad060d60485b6f1d17f792ddf6d282b7959a4ad6621c5a04a7e192dea4b";
    static final long SPLIT_SEED_OFFSET = 13000;
    static final int GALLERY = 6;
    static final int REG_MIN = 10;
    static final int OWN_MIN = 5;
    static final double R = 0.387;

    /** Отрезок человека: начало и конец (включительно). */
    record Run(int from, int to) {
        int len() {
            return to - from + 1;
        }
    }

    /** Отрезки по меткам групп: подряд идущие номера одной группы. */
    static Map<String, List<Run>> runs(int[] grp) {
        Map<String, List<Run>> out = new TreeMap<>();
        for (int i = 0; i < grp.length; ) {
            int j = i;
            while (j + 1 < grp.length && grp[j + 1] == grp[i]) j++;
            out.computeIfAbsent(person(grp[i]), k -> new ArrayList<>()).add(new Run(i, j));
            i = j + 1;
        }
        return out;
    }

    static String person(int g) {
        return String.format(Locale.ROOT, "A%02d", g);
    }

    /** Отрезок регистрации: первый по номерам длиной ≥ 10; нет – −1. */
    static int regRun(List<Run> rs) {
        for (int i = 0; i < rs.size(); i++) if (rs.get(i).len() >= REG_MIN) return i;
        return -1;
    }

    /** Номера галереи: k·(L − 1)/5 с округлением, k = 0…5; негодный или занятый – следующий по номеру в отрезке. Не хватило – null. */
    static List<Integer> gallery(Run r, Set<Integer> valid) {
        List<Integer> out = new ArrayList<>();
        for (int k = 0; k < GALLERY; k++) {
            int i = r.from() + (int) Math.round(k * (r.len() - 1) / (double) (GALLERY - 1));
            while (i <= r.to() && (!valid.contains(i) || out.contains(i))) i++;
            if (i > r.to()) return null;
            out.add(i);
        }
        return out;
    }

    /** Деление своих: перестановка по возрастанию номеров одним генератором, первая половина – ⌈n/2⌉. */
    static List<List<String>> split(List<String> own, long seed) {
        List<String> a = new ArrayList<>(own);
        Collections.sort(a);
        Collections.shuffle(a, new Random(seed));
        int h = (a.size() + 1) / 2;
        List<String> h1 = new ArrayList<>(a.subList(0, h));
        List<String> h2 = new ArrayList<>(a.subList(h, a.size()));
        Collections.sort(h1);
        Collections.sort(h2);
        return List.of(h1, h2);
    }

    static Path image(Path img, int i) {
        return img.resolve(String.format(Locale.ROOT, "BioID_%04d.pgm", i));
    }

    static String key(int i) {
        return KEY + String.format(Locale.ROOT, "%04d", i);
    }

    /** Детекции всех снимков: YuNet в два прохода, наибольшая рамка (как cp-run); кэш derived\bioid_detections.tsv, несколько лиц – derived\bioid_multi.tsv. */
    static Map<String, double[]> detections(Path root, SettingsStore settings, Set<String> multi) throws IOException {
        Path der = root.resolve("derived");
        Files.createDirectories(der);
        Path cf = der.resolve(CACHE);
        Path mf = der.resolve("bioid_multi.tsv");
        Map<String, double[]> rows = Files.exists(cf) ? VidTimitData.readDetections(cf) : new HashMap<>();
        if (Files.exists(mf)) multi.addAll(Files.readAllLines(mf, StandardCharsets.UTF_8));
        boolean todo = false;
        for (int i = 0; i < N; i++) todo |= !rows.containsKey(key(i));
        if (!todo) return rows;
        FaceDetectorYN det = OwnDetectorThreshold.create(Paths.get(settings.loadFacesModelsDir()), FeretData.SCORE);
        try (Writer c = Files.newBufferedWriter(cf, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
             Writer m = Files.newBufferedWriter(mf, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            for (int i = 0; i < N; i++) {
                String k = key(i);
                if (rows.containsKey(k)) continue;
                // PGM (серый) читается в три канала – COLOR_GRAY2BGR.
                Mat color = Imgcodecs.imread(image(root.resolve("images"), i).toString(), Imgcodecs.IMREAD_COLOR);
                if (color.empty() || color.cols() != 384 || color.rows() != 286) throw new IOException("BioID: нет снимка или не 384×286 – " + i);
                boolean[] mm = new boolean[1];
                double[] r = XBaseChokeRun.detect(det, color, color, 1.0, mm);
                color.release();
                StringBuilder l = new StringBuilder(k).append('\t').append(r == null ? "0" : "1");
                for (int j = 0; j < 15; j++) l.append('\t').append(r == null ? "" : Double.toString(r[j]));
                c.write(l.append('\n').toString());
                if (mm[0]) {
                    multi.add(k);
                    m.write(k + "\n");
                }
                rows.put(k, r);
            }
        }
        return rows;
    }

    static int[] groups(Path file) throws IOException {
        String sha = FrameB.sha256(file);
        if (!sha.equals(GROUPS_SHA)) throw new IllegalStateException("bioid: " + file + " – SHA-256 " + sha + " вместо " + GROUPS_SHA);
        int[] g = new int[N];
        int n = 0;
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#")) continue;
            String[] c = l.split("\t");
            if (Integer.parseInt(c[0]) != n) throw new IllegalStateException("bioid: номер " + c[0] + " на месте " + n);
            g[n++] = Integer.parseInt(c[1]);
        }
        if (n != N) throw new IllegalStateException("bioid: снимков " + n);
        return g;
    }

    static String runText(Run r) {
        return r.from() == r.to() ? String.format(Locale.ROOT, "%04d", r.from()) : String.format(Locale.ROOT, "%04d–%04d", r.from(), r.to());
    }

    static String nums(List<Integer> l) {
        StringBuilder s = new StringBuilder();
        for (int i : l) s.append(s.length() == 0 ? "" : ",").append(String.format(Locale.ROOT, "%04d", i));
        return s.toString();
    }

    /** Роли по постановке: группы, отрезки, годность снимков, свои и только чужие, галерея, пробы, деление. */
    static final class Roles {
        int[] grp;
        Map<String, List<Run>> runs;
        Set<String> multi;
        Map<String, double[]> det;
        Set<Integer> valid;
        int noFace;
        int bigR;
        int noAlign;
        StringBuilder bad;
        List<String> own;
        List<String> only;
        Map<String, Integer> reg;
        Map<String, List<Integer>> gal;
        Map<String, List<Integer>> probes;
        Map<String, Integer> laterRaw;
        Map<String, List<Integer>> allValid;
        long seed;
        List<List<String>> halves;
    }

    static Roles roles(Path root, Path rep, SettingsStore settings) throws IOException {
        int[] grp = groups(rep.resolve("bioid_groups.tsv"));
        Map<String, List<Run>> runs = runs(grp);
        Set<String> multi = new HashSet<>();
        Map<String, double[]> det = detections(root, settings, multi);

        // Годность снимка: лицо найдено, |r| ≤ R, выравнивание возможно.
        Set<Integer> valid = new HashSet<>();
        int noFace = 0;
        int bigR = 0;
        int noAlign = 0;
        StringBuilder bad = new StringBuilder();
        for (int i = 0; i < N; i++) {
            double[] r = det.get(key(i));
            String why = null;
            if (r == null) {
                noFace++;
                why = "нет лица";
            } else if (Math.abs(FeiDataset.pose(r)) > R) {
                bigR++;
                why = String.format(Locale.ROOT, "|r| = %.3f", Math.abs(FeiDataset.pose(r)));
            } else if (!XBaseChokeRun.alignable(r)) {
                noAlign++;
                why = "не выравнивается";
            } else valid.add(i);
            if (why != null) bad.append(String.format(Locale.ROOT, "%04d (%s, %s)", i, person(grp[i]), why)).append("; ");
        }

        // Роли.
        List<String> own = new ArrayList<>();
        List<String> only = new ArrayList<>();
        Map<String, Integer> reg = new TreeMap<>();
        Map<String, List<Integer>> gal = new TreeMap<>();
        Map<String, List<Integer>> probes = new TreeMap<>();
        Map<String, Integer> laterRaw = new TreeMap<>();
        Map<String, List<Integer>> allValid = new TreeMap<>();
        for (Map.Entry<String, List<Run>> e : runs.entrySet()) {
            String q = e.getKey();
            List<Run> rs = e.getValue();
            List<Integer> av = new ArrayList<>();
            for (Run r : rs) for (int i = r.from(); i <= r.to(); i++) if (valid.contains(i)) av.add(i);
            allValid.put(q, av);
            int ri = regRun(rs);
            reg.put(q, ri);
            int later = 0;
            List<Integer> pr = new ArrayList<>();
            if (ri >= 0) for (int j = ri + 1; j < rs.size(); j++) {
                later += rs.get(j).len();
                for (int i = rs.get(j).from(); i <= rs.get(j).to(); i++) if (valid.contains(i)) pr.add(i);
            }
            laterRaw.put(q, later);
            if (ri >= 0 && later >= OWN_MIN) {
                List<Integer> g = gallery(rs.get(ri), valid);
                if (g == null) throw new IllegalStateException("bioid-inventory: у " + q + " не набралось " + GALLERY + " годных снимков в отрезке регистрации");
                own.add(q);
                gal.put(q, g);
                probes.put(q, pr);
            } else only.add(q);
        }
        long seed = settings.loadFacesSeed() + SPLIT_SEED_OFFSET;
        List<List<String>> halves = split(own, seed);
        Roles x = new Roles();
        x.grp = grp;
        x.runs = runs;
        x.multi = multi;
        x.det = det;
        x.valid = valid;
        x.noFace = noFace;
        x.bigR = bigR;
        x.noAlign = noAlign;
        x.bad = bad;
        x.own = own;
        x.only = only;
        x.reg = reg;
        x.gal = gal;
        x.probes = probes;
        x.laterRaw = laterRaw;
        x.allValid = allValid;
        x.seed = seed;
        x.halves = halves;
        return x;
    }

    static void inventory(SettingsStore settings) throws IOException {
        String d = System.getenv("FACES_BIOID_DIR");
        String t37 = System.getenv("FACES_P4_TABLE");
        if (d == null || t37 == null) throw new IllegalStateException("bioid-inventory: задайте FACES_BIOID_DIR, FACES_P4_TABLE");
        Path root = Paths.get(d);
        Path der = root.resolve("derived");
        Path rep = root.resolve("reports");
        Files.createDirectories(der);
        Roles x = roles(root, rep, settings);
        Map<String, List<Run>> runs = x.runs;
        Set<String> multi = x.multi;
        Set<Integer> valid = x.valid;
        StringBuilder bad = x.bad;
        List<String> own = x.own;
        List<String> only = x.only;
        Map<String, Integer> reg = x.reg;
        Map<String, List<Integer>> gal = x.gal;
        Map<String, List<Integer>> probes = x.probes;
        Map<String, Integer> laterRaw = x.laterRaw;
        Map<String, List<Integer>> allValid = x.allValid;
        long seed = x.seed;
        List<List<String>> halves = x.halves;
        int noFace = x.noFace;
        int bigR = x.bigR;
        int noAlign = x.noAlign;

        // Таблица людей.
        StringBuilder pt = new StringBuilder("person\truns\timages\tvalid\treg_run\trole\thalf\tgallery\tgallery_replaced\tlater_images\tprobes\tprobes_excluded\n");
        for (Map.Entry<String, List<Run>> e : runs.entrySet()) {
            String q = e.getKey();
            List<Run> rs = e.getValue();
            StringBuilder rt = new StringBuilder();
            int imgs = 0;
            for (Run r : rs) {
                rt.append(rt.length() == 0 ? "" : ",").append(runText(r).replace('–', '-'));
                imgs += r.len();
            }
            int ri = reg.get(q);
            boolean isOwn = own.contains(q);
            String half = isOwn ? (halves.get(0).contains(q) ? "1" : "2") : "-";
            int replaced = 0;
            if (isOwn) {
                Run r = rs.get(ri);
                for (int k = 0; k < GALLERY; k++) if (gal.get(q).get(k) != r.from() + (int) Math.round(k * (r.len() - 1) / (double) (GALLERY - 1))) replaced++;
            }
            int pr = isOwn ? probes.get(q).size() : 0;
            pt.append(String.join("\t", q, rt, Integer.toString(imgs), Integer.toString(allValid.get(q).size()), ri < 0 ? "-" : runText(rs.get(ri)).replace('–', '-'),
                    isOwn ? "own" : "impostor_only", half, isOwn ? nums(gal.get(q)) : "-", isOwn ? Integer.toString(replaced) : "-", Integer.toString(laterRaw.get(q)),
                    isOwn ? Integer.toString(pr) : "-", isOwn ? Integer.toString(laterRaw.get(q) - pr) : "-")).append('\n');
        }

        // Попытки по проходам.
        int[][] att = new int[2][4]; // свои: людей, попыток; чужие: людей, попыток
        StringBuilder ap = new StringBuilder("pass\trole\tperson\tattempts\n");
        for (int p = 0; p < 2; p++) {
            for (String q : halves.get(p)) {
                att[p][0]++;
                att[p][1] += probes.get(q).size();
                ap.append(p + 1).append("\town\t").append(q).append('\t').append(probes.get(q).size()).append('\n');
            }
            List<String> ctrl = new ArrayList<>(halves.get(1 - p));
            ctrl.addAll(only);
            Collections.sort(ctrl);
            for (String q : ctrl) {
                att[p][2]++;
                att[p][3] += allValid.get(q).size();
                ap.append(p + 1).append("\tctrl\t").append(q).append('\t').append(allValid.get(q).size()).append('\n');
            }
        }

        // Узел порога.
        Map<Integer, XBasePart4.TRow> tb = XBasePart4.readTable(Paths.get(t37));
        StringBuilder tt = new StringBuilder("pass\tN\tt37\tnode\n");
        for (int p = 0; p < 2; p++) {
            int n = halves.get(p).size();
            double[] a = XBaseFinalReport.tOf(tb, n);
            tt.append(String.format(Locale.ROOT, "%d\t%d\t%.2f\t%d%n", p + 1, n, a[0], (int) a[1]).replace("\r", ""));
        }

        Files.write(der.resolve("bioid_roles.tsv"), pt.toString().getBytes(StandardCharsets.UTF_8));
        Files.write(der.resolve("bioid_attempts.tsv"), ap.toString().getBytes(StandardCharsets.UTF_8));
        Files.write(der.resolve("bioid_thresholds.tsv"), tt.toString().getBytes(StandardCharsets.UTF_8));

        StringBuilder r = new StringBuilder("BioID, этап 1 (bioid-inventory): опись ролей по постановке 07.11 (SHA-256 0ef9a14e…). Без распознавания; YuNet – только для отбора |r| ≤ R и замены в галерее.\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append("\n");
        r.append("Люди – группы (а): ").append(rep.resolve("bioid_groups.tsv")).append(", SHA-256 ").append(GROUPS_SHA).append("\n");
        r.append("Детекция: снимок PGM → три канала (IMREAD_COLOR), YuNet ").append(FeretData.SCORE).append(" в два прохода, наибольшая рамка (как cp-run); кэш ").append(der.resolve(CACHE)).append("\n\n");
        r.append(String.format(Locale.ROOT, "Снимков %d; годных %d; нет лица %d; |r| > R = %.3f – %d; не выравнивается %d; несколько лиц в первом проходе %d%n", N, valid.size(), noFace, R, bigR, noAlign, multi.size()));
        r.append("Негодные: ").append(bad.length() == 0 ? "нет" : bad.substring(0, bad.length() - 2)).append("\n\n");
        r.append(String.format(Locale.ROOT, "Свои (отрезок регистрации ≥ %d снимков и ≥ %d снимков в более поздних отрезках) – %d: %s%n", REG_MIN, OWN_MIN, own.size(), own));
        r.append(String.format(Locale.ROOT, "Только чужие – %d: %s%n", only.size(), only));
        r.append(String.format(Locale.ROOT, "Деление, seed %d: половина 1 (%d) %s; половина 2 (%d) %s%n", seed, halves.get(0).size(), halves.get(0), halves.get(1).size(), halves.get(1)));
        for (int p = 0; p < 2; p++)
            r.append(String.format(Locale.ROOT, "Проход %d: свои – %d чел., %d попыток; контрольные чужие – %d чел., %d попыток%n", p + 1, att[p][0], att[p][1], att[p][2], att[p][3]));
        r.append(String.format(Locale.ROOT, "Итого за два прохода: свои – %d чел., %d попыток; чужие – %d единиц «человек в проходе», %d попыток%n",
                att[0][0] + att[1][0], att[0][1] + att[1][1], att[0][2] + att[1][2], att[0][3] + att[1][3]));
        r.append("\nЛюди (person | отрезки | снимков | годных | отрезок регистрации | роль | половина | галерея | замен | снимков позже | проб | исключено из проб):\n");
        r.append(pt.toString().replace("\t", " | "));
        r.append("\nУзел порога (XBaseFinalReport.tOf; N меньше 10 – t(10)):\n").append(tt);
        r.append("Таблица (37) – ").append(t37).append(", SHA-256 ").append(FrameB.sha256(Paths.get(t37))).append('\n');
        r.append("\nSHA-256 ролей bioid_roles.tsv ").append(FrameB.sha256(der.resolve("bioid_roles.tsv"))).append('\n');
        r.append("SHA-256 попыток bioid_attempts.tsv ").append(FrameB.sha256(der.resolve("bioid_attempts.tsv"))).append('\n');
        r.append("SHA-256 узлов bioid_thresholds.tsv ").append(FrameB.sha256(der.resolve("bioid_thresholds.tsv"))).append('\n');
        r.append("SHA-256 детекций ").append(CACHE).append(' ').append(FrameB.sha256(der.resolve(CACHE))).append('\n');
        Files.createDirectories(rep);
        Files.write(rep.resolve("bioid_roles.txt"), r.toString().getBytes(StandardCharsets.UTF_8));
        System.out.print(r);
    }

    // ---------------------------------------------------------------- прогон (bioid-run)

    static final double REF_X = 25.9;
    static final double REF_IN = 3.6;

    /** План прогона: план cp-run (для общих мер) плюс пробы своих, все снимки человека и номер отрезка снимка. */
    static final class BPlan {
        XBaseChokeRun.Plan p = new XBaseChokeRun.Plan();
        final Map<String, List<String>> ownPro = new TreeMap<>();
        final Map<String, List<String>> allImg = new TreeMap<>();
        /** Отрезок снимка относительно регистрации: «до», «1» (регистрация), «2», «3+»; у людей без регистрации – «-». */
        final Map<String, String> runOf = new HashMap<>();
    }

    static String runLabel(int j, int ri) {
        return ri < 0 ? "-" : j < ri ? "до" : j == ri ? "1" : j == ri + 1 ? "2" : "3+";
    }

    /** Кадры Б, k = 2 и |r| годных снимков; файлы – derived\a_b, derived\c. */
    static void frames(Roles x, Path root, XBaseChokeRun.Plan p, StringBuilder pts) throws IOException {
        Path der = root.resolve("derived");
        for (Path d : List.of(der.resolve(FrameB.DIR), der.resolve(FrameK2.DIR))) Files.createDirectories(d);
        for (int i = 0; i < N; i++) {
            if (!x.valid.contains(i)) continue;
            String k = key(i);
            double[] row = x.det.get(k);
            Mat color = Imgcodecs.imread(image(root.resolve("images"), i).toString(), Imgcodecs.IMREAD_COLOR);
            if (color.empty()) throw new IOException("BioID: нет снимка " + i);
            String name = String.format(Locale.ROOT, "bioid_%04d.png", i);
            Mat b = FrameB.frame(color, row, true);
            Mat gray = new Mat();
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
            FrameK2.F f = FrameK2.fromFull(gray, FrameK2.lm(row));
            FeretData.writePng(der.resolve(FrameB.DIR).resolve(name), b);
            FeretData.writePng(der.resolve(FrameK2.DIR).resolve(name), f.frame());
            pts.append(FrameK2.pointsLine(k, f.pts()));
            p.fr.put(k, b);
            p.k2.put(k, f);
            p.rOf.put(k, Math.abs(FeiDataset.pose(row)));
            gray.release();
            color.release();
        }
    }

    static BPlan realPlan(Path root, SettingsStore settings) throws IOException {
        Path rep = root.resolve("reports");
        Roles x = roles(root, rep, settings);
        BPlan bp = new BPlan();
        XBaseChokeRun.Plan p = bp.p;
        p.dir = rep.resolve("bioid");
        p.halves = x.halves;
        p.only2 = new ArrayList<>(x.only);
        p.all.addAll(x.runs.keySet());
        StringBuilder pts = new StringBuilder();
        frames(x, root, p, pts);
        for (Map.Entry<String, List<Run>> e : x.runs.entrySet()) {
            String q = e.getKey();
            List<Run> rs = e.getValue();
            int ri = x.reg.get(q);
            List<String> all = new ArrayList<>();
            for (int j = 0; j < rs.size(); j++)
                for (int i = rs.get(j).from(); i <= rs.get(j).to(); i++) {
                    if (!x.valid.contains(i)) continue;
                    String k = key(i);
                    all.add(k);
                    bp.runOf.put(k, runLabel(j, ri));
                    p.atts.add(new XBaseChokeRun.Att(q, runLabel(j, ri), "-", 0, "", k, XBaseChokeRun.iod(x.det.get(k))));
                }
            bp.allImg.put(q, all);
            if (x.own.contains(q)) {
                p.gal.put(q, x.gal.get(q).stream().map(XBaseBioId::key).toList());
                bp.ownPro.put(q, x.probes.get(q).stream().map(XBaseBioId::key).toList());
            }
        }
        int ex = 0;
        for (String q : x.own) ex += x.laterRaw.get(q) - x.probes.get(q).size();
        p.stats.append(String.format(Locale.ROOT, "%nОбработка: снимков %d, годных %d; нет лица %d, |r| > R %d, не выравнивается %d; несколько лиц в первом проходе %d.%n", N, x.valid.size(), x.noFace, x.bigR, x.noAlign, x.multi.size()));
        p.stats.append("Негодные: ").append(x.bad.length() == 0 ? "нет" : x.bad.substring(0, x.bad.length() - 2)).append('\n');
        p.stats.append(String.format(Locale.ROOT, "Свои %d %s, только чужие %d %s; половина 1 %s, половина 2 %s (seed %d); исключено из проб своих %d.%n",
                x.own.size(), x.own, x.only.size(), x.only, x.halves.get(0), x.halves.get(1), x.seed, ex));
        Files.writeString(root.resolve("derived").resolve("bioid_c_points.tsv"), pts.toString(), StandardCharsets.UTF_8);
        return bp;
    }

    /** Сухой прогон пути на GT по плану cp-run (BioID не читается): пробы своих – все попытки, номер отрезка – первые 4 попытки «2», остальные «3+». */
    static BPlan dryPlan(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out, XBaseA0.FeretSetup su) throws IOException {
        BPlan bp = new BPlan();
        bp.p = XBaseChokeRun.dryPlan(export, dir, feiKeys, selLines, settings, out, su);
        XBaseChokeRun.Plan p = bp.p;
        p.dir = out.resolve("bioid_dry");
        p.still.clear();
        Map<String, Integer> seen = new HashMap<>();
        for (XBaseChokeRun.Att a : p.atts) {
            if (a.key() == null) continue;
            int n = seen.merge(a.person(), 1, Integer::sum);
            bp.runOf.put(a.key(), n <= 4 ? "2" : "3+");
            bp.allImg.computeIfAbsent(a.person(), k -> new ArrayList<>()).add(a.key());
            if (p.gal.containsKey(a.person())) bp.ownPro.computeIfAbsent(a.person(), k -> new ArrayList<>()).add(a.key());
        }
        p.gal.forEach((q, ks) -> bp.allImg.computeIfAbsent(q, k -> new ArrayList<>()).addAll(ks));
        return bp;
    }

    /** Два прохода основной галереи; W и модели частей заново; части (Т2) – Random(faces.seed + 13010 + проход). */
    static XBaseFinal.Run setup(BPlan bp, XBaseA0.FeretSetup su, List<List<String>> train, SettingsStore settings, double r0, double rLimit) {
        XBaseChokeRun.Plan p = bp.p;
        long seed = settings.loadFacesSeed();
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < p.all.size(); i++) pIdx.put(p.all.get(i), i);
        Map<String, Integer> numOf = new HashMap<>();
        for (String k : new TreeSet<>(p.rOf.keySet())) numOf.put(k, numOf.size());
        XBaseVec.Setup u = new XBaseVec.Setup();
        u.base = "bioid";
        u.su = su;
        u.persons = p.all.size();
        u.fr = p.fr;
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.seed = seed;
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.params = IlluminationNorm.NormParams.of(settings);
        tpl.r0 = r0;
        tpl.rLimit = rLimit;
        tpl.persons = p.all;
        tpl.pIdx = pIdx;
        tpl.imageNo = numOf;
        tpl.rOf = p.rOf;
        for (String q : p.all) tpl.byPerson.put(q, new TreeMap<>(Map.of(1, bp.allImg.getOrDefault(q, List.of()))));
        u.tpl = tpl;
        u.groups.add("основная");
        for (int pass = 0; pass < 2; pass++) {
            List<String> own = new ArrayList<>(p.halves.get(pass));
            List<String> ctrl = new ArrayList<>(p.halves.get(1 - pass));
            ctrl.addAll(p.only2);
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            for (String q : own) {
                gal.put(q, p.gal.get(q));
                pro.put(q, bp.ownPro.getOrDefault(q, List.of()));
            }
            XBaseEval.Gt g = XBaseAbl.copy(tpl, train, p.rOf, r0);
            g.own = own;
            g.ctrl = ctrl;
            g.persons = p.all;
            g.pIdx = pIdx;
            g.imageNo = numOf;
            List<List<String>> ps = parts(own, seed, pass + 1);
            List<XBaseEval.Probe> probes = XBaseEval.probes(g, pro);
            u.blocks.add(new XBaseVec.Blk("p" + pass, "основная, %s, проход " + (pass + 1) + ", ", g, gal, pro, ps, probes, XBaseThr.extra(g, pro, ps), pass + 1, "основная"));
        }
        return new XBaseFinal.Run(u, p.k2, Map.of(), p.dir, false, 0);
    }

    static List<List<String>> parts(List<String> own, long seed, int pass) {
        return XBaseOther.cvParts(own, seed + 13010 + pass - FeiMethods.CV_SEED_OFFSET, 0);
    }

    /**
     * Итог разд. 4: интервал ошибки своих по людям против GT межс. 25,9 % и внутри сессии 3,6 %. «Содержит 25,9 % или выше» – верхняя граница
     * ≥ 25,9; «содержит 3,6 % или ниже» – нижняя граница ≤ 3,6; оба условия сразу – выводится как есть, без выбора слова.
     */
    static String verdict(double point, double lo, double hi) {
        boolean a = hi >= REF_X;
        boolean b = lo <= REF_IN;
        String word = a && !b ? "ближе к межсессионному" : b && !a ? "ближе к внутрисессионному" : !a ? "между" : "выполнены оба условия – правилом разд. 4 не различается";
        return String.format(Locale.ROOT, "ошибка своих по людям %.1f %% [%.1f; %.1f]; содержит %.1f %% или выше – %s; содержит %.1f %% или ниже – %s: %s", point, lo, hi, REF_X, a ? "да" : "нет", REF_IN, b ? "да" : "нет", word);
    }

    static void evaluate(BPlan bp, XBaseFinal.Run r, Map<Integer, XBasePart4.TRow> tb37, String head, String scoresName, String reportName) throws IOException {
        long t0 = System.nanoTime();
        XBaseChokeRun.Plan p = bp.p;
        XBaseVec.Setup u = r.u();
        Map<String, Double> sec = XBaseFinal.compute(r);
        for (Mat m : p.fr.values()) m.release();
        for (FrameK2.F k : p.k2.values()) k.frame().release();
        Map<String, Integer> pIdx = u.tpl.pIdx;
        int persons = u.persons;
        int[] unit = FeiBootstrap.unit(persons);
        int[][] w = u.tpl.b > 0 ? FeiBootstrap.weights(persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] clamped = new int[2];
        StringBuilder sc = new StringBuilder("# Файл оценок BioID (XBaseBioId); строки – формат FeiScores\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder t = new StringBuilder(head);
        t.append("Код: коммит ").append(FarMethods.commit()).append("; модели – ").append(r.dir().resolve("vec")).append(String.format(Locale.ROOT, "; бутстреп по людям B = %d, seed %d; людей %d%n", u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET, persons));
        t.append(p.stats);
        List<XBaseChokeRun.Res> r37 = new ArrayList<>();
        StringBuilder nodes = new StringBuilder();
        List<XBaseThr.Unit> uFus = new ArrayList<>();
        List<XBaseVec.Blk> main = u.blocks;
        for (int i = 0; i < 2; i++) {
            XBaseVec.Blk b = main.get(i);
            int n = b.g().own.size();
            double[] a = XBaseFinalReport.tOf(tb37, n);
            nodes.append(String.format(Locale.ROOT, "Проход %d: N = %d; t (37) %.2f (узел %d)%n", i + 1, n, a[0], (int) a[1]));
            r37.addAll(XBaseChokeRun.atT(b, i, r, a[0], p, false));
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, XBaseFinal.TT1, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            String fl = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, false), XBaseVec.fuseStats(mx, false), b.g().rOf, b.g().r0, clamped);
            sc.append("# слияние | ").append(b.name()).append('\n').append(fl);
            uFus.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), fl, null, true));
        }
        Files.createDirectories(r.dir());
        Path scf = r.dir().resolve(scoresName);
        Writer sw = XBaseVec.writer(scf, sc.toString());
        XBaseVec.finish(sw, scf);
        t.append("\n=== Пороги ===\n").append(nodes);
        t.append("\n=== Главные меры при рабочем пороге ===\nстрока | ошибка своих | «чужое имя» | FAR по попыткам | FAR по людям | свои по людям и по попыткам, 95 % интервал | FAR | верный лучший\n");
        t.append(XBaseChokeRun.mainRow("слияние (32), t по (37)", r37, pIdx, w, persons));
        t.append("\n=== Итог по разд. 4 постановки ===\n");
        double far = XBaseChokeRun.farAtt(r37, pIdx, unit);
        t.append(String.format(Locale.ROOT, "FAR по попыткам %.2f %%: %s (держит при ≤ 1 %% в точке).%n", far, far <= 1 ? "держит" : "не держит"));
        if (w.length > 0) {
            double[] d = FeiBootstrap.sorted(Arrays.stream(XBaseChokeRun.reps(a -> XBaseChokeRun.errPersons(r37, pIdx, a), w)).filter(v -> !Double.isNaN(v)).toArray());
            t.append(verdict(XBaseChokeRun.errPersons(r37, pIdx, unit), FeiBootstrap.quantile(d, 0.025), FeiBootstrap.quantile(d, 0.975))).append('\n');
        }
        t.append("\n=== Разбивка по номеру отрезка пробы (справочно) ===\n");
        t.append(XBaseChokeRun.mainRow("свои: второй отрезок; все чужие", r37.stream().filter(v -> !v.own() || "2".equals(bp.runOf.get(v.att().key()))).toList(), pIdx, w, persons));
        t.append(XBaseChokeRun.mainRow("свои: третий и далее; все чужие", r37.stream().filter(v -> !v.own() || "3+".equals(bp.runOf.get(v.att().key()))).toList(), pIdx, w, persons));
        t.append("\n=== Меры без порога (справочно; бутстреп по людям) ===\n");
        XBaseThr.Out oFus = XBaseThr.evaluate("слияние", uFus, persons, u.tpl);
        XBaseFinalReport.M mFus = XBaseFinalReport.measure(XBaseChokeRun.withFails(oFus.res.get(XBaseThr.WAYS[0]).cfgs, main, p, false, pIdx), persons, w);
        t.append(XBaseChokeRun.sepRow("слияние (32)", mFus));
        t.append("\n=== Время, с ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f%n", k, v)));
        t.append(String.format(Locale.ROOT, "модели, оценка и отчёт: %.0f%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(r.dir().resolve(reportName), t);
        System.out.print(t);
    }

    /** bioid-run: FACES_BIOID_DRY=1 – сухой прогон пути на GT; иначе однократный прогон. */
    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("bioid-run: FACES_FRAMES=b");
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("bioid-run: задайте FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        double rLimit = XBaseEval.poseLimit(selLines);
        if (r0 != 0.12 || Math.abs(rLimit - R) > 5e-4) throw new IllegalStateException("bioid-run: r0 = " + r0 + ", R = " + rLimit + " не совпали с постановкой");
        long seed = settings.loadFacesSeed();
        long t0 = System.nanoTime();
        Map<Integer, XBasePart4.TRow> tb37 = XBasePart4.readTable(Paths.get(System.getenv("FACES_P4_TABLE")));
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        if ("1".equals(System.getenv("FACES_BIOID_DRY"))) {
            BPlan bp = dryPlan(export, dir, feiKeys, selLines, settings, out, su);
            List<List<String>> train = XBaseChokeRun.train(bp.p, su, export, dir, feiKeys, settings, seed, 15);
            evaluate(bp, setup(bp, su, train, settings, r0, rLimit), tb37, "BioID – СУХОЙ ПРОГОН НА GT (числа смысла не имеют)\n", "bioid_dry_scores.tsv.gz", "bioid_dry_report.txt");
            System.out.println("DONE");
            return;
        }
        String root = System.getenv("FACES_BIOID_DIR");
        if (root == null) throw new IllegalStateException("bioid-run: задайте FACES_BIOID_DIR");
        BPlan bp = realPlan(Paths.get(root), settings);
        System.out.printf(Locale.ROOT, "bioid: кадры за %.0f с%n", (System.nanoTime() - t0) / 1e9);
        List<List<String>> train = XBaseChokeRun.train(bp.p, su, export, dir, feiKeys, settings, seed, 0);
        String head = "BioID – справочная проверка «другая серия» (постановка 07.11, SHA-256 0ef9a14e…0d90a; XBaseEval bioid-run); кадры изменены: выровнены\nДанные: BioID Face Database V1.2 (BioID AG); люди – группы (а) по SFace, проверены Хозяином\n";
        evaluate(bp, setup(bp, su, train, settings, r0, rLimit), tb37, head, "bioid_scores.tsv.gz", "bioid_report.txt");
        System.out.printf(Locale.ROOT, "Всего: %.0f с%n", (System.nanoTime() - t0) / 1e9);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- внешние чужие (bioid-ext, постановка 08.11)

    /** Снимок внешнего чужого: база, человек, ключ, |r|. */
    record Ext(String base, String person, String key, double r) {}

    /** Опись E: отобранные снимки и исключения по базам. */
    static final class ExtSet {
        final List<Ext> list = new ArrayList<>();
        final Map<String, int[]> count = new java.util.LinkedHashMap<>(); // база → {людей всего, людей исключено, снимков всего, нет лица, |r| > R, не выравнивается, отобрано}
        final Map<String, java.util.TreeSet<String>> persons = new TreeMap<>();
        final Map<String, String> excludedPersons = new java.util.LinkedHashMap<>();
    }

    static void addExt(ExtSet e, String base, String person, String key, double[] row) {
        int[] c = e.count.computeIfAbsent(base, b -> new int[7]);
        c[2]++;
        if (row == null) {
            c[3]++;
            return;
        }
        double r = Math.abs(FeiDataset.pose(row));
        if (r > R) {
            c[4]++;
            return;
        }
        if (!XBaseChokeRun.alignable(row)) {
            c[5]++;
            return;
        }
        c[6]++;
        e.list.add(new Ext(base, person, key, r));
        e.persons.computeIfAbsent(base, b -> new java.util.TreeSet<>()).add(person);
    }

    /** s = 2ℓ₁ − ℓ₂ слияния (32) по строкам моделей (как XBaseVec.fuseLines): {s, лучший}; второго нет – s = +∞. */
    static double[] fuseS(XBaseVec.Row rt, XBaseVec.Row rx, double[] st, double[] sx) {
        int ng = rt.dg().length;
        int b = -1;
        int c = -1;
        double[] ln = new double[ng];
        for (int g = 0; g < ng; g++) {
            ln[g] = Double.isInfinite(rt.dg()[g]) || Double.isInfinite(rx.dg()[g]) ? Double.POSITIVE_INFINITY
                    : XBaseFeat2.W_FUSE * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
            if (b < 0 || ln[g] < ln[b]) {
                c = b;
                b = g;
            } else if (c < 0 || ln[g] < ln[c]) c = g;
        }
        if (c < 0 || Double.isInfinite(ln[b]) || Double.isInfinite(ln[c])) return new double[] {Double.POSITIVE_INFINITY, b};
        return new double[] {2 * ln[b] - ln[c], b};
    }

    /** Наибольший t, при котором доля s ≤ t не больше far: следующее вниз от (⌊far·n⌋ + 1)-го наименьшего s. */
    static double tAt(double[] s, double far) {
        double[] v = s.clone();
        java.util.Arrays.sort(v);
        int k = (int) Math.floor(far * v.length + 1e-9);
        return k >= v.length ? Double.POSITIVE_INFINITY : Math.nextDown(v[k]);
    }

    /** Попытка BioID или E: проход, роль (own / ctrl / ext), человек, база E, s, верный ли лучший. */
    record Sc(int pass, String role, String person, String base, double s, boolean right) {}

    static void ext(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("bioid-ext: FACES_FRAMES=b");
        double r0 = FeiPose.r0(settings.loadFacesFeiPoseR0(), "+pose");
        double rLimit = XBaseEval.poseLimit(selLines);
        if (r0 != 0.12 || Math.abs(rLimit - R) > 5e-4) throw new IllegalStateException("bioid-ext: r0 = " + r0 + ", R = " + rLimit);
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        Path root = Paths.get(System.getenv("FACES_BIOID_DIR"));
        Path od = root.resolve("reports").resolve("bioid").resolve("ext");
        Files.createDirectories(od);
        Path fd = Paths.get(System.getenv("FACES_FERET_DIR")).resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Map<Integer, XBasePart4.TRow> tb37 = XBasePart4.readTable(Paths.get(System.getenv("FACES_P4_TABLE")));
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        BPlan bp = realPlan(root, settings);
        List<List<String>> train = XBaseChokeRun.train(bp.p, su, export, dir, feiKeys, settings, seed, 0);
        XBaseFinal.Run run = setup(bp, su, train, settings, r0, rLimit);
        XBaseVec.Setup u = run.u();

        // Опись E.
        ExtSet e = new ExtSet();
        Set<String> cohort = new java.util.HashSet<>(XBaseEval.cohortPersons(feiKeys, seed));
        Set<String> pool = new java.util.HashSet<>(su.ro.own1);
        Map<String, double[]> expRows = FrameK2.exportRows(dir);
        Map<String, GalleryEvaluation.Sample> bySample = new HashMap<>();
        for (String[] bs : new String[][] {{"GT", ""}, {"MUCT", "429"}, {"FEI", "071"}}) {
            Map<String, List<GalleryEvaluation.Sample>> m = bs[0].equals("GT") ? export.data().gt() : bs[0].equals("MUCT") ? export.data().muct() : export.data().fei();
            int[] c = e.count.computeIfAbsent(bs[0], b -> new int[7]);
            for (Map.Entry<String, List<GalleryEvaluation.Sample>> en : new TreeMap<>(m).entrySet()) {
                String q = en.getKey();
                c[0]++;
                String why = q.equals(bs[1]) || q.endsWith("/" + bs[1]) ? "дубль (38)" : bs[0].equals("FEI") && cohort.contains(q) ? "когорта прогона" : null;
                if (why != null) {
                    c[1]++;
                    e.excludedPersons.put(bs[0] + " " + q, why);
                    continue;
                }
                for (GalleryEvaluation.Sample s : en.getValue()) {
                    String k = s.file().toString();
                    bySample.put(k, s);
                    addExt(e, bs[0], q, k, expRows.get(k));
                }
            }
        }
        Map<String, List<FeretData.Img>> fp = new TreeMap<>();
        for (FeretData.Img im : su.imgs) fp.computeIfAbsent(im.person(), p -> new ArrayList<>()).add(im);
        int[] cf = e.count.computeIfAbsent("FERET", b -> new int[7]);
        for (Map.Entry<String, List<FeretData.Img>> en : fp.entrySet()) {
            String q = en.getKey();
            cf[0]++;
            String why = q.equals("00700") ? "дубль (38)" : pool.contains(q) ? "пул прогона" : null;
            if (why != null) {
                cf[1]++;
                if (!why.equals("пул прогона")) e.excludedPersons.put("FERET " + q, why);
                continue;
            }
            for (FeretData.Img im : en.getValue()) if (im.frontal()) addExt(e, "FERET", q, im.key(), su.det.get(im.key()));
        }

        // Кадры Б и k = 2 для E (нет кадра – снимок исключается, числом).
        Map<String, Integer> noFrame = new TreeMap<>();
        List<Ext> use = new ArrayList<>();
        List<String> feretK2 = new ArrayList<>();
        List<String> expK2 = new ArrayList<>();
        List<GalleryEvaluation.Sample> muctK2 = new ArrayList<>();
        for (Ext x : e.list) {
            Mat f;
            try {
                f = x.base().equals("FERET") ? FeretData.frame(su.dir, "a", x.key()) : export.frame(bySample.get(x.key()), GalleryEvaluation.Variant.A);
            } catch (IOException ex) {
                f = null;
            }
            if (f == null || f.empty()) {
                noFrame.merge(x.base(), 1, Integer::sum);
                continue;
            }
            bp.p.fr.put(x.key(), f);
            use.add(x);
            if (x.base().equals("FERET")) feretK2.add(x.key());
            else if (x.base().equals("MUCT")) muctK2.add(bySample.get(x.key()));
            else expK2.add(x.key());
        }
        List<String> fe = new ArrayList<>(feretK2);
        fe.addAll(expK2);
        bp.p.k2.putAll(FrameK2.load(fe, su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir())));
        bp.p.k2.putAll(DuelThr.k2Raw("muct", muctK2, export, dir, Paths.get(settings.loadFacesMuctDir())));
        int next = 100000;
        for (Ext x : use) {
            bp.p.rOf.put(x.key(), x.r());
            u.tpl.imageNo.put(x.key(), next++);
        }
        System.out.printf(Locale.ROOT, "bioid-ext: E – %d снимков, кадры за %.0f с%n", use.size(), (System.nanoTime() - t0) / 1e9);

        // Модели: W основной модели каждого прохода заново (XBaseVec.model, MldaScorer – как прогон), сверка строк BioID с файлами прогона.
        List<Sc> sc = new ArrayList<>();
        StringBuilder chk = new StringBuilder();
        Map<String, Double> sec = new java.util.LinkedHashMap<>();
        Map<XBaseFinal.Cfg, XBaseEval> xs = new java.util.LinkedHashMap<>();
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            long tv = System.nanoTime();
            xs.put(c, XBasePart4.vectorsB(bp.p.fr, bp.p.k2, XBaseFinal.param(c, u.tpl.params), c.f(), u.tpl.threads));
            sec.put(c.id() + " признаки", (System.nanoTime() - tv) / 1e9);
        }
        for (Mat m : bp.p.fr.values()) m.release();
        for (int pass = 0; pass < 2; pass++) {
            XBaseVec.Blk b = u.blocks.get(pass);
            List<XBaseEval.Probe> probes = new ArrayList<>(b.probes());
            for (Ext x : use) probes.add(new XBaseEval.Probe(FeiScores.CTRL, "ext:" + x.base() + ":" + x.person(), u.tpl.imageNo.get(x.key()), x.key(), -1));
            List<List<String>> gallery = b.g().own.stream().map(b.gal()::get).toList();
            List<java.util.concurrent.Callable<XBaseVec.Model>> jobs = new ArrayList<>();
            XBaseEval xt = xs.get(XBaseFinal.TT1);
            XBaseEval xx = xs.get(XBaseFinal.XS);
            String mx = XBaseEval.V.L1.id;
            String mm = mx + FeiMethods.MIRROR;
            jobs.add(() -> XBaseVec.model(xt, false, mx, b.s(), b.g().own, gallery, b.g().cohort, u.cin, u.cout, probes, "main"));
            jobs.add(() -> XBaseVec.model(xt, true, mm, b.s(), b.g().own, gallery, b.g().cohort, u.cin, u.cout, probes, "main"));
            jobs.add(() -> XBaseVec.model(xx, true, mm, b.s(), b.g().own, gallery, b.g().cohort, u.cin, u.cout, probes, "main"));
            long tm = System.nanoTime();
            List<XBaseVec.Model> done = XBaseVec.run(List.of(jobs), u.tpl.threads).get(0);
            sec.put("W прохода " + (pass + 1), (System.nanoTime() - tm) / 1e9);
            List<XBaseVec.Model> ft = XBaseFinal.models(run, b, XBaseFinal.TT1, false);
            List<XBaseVec.Model> fx = XBaseFinal.models(run, b, XBaseFinal.XS, false);
            XBaseVec.Model[] file = {XBaseVec.find(ft, "main", false), XBaseVec.find(ft, "main", true), XBaseVec.find(fx, "main", true)};
            String[] name = {"L1-tt-xs-F X", "L1-tt-xs-F X+mirror", "L1-xs-F X+mirror"};
            for (int k = 0; k < 3; k++) {
                List<XBaseVec.Row> mine = done.get(k).rows();
                List<XBaseVec.Row> ref = file[k].rows();
                int bad = 0;
                if (!done.get(k).gallery().equals(file[k].gallery())) bad = -1;
                for (int i = 0; i < ref.size() && bad >= 0; i++) {
                    XBaseVec.Row a = mine.get(i);
                    XBaseVec.Row r = ref.get(i);
                    boolean eq = a.key().equals(r.key()) && a.role().equals(r.role()) && a.image() == r.image() && a.dg().length == r.dg().length;
                    for (int g = 0; eq && g < r.dg().length; g++) eq = Double.doubleToLongBits(a.dg()[g]) == Double.doubleToLongBits(r.dg()[g]);
                    if (!eq) bad++;
                }
                chk.append(String.format(Locale.ROOT, "Проход %d, %s: строк BioID %d, совпало побайтно %s%n", pass + 1, name[k], ref.size(), bad == 0 ? "все" : bad < 0 ? "НЕТ (галерея)" : "НЕТ у " + bad));
                if (bad != 0) {
                    FarMethods.write(od.resolve("bioid_ext_check.txt"), chk);
                    throw new IllegalStateException("bioid-ext: сверка с прогоном не прошла – " + chk);
                }
            }
            double[] st = XBaseVec.fuseStats(ft, false);
            double[] sx = XBaseVec.fuseStats(fx, false);
            List<XBaseVec.Row> rX = done.get(0).rows(), rM = done.get(1).rows(), rx = done.get(2).rows();
            for (int i = 0; i < rX.size(); i++) {
                XBaseVec.Row a = rX.get(i);
                XBaseVec.Row rt = bp.p.rOf.get(a.key()) <= r0 ? a : rM.get(i);
                double[] d = fuseS(rt, rx.get(i), st, sx);
                boolean extRow = a.person().startsWith("ext:");
                String base = extRow ? a.person().split(":")[1] : "";
                String person = extRow ? a.person().substring(4) : a.person();
                sc.add(new Sc(pass, extRow ? "ext" : a.role(), person, base, d[0], a.trueIdx() >= 0 && (int) d[1] == a.trueIdx()));
            }
        }

        // Пороги и меры.
        Map<String, Integer> pIdx = u.tpl.pIdx;
        int persons = u.persons;
        int[][] w = FeiBootstrap.weights(persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        List<String> bases = new ArrayList<>(List.of("все"));
        for (String b : e.count.keySet()) if (use.stream().anyMatch(x -> x.base().equals(b))) bases.add(b);
        StringBuilder t = new StringBuilder("BioID – порог под свою галерею по внешним чужим (постановка 08.11, SHA-256 12680c83…; XBaseEval bioid-ext; уточнения Жорика №2 от 08.11)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append("; модели прогона – ").append(run.dir().resolve("vec")).append(" (код 1f9990a); (Т2) – из файлов прогона; бутстреп по людям BioID B = ")
                .append(u.tpl.b).append(", seed ").append(u.tpl.seed + FeiBootstrap.SEED_OFFSET).append("\n\n");
        t.append("=== 1. Внешние чужие E (кадры Б; отбор как у проб (32): лицо, |r| ≤ 0,387, выравнивание) ===\n");
        t.append("База | людей в базе | исключено людей | снимков у оставшихся | нет лица | |r| > R | не выравнивается | отобрано | нет кадра Б | в расчёте (снимков / людей)\n");
        for (Map.Entry<String, int[]> en : e.count.entrySet()) {
            int[] c = en.getValue();
            int nf = noFrame.getOrDefault(en.getKey(), 0);
            long nu = use.stream().filter(x -> x.base().equals(en.getKey())).count();
            long np = use.stream().filter(x -> x.base().equals(en.getKey())).map(Ext::person).distinct().count();
            t.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d | %d | %d | %d | %d | %d / %d%n", en.getKey(), c[0], c[1], c[2], c[3], c[4], c[5], c[6], nf, nu, np));
        }
        t.append("Исключённые люди (кроме пула FERET – ").append(pool.size()).append(" человек): ").append(e.excludedPersons).append('\n');
        t.append("\n=== Сверка повторной W с прогоном (строки проб BioID, расстояния побайтно) ===\n").append(chk);
        // Порог t_E по проходу и мерам.
        Map<String, double[]> tE = new java.util.LinkedHashMap<>(); // ключ «база|far» → {t прохода 1, t прохода 2}
        for (String b : bases)
            for (double far : new double[] {0.01, 0.05}) {
                double[] tt = new double[2];
                for (int pass = 0; pass < 2; pass++) {
                    int pp = pass;
                    double[] s = sc.stream().filter(x -> x.pass() == pp && x.role().equals("ext") && (b.equals("все") || x.base().equals(b))).mapToDouble(Sc::s).toArray();
                    tt[pass] = tAt(s, far);
                }
                tE.put(b + "|" + far, tt);
            }
        t.append("\n=== 2. Порог t_E (наибольший, при котором доля принятых снимков E ≤ FAR; по проходу) ===\n");
        t.append("E | FAR E | t прохода 1 | t прохода 2 | принято E (по сумме проходов)\n");
        for (Map.Entry<String, double[]> en : tE.entrySet()) {
            String[] k = en.getKey().split("\\|");
            long acc = 0, n = 0;
            for (Sc x : sc) if (x.role().equals("ext") && (k[0].equals("все") || x.base().equals(k[0]))) {
                n++;
                if (x.s() <= en.getValue()[x.pass()]) acc++;
            }
            t.append(String.format(Locale.ROOT, "%s | %.0f %% | %.4f | %.4f | %d/%d = %.2f %%%n", k[0], 100 * Double.parseDouble(k[1]), en.getValue()[0], en.getValue()[1], acc, n, 100.0 * acc / n));
        }
        t.append("\n=== 3. Меры на BioID (настоящие чужие и свои – как в прогоне; бутстреп по людям BioID) ===\n");
        t.append("порог | ошибка своих | «чужое имя» | FAR по попыткам | FAR по людям | свои по людям и по попыткам, 95 % интервал | FAR | верный лучший\n");
        List<String[]> rows = new ArrayList<>();
        double t10 = XBaseFinalReport.tOf(tb37, 8)[0];
        rows.add(new String[] {String.format(Locale.ROOT, "рабочий t(10) = %.2f", t10), Double.toString(t10), Double.toString(t10)});
        for (Map.Entry<String, double[]> en : tE.entrySet()) {
            String[] k = en.getKey().split("\\|");
            rows.add(new String[] {String.format(Locale.ROOT, "t_E %s, FAR E %.0f %%", k[0], 100 * Double.parseDouble(k[1])), Double.toString(en.getValue()[0]), Double.toString(en.getValue()[1])});
        }
        for (String[] rr : rows) {
            double[] tt = {Double.parseDouble(rr[1]), Double.parseDouble(rr[2])};
            List<XBaseChokeRun.Res> l = new ArrayList<>();
            for (Sc x : sc) if (!x.role().equals("ext")) l.add(new XBaseChokeRun.Res(x.person(), x.pass(), x.role().equals("own"), x.s() <= tt[x.pass()], x.right(), null));
            t.append(XBaseChokeRun.mainRow(rr[0], l, pIdx, w, persons));
        }
        long acc10 = sc.stream().filter(x -> x.role().equals("ctrl") && x.s() <= t10).count();
        long err10 = sc.stream().filter(x -> x.role().equals("own") && !(x.s() <= t10 && x.right())).count();
        t.append(String.format(Locale.ROOT, "Сверка с прогоном при t(10): принято чужих %d (прогон 46), ошибок своих %d (прогон 239).%n", acc10, err10));
        t.append("Предел (прогон, те же модели): разделимость при фактическом FAR 1 % по настоящим чужим – 37,8 % [4,3; 60,4] (факт. 0,90 %).\n");
        double far1 = Double.NaN;
        {
            double[] tt = tE.get("все|0.01");
            long a = 0, n = 0;
            for (Sc x : sc) if (x.role().equals("ctrl")) {
                n++;
                if (x.s() <= tt[x.pass()]) a++;
            }
            far1 = 100.0 * a / n;
        }
        t.append(String.format(Locale.ROOT, "%n=== 4. Итог по разд. 4 ===%nFAR настоящих чужих BioID при t_E (все внешние, FAR E 1 %%) по попыткам в точке %.2f %%: %s.%n", far1,
                far1 <= 1 ? "способ годится на этой базе (≤ 1 %); цена – ошибка своих при t_E (строка выше) против разделимости 37,8 %" : "внешние чужие не заменяют местных на этой базе (> 1 %)"));
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f с%n", k, v)));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(od.resolve("bioid_ext.txt"), t);
        // Оценки: BioID и E не FERET – в ext\, E FERET – только в Feret\reports\eq_b\.
        StringBuilder a = new StringBuilder("# проход\tроль\tчеловек\tбаза E\ts\tверный лучший\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder bF = new StringBuilder(a);
        for (Sc x : sc) (x.base().equals("FERET") ? bF : a).append(String.format(Locale.ROOT, "%d\t%s\t%s\t%s\t%.17g\t%b%n", x.pass() + 1, x.role(), x.person(), x.base(), x.s(), x.right()));
        Files.writeString(od.resolve("bioid_ext_scores.tsv"), a, StandardCharsets.UTF_8);
        Files.writeString(fd.resolve("bioid_ext_feret_scores.tsv"), bF, StandardCharsets.UTF_8);
        System.out.print(t);
        System.out.println("DONE");
    }
}
