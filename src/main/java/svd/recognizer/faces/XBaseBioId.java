package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
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

    static void inventory(SettingsStore settings) throws IOException {
        String d = System.getenv("FACES_BIOID_DIR");
        String t37 = System.getenv("FACES_P4_TABLE");
        if (d == null || t37 == null) throw new IllegalStateException("bioid-inventory: задайте FACES_BIOID_DIR, FACES_P4_TABLE");
        Path root = Paths.get(d);
        Path der = root.resolve("derived");
        Path rep = root.resolve("reports");
        Files.createDirectories(der);
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
}
