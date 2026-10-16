package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseMpPts.Mp;
import svd.recognizer.storage.SettingsStore;

/**
 * Пункт (8) MediaPipe, этап 2 (PLAN.md п. 5.5; план – №8–№9 Жорика от 16.10.2026, вариант «оба вместе»): распознавание при
 * схеме «МП» против базы. Схема «МП»: МП на вырезке по рамке YuNet (XBaseMpPts, mp_points_crop3.tsv), кадр 92×112 – подобие по
 * радужкам 468/473 и углам рта 61/291 МП, нос – YuNet; отбор |yaw| ≤ R_МП и выбор модели X / X+mirror по |yaw| ≤ r0_МП (yaw МП,
 * градусы; R_МП – 95-й процентиль |yaw| снимков своей базы, r0_МП – 95-й процентиль |yaw| №11–14 FEI). База – кадр «а», r
 * YuNet, R = 0,387, r0 = 0,12.
 * <p>
 * Наборы: U – YuNet и МП нашли лицо; I – U, отобранные обеими схемами. Роли, галереи, обучение W и 5 моделей частей – на I,
 * одни для обеих схем. (б) – пробы из I; (а) «как в ПО» – пробы из U, проба, отвергнутая отбором своей схемы, у своего – отказ,
 * у чужого – не принят (d₂ = +∞, FeiScores.ratio, п. 4). Остановка до счёта: I меньше прежнего набора больше чем на 5 % снимков
 * галереи или проб своих в любой постановке. Строки базы на прежних ролях – побайтно с feret-xs2 / feret-subsp / gt-subsp
 * (k = 0, строки L1 и +mirror).
 * <p>
 * feret-mp2 – FERET eq: межс. (1 дата), внутри, G2 (2 даты), половины №2, два прохода; отчёт и оценки – только
 * reports/eq/xbase_feret_mp2*. gt-mp2 – GT межс. N = 6, внутри N = 3, пул FERET; reports/faces/xbase/xbase_gt_mp2*.
 *
 * @author ssv
 */
final class XBaseMp2 {

    static final String CROP_FILE = "mp_points_crop3.tsv";
    static final double STOP_SHARE = 0.05;
    static final String[] SCHEMES = {"база", "МП"};

    private XBaseMp2() {
    }

    // ---------------------------------------------------------------- пороги и кадры

    /** Пороги схемы «МП»: R_МП, r0_МП и объёмы, по которым они посчитаны. */
    record Thr(double bigR, int ownN, int ownFound, double r0, int feiN, int feiFound) {}

    /** R_МП – снимки протокола своей базы (рамки mp_boxes.tsv), где МП нашёл лицо на вырезке; r0_МП – №11–14 FEI (все люди). */
    static Thr thresholds(SettingsStore settings) throws IOException {
        Path own = Paths.get(settings.loadFacesOwnDir());
        Map<String, Mp> mo = XBaseMpPts.read(own.resolve(CROP_FILE), new ArrayList<>());
        List<String> boxes = Files.readAllLines(own.resolve(XBaseMpPts.BOXES), StandardCharsets.UTF_8).stream().filter(l -> !l.startsWith("#") && !l.isBlank())
                .map(l -> l.substring(0, l.indexOf('\t'))).toList();
        double[] y = boxes.stream().map(mo::get).filter(Mp::found).mapToDouble(m -> Math.abs(m.yaw())).sorted().toArray();
        Map<String, Mp> mf = XBaseMpPts.read(Paths.get(settings.loadFacesFeiDir()).resolve(CROP_FILE), new ArrayList<>());
        List<String> front = mf.keySet().stream().filter(k -> {
            int n = Integer.parseInt(k.substring(k.indexOf('-') + 1));
            return n >= 11 && n <= 14;
        }).toList();
        double[] f = front.stream().map(mf::get).filter(Mp::found).mapToDouble(m -> Math.abs(m.yaw())).sorted().toArray();
        return new Thr(FeiDataset.percentile(y, FeiDataset.POSE_PERCENTILE), boxes.size(), y.length, FeiDataset.percentile(f, FeiDataset.POSE_PERCENTILE),
                front.size(), f.length);
    }

    /** Кадр «МП» 92×112: строка YuNet с глазами (радужки) и углами рта МП, нос – YuNet; выравнивание – как кадр «а». */
    static Mat frameMp(Mat color, double[] yunet, Mp m) {
        double[] row = yunet.clone();
        double[][] p = {m.p().get("iris_re"), m.p().get("iris_le"), m.p().get("mouth_r"), m.p().get("mouth_l")};
        int[] at = {4, 6, 10, 12};
        for (int i = 0; i < 4; i++) {
            row[at[i]] = p[i][0];
            row[at[i] + 1] = p[i][1];
        }
        return FeretData.frameA(color, row);
    }

    /** Кадры «МП» FERET: кэш derived/a_mp (уже записанные не пересчитываются). */
    static Map<String, Mat> feretMpFrames(Path dir, Set<String> keys, Map<String, Img> byKey, Map<String, double[]> det, Map<String, Mp> mp) throws IOException {
        Path d = dir.resolve(FeretData.DERIVED).resolve("a_mp");
        Files.createDirectories(d);
        Map<String, Mat> out = new LinkedHashMap<>();
        int n = 0;
        for (String k : keys) {
            Path f = d.resolve(k + ".png");
            if (!Files.exists(f)) {
                Mat color = FeretData.decode(byKey.get(k).file());
                Mat a = frameMp(color, det.get(k), mp.get(k));
                FeretData.writePng(f, a);
                a.release();
                color.release();
                if (++n % 500 == 0) System.out.printf(Locale.ROOT, "mp2: кадры МП FERET – %d новых%n", n);
            }
            out.put(k, FeretData.frame(dir, "a_mp", k));
        }
        return out;
    }

    /** Кадры «МП» когорты FEI (в памяти): исходный снимок, YuNet экспорта, МП FEI. Ключ – как у кадров экспорта. */
    static Map<String, Mat> feiMpFrames(FarExport.Loaded export, Path exportDir, Set<String> cohortKeys, SettingsStore settings) throws IOException {
        Path feiDir = Paths.get(settings.loadFacesFeiDir());
        Map<String, Mp> mf = XBaseMpPts.read(feiDir.resolve(CROP_FILE), new ArrayList<>());
        Map<String, Path> raw = new HashMap<>();
        for (List<Sample> l : FeiDataset.originals(feiDir).values()) for (Sample s : l) raw.put(s.file().getFileName().toString().replaceAll("(?i)\\.jpg$", ""), s.file());
        Map<String, double[]> rows = XBaseMpPts.exportRows(exportDir, "fei");
        Map<String, Mat> out = new LinkedHashMap<>();
        for (List<Sample> l : export.data().fei().values()) {
            for (Sample s : l) {
                String k = s.file().toString();
                if (!cohortKeys.contains(k)) continue;
                String name = s.file().getFileName().toString();
                Mp m = mf.get(name);
                double[] row = rows.get(s.person() + "/" + name);
                if (m == null || !m.found() || row == null) throw new IllegalStateException("mp2: снимок когорты FEI без МП или YuNet: " + name);
                Mat color = Imgcodecs.imread(raw.get(name).toString(), Imgcodecs.IMREAD_COLOR);
                out.put(k, frameMp(color, row, m));
                color.release();
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- оценки (а)

    /** Отказ отбором: у строк проб из rejected (роль, человек, номер снимка) второе расстояние = +∞ (FeiScores.ratio, п. 4). */
    static String refuse(String lines, Set<String> rejected) {
        StringBuilder b = new StringBuilder();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (rejected.contains(c[3] + "\t" + c[4] + "\t" + c[5])) {
                c[10] = "Infinity";
                b.append(String.join("\t", c));
            } else {
                b.append(l);
            }
            b.append('\n');
        }
        return b.toString();
    }

    /** Строки проб из keep (роль, человек, номер снимка) и всех строк частей cv – для сверки с прежними файлами. */
    static String only(String lines, Set<String> keep) {
        StringBuilder b = new StringBuilder();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (c[3].startsWith(FeiMethods.CV_ROLE) || keep.contains(c[3] + "\t" + c[4] + "\t" + c[5])) b.append(l).append('\n');
        }
        return b.toString();
    }

    static Set<String> ids(List<Probe> probes) {
        Set<String> out = new HashSet<>();
        for (Probe q : probes) out.add(q.role() + "\t" + q.person() + "\t" + q.image());
        return out;
    }

    /** Накопитель результатов строки: конфигурации и «чужие» по проходам. */
    static final class Acc {
        final List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        final List<FeiThreshold.Imp> imps = new ArrayList<>();

        void add(XBaseEval.Parsed p) {
            cfgs.add(p.cfg.get(V.L1));
            imps.add(p.imp.get(V.L1));
        }
    }

    static int size(Map<String, List<String>> m, List<String> persons) {
        int n = 0;
        for (String p : persons) n += m.getOrDefault(p, List.of()).size();
        return n;
    }

    // ---------------------------------------------------------------- FERET

    /** Набор FERET: U, I, роли на I и U, позы схем. */
    static final class FeretSets {
        XBaseA0.FeretSetup su;
        Thr th;
        Map<String, Mp> mp;
        List<String> mpHead = new ArrayList<>();
        Set<String> u = new TreeSet<>();
        Set<String> i = new TreeSet<>();
        int detected;
        int mpMissing;
        Map<String, Double> rB = new HashMap<>();
        Map<String, Double> yM = new HashMap<>();
        XBaseFeret.Roles roI;
        XBaseFeret.Roles roU;
        XBaseFeret.Roles ro;
        XBaseFeret.Roles ro0;
        Map<String, List<String>> ctrlKeysU = new TreeMap<>();
        Map<String, List<List<String>>> pool;
        Map<String, List<List<String>>> poolU;

        boolean rejB(String k) {
            return rB.get(k) > su.tpl.rLimit;
        }

        boolean rejM(String k) {
            return yM.get(k) > th.bigR();
        }
    }

    static FeretSets feretSets(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        FeretSets fs = new FeretSets();
        fs.su = XBaseA0.feretSetup(exportDir, selLines, settings);
        fs.th = thresholds(settings);
        XBaseA0.FeretSetup su = fs.su;
        fs.mp = XBaseMpPts.read(su.dir.resolve(FeretData.DERIVED).resolve(CROP_FILE), fs.mpHead);
        Map<String, Map<String, List<Img>>> keptI = new TreeMap<>();
        Map<String, Map<String, List<Img>>> keptU = new TreeMap<>();
        for (Img im : su.imgs) {
            double[] row = su.det.get(im.key());
            if (row == null) continue;
            fs.detected++;
            Mp m = fs.mp.get(im.key());
            if (m == null) throw new IllegalStateException("mp2: нет строки МП (вырезка) для " + im.key());
            if (!m.found()) {
                fs.mpMissing++;
                continue;
            }
            fs.rB.put(im.key(), Math.abs(FeiDataset.pose(row)));
            fs.yM.put(im.key(), Math.abs(m.yaw()));
            fs.u.add(im.key());
            keptU.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
            if (!fs.rejB(im.key()) && !fs.rejM(im.key())) {
                fs.i.add(im.key());
                keptI.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
            }
        }
        Map<String, TreeSet<String>> dates = XBaseFeret.dates(su.imgs);
        long seed = su.tpl.seed;
        fs.roI = XBaseFeret.roles(dates, keptI, seed);
        fs.roU = XBaseFeret.roles(dates, keptU, seed);
        fs.ro = XBaseFeret.equalize(fs.roI, su.byKey);
        fs.ro0 = XBaseFeret.equalize(su.ro, su.byKey);
        for (String p : fs.ro.ctrl) {
            List<String> ks = new ArrayList<>();
            keptU.getOrDefault(p, Map.of()).values().forEach(l -> l.forEach(im -> ks.add(im.key())));
            fs.ctrlKeysU.put(p, XBaseFeret.sorted(ks));
        }
        fs.pool = XBaseXsF.pool(fs.roI, su.byKey);
        fs.poolU = XBaseXsF.pool(fs.roU, su.byKey);
        return fs;
    }

    static List<XBaseXsF.Conf> confs(XBaseFeret.Roles r) {
        return List.of(new XBaseXsF.Conf("межс.", 0, r.own1, r.gal1, r.pro1), new XBaseXsF.Conf("внутри", 1, r.own1in, r.gal1in, r.pro1in),
                new XBaseXsF.Conf("G2", 3, r.own2, r.g2, r.pro2));
    }

    /** Пробы своих по U для постановки (люди и даты – по ролям на I). */
    static Map<String, List<String>> proU(XBaseFeret.Roles roU, String conf, List<String> own) {
        Map<String, List<String>> src = switch (conf) {
            case "межс." -> roU.pro1;
            case "внутри" -> roU.pro1in;
            default -> roU.pro2;
        };
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (String p : own) out.put(p, src.get(p));
        return out;
    }

    /** Таблица размеров и цены пересъёмки FERET; true – условие остановки сработало. */
    static boolean feretSizes(StringBuilder r, FeretSets fs) {
        XBaseA0.FeretSetup su = fs.su;
        r.append(String.format(Locale.ROOT, "%nСнимки fa/fb: %d; YuNet нашёл %d; из них МП (вырезка) не нашёл %d – отказ, в U не входят; U = %d; отобрано базой (|r| ≤ %.4f) "
                + "%d, МП (|yaw| ≤ %.4f°) %d; I = %d.%n", su.imgs.size(), fs.detected, fs.mpMissing, fs.u.size(), su.tpl.rLimit,
                fs.u.stream().filter(k -> !fs.rejB(k)).count(), fs.th.bigR(), fs.u.stream().filter(k -> !fs.rejM(k)).count(), fs.i.size()));
        r.append("\n=== Размеры: прежний набор (роли feret-xs2 / feret-subsp) против I; пробы (а) – U ===\n");
        r.append("постановка | своих прежде / I | снимков галереи прежде / I | проб своих прежде / I / U | контроль: людей, снимков прежде / I / U | сокращение I, %\n");
        boolean stop = false;
        List<XBaseXsF.Conf> c0 = confs(fs.ro0);
        List<XBaseXsF.Conf> c1 = confs(fs.ro);
        int ctrl0 = size(fs.ro0.ctrlKeys, fs.ro0.ctrl);
        int ctrlI = size(fs.ro.ctrlKeys, fs.ro.ctrl);
        int ctrlU = size(fs.ctrlKeysU, fs.ro.ctrl);
        for (int j = 0; j < c0.size(); j++) {
            XBaseXsF.Conf a = c0.get(j);
            XBaseXsF.Conf b = c1.get(j);
            int g0 = size(a.gal(), a.own());
            int g1 = size(b.gal(), b.own());
            int p0 = size(a.pro(), a.own());
            int p1 = size(b.pro(), b.own());
            int pu = size(proU(fs.roU, b.name(), b.own()), b.own());
            double cut = 100.0 * Math.max(1 - (double) g1 / g0, 1 - (double) p1 / p0);
            if (cut > 100 * STOP_SHARE) stop = true;
            r.append(String.format(Locale.ROOT, "%s | %d / %d | %d / %d | %d / %d / %d | %d, %d / %d / %d | %.1f%n", a.name(), a.own().size(), b.own().size(), g0, g1, p0, p1,
                    pu, fs.ro.ctrl.size(), ctrl0, ctrlI, ctrlU, cut));
        }
        r.append(String.format(Locale.ROOT, "Контрольные люди I и прежние совпадают: %s; люди пула прежде / I: %d / %d.%n", fs.ro.ctrl.equals(fs.ro0.ctrl) ? "да" : "нет",
                XBaseXsF.pool(su.ro, su.byKey).size(), fs.pool.size()));

        r.append("\n=== Цена пересъёмки: снимки U в ролях галереи и обучения, которые отверг бы отбор каждой схемы ===\n");
        r.append("набор | снимков U | отверг бы отбор базы | отверг бы отбор МП\n");
        for (XBaseXsF.Conf b : c1) {
            Set<String> galDates = new TreeSet<>();
            List<String> ks = new ArrayList<>();
            for (String p : b.own()) {
                for (String k : b.gal().get(p)) galDates.add(p + "\t" + su.byKey.get(k).date());
            }
            for (String k : fs.u) {
                Img im = su.byKey.get(k);
                if (galDates.contains(im.person() + "\t" + im.date())) ks.add(k);
            }
            r.append(String.format(Locale.ROOT, "%s: даты галереи своих | %d | %d | %d%n", b.name(), ks.size(), ks.stream().filter(fs::rejB).count(),
                    ks.stream().filter(fs::rejM).count()));
        }
        List<String> poolK = new ArrayList<>();
        fs.poolU.values().forEach(ds -> ds.forEach(poolK::addAll));
        r.append(String.format(Locale.ROOT, "пул FERET (обучение W) | %d | %d | %d%n", poolK.size(), poolK.stream().filter(fs::rejB).count(),
                poolK.stream().filter(fs::rejM).count()));
        return stop;
    }

    static XBaseEval.Gt withR0(XBaseEval.Gt t, double r0) {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.dir = t.dir;
        g.seed = t.seed;
        g.r0 = r0;
        g.b = t.b;
        g.threads = t.threads;
        g.rLimit = t.rLimit;
        g.params = t.params;
        return g;
    }

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-mp2: только равные галереи (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        FeretSets fs = feretSets(exportDir, selLines, settings);
        XBaseA0.FeretSetup su = fs.su;
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(out);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (8) MediaPipe, этап 2 – распознавание, схема «МП» против базы, FERET (XBaseEval feret-mp2): Color FERET — только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры «а» – derived/; МП – derived/").append(CROP_FILE).append(", кадры «МП» – derived/a_mp); когорта "
                + "FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        for (String h : fs.mpHead) r.append("Кэш МП: ").append(h.substring(1).trim()).append('\n');
        header(r, fs.th, tpl);
        boolean stop = feretSizes(r, fs);
        System.out.print(r);
        System.out.flush();
        if (stop) {
            r.append(String.format(Locale.ROOT, "%nОСТАНОВКА: I меньше прежнего набора больше чем на %.0f %% снимков галереи или проб своих – счёт не выполнялся "
                    + "(условие Жорика, №9).%n", 100 * STOP_SHARE));
            FarMethods.write(out.resolve("xbase_feret_mp2.txt"), r);
            System.out.print(r);
            return;
        }
        XBaseFeret.Roles ro = fs.ro;
        List<String> perm = new ArrayList<>(fs.pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        List<Map<String, List<List<String>>>> passPool = new ArrayList<>();
        for (int ps = 0; ps < 2; ps++) {
            Map<String, List<List<String>>> m = new TreeMap<>();
            for (String p : halves.get(1 - ps)) m.put(p, fs.pool.get(p));
            passPool.add(m);
        }
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        TreeSet<String> ps0 = new TreeSet<>(fs.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        List<XBaseXsF.Conf> confs = confs(ro);
        Set<String> used = new TreeSet<>();
        for (XBaseXsF.Conf c : confs) {
            for (String p : c.own()) {
                used.addAll(c.gal().get(p));
                used.addAll(proU(fs.roU, c.name(), c.own()).get(p));
            }
        }
        fs.ctrlKeysU.values().forEach(used::addAll);
        fs.pool.values().forEach(ds -> ds.forEach(used::addAll));
        boolean sameRoles = sameRoles(fs);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-mp2 %s; U %d; I %d; R_МП %s; r0_МП %s; seed %d; r0 %s; пул %d", dir,
                fs.u.size(), fs.i.size(), fs.th.bigR(), fs.th.r0(), tpl.seed, tpl.r0, fs.pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_mp2_eq"), fp);
        Map<String, String> xs2 = XBaseSubsp.blocks(out.resolve("xbase_feret_xs2_scores.tsv.gz"));
        Map<String, String> subsp = XBaseSubsp.blocks(out.resolve("xbase_feret_subsp_scores.tsv.gz"));
        StringBuilder check = new StringBuilder("\n=== Проверка базы: строки L1 и +mirror (k = 0) на прежних ролях – побайтно с файлами оценок feret-xs2 / feret-subsp ===\n");
        if (!sameRoles) check.append("Роли I отличаются от прежних – база (б) на I прежним файлам не равна; проверка – отдельным прогоном базы на прежних ролях.\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Acc> acc = new LinkedHashMap<>();
        double fitS = 0;
        for (int sc = 0; sc < 2; sc++) {
            boolean mpScheme = sc == 1;
            Map<String, Mat> fr = new LinkedHashMap<>();
            if (mpScheme) {
                fr.putAll(feretMpFrames(dir, used, su.byKey, su.det, fs.mp));
                fr.putAll(feiMpFrames(export, exportDir, cohortKeys, settings));
            } else {
                for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
                List<Sample> fei = new ArrayList<>();
                export.data().fei().values().forEach(fei::addAll);
                fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            }
            XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
            xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
            xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
            fr.values().forEach(Mat::release);
            Map<String, Double> rOf = mpScheme ? fs.yM : su.rOf;
            XBaseEval.Gt t = withR0(tpl, mpScheme ? fs.th.r0() : tpl.r0);
            XBaseEval.Gt baseI = XBaseFeret.gt(t, persons, ro.ctrl, ro.ctrlKeys, cohort, rOf, su.imageNo);
            XBaseEval.Gt baseU = XBaseFeret.gt(t, persons, ro.ctrl, fs.ctrlKeysU, cohort, rOf, su.imageNo);
            Predicate<String> rej = mpScheme ? fs::rejM : fs::rejB;
            for (XBaseXsF.Conf c : confs) {
                for (int f = 0; f < XBaseXsF.FEATS.length; f++) {
                    String rowLabel = XBaseXsF.label(f, XBaseXsF.Kind.XS);
                    for (int ps = 0; ps < 2; ps++) {
                        Set<String> mine = halves.get(ps);
                        List<String> own = c.own().stream().filter(mine::contains).toList();
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        Map<String, List<String>> pro = new LinkedHashMap<>();
                        for (String p : own) {
                            gal.put(p, c.gal().get(p));
                            pro.put(p, c.pro().get(p));
                        }
                        Map<String, List<String>> proUa = proU(fs.roU, c.name(), own);
                        List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), passPool.get(ps), Set.of(), Map.of()).extra();
                        List<List<String>> aug = new ArrayList<>(cohort);
                        aug.addAll(extra);
                        XBaseEval.Gt gI = XBaseFeret.withOwn(baseI, own);
                        gI.cohort = aug;
                        XBaseEval.Gt gU = XBaseFeret.withOwn(baseU, own);
                        gU.cohort = aug;
                        List<Probe> probesI = XBaseEval.probes(gI, pro);
                        List<Probe> probesU = XBaseEval.probes(gU, proUa);
                        List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                        String name = String.format(Locale.ROOT, "%d_%d_%d_%d", sc, c.s(), f, ps);
                        String ls = cp.read(name);
                        if (ls == null) {
                            long t1 = System.nanoTime();
                            ls = XBaseEval.fitLines(gU, xs[f], EnumSet.of(V.L1), gal, probesU, parts, c.s());
                            double s = (System.nanoTime() - t1) / 1e9;
                            fitS += s;
                            cp.save(name, ls);
                            System.out.printf(Locale.ROOT, "feret-mp2: %s, %s, %s, проход %d ready (%.0f s)%n", SCHEMES[sc], c.name(), rowLabel, ps + 1, s);
                        }
                        lines.append("# схема ").append(SCHEMES[sc]).append(", проход ").append(ps + 1).append(", конфигурация ").append(c.name()).append(", строка ")
                                .append(rowLabel).append('\n').append(ls);
                        if (!mpScheme && sameRoles) {
                            String old = c.name().equals("G2") ? subsp.get("G2, проекция " + rowLabel + ", проход " + (ps + 1) + ", k = 0")
                                    : xs2.get("проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + rowLabel);
                            boolean same = old != null && XBaseSubsp.l1Rows(old).equals(XBaseSubsp.l1Rows(only(ls, ids(probesI))));
                            check.append(String.format(Locale.ROOT, "%s %s, проход %d: %s (%d строк)%n", c.name(), rowLabel, ps + 1, same ? "совпало" : "НЕ СОВПАЛО",
                                    XBaseSubsp.l1Rows(only(ls, ids(probesI))).size()));
                            if (!same) throw new IllegalStateException("feret-mp2: база не совпала с прежним файлом оценок: " + c.name() + " " + rowLabel + " проход " + (ps + 1));
                        }
                        Set<String> rejected = new HashSet<>();
                        for (Probe q : probesU) if (rej.test(q.key())) rejected.add(q.role() + "\t" + q.person() + "\t" + q.image());
                        acc.computeIfAbsent(c.name() + " " + rowLabel + " " + SCHEMES[sc] + " (б)", x -> new Acc())
                                .add(XBaseEval.parse(gI, EnumSet.of(V.L1), gal, probesI, parts, ls));
                        acc.computeIfAbsent(c.name() + " " + rowLabel + " " + SCHEMES[sc] + " (а)", x -> new Acc())
                                .add(XBaseEval.parse(gU, EnumSet.of(V.L1), gal, probesU, parts, refuse(ls, rejected)));
                    }
                }
            }
        }
        if (!sameRoles) checkOldRoles(check, fs, export, exportDir, feiKeys, xs2, subsp, cohort, cohortKeys);
        r.append(check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        tablesOut(r, acc, persons.size(), tpl, List.of(List.of("межс.", "внутри"), List.of("G2")));
        XBaseOther.writeScores(out.resolve("xbase_feret_mp2_scores.tsv.gz"), "# FERET, (8) MediaPipe этап 2, XBaseMp2 feret-mp2, только локально; блоки — схема, "
                + "проход, конфигурация, строка; пробы U (отказ отбором не применён)\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("xbase_feret_mp2.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /** Роли на I совпадают с прежними для всех постановок, контроля и пула. */
    static boolean sameRoles(FeretSets fs) {
        List<XBaseXsF.Conf> a = confs(fs.ro0);
        List<XBaseXsF.Conf> b = confs(fs.ro);
        for (int j = 0; j < a.size(); j++) {
            if (!a.get(j).own().equals(b.get(j).own()) || !a.get(j).gal().equals(b.get(j).gal()) || !a.get(j).pro().equals(b.get(j).pro())) return false;
        }
        return fs.ro0.ctrl.equals(fs.ro.ctrl) && fs.ro0.ctrlKeys.equals(fs.ro.ctrlKeys) && XBaseXsF.pool(fs.su.ro, fs.su.byKey).equals(fs.pool);
    }

    /** Проверка базы на прежних ролях (если роли I отличаются): тот же счёт, что feret-xs2 / feret-subsp, только сверка. */
    static void checkOldRoles(StringBuilder check, FeretSets fs, FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys,
                              Map<String, String> xs2, Map<String, String> subsp, List<List<String>> cohort, Set<String> cohortKeys) throws IOException {
        XBaseA0.FeretSetup su = fs.su;
        XBaseEval.Gt tpl = su.tpl;
        XBaseFeret.Roles ro = fs.ro0;
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        Set<String> used = new TreeSet<>();
        for (XBaseXsF.Conf c : confs(ro)) for (String p : c.own()) {
            used.addAll(c.gal().get(p));
            used.addAll(c.pro().get(p));
        }
        ro.ctrlKeys.values().forEach(used::addAll);
        pool.values().forEach(ds -> ds.forEach(used::addAll));
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        for (XBaseXsF.Conf c : confs(ro)) {
            for (int f = 0; f < XBaseXsF.FEATS.length; f++) {
                String rowLabel = XBaseXsF.label(f, XBaseXsF.Kind.XS);
                for (int ps = 0; ps < 2; ps++) {
                    List<String> own = c.own().stream().filter(halves.get(ps)::contains).toList();
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Map<String, List<String>> pro = new LinkedHashMap<>();
                    for (String p : own) {
                        gal.put(p, c.gal().get(p));
                        pro.put(p, c.pro().get(p));
                    }
                    Map<String, List<List<String>>> pp = new TreeMap<>();
                    for (String p : halves.get(1 - ps)) pp.put(p, pool.get(p));
                    List<List<String>> aug = new ArrayList<>(cohort);
                    aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp, Set.of(), Map.of()).extra());
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                    g.cohort = aug;
                    String ls = XBaseEval.fitLines(g, xs[f], EnumSet.of(V.L1), gal, XBaseEval.probes(g, pro), XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s()),
                            c.s());
                    String old = c.name().equals("G2") ? subsp.get("G2, проекция " + rowLabel + ", проход " + (ps + 1) + ", k = 0")
                            : xs2.get("проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + rowLabel);
                    boolean same = old != null && XBaseSubsp.l1Rows(old).equals(XBaseSubsp.l1Rows(ls));
                    check.append(String.format(Locale.ROOT, "%s %s, проход %d (прежние роли): %s (%d строк)%n", c.name(), rowLabel, ps + 1, same ? "совпало" : "НЕ СОВПАЛО",
                            XBaseSubsp.l1Rows(ls).size()));
                    if (!same) throw new IllegalStateException("feret-mp2: база на прежних ролях не совпала: " + c.name() + " " + rowLabel + " проход " + (ps + 1));
                }
            }
        }
    }

    static void header(StringBuilder r, Thr th, XBaseEval.Gt tpl) {
        r.append(String.format(Locale.ROOT, "МП – на вырезке по рамке YuNet: квадрат со стороной 3 · max(w, h) рамки, центр – центр рамки, обрезка по краю снимка (одно "
                + "правило для всех баз).%n"));
        r.append(String.format(Locale.ROOT, "Пороги схемы «МП» (yaw МП, градусы; правило FeiDataset – элемент round(0,95·(n − 1))): R_МП = %.4f° – снимки протокола своей "
                + "базы с рамкой YuNet %d, МП нашёл лицо на %d (не найденные не входят); r0_МП = %.4f° – снимки №11–14 FEI %d, МП нашёл на %d. База: R = %.4f, "
                + "r0 = %s.%n", th.bigR(), th.ownN(), th.ownFound(), th.r0(), th.feiN(), th.feiFound(), tpl.rLimit, tpl.r0));
        r.append("Кадр «МП» 92×112: подобие (как кадр «а», шаблон ownTemplate) по радужкам 468/473 и углам рта 61/291 МП, нос – YuNet. Кандидаты: L1-xs-F, L1-tt-xs-F.\n"
                + "Роли, галереи, пул, обучение W и 5 моделей частей – на I (одни для обеих схем; галерея и обучение в ПО – под контролем качества с пересъёмкой);\n"
                + "(б) – пробы из I; (а) – пробы из U, отвергнутая отбором своей схемы проба: свой – отказ, чужой – не принят. Когорта FEI – фиксированная (те же\n"
                + "люди и снимки, кадры – по схеме). Разности «МП − база» – бутстреп по людям (одни веса для обеих схем).\n");
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
    }

    /** Таблицы: группы постановок; строки «постановка признак схема (а|б)», пары «МП − база». */
    static void tablesOut(StringBuilder r, Map<String, Acc> acc, int persons, XBaseEval.Gt tpl, List<List<String>> groups) {
        for (String tbl : new String[] {"(а)", "(б)"}) {
            for (List<String> grp : groups) {
                Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                for (Map.Entry<String, Acc> e : acc.entrySet()) {
                    String l = e.getKey();
                    if (!l.endsWith(tbl) || grp.stream().noneMatch(g -> l.startsWith(g + " "))) continue;
                    res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), e.getValue().cfgs, e.getValue().imps, persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    if (l.contains(" МП ")) pairs.add(new String[] {l, l.replace(" МП ", " база ")});
                }
                XBaseXsF.flush(r, res, pairs, persons, tpl, "Таблица " + tbl + ": " + String.join(", ", grp));
            }
        }
    }

    // ---------------------------------------------------------------- проверки до счёта

    /** Строка «вырезка − полный кадр» по набору ключей: медиана, 95 %, наибольшее (точки – доли межглазья YuNet, yaw – градусы). */
    static String cropLine(String name, List<String> keys, Map<String, Mp> full, Map<String, Mp> crop, Map<String, double[]> det, double[] eyesMed) {
        List<double[]> d = new ArrayList<>();
        for (String k : keys) {
            Mp a = full.get(k);
            Mp b = crop.get(k);
            double[] row = det.get(k);
            if (a == null || b == null || row == null || !a.found() || !b.found()) continue;
            double iod = Math.hypot(row[6] - row[4], row[7] - row[5]);
            double eyes = (XBaseMpPts.dist(a.p().get("iris_re"), b.p().get("iris_re")) + XBaseMpPts.dist(a.p().get("iris_le"), b.p().get("iris_le"))) / 2 / iod;
            double mouth = XBaseMpPts.dist(a.mid("mouth_r", "mouth_l"), b.mid("mouth_r", "mouth_l")) / iod;
            d.add(new double[] {eyes, mouth, Math.abs(b.yaw() - a.yaw())});
        }
        StringBuilder s = new StringBuilder(name).append(" | ").append(d.size());
        for (int j = 0; j < 3; j++) {
            int jj = j;
            double[] v = FeiBootstrap.sorted(d.stream().mapToDouble(x -> x[jj]).toArray());
            s.append(String.format(Locale.ROOT, j < 2 ? " | %.4f / %.4f / %.4f" : " | %.2f / %.2f / %.2f", FeiBootstrap.quantile(v, 0.5), FeiBootstrap.quantile(v, 0.95),
                    v[v.length - 1]));
            if (j == 0 && eyesMed != null) eyesMed[0] = FeiBootstrap.quantile(v, 0.5);
        }
        return s.append('\n').toString();
    }

    /**
     * Режим mp-crop-check: расхождение точек МП «вырезка − полный кадр» (этап 1) – FERET (fa/fb и все позы) и GT; на fa/fb FERET
     * медиана по глазам больше 0,01 межглазья – пометка «сказать до счёта». Отчёты: FERET – только reports/eq/xbase_feret_mp_crop.txt,
     * GT – reports/faces/xbase/xbase_gt_mp_crop.txt.
     */
    static void cropCheck(Path exportDir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        FeretSets fs = feretSets(exportDir, selLines, settings);
        Path dir = fs.su.dir;
        Map<String, Mp> full = XBaseMpPts.read(dir.resolve(FeretData.DERIVED).resolve(XBaseMpPts.FILE), new ArrayList<>());
        Map<String, double[]> all = new HashMap<>();
        for (String l : Files.readAllLines(dir.resolve(FeretData.DERIVED).resolve(XBaseMpPts.BOXES), StandardCharsets.UTF_8)) {
            if (l.startsWith("#")) continue;
            all.put(l.substring(0, l.indexOf('\t')), null);
        }
        Map<String, double[]> det = new HashMap<>(fs.su.det);
        Map<String, double[]> allPose = XBaseMpPts.detectAllPose(XBaseMpPts.allFiles(dir), fs.su.det, dir, Paths.get(settings.loadFacesModelsDir()));
        det.putAll(allPose);
        String head = "набор | снимков (МП нашёл и там, и там) | глаза (среднее по двум радужкам): медиана / 95 % / наибольшее, доли межглазья YuNet | рот (середина "
                + "углов) | |yaw|, градусы\n";
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, (8) MediaPipe, этап 2: точки МП на вырезке по рамке YuNet (×3) против полного кадра (этап 1), FERET (XBaseEval "
                + "mp-crop-check): только локально\nКод: коммит " + FarMethods.commit() + "\n\n=== Вырезка − полный кадр ===\n" + head);
        double[] med = new double[1];
        r.append(cropLine("fa, fb", fs.su.imgs.stream().map(Img::key).toList(), full, fs.mp, det, med));
        r.append(cropLine("все позы", new ArrayList<>(allPose.keySet()), full, fs.mp, det, null));
        r.append(String.format(Locale.ROOT, "Условие Жорика (fa/fb, глаза, медиана > 0,01 межглазья – сказать до счёта): медиана %.4f – %s.%n", med[0],
                med[0] > 0.01 ? "ПРЕВЫШЕНО" : "не превышено"));
        FarMethods.write(dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve("xbase_feret_mp_crop.txt"), r);
        System.out.print(r);
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, Mp> gFull = XBaseMpPts.read(gtDir.getParent().resolve(XBaseMpPts.FILE), new ArrayList<>());
        Map<String, Mp> gCrop = XBaseMpPts.read(gtDir.getParent().resolve(CROP_FILE), new ArrayList<>());
        Map<String, double[]> gDet = XBaseMpPts.exportRows(exportDir, "gt");
        StringBuilder g = new StringBuilder("PLAN.md п. 5.5, (8) MediaPipe, этап 2: точки МП на вырезке по рамке YuNet (×3) против полного кадра (этап 1), GT (XBaseEval "
                + "mp-crop-check)\nКод: коммит " + FarMethods.commit() + "\n\n=== Вырезка − полный кадр ===\n" + head);
        g.append(cropLine("GT, все", new ArrayList<>(gDet.keySet()), gFull, gCrop, gDet, null));
        FarMethods.write(out.resolve("xbase_gt_mp_crop.txt"), g);
        System.out.print(g);
    }

    /**
     * Режим mp2-sheet: лист кадров «а» | «МП» (по парам) – 12 снимков FERET разных поз (только локально, reports/eq/mp2_frames.png)
     * и 8 GT (reports/faces/xbase/mp2_gt_frames.png). Кадры МП FERET – кэш derived/a_mp.
     */
    static void sheet(FarExport.Loaded export, Path exportDir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        FeretSets fs = feretSets(exportDir, selLines, settings);
        List<String> keys = new ArrayList<>(fs.i);
        java.util.Collections.shuffle(keys, new java.util.Random(fs.su.tpl.seed + 9995));
        keys = new ArrayList<>(keys.subList(0, 12));
        Map<String, Mat> mpF = feretMpFrames(fs.su.dir, new TreeSet<>(keys), fs.su.byKey, fs.su.det, fs.mp);
        List<Mat> tiles = new ArrayList<>();
        for (String k : new TreeSet<>(keys)) tiles.add(pair(FeretData.frame(fs.su.dir, "a", k), mpF.get(k)));
        writeSheet(fs.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve("mp2_frames.png"), tiles);
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, Mp> mp = XBaseMpPts.read(gtDir.getParent().resolve(CROP_FILE), new ArrayList<>());
        Map<String, double[]> det = XBaseMpPts.exportRows(exportDir, "gt");
        List<Sample> gs = new ArrayList<>();
        export.data().gt().values().forEach(gs::addAll);
        gs.sort((a, b) -> a.file().toString().compareTo(b.file().toString()));
        java.util.Collections.shuffle(gs, new java.util.Random(fs.su.tpl.seed + 9996));
        tiles = new ArrayList<>();
        for (Sample s : gs.subList(0, 8)) {
            String pk = s.person() + "/" + s.file().getFileName().toString();
            Mat color = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            tiles.add(pair(export.frame(s, GalleryEvaluation.Variant.A), frameMp(color, det.get(pk), mp.get(pk))));
            color.release();
        }
        writeSheet(out.resolve("mp2_gt_frames.png"), tiles);
        System.out.println("mp2-sheet: листы записаны");
    }

    static Mat pair(Mat a, Mat b) {
        Mat m = new Mat();
        org.opencv.core.Core.hconcat(List.of(a, b), m);
        Mat big = new Mat();
        org.opencv.imgproc.Imgproc.resize(m, big, new org.opencv.core.Size(m.cols() * 3, m.rows() * 3), 0, 0, org.opencv.imgproc.Imgproc.INTER_NEAREST);
        return big;
    }

    static void writeSheet(Path f, List<Mat> tiles) throws IOException {
        List<Mat> rows = new ArrayList<>();
        for (int i = 0; i < tiles.size(); i += 4) {
            Mat row = new Mat();
            org.opencv.core.Core.hconcat(tiles.subList(i, Math.min(tiles.size(), i + 4)), row);
            rows.add(row);
        }
        Mat all = new Mat();
        org.opencv.core.Core.vconcat(rows, all);
        Files.createDirectories(f.getParent());
        FeretData.writePng(f, all);
    }

    // ---------------------------------------------------------------- GT

    /** Роли GT по образцу XBaseEval.prepare (без векторов): снимки после keep («человек/снимок»). */
    static XBaseEval.Gt gtRoles(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, SettingsStore settings, Predicate<String> keep,
                                Map<String, Double> rOfPk, double r0) throws IOException {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.dir = dir;
        g.seed = settings.loadFacesSeed();
        g.r0 = r0;
        g.b = settings.loadFacesFeiBootstrap();
        g.threads = settings.loadFacesFeiCvThreads();
        g.params = IlluminationNorm.NormParams.of(settings);
        Map<String, Integer> sess = XBaseEval.sessions(dir.resolve(XBaseEval.SESSIONS));
        for (Map.Entry<String, List<Sample>> e : new TreeMap<>(export.data().gt()).entrySet()) {
            List<Sample> list = new ArrayList<>(e.getValue());
            list.sort((x, y) -> x.file().getFileName().toString().compareTo(y.file().getFileName().toString()));
            for (Sample s : list) {
                g.total++;
                String name = s.file().getFileName().toString();
                String pk = s.person() + "/" + name;
                if (!keep.test(pk)) {
                    g.dropped.add(pk);
                    continue;
                }
                String k = s.file().toString();
                Integer d = sess.get(pk);
                if (d == null) throw new IllegalStateException("Нет сессии снимка GT " + s.file());
                g.byPerson.computeIfAbsent(s.person(), x -> new TreeMap<>()).computeIfAbsent(d, x -> new ArrayList<>()).add(k);
                g.samples.put(k, s);
                g.rOf.put(k, rOfPk.get(pk));
            }
        }
        Map<String, Set<Integer>> datesAll = new TreeMap<>();
        sess.forEach((pk, d) -> {
            if (d > 0) datesAll.computeIfAbsent(pk.substring(0, pk.indexOf('/')), x -> new HashSet<>()).add(d);
        });
        for (String p : g.byPerson.keySet()) (datesAll.getOrDefault(p, Set.of()).size() >= 2 ? g.own : g.ctrl).add(p);
        g.persons = new ArrayList<>(g.byPerson.keySet());
        for (int i = 0; i < g.persons.size(); i++) g.pIdx.put(g.persons.get(i), i);
        for (String p : g.own) {
            Map<Integer, List<String>> bd = g.byPerson.get(p);
            int gd = XBaseEval.galleryDate(bd);
            g.galleryDateOf.put(p, gd);
            g.galleryKeys.put(p, bd.get(gd));
            List<String> pr = new ArrayList<>();
            bd.forEach((d, ks) -> {
                if (d != gd) pr.addAll(ks);
            });
            pr.sort(null);
            if (pr.isEmpty()) throw new IllegalStateException("У своего " + p + " после отбора нет проб");
            g.probeKeys.put(p, pr);
        }
        g.galleryAll = new LinkedHashMap<>(g.galleryKeys);
        g.cohortP = XBaseEval.cohortPersons(feiKeys, g.seed);
        for (String p : g.cohortP) g.cohort.add(new ArrayList<>(feiKeys.get(p).values()));
        return g;
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-mp2: только равные галереи FERET (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        FeretSets fs = feretSets(dir, selLines, settings);
        Thr th = fs.th;
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("gt-mp2: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        List<String> mpHead = new ArrayList<>();
        Map<String, Mp> mp = XBaseMpPts.read(gtDir.getParent().resolve(CROP_FILE), mpHead);
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rB = new HashMap<>();
        Map<String, Double> yM = new HashMap<>();
        Set<String> u = new TreeSet<>();
        int mpMissing = 0;
        for (Map.Entry<String, double[]> e : det.entrySet()) {
            if (e.getValue() == null) continue;
            Mp m = mp.get(e.getKey());
            if (m == null) throw new IllegalStateException("gt-mp2: нет строки МП для " + e.getKey());
            if (!m.found()) {
                mpMissing++;
                continue;
            }
            rB.put(e.getKey(), Math.abs(FeiDataset.pose(e.getValue())));
            yM.put(e.getKey(), Math.abs(m.yaw()));
            u.add(e.getKey());
        }
        Predicate<String> rejB = pk -> rB.get(pk) > rLimit;
        Predicate<String> rejM = pk -> yM.get(pk) > th.bigR();
        Predicate<String> inI = pk -> u.contains(pk) && !rejB.test(pk) && !rejM.test(pk);
        Map<String, Double> rOld = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        XBaseEval.Gt[] gI = {gtRoles(export, dir, feiKeys, settings, inI, rB, r0), gtRoles(export, dir, feiKeys, settings, inI, yM, th.r0())};
        long eqSeed = gI[0].seed + XBaseEval.EQ_SEED_OFFSET;

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (8) MediaPipe, этап 2 – распознавание, схема «МП» против базы, GT (XBaseEval gt-mp2)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, YuNet – detections.tsv; когорта – a/fei); "
                + "снимки GT – ").append(gtDir).append("; МП – ").append(gtDir.getParent().resolve(CROP_FILE)).append("; пул L1-xs-F – Color FERET (только обучение, I FERET; "
                + "имён снимков FERET нет)\n");
        for (String h : mpHead) r.append("Кэш МП: ").append(h.substring(1).trim()).append('\n');
        XBaseEval.Gt tplH = withR0(gI[0], r0);
        tplH.rLimit = rLimit;
        header(r, th, tplH);
        r.append(String.format(Locale.ROOT, "%nСнимки GT: %d; YuNet нашёл %d; МП (вырезка) не нашёл %d – отказ, в U не входят; U = %d; отобрано базой %d, МП %d; I = %d.%n",
                det.size(), rOld.size(), mpMissing, u.size(), u.stream().filter(rejB.negate()).count(), u.stream().filter(rejM.negate()).count(),
                u.stream().filter(inI).count()));

        // Галереи по правилу №8 (как gt-subsp): межс. N = 6, внутри N = 3; роли – на I (одни для обеих схем).
        XBaseEval.Gt b = gI[0];
        Map<String, List<String>> galX = XBaseEval.equalize(b.own, b.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(b.own, b.galleryAll, b.seed);
        Map<String, List<String>> galIn = XBaseEval.equalize(b.own, split.get(0), eqSeed);
        Map<String, List<String>> galX0 = XBaseEval.equalize(g0.own, g0.galleryAll, eqSeed);
        List<Map<String, List<String>>> split0 = XBaseEval.withinSplit(g0.own, g0.galleryAll, g0.seed);
        Map<String, List<String>> galIn0 = XBaseEval.equalize(g0.own, split0.get(0), eqSeed);

        // Пробы (а): U-снимки своих (межс. – вне даты галереи, внутри – на дате галереи вне галереи) и все U-снимки контрольных.
        Map<String, String> keyOf = new HashMap<>();
        Map<String, Sample> sampleOf = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) {
            for (Sample s : l) {
                keyOf.put(s.person() + "/" + s.file().getFileName().toString(), s.file().toString());
                sampleOf.put(s.file().toString(), s);
            }
        }
        Map<String, Integer> sess = XBaseEval.sessions(dir.resolve(XBaseEval.SESSIONS));
        Map<String, List<String>> proXU = new LinkedHashMap<>();
        Map<String, List<String>> proInU = new LinkedHashMap<>();
        for (String p : b.own) {
            List<String> x = new ArrayList<>(b.probeKeys.get(p));
            List<String> in = new ArrayList<>(split.get(1).get(p));
            for (String pk : u) {
                if (!pk.startsWith(p + "/") || inI.test(pk)) continue;
                (sess.get(pk) == b.galleryDateOf.get(p) ? in : x).add(keyOf.get(pk));
            }
            x.sort(null);
            in.sort(null);
            proXU.put(p, x);
            proInU.put(p, in);
        }
        Object[][] modes = {{"межс.", galX, b.probeKeys, proXU, galX0, g0.probeKeys, "xbase_gt_subsp_scores.tsv.gz"},
                {"внутри", galIn, split.get(1), proInU, galIn0, split0.get(1), "xbase_gt_subsp_scores.tsv.gz"}};

        // Размеры и остановка.
        r.append("\n=== Размеры: прежний набор (gt-subsp) против I; пробы (а) – U ===\n");
        r.append("постановка | своих прежде / I | снимков галереи прежде / I | проб своих прежде / I / U | контроль: людей прежде / I, снимков прежде / I / U | сокращение I, %\n");
        int ctrl0 = g0.ctrl.stream().mapToInt(p -> g0.byPerson.get(p).values().stream().mapToInt(List::size).sum()).sum();
        int ctrlI = b.ctrl.stream().mapToInt(p -> b.byPerson.get(p).values().stream().mapToInt(List::size).sum()).sum();
        int ctrlU = (int) u.stream().filter(pk -> b.ctrl.contains(pk.substring(0, pk.indexOf('/')))).count();
        boolean stop = false;
        for (Object[] md : modes) {
            @SuppressWarnings("unchecked")
            Map<String, List<String>>[] m = new Map[] {(Map<String, List<String>>) md[1], (Map<String, List<String>>) md[2], (Map<String, List<String>>) md[3],
                    (Map<String, List<String>>) md[4], (Map<String, List<String>>) md[5]};
            int g0n = size(m[3], g0.own);
            int g1n = size(m[0], b.own);
            int p0n = size(m[4], g0.own);
            int p1n = size(m[1], b.own);
            double cut = 100.0 * Math.max(1 - (double) g1n / g0n, 1 - (double) p1n / p0n);
            if (cut > 100 * STOP_SHARE) stop = true;
            r.append(String.format(Locale.ROOT, "%s | %d / %d | %d / %d | %d / %d / %d | %d / %d, %d / %d / %d | %.1f%n", md[0], g0.own.size(), b.own.size(), g0n, g1n, p0n,
                    p1n, size(m[2], b.own), g0.ctrl.size(), b.ctrl.size(), ctrl0, ctrlI, ctrlU, cut));
        }
        r.append("\n=== Цена пересъёмки: снимки U в ролях галереи и обучения, которые отверг бы отбор каждой схемы ===\nнабор | снимков U | отверг бы отбор базы | "
                + "отверг бы отбор МП\n");
        List<String> galDateU = u.stream().filter(pk -> {
            String p = pk.substring(0, pk.indexOf('/'));
            return b.own.contains(p) && sess.get(pk).equals(b.galleryDateOf.get(p));
        }).toList();
        r.append(String.format(Locale.ROOT, "GT: дата галереи своих | %d | %d | %d%n", galDateU.size(), galDateU.stream().filter(rejB).count(),
                galDateU.stream().filter(rejM).count()));
        List<String> poolK = new ArrayList<>();
        fs.poolU.values().forEach(ds -> ds.forEach(poolK::addAll));
        r.append(String.format(Locale.ROOT, "пул FERET (обучение W) | %d | %d | %d%n", poolK.size(), poolK.stream().filter(fs::rejB).count(),
                poolK.stream().filter(fs::rejM).count()));
        Path outFile = out.resolve("xbase_gt_mp2.txt");
        if (stop) {
            r.append(String.format(Locale.ROOT, "%nОСТАНОВКА: I меньше прежнего набора больше чем на %.0f %% снимков галереи или проб своих – счёт не выполнялся.%n",
                    100 * STOP_SHARE));
            FarMethods.write(outFile, r);
            System.out.print(r);
            return;
        }
        boolean samePool = XBaseXsF.pool(fs.su.ro, fs.su.byKey).equals(fs.pool);
        boolean sameRoles = g0.own.equals(b.own) && g0.ctrl.equals(b.ctrl) && galX0.equals(galX) && galIn0.equals(galIn) && g0.probeKeys.equals(b.probeKeys)
                && split0.get(1).equals(split.get(1)) && g0.byPerson.equals(b.byPerson) && samePool;

        // Векторы схем: GT, когорта FEI, пул FERET.
        Set<String> cohortKeys = new HashSet<>();
        b.cohort.forEach(cohortKeys::addAll);
        Map<String, List<List<String>>> pool = fs.pool;
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-mp2; U %d; R_МП %s; r0_МП %s; seed %d; r0 %s; пул FERET %d", u.size(), th.bigR(), th.r0(),
                b.seed, r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_mp2"), fp);
        Map<String, String> old = XBaseSubsp.blocks(out.resolve("xbase_gt_subsp_scores.tsv.gz"));
        StringBuilder check = new StringBuilder("\n=== Проверка базы: строки L1 и +mirror (k = 0) – побайтно с файлом оценок gt-subsp ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Acc> acc = new LinkedHashMap<>();
        List<List<String>> parts = FeiMethods.cvParts(b.own, b.seed, XBaseEval.CV_S);
        double fitS = 0;
        String[][] projs = {{"L1-xs-F", "0"}, {"L1-tt-xs-F", "1"}};
        for (int sc = 0; sc < 2; sc++) {
            boolean mpScheme = sc == 1;
            Map<String, Mat> fr = new LinkedHashMap<>();
            Set<String> gtKeys = new TreeSet<>();
            for (String pk : u) gtKeys.add(keyOf.get(pk));
            if (mpScheme) {
                for (String k : gtKeys) {
                    Sample s = sampleOf.get(k);
                    String pk = s.person() + "/" + s.file().getFileName().toString();
                    Mat color = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                    fr.put(k, frameMp(color, det.get(pk), mp.get(pk)));
                    color.release();
                }
                fr.putAll(feiMpFrames(export, dir, cohortKeys, settings));
                fr.putAll(feretMpFrames(fs.su.dir, poolKeys, fs.su.byKey, fs.su.det, fs.mp));
            } else {
                fr.putAll(XBaseEval.frames(export, gtKeys.stream().map(sampleOf::get).toList(), null));
                List<Sample> fei = new ArrayList<>();
                export.data().fei().values().forEach(fei::addAll);
                fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
                for (String k : poolKeys) fr.put(k, FeretData.frame(fs.su.dir, "a", k));
            }
            XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
            xs[0].vectors(fr, EnumSet.of(V.L1), b.params);
            xs[1].vectors(fr, EnumSet.of(V.L1), b.params, IlluminationNorm.TAN_TRIGGS);
            fr.values().forEach(Mat::release);
            XBaseEval.Gt g = gI[sc];
            // gU: g с U-снимками контрольных и своих (для проб (а)); rOf – по схеме.
            XBaseEval.Gt gUx = gtRoles(export, dir, feiKeys, settings, pk -> u.contains(pk), mpScheme ? yM : rB, mpScheme ? th.r0() : r0);
            XBaseEval.Gt gU = withR0(g, g.r0);
            gU.own = g.own;
            gU.ctrl = g.ctrl;
            gU.persons = g.persons;
            gU.pIdx = g.pIdx;
            gU.cohort = g.cohort;
            gU.samples = gUx.samples;
            gU.rOf = gUx.rOf;
            for (String p : g.ctrl) gU.byPerson.put(p, gUx.byPerson.get(p));
            for (String p : g.own) gU.byPerson.put(p, gUx.byPerson.get(p));
            Predicate<String> rej = mpScheme ? rejM : rejB;
            for (Object[] md : modes) {
                String mode = (String) md[0];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> proUa = (Map<String, List<String>>) md[3];
                List<Probe> probesI = XBaseEval.probes(g, pro);
                List<Probe> probesU = XBaseEval.probes(gU, proUa);
                for (String[] pj : projs) {
                    int f = Integer.parseInt(pj[1]);
                    List<List<String>> aug = new ArrayList<>(b.cohort);
                    aug.addAll(poolCls);
                    g.cohort = aug;
                    gU.cohort = aug;
                    String name = String.format(Locale.ROOT, "%d_%s_%d", sc, mode.equals("межс.") ? "x" : "i", f);
                    String ls = cp.read(name);
                    if (ls == null) {
                        long t1 = System.nanoTime();
                        ls = XBaseEval.fitLines(gU, xs[f], EnumSet.of(V.L1), gal, probesU, parts, 0);
                        double s = (System.nanoTime() - t1) / 1e9;
                        fitS += s;
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "gt-mp2: %s, %s, %s ready (%.0f s)%n", SCHEMES[sc], mode, pj[0], s);
                    }
                    lines.append("# схема ").append(SCHEMES[sc]).append(", ").append(mode).append(", проекция ").append(pj[0]).append('\n').append(ls);
                    if (!mpScheme) {
                        if (sameRoles) {
                            String ob = old.get(mode + ", проекция " + pj[0] + ", k = 0");
                            boolean same = ob != null && XBaseSubsp.l1Rows(ob).equals(XBaseSubsp.l1Rows(only(ls, ids(probesI))));
                            check.append(String.format(Locale.ROOT, "%s %s: %s (%d строк)%n", mode, pj[0], same ? "совпало" : "НЕ СОВПАЛО",
                                    XBaseSubsp.l1Rows(only(ls, ids(probesI))).size()));
                            if (!same) throw new IllegalStateException("gt-mp2: база не совпала с gt-subsp: " + mode + " " + pj[0]);
                        } else {
                            check.append(mode).append(' ').append(pj[0]).append(": роли I отличаются от прежних – сверка с gt-subsp не делалась (НЕ ПРОВЕРЕНО)\n");
                        }
                    }
                    Set<String> rejected = new HashSet<>();
                    for (Probe q : probesU) {
                        Sample s = sampleOf.get(q.key());
                        if (rej.test(s.person() + "/" + s.file().getFileName().toString())) rejected.add(q.role() + "\t" + q.person() + "\t" + q.image());
                    }
                    acc.computeIfAbsent(mode + " " + pj[0] + " " + SCHEMES[sc] + " (б)", x -> new Acc()).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probesI, parts, ls));
                    acc.computeIfAbsent(mode + " " + pj[0] + " " + SCHEMES[sc] + " (а)", x -> new Acc())
                            .add(XBaseEval.parse(gU, EnumSet.of(V.L1), gal, probesU, parts, refuse(ls, rejected)));
                }
            }
        }
        r.append(check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, b.threads));
        tablesOut(r, acc, b.persons.size(), tplH, List.of(List.of("межс."), List.of("внутри")));
        XBaseOther.writeScores(out.resolve("xbase_gt_mp2_scores.tsv.gz"), "# GT, (8) MediaPipe этап 2 (XBaseMp2 gt-mp2); блоки — схема, постановка, проекция; пробы U\n"
                + "# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(outFile, r);
        System.out.print(r);
        cp.delete();
    }
}
