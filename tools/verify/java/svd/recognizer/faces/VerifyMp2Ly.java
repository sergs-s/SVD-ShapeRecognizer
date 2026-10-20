package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseMp2.Acc;
import svd.recognizer.faces.XBaseMp2.FeretSets;
import svd.recognizer.faces.XBaseMp2.Thr;
import svd.recognizer.storage.SettingsStore;

/**
 * Проверка вида ЛЮ «рамка YuNet + точки LBF» на распознавании (tools/verify, ветка cloud-verify; Жорик №15, №16 от 20.10.2026):
 * протокол feret-mp2 (XBaseMp2.feret) – тот же цикл, те же методы проекта; код проекта не меняется.
 * <p>
 * repro – воспроизведение схемы «база» feret-mp2: наборы U и I как в XBaseMp2.feretSets, пороги МП – из шапки отчёта
 * xbase_feret_mp2.txt (R_МП, r0_МП; в облаке нет своей базы и МП FEI для пересчёта); строки оценок сверяются с блоками
 * xbase_feret_mp2_scores.tsv.gz побайтно, таблицы печатаются в формате XBaseMp2.tablesOut.
 * <p>
 * ly – вариант (в) (решение Хозяина 20.10, №16): кадр ЛЮ – кадр «а» (FeretData.frameA) по строке YuNet, в которой глаза
 * (поля 4–7) заменены центрами глаз LBF по рамке YuNet (LbfFrames.Lm.five()); отбор и выбор модели – r YuNet (как в базе);
 * когорта FEI – кадры «а» экспорта в обоих видах; U – YuNet нашёл, I – отбор базы. Пары «ЛЮ − база»; отдельно – пробы своих
 * в очках и без очков (ground_truths, поле glasses), контроль и «чужие» П2/П3 – все.
 * <p>
 * Аргументы: repro | ly; каталог с прежними отчётами и файлами оценок FERET (xbase_feret_mp2.txt, *_scores.tsv.gz);
 * кэш точек LBF по рамке YuNet (ly; иначе «-»); выходной каталог (отчёт, строки оценок, контрольная точка блоков).
 * Остальное – как XBaseEval: FACES_EXPORT_DIR, FACES_FERET_DIR, FACES_MODELS_DIR, FACES_FEI_POSE_R0, FACES_FEI_BOOTSTRAP.
 */
public final class VerifyMp2Ly {

    static final String[] SCHEMES = {"база", "ЛЮ"};

    private VerifyMp2Ly() {
    }

    /** R_МП и r0_МП из шапки отчёта feret-mp2 (4 знака после запятой – как напечатано). */
    static Thr thrFromReport(Path report) throws IOException {
        String s = Files.readString(report, StandardCharsets.UTF_8);
        Matcher m = Pattern.compile("R_МП = ([0-9.]+)° – снимки протокола своей базы с рамкой YuNet (\\d+), МП нашёл лицо на (\\d+).*?r0_МП = ([0-9.]+)° – "
                + "снимки №11–14 FEI (\\d+), МП нашёл на (\\d+)").matcher(s);
        if (!m.find()) throw new IllegalStateException("нет порогов МП в " + report);
        return new Thr(Double.parseDouble(m.group(1)), Integer.parseInt(m.group(2)), Integer.parseInt(m.group(3)), Double.parseDouble(m.group(4)),
                Integer.parseInt(m.group(5)), Integer.parseInt(m.group(6)));
    }

    /** Наборы: repro – как XBaseMp2.feretSets (пороги МП заданы); ly – U – YuNet нашёл, I – отбор базы, МП не участвует. */
    static FeretSets sets(XBaseA0.FeretSetup su, Thr th, boolean ly) throws IOException {
        FeretSets fs = new FeretSets();
        fs.su = su;
        fs.th = th;
        fs.mp = ly ? new HashMap<>() : XBaseMpPts.read(su.dir.resolve(FeretData.DERIVED).resolve(XBaseMp2.CROP_FILE), fs.mpHead);
        Map<String, Map<String, List<Img>>> keptI = new TreeMap<>();
        Map<String, Map<String, List<Img>>> keptU = new TreeMap<>();
        for (Img im : su.imgs) {
            double[] row = su.det.get(im.key());
            if (row == null) continue;
            fs.detected++;
            double yaw = 0;
            if (!ly) {
                XBaseMpPts.Mp m = fs.mp.get(im.key());
                if (m == null) throw new IllegalStateException("нет строки МП (вырезка) для " + im.key());
                if (!m.found()) {
                    fs.mpMissing++;
                    continue;
                }
                yaw = Math.abs(m.yaw());
            }
            fs.rB.put(im.key(), Math.abs(FeiDataset.pose(row)));
            fs.yM.put(im.key(), yaw);
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

    /** Очки по ground_truths (поле glasses=Yes); снимки без поля – в очки не входят, их число печатается. */
    static Map<String, Boolean> glasses(Path feret, List<Img> imgs) throws IOException {
        Map<String, Boolean> out = new HashMap<>();
        for (Img im : imgs) {
            Path gt = im.file().getParent().getParent().getParent().resolve("ground_truths").resolve("name_value").resolve(im.person()).resolve(im.key() + ".txt");
            Boolean g = null;
            if (Files.exists(gt)) for (String l : Files.readAllLines(gt, StandardCharsets.UTF_8)) if (l.startsWith("glasses=")) g = l.substring(8).trim().equals("Yes");
            out.put(im.key(), g);
        }
        return out;
    }

    /** Кадры ЛЮ: кадр «а» по строке YuNet с глазами LBF (центры – Lm.five(), поля 4–7). */
    static Map<String, Mat> lyFrames(Set<String> keys, XBaseA0.FeretSetup su, Map<String, LbfFrames.Lm> lbf) throws IOException {
        Map<String, Mat> out = new LinkedHashMap<>();
        int n = 0;
        for (String k : keys) {
            LbfFrames.Lm lm = lbf.get(k);
            if (lm == null) throw new IllegalStateException("нет точек LBF по рамке YuNet: " + k);
            double[] row = su.det.get(k).clone();
            double[] f = lm.five();
            for (int i = 4; i < 8; i++) row[i] = f[i];
            Mat color = FeretData.decode(su.byKey.get(k).file());
            out.put(k, FeretData.frameA(color, row));
            color.release();
            if (++n % 500 == 0) System.out.printf(Locale.ROOT, "ly: кадры ЛЮ – %d / %d%n", n, keys.size());
        }
        return out;
    }

    /** Сверка: кадр «а» из derived и FeretData.frameA по исходной строке YuNet – побайтно (первые n ключей). */
    static String frameCheck(Set<String> keys, XBaseA0.FeretSetup su, int n) throws IOException {
        int same = 0;
        int all = 0;
        for (String k : keys) {
            if (all == n) break;
            Mat color = FeretData.decode(su.byKey.get(k).file());
            Mat a = FeretData.frameA(color, su.det.get(k));
            Mat b = FeretData.frame(su.dir, "a", k);
            Mat d = new Mat();
            Core.absdiff(a, b, d);
            if (Core.countNonZero(d) == 0) same++;
            all++;
            color.release();
            a.release();
            b.release();
            d.release();
        }
        return String.format(Locale.ROOT, "Кадр «а» derived/a против FeretData.frameA по строке YuNet: побайтно %d из %d (первые ключи).%n", same, all);
    }

    static String sortedRows(String lines) {
        List<String> l = new ArrayList<>(Arrays.asList(lines.split("\n")));
        l.removeIf(String::isEmpty);
        java.util.Collections.sort(l);
        return String.join("\n", l);
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        String mode = a[0];
        boolean ly = mode.equals("ly");
        Path old = Paths.get(a[1]);
        Path out = Paths.get(a[3]);
        Files.createDirectories(out.resolve("blocks"));
        SettingsStore settings = new SettingsStore();
        Path exportDir = Paths.get(settings.loadFacesExportDir());
        FarExport.Loaded export = FarExport.read(exportDir);
        List<String> selLines = Files.readAllLines(exportDir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Thr th = ly ? new Thr(Double.POSITIVE_INFINITY, 0, 0, Double.NaN, 0, 0) : thrFromReport(old.resolve("xbase_feret_mp2.txt"));
        FeretSets fs = sets(su, th, ly);
        Map<String, LbfFrames.Lm> lbf = ly ? VerifyLbfYunet.lbf(Paths.get(a[2])) : Map.of();
        Map<String, Boolean> gl = glasses(su.dir, su.imgs);

        StringBuilder r = new StringBuilder();
        r.append(ly ? "Вид ЛЮ «рамка YuNet + точки LBF» против Ю5 (база) на протоколе feret-mp2, вариант (в) – Жорик №15, №16 от 20.10.2026\n"
                : "Воспроизведение схемы «база» feret-mp2 (XBaseMp2) – Жорик №15, №16 от 20.10.2026\n");
        r.append("Код: проект ").append(FarMethods.commit()).append(" (не менялся), запускалка tools/verify VerifyMp2Ly\n");
        if (!ly) {
            r.append(String.format(Locale.ROOT, "Пороги МП – из шапки xbase_feret_mp2.txt (4 знака): R_МП = %.4f°, r0_МП = %.4f°.%n", th.bigR(), th.r0()));
            XBaseMp2.header(r, th, tpl);
        } else {
            r.append("Кадр ЛЮ 92×112 – FeretData.frameA по строке YuNet, глаза (поля 4–7) – центры глаз LBF по рамке YuNet (LbfFrames.Lm.five(): средние 36–41 и\n"
                    + "42–47, LbfFrames.java:64–72); нос и углы рта – YuNet. Отбор |r| ≤ R и выбор модели X / X+mirror по |r| ≤ r0 – r YuNet в обоих видах\n"
                    + "(FeiDataset.pose по строке YuNet, FeiDataset.java:96–97; XBaseA0.feretSetup). Когорта FEI в обоих видах – кадры «а» экспорта, выровнены по\n"
                    + "YuNet (оговорка №16). Пул, роли, галереи, обучение W и 5 моделей частей – как в базе (U – YuNet нашёл, I – отбор базы).\n"
                    + String.format(Locale.ROOT, "База: R = %.4f, r0 = %s. Кандидаты: L1-xs-F, L1-tt-xs-F. Разности «ЛЮ − база» – бутстреп по людям (одни веса).%n",
                    tpl.rLimit, tpl.r0));
            if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", tpl.b,
                    tpl.seed + FeiBootstrap.SEED_OFFSET));
            r.append("Очки – ground_truths, поле glasses (Yes / прочее); группы «очки» / «без очков» – только пробы своих, контроль и «чужие» П2/П3 – все.\n");
        }
        boolean stop = XBaseMp2.feretSizes(r, fs);
        System.out.print(r);
        System.out.flush();
        if (stop) throw new IllegalStateException("условие остановки feret-mp2 сработало");

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
        List<XBaseXsF.Conf> confs = XBaseMp2.confs(ro);
        Set<String> used = new TreeSet<>();
        for (XBaseXsF.Conf c : confs) {
            for (String p : c.own()) {
                used.addAll(c.gal().get(p));
                used.addAll(XBaseMp2.proU(fs.roU, c.name(), c.own()).get(p));
            }
        }
        fs.ctrlKeysU.values().forEach(used::addAll);
        fs.pool.values().forEach(ds -> ds.forEach(used::addAll));
        boolean sameRoles = XBaseMp2.sameRoles(fs);
        r.append(String.format(Locale.ROOT, "Роли на I совпадают с прежними (feret-xs2 / feret-subsp): %s. Снимков FERET в счёте: %d.%n", sameRoles ? "да" : "нет", used.size()));
        long glN = used.stream().filter(k -> Boolean.TRUE.equals(gl.get(k))).count();
        long glNull = used.stream().filter(k -> gl.get(k) == null).count();
        r.append(String.format(Locale.ROOT, "Из них в очках (glasses=Yes): %d; без поля glasses: %d.%n", glN, glNull));
        if (ly) r.append(frameCheck(used, su, 200));

        Map<String, String> mp2 = XBaseSubsp.blocks(old.resolve("xbase_feret_mp2_scores.tsv.gz"));
        Map<String, String> xs2 = XBaseSubsp.blocks(old.resolve("xbase_feret_xs2_scores.tsv.gz"));
        Map<String, String> subsp = XBaseSubsp.blocks(old.resolve("xbase_feret_subsp_scores.tsv.gz"));
        StringBuilder check = new StringBuilder(ly ? "\n=== Проверка базы: строки L1 и +mirror (k = 0) – побайтно с файлами оценок feret-xs2 / feret-subsp ===\n"
                : "\n=== Проверка: строки оценок схемы «база» – побайтно (все строки блока, по сортировке) с xbase_feret_mp2_scores.tsv.gz ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Acc> acc = new LinkedHashMap<>();
        Map<String, int[]> grpN = new LinkedHashMap<>();
        double fitS = 0;
        int schemes = ly ? 2 : 1;
        for (int sc = 0; sc < schemes; sc++) {
            boolean lyScheme = sc == 1;
            Map<String, Mat> fr = new LinkedHashMap<>();
            if (lyScheme) {
                fr.putAll(lyFrames(used, su, lbf));
            } else {
                for (String k : used) fr.put(k, FeretData.frame(su.dir, "a", k));
            }
            List<Sample> fei = new ArrayList<>();
            export.data().fei().values().forEach(fei::addAll);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
            xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
            xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
            fr.values().forEach(Mat::release);
            XBaseEval.Gt baseI = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
            XBaseEval.Gt baseU = XBaseFeret.gt(tpl, persons, ro.ctrl, fs.ctrlKeysU, cohort, su.rOf, su.imageNo);
            Predicate<String> rej = fs::rejB;
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
                        Map<String, List<String>> proUa = XBaseMp2.proU(fs.roU, c.name(), own);
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
                        String name = String.format(Locale.ROOT, "%s_%d_%d_%d_%d", mode, sc, c.s(), f, ps);
                        Path bf = out.resolve("blocks").resolve(name + ".tsv");
                        String ls;
                        if (Files.exists(bf)) {
                            ls = Files.readString(bf, StandardCharsets.UTF_8);
                        } else {
                            long t1 = System.nanoTime();
                            ls = XBaseEval.fitLines(gU, xs[f], EnumSet.of(V.L1), gal, probesU, parts, c.s());
                            double s = (System.nanoTime() - t1) / 1e9;
                            fitS += s;
                            Path tmp = bf.resolveSibling(name + ".tmp");
                            Files.writeString(tmp, ls, StandardCharsets.UTF_8);
                            Files.move(tmp, bf, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                            System.out.printf(Locale.ROOT, "VerifyMp2Ly: %s, %s, %s, проход %d ready (%.0f s)%n", SCHEMES[sc], c.name(), rowLabel, ps + 1, s);
                        }
                        String blockHead = "схема " + SCHEMES[sc] + ", проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + rowLabel;
                        lines.append("# ").append(blockHead).append('\n').append(ls);
                        if (!ly) {
                            String o = mp2.get(blockHead);
                            boolean same = o != null && sortedRows(o).equals(sortedRows(ls));
                            check.append(String.format(Locale.ROOT, "%s %s, проход %d: %s (%d строк)%n", c.name(), rowLabel, ps + 1, same ? "совпало" : "НЕ СОВПАЛО",
                                    ls.split("\n").length));
                        } else if (!lyScheme && sameRoles) {
                            String o = c.name().equals("G2") ? subsp.get("G2, проекция " + rowLabel + ", проход " + (ps + 1) + ", k = 0")
                                    : xs2.get("проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + rowLabel);
                            boolean same = o != null && XBaseSubsp.l1Rows(o).equals(XBaseSubsp.l1Rows(XBaseMp2.only(ls, XBaseMp2.ids(probesI))));
                            check.append(String.format(Locale.ROOT, "%s %s, проход %d: %s (%d строк)%n", c.name(), rowLabel, ps + 1, same ? "совпало" : "НЕ СОВПАЛО",
                                    XBaseSubsp.l1Rows(XBaseMp2.only(ls, XBaseMp2.ids(probesI))).size()));
                        }
                        Set<String> rejected = new HashSet<>();
                        for (Probe q : probesU) if (rej.test(q.key())) rejected.add(q.role() + "\t" + q.person() + "\t" + q.image());
                        String lsA = XBaseMp2.refuse(ls, rejected);
                        String base = c.name() + " " + rowLabel + " " + SCHEMES[sc];
                        acc.computeIfAbsent(base + " (б)", x -> new Acc()).add(XBaseEval.parse(gI, EnumSet.of(V.L1), gal, probesI, parts, ls));
                        acc.computeIfAbsent(base + " (а)", x -> new Acc()).add(XBaseEval.parse(gU, EnumSet.of(V.L1), gal, probesU, parts, lsA));
                        if (ly) {
                            for (String tbl : new String[] {"(б)", "(а)"}) {
                                List<Probe> src = tbl.equals("(б)") ? probesI : probesU;
                                XBaseEval.Gt g = tbl.equals("(б)") ? gI : gU;
                                String l = tbl.equals("(б)") ? ls : lsA;
                                for (boolean wantG : new boolean[] {true, false}) {
                                    List<Probe> sub = src.stream().filter(q -> !q.role().equals(XBaseEval.OWN) || Boolean.TRUE.equals(gl.get(q.key())) == wantG).toList();
                                    String key = base + " " + tbl + (wantG ? " очки" : " без очков");
                                    acc.computeIfAbsent(key, x -> new Acc()).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, sub, parts, l));
                                    int ownN = (int) sub.stream().filter(q -> q.role().equals(XBaseEval.OWN)).count();
                                    grpN.computeIfAbsent(key, x -> new int[1])[0] += ownN;
                                }
                            }
                        }
                    }
                }
            }
        }
        r.append(check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        if (!ly) {
            XBaseMp2.tablesOut(r, acc, persons.size(), tpl, List.of(List.of("межс.", "внутри"), List.of("G2")));
        } else {
            r.append("\nПробы своих по группам очков (сумма по проходам; (б) – I, (а) – U):\n");
            grpN.forEach((k, v) -> {
                if (k.contains(" база ") && k.contains("L1-xs-F")) r.append(k.replace(" L1-xs-F база", "")).append(": ").append(v[0]).append('\n');
            });
            for (String tbl : new String[] {"(а)", "(б)", "(а) очки", "(а) без очков", "(б) очки", "(б) без очков"}) {
                for (List<String> grp : List.of(List.of("межс.", "внутри"), List.of("G2"))) {
                    Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
                    List<String[]> pairs = new ArrayList<>();
                    for (Map.Entry<String, Acc> e : acc.entrySet()) {
                        String l = e.getKey();
                        if (!l.endsWith(" " + tbl) || l.endsWith(" без очков") && !tbl.endsWith("без очков") || grp.stream().noneMatch(g -> l.startsWith(g + " ")))
                            continue;
                        res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), e.getValue().cfgs, e.getValue().imps, persons.size(), tpl.b,
                                tpl.seed + FeiBootstrap.SEED_OFFSET));
                        if (l.contains(" ЛЮ ")) pairs.add(new String[] {l, l.replace(" ЛЮ ", " база ")});
                    }
                    XBaseXsF.flush(r, res, pairs, persons.size(), tpl, "Таблица " + tbl + ": " + String.join(", ", grp));
                }
            }
        }
        XBaseOther.writeScores(out.resolve("verify_" + mode + "_scores.tsv.gz"), "# FERET, VerifyMp2Ly " + mode + "; блоки — схема, проход, конфигурация, строка; "
                + "пробы U (отказ отбором не применён)\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("verify_" + mode + ".txt"), r);
        System.out.print(r);
    }
}
