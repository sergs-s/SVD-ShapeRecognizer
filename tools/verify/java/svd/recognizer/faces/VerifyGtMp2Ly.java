package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
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
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.face.Facemark;
import org.opencv.imgcodecs.Imgcodecs;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseMp2.Acc;
import svd.recognizer.faces.XBaseMp2.FeretSets;
import svd.recognizer.faces.XBaseMp2.Thr;
import svd.recognizer.faces.XBaseMpPts.Mp;
import svd.recognizer.storage.SettingsStore;

/**
 * Вид ЛЮ на GT по протоколу gt-mp2 (XBaseMp2.gt, XBaseMp2.java:776–1061) – tools/verify, ветка cloud-verify; Жорик №17, №18 от
 * 20.10.2026. Тот же цикл, методы проекта; код проекта не меняется.
 * <p>
 * lbf – точки LBF по рамке YuNet (detections.tsv экспорта, поля 0–3) на сырых снимках GT (faces.gt.dir), кэш в формате
 * LbfFrames.cached; вход – HaarFrames.input(цвет, 1,0), как GT в HaarFrames.bases.
 * <p>
 * repro – схема «база» gt-mp2: U, I по МП (кэш mp_points_crop3.tsv рядом с gt_db, как XBaseMp2.java:788), пороги МП – из
 * шапки xbase_gt_mp2.txt; пул FERET – как VerifyMp2Ly repro (I FERET по МП FERET). Строки – побайтно с блоками
 * xbase_gt_mp2_scores.tsv.gz, таблицы – XBaseMp2.tablesOut.
 * <p>
 * ly – вариант (в): I – отбор базы (r YuNet), роли = gt-subsp (строки базы сверяются с xbase_gt_subsp_scores.tsv.gz, как
 * XBaseMp2.java:1005–1011); кадр ЛЮ – FeretData.frameA по строке YuNet с глазами LBF (Lm.five(), поля 4–7); отбор и модель – r
 * YuNet; когорта FEI и пул FERET – кадры «а» в обоих видах. Пары «ЛЮ − база».
 * <p>
 * Аргументы: lbf &lt;кэш&gt; | repro &lt;каталог отчёта gt-mp2&gt; &lt;каталог FERET-отчёта feret-mp2&gt; &lt;выход&gt; |
 * ly &lt;кэш LBF&gt; &lt;каталог gt-subsp&gt; &lt;выход&gt;. Ключи – как XBaseEval (FACES_EXPORT_DIR, FACES_GT_DIR, FACES_FERET_DIR,
 * FACES_MODELS_DIR, FACES_FEI_POSE_R0, FACES_FEI_BOOTSTRAP).
 */
public final class VerifyGtMp2Ly {

    static final String[] SCHEMES = {"база", "ЛЮ"};

    private VerifyGtMp2Ly() {
    }

    static void lbf(SettingsStore settings, Path dir, Path cache) throws IOException {
        Path models = Paths.get(settings.loadFacesModelsDir());
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Facemark fm = LbfFrames.load(models);
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        try (Writer w = Files.newBufferedWriter(cache, StandardCharsets.UTF_8)) {
            w.write("# lbf по рамке YuNet (detections.tsv экспорта, поля 0–3)|" + LbfFrames.signature(models) + "\n");
            for (Map.Entry<String, double[]> e : det.entrySet()) {
                double[] row = e.getValue();
                LbfFrames.Lm lm = null;
                if (row != null) {
                    Mat color = Imgcodecs.imread(gtDir.resolve(e.getKey() + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                    if (color.empty()) throw new IOException("нет снимка GT " + e.getKey());
                    Mat in = HaarFrames.input(color, 1.0);
                    color.release();
                    lm = LbfFrames.fit(fm, in, 1.0, new HaarFrames.Det(1, new double[] {row[0], row[1], row[2], row[3]}, null, null));
                    in.release();
                }
                StringBuilder m = new StringBuilder(e.getKey()).append('\t').append(lm == null ? "0" : "1");
                for (int i = 0; i < LbfFrames.POINTS; i++)
                    m.append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][0])).append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][1]));
                w.write(m.append('\n').toString());
            }
        }
    }

    static String sortedRows(String lines) {
        return VerifyMp2Ly.sortedRows(lines);
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        String mode = a[0];
        SettingsStore settings = new SettingsStore();
        Path dir = Paths.get(settings.loadFacesExportDir());
        if (mode.equals("lbf")) {
            lbf(settings, dir, Paths.get(a[1]));
            return;
        }
        boolean ly = mode.equals("ly");
        Path out = Paths.get(a[3]);
        Files.createDirectories(out.resolve("blocks"));
        FarExport.Loaded export = FarExport.read(dir);
        List<String> selLines = Files.readAllLines(dir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        long t0 = System.nanoTime();

        // FERET (пул): как VerifyMp2Ly – repro: I FERET по МП FERET с порогами из шапки feret-mp2; ly: I – отбор базы.
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Thr th = ly ? new Thr(Double.POSITIVE_INFINITY, 0, 0, Double.NaN, 0, 0) : VerifyMp2Ly.thrFromReport(Paths.get(a[2]).resolve("xbase_feret_mp2.txt"));
        if (!ly) {
            Thr thG = VerifyMp2Ly.thrFromReport(Paths.get(a[1]).resolve("xbase_gt_mp2.txt"));
            if (thG.bigR() != th.bigR() || thG.r0() != th.r0()) throw new IllegalStateException("пороги МП в шапках feret-mp2 и gt-mp2 различаются");
        }
        FeretSets fs = VerifyMp2Ly.sets(su, th, ly);

        double rLimit = XBaseEval.poseLimit(selLines);
        double r0 = FeiPose.r0(settings.loadFacesFeiPoseR0(), "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        List<String> mpHead = new ArrayList<>();
        Map<String, Mp> mp = ly ? Map.of() : XBaseMpPts.read(gtDir.getParent().resolve(XBaseMp2.CROP_FILE), mpHead);
        Map<String, LbfFrames.Lm> lbf = ly ? VerifyLbfYunet.lbf(Paths.get(a[1])) : Map.of();
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rB = new HashMap<>();
        Map<String, Double> yM = new HashMap<>();
        Set<String> u = new TreeSet<>();
        int mpMissing = 0;
        for (Map.Entry<String, double[]> e : det.entrySet()) {
            if (e.getValue() == null) continue;
            double yaw = 0;
            if (!ly) {
                Mp m = mp.get(e.getKey());
                if (m == null) throw new IllegalStateException("нет строки МП для " + e.getKey());
                if (!m.found()) {
                    mpMissing++;
                    continue;
                }
                yaw = Math.abs(m.yaw());
            }
            rB.put(e.getKey(), Math.abs(FeiDataset.pose(e.getValue())));
            yM.put(e.getKey(), yaw);
            u.add(e.getKey());
        }
        Predicate<String> rejB = pk -> rB.get(pk) > rLimit;
        Predicate<String> rejM = pk -> yM.get(pk) > th.bigR();
        Predicate<String> inI = pk -> u.contains(pk) && !rejB.test(pk) && !rejM.test(pk);
        Map<String, Double> rOld = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        XBaseEval.Gt[] gI = {XBaseMp2.gtRoles(export, dir, feiKeys, settings, inI, rB, r0), XBaseMp2.gtRoles(export, dir, feiKeys, settings, inI, rB, r0)};
        long eqSeed = gI[0].seed + XBaseEval.EQ_SEED_OFFSET;

        StringBuilder r = new StringBuilder();
        r.append(ly ? "Вид ЛЮ «рамка YuNet + точки LBF» против Ю5 (база) на протоколе gt-mp2, вариант (в) – Жорик №17, №18 от 20.10.2026\n"
                : "Воспроизведение схемы «база» gt-mp2 (XBaseMp2.gt) – Жорик №17, №18 от 20.10.2026\n");
        r.append("Код: проект ").append(FarMethods.commit()).append(" (не менялся), запускалка tools/verify VerifyGtMp2Ly\n");
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; снимки GT – ").append(gtDir).append('\n');
        for (String h : mpHead) r.append("Кэш МП: ").append(h.substring(1).trim()).append('\n');
        XBaseEval.Gt tplH = XBaseMp2.withR0(gI[0], r0);
        tplH.rLimit = rLimit;
        if (!ly) {
            r.append(String.format(Locale.ROOT, "Пороги МП – из шапок xbase_gt_mp2.txt и xbase_feret_mp2.txt (4 знака): R_МП = %.4f°, r0_МП = %.4f°.%n", th.bigR(), th.r0()));
            XBaseMp2.header(r, th, tplH);
        } else {
            r.append("Кадр ЛЮ 92×112 – FeretData.frameA по строке YuNet, глаза (поля 4–7) – центры глаз LBF по рамке YuNet (LbfFrames.Lm.five(), LbfFrames.java:64–72);\n"
                    + "нос и углы рта – YuNet. Отбор |r| ≤ R и выбор модели X / X+mirror по |r| ≤ r0 – r YuNet в обоих видах. Когорта FEI и пул FERET в обоих видах –\n"
                    + "кадры «а», выровнены по YuNet (оговорка №16). Роли, галереи (межс. N = 6, внутри N = 3), обучение W и 5 моделей частей – как в базе (gt-subsp).\n");
            r.append(String.format(Locale.ROOT, "База: R = %.4f, r0 = %s. Бутстреп: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", rLimit, r0, tplH.b,
                    tplH.seed + FeiBootstrap.SEED_OFFSET));
        }
        r.append(String.format(Locale.ROOT, "%nСнимки GT: %d; YuNet нашёл %d; МП (вырезка) не нашёл %d; U = %d; отобрано базой %d, МП %d; I = %d.%n", det.size(),
                rOld.size(), mpMissing, u.size(), u.stream().filter(rejB.negate()).count(), u.stream().filter(rejM.negate()).count(), u.stream().filter(inI).count()));

        XBaseEval.Gt b = gI[0];
        Map<String, List<String>> galX = XBaseEval.equalize(b.own, b.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(b.own, b.galleryAll, b.seed);
        Map<String, List<String>> galIn = XBaseEval.equalize(b.own, split.get(0), eqSeed);
        Map<String, List<String>> galX0 = XBaseEval.equalize(g0.own, g0.galleryAll, eqSeed);
        List<Map<String, List<String>>> split0 = XBaseEval.withinSplit(g0.own, g0.galleryAll, g0.seed);
        Map<String, List<String>> galIn0 = XBaseEval.equalize(g0.own, split0.get(0), eqSeed);
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
        Object[][] modes = {{"межс.", galX, b.probeKeys, proXU}, {"внутри", galIn, split.get(1), proInU}};
        boolean samePool = XBaseXsF.pool(fs.su.ro, fs.su.byKey).equals(fs.pool);
        boolean sameRoles = g0.own.equals(b.own) && g0.ctrl.equals(b.ctrl) && galX0.equals(galX) && galIn0.equals(galIn) && g0.probeKeys.equals(b.probeKeys)
                && split0.get(1).equals(split.get(1)) && g0.byPerson.equals(b.byPerson) && samePool;
        r.append(String.format(Locale.ROOT, "Своих %d, контрольных %d; роли на I совпадают с gt-subsp: %s; пул FERET %d людей (совпадает с прежним: %s).%n", b.own.size(),
                b.ctrl.size(), sameRoles ? "да" : "нет", fs.pool.size(), samePool ? "да" : "нет"));

        Set<String> cohortKeys = new HashSet<>();
        b.cohort.forEach(cohortKeys::addAll);
        Map<String, List<List<String>>> pool = fs.pool;
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        Map<String, String> oldMp2 = ly ? Map.of() : XBaseSubsp.blocks(Paths.get(a[1]).resolve("xbase_gt_mp2_scores.tsv.gz"));
        Map<String, String> oldSubsp = ly ? XBaseSubsp.blocks(Paths.get(a[2]).resolve("xbase_gt_subsp_scores.tsv.gz")) : Map.of();
        StringBuilder check = new StringBuilder(ly ? "\n=== Проверка базы: строки L1 и +mirror (k = 0) – побайтно с файлом оценок gt-subsp ===\n"
                : "\n=== Проверка: строки оценок схемы «база» – побайтно (все строки блока, по сортировке) с xbase_gt_mp2_scores.tsv.gz ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Acc> acc = new LinkedHashMap<>();
        List<List<String>> parts = FeiMethods.cvParts(b.own, b.seed, XBaseEval.CV_S);
        double fitS = 0;
        String[][] projs = {{"L1-xs-F", "0"}, {"L1-tt-xs-F", "1"}};
        List<List<String>> cohort0 = new ArrayList<>(b.cohort);
        String frameCheck = "";
        for (int sc = 0; sc < (ly ? 2 : 1); sc++) {
            boolean lyScheme = sc == 1;
            Map<String, Mat> fr = new LinkedHashMap<>();
            Set<String> gtKeys = new TreeSet<>();
            for (String pk : u) gtKeys.add(keyOf.get(pk));
            if (lyScheme) {
                for (String k : gtKeys) {
                    Sample s = sampleOf.get(k);
                    String pk = s.person() + "/" + s.file().getFileName().toString();
                    LbfFrames.Lm lm = lbf.get(pk);
                    if (lm == null) throw new IllegalStateException("нет точек LBF по рамке YuNet: " + pk);
                    double[] row = det.get(pk).clone();
                    double[] f = lm.five();
                    for (int i = 4; i < 8; i++) row[i] = f[i];
                    Mat color = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                    fr.put(k, FeretData.frameA(color, row));
                    color.release();
                }
            } else {
                fr.putAll(XBaseEval.frames(export, gtKeys.stream().map(sampleOf::get).toList(), null));
                if (ly) {
                    // сверка: кадр «а» экспорта GT и FeretData.frameA по исходной строке YuNet
                    int same = 0;
                    for (String k : gtKeys) {
                        Sample s = sampleOf.get(k);
                        String pk = s.person() + "/" + s.file().getFileName().toString();
                        Mat color = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                        Mat fa = FeretData.frameA(color, det.get(pk));
                        Mat d = new Mat();
                        Core.absdiff(fa, fr.get(k), d);
                        if (Core.countNonZero(d) == 0) same++;
                        color.release();
                        fa.release();
                        d.release();
                    }
                    frameCheck = String.format(Locale.ROOT, "Кадр «а» экспорта GT против FeretData.frameA по строке YuNet: побайтно %d из %d.%n", same, gtKeys.size());
                }
            }
            List<Sample> fei = new ArrayList<>();
            export.data().fei().values().forEach(fei::addAll);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            for (String k : poolKeys) fr.put(k, FeretData.frame(fs.su.dir, "a", k));
            XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
            xs[0].vectors(fr, EnumSet.of(V.L1), b.params);
            xs[1].vectors(fr, EnumSet.of(V.L1), b.params, IlluminationNorm.TAN_TRIGGS);
            fr.values().forEach(Mat::release);
            XBaseEval.Gt g = gI[sc];
            XBaseEval.Gt gUx = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> u.contains(pk), rB, r0);
            XBaseEval.Gt gU = XBaseMp2.withR0(g, g.r0);
            gU.own = g.own;
            gU.ctrl = g.ctrl;
            gU.persons = g.persons;
            gU.pIdx = g.pIdx;
            gU.cohort = g.cohort;
            gU.samples = gUx.samples;
            gU.rOf = gUx.rOf;
            for (String p : g.ctrl) gU.byPerson.put(p, gUx.byPerson.get(p));
            for (String p : g.own) gU.byPerson.put(p, gUx.byPerson.get(p));
            for (Object[] md : modes) {
                String mname = (String) md[0];
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
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    aug.addAll(poolCls);
                    g.cohort = aug;
                    gU.cohort = aug;
                    String name = String.format(Locale.ROOT, "%s_%d_%s_%d", mode, sc, mname.equals("межс.") ? "x" : "i", f);
                    Path bf = out.resolve("blocks").resolve(name + ".tsv");
                    String ls;
                    if (Files.exists(bf)) {
                        ls = Files.readString(bf, StandardCharsets.UTF_8);
                    } else {
                        long t1 = System.nanoTime();
                        ls = XBaseEval.fitLines(gU, xs[f], EnumSet.of(V.L1), gal, probesU, parts, 0);
                        double s = (System.nanoTime() - t1) / 1e9;
                        fitS += s;
                        Path tmp = bf.resolveSibling(name + ".tmp");
                        Files.writeString(tmp, ls, StandardCharsets.UTF_8);
                        Files.move(tmp, bf, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                        System.out.printf(Locale.ROOT, "VerifyGtMp2Ly: %s, %s, %s ready (%.0f s)%n", SCHEMES[sc], mname, pj[0], s);
                    }
                    String head = "схема " + SCHEMES[sc] + ", " + mname + ", проекция " + pj[0];
                    lines.append("# ").append(head).append('\n').append(ls);
                    if (!ly) {
                        String o = oldMp2.get(head);
                        boolean same = o != null && sortedRows(o).equals(sortedRows(ls));
                        check.append(String.format(Locale.ROOT, "%s %s: %s (%d строк)%n", mname, pj[0], same ? "совпало" : "НЕ СОВПАЛО", ls.split("\n").length));
                    } else if (!lyScheme) {
                        String o = oldSubsp.get(mname + ", проекция " + pj[0] + ", k = 0");
                        List<String> mine = XBaseSubsp.l1Rows(XBaseMp2.only(ls, XBaseMp2.ids(probesI)));
                        boolean same = sameRoles && o != null && XBaseSubsp.l1Rows(o).equals(mine);
                        check.append(String.format(Locale.ROOT, "%s %s: %s (%d строк)%n", mname, pj[0], same ? "совпало" : "НЕ СОВПАЛО", mine.size()));
                    }
                    Set<String> rejected = new HashSet<>();
                    for (Probe q : probesU) {
                        Sample s = sampleOf.get(q.key());
                        if (rejB.test(s.person() + "/" + s.file().getFileName().toString())) rejected.add(q.role() + "\t" + q.person() + "\t" + q.image());
                    }
                    acc.computeIfAbsent(mname + " " + pj[0] + " " + SCHEMES[sc] + " (б)", x -> new Acc()).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probesI, parts, ls));
                    acc.computeIfAbsent(mname + " " + pj[0] + " " + SCHEMES[sc] + " (а)", x -> new Acc())
                            .add(XBaseEval.parse(gU, EnumSet.of(V.L1), gal, probesU, parts, XBaseMp2.refuse(ls, rejected)));
                }
            }
        }
        r.append(frameCheck);
        r.append(check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, b.threads));
        if (!ly) {
            XBaseMp2.tablesOut(r, acc, b.persons.size(), tplH, List.of(List.of("межс."), List.of("внутри")));
        } else {
            for (String tbl : new String[] {"(а)", "(б)"}) {
                for (List<String> grp : List.of(List.of("межс."), List.of("внутри"))) {
                    Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
                    List<String[]> pairs = new ArrayList<>();
                    for (Map.Entry<String, Acc> e : acc.entrySet()) {
                        String l = e.getKey();
                        if (!l.endsWith(tbl) || grp.stream().noneMatch(x -> l.startsWith(x + " "))) continue;
                        res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), e.getValue().cfgs, e.getValue().imps, b.persons.size(), tplH.b,
                                tplH.seed + FeiBootstrap.SEED_OFFSET));
                        if (l.contains(" ЛЮ ")) pairs.add(new String[] {l, l.replace(" ЛЮ ", " база ")});
                    }
                    XBaseXsF.flush(r, res, pairs, b.persons.size(), tplH, "Таблица " + tbl + ": " + String.join(", ", grp));
                }
            }
        }
        XBaseOther.writeScores(out.resolve("verify_gt_" + mode + "_scores.tsv.gz"), "# GT, VerifyGtMp2Ly " + mode + "; блоки — схема, постановка, проекция; пробы U\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(out.resolve("verify_gt_" + mode + ".txt"), r);
        System.out.print(r);
    }
}
