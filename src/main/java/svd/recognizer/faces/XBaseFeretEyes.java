package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Mat;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseFeret.Conf;
import svd.recognizer.storage.SettingsStore;

/**
 * Диагностика выравнивания на FERET (PLAN.md п. 5.5; план подтверждён Хозяином 12.10.2026): L1 и L1-tt на кадрах «а»
 * 92×112 трёх видов — Р (ручные 4 точки разметки: глаза, нос, центр рта), Ю4 (те же 4 точки YuNet: центр рта — середина
 * уголков), Ю5 (нынешние 5 точек YuNet); подобие одно (FaceAlignment.similarity, LMEDS) на шаблон FaceAlignment.ownTemplate
 * (в 4 точках — середина уголков рта). Только снимки с ручной разметкой (и найденным лицом, |r| ≤ R); роли — XBaseFeret.roles
 * по этим снимкам; контрольные чужие — те же 200, что в основном прогоне FERET, с их размеченными снимками; когорта FEI 20
 * общая; выбор модели по позе — по r YuNet во всех видах. Главная разность — Р − Ю4, справочно Ю4 − Ю5. Отчёт и файл оценок —
 * только в каталоге базы (reports/xbase_feret_eyes*.txt|tsv.gz), кэш кадров — derived/a_manual, derived/a_y4.
 *
 * @author ssv
 */
final class XBaseFeretEyes {

    /** Виды кадров: подпись и каталог derived. */
    static final String[][] KINDS = {{"Р", FeretData.MANUAL}, {"Ю4", FeretData.Y4}, {"Ю5", "a"}};
    /** Методы: подпись и набор векторов (0 — none, 1 — tt). */
    static final String[] METHODS = {"L1", "L1-tt"};

    private XBaseFeretEyes() {
    }

    /**
     * Вариант прогона: tag — имя отчёта, файла оценок и контрольной точки (xbase_&lt;tag&gt;, checkpoint_&lt;tag&gt;); fp — начало
     * отпечатка; kinds — виды кадров {подпись, каталог derived}; pairs — пары видов (уменьшаемое, вычитаемое); note — строка
     * шапки о дополнительных видах или null.
     */
    record Variant(String tag, String fp, String[][] kinds, String[][] pairs, String note) {}

    /** Прежний прогон feret-eyes: Р, Ю4, Ю5; пары Р − Ю4 и Ю4 − Ю5. */
    static final Variant EYES = new Variant("feret_eyes", "feret-eyes", KINDS, new String[][] {{"Р", "Ю4"}, {"Ю4", "Ю5"}}, null);

    /** (а2б) этап 2, кандидат 0б: Р, К4, Ю4; пары К4 − Ю4 и Р − К4. */
    static final Variant EYES2 = new Variant("feret_eyes2", "feret-eyes2", new String[][] {{"Р", FeretData.MANUAL}, {"К4", FeretData.K4_0B},
            {"Ю4", FeretData.Y4}}, new String[][] {{"К4", "Ю4"}, {"Р", "К4"}}, "  К4 — (а2б) этап 2: точки глаз кандидата 0б (каскад "
            + "haarcascade_eye_tree_eyeglasses, derived/eyepts.tsv; не найден — точка YuNet), нос и центр рта — YuNet (как Ю4).\n");

    static void run(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        run(export, exportDir, feiKeys, selLines, settings, EYES);
    }

    static void run(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Variant va) throws IOException {
        String[][] kinds = va.kinds();
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-eyes: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval feret-eyes: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        tpl.r0 = FeiPose.r0(rawR0, "+pose");
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);
        Integer eq = settings.loadFacesFeretGalleryEqual() ? 0 : null; // №9: равные галереи — по умолчанию, old — прежние
        Path out = eq == null ? dir.resolve("reports") : dir.resolve("reports").resolve(XBaseEval.eqDir(eq, true));
        Files.createDirectories(out);

        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        long t0 = System.nanoTime();
        FeretData.frames4(imgs, det, dir);
        if (va == EYES2) FeretData.framesK4(imgs, det, dir);
        double frS = (System.nanoTime() - t0) / 1e9;

        // Отбор: все (как в основном прогоне) и только размеченные.
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> keptAll = new TreeMap<>();
        Map<String, Map<String, List<Img>>> keptMarked = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        int marked = 0;
        for (Img im : imgs) {
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) continue;
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) continue;
            keptAll.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
            if (FeretData.manual4(im) == null) continue;
            marked++;
            keptMarked.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
        }
        Map<String, TreeSet<String>> dates = XBaseFeret.dates(imgs);
        XBaseFeret.Roles all = XBaseFeret.roles(dates, keptAll, tpl.seed);
        XBaseFeret.Roles ro = XBaseFeret.roles(dates, keptMarked, tpl.seed);
        XBaseFeret.Roles roAll = ro;
        if (eq != null) {
            Map<String, Img> byKey = new HashMap<>();
            for (Img im : imgs) byKey.put(im.key(), im);
            ro = XBaseFeret.equalize(ro, byKey);
        }
        // Контрольные чужие — те же 200, что в основном прогоне, с их размеченными снимками.
        List<String> ctrl = new ArrayList<>();
        Map<String, List<String>> ctrlKeys = new TreeMap<>();
        for (String p : all.ctrl) {
            List<String> ks = new ArrayList<>();
            keptMarked.getOrDefault(p, Map.of()).values().forEach(l -> ks.addAll(XBaseFeret.keys(l)));
            if (ks.isEmpty()) continue;
            ctrl.add(p);
            ctrlKeys.put(p, XBaseFeret.sorted(ks));
        }

        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        // №8: векторы — по прежним ролям (как в XBaseFeret.run).
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.g2, roAll.pro2, ctrlKeys)) m.values().forEach(used::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        Map<String, Mat> cohortFr = XBaseEval.frames(export, fei, cohortKeys);
        // [вид][набор]: векторы L1 (none) и L1 (tt).
        XBaseEval[][] xs = new XBaseEval[kinds.length][2];
        for (int k = 0; k < kinds.length; k++) {
            Map<String, Mat> fr = new LinkedHashMap<>();
            for (String key : used) fr.put(key, FeretData.frame(dir, kinds[k][1], key));
            fr.putAll(cohortFr);
            xs[k][0] = new XBaseEval();
            xs[k][0].vectors(fr, EnumSet.of(V.L1), tpl.params);
            xs[k][1] = new XBaseEval();
            xs[k][1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
            for (String key : used) fr.get(key).release();
        }
        cohortFr.values().forEach(Mat::release);

        List<Conf> confs1 = List.of(new Conf("межс.", 0, ro.own1, ro.gal1, ro.pro1, null), new Conf("внутри", 1, ro.own1in, ro.gal1in, ro.pro1in, null));
        List<Conf> confs2 = List.of(new Conf("G1", 2, ro.own2, ro.g1, ro.pro2, null), new Conf("G2-i", 3, ro.own2, ro.g2, ro.pro2, null),
                new Conf("G2-ii", 4, ro.own2, ro.g2, ro.pro2, ro.g2s), new Conf("G2'", 5, ro.own2, ro.g2p, ro.pro2, null));
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "%s %s; снимков %d; размеченных отобранных %d; seed %d; r0 %s%s", va.fp(), dir,
                imgs.size(), marked, tpl.seed, tpl.r0, eq == null ? "" : "; №8 галерея fa/fb"));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_" + va.tag() + (eq == null ? "" : "_eq")), fp);

        StringBuilder r = new StringBuilder();
        if (eq != null) r.append(XBaseFeret.EQ_NOTE);
        r.append("PLAN.md п. 5.5: диагностика выравнивания на FERET — кадры по ручным точкам разметки против точек YuNet (XBaseEval feret-eyes); "
                + "только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append("Кадры «а» 92×112, одно подобие (FaceAlignment.similarity, LMEDS) на шаблон ownTemplate(92, 112):\n"
                + "  Р — ручные 4 точки разметки: правый глаз человека (right_eye, на снимке слева), левый (left_eye), нос, центр рта; в шаблоне\n"
                + "      центр рта — середина уголков; Ю4 — те же 4 точки YuNet (центр рта — середина уголков); Ю5 — нынешние 5 точек YuNet.\n"
                + "Методы: L1 (LBP u2 (8, 2) 7×7 + MLDA, none) и L1-tt (то же на выходе Tan – Triggs); зеркала (способ а) и выбор модели по позе\n"
                + "r0 = " + tpl.r0 + " — по r YuNet во всех видах; когорта MLDA — FEI 20; порог П2 / П3 по «чужим» галереи своей конфигурации.\n");
        if (va.note() != null) r.append(va.note());
        r.append(String.format(Locale.ROOT, "Снимков fa/fb: %d; с ручной разметкой (глаза, нос, рот), найденным лицом и |r| ≤ R = %.4f: %d; кадры Р и Ю4 — %.0f с "
                + "(из кэша derived/ — быстрее).%n", imgs.size(), tpl.rLimit, marked, frS));
        r.append(String.format(Locale.ROOT, "Роль (1): своих %d (выбыли %d), галерея %d, пробы %d; внутри сессии — своих %d, галерея fa %d, пробы fb %d.%n", ro.own1.size(),
                ro.dropped[0], XBaseFeret.sum(ro.gal1), XBaseFeret.sum(ro.pro1), ro.own1in.size(), XBaseFeret.sum(ro.gal1in), XBaseFeret.sum(ro.pro1in)));
        r.append(String.format(Locale.ROOT, "Роль (2): своих %d (выбыли %d); G1 %d, G2 %d, G2' %d; пробы %d.%n", ro.own2.size(), ro.dropped[1], XBaseFeret.sum(ro.g1),
                XBaseFeret.sum(ro.g2), XBaseFeret.sum(ro.g2p), XBaseFeret.sum(ro.pro2)));
        r.append(String.format(Locale.ROOT, "Контрольные чужие — те же, что в основном прогоне (200), с размеченными снимками: людей %d, снимков %d.%n", ctrl.size(),
                XBaseFeret.sum(ctrlKeys)));
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям роли и контроля: B = %d, seed = %d; одни веса для всех строк роли.%n", tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder lines = new StringBuilder();
        for (int ex = 0; ex < 2; ex++) {
            List<Conf> confs = ex == 0 ? confs1 : confs2;
            TreeSet<String> ps = new TreeSet<>(ex == 0 ? ro.own1 : ro.own2);
            ps.addAll(ctrl);
            List<String> persons = new ArrayList<>(ps);
            XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ctrl, ctrlKeys, cohort, rOf, imageNo);
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            for (Conf c : confs) {
                XBaseEval.Gt g = XBaseFeret.withOwn(base, c.own());
                List<XBaseEval.Probe> probes = XBaseEval.probes(g, c.pro());
                List<List<String>> parts = XBaseOther.cvParts(c.own(), tpl.seed, XBaseEval.CV_S + c.s());
                for (int k = 0; k < kinds.length; k++) {
                    for (int set = 0; set < 2; set++) {
                        String name = String.format(Locale.ROOT, "conf_%d_%d_%d", c.s(), k, set);
                        String ls = cp.read(name);
                        Set<V> vs = EnumSet.of(V.L1);
                        if (ls == null) {
                            ls = c.sessions() == null ? XBaseEval.fitLines(g, xs[k][set], vs, c.gal(), probes, parts, c.s())
                                    : XBaseFeret.fitLinesSessions(g, xs[k][set], vs, c.sessions(), c.gal(), probes, parts, c.s());
                            cp.save(name, ls);
                            System.out.printf(Locale.ROOT, "FERET-eyes: %s %s %s готово%n", c.name(), kinds[k][0], METHODS[set]);
                        }
                        lines.append("# конфигурация ").append(c.name()).append(", кадры ").append(kinds[k][0]).append(", ").append(METHODS[set]).append('\n').append(ls);
                        XBaseEval.Parsed p = XBaseEval.parse(g, vs, c.gal(), probes, parts, ls);
                        String label = c.name() + " " + kinds[k][0] + " " + METHODS[set];
                        res.put(label, FeiThreshold.evaluate(label, V.L1.ratioId(), List.of(p.cfg.get(V.L1)), List.of(p.imp.get(V.L1)), persons.size(), tpl.b,
                                tpl.seed + FeiBootstrap.SEED_OFFSET));
                    }
                }
                for (String m : METHODS) {
                    for (String[] pk : va.pairs()) pairs.add(new String[] {c.name() + " " + pk[0] + " " + m, c.name() + " " + pk[1] + " " + m});
                }
            }
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            int[] unit = FeiBootstrap.unit(persons.size());
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
            r.append("\n\n######## ").append(ex == 0 ? "Роль (1): межсессионно и опора внутри сессии" : "Роль (2): регистрация по нескольким дням")
                    .append(String.format(Locale.ROOT, " — бутстреп по %d людям (свои и контроль) ########%n", persons.size()));
            XBaseOther.tables(r, res, sep, tpl.b, pairs);
        }
        XBaseOther.writeScores(out.resolve("xbase_" + va.tag() + "_scores.tsv.gz"), "# FERET, диагностика выравнивания (XBaseFeretEyes), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_" + va.tag() + ".txt"), r);
        System.out.print(r);
        cp.delete();
    }
}
