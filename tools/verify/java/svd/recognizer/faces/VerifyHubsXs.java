package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
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
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseExcl.Data;
import svd.recognizer.storage.SettingsStore;

/**
 * Число сессий в галерее для L1-xs-F и L1-tt-xs-F (tools/verify, ветка cloud-verify; Жорик №6 от 21.10.2026): протокол части 1 (ii)
 * feret-hubs (XBaseHubs.java:473–541) – те же 25 человек с ≥ 4 датами, галерея – первые M = 1, 2, 3 даты (правило равных галерей
 * fa/fb), пробы – даты с 4-й, контроль – те же 200, части П2/П3 – конфигурации 12, 13, 14. Код проекта не меняется.
 * <p>
 * Сначала – воспроизведение строк L1, L1-tt и SFace части 1 (ii) (те же методы проекта: XBaseEval.fitLines,
 * XBaseFeret.fitLinesSessions, FeiThreshold.evaluate, XBaseOther.tables). Затем строки L1-xs-F и L1-tt-xs-F: к классам обучения MLDA
 * (галерея своих при данном M и когорта FEI) добавлены классы пула FERET – люди с ≥ 2 датами (XBaseXsF.pool), кроме 25 своих
 * (XBaseXsF.train, Kind.XS, excluded – свои), как в (4а) пул не пересекается со своими. W и 5 моделей частей обучаются заново при
 * каждом M на текущей галерее (так в XBaseHubs: каждый блок – свой XBaseEval.fitLines; схема (Wа) – все люди галереи в W).
 * <p>
 * Аргументы: каталог отчёта xbase_feret_hubs.txt, выходной каталог. Ключи – как XBaseEval (FACES_EXPORT_DIR, FACES_FERET_DIR,
 * FACES_MODELS_DIR, FACES_FEI_POSE_R0, FACES_FEI_BOOTSTRAP); правило равных галерей – по умолчанию (eq).
 */
public final class VerifyHubsXs {

    static final int[] CONF_II = XBaseHubs.CONF_II;
    static final String[] SET_NAMES = {"none", "tt", "SFace"};

    private VerifyHubsXs() {
    }

    interface Fit {
        String fit() throws IOException;
    }

    static String block(Path dir, String name, Fit f) throws IOException {
        Path bf = dir.resolve(name + ".tsv");
        if (Files.exists(bf)) return Files.readString(bf, StandardCharsets.UTF_8);
        long t1 = System.nanoTime();
        String l = f.fit();
        Path tmp = dir.resolve(name + ".tmp");
        Files.writeString(tmp, l, StandardCharsets.UTF_8);
        Files.move(tmp, bf, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        System.out.printf(Locale.ROOT, "VerifyHubsXs: блок %s (%.0f с)%n", name, (System.nanoTime() - t1) / 1e9);
        return l;
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        Path oldDir = Paths.get(a[0]);
        Path out = Paths.get(a[1]);
        Path blocks = out.resolve("blocks");
        Files.createDirectories(blocks);
        SettingsStore settings = new SettingsStore();
        Path exportDir = Paths.get(settings.loadFacesExportDir());
        FarExport.Loaded export = FarExport.read(exportDir);
        List<String> selLines = Files.readAllLines(exportDir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        Path dir = Paths.get(settings.loadFacesFeretDir());
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        tpl.r0 = FeiPose.r0(settings.loadFacesFeiPoseR0(), "+pose");
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);

        // Снимки, детекции, роли, сессии – как XBaseHubs.run (XBaseHubs.java:345–378).
        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> kept = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        for (Img im : imgs) {
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) continue;
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) continue;
            kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), x -> new ArrayList<>()).add(im);
        }
        XBaseFeret.Roles ro = XBaseFeret.roles(XBaseFeret.dates(imgs), kept, tpl.seed);
        Map<String, List<List<String>>> sess = new LinkedHashMap<>();
        Map<String, List<LocalDate>> when = new LinkedHashMap<>();
        for (String p : ro.own1) {
            List<List<String>> ds = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (Map.Entry<String, List<Img>> e : kept.get(p).entrySet()) {
                ds.add(XBaseFeret.keys(e.getValue()));
                ws.add(FeretData.date(e.getKey()));
            }
            sess.put(p, ds);
            when.put(p, ws);
        }
        Map<String, Img> byKey = new HashMap<>();
        for (Img im : imgs) byKey.put(im.key(), im);
        Data d = new Data(ro.own1, sess, when, XBaseExcl.pairSessions(sess, byKey));

        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        // Пул FERET (как (4а): XBaseXsF.pool – люди роли (1), снимки по правилу равных галерей) без своих части 1 (ii).
        List<String> own = d.own().stream().filter(p -> d.sess().get(p).size() >= 4).toList();
        Map<String, List<List<String>>> pool = XBaseXsF.pool(ro, byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, new HashSet<>(own), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        poolCls.forEach(poolKeys::addAll);

        Set<String> used = new TreeSet<>();
        sess.values().forEach(ds -> ds.forEach(used::addAll));
        ro.ctrlKeys.values().forEach(used::addAll);
        used.addAll(poolKeys);
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String k : used) {
            fr.put(k, FeretData.frame(dir, "a", k));
            sf.put(k, FeretData.frame(dir, "sface", k));
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        xs[2].sfaceMin = 1;
        xs[2].sfaceVectors(sf, sface);
        fr.values().forEach(Mat::release);
        sf.values().forEach(Mat::release);

        TreeSet<String> ps = new TreeSet<>(own);
        ps.addAll(ro.ctrl);
        XBaseEval.Gt g = XBaseFeret.withOwn(XBaseFeret.gt(tpl, new ArrayList<>(ps), ro.ctrl, ro.ctrlKeys, cohort, rOf, imageNo), own);
        List<List<String>> augCohort = new ArrayList<>(cohort);
        augCohort.addAll(poolCls);
        List<Probe> probes = XBaseEval.probes(g, XBaseHubs.probesFrom(d, own, 3));

        // Строки: (метка, признак, набор V, с пулом).
        Object[][] rows = {{"L1", 0, EnumSet.of(V.L1), false}, {"L1-tt", 1, EnumSet.of(V.L1), false}, {"SFace", 2, EnumSet.of(V.SFACE), false},
            {"L1-xs-F", 0, EnumSet.of(V.L1), true}, {"L1-tt-xs-F", 1, EnumSet.of(V.L1), true}};
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        StringBuilder allLines = new StringBuilder();
        for (int sv = 1; sv <= 3; sv++) {
            int cs = CONF_II[sv - 1];
            Map<String, List<List<String>>> fs = XBaseHubs.firstSessions(d, own, sv);
            Map<String, List<String>> gal = XBaseHubs.keys(fs);
            List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + cs);
            for (String mode : sv == 1 ? List.of("") : List.of(" (м)", " (с)")) {
                for (Object[] rw : rows) {
                    String label = (String) rw[0];
                    int f = (Integer) rw[1];
                    @SuppressWarnings("unchecked") Set<V> vs = (Set<V>) rw[2];
                    boolean withPool = (Boolean) rw[3];
                    g.cohort = withPool ? augCohort : cohort;
                    String name = String.format(Locale.ROOT, "s%d%s_%s", sv, mode.isEmpty() ? "" : mode.contains("м") ? "m" : "c", label);
                    String ls = block(blocks, name, mode.contains("с") ? () -> XBaseFeret.fitLinesSessions(g, xs[f], vs, fs, gal, probes, parts, cs)
                            : () -> XBaseEval.fitLines(g, xs[f], vs, gal, probes, parts, cs));
                    allLines.append("# (ii) s").append(sv).append(mode).append(' ').append(label).append('\n').append(ls);
                    V v = vs.iterator().next();
                    XBaseEval.Parsed p = XBaseEval.parse(g, vs, gal, probes, parts, ls);
                    String l = "(ii) s" + sv + mode + " " + label;
                    res.put(l, FeiThreshold.evaluate(l, v.ratioId(), List.of(p.cfg.get(v)), List.of(p.imp.get(v)), g.persons.size(), g.b,
                            g.seed + FeiBootstrap.SEED_OFFSET));
                }
            }
        }
        g.cohort = cohort;
        // Пары – как XBaseHubs.java:523–530.
        List<String[]> pairs = new ArrayList<>();
        for (Object[] rw : rows) {
            String l = " " + rw[0];
            pairs.add(new String[] {"(ii) s2 (м)" + l, "(ii) s1" + l});
            pairs.add(new String[] {"(ii) s2 (с)" + l, "(ii) s1" + l});
            pairs.add(new String[] {"(ii) s3 (м)" + l, "(ii) s2 (м)" + l});
            pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s2 (с)" + l});
            pairs.add(new String[] {"(ii) s3 (м)" + l, "(ii) s1" + l});
            pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s1" + l});
            pairs.add(new String[] {"(ii) s2 (с)" + l, "(ii) s2 (м)" + l});
            pairs.add(new String[] {"(ii) s3 (с)" + l, "(ii) s3 (м)" + l});
        }
        // Порядок строк – по s, затем (м), (с), по строкам.
        Map<String, FeiThreshold.Res> ordered = new LinkedHashMap<>();
        for (int sv = 1; sv <= 3; sv++) {
            for (String mode : sv == 1 ? List.of("") : List.of(" (м)", " (с)")) {
                for (Object[] rw : rows) ordered.put("(ii) s" + sv + mode + " " + rw[0], res.get("(ii) s" + sv + mode + " " + rw[0]));
            }
        }
        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : ordered.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(g.persons.size()), w));
        StringBuilder t = new StringBuilder();
        XBaseOther.tables(t, ordered, sep, g.b, pairs);

        // Воспроизведение: строки L1, L1-tt, SFace – против раздела части 1 (ii) отчёта.
        String rep = Files.readString(oldDir.resolve("xbase_feret_hubs.txt"), StandardCharsets.UTF_8).replace("\r", "");
        int i0 = rep.indexOf("######## Часть 1 (ii)");
        int i1 = rep.indexOf("######## Часть 2");
        Set<String> ref = new HashSet<>(List.of(rep.substring(i0, i1).split("\n")));
        int n = 0;
        int ok = 0;
        List<String> bad = new ArrayList<>();
        for (String l : t.toString().split("\n")) {
            if (!l.startsWith("(ii) ") || l.contains("xs-F")) continue;
            n++;
            if (ref.contains(l)) ok++;
            else if (bad.size() < 20) bad.add(l);
        }
        StringBuilder r = new StringBuilder();
        r.append("Число сессий в галерее: L1-xs-F и L1-tt-xs-F на протоколе части 1 (ii) feret-hubs (Жорик №6 от 21.10.2026) – VerifyHubsXs\n");
        r.append("Код: проект ").append(FarMethods.commit()).append(" (не менялся)\n");
        r.append(String.format(Locale.ROOT, "Своих (≥ 4 дат) %d, проб своих %d, контроль %d людей, люди бутстрепа %d; пул FERET без своих: %d людей, %d снимков; "
                + "когорта FEI 20.%n", own.size(), probes.stream().filter(q -> q.role().equals(FeiScores.OWN)).count(), ro.ctrl.size(), g.persons.size(),
                poolCls.size(), poolKeys.size()));
        r.append("Галерея – первые M дат (правило равных галерей fa/fb), пробы – даты с 4-й; (м) – средний шаблон, (с) – шаблоны по сессиям (как G2-ii).\n");
        r.append("L1-xs-F / L1-tt-xs-F: классы обучения MLDA – галерея своих при данном M, когорта FEI и пул FERET (XBaseXsF.train, Kind.XS); W и 5 моделей\n"
                + "частей обучаются заново при каждом M (как XBaseHubs: блок – свой XBaseEval.fitLines / XBaseFeret.fitLinesSessions).\n");
        if (g.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET));
        r.append(String.format(Locale.ROOT, "%n=== Воспроизведение: строки L1, L1-tt, SFace части 1 (ii) против xbase_feret_hubs.txt – побайтно: %d из %d ===%n", ok, n));
        for (String l : bad) r.append("НЕ СОВПАЛО: ").append(l).append('\n');
        r.append(t);
        XBaseOther.writeScores(out.resolve("verify_hubs_xs_scores.tsv.gz"), "# FERET, VerifyHubsXs; блоки – s, шаблон, строка\n# Код: коммит " + FarMethods.commit()
                + "\n", allLines.toString());
        FarMethods.write(out.resolve("verify_hubs_xs.txt"), r);
        System.out.print(r);
    }
}
