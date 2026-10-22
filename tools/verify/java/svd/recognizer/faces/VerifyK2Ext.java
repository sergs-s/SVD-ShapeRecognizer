package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseMp2.FeretSets;
import svd.recognizer.faces.XBaseMp2.Thr;
import svd.recognizer.storage.SettingsStore;

/**
 * Внешние чужие для порога К2 на FERET (tools/verify, ветка cloud-verify; Жорик №4 от 22.10.2026): люди FERET с одной датой, не
 * входящие ни в одну роль (не свои ≥ 2 дат – пул, не контроль 200). Их снимки оцениваются основной моделью как дополнительные пробы
 * роли «cal» – как XBaseCalib.pass (XBaseCalib.java, CAL_ROLE). Код проекта не меняется.
 * <p>
 * L1-xs-F и L1-tt-xs-F – роли (4а) = feret-xs2 (половины №2, два прохода, межс. и внутри; цикл VerifyMp2Ly / XBaseMp2.feret, схема
 * «база»); SFace – протокол feret (XBaseFeret.run: роль (1), межс. s = 0 и внутри s = 1). Проверка: строки своих, контроля и
 * «чужих» галереи (без cal) – побайтно с блоками xbase_feret_xs2_scores.tsv.gz (строки L1 и +mirror, как XBaseMp2) и
 * xbase_feret_scores.tsv.gz (все строки блока SFace).
 * <p>
 * Аргументы: каталог прежних файлов оценок FERET, выходной каталог. Ключи – как XBaseEval.
 */
public final class VerifyK2Ext {

    private VerifyK2Ext() {
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
        System.out.printf(Locale.ROOT, "VerifyK2Ext: блок %s (%.0f с)%n", name, (System.nanoTime() - t1) / 1e9);
        return l;
    }

    static String withoutCal(String lines) {
        StringBuilder b = new StringBuilder();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            if (l.split("\t", -1)[3].equals(XBaseCalib.CAL_ROLE)) continue;
            b.append(l).append('\n');
        }
        return b.toString();
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        Path old = Paths.get(a[0]);
        Path out = Paths.get(a[1]);
        Path blocks = out.resolve("blocks");
        Files.createDirectories(blocks);
        SettingsStore settings = new SettingsStore();
        Path exportDir = Paths.get(settings.loadFacesExportDir());
        FarExport.Loaded export = FarExport.read(exportDir);
        List<String> selLines = Files.readAllLines(exportDir.resolve(FarExport.FEI_SELECTION), StandardCharsets.UTF_8);
        Map<String, Map<Integer, String>> feiKeys = FeiMethods.selection(selLines, export.data().fei());
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("только eq");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        FeretSets fs = VerifyMp2Ly.sets(su, new Thr(Double.POSITIVE_INFINITY, 0, 0, Double.NaN, 0, 0), true);
        XBaseFeret.Roles ro = fs.ro;
        if (!XBaseMp2.sameRoles(fs)) throw new IllegalStateException("роли не совпали с прежними");

        // Внешние чужие: люди с отобранными снимками, не свои роли (1) (пул) и не контроль.
        Set<String> inRoles = new TreeSet<>(ro.own1);
        inRoles.addAll(ro.ctrl);
        Map<String, List<String>> ext = new TreeMap<>();
        for (Img im : su.imgs) {
            Double r = su.rOf.get(im.key());
            if (r == null || r > tpl.rLimit || inRoles.contains(im.person())) continue;
            ext.computeIfAbsent(im.person(), p -> new ArrayList<>()).add(im.key());
        }
        Map<String, Set<String>> extDates = new TreeMap<>();
        for (Map.Entry<String, List<String>> e : ext.entrySet()) for (String k : e.getValue()) extDates.computeIfAbsent(e.getKey(), p -> new TreeSet<>()).add(su.byKey.get(k).date());
        List<String> extKeys = new ArrayList<>();
        ext.values().forEach(extKeys::addAll);
        java.util.Collections.sort(extKeys);
        StringBuilder r = new StringBuilder();
        r.append("Внешние чужие К2 на FERET (Жорик №4 от 22.10.2026) – VerifyK2Ext\nКод: проект ").append(FarMethods.commit()).append(" (не менялся)\n");
        long multi = extDates.values().stream().filter(s -> s.size() > 1).count();
        r.append(String.format(Locale.ROOT, "Внешние чужие: людей %d (с одной датой: %d), снимков %d; пул (свои роли (1)) %d, контроль %d.%n", ext.size(),
                ext.size() - multi, extKeys.size(), ro.own1.size(), ro.ctrl.size()));
        StringBuilder list = new StringBuilder("# человек\tснимков\n");
        ext.forEach((p, ks) -> list.append(p).append('\t').append(ks.size()).append('\n'));
        Files.writeString(out.resolve("ext_people.tsv"), list.toString(), StandardCharsets.UTF_8);

        // Векторы: снимки ролей, пул, внешние, когорта.
        Set<String> used = new TreeSet<>(extKeys);
        for (Map<String, List<String>> m : List.of(ro.gal1, ro.pro1, ro.gal1in, ro.pro1in, ro.ctrlKeys)) m.values().forEach(used::addAll);
        fs.pool.values().forEach(ds -> ds.forEach(used::addAll));
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String k : used) {
            fr.put(k, FeretData.frame(su.dir, "a", k));
            sf.put(k, FeretData.frame(su.dir, "sface", k));
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        xs[2].sfaceMin = 1;
        xs[2].sfaceVectors(sf, su.sface);
        fr.values().forEach(Mat::release);
        sf.values().forEach(Mat::release);

        StringBuilder lines = new StringBuilder();
        StringBuilder check = new StringBuilder("\n=== Проверка: строки без cal – побайтно с прежними файлами оценок ===\n");

        // L1-xs-F, L1-tt-xs-F – роли (4а), как VerifyMp2Ly (схема «база»).
        List<String> perm = new ArrayList<>(fs.pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(fs.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt baseI = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        Map<String, String> xs2 = XBaseSubsp.blocks(old.resolve("xbase_feret_xs2_scores.tsv.gz"));
        for (XBaseXsF.Conf c : XBaseMp2.confs(ro)) {
            if (c.name().equals("G2")) continue;
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
                    Map<String, List<List<String>>> pp = new TreeMap<>();
                    for (String p : halves.get(1 - ps)) pp.put(p, fs.pool.get(p));
                    List<List<String>> aug = new ArrayList<>(cohort);
                    aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp, Set.of(), Map.of()).extra());
                    XBaseEval.Gt g = XBaseFeret.withOwn(baseI, own);
                    g.cohort = aug;
                    List<Probe> probes = XBaseEval.probes(g, pro);
                    List<Probe> all = new ArrayList<>(probes);
                    for (String k : extKeys) all.add(new Probe(XBaseCalib.CAL_ROLE, su.byKey.get(k).person(), g.img(k), k, -1));
                    List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                    int ff = f;
                    String ls = block(blocks, String.format(Locale.ROOT, "xs_%d_%d_%d", c.s(), f, ps),
                            () -> XBaseEval.fitLines(g, xs[ff], EnumSet.of(V.L1), gal, all, parts, c.s()));
                    String head = "проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + rowLabel;
                    lines.append("# ").append(head).append('\n').append(ls);
                    String o = xs2.get(head);
                    boolean same = o != null && XBaseSubsp.l1Rows(o).equals(XBaseSubsp.l1Rows(withoutCal(ls)));
                    check.append(String.format(Locale.ROOT, "%s: %s (%d строк без cal)%n", head, same ? "совпало" : "НЕ СОВПАЛО", XBaseSubsp.l1Rows(withoutCal(ls)).size()));
                }
            }
        }

        // SFace – протокол feret, роль (1).
        TreeSet<String> ps1 = new TreeSet<>(ro.own1);
        ps1.addAll(ro.ctrl);
        XBaseEval.Gt base1 = XBaseFeret.gt(tpl, new ArrayList<>(ps1), ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        Map<String, String> fer = XBaseSubsp.blocks(old.resolve("xbase_feret_scores.tsv.gz"));
        Object[][] confs1 = {{"межс.", 0, ro.own1, ro.gal1, ro.pro1}, {"внутри", 1, ro.own1in, ro.gal1in, ro.pro1in}};
        for (Object[] c : confs1) {
            String name = (String) c[0];
            int s = (Integer) c[1];
            @SuppressWarnings("unchecked") List<String> own = (List<String>) c[2];
            @SuppressWarnings("unchecked") Map<String, List<String>> gal = (Map<String, List<String>>) c[3];
            @SuppressWarnings("unchecked") Map<String, List<String>> pro = (Map<String, List<String>>) c[4];
            XBaseEval.Gt g = XBaseFeret.withOwn(base1, own);
            List<Probe> probes = XBaseEval.probes(g, pro);
            List<Probe> all = new ArrayList<>(probes);
            for (String k : extKeys) all.add(new Probe(XBaseCalib.CAL_ROLE, su.byKey.get(k).person(), g.img(k), k, -1));
            List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + s);
            String ls = block(blocks, "sface_" + s, () -> XBaseEval.fitLines(g, xs[2], EnumSet.of(V.SFACE), gal, all, parts, s));
            String head = "конфигурация " + name + ", набор SFace";
            lines.append("# ").append(head).append('\n').append(ls);
            String o = fer.get(head);
            boolean same = o != null && VerifyMp2Ly.sortedRows(o).equals(VerifyMp2Ly.sortedRows(withoutCal(ls)));
            check.append(String.format(Locale.ROOT, "%s: %s (%d строк без cal)%n", head, same ? "совпало" : "НЕ СОВПАЛО", withoutCal(ls).split("\n").length));
        }
        r.append(check);
        XBaseOther.writeScores(out.resolve("verify_k2_ext_scores.tsv.gz"), "# FERET, VerifyK2Ext; блоки – как xs2 и feret; роль cal – внешние чужие\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("verify_k2_ext.txt"), r);
        System.out.print(r);
    }
}
