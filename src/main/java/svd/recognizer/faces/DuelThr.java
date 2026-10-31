package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
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
import java.util.concurrent.Callable;
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№1 (дуэль, 30.10.2026; PLAN.md Д21–Д25): фиксированный порог классики (Д) – дуэльная конфигурация duel-eval-24: признак LBP (L1-xs-F –
 * 7×7 (8, 2), L1-tt-xs-F – Тан – Триггс, 9×9 (8, 2)), 1 − cos в W до центров людей галереи, модель X / X+mirror по |r| ≤ r0, решающая оценка
 * d₁/d₂. Способ – как у порога SFace (verify_sface_thr_cafb7fc): строки FERET межс. и внутри (половины №2, два прохода, как feret-xs2), GT межс.
 * и внутри (N = 6 / 3, eq9800), FEI, MUCT, ORL (шаблон – ⌈n/2⌉ снимков по делению Ахмеда; файл деления – src/main/python/duel_thr.py split).
 * <p>
 * Режим duel-thr-vec – одна W на строку (все люди галереи строки + когорта FEI 20 (кроме строки FEI) + пул FERET), векторы 1 − cos каждой
 * пробы до центров всех людей галереи (модель по позе) → файлы &lt;строка&gt;_f&lt;признак&gt;.tsv.gz (FERET – только D:\data\Feret\reports\eq\
 * duel_thr\, прочие – reports/faces/duel/thr/); подвыборки галереи по N – по векторам, без переобучения (duel_thr.py table). Сверка:
 * строки d₁/d₂ по векторам – с XBaseReg.lines (внутри XBaseVec.model), у FERET межс. и GT – ещё с базой LBP файлов оценок feret-poem / gt-poem.
 * Режим duel-thr-b – проверка (Б): W заново на 10 людях подвыборки + когорта + пул, 10 подвыборок N = 10 (seed faces.seed + 10700 + 100 ·
 * строка + k), GT межс. и FERET межс. (проход 1) → b_&lt;строка&gt;_f&lt;признак&gt;.tsv.gz там же.
 *
 * @author ssv
 */
final class DuelThr {

    /** Сдвиг seed подвыборок проверки (Б). */
    static final long B_SEED_OFFSET = 10700;
    static final int B_N = 10;
    static final int B_REPS = 10;

    private DuelThr() {
    }

    /** Блок: строка протокола, проход, Gt (когорта – с пулом), галерея, пробы. */
    record Blk(String row, String file, String title, List<String> own, Map<String, List<String>> gal, List<List<String>> cohort, List<Probe> probes,
               Map<String, Double> rOf) {}

    /** Строки протокола и кадры. */
    static final class Setup {
        List<Blk> blocks = new ArrayList<>();
        Map<String, Mat> fr = new LinkedHashMap<>();
        IlluminationNorm.NormParams params;
        double r0;
        int threads;
        long seed;
        StringBuilder log = new StringBuilder();
        /** Файлы оценок для сверки базы LBP: блок → путь. */
        Map<String, Path> refFile = new HashMap<>();
        Map<String, String> refBlock = new HashMap<>();
    }

    static int threads(SettingsStore settings) {
        String t = System.getenv("FACES_DUELTHR_THREADS");
        return t != null ? Integer.parseInt(t.trim()) : Math.min(5, settings.loadFacesFeiCvThreads());
    }

    // ---------------------------------------------------------------- FERET

    static void feret(Setup u, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseVec.Setup v = XBaseVec.feretSetup(export, dir, feiKeys, selLines, settings, 0);
        XBaseFeret.Roles ro = XBaseFeret.equalize(v.su.ro, v.su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(v.su.ro, v.su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(v.tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        Path ref = v.out.resolve("xbase_feret_poem_scores.tsv.gz");
        for (XBaseVec.Blk b : v.blocks) {
            if (!b.group().equals("межс.")) continue;
            int ps = b.name().endsWith("p0") ? 0 : 1;
            String file = "feret_x_p" + ps;
            u.blocks.add(new Blk("FERET межс.", file, "FERET межс., проход " + (ps + 1), b.g().own, b.gal(), b.g().cohort, b.probes(), v.su.rOf));
            for (int f = 0; f < 2; f++) {
                u.refFile.put(file + "_f" + f, ref);
                u.refBlock.put(file + "_f" + f, String.format(Locale.ROOT, "межс., %s, проход %d, база LBP %s, косинус", XBaseRule.FEATS[f], ps + 1,
                        f == 0 ? "7×7 (8, 2)" : "9×9 (8, 2)"));
            }
            Set<String> mine = halves.get(ps);
            List<String> own = ro.own1in.stream().filter(mine::contains).toList();
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            for (String p : own) {
                gal.put(p, ro.gal1in.get(p));
                pro.put(p, ro.pro1in.get(p));
                for (String k : ro.gal1in.get(p)) if (!v.fr.containsKey(k)) v.fr.put(k, FeretData.frame(v.su.dir, "a", k));
                for (String k : ro.pro1in.get(p)) if (!v.fr.containsKey(k)) v.fr.put(k, FeretData.frame(v.su.dir, "a", k));
            }
            XBaseEval.Gt g = XBaseFeret.withOwn(b.g(), own);
            u.blocks.add(new Blk("FERET внутри", "feret_i_p" + ps, "FERET внутри, проход " + (ps + 1), own, gal, b.g().cohort, XBaseEval.probes(g, pro),
                    v.su.rOf));
        }
        u.fr.putAll(v.fr);
        u.params = v.tpl.params;
        u.r0 = v.tpl.r0;
        u.seed = v.tpl.seed;
    }

    // ---------------------------------------------------------------- GT

    static void gt(Setup u, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        XBaseVec.Setup v = XBaseVec.gtSetup(export, dir, feiKeys, selLines, settings, out, 0);
        Path ref = Paths.get(System.getProperty("user.dir"), "reports", "faces", XBaseEval.DIR, "xbase_gt_poem_scores.tsv.gz");
        String rg = System.getenv("FACES_DUELTHR_GT_REF");
        if (rg != null) ref = Paths.get(rg);
        for (XBaseVec.Blk b : v.blocks) {
            String file = "gt_" + b.name();
            u.blocks.add(new Blk("GT " + b.group(), file, "GT " + b.group(), b.g().own, b.gal(), b.g().cohort, b.probes(), b.g().rOf));
            for (int f = 0; f < 2; f++) {
                u.refFile.put(file + "_f" + f, ref);
                u.refBlock.put(file + "_f" + f, String.format(Locale.ROOT, "%s, %s, база LBP %s, косинус", b.group(), XBaseRule.FEATS[f],
                        f == 0 ? "7×7 (8, 2)" : "9×9 (8, 2)"));
            }
        }
        u.fr.putAll(v.fr);
        u.params = v.tpl.params;
        u.r0 = v.tpl.r0;
        u.seed = v.tpl.seed;
    }

    // ---------------------------------------------------------------- FEI, MUCT, ORL

    /** Деление «шаблон / пробы» (duel_thr.py split): база → человек → [ключи шаблона, ключи проб] (ключ – «человек/имя без расширения»). */
    static Map<String, Map<String, List<List<String>>>> readSplit(Path f) throws IOException {
        Map<String, Map<String, List<List<String>>>> out = new TreeMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("base\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            List<List<String>> tp = out.computeIfAbsent(c[0], x -> new TreeMap<>()).computeIfAbsent(c[1], x -> List.of(new ArrayList<>(), new ArrayList<>()));
            tp.get(c[3].equals("t") ? 0 : 1).add(c[2]);
        }
        return out;
    }

    static String stem(Path file) {
        String n = file.getFileName().toString();
        int i = n.lastIndexOf('.');
        return i < 0 ? n : n.substring(0, i);
    }

    static void other(Setup u, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        other(u, export, dir, feiKeys, selLines, settings, null);
    }

    /** То же только для базы only (fei, muct, orl; null – все три). */
    static void other(Setup u, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                      String only) throws IOException {
        String rs = System.getenv("FACES_DUELTHR_SPLIT");
        if (rs == null) throw new IllegalStateException("duel-thr-vec: задайте FACES_DUELTHR_SPLIT (файл деления duel_thr.py split)");
        Path splitFile = Paths.get(rs);
        Map<String, Map<String, List<List<String>>>> split = readSplit(splitFile);
        u.log.append("Деление FEI, MUCT, ORL: ").append(splitFile).append(", SHA-256 ").append(DuelSet.sha256(splitFile)).append('\n');
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        u.params = su.tpl.params;
        u.r0 = su.tpl.r0;
        u.seed = su.tpl.seed;
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) u.fr.put(k, FeretData.frame(su.dir, "a", k));
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, u.seed);
        List<List<String>> feiCohort = new ArrayList<>();
        Set<String> feiCohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            feiCohort.add(ks);
            feiCohortKeys.addAll(ks);
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        u.fr.putAll(XBaseEval.frames(export, fei, feiCohortKeys));
        Map<String, Double> pose = XBaseEval.poses(dir.resolve("detections.tsv"));
        double rLimit = XBaseEval.poseLimit(selLines);
        String[][] bases = {{"fei", "FEI"}, {"muct", "MUCT"}, {"orl", "ORL"}};
        for (String[] bs : bases) {
            if (only != null && !only.equals(bs[0])) continue;
            Map<String, List<Sample>> data = switch (bs[0]) {
                case "fei" -> export.data().fei();
                case "muct" -> export.data().muct();
                default -> export.data().orl();
            };
            Map<String, Sample> byRel = new HashMap<>();
            for (Map.Entry<String, List<Sample>> e : data.entrySet()) for (Sample s : e.getValue()) byRel.put(s.person() + "/" + stem(s.file()), s);
            Map<String, List<List<String>>> sp = split.get(bs[0]);
            if (sp == null) throw new IllegalStateException("duel-thr-vec: в " + splitFile + " нет базы " + bs[0]);
            Map<String, Double> rOf = new HashMap<>();
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            List<Sample> used = new ArrayList<>();
            int poseOut = 0;
            List<String> fewTpl = new ArrayList<>();
            for (Map.Entry<String, List<List<String>>> e : sp.entrySet()) {
                List<List<String>> kept = List.of(new ArrayList<>(), new ArrayList<>());
                List<Sample> mine = new ArrayList<>();
                for (int t = 0; t < 2; t++) {
                    for (String rel : e.getValue().get(t)) {
                        Sample s = byRel.get(rel);
                        if (s == null) throw new IllegalStateException("duel-thr-vec: нет снимка " + bs[0] + "/" + rel + " в экспорте");
                        Double r = pose.get(bs[0] + "/" + rel);
                        if (r == null) throw new IllegalStateException("duel-thr-vec: нет позы " + bs[0] + "/" + rel);
                        if (Math.abs(r) > rLimit) {
                            poseOut++;
                            continue;
                        }
                        String k = s.file().toString();
                        rOf.put(k, Math.abs(r));
                        kept.get(t).add(k);
                        mine.add(s);
                    }
                }
                if (kept.get(0).size() < 2) {
                    fewTpl.add(e.getKey() + " (" + kept.get(0).size() + ")");
                    continue;
                }
                gal.put(e.getKey(), kept.get(0));
                pro.put(e.getKey(), kept.get(1));
                used.addAll(mine);
            }
            u.fr.putAll(XBaseEval.frames(export, used, null));
            List<String> own = new ArrayList<>(gal.keySet());
            List<Probe> probes = new ArrayList<>();
            int img = 0;
            for (int i = 0; i < own.size(); i++) for (String k : pro.get(own.get(i))) probes.add(new Probe(FeiScores.OWN, own.get(i), ++img, k, i));
            List<List<String>> cohort = new ArrayList<>(bs[0].equals("fei") ? List.of() : feiCohort);
            cohort.addAll(poolCls);
            u.blocks.add(new Blk(bs[1], bs[0], bs[1], own, gal, cohort, probes, rOf));
            u.log.append(String.format(Locale.ROOT, "%s: людей галереи %d, проб %d; снимков вне отбора |r| ≤ R (%.3f) – %d; шаблон < 2 снимков после отбора – %s%n",
                    bs[1], own.size(), probes.size(), rLimit, poseOut, fewTpl.isEmpty() ? "нет" : String.join(", ", fewTpl)));
        }
    }

    // ---------------------------------------------------------------- векторы

    /** Модели X и X+mirror блока на галерее own (W – галерея own + когорта блока). */
    static List<Callable<XBaseVec.Model>> jobs(XBaseEval x, Blk b, List<String> own, List<Probe> probes, String task) {
        List<List<String>> gallery = own.stream().map(b.gal()::get).toList();
        List<Callable<XBaseVec.Model>> out = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            out.add(() -> XBaseVec.model(x, m, method, 0, own, gallery, b.cohort(), List.of(), List.of(), probes, task));
        }
        return out;
    }

    /** Запись строк проб: модель по позе (|r| ≤ r0 – X, иначе X+mirror). */
    static void write(Writer w, String title, List<XBaseVec.Model> ms, Map<String, Double> rOf, double r0) throws IOException {
        XBaseVec.Model mx = ms.get(0);
        XBaseVec.Model mm = ms.get(1);
        w.write("#S\t" + title + "\n");
        w.write("#G\t" + String.join("\t", mx.gallery()) + "\n");
        for (int i = 0; i < mx.rows().size(); i++) {
            XBaseVec.Row a = mx.rows().get(i);
            Double r = rOf.get(a.key());
            if (r == null) throw new IllegalStateException("duel-thr: нет позы пробы " + a.key());
            boolean frontal = r <= r0;
            XBaseVec.Row c = frontal ? a : mm.rows().get(i);
            if (!c.key().equals(a.key())) throw new IllegalStateException("duel-thr: порядок проб моделей X и X+mirror разошёлся");
            StringBuilder b = new StringBuilder("R\t").append(c.role()).append('\t').append(c.person()).append('\t').append(c.key()).append('\t')
                    .append(frontal ? "X" : "XM");
            for (double v : c.dg()) b.append('\t').append(XBaseVec.num(v));
            w.write(b.append('\n').toString());
        }
    }

    /** Сверка строк основной модели (own, ctrl) с блоком файла оценок; номер конфигурации s не сравнивается. */
    static String check(String key, String lines, Setup u) throws IOException {
        Path ref = u.refFile.get(key);
        if (ref == null) return "";
        if (!Files.exists(ref)) return key + ": файла сверки нет – " + ref + "\n";
        String bn = u.refBlock.get(key);
        String old = XBaseSubsp.blocks(ref).get(bn);
        if (old == null) throw new IllegalStateException("duel-thr: нет блока «" + bn + "» в " + ref);
        List<String> a = norm(old);
        List<String> b = norm(lines);
        if (!a.equals(b)) throw new IllegalStateException("duel-thr: строки d₁/d₂ не совпали с " + ref + ": " + bn + " (" + a.size() + " / " + b.size() + ")");
        return String.format(Locale.ROOT, "%s: совпало с «%s» (%s), %d строк%n", key, bn, ref.getFileName(), a.size());
    }

    static List<String> norm(String lines) {
        List<String> out = new ArrayList<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (!c[3].equals(FeiScores.OWN) && !c[3].equals(FeiScores.CTRL)) continue;
            c[1] = "*";
            out.add(String.join("\t", c));
        }
        Collections.sort(out);
        return out;
    }

    static Path outDir(String base, Path out) {
        return base.equals("feret") ? Paths.get(System.getenv("FACES_FERET_DIR"), "reports", "eq", "duel_thr") : out.getParent().resolve("duel").resolve("thr");
    }

    static Setup setup(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out) throws IOException {
        Setup u = new Setup();
        u.threads = threads(settings);
        switch (base) {
            case "feret" -> feret(u, export, dir, feiKeys, selLines, settings);
            case "gt" -> gt(u, export, dir, feiKeys, selLines, settings, out);
            case "other" -> other(u, export, dir, feiKeys, selLines, settings);
            default -> throw new IllegalArgumentException("duel-thr: база feret | gt | other, а не " + base);
        }
        return u;
    }

    /** Режим duel-thr-vec &lt;feret | gt | other&gt;. */
    static void vec(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        long t0 = System.nanoTime();
        Setup u = setup(base, export, dir, feiKeys, selLines, settings, out);
        Path vd = outDir(base, out);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        StringBuilder rep = new StringBuilder("Векторы 1 − cos, дуэльная конфигурация (Д) – XBaseEval duel-thr-vec " + base + "\n");
        rep.append("Код: коммит ").append(commit).append(" (с правками рабочей копии)\nДанные: ").append(FarMethods.dataCommit(dir)).append('\n');
        rep.append(String.format(Locale.ROOT, "r0 = %s; потоков %d; каталог %s%n", u.r0, u.threads, vd));
        rep.append(u.log);
        for (int f = 1; f >= 0; f--) {
            long tv = System.nanoTime();
            XBaseEval x = XBasePoem.vectors(u.fr, u.params, f, new int[] {0})[0];
            System.out.printf(Locale.ROOT, "duel-thr-vec %s: f%d vectors (%.0f s)%n", base, f, (System.nanoTime() - tv) / 1e9);
            for (Blk b : u.blocks) {
                String key = b.file() + "_f" + f;
                Path p = vd.resolve(key + ".tsv.gz");
                if (Files.exists(p)) {
                    rep.append(key).append(": из каталога (контрольная точка)\n");
                    continue;
                }
                long ts = System.nanoTime();
                List<XBaseVec.Model> ms = XBaseVec.run(List.of(jobs(x, b, b.own(), b.probes(), "main")), u.threads).get(0);
                rep.append(check(key, XBaseVec.lines(ms), u));
                Writer w = XBaseVec.writer(p, "# DuelThr duel-thr-vec: " + b.title() + ", " + XBaseRule.FEATS[f] + ", LBP " + (f == 0 ? "7×7" : "9×9")
                        + " (8, 2), 1 − cos в W, модель по позе (r0 = " + u.r0 + "), людей галереи " + b.own().size() + ", проб " + b.probes().size()
                        + "\n# Код: коммит " + commit + "\n");
                write(w, b.title(), ms, b.rOf(), u.r0);
                XBaseVec.finish(w, p);
                rep.append(String.format(Locale.ROOT, "%s: %s, людей %d, проб %d, %.0f с%n", key, b.title(), b.own().size(), b.probes().size(),
                        (System.nanoTime() - ts) / 1e9));
                System.out.printf(Locale.ROOT, "duel-thr-vec %s: %s ready (%.0f s)%n", base, key, (System.nanoTime() - ts) / 1e9);
            }
        }
        for (Mat m : u.fr.values()) m.release();
        rep.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(vd.resolve("duel_thr_vec_" + base + ".txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    /** Режим duel-thr-b &lt;feret | gt&gt;: проверка (Б) – W заново на 10 людях подвыборки. */
    static void checkB(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out) throws IOException {
        long t0 = System.nanoTime();
        Setup u = setup(base, export, dir, feiKeys, selLines, settings, out);
        Blk b = u.blocks.stream().filter(x -> x.file().equals(base.equals("feret") ? "feret_x_p0" : "gt_x")).findFirst().orElseThrow();
        int rowNo = base.equals("feret") ? 0 : 1;
        Path vd = outDir(base, out);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        List<List<String>> subs = new ArrayList<>();
        for (int k = 0; k < B_REPS; k++) {
            List<String> perm = new ArrayList<>(b.own());
            Collections.shuffle(perm, new Random(u.seed + B_SEED_OFFSET + 100L * rowNo + k));
            List<String> g = new ArrayList<>(perm.subList(0, B_N));
            Collections.sort(g);
            subs.add(g);
        }
        StringBuilder rep = new StringBuilder("Проверка (Б) – W заново на " + B_N + " людях, XBaseEval duel-thr-b " + base + "\nКод: коммит " + commit
                + " (с правками рабочей копии)\n");
        for (int f = 1; f >= 0; f--) {
            Path p = vd.resolve("b_" + b.file() + "_f" + f + ".tsv.gz");
            if (Files.exists(p)) {
                rep.append(p.getFileName()).append(": из каталога\n");
                continue;
            }
            XBaseEval x = XBasePoem.vectors(u.fr, u.params, f, new int[] {0})[0];
            List<List<Callable<XBaseVec.Model>>> js = new ArrayList<>();
            for (List<String> g : subs) {
                Map<String, Integer> idx = new HashMap<>();
                for (int i = 0; i < g.size(); i++) idx.put(g.get(i), i);
                List<Probe> pr = new ArrayList<>();
                for (Probe q : b.probes()) {
                    if (q.role().equals(FeiScores.CTRL)) pr.add(q);
                    else if (idx.containsKey(q.person())) pr.add(new Probe(q.role(), q.person(), q.image(), q.key(), idx.get(q.person())));
                }
                js.add(jobs(x, b, g, pr, "b"));
            }
            long ts = System.nanoTime();
            List<List<XBaseVec.Model>> done = XBaseVec.run(js, u.threads);
            Writer w = XBaseVec.writer(p, "# DuelThr duel-thr-b: " + b.title() + ", " + XBaseRule.FEATS[f] + ", W заново на " + B_N + " людях, " + B_REPS
                    + " подвыборок (seed faces.seed + " + B_SEED_OFFSET + " + 100 · " + rowNo + " + k)\n# Код: коммит " + commit + "\n");
            for (int k = 0; k < subs.size(); k++) write(w, b.title() + ", подвыборка " + k, done.get(k), b.rOf(), u.r0);
            XBaseVec.finish(w, p);
            rep.append(String.format(Locale.ROOT, "%s: %d подвыборок, %.0f с%n", p.getFileName(), subs.size(), (System.nanoTime() - ts) / 1e9));
        }
        for (Mat m : u.fr.values()) m.release();
        rep.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(vd.resolve("duel_thr_b_" + base + ".txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- (Н): признак (П2) + MS-LBP (решение (32)), слияние (31), β = 1 (30)

    /** Деление людей вне W на (П) – приведение ℓ и (З) – замер FAR (ответ Жорика (дуэль) на №3): seed faces.seed + 10710 + номер строки. */
    static final long N_SPLIT_OFFSET = 10710;
    static final int V8 = 8;
    /** Номера строк (Н): FERET межс. 0, внутри 1, GT межс. 2, внутри 3, FEI 4, MUCT 5, ORL 6; людей в W / (П) у FEI, MUCT, ORL. */
    static final Map<String, int[]> N_SPLIT = Map.of("fei", new int[] {4, 120, 40}, "muct", new int[] {5, 200, 37}, "orl", new int[] {6, 20, 10});

    /** Деление списка людей (по возрастанию) перестановкой seed: [W, (П), (З)]; nW &lt; 0 – W нет (контрольные пополам). */
    static List<List<String>> splitPersons(List<String> persons, long seed, int nW, int nP) {
        List<String> perm = new ArrayList<>(new TreeSet<>(persons));
        Collections.shuffle(perm, new Random(seed));
        int w = Math.max(0, nW);
        int p = nW < 0 ? perm.size() / 2 : nP;
        List<List<String>> out = new ArrayList<>();
        for (int[] r : new int[][] {{0, w}, {w, w + p}, {w + p, perm.size()}}) {
            List<String> l = new ArrayList<>(perm.subList(r[0], r[1]));
            Collections.sort(l);
            out.add(l);
        }
        return out;
    }

    /** Признаки (Н): кадры k = 2 – готовые; BSIF не нужен (вариант 8 – (П2) + MS-LBP). */
    static XBaseFeat2.Feats nFeats(IlluminationNorm.NormParams params, Map<String, FrameK2.F> k2, int threads) {
        XBaseFeat2.Feats fs = new XBaseFeat2.Feats();
        fs.params = params;
        fs.paramsK2 = XBaseFeat2.scaledTt(params, FrameK2.K);
        fs.threads = threads;
        fs.k2 = k2;
        return fs;
    }

    /**
     * Кадр k = 2 и точки по исходному снимку базы (шаги FrameK2.load для ключей экспорта: детекция экспорта, FrameK2.fromFull по серому
     * исходному снимку, сверка кадра с кадром «в» экспорта – не совпал – остановка); MUCT – &lt;faces.muct.dir&gt;/&lt;имя&gt;.jpg, ORL – s&lt;N&gt;/&lt;имя&gt;.pgm,
     * GT и FEI – как FrameK2 (для сверки с FrameK2.load).
     */
    static Map<String, FrameK2.F> k2Raw(String base, List<Sample> samples, FarExport.Loaded export, Path exportDir, Path rawDir) throws IOException {
        Map<String, double[]> rows = FrameK2.exportRows(exportDir);
        Map<String, Path> feiRaw = base.equals("fei") ? FrameK2.feiRaw(rawDir) : null;
        Map<String, FrameK2.F> out = new LinkedHashMap<>();
        for (Sample s : samples) {
            String k = s.file().toString();
            double[] row = rows.get(k);
            if (row == null) throw new IllegalStateException("duel-thr-n: нет детекции экспорта " + k);
            Path p = Paths.get(k);
            String person = p.getName(1).toString();
            String name = p.getName(2).toString();
            Path raw = switch (base) {
                case "muct" -> rawDir.resolve(name + ".jpg");
                case "orl" -> rawDir.resolve("s" + Integer.parseInt(person.substring(1))).resolve(name + ".pgm");
                case "gt" -> rawDir.resolve(person).resolve(name + ".jpg");
                default -> feiRaw.get(name);
            };
            if (raw == null || !Files.exists(raw)) throw new IllegalStateException("duel-thr-n: нет исходного снимка " + k + " (" + raw + ")");
            Mat color = org.opencv.imgcodecs.Imgcodecs.imdecode(new org.opencv.core.MatOfByte(Files.readAllBytes(raw)), org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR);
            Mat gray = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(color, gray, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);
            color.release();
            FrameK2.F f = FrameK2.fromFull(gray, FrameK2.lm(row));
            gray.release();
            Mat e = export.frame(s, GalleryEvaluation.Variant.C);
            if (e == null || !FrameK2.same(e, f.frame())) throw new IllegalStateException("duel-thr-n: кадр k = 2 из исходного снимка ≠ кадр «в» экспорта: " + k);
            f.frame().release();
            out.put(k, new FrameK2.F(e, f.pts()));
        }
        return out;
    }

    /** Режим duel-thr-n-k2check: точки k2Raw на GT и FEI побайтно = FrameK2.load; кадры MUCT и ORL = «в» экспорта (внутри k2Raw). */
    static void k2check(FarExport.Loaded export, Path dir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        StringBuilder rep = new StringBuilder("Сверка кадров k = 2 и точек для (Н) – XBaseEval duel-thr-n-k2check\nКод: коммит " + FarMethods.commit()
                + " (с правками рабочей копии)\n");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Path feiDir = Paths.get(settings.loadFacesFeiDir());
        for (String base : List.of("gt", "fei")) {
            List<Sample> ss = new ArrayList<>();
            (base.equals("gt") ? export.data().gt() : export.data().fei()).values().forEach(ss::addAll);
            Map<String, FrameK2.F> mine = k2Raw(base, ss, export, dir, base.equals("gt") ? gtDir : feiDir);
            Map<String, FrameK2.F> ref = FrameK2.load(mine.keySet(), su, su.dir, export, dir, gtDir, feiDir);
            int bad = 0;
            for (Map.Entry<String, FrameK2.F> e : mine.entrySet()) {
                FrameK2.F r = ref.get(e.getKey());
                boolean eq = FrameK2.same(r.frame(), e.getValue().frame());
                for (int j = 0; j < 5 && eq; j++) for (int c = 0; c < 2; c++) eq &= Double.doubleToLongBits(r.pts()[j][c]) == Double.doubleToLongBits(e.getValue().pts()[j][c]);
                if (!eq) bad++;
            }
            mine.values().forEach(f -> f.frame().release());
            ref.values().forEach(f -> f.frame().release());
            rep.append(String.format(Locale.ROOT, "%s: снимков %d, кадр и точки побайтно = FrameK2.load – %s%n", base, mine.size(), bad == 0 ? "да" : "НЕТ у " + bad));
            if (bad > 0) throw new IllegalStateException("duel-thr-n-k2check: " + base + " – расхождение с FrameK2.load у " + bad + " снимков\n" + rep);
        }
        for (String base : List.of("muct", "orl")) {
            List<Sample> ss = new ArrayList<>();
            (base.equals("muct") ? export.data().muct() : export.data().orl()).values().forEach(ss::addAll);
            Path raw = Paths.get(base.equals("muct") ? settings.loadFacesMuctDir() : settings.loadFacesDatasetDir());
            Map<String, FrameK2.F> mine = k2Raw(base, ss, export, dir, raw);
            mine.values().forEach(f -> f.frame().release());
            rep.append(String.format(Locale.ROOT, "%s: снимков %d, кадр k = 2 из исходного снимка = кадр «в» экспорта у всех%n", base, mine.size()));
        }
        rep.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        Path o = out.getParent().resolve("duel").resolve("thr_n");
        Files.createDirectories(o);
        FarMethods.write(o.resolve("duel_thr_n_k2check.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    /** Блок (Н): строка, файл, подпись, номер строки, галерея (W), когорта, пробы, поза, люди (П) и (З). */
    record NB(String file, String title, int rowNo, List<String> own, Map<String, List<String>> gal, List<List<String>> cohort, List<Probe> probes,
              Map<String, Double> rOf, Set<String> pp, Set<String> zp) {}

    /** Строки (Н): у L1-tt-xs-F (f = 1) – модель по позе, у L1-xs-F (f = 0) – всегда X+mirror; роль own / P / Z. */
    static void writeN(Writer w, String title, XBaseVec.Model mx, XBaseVec.Model mm, int f, Map<String, Double> rOf, double r0, Set<String> pp, Set<String> zp,
                       double[] t2) throws IOException {
        w.write("#S\t" + title + "\n");
        if (t2 != null) w.write("#T2\t" + XBaseVec.num(t2[0]) + "\t" + XBaseVec.num(t2[1]) + "\n");
        w.write("#G\t" + String.join("\t", mm.gallery()) + "\n");
        for (int i = 0; i < mm.rows().size(); i++) {
            XBaseVec.Row b = mm.rows().get(i);
            XBaseVec.Row c = b;
            String model = "XM";
            if (f == 1) {
                XBaseVec.Row a = mx.rows().get(i);
                if (!a.key().equals(b.key())) throw new IllegalStateException("duel-thr-n: порядок проб моделей X и X+mirror разошёлся");
                Double r = rOf.get(a.key());
                if (r == null) throw new IllegalStateException("duel-thr-n: нет позы пробы " + a.key());
                if (r <= r0) {
                    c = a;
                    model = "X";
                }
            }
            String role = c.trueIdx() >= 0 ? FeiScores.OWN : pp.contains(c.person()) ? "P" : zp.contains(c.person()) ? "Z" : null;
            if (role == null) throw new IllegalStateException("duel-thr-n: чужой " + c.person() + " – ни в (П), ни в (З)");
            StringBuilder s = new StringBuilder("R\t").append(role).append('\t').append(c.person()).append('\t').append(c.key()).append('\t').append(model);
            for (double v : c.dg()) s.append('\t').append(XBaseVec.num(v));
            w.write(s.append('\n').toString());
        }
    }

    /** Модели блока (Н) по признаку f: L1-tt-xs-F – X и X+mirror, L1-xs-F – только X+mirror (вместо X – null). */
    static List<Callable<XBaseVec.Model>> nJobs(XBaseEval x, int f, NB b, List<String> own, List<Probe> probes, String task) {
        List<List<String>> gallery = own.stream().map(b.gal()::get).toList();
        List<Callable<XBaseVec.Model>> out = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            if (f == 0 && !m) {
                out.add(() -> null);
                continue;
            }
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            out.add(FAST ? () -> nModelFast(x, m, method, own, gallery, b.cohort(), probes, task)
                    : () -> XBaseVec.model(x, m, method, 0, own, gallery, b.cohort(), List.of(), List.of(), probes, task));
        }
        return out;
    }

    /** FACES_DUELTHR_FAST=1 – обучение W через MldaFast (LAPACK; разрешение Хозяина 30.10, вариант (в)), векторы – в каталог …_fast. */
    static final boolean FAST = "1".equals(System.getenv("FACES_DUELTHR_FAST"));

    /** Как XBaseVec.model (без когорт приведения), обучение – MldaFast вместо MldaScorer; сверка строк с XBaseReg.lines – та же. */
    static XBaseVec.Model nModelFast(XBaseEval x, boolean mirror, String method, List<String> own, List<List<String>> gallery, List<List<String>> cohort,
                                     List<Probe> probes, String task) {
        List<List<double[]>> cls = new ArrayList<>();
        for (List<List<String>> group : List.of(gallery, cohort)) {
            for (List<String> keys : group) {
                List<double[]> list = new ArrayList<>();
                for (String k : keys) list.add(x.vec.get(V.L1).get(k));
                if (mirror) for (String k : keys) list.add(x.mir.get(V.L1).get(k));
                cls.add(list);
            }
        }
        LdaModel lm = MldaFast.fit(cls, gallery.size(), 1.0);
        double[][] c = lm.centers();
        List<XBaseVec.Row> rows = new ArrayList<>();
        StringBuilder mine = new StringBuilder();
        for (Probe p : probes) {
            double[] y = lm.embed(x.vec.get(V.L1).get(p.key()));
            double[] dg = new double[c.length];
            for (int g = 0; g < c.length; g++) dg[g] = c[g] == null ? Double.POSITIVE_INFINITY : XBaseVec.cos(y, c[g]);
            XBaseVec.Row r = new XBaseVec.Row(p.role(), p.person(), p.image(), p.key(), p.trueIdx(), dg, new double[0], new double[0]);
            rows.add(r);
            mine.append(XBaseVec.line(method, 0, own, r));
        }
        String ref = XBaseReg.lines(x, method, 0, lm, own, probes)[1];
        if (!ref.equals(mine.toString())) throw new IllegalStateException("DuelThr: строки по векторам ≠ XBaseReg.lines (" + task + ", " + method + ")");
        return new XBaseVec.Model(task, method, 0, own, rows, new double[c.length][]);
    }

    /**
     * Переобучение W на подвыборках (решение Жорика (дуэль) / основного Жорика №10, 30.10): FACES_DUELTHR_SUBS – файл подвыборок
     * (duel_thr.py nsubs: «файл блока, номер, люди через запятую»; те же подвыборки, что у общей W в ntable), FACES_DUELTHR_SUBS_TAG – префикс
     * выходных файлов (r10, r50), FACES_DUELTHR_RTHREADS – подвыборок параллельно (по умолчанию 3).
     */
    static final String SUBS = System.getenv("FACES_DUELTHR_SUBS");
    private static Map<String, List<List<String>>> subsMap;

    static synchronized List<List<String>> subs(String file) throws IOException {
        if (subsMap == null) {
            subsMap = new LinkedHashMap<>();
            for (String line : Files.readAllLines(Paths.get(SUBS), StandardCharsets.UTF_8)) {
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] c = line.split("\t");
                List<List<String>> l = subsMap.computeIfAbsent(c[0], k -> new ArrayList<>());
                if (Integer.parseInt(c[1]) != l.size()) throw new IllegalStateException("duel-thr-n: номера подвыборок в " + SUBS + " не по порядку: " + line);
                l.add(List.of(c[2].split(",")));
            }
        }
        return subsMap.get(file);
    }

    /** Подвыборки блока b: W заново (подвыборка + пул + когорта), по одному файлу на подвыборку в каталоге частей, затем склейка. */
    static String retrain(XBaseEval x, int f, NB b, double r0, String commit, Path vd) throws IOException {
        String tag = System.getenv("FACES_DUELTHR_SUBS_TAG");
        if (tag == null) throw new IllegalStateException("duel-thr-n: задайте FACES_DUELTHR_SUBS_TAG (r10, r50)");
        String rt = System.getenv("FACES_DUELTHR_RTHREADS");
        int par = rt == null ? 3 : Integer.parseInt(rt.trim());
        List<List<String>> ss = subs(b.file());
        Path p = vd.resolve(tag + "_" + b.file() + "_f" + f + "_n.tsv.gz");
        if (Files.exists(p)) return p.getFileName() + ": из каталога\n";
        Path parts = vd.resolve(tag + "_" + b.file() + "_f" + f + "_parts");
        Files.createDirectories(parts);
        long ts = System.nanoTime();
        for (int k0 = 0; k0 < ss.size(); k0 += par) {
            List<Integer> ks = new ArrayList<>();
            List<List<Callable<XBaseVec.Model>>> js = new ArrayList<>();
            for (int k = k0; k < Math.min(ss.size(), k0 + par); k++) {
                if (Files.exists(parts.resolve(String.format(Locale.ROOT, "k%03d.tsv.gz", k)))) continue;
                List<String> g = new ArrayList<>(ss.get(k));
                for (String q : g) if (!b.gal().containsKey(q)) throw new IllegalStateException("duel-thr-n: " + q + " подвыборки " + k + " нет в галерее " + b.file());
                Map<String, Integer> idx = new HashMap<>();
                for (int i = 0; i < g.size(); i++) idx.put(g.get(i), i);
                List<Probe> pr = new ArrayList<>();
                for (Probe q : b.probes()) {
                    if (q.trueIdx() < 0) pr.add(q);
                    else if (idx.containsKey(q.person())) pr.add(new Probe(q.role(), q.person(), q.image(), q.key(), idx.get(q.person())));
                }
                ks.add(k);
                js.add(nJobs(x, f, b, g, pr, tag));
            }
            if (js.isEmpty()) continue;
            List<List<XBaseVec.Model>> done = XBaseVec.run(js, par);
            for (int i = 0; i < ks.size(); i++) {
                Path pp = parts.resolve(String.format(Locale.ROOT, "k%03d.tsv.gz", ks.get(i)));
                Writer w = XBaseVec.writer(pp, "# part\n");
                writeN(w, b.title() + ", подвыборка " + ks.get(i), done.get(i).get(0), done.get(i).get(1), f, b.rOf(), r0, b.pp(), b.zp(), null);
                XBaseVec.finish(w, pp);
            }
            System.out.printf(Locale.ROOT, "duel-thr-n %s f%d: подвыборок %d из %d (%.0f s)%n", b.file(), f, Math.min(ss.size(), k0 + par), ss.size(),
                    (System.nanoTime() - ts) / 1e9);
        }
        Writer w = XBaseVec.writer(p, "# DuelThr duel-thr-n (переобучение " + tag + "): " + b.title() + ", " + XBaseRule.FEATS[f] + ", (П2) + MS-LBP (v8), W заново "
                + "на подвыборке + пул + когорта, " + (FAST ? "MldaFast (LAPACK)" : "MldaScorer") + ", подвыборок " + ss.size() + " (файл " + SUBS + ")\n# Код: коммит "
                + commit + "\n");
        for (int k = 0; k < ss.size(); k++) {
            Path pp = parts.resolve(String.format(Locale.ROOT, "k%03d.tsv.gz", k));
            try (BufferedReader r = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(pp)), StandardCharsets.UTF_8))) {
                for (String line; (line = r.readLine()) != null; ) if (!line.startsWith("# ")) w.write(line + "\n");
            }
        }
        XBaseVec.finish(w, p);
        for (int k = 0; k < ss.size(); k++) Files.delete(parts.resolve(String.format(Locale.ROOT, "k%03d.tsv.gz", k)));
        Files.delete(parts);
        return String.format(Locale.ROOT, "%s: %s, подвыборок %d, %.0f с%n", p.getFileName(), b.title(), ss.size(), (System.nanoTime() - ts) / 1e9);
    }

    static Path nDir(String base, Path out) {
        String sfx = FAST && SUBS == null ? "_fast" : "";
        return base.equals("feret") ? Paths.get(System.getenv("FACES_FERET_DIR"), "reports", "eq", "duel_thr_n" + sfx)
                : out.getParent().resolve("duel").resolve("thr_n" + sfx);
    }

    /** Контрольные: (П) и (З) пополам (seed faces.seed + 10710 + номер строки). */
    static List<List<String>> ctrlSplit(List<Probe> probes, long seed, int rowNo) {
        List<String> ctrl = probes.stream().filter(q -> q.trueIdx() < 0).map(Probe::person).distinct().toList();
        return splitPersons(ctrl, seed + N_SPLIT_OFFSET + rowNo, -1, 0);
    }

    /** Перенос векторов Васи v8 (main X / X+mirror) в формат (Н) и (Т2) по моделям без части (как XBaseVec.fuseStats). */
    static String importV8(Path src, Path dst, String title, int f, Map<String, Double> rOf, double r0, List<List<String>> pz, String commit) throws IOException {
        if (Files.exists(dst)) return dst.getFileName() + ": из каталога\n";
        Map<String, List<XBaseVec.Model>> bl = XBaseVec.read(src);
        if (bl.size() != 1) throw new IllegalStateException("duel-thr-n: в " + src + " блоков " + bl.size());
        List<XBaseVec.Model> ms = bl.values().iterator().next();
        double[] t2 = XBaseVec.fuseStats(ms, false);
        XBaseVec.Model mx = XBaseVec.find(ms, "main", false);
        XBaseVec.Model mm = XBaseVec.find(ms, "main", true);
        Writer w = XBaseVec.writer(dst, "# DuelThr duel-thr-n: " + title + ", " + XBaseRule.FEATS[f] + ", (П2) + MS-LBP (v8) – векторы Васи " + src.getFileName()
                + " (SHA-256 " + DuelSet.sha256(src) + "), (Т2) – XBaseVec.fuseStats\n# Код: коммит " + commit + "\n");
        writeN(w, title, mx, mm, f, rOf, r0, new HashSet<>(pz.get(1)), new HashSet<>(pz.get(2)), t2);
        XBaseVec.finish(w, dst);
        return String.format(Locale.ROOT, "%s ← %s (SHA-256 %s); (Т2) μ = %.6f, σ = %.6f%n", dst.getFileName(), src, DuelSet.sha256(src), t2[0], t2[1]);
    }

    /** Режим duel-thr-n &lt;feret | gt | fei | muct | orl&gt;. */
    static void nVec(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        long t0 = System.nanoTime();
        Path vd = nDir(base, out);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        StringBuilder rep = new StringBuilder("Векторы (Н) – XBaseEval duel-thr-n " + base + "\nКод: коммит " + commit + " (с правками рабочей копии)\nДанные: "
                + FarMethods.dataCommit(dir) + "\n");
        List<NB> todo = new ArrayList<>();
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, FrameK2.F> k2 = new LinkedHashMap<>();
        IlluminationNorm.NormParams params;
        double r0;
        int threads = threads(settings);
        long seed;
        if (base.equals("feret") || base.equals("gt")) {
            Setup u = setup(base, export, dir, feiKeys, selLines, settings, out);
            params = u.params;
            r0 = u.r0;
            seed = u.seed;
            String rv = System.getenv("FACES_DUELTHR_V8_" + base.toUpperCase(Locale.ROOT));
            if (rv == null) throw new IllegalStateException("duel-thr-n: задайте FACES_DUELTHR_V8_" + base.toUpperCase(Locale.ROOT) + " (каталог векторов Васи v8)");
            Path v8 = Paths.get(rv);
            for (Blk b : u.blocks) {
                int rowNo = b.file().startsWith("feret_x") ? 0 : b.file().startsWith("feret_i") ? 1 : b.file().equals("gt_x") ? 2 : 3;
                List<List<String>> pz = ctrlSplit(b.probes(), seed, rowNo);
                if (SUBS != null) {
                    // Переобучение: только блоки из файла подвыборок (FERET межс. – оба прохода, GT – межс.).
                    if (subs(b.file()) != null) todo.add(new NB(b.file(), b.title(), rowNo, b.own(), b.gal(), b.cohort(), b.probes(), b.rOf(), new HashSet<>(pz.get(1)),
                            new HashSet<>(pz.get(2))));
                    continue;
                }
                if (rowNo != 1) {
                    String src = base.equals("gt") ? b.file().substring(3) : "c0_" + b.file().substring(8);
                    for (int f = 1; f >= 0; f--) rep.append(importV8(v8.resolve(src + "_f" + f + "_v8.tsv.gz"), vd.resolve(b.file() + "_f" + f + "_n.tsv.gz"), b.title(), f,
                            b.rOf(), r0, pz, commit));
                    continue;
                }
                todo.add(new NB(b.file(), b.title(), rowNo, b.own(), b.gal(), b.cohort(), b.probes(), b.rOf(), new HashSet<>(pz.get(1)), new HashSet<>(pz.get(2))));
                rep.append(String.format(Locale.ROOT, "%s: (П) %d человек, (З) %d человек%n", b.title(), pz.get(1).size(), pz.get(2).size()));
            }
            if (!todo.isEmpty()) {
                Set<String> need = new TreeSet<>();
                for (NB b : todo) {
                    b.gal().values().forEach(need::addAll);
                    b.cohort().forEach(need::addAll);
                    b.probes().forEach(q -> need.add(q.key()));
                }
                for (String k : need) fr.put(k, u.fr.get(k));
                for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (!need.contains(e.getKey())) e.getValue().release();
                Map<String, double[][]> cache = FrameK2.readPoints(Paths.get(System.getenv("FACES_FERET_DIR")).resolve(FeretData.DERIVED).resolve(FrameK2.POINTS));
                List<String> miss = need.stream().filter(k -> !k.contains("\\") && !k.contains("/") && !cache.containsKey(k)).toList();
                if (!miss.isEmpty()) throw new IllegalStateException("duel-thr-n: кадров k = 2 FERET нет в кэше derived (" + miss.size() + ", например " + miss.get(0)
                        + ") – FrameK2 дописал бы общий кэш; остановка");
                XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
                k2.putAll(FrameK2.load(need, su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir())));
            } else {
                for (Mat m : u.fr.values()) m.release();
            }
        } else {
            int[] sp = N_SPLIT.get(base);
            if (sp == null) throw new IllegalArgumentException("duel-thr-n: база feret | gt | fei | muct | orl, а не " + base);
            Setup u = new Setup();
            u.threads = threads;
            other(u, export, dir, feiKeys, selLines, settings, base);
            params = u.params;
            r0 = u.r0;
            seed = u.seed;
            Blk b = u.blocks.get(0);
            List<List<String>> wpz = splitPersons(b.own(), seed + N_SPLIT_OFFSET + sp[0], sp[1], sp[2]);
            List<String> own = wpz.get(0);
            Map<String, Integer> idx = new HashMap<>();
            for (int i = 0; i < own.size(); i++) idx.put(own.get(i), i);
            Set<String> outW = new HashSet<>(wpz.get(1));
            outW.addAll(wpz.get(2));
            List<Probe> probes = new ArrayList<>();
            int img = 0;
            for (Probe q : b.probes()) {
                if (idx.containsKey(q.person())) probes.add(new Probe(FeiScores.OWN, q.person(), ++img, q.key(), idx.get(q.person())));
            }
            for (String p : new TreeSet<>(outW)) {
                List<String> ks = new ArrayList<>(b.gal().get(p));
                for (Probe q : b.probes()) if (q.person().equals(p)) ks.add(q.key());
                for (String k : ks) probes.add(new Probe(FeiScores.CTRL, p, ++img, k, -1));
            }
            Map<String, List<String>> gal = new LinkedHashMap<>();
            for (String p : own) gal.put(p, b.gal().get(p));
            NB nb = new NB(base, b.title(), sp[0], own, gal, b.cohort(), probes, b.rOf(), new HashSet<>(wpz.get(1)), new HashSet<>(wpz.get(2)));
            todo.add(nb);
            rep.append(String.format(Locale.ROOT, "%s: людей в W %d, (П) %d, (З) %d; проб своих %d, (П) %d, (З) %d%n", b.title(), own.size(), wpz.get(1).size(),
                    wpz.get(2).size(), probes.stream().filter(q -> q.trueIdx() >= 0).count(), probes.stream().filter(q -> nb.pp().contains(q.person())).count(),
                    probes.stream().filter(q -> nb.zp().contains(q.person())).count()));
            Set<String> need = new TreeSet<>();
            gal.values().forEach(need::addAll);
            b.cohort().forEach(need::addAll);
            probes.forEach(q -> need.add(q.key()));
            for (String k : need) fr.put(k, u.fr.get(k));
            for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (!need.contains(e.getKey())) e.getValue().release();
            // Кадры k = 2: пул FERET и когорта FEI – FrameK2.load; снимки базы – FrameK2.load (FEI) или k2Raw (MUCT, ORL).
            Map<String, Sample> byKey = new HashMap<>();
            for (Map<String, List<Sample>> m : List.of(export.data().fei(), export.data().muct(), export.data().orl())) for (List<Sample> l : m.values()) for (Sample s : l)
                byKey.put(s.file().toString(), s);
            List<String> viaLoad = new ArrayList<>();
            List<Sample> viaRaw = new ArrayList<>();
            for (String k : need) {
                Sample s = byKey.get(k);
                if (s != null && (k.startsWith("muct") || k.startsWith("orl"))) viaRaw.add(s);
                else viaLoad.add(k);
            }
            XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
            Map<String, double[][]> cache = FrameK2.readPoints(su.dir.resolve(FeretData.DERIVED).resolve(FrameK2.POINTS));
            List<String> miss = viaLoad.stream().filter(k -> !k.contains("\\") && !k.contains("/") && !cache.containsKey(k)).toList();
            if (!miss.isEmpty()) throw new IllegalStateException("duel-thr-n: кадров k = 2 FERET нет в кэше derived (" + miss.size() + ") – остановка");
            k2.putAll(FrameK2.load(viaLoad, su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir())));
            if (!viaRaw.isEmpty()) k2.putAll(k2Raw(base, viaRaw, export, dir, Paths.get(base.equals("muct") ? settings.loadFacesMuctDir() : settings.loadFacesDatasetDir())));
        }
        if (!todo.isEmpty()) {
            XBaseFeat2.Feats fs = nFeats(params, k2, threads);
            for (int f = 1; f >= 0; f--) {
                long tv = System.nanoTime();
                XBaseEval x = XBaseFeat2.vectors(fr, fs, f, V8);
                System.out.printf(Locale.ROOT, "duel-thr-n %s: f%d v8 vectors (%.0f s)%n", base, f, (System.nanoTime() - tv) / 1e9);
                if (SUBS != null) {
                    for (NB b : todo) rep.append(retrain(x, f, b, r0, commit, vd));
                    continue;
                }
                for (NB b : todo) {
                    Path p = vd.resolve(b.file() + "_f" + f + "_n.tsv.gz");
                    if (Files.exists(p)) {
                        rep.append(p.getFileName()).append(": из каталога\n");
                        continue;
                    }
                    long ts = System.nanoTime();
                    List<XBaseVec.Model> ms = XBaseVec.run(List.of(nJobs(x, f, b, b.own(), b.probes(), "main")), threads).get(0);
                    Writer w = XBaseVec.writer(p, "# DuelThr duel-thr-n: " + b.title() + ", " + XBaseRule.FEATS[f] + ", (П2) + MS-LBP (v8), 1 − cos в W, "
                            + (f == 1 ? "модель по позе (r0 = " + r0 + ")" : "всегда X+mirror") + ", людей в W " + b.own().size() + ", проб " + b.probes().size()
                            + "\n# Код: коммит " + commit + "\n");
                    writeN(w, b.title(), ms.get(0), ms.get(1), f, b.rOf(), r0, b.pp(), b.zp(), null);
                    XBaseVec.finish(w, p);
                    rep.append(String.format(Locale.ROOT, "%s: %s, людей в W %d, проб %d, %.0f с%n", p.getFileName(), b.title(), b.own().size(), b.probes().size(),
                            (System.nanoTime() - ts) / 1e9));
                }
                x = null;
            }
        }
        for (Mat m : fr.values()) m.release();
        for (FrameK2.F f : k2.values()) f.frame().release();
        rep.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(vd.resolve("duel_thr_n_" + (SUBS != null ? System.getenv("FACES_DUELTHR_SUBS_TAG") + "_" : "") + base + ".txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    /** Режим duel-thr-n-b &lt;feret | gt&gt;: проверка (Б) для (Н) – W заново на 10 людях подвыборки (подвыборки – те же, что в duel-thr-b). */
    static void nCheckB(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                        Path out) throws IOException {
        long t0 = System.nanoTime();
        Setup u = setup(base, export, dir, feiKeys, selLines, settings, out);
        Blk b = u.blocks.stream().filter(x -> x.file().equals(base.equals("feret") ? "feret_x_p0" : "gt_x")).findFirst().orElseThrow();
        int rowNo = base.equals("feret") ? 0 : 2;
        int bRow = base.equals("feret") ? 0 : 1;
        List<List<String>> pz = ctrlSplit(b.probes(), u.seed, rowNo);
        NB nb = new NB(b.file(), b.title(), rowNo, b.own(), b.gal(), b.cohort(), b.probes(), b.rOf(), new HashSet<>(pz.get(1)), new HashSet<>(pz.get(2)));
        Path vd = nDir(base, out);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        List<List<String>> subs = new ArrayList<>();
        for (int k = 0; k < B_REPS; k++) {
            List<String> perm = new ArrayList<>(b.own());
            Collections.shuffle(perm, new Random(u.seed + B_SEED_OFFSET + 100L * bRow + k));
            List<String> g = new ArrayList<>(perm.subList(0, B_N));
            Collections.sort(g);
            subs.add(g);
        }
        Set<String> need = new TreeSet<>();
        for (List<String> g : subs) for (String p : g) need.addAll(b.gal().get(p));
        b.cohort().forEach(need::addAll);
        Set<String> sub = new HashSet<>();
        subs.forEach(sub::addAll);
        for (Probe q : b.probes()) if (q.trueIdx() < 0 || sub.contains(q.person())) need.add(q.key());
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : need) fr.put(k, u.fr.get(k));
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (!need.contains(e.getKey())) e.getValue().release();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, double[][]> cache = FrameK2.readPoints(su.dir.resolve(FeretData.DERIVED).resolve(FrameK2.POINTS));
        List<String> miss = need.stream().filter(k -> !k.contains("\\") && !k.contains("/") && !cache.containsKey(k)).toList();
        if (!miss.isEmpty()) throw new IllegalStateException("duel-thr-n-b: кадров k = 2 FERET нет в кэше derived (" + miss.size() + ") – остановка");
        Map<String, FrameK2.F> k2 = FrameK2.load(need, su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        XBaseFeat2.Feats fs = nFeats(u.params, k2, u.threads);
        StringBuilder rep = new StringBuilder("Проверка (Б) для (Н) – W заново на " + B_N + " людях, XBaseEval duel-thr-n-b " + base + "\nКод: коммит " + commit
                + " (с правками рабочей копии)\n");
        for (int f = 1; f >= 0; f--) {
            Path p = vd.resolve("b_" + b.file() + "_f" + f + "_n.tsv.gz");
            if (Files.exists(p)) {
                rep.append(p.getFileName()).append(": из каталога\n");
                continue;
            }
            XBaseEval x = XBaseFeat2.vectors(fr, fs, f, V8);
            List<List<Callable<XBaseVec.Model>>> js = new ArrayList<>();
            for (List<String> g : subs) {
                Map<String, Integer> idx = new HashMap<>();
                for (int i = 0; i < g.size(); i++) idx.put(g.get(i), i);
                List<Probe> pr = new ArrayList<>();
                for (Probe q : b.probes()) {
                    if (q.trueIdx() < 0) pr.add(q);
                    else if (idx.containsKey(q.person())) pr.add(new Probe(q.role(), q.person(), q.image(), q.key(), idx.get(q.person())));
                }
                js.add(nJobs(x, f, nb, g, pr, "b"));
            }
            long ts = System.nanoTime();
            List<List<XBaseVec.Model>> done = XBaseVec.run(js, u.threads);
            Writer w = XBaseVec.writer(p, "# DuelThr duel-thr-n-b: " + b.title() + ", " + XBaseRule.FEATS[f] + ", (П2) + MS-LBP (v8), W заново на " + B_N + " людях, "
                    + B_REPS + " подвыборок (seed faces.seed + " + B_SEED_OFFSET + " + 100 · " + bRow + " + k)\n# Код: коммит " + commit + "\n");
            for (int k = 0; k < subs.size(); k++) writeN(w, b.title() + ", подвыборка " + k, done.get(k).get(0), done.get(k).get(1), f, b.rOf(), u.r0, nb.pp(), nb.zp(),
                    null);
            XBaseVec.finish(w, p);
            rep.append(String.format(Locale.ROOT, "%s: %d подвыборок, %.0f с%n", p.getFileName(), subs.size(), (System.nanoTime() - ts) / 1e9));
        }
        for (Mat m : fr.values()) m.release();
        for (FrameK2.F f : k2.values()) f.frame().release();
        rep.append(String.format(Locale.ROOT, "Время: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(vd.resolve("duel_thr_n_b_" + base + ".txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- п. 2.3: сверка симуляции метода Прокаева с программой коллеги

    /** Вариант коллеги (Д20): имя листа без «а» / «б», вариант доли, l. */
    record PV(String name, ProkaevMethod.Variant v, int l) {}

    static final List<PV> PVS = List.of(new PV("1", ProkaevMethod.Variant.S, 3), new PV("2", ProkaevMethod.Variant.S, 5),
            new PV("3", ProkaevMethod.Variant.K, 5));
    /** Порог коллеги в «б»: ни один свой не набрал PB – гость. */
    static final double PB = 0.6;
    static final int PK = 7;
    static final int SCALE = 10;
    static final String[] WAYS = {"area", "block", "matlab", "matlab-d"};

    /** Серый кадр, уменьшенный в SCALE раз способом way, строкой пикселей. */
    static float[] small(Mat color, String way) {
        int w = color.cols() / SCALE;
        int h = color.rows() / SCALE;
        if (way.equals("area")) return DuelEval.grayAt(color, new org.opencv.core.Size(w, h));
        int rows = color.rows();
        int cols = color.cols();
        byte[] px = new byte[rows * cols * 3];
        color.get(0, 0, px);
        double[][] g = new double[rows][cols];
        if (way.equals("block")) {
            Mat gm = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(color, gm, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);
            byte[] gb = new byte[rows * cols];
            gm.get(0, 0, gb);
            gm.release();
            float[] out = new float[w * h];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                double s = 0;
                for (int dy = 0; dy < SCALE; dy++) for (int dx = 0; dx < SCALE; dx++) s += gb[(y * SCALE + dy) * cols + x * SCALE + dx] & 0xff;
                out[y * w + x] = (float) (s / (SCALE * SCALE));
            }
            return out;
        }
        // MATLAB: rgb2gray (uint8, округление), imresize(…, 1/10) по умолчанию – бикубическое со сглаживанием, сначала по строкам (размерность 1),
        // затем по столбцам; matlab – промежуточный и итог округлены до uint8, matlab-d – без округления.
        boolean round = way.equals("matlab");
        for (int y = 0; y < rows; y++) for (int x = 0; x < cols; x++) {
            int o = (y * cols + x) * 3;
            double v = 0.298936021293775 * (px[o + 2] & 0xff) + 0.587043074451121 * (px[o + 1] & 0xff) + 0.114020904255103 * (px[o] & 0xff);
            g[y][x] = Math.min(255, Math.max(0, Math.round(v)));
        }
        double[][] a = new double[h][cols];
        double[][] wy = weights(rows, h);
        int[][] iy = idx(rows, h);
        for (int x = 0; x < cols; x++) for (int u = 0; u < h; u++) {
            double s = 0;
            for (int t = 0; t < wy[u].length; t++) s += wy[u][t] * g[iy[u][t]][x];
            a[u][x] = round ? Math.min(255, Math.max(0, Math.round(s))) : s;
        }
        double[][] wx = weights(cols, w);
        int[][] ix = idx(cols, w);
        float[] out = new float[w * h];
        for (int u = 0; u < h; u++) for (int v = 0; v < w; v++) {
            double s = 0;
            for (int t = 0; t < wx[v].length; t++) s += wx[v][t] * a[u][ix[v][t]];
            out[u * w + v] = (float) (round ? Math.min(255, Math.max(0, Math.round(s))) : s);
        }
        return out;
    }

    static double cubic(double x) {
        double ax = Math.abs(x);
        if (ax <= 1) return 1.5 * ax * ax * ax - 2.5 * ax * ax + 1;
        if (ax <= 2) return -0.5 * ax * ax * ax + 2.5 * ax * ax - 4 * ax + 2;
        return 0;
    }

    /** Веса и индексы imresize (contributions): масштаб s = out / in &lt; 1 – ядро s · cubic(s · x), ширина 4 / s, края – отражение. */
    static double[][] weights(int in, int out) {
        double s = (double) out / in;
        double kw = 4 / s;
        int p = (int) Math.ceil(kw) + 2;
        double[][] wt = new double[out][p];
        for (int u = 1; u <= out; u++) {
            double x = u / s + 0.5 * (1 - 1 / s);
            int left = (int) Math.floor(x - kw / 2);
            double sum = 0;
            for (int t = 0; t < p; t++) {
                wt[u - 1][t] = s * cubic(s * (x - (left + t)));
                sum += wt[u - 1][t];
            }
            for (int t = 0; t < p; t++) wt[u - 1][t] /= sum;
        }
        return wt;
    }

    static int[][] idx(int in, int out) {
        double s = (double) out / in;
        double kw = 4 / s;
        int p = (int) Math.ceil(kw) + 2;
        int[][] ix = new int[out][p];
        for (int u = 1; u <= out; u++) {
            double x = u / s + 0.5 * (1 - 1 / s);
            int left = (int) Math.floor(x - kw / 2);
            for (int t = 0; t < p; t++) {
                int j = left + t; // 1-based, отражение как в imresize: [1:in, in:-1:1]
                int m = Math.floorMod(j - 1, 2 * in);
                ix[u - 1][t] = m < in ? m : 2 * in - 1 - m;
            }
        }
        return ix;
    }

    /** Режим duel-prokaev-check: доли по каждой пробе, оба комплекта, способы уменьшения WAYS, варианты 1–3, «а» и «б». */
    static void prokaevCheck(SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        String rawDuel = System.getenv("FACES_DUEL_DIR");
        if (rawDuel == null) throw new IllegalStateException("duel-prokaev-check: задайте FACES_DUEL_DIR (каталог набора GT_24.10)");
        Path duel = Paths.get(rawDuel);
        String rr = System.getenv("FACES_DUEL_REP");
        Path rep = rr != null ? Paths.get(rr) : out.getParent().resolve("duel");
        List<String> inv = DuelEval.verify(duel, rep.resolve("set_sha256_" + DuelSet.TAG24 + ".txt"), "duel-prokaev-check");
        Map<String, String> fileOf = new HashMap<>();
        List<DuelSet.Item> key = DuelEval.readKey24(rep.resolve("key_" + DuelSet.TAG24 + ".tsv"), fileOf);
        StringBuilder tsv = new StringBuilder("set\tway\tsheet\tprobe");
        for (int c = 1; c <= 10; c++) tsv.append("\tc").append(c);
        tsv.append("\tguest\n");
        for (int s = 0; s < DuelSet.SETS.length; s++) {
            String set = DuelSet.SETS[s];
            Map<String, Mat> color = new LinkedHashMap<>();
            for (DuelSet.Item it : key) color.put(it.code(), DuelSet.readColor(duel.resolve(set).resolve(fileOf.get(it.code()))));
            for (String way : WAYS) {
                Map<String, float[]> gray = new HashMap<>();
                color.forEach((c, m) -> gray.put(c, small(m, way)));
                List<float[]> a = new ArrayList<>();
                List<String> lab = new ArrayList<>();
                for (DuelSet.Item it : key) if (it.code().startsWith("P")) {
                    a.add(gray.get(it.code()));
                    lab.add(it.code().substring(0, 3));
                }
                ProkaevMethod pm = ProkaevMethod.fit(a.toArray(float[][]::new), lab.toArray(String[]::new), null, PK);
                for (PV pv : PVS) for (boolean b : new boolean[] {false, true}) {
                    for (DuelSet.Item it : key) {
                        if (!DuelEval.probe(it)) continue;
                        ProkaevMethod.Result r = pm.decide(pm.project(gray.get(it.code()), pv.v()), pv.v(), PK, pv.l());
                        double[] sh = new double[11];
                        double mx = 0;
                        for (int c = 0; c < 10; c++) {
                            sh[c] = r.share().getOrDefault(String.format(Locale.ROOT, "P%02d", c + 1), 0.0);
                            mx = Math.max(mx, sh[c]);
                        }
                        if (b && mx < PB) {
                            java.util.Arrays.fill(sh, 0);
                            sh[10] = 1;
                        }
                        String f = fileOf.get(it.code());
                        tsv.append(set).append('\t').append(way).append('\t').append(pv.name()).append(b ? "б" : "а").append('\t')
                                .append(f.substring(f.lastIndexOf('/') + 1, f.length() - 4));
                        for (double v : sh) tsv.append('\t').append(v);
                        tsv.append('\n');
                    }
                }
            }
            color.values().forEach(Mat::release);
        }
        Path o = out.getParent().resolve("duel").resolve("duel_24.10_prokaev_check.tsv");
        Files.createDirectories(o.getParent());
        FarMethods.write(o, tsv);
        System.out.printf(Locale.ROOT, "duel-prokaev-check: файлов набора %d (опись сверена), %s, %.0f с%nDONE%n", inv.size(), o, (System.nanoTime() - t0) / 1e9);
    }

    /** Чтение файла векторов (для тестов и проверок): подпись блока → [галерея, строки]. */
    static Map<String, List<String[]>> read(Path p) throws IOException {
        Map<String, List<String[]>> out = new LinkedHashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(p)), StandardCharsets.UTF_8))) {
            List<String[]> cur = null;
            for (String l = in.readLine(); l != null; l = in.readLine()) {
                if (l.startsWith("#S\t")) {
                    cur = new ArrayList<>();
                    out.put(l.substring(3), cur);
                } else if (cur != null && (l.startsWith("#G") || l.startsWith("R\t"))) {
                    cur.add(l.split("\t", -1));
                }
            }
        }
        return out;
    }
}
