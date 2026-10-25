package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseKiss.Pass;
import svd.recognizer.storage.SettingsStore;

/**
 * Диагностика KISSME (п. 3; ответ Жорика 25.10.2026 на итоги, план подтверждён Хозяином). kiss-nofloor – строки «без пола у Σ_S»
 * по готовым файлам оценок feret-kiss и gt-kiss (без обучения): разделимость, EER, П3 и П2, пары «строка − база». kiss-ctrl –
 * контрольные строки на той же PCA, что у (К1), m = 100 и 300: (Е) M = I, (С) M = Σ_S⁻¹ с полом λ̄, (С0) M = Σ_S⁻¹ без пола
 * (псевдообратная по рангу); оценка до центра класса, d₁/d₂, модель X / X+mirror по |r|; FERET межс. и GT внутри сессии; рядом –
 * доля энергии PCA в первых m компонентах. Режимы XBaseEval feret-kiss-nofloor, gt-kiss-nofloor, feret-kiss-ctrl, gt-kiss-ctrl.
 *
 * @author ssv
 */
final class XBaseKissDiag {

    static final int[] CM = {100, 300};
    /** Строки kiss-ctrl: (Е), (С), (С0) при m = 100 и 300. */
    static final String[] CNAME = {"(Е) евклид в PCA, m = 100", "(Е) евклид в PCA, m = 300", "(С) Σ_S⁻¹ с полом, m = 100", "(С) Σ_S⁻¹ с полом, m = 300",
        "(С0) Σ_S⁻¹ без пола, m = 100", "(С0) Σ_S⁻¹ без пола, m = 300"};
    static final int NC = CNAME.length;

    private XBaseKissDiag() {
    }

    // ---------------------------------------------------------------- постановки (как XBaseKiss.feret / gt)

    /** FERET: постановка post (0 – межс., 1 – (ii) s3), проходы – как в XBaseKiss.feret. */
    static List<Pass> feretPasses(XBaseRule.FSetup s, XBaseFeret.Roles ro0, XBaseFeret.Roles ro, XBaseEval.Gt base, Map<String, List<List<String>>> kept, int post) {
        XBaseEval.Gt tpl = s.tpl;
        int conf = post == 0 ? 0 : 71;
        List<Pass> passes = new ArrayList<>();
        for (int ps = 0; ps < 2; ps++) {
            Set<String> mine = s.halves.get(ps);
            List<String> own;
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            if (post == 0) {
                own = XBaseKiss.trim(ro.own1.stream().filter(mine::contains).toList());
                for (String p : own) {
                    gal.put(p, ro.gal1.get(p));
                    pro.put(p, ro.pro1.get(p));
                }
            } else {
                own = XBaseKiss.trim(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList());
                for (String p : own) {
                    List<String> g = new ArrayList<>();
                    for (int j = 0; j < 3; j++) g.addAll(s.pool.get(p).get(j));
                    gal.put(p, XBaseFeret.sorted(g));
                    List<String> pr = new ArrayList<>();
                    for (int j = 3; j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                    pro.put(p, XBaseFeret.sorted(pr));
                }
            }
            Map<String, List<List<String>>> projPool = new TreeMap<>();
            for (String p : s.halves.get(1 - ps)) if (XBaseKiss.smoke() != 1 || projPool.size() < XBaseDates.SMOKE_POOL) projPool.put(p, s.pool.get(p));
            XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
            List<List<String>> aug = new ArrayList<>(s.cohort);
            aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
            g.cohort = aug;
            List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + conf);
            List<Probe> probes = XBaseEval.probes(g, pro);
            passes.add(new Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
        }
        return passes;
    }

    /** Все отобранные снимки своих по датам – как XBaseKiss.feret. */
    static Map<String, List<List<String>>> kept(XBaseFeret.Roles ro0, Map<String, Img> byKey) {
        Map<String, List<List<String>>> kept = new TreeMap<>();
        for (String p : ro0.own1) {
            Map<String, List<String>> bd = new TreeMap<>();
            List<String> al = new ArrayList<>(ro0.gal1.get(p));
            al.addAll(ro0.pro1.get(p));
            for (String k : al) bd.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ks = new ArrayList<>();
            for (List<String> l : bd.values()) ks.add(XBaseFeret.sorted(l));
            kept.put(p, ks);
        }
        return kept;
    }

    /** GT: данные и постановки (режим 0 – межс., 1 – внутри) – как XBaseKiss.gt. */
    static final class GtSetup {
        XBaseEval.Gt g;
        XBaseEval.Gt ga;
        XBaseA0.FeretSetup su;
        Map<String, List<List<String>>> pool;
        List<List<String>> parts;
        Object[][] modes;
    }

    static GtSetup gtSetup(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-kiss-*: только равные галереи FERET (faces.feret.gallery = eq)");
        GtSetup t = new GtSetup();
        t.g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        t.su = XBaseA0.feretSetup(dir, selLines, settings);
        t.pool = XBaseXsF.pool(t.su.ro, t.su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), t.pool, Set.of(), Map.of()).extra();
        if (XBaseKiss.smoke() == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));
        long eqSeed = t.g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(t.g.own, t.g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(t.g.own, t.g.galleryAll, t.g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(t.g.own, split.get(0), eqSeed);
        List<String> own = XBaseKiss.trim(t.g.own);
        t.ga = XBaseDates.withOwn(t.g, own);
        List<List<String>> aug = new ArrayList<>(t.g.cohort);
        aug.addAll(poolCls);
        t.ga.cohort = aug;
        t.parts = XBaseKiss.smoke() == 1 ? XBaseOther.cvParts(own, t.g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(t.g.own, t.g.seed, XBaseEval.CV_S);
        t.modes = new Object[][] {{"межс.", galX, t.g.probeKeys}, {"внутри", galI, split.get(1)}};
        return t;
    }

    static Pass gtPass(GtSetup t, int mi) {
        @SuppressWarnings("unchecked")
        Map<String, List<String>> gal = (Map<String, List<String>>) t.modes[mi][1];
        @SuppressWarnings("unchecked")
        Map<String, List<String>> pro = (Map<String, List<String>>) t.modes[mi][2];
        List<Probe> probes = XBaseEval.probes(t.ga, pro);
        return new Pass(t.ga, gal, pro, t.parts, probes, mi == 0 ? XBaseThr.extra(t.ga, pro, t.parts) : null);
    }

    // ---------------------------------------------------------------- таблицы

    /** Таблицы одной постановки: строки label → строки оценок по проходам; пары «строка − база» (база – первая строка). */
    static void table(StringBuilder t, String title, List<Pass> passes, Map<String, List<String>> rows, int persons, XBaseEval.Gt tpl, boolean t2) {
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        String bl = null;
        for (Map.Entry<String, List<String>> e : rows.entrySet()) {
            if (bl == null) bl = e.getKey();
            List<XBaseThr.Unit> us = new ArrayList<>();
            for (int i = 0; i < passes.size(); i++) {
                Pass p = passes.get(i);
                us.add(new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), e.getValue().get(i), null, t2));
            }
            outs.put(e.getKey(), XBaseThr.evaluate(e.getKey(), us, persons, tpl));
        }
        List<String[]> pairs = new ArrayList<>();
        for (String k : outs.keySet()) if (!k.equals(bl)) pairs.add(new String[] {k, bl});
        XBaseLbp.tables(t, title, outs, persons, tpl, pairs, t2);
    }

    /** Строки файла оценок KISSME: блок → строки. */
    static String block(Map<String, String> blocks, String name, Path src) {
        String b = blocks.get(name);
        if (b == null) throw new IllegalStateException("XBaseKissDiag: нет блока «" + name + "» в " + src);
        return b;
    }

    // ---------------------------------------------------------------- kiss-nofloor

    static void feretNofloor(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, List<List<String>>> kept = kept(ro0, s.su.byKey);
        Path kissScores = s.out.resolve("xbase_feret_kiss_scores.tsv.gz");
        Path lbpScores = s.out.resolve("xbase_feret_lbp_scores.tsv.gz");
        Path datesScores = s.out.resolve("xbase_feret_dates_scores.tsv.gz");
        for (Path f : List.of(kissScores, lbpScores, datesScores)) if (!Files.exists(f)) throw new IllegalStateException("feret-kiss-nofloor: нужен " + f);
        Map<String, String> kb = XBaseSubsp.blocks(kissScores);
        Map<String, String> lb = XBaseSubsp.blocks(lbpScores);
        Map<String, String> db = XBaseSubsp.blocks(datesScores);
        StringBuilder t = new StringBuilder();
        t.append("П. 3, диагностика KISSME: строки без пола у Σ_S по файлу оценок feret-kiss (XBaseEval feret-kiss-nofloor): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; файл оценок ").append(kissScores.getFileName()).append(" (").append(FeiThreshold.sha(kissScores)).append(")\n");
        t.append("Постановки и проходы – как feret-kiss (строятся тем же кодом); обучения нет. Строки: база (косинус), варианты с полом у Σ_S (как в отчёте "
                + "feret-kiss) и без пола;\nпары «строка − база».\n");
        t.append(XBaseThr.NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: строки базы файла feret-kiss = feret-lbp («межс.») и feret-dates ((ii) s3 (м)), побайтно ===\n");
        StringBuilder body = new StringBuilder();
        for (int post = 0; post < 2; post++) {
            String pname = post == 0 ? "межс." : "(ii) s3";
            List<Pass> passes = feretPasses(s, ro0, ro, base, kept, post);
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                Map<String, List<String>> rows = new LinkedHashMap<>();
                for (int v : order()) {
                    List<String> ls = new ArrayList<>();
                    for (int ps = 0; ps < 2; ps++) ls.add(block(kb, pname + ", " + feat + ", проход " + (ps + 1) + ", " + XBaseKiss.rowName(v), kissScores));
                    rows.put(XBaseKiss.rowName(v), ls);
                }
                for (int ps = 0; ps < 2; ps++) {
                    String r0 = rows.get(XBaseKiss.rowName(0)).get(ps);
                    if (post == 0) check.append(XBaseReg.check("проход " + (ps + 1) + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", r0, lb,
                            lbpScores));
                    else check.append(XBaseReg.check("(i)=(ii) s3 (м) " + feat + ", проход " + (ps + 1), r0, db, datesScores));
                }
                table(body, "FERET " + pname + ", " + feat, passes, rows, persons.size(), tpl, true);
            }
        }
        t.append(check).append(body);
        FarMethods.write(s.out.resolve("xbase_feret_kiss_nofloor.txt"), t);
        System.out.print(t);
    }

    /** Порядок строк: база, (К2) с полом и без, (К1) m = 50…300 с полом и без. */
    static int[] order() {
        int[] o = new int[XBaseKiss.ROWS];
        int i = 0;
        o[i++] = 0;
        for (int v = 1; v < XBaseKiss.NV; v++) {
            o[i++] = v;
            o[i++] = XBaseKiss.NO + v;
        }
        return o;
    }

    static void gtNofloor(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        GtSetup gs = gtSetup(export, dir, feiKeys, selLines, settings);
        XBaseEval.Gt g = gs.g;
        Path kissScores = out.resolve("xbase_gt_kiss_scores.tsv.gz");
        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        for (Path f : List.of(kissScores, lbpScores)) if (!Files.exists(f)) throw new IllegalStateException("gt-kiss-nofloor: нужен " + f);
        Map<String, String> kb = XBaseSubsp.blocks(kissScores);
        Map<String, String> lb = XBaseSubsp.blocks(lbpScores);
        StringBuilder t = new StringBuilder();
        t.append("П. 3, диагностика KISSME: строки без пола у Σ_S по файлу оценок gt-kiss (XBaseEval gt-kiss-nofloor)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; файл оценок ").append(kissScores.getFileName())
                .append(" (").append(FeiThreshold.sha(kissScores)).append(")\n");
        t.append("Постановки – как gt-kiss (межс. N = 6, порог (Т2), рядом (Т0); внутри N = 3, порог (Т0)); обучения нет. Строки: база, варианты с полом и без пола; "
                + "пары «строка − база».\n");
        t.append(XBaseThr.NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: строки базы файла gt-kiss = gt-lbp, побайтно ===\n");
        StringBuilder body = new StringBuilder();
        for (int mi = 0; mi < 2; mi++) {
            String mode = (String) gs.modes[mi][0];
            Pass pass = gtPass(gs, mi);
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                Map<String, List<String>> rows = new LinkedHashMap<>();
                for (int v : order()) rows.put(XBaseKiss.rowName(v), List.of(block(kb, mode + ", " + feat + ", " + XBaseKiss.rowName(v), kissScores)));
                check.append(XBaseReg.check(mode + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", rows.get(XBaseKiss.rowName(0)).get(0), lb,
                        lbpScores));
                table(body, "GT " + mode + ", " + feat, List.of(pass), rows, g.persons.size(), g, mi == 0);
            }
        }
        t.append(check).append(body);
        FarMethods.write(out.resolve("xbase_gt_kiss_nofloor.txt"), t);
        System.out.print(t);
    }

    // ---------------------------------------------------------------- kiss-ctrl

    /** Строки контрольной модели: NC строк и сведения (энергия PCA и ранги). */
    static String[] ctrlModel(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> rest,
                              List<Probe> probes, Map<String, String> dateOf, long seed, String tag) {
        List<List<String>> classes = new ArrayList<>(gallery);
        classes.addAll(rest);
        XBaseKiss.Pairs pr = XBaseKiss.pairs(x, mirror, classes, dateOf, seed);
        LdaMath.Data dt = LdaMath.Data.of(List.of(pr.vec()));
        int mMax = CM[CM.length - 1];
        LdaMath.Span sp = LdaMath.span(dt, XBaseKiss.RANK_TOL, mMax);
        int r = sp.basis().length;
        double[] mean = dt.mean();
        double[][] basis = sp.basis();
        double total = 0;
        for (double[] v : pr.vec()) for (int i = 0; i < v.length; i++) total += (v[i] - mean[i]) * (v[i] - mean[i]);
        java.util.function.Function<double[], double[]> pca = v -> {
            double[] z = new double[r];
            for (int j = 0; j < r; j++) {
                double a = 0;
                double[] b = basis[j];
                for (int i = 0; i < v.length; i++) a += b[i] * (v[i] - mean[i]);
                z[j] = a;
            }
            return z;
        };
        double[][] z = new double[pr.vec().size()][];
        for (int i = 0; i < z.length; i++) z[i] = pca.apply(pr.vec().get(i));
        double[][] csP = XBaseKiss.cov(z, pr.s());
        double[][] cP = XBaseKiss.centers(z, gallery, gallery.size(), mirror);
        Map<String, double[]> zp = new HashMap<>();
        for (Probe p : probes) zp.computeIfAbsent(p.key(), key -> pca.apply(x.vec.get(V.L1).get(key)));
        String[] out = new String[NC + 1];
        StringBuilder diag = new StringBuilder();
        for (int mi = 0; mi < CM.length; mi++) {
            int m = Math.min(CM[mi], r);
            double en = 0;
            for (int j = 0; j < m; j++) en += sp.sigma()[j] * sp.sigma()[j];
            double[][] cm = new double[cP.length][];
            for (int g = 0; g < cP.length; g++) cm[g] = cP[g] == null ? null : Arrays.copyOf(cP[g], m);
            java.util.function.Function<String, double[]> pm = key -> Arrays.copyOf(zp.get(key), m);
            double[][] eye = new double[m][m];
            for (int i = 0; i < m; i++) eye[i][i] = 1;
            out[mi] = XBaseKiss.lines(method, s, own, probes, eye, cm, pm);
            LdaMath.Eigen es = LdaMath.eigen(XBaseKiss.lead(csP, m));
            double maxS = Arrays.stream(es.values()).max().orElse(0);
            double tr = 0;
            int rankS = 0;
            for (double v : es.values()) {
                tr += v;
                if (v > XBaseKiss.RANK_TOL * maxS) rankS++;
            }
            double bar = tr / m;
            int raised = 0;
            for (double v : es.values()) if (v < bar) raised++;
            for (boolean fl : new boolean[] {true, false}) {
                List<double[]> rows = new ArrayList<>();
                for (int k = 0; k < m; k++) {
                    double v = es.values()[k];
                    if (fl) v = Math.max(v, bar);
                    else if (v <= XBaseKiss.RANK_TOL * maxS) continue;
                    double[] u = new double[m];
                    for (int i = 0; i < m; i++) u[i] = es.vectors()[i][k] / Math.sqrt(v);
                    rows.add(u);
                }
                out[(fl ? 2 : 4) + mi] = XBaseKiss.lines(method, s, own, probes, rows.toArray(new double[0][]), cm, pm);
            }
            diag.append(String.format(Locale.ROOT, "%s\tm = %d\t%d\t%.4f\t%d\t%d\t%d%n", tag, m, pr.s().length, en / total, r, rankS, raised));
        }
        out[NC] = diag.toString();
        return out;
    }

    /** Задачи блока – как XBaseKiss.job (тот же порядок моделей, seed пар). */
    static List<Callable<String[]>> ctrlJob(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts, int s,
                                            List<List<Probe>> extra, Map<String, String> dateOf, long seed) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<String[]>> tasks = new ArrayList<>();
        int mno = 0;
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            long s0 = seed + XBaseKiss.SEED + mno++;
            String t0 = (m ? "X+mirror" : "X") + " основная";
            tasks.add(() -> ctrlModel(x, m, method, s, own, fullGallery, cohort, main, dateOf, s0, t0));
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                long sq = seed + XBaseKiss.SEED + mno++;
                String tq = (m ? "X+mirror" : "X") + " без части " + q;
                tasks.add(() -> ctrlModel(x, m, method, s, g, gk, cohort, pr, dateOf, sq, tq));
            }
        }
        return tasks;
    }

    /** Блоки задач разом; [блок][NC строк + сведения]. */
    static List<String[]> run(List<List<Callable<String[]>>> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (List<Callable<String[]>> j : jobs) flat.addAll(j);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<String[]> out = new ArrayList<>();
            int i = 0;
            for (List<Callable<String[]>> j : jobs) {
                StringBuilder[] b = new StringBuilder[NC + 1];
                for (int r = 0; r <= NC; r++) b[r] = new StringBuilder();
                for (int k = 0; k < j.size(); k++) {
                    String[] r = fs.get(i++).get();
                    for (int q = 0; q <= NC; q++) b[q].append(r[q]);
                }
                String[] o = new String[NC + 1];
                for (int r = 0; r <= NC; r++) o[r] = b[r].toString();
                out.add(o);
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseKissDiag: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    static String[] read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[] o = new String[NC + 1];
        for (int r = 0; r <= NC; r++) {
            o[r] = cp.read(tag + "_" + r);
            if (o[r] == null) return null;
        }
        return o;
    }

    static void save(XBaseOther.Checkpoint cp, String tag, String[] o) throws IOException {
        for (int r = 0; r <= NC; r++) cp.save(tag + "_" + r, o[r]);
    }

    static final String DIAG_HEAD = "\n=== Сведения контрольных моделей (PCA выборки, как у (К1)) ===\nблок | модель | m | |S| | доля энергии PCA в первых m | ранг PCA "
            + "(≤ 300) | ранг Σ_S | поднято до пола λ̄ у Σ_S\n";

    static final String CTRL_NOTE = "Контрольные строки на той же PCA, что у (К1) (выборка W: галерея, когорта FEI, пул FERET, у X+mirror – с отражениями; PCA – одна на "
            + "модель, m = 100 и 300 –\nпервые компоненты): (Е) M = I – евклид в PCA; (С) M = Σ_S⁻¹, пол λ̄ = tr/n; (С0) M = Σ_S⁻¹ без пола (псевдообратная по "
            + "рангу). Σ_S – по тем же межсессионным\nпарам S, что у KISSME (seed faces.seed + 10400 + номер модели); оценка до центра класса, d₁/d₂, модель X / "
            + "X+mirror по |r|. Рядом – база и (К1) m = 100, 300\nс полом и без (из файла оценок KISSME).\n";

    static Map<String, String> dates(Map<String, Img> byKey) {
        return XBaseKiss.dates(byKey);
    }

    static void feretCtrl(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = XBaseKiss.smoke();
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, List<List<String>>> kept = kept(ro0, s.su.byKey);
        Map<String, String> dateOf = dates(s.su.byKey);
        Path kissScores = s.out.resolve("xbase_feret_kiss_scores.tsv.gz");
        if (!Files.exists(kissScores)) throw new IllegalStateException("feret-kiss-ctrl: нужен " + kissScores);
        Map<String, String> kb = sm == 1 ? Map.of() : XBaseSubsp.blocks(kissScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-kiss-ctrl %s; seed %d; r0 %s; пул %d; пробный %d", s.su.dir, tpl.seed, tpl.r0,
                s.pool.size(), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_kiss_ctrl_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);
        XBaseEval[] xv = XBaseKiss.vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();
        List<Pass> passes = feretPasses(s, ro0, ro, base, kept, 0);
        StringBuilder t = new StringBuilder();
        t.append("П. 3, диагностика KISSME: контрольные строки на PCA (К1), FERET межс. (XBaseEval feret-kiss-ctrl): Color FERET — только локально")
                .append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append("; база и (К1) – файл оценок ").append(kissScores.getFileName()).append(" (").append(FeiThreshold.sha(kissScores)).append(")\n");
        t.append("Постановка «межс.» – как feret-kiss (половины №2, части – номер 0).\n");
        t.append(CTRL_NOTE);
        t.append(XBaseThr.NOTE);
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            List<String[]> rows = new ArrayList<>();
            List<List<Callable<String[]>>> jobs = new ArrayList<>();
            List<Integer> todo = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                String tag = String.format(Locale.ROOT, "p%d_f%d", ps, f);
                String[] r = read(cp, tag);
                rows.add(r);
                if (r != null) continue;
                Pass p = passes.get(ps);
                jobs.add(ctrlJob(p.g(), xv[f], p.gal(), p.probes(), p.parts(), 0, p.extra(), dateOf, tpl.seed));
                todo.add(ps);
            }
            if (!jobs.isEmpty()) {
                long t0 = System.nanoTime();
                List<String[]> done = run(jobs, tpl.threads);
                double sec = (System.nanoTime() - t0) / 1e9;
                fitS += sec;
                for (int i = 0; i < todo.size(); i++) {
                    rows.set(todo.get(i), done.get(i));
                    save(cp, String.format(Locale.ROOT, "p%d_f%d", todo.get(i), f), done.get(i));
                }
                System.out.printf(Locale.ROOT, "feret-kiss-ctrl: %s ready, %d blocks (%.0f s)%n", feat, jobs.size(), sec);
            }
            for (int ps = 0; ps < 2; ps++) {
                dg.append(XBaseKiss.diag("межс. " + feat + " проход " + (ps + 1), rows.get(ps)[NC]));
                for (int v = 0; v < NC; v++) lines.append("# межс., ").append(feat).append(", проход ").append(ps + 1).append(", ").append(CNAME[v]).append('\n')
                        .append(rows.get(ps)[v]);
            }
            if (sm == 2) {
                t.append(dg).append(String.format(Locale.ROOT, "Замер: L1-tt-xs-F, 2 прохода – %.0f с; блоков всего 4 (2 признака × 2 прохода).%n", fitS));
                FarMethods.write(s.out.resolve("xbase_feret_kiss_ctrl" + sfx + ".txt"), t);
                System.out.print(t);
                cp.delete();
                return;
            }
            Map<String, List<String>> tr = new LinkedHashMap<>();
            if (sm != 1) {
                for (int v : new int[] {0, 3, XBaseKiss.NO + 3, 5, XBaseKiss.NO + 5}) {
                    List<String> ls = new ArrayList<>();
                    for (int ps = 0; ps < 2; ps++) ls.add(block(kb, "межс., " + feat + ", проход " + (ps + 1) + ", " + XBaseKiss.rowName(v), kissScores));
                    tr.put(XBaseKiss.rowName(v), ls);
                }
            } else {
                tr.put(CNAME[1] + " (вместо базы – пробный режим)", List.of(rows.get(0)[1], rows.get(1)[1]));
            }
            for (int v = 0; v < NC; v++) tr.put(CNAME[v], List.of(rows.get(0)[v], rows.get(1)[v]));
            table(body, "FERET межс., " + feat, passes, tr, persons.size(), tpl, true);
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body).append(dg);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_kiss_ctrl" + sfx + "_scores.tsv.gz"), "# FERET, KISSME – контрольные строки (XBaseKissDiag feret-kiss-ctrl), "
                + "только локально\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_kiss_ctrl" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    /** (Е0): евклид до центра класса в исходном пространстве признака (без PCA); центр – среднее снимков галереи (у X+mirror – с отражениями). */
    static String rawLines(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<Probe> probes) {
        double[][] c = new double[gallery.size()][];
        for (int g = 0; g < gallery.size(); g++) {
            List<double[]> vs = new ArrayList<>();
            for (String k : gallery.get(g)) {
                vs.add(x.vec.get(V.L1).get(k));
                if (mirror) vs.add(x.mir.get(V.L1).get(k));
            }
            if (vs.isEmpty()) continue;
            double[] m = new double[vs.get(0).length];
            for (double[] v : vs) for (int i = 0; i < m.length; i++) m[i] += v[i];
            for (int i = 0; i < m.length; i++) m[i] /= vs.size();
            c[g] = m;
        }
        StringBuilder out = new StringBuilder();
        for (Probe p : probes) {
            double[] y = x.vec.get(V.L1).get(p.key());
            double[] sc = new double[c.length];
            for (int g = 0; g < c.length; g++) {
                if (c[g] == null) {
                    sc[g] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double d = 0;
                for (int i = 0; i < y.length; i++) d += (y[i] - c[g][i]) * (y[i] - c[g][i]);
                sc[g] = d;
            }
            out.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), own, sc, false));
        }
        return out.toString();
    }

    /** gt-kiss-raw: (Е0) на GT внутри сессии рядом с базой и (Е) m = 100, 300 из файлов оценок gt-kiss и gt-kiss-ctrl. */
    static void gtRaw(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        GtSetup gs = gtSetup(export, dir, feiKeys, selLines, settings);
        XBaseEval.Gt g = gs.g;
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        XBaseEval[] xv = XBaseKiss.vectors(fr, g.params);
        for (Mat m : fr.values()) m.release();
        Path kissScores = out.resolve("xbase_gt_kiss_scores.tsv.gz");
        Path ctrlScores = out.resolve("xbase_gt_kiss_ctrl_scores.tsv.gz");
        for (Path f : List.of(kissScores, ctrlScores)) if (!Files.exists(f)) throw new IllegalStateException("gt-kiss-raw: нужен " + f);
        Map<String, String> kb = XBaseSubsp.blocks(kissScores);
        Map<String, String> cb = XBaseSubsp.blocks(ctrlScores);
        Pass pass = gtPass(gs, 1);
        StringBuilder t = new StringBuilder();
        t.append("П. 3, диагностика KISSME: евклид до центра класса без PCA, GT внутри сессии (XBaseEval gt-kiss-raw)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; база – ").append(kissScores.getFileName())
                .append(" (").append(FeiThreshold.sha(kissScores)).append("), (Е) – ").append(ctrlScores.getFileName()).append(" (").append(FeiThreshold.sha(ctrlScores))
                .append(")\n");
        t.append("(Е0) – евклид в исходном пространстве признака (корень гистограмм LBP, у L1-tt-xs-F – после Tan – Triggs), центр – среднее снимков галереи (у "
                + "X+mirror – с отражениями),\nбез обучения; d₁/d₂, модель X / X+mirror по |r|; основная модель и 5 моделей частей (только центры). Постановка – как "
                + "gt-kiss «внутри» (N = 3, порог (Т0)).\n");
        StringBuilder body = new StringBuilder();
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            StringBuilder raw = new StringBuilder();
            List<String> own = gs.ga.own;
            for (boolean m : new boolean[] {false, true}) {
                String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
                raw.append(rawLines(xv[f], m, method, 0, own, own.stream().map(pass.gal()::get).toList(), pass.probes()));
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(pass.parts().get(q));
                    List<String> gq = own.stream().filter(p -> !part.contains(p)).toList();
                    List<Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : pass.gal().get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gs.ga.img(k), k, -1));
                    raw.append(rawLines(xv[f], m, method, 0, gq, gq.stream().map(pass.gal()::get).toList(), pr));
                }
            }
            Map<String, List<String>> tr = new LinkedHashMap<>();
            tr.put(XBaseKiss.rowName(0), List.of(block(kb, "внутри, " + feat + ", " + XBaseKiss.rowName(0), kissScores)));
            tr.put("(Е0) евклид без PCA", List.of(raw.toString()));
            for (int v = 0; v < 2; v++) tr.put(CNAME[v], List.of(block(cb, "внутри, " + feat + ", " + CNAME[v], ctrlScores)));
            table(body, "GT внутри, " + feat, List.of(pass), tr, g.persons.size(), g, false);
        }
        t.append(body);
        FarMethods.write(out.resolve("xbase_gt_kiss_raw.txt"), t);
        System.out.print(t);
    }

    static void gtCtrl(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        int sm = XBaseKiss.smoke();
        GtSetup gs = gtSetup(export, dir, feiKeys, selLines, settings);
        XBaseEval.Gt g = gs.g;
        Map<String, String> dateOf = dates(gs.su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : gs.pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(gs.su.dir, "a", k));
        XBaseEval[] xv = XBaseKiss.vectors(fr, g.params);
        for (Mat m : fr.values()) m.release();
        Path kissScores = out.resolve("xbase_gt_kiss_scores.tsv.gz");
        if (!Files.exists(kissScores)) throw new IllegalStateException("gt-kiss-ctrl: нужен " + kissScores);
        Map<String, String> kb = sm == 1 ? Map.of() : XBaseSubsp.blocks(kissScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-kiss-ctrl; seed %d; r0 %s; пул FERET %d; пробный %d", g.seed, g.r0, gs.pool.size(), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_kiss_ctrl" + sfx), fp);
        Pass pass = gtPass(gs, 1);
        StringBuilder t = new StringBuilder();
        t.append("П. 3, диагностика KISSME: контрольные строки на PCA (К1), GT внутри сессии (XBaseEval gt-kiss-ctrl)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                .append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; база и (К1) – файл оценок ")
                .append(kissScores.getFileName()).append(" (").append(FeiThreshold.sha(kissScores)).append(")\n");
        t.append("Постановка «внутри» – как gt-kiss (N = 3, порог (Т0)).\n");
        t.append(CTRL_NOTE);
        t.append(XBaseThr.NOTE);
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            String tag = "i_f" + f;
            String[] r = read(cp, tag);
            if (r == null) {
                long t0 = System.nanoTime();
                r = run(List.of(ctrlJob(gs.ga, xv[f], pass.gal(), pass.probes(), pass.parts(), 0, null, dateOf, g.seed)), g.threads).get(0);
                double sec = (System.nanoTime() - t0) / 1e9;
                fitS += sec;
                save(cp, tag, r);
                System.out.printf(Locale.ROOT, "gt-kiss-ctrl: %s ready (%.0f s)%n", feat, sec);
            }
            dg.append(XBaseKiss.diag("внутри " + feat, r[NC]));
            for (int v = 0; v < NC; v++) lines.append("# внутри, ").append(feat).append(", ").append(CNAME[v]).append('\n').append(r[v]);
            if (sm == 2) {
                t.append(dg).append(String.format(Locale.ROOT, "Замер: L1-tt-xs-F – %.0f с; блоков всего 2 (2 признака).%n", fitS));
                FarMethods.write(out.resolve("xbase_gt_kiss_ctrl" + sfx + ".txt"), t);
                System.out.print(t);
                cp.delete();
                return;
            }
            Map<String, List<String>> tr = new LinkedHashMap<>();
            if (sm != 1) {
                for (int v : new int[] {0, 3, XBaseKiss.NO + 3, 5, XBaseKiss.NO + 5}) tr.put(XBaseKiss.rowName(v), List.of(block(kb, "внутри, " + feat + ", "
                        + XBaseKiss.rowName(v), kissScores)));
            } else {
                tr.put(CNAME[1] + " (вместо базы – пробный режим)", List.of(r[1]));
            }
            for (int v = 0; v < NC; v++) tr.put(CNAME[v], List.of(r[v]));
            table(body, "GT внутри, " + feat, List.of(pass), tr, g.persons.size(), g, false);
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body).append(dg);
        XBaseOther.writeScores(out.resolve("xbase_gt_kiss_ctrl" + sfx + "_scores.tsv.gz"), "# GT, KISSME – контрольные строки (XBaseKissDiag gt-kiss-ctrl)\n# Код: "
                + "коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_kiss_ctrl" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
