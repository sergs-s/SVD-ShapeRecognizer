package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Параметры LBP под кадр 92×112 (ревизия 1.4, ступень 1; задание №8 Жорика от 24.10.2026, план подтверждён Хозяином, №9):
 * база 7×7 LBP u2 (8, 2) и варианты по одному фактору – (1) сетка 5×5, (2) 9×9, (3) радиус 1, (4) (8, 1) + (8, 2) подряд.
 * Обработка как сейчас: корень гистограмм, Tan – Triggs для L1-tt-xs-F, MLDA с пулом FERET, пол λ̄ (c = 1), 1 − cos по всем
 * осям, зеркала и выбор модели по |r|. Режимы: pool-lbp – выбор варианта по пулу FERET (схема pool-rule без изменений);
 * feret-lbp – роль «межс.» feret-xs2; gt-lbp – GT межс. N = 6 и внутри N = 3. Порог П2/П3 – (Т2), рядом (Т0). База сверяется
 * побайтно со строками косинуса c = 1 pool-reg, feret-reg, gt-reg.
 *
 * @author ssv
 */
final class XBaseLbp {

    static final LbpFeature.Params[] VARS = {
        LbpFeature.Params.DEFAULT,
        new LbpFeature.Params(8, new double[] {2}, 5),
        new LbpFeature.Params(8, new double[] {2}, 9),
        new LbpFeature.Params(8, new double[] {1}, 7),
        new LbpFeature.Params(8, new double[] {1, 2}, 7)};
    static final String[] VNAME = {"база 7×7 (8, 2)", "(1) 5×5 (8, 2)", "(2) 9×9 (8, 2)", "(3) 7×7 (8, 1)", "(4) 7×7 (8, 1)+(8, 2)"};
    static final int BASE = 0;
    static final String SELECT = "xbase_pool_lbp_select.tsv";
    static final String CHOICE = "выбор по пулу";
    /** Блок оценок c = 1 (косинус) в файлах XBaseReg. */
    static final String REG_C1 = XBaseReg.cName(XBaseReg.BASE_C) + ", косинус";

    private XBaseLbp() {
    }

    /**
     * Правило выбора (№8 Жорика): база, если в пределах NEAR п. п. от лучшего; иначе среди вариантов в пределах NEAR – с
     * меньшей размерностью признака, при равной – меньший номер (база раньше).
     */
    static int choose(double[] crit, int[] dims) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        if (crit[BASE] <= best + XBaseRule.NEAR) return BASE;
        int pick = -1;
        for (int i = 0; i < crit.length; i++) {
            if (crit[i] > best + XBaseRule.NEAR) continue;
            if (pick < 0 || dims[i] < dims[pick]) pick = i;
        }
        if (pick < 0) throw new IllegalStateException("XBaseLbp: выбор не определён");
        return pick;
    }

    static int[] dims() {
        int[] d = new int[VARS.length];
        for (int v = 0; v < VARS.length; v++) d[v] = VARS[v].length();
        return d;
    }

    /** Признаки варианта: [0] без нормировки, [1] Tan – Triggs (только V.L1). */
    static XBaseEval[] vectors(Map<String, Mat> fr, IlluminationNorm.NormParams params, LbpFeature.Params lp) {
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, params, IlluminationNorm.NONE, lp);
        xs[1].vectors(fr, params, IlluminationNorm.TAN_TRIGGS, lp);
        return xs;
    }

    /** Строки одной модели: [евклид, косинус], сведения основной модели X / X+mirror и число осей. */
    record One(String[] rows, String[] info, int[] axes) {}

    /** Задачи одного блока (как XBaseReg.fitLines при одном c = 1) и место для сведений основной модели. */
    record Job(List<Callable<String[]>> tasks, String[] info, int[] axes) {}

    /** Блоки jobs – все задачи разом в threads потоках (порядок строк внутри блока – как в XBaseReg.fitLines). */
    static List<One> run(List<Job> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (Job j : jobs) flat.addAll(j.tasks());
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<One> out = new ArrayList<>();
            int i = 0;
            for (Job j : jobs) {
                StringBuilder[] b = {new StringBuilder(), new StringBuilder()};
                for (int k = 0; k < j.tasks().size(); k++) {
                    String[] r = fs.get(i++).get();
                    b[0].append(r[0]);
                    b[1].append(r[1]);
                }
                out.add(new One(new String[] {b[0].toString(), b[1].toString()}, j.info(), j.axes()));
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseLbp: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    /** Задачи блока – как XBaseReg.fitLines при одном c = 1 (MldaScorer() – прежний пол), тот же порядок задач. */
    static Job job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, int s,
                   List<List<Probe>> extra) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        String[] info = new String[2];
        int[] axes = new int[2];
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> {
                XBaseRule.Fit f = XBaseRule.fit(x, m, fullGallery, cohort, null);
                info[m ? 1 : 0] = f.lm.info();
                axes[m ? 1 : 0] = f.axes();
                return XBaseReg.lines(x, method, s, f.lm, own, mainProbes);
            });
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                tasks.add(() -> XBaseReg.lines(x, method, s, XBaseRule.fit(x, m, gk, cohort, null).lm, g, pr));
            }
        }
        return new Job(tasks, info, axes);
    }

    static One read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[] rows = new String[2];
        for (int r = 0; r < 2; r++) {
            rows[r] = cp.read(tag + "_" + r);
            if (rows[r] == null) return null;
        }
        String inf = cp.read(tag + "_info");
        if (inf == null) return null;
        String[] p = inf.split("\n");
        String[] info = new String[2];
        int[] axes = new int[2];
        for (int m = 0; m < 2; m++) {
            String[] f = p[m].split("\t", 2);
            axes[m] = Integer.parseInt(f[0]);
            info[m] = f[1];
        }
        return new One(rows, info, axes);
    }

    static void save(XBaseOther.Checkpoint cp, String tag, One l) throws IOException {
        for (int r = 0; r < 2; r++) cp.save(tag + "_" + r, l.rows()[r]);
        cp.save(tag + "_info", l.axes()[0] + "\t" + l.info()[0] + "\n" + l.axes()[1] + "\t" + l.info()[1] + "\n");
    }

    private static final Pattern HULL = Pattern.compile("в оболочке (\\d+) из (\\d+)");

    /** [заменено, ранг оболочки] из строки MLDA. */
    static int[] hull(String info) {
        Matcher m = HULL.matcher(info);
        if (!m.find()) throw new IllegalStateException("XBaseLbp: нет сведений об оболочке в «" + info + "»");
        return new int[] {Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2))};
    }

    static String infoRow(String head, int v, One l) {
        int[] a = hull(l.info()[0]);
        int[] b = hull(l.info()[1]);
        return String.format(Locale.ROOT, "%s | %s | %d | %d / %d | %d / %d | %d / %d%n", head, VNAME[v], VARS[v].length(), a[1], b[1], a[0], b[0], l.axes()[0],
                l.axes()[1]);
    }

    static final String INFO_HEAD = "\n=== Модели (основная модель): размерность признака | ранг оболочки X / X+mirror | заменено собственных значений S_w в "
            + "оболочке X / X+mirror | осей MLDA X / X+mirror ===\n";

    static final String NOTE = "Варианты LBP (по одному фактору от базы; кадр 92×112, корень гистограмм, нормировка блока на сумму 1): база – сетка 7×7, LBP u2 "
            + "(8, 2), 2891;\n(1) сетка 5×5, (8, 2), 1475; (2) сетка 9×9, (8, 2), 4779; (3) сетка 7×7, (8, 1), 2891; (4) сетка 7×7, гистограммы (8, 1) и (8, 2) "
            + "подряд, 5782.\nНа каждый вариант обучаются заново основные модели X / X+mirror и 5 моделей частей; MLDA с пулом FERET, пол λ̄ (c = 1), правило "
            + "оценки – 1 − cos\nдо центра класса по всем осям (решение (26)); решение – d₁/d₂, модель X / X+mirror по |r| пробы.\n";

    // ---------------------------------------------------------------- pool-lbp

    /** Кадры FERET (пул и при roles – роли) и когорты FEI, как XBaseRule.fsetup; не освобождаются. */
    static Map<String, Mat> frames(XBaseRule.FSetup s, FarExport.Loaded export, boolean roles) throws IOException {
        Set<String> cohortKeys = new HashSet<>();
        s.cohort.forEach(cohortKeys::addAll);
        Set<String> used = new TreeSet<>();
        if (roles) {
            XBaseFeret.Roles r = s.su.ro;
            for (Map<String, List<String>> m : List.of(r.gal1, r.pro1, r.g2, r.g2p, r.pro2, r.ctrlKeys)) m.values().forEach(used::addAll);
        }
        for (List<List<String>> ds : s.pool.values()) for (List<String> d : ds) used.addAll(d);
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(s.su.dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        return fr;
    }

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        List<String> setNames = new ArrayList<>(sets.keySet());
        Path regScores = s.out.resolve("xbase_pool_reg_scores.tsv.gz");
        if (!Files.exists(regScores)) throw new IllegalStateException("pool-lbp: для сверки базы нужен " + regScores + " (pool-reg)");
        Map<String, String> blocks = XBaseSubsp.blocks(regScores);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-lbp %s; seed %d; r0 %s; пул %d; половины %d/%d; варианты %s", s.su.dir,
                tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", VNAME)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_lbp_eq"), fp);
        Map<String, Mat> fr = frames(s, export, false);
        StringBuilder head = new StringBuilder();
        head.append("Ревизия 1.4, ступень 1: выбор параметров LBP по пулу FERET (XBaseEval pool-lbp)\n");
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule: GT – весь пул (выбор для gt-lbp); FERET-1 – половина B (%d; проход 1 feret-lbp); "
                + "FERET-2 – половина A (%d; проход 2);%n5 частей по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая "
                + "дата пула; обучение MLDA – галерея своих + когорта FEI 20 + люди 4 частей.%n", s.halves.get(1).size(), s.halves.get(0).size(),
                XBaseRule.PART_SEED_OFFSET, XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %% (порог по объединённым контрольным); база, "
                + "если в пределах %.0f п. п. от лучшего,%nиначе среди вариантов в пределах %.0f п. п. – с меньшей размерностью признака (при равной – база).%n",
                XBaseRule.NEAR, XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = pool-reg c = 1 (все строки блока, побайтно) ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("набор, признак, часть, смена | вариант | размерность | ранг оболочки | заменено | осей\n");
        // [набор][признак][вариант] → конфигурации по частям и сменам.
        List<List<List<List<FeiDecompose.Cfg>>>> cfgs = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            List<List<List<FeiDecompose.Cfg>>> a = new ArrayList<>();
            for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                List<List<FeiDecompose.Cfg>> b = new ArrayList<>();
                for (int v = 0; v < VARS.length; v++) b.add(new ArrayList<>());
                a.add(b);
            }
            cfgs.add(a);
        }
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        for (int v = 0; v < VARS.length; v++) {
            XBaseEval[] xv = null;
            for (int si = 0; si < setNames.size(); si++) {
                String set = setNames.get(si);
                Set<String> members = sets.get(set);
                List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
                for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                    // Блоки (часть, смена): сначала все описания, недостающие в контрольной точке – одной пачкой задач.
                    List<Object[]> blk = new ArrayList<>();
                    for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                        List<String> held = new ArrayList<>(parts.get(q));
                        Set<String> train = new TreeSet<>(members);
                        held.forEach(train::remove);
                        Collections.shuffle(held, new Random(tpl.seed + XBaseRule.HALF_SEED_OFFSET + q));
                        int h0 = (held.size() + 1) / 2;
                        List<List<String>> hv = List.of(new ArrayList<>(new TreeSet<>(held.subList(0, h0))), new ArrayList<>(new TreeSet<>(held.subList(h0, held.size()))));
                        Map<String, List<List<String>>> trainPool = new TreeMap<>();
                        for (String p : train) trainPool.put(p, s.pool.get(p));
                        List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), trainPool, Set.of(), Map.of()).extra();
                        for (int h = 0; h < 2; h++) {
                            List<String> own = hv.get(h);
                            List<String> ctrl = hv.get(1 - h);
                            XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                            List<List<String>> aug = new ArrayList<>(s.cohort);
                            aug.addAll(extra);
                            g.cohort = aug;
                            Map<String, List<String>> gal = new LinkedHashMap<>();
                            List<Probe> probes = new ArrayList<>();
                            for (int i = 0; i < own.size(); i++) {
                                List<List<String>> ds = s.pool.get(own.get(i));
                                gal.put(own.get(i), ds.get(0));
                                for (int d = 1; d < ds.size(); d++) for (String k : ds.get(d)) probes.add(new Probe(FeiScores.OWN, own.get(i), g.img(k), k, i));
                            }
                            for (String p : ctrl) for (List<String> d : s.pool.get(p)) for (String k : d) probes.add(new Probe(FeiScores.CTRL, p, g.img(k), k, -1));
                            String tag = String.format(Locale.ROOT, "v%d_%s_q%d_h%d_f%d", v, set, q, h, f);
                            blk.add(new Object[] {q, h, g, gal, probes, tag, read(cp, tag)});
                        }
                    }
                    List<Job> jobs = new ArrayList<>();
                    List<Object[]> todo = new ArrayList<>();
                    for (Object[] b : blk) {
                        if (b[6] != null) continue;
                        if (xv == null) xv = vectors(fr, tpl.params, VARS[v]);
                        @SuppressWarnings("unchecked")
                        Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                        @SuppressWarnings("unchecked")
                        List<Probe> probes = (List<Probe>) b[4];
                        jobs.add(job((XBaseEval.Gt) b[2], xv[f], gal, probes, null, 2 * (int) b[0] + (int) b[1], null));
                        todo.add(b);
                    }
                    if (!jobs.isEmpty()) {
                        long t0 = System.nanoTime();
                        List<One> done = run(jobs, tpl.threads);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        for (int i = 0; i < todo.size(); i++) {
                            todo.get(i)[6] = done.get(i);
                            save(cp, (String) todo.get(i)[5], done.get(i));
                        }
                        System.out.printf(Locale.ROOT, "pool-lbp: v%d %s f%d ready, %d blocks (%.0f s)%n", v, set, f, jobs.size(), sec);
                    }
                    for (Object[] b : blk) {
                        int q = (int) b[0];
                        int h = (int) b[1];
                        XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                        @SuppressWarnings("unchecked")
                        Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                        @SuppressWarnings("unchecked")
                        List<Probe> probes = (List<Probe>) b[4];
                        One ls = (One) b[6];
                        String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                        if (v == BASE) check.append(XBaseReg.check(bp + REG_C1, ls.rows()[1], blocks, regScores));
                        info.append(infoRow(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], q, h), v, ls));
                        List<List<String>> noParts = new ArrayList<>();
                        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
                        scores.append("# ").append(bp).append(VNAME[v]).append(", косинус\n").append(ls.rows()[1]);
                        cfgs.get(si).get(f).get(v).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls.rows()[1]).cfg.get(V.L1));
                    }
                }
            }
        }
        for (Mat m : fr.values()) m.release();
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseLbp pool-lbp): набор, признак, номер варианта, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        int[] dims = dims();
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("вариант | размерность | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным "
                    + "лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                double[] crit = new double[VARS.length];
                for (int v = 0; v < VARS.length; v++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(v))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[v] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %d | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", VNAME[v], dims[v], XBaseRule.FEATS[f],
                            XBaseRule.fmt(sp.err[0]), XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[v]),
                            XBaseRule.fmt(sp.eer[3]), sp.medOwn, sp.medCtrl));
                }
                int c = choose(crit, dims);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], VNAME[c],
                        crit[c], crit[BASE], java.util.Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(VNAME[c]).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check).append(info)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body);
        Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_lbp_scores.tsv.gz"), "# Выбор параметров LBP по пулу FERET (XBaseLbp pool-lbp), только локально\n# Код: "
                + "коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_lbp.txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_lbp_gt.txt"), new StringBuilder(head).append(gtBody));
        System.out.print(t);
        cp.delete();
    }

    /** Выбор по пулу: набор → признак → номер варианта. */
    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseLbp: нет файла выбора " + f + " (сначала pool-lbp)");
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int v = Integer.parseInt(p[2]);
            if (v < 0 || v >= VARS.length || !VNAME[v].equals(p[3])) throw new IllegalStateException("XBaseLbp: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    // ---------------------------------------------------------------- таблицы на своих

    /** Как XBaseReg.tables: Res (Т2) (у «внутри» – (Т0)) для таблиц разделимости, пары «вариант − база», способы (Т2) и (Т0). */
    static void tables(StringBuilder t, String title, Map<String, XBaseThr.Out> outs, int persons, XBaseEval.Gt tpl, List<String[]> pairs, boolean t2) {
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        String way = t2 ? XBaseThr.WAYS[2] : XBaseThr.WAYS[0];
        for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) res.put(e.getKey(), e.getValue().res.get(way));
        XBaseRule.tables(t, title + "; П3 в первой таблице – по «чужим» " + way, res, persons, tpl, pairs);
        if (tpl.b > 0) {
            t.append("\n=== Парные разности «вариант − база» при порогах (Т2) и (Т0), П3 и П2, п. п. ===\n");
            t.append("пара | способ | порог | ").append(LbpNormReport.DIFF_HEAD).append('\n');
            for (String[] p : pairs) {
                for (String w : t2 ? new String[] {XBaseThr.WAYS[2], XBaseThr.WAYS[0]} : new String[] {XBaseThr.WAYS[0]}) {
                    FeiThreshold.Res a = outs.get(p[0]).res.get(w);
                    FeiThreshold.Res z = outs.get(p[1]).res.get(w);
                    for (int m = 3; m >= 2; m--) t.append(LbpNormReport.diffRow(List.of(p[0] + " − " + p[1], w, XBaseThr.M_NAME[m]), a, z, m));
                }
            }
        }
        Map<String, XBaseThr.Out> shown = new LinkedHashMap<>();
        for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) {
            XBaseThr.Out o = e.getValue();
            o.res.remove(XBaseThr.WAYS[1]);
            o.imp1.clear();
            shown.put(e.getKey(), o);
        }
        XBaseThr.tables(t, title + ": способы порога (Т2) и (Т0)", shown, persons, tpl);
    }

    // ---------------------------------------------------------------- feret-lbp

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        XBaseFeret.Roles ro = XBaseFeret.equalize(s.su.ro, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Path regScores = s.out.resolve("xbase_feret_reg_scores.tsv.gz");
        Path regTxt = s.out.resolve("xbase_feret_reg.txt");
        if (!Files.exists(regScores) || !Files.exists(regTxt)) throw new IllegalStateException("feret-lbp: нужны " + regScores + " и " + regTxt + " (feret-reg)");
        Map<String, String> blocks = XBaseSubsp.blocks(regScores);
        List<String> report = Files.readAllLines(regTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-lbp %s; seed %d; r0 %s; пул %d; половины %d/%d; варианты %s", s.su.dir,
                tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", VNAME)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_lbp_eq"), fp);
        Map<String, Mat> fr = frames(s, export, true);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.4, ступень 1: параметры LBP, FERET межсессионно (XBaseEval feret-lbp): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("Роль «межс.» feret-xs2 (как feret-rule, feret-thr, feret-reg): половины №2, проход 1 – свои A, пул проекции – B, проход 2 – наоборот; "
                + "контрольные 200.\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        t.append("Порог – по «чужим» (Т2), рядом (Т0). Сетка вариантов на своих – диагностика; «выбор по пулу» – вариант, выбранный pool-lbp\n"
                + "(проход 1 – FERET-1, проход 2 – FERET-2), прогон на своих – один.\n");
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для обоих проходов, всех строк и способов: B = %d, "
                + "seed = %d.%n", persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = feret-reg c = 1 (все роли, побайтно); (Т0), (Т2) П3 FAR 1 % = отчёт "
                + "feret-reg ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("проход, признак | вариант | размерность | ранг оболочки | заменено | осей\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder lines = new StringBuilder();
        // [признак][вариант][проход]
        XBaseThr.Unit[][][] units = new XBaseThr.Unit[XBaseRule.FEATS.length][VARS.length][2];
        Object[][] passes = new Object[2][];
        for (int ps = 0; ps < 2; ps++) {
            Set<String> mine = s.halves.get(ps);
            List<String> own = ro.own1.stream().filter(mine::contains).toList();
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            for (String p : own) {
                gal.put(p, ro.gal1.get(p));
                pro.put(p, ro.pro1.get(p));
            }
            Map<String, List<List<String>>> projPool = new TreeMap<>();
            for (String p : s.halves.get(1 - ps)) projPool.put(p, s.pool.get(p));
            XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
            List<List<String>> aug = new ArrayList<>(s.cohort);
            aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
            g.cohort = aug;
            List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S);
            passes[ps] = new Object[] {g, gal, pro, parts, XBaseEval.probes(g, pro), XBaseThr.extra(g, pro, parts)};
        }
        double fitS = 0;
        for (int v = 0; v < VARS.length; v++) {
            // Блоки варианта (признак, проход): недостающие в контрольной точке – одной пачкой задач.
            One[][] got = new One[XBaseRule.FEATS.length][2];
            List<Job> jobs = new ArrayList<>();
            List<int[]> todo = new ArrayList<>();
            XBaseEval[] xv = null;
            for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                for (int ps = 0; ps < 2; ps++) {
                    got[f][ps] = read(cp, String.format(Locale.ROOT, "v%d_p%d_f%d", v, ps, f));
                    if (got[f][ps] != null) continue;
                    if (xv == null) xv = vectors(fr, tpl.params, VARS[v]);
                    Object[] pz = passes[ps];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) pz[1];
                    @SuppressWarnings("unchecked")
                    List<List<String>> parts = (List<List<String>>) pz[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) pz[4];
                    @SuppressWarnings("unchecked")
                    List<List<Probe>> extra = (List<List<Probe>>) pz[5];
                    jobs.add(job((XBaseEval.Gt) pz[0], xv[f], gal, probes, parts, 0, extra));
                    todo.add(new int[] {f, ps});
                }
            }
            if (!jobs.isEmpty()) {
                long t0 = System.nanoTime();
                List<One> done = run(jobs, tpl.threads);
                double sec = (System.nanoTime() - t0) / 1e9;
                fitS += sec;
                for (int i = 0; i < todo.size(); i++) {
                    int[] fp2 = todo.get(i);
                    got[fp2[0]][fp2[1]] = done.get(i);
                    save(cp, String.format(Locale.ROOT, "v%d_p%d_f%d", v, fp2[1], fp2[0]), done.get(i));
                }
                System.out.printf(Locale.ROOT, "feret-lbp: %s ready, %d blocks (%.0f s)%n", VNAME[v], jobs.size(), sec);
            }
            for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                for (int ps = 0; ps < 2; ps++) {
                    Object[] pz = passes[ps];
                    XBaseEval.Gt g = (XBaseEval.Gt) pz[0];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) pz[1];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> pro = (Map<String, List<String>>) pz[2];
                    @SuppressWarnings("unchecked")
                    List<List<String>> parts = (List<List<String>>) pz[3];
                    One ls = got[f][ps];
                    String bp = "проход " + (ps + 1) + ", " + XBaseRule.FEATS[f] + ", ";
                    if (v == BASE) check.append(XBaseReg.check(bp + REG_C1, ls.rows()[1], blocks, regScores));
                    info.append(infoRow("проход " + (ps + 1) + ", " + XBaseRule.FEATS[f], v, ls));
                    lines.append("# ").append(bp).append(VNAME[v]).append(", косинус\n").append(ls.rows()[1]);
                    units[f][v][ps] = new XBaseThr.Unit(g, gal, pro, parts, ls.rows()[1], null, true);
                }
            }
        }
        for (Mat m : fr.values()) m.release();
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        for (int f = 0; f < XBaseRule.FEATS.length; f++) {
            String bl = "межс. " + XBaseRule.FEATS[f] + " " + VNAME[BASE];
            List<XBaseThr.Unit> chosen = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                int c = select.get("FERET-" + (ps + 1)).get(XBaseRule.FEATS[f]);
                sel.append(String.format(Locale.ROOT, "%s проход %d: %s%n", XBaseRule.FEATS[f], ps + 1, VNAME[c]));
                chosen.add(units[f][c][ps]);
            }
            for (int v = 0; v <= VARS.length; v++) {
                String l = "межс. " + XBaseRule.FEATS[f] + " " + (v < VARS.length ? VNAME[v] : CHOICE);
                XBaseThr.Out o = XBaseThr.evaluate(l, v < VARS.length ? List.of(units[f][v]) : chosen, persons.size(), tpl);
                if (v == BASE) {
                    String rl = "межс. " + XBaseRule.FEATS[f] + " " + XBaseReg.cName(XBaseReg.BASE_C);
                    for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), report, tpl.b, regTxt));
                }
                outs.put(l, o);
                if (v != BASE) pairs.add(new String[] {l, bl});
            }
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        tables(t, "FERET межсессионно (роль «межс.» feret-xs2, оба прохода)", outs, persons.size(), tpl, pairs, true);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_lbp_scores.tsv.gz"), "# FERET, параметры LBP (XBaseLbp feret-lbp), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_lbp.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-lbp

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-lbp: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Path regScores = out.resolve("xbase_gt_reg_scores.tsv.gz");
        Path regTxt = out.resolve("xbase_gt_reg.txt");
        if (!Files.exists(regScores) || !Files.exists(regTxt)) throw new IllegalStateException("gt-lbp: нужны " + regScores + " и " + regTxt + " (gt-reg)");
        Map<String, String> blocks = XBaseSubsp.blocks(regScores);
        List<String> report = Files.readAllLines(regTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-lbp; seed %d; r0 %s; пул FERET %d; варианты %s", g.seed, g.r0, pool.size(),
                String.join("; ", VNAME)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_lbp"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.4, ступень 1: параметры LBP, GT (XBaseEval gt-lbp)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F – Color "
                + "FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-rule, gt-thr, gt-reg: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6; внутри сессии N = 3 – контроль "
                + "(Δ = 3 п. п.), порог только (Т0).\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        t.append("Порог – по «чужим» (Т2), рядом (Т0). Сетка вариантов на своих – диагностика; «выбор по пулу» – вариант, выбранный pool-lbp\n"
                + "(набор GT – весь пул FERET), прогон на своих – один.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET,
                g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = gt-reg c = 1 (все роли, побайтно); (Т0), (Т2) П3 FAR 1 % = отчёт gt-reg "
                + "===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("режим, признак | вариант | размерность | ранг оболочки | заменено | осей\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (набор GT) ===\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double fitS = 0;
        List<List<String>> cohort0 = g.cohort;
        // [режим][признак][вариант] → строки косинуса.
        String[][][] rows = new String[modes.length][XBaseRule.FEATS.length][VARS.length];
        try {
            List<List<String>> aug = new ArrayList<>(cohort0);
            aug.addAll(poolCls);
            g.cohort = aug;
            for (int v = 0; v < VARS.length; v++) {
                // Блоки варианта (режим, признак): недостающие в контрольной точке – одной пачкой задач.
                One[][] got = new One[modes.length][XBaseRule.FEATS.length];
                List<Job> jobs = new ArrayList<>();
                List<int[]> todo = new ArrayList<>();
                XBaseEval[] xv = null;
                for (int mi = 0; mi < modes.length; mi++) {
                    boolean x = mi == 0;
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                    List<Probe> probes = XBaseEval.probes(g, pro);
                    for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                        got[mi][f] = read(cp, "v" + v + "_" + (x ? "x" : "i") + "_f" + f);
                        if (got[mi][f] != null) continue;
                        if (xv == null) xv = vectors(fr, g.params, VARS[v]);
                        jobs.add(job(g, xv[f], gal, probes, parts, 0, x ? XBaseThr.extra(g, pro, parts) : null));
                        todo.add(new int[] {mi, f});
                    }
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<One> done = run(jobs, g.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        int[] mf = todo.get(i);
                        got[mf[0]][mf[1]] = done.get(i);
                        save(cp, "v" + v + "_" + (mf[0] == 0 ? "x" : "i") + "_f" + mf[1], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "gt-lbp: %s ready, %d blocks (%.0f s)%n", VNAME[v], jobs.size(), sec);
                }
                for (int mi = 0; mi < modes.length; mi++) {
                    String mode = (String) modes[mi][0];
                    for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                        One ls = got[mi][f];
                        String bp = mode + ", " + XBaseRule.FEATS[f] + ", ";
                        if (v == BASE) check.append(XBaseReg.check(bp + REG_C1, ls.rows()[1], blocks, regScores));
                        info.append(infoRow(mode + ", " + XBaseRule.FEATS[f], v, ls));
                        lines.append("# ").append(bp).append(VNAME[v]).append(", косинус\n").append(ls.rows()[1]);
                        rows[mi][f][v] = ls.rows()[1];
                    }
                }
            }
            for (Mat m : fr.values()) m.release();
            for (int mi = 0; mi < modes.length; mi++) {
                String mode = (String) modes[mi][0];
                boolean x = mi == 0;
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                    int c = select.get("GT").get(XBaseRule.FEATS[f]);
                    if (x) sel.append(String.format(Locale.ROOT, "%s: %s%n", XBaseRule.FEATS[f], VNAME[c]));
                    String bl = mode + " " + XBaseRule.FEATS[f] + " " + VNAME[BASE];
                    for (int v = 0; v <= VARS.length; v++) {
                        String b0 = rows[mi][f][v < VARS.length ? v : c];
                        String l = mode + " " + XBaseRule.FEATS[f] + " " + (v < VARS.length ? VNAME[v] : CHOICE);
                        XBaseThr.Out o = XBaseThr.evaluate(l, List.of(new XBaseThr.Unit(g, gal, pro, parts, b0, null, x)), g.persons.size(), g);
                        if (v == BASE) {
                            String rl = mode + " " + XBaseRule.FEATS[f] + " " + XBaseReg.cName(XBaseReg.BASE_C);
                            for (int w : x ? new int[] {0, 2} : new int[] {0}) {
                                check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), report, g.b, regTxt));
                            }
                        }
                        outs.put(l, o);
                        if (v != BASE) pairs.add(new String[] {l, bl});
                    }
                }
                tables(body, "GT " + mode, outs, g.persons.size(), g, pairs, x);
            }
        } finally {
            g.cohort = cohort0;
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_lbp_scores.tsv.gz"), "# GT, параметры LBP (XBaseLbp gt-lbp)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_lbp.txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
