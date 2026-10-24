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
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Регуляризация MLDA (ревизия 1.3; задание №6 Жорика от 24.10.2026, план подтверждён Хозяином, №7): пол собственных
 * значений S_w λ̄·c, c ∈ {0,3; 1; 3; 10} (c = 1 – прежнее правило), правило оценки – 1 − cos по всем осям (решение (26)).
 * На каждое c модели обучаются заново (MldaScorer(c)). Режимы: pool-reg – выбор c по пулу FERET перекрёстно по людям
 * (схема pool-rule без изменений); feret-reg – роль «межс.» feret-xs2; gt-reg – GT межс. N = 6 и внутри N = 3. Порог П2/П3 –
 * по «чужим» (Т2), рядом (Т0) (XBaseThr). При c = 1 строки сверяются побайтно с pool-rule, feret-thr, gt-thr.
 *
 * @author ssv
 */
final class XBaseReg {

    static final double[] CS = {0.3, 1, 3, 10};
    static final int BASE_C = 1;
    /** Строки на одно c: 0 – евклид по всем осям, 1 – косинус по всем осям. */
    static final String[] ROW = {"(0) база", "(б) косинус, все оси"};
    static final String SELECT = "xbase_pool_reg_select.tsv";
    static final String CHOICE = "выбор по пулу";

    private XBaseReg() {
    }

    static String cName(int ci) {
        return "c = " + String.valueOf(CS[ci]).replace(".0", "").replace('.', ',');
    }

    /**
     * Правило выбора (№6 Жорика): c = 1, если в пределах NEAR п. п. от лучшего; иначе среди c в пределах NEAR – наименьшее
     * |ln c|, при равенстве – меньшее c.
     */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        if (crit[BASE_C] <= best + XBaseRule.NEAR) return BASE_C;
        int pick = -1;
        for (int i = 0; i < CS.length; i++) {
            if (crit[i] > best + XBaseRule.NEAR) continue;
            if (pick < 0) {
                pick = i;
                continue;
            }
            double a = Math.abs(Math.log(CS[i]));
            double z = Math.abs(Math.log(CS[pick]));
            if (a < z || a == z && CS[i] < CS[pick]) pick = i;
        }
        if (pick < 0) throw new IllegalStateException("XBaseReg: выбор не определён");
        return pick;
    }

    /** Строки модели: евклид и косинус по всем осям (как XBaseRule.Fit.scores для (0) и (б)). */
    static String[] lines(XBaseEval x, String method, int s, LdaModel lm, List<String> gallery, List<Probe> probes) {
        StringBuilder[] out = {new StringBuilder(), new StringBuilder()};
        double[][] c = lm.centers();
        for (Probe p : probes) {
            double[] y = lm.embed(x.vec.get(V.L1).get(p.key()));
            double[][] sc = new double[2][c.length];
            for (int g = 0; g < c.length; g++) {
                if (c[g] == null) {
                    sc[0][g] = Double.POSITIVE_INFINITY;
                    sc[1][g] = Double.POSITIVE_INFINITY;
                    continue;
                }
                sc[0][g] = Math.sqrt(NearestVectorScorer.distance2(y, c[g]));
                sc[1][g] = XBaseRule.cos(y, c[g], y.length, null);
            }
            for (int r = 0; r < 2; r++) out[r].append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc[r], false));
        }
        return new String[] {out[0].toString(), out[1].toString()};
    }

    /** Строки по c: [c][строка] и сведения основных моделей [c][X, X+mirror]: осей и строка MLDA. */
    record Lines(String[][] byC, String[][] info, int[][] axes) {}

    /**
     * Как XBaseRule.fitLines (тот же порядок задач), для всех c: основная модель и модели без части q, оба зеркала; parts ==
     * null – только основная модель (выбор по пулу); extra – пробы моделей без части q (роль od q, XBaseThr), null – нет.
     */
    static Lines fitLines(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, int s,
                          List<List<Probe>> extra) {
        List<String> own = gt.own;
        List<List<String>> cohort = gt.cohort;
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        String[][] info = new String[CS.length][2];
        int[][] axes = new int[CS.length][2];
        List<List<Callable<String[]>>> tasks = new ArrayList<>();
        for (int ci = 0; ci < CS.length; ci++) {
            double c = CS[ci];
            int cc = ci;
            List<Callable<String[]>> tc = new ArrayList<>();
            for (boolean m : new boolean[] {false, true}) {
                String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
                tc.add(() -> {
                    XBaseRule.Fit f = XBaseRule.fit(x, m, fullGallery, cohort, null, c);
                    info[cc][m ? 1 : 0] = f.lm.info();
                    axes[cc][m ? 1 : 0] = f.axes();
                    return lines(x, method, s, f.lm, own, mainProbes);
                });
                if (parts == null) continue;
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    Set<String> part = new HashSet<>(parts.get(q));
                    List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                    List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                    List<Probe> pr = new ArrayList<>();
                    for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                    if (extra != null) pr.addAll(extra.get(q));
                    tc.add(() -> lines(x, method, s, XBaseRule.fit(x, m, gk, cohort, null, c).lm, g, pr));
                }
            }
            tasks.add(tc);
        }
        List<Callable<String[]>> flat = new ArrayList<>();
        tasks.forEach(flat::addAll);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(gt.threads);
        String[][] byC = new String[CS.length][2];
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            int i = 0;
            for (int ci = 0; ci < CS.length; ci++) {
                StringBuilder[] b = {new StringBuilder(), new StringBuilder()};
                for (int k = 0; k < tasks.get(ci).size(); k++) {
                    String[] r = fs.get(i++).get();
                    b[0].append(r[0]);
                    b[1].append(r[1]);
                }
                byC[ci] = new String[] {b[0].toString(), b[1].toString()};
            }
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseReg: " + e, e);
        } finally {
            ex.shutdown();
        }
        return new Lines(byC, info, axes);
    }

    static Lines read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[][] byC = new String[CS.length][2];
        String[][] info = new String[CS.length][2];
        int[][] axes = new int[CS.length][2];
        for (int ci = 0; ci < CS.length; ci++) {
            for (int r = 0; r < 2; r++) {
                byC[ci][r] = cp.read(tag + "_" + ci + "_" + r);
                if (byC[ci][r] == null) return null;
            }
            String inf = cp.read(tag + "_" + ci + "_info");
            if (inf == null) return null;
            String[] p = inf.split("\n");
            for (int m = 0; m < 2; m++) {
                String[] f = p[m].split("\t", 2);
                axes[ci][m] = Integer.parseInt(f[0]);
                info[ci][m] = f[1];
            }
        }
        return new Lines(byC, info, axes);
    }

    static void save(XBaseOther.Checkpoint cp, String tag, Lines l) throws IOException {
        for (int ci = 0; ci < CS.length; ci++) {
            for (int r = 0; r < 2; r++) cp.save(tag + "_" + ci + "_" + r, l.byC()[ci][r]);
            cp.save(tag + "_" + ci + "_info", l.axes()[ci][0] + "\t" + l.info()[ci][0] + "\n" + l.axes()[ci][1] + "\t" + l.info()[ci][1] + "\n");
        }
    }

    static String infoRow(String head, Lines l) {
        StringBuilder t = new StringBuilder();
        for (int ci = 0; ci < CS.length; ci++) {
            t.append(String.format(Locale.ROOT, "%s | %s | %d / %d | %s | %s%n", head, cName(ci), l.axes()[ci][0], l.axes()[ci][1], XBaseRule.replaced(l.info()[ci][0]),
                    XBaseRule.replaced(l.info()[ci][1])));
        }
        return t.toString();
    }

    static final String INFO_HEAD = "\n=== Модели (основная модель): осей MLDA X / X+mirror | заменено собственных значений S_w (X) | (X+mirror) ===\n";

    static final String NOTE = "Пол собственных значений S_w в MLDA – λ̄·c, λ̄ = tr(S_w)/n (n – длина вектора L1), c ∈ {0,3; 1; 3; 10}; c = 1 – прежнее правило.\n"
            + "На каждое c обучаются заново основные модели X / X+mirror и 5 моделей частей; правило оценки – 1 − cos до центра класса по всем осям\n"
            + "MLDA (решение (26)); решение – d₁/d₂, модель X / X+mirror по |r| пробы.\n";

    /** Сравнение блока с прежним файлом оценок: все строки, отсортированные. */
    static String check(String bn, String lines, Map<String, String> blocks, Path src) {
        String old = blocks.get(bn);
        if (old == null) throw new IllegalStateException("XBaseReg: нет блока «" + bn + "» в " + src);
        List<String> a = XBaseThr.sorted(old, false);
        if (!a.equals(XBaseThr.sorted(lines, false))) throw new IllegalStateException("XBaseReg: строки не совпали с " + src + ": " + bn);
        return String.format(Locale.ROOT, "%s: совпало (%d строк)%n", bn, a.size());
    }

    /** Ячейки способа way при П3 FAR 1 % – как строка сводки XBaseThr (до столбца «держит»). */
    static String checkThr(String label, String way, FeiThreshold.Res r, List<String> report, int b, Path src) {
        int x = (int) r.far[3][0][0];
        int n = (int) r.far[3][0][1];
        String row = String.format(Locale.ROOT, "%s | %s | %s | %.0f %% | %s | %.2f %%; КП %d/%d ↑%.2f %% |", label, way, "П3", 1.0, FeiThreshold.errCell(r, 3, 0, 0, b),
                FeiThreshold.farPct(r, 3, 0), x, n, 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95));
        for (String l : report) if (l.startsWith(row)) return String.format(Locale.ROOT, "%s %s П3 FAR 1 %%: совпало%n", label, way);
        throw new IllegalStateException("XBaseReg: нет строки «" + row + "» в " + src);
    }

    // ---------------------------------------------------------------- pool-reg

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        XBaseEval.Gt tpl = s.tpl;
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        Path ruleScores = s.out.resolve("xbase_pool_rule_scores.tsv.gz");
        if (!Files.exists(ruleScores)) throw new IllegalStateException("pool-reg: для сверки c = 1 нужен " + ruleScores + " (pool-rule)");
        Map<String, String> blocks = XBaseSubsp.blocks(ruleScores);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-reg %s; seed %d; r0 %s; пул %d; половины %d/%d; c %s", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), java.util.Arrays.toString(CS)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_reg_eq"), fp);
        StringBuilder head = new StringBuilder();
        head.append("Ревизия 1.3: выбор пола MLDA λ̄·c по пулу FERET (XBaseEval pool-reg)\n");
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule: GT – весь пул (выбор для gt-reg); FERET-1 – половина B (%d; проход 1 feret-reg); "
                + "FERET-2 – половина A (%d; проход 2);%n5 частей по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая "
                + "дата пула; обучение MLDA – галерея своих + когорта FEI 20 + люди 4 частей.%n", s.halves.get(1).size(), s.halves.get(0).size(),
                XBaseRule.PART_SEED_OFFSET, XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %% (порог по объединённым контрольным); c = 1, "
                + "если в пределах %.0f п. п. от лучшего,%nиначе среди c в пределах %.0f п. п. – наименьшее |ln c| (при равенстве – меньшее c).%n", XBaseRule.NEAR,
                XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: c = 1, строки (0) и (б) косинус = pool-rule (все строки блока, побайтно) ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("набор, признак, часть, смена | c | осей | заменено (X) | заменено (X+mirror)\n");
        StringBuilder body = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseReg pool-reg): набор, признак, c\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder gtBody = new StringBuilder();
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        for (Map.Entry<String, Set<String>> se : sets.entrySet()) {
            String set = se.getKey();
            List<List<String>> parts = XBaseRule.parts5(se.getValue(), tpl.seed + XBaseRule.PART_SEED_OFFSET);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(se.getValue().size()).append(" человек) ########\n");
            tb.append("c | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                List<List<FeiDecompose.Cfg>> cfgs = new ArrayList<>();
                for (int ci = 0; ci < CS.length; ci++) cfgs.add(new ArrayList<>());
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    List<String> held = new ArrayList<>(parts.get(q));
                    Set<String> train = new TreeSet<>(se.getValue());
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
                        String tag = String.format(Locale.ROOT, "%s_q%d_h%d_f%d", set, q, h, f);
                        Lines ls = read(cp, tag);
                        if (ls == null) {
                            long t0 = System.nanoTime();
                            ls = fitLines(g, s.xs[f], gal, probes, null, 2 * q + h, null);
                            double sec = (System.nanoTime() - t0) / 1e9;
                            fitS += sec;
                            save(cp, tag, ls);
                            System.out.printf(Locale.ROOT, "pool-reg: %s ready (%.0f s)%n", tag, sec);
                        }
                        String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                        for (int r = 0; r < 2; r++) check.append(check(bp + ROW[r], ls.byC()[BASE_C][r], blocks, ruleScores));
                        info.append(infoRow(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], q, h), ls));
                        List<List<String>> noParts = new ArrayList<>();
                        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
                        for (int ci = 0; ci < CS.length; ci++) {
                            scores.append("# ").append(bp).append(cName(ci)).append(", косинус\n").append(ls.byC()[ci][1]);
                            cfgs.get(ci).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls.byC()[ci][1]).cfg.get(V.L1));
                        }
                    }
                }
                double[] crit = new double[CS.length];
                for (int ci = 0; ci < CS.length; ci++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(ci))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[ci] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", cName(ci), XBaseRule.FEATS[f], XBaseRule.fmt(sp.err[0]),
                            XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[ci]), XBaseRule.fmt(sp.eer[3]), sp.medOwn,
                            sp.medCtrl));
                }
                int c = choose(crit);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; c = 1 – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], cName(c),
                        crit[c], crit[BASE_C], java.util.Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(CS[c]).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check).append(info)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body);
        Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_reg_scores.tsv.gz"), "# Выбор пола MLDA по пулу FERET (XBaseReg pool-reg), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_reg.txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_reg_gt.txt"), new StringBuilder(head).append(gtBody));
        System.out.print(t);
        cp.delete();
    }

    /** Выбор по пулу: набор → признак → индекс c. */
    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseReg: нет файла выбора " + f + " (сначала pool-reg)");
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            double c = Double.parseDouble(p[2]);
            int ci = -1;
            for (int i = 0; i < CS.length; i++) if (CS[i] == c) ci = i;
            if (ci < 0) throw new IllegalStateException("XBaseReg: c " + c + " не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], ci);
        }
        return out;
    }

    // ---------------------------------------------------------------- таблицы на своих

    /** Группа строк: подпись → проходы по c; Res (Т2) для таблиц разделимости и XBaseThr.Out для способов. */
    static void tables(StringBuilder t, String title, Map<String, XBaseThr.Out> outs, int persons, XBaseEval.Gt tpl, List<String[]> pairs, boolean t2) {
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        String way = t2 ? XBaseThr.WAYS[2] : XBaseThr.WAYS[0];
        for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) res.put(e.getKey(), e.getValue().res.get(way));
        XBaseRule.tables(t, title + "; П3 в первой таблице – по «чужим» " + way, res, persons, tpl, pairs);
        if (tpl.b > 0) {
            t.append("\n=== Парные разности «строка − база (c = 1)» при порогах (Т2) и (Т0), П3 и П2, п. п. ===\n");
            t.append("пара | способ | порог | ").append(LbpNormReport.DIFF_HEAD).append('\n');
            for (String[] p : pairs) {
                for (String w : t2 ? new String[] {XBaseThr.WAYS[2], XBaseThr.WAYS[0]} : new String[] {XBaseThr.WAYS[0]}) {
                    FeiThreshold.Res a = outs.get(p[0]).res.get(w);
                    FeiThreshold.Res z = outs.get(p[1]).res.get(w);
                    for (int m = 3; m >= 2; m--) t.append(LbpNormReport.diffRow(List.of(p[0] + " − " + p[1], w, XBaseThr.M_NAME[m]), a, z, m));
                }
            }
        }
        // Способы (Т2) и (Т0) рядом: сводка, группы, FAR, квантили (XBaseThr); (Т1) – не показывается.
        Map<String, XBaseThr.Out> shown = new LinkedHashMap<>();
        for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) {
            XBaseThr.Out o = e.getValue();
            o.res.remove(XBaseThr.WAYS[1]);
            o.imp1.clear();
            shown.put(e.getKey(), o);
        }
        XBaseThr.tables(t, title + ": способы порога (Т2) и (Т0)", shown, persons, tpl);
    }

    // ---------------------------------------------------------------- feret-reg

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        XBaseEval.Gt tpl = s.tpl;
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        XBaseFeret.Roles ro = XBaseFeret.equalize(s.su.ro, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Path thrScores = s.out.resolve("xbase_feret_thr_scores.tsv.gz");
        Path thrTxt = s.out.resolve("xbase_feret_thr.txt");
        if (!Files.exists(thrScores) || !Files.exists(thrTxt)) throw new IllegalStateException("feret-reg: нужны " + thrScores + " и " + thrTxt + " (feret-thr)");
        Map<String, String> blocks = XBaseSubsp.blocks(thrScores);
        List<String> report = Files.readAllLines(thrTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-reg %s; seed %d; r0 %s; пул %d; половины %d/%d; c %s", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), java.util.Arrays.toString(CS)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_reg_eq"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.3: регуляризация MLDA (пол λ̄·c), FERET межсессионно (XBaseEval feret-reg): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("Роль «межс.» feret-xs2 (как feret-rule, feret-thr): половины №2, проход 1 – свои A, пул проекции – B, проход 2 – наоборот; контрольные 200.\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        t.append("Порог – по «чужим» (Т2) (решение Жорика 24.10.2026), рядом (Т0). Сетка c на своих – диагностика; «выбор по пулу» – c, выбранное pool-reg\n"
                + "(проход 1 – FERET-1, проход 2 – FERET-2), прогон на своих – один.\n");
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для обоих проходов, всех строк и способов: B = %d, "
                + "seed = %d.%n", persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: c = 1 – строки = feret-thr (все роли, побайтно); (Т0), (Т2) П3 FAR 1 % = отчёт feret-thr ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("проход, признак | c | осей | заменено (X) | заменено (X+mirror)\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        double fitS = 0;
        for (int f = 0; f < XBaseRule.FEATS.length; f++) {
            List<List<XBaseThr.Unit>> units = new ArrayList<>();
            for (int ci = 0; ci <= CS.length; ci++) units.add(new ArrayList<>());
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
                List<Probe> probes = XBaseEval.probes(g, pro);
                String tag = String.format(Locale.ROOT, "p%d_f%d", ps, f);
                Lines ls = read(cp, tag);
                if (ls == null) {
                    long t0 = System.nanoTime();
                    ls = fitLines(g, s.xs[f], gal, probes, parts, 0, XBaseThr.extra(g, pro, parts));
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    save(cp, tag, ls);
                    System.out.printf(Locale.ROOT, "feret-reg: pass %d, %s ready (%.0f s)%n", ps + 1, XBaseRule.FEATS[f], sec);
                }
                String bp = "проход " + (ps + 1) + ", " + XBaseRule.FEATS[f] + ", ";
                for (int r = 0; r < 2; r++) check.append(check(bp + ROW[r], ls.byC()[BASE_C][r], blocks, thrScores));
                info.append(infoRow("проход " + (ps + 1) + ", " + XBaseRule.FEATS[f], ls));
                int c = select.get("FERET-" + (ps + 1)).get(XBaseRule.FEATS[f]);
                sel.append(String.format(Locale.ROOT, "%s проход %d: %s%n", XBaseRule.FEATS[f], ps + 1, cName(c)));
                for (int ci = 0; ci < CS.length; ci++) {
                    lines.append("# ").append(bp).append(cName(ci)).append(", косинус\n").append(ls.byC()[ci][1]);
                    units.get(ci).add(new XBaseThr.Unit(g, gal, pro, parts, ls.byC()[ci][1], null, true));
                }
                units.get(CS.length).add(new XBaseThr.Unit(g, gal, pro, parts, ls.byC()[c][1], null, true));
            }
            for (int ci = 0; ci <= CS.length; ci++) {
                String l = "межс. " + XBaseRule.FEATS[f] + " " + (ci < CS.length ? cName(ci) : CHOICE);
                XBaseThr.Out o = XBaseThr.evaluate(l, units.get(ci), persons.size(), tpl);
                if (ci == BASE_C) {
                    String tl = "межс. " + XBaseRule.FEATS[f] + " " + XBaseThr.RV_NAME[1];
                    for (int w : new int[] {0, 2}) check.append(checkThr(tl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), report, tpl.b, thrTxt));
                }
                outs.put(l, o);
                if (ci != BASE_C) pairs.add(new String[] {l, "межс. " + XBaseRule.FEATS[f] + " " + cName(BASE_C)});
            }
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        tables(t, "FERET межсессионно (роль «межс.» feret-xs2, оба прохода)", outs, persons.size(), tpl, pairs, true);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_reg_scores.tsv.gz"), "# FERET, регуляризация MLDA (XBaseReg feret-reg), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_reg.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-reg

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-reg: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        xs[1].vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        Map<String, Mat> pf = new LinkedHashMap<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) pf.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval p0 = new XBaseEval();
        p0.vectors(pf, EnumSet.of(V.L1), g.params);
        XBaseEval p1 = new XBaseEval();
        p1.vectors(pf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : pf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(p0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(p0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(p1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(p1.mir.get(V.L1));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Path thrScores = out.resolve("xbase_gt_thr_scores.tsv.gz");
        Path thrTxt = out.resolve("xbase_gt_thr.txt");
        if (!Files.exists(thrScores) || !Files.exists(thrTxt)) throw new IllegalStateException("gt-reg: нужны " + thrScores + " и " + thrTxt + " (gt-thr)");
        Map<String, String> blocks = XBaseSubsp.blocks(thrScores);
        List<String> report = Files.readAllLines(thrTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-reg; seed %d; r0 %s; пул FERET %d; c %s", g.seed, g.r0, pool.size(),
                java.util.Arrays.toString(CS)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_reg"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.3: регуляризация MLDA (пол λ̄·c), GT (XBaseEval gt-reg)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F – Color "
                + "FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-rule, gt-thr: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6; внутри сессии N = 3 – контроль "
                + "(Δ = 3 п. п.), порог только (Т0).\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        t.append("Порог – по «чужим» (Т2) (решение Жорика 24.10.2026), рядом (Т0). Сетка c на своих – диагностика; «выбор по пулу» – c, выбранное pool-reg\n"
                + "(набор GT – весь пул FERET), прогон на своих – один.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET,
                g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: c = 1 – строки = gt-thr (все роли, побайтно); (Т0), (Т2) П3 FAR 1 % = отчёт gt-thr ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD).append("режим, признак | c | осей | заменено (X) | заменено (X+mirror)\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (набор GT) ===\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double fitS = 0;
        List<List<String>> cohort0 = g.cohort;
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                boolean x = mode.equals("межс.");
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Probe> probes = XBaseEval.probes(g, pro);
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    aug.addAll(poolCls);
                    g.cohort = aug;
                    String tag = (x ? "x" : "i") + "_f" + f;
                    Lines ls = read(cp, tag);
                    if (ls == null) {
                        long t0 = System.nanoTime();
                        ls = fitLines(g, xs[f], gal, probes, parts, 0, x ? XBaseThr.extra(g, pro, parts) : null);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        save(cp, tag, ls);
                        System.out.printf(Locale.ROOT, "gt-reg: %s ready (%.0f s)%n", tag, sec);
                    }
                    String bp = mode + ", " + XBaseRule.FEATS[f] + ", ";
                    for (int r = 0; r < 2; r++) check.append(check(bp + ROW[r], ls.byC()[BASE_C][r], blocks, thrScores));
                    info.append(infoRow(mode + ", " + XBaseRule.FEATS[f], ls));
                    int c = select.get("GT").get(XBaseRule.FEATS[f]);
                    if (x) sel.append(String.format(Locale.ROOT, "%s: %s%n", XBaseRule.FEATS[f], cName(c)));
                    for (int ci = 0; ci <= CS.length; ci++) {
                        String b0 = ls.byC()[ci < CS.length ? ci : c][1];
                        String l = mode + " " + XBaseRule.FEATS[f] + " " + (ci < CS.length ? cName(ci) : CHOICE);
                        if (ci < CS.length) lines.append("# ").append(bp).append(cName(ci)).append(", косинус\n").append(b0);
                        XBaseThr.Out o = XBaseThr.evaluate(l, List.of(new XBaseThr.Unit(g, gal, pro, parts, b0, null, x)), g.persons.size(), g);
                        if (ci == BASE_C) {
                            String tl = mode + " " + XBaseRule.FEATS[f] + " " + XBaseThr.RV_NAME[1];
                            for (int w : x ? new int[] {0, 2} : new int[] {0}) {
                                check.append(checkThr(tl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), report, g.b, thrTxt));
                            }
                        }
                        outs.put(l, o);
                        if (ci != BASE_C) pairs.add(new String[] {l, mode + " " + XBaseRule.FEATS[f] + " " + cName(BASE_C)});
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
        XBaseOther.writeScores(out.resolve("xbase_gt_reg_scores.tsv.gz"), "# GT, регуляризация MLDA (XBaseReg gt-reg)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_reg.txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
