package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Сочетание сигналов повтора (4а) и совмещение d₁ и d₁/d₂ (решение Хозяина (29), постановка Жорика №3 от 27.10.2026, план принят
 * №4) – без обучения, по готовым файлам оценок pool-abl2, feret-abl2, gt-abl2 (код 0e41217). Часть 2, оба кандидата: строка оценок
 * переписывается в d₁′ = d₁^(1+β) · d₂^(−β) при втором расстоянии XBaseAbl.ABS_C, β ∈ {∞ (база), 0, 0,5, 1, 2}; β = 0 – побайтно
 * строка (4) pool-abl2. Часть 1, L1-xs-F: база, (М) – база при r0 = −1, (Т) – строка (3) (с Тан – Триггс), (Т+М) – строка (3) при r0 =
 * −1 (модель X+mirror у (Т) обучена в pool-abl2). Выбор по пулу: часть 2 – β = ∞ в пределах 1 п. п. от лучшего, иначе β с
 * наименьшим критерием; часть 1 – база в пределах 1 п. п., иначе первый по простоте (база < (М) < (Т) < (Т+М)) в пределах 1 п. п.
 *
 * @author ssv
 */
final class XBaseAbl3 {

    static final double[] BETAS = {Double.POSITIVE_INFINITY, 0, 0.5, 1, 2};
    static final String[] PART1 = {"база (П2)", "(М) всегда X+mirror", "(Т) с Тан – Триггс", "(Т+М) с Тан – Триггс, всегда X+mirror"};
    /** Строка pool-abl2 / feret-abl2 / gt-abl2 и r0 (NaN – прежний) у вариантов части 1. */
    static final int[] PART1_ROW = {0, 0, 3, 3};
    static final boolean[] PART1_MIRROR = {false, true, false, true};
    static final String SELECT = "xbase_pool_comb_select.tsv";

    private XBaseAbl3() {
    }

    static String bname(int i) {
        double b = BETAS[i];
        if (Double.isInfinite(b)) return "β = ∞ (d₁/d₂, база)";
        return "β = " + (b == 0.5 ? "0,5" : String.valueOf((int) b)) + (b == 0 ? " (d₁)" : "");
    }

    /**
     * Строки при β: β = ∞ – без изменений; иначе d₁′ = d₁^(1+β) · d₂^(−β) в столбце лучшего, второе – XBaseAbl.ABS_C; при β = 0 столбец
     * лучшего не переписывается (побайтно XBaseAbl.abs); строки с d₂ = +∞ не меняются; d₁ = 0 – d₁′ = 0.
     */
    static String beta(String lines, double b) {
        if (Double.isInfinite(b)) return lines;
        StringBuilder out = new StringBuilder();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (!c[10].equals("Infinity")) {
                double d1 = Double.parseDouble(c[8]);
                double d2 = Double.parseDouble(c[10]);
                if (b != 0) {
                    double v = d1 == 0 ? 0 : Math.pow(d1, 1 + b) * Math.pow(d2, -b);
                    if (!(v >= 0) || v >= XBaseAbl.ABS_C) throw new IllegalStateException("XBaseAbl3: d₁′ вне диапазона: " + l);
                    c[8] = Double.toString(v);
                } else if (d1 >= XBaseAbl.ABS_C || (d1 > 0 && d1 / XBaseAbl.ABS_C < 1e-12)) {
                    throw new IllegalStateException("XBaseAbl3: d₁ вне диапазона постоянной: " + l);
                }
                c[10] = Double.toString(XBaseAbl.ABS_C);
            }
            out.append(String.join("\t", c)).append('\n');
        }
        return out.toString();
    }

    /** Выбор β: β = ∞, если в пределах NEAR от лучшего; иначе наименьший критерий (при равенстве – меньший номер). */
    static int chooseBeta(double[] crit) {
        double best = Arrays.stream(crit).min().orElseThrow();
        if (crit[0] <= best + XBaseRule.NEAR) return 0;
        int pick = 0;
        for (int i = 1; i < crit.length; i++) if (crit[i] < crit[pick]) pick = i;
        return pick;
    }

    /** Выбор части 1: первый по простоте (порядок PART1) в пределах NEAR от лучшего (база – первая). */
    static int choose1(double[] crit) {
        double best = Arrays.stream(crit).min().orElseThrow();
        for (int i = 0; i < crit.length; i++) if (crit[i] <= best + XBaseRule.NEAR) return i;
        throw new IllegalStateException("XBaseAbl3: выбор не определён");
    }

    static XBaseEval.Gt mirrorGt(XBaseEval.Gt g, boolean m) {
        return m ? XBaseAbl.copy(g, g.cohort, g.rOf, -1) : g;
    }

    static String need(Map<String, String> blocks, String name, Path src) {
        String s = blocks.get(name);
        if (s == null) throw new IllegalStateException("XBaseAbl3: нет блока «" + name + "» в " + src);
        return s;
    }

    /** Строка таблицы пула (как в pool-abl2, до столбца медианы контрольных включительно). */
    static String poolRow(String name, String feat, XBaseRule.Sep sp) {
        return String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f", name, feat, XBaseRule.fmt(sp.err[0]), XBaseRule.fmt(sp.err[1]),
                XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(sp.crit()), XBaseRule.fmt(sp.eer[3]), sp.medOwn, sp.medCtrl);
    }

    /** «Держит ли П3» способа way при FAR 1 % и 5 %: FAR контроля и верхняя граница Клоппера – Пирсона. */
    static String holds(String label, XBaseThr.Out o, String way) {
        FeiThreshold.Res r = o.res.get(way);
        StringBuilder b = new StringBuilder(label + " | " + way + " | П3");
        double[] nom = {0.01, 0.05};
        for (int k = 0; k < 2; k++) {
            int x = (int) r.far[3][k][0];
            int n = (int) r.far[3][k][1];
            double up = FaceEvaluation.binomialUpperBound(x, n, 0.95);
            b.append(String.format(Locale.ROOT, " | FAR %.0f %%: %d/%d = %.2f %%, КП ↑%.2f %% – %s", 100 * nom[k], x, n, 100.0 * x / n, 100 * up, up <= nom[k]
                    ? "держит" : "не держит"));
        }
        return b.append('\n').toString();
    }

    static final String NOTE = "Без обучения – по готовым файлам оценок pool-abl2 / feret-abl2 / gt-abl2 (код 0e41217; база – (П2) LBP + POEM · 1/√3, MLDA с пулом, "
            + "c = 1, косинус).\nЧасть 2 (оба кандидата): d₁′ = d₁^(1+β) · d₂^(−β), второе расстояние – постоянная 1e6 (как строка (4)); β = ∞ – прежнее d₁/d₂, β = 0 – "
            + "d₁ (= строка (4));\nлучший кандидат не меняется (порядок по d₁); строки с d₂ = +∞ не меняются; d₁ = 0 – d₁′ = 0 (далее ρ_min).\nЧасть 1 (L1-xs-F): "
            + "(М) – база при r0 = −1 (всегда X+mirror); (Т) – строка (3) pool-abl2 (с Тан – Триггс); (Т+М) – строка (3) при r0 = −1.\n";

    // ---------------------------------------------------------------- pool-comb

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
        Path src = s.out.resolve("xbase_pool_abl2_scores.tsv.gz");
        Path srcTxt = s.out.resolve("xbase_pool_abl2.txt");
        if (!Files.exists(src) || !Files.exists(srcTxt)) throw new IllegalStateException("pool-comb: нужны " + src + " и " + srcTxt + " (pool-abl2)");
        Map<String, String> blocks = XBaseSubsp.blocks(src);
        List<String> report = Files.readAllLines(srcTxt, java.nio.charset.StandardCharsets.UTF_8);
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        StringBuilder head = new StringBuilder();
        head.append("Сочетание (Т), (М) у L1-xs-F и совмещение d₁ и d₁/d₂: выбор по пулу FERET (XBaseEval pool-comb)\n");
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: файл оценок ").append(src.getFileName()).append(" (").append(Files.readAllLines(srcTxt).get(1)).append("); пул Color FERET (")
                .append(s.pool.size()).append("); когорта FEI – экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append("Наборы и схема – как pool-abl2. Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %. Часть 2: β = ∞, если в пределах 1 п. п. "
                + "от лучшего,\nиначе β с наименьшим критерием. Часть 1: база в пределах 1 п. п. от лучшего – база, иначе первый по простоте (база < (М) < (Т) < (Т+М)) "
                + "в пределах 1 п. п.\n");
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: β = 0 – строки (4) pool-abl2 побайтно; база, (Т), (М) – строки таблиц pool-abl2 ===\n");
        // [набор][f][вариант]: части 2 (BETAS) и части 1 (PART1, только f = 0).
        List<List<List<List<FeiDecompose.Cfg>>>> c2 = new ArrayList<>();
        List<List<List<FeiDecompose.Cfg>>> c1 = new ArrayList<>();
        List<String> setNames = new ArrayList<>(sets.keySet());
        int beta0 = 0;
        for (String set : setNames) {
            Set<String> members = sets.get(set);
            List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
            List<List<List<FeiDecompose.Cfg>>> a2 = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                List<List<FeiDecompose.Cfg>> b = new ArrayList<>();
                for (int i = 0; i < BETAS.length; i++) b.add(new ArrayList<>());
                a2.add(b);
            }
            List<List<FeiDecompose.Cfg>> a1 = new ArrayList<>();
            for (int i = 0; i < PART1.length; i++) a1.add(new ArrayList<>());
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
                    for (int f = 0; f < 2; f++) {
                        String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                        String bl = need(blocks, bp + XBaseAbl2.rname(f, 0), src);
                        if (!beta(bl, 0).equals(need(blocks, bp + XBaseAbl2.rname(f, 4), src))) throw new IllegalStateException("pool-comb: β = 0 ≠ строка (4): " + bp);
                        beta0++;
                        for (int i = 0; i < BETAS.length; i++) a2.get(f).get(i).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, beta(bl, BETAS[i])).cfg
                                .get(V.L1));
                        if (f == 0) {
                            for (int i = 0; i < PART1.length; i++) {
                                String l = need(blocks, bp + XBaseAbl2.rname(f, PART1_ROW[i]), src);
                                a1.get(i).add(XBaseEval.parse(mirrorGt(g, PART1_MIRROR[i]), EnumSet.of(V.L1), gal, probes, noParts, l).cfg.get(V.L1));
                            }
                        }
                    }
                }
            }
            c2.add(a2);
            c1.add(a1);
        }
        check.append(String.format(Locale.ROOT, "β = 0 = строка (4): совпало (%d блоков)%n", beta0));
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseAbl3 pool-comb): часть, набор, признак, номер, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        String[] r6 = {XBaseAbl2.rname(0, 0), XBaseAbl2.rname(0, 6), XBaseAbl2.rname(0, 3)};
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("Часть 2 – вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | "
                    + "контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < 2; f++) {
                double[] crit = new double[BETAS.length];
                for (int i = 0; i < BETAS.length; i++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(c2.get(si).get(f).get(i))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[i] = sp.crit();
                    tb.append(poolRow(bname(i), XBaseRule.FEATS[f], sp)).append('\n');
                }
                int c = chooseBeta(crit);
                choice.append(String.format(Locale.ROOT, "Часть 2, выбор %s %s: %s (критерий %.1f %%; β = ∞ – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f],
                        bname(c), crit[c], crit[0], Arrays.stream(crit).min().orElseThrow()));
                sel.append("2\t").append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(bname(c)).append('\n');
            }
            tb.append("Часть 1 (L1-xs-F) – вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂ | "
                    + "контрольные\n");
            double[] crit = new double[PART1.length];
            for (int i = 0; i < PART1.length; i++) {
                XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(c1.get(si).get(i))), FeiBootstrap.unit(all.size()), new int[0][]);
                crit[i] = sp.crit();
                tb.append(poolRow(PART1[i], XBaseRule.FEATS[0], sp)).append('\n');
                if (i < 3) {
                    String pref = poolRow(r6[i == 1 ? 1 : i == 2 ? 2 : 0], XBaseRule.FEATS[0], sp) + " |";
                    boolean ok = false;
                    for (String l : report) ok |= l.startsWith(pref);
                    if (!ok) throw new IllegalStateException("pool-comb: нет строки «" + pref + "» в " + srcTxt);
                    check.append(String.format(Locale.ROOT, "%s, L1-xs-F, %s = %s pool-abl2: совпало%n", set, PART1[i], r6[i == 1 ? 1 : i == 2 ? 2 : 0]));
                }
            }
            int c = choose1(crit);
            choice.append(String.format(Locale.ROOT, "Часть 1, выбор %s L1-xs-F: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, PART1[c], crit[c],
                    crit[0], Arrays.stream(crit).min().orElseThrow()));
            sel.append("1\t").append(set).append('\t').append(XBaseRule.FEATS[0]).append('\t').append(c).append('\t').append(PART1[c]).append('\n');
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check).append(body);
        Files.writeString(s.out.resolve(SELECT), sel.toString());
        FarMethods.write(s.out.resolve("xbase_pool_comb.txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_comb_gt.txt"), new StringBuilder(head).append(gtBody));
        System.out.print(t);
    }

    /** Выбор части 1: true, если хоть в одном наборе не база. */
    static boolean part1Own(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseAbl3: нет файла выбора " + f + " (сначала pool-comb)");
        boolean any = false;
        for (String l : Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int i = Integer.parseInt(p[3]);
            boolean ok = p[0].equals("2") ? i >= 0 && i < BETAS.length && bname(i).equals(p[4]) : p[0].equals("1") && i >= 0 && i < PART1.length && PART1[i]
                    .equals(p[4]);
            if (!ok) throw new IllegalStateException("XBaseAbl3: строка «" + l + "» не из сетки");
            if (p[0].equals("1") && i != 0) any = true;
        }
        return any;
    }

    /** Итоги одной постановки своих: части 2 (оба признака) и 1 (L1-xs-F, при own1). */
    static void ownTables(StringBuilder body, StringBuilder hold, StringBuilder dl, String title, String prefix, int f, List<List<XBaseThr.Unit>> u2,
                          List<List<XBaseThr.Unit>> u1, int persons, XBaseEval.Gt tpl, boolean t2, boolean within) {
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        String bl = prefix + bname(0);
        for (int i = 0; i < BETAS.length; i++) {
            String l = prefix + bname(i);
            XBaseThr.Out o = XBaseThr.evaluate(l, u2.get(i), persons, tpl);
            outs.put(l, o);
            if (i > 0) pairs.add(new String[] {l, bl});
            if (t2) hold.append(holds(l, o, XBaseThr.WAYS[2]));
        }
        XBaseLbp.tables(body, title + " – часть 2 (β)", outs, persons, tpl, pairs, t2);
        if (within) for (String[] p : pairs) {
            double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], persons, tpl);
            dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                    : "НЕ выполнено"));
        }
        if (u1 == null || f != 0) return;
        Map<String, XBaseThr.Out> o1 = new LinkedHashMap<>();
        List<String[]> p1 = new ArrayList<>();
        String b1 = prefix + PART1[0];
        for (int i = 0; i < PART1.length; i++) {
            String l = prefix + PART1[i];
            o1.put(l, XBaseThr.evaluate(l, u1.get(i), persons, tpl));
            if (i > 0) p1.add(new String[] {l, b1});
        }
        XBaseLbp.tables(body, title + " – часть 1 (Т, М)", o1, persons, tpl, p1, t2);
        if (within) for (String[] p : p1) {
            double[] d = XBaseVt.delta(o1.get(p[0]), o1.get(b1), XBaseThr.WAYS[0], persons, tpl);
            dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                    : "НЕ выполнено"));
        }
    }

    // ---------------------------------------------------------------- feret-comb

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        boolean own1 = part1Own(s.out.resolve(SELECT));
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
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
        Path src = s.out.resolve("xbase_feret_abl2_scores.tsv.gz");
        Path srcTxt = s.out.resolve("xbase_feret_abl2.txt");
        if (!Files.exists(src) || !Files.exists(srcTxt)) throw new IllegalStateException("feret-comb: нужны " + src + " и " + srcTxt + " (feret-abl2)");
        Map<String, String> blocks = XBaseSubsp.blocks(src);
        List<String> report = Files.readAllLines(srcTxt, java.nio.charset.StandardCharsets.UTF_8);
        StringBuilder t = new StringBuilder();
        t.append("Сочетание (Т), (М) у L1-xs-F и совмещение d₁ и d₁/d₂, FERET (XBaseEval feret-comb): Color FERET – только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: файл оценок ").append(src.getFileName()).append(" (").append(report.get(1)).append("); ").append(s.su.dir).append("; когорта FEI — экспорт ")
                .append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        t.append(XBaseFeret.EQ_NOTE).append(NOTE);
        t.append("Постановки – как feret-abl2: «межс.» и (ii) s = 3 (справочно), контрольные – те же 200. Часть 1 на своих – ").append(own1 ? "считается (выбор по пулу "
                + "хоть в одном наборе не база)" : "не считается (выбор по пулу – база во всех наборах)").append(".\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: β = 0 – строки (4) feret-abl2 побайтно; база, (Т), (М) – П3 FAR 1 % (Т0), (Т2) как в feret-abl2 ===\n");
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder body = new StringBuilder();
        for (int post = 0; post < 2; post++) {
            String pname = post == 0 ? "межс." : "(ii) s3";
            List<Object[]> passes = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (post == 0) {
                    own = ro.own1.stream().filter(mine::contains).toList();
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else {
                    own = ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList();
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
                for (String p : s.halves.get(1 - ps)) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + (post == 0 ? 0 : 71));
                passes.add(new Object[] {g, gal, pro, parts});
            }
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                List<List<XBaseThr.Unit>> u2 = new ArrayList<>();
                for (int i = 0; i < BETAS.length; i++) u2.add(new ArrayList<>());
                List<List<XBaseThr.Unit>> u1 = null;
                if (own1 && f == 0) {
                    u1 = new ArrayList<>();
                    for (int i = 0; i < PART1.length; i++) u1.add(new ArrayList<>());
                }
                for (int ps = 0; ps < 2; ps++) {
                    Object[] p = passes.get(ps);
                    XBaseEval.Gt g = (XBaseEval.Gt) p[0];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) p[1];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> pro = (Map<String, List<String>>) p[2];
                    @SuppressWarnings("unchecked")
                    List<List<String>> parts = (List<List<String>>) p[3];
                    String bp = pname + ", " + feat + ", проход " + (ps + 1) + ", ";
                    String bl = need(blocks, bp + XBaseAbl2.rname(f, 0), src);
                    if (!beta(bl, 0).equals(need(blocks, bp + XBaseAbl2.rname(f, 4), src))) throw new IllegalStateException("feret-comb: β = 0 ≠ строка (4): " + bp);
                    check.append(bp).append("β = 0 = строка (4): совпало\n");
                    for (int i = 0; i < BETAS.length; i++) u2.get(i).add(new XBaseThr.Unit(g, gal, pro, parts, beta(bl, BETAS[i]), null, true));
                    if (u1 != null) for (int i = 0; i < PART1.length; i++) u1.get(i).add(new XBaseThr.Unit(mirrorGt(g, PART1_MIRROR[i]), gal, pro, parts, need(blocks,
                            bp + XBaseAbl2.rname(f, PART1_ROW[i]), src), null, true));
                }
                // Сверка с feret-abl2: база (β = ∞) и (Т), (М) – П3 FAR 1 % при (Т0) и (Т2).
                XBaseThr.Out ob = XBaseThr.evaluate("b", u2.get(0), persons.size(), tpl);
                for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(pname + " " + feat + " " + XBaseAbl2.rname(f, 0), XBaseThr.WAYS[w], ob.res.get(
                        XBaseThr.WAYS[w]), XBaseNap.relabel(report, feat + " " + XBaseAbl2.rname(f, 0), pname + " " + feat + " " + XBaseAbl2.rname(f, 0)), tpl.b, srcTxt));
                if (u1 != null) for (int i : new int[] {1, 2}) {
                    String rn = XBaseAbl2.rname(f, i == 1 ? 6 : 3);
                    XBaseThr.Out o = XBaseThr.evaluate("c", u1.get(i), persons.size(), tpl);
                    for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(pname + " " + feat + " " + rn, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]),
                            XBaseNap.relabel(report, feat + " " + rn, pname + " " + feat + " " + rn), tpl.b, srcTxt));
                }
                ownTables(body, hold, null, "FERET " + pname + ", " + feat + (post == 1 ? " (справочно; 25 человек)" : ""), pname + " " + feat + " ", f, u2, u1,
                        persons.size(), tpl, true, false);
            }
        }
        t.append(check).append(hold).append(body);
        FarMethods.write(s.out.resolve("xbase_feret_comb.txt"), t);
        System.out.print(t);
    }

    // ---------------------------------------------------------------- gt-comb

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-comb: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        boolean own1 = part1Own(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, g.own);
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path src = out.resolve("xbase_gt_abl2_scores.tsv.gz");
        Path srcTxt = out.resolve("xbase_gt_abl2.txt");
        if (!Files.exists(src) || !Files.exists(srcTxt)) throw new IllegalStateException("gt-comb: нужны " + src + " и " + srcTxt + " (gt-abl2)");
        Map<String, String> blocks = XBaseSubsp.blocks(src);
        List<String> report = Files.readAllLines(srcTxt, java.nio.charset.StandardCharsets.UTF_8);
        StringBuilder t = new StringBuilder();
        t.append("Сочетание (Т), (М) у L1-xs-F и совмещение d₁ и d₁/d₂, GT (XBaseEval gt-comb)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: файл оценок ").append(src.getFileName()).append(" (").append(report.get(1)).append("); экспорт ").append(dir).append(", коммит данных ")
                .append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ").append(pool.size()).append(" человек\n");
        t.append("Протокол как gt-abl2: межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, порог (Т0); условие Δ – верхняя граница 95 % "
                + "интервала разности\n«вариант − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append("Часть 1 на своих – ").append(own1 ? "считается (выбор по пулу хоть в одном наборе не база)" : "не считается (выбор по пулу – база во всех наборах)")
                .append(".\n");
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: β = 0 – строки (4) gt-abl2 побайтно; база, (Т), (М) – П3 FAR 1 % как в gt-abl2 ===\n");
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (межс.; контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        StringBuilder body = new StringBuilder();
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            for (int mi = 0; mi < modes.length; mi++) {
                String mode = (String) modes[mi][0];
                boolean xm = mi == 0;
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                String bp = mode + ", " + feat + ", ";
                String bl = need(blocks, bp + XBaseAbl2.rname(f, 0), src);
                if (!beta(bl, 0).equals(need(blocks, bp + XBaseAbl2.rname(f, 4), src))) throw new IllegalStateException("gt-comb: β = 0 ≠ строка (4): " + bp);
                check.append(bp).append("β = 0 = строка (4): совпало\n");
                List<List<XBaseThr.Unit>> u2 = new ArrayList<>();
                for (int i = 0; i < BETAS.length; i++) u2.add(List.of(new XBaseThr.Unit(ga, gal, pro, parts, beta(bl, BETAS[i]), null, xm)));
                List<List<XBaseThr.Unit>> u1 = null;
                if (own1 && f == 0) {
                    u1 = new ArrayList<>();
                    for (int i = 0; i < PART1.length; i++) u1.add(List.of(new XBaseThr.Unit(mirrorGt(ga, PART1_MIRROR[i]), gal, pro, parts, need(blocks, bp
                            + XBaseAbl2.rname(f, PART1_ROW[i]), src), null, xm)));
                }
                int[] ways = xm ? new int[] {0, 2} : new int[] {0};
                XBaseThr.Out ob = XBaseThr.evaluate("b", u2.get(0), g.persons.size(), g);
                for (int w : ways) check.append(XBaseReg.checkThr(mode + " " + feat + " " + XBaseAbl2.rname(f, 0), XBaseThr.WAYS[w], ob.res.get(XBaseThr.WAYS[w]),
                        report, g.b, srcTxt));
                if (u1 != null) for (int i : new int[] {1, 2}) {
                    String rn = XBaseAbl2.rname(f, i == 1 ? 6 : 3);
                    XBaseThr.Out o = XBaseThr.evaluate("c", u1.get(i), g.persons.size(), g);
                    for (int w : ways) check.append(XBaseReg.checkThr(mode + " " + feat + " " + rn, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), report, g.b,
                            srcTxt));
                }
                ownTables(body, hold, dl, "GT " + mode + ", " + feat, mode + " " + feat + " ", f, u2, u1, g.persons.size(), g, xm, !xm);
            }
        }
        t.append(check).append(hold).append(dl).append(body);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_gt_comb.txt"), t);
        System.out.print(t);
    }
}
