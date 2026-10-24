package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Порог П2/П3 при косинусе (ревизия 1.14; задание №4 Жорика от 24.10.2026, план подтверждён Хозяином, дополнение №5).
 * Модели, части и пробы – как feret-rule (роль «межс.» feret-xs2, оба прохода) и gt-rule (межс. N = 6, внутри N = 3); ничего
 * не переобучается. Способы набора «чужих» порога: (Т0) – как сейчас, снимки галереи людей части q против модели без части q
 * (роль cv q); (Т1) – межсессионные пробы людей части q (все их пробы роли «свои») против той же модели (роль od q, аналог
 * журнала подтверждённых проходов); (Т2) – объединение. FERET: (Т1н) – без пересечения ролей: у людей с ≥ 3 датами одна дата
 * проб (Random(faces.seed + 10300 + проход)) – в «чужие», остальные – в оценку своих («сокращённый набор»); на нём же (Т0) и
 * (Т1). Строки: (0) евклид и (б) косинус по всем осям, L1-xs-F и L1-tt-xs-F; GT внутри – только (Т0). Строки ролей own, ctrl,
 * cv q сверяются побайтно с файлами оценок feret-rule / gt-rule, итог (Т0) при П3 FAR 1 % – с их отчётами.
 *
 * @author ssv
 */
final class XBaseThr {

    static final String T1_ROLE = "od";
    static final long T1N_SEED_OFFSET = 10300;
    /** Строки XBaseRule.RVS: (0) евклид, (б) косинус по всем осям. */
    static final int[] RV = {0, 1};
    static final String[] RV_NAME = {"евклид", "косинус"};
    static final double[] QS = {0.05, 0.25, 0.5};
    static final String[] WAYS = {"(Т0)", "(Т1)", "(Т2)"};
    static final String[] WAYS_RED = {"(Т0) сокр.", "(Т1) сокр.", "(Т1н) сокр."};

    private XBaseThr() {
    }

    /** Проход: Gt, галерея, пробы, части, строки оценок; proRed – пробы своих сокращённого набора (null – нет); t1 – есть роль od. */
    record Unit(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, String lines,
                Map<String, List<String>> proRed, boolean t1) {}

    /** Дополнительные пробы моделей без части q: пробы своих людей части (роль od q). */
    static List<List<Probe>> extra(XBaseEval.Gt g, Map<String, List<String>> pro, List<List<String>> parts) {
        List<List<Probe>> out = new ArrayList<>();
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
            List<Probe> l = new ArrayList<>();
            for (String p : parts.get(q)) for (String k : pro.get(p)) l.add(new Probe(T1_ROLE + q, p, g.img(k), k, -1));
            out.add(l);
        }
        return out;
    }

    static Map<String, Row> rows(String lines) {
        Map<String, Row> rows = new HashMap<>();
        for (String l : lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        return rows;
    }

    /** «Чужие» (Т1) или (Т1н): по людям частей, снимки keys(p) (роль od q), модель по |r| пробы. */
    static FeiThreshold.Imp t1(Unit u, Map<String, Row> rows, Map<String, List<String>> keys) {
        List<double[]> v = new ArrayList<>();
        int front = 0;
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
            for (String p : u.parts().get(q)) {
                for (String k : keys.getOrDefault(p, List.of())) {
                    boolean f = u.g().rOf.get(k) <= u.g().r0;
                    if (f) front++;
                    Row r = XBaseEval.choose(rows, V.L1, T1_ROLE + q, p, u.g().img(k), f);
                    v.add(new double[] {u.g().pIdx.get(p), FeiScores.ratio(r, null)});
                }
            }
        }
        return FeiThreshold.Imp.of(v, front);
    }

    /** Снятые в (Т1н) пробы: pro − proRed. */
    static Map<String, List<String>> dropped(Unit u) {
        Map<String, List<String>> d = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : u.pro().entrySet()) {
            Set<String> keep = new HashSet<>(u.proRed().get(e.getKey()));
            List<String> l = e.getValue().stream().filter(k -> !keep.contains(k)).toList();
            if (!l.isEmpty()) d.put(e.getKey(), l);
        }
        return d;
    }

    /** Итог строки: Res по способам (полный набор и сокращённый), «чужие» для квантилей, объём. */
    static final class Out {
        final Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        final List<double[]> imp0 = new ArrayList<>();
        final List<double[]> imp1 = new ArrayList<>();
        final List<double[]> imp1n = new ArrayList<>();
        final StringBuilder vol = new StringBuilder();
    }

    static double[] rs(List<FeiThreshold.Imp> l) {
        return l.stream().flatMapToDouble(i -> java.util.Arrays.stream(i.r())).toArray();
    }

    /** Разбор строки по проходам: способы (Т0)–(Т2), при proRed – и сокращённый набор. */
    static Out evaluate(String label, List<Unit> units, int persons, XBaseEval.Gt tpl) {
        Out o = new Out();
        List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        List<FeiDecompose.Cfg> red = new ArrayList<>();
        List<List<FeiThreshold.Imp>> imps = List.of(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        List<FeiThreshold.Imp> imp1n = new ArrayList<>();
        boolean t1 = units.get(0).t1();
        boolean hasRed = units.get(0).proRed() != null;
        long seed = tpl.seed + FeiBootstrap.SEED_OFFSET;
        for (int i = 0; i < units.size(); i++) {
            Unit u = units.get(i);
            XBaseEval.Parsed p = XBaseEval.parse(u.g(), EnumSet.of(V.L1), u.gal(), XBaseEval.probes(u.g(), u.pro()), u.parts(), u.lines());
            cfgs.add(p.cfg.get(V.L1));
            imps.get(0).add(p.imp.get(V.L1));
            o.vol.append(String.format(Locale.ROOT, "%s | проход %d | (Т0) %d снимков, %d людей", label, i + 1, p.imp.get(V.L1).r().length,
                    p.imp.get(V.L1).persons()));
            if (t1) {
                Map<String, Row> rows = rows(u.lines());
                FeiThreshold.Imp a = t1(u, rows, u.pro());
                imps.get(1).add(a);
                imps.get(2).add(XBaseCalib.union(p.imp.get(V.L1), a));
                o.vol.append(String.format(Locale.ROOT, " | (Т1) %d снимков, %d людей", a.r().length, a.persons()));
                if (hasRed) {
                    FeiThreshold.Imp n = t1(u, rows, dropped(u));
                    imp1n.add(n);
                    FeiDecompose.Cfg c = XBaseEval.parse(u.g(), EnumSet.of(V.L1), u.gal(), XBaseEval.probes(u.g(), u.proRed()), u.parts(), u.lines()).cfg
                            .get(V.L1);
                    red.add(c);
                    long people = java.util.Arrays.stream(c.ownP).distinct().count();
                    o.vol.append(String.format(Locale.ROOT, " | (Т1н) %d снимков, %d людей | сокр. набор своих: %d проб, %d людей (полный %d проб)", n.r().length,
                            n.persons(), c.ownR.length, people, p.cfg.get(V.L1).ownR.length));
                }
            }
            o.vol.append('\n');
        }
        int ways = t1 ? 3 : 1;
        for (int w = 0; w < ways; w++) o.res.put(WAYS[w], FeiThreshold.evaluate(label + " " + WAYS[w], V.L1.ratioId(), cfgs, imps.get(w), persons, tpl.b, seed));
        if (hasRed) {
            o.res.put(WAYS_RED[0], FeiThreshold.evaluate(label + " " + WAYS_RED[0], V.L1.ratioId(), red, imps.get(0), persons, tpl.b, seed));
            o.res.put(WAYS_RED[1], FeiThreshold.evaluate(label + " " + WAYS_RED[1], V.L1.ratioId(), red, imps.get(1), persons, tpl.b, seed));
            o.res.put(WAYS_RED[2], FeiThreshold.evaluate(label + " " + WAYS_RED[2], V.L1.ratioId(), red, imp1n, persons, tpl.b, seed));
        }
        for (double r : rs(imps.get(0))) o.imp0.add(new double[] {r});
        if (t1) for (double r : rs(imps.get(1))) o.imp1.add(new double[] {r});
        for (double r : rs(imp1n)) o.imp1n.add(new double[] {r});
        return o;
    }

    static String quant(List<double[]> v) {
        if (v.isEmpty()) return "–";
        double[] s = FeiBootstrap.sorted(v.stream().mapToDouble(a -> a[0]).toArray());
        List<String> c = new ArrayList<>();
        for (double q : QS) c.add(String.format(Locale.ROOT, "%.4f", FeiBootstrap.quantile(s, q)));
        return String.join(" / ", c) + " (n = " + s.length + ")";
    }

    /** Разрыв «ошибка своих при П3 − разделимость при фактическом FAR контроля П3» (FAR k), точка и реплики. */
    static String gap(FeiThreshold.Res r, int m, int k, int persons, int[][] w) {
        double far = FeiThreshold.farPct(r, m, k);
        double err = FeiThreshold.err(r, m, k, 0);
        double sep = XBaseOther.errAtFar(r.cfgs, FeiBootstrap.unit(persons), far / 100)[0];
        List<Double> sepRep = new ArrayList<>();
        List<Double> gapRep = new ArrayList<>();
        for (int q = 0; q < w.length; q++) {
            double f = r.farRep[m][k][q];
            double e = r.errRep[m][k][0][q];
            if (Double.isNaN(f) || Double.isNaN(e)) continue;
            double s = XBaseOther.errAtFar(r.cfgs, w[q], f / 100)[0];
            if (Double.isNaN(s)) continue;
            sepRep.add(s);
            gapRep.add(e - s);
        }
        double[] sr = sepRep.stream().mapToDouble(Double::doubleValue).toArray();
        double[] gr = gapRep.stream().mapToDouble(Double::doubleValue).toArray();
        String g = gr.length == 0 ? String.format(Locale.ROOT, "%+.1f", err - sep) : FeiBootstrap.diff(err - sep, gr, 1);
        return String.format(Locale.ROOT, "%s | %s | %s", XBaseSepLtp.cell(sep, sr), g, gr.length == w.length ? "" : "реплик " + gr.length);
    }

    static final String[] M_NAME = {"", "", "П2", "П3"};

    /** Таблицы группы строк: подпись строки → Out. */
    static void tables(StringBuilder t, String title, Map<String, Out> outs, int persons, XBaseEval.Gt tpl) {
        int b = tpl.b;
        int[][] w = b > 0 ? FeiBootstrap.weights(persons, b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        t.append("\n\n######## ").append(title).append(" ########\n");
        t.append("\n=== Объём «чужих» порога по проходам ===\n");
        for (Out o : outs.values()) t.append(o.vol);
        t.append("\n=== Сводка: П3 и П2, FAR 1 % и 5 %; FAR контроля по снимкам (КП – верхняя 95 % граница Клоппера – Пирсона); «держит» – КП ↑ ≤ "
                + "номинала;\nразделимость – ошибка своих при пороге по контрольным чужим, дающем тот же фактический FAR контроля, что и способ; "
                + "разрыв = ошибка − разделимость ===\n");
        t.append("строка | способ | порог | FAR | ошибка своих | FAR контроля; КП | держит | ноль недопустим (конфигураций) | разделимость при том же FAR | "
                + "разрыв, п. п. | примечание\n");
        for (Map.Entry<String, Out> e : outs.entrySet()) {
            for (Map.Entry<String, FeiThreshold.Res> r : e.getValue().res.entrySet()) {
                for (int m = 3; m >= 2; m--) {
                    for (int k = 0; k < FeiThreshold.KS; k++) {
                        FeiThreshold.Res rr = r.getValue();
                        int x = (int) rr.far[m][k][0];
                        int n = (int) rr.far[m][k][1];
                        double up = 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95);
                        double nom = 100 * FeiScores.FARS[k];
                        t.append(String.format(Locale.ROOT, "%s | %s | %s | %.0f %% | %s | %.2f %%; КП %d/%d ↑%.2f %% | %s | %d | %s%n", e.getKey(), r.getKey(),
                                M_NAME[m], nom, FeiThreshold.errCell(rr, m, k, 0, b), FeiThreshold.farPct(rr, m, k), x, n, up, up <= nom ? "да" : "нет",
                                rr.zeroFail[m][k], gap(rr, m, k, persons, w)));
                    }
                }
            }
        }
        t.append("\n=== Ошибка своих по группам (как в LbpNormReport): всего | анфас | поворот | (i)–(iv) | верный кандидат ===\n");
        for (Map.Entry<String, Out> e : outs.entrySet())
            for (Map.Entry<String, FeiThreshold.Res> r : e.getValue().res.entrySet())
                for (int m = 3; m >= 2; m--)
                    for (int k = 0; k < FeiThreshold.KS; k++)
                        t.append(LbpNormReport.errRow(List.of(e.getKey(), r.getKey(), M_NAME[m], String.format(Locale.ROOT, "FAR %.0f %%", 100 * FeiScores.FARS[k])),
                                r.getValue(), m, k, b));
        t.append("\n=== FAR контроля: ").append(LbpNormReport.FAR_HEAD).append(" ===\n");
        for (Map.Entry<String, Out> e : outs.entrySet())
            for (Map.Entry<String, FeiThreshold.Res> r : e.getValue().res.entrySet())
                for (int m = 3; m >= 2; m--)
                    for (int k = 0; k < FeiThreshold.KS; k++)
                        t.append(LbpNormReport.farRow(List.of(e.getKey(), r.getKey(), M_NAME[m], String.format(Locale.ROOT, "FAR %.0f %%", 100 * FeiScores.FARS[k])),
                                r.getValue(), m, k, b));
        t.append("\n=== Диагностика: квантили 5 / 25 / 50 % d₁/d₂ (все проходы) ===\nстрока | «чужие» (Т0) | «чужие» (Т1) | «чужие» (Т1н) | контрольные чужие\n");
        for (Map.Entry<String, Out> e : outs.entrySet()) {
            Out o = e.getValue();
            List<double[]> ctrl = new ArrayList<>();
            for (FeiDecompose.Cfg c : o.res.get(WAYS[0]).cfgs) for (double r : c.ctrlR) ctrl.add(new double[] {r});
            t.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s%n", e.getKey(), quant(o.imp0), quant(o.imp1), quant(o.imp1n), quant(ctrl)));
        }
        if (b > 0) {
            t.append("\n=== Парные разности способов, п. п. (одни веса бутстрепа; «*» – интервал [2,5; 97,5] не содержит 0) ===\n");
            t.append("строка | пара | порог | ").append(LbpNormReport.DIFF_HEAD).append('\n');
            String[][] pairs = {{WAYS[1], WAYS[0]}, {WAYS[2], WAYS[0]}, {WAYS_RED[2], WAYS_RED[0]}, {WAYS_RED[1], WAYS_RED[2]}};
            for (Map.Entry<String, Out> e : outs.entrySet()) {
                for (String[] p : pairs) {
                    FeiThreshold.Res a = e.getValue().res.get(p[0]);
                    FeiThreshold.Res z = e.getValue().res.get(p[1]);
                    if (a == null || z == null) continue;
                    for (int m = 3; m >= 2; m--) t.append(LbpNormReport.diffRow(List.of(e.getKey(), p[0] + " − " + p[1], M_NAME[m]), a, z, m));
                }
            }
        }
    }

    /** Сверка с прежними: строки ролей кроме od – побайтно с блоком; (Т0) П3 FAR 1 % – с ячейками строки отчёта. */
    static String check(String bn, String lines, Map<String, String> blocks, Path src) {
        String old = blocks.get(bn);
        if (old == null) throw new IllegalStateException("XBaseThr: нет блока «" + bn + "» в " + src);
        List<String> a = sorted(old, false);
        List<String> z = sorted(lines, true);
        if (!a.equals(z)) throw new IllegalStateException("XBaseThr: строки не совпали с " + src + ": " + bn);
        return String.format(Locale.ROOT, "%s: совпало (%d строк)%n", bn, a.size());
    }

    static List<String> sorted(String lines, boolean dropT1) {
        List<String> out = new ArrayList<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            if (dropT1 && l.split("\t", -1)[3].startsWith(T1_ROLE)) continue;
            out.add(l);
        }
        Collections.sort(out);
        return out;
    }

    static String checkRes(String ruleLabel, FeiThreshold.Res r, List<String> report, int b, Path src) {
        String err = FeiThreshold.errCell(r, 3, 0, 0, b);
        String far = FeiThreshold.farCell(r, 3, 0, b);
        for (String l : report) {
            if (!l.startsWith(ruleLabel + " | ")) continue;
            String[] f = l.split(" \\| ");
            if (f.length > 7 && f[6].equals(err) && f[7].equals(far)) return String.format(Locale.ROOT, "%s (Т0) П3 FAR 1 %%: совпало (%s; %s)%n", ruleLabel, err, far);
            throw new IllegalStateException("XBaseThr: (Т0) не совпала с " + src + " для «" + ruleLabel + "»: " + err + " / " + far + " против " + l);
        }
        throw new IllegalStateException("XBaseThr: нет строки «" + ruleLabel + "» в " + src);
    }

    static final String NOTE = "Способы набора «чужих» порога (П2 – Нейман – Пирсон, П3 – с поправкой на корреляцию по людям; FAR 1 % и 5 %): (Т0) – как сейчас,\n"
            + "снимки галереи людей части q против модели без части q; (Т1) – межсессионные пробы людей части q (все их пробы роли «свои», другие\n"
            + "даты) против той же модели (роль od q; аналог журнала подтверждённых проходов); (Т2) – объединение (Т0) и (Т1). Модели, 5 частей\n"
            + "(seed faces.seed + 8000 + 100), пробы и контроль прежние, ничего не переобучается; порог – свой на проход, итог – объединение.\n"
            + "Оговорка: «чужие» (Т1) – те же снимки, что пробы своих (против другой галереи, без своей части), то есть совпадают по снимку,\n"
            + "человеку и дню с оцениваемыми пробами.\n";

    // ---------------------------------------------------------------- feret-thr

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        XBaseEval.Gt tpl = s.tpl;
        XBaseFeret.Roles ro = XBaseFeret.equalize(s.su.ro, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Path ruleScores = s.out.resolve("xbase_feret_rule_scores.tsv.gz");
        Path ruleTxt = s.out.resolve("xbase_feret_rule.txt");
        if (!Files.exists(ruleScores) || !Files.exists(ruleTxt)) throw new IllegalStateException("feret-thr: нужны " + ruleScores + " и " + ruleTxt + " (feret-rule)");
        Map<String, String> blocks = XBaseSubsp.blocks(ruleScores);
        List<String> report = Files.readAllLines(ruleTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-thr %s; seed %d; r0 %s; пул %d; половины %d/%d", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_thr_eq"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.14: порог П2/П3 при косинусе, FERET межсессионно (XBaseEval feret-thr): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("Роль «межс.» feret-xs2 (как feret-rule): половины №2, проход 1 – свои A, пул проекции – B, проход 2 – наоборот; контрольные 200.\n");
        t.append(NOTE);
        t.append("(Т1н) – без пересечения ролей: у людей с ≥ 3 датами (галерея и ≥ 2 даты проб) одна дата проб (java.util.Random(faces.seed + "
                + T1N_SEED_OFFSET + " + проход),\nпо людям в порядке списка своих) идёт в «чужие» (Т1н), остальные даты – в оценку своих; люди с 2 датами "
                + "оцениваются полностью\n(«сокращённый набор»). На сокращённом наборе рядом – (Т0) и (Т1) (порог по полному (Т1)); «(Т1) − (Т1н)» на одних "
                + "пробах – мера\nоптимизма от пересечения ролей. При малом числе «чужих» П3 FAR 1 % недостижим и при нуле принятых (столбец «ноль недопустим»).\n");
        t.append("Строки: (0) евклид и (б) косинус по всем осям MLDA (XBaseRule).\n");
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для обоих проходов, всех строк и способов: B = %d, "
                + "seed = %d.%n", persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: строки ролей own, ctrl, cv = feret-rule (побайтно); (Т0) П3 FAR 1 % = отчёт feret-rule ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Out> outs = new LinkedHashMap<>();
        double fitS = 0;
        for (int f = 0; f < XBaseRule.FEATS.length; f++) {
            List<List<Unit>> units = List.of(new ArrayList<>(), new ArrayList<>());
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
                // Сокращённый набор: у людей с ≥ 2 датами проб одна дата – в (Т1н).
                Random rnd = new Random(tpl.seed + T1N_SEED_OFFSET + ps);
                Map<String, List<String>> red = new LinkedHashMap<>();
                for (String p : own) {
                    List<String> dates = new ArrayList<>(new TreeSet<>(pro.get(p).stream().map(k -> s.su.byKey.get(k).date()).toList()));
                    if (dates.size() < 2) {
                        red.put(p, pro.get(p));
                        continue;
                    }
                    String d = dates.get(rnd.nextInt(dates.size()));
                    red.put(p, pro.get(p).stream().filter(k -> !s.su.byKey.get(k).date().equals(d)).toList());
                }
                List<Probe> probes = XBaseEval.probes(g, pro);
                String tag = String.format(Locale.ROOT, "p%d_f%d", ps, f);
                XBaseRule.Lines ls = XBaseRule.read(cp, tag);
                if (ls == null) {
                    long t0 = System.nanoTime();
                    List<String> pk = XBaseRule.keys(s.pool, s.halves.get(1 - ps));
                    XBaseRule.Pca[] pcas = {XBaseRule.pca(s.xs[f], pk, false), XBaseRule.pca(s.xs[f], pk, true)};
                    ls = XBaseRule.fitLines(g, s.xs[f], gal, probes, parts, 0, pcas, extra(g, pro, parts));
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    XBaseRule.save(cp, tag, ls);
                    System.out.printf(Locale.ROOT, "feret-thr: pass %d, %s ready (%.0f s)%n", ps + 1, XBaseRule.FEATS[f], sec);
                }
                for (int i = 0; i < RV.length; i++) {
                    String body = ls.byRv().get(RV[i]);
                    check.append(check("проход " + (ps + 1) + ", " + XBaseRule.FEATS[f] + ", " + XBaseRule.RVS.get(RV[i]).label(), body, blocks, ruleScores));
                    lines.append("# проход ").append(ps + 1).append(", ").append(XBaseRule.FEATS[f]).append(", ").append(XBaseRule.RVS.get(RV[i]).label())
                            .append('\n').append(body);
                    units.get(i).add(new Unit(g, gal, pro, parts, body, red, true));
                }
            }
            for (int i = 0; i < RV.length; i++) {
                String l = "межс. " + XBaseRule.FEATS[f] + " " + RV_NAME[i];
                Out o = evaluate(l, units.get(i), persons.size(), tpl);
                check.append(checkRes("межс. " + XBaseRule.FEATS[f] + " " + XBaseRule.RVS.get(RV[i]).label(), o.res.get(WAYS[0]), report, tpl.b, ruleTxt));
                outs.put(l, o);
            }
        }
        t.append(check);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        tables(t, "FERET межсессионно (роль «межс.» feret-xs2, оба прохода)", outs, persons.size(), tpl);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_thr_scores.tsv.gz"), "# FERET, порог при косинусе (XBaseThr feret-thr), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_thr.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-thr

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-thr: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
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
        List<String> poolKeys = XBaseRule.keys(pool, pool.keySet());

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Path ruleScores = out.resolve("xbase_gt_rule_scores.tsv.gz");
        Path ruleTxt = out.resolve("xbase_gt_rule.txt");
        if (!Files.exists(ruleScores) || !Files.exists(ruleTxt)) throw new IllegalStateException("gt-thr: нужны " + ruleScores + " и " + ruleTxt + " (gt-rule)");
        Map<String, String> blocks = XBaseSubsp.blocks(ruleScores);
        List<String> report = Files.readAllLines(ruleTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-thr; seed %d; r0 %s; пул FERET %d", g.seed, g.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_thr"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 1.14: порог П2/П3 при косинусе, GT (XBaseEval gt-thr)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F и "
                + "PCA – Color FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-rule: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6; внутри сессии N = 3 – только (Т0), как контроль.\n");
        t.append(NOTE);
        t.append("(Т1н) на GT не считается: людей с ≥ 3 датами трое (gt_sessions.tsv), деление проб одной даты пересечения по дню не снимает.\n");
        t.append("Строки: (0) евклид и (б) косинус по всем осям MLDA (XBaseRule).\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET,
                g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: строки ролей own, ctrl, cv = gt-rule (побайтно); (Т0) П3 FAR 1 % = отчёт gt-rule ===\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double fitS = 0;
        List<List<String>> cohort0 = g.cohort;
        XBaseRule.Pca[][] pcas = new XBaseRule.Pca[XBaseRule.FEATS.length][];
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                boolean x = mode.equals("межс.");
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Probe> probes = XBaseEval.probes(g, pro);
                Map<String, Out> outs = new LinkedHashMap<>();
                for (int f = 0; f < XBaseRule.FEATS.length; f++) {
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    aug.addAll(poolCls);
                    g.cohort = aug;
                    String tag = (x ? "x" : "i") + "_f" + f;
                    XBaseRule.Lines ls = XBaseRule.read(cp, tag);
                    if (ls == null) {
                        if (pcas[f] == null) pcas[f] = new XBaseRule.Pca[] {XBaseRule.pca(xs[f], poolKeys, false), XBaseRule.pca(xs[f], poolKeys, true)};
                        long t0 = System.nanoTime();
                        ls = XBaseRule.fitLines(g, xs[f], gal, probes, parts, 0, pcas[f], x ? extra(g, pro, parts) : null);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        XBaseRule.save(cp, tag, ls);
                        System.out.printf(Locale.ROOT, "gt-thr: %s ready (%.0f s)%n", tag, sec);
                    }
                    for (int i = 0; i < RV.length; i++) {
                        String b0 = ls.byRv().get(RV[i]);
                        String rl = XBaseRule.RVS.get(RV[i]).label();
                        check.append(check(mode + ", " + XBaseRule.FEATS[f] + ", " + rl, b0, blocks, ruleScores));
                        lines.append("# ").append(mode).append(", ").append(XBaseRule.FEATS[f]).append(", ").append(rl).append('\n').append(b0);
                        String l = mode + " " + XBaseRule.FEATS[f] + " " + RV_NAME[i];
                        Out o = evaluate(l, List.of(new Unit(g, gal, pro, parts, b0, null, x)), g.persons.size(), g);
                        check.append(checkRes(mode + " " + XBaseRule.FEATS[f] + " " + rl, o.res.get(WAYS[0]), report, g.b, ruleTxt));
                        outs.put(l, o);
                    }
                }
                tables(body, "GT " + mode, outs, g.persons.size(), g);
            }
        } finally {
            g.cohort = cohort0;
        }
        t.append(check);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_thr_scores.tsv.gz"), "# GT, порог при косинусе (XBaseThr gt-thr)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_thr.txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
