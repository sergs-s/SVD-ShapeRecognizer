package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.storage.SettingsStore;

/**
 * Итоговый прогон перед ПО – отчёт (постановка, разд. 2–4): final-report &lt;feret | gt | fei | own | vt&gt; по моделям и файлам оценок Ф1 / Ф3
 * (кадры Б, выбранная σ₀ – final_sigma.tsv) и контроля на кадрах «а». Меры разд. 3: разделимость при фактическом FAR контроля 1, 2, 5, 10 %,
 * EER, частичная площадь под ROC при FAR ≤ 10 % (pAUC, нормированная: среднее (1 − ошибка своих) по FAR 1…10 % шагом 1 %), П3 по «чужим»
 * (Т2) / (Т0) – таблицы XBaseLbp; рабочий порог (34) – t(N) таблицы (Н) (FACES_P4_TABLE) с приведениями по (Т2) своей галереи, между
 * узлами – меньший t, N &lt; 10 – t(10), N &gt; 200 – t(200) с пометкой; ошибка своих, FAR контроля по попыткам и по людям с верхней границей
 * Клоппера – Пирсона, доля «чужое имя»; бутстреп по людям B (FACES_FEI_BOOTSTRAP), seed faces.seed + 7000; сравнения разд. 3 – интервалы
 * 95 % и 99,5 %. Ф2 (GT межс.) – FAR при рабочем пороге по источникам; Ф5 (GT межс.) – вторая дата ≤ 7 и &gt; 30 дней; 1:1 – худший случай
 * (заявлен ближайший, XBaseAbl3.oneLines) и случайно заявленный (Random(faces.seed + 7000)).
 *
 * @author ssv
 */
final class XBaseFinalReport {

    private XBaseFinalReport() {
    }

    static final double[] PAUC_FARS = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10};

    /** pAUC (нормированная, %) по конфигурациям при весах людей w. */
    static double pauc(List<FeiDecompose.Cfg> cs, int[] w) {
        double s = 0;
        for (double f : PAUC_FARS) s += 100 - XBaseOther.errAtFar(cs, w, f)[0];
        return s / PAUC_FARS.length;
    }

    /** Строка с интервалами 95 % и 99,5 % по повторам. */
    static String ci(double point, double[] reps) {
        if (reps.length == 0) return String.format(Locale.ROOT, "%+.1f", point);
        double[] d = FeiBootstrap.sorted(reps);
        return String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f] {%+.1f; %+.1f}%s", point, FeiBootstrap.quantile(d, 0.025), FeiBootstrap.quantile(d, 0.975),
                FeiBootstrap.quantile(d, 0.0025), FeiBootstrap.quantile(d, 0.9975), FeiBootstrap.quantile(d, 0.975) < 0 || FeiBootstrap.quantile(d, 0.025) > 0
                        ? (FeiBootstrap.quantile(d, 0.9975) < 0 || FeiBootstrap.quantile(d, 0.0025) > 0 ? " **" : " *") : "");
    }

    /** Мера строки: разделимость, EER, pAUC с повторами. */
    static final class M {
        XBaseRule.Sep sep;
        double pauc;
        double[] paucRep;
    }

    static M measure(List<FeiDecompose.Cfg> cs, int persons, int[][] w) {
        M m = new M();
        m.sep = XBaseRule.sep(cs, FeiBootstrap.unit(persons), w);
        m.pauc = pauc(cs, FeiBootstrap.unit(persons));
        m.paucRep = new double[w.length];
        for (int q = 0; q < w.length; q++) m.paucRep[q] = pauc(cs, w[q]);
        return m;
    }

    /** t(N) по правилу (34). */
    static double[] tOf(Map<Integer, XBasePart4.TRow> table, int n) {
        int[] nodes = XBasePart4.NS;
        if (n <= nodes[0]) return new double[] {table.get(nodes[0]).t(), nodes[0], 0};
        if (n >= nodes[nodes.length - 1]) return new double[] {table.get(nodes[nodes.length - 1]).t(), nodes[nodes.length - 1], n > nodes[nodes.length - 1] ? 1 : 0};
        for (int i = 0; i + 1 < nodes.length; i++) {
            if (n == nodes[i]) return new double[] {table.get(nodes[i]).t(), nodes[i], 0};
            if (n > nodes[i] && n < nodes[i + 1]) {
                double a = table.get(nodes[i]).t();
                double b = table.get(nodes[i + 1]).t();
                return new double[] {Math.min(a, b), a <= b ? nodes[i] : nodes[i + 1], 0};
            }
        }
        return new double[] {table.get(n).t(), n, 0};
    }

    /** Исход пробы при пороге t: роль, человек, принят ли, верный ли лучший. */
    record Out1(String role, String person, boolean own, boolean accepted, boolean right) {}

    /** Пробы блока при пороге t на s = 2ℓ₁ − ℓ₂ (как XBasePart4.countT); keep – отбор строк по роли. */
    static List<Out1> atT(List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, double[] st, double[] sx, Map<String, Double> rOf, double r0, double t,
                          java.util.function.Predicate<String> keep) {
        XBaseVec.Model tX = XBaseVec.find(tt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(tt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(xs, "main", true);
        List<Out1> out = new ArrayList<>();
        for (int i = 0; i < tX.rows().size(); i++) {
            XBaseVec.Row rx = xM.rows().get(i);
            XBaseVec.Row rt = rOf.get(tX.rows().get(i).key()) <= r0 ? tX.rows().get(i) : tM.rows().get(i);
            if (!rx.key().equals(rt.key())) throw new IllegalStateException("final-report: пробы кандидатов разошлись");
            if (!keep.test(rt.role())) continue;
            int ng = rt.dg().length;
            int b = -1;
            int s2 = -1;
            double[] ln = new double[ng];
            for (int g = 0; g < ng; g++) {
                ln[g] = XBaseFeat2.W_FUSE * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                if (b < 0 || ln[g] < ln[b]) {
                    s2 = b;
                    b = g;
                } else if (s2 < 0 || ln[g] < ln[s2]) {
                    s2 = g;
                }
            }
            double s = 2 * ln[b] - ln[s2];
            out.add(new Out1(rt.role(), rt.person(), rt.trueIdx() >= 0, s <= t, b == rt.trueIdx()));
        }
        return out;
    }

    /** Сводка исходов: ошибка своих, «чужое имя», FAR по попыткам и по людям с КП ↑95. */
    static String summary(List<Out1> l) {
        long own = 0, err = 0, wrong = 0, ca = 0, cn = 0;
        Map<String, Boolean> per = new TreeMap<>();
        for (Out1 o : l) {
            if (o.own()) {
                own++;
                if (!(o.accepted() && o.right())) err++;
                if (o.accepted() && !o.right()) wrong++;
            } else {
                cn++;
                if (o.accepted()) ca++;
                per.merge(o.person(), o.accepted(), Boolean::logicalOr);
            }
        }
        long pa = per.values().stream().filter(x -> x).count();
        return String.format(Locale.ROOT, "%s | %s | %d/%d = %.2f %%, КП ↑%.2f %% | %d/%d = %.1f %%, КП ↑%.1f %% | %s", own == 0 ? "–" : String.format(Locale.ROOT,
                "%d/%d = %.1f %%", err, own, 100.0 * err / own), own == 0 ? "–" : String.format(Locale.ROOT, "%d (%.1f %%)", wrong, 100.0 * wrong / own), ca, cn,
                cn == 0 ? 0 : 100.0 * ca / cn, cn == 0 ? 0 : 100 * FaceEvaluation.binomialUpperBound((int) ca, (int) cn, 0.95), pa, per.size(), per.isEmpty() ? 0
                        : 100.0 * pa / per.size(), per.isEmpty() ? 0 : 100 * FaceEvaluation.binomialUpperBound((int) pa, per.size(), 0.95), cn == 0 ? "–"
                        : 100 * FaceEvaluation.binomialUpperBound((int) ca, (int) cn, 0.95) <= 1.1 ? "держит (КП ≤ 1,1 %)" : "не держит");
    }

    static final String SUM_HEAD = "ошибка своих | «чужое имя» | FAR по попыткам | FAR по людям | FAR 1 % (КП ≤ 1,1 %)";

    /** Отчёт по базе. */
    static void report(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("final-report: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        String tf = System.getenv("FACES_P4_TABLE");
        if (tf == null) throw new IllegalStateException("final-report: задайте FACES_P4_TABLE – таблица (Н) итогового прогона");
        Map<Integer, XBasePart4.TRow> table = XBasePart4.readTable(Paths.get(tf));
        double sigma = XBaseFinal.chosenSigma(out);
        XBaseFinal.Run r = XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        XBaseFinal.Cfg tt = sigma == 1 ? XBaseFinal.TT1 : XBaseFinal.TT05;
        XBaseFinal.Cfg tt0 = sigma == 1 ? XBaseFinal.TT05 : XBaseFinal.TT1;
        boolean hasOther = XBaseFinal.cfgs(false, base).contains(tt0);
        // кадры «а»: тот же блок в final_a (FERET – reports\eq\final_a)
        Path dirA = base.equals("feret") ? u.su.dir.resolve("reports").resolve("eq").resolve("final_a") : out.resolve("final_a");
        if (r.sm() > 0) dirA = dirA.resolveSibling(dirA.getFileName() + "_smoke" + r.sm());
        XBaseFinal.Run ra = new XBaseFinal.Run(u, r.k2(), Map.of(), dirA, true, 0);
        String fl = "слияние σ₀ = " + (sigma == 1 ? "1" : "0,5");
        String flo = "слияние σ₀ = " + (sigma == 1 ? "0,5" : "1");
        Path scf = r.dir().resolve("final_" + base + "_scores.tsv.gz");
        Map<String, Map<String, List<String>>> sfLines = sfaceLines(scf);
        int[][] w = u.tpl.b > 0 ? FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        StringBuilder t = new StringBuilder("Итоговый прогон перед ПО – отчёт – " + base + " (XBaseEval final-report " + base + ")\nКод: коммит " + FarMethods.commit()
                + "; модели – " + r.dir().resolve("vec") + "; кадры «а» – " + dirA + "\nТаблица (Н) – " + tf + " (SHA-256 " + FrameB.sha256(Paths.get(tf)) + "); σ₀ = "
                + sigma + " (Ф1б); бутстреп по людям B = " + u.tpl.b + ", seed " + (u.tpl.seed + FeiBootstrap.SEED_OFFSET) + "; людей " + u.persons + "\n");
        t.append("Интервалы сравнений: [2,5; 97,5] – 95 %, {0,25; 99,75} – 99,5 % (Бонферрони, k = 10); «*» – 95 % без нуля, «**» – и 99,5 %.\n");
        if (base.equals("feret")) t.append("FERET – база подбора конфигурации (не меньше 14 выборов на пуле FERET); независимые проверки – GT и VidTIMIT.\n");
        if (base.equals("own")) t.append("Своя база – одна сессия, оценка оптимистична.\n");
        if (base.equals("vt")) t.append("VidTIMIT – мягкая межсессионная проверка (неделя, та же камера).\n");
        StringBuilder body = new StringBuilder();
        StringBuilder work = new StringBuilder("\n=== Рабочий порог (34): t(N) таблицы (Н), приведения по (Т2) своей галереи; s ≤ t – приём ===\nгруппа | строка | N (узел) | t | "
                + "t выбран | " + SUM_HEAD + "\n");
        StringBuilder cmp = new StringBuilder("\n=== Сравнения разд. 3 (разделимость при FAR 1 % | EER | pAUC ≤ 10 %; п. п.) ===\nпара | FAR 1 % | EER | pAUC\n");
        StringBuilder meas = new StringBuilder("\n=== Меры разд. 3: разделимость при фактическом FAR контроля 1, 2, 5, 10 %, EER, pAUC ≤ 10 % (95 % интервал) ===\nгруппа | строка | 1 % | "
                + "2 % | 5 % | 10 % | EER | pAUC\n");
        Map<String, M> all = new LinkedHashMap<>();
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> b.group().equals(grp)).toList();
            boolean hasA = Files.exists(XBaseFinal.vfile(ra, bs.get(0), XBaseFinal.TT1));
            Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
            Map<String, M> ms = new LinkedHashMap<>();
            Map<String, List<Out1>> wt = new LinkedHashMap<>();
            Map<String, double[]> tUsed = new LinkedHashMap<>();
            List<Object[]> rows = new ArrayList<>();
            rows.add(new Object[] {fl, r, tt, true});
            if (hasOther) rows.add(new Object[] {flo, r, tt0, true});
            rows.add(new Object[] {tt.name(), r, tt, false});
            rows.add(new Object[] {XBaseFinal.XS.name(), r, XBaseFinal.XS, false});
            if (hasA) rows.add(new Object[] {"слияние, кадры «а»", ra, XBaseFinal.TT1, true});
            int[] clamped = new int[2];
            for (Object[] row : rows) {
                String lab = grp + " " + row[0];
                XBaseFinal.Run rr = (XBaseFinal.Run) row[1];
                XBaseFinal.Cfg c = (XBaseFinal.Cfg) row[2];
                boolean fus = (Boolean) row[3];
                List<XBaseThr.Unit> us = new ArrayList<>();
                List<Out1> o1 = new ArrayList<>();
                double[] tu = null;
                for (XBaseVec.Blk b : bs) {
                    List<XBaseVec.Model> mt = XBaseFinal.models(rr, b, c, false);
                    List<XBaseVec.Model> mx = XBaseFinal.models(rr, b, XBaseFinal.XS, false);
                    String lines;
                    XBaseEval.Gt g = b.g();
                    if (fus) {
                        double[] st = XBaseVec.fuseStats(mt, false);
                        double[] sx = XBaseVec.fuseStats(mx, false);
                        lines = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, st, sx, g.rOf, g.r0, clamped);
                        tu = tOf(table, b.g().own.size());
                        o1.addAll(atT(mt, mx, st, sx, g.rOf, g.r0, tu[0], ro -> ro.equals(FeiScores.OWN) || ro.equals(FeiScores.CTRL)));
                    } else {
                        lines = XBaseVec.normLines(mt, null, clamped);
                        g = XBaseAbl3.mirrorGt(b.g(), c.f() == 0);
                    }
                    us.add(new XBaseThr.Unit(g, b.gal(), b.pro(), b.parts(), lines, null, t2));
                }
                XBaseThr.Out o = XBaseThr.evaluate(lab, us, u.persons, u.tpl);
                of.put(lab, o);
                ms.put(lab, measure(o.res.get(XBaseThr.WAYS[0]).cfgs, u.persons, w));
                if (fus) {
                    wt.put(lab, o1);
                    tUsed.put(lab, tu);
                }
            }
            if (sfLines.containsKey(grp)) {
                List<FeiDecompose.Cfg> cs = new ArrayList<>();
                List<String> ls = sfLines.get(grp).get("SFace");
                for (int i = 0; i < bs.size(); i++) {
                    XBaseVec.Blk b = bs.get(i);
                    cs.add(XBaseEval.parse(b.g(), EnumSet.of(XBaseEval.V.SFACE), b.gal(), b.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList(),
                            b.parts(), ls.get(i)).cfg.get(XBaseEval.V.SFACE));
                }
                ms.put(grp + " SFace", measure(cs, u.persons, w));
            }
            for (Map.Entry<String, M> e : ms.entrySet()) {
                XBaseRule.Sep s = e.getValue().sep;
                List<String> row = new ArrayList<>(List.of(grp, e.getKey().substring(grp.length() + 1)));
                for (int k = 0; k < XBaseRule.FARS.length; k++) row.add(XBaseSepLtp.cell(s.err[k], s.errRep[k]));
                row.add(XBaseSepLtp.cell(s.eer[3], s.eerRep));
                row.add(XBaseSepLtp.cell(e.getValue().pauc, e.getValue().paucRep));
                meas.append(String.join(" | ", row)).append('\n');
            }
            for (Map.Entry<String, List<Out1>> e : wt.entrySet()) {
                double[] tu = tUsed.get(e.getKey());
                String by = table.get((int) tu[1]).by();
                boolean self = by.toLowerCase(Locale.ROOT).startsWith(base.equals("feret") ? "feret " + grp.toLowerCase(Locale.ROOT) : base.equals("gt") ? "gt "
                        + grp.toLowerCase(Locale.ROOT) : base);
                work.append(String.format(Locale.ROOT, "%s | %s | %d (%d)%s | %.2f | %s | %s%n", grp, e.getKey().substring(grp.length() + 1), bs.get(0).g().own.size(),
                        (int) tu[1], tu[2] > 0 ? " – N > 200, предупреждение" : "", tu[0], (self ? "по этой строке" : "по другой") + " (" + by + ")", summary(e.getValue())));
            }
            all.putAll(ms);
            // сравнения внутри группы
            List<String[]> pairs = new ArrayList<>();
            pairs.add(new String[] {grp + " " + fl, grp + " SFace"});
            pairs.add(new String[] {grp + " " + fl, grp + " " + tt.name()});
            if (hasA) pairs.add(new String[] {grp + " " + fl, grp + " слияние, кадры «а»"});
            if (hasOther) pairs.add(new String[] {grp + " слияние σ₀ = 0,5", grp + " слияние σ₀ = 1"});
            for (String[] p : pairs) {
                M a = ms.get(p[0]);
                M z = ms.get(p[1]);
                if (a == null || z == null) continue;
                cmp.append(p[0]).append(" − ").append(p[1].substring(grp.length() + 1)).append(" | ").append(ci(a.sep.err[0] - z.sep.err[0], FeiNorm.diff(a.sep.errRep[0],
                        z.sep.errRep[0]))).append(" | ").append(ci(a.sep.eer[3] - z.sep.eer[3], FeiNorm.diff(a.sep.eerRep, z.sep.eerRep))).append(" | ").append(ci(a.pauc
                        - z.pauc, FeiNorm.diff(a.paucRep, z.paucRep))).append('\n');
            }
            XBaseLbp.tables(body, (base.equals("feret") ? "FERET " : base.toUpperCase(Locale.ROOT) + " ") + grp + " – П3", of, u.persons, u.tpl, List.of(), t2);
            if (base.equals("gt") && grp.equals("межс.")) {
                body.append(phase2(r, bs, table, tt));
                body.append(phase5(r, bs, table, tt, dir));
            }
            if (t2 || base.equals("vt")) body.append(oneToOne(r, bs, tt, grp, u, t2));
        }
        if (base.equals("vt")) {
            M a = all.get("M = 2 " + fl);
            M z = all.get("M = 1 " + fl);
            cmp.append("M = 2 − M = 1, ").append(fl).append(" | ").append(ci(a.sep.err[0] - z.sep.err[0], FeiNorm.diff(a.sep.errRep[0], z.sep.errRep[0]))).append(" | ")
                    .append(ci(a.sep.eer[3] - z.sep.eer[3], FeiNorm.diff(a.sep.eerRep, z.sep.eerRep))).append(" | ").append(ci(a.pauc - z.pauc, FeiNorm.diff(a.paucRep,
                    z.paucRep))).append("\n");
        }
        if (base.equals("vt")) t.append(vtParts(r, u, table, tt));
        t.append(work).append(meas).append(cmp).append(body);
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Path rf = r.dir().resolve("final_report_" + base + ".txt");
        FarMethods.write(rf, t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Строки SFace файла оценок Ф1: группа → подпись → строки по блокам (в порядке блоков). */
    static Map<String, Map<String, List<String>>> sfaceLines(Path f) throws IOException {
        Map<String, Map<String, List<String>>> m = new LinkedHashMap<>();
        if (!Files.exists(f)) return m;
        String grp = null;
        String lab = null;
        StringBuilder cur = null;
        try (java.io.BufferedReader in = new java.io.BufferedReader(new java.io.InputStreamReader(new java.util.zip.GZIPInputStream(Files.newInputStream(f)),
                StandardCharsets.UTF_8))) {
            for (String l; (l = in.readLine()) != null;) {
                if (l.startsWith("# ") && l.contains(" | ")) {
                    if (cur != null && lab.equals("SFace")) m.computeIfAbsent(grp, k -> new LinkedHashMap<>()).computeIfAbsent(lab, k -> new ArrayList<>()).add(cur.toString());
                    String[] c = l.substring(2).split(" \\| ");
                    grp = c[0];
                    lab = c[1];
                    cur = new StringBuilder();
                } else if (cur != null && !l.startsWith("#")) {
                    cur.append(l).append('\n');
                }
            }
        }
        if (cur != null && "SFace".equals(lab)) m.computeIfAbsent(grp, k -> new LinkedHashMap<>()).computeIfAbsent(lab, k -> new ArrayList<>()).add(cur.toString());
        return m;
    }

    /** Ф2: контрольные чужие FEI и MUCT вне W – FAR при рабочем пороге по источникам, по попыткам и по людям; «вместе» – среднее по источникам. */
    static String phase2(XBaseFinal.Run r, List<XBaseVec.Blk> bs, Map<Integer, XBasePart4.TRow> table, XBaseFinal.Cfg tt) throws IOException {
        Map<String, List<Out1>> by = new TreeMap<>();
        for (XBaseVec.Blk b : bs) {
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, tt, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            double[] st = XBaseVec.fuseStats(mt, false);
            double[] sx = XBaseVec.fuseStats(mx, false);
            double t = tOf(table, b.g().own.size())[0];
            List<XBaseVec.Model> et = XBaseFinal.models(r, b, tt, true);
            List<XBaseVec.Model> ex = XBaseFinal.models(r, b, XBaseFinal.XS, true);
            for (Out1 o : atT(et, ex, st, sx, b.g().rOf, b.g().r0, t, ro -> true)) by.computeIfAbsent(o.person().substring(0, o.person().indexOf(':')), k -> new ArrayList<>()).add(o);
            for (Out1 o : atT(mt, mx, st, sx, b.g().rOf, b.g().r0, t, ro -> ro.equals(FeiScores.CTRL))) by.computeIfAbsent("gt", k -> new ArrayList<>()).add(o);
        }
        StringBuilder s = new StringBuilder("\n=== Ф2: расширенный контроль GT межс. – FAR при рабочем пороге по источникам (камеры другие; абсолютный FAR GT – по своему "
                + "контролю) ===\nисточник | " + SUM_HEAD + "\n");
        double sum = 0;
        for (Map.Entry<String, List<Out1>> e : by.entrySet()) {
            s.append(e.getKey()).append(" | ").append(summary(e.getValue())).append('\n');
            long a = e.getValue().stream().filter(Out1::accepted).count();
            sum += 100.0 * a / e.getValue().size();
        }
        s.append(String.format(Locale.ROOT, "вместе (равновзвешенно по источникам, по попыткам): %.2f %%%n", sum / by.size()));
        return s.toString();
    }

    /** Ф5: GT межс. – разбивка своих по интервалу «дата пробы − дата галереи» (≤ 7 дней, > 30 дней; gt_sessions.tsv) при рабочем пороге. */
    static String phase5(XBaseFinal.Run r, List<XBaseVec.Blk> bs, Map<Integer, XBasePart4.TRow> table, XBaseFinal.Cfg tt, Path dir) throws IOException {
        Map<String, java.time.LocalDate> date = new HashMap<>();
        for (String l : Files.readAllLines(dir.resolve("reports").resolve("gt_sessions.tsv"), StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t")) continue;
            String[] c = l.split("\t");
            if (c[2].length() < 10 || !Character.isDigit(c[2].charAt(0))) continue;
            date.put(c[0] + "/" + c[1], java.time.LocalDate.parse(c[2].substring(0, 10).replace(':', '-')));
        }
        Map<String, List<Out1>> by = new LinkedHashMap<>();
        by.put("≤ 7 дней", new ArrayList<>());
        by.put("8–30 дней", new ArrayList<>());
        by.put("> 30 дней", new ArrayList<>());
        int miss = 0;
        for (XBaseVec.Blk b : bs) {
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, tt, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            double[] st = XBaseVec.fuseStats(mt, false);
            double[] sx = XBaseVec.fuseStats(mx, false);
            double t = tOf(table, b.g().own.size())[0];
            XBaseVec.Model tX = XBaseVec.find(mt, "main", false);
            List<Out1> o = atT(mt, mx, st, sx, b.g().rOf, b.g().r0, t, ro -> ro.equals(FeiScores.OWN));
            int i = 0;
            for (XBaseVec.Row row : tX.rows()) {
                if (!row.role().equals(FeiScores.OWN)) continue;
                java.time.LocalDate dp = date.get(gtKey(row.key()));
                java.time.LocalDate dg = date.get(gtKey(b.gal().get(row.person()).get(0)));
                Out1 x = o.get(i++);
                if (dp == null || dg == null) {
                    miss++;
                    continue;
                }
                long days = Math.abs(java.time.temporal.ChronoUnit.DAYS.between(dg, dp));
                by.get(days <= 7 ? "≤ 7 дней" : days <= 30 ? "8–30 дней" : "> 30 дней").add(x);
            }
        }
        StringBuilder s = new StringBuilder("\n=== Ф5: GT межс. – ошибка своих при рабочем пороге по интервалу «дата пробы − дата галереи» (gt_sessions.tsv) ===\n"
                + "интервал | " + SUM_HEAD + "\n");
        for (Map.Entry<String, List<Out1>> e : by.entrySet()) s.append(e.getKey()).append(" | ").append(summary(e.getValue())).append('\n');
        if (miss > 0) s.append("без даты: ").append(miss).append(" проб\n");
        return s.toString();
    }

    /** Ключ GT экспорта (gt\sNN\MM) → «sNN/MM» gt_sessions.tsv. */
    static String gtKey(String k) {
        Path p = Paths.get(k);
        return p.getName(p.getNameCount() - 2) + "/" + p.getName(p.getNameCount() - 1);
    }

    /** 1:1 – худший случай (XBaseAbl3.oneLines по строкам слияния) рядом с 1:N; случайно заявленный – разделимость по моделям. */
    static String oneToOne(XBaseFinal.Run r, List<XBaseVec.Blk> bs, XBaseFinal.Cfg tt, String grp, XBaseVec.Setup u, boolean t2) throws IOException {
        List<List<XBaseThr.Unit>> us = List.of(new ArrayList<>(), new ArrayList<>());
        int[] clamped = new int[2];
        for (XBaseVec.Blk b : bs) {
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, tt, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            String lines = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, false), XBaseVec.fuseStats(mx, false), b.g().rOf, b.g().r0, clamped);
            us.get(0).add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), lines, null, t2));
            us.get(1).add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), XBaseAbl3.oneLines(lines), null, t2));
        }
        StringBuilder body = new StringBuilder();
        XBaseAbl3.own11(body, new StringBuilder(), "Режим 1:1 (отдельная таблица) – " + grp, grp + " ", us, u.persons, u.tpl, t2);
        return body.toString();
    }

    /** Ф3: строки «фразы», «поворот» и вместе при рабочем пороге N = 22; M = 2 − M = 1 (слияние) – по ротациям вместе. */
    static String vtParts(XBaseFinal.Run r, XBaseVec.Setup u, Map<Integer, XBasePart4.TRow> table, XBaseFinal.Cfg tt) throws IOException {
        StringBuilder s = new StringBuilder("\n=== Ф3: при рабочем пороге (N = 22 – между узлами 10 и 50, меньший t) – фразы, поворот, вместе ===\nM | пробы | " + SUM_HEAD
                + "\n");
        for (String grp : u.groups) {
            Map<String, List<Out1>> by = new LinkedHashMap<>();
            by.put("фразы", new ArrayList<>());
            by.put("поворот", new ArrayList<>());
            by.put("вместе", new ArrayList<>());
            for (XBaseVec.Blk b : u.blocks.stream().filter(x -> x.group().equals(grp)).toList()) {
                List<XBaseVec.Model> mt = XBaseFinal.models(r, b, tt, false);
                List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
                double[] st = XBaseVec.fuseStats(mt, false);
                double[] sx = XBaseVec.fuseStats(mx, false);
                double t = tOf(table, b.g().own.size())[0];
                XBaseVec.Model tX = XBaseVec.find(mt, "main", false);
                List<Out1> o = atT(mt, mx, st, sx, b.g().rOf, b.g().r0, t, ro -> ro.equals(FeiScores.OWN) || ro.equals(FeiScores.CTRL));
                int i = 0;
                for (XBaseVec.Row row : tX.rows()) {
                    if (!(row.role().equals(FeiScores.OWN) || row.role().equals(FeiScores.CTRL))) continue;
                    Out1 x = o.get(i++);
                    by.get(row.key().contains("/head3/") ? "поворот" : "фразы").add(x);
                    by.get("вместе").add(x);
                }
            }
            for (Map.Entry<String, List<Out1>> e : by.entrySet()) s.append(grp).append(" | ").append(e.getKey()).append(" | ").append(summary(e.getValue())).append('\n');
        }
        return s.toString();
    }
}
