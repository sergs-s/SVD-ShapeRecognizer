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

    // ---------------------------------------------------------------- досчитано 04.11 (Жорик №1 от 04.11)

    static final String ADDON_MARK = "######## Досчитано 04.11";

    /**
     * Строки 1:1 слияния: у своего заявлен свой; у чужого – seed &lt; 0: худший случай (ближайший по ℓ), иначе случайный человек галереи
     * (Random как XBaseVec.oneRandom, по порядку проб); оценка – ℓ(заявленного) после приведения по (Т2), сжатие XBaseVec.squash; второе –
     * постоянная. Модель L1-tt-xs-F – по позе, L1-xs-F – X+mirror; строка – под обоими методами, как XBaseVec.fuseLines.
     */
    static String fuseOne(List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, double w, double[] st, double[] sx, Map<String, Double> rOf, double r0, long seed) {
        StringBuilder b = new StringBuilder();
        Map<String, XBaseVec.Model> byTask = new LinkedHashMap<>();
        for (XBaseVec.Model m : tt) byTask.putIfAbsent(m.task(), m);
        for (String task : byTask.keySet()) {
            XBaseVec.Model tX = XBaseVec.find(tt, task, false);
            XBaseVec.Model tM = XBaseVec.find(tt, task, true);
            XBaseVec.Model xM = XBaseVec.find(xs, task, true);
            java.util.Random rnd = new java.util.Random(seed + task.hashCode() * 31L + tX.method().hashCode());
            for (int i = 0; i < tX.rows().size(); i++) {
                XBaseVec.Row rx = xM.rows().get(i);
                XBaseVec.Row r0r = tX.rows().get(i);
                if (!rx.key().equals(r0r.key()) || !rx.role().equals(r0r.role())) throw new IllegalStateException("final-addon: слияние – разные пробы");
                XBaseVec.Row rt = rOf.get(r0r.key()) <= r0 ? r0r : tM.rows().get(i);
                int ng = rt.dg().length;
                double[] ln = new double[ng];
                List<Integer> ok = new ArrayList<>();
                for (int g = 0; g < ng; g++) {
                    if (Double.isInfinite(rt.dg()[g]) || Double.isInfinite(rx.dg()[g])) {
                        ln[g] = Double.POSITIVE_INFINITY;
                        continue;
                    }
                    ln[g] = w * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - w) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                    ok.add(g);
                }
                int c = rt.trueIdx();
                if (c < 0) {
                    if (seed >= 0) {
                        c = ok.get(rnd.nextInt(ok.size()));
                    } else {
                        c = ok.get(0);
                        for (int g : ok) if (ln[g] < ln[c]) c = g;
                    }
                }
                double[] v = new double[ng];
                java.util.Arrays.fill(v, XBaseAbl.ABS_C);
                v[c] = XBaseAbl.ABS_C * XBaseVec.squash(ln[c], null);
                for (XBaseVec.Model m : List.of(tX, tM)) b.append(FeiScores.line(m.method(), m.s(), 0, rt.role(), rt.person(), rt.image(), rt.trueIdx(), m.gallery(), v, false));
            }
        }
        return b.toString();
    }

    /** Строка таблицы 1:1: разделимость, EER, П3. */
    static String oneRow(String lab, XBaseThr.Out o, String way, int persons, XBaseEval.Gt tpl, int[][] w, double[][] acc) {
        XBaseRule.Sep s = XBaseRule.sep(o.res.get(XBaseThr.WAYS[0]).cfgs, FeiBootstrap.unit(persons), w);
        FeiThreshold.Res r = o.res.get(way);
        List<String> row = new ArrayList<>(List.of(lab));
        for (int k = 0; k < XBaseRule.FARS.length; k++) row.add(XBaseSepLtp.cell(s.err[k], s.errRep[k]));
        row.add(XBaseSepLtp.cell(s.eer[3], s.eerRep));
        row.add(FeiThreshold.errCell(r, 3, 0, 0, tpl.b));
        row.add(FeiThreshold.farCell(r, 3, 0, tpl.b));
        if (acc != null) {
            double[] cnt = r.counts[3][0][0];
            double own = java.util.Arrays.stream(cnt).sum();
            for (int k = 0; k < XBaseRule.FARS.length; k++) acc[0][k] += s.err[k];
            acc[0][XBaseRule.FARS.length] += s.eer[3];
            acc[1][0] += 100 * (own - cnt[FeiDecompose.G_OK]) / own;
            acc[1][1] += 100 * r.far[3][0][0] / r.far[3][0][1];
        }
        return String.join(" | ", row) + "\n";
    }

    /** 1:1 группы: 1:N, худший случай, 5 случайных повторов (seed faces.seed + 10600 + повтор, как пачка 1) и их среднее – для слияния и L1-tt-xs-F. */
    static String oneTable(XBaseFinal.Run r, List<XBaseVec.Blk> bs, XBaseFinal.Cfg tt, String grp, XBaseVec.Setup u, boolean t2, int[][] w) throws IOException {
        StringBuilder s = new StringBuilder("\n=== 1:1 – " + grp + " (П3 FAR 1 % по «чужим» " + (t2 ? "(Т2)" : "(Т0)") + "; в 1:1 «чужие» порога заявляют так же, как контроль) ===\nстрока | разд. 1 % | 2 % | 5 % | "
                + "10 % | EER | П3: ошибка своих | FAR контроля\n");
        String way = XBaseThr.WAYS[t2 ? 2 : 0];
        int[] clamped = new int[2];
        for (int f = 0; f < 2; f++) {
            String feat = f == 0 ? "слияние" : "L1-tt-xs-F";
            double[][] acc = new double[2][XBaseRule.FARS.length + 1];
            for (int rep = -1; rep <= 5; rep++) {
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (XBaseVec.Blk b : bs) {
                    List<XBaseVec.Model> mt = XBaseFinal.models(r, b, tt, false);
                    XBaseEval.Gt g = f == 0 ? b.g() : XBaseAbl3.mirrorGt(b.g(), false);
                    String ls;
                    if (f == 0) {
                        List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
                        double[] st = XBaseVec.fuseStats(mt, false);
                        double[] sx = XBaseVec.fuseStats(mx, false);
                        ls = rep < 0 ? XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, st, sx, g.rOf, g.r0, clamped) : fuseOne(mt, mx, XBaseFeat2.W_FUSE, st, sx, g.rOf, g.r0,
                                rep == 0 ? -1 : u.tpl.seed + 10600 + rep);
                    } else {
                        ls = rep < 0 ? XBaseVec.normLines(mt, null, clamped) : rep == 0 ? XBaseAbl3.oneLines(XBaseVec.lines(mt)) : XBaseVec.oneRandom(mt, u.tpl.seed + 10600 + rep);
                    }
                    us.add(new XBaseThr.Unit(g, b.gal(), b.pro(), b.parts(), ls, null, t2));
                }
                String lab = grp + " " + feat + (rep < 0 ? " 1:N" : rep == 0 ? " 1:1, заявлен ближайший (худший случай)" : " 1:1 случайный, повтор " + rep);
                XBaseThr.Out o = XBaseThr.evaluate(lab, us, u.persons, u.tpl);
                s.append(oneRow(lab, o, way, u.persons, u.tpl, w, rep > 0 ? acc : null));
            }
            s.append(String.format(Locale.ROOT, "%s %s 1:1 случайный, среднее 5 повторов | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.2f %%%n", grp, feat, acc[0][0] / 5, acc[0][1] / 5,
                    acc[0][2] / 5, acc[0][3] / 5, acc[0][4] / 5, acc[1][0] / 5, acc[1][1] / 5));
        }
        return s.toString();
    }

    /** SFace VidTIMIT: кадры alignCrop из исходных кадров по строкам YuNet Ф3, строки оценок по блокам (без обучения); файл оценок – для пересчёта. */
    static Map<String, List<String>> sfaceVt(XBaseFinal.Run r, SettingsStore settings) throws IOException {
        Path vdir = VidTimitData.dir(settings);
        Path der = vdir.resolve(VidTimitData.DERIVED);
        Map<String, double[]> rows = VidTimitData.readDetections(der.resolve(VidTimitData.DETECTIONS));
        rows.putAll(VidTimitData.readDetections(der.resolve(XBaseFinal.VT_HEAD_DET)));
        org.opencv.objdetect.FaceRecognizerSF sf = org.opencv.objdetect.FaceRecognizerSF.create(Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE).toString(), "");
        Map<String, Mat> crops = new LinkedHashMap<>();
        for (String l : Files.readAllLines(der.resolve(XBaseFinal.VT_LIST), StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t") || l.isBlank()) continue;
            String k = l.split("\t", -1)[4];
            String dk = k.substring(VidTimitData.KEY_PREFIX.length());
            String[] q = dk.split("/");
            Mat color = org.opencv.imgcodecs.Imgcodecs.imread(vdir.resolve(q[0]).resolve("video").resolve(q[1]).resolve(q[2]).toString(), org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR);
            if (color.empty() || rows.get(dk) == null) throw new IOException("final-addon: нет кадра или детекции " + dk);
            crops.put(k, FeretData.frameSface(color, rows.get(dk), sf));
            color.release();
        }
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        xsf.sfaceVectors(crops, sf);
        crops.values().forEach(Mat::release);
        Map<String, List<String>> out = new LinkedHashMap<>();
        StringBuilder sc = new StringBuilder("# SFace VidTIMIT (XBaseFinalReport final-addon, досчитано 04.11; alignCrop по строкам YuNet Ф3, без обучения)\n# Код: коммит " + FarMethods.commit() + "\n");
        for (XBaseVec.Blk b : r.u().blocks) {
            String ls = XBaseFinal.sfaceLines(r, b, xsf);
            out.computeIfAbsent(b.group(), g -> new ArrayList<>()).add(ls);
            sc.append("# ").append(b.group()).append(" | SFace | ").append(b.name()).append('\n').append(ls);
        }
        Path f = r.dir().resolve("final_vt_sface_scores.tsv.gz");
        java.io.Writer w = XBaseVec.writer(f, sc.toString());
        XBaseVec.finish(w, f);
        return out;
    }

    /** final-addon &lt;база&gt;: раздел «досчитано 04.11» в конце final_report_&lt;база&gt;.txt (повтор заменяет раздел; основная часть отчёта не меняется). */
    static void addon(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("final-addon: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        double sigma = XBaseFinal.chosenSigma(out);
        XBaseFinal.Run r = XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        XBaseFinal.Cfg tt = sigma == 1 ? XBaseFinal.TT1 : XBaseFinal.TT05;
        int[][] w = u.tpl.b > 0 ? FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        StringBuilder t = new StringBuilder(ADDON_MARK + " (вне постановки, по её списку; Жорик №1 от 04.11) ########\nКод: коммит " + FarMethods.commit() + "; σ₀ = " + sigma + "; B = " + u.tpl.b
                + ", seed " + (u.tpl.seed + FeiBootstrap.SEED_OFFSET) + "; без обучения.\n");
        if (r.sm() == 0) t.append("Таблицы «Режим 1:1» основной части отчёта (код 8d36075) не использовать: там у своих сырое d₁, у чужих – d₁ после β, шкалы разные. Действует таблица этого раздела.\n");
        Map<String, List<String>> sf = new LinkedHashMap<>();
        if (base.equals("vt")) {
            sf = sfaceVt(r, settings);
        } else {
            for (Map.Entry<String, Map<String, List<String>>> e : sfaceLines(r.dir().resolve("final_" + base + "_scores.tsv.gz")).entrySet()) sf.put(e.getKey(), e.getValue().get("SFace"));
        }
        t.append("\n=== SFace при пороге таблицы (24): 1 − cos с лучшим шаблоном ≤ t; t*(N) = 0,53 / 0,46 / 0,41 / 0,35 при N = 10, 50, 100, 200, между узлами – меньший ===\nгруппа | N | t | ").append(SUM_HEAD)
                .append('\n');
        for (String grp : u.groups) {
            List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> b.group().equals(grp)).toList();
            List<String> ls = sf.get(grp);
            if (ls == null) continue;
            int n = bs.get(0).g().own.size();
            List<Out1> so = new ArrayList<>();
            for (int i = 0; i < bs.size(); i++) so.addAll(sfaceAtT(ls.get(i), sfaceT(bs.get(i).g().own.size())));
            t.append(String.format(Locale.ROOT, "%s | %d | %.2f | %s%n", grp, n, sfaceT(n), summary(so)));
        }
        if (base.equals("feret")) {
            Path c = u.out.resolve("final").resolve("final_cmp10.txt");
            if (Files.exists(c)) t.append("\n=== Сравнение 10 ===\n").append(Files.readString(c, StandardCharsets.UTF_8));
        }
        for (String grp : u.groups) t.append(oneTable(r, u.blocks.stream().filter(b -> b.group().equals(grp)).toList(), tt, grp, u, !grp.equals("внутри"), w));
        t.append(String.format(Locale.ROOT, "%nВремя раздела: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Path rf = r.dir().resolve("final_report_" + base + ".txt");
        String old = Files.exists(rf) ? Files.readString(rf, StandardCharsets.UTF_8) : "";
        int at = old.indexOf(ADDON_MARK);
        if (at >= 0) old = old.substring(0, at);
        FarMethods.write(rf, new StringBuilder(old).append(old.isEmpty() ? "" : "\n\n").append(t));
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Порог SFace по решению (24): t*(N) = 0,53 / 0,46 / 0,41 / 0,35 при N = 10, 50, 100, 200; между узлами – меньший, N &lt; 10 – t(10), N &gt; 200 – t(200). */
    static double sfaceT(int n) {
        int[] nodes = {10, 50, 100, 200};
        double[] t = {0.53, 0.46, 0.41, 0.35};
        for (int i = 0; i < nodes.length; i++) if (n <= nodes[i]) return t[i];
        return t[t.length - 1];
    }

    /** Исходы SFace при пороге t на 1 − cos с лучшим шаблоном: только основные пробы (свои и контроль). */
    static List<Out1> sfaceAtT(String lines, double t) {
        List<Out1> out = new ArrayList<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            boolean own = c[3].equals(FeiScores.OWN);
            if (!own && !c[3].equals(FeiScores.CTRL)) continue;
            out.add(new Out1(c[3], c[4], own, Double.parseDouble(c[8]) <= t, own && c[6].equals(c[7])));
        }
        return out;
    }

    /**
     * final-cmp10 (сравнение 10 разд. 3): рабочий порог (34) − П3 по (Т2), ошибка своих, FERET межс., N = 50, галереи и модели feret-p4d (40
     * галерей). Бутстреп по людям: одни веса на все галереи, B и seed – как в отчётах; пороги галерей в репликах не пересчитываются. Сверка
     * точки с итогом feret-p4d – по печатаемым счётам.
     */
    static void cmp10(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("final-cmp10: FACES_FRAMES=b");
        String tf = System.getenv("FACES_P4_TABLE");
        if (tf == null) throw new IllegalStateException("final-cmp10: задайте FACES_P4_TABLE");
        double t = XBasePart4.readTable(Paths.get(tf)).get(50).t();
        XBaseVec.Setup u = XBaseVec.setup("feret", export, dir, feiKeys, selLines, settings, out, 0);
        for (Mat m : u.fr.values()) m.release();
        Path cd = u.out.resolve("xbase_feret_p4d");
        XBaseEval.Gt tpl0 = XBaseAbl.copy(u.tpl, u.tpl.cohort, u.tpl.rOf, u.tpl.r0);
        tpl0.b = 0;
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        double[][] e = new double[2][w.length];
        double[] n = new double[w.length];
        long[] pe = new long[2];
        long po = 0;
        int row = 0;
        for (XBaseVec.Blk b : u.blocks) {
            if (!b.group().equals("межс.")) continue;
            long seed = u.tpl.seed + XBasePart4.D_SEED_OFFSET + 100L * row + 50;
            row++;
            java.util.Random rnd = new java.util.Random(seed);
            Map<String, List<XBaseVec.Model>> mt = XBaseVec.read(cd.resolve(b.name() + "_n50_f1.tsv.gz"));
            Map<String, List<XBaseVec.Model>> mx = XBaseVec.read(cd.resolve(b.name() + "_n50_f0.tsv.gz"));
            for (int k = 0; k < XBasePart4.GALLERIES; k++) {
                List<String> perm = new ArrayList<>(b.g().own);
                java.util.Collections.shuffle(perm, rnd);
                List<String> own = new ArrayList<>(perm.subList(0, 50));
                java.util.Collections.sort(own);
                XBasePart4.DG d = XBasePart4.gallery(b, own, seed, k, true);
                List<XBaseVec.Model> tt = mt.get("g" + k);
                List<XBaseVec.Model> xs = mx.get("g" + k);
                double[] st = XBaseVec.fuseStats(tt, false);
                double[] sx = XBaseVec.fuseStats(xs, false);
                String lines = XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, st, sx, d.g().rOf, d.g().r0, null);
                FeiThreshold.Res r = XBaseThr.evaluate("g" + k, List.of(new XBaseThr.Unit(d.g(), d.gal(), d.pro(), d.parts(), lines, null, true)), u.persons, tpl0).res
                        .get(XBaseThr.WAYS[2]);
                FeiDecompose.Cfg c = r.cfgs.get(0);
                double th = r.theta[3][0][0];
                for (int i = 0; i < c.ownR.length; i++) {
                    boolean err = !(c.ownOk[i] && c.ownR[i] <= th);
                    if (err) pe[0]++;
                    po++;
                    for (int q = 0; q < w.length; q++) {
                        int wt = w[q][c.ownP[i]];
                        n[q] += wt;
                        if (err) e[0][q] += wt;
                    }
                }
                for (Out1 o : atT(tt, xs, st, sx, d.g().rOf, d.g().r0, t, ro -> ro.equals(FeiScores.OWN))) {
                    boolean err = !(o.accepted() && o.right());
                    if (!err) continue;
                    pe[1]++;
                    int p = d.g().pIdx.get(o.person());
                    for (int q = 0; q < w.length; q++) e[1][q] += w[q][p];
                }
            }
        }
        double[] rep = new double[w.length];
        for (int q = 0; q < w.length; q++) rep[q] = 100 * (e[1][q] - e[0][q]) / n[q];
        double pa = 100.0 * pe[0] / po;
        double pb = 100.0 * pe[1] / po;
        StringBuilder s = new StringBuilder("Итоговый прогон, сравнение 10 разд. 3 (XBaseEval final-cmp10)\nКод: коммит " + FarMethods.commit() + "; модели – " + cd + "; t(50) = " + t + "\n");
        s.append("FERET межс., N = 50, 40 галерей; бутстреп по людям B = ").append(u.tpl.b).append(", seed ").append(u.tpl.seed + FeiBootstrap.SEED_OFFSET)
                .append("; пороги галерей в репликах не пересчитываются (интервал – по составу людей при данных порогах).\n");
        s.append(String.format(Locale.ROOT, "П3 по (Т2): %d/%d = %.1f %%; рабочий порог (34): %d/%d = %.1f %%%n", pe[0], po, pa, pe[1], po, pb));
        s.append("рабочий порог − П3, ошибка своих, п. п.: ").append(ci(pb - pa, rep)).append('\n');
        Path f = u.out.resolve("final").resolve("final_cmp10.txt");
        FarMethods.write(f, s);
        System.out.print(s);
        System.out.println("DONE");
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
