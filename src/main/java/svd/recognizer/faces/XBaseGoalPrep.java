package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Этап «к цели», подготовка (Жорик №12 от 04.11, разбор Зануды, разд. 5.1): goal-prep &lt;gt | feret&gt; – только по готовым моделям и файлам оценок
 * итогового прогона, без обучения. П. 1 – полуширина 95 % интервала парных разностей известных пар (наименьший различимый эффект мерки) и объём
 * контроля. П. 2 (О-11) – итоговая конфигурация против конфигурации 23.10 (L1-tt-xs-F, евклид, 7×7: gt-subsp k = 0, роль «межс.» feret-xs2; те же
 * пробы, кадры «а»). П. 3 (О-10) – поза r с поправкой на наклон: доля проб, у которых меняется выбор X / X+mirror, и слияние при новом выборе
 * (обе модели уже посчитаны; состав проб – по прежнему отбору R, приведения те же). Бутстреп по людям B (FACES_FEI_BOOTSTRAP), seed faces.seed + 7000.
 *
 * @author ssv
 */
final class XBaseGoalPrep {

    private XBaseGoalPrep() {
    }

    /** Меры в порядке вывода: подпись, значение, повторы. */
    private static final String[] MEASURES = {"FAR 1 %", "FAR 2 %", "FAR 5 %", "FAR 10 %", "EER", "pAUC ≤ 10 %"};

    private static double[] point(XBaseFinalReport.M m) {
        return new double[] {m.sep.err[0], m.sep.err[1], m.sep.err[2], m.sep.err[3], m.sep.eer[3], m.pauc};
    }

    private static double[][] reps(XBaseFinalReport.M m) {
        return new double[][] {m.sep.errRep[0], m.sep.errRep[1], m.sep.errRep[2], m.sep.errRep[3], m.sep.eerRep, m.paucRep};
    }

    /** Разность a − z: точка, 95 % интервал, полуширина, ст. откл. по повторам; повторы с NaN (порог не определён) отбрасываются и считаются. */
    private static String diffRow(String lab, XBaseFinalReport.M a, XBaseFinalReport.M z, Map<String, List<Double>> widths) {
        StringBuilder b = new StringBuilder(lab);
        double[] pa = point(a);
        double[] pz = point(z);
        double[][] ra = reps(a);
        double[][] rz = reps(z);
        for (int k = 0; k < MEASURES.length; k++) {
            double[] d = FeiNorm.diff(ra[k], rz[k]);
            double[] ok = Arrays.stream(d).filter(x -> !Double.isNaN(x)).toArray();
            double lo = Double.NaN;
            double hi = Double.NaN;
            String sign = "";
            if (ok.length > 0) {
                double[] s = FeiBootstrap.sorted(ok);
                lo = FeiBootstrap.quantile(s, 0.025);
                hi = FeiBootstrap.quantile(s, 0.975);
                sign = lo > 0 || hi < 0 ? " *" : "";
            }
            double half = (hi - lo) / 2;
            if (widths != null) widths.computeIfAbsent(MEASURES[k], x -> new ArrayList<>()).add(half);
            b.append(String.format(Locale.ROOT, " | %+.2f [%+.2f; %+.2f]%s ±%.2f%s", pa[k] - pz[k], lo, hi, sign, half, ok.length < d.length ? " (NaN " + (d.length - ok.length) + ")" : ""));
        }
        return b.append('\n').toString();
    }

    private static String levelRow(String lab, XBaseFinalReport.M m) {
        StringBuilder b = new StringBuilder(lab);
        double[] p = point(m);
        double[][] r = reps(m);
        for (int k = 0; k < MEASURES.length; k++) b.append(" | ").append(XBaseSepLtp.cell(p[k], r[k]));
        return b.append('\n').toString();
    }

    private static String head() {
        return String.join(" | ", MEASURES);
    }

    /** Оценки основной модели как конфигурации разбора (порог по «чужим» здесь не нужен: части пусты). */
    private static FeiDecompose.Cfg cfg(XBaseVec.Blk b, XBaseEval.Gt g, String lines) {
        List<List<String>> none = new ArrayList<>();
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) none.add(List.of());
        return XBaseEval.parse(g, EnumSet.of(V.L1), b.gal(), XBaseEval.probes(g, b.pro()), none, lines).cfg.get(V.L1);
    }

    private static String fused(XBaseFinal.Run r, XBaseVec.Blk b, XBaseFinal.Cfg c, Map<String, Double> rOf, double r0) throws IOException {
        List<XBaseVec.Model> mt = XBaseFinal.models(r, b, c, false);
        List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
        return XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, false), XBaseVec.fuseStats(mx, false), rOf, r0, new int[2]);
    }

    /** |r′|: смещение носа от середины глаз вдоль линии глаз, в долях межглазья (поправка на наклон головы; без поправки – FeiDataset.pose). */
    static double poseTilt(double[] row) {
        double dx = row[6] - row[4];
        double dy = row[7] - row[5];
        return Math.abs(((row[8] - (row[4] + row[6]) / 2) * dx + (row[9] - (row[5] + row[7]) / 2) * dy) / (dx * dx + dy * dy));
    }

    /** Строки детекций YuNet: FERET – кэш детекций, прочие базы – экспорт. */
    private static double[] detRow(XBaseVec.Setup u, Map<String, double[]> exportRows, String k) {
        double[] row = u.base.equals("feret") ? u.su.det.get(k) : exportRows.get(k);
        if (row == null) throw new IllegalStateException("goal-prep: нет строки YuNet для " + k);
        return row;
    }

    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("goal-prep: FACES_FRAMES=b");
        if (!base.equals("gt") && !base.equals("feret")) throw new IllegalArgumentException("goal-prep: gt | feret");
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup(base, false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        Path dirA = base.equals("feret") ? u.su.dir.resolve("reports").resolve("eq").resolve("final_a") : out.resolve("final_a");
        XBaseFinal.Run ra = new XBaseFinal.Run(u, r.k2(), Map.of(), dirA, true, 0);
        if (u.tpl.b <= 0) throw new IllegalStateException("goal-prep: задайте FACES_FEI_BOOTSTRAP");
        int[][] w = FeiBootstrap.weights(u.persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET);
        Map<String, double[]> rows = FrameK2.exportRows(dir);

        // r0′ тем же правилом, что r0 (решение Хозяина 02.10): 95-й процентиль по снимкам анфаса №11–14 FEI
        List<Double> r14 = new ArrayList<>();
        List<Double> r14t = new ArrayList<>();
        for (Map<Integer, String> m : feiKeys.values()) {
            for (int no = 11; no <= 14; no++) {
                String k = m.get(no);
                double[] row = k == null ? null : rows.get(k);
                if (row == null) continue;
                r14.add(Math.abs(FeiDataset.pose(row)));
                r14t.add(poseTilt(row));
            }
        }
        double[] s14 = r14.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double[] s14t = r14t.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double r95 = FeiDataset.percentile(s14, 0.95);
        double r0t = FeiDataset.percentile(s14t, 0.95);

        StringBuilder t = new StringBuilder("Этап «к цели», подготовка – " + base + " (XBaseEval goal-prep " + base + "; Жорик №12 от 04.11)\nКод: коммит " + FarMethods.commit()
                + "; модели – " + r.dir().resolve("vec") + "; кадры «а» – " + dirA + "\nБез обучения. Бутстреп по людям B = " + u.tpl.b + ", seed " + (u.tpl.seed + FeiBootstrap.SEED_OFFSET)
                + "; людей " + u.persons + ". Разделимость – ошибка своих, п. п. (порог по контрольным чужим своей конфигурации); запись: разность [95 %] ±полуширина; "
                + "«*» – интервал без нуля. У pAUC больше – лучше, у прочих мер меньше – лучше.\n");
        Map<String, List<Double>> widthsAll = new LinkedHashMap<>();
        StringBuilder vol = new StringBuilder("\n=== П. 1. Объём контроля и шаг фактического FAR ===\nгруппа | блок | свои: людей, попыток (шаг ошибки своих, п. п.) | контрольные: людей, попыток | шаг FAR, % | факт. FAR при 1 / 2 / 5 / 10 %\n");
        StringBuilder lev = new StringBuilder("\n=== Уровни мер (95 % интервал) ===\nгруппа | строка | " + head() + "\n");
        StringBuilder pow = new StringBuilder("\n=== П. 1. Парные разности известных пар ===\nгруппа | пара | " + head() + "\n");
        StringBuilder o11 = new StringBuilder();
        StringBuilder o10 = new StringBuilder();
        for (String grp : List.of("межс.", "(ii) s1")) {
            if (!u.groups.contains(grp)) continue;
            List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> b.group().equals(grp)).toList();
            boolean hasA = java.nio.file.Files.exists(XBaseFinal.vfile(ra, bs.get(0), XBaseFinal.TT1));
            List<FeiDecompose.Cfg> csF = new ArrayList<>();
            List<FeiDecompose.Cfg> csO = new ArrayList<>();
            List<FeiDecompose.Cfg> csT = new ArrayList<>();
            List<FeiDecompose.Cfg> csA = new ArrayList<>();
            for (XBaseVec.Blk b : bs) {
                csF.add(cfg(b, b.g(), fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
                csO.add(cfg(b, b.g(), fused(r, b, XBaseFinal.TT05, b.g().rOf, b.g().r0)));
                csT.add(cfg(b, b.g(), XBaseVec.normLines(XBaseFinal.models(r, b, XBaseFinal.TT1, false), null, new int[2])));
                if (hasA) csA.add(cfg(b, b.g(), fused(ra, b, XBaseFinal.TT1, b.g().rOf, b.g().r0)));
            }
            XBaseFinalReport.M mF = XBaseFinalReport.measure(csF, u.persons, w);
            XBaseFinalReport.M mO = XBaseFinalReport.measure(csO, u.persons, w);
            XBaseFinalReport.M mT = XBaseFinalReport.measure(csT, u.persons, w);
            XBaseFinalReport.M mA = hasA ? XBaseFinalReport.measure(csA, u.persons, w) : null;
            for (int i = 0; i < bs.size(); i++) {
                FeiDecompose.Cfg c = csF.get(i);
                long pOwn = Arrays.stream(c.ownP).distinct().count();
                long pCtrl = Arrays.stream(c.ctrlP).distinct().count();
                vol.append(String.format(Locale.ROOT, "%s | %s | %d, %d (%.2f) | %d, %d | %.3f | %.2f / %.2f / %.2f / %.2f%n", grp, bs.get(i).name(), pOwn, c.ownR.length, 100.0 / c.ownR.length,
                        pCtrl, c.ctrlR.length, 100.0 / c.ctrlR.length, XBaseOther.errAtFar(List.of(c), FeiBootstrap.unit(u.persons), 0.01)[1],
                        XBaseOther.errAtFar(List.of(c), FeiBootstrap.unit(u.persons), 0.02)[1], XBaseOther.errAtFar(List.of(c), FeiBootstrap.unit(u.persons), 0.05)[1],
                        XBaseOther.errAtFar(List.of(c), FeiBootstrap.unit(u.persons), 0.10)[1]));
            }
            lev.append(levelRow(grp + " | слияние σ₀ = 1", mF)).append(levelRow(grp + " | слияние σ₀ = 0,5", mO)).append(levelRow(grp + " | L1-tt-xs-F σ₀ = 1", mT));
            if (hasA) lev.append(levelRow(grp + " | слияние, кадры «а»", mA));
            Map<String, List<Double>> widths = new LinkedHashMap<>();
            pow.append(diffRow(grp + " | σ₀ = 0,5 − σ₀ = 1", mO, mF, widths));
            if (hasA) pow.append(diffRow(grp + " | кадры Б − «а»", mF, mA, widths));
            else pow.append(grp).append(" | кадры Б − «а» | нет моделей на кадрах «а» для этой строки (контроль замораживания: FERET межс., (ii) s3; GT межс., внутри)\n");
            pow.append(diffRow(grp + " | слияние − L1-tt-xs-F", mF, mT, widths));
            pow.append(grp).append(" | средняя полуширина по парам");
            for (String ms : MEASURES) pow.append(String.format(Locale.ROOT, " | ±%.2f", widths.get(ms).stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN)));
            pow.append('\n');
            widthsAll.putAll(widths);

            // П. 2 (О-11): только межс. – те же пробы в файлах 23.10
            if (grp.equals("межс.")) {
                Path old = base.equals("gt") ? out.resolve("xbase_gt_subsp_scores.tsv.gz") : u.su.dir.resolve("reports").resolve("eq").resolve("xbase_feret_xs2_scores.tsv.gz");
                Map<String, String> ob = XBaseSubsp.blocks(old);
                List<FeiDecompose.Cfg> cs23 = new ArrayList<>();
                for (int i = 0; i < bs.size(); i++) {
                    String bn = base.equals("gt") ? grp + ", проекция L1-tt-xs-F, k = 0" : "проход " + (i + 1) + ", конфигурация межс., строка L1-tt-xs-F";
                    String ls = ob.get(bn);
                    if (ls == null) throw new IllegalStateException("goal-prep: нет блока «" + bn + "» в " + old);
                    FeiDecompose.Cfg c = cfg(bs.get(i), bs.get(i).g(), ls);
                    if (!Arrays.equals(c.ownP, csF.get(i).ownP) || !Arrays.equals(c.ctrlP, csF.get(i).ctrlP)) throw new IllegalStateException("goal-prep: пробы 23.10 и итогового прогона разошлись: " + bn);
                    cs23.add(c);
                }
                XBaseFinalReport.M m23 = XBaseFinalReport.measure(cs23, u.persons, w);
                o11.append("\n=== П. 2 (О-11). Итоговая конфигурация − конфигурация 23.10 (L1-tt-xs-F, евклид, 7×7, кадры «а»; ").append(old.getFileName()).append(", те же пробы) ===\nстрока | ").append(head()).append('\n');
                o11.append(levelRow(grp + " конфигурация 23.10", m23));
                o11.append("пара | ").append(head()).append('\n');
                o11.append(diffRow("слияние (Б) − 23.10", mF, m23, null));
                if (hasA) o11.append(diffRow("слияние («а») − 23.10", mA, m23, null));
                o11.append(diffRow("L1-tt-xs-F (Б) − 23.10", mT, m23, null));

                // П. 3 (О-10)
                o10.append(String.format(Locale.ROOT, "%n=== П. 3 (О-10). Выбор X / X+mirror по r и по r′ (с поправкой на наклон) ===%nFEI №11–14, снимков %d: 95-й процентиль |r| %.4f (r0 решения – %.2f), |r′| %.4f = r0′%n", s14.length, r95, bs.get(0).g().r0, r0t));
                o10.append("блок | проб (свои / чужие) | меняется при r′, r0 = r0 | меняется при r′, r0′ | из них свои / чужие (r0′)\n");
                long[] tot = new long[3];
                for (XBaseVec.Blk b : bs) {
                    double r0 = b.g().r0;
                    int nOwn = 0, nCt = 0, ch0 = 0, chT = 0, chOwn = 0, chCt = 0;
                    for (XBaseEval.Probe p : XBaseEval.probes(b.g(), b.pro())) {
                        double[] row = detRow(u, rows, p.key());
                        double a = Math.abs(FeiDataset.pose(row));
                        double tl = poseTilt(row);
                        if (Math.abs(a - b.g().rOf.get(p.key())) > 1e-12) throw new IllegalStateException("goal-prep: r не сошлось с rOf: " + p.key());
                        boolean own = p.role().equals(FeiScores.OWN);
                        if (own) nOwn++; else nCt++;
                        if ((a <= r0) != (tl <= r0)) ch0++;
                        if ((a <= r0) != (tl <= r0t)) {
                            chT++;
                            if (own) chOwn++; else chCt++;
                        }
                    }
                    int n = nOwn + nCt;
                    tot[0] += n;
                    tot[1] += ch0;
                    tot[2] += chT;
                    o10.append(String.format(Locale.ROOT, "%s | %d (%d / %d) | %d = %.1f %% | %d = %.1f %% | %d / %d%n", b.name(), n, nOwn, nCt, ch0, 100.0 * ch0 / n, chT, 100.0 * chT / n, chOwn, chCt));
                }
                o10.append(String.format(Locale.ROOT, "итого | %d | %d = %.1f %% | %d = %.1f %%%n", tot[0], tot[1], 100.0 * tot[1] / tot[0], tot[2], 100.0 * tot[2] / tot[0]));
                // Перевыбор модели по r′: признаки и W не пересчитываются, обе модели уже в векторах; rOf у всех ключей блока – r′
                List<FeiDecompose.Cfg> csT0 = new ArrayList<>();
                List<FeiDecompose.Cfg> csTt = new ArrayList<>();
                for (XBaseVec.Blk b : bs) {
                    Map<String, Double> rr = new HashMap<>();
                    TreeSet<String> keys = new TreeSet<>();
                    for (XBaseVec.Model m : XBaseFinal.models(r, b, XBaseFinal.TT1, false)) for (XBaseVec.Row q : m.rows()) keys.add(q.key());
                    for (String k : keys) rr.put(k, poseTilt(detRow(u, rows, k)));
                    csT0.add(cfg(b, b.g(), fused(r, b, XBaseFinal.TT1, rr, b.g().r0)));
                    csTt.add(cfg(b, b.g(), fused(r, b, XBaseFinal.TT1, rr, r0t)));
                }
                XBaseFinalReport.M m0 = XBaseFinalReport.measure(csT0, u.persons, w);
                XBaseFinalReport.M mt = XBaseFinalReport.measure(csTt, u.persons, w);
                o10.append("\nСлияние при выборе модели по r′ (пересчёт признаков не нужен: обе модели посчитаны; отбор проб по |r| ≤ R прежний; при R′ состав проб изменился бы – для этого нужны кадры и признаки новых проб)\nпара | ").append(head()).append('\n');
                o10.append(diffRow("r′, r0 = " + bs.get(0).g().r0 + " − r", m0, mF, null));
                o10.append(diffRow(String.format(Locale.ROOT, "r′, r0′ = %.4f − r", r0t), mt, mF, null));
            }
        }
        t.append(vol).append(lev).append(pow).append(o11).append(o10);
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(r.dir().resolve("goal_prep_" + base + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }
}
