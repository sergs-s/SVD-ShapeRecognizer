package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.storage.SettingsStore;

/**
 * Разделимость при равном фактическом FAR контроля (PLAN.md, п. 4.2, решения Хозяина 11.10.2026 по итогам LTP, п. А;
 * диагностика, не приёмка) по имеющимся файлам оценок, без обучения: L1, L1-in (xbase_gt_scores / xbase_gt_in_scores) и
 * L1-tt, L1-in-tt, LTP-tt, LTP-in-tt (xbase_gt_ltp_scores / xbase_gt_in_ltp_scores), межсессионно и внутри сессии. Выбор
 * модели по позе — как в XBaseSep (X / X+mirror по |r| ≤ r0). Для каждой строки: точка равных ошибок по контрольным чужим
 * (XBaseSep.eer); порог по контрольным чужим — наибольший θ, при котором фактический FAR контроля по снимкам не выше 1 % и
 * 5 % (FeiBootstrap.threshold по контрольным чужим; свойство разделимости, не калибровки) и ошибка своих при нём (отказ или
 * неверный кандидат); квартили d₁/d₂ своих и контрольных чужих. Бутстреп по людям GT (те же веса, что в XBaseEval; порог —
 * заново в каждой реплике), парные разности. Отчёт xbase_gt_sep_ltp.txt.
 *
 * @author ssv
 */
final class XBaseSepLtp {

    static final double[] FARS = {0.01, 0.05};
    static final String[] FAR_NAMES = {"FAR контроля 1 %", "FAR контроля 5 %"};
    static final List<String> ORDER = List.of("L1", "L1-in", "L1-tt", "L1-in-tt", "LTP-tt", "LTP-in-tt");
    static final String[][] PAIRS = {{"LTP-in-tt", "L1-in"}, {"LTP-in-tt", "L1-in-tt"}, {"LTP-tt", "L1"}, {"LTP-tt", "L1-tt"}};
    static final String[] MODES = {"межсессионно", "внутри сессии"};

    private XBaseSepLtp() {
    }

    /** Порог по контрольным чужим: наибольший θ, при котором взвешенная доля контрольных снимков с r ≤ θ не больше far. */
    static double ctrlThreshold(XBaseSep.Data d, int[] w, double far) {
        Integer[] idx = new Integer[d.ctrlR.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(d.ctrlR[a], d.ctrlR[b]));
        double[] s = new double[idx.length];
        int[] p = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            s[i] = d.ctrlR[idx[i]];
            p[i] = d.ctrlP[idx[i]];
        }
        return FeiBootstrap.threshold(s, p, w, far);
    }

    /** Итоги строки в режиме: [k] ошибка своих при FAR контроля FARS[k], EER; точки и реплики. */
    static final class Res {
        double[] th = new double[FARS.length];
        double[] far = new double[FARS.length];
        double[] err = new double[FARS.length];
        double[][] errRep;
        double[] eer;
        double[] eerRep;
        double[][] q;
    }

    static Res evaluate(XBaseSep.Data d, int[] unit, int[][] w) {
        Res r = new Res();
        int b = w == null ? 0 : w.length;
        r.errRep = new double[FARS.length][b];
        for (int k = 0; k < FARS.length; k++) {
            r.th[k] = ctrlThreshold(d, unit, FARS[k]);
            r.far[k] = XBaseSep.ctrlFar(d, unit, r.th[k]);
            r.err[k] = XBaseSep.ownErr(d, unit, r.th[k]);
            for (int q = 0; q < b; q++) r.errRep[k][q] = XBaseSep.ownErr(d, w[q], ctrlThreshold(d, w[q], FARS[k]));
        }
        r.eer = XBaseSep.eer(d, unit);
        r.eerRep = new double[b];
        for (int q = 0; q < b; q++) r.eerRep[q] = XBaseSep.eer(d, w[q])[3];
        r.q = new double[][] {XBaseSep.quartiles(d.ownR), XBaseSep.quartiles(d.ctrlR)};
        return r;
    }

    static String cell(double point, double[] reps) {
        return reps.length == 0 ? String.format(Locale.ROOT, "%.1f %%", point) : XBaseSep.ci(point, reps);
    }

    static String diff(double point, double[] a, double[] z) {
        return a.length == 0 ? String.format(Locale.ROOT, "%+.1f", point) : FeiBootstrap.diff(point, FeiNorm.diff(a, z), 1);
    }

    static void run(Path dir, SettingsStore settings, Path out) throws IOException {
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval gt-sep-ltp: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        int b = settings.loadFacesFeiBootstrap();
        long seed = settings.loadFacesSeed() + FeiBootstrap.SEED_OFFSET;
        Map<String, Double> pose = XBaseEval.poses(dir.resolve("detections.tsv"));
        Map<String, Double> rOf = new HashMap<>();
        pose.forEach((k, v) -> {
            if (!k.startsWith("gt/")) return;
            String[] c = k.split("/");
            rOf.put(c[1] + "/" + Integer.parseInt(c[2].replaceAll("\\D", "")), Math.abs(v));
        });
        // [режим] файлы: none (L1, L1-in), tt (L1-tt, L1-in-tt, LTP-tt, LTP-in-tt).
        String[][] files = {{"xbase_gt_scores.tsv.gz", "xbase_gt_ltp_scores.tsv.gz"}, {"xbase_gt_in_scores.tsv.gz", "xbase_gt_in_ltp_scores.tsv.gz"}};
        List<List<Map<String, Row>>> loaded = new ArrayList<>();
        TreeSet<String> persons = new TreeSet<>();
        for (String[] pair : files) {
            List<Map<String, Row>> l = new ArrayList<>();
            for (String f : pair) {
                Map<String, Row> rows = XBaseSep.read(out.resolve(f));
                for (Row r : rows.values()) persons.add(r.person());
                l.add(rows);
            }
            loaded.add(l);
        }
        Map<String, Integer> pIdx = new HashMap<>();
        for (String p : persons) pIdx.put(p, pIdx.size());
        int[] unit = FeiBootstrap.unit(persons.size());
        int[][] w = b > 0 ? FeiBootstrap.weights(persons.size(), b, seed) : null;
        // Строка → {файл (0 — none, 1 — tt), метод}.
        Map<String, Object[]> src = new LinkedHashMap<>();
        src.put("L1", new Object[] {0, FeiMethods.LBP_MLDA});
        src.put("L1-in", new Object[] {0, FeiMethods.LBP_MLDA_IN});
        src.put("L1-tt", new Object[] {1, FeiMethods.LBP_MLDA});
        src.put("L1-in-tt", new Object[] {1, FeiMethods.LBP_MLDA_IN});
        src.put("LTP-tt", new Object[] {1, XBaseEval.V.LTP.id});
        src.put("LTP-in-tt", new Object[] {1, XBaseEval.V.LTPIN.id});
        List<Map<String, Res>> res = new ArrayList<>();
        for (int md = 0; md < 2; md++) {
            Map<String, Res> m = new LinkedHashMap<>();
            for (String l : ORDER) {
                Object[] s = src.get(l);
                m.put(l, evaluate(XBaseSep.data(loaded.get(md).get((int) s[0]), (String) s[1], true, rOf, r0, pIdx), unit, w));
            }
            res.add(m);
        }

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2 (решения 11.10.2026 по LTP, п. А): разделимость при равном фактическом FAR контроля — GT, межсессионно и\n"
                + "внутри сессии (XBaseEval gt-sep-ltp, по файлам оценок, без обучения; диагностика, не приёмка)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (detections.tsv — позы)\n");
        for (String[] pair : files) {
            for (String f : pair) t.append("Файл оценок: ").append(out.resolve(f)).append(" (").append(FeiThreshold.sha(out.resolve(f))).append(")\n");
        }
        t.append("Строки: L1, L1-in — none (xbase_gt_scores / xbase_gt_in_scores); L1-tt, L1-in-tt, LTP-tt, LTP-in-tt — tt (xbase_gt_ltp_scores /\n"
                + "xbase_gt_in_ltp_scores; t = 0,1). Выбор модели по позе пробы: X / X+mirror по |r| ≤ r0 = ").append(r0)
                .append("; оценка d₁/d₂ — FeiScores.ratio.\n");
        t.append("Порог по контрольным чужим — наибольший θ, при котором фактический FAR контроля по снимкам не выше 1 % / 5 % (Нейман – Пирсон\n"
                + "по контрольным чужим, FeiBootstrap.threshold); ошибка своих — отказ или неверный кандидат при этом θ. Точка равных ошибок —\n"
                + "по контрольным чужим (XBaseSep.eer: первый θ с наименьшей |ошибка своих − FAR контроля|; EER — их среднее).\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d (те же веса, что в XBaseEval); порог — "
                    + "заново в каждой реплике;\n[2,5; 97,5] — процентили; «*» — интервал разности не содержит 0.%n", b, seed, persons.size()));
        }
        t.append("Распределения — квартили 25 / 50 / 75 % без весов.\n");

        t.append("\n=== Ошибка своих при пороге по контрольным чужим (θ; фактический FAR контроля по снимкам, точка) ===\n");
        t.append("режим | строка");
        for (String n : FAR_NAMES) t.append(" | ").append(n).append(": ошибка своих | θ | FAR контроля");
        t.append('\n');
        for (int md = 0; md < 2; md++) {
            for (String l : ORDER) {
                Res r = res.get(md).get(l);
                List<String> row = new ArrayList<>(List.of(MODES[md], l));
                for (int k = 0; k < FARS.length; k++) {
                    row.add(cell(r.err[k], r.errRep[k]));
                    row.add(String.format(Locale.ROOT, "%.6f", r.th[k]));
                    row.add(String.format(Locale.ROOT, "%.2f %%", r.far[k]));
                }
                t.append(String.join(" | ", row)).append('\n');
            }
        }
        t.append("\n=== Точка равных ошибок (контрольные чужие) ===\nрежим | строка | θ (d₁/d₂) | ошибка своих | FAR контроля | EER\n");
        for (int md = 0; md < 2; md++) {
            for (String l : ORDER) {
                Res r = res.get(md).get(l);
                t.append(String.format(Locale.ROOT, "%s | %s | %.6f | %.1f %% | %.2f %% | %s%n", MODES[md], l, r.eer[0], r.eer[1], r.eer[2], cell(r.eer[3], r.eerRep)));
            }
        }
        t.append("\n=== Распределения d₁/d₂ (квартили 25 / 50 / 75 %) ===\nрежим | строка | группа | 25 % | медиана | 75 %\n");
        String[] groups = {"свои", "контрольные чужие"};
        for (int md = 0; md < 2; md++) {
            for (String l : ORDER) {
                Res r = res.get(md).get(l);
                for (int g = 0; g < 2; g++) {
                    t.append(String.format(Locale.ROOT, "%s | %s | %s | %.4f | %.4f | %.4f%n", MODES[md], l, groups[g], r.q[g][0], r.q[g][1], r.q[g][2]));
                }
            }
        }
        t.append("\n=== Парные разности, п. п. (те же реплики) ===\nрежим | пара | ошибка своих при FAR контроля 1 % | при FAR контроля 5 % | EER\n");
        for (int md = 0; md < 2; md++) {
            for (String[] p : PAIRS) {
                Res a = res.get(md).get(p[0]);
                Res z = res.get(md).get(p[1]);
                List<String> row = new ArrayList<>(List.of(MODES[md], p[0] + " − " + p[1]));
                for (int k = 0; k < FARS.length; k++) row.add(diff(a.err[k] - z.err[k], a.errRep[k], z.errRep[k]));
                row.add(diff(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                t.append(String.join(" | ", row)).append('\n');
            }
        }
        FarMethods.write(out.resolve("xbase_gt_sep_ltp.txt"), t);
        System.out.print(t);
    }
}
