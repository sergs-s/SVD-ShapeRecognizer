package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.storage.SettingsStore;

/**
 * Разделимость на GT (PLAN.md, п. 4.2, «Проверка выравнивания и разделимости на GT», п. 2) по имеющимся файлам оценок
 * XBaseEval, без обучения: L1 (xbase_gt_scores / xbase_gt_in_scores) и SFace справочно (xbase_gt_diag_scores /
 * xbase_gt_in_diag_scores), межсессионно и внутри сессии. Выбор модели по позе — как в XBaseEval (L1: X / X+mirror по
 * |r| ≤ r0; SFace — одна модель). Порог П2 (Нейман – Пирсон по «чужим» галереи) при FAR 1, 5, 10, 20 %: ошибка своих
 * (отказ или неверный кандидат) и FAR контроля по снимкам; точка равных ошибок — по контрольным чужим (порог по d₁/d₂, при
 * котором |ошибка своих − FAR контроля| наименьшая; EER — их среднее). Интервалы — бутстреп по людям GT (те же веса, что в
 * XBaseEval). Распределения (квартили, без весов): свои — d₁ = расстояние до истинного, d₂ = до ближайшего постороннего,
 * r = d₁/d₂ оценки; контрольные чужие — d₁ = лучший, r. Отчёт xbase_gt_sep.txt.
 *
 * @author ssv
 */
final class XBaseSep {

    static final double[] FARS = {0.01, 0.05, 0.10, 0.20};
    static final String[] FAR_NAMES = {"FAR 1 %", "FAR 5 %", "FAR 10 %", "FAR 20 %"};

    private XBaseSep() {
    }

    /** Попытки одного метода и режима: свои, контрольные чужие, «чужие» галереи (П2). */
    static final class Data {
        int[] ownP;
        double[] ownR;
        boolean[] ownOk;
        double[] ownD1;
        double[] ownD2;
        int[] ctrlP;
        double[] ctrlR;
        double[] ctrlD1;
        double[] cvR;
        int[] cvP;
    }

    /** Строки файла оценок: «метод\tроль\tчеловек\tснимок» → строка. */
    static Map<String, Row> read(Path file) throws IOException {
        Map<String, Row> rows = new HashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") || l.isEmpty() || l.equals(FeiScores.COLUMNS)) continue;
                Row r = FeiScores.parse(l);
                rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
            }
        }
        return rows;
    }

    /**
     * Попытки метода base (mirror — есть ли модель +mirror: выбор по позе) из строк файла; rOf — |r| по «человек/снимок»;
     * pIdx — номер человека GT.
     */
    static Data data(Map<String, Row> rows, String base, boolean mirror, Map<String, Double> rOf, double r0, Map<String, Integer> pIdx) {
        List<double[]> own = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        List<double[]> cv = new ArrayList<>();
        for (Row r : rows.values()) {
            if (!r.method().equals(base)) continue;
            boolean front = rOf.get(r.person() + "/" + r.image()) <= r0;
            Row c = mirror && !front ? rows.get(base + FeiMethods.MIRROR + "\t" + r.role() + "\t" + r.person() + "\t" + r.image()) : r;
            if (c == null) throw new IllegalStateException("Нет строки +mirror для " + r);
            double ratio = FeiScores.ratio(c, null);
            int p = pIdx.get(c.person());
            if (c.role().equals(FeiScores.OWN)) {
                boolean ok = c.bestId().equals(c.trueId());
                own.add(new double[] {p, ratio, ok ? 1 : 0, c.trueScore(), ok ? c.second() : c.best()});
            } else if (c.role().equals(FeiScores.CTRL)) {
                ctrl.add(new double[] {p, ratio, c.best()});
            } else if (c.role().startsWith(FeiMethods.CV_ROLE)) {
                cv.add(new double[] {p, ratio});
            } else {
                throw new IllegalStateException("Роль " + c.role());
            }
        }
        if (own.isEmpty() || ctrl.isEmpty() || cv.isEmpty()) throw new IllegalStateException("Нет попыток метода " + base);
        Data d = new Data();
        d.ownP = own.stream().mapToInt(a -> (int) a[0]).toArray();
        d.ownR = own.stream().mapToDouble(a -> a[1]).toArray();
        d.ownOk = new boolean[own.size()];
        for (int i = 0; i < own.size(); i++) d.ownOk[i] = own.get(i)[2] == 1;
        d.ownD1 = own.stream().mapToDouble(a -> a[3]).toArray();
        d.ownD2 = own.stream().mapToDouble(a -> a[4]).toArray();
        d.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
        d.ctrlR = ctrl.stream().mapToDouble(a -> a[1]).toArray();
        d.ctrlD1 = ctrl.stream().mapToDouble(a -> a[2]).toArray();
        cv.sort((a, b) -> Double.compare(a[1], b[1]));
        d.cvR = cv.stream().mapToDouble(a -> a[1]).toArray();
        d.cvP = cv.stream().mapToInt(a -> (int) a[0]).toArray();
        return d;
    }

    /** Ошибка своих при пороге θ (отказ или неверный кандидат), % с весами. */
    static double ownErr(Data d, int[] w, double theta) {
        if (Double.isNaN(theta)) return Double.NaN;
        double e = 0;
        double n = 0;
        for (int i = 0; i < d.ownR.length; i++) {
            int wt = w[d.ownP[i]];
            n += wt;
            if (!(d.ownOk[i] && d.ownR[i] <= theta)) e += wt;
        }
        return 100 * e / n;
    }

    /** FAR контроля по снимкам при θ, % с весами. */
    static double ctrlFar(Data d, int[] w, double theta) {
        if (Double.isNaN(theta)) return Double.NaN;
        double x = 0;
        double n = 0;
        for (int i = 0; i < d.ctrlR.length; i++) {
            int wt = w[d.ctrlP[i]];
            n += wt;
            if (d.ctrlR[i] <= theta) x += wt;
        }
        return 100 * x / n;
    }

    /**
     * Точка равных ошибок по контрольным чужим: перебор порогов θ — все значения r своих и контрольных чужих; выбирается
     * первый θ с наименьшей |ошибка своих − FAR контроля|. Возвращает {θ, ошибка своих %, FAR контроля %, EER = среднее}.
     */
    static double[] eer(Data d, int[] w) {
        int no = d.ownR.length;
        int nc = d.ctrlR.length;
        Integer[] oi = new Integer[no];
        Integer[] ci = new Integer[nc];
        for (int i = 0; i < no; i++) oi[i] = i;
        for (int i = 0; i < nc; i++) ci[i] = i;
        Arrays.sort(oi, (a, b) -> Double.compare(d.ownR[a], d.ownR[b]));
        Arrays.sort(ci, (a, b) -> Double.compare(d.ctrlR[a], d.ctrlR[b]));
        double ownN = 0;
        double okN = 0;
        for (int i = 0; i < no; i++) ownN += w[d.ownP[i]];
        double ctrlN = 0;
        for (int i = 0; i < nc; i++) ctrlN += w[d.ctrlP[i]];
        // Пустая реплика (нет своих или контрольных с весом > 0) – EER не определён (NaN; решение Хозяина 20.10.2026).
        if (ownN == 0 || ctrlN == 0) return new double[] {Double.NaN, Double.NaN, Double.NaN, Double.NaN};
        TreeSet<Double> cand = new TreeSet<>();
        for (double v : d.ownR) cand.add(v);
        for (double v : d.ctrlR) cand.add(v);
        double[] best = {Double.NEGATIVE_INFINITY, 100, 0, 50};
        double bestGap = 100;
        int po = 0;
        int pc = 0;
        double acc = 0;
        for (double th : cand) {
            while (po < no && d.ownR[oi[po]] <= th) {
                if (d.ownOk[oi[po]]) okN += w[d.ownP[oi[po]]];
                po++;
            }
            while (pc < nc && d.ctrlR[ci[pc]] <= th) {
                acc += w[d.ctrlP[ci[pc]]];
                pc++;
            }
            double frr = 100 * (ownN - okN) / ownN;
            double far = 100 * acc / ctrlN;
            double gap = Math.abs(frr - far);
            if (gap < bestGap) {
                bestGap = gap;
                best = new double[] {th, frr, far, (frr + far) / 2};
            }
        }
        return best;
    }

    /** Квартили (25, 50, 75 %) по FeiBootstrap.quantile. */
    static double[] quartiles(double[] v) {
        double[] s = FeiBootstrap.sorted(v);
        return new double[] {FeiBootstrap.quantile(s, 0.25), FeiBootstrap.quantile(s, 0.5), FeiBootstrap.quantile(s, 0.75)};
    }

    static String ci(double point, double[] reps) {
        double[] c = FeiNorm.ci(reps);
        return String.format(Locale.ROOT, "%.1f %% [%.1f; %.1f]", point, c[0], c[1]) + (Double.isNaN(point) ? "" : FeiBootstrap.note(reps));
    }

    static void run(Path dir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval gt-sep: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
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
        String[][] files = {{"xbase_gt_scores.tsv.gz", "xbase_gt_in_scores.tsv.gz"}, {"xbase_gt_diag_scores.tsv.gz", "xbase_gt_in_diag_scores.tsv.gz"}};
        String[] methods = {FeiMethods.LBP_MLDA, "0-sface"};
        String[] labels = {"L1", "SFace"};
        String[] modes = {"межсессионно", "внутри сессии"};
        List<Map<String, Row>> loaded = new ArrayList<>();
        TreeSet<String> persons = new TreeSet<>();
        for (String[] pair : files) {
            for (String f : pair) {
                Map<String, Row> rows = read(out.resolve(f));
                for (Row r : rows.values()) persons.add(r.person());
                loaded.add(rows);
            }
        }
        Map<String, Integer> pIdx = new HashMap<>();
        for (String p : persons) pIdx.put(p, pIdx.size());
        int[] unit = FeiBootstrap.unit(persons.size());
        int[][] w = b > 0 ? FeiBootstrap.weights(persons.size(), b, seed) : null;

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: разделимость на GT — L1 и SFace, межсессионно и внутри сессии (XBaseEval gt-sep, по файлам оценок)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (detections.tsv — позы)\n");
        for (String[] pair : files) {
            for (String f : pair) t.append("Файл оценок: ").append(out.resolve(f)).append(" (").append(FeiThreshold.sha(out.resolve(f))).append(")\n");
        }
        t.append("Выбор модели по позе пробы: L1 — X / X+mirror по |r| ≤ r0 = ").append(r0).append(", SFace — одна модель; оценка d₁/d₂ — FeiScores.ratio.\n");
        t.append("Порог П2 — Нейман – Пирсон по «чужим» галереи (5 частей), при FAR 1, 5, 10, 20 %; ошибка своих — отказ или неверный кандидат;\n"
                + "точка равных ошибок — по контрольным чужим (порог по d₁/d₂ — значения r своих и контрольных; первый с наименьшей\n"
                + "|ошибка своих − FAR контроля|; EER — их среднее).\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d (те же веса, что в XBaseEval); порог — "
                    + "заново в каждой реплике; [2,5; 97,5] — процентили.%n", b, seed, persons.size()));
        }
        t.append("Распределения — квартили 25 / 50 / 75 % без весов; свои: d₁ — до истинного человека, d₂ — до ближайшего постороннего\n"
                + "(второй кандидат, если лучший — истинный, иначе лучший), r — d₁/d₂ оценки (лучший / второй); контрольные чужие: d₁ —\n"
                + "лучший кандидат, r. Шкалы d у L1 и SFace разные: d сравнивается только межсессионно против внутри сессии одного метода.\n");

        StringBuilder err = new StringBuilder("\n=== Ошибка своих при пороге П2 (в скобках — FAR контроля по снимкам, точка) ===\nметод | режим");
        for (String n : FAR_NAMES) err.append(" | ").append(n);
        err.append('\n');
        StringBuilder eq = new StringBuilder("\n=== Точка равных ошибок (контрольные чужие) ===\nметод | режим | θ (d₁/d₂) | ошибка своих | FAR контроля | EER\n");
        StringBuilder dist = new StringBuilder("\n=== Распределения (квартили 25 / 50 / 75 %) ===\nметод | режим | группа | величина | 25 % | медиана | 75 %\n");
        for (int mth = 0; mth < 2; mth++) {
            for (int md = 0; md < 2; md++) {
                Data d = data(loaded.get(mth * 2 + md), methods[mth], mth == 0, rOf, r0, pIdx);
                List<String> row = new ArrayList<>(List.of(labels[mth], modes[md]));
                for (double far : FARS) {
                    double th = FeiBootstrap.threshold(d.cvR, d.cvP, unit, far);
                    String s;
                    if (b > 0) {
                        double[] reps = new double[b];
                        for (int q = 0; q < b; q++) reps[q] = ownErr(d, w[q], FeiBootstrap.threshold(d.cvR, d.cvP, w[q], far));
                        s = ci(ownErr(d, unit, th), reps);
                    } else {
                        s = String.format(Locale.ROOT, "%.1f %%", ownErr(d, unit, th));
                    }
                    row.add(s + String.format(Locale.ROOT, " (%.2f %%)", ctrlFar(d, unit, th)));
                }
                err.append(String.join(" | ", row)).append('\n');
                double[] e = eer(d, unit);
                String eerCell = String.format(Locale.ROOT, "%.1f %%", e[3]);
                if (b > 0) {
                    double[] reps = new double[b];
                    for (int q = 0; q < b; q++) reps[q] = eer(d, w[q])[3];
                    eerCell = ci(e[3], reps);
                }
                eq.append(String.format(Locale.ROOT, "%s | %s | %.6f | %.1f %% | %.2f %% | %s%n", labels[mth], modes[md], e[0], e[1], e[2], eerCell));
                Object[][] groups = {{"свои", "d₁", d.ownD1}, {"свои", "d₂", d.ownD2}, {"свои", "r", d.ownR}, {"контрольные чужие", "d₁", d.ctrlD1},
                    {"контрольные чужие", "r", d.ctrlR}};
                for (Object[] g : groups) {
                    double[] qv = quartiles((double[]) g[2]);
                    dist.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %.4f | %.4f | %.4f%n", labels[mth], modes[md], g[0], g[1], qv[0], qv[1], qv[2]));
                }
            }
        }
        t.append(err).append(eq).append(dist);
        FarMethods.write(out.resolve("xbase_gt_sep.txt"), t);
        System.out.print(t);
    }
}
