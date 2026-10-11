package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.storage.SettingsStore;

/**
 * LTP – устойчивость LBP к шуму (PLAN.md, п. 4.2; решения Хозяина 11.10.2026): цепочка статьи Tan, Triggs (IEEE TIP, 2010) —
 * Tan – Triggs с параметрами статьи (IlluminationNorm.TAN_TRIGGS, края свёртки — REFLECT_101 OpenCV, как прежде) и LTP с
 * порогом t = faces.ltp.t (0,1 — середина диапазона Table I статьи, не перебирается); остальное как у L1: корень гистограмм
 * (LtpFeature: u2 (8, 2), 7×7, на сумму 1 по половинам, [верх | низ]), MLDA, когорта FEI 20, d₁/d₂, зеркала (способ а) и
 * выбор X / X+mirror по |r| пробы, П2 / П3 по «чужим» галереи. Строки: L1, L1-in (none, опорные, как в xbase_gt.txt),
 * L1-tt, L1-in-tt (как в xbase_gt_norm.txt), LTP-tt, справочно LTP-in-tt; межсессионно и внутри сессии (деление gt-in), одни
 * веса бутстрепа. До распознавания — проверка шкалы t (не подбор): по кадрам tt межсессионной галереи GT доля |сосед −
 * центр| &lt; t; вне [5; 95] % — остановка с ошибкой. Отчёт xbase_gt_ltp.txt, оценки строк tt — xbase_gt_ltp_scores.tsv.gz и
 * xbase_gt_in_ltp_scores.tsv.gz (5-lbp-mlda* = L1-tt, 5-ltp-mlda* = LTP-tt).
 *
 * @author ssv
 */
final class XBaseLtp {

    /** Допустимая доля |сосед − центр| &lt; t, %: вне [SHARE_MIN; SHARE_MAX] — остановка до распознавания. */
    static final double SHARE_MIN = 5;
    static final double SHARE_MAX = 95;
    static final Set<XBaseEval.V> TT_VARIANTS = EnumSet.of(XBaseEval.V.L1, XBaseEval.V.L1IN, XBaseEval.V.LTP, XBaseEval.V.LTPIN);
    static final List<String> ORDER = List.of("L1", "L1-in", "L1-tt", "L1-in-tt", "LTP-tt", "LTP-in-tt");

    private XBaseLtp() {
    }

    /** Порог t из faces.ltp.t / FACES_LTP_T: обязателен, конечное число &gt; 0. */
    static double threshold(String raw) {
        if (raw == null) throw new IllegalStateException("XBaseEval gt-ltp: задайте faces.ltp.t / FACES_LTP_T (решение Хозяина 11.10.2026 — 0.1)");
        double t;
        try {
            t = Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalStateException("faces.ltp.t: не число «" + raw + "»");
        }
        if (!(t > 0) || Double.isInfinite(t)) throw new IllegalStateException("faces.ltp.t: нужно конечное число > 0, задано «" + raw + "»");
        return t;
    }

    /** Проверка шкалы: {доля |разность| &lt; t, %; 25 %; медиана; 75 %; число разностей} по отсортированным |разностям|. */
    static double[] scale(double[] sortedAbs, double t) {
        int below = 0;
        while (below < sortedAbs.length && sortedAbs[below] < t) below++;
        return new double[] {100.0 * below / sortedAbs.length, FeiBootstrap.quantile(sortedAbs, 0.25), FeiBootstrap.quantile(sortedAbs, 0.5),
                FeiBootstrap.quantile(sortedAbs, 0.75), sortedAbs.length};
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        double t = threshold(settings.loadFacesLtpT());
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);

        // Кадры: GT после отбора и когорта FEI (как в prepare).
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));

        // Проверка шкалы t до распознавания: кадры tt межсессионной галереи GT.
        List<String> galKeys = g.galleryKeys.values().stream().flatMap(List::stream).sorted().toList();
        List<double[]> parts = new ArrayList<>();
        int total = 0;
        for (String k : galKeys) {
            Mat f = fr.get(k);
            double[] d = LtpFeature.absDiffs(IlluminationNorm.TAN_TRIGGS.vector(f, g.params), f.cols(), f.rows());
            parts.add(d);
            total += d.length;
        }
        double[] all = new double[total];
        int pos = 0;
        for (double[] d : parts) {
            System.arraycopy(d, 0, all, pos, d.length);
            pos += d.length;
        }
        Arrays.sort(all);
        double[] sc = scale(all, t);
        String scaleText = String.format(Locale.ROOT, "Проверка шкалы t (до распознавания, не подбор): кадры tt межсессионной галереи GT — %d снимков, %d разностей "
                + "«сосед − центр»\n(билинейные соседи (8, 2), пиксели с полной окрестностью): доля |разность| < t = %s — %.2f %%; |разность|: 25 %% — %.4f, "
                + "медиана — %.4f,\n75 %% — %.4f (допустимо %.0f–%.0f %%: %s).%n", galKeys.size(), (long) sc[4], t, sc[0], sc[1], sc[2], sc[3], SHARE_MIN,
                SHARE_MAX, sc[0] >= SHARE_MIN && sc[0] <= SHARE_MAX ? "в пределах, продолжаем" : "ВНЕ ПРЕДЕЛОВ — остановка");
        System.out.print(scaleText);
        if (sc[0] < SHARE_MIN || sc[0] > SHARE_MAX) {
            fr.values().forEach(Mat::release);
            FarMethods.write(out.resolve("xbase_gt_ltp.txt"), new StringBuilder("PLAN.md п. 4.2: LTP на GT (XBaseEval gt-ltp) — остановлено до распознавания\n")
                    .append("Код: коммит ").append(FarMethods.commit()).append('\n').append(scaleText));
            throw new IllegalStateException("gt-ltp: доля |разность| < t вне " + SHARE_MIN + "–" + SHARE_MAX + " % — доложить до прогона");
        }

        // Векторы tt: L1-tt, L1-in-tt (как в xbase_gt_norm.txt) и LTP-tt, LTP-in-tt.
        XBaseEval xt = new XBaseEval();
        xt.vectors(fr, XBaseEval.NORM_VARIANTS, g.params, IlluminationNorm.TAN_TRIGGS);
        xt.ltpVectors(fr, g.params, t);
        fr.values().forEach(Mat::release);

        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryKeys, g.seed);
        XBaseEval.Eval cross0 = XBaseEval.evaluate(g, g.x, XBaseEval.NORM_VARIANTS, g.galleryKeys, g.probeKeys);
        XBaseEval.Eval in0 = XBaseEval.evaluate(g, g.x, XBaseEval.NORM_VARIANTS, split.get(0), split.get(1));
        XBaseEval.Eval crossT = XBaseEval.evaluate(g, xt, TT_VARIANTS, g.galleryKeys, g.probeKeys);
        XBaseEval.Eval inT = XBaseEval.evaluate(g, xt, TT_VARIANTS, split.get(0), split.get(1));
        String what = String.format(Locale.ROOT, " — LTP, нормировка перед LBP tt, t = %s (5-lbp-mlda* — L1-tt, 5-ltp-mlda* — LTP-tt)", t);
        XBaseEval.writeScores(out.resolve("xbase_gt_ltp_scores.tsv.gz"), what + ", межсессионно", g, crossT);
        XBaseEval.writeScores(out.resolve("xbase_gt_in_ltp_scores.tsv.gz"), what + ", внутри сессии", g, inT);

        Map<String, FeiThreshold.Res> cm = results(cross0, crossT);
        Map<String, FeiThreshold.Res> wm = results(in0, inT);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 4.2: LTP – устойчивость LBP к шуму — Georgia Tech, межсессионно и внутри сессии (XBaseEval gt-ltp)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt — кадр а 92×112; когорта — a/fei)\n");
        r.append("Цепочка (решения Хозяина 11.10.2026; выбрана до прогона, по результатам не подбирается; одна для всех кадров: галерея, пробы,\n"
                + "«чужие» П2/П3, контроль, когорта FEI, отражённые кадры — после отражения и повторного выравнивания):\n");
        r.append("  ").append(FeiMethods.lbpNormText(IlluminationNorm.TAN_TRIGGS, g.params)).append(";\n");
        r.append("  параметры Tan – Triggs — как в статье (Tan, Triggs, IEEE TIP, 2010, Table I); различие: края свёртки DoG — REFLECT_101\n"
                + "  (умолчание OpenCV GaussianBlur), статья для вырезанного лица советует продолжение края постоянным значением (разд. II.A);\n");
        r.append(String.format(Locale.ROOT, "  LTP: t = %s (faces.ltp.t; середина диапазона 0–0,2 Table I статьи, шкала — выход Tan – Triggs в (−τ; τ));\n"
                + "  сосед u при центре c: +1 при u ≥ c + t, −1 при u ≤ c − t, иначе 0 (формула (4)); верхний и нижний двоичные коды — отдельные\n"
                + "  каналы (рис. 8): по каждому гистограммы u2 (8, 2) по блокам 7×7 на сумму 1, вектор [верх | низ] = 5782, корень, MLDA.%n", t));
        r.append("Строки: L1 = 5-lbp-mlda-ratio+pose, L1-in = 5-lbp-mlda-in-ratio+pose — none, опорные (как в xbase_gt.txt / xbase_gt_in.txt);\n"
                + "  L1-tt, L1-in-tt — те же на выходе tt (как в xbase_gt_norm.txt / xbase_gt_in_norm.txt); LTP-tt = 5-ltp-mlda-ratio+pose;\n"
                + "  LTP-in-tt = 5-ltp-mlda-in-ratio+pose — внутренние блоки 1–5 в каждой половине (справочно).\n");
        r.append("Протокол — как в xbase_gt.txt (межсессионно) и xbase_gt_in.txt (внутри сессии): r0 = ").append(g.r0)
                .append(", d₁/d₂ — FeiScores.ratio, порог П2 / П3 по «чужим» галереи, 5 частей.\n");
        r.append(scaleText);
        r.append(String.format(Locale.ROOT, "Межсессионно: пробы своих %d, «чужие» П2/П3 %d снимков; внутри сессии: пробы своих %d, «чужие» П2/П3 %d; "
                + "контроль — %d снимков %d людей; свои — %d человек.%n", crossT.probes.stream().filter(p -> p.role().equals(XBaseEval.OWN)).count(),
                crossT.cvImages, inT.probes.stream().filter(p -> p.role().equals(XBaseEval.OWN)).count(), inT.cvImages, crossT.ctrlImages, g.ctrl.size(),
                g.own.size()));
        if (g.b > 0) {
            r.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d; одни веса для всех строк и обоих "
                    + "режимов.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        r.append(String.format(Locale.ROOT, "Обучение и оценки: none — межсессионно %.0f с, внутри %.0f с; tt — межсессионно %.0f с, внутри %.0f с; "
                + "потоков %d.%n", cross0.fitS, in0.fitS, crossT.fitS, inT.fitS, g.threads));
        r.append("Правило d₁/d₂ у «чужих» П2 (L1-tt; случаи 1–6), межсессионно: ");
        for (int k = 0; k < FeiScores.CASES; k++) r.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(crossT.cases[k]);
        r.append('\n');
        r.append("Оговорка: 30 своих — различимы только крупные эффекты; внутри сессии снимки почти дубли (оценка «внутри» оптимистична).\n");
        LbpNormReport.diag(r, cm, wm, ORDER, "L1", List.of("LTP-tt", "L1-tt"), g.b);
        if (g.b > 0) {
            LbpNormReport.dropLevel(r, cm, wm, "L1-tt", List.of("LTP-tt"));
            LbpNormReport.dropLevel(r, cm, wm, "L1-in", List.of("LTP-in-tt", "L1-in-tt"));
            LbpNormReport.dropLevel(r, cm, wm, "L1-in-tt", List.of("LTP-in-tt"));
            pairs(r, cm, wm);
        }
        conditions(r, cm, wm, g.b);
        FarMethods.write(out.resolve("xbase_gt_ltp.txt"), r);
        System.out.print(r);
    }

    /** Итоги строк по меткам: none (L1, L1-in), tt (L1-tt, L1-in-tt, LTP-tt, LTP-in-tt). */
    static Map<String, FeiThreshold.Res> results(XBaseEval.Eval none, XBaseEval.Eval tt) {
        Map<String, FeiThreshold.Res> m = new LinkedHashMap<>();
        m.put("L1", none.results.get(0));
        m.put("L1-in", none.results.get(1));
        m.put("L1-tt", tt.results.get(0));
        m.put("L1-in-tt", tt.results.get(1));
        m.put("LTP-tt", tt.results.get(2));
        m.put("LTP-in-tt", tt.results.get(3));
        return m;
    }

    /** Пары парных разностей по уровню: {X, опорная}. */
    static final String[][] PAIRS = {{"LTP-tt", "L1"}, {"LTP-tt", "L1-tt"}, {"LTP-in-tt", "L1-in"}, {"LTP-in-tt", "L1-in-tt"}};

    /** Парные разности по уровню в каждом режиме (ошибка своих, верный кандидат, FAR контроля; те же реплики). */
    static void pairs(StringBuilder t, Map<String, FeiThreshold.Res> cross, Map<String, FeiThreshold.Res> within) {
        t.append("\n=== Парные разности по уровню, п. п. (те же реплики; «*» — интервал [2,5; 97,5] не содержит 0) ===\n");
        t.append("режим | пара | способ | ").append(LbpNormReport.DIFF_HEAD).append('\n');
        String[] modes = {"межсессионно", "внутри сессии"};
        List<Map<String, FeiThreshold.Res>> both = List.of(cross, within);
        for (int i = 0; i < 2; i++) {
            for (String[] p : PAIRS) {
                for (int m : LbpNormReport.MS) {
                    t.append(LbpNormReport.diffRow(List.of(modes[i], p[0] + " − " + p[1], FeiThreshold.METHODS[m]), both.get(i).get(p[0]), both.get(i).get(p[1]), m));
                }
            }
        }
    }

    /**
     * Условия приёмки (порядок — как у нормировки освещения): 1) GT межсессионно — ошибка своих LTP-tt при FAR 1 % значимо
     * ниже, чем у L1 (верхний край интервала разности &lt; 0), и при П2, и при П3; справочно — то же против L1-tt и для -in.
     * 3) FAR контроля в допуске (точка) у LTP-tt и LTP-in-tt, П2 и П3, FAR 1 и 5 %, в обоих режимах. Условие 2 (FEI и GT
     * внутри сессии «не хуже») — только если выполнено условие 1.
     */
    static void conditions(StringBuilder t, Map<String, FeiThreshold.Res> cross, Map<String, FeiThreshold.Res> within, int b) {
        t.append("\n=== Условия приёмки (выбор — за Хозяином) ===\n");
        if (b == 0) {
            t.append("Без бутстрепа (faces.fei.bootstrap = 0) значимость не проверяется — условие 1 не оценивается.\n");
        } else {
            for (String[] p : PAIRS) {
                boolean ok = true;
                StringBuilder s = new StringBuilder();
                for (int m : LbpNormReport.MS) {
                    double[] ci = LbpNormReport.errCi(cross.get(p[0]), cross.get(p[1]), m, 0);
                    boolean lower = ci[1] < 0;
                    ok &= lower;
                    s.append(s.length() > 0 ? "; " : "").append(String.format(Locale.ROOT, "%s %+.1f [%+.1f; %+.1f] — %s", FeiThreshold.METHODS[m],
                            FeiThreshold.err(cross.get(p[0]), m, 0, 0) - FeiThreshold.err(cross.get(p[1]), m, 0, 0), ci[0], ci[1],
                            lower ? "значимо ниже" : "не значимо ниже"));
                }
                boolean main = p[0].equals("LTP-tt") && p[1].equals("L1");
                t.append(main ? "условие 1" : "справочно").append(" (GT межсессионно: ошибка своих ").append(p[0]).append(" при FAR 1 % значимо ниже, чем у ")
                        .append(p[1]).append(", П2 и П3): ").append(ok ? "ВЫПОЛНЕНО" : "НЕ выполнено").append(" (").append(s).append(")\n");
            }
        }
        String[] modes = {"межсессионно", "внутри сессии"};
        List<Map<String, FeiThreshold.Res>> both = List.of(cross, within);
        for (String l : List.of("LTP-tt", "LTP-in-tt", "L1")) {
            List<String> outside = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                FeiThreshold.Res r = both.get(i).get(l);
                for (int m : LbpNormReport.MS) {
                    for (int k = 0; k < FeiThreshold.KS; k++) {
                        if (!LbpNormReport.admissible(r, m, k)) {
                            outside.add(String.format(Locale.ROOT, "%s %s %s: %.2f %%", modes[i], FeiThreshold.METHODS[m], LbpNormReport.FAR_NAME[k],
                                    FeiThreshold.farPct(r, m, k)));
                        }
                    }
                }
            }
            t.append(l.equals("L1") ? "справочно" : "условие 3").append(" (FAR контроля ").append(l).append(" в допуске 1,1 / 5,5 %, точка; П2 и П3, оба режима): ")
                    .append(outside.isEmpty() ? "ВЫПОЛНЕНО" : "вне допуска: " + String.join("; ", outside)).append(l.equals("L1") ? " (L1 — опорная)" : "")
                    .append('\n');
        }
    }
}
