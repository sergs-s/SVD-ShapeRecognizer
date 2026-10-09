package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.apache.commons.math3.distribution.BetaDistribution;

/**
 * Отчёт «Нормировка освещения перед LBP» (PLAN.md, п. 4.2; решение Хозяина 09.10.2026): для нормировок none / tt / clahe,
 * вариантов L1 и L1-in, способов порога П2 и П3, FAR 1 % и 5 % — ошибка своих с интервалом и разложением (i)–(iv), верный
 * кандидат (i + ii); FAR контроля по снимкам (точка, [2,5; 97,5], ↑95), верхняя 95 % граница Клоппера – Пирсона по
 * n_eff = n/deff (как в П1), FAR по людям, допуск; парные разности с none (те же веса бутстрепа) и выполнение условий
 * приёмки. Итоги вариантов — FeiThreshold.Res (точка, реплики, конфигурации).
 *
 * @author ssv
 */
final class LbpNormReport {

    /** Способы порога в отчёте: П2, П3 (индексы FeiThreshold.METHODS). */
    static final int[] MS = {2, 3};
    static final String[] FAR_NAME = {"FAR 1 %", "FAR 5 %"};

    /** Какие условия приёмки проверяются: GT межсессионно — 1 и 3; GT внутри сессии и FEI — 2 и 3. */
    enum Kind { CROSS, WITHIN, FEI }

    private LbpNormReport() {
    }

    /**
     * FAR контроля по людям в точке: {принятых, всего}; единица — человек в конфигурации (принят, если принят хотя бы один
     * его снимок при пороге этой конфигурации).
     */
    static int[] farPeople(FeiThreshold.Res r, int m, int k) {
        int x = 0;
        int n = 0;
        for (int ci = 0; ci < r.cfgs.size(); ci++) {
            FeiDecompose.Cfg c = r.cfgs.get(ci);
            double th = r.theta[m][k][ci];
            Map<Integer, Boolean> acc = new TreeMap<>();
            for (int i = 0; i < c.ctrlR.length; i++) acc.merge(c.ctrlP[i], c.ctrlR[i] <= th, Boolean::logicalOr);
            n += acc.size();
            for (boolean a : acc.values()) if (a) x++;
        }
        return new int[] {x, n};
    }

    /**
     * FAR контроля по снимкам с учётом людей: {x, n, deff, n_eff, верхняя односторонняя 95 % граница}. Снимки сгруппированы
     * по людям по всем конфигурациям; deff — FeiThreshold.deff (веса 1); граница — Клоппер – Пирсон по x_eff = p̂·n_eff,
     * n_eff (при x = 0 — 1 − 0,05^(1/n_eff)).
     */
    static double[] farEff(FeiThreshold.Res r, int m, int k) {
        Map<Integer, double[]> per = new HashMap<>();
        for (int ci = 0; ci < r.cfgs.size(); ci++) {
            FeiDecompose.Cfg c = r.cfgs.get(ci);
            double th = r.theta[m][k][ci];
            for (int i = 0; i < c.ctrlR.length; i++) {
                double[] a = per.computeIfAbsent(c.ctrlP[i], q -> new double[2]);
                a[1]++;
                if (c.ctrlR[i] <= th) a[0]++;
            }
        }
        double x = 0;
        double n = 0;
        double sxx = 0;
        double sxn = 0;
        double snn = 0;
        for (double[] a : per.values()) {
            x += a[0];
            n += a[1];
            sxx += a[0] * a[0];
            sxn += a[0] * a[1];
            snn += a[1] * a[1];
        }
        if (x != r.far[m][k][0] || n != r.far[m][k][1]) throw new IllegalStateException("FAR контроля по людям ≠ по снимкам: " + r.label);
        double d = FeiThreshold.deff(x, n, per.size(), sxx, sxn, snn);
        double ne = n / d;
        double up = x == 0 ? 1 - Math.pow(1 - FeiThreshold.CONFIDENCE, 1 / ne)
                : Double.isInfinite(d) ? 1 : new BetaDistribution(x / n * ne + 1, ne - x / n * ne).inverseCumulativeProbability(FeiThreshold.CONFIDENCE);
        return new double[] {x, n, d, ne, up};
    }

    /** Доля верного кандидата (i) + (ii), %: точка и реплики (null — без бутстрепа). */
    static double correct(FeiThreshold.Res r, int m, int k) {
        double[] c = r.counts[m][k][0];
        return 100.0 * (c[FeiDecompose.G_OK] + c[FeiDecompose.G_REJ]) / Arrays.stream(c).sum();
    }

    static double[] correctRep(FeiThreshold.Res r, int m, int k) {
        if (r.shareRep == null) return null;
        double[] a = r.shareRep[m][k][0][FeiDecompose.G_OK];
        double[] b = r.shareRep[m][k][0][FeiDecompose.G_REJ];
        double[] s = new double[a.length];
        for (int q = 0; q < s.length; q++) s[q] = a[q] + b[q];
        return s;
    }

    /**
     * Таблицы и условия приёмки.
     *
     * @param res  нормировка → итоги вариантов (в одном порядке у всех нормировок; первая — none, опорная)
     * @param b    реплик бутстрепа (0 — без интервалов и без проверки значимости)
     * @param kind какие условия проверяются
     */
    static void tables(StringBuilder t, Map<IlluminationNorm, List<FeiThreshold.Res>> res, int b, Kind kind) {
        List<IlluminationNorm> norms = new ArrayList<>(res.keySet());
        if (norms.get(0) != IlluminationNorm.NONE) throw new IllegalArgumentException("Первая нормировка отчёта — не none");

        t.append("\n=== Ошибка своих, разложение, верный кандидат ===\n");
        t.append("нормировка | вариант | способ | порог | ошибка своих всего | анфас | поворот | (i) | (ii) | (iii) | (iv) | верный кандидат (i)+(ii)\n");
        for (IlluminationNorm n : norms) {
            for (FeiThreshold.Res r : res.get(n)) {
                for (int m : MS) {
                    for (int k = 0; k < FeiThreshold.KS; k++) {
                        List<String> c = new ArrayList<>(List.of(n.id, r.label, FeiThreshold.METHODS[m], FAR_NAME[k]));
                        for (int sp = 0; sp < FeiThreshold.SP; sp++) c.add(FeiThreshold.errCell(r, m, k, sp, b));
                        double[] cnt = r.counts[m][k][0];
                        double all = Arrays.stream(cnt).sum();
                        for (int g = FeiDecompose.G_OK; g <= FeiDecompose.G_WRONG_ACC; g++) c.add(FeiNorm.cell(cnt[g], all, b > 0 ? r.shareRep[m][k][0][g] : null));
                        c.add(FeiNorm.cell(cnt[FeiDecompose.G_OK] + cnt[FeiDecompose.G_REJ], all, correctRep(r, m, k)));
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }

        t.append("\n=== FAR контроля ===\n");
        t.append("по снимкам: x/n = % [2,5; 97,5] ↑95 (бутстреп); КП_eff — верхняя 95 % граница Клоппера – Пирсона по n_eff = n/deff (как П1);\n"
                + "по людям: принят хотя бы один снимок человека (единица — человек в конфигурации), КП — верхняя 95 % граница; допуск — точка по\n"
                + "снимкам не выше 1,1 % при FAR 1 % и 5,5 % при FAR 5 %.\n");
        t.append("нормировка | вариант | способ | порог | по снимкам | КП_eff (deff; n_eff) | по людям (КП) | допуск\n");
        for (IlluminationNorm n : norms) {
            for (FeiThreshold.Res r : res.get(n)) {
                for (int m : MS) {
                    for (int k = 0; k < FeiThreshold.KS; k++) {
                        List<String> c = new ArrayList<>(List.of(n.id, r.label, FeiThreshold.METHODS[m], FAR_NAME[k]));
                        c.add(farSnap(r, m, k, b));
                        double[] e = farEff(r, m, k);
                        c.add(String.format(Locale.ROOT, "↑%.2f %% (%.2f; %.1f)", 100 * e[4], e[2], e[3]));
                        int[] p = farPeople(r, m, k);
                        c.add(String.format(Locale.ROOT, "%d/%d = %.1f %% (↑%.1f %%)", p[0], p[1], 100.0 * p[0] / p[1],
                                100 * FaceEvaluation.binomialUpperBound(p[0], p[1], FeiThreshold.CONFIDENCE)));
                        c.add(admissible(r, m, k) ? "да" : "нет");
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }

        if (b > 0) {
            t.append("\n=== Парные разности с none, п. п. (те же реплики; «*» — интервал [2,5; 97,5] не содержит 0) ===\n");
            t.append("пара | вариант | способ | ошибка FAR 1 % всего | анфас | поворот | ошибка FAR 5 % всего | верный кандидат | FAR контроля 1 % | "
                    + "FAR контроля 5 %\n");
            for (IlluminationNorm n : norms.subList(1, norms.size())) {
                for (int v = 0; v < res.get(n).size(); v++) {
                    FeiThreshold.Res a = res.get(n).get(v);
                    FeiThreshold.Res z = res.get(IlluminationNorm.NONE).get(v);
                    for (int m : MS) {
                        List<String> c = new ArrayList<>(List.of(n.id + " − none", a.label, FeiThreshold.METHODS[m]));
                        for (int sp = 0; sp < FeiThreshold.SP; sp++) c.add(errDiff(a, z, m, 0, sp));
                        c.add(errDiff(a, z, m, 1, 0));
                        c.add(FeiBootstrap.diff(correct(a, m, 0) - correct(z, m, 0), FeiNorm.diff(correctRep(a, m, 0), correctRep(z, m, 0)), 1));
                        for (int k = 0; k < FeiThreshold.KS; k++) {
                            c.add(FeiBootstrap.diff(FeiThreshold.farPct(a, m, k) - FeiThreshold.farPct(z, m, k), FeiNorm.diff(a.farRep[m][k], z.farRep[m][k]), 2));
                        }
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }
        conditions(t, res, norms, b, kind);
    }

    static String farSnap(FeiThreshold.Res r, int m, int k, int b) {
        int x = (int) r.far[m][k][0];
        int n = (int) r.far[m][k][1];
        String s = String.format(Locale.ROOT, "%d/%d = %.2f %%", x, n, FeiThreshold.farPct(r, m, k));
        if (b == 0) return s;
        double[] ci = FeiNorm.ci(r.farRep[m][k]);
        return s + String.format(Locale.ROOT, " [%.2f; %.2f] ↑%.2f %%", ci[0], ci[1], FeiBootstrap.quantile(FeiBootstrap.sorted(r.farRep[m][k]), 0.95));
    }

    static boolean admissible(FeiThreshold.Res r, int m, int k) {
        return FeiThreshold.withinTolerance(r.far[m][k][0], r.far[m][k][1], FeiScores.FARS[k]);
    }

    static String errDiff(FeiThreshold.Res a, FeiThreshold.Res z, int m, int k, int sp) {
        return FeiBootstrap.diff(FeiThreshold.err(a, m, k, sp) - FeiThreshold.err(z, m, k, sp), FeiNorm.diff(a.errRep[m][k][sp], z.errRep[m][k][sp]), 1);
    }

    /** Интервал [2,5; 97,5] разности ошибки своих (нормировка − none). */
    static double[] errCi(FeiThreshold.Res a, FeiThreshold.Res z, int m, int k) {
        return FeiNorm.ci(FeiNorm.diff(a.errRep[m][k][0], z.errRep[m][k][0]));
    }

    /**
     * Условия приёмки (утверждены Хозяином 09.10.2026), по каждой нормировке ≠ none отдельно; выбор нормировки — за
     * Хозяином. 1) GT межсессионно: ошибка своих L1 при FAR 1 % значимо ниже, чем при none, и при П2, и при П3 (верхний
     * край интервала разности < 0). 2) FEI и GT внутри сессии: не хуже в пределах интервалов — разность с none не значимо
     * больше нуля (нижний край интервала ≤ 0) у L1 и L1-in, П2 и П3, FAR 1 % и 5 %. 3) FAR контроля в допуске (точка) у
     * L1 и L1-in при П2 и П3, FAR 1 % и 5 % — перечисляются строки вне допуска.
     */
    static void conditions(StringBuilder t, Map<IlluminationNorm, List<FeiThreshold.Res>> res, List<IlluminationNorm> norms, int b, Kind kind) {
        t.append("\n=== Условия приёмки (по каждой нормировке отдельно; выбор — за Хозяином) ===\n");
        if (b == 0) {
            t.append("Без бутстрепа (faces.fei.bootstrap = 0) значимость не проверяется — условия 1 и 2 не оцениваются.\n");
        }
        Map<IlluminationNorm, List<String>> out = new LinkedHashMap<>();
        for (IlluminationNorm n : norms) {
            List<String> lines = new ArrayList<>();
            List<FeiThreshold.Res> rs = res.get(n);
            List<FeiThreshold.Res> zs = res.get(IlluminationNorm.NONE);
            if (n != IlluminationNorm.NONE && b > 0) {
                if (kind == Kind.CROSS) {
                    boolean ok = true;
                    StringBuilder s = new StringBuilder();
                    for (int m : MS) {
                        double[] ci = errCi(rs.get(0), zs.get(0), m, 0);
                        boolean lower = ci[1] < 0;
                        ok &= lower;
                        s.append(s.length() > 0 ? "; " : "").append(String.format(Locale.ROOT, "%s %+.1f [%+.1f; %+.1f] — %s", FeiThreshold.METHODS[m],
                                FeiThreshold.err(rs.get(0), m, 0, 0) - FeiThreshold.err(zs.get(0), m, 0, 0), ci[0], ci[1], lower ? "значимо ниже" : "не значимо ниже"));
                    }
                    lines.add("условие 1 (ошибка своих " + rs.get(0).label + " при FAR 1 % значимо ниже, чем при none, П2 и П3): " + (ok ? "ВЫПОЛНЕНО" : "НЕ выполнено")
                            + " (" + s + ")");
                } else {
                    List<String> worse = new ArrayList<>();
                    for (int v = 0; v < rs.size(); v++) {
                        for (int m : MS) {
                            for (int k = 0; k < FeiThreshold.KS; k++) {
                                double[] ci = errCi(rs.get(v), zs.get(v), m, k);
                                if (ci[0] > 0) {
                                    worse.add(String.format(Locale.ROOT, "%s %s %s: %+.1f [%+.1f; %+.1f]", rs.get(v).label, FeiThreshold.METHODS[m], FAR_NAME[k],
                                            FeiThreshold.err(rs.get(v), m, k, 0) - FeiThreshold.err(zs.get(v), m, k, 0), ci[0], ci[1]));
                                }
                            }
                        }
                    }
                    lines.add("условие 2 (" + (kind == Kind.FEI ? "FEI" : "GT внутри сессии") + ": не хуже none в пределах интервалов — разность ошибки своих "
                            + "не значимо больше 0; L1 и L1-in, П2 и П3, FAR 1 и 5 %): " + (worse.isEmpty() ? "ВЫПОЛНЕНО" : "НЕ выполнено — значимо хуже: "
                            + String.join("; ", worse)));
                }
            }
            List<String> outside = new ArrayList<>();
            for (FeiThreshold.Res r : rs) {
                for (int m : MS) {
                    for (int k = 0; k < FeiThreshold.KS; k++) {
                        if (!admissible(r, m, k)) outside.add(String.format(Locale.ROOT, "%s %s %s: %.2f %%", r.label, FeiThreshold.METHODS[m], FAR_NAME[k],
                                FeiThreshold.farPct(r, m, k)));
                    }
                }
            }
            lines.add("условие 3 (FAR контроля в допуске 1,1 / 5,5 %, точка; L1 и L1-in, П2 и П3): " + (outside.isEmpty() ? "ВЫПОЛНЕНО"
                    : "вне допуска: " + String.join("; ", outside)) + (n == IlluminationNorm.NONE ? " (none — опорная)" : ""));
            out.put(n, lines);
        }
        out.forEach((n, lines) -> {
            t.append(n.id).append(":\n");
            for (String l : lines) t.append("  ").append(l).append('\n');
        });
    }
}
