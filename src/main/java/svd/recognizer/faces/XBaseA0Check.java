package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseA0.Att;
import svd.recognizer.faces.XBaseA0.Pair;
import svd.recognizer.faces.XBaseA0.Source;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Проверки по итогам просмотра листов (а0) (PLAN.md п. 5.5, «Поставленные проверки»; задание №7 от 13.10.2026), только по
 * готовым файлам оценок №4, без распознавания. Решения — XBaseA0.decide (модель по позе, П2 FAR 1 %).
 * <ul>
 *   <li>П-а.1: распределения d₁ (оценка лучшего кандидата) по группам (i), отказы своим (ii)+(iii), (iv), принятые
 *       контрольные чужие; разбивки по позе (|r| ≤ 0,25 / &gt; 0,25), асимметрии освещения A = |ln(m_L / m_R)| (средние
 *       яркости левой и правой половин кадра «а» во внутренней области: столбцы 10–81, строки 20–100) и шуму (σ Immerkær,
 *       XBaseQuality.frameMeasures) — по медиане всех проб базы; доля отказов своим, где лучший — свой; критерий зазора
 *       (решение Жорика 13.10): «принятые чужие» — контрольные и «чужие» галереи, прошедшие порог по d₁/d₂; зазор есть,
 *       если найдётся θ₂ по d₁, отсекающий ≥ 1/3 принятых чужих при потере ≤ 2 % проб (i); при n &lt; 20 — «мало данных».</li>
 *   <li>П-б.2 (FERET G1): принятые контрольные чужие по дате пробы (940422 и прочие), за 00383 и за прочих людей с
 *       галереей той же даты.</li>
 *   <li>П-в: решения SFace (свой порог П2 FAR 1 %) на отказах своим, (iv) и принятых чужих L1 / L1-tt — пробы листов (а0)
 *       и вся группа.</li>
 * </ul>
 * Режимы XBaseEval gt-a0-check (отчёт reports/faces/xbase/xbase_gt_a0_check.txt) и feret-a0-check (только
 * D:\data\Feret\reports\xbase_feret_a0_check.txt).
 *
 * @author ssv
 */
final class XBaseA0Check {

    static final double POSE = 0.25;
    static final double GAP_LOSS = 0.02;
    static final double GAP_CUT = 1.0 / 3;
    static final int GAP_MIN_N = 20;
    static final int BINS = 20;
    static final String HUB = "00383";
    static final String HUB_DATE = "940422";
    static final String[] GROUPS = {"(i) свои верные", "свои отвергнутые (ii)+(iii)", "(iv) свои под чужим именем", "контрольные чужие приняты"};

    private XBaseA0Check() {
    }

    /** Меры кадра «а» пробы: {A, σ шума}. */
    static double[] measures(Source src, String key, Map<String, double[]> cache) throws IOException {
        double[] m = cache.get(key);
        if (m != null) return m;
        Mat a = src.frameA(key);
        int w = a.cols();
        double l = 0;
        double r = 0;
        int nl = 0;
        int nr = 0;
        byte[] px = new byte[(int) a.total()];
        a.get(0, 0, px);
        for (int y = 20; y <= 100; y++) {
            for (int x = 10; x <= 81; x++) {
                int v = px[y * w + x] & 0xff;
                if (x <= 45) {
                    l += v;
                    nl++;
                } else {
                    r += v;
                    nr++;
                }
            }
        }
        double ml = Math.max(l / nl, 1e-9);
        double mr = Math.max(r / nr, 1e-9);
        double noise = XBaseQuality.frameMeasures(a)[2];
        a.release();
        m = new double[] {Math.abs(Math.log(ml / mr)), noise};
        cache.put(key, m);
        return m;
    }

    /** Номер группы П-а по попытке: 0 (i), 1 отказ своему, 2 (iv), 3 принятый контрольный чужой, −1 — прочее. */
    static int group(Att a) {
        return switch (a.group()) {
            case 0 -> 0;
            case 1, 2 -> 1;
            case 3 -> 2;
            case 5 -> 3;
            default -> -1;
        };
    }

    /** Квантиль (линейная интерполяция, тип 7) по отсортированному массиву. */
    static double q(double[] s, double p) {
        if (s.length == 0) return Double.NaN;
        double h = (s.length - 1) * p;
        int lo = (int) Math.floor(h);
        int hi = Math.min(s.length - 1, lo + 1);
        return s[lo] + (h - lo) * (s[hi] - s[lo]);
    }

    static String quant(List<Double> v) {
        double[] s = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        if (s.length == 0) return "0 | – | – | – | – | –";
        return String.format(Locale.ROOT, "%d | %.4f | %.4f | %.4f | %.4f | %.4f", s.length, q(s, 0.05), q(s, 0.25), q(s, 0.5), q(s, 0.75), q(s, 0.95));
    }

    static double median(List<Double> v) {
        double[] s = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return q(s, 0.5);
    }

    /** П-а.1 для одной пары; возвращает строку вывода о зазоре. */
    static String partA(StringBuilder t, Pair p, Source src, Map<String, double[]> cache) throws IOException {
        String name = p.base + " " + p.label;
        List<Att> atts = new ArrayList<>();
        for (Att a : p.atts) if (group(a) >= 0) atts.add(a);
        List<Double> allA = new ArrayList<>();
        List<Double> allN = new ArrayList<>();
        for (Att a : p.atts) {
            double[] m = measures(src, a.probe().key(), cache);
            allA.add(m[0]);
            allN.add(m[1]);
        }
        double medA = median(allA);
        double medN = median(allN);
        t.append(String.format(Locale.ROOT, "%n=== П-а.1: %s — d₁ (оценка лучшего кандидата), θ П2 FAR 1 %% = %.6f ===%n", name, p.theta));
        t.append(String.format(Locale.ROOT, "Медианы по всем пробам базы (свои и контроль, %d): A = %.4f, σ шума = %.4f%n", p.atts.size(), medA, medN));
        t.append("группа | разбивка | n | 5 % | 25 % | медиана | 75 % | 95 %\n");
        String[] splits = {"все", "|r| ≤ 0,25", "|r| > 0,25", "A ≤ медианы", "A > медианы", "шум ≤ медианы", "шум > медианы"};
        for (int gi = 0; gi < GROUPS.length; gi++) {
            for (int sp = 0; sp < splits.length; sp++) {
                List<Double> v = new ArrayList<>();
                for (Att a : atts) {
                    if (group(a) != gi) continue;
                    double[] m = measures(src, a.probe().key(), cache);
                    double r = src.absR(a.probe().key());
                    boolean in = switch (sp) {
                        case 1 -> r <= POSE;
                        case 2 -> r > POSE;
                        case 3 -> m[0] <= medA;
                        case 4 -> m[0] > medA;
                        case 5 -> m[1] <= medN;
                        case 6 -> m[1] > medN;
                        default -> true;
                    };
                    if (in) v.add(a.row().best());
                }
                t.append(GROUPS[gi]).append(" | ").append(splits[sp]).append(" | ").append(quant(v)).append('\n');
            }
        }
        // Гистограмма d₁ по группам (все пробы), общий диапазон.
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (Att a : atts) {
            lo = Math.min(lo, a.row().best());
            hi = Math.max(hi, a.row().best());
        }
        int[][] h = new int[GROUPS.length][BINS];
        for (Att a : atts) {
            int b = (int) Math.min(BINS - 1, Math.floor((a.row().best() - lo) / (hi - lo) * BINS));
            h[group(a)][b]++;
        }
        t.append("Гистограмма d₁ (корзина: от – до | (i) | отказы | (iv) | чужие приняты)\n");
        for (int b = 0; b < BINS; b++) {
            t.append(String.format(Locale.ROOT, "%.4f – %.4f | %d | %d | %d | %d%n", lo + (hi - lo) * b / BINS, lo + (hi - lo) * (b + 1) / BINS, h[0][b], h[1][b],
                    h[2][b], h[3][b]));
        }
        // Доля отказов своим, где лучший — свой.
        t.append("Отказы своим, лучший кандидат — свой ((ii) от (ii)+(iii)):\n");
        for (int sp = 0; sp < 3; sp++) {
            int ii = 0;
            int all = 0;
            for (Att a : p.atts) {
                if (a.group() != 1 && a.group() != 2) continue;
                double r = src.absR(a.probe().key());
                if (sp == 1 && r > POSE || sp == 2 && r <= POSE) continue;
                all++;
                if (a.group() == 1) ii++;
            }
            t.append(String.format(Locale.ROOT, "  %s: %d/%d = %.1f %%%n", splits[sp], ii, all, all > 0 ? 100.0 * ii / all : Double.NaN));
        }
        // Критерий зазора.
        List<Double> own = new ArrayList<>();
        List<Double> str = new ArrayList<>();
        int strCtrl = 0;
        int strCv = 0;
        for (Att a : p.atts) {
            if (a.group() == 0) own.add(a.row().best());
            if (a.group() == 5) {
                str.add(a.row().best());
                strCtrl++;
            }
        }
        for (int i = 0; i < p.cvR.length; i++) {
            if (p.cvR[i] <= p.theta) {
                str.add(p.cvRows.get(i).best());
                strCv++;
            }
        }
        double[] os = own.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double[] ss = str.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        t.append(String.format(Locale.ROOT, "Зазор: (i) %d проб; принятые чужие %d (контрольные %d, «чужие» галереи П2 %d)%n", os.length, ss.length, strCtrl, strCv));
        t.append("Принятые чужие, d₁: ").append(quant(str)).append("  (n | 5 % | 25 % | медиана | 75 % | 95 %)\n");
        // Перекрытие: доля пар (свой (i), принятый чужой) с d₁ чужого > d₁ своего.
        long gt = 0;
        long eq = 0;
        for (double o : os) {
            for (double s : ss) {
                if (s > o) gt++;
                else if (s == o) eq++;
            }
        }
        double auc = os.length * ss.length > 0 ? (gt + 0.5 * eq) / ((double) os.length * ss.length) : Double.NaN;
        t.append(String.format(Locale.ROOT, "P(d₁ чужого > d₁ своего (i)) = %.3f (0,5 — распределения совпадают, 1 — полностью разделены)%n", auc));
        // Наибольшая доля отсечённых чужих при потере (i) ≤ 2 %: θ₂ — значения d₁ (i) и чужих.
        double bestCut = 0;
        double bestTh = Double.NaN;
        double bestLoss = 0;
        TreeSet<Double> cand = new TreeSet<>();
        for (double o : os) cand.add(o);
        for (double s : ss) cand.add(s);
        for (double th : cand) {
            double loss = (double) (os.length - upper(os, th)) / os.length;
            if (loss > GAP_LOSS) continue;
            double cut = ss.length > 0 ? (double) (ss.length - upper(ss, th)) / ss.length : 0;
            if (cut > bestCut) {
                bestCut = cut;
                bestTh = th;
                bestLoss = loss;
            }
        }
        String verdict;
        if (ss.length < GAP_MIN_N) verdict = "мало данных (принятых чужих " + ss.length + " < " + GAP_MIN_N + ")";
        else verdict = bestCut >= GAP_CUT ? "зазор есть" : "зазора нет";
        t.append(String.format(Locale.ROOT, "Наибольшая доля отсечённых принятых чужих при потере (i) ≤ 2 %%: %.1f %% (θ₂ = %.4f, потеря (i) %.1f %%); "
                + "критерий ≥ 33,3 %% — %s%n", 100 * bestCut, bestTh, 100 * bestLoss, verdict));
        return name + ": " + verdict + String.format(Locale.ROOT, " (отсечено %.1f %% при потере (i) %.1f %%; P = %.3f)", 100 * bestCut, 100 * bestLoss, auc);
    }

    /** Число элементов отсортированного s, не больших x. */
    static int upper(double[] s, double x) {
        int i = Arrays.binarySearch(s, x);
        if (i < 0) return -i - 1;
        while (i + 1 < s.length && s[i + 1] == x) i++;
        return i + 1;
    }

    /** П-в: решения SFace на группах пары p (L1 / L1-tt); sf — пара SFace на тех же пробах. */
    static void partC(StringBuilder t, Pair p, Pair sf) {
        Map<String, Att> byKey = new HashMap<>();
        for (Att a : sf.atts) byKey.put(a.probe().role() + "\t" + a.probe().key(), a);
        String[] names = {"отказы своим (ii)+(iii)", "(iv) свои под чужим именем", "контрольные чужие приняты"};
        int[][] groupsOf = {{1, 2}, {3}, {5}};
        t.append(String.format(Locale.ROOT, "%n=== П-в: SFace на ошибках %s %s (SFace θ П2 FAR 1 %% = %.6f; верно — свой принят как свой, чужой "
                + "отвергнут) ===%n", p.base, p.label, sf.theta));
        t.append("группа | пробы листов (а0): SFace верно / n | вся группа: SFace верно / n | вся группа: SFace принял под чужим именем\n");
        for (int gi = 0; gi < names.length; gi++) {
            Set<String> sheet = new java.util.HashSet<>();
            for (XBaseA0.Case c : p.sel.get(gi)) sheet.add(c.probe().role() + "\t" + c.probe().key());
            int[] ok = new int[2];
            int[] n = new int[2];
            int wrongAcc = 0;
            for (Att a : p.atts) {
                if (Arrays.stream(groupsOf[gi]).noneMatch(x -> x == a.group())) continue;
                String k = a.probe().role() + "\t" + a.probe().key();
                Att s = byKey.get(k);
                if (s == null) throw new IllegalStateException("П-в: нет попытки SFace " + k);
                boolean own = a.probe().role().equals(FeiScores.OWN);
                boolean right = own ? s.group() == 0 : s.group() == 4;
                if (own && s.group() == 3 || !own && s.group() == 5) wrongAcc++;
                for (int w = 0; w < 2; w++) {
                    if (w == 0 && !sheet.contains(k)) continue;
                    n[w]++;
                    if (right) ok[w]++;
                }
            }
            t.append(String.format(Locale.ROOT, "%s | %d/%d | %d/%d | %d%n", names[gi], ok[0], n[0], ok[1], n[1], wrongAcc));
        }
        int[] sfG = new int[6];
        for (Att a : sf.atts) sfG[a.group()]++;
        t.append(String.format(Locale.ROOT, "SFace в целом: свои %d, ошибка своих %d (%.1f %%); контрольные чужие приняты %d/%d%n", sf.ownN,
                sf.ownN - sfG[0], 100.0 * (sf.ownN - sfG[0]) / sf.ownN, sfG[5], sfG[4] + sfG[5]));
    }

    // ---------------------------------------------------------------- GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        XBaseA0.GtSetup su = XBaseA0.gtSetup(export, dir, feiKeys, selLines, settings);
        StringBuilder t = new StringBuilder();
        t.append("Проверки П-а.1 и П-в по итогам (а0), GT межсессионно (XBaseEval gt-a0-check; PLAN.md п. 5.5), по файлам оценок №4\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append('\n');
        Path sfFile = out.resolve("xbase_gt_diag_scores.tsv.gz");
        Pair sf = XBaseA0.decide("GT", "SFace", IlluminationNorm.NONE, V.SFACE, XBaseA0.read(sfFile, "0-sface", null), su.g, su.g.galleryKeys,
                su.probes, su.parts);
        t.append("Файл оценок SFace: ").append(sfFile).append(" (").append(FeiThreshold.sha(sfFile)).append(")\n");
        header(t);
        List<String> verdicts = new ArrayList<>();
        Map<String, double[]> cache = new HashMap<>();
        StringBuilder c = new StringBuilder();
        for (Object[] pp : XBaseA0.GT_PAIRS) {
            Path f = out.resolve((String) pp[0]);
            t.append("Файл оценок ").append(pp[1]).append(": ").append(f).append(" (").append(FeiThreshold.sha(f)).append(")\n");
            Pair p = XBaseA0.decide("GT", (String) pp[1], (IlluminationNorm) pp[2], XBaseA0.read(f, FeiMethods.LBP_MLDA, null), su.g, su.g.galleryKeys,
                    su.probes, su.parts);
            verdicts.add(partA(c, p, su.src, cache));
            partC(c, p, sf);
        }
        t.append(c);
        t.append("\n=== Итог П-а.1 (зазор) ===\n");
        for (String v : verdicts) t.append(v).append('\n');
        FarMethods.write(out.resolve("xbase_gt_a0_check.txt"), t);
        System.out.print(t);
    }

    static void header(StringBuilder t) {
        t.append("Решения: XBaseA0.decide — модель по позе пробы (X / X+mirror по |r| ≤ r0; SFace — одна модель), порог П2 FAR 1 % по «чужим» "
                + "галереи (FeiBootstrap.threshold, веса 1).\n");
        t.append("d₁ — оценка лучшего кандидата (best_score файла оценок; у верного своего — расстояние до своего шаблона).\n");
        t.append("A = |ln(m_L / m_R)|, m_L, m_R — средняя яркость левой (столбцы 10–45) и правой (46–81) половин кадра «а», строки 20–100; "
                + "шум — σ по Immerkær (XBaseQuality.frameMeasures); разбивка по медиане всех проб базы.\n");
        t.append("Зазор (решение Жорика 13.10): принятые чужие — контрольные и «чужие» галереи П2 с d₁/d₂ ≤ θ; зазор есть, если найдётся θ₂ по d₁, "
                + "отсекающий ≥ 1/3 принятых чужих при потере ≤ 2 % проб (i); при n < 20 — «мало данных».\n");
    }

    // ---------------------------------------------------------------- FERET

    static void feret(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        Path f = su.dir.resolve("reports").resolve(XBaseA0.FERET_SCORES);
        StringBuilder t = new StringBuilder();
        t.append("Проверки П-а.1, П-б.2, П-в по итогам (а0), FERET G1 (XBaseEval feret-a0-check; PLAN.md п. 5.5), по файлам оценок №4; только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл оценок: ").append(f).append(" (").append(FeiThreshold.sha(f)).append("), блоки «G1, набор tt» и «G1, набор SFace»\n");
        header(t);
        Pair p = XBaseA0.decide("FERET-G1", "L1-tt", IlluminationNorm.TAN_TRIGGS, XBaseA0.read(f, FeiMethods.LBP_MLDA, "# конфигурация G1, набор tt"), su.g,
                su.ro.g1, su.probes, su.parts);
        Pair sf = XBaseA0.decide("FERET-G1", "SFace", IlluminationNorm.NONE, V.SFACE, XBaseA0.read(f, "0-sface", "# конфигурация G1, набор SFace"), su.g,
                su.ro.g1, su.probes, su.parts);
        Map<String, double[]> cache = new HashMap<>();
        String verdict = partA(t, p, su.src, cache);
        partB(t, p, su);
        partC(t, p, sf);
        t.append("\n=== Итог П-а.1 (зазор) ===\n").append(verdict).append('\n');
        FarMethods.write(su.dir.resolve("reports").resolve("xbase_feret_a0_check.txt"), t);
        System.out.print(t);
    }

    /** П-б.2: принятые контрольные чужие (и (iv)) по дате пробы. */
    static void partB(StringBuilder t, Pair p, XBaseA0.FeretSetup su) {
        Set<String> galDay = new TreeSet<>();
        for (Map.Entry<String, List<String>> e : su.ro.g1.entrySet()) {
            for (String k : e.getValue()) if (su.byKey.get(k).date().equals(HUB_DATE)) galDay.add(e.getKey());
        }
        t.append(String.format(Locale.ROOT, "%n=== П-б.2: хаб %s, дата %s (G1 L1-tt, θ = %.6f) ===%n", HUB, HUB_DATE, p.theta));
        t.append("Люди галереи G1 со снимками даты ").append(HUB_DATE).append(": ").append(galDay.size()).append(' ').append(galDay).append('\n');
        t.append("роль | дата пробы | проб | принято всего | за " + HUB + " | за прочих людей с галереей " + HUB_DATE + " | за остальных\n");
        for (String role : new String[] {FeiScores.CTRL, FeiScores.OWN}) {
            for (boolean day : new boolean[] {true, false}) {
                int n = 0;
                int acc = 0;
                int hub = 0;
                int sameDay = 0;
                for (Att a : p.atts) {
                    if (!a.probe().role().equals(role)) continue;
                    if (su.byKey.get(a.probe().key()).date().equals(HUB_DATE) != day) continue;
                    n++;
                    boolean accepted = role.equals(FeiScores.CTRL) ? a.group() == 5 : a.group() == 3;
                    if (!accepted) continue;
                    acc++;
                    String b = a.row().bestId();
                    if (b.equals(HUB)) hub++;
                    else if (galDay.contains(b)) sameDay++;
                }
                t.append(String.format(Locale.ROOT, "%s | %s | %d | %d (%.1f %%) | %d (%.1f %%) | %d (%.1f %%) | %d%n",
                        role.equals(FeiScores.CTRL) ? "контрольные чужие" : "свои, (iv)", day ? HUB_DATE : "прочие", n, acc, pct(acc, n), hub, pct(hub, n),
                        sameDay, pct(sameDay, n), acc - hub - sameDay));
            }
        }
        // Лучший кандидат у всех контрольных чужих (принятых и нет): как часто 00383 — первый, по дате пробы.
        t.append("Лучший кандидат " + HUB + " у контрольных чужих (принятых и отвергнутых):\n");
        for (boolean day : new boolean[] {true, false}) {
            int n = 0;
            int hub = 0;
            for (Att a : p.atts) {
                if (!a.probe().role().equals(FeiScores.CTRL)) continue;
                if (su.byKey.get(a.probe().key()).date().equals(HUB_DATE) != day) continue;
                n++;
                if (a.row().bestId().equals(HUB)) hub++;
            }
            t.append(String.format(Locale.ROOT, "  пробы %s: %d/%d = %.1f %%%n", day ? HUB_DATE : "прочих дат", hub, n, pct(hub, n)));
        }
    }

    static double pct(int x, int n) {
        return n > 0 ? 100.0 * x / n : Double.NaN;
    }

    // ---------------------------------------------------------------- П-б.1

    /** Строки оценок метода base (и base+mirror) из текста fitLines. */
    static Map<String, FeiScores.Row> rows(String lines, String base) {
        Map<String, FeiScores.Row> out = new HashMap<>();
        for (String l : lines.lines().toList()) {
            FeiScores.Row r = FeiScores.parse(l);
            if (!r.method().equals(base) && !r.method().equals(base + FeiMethods.MIRROR)) continue;
            out.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        return out;
    }

    /** Строки пары при пороге theta: ошибка своих, (iv) всего / за хаб, принятые контрольные чужие всего / за хаб. */
    static String counts(String name, Pair p, double theta) {
        int err = 0;
        int iv = 0;
        int ivHub = 0;
        int acc = 0;
        int accHub = 0;
        int ctrl = 0;
        for (Att a : p.atts) {
            boolean in = a.ratio() <= theta;
            boolean ok = a.row().bestId().equals(a.row().trueId());
            boolean hub = a.row().bestId().equals(HUB);
            if (a.probe().role().equals(FeiScores.OWN)) {
                if (!(ok && in)) err++;
                if (!ok && in) {
                    iv++;
                    if (hub) ivHub++;
                }
            } else {
                ctrl++;
                if (in) {
                    acc++;
                    if (hub) accHub++;
                }
            }
        }
        return String.format(Locale.ROOT, "%s | %.6f | %d/%d = %.1f %% | %d | %d | %d/%d = %.2f %% | %d%n", name, theta, err, p.ownN, pct(err, p.ownN), iv, ivHub,
                acc, ctrl, pct(acc, ctrl), accHub);
    }

    /**
     * П-б.1: G1 (L1-tt), галерея 00383 только из fa и fb (без fa_a, fb_a), прочее — как в XBaseEval feret (роли, части
     * «чужих», когорта FEI 20, векторы L1 на выходе Tan – Triggs, fitLines). Контроль: исходная G1, пересчитанная здесь,
     * побайтно совпадает с блоком «G1, набор tt» файла оценок (строки 5-lbp-mlda и +mirror), иначе остановка.
     */
    static void hub2(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, su.tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new java.util.HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        su.base.cohort = cohort;
        su.g.cohort = cohort;
        // Векторы L1 на выходе tt — как xs[1].vectors в XBaseFeret.run (все отобранные снимки ролей и когорта).
        Set<String> used = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(su.ro.gal1, su.ro.pro1, su.ro.g2, su.ro.pro2, su.ro.ctrlKeys)) m.values().forEach(used::addAll);
        Map<String, Mat> fr = new java.util.LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<GalleryEvaluation.Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval x = new XBaseEval();
        x.vectors(fr, java.util.EnumSet.of(V.L1), su.tpl.params, IlluminationNorm.TAN_TRIGGS);
        fr.values().forEach(Mat::release);

        Path f = su.dir.resolve("reports").resolve(XBaseA0.FERET_SCORES);
        String block = "# конфигурация G1, набор tt";
        Map<String, FeiScores.Row> fileRows = XBaseA0.read(f, FeiMethods.LBP_MLDA, block);
        Set<V> vs = java.util.EnumSet.of(V.L1);
        String base = XBaseEval.fitLines(su.g, x, vs, su.ro.g1, su.probes, su.parts, XBaseA0.G1_S);
        Map<String, FeiScores.Row> baseRows = rows(base, FeiMethods.LBP_MLDA);
        if (!baseRows.equals(fileRows)) throw new IllegalStateException("П-б.1: исходная G1 не совпала с блоком файла оценок (" + baseRows.size() + " / "
                + fileRows.size() + " строк)");
        Map<String, List<String>> g1h = new java.util.LinkedHashMap<>(su.ro.g1);
        List<String> hubKeys = new ArrayList<>();
        for (String k : su.ro.g1.get(HUB)) if (su.byKey.get(k).flag().equals("-")) hubKeys.add(k);
        g1h.put(HUB, hubKeys);
        String mod = XBaseEval.fitLines(su.g, x, vs, g1h, su.probes, su.parts, XBaseA0.G1_S);

        Pair p0 = XBaseA0.decide("FERET-G1", "L1-tt", IlluminationNorm.TAN_TRIGGS, baseRows, su.g, su.ro.g1, su.probes, su.parts);
        Pair p1 = XBaseA0.decide("FERET-G1", "L1-tt", IlluminationNorm.TAN_TRIGGS, rows(mod, FeiMethods.LBP_MLDA), su.g, g1h, su.probes, su.parts);
        StringBuilder t = new StringBuilder();
        t.append("П-б.1: хаб FERET 00383 — G1 L1-tt с галереей 00383 только из fa и fb (XBaseEval feret-hub2; PLAN.md п. 5.5); только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append(" (когорта FEI 20)\n");
        t.append("Файл оценок (контроль исходной G1): ").append(f).append(" (").append(FeiThreshold.sha(f)).append("), блок «G1, набор tt»\n");
        t.append("Контроль: исходная G1, пересчитанная здесь (векторы L1 на выходе tt, fitLines, части seed + 8000 + 100 + 2), — строки 5-lbp-mlda и "
                + "+mirror совпали с блоком файла оценок (").append(baseRows.size()).append(" строк).\n");
        t.append("Галерея 00383: прежняя ").append(su.ro.g1.get(HUB)).append(" → ").append(hubKeys).append("; роли, пробы, части «чужих» П2, когорта — "
                + "прежние (у части с 00383 меньше «чужих» П2: ").append(p0.cvR.length).append(" → ").append(p1.cvR.length).append(").\n");
        t.append("\nвариант | θ | ошибка своих | (iv) | (iv) за 00383 | контрольные чужие приняты | за 00383\n");
        t.append(counts("прежняя галерея (4 снимка 00383), θ П2", p0, p0.theta));
        t.append(counts("галерея 00383 из fa, fb, θ П2 пересчитан", p1, p1.theta));
        t.append(counts("галерея 00383 из fa, fb, прежний θ", p1, p0.theta));
        FarMethods.write(su.dir.resolve("reports").resolve("xbase_feret_hub2.txt"), t);
        System.out.print(t);
    }
}
