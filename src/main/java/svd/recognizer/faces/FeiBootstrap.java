package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeiScores.Transform;

/**
 * Этап 5б: бутстреп по людям для разбора файла оценок faces-fei (PLAN.md, п. 3а). В каждой реплике люди пересэмплируются
 * целиком, со всеми своими снимками, отдельно в каждом разбиении s и в каждой роли (свои 100, пороговые 40, контроль 40):
 * человек получает вес — сколько раз он вытянут (выборка с возвращением). Веса общие для всех строк отчёта (парные
 * разности) и для всех 8 конфигураций разбиения. В каждой реплике и конфигурации порог заново — Нейман – Пирсон по
 * снимкам пороговых чужих с весами (наибольший θ, при котором взвешенное число принятых снимков не больше ⌊FAR·W⌋, W —
 * взвешенное число снимков); при весах 1 — ровно порог FeiScores. Интервал — процентили 2,5 и 97,5 %, для FAR контроля —
 * верхняя 95 % граница (процентиль 95 %), рядом граница Клоппера – Пирсона по снимкам (оптимистична).
 *
 * Включается ключом faces.fei.bootstrap (число реплик B, FACES_FEI_BOOTSTRAP); seed — faces.seed + 7000. Отчёт —
 * fei_bootstrap.txt; fei_scores.txt не меняется.
 *
 * @author ssv
 */
final class FeiBootstrap {

    static final String FILE = "fei_bootstrap.txt";
    static final long SEED_OFFSET = 7000;
    /** Роли (индексы весов). */
    static final String[] ROLES = {FeiScores.OWN, FeiScores.THR, FeiScores.CTRL};
    static final int OWN = 0;
    static final int THR = 1;
    static final int CTRL = 2;

    /** Метрики реплики. */
    static final int ERR1 = 0;
    static final int ERR5 = 1;
    static final int ERR5_FRONTAL = 2;
    static final int ERR5_TURN = 3;
    static final int FAR1 = 4;
    static final int FAR5 = 5;
    static final int FARP1 = 6;
    static final int METRICS = 7;

    /** Пары по суффиксу: «X + суффикс − X» для всех X, у которых в файле есть обе строки. */
    static final List<String> PAIR_SUFFIXES = List.of("+mirror");
    /** Прочие пары (A − B). */
    static final List<String[]> EXTRA_PAIRS = List.<String[]>of(new String[] {"2c-mlda-ratio", "2a95-fisher-ratio"});

    /** Попытки одной строки отчёта в одной конфигурации; люди — индексы в своей роли и разбиении. */
    static final class Cfg {
        final int s;
        final boolean frontal;
        int[] ownP;
        double[] ownS;
        boolean[] ownOk;
        /** Пороговые и контроль: оценки по возрастанию, человек каждого снимка. */
        int[] thrP;
        double[] thrS;
        int[] ctrlP;
        double[] ctrlS;

        Cfg(int s, boolean frontal) {
            this.s = s;
            this.frontal = frontal;
        }
    }

    final Map<String, List<Cfg>> rows = new LinkedHashMap<>();
    /** [s][роль]: человек → индекс по первому появлению (после finish — по возрастанию id). */
    private final List<List<Map<String, Integer>>> persons = new ArrayList<>();
    private boolean finished;

    /** Попытки одной конфигурации строки id (все строки группы — один метод, s, j). */
    void add(String id, Transform t, List<Row> group) {
        if (finished) throw new IllegalStateException("Бутстреп уже завершён");
        Row first = group.get(0);
        Cfg c = new Cfg(first.s(), FeiProtocol.frontal(first.j()));
        List<double[]> own = new ArrayList<>();
        List<double[]> thr = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        for (Row r : group) {
            double sc = t.apply(r.best(), r.second());
            if (Double.isNaN(sc)) throw new IllegalArgumentException("NaN в оценке: " + id + " " + r);
            switch (r.role()) {
                case FeiScores.OWN -> {
                    double s = Double.isInfinite(r.trueScore()) ? Double.POSITIVE_INFINITY : sc;
                    own.add(new double[] {person(c.s, OWN, r.person()), s, r.bestId().equals(r.trueId()) ? 1 : 0});
                }
                case FeiScores.THR -> thr.add(new double[] {person(c.s, THR, r.person()), sc});
                case FeiScores.CTRL -> ctrl.add(new double[] {person(c.s, CTRL, r.person()), sc});
                default -> throw new IllegalArgumentException("Роль: " + r.role());
            }
        }
        c.ownP = new int[own.size()];
        c.ownS = new double[own.size()];
        c.ownOk = new boolean[own.size()];
        for (int i = 0; i < own.size(); i++) {
            c.ownP[i] = (int) own.get(i)[0];
            c.ownS[i] = own.get(i)[1];
            c.ownOk[i] = own.get(i)[2] == 1;
        }
        thr.sort((a, b) -> Double.compare(a[1], b[1]));
        ctrl.sort((a, b) -> Double.compare(a[1], b[1]));
        c.thrP = thr.stream().mapToInt(a -> (int) a[0]).toArray();
        c.thrS = thr.stream().mapToDouble(a -> a[1]).toArray();
        c.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
        c.ctrlS = ctrl.stream().mapToDouble(a -> a[1]).toArray();
        rows.computeIfAbsent(id, k -> new ArrayList<>()).add(c);
    }

    private int person(int s, int role, String id) {
        while (persons.size() <= s) {
            List<Map<String, Integer>> l = new ArrayList<>();
            for (int k = 0; k < ROLES.length; k++) l.add(new HashMap<>());
            persons.add(l);
        }
        Map<String, Integer> m = persons.get(s).get(role);
        return m.computeIfAbsent(id, k -> m.size());
    }

    /** Число людей [s][роль]. */
    int[][] sizes() {
        int[][] n = new int[persons.size()][ROLES.length];
        for (int s = 0; s < n.length; s++) for (int k = 0; k < ROLES.length; k++) n[s][k] = persons.get(s).get(k).size();
        return n;
    }

    /** Индексы людей — по возрастанию id (выборка не зависит от порядка строк в файле). */
    void finish() {
        if (finished) return;
        finished = true;
        List<List<int[]>> remap = new ArrayList<>();
        for (List<Map<String, Integer>> bySplit : persons) {
            List<int[]> r = new ArrayList<>();
            for (Map<String, Integer> m : bySplit) {
                String[] ids = m.keySet().toArray(String[]::new);
                Arrays.sort(ids);
                int[] map = new int[ids.length];
                for (int i = 0; i < ids.length; i++) map[m.get(ids[i])] = i;
                r.add(map);
            }
            remap.add(r);
        }
        for (List<Cfg> cs : rows.values()) {
            for (Cfg c : cs) {
                List<int[]> r = remap.get(c.s);
                remapAll(c.ownP, r.get(OWN));
                remapAll(c.thrP, r.get(THR));
                remapAll(c.ctrlP, r.get(CTRL));
            }
        }
    }

    private static void remapAll(int[] p, int[] map) {
        for (int i = 0; i < p.length; i++) p[i] = map[p[i]];
    }

    /** Веса реплик [b][s][роль][человек]; реплики по порядку, разбиения по порядку, роли own, thr, ctrl. */
    static int[][][][] weights(int[][] sizes, int b, long seed) {
        Random rnd = new Random(seed);
        int[][][][] w = new int[b][sizes.length][ROLES.length][];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < sizes.length; s++) {
                for (int k = 0; k < ROLES.length; k++) {
                    int n = sizes[s][k];
                    int[] v = new int[n];
                    for (int d = 0; d < n; d++) v[rnd.nextInt(n)]++;
                    w[i][s][k] = v;
                }
            }
        }
        return w;
    }

    /** Веса 1 (исходная выборка). */
    static int[][][] unit(int[][] sizes) {
        int[][][] w = new int[sizes.length][ROLES.length][];
        for (int s = 0; s < sizes.length; s++) {
            for (int k = 0; k < ROLES.length; k++) {
                w[s][k] = new int[sizes[s][k]];
                Arrays.fill(w[s][k], 1);
            }
        }
        return w;
    }

    /**
     * Порог Неймана – Пирсона по снимкам с весами людей: наибольший θ, при котором взвешенное число снимков с оценкой ≤ θ
     * не больше ⌊far·W⌋. При весах 1 совпадает с FaceEvaluation.neymanPearsonThreshold.
     *
     * @param sorted оценки по возрастанию
     * @param person человек каждого снимка
     */
    static double threshold(double[] sorted, int[] person, int[] w, double far) {
        long total = 0;
        for (int p : person) total += w[p];
        long allowed = (long) Math.floor(far * total + 1e-9);
        long cum = 0;
        for (int i = 0; i < sorted.length; i++) {
            cum += w[person[i]];
            if (cum > allowed) return Math.nextDown(sorted[i]);
        }
        return Double.POSITIVE_INFINITY;
    }

    /** Метрики строки при весах w[s][роль][человек], проценты (100,0·x/n — как округляет FeiScores). */
    static double[] metrics(List<Cfg> cs, int[][][] w) {
        double[] err = new double[2];
        double[] err5By = new double[2];
        double[] ownN = new double[2];
        double[] farX = new double[2];
        double farN = 0;
        double farPX = 0;
        double farPN = 0;
        for (Cfg c : cs) {
            int[] wo = w[c.s][OWN];
            int[] wt = w[c.s][THR];
            int[] wc = w[c.s][CTRL];
            double t1 = threshold(c.thrS, c.thrP, wt, FeiScores.FARS[0]);
            double t5 = threshold(c.thrS, c.thrP, wt, FeiScores.FARS[1]);
            int fr = c.frontal ? 1 : 0;
            for (int i = 0; i < c.ownP.length; i++) {
                int q = wo[c.ownP[i]];
                if (q == 0) continue;
                ownN[fr] += q;
                if (c.ownS[i] > t1 || !c.ownOk[i]) err[0] += q;
                if (c.ownS[i] > t5 || !c.ownOk[i]) {
                    err[1] += q;
                    err5By[fr] += q;
                }
            }
            boolean[] accP = new boolean[wc.length];
            for (int i = 0; i < c.ctrlS.length; i++) {
                int q = wc[c.ctrlP[i]];
                farN += q;
                if (c.ctrlS[i] <= t1) {
                    farX[0] += q;
                    accP[c.ctrlP[i]] = true;
                }
                if (c.ctrlS[i] <= t5) farX[1] += q;
            }
            for (int p = 0; p < wc.length; p++) {
                farPN += wc[p];
                if (accP[p]) farPX += wc[p];
            }
        }
        double[] m = new double[METRICS];
        double n = ownN[0] + ownN[1];
        m[ERR1] = 100.0 * err[0] / n;
        m[ERR5] = 100.0 * err[1] / n;
        m[ERR5_FRONTAL] = 100.0 * err5By[1] / ownN[1];
        m[ERR5_TURN] = 100.0 * err5By[0] / ownN[0];
        m[FAR1] = 100.0 * farX[0] / farN;
        m[FAR5] = 100.0 * farX[1] / farN;
        m[FARP1] = 100.0 * farPX / farPN;
        return m;
    }

    /** Без весов: принятые снимки контроля и все снимки, сумма по конфигурациям (для Клоппера – Пирсона). */
    static int[] ctrlCounts(List<Cfg> cs, int[][][] unit, int k) {
        int x = 0;
        int n = 0;
        for (Cfg c : cs) {
            double t = threshold(c.thrS, c.thrP, unit[c.s][THR], FeiScores.FARS[k]);
            for (double v : c.ctrlS) {
                n++;
                if (v <= t) x++;
            }
        }
        return new int[] {x, n};
    }

    /** Результат: точечные метрики и реплики [метрика][b] по строкам. */
    static final class Result {
        final int b;
        final long seed;
        final Map<String, double[]> point = new LinkedHashMap<>();
        final Map<String, double[][]> reps = new LinkedHashMap<>();
        final Map<String, int[][]> cp = new LinkedHashMap<>();

        Result(int b, long seed) {
            this.b = b;
            this.seed = seed;
        }
    }

    Result run(int b, long seed) {
        finish();
        int[][] sizes = sizes();
        int[][][][] w = weights(sizes, b, seed);
        int[][][] u = unit(sizes);
        Result res = new Result(b, seed);
        List<String> ids = new ArrayList<>(rows.keySet());
        double[][][] all = new double[ids.size()][][];
        java.util.stream.IntStream.range(0, ids.size()).parallel().forEach(r -> {
            List<Cfg> cs = rows.get(ids.get(r));
            double[][] v = new double[METRICS][b];
            for (int i = 0; i < b; i++) {
                double[] m = metrics(cs, w[i]);
                for (int k = 0; k < METRICS; k++) v[k][i] = m[k];
            }
            all[r] = v;
        });
        for (int r = 0; r < ids.size(); r++) {
            List<Cfg> cs = rows.get(ids.get(r));
            res.point.put(ids.get(r), metrics(cs, u));
            res.reps.put(ids.get(r), all[r]);
            res.cp.put(ids.get(r), new int[][] {ctrlCounts(cs, u, 0), ctrlCounts(cs, u, 1)});
        }
        return res;
    }

    /** Процентиль p ∈ (0, 1) отсортированной выборки: элемент с номером ⌈p·B⌉ (с 1). */
    static double quantile(double[] sorted, double p) {
        int i = (int) Math.ceil(p * sorted.length - 1e-9) - 1;
        return sorted[Math.max(0, Math.min(sorted.length - 1, i))];
    }

    static double[] sorted(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s;
    }

    /** Пары (A, B) для разностей A − B: по суффиксам, затем прочие; только если обе строки есть. */
    static List<String[]> pairs(Iterable<String> ids, Map<String, ?> present) {
        List<String[]> p = new ArrayList<>();
        for (String suffix : PAIR_SUFFIXES) {
            for (String id : ids) {
                if (id.endsWith(suffix)) {
                    String base = id.substring(0, id.length() - suffix.length());
                    if (present.containsKey(base)) p.add(new String[] {id, base});
                }
            }
        }
        for (String[] e : EXTRA_PAIRS) if (present.containsKey(e[0]) && present.containsKey(e[1])) p.add(e);
        return p;
    }

    // ---------------------------------------------------------------- отчёт

    static StringBuilder report(Result r, List<String> header, String source) {
        StringBuilder t = new StringBuilder();
        t.append("Этап 5б: бутстреп по людям, разбор файла оценок faces-fei (FeiBootstrap, faces-fei-scores с faces.fei.bootstrap)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл: ").append(source).append('\n');
        t.append("Шапка файла:\n");
        for (String h : header) t.append("  ").append(h).append('\n');
        t.append(String.format(Locale.ROOT, "Реплик B = %d, seed = %d (faces.seed + %d).%n", r.b, r.seed, SEED_OFFSET));
        t.append("Реплика: в каждом разбиении s люди каждой роли (свои 100, пороговые 40, контроль 40) вытягиваются с возвращением\n"
                + "целиком, со всеми снимками; вес человека — число вытягиваний; веса общие для всех строк (разности парные) и для 8\n"
                + "конфигураций разбиения. Порог в каждой реплике и конфигурации — Нейман – Пирсон по снимкам пороговых с весами\n"
                + "(взвешенное число принятых снимков ≤ ⌊FAR·W⌋), FAR 1 % и 5 %. Метрики — по сумме 40 конфигураций с весами.\n"
                + "Точка — исходная выборка (веса 1; совпадает с fei_scores.txt); [2,5; 97,5] — процентили реплик; ↑95 — процентиль\n"
                + "95 % (односторонняя верхняя граница FAR контроля); КП — граница Клоппера – Пирсона по снимкам (сумма 40\n"
                + "конфигураций; оптимистична: снимки одного человека коррелируют). FAR контроля по людям (чужой принят, если принят\n"
                + "хотя бы один его снимок, сумма по конфигурациям) при пороге по снимкам 1 % — справочно.\n");

        t.append("\n=== Ошибка своих и FAR контроля с интервалами ===\n");
        t.append("метод | ошибка своих FAR 1 % [2,5; 97,5] | FAR 5 % [2,5; 97,5] | анфас FAR 5 % [..] | поворот FAR 5 % [..] | "
                + "FAR контроля снимки 1 %: точка [2,5; 97,5] ↑95; КП | 5 %: точка [..] ↑95; КП | FAR контроля люди 1 % [..]\n");
        for (String id : r.point.keySet()) {
            double[] p = r.point.get(id);
            double[][] v = r.reps.get(id);
            int[][] cp = r.cp.get(id);
            t.append(String.join(" | ", id,
                    ci(p[ERR1], v[ERR1], 1), ci(p[ERR5], v[ERR5], 1), ci(p[ERR5_FRONTAL], v[ERR5_FRONTAL], 1), ci(p[ERR5_TURN], v[ERR5_TURN], 1),
                    far(p[FAR1], v[FAR1], cp[0]), far(p[FAR5], v[FAR5], cp[1]), ci(p[FARP1], v[FARP1], 1))).append('\n');
        }

        t.append("\n=== Парные разности A − B, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
        t.append("A − B | ошибка своих FAR 1 % [2,5; 97,5] | FAR 5 % | анфас FAR 5 % | поворот FAR 5 % | FAR контроля снимки 1 % | "
                + "5 %\n");
        int[] ks = {ERR1, ERR5, ERR5_FRONTAL, ERR5_TURN, FAR1, FAR5};
        int[] dec = {1, 1, 1, 1, 2, 2};
        for (String[] pr : pairs(r.point.keySet(), r.point)) {
            double[] pa = r.point.get(pr[0]);
            double[] pb = r.point.get(pr[1]);
            double[][] va = r.reps.get(pr[0]);
            double[][] vb = r.reps.get(pr[1]);
            List<String> cells = new ArrayList<>();
            cells.add(pr[0] + " − " + pr[1]);
            for (int i = 0; i < ks.length; i++) {
                int k = ks[i];
                double[] d = new double[r.b];
                for (int q = 0; q < r.b; q++) d[q] = va[k][q] - vb[k][q];
                cells.add(diff(pa[k] - pb[k], d, dec[i]));
            }
            t.append(String.join(" | ", cells)).append('\n');
        }
        return t;
    }

    /** «p % [lo; hi]». */
    static String ci(double point, double[] reps, int dec) {
        double[] s = sorted(reps);
        String f = "%." + dec + "f";
        return String.format(Locale.ROOT, f + " %% [" + f + "; " + f + "]", point, quantile(s, 0.025), quantile(s, 0.975));
    }

    /** FAR контроля: «p % [lo; hi] ↑u %; КП x/n ↑c %». */
    static String far(double point, double[] reps, int[] xn) {
        double[] s = sorted(reps);
        return String.format(Locale.ROOT, "%.2f %% [%.2f; %.2f] ↑%.2f %%; КП %d/%d ↑%.2f %%", point, quantile(s, 0.025),
                quantile(s, 0.975), quantile(s, 0.95), xn[0], xn[1], 100 * FaceEvaluation.binomialUpperBound(xn[0], xn[1], 0.95));
    }

    /** Разность, п. п.: «+d [lo; hi]», «*» — интервал не содержит 0. */
    static String diff(double point, double[] reps, int dec) {
        double[] s = sorted(reps);
        double lo = quantile(s, 0.025);
        double hi = quantile(s, 0.975);
        String f = "%+." + dec + "f";
        return String.format(Locale.ROOT, f + " [" + f + "; " + f + "]%s", point, lo, hi, lo > 0 || hi < 0 ? " *" : "");
    }
}
