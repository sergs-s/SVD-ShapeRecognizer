package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeiScores.Stat;
import svd.recognizer.faces.FeiScores.Transform;
import svd.recognizer.storage.SettingsStore;

/**
 * PLAN.md, п. 4.2, «Нормировка оценки»: по полным векторам расстояний faces-fei (FeiVectors) — варианты оценки для порога
 * у частей A = 2c-mlda-ratio+pose, B = 4-wpca-cos-0-ratio+pose, C = 2a95-fisher-ratio+pose, в протоколе fei_scores.txt
 * (порог Нейман – Пирсон по снимкам 40 пороговых, FAR 1 % и 5 %; бутстреп по людям FEI целиком, те же веса).
 *
 * У каждой пробы вектор модели X или X+mirror по позе (|r| ≤ r0 — X), как в +pose. Кандидат c — argmin сырых d (как в
 * файле оценок) во всех вариантах; меняется только оценка «свой/чужой». Над ℓ_p = log d_p:
 * база — log(d₁/d₂) (FeiScores.ratio); T — (ℓ_c − μ)/σ, μ и σ (n − 1) по 99 людям галереи кроме c; Z — (ℓ_c − μ_c,m)/σ_c,m,
 * статистика модели c в режиме m (X или X+mirror — по позе пробы) по снимкам 40 пороговых чужих конфигурации,
 * попавшим в тот же режим; ZT — сначала Z для всех людей галереи (режим пробы), затем T по z без c. Параметров нет,
 * кроме зафиксированных до прогона: d < 1e-12 (в т. ч. 0) → 1e-12; 1 − cos ≥ −1e-12 и < 0 → 0; σ = 0, d < 0, NaN, +∞ —
 * исключение. В бутстрепе Z-статистики и порог — заново в каждой реплике по взвешенным пороговым.
 *
 * Контроль на входе: лучший кандидат, d₁ и d₂ по вектору каждой попытки — ровно как в файле оценок; база — те же
 * ошибки и FAR, что FeiScores.Stat строки *-ratio+pose (иначе исключение).
 *
 * Запуск: {@code mvn compile exec:java@faces-fei-norm "-Dexec.args=<каталог векторов> <файл оценок>"} при заданных
 * faces.fei.pose.r0 и faces.export.dir (fei_selection.tsv); бутстреп — faces.fei.bootstrap. Отчёт
 * reports/faces/fei/fei_norm.txt.
 *
 * @author ssv
 */
public final class FeiNorm {

    static final String FILE = "fei_norm.txt";
    static final double D_MIN = 1e-12;
    static final int V_BASE = 0;
    static final int V_T = 1;
    static final int V_Z = 2;
    static final int V_ZT = 3;
    static final int NV = 4;
    static final String[] V_NAMES = {"база log(d₁/d₂)", "T", "Z", "ZT"};
    /** Базовые методы частей A, B, C (строки X и X+mirror). */
    static final String[] PART_BASES = {"2c-mlda", "4-wpca-cos-0", "2a95-fisher"};
    /** Случаи log d: d < d_min → d_min; 1 − cos округлено к 0. */
    static final int L_FLOOR = 0;
    static final int L_COS = 1;
    static final String COMMAND = "FACES_FEI_MIRROR=true FACES_FEI_VECTORS=2c-mlda,4-wpca-cos-0,2a95-fisher FACES_EXPORT_DIR=<экспорт> "
            + "mvn compile exec:exec@faces-fei";

    /** Попытка: выбранный по позе режим и вектор (log d), кандидат, база, T, ранг истинного. */
    static final class Att {
        String role;
        String person;
        int image;
        int p;
        /** Индекс истинного в галерее; −1 — чужой. */
        int t = -1;
        int mode;
        boolean poseFrontal;
        double[] l;
        int cand;
        double ratio;
        boolean noSecond;
        double tScore;
        int rank;
    }

    /** Конфигурация части: попытки и суммы для Z-статистик по людям пороговых. */
    static final class Conf {
        int s;
        int j;
        boolean numberFrontal;
        final List<Att> all = new ArrayList<>();
        /** [режим] → люди пороговых (индексы FEI), число их снимков в режиме, суммы (ℓ − shift) и квадратов по людям галереи. */
        int[][] zPersons;
        int[][] zCount;
        double[][][] s1;
        double[][][] s2;
        double[][] shift;
        int[] total = new int[2];
        boolean[] used = new boolean[2];
    }

    /** Итог части: строки вариантов (точка, реплики), ранги, контроль. */
    static final class Part {
        String label;
        /** Базовый метод (строки X и X+mirror). */
        String base;
        String id;
        List<Conf> confs = new ArrayList<>();
        int[] lCases = new int[2];
        int[] rCases = new int[FeiScores.CASES];
        long[] checked = new long[2];
        String[] files = new String[2];
        /** [вариант]: доли групп точки [порог][деление][группа], числа; реплики. */
        double[][][][] pointCounts = new double[NV][][][];
        double[][][][][] repShares = new double[NV][][][][];
        /** [вариант]: FeiBootstrap.metrics точки и реплик [метрика][b]; КП [порог] {x, n}. */
        double[][] pointFar = new double[NV][];
        double[][][] repFar = new double[NV][][];
        int[][][] cp = new int[NV][][];
        /** [вариант][порог][бин]: ранги в группе (iii); [бин] — все неверные argmin. */
        int[][][] rankIii = new int[NV][2][3];
        int[] rankWrong = new int[3];
        int zMin = Integer.MAX_VALUE;
        int[] zMinMode = {Integer.MAX_VALUE, Integer.MAX_VALUE};
        String zMinWhere = "";
    }

    private FeiNorm() {
    }

    // ---------------------------------------------------------------- оценки

    /** ℓ = log d по правилу: NaN, +∞, d < 0 (кроме округления 1 − cos) — исключение; d < d_min → d_min. */
    static double logd(double d, boolean cosine, int[] cases, String where) {
        if (Double.isNaN(d) || d == Double.POSITIVE_INFINITY) throw new IllegalStateException("log d: " + d + "; попытка " + where);
        if (cosine && d < 0 && d >= -FeiScores.COS_EPS) {
            d = 0;
            cases[L_COS]++;
        }
        if (d < 0) throw new IllegalStateException("log d: отрицательное расстояние " + d + "; попытка " + where);
        if (d < D_MIN) {
            cases[L_FLOOR]++;
            d = D_MIN;
        }
        return Math.log(d);
    }

    /** (x_c − μ)/σ, μ и σ (n − 1) по всем x, кроме c; σ = 0 — исключение. */
    static double tNorm(double[] x, int c) {
        double s = 0;
        int n = 0;
        for (int p = 0; p < x.length; p++) {
            if (p == c) continue;
            s += x[p];
            n++;
        }
        double m = s / n;
        double v = 0;
        for (int p = 0; p < x.length; p++) {
            if (p == c) continue;
            double d = x[p] - m;
            v += d * d;
        }
        double sd = Math.sqrt(v / (n - 1));
        if (!(sd > 0)) throw new IllegalStateException("T-норма: σ = 0");
        return (x[c] - m) / sd;
    }

    /** Z-статистики конфигурации {μ[режим][p], σ[режим][p]} по взвешенным пороговым (w null — веса 1). */
    static double[][][] zStats(Conf c, int[] w) {
        int g = c.s1[0].length == 0 ? 0 : c.s1[0][0].length;
        double[][] mu = new double[2][g];
        double[][] sd = new double[2][g];
        for (int m = 0; m < 2; m++) {
            double wt = 0;
            for (int q = 0; q < c.zPersons[m].length; q++) wt += (w == null ? 1 : w[c.zPersons[m][q]]) * c.zCount[m][q];
            for (int p = 0; p < g; p++) {
                double a = 0;
                double b = 0;
                for (int q = 0; q < c.zPersons[m].length; q++) {
                    int k = w == null ? 1 : w[c.zPersons[m][q]];
                    if (k == 0) continue;
                    a += k * c.s1[m][q][p];
                    b += k * c.s2[m][q][p];
                }
                double mean = a / wt;
                double var = (b - wt * mean * mean) / (wt - 1);
                mu[m][p] = c.shift[m][p] + mean;
                sd[m][p] = wt < 2 ? Double.NaN : Math.sqrt(Math.max(var, 0));
            }
        }
        return new double[][][] {mu, sd};
    }

    /** Оценка попытки в варианте v (z — Z-статистики конфигурации). */
    static double score(Att a, int v, double[][][] z, String where) {
        switch (v) {
            case V_BASE:
                return a.ratio;
            case V_T:
                return a.tScore;
            case V_Z: {
                double sd = z[1][a.mode][a.cand];
                if (!(sd > 0)) throw new IllegalStateException("Z-норма: σ = " + sd + " (режим " + a.mode + "); " + where);
                return (a.l[a.cand] - z[0][a.mode][a.cand]) / sd;
            }
            default: {
                double[] zs = new double[a.l.length];
                for (int p = 0; p < zs.length; p++) {
                    double sd = z[1][a.mode][p];
                    if (!(sd > 0)) throw new IllegalStateException("ZT-норма: σ = " + sd + " (режим " + a.mode + "); " + where);
                    zs[p] = (a.l[p] - z[0][a.mode][p]) / sd;
                }
                return tNorm(zs, a.cand);
            }
        }
    }

    /** Конфигурации варианта v при весах w (Z-статистики заново): для FeiDecompose и FeiBootstrap. */
    static Object[] cfgs(List<Conf> confs, int v, int[] w) {
        List<FeiDecompose.Cfg> dc = new ArrayList<>(confs.size());
        List<FeiBootstrap.Cfg> bc = new ArrayList<>(confs.size());
        for (Conf c : confs) {
            double[][][] z = v == V_Z || v == V_ZT ? zStats(c, w) : null;
            List<Att> own = new ArrayList<>();
            List<double[]> thr = new ArrayList<>();
            List<double[]> ctrl = new ArrayList<>();
            List<Double> ownS = new ArrayList<>();
            for (Att a : c.all) {
                double sc = score(a, v, z, a.person + " s=" + c.s + " j=" + c.j);
                switch (a.role) {
                    case FeiScores.OWN -> {
                        own.add(a);
                        ownS.add(sc);
                    }
                    case FeiScores.THR -> thr.add(new double[] {a.p, sc});
                    default -> ctrl.add(new double[] {a.p, sc});
                }
            }
            thr.sort((x, y) -> Double.compare(x[1], y[1]));
            ctrl.sort((x, y) -> Double.compare(x[1], y[1]));
            int n = own.size();
            FeiDecompose.Cfg d = new FeiDecompose.Cfg();
            d.numberFrontal = c.numberFrontal;
            d.ownP = new int[n];
            d.ownR = new double[n];
            d.ownOk = new boolean[n];
            d.noSecond = new boolean[n];
            d.trueInf = new boolean[n];
            d.poseFrontal = new boolean[n];
            FeiBootstrap.Cfg b = new FeiBootstrap.Cfg(c.numberFrontal);
            b.ownP = new int[n];
            b.ownS = new double[n];
            b.ownOk = new boolean[n];
            for (int q = 0; q < n; q++) {
                Att a = own.get(q);
                d.ownP[q] = a.p;
                d.ownR[q] = ownS.get(q);
                d.ownOk[q] = a.cand == a.t;
                d.noSecond[q] = v == V_BASE && a.noSecond;
                d.poseFrontal[q] = a.poseFrontal;
                b.ownP[q] = a.p;
                b.ownS[q] = ownS.get(q);
                b.ownOk[q] = d.ownOk[q];
            }
            d.thrP = thr.stream().mapToInt(x -> (int) x[0]).toArray();
            d.thrR = thr.stream().mapToDouble(x -> x[1]).toArray();
            b.thrP = d.thrP;
            b.thrS = d.thrR;
            b.ctrlP = ctrl.stream().mapToInt(x -> (int) x[0]).toArray();
            b.ctrlS = ctrl.stream().mapToDouble(x -> x[1]).toArray();
            b.ctrlPeople = java.util.Arrays.stream(b.ctrlP).distinct().sorted().toArray();
            dc.add(d);
            bc.add(b);
        }
        return new Object[] {dc, bc};
    }

    // ---------------------------------------------------------------- чтение

    static String key(String method, int s, int j, String role, String person, int image) {
        return method + "\t" + s + "\t" + j + "\t" + role + "\t" + person + "\t" + image;
    }

    /** Строки файла оценок для методов (в порядке файла). */
    static Map<String, Row> reference(Path file, List<String> methods, Map<String, List<Row>> groups) throws IOException {
        Map<String, Row> out = new HashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8), 1 << 16)) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") || l.isEmpty() || l.equals(FeiScores.COLUMNS)) continue;
                int tab = l.indexOf('\t');
                if (!methods.contains(l.substring(0, tab))) continue;
                Row r = FeiScores.parse(l);
                out.put(key(r.method(), r.s(), r.j(), r.role(), r.person(), r.image()), r);
                groups.computeIfAbsent(r.method() + "\t" + r.s() + "\t" + r.j(), k -> new ArrayList<>()).add(r);
            }
        }
        return out;
    }

    /** Лучший, второй, d₁, d₂ вектора — ровно как в строке файла оценок; иначе исключение. */
    static void check(FeiVectors.Row v, List<String> gallery, Row ref) {
        String where = v.method() + " s=" + v.s() + " j=" + v.j() + " " + v.role() + " " + v.person() + " №" + v.image();
        if (ref == null) throw new IllegalStateException("Контроль векторов: нет строки в файле оценок для " + where);
        int[] bt = FeiVectors.bestTwo(v.d());
        if (!gallery.get(bt[0]).equals(ref.bestId()) || !gallery.get(bt[1]).equals(ref.secondId()) || Double.compare(v.d()[bt[0]], ref.best()) != 0
                || Double.compare(v.d()[bt[1]], ref.second()) != 0 || !v.trueId().equals(ref.trueId())) {
            throw new IllegalStateException("Контроль векторов: не совпало с файлом оценок: " + where);
        }
    }

    /** Часть i: векторы X и X+mirror, контроль, попытки по позе, Z-суммы. */
    static Part load(int i, Path dir, Map<String, Row> ref, Map<String, List<Row>> groups, FeiPose pose, double r0, Map<String, Integer> persons)
            throws IOException {
        return load(PART_BASES[i], FeiFusion.PARTS.get(i), FeiFusion.letter(i), dir, ref, groups, pose, r0, persons);
    }

    /** Часть по базовому методу x (строки x и x+mirror), строка id (*-ratio+pose) и метка label. */
    static Part load(String x, String id, String label, Path dir, Map<String, Row> ref, Map<String, List<Row>> groups, FeiPose pose, double r0,
                     Map<String, Integer> persons) throws IOException {
        Part part = new Part();
        part.label = label;
        part.id = id;
        part.base = x;
        String[] ms = {x, x + FeiMethods.MIRROR};
        boolean cosine = x.contains(FeiScores.COSINE);
        FeiVectors.Loaded[] f = new FeiVectors.Loaded[2];
        Map<String, FeiVectors.Row> mir = new HashMap<>();
        for (int m = 0; m < 2; m++) {
            Path file = dir.resolve(FeiVectors.file(ms[m]));
            f[m] = FeiVectors.read(file);
            part.files[m] = String.format(Locale.ROOT, "%s: %d байт, SHA-256 %s", file.getFileName(), f[m].bytes(), f[m].sha256());
            // Попыток в файле оценок — по конфигурациям, которые есть в файле векторов (все 40 или первые N при отладке).
            java.util.Set<String> cfgKeys = new java.util.HashSet<>();
            for (FeiVectors.Row v : f[m].rows()) cfgKeys.add(ms[m] + "\t" + v.s() + "\t" + v.j());
            long refCount = cfgKeys.stream().mapToLong(k -> groups.getOrDefault(k, List.of()).size()).sum();
            for (FeiVectors.Row v : f[m].rows()) {
                if (!v.method().equals(ms[m])) throw new IllegalStateException("Файл " + file + ": строка метода " + v.method());
                check(v, f[m].galleries().get(v.s()), ref.get(key(v.method(), v.s(), v.j(), v.role(), v.person(), v.image())));
                if (m == 1) mir.put(key("", v.s(), v.j(), v.role(), v.person(), v.image()), v);
            }
            if (f[m].rows().size() != refCount) {
                throw new IllegalStateException("Контроль векторов " + ms[m] + ": строк " + f[m].rows().size() + ", в файле оценок " + refCount);
            }
            part.checked[m] = refCount;
        }
        Map<String, Conf> byCfg = new LinkedHashMap<>();
        for (FeiVectors.Row xr : f[0].rows()) {
            Conf c = byCfg.computeIfAbsent(xr.s() + "\t" + xr.j(), k -> {
                Conf n = new Conf();
                n.s = xr.s();
                n.j = xr.j();
                n.numberFrontal = FeiProtocol.frontal(xr.j());
                return n;
            });
            FeiVectors.Row mr = mir.get(key("", xr.s(), xr.j(), xr.role(), xr.person(), xr.image()));
            if (mr == null) throw new IllegalStateException("Нет попытки X+mirror для " + xr.person() + " №" + xr.image());
            List<String> gallery = f[0].galleries().get(xr.s());
            Att a = new Att();
            a.role = xr.role();
            a.person = xr.person();
            a.image = xr.image();
            Integer pi = persons.get(xr.person());
            if (pi == null) throw new IllegalStateException("Человек " + xr.person() + " не входит в людей FEI");
            a.p = pi;
            if (a.role.equals(FeiScores.OWN)) a.t = gallery.indexOf(xr.trueId());
            a.poseFrontal = pose.absR(xr.person(), xr.image()) <= r0;
            a.mode = a.poseFrontal ? 0 : 1;
            double[] d = a.mode == 0 ? xr.d() : mr.d();
            String where = ms[a.mode] + " s=" + xr.s() + " j=" + xr.j() + " " + xr.role() + " " + xr.person() + " №" + xr.image();
            a.l = new double[d.length];
            for (int p = 0; p < d.length; p++) a.l[p] = logd(d[p], cosine, part.lCases, where);
            int[] bt = FeiVectors.bestTwo(d);
            a.cand = bt[0];
            int before = part.rCases[FeiScores.CASE_NO_SECOND];
            a.ratio = FeiScores.ratio(d[bt[0]], d[bt[1]], cosine, () -> where, part.rCases);
            a.noSecond = part.rCases[FeiScores.CASE_NO_SECOND] > before;
            a.tScore = tNorm(a.l, a.cand);
            if (a.t >= 0) a.rank = FeiVectors.rank(d, a.t);
            c.all.add(a);
        }
        for (Conf c : byCfg.values()) {
            sums(c);
            part.confs.add(c);
            for (int m = 0; m < 2; m++) {
                if (!c.used[m]) continue;
                if (c.total[m] < 2) throw new IllegalStateException("Z-статистика: в режиме " + m + " конфигурации " + c.s + " " + c.j + " снимков пороговых "
                        + c.total[m]);
                if (c.total[m] < part.zMinMode[m]) part.zMinMode[m] = c.total[m];
                if (c.total[m] < part.zMin) {
                    part.zMin = c.total[m];
                    part.zMinWhere = "s=" + c.s + " j=" + c.j + ", режим " + (m == 0 ? "X" : "X+mirror");
                }
            }
        }
        controlBase(part, groups, pose, r0);
        return part;
    }

    /** Z-суммы конфигурации по людям пороговых, со сдвигом — средним ℓ по всем пороговым режима. */
    static void sums(Conf c) {
        int g = c.all.get(0).l.length;
        c.zPersons = new int[2][];
        c.zCount = new int[2][];
        c.s1 = new double[2][][];
        c.s2 = new double[2][][];
        c.shift = new double[2][g];
        for (Att a : c.all) c.used[a.mode] = true;
        for (int m = 0; m < 2; m++) {
            Map<Integer, Integer> idx = new LinkedHashMap<>();
            List<Att> thr = new ArrayList<>();
            for (Att a : c.all) {
                if (a.role.equals(FeiScores.THR) && a.mode == m) {
                    thr.add(a);
                    idx.putIfAbsent(a.p, idx.size());
                }
            }
            c.total[m] = thr.size();
            int nq = idx.size();
            c.zPersons[m] = idx.keySet().stream().mapToInt(Integer::intValue).toArray();
            c.zCount[m] = new int[nq];
            c.s1[m] = new double[nq][g];
            c.s2[m] = new double[nq][g];
            for (Att a : thr) for (int p = 0; p < g; p++) c.shift[m][p] += a.l[p] / thr.size();
            for (Att a : thr) {
                int q = idx.get(a.p);
                c.zCount[m][q]++;
                for (int p = 0; p < g; p++) {
                    double d = a.l[p] - c.shift[m][p];
                    c.s1[m][q][p] += d;
                    c.s2[m][q][p] += d * d;
                }
            }
        }
    }

    /** База по векторам = FeiScores.Stat строки *-ratio+pose из файла оценок (ошибки и FAR контроля); иначе исключение. */
    @SuppressWarnings("unchecked")
    static void controlBase(Part part, Map<String, List<Row>> groups, FeiPose pose, double r0) {
        String x = part.base;
        Stat st = new Stat(part.id, Transform.RATIO);
        for (Conf c : part.confs) {
            List<Row> b = groups.get(x + "\t" + c.s + "\t" + c.j);
            List<Row> m = groups.get(x + FeiMethods.MIRROR + "\t" + c.s + "\t" + c.j);
            st.add(FeiScores.choose(b, m, pose, r0, x + "+pose"));
        }
        Object[] cs = cfgs(part.confs, V_BASE, null);
        double[][][] cnt = FeiDecompose.counts((List<FeiDecompose.Cfg>) cs[0], null);
        int[] ks = {FeiScores.T1, FeiScores.T5};
        int[] unit = FeiBootstrap.unit(maxPerson(part) + 1);
        for (int k = 0; k < 2; k++) {
            double total = java.util.Arrays.stream(cnt[k][0]).sum();
            double err = total - cnt[k][0][FeiDecompose.G_OK];
            int[] ctrl = FeiBootstrap.ctrlCounts((List<FeiBootstrap.Cfg>) cs[1], unit, k);
            int[][] far = st.far(st.farCtrl, ks[k]);
            int x0 = 0;
            for (int[] r : far) x0 += r[0];
            if (err != st.errors(ks[k]) || total != st.att || ctrl[0] != x0) {
                throw new IllegalStateException("Контроль базы " + part.id + ": ошибка " + err + " ≠ " + st.errors(ks[k]) + " или FAR контроля " + ctrl[0]
                        + " ≠ " + x0);
            }
        }
    }

    private static int maxPerson(Part part) {
        int m = 0;
        for (Conf c : part.confs) for (Att a : c.all) m = Math.max(m, a.p);
        return m;
    }

    // ---------------------------------------------------------------- расчёт

    static void evaluate(Part part, int persons, int b, long seed) {
        evaluate(part, persons, b, seed, NV);
    }

    /** Варианты 0…nv − 1 (nv = 1 — только база). */
    @SuppressWarnings("unchecked")
    static void evaluate(Part part, int persons, int b, long seed, int nv) {
        int[] u = FeiBootstrap.unit(persons);
        int[][] w = b > 0 ? FeiBootstrap.weights(persons, b, seed) : null;
        for (int v = 0; v < nv; v++) {
            Object[] pc = cfgs(part.confs, v, null);
            List<FeiDecompose.Cfg> dc = (List<FeiDecompose.Cfg>) pc[0];
            List<FeiBootstrap.Cfg> bc = (List<FeiBootstrap.Cfg>) pc[1];
            part.pointCounts[v] = FeiDecompose.counts(dc, null);
            part.pointFar[v] = FeiBootstrap.metrics(bc, u);
            part.cp[v] = new int[][] {FeiBootstrap.ctrlCounts(bc, u, 0), FeiBootstrap.ctrlCounts(bc, u, 1)};
            ranks(part, v, dc);
            if (w == null) continue;
            boolean fixed = v == V_BASE || v == V_T;
            double[][][][] sh = new double[b][][][];
            double[][] far = new double[FeiBootstrap.METRICS][b];
            int vv = v;
            IntStream.range(0, b).parallel().forEach(q -> {
                List<FeiDecompose.Cfg> d = dc;
                List<FeiBootstrap.Cfg> bb = bc;
                if (!fixed) {
                    Object[] r = cfgs(part.confs, vv, w[q]);
                    d = (List<FeiDecompose.Cfg>) r[0];
                    bb = (List<FeiBootstrap.Cfg>) r[1];
                }
                sh[q] = FeiDecompose.shares(FeiDecompose.counts(d, w[q]));
                double[] m = FeiBootstrap.metrics(bb, w[q]);
                for (int k = 0; k < FeiBootstrap.METRICS; k++) far[k][q] = m[k];
            });
            part.repShares[v] = sh;
            part.repFar[v] = far;
        }
    }

    /** Ранги истинного (2 / 3–5 / >5) в группе (iii) варианта v при порогах точки и у всех неверных argmin (один раз). */
    static void ranks(Part part, int v, List<FeiDecompose.Cfg> dc) {
        for (int ci = 0; ci < dc.size(); ci++) {
            FeiDecompose.Cfg d = dc.get(ci);
            List<Att> own = part.confs.get(ci).all.stream().filter(a -> a.role.equals(FeiScores.OWN)).toList();
            for (int k = 0; k < 2; k++) {
                double th = FeiDecompose.threshold(d, null, k);
                for (int q = 0; q < own.size(); q++) {
                    if (FeiDecompose.group(d, q, th) == FeiDecompose.G_WRONG_REJ) part.rankIii[v][k][bin(own.get(q).rank)]++;
                }
            }
            if (v == V_BASE) for (Att a : own) if (a.cand != a.t) part.rankWrong[bin(a.rank)]++;
        }
    }

    static int bin(int rank) {
        if (rank < 2) throw new IllegalStateException("Ранг истинного " + rank + " у неверного кандидата");
        return rank == 2 ? 0 : rank <= 5 ? 1 : 2;
    }

    /** Ошибка своих, %: точка или реплика q; порог k, деление sp. */
    static double err(double[][][] shares, int k, int sp) {
        return 100 - shares[k][sp][FeiDecompose.G_OK];
    }

    /** Реплики ошибки варианта v. */
    static double[] errReps(Part p, int v, int k, int sp) {
        double[] r = new double[p.repShares[v].length];
        for (int q = 0; q < r.length; q++) r[q] = err(p.repShares[v][q], k, sp);
        return r;
    }

    /** Интервал [2,5; 97,5] разности реплик. */
    static double[] ci(double[] reps) {
        double[] s = FeiBootstrap.sorted(reps);
        return new double[] {FeiBootstrap.quantile(s, 0.025), FeiBootstrap.quantile(s, 0.975)};
    }

    /**
     * Критерий выбора (зафиксирован заранее): S — варианты (T, Z, ZT) со значимо меньшей ошибкой своих A при FAR 1 %, чем
     * база; S пусто — база; иначе v* — наименьшая ошибка в S; более простой вариант из S (порядок T, Z, ZT), значимо не
     * отличающийся от v*, выбирается вместо него.
     */
    static int choose(double[] point, double[][] reps) {
        List<Integer> s = new ArrayList<>();
        for (int v = V_T; v < NV; v++) {
            double[] d = diff(reps[v], reps[V_BASE]);
            if (point[v] < point[V_BASE] && ci(d)[1] < 0) s.add(v);
        }
        if (s.isEmpty()) return V_BASE;
        int best = s.get(0);
        for (int v : s) if (point[v] < point[best]) best = v;
        for (int v : s) {
            if (v >= best) break;
            double[] c = ci(diff(reps[best], reps[v]));
            if (c[0] <= 0 && c[1] >= 0) return v;
        }
        return best;
    }

    static double[] diff(double[] a, double[] b) {
        double[] d = new double[a.length];
        for (int q = 0; q < a.length; q++) d[q] = a[q] - b[q];
        return d;
    }

    // ---------------------------------------------------------------- запуск и отчёт

    public static void main(String[] args) throws IOException {
        String[] a = args.length == 1 ? args[0].trim().split("\\s+") : args;
        if (a.length < 2 || a[0].isBlank()) throw new IllegalArgumentException("faces-fei-norm: -Dexec.args=\"<каталог векторов> <файл оценок>\"");
        Path dir = Paths.get(a[0].trim());
        Path scores = Paths.get(a[1].trim());
        SettingsStore settings = new SettingsStore();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("faces-fei-norm: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        FeiPose sel = FeiScores.selection(settings);
        List<String> people = sel.persons();
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < people.size(); i++) idx.put(people.get(i), i);
        int b = settings.loadFacesFeiBootstrap();
        long seed = settings.loadFacesSeed() + FeiBootstrap.SEED_OFFSET;
        List<String> methods = new ArrayList<>();
        for (String x : PART_BASES) {
            methods.add(x);
            methods.add(x + FeiMethods.MIRROR);
        }
        long t0 = System.nanoTime();
        Map<String, List<Row>> groups = new HashMap<>();
        Map<String, Row> ref = reference(scores, methods, groups);
        List<Part> parts = new ArrayList<>();
        for (int i = 0; i < PART_BASES.length; i++) {
            Part p = load(i, dir, ref, groups, sel, r0, idx);
            evaluate(p, people.size(), b, seed);
            p.confs = List.of();
            parts.add(p);
        }
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(out);
        FarMethods.write(out.resolve(FILE), report(parts, dir.toString(), scores.toString(), r0, b, seed, people.size()));
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "Нормировка: %.0f с; отчёт %s",
                (System.nanoTime() - t0) / 1e9, out.resolve(FILE)));
    }

    static String cell(double x, double n, double[] reps) {
        String s = FeiMethods.pctOf((int) x, (int) n);
        if (reps == null) return s;
        double[] c = ci(reps);
        return s + String.format(Locale.ROOT, " [%.1f; %.1f]", c[0], c[1]) + FeiBootstrap.note(reps);
    }

    static StringBuilder report(List<Part> parts, String dir, String scores, double r0, int b, long seed, int persons) {
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: нормировка оценки по полным векторам расстояний (FeiNorm, faces-fei-norm)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Векторы: ").append(dir).append(" (вне git; получены командой: ").append(COMMAND).append(")\n");
        for (Part p : parts) for (String f : p.files) t.append("  ").append(f).append('\n');
        t.append("Файл оценок для контроля: ").append(scores).append('\n');
        t.append("Части: ");
        for (Part p : parts) t.append(p.label).append(" = ").append(p.id).append("; ");
        t.append("r0 = ").append(r0).append(" (|r| пробы ≤ r0 — вектор модели X, иначе X+mirror).\n");
        t.append("Кандидат c — argmin сырых d во всех вариантах (группы (iii), (iv) и ошибка argmin от варианта не зависят). Над ℓ = log d:\n"
                + "база — log(d₁/d₂) (FeiScores.ratio); T — (ℓ_c − μ)/σ по 99 людям галереи кроме c (σ — n − 1); Z — (ℓ_c − μ_c,m)/σ_c,m,\n"
                + "статистика модели c в режиме m пробы (X или X+mirror) по снимкам 40 пороговых чужих конфигурации того же режима; ZT —\n"
                + "сначала Z всех людей галереи (режим пробы), затем T по z без c. Параметров нет, кроме зафиксированных заранее: d < 1e-12\n"
                + "→ 1e-12; 1 − cos в [−1e-12; 0) → 0; σ = 0, d < 0, NaN, +∞ — исключение. Порог — как в fei_scores.txt (Нейман – Пирсон по\n"
                + "снимкам 40 пороговых, FAR 1 % и 5 %). Группы (i)–(iv) — как fei_decompose.txt; (v-а) — случай 4 правила d₁/d₂ (только\n"
                + "база). Анфас/поворот — по |r| ≤ r0.\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям FEI целиком: B = %d, seed = %d, людей %d, те же веса, что в fei_bootstrap.txt; "
                    + "Z-статистики и порог — заново\nв каждой реплике и конфигурации. [2,5; 97,5] — процентили; «*» — интервал разности не "
                    + "содержит 0.\n", b, seed, persons));
        } else {
            t.append("Бутстреп выключен (faces.fei.bootstrap не задан): интервалов, разностей и выбора варианта нет.\n");
        }
        t.append("\n=== Контроль ===\n");
        for (Part p : parts) {
            t.append(String.format(Locale.ROOT, "%s: векторы X и X+mirror — %d и %d попыток, лучший, второй, d₁, d₂ — ровно как в файле оценок; база — "
                    + "те же ошибки своих и FAR контроля, что FeiScores.Stat %s (fei_scores.txt).%n".replace("%n", "\n"), p.label, p.checked[0], p.checked[1],
                    p.id));
            t.append(String.format(Locale.ROOT, "  случаи log d: d < 1e-12 — %d, 1 − cos округлено — %d; правило d₁/d₂: ", p.lCases[L_FLOOR], p.lCases[L_COS]));
            for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(p.rCases[k]);
            t.append(String.format(Locale.ROOT, "%n  Z-статистика: наименьшее число снимков пороговых (модель × режим) — %d (%s); по режимам: X %d, X+mirror %d%n"
                    .replace("%n", "\n"), p.zMin, p.zMinWhere, p.zMinMode[0], p.zMinMode[1]));
        }

        String[] farName = {"FAR 1 %", "FAR 5 %"};
        t.append("\n=== Ошибка своих и FAR контроля ===\n");
        t.append("часть, вариант | ошибка своих FAR 1 % | FAR 5 % | анфас FAR 1 % | поворот FAR 1 % | анфас FAR 5 % | поворот FAR 5 % | "
                + "FAR контроля снимки 1 %: точка ↑95; КП | 5 %\n");
        for (Part p : parts) {
            for (int v = 0; v < NV; v++) {
                List<String> c = new ArrayList<>();
                c.add(p.label + ", " + V_NAMES[v]);
                for (int[] ks : new int[][] {{0, 0}, {1, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}}) {
                    double[][][] cnt = p.pointCounts[v];
                    double n = java.util.Arrays.stream(cnt[ks[0]][ks[1]]).sum();
                    c.add(cell(n - cnt[ks[0]][ks[1]][FeiDecompose.G_OK], n, b > 0 ? errReps(p, v, ks[0], ks[1]) : null));
                }
                for (int k = 0; k < 2; k++) {
                    int[] xn = p.cp[v][k];
                    String s = String.format(Locale.ROOT, "%.2f %%", p.pointFar[v][k == 0 ? FeiBootstrap.FAR1 : FeiBootstrap.FAR5]);
                    if (b > 0) {
                        s += String.format(Locale.ROOT, " ↑%.2f %%", FeiBootstrap.quantile(FeiBootstrap.sorted(p.repFar[v][k == 0 ? FeiBootstrap.FAR1
                                : FeiBootstrap.FAR5]), 0.95));
                    }
                    s += String.format(Locale.ROOT, "; КП %d/%d ↑%.2f %%", xn[0], xn[1], 100 * FaceEvaluation.binomialUpperBound(xn[0], xn[1], 0.95));
                    c.add(s);
                }
                t.append(String.join(" | ", c)).append('\n');
            }
        }

        t.append("\n=== Разложение ошибки своих: (i) верный принят, (ii) верный отказ, (iii) неверный отказ, (iv) неверный принят, (v-а) "
                + "случай 4 ===\n");
        for (Part p : parts) {
            for (int k = 0; k < 2; k++) {
                t.append("-- ").append(p.label).append(", ").append(farName[k]).append('\n');
                t.append("вариант | деление | (i) | (ii) | (iii) | (iv) | (v-а)\n");
                for (int v = 0; v < NV; v++) {
                    for (int sp = 0; sp < 3; sp++) {
                        double[][][] cnt = p.pointCounts[v];
                        double n = java.util.Arrays.stream(cnt[k][sp]).sum();
                        List<String> c = new ArrayList<>(List.of(V_NAMES[v], FeiDecompose.SPLIT_NAMES[sp]));
                        for (int g = 0; g <= FeiDecompose.G_NO_SECOND; g++) {
                            double[] r = null;
                            if (b > 0) {
                                r = new double[b];
                                for (int q = 0; q < b; q++) r[q] = p.repShares[v][q][k][sp][g];
                            }
                            c.add(cell(cnt[k][sp][g], n, r));
                        }
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }

        if (b > 0) {
            t.append("\n=== Парные разности ошибки своих, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("часть, A − B | FAR 1 % всего | анфас | поворот | FAR 5 % всего | анфас | поворот | FAR контроля снимки 1 %\n");
            int[][] pairs = {{V_T, V_BASE}, {V_Z, V_BASE}, {V_ZT, V_BASE}, {V_Z, V_T}, {V_ZT, V_T}, {V_ZT, V_Z}};
            for (Part p : parts) {
                for (int[] pr : pairs) {
                    List<String> c = new ArrayList<>(List.of(p.label + ", " + V_NAMES[pr[0]] + " − " + V_NAMES[pr[1]]));
                    for (int[] ks : new int[][] {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}) {
                        double pt = err(FeiDecompose.shares(p.pointCounts[pr[0]]), ks[0], ks[1]) - err(FeiDecompose.shares(p.pointCounts[pr[1]]), ks[0], ks[1]);
                        c.add(FeiBootstrap.diff(pt, diff(errReps(p, pr[0], ks[0], ks[1]), errReps(p, pr[1], ks[0], ks[1])), 1));
                    }
                    c.add(FeiBootstrap.diff(p.pointFar[pr[0]][FeiBootstrap.FAR1] - p.pointFar[pr[1]][FeiBootstrap.FAR1],
                            diff(p.repFar[pr[0]][FeiBootstrap.FAR1], p.repFar[pr[1]][FeiBootstrap.FAR1]), 2));
                    t.append(String.join(" | ", c)).append('\n');
                }
            }
            Part a = parts.get(0);
            double[] point = new double[NV];
            double[][] reps = new double[NV][];
            for (int v = 0; v < NV; v++) {
                point[v] = err(FeiDecompose.shares(a.pointCounts[v]), 0, 0);
                reps[v] = errReps(a, v, 0, 0);
            }
            int ch = choose(point, reps);
            t.append("\n=== Выбор варианта по критерию (зафиксирован заранее) ===\n");
            t.append("Наименьшая ошибка своих у A при FAR 1 %; принимается, только если разность с базой значима; при незначимой разности с более\n"
                    + "простым значимым вариантом (порядок T, Z, ZT) — более простой; значимых нет — база.\n");
            t.append("Выбрано: ").append(V_NAMES[ch]).append(String.format(Locale.ROOT, " (ошибка своих A при FAR 1 %%: %.1f %%).%n".replace("%n", "\n"),
                    point[ch]));
        }

        t.append("\n=== Ранг истинного в сыром векторе (выбранном по позе) у неверного кандидата: 2 / 3–5 / >5 ===\n");
        t.append("часть | все неверные argmin | группа (iii) по вариантам: FAR 1 % ; FAR 5 %\n");
        for (Part p : parts) {
            int n = p.rankWrong[0] + p.rankWrong[1] + p.rankWrong[2];
            StringBuilder c = new StringBuilder(p.label + " | " + bins(p.rankWrong, n));
            for (int v = 0; v < NV; v++) {
                c.append(" | ").append(V_NAMES[v]).append(": ");
                for (int k = 0; k < 2; k++) {
                    int[] r = p.rankIii[v][k];
                    c.append(k > 0 ? " ; " : "").append(bins(r, r[0] + r[1] + r[2]));
                }
            }
            t.append(c).append('\n');
        }
        return t;
    }

    static String bins(int[] r, int n) {
        return String.format(Locale.ROOT, "%s / %s / %s", FeiMethods.pctOf(r[0], n), FeiMethods.pctOf(r[1], n), FeiMethods.pctOf(r[2], n));
    }
}
