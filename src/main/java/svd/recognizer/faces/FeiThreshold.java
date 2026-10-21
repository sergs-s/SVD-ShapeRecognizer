package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import org.apache.commons.math3.special.Beta;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.storage.SettingsStore;

/**
 * PLAN.md, п. 4.2, «Калибровка порога»: способы задания порога Нейман – Пирсон по снимкам (номинал FAR 1 % и 5 %) для
 * L1 = 5-lbp-mlda-ratio+pose и L1-in = 5-lbp-mlda-in-ratio+pose; оценки своих и контрольных чужих — из файла оценок,
 * меняется только порог. П0 — по 40 пороговым чужим, точечно; П1 — по ним же с запасом (верхняя односторонняя 95 %
 * граница FAR не выше номинала, с учётом группировки снимков по людям); П2 — перекрёстно по галерее (5 частей, «чужие» —
 * снимки людей части против модели без неё; файл FeiMethods faces.fei.cv), точечно; П3 — П2 с запасом. Бутстреп по
 * людям FEI целиком (те же веса, что fei_bootstrap.txt), порог — заново своим способом в каждой реплике и конфигурации.
 * Выбор способа — по критерию, зафиксированному заранее (choose).
 *
 * Запуск: {@code mvn compile exec:java@faces-fei-threshold "-Dexec.args=<файл оценок> <файл перекрёстной оценки>"} при
 * заданных faces.fei.pose.r0 и faces.export.dir; бутстреп — faces.fei.bootstrap. Отчёт reports/faces/fei/fei_threshold.txt.
 *
 * @author ssv
 */
public final class FeiThreshold {

    static final String FILE = "fei_threshold.txt";
    /** Варианты: строка файла оценок (+pose), базовый метод перекрёстной оценки, метка. */
    static final String[] VARIANTS = {"5-lbp-mlda-ratio+pose", "5-lbp-mlda-in-ratio+pose"};
    static final String[] CV_BASES = {FeiMethods.LBP_MLDA, FeiMethods.LBP_MLDA_IN};
    static final String[] LABELS = {"L1", "L1-in"};
    /** Способы в порядке простоты. */
    static final String[] METHODS = {"П0", "П1", "П2", "П3"};
    static final int M = METHODS.length;
    /** Доверие границы П1 / П3 (фиксировано заранее). */
    static final double CONFIDENCE = 0.95;
    static final int KS = FeiDecompose.KS;
    /** Деления в таблицах: всего, анфас, поворот (по |r| ≤ r0). */
    static final int SP = 3;

    private FeiThreshold() {
    }

    // ---------------------------------------------------------------- граница с учётом людей

    /**
     * Верхняя односторонняя граница Клоппера – Пирсона для доли x/n (x, n — не обязательно целые) не выше far:
     * P(Beta(x + 1, n − x) ≤ far) ≥ confidence; при x = 0 — 1 − (1 − confidence)^(1/n) ≤ far.
     */
    static boolean boundOk(double x, double n, double far, double confidence) {
        if (n <= x) return false;
        return Beta.regularizedBeta(far, x + 1, n - x) >= confidence;
    }

    /**
     * Эффект группировки (deff) доли X/n по людям: k — число людей (с весами), sxx = Σ wᵢxᵢ², sxn = Σ wᵢxᵢnᵢ, snn = Σ wᵢnᵢ²;
     * кластерная дисперсия v = k/(k − 1)·Σ wᵢ(xᵢ − p̂nᵢ)²/n², deff = max(1, v / (p̂(1 − p̂)/n)); при X = 0 — 1; при k ≤ 1 или
     * p̂ = 1 — +∞.
     */
    static double deff(double x, double n, double k, double sxx, double sxn, double snn) {
        if (x == 0) return 1;
        double p = x / n;
        if (k <= 1 || p >= 1) return Double.POSITIVE_INFINITY;
        double ss = sxx - 2 * p * sxn + p * p * snn;
        double v = k / (k - 1) * Math.max(0, ss) / (n * n);
        return Math.max(1, v / (p * (1 - p) / n));
    }

    /** Граница доли X/n с учётом людей не выше far: n_eff = n/deff, x_eff = p̂·n_eff. */
    static boolean marginOk(double x, double n, double k, double sxx, double sxn, double snn, double far) {
        double d = deff(x, n, k, sxx, sxn, snn);
        if (Double.isInfinite(d)) return false;
        double ne = n / d;
        return boundOk(x / n * ne, ne, far, CONFIDENCE);
    }

    /** Порог с запасом: θ и признак «недопустим даже ноль принятых» (тогда θ — ниже наименьшей оценки чужих). */
    record Th(double theta, boolean zeroFail) {}

    /**
     * Порог с запасом (П1, П3) по чужим r (по возрастанию) людей p при весах людей w: проход по числу принятых снимков m = 0,
     * 1, … (только на границах между разными оценками) до первого m, при котором граница FAR с учётом людей выше far; порог
     * — последний допустимый (θ = nextDown(r[m]), при m = n — +∞).
     */
    static Th margin(double[] r, int[] p, int[] w, double far) {
        Map<Integer, Integer> ni = new HashMap<>();
        for (int q : p) ni.merge(q, 1, Integer::sum);
        double k = 0;
        double n = 0;
        double snn = 0;
        for (Map.Entry<Integer, Integer> e : ni.entrySet()) {
            int wt = w[e.getKey()];
            k += wt;
            n += (double) wt * e.getValue();
            snn += (double) wt * e.getValue() * e.getValue();
        }
        if (r.length == 0) return new Th(Double.POSITIVE_INFINITY, false);
        // Пустая реплика: у чужих порога суммарный вес 0 – порог не определён (NaN; решение Хозяина 20.10.2026, №6 Жорика).
        if (n == 0) return new Th(Double.NaN, false);
        // Порог – только по оценкам людей с весом > 0 (люди с весом 0 в реплике не вытянуты; исправление 20.10.2026, №4 Жорика).
        int first = nextWeighted(r, p, w, -1);
        if (!marginOk(0, n, k, 0, 0, snn, far)) return new Th(Math.nextDown(r[first]), true);
        Map<Integer, Integer> xi = new HashMap<>();
        double x = 0;
        double sxx = 0;
        double sxn = 0;
        double theta = Math.nextDown(r[first]);
        for (int i = 0; i < r.length; i++) {
            int q = p[i];
            int wt = w[q];
            int xq = xi.getOrDefault(q, 0);
            x += wt;
            sxx += (double) wt * (2 * xq + 1);
            sxn += (double) wt * ni.get(q);
            xi.put(q, xq + 1);
            if (i + 1 < r.length && r[i + 1] == r[i]) continue;
            if (!marginOk(x, n, k, sxx, sxn, snn, far)) break;
            int nx = nextWeighted(r, p, w, i);
            theta = nx < r.length ? Math.nextDown(r[nx]) : Double.POSITIVE_INFINITY;
        }
        return new Th(theta, false);
    }

    /** Индекс первой оценки после i (i = −1 – с начала) у человека с весом > 0; r.length – нет такой. */
    static int nextWeighted(double[] r, int[] p, int[] w, int i) {
        int j = i + 1;
        while (j < r.length && w[p[j]] == 0) j++;
        return j;
    }

    // ---------------------------------------------------------------- критерий

    /** Допуск FAR контроля сверх номинала, доля номинала (решение Хозяина 04.10.2026: 1,1 % при 1 %, 5,5 % при 5 %). */
    static final double TOLERANCE = 0.10;

    /** Доля x/n не выше far·(1 + TOLERANCE). */
    static boolean withinTolerance(double x, double n, double far) {
        return x <= far * (1 + TOLERANCE) * n + 1e-9;
    }

    /**
     * Критерий выбора (решение Хозяина 04.10.2026; исправлен 04.10.2026 — допуск «не выше номинала» был ошибкой постановки):
     * для каждого варианта отдельно допустимы способы с фактическим FAR контроля (точка) не выше номинала + 10 % от него при
     * 1 % и 5 %; из них m* — наименьшая ошибка своих при FAR 1 %; выбирается самый простой (порядок П0…П3) допустимый способ,
     * у которого разность с m* незначима (интервал [2,5; 97,5] реплик содержит 0), иначе m*. Нет допустимых — −1.
     *
     * @param err  точечная ошибка своих варианта при FAR 1 % по способам
     * @param reps реплики этой ошибки по способам (null — без бутстрепа: разность считается значимой)
     */
    static int choose(double[] err, double[][] reps, boolean[] admissible) {
        int best = -1;
        for (int m = 0; m < err.length; m++) if (admissible[m] && (best < 0 || err[m] < err[best])) best = m;
        if (best < 0) return -1;
        for (int m = 0; m < best; m++) {
            if (!admissible[m] || reps == null) continue;
            double[] c = FeiNorm.ci(FeiNorm.diff(reps[m], reps[best]));
            if (c[0] <= 0 && c[1] >= 0) return m;
        }
        return best;
    }

    // ---------------------------------------------------------------- данные

    /** Чужие для порога одной конфигурации: r по возрастанию и человек каждого снимка. */
    record Imp(double[] r, int[] p, int frontal) implements java.io.Serializable {
        static Imp of(List<double[]> v, int frontal) {
            v.sort((a, b) -> Double.compare(a[1], b[1]));
            return new Imp(v.stream().mapToDouble(a -> a[1]).toArray(), v.stream().mapToInt(a -> (int) a[0]).toArray(), frontal);
        }

        int persons() {
            return (int) Arrays.stream(p).distinct().count();
        }
    }

    /** Перекрёстная оценка: базовый метод → «s j» → чужие (выбор модели по позе); случаи правила d₁/d₂; галереи. */
    static final class Cv {
        final Map<String, Map<String, Imp>> imp = new LinkedHashMap<>();
        final int[] cases = new int[FeiScores.CASES];
        /** «s j» → люди частей (все роли cv). */
        final Map<String, Set<String>> persons = new HashMap<>();
        long lines;
    }

    static Cv readCv(Path file, FeiPose pose, double r0, FeiBootstrap index) throws IOException {
        Map<String, Map<String, Row[]>> byCfg = new LinkedHashMap<>();
        Cv cv = new Cv();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8), 1 << 16)) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") || l.isEmpty() || l.equals(FeiScores.COLUMNS)) continue;
                Row r = FeiScores.parse(l);
                cv.lines++;
                if (!r.role().startsWith(FeiMethods.CV_ROLE)) throw new IllegalStateException("Перекрёстная оценка: роль " + r.role());
                boolean mir = r.method().endsWith(FeiMethods.MIRROR);
                String base = mir ? r.method().substring(0, r.method().length() - FeiMethods.MIRROR.length()) : r.method();
                String key = base + "\t" + r.s() + "\t" + r.j();
                Row[] pair = byCfg.computeIfAbsent(key, x -> new LinkedHashMap<>()).computeIfAbsent(r.person() + "\t" + r.image(), x -> new Row[2]);
                if (pair[mir ? 1 : 0] != null) throw new IllegalStateException("Повтор попытки перекрёстной оценки: " + l);
                pair[mir ? 1 : 0] = r;
                cv.persons.computeIfAbsent(r.s() + " " + r.j(), x -> new TreeSet<>()).add(r.person());
            }
        }
        for (Map.Entry<String, Map<String, Row[]>> e : byCfg.entrySet()) {
            String[] k = e.getKey().split("\t");
            List<double[]> v = new ArrayList<>();
            int frontal = 0;
            for (Row[] pair : e.getValue().values()) {
                if (pair[0] == null || pair[1] == null) throw new IllegalStateException("Перекрёстная оценка: нет пары X / X+mirror в " + e.getKey());
                boolean front = pose.absR(pair[0].person(), pair[0].image()) <= r0;
                if (front) frontal++;
                Row c = front ? pair[0] : pair[1];
                v.add(new double[] {index.person(c.person()), FeiScores.ratio(c, cv.cases)});
            }
            cv.imp.computeIfAbsent(k[0], x -> new LinkedHashMap<>()).put(k[1] + " " + k[2], Imp.of(v, frontal));
        }
        return cv;
    }

    // ---------------------------------------------------------------- оценка

    /** Итоги одного варианта: точка и реплики по способам. */
    static final class Res {
        String label;
        String id;
        /** [способ]: числа [порог][деление][группа] точки; FAR контроля [порог] {x, n}; θ по конфигурациям [порог][c]. */
        double[][][][] counts = new double[M][][][];
        double[][][] far = new double[M][KS][2];
        double[][][] theta = new double[M][KS][];
        /** [способ][порог]: конфигураций с «недопустим даже ноль» (точка); в репликах — сумма по репликам и конфигурациям. */
        int[][] zeroFail = new int[M][KS];
        long[][] zeroFailRep = new long[M][KS];
        /** Реплики: ошибка [способ][порог][деление][b], доли групп [способ][порог][деление][группа][b], FAR [способ][порог][b]. */
        double[][][][] errRep;
        double[][][][][] shareRep;
        double[][][] farRep;
        /** Объём чужих порога: [способ]: снимков всего, мин. и макс. на конфигурацию, людей мин. и макс. */
        int[][] volume = new int[M][5];
        /** Конфигурации точки (для FAR контроля по людям и границы с учётом людей, LbpNormReport). */
        List<FeiDecompose.Cfg> cfgs;
    }

    /** Порог способа m в конфигурации c при весах w. */
    static Th threshold(int m, FeiDecompose.Cfg c, Imp g, int[] w, double far) {
        return switch (m) {
            case 0 -> new Th(FeiBootstrap.threshold(c.thrR, c.thrP, w, far), false);
            case 1 -> margin(c.thrR, c.thrP, w, far);
            case 2 -> new Th(FeiBootstrap.threshold(g.r(), g.p(), w, far), false);
            default -> margin(g.r(), g.p(), w, far);
        };
    }

    /** Числа [порог][деление][группа] и FAR контроля [порог] {x, n} способа m при весах w; zero[порог] — счётчик «ноль недопустим». */
    static double[][][] counts(int m, List<FeiDecompose.Cfg> cs, List<Imp> gs, int[] w, double[][] far, long[] zero, double[][] theta) {
        double[][][] t = new double[KS][FeiDecompose.SPLITS][FeiDecompose.GROUPS];
        boolean[] empty = new boolean[KS];
        for (int ci = 0; ci < cs.size(); ci++) {
            FeiDecompose.Cfg c = cs.get(ci);
            for (int k = 0; k < KS; k++) {
                Th th = threshold(m, c, gs.get(ci), w, FeiScores.FARS[k]);
                if (th.zeroFail()) zero[k]++;
                if (theta != null) theta[k][ci] = th.theta();
                for (int q = 0; q < c.ownR.length; q++) {
                    int wt = w[c.ownP[q]];
                    if (wt == 0) continue;
                    int g = FeiDecompose.group(c, q, th.theta());
                    for (int sp = 0; sp < FeiDecompose.SPLITS; sp++) if (FeiDecompose.in(c, q, sp)) t[k][sp][g] += wt;
                }
                for (int i = 0; i < c.ctrlR.length; i++) {
                    int wt = w[c.ctrlP[i]];
                    far[k][1] += wt;
                    if (c.ctrlR[i] <= th.theta()) far[k][0] += wt;
                }
                if (Double.isNaN(th.theta())) empty[k] = true;
            }
        }
        // Порог не определён хотя бы в одной конфигурации – реплика для этого порога пустая (NaN).
        for (int k = 0; k < KS; k++) {
            if (!empty[k]) continue;
            for (double[] sp : t[k]) java.util.Arrays.fill(sp, Double.NaN);
            far[k][0] = Double.NaN;
        }
        return t;
    }

    static Res evaluate(String label, String id, List<FeiDecompose.Cfg> cs, List<Imp> gs, int persons, int b, long seed) {
        Res res = new Res();
        res.label = label;
        res.id = id;
        res.cfgs = cs;
        int[] u = FeiBootstrap.unit(persons);
        for (int m = 0; m < M; m++) {
            long[] zero = new long[KS];
            res.theta[m] = new double[KS][cs.size()];
            res.counts[m] = counts(m, cs, gs, u, res.far[m], zero, res.theta[m]);
            for (int k = 0; k < KS; k++) res.zeroFail[m][k] = (int) zero[k];
            int[] vol = {0, Integer.MAX_VALUE, 0, Integer.MAX_VALUE, 0};
            for (int ci = 0; ci < cs.size(); ci++) {
                int n = m < 2 ? cs.get(ci).thrR.length : gs.get(ci).r().length;
                int pp = m < 2 ? (int) Arrays.stream(cs.get(ci).thrP).distinct().count() : gs.get(ci).persons();
                vol[0] += n;
                vol[1] = Math.min(vol[1], n);
                vol[2] = Math.max(vol[2], n);
                vol[3] = Math.min(vol[3], pp);
                vol[4] = Math.max(vol[4], pp);
            }
            res.volume[m] = vol;
        }
        if (b == 0) return res;
        int[][] w = FeiBootstrap.weights(persons, b, seed);
        res.errRep = new double[M][KS][SP][b];
        res.shareRep = new double[M][KS][SP][FeiDecompose.GROUPS][b];
        res.farRep = new double[M][KS][b];
        long[][][] zr = new long[b][M][KS];
        IntStream.range(0, b).parallel().forEach(q -> {
            for (int m = 0; m < M; m++) {
                double[][] far = new double[KS][2];
                double[][][] sh = FeiDecompose.shares(counts(m, cs, gs, w[q], far, zr[q][m], null));
                for (int k = 0; k < KS; k++) {
                    res.farRep[m][k][q] = 100.0 * far[k][0] / far[k][1];
                    for (int sp = 0; sp < SP; sp++) {
                        res.errRep[m][k][sp][q] = 100 - sh[k][sp][FeiDecompose.G_OK];
                        for (int g = 0; g < FeiDecompose.GROUPS; g++) res.shareRep[m][k][sp][g][q] = sh[k][sp][g];
                    }
                }
            }
        });
        for (long[][] z : zr) for (int m = 0; m < M; m++) for (int k = 0; k < KS; k++) res.zeroFailRep[m][k] += z[m][k];
        return res;
    }

    /** Ошибка своих точки, %. */
    static double err(Res r, int m, int k, int sp) {
        double[] t = r.counts[m][k][sp];
        double n = Arrays.stream(t).sum();
        return 100.0 * (n - t[FeiDecompose.G_OK]) / n;
    }

    /** Допуск: FAR контроля (точка) не выше номинала + TOLERANCE от него при 1 % и 5 % (1,1 % и 5,5 %). */
    static boolean admissible(Res r, int m) {
        for (int k = 0; k < KS; k++) if (!withinTolerance(r.far[m][k][0], r.far[m][k][1], FeiScores.FARS[k])) return false;
        return true;
    }

    // ---------------------------------------------------------------- запуск

    public static void main(String[] args) throws IOException {
        String[] a = args.length == 1 ? args[0].trim().split("\\s+") : args;
        if (a.length < 2) throw new IllegalArgumentException("faces-fei-threshold: -Dexec.args=\"<файл оценок> <файл перекрёстной оценки>\"");
        SettingsStore settings = new SettingsStore();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("faces-fei-threshold: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        FeiPose sel = FeiScores.selection(settings);
        List<String> people = sel.persons();
        FeiBootstrap index = new FeiBootstrap(people);
        int b = settings.loadFacesFeiBootstrap();
        long seed = settings.loadFacesSeed() + FeiBootstrap.SEED_OFFSET;
        long t0 = System.nanoTime();
        if (a[0].equals("norm")) {
            normMain(a, sel, r0, index, people, b, seed, t0);
            return;
        }
        Path scores = Paths.get(a[0]);
        Path cvFile = Paths.get(a[1]);
        Loaded ld = load(scores, cvFile, sel, r0, index, people, b, seed);
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(out);
        FarMethods.write(out.resolve(FILE), report(ld.res(), ld.dec(), ld.cv(), ld.control(), scores, cvFile, r0, b, seed, people.size()));
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "Калибровка порога: %.0f с; отчёт %s",
                (System.nanoTime() - t0) / 1e9, out.resolve(FILE)));
    }

    /** Итоги пары «файл оценок + файл перекрёстной оценки»: варианты L1, L1-in, разбор, перекрёстная оценка, строки контроля. */
    record Loaded(List<Res> res, FeiDecompose dec, Cv cv, List<String> control) {}

    static Loaded load(Path scores, Path cvFile, FeiPose sel, double r0, FeiBootstrap index, List<String> people, int b, long seed) throws IOException {
        FeiScores.Analysis an = new FeiScores.Analysis();
        an.pose = sel;
        an.r0 = r0;
        FeiDecompose dec = new FeiDecompose(List.of(VARIANTS), sel, r0, index);
        an.decompose = dec;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(scores), 1 << 16),
                StandardCharsets.UTF_8))) {
            an.read(in);
        }
        dec.check();
        Cv cv = readCv(cvFile, sel, r0, index);

        List<String> control = new ArrayList<>();
        List<Res> res = new ArrayList<>();
        int debug = 0;
        for (int v = 0; v < VARIANTS.length; v++) {
            if (dec.cfgs.get(v).size() != FeiProtocol.CONFIGS) throw new IllegalStateException(VARIANTS[v] + ": конфигураций " + dec.cfgs.get(v).size());
            Map<String, Imp> g = cv.imp.get(CV_BASES[v]);
            if (g == null) throw new IllegalStateException("В файле перекрёстной оценки нет " + CV_BASES[v]);
            // Отладка: в файле перекрёстной оценки не все конфигурации — только они (сверка с FeiScores.Stat не делается).
            boolean full = g.size() == FeiProtocol.CONFIGS;
            List<FeiDecompose.Cfg> cs = dec.cfgs.get(v).stream().filter(c -> g.containsKey(c.s + " " + c.j)).toList();
            if (!full) debug = g.size();
            List<Imp> gs = new ArrayList<>();
            for (FeiDecompose.Cfg c : cs) {
                if (c.poseUnknown > 0) throw new IllegalStateException("Нет позы у " + c.poseUnknown + " снимков чужих s=" + c.s + " j=" + c.j);
                Imp imp = g.get(c.s + " " + c.j);
                if (imp == null) throw new IllegalStateException("Перекрёстная оценка: нет конфигурации s=" + c.s + " j=" + c.j);
                Set<String> own = new TreeSet<>();
                for (int p : c.ownP) own.add(people.get(p));
                if (!own.equals(cv.persons.get(c.s + " " + c.j)) || imp.r().length != own.size() * FeiProtocol.NUMBERS.length) {
                    throw new IllegalStateException("Перекрёстная оценка s=" + c.s + " j=" + c.j + ": люди частей ≠ своим галереи или не по 8 снимков");
                }
                gs.add(imp);
            }
            Res r = evaluate(LABELS[v], VARIANTS[v], cs, gs, people.size(), b, seed);
            // Контроль П0: числа = FeiDecompose.counts (точка и все реплики) и FAR контроля = FeiScores.Stat.
            if (!Arrays.deepEquals(r.counts[0], FeiDecompose.counts(cs, null))) throw new IllegalStateException("П0 ≠ FeiDecompose " + VARIANTS[v]);
            FeiScores.Stat st = an.stats.get(VARIANTS[v]);
            int[] ks = {FeiScores.T1, FeiScores.T5};
            for (int k = 0; k < KS && full; k++) {
                long x = 0;
                long n = 0;
                for (int[] f : st.far(st.farCtrl, ks[k])) {
                    x += f[0];
                    n += f[1];
                }
                if (x != r.far[0][k][0] || n != r.far[0][k][1]) throw new IllegalStateException("П0: FAR контроля ≠ FeiScores.Stat " + VARIANTS[v]);
            }
            if (b > 0) {
                int[][] w = FeiBootstrap.weights(people.size(), b, seed);
                for (int q = 0; q < b; q++) {
                    double[][][] sh = FeiDecompose.shares(FeiDecompose.counts(cs, w[q]));
                    for (int k = 0; k < KS; k++) for (int sp = 0; sp < SP; sp++) {
                        if (Double.compare(r.errRep[0][k][sp][q], 100 - sh[k][sp][FeiDecompose.G_OK]) != 0) {
                            throw new IllegalStateException("П0: реплика " + q + " ≠ FeiDecompose " + VARIANTS[v]);
                        }
                    }
                }
            }
            control.add(String.format(Locale.ROOT, "%s: П0 — числа групп (точка%s) = FeiDecompose.counts; ошибка своих и FAR контроля = FeiScores.Stat %s "
                    + "(FAR 1 %%: %d + %d = %d/%d, FAR контроля %d/%d; FAR 5 %%: %d/%d, FAR контроля %d/%d); перекрёстная оценка — по %d "
                    + "конфигурациям, люди частей = своим галереи, по 8 снимков (%d)", LABELS[v], b > 0 ? " и все " + b + " реплик" : "", VARIANTS[v],
                    st.frr(FeiScores.T1), st.misaccepted(FeiScores.T1), st.errors(FeiScores.T1), st.att, (long) r.far[0][0][0], (long) r.far[0][0][1],
                    st.errors(FeiScores.T5), st.att, (long) r.far[0][1][0], (long) r.far[0][1][1], cs.size(), gs.get(0).r().length));
            res.add(r);
        }
        if (debug > 0) control.add(0, "ОТЛАДКА: в файле перекрёстной оценки конфигураций " + debug + " из " + FeiProtocol.CONFIGS
                + " — только они; FeiScores.Stat не сверялся, числа П0 — не как в fei_lbp.txt.");
        return new Loaded(res, dec, cv, control);
    }

    static final String NORM_FILE = "fei_lbp_norm.txt";
    /** Строка шапки файлов оценок и перекрёстной оценки faces-fei с нормировкой перед LBP (FeiMethods.lbpNormText). */
    static final String NORM_HEADER = "# Нормировка освещения перед LBP (faces.lbp.norm): ";

    /** Нормировка перед LBP по шапке файла (строка NORM_HEADER; нет строки — none). */
    static IlluminationNorm normOf(Path file) throws IOException {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null && l.startsWith("#")) {
                if (l.startsWith(NORM_HEADER)) return IlluminationNorm.lbpNorm(l.substring(NORM_HEADER.length()).split(" ")[0]);
            }
        }
        return IlluminationNorm.NONE;
    }

    /**
     * Режим norm (PLAN.md, п. 4.2, «Нормировка освещения перед LBP»): пары «файл оценок, файл перекрёстной оценки» — первая
     * none, далее tt и/или clahe (нормировка — по шапкам файлов, у обоих файлов пары одна); отчёт fei_lbp_norm.txt
     * (LbpNormReport, условия 2 и 3), те же веса бутстрепа у всех пар.
     */
    static void normMain(String[] a, FeiPose sel, double r0, FeiBootstrap index, List<String> people, int b, long seed, long t0) throws IOException {
        if (a.length < 5 || a.length % 2 == 0) {
            throw new IllegalArgumentException("faces-fei-threshold norm: -Dexec.args=\"norm <оценки none> <перекрёстная none> <оценки tt|clahe> "
                    + "<перекрёстная tt|clahe> …\"");
        }
        Map<IlluminationNorm, List<Res>> res = new LinkedHashMap<>();
        StringBuilder head = new StringBuilder();
        for (int i = 1; i < a.length; i += 2) {
            Path scores = Paths.get(a[i]);
            Path cvFile = Paths.get(a[i + 1]);
            IlluminationNorm n = normOf(scores);
            if (normOf(cvFile) != n) throw new IllegalStateException("Нормировка в шапках различается: " + scores + " и " + cvFile);
            if (i == 1 && n != IlluminationNorm.NONE) throw new IllegalStateException("Первая пара — не none: " + n.id);
            if (res.containsKey(n)) throw new IllegalStateException("Нормировка " + n.id + " задана дважды");
            Loaded ld = load(scores, cvFile, sel, r0, index, people, b, seed);
            if (ld.control().get(0).startsWith("ОТЛАДКА")) throw new IllegalStateException("Не все конфигурации в " + cvFile);
            res.put(n, ld.res());
            head.append(n.id).append(": оценки ").append(scores).append(" (").append(sha(scores)).append(");\n  перекрёстная ").append(cvFile)
                    .append(" (").append(sha(cvFile)).append(")\n");
            for (String c : ld.control()) head.append("  контроль: ").append(c).append('\n');
        }
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: нормировка освещения перед LBP — FEI (FeiThreshold norm, faces-fei-threshold)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Протокол FEI (п. 3а), 40 конфигураций; варианты L1 = 5-lbp-mlda-ratio+pose, L1-in = 5-lbp-mlda-in-ratio+pose; выбор модели по\n"
                + "позе r0 = ").append(r0).append("; порог П2 / П3 — как fei_threshold.txt. Нормировка — по шапкам файлов (FeiMethods, faces.lbp.norm).\n");
        t.append("Оговорка: FEI — одна сессия на человека, результаты внутрисессионные и оптимистичны.\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям FEI целиком: B = %d, seed = %d, людей %d; одни веса для всех нормировок, строк и "
                    + "способов.%n", b, seed, people.size()));
        }
        t.append(head);
        LbpNormReport.tables(t, res, b, LbpNormReport.Kind.FEI);
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(out);
        FarMethods.write(out.resolve(NORM_FILE), t);
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "Нормировка перед LBP (FEI): %.0f с; отчёт %s",
                (System.nanoTime() - t0) / 1e9, out.resolve(NORM_FILE)));
    }

    static String sha(Path f) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        try (InputStream in = new DigestInputStream(Files.newInputStream(f), md)) {
            in.transferTo(java.io.OutputStream.nullOutputStream());
        }
        return String.format(Locale.ROOT, "%d байт, SHA-256 %s", Files.size(f), HexFormat.of().formatHex(md.digest()));
    }

    // ---------------------------------------------------------------- отчёт

    static StringBuilder report(List<Res> res, FeiDecompose dec, Cv cv, List<String> control, Path scores, Path cvFile, double r0, int b,
                                long seed, int persons) throws IOException {
        StringBuilder t = new StringBuilder();
        String[] farName = {"FAR 1 %", "FAR 5 %"};
        t.append("PLAN.md п. 4.2: калибровка порога — способы задания порога (FeiThreshold, faces-fei-threshold)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл оценок: ").append(scores).append(" (").append(sha(scores)).append(")\n");
        t.append("Перекрёстная оценка: ").append(cvFile).append(" (").append(sha(cvFile)).append("; строк ").append(cv.lines).append(")\n");
        t.append("Варианты: L1 = 5-lbp-mlda-ratio+pose, L1-in = 5-lbp-mlda-in-ratio+pose (как в fei_lbp.txt и fei_lbp_mask.txt); выбор модели по\n"
                + "позе r0 = ").append(r0).append(", оценка d₁/d₂ (FeiScores.ratio). Оценки своих и контрольных чужих — из файла оценок, меняется только порог.\n");
        t.append("Способы (Нейман – Пирсон по снимкам, номинал FAR 1 % и 5 %; порог — в каждой конфигурации):\n"
                + "  П0 — по снимкам 40 пороговых чужих, точечно (как fei_scores.txt);\n"
                + "  П1 — по ним же с запасом: наибольший θ (проход по числу принятых снимков до первого нарушения), при котором верхняя\n"
                + "       односторонняя 95 % граница FAR не выше номинала; граница — Клоппер – Пирсон по эффективному объёму n_eff = n/deff,\n"
                + "       x_eff = p̂·n_eff, deff = max(1, v/(p̂(1 − p̂)/n)), v = k/(k − 1)·Σ(xᵢ − p̂nᵢ)²/n² — кластерная дисперсия по людям (xᵢ, nᵢ —\n"
                + "       принятые и все снимки человека, k — людей), при p̂ = 0 deff = 1; если недопустим и ноль принятых — порог ниже\n"
                + "       наименьшей оценки чужих (случаи считаются);\n"
                + "  П2 — перекрёстно по галерее: 100 своих разбиения — 5 частей по 20 (перестановка, seed faces.seed + 8000 + s); для части\n"
                + "       модель (MLDA, X и X+mirror) обучена без неё на остальных 80 своих и той же когорте 20; «чужие» — все 8 снимков людей\n"
                + "       части, модель по позе, d₁/d₂ по 80; порог — по объединённым «чужим» 5 частей (800 снимков, 100 людей), точечно;\n"
                + "       пороговые 40 не используются;\n"
                + "  П3 — П2 с запасом, как П1.\n");
        t.append("Оговорка: у «чужих» П2/П3 d₁/d₂ считается по галерее из 80 своих, у проб — по 100 (отдельно не исправляется; судим по\n"
                + "FAR контроля).\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям FEI целиком: B = %d, seed = %d, людей %d, одни веса для всех строк и способов (как "
                    + "fei_bootstrap.txt);\nпорог — заново своим способом в каждой реплике и конфигурации; у П2/П3 «чужие» — люди галереи с их весами, "
                    + "без\nпереобучения. [2,5; 97,5] — процентили; «*» — интервал разности не содержит 0.%n", b, seed, persons));
        }
        t.append("Критерий выбора (решение Хозяина 04.10.2026, зашит в код; исправлен 04.10.2026 — ошибка постановки: допуск «не выше\n"
                + "номинала» отсекал несмещённый способ): для каждого варианта отдельно допустимы способы с фактическим FAR контроля (точка)\n"
                + "не выше номинала + 10 % от него и при 1 %, и при 5 % (1,1 % и 5,5 %); из них — наименьшая ошибка своих при FAR 1 %;\n"
                + "при незначимой разности — более простой (порядок П0, П1, П2, П3). Нет допустимых — доложить, ничего не подбирать.\n");

        t.append("\n=== Контроль ===\n");
        for (String c : control) t.append(c).append('\n');
        t.append("Правило d₁/d₂ у «чужих» П2 (случаи 1–6): ");
        for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(cv.cases[k]);
        t.append('\n');

        t.append("\n=== Объём «чужих», по которым задан порог, и позы ===\n");
        t.append("способ | снимков всего (40 конфигураций) | снимков на конфигурацию мин.–макс. | людей на конфигурацию мин.–макс.\n");
        Res r0r = res.get(0);
        for (int m = 0; m < M; m++) {
            int[] v = r0r.volume[m];
            t.append(String.join(" | ", METHODS[m], Integer.toString(v[0]), v[1] + "–" + v[2], v[3] + "–" + v[4])).append('\n');
        }
        long ownN = 0;
        long ownF = 0;
        long thrN = 0;
        long thrF = 0;
        long ctrlN = 0;
        long ctrlF = 0;
        for (FeiDecompose.Cfg c : dec.cfgs.get(0)) {
            ownN += c.ownR.length;
            for (boolean f : c.poseFrontal) if (f) ownF++;
            thrN += c.thrR.length;
            thrF += c.thrFrontal;
            ctrlN += c.ctrlR.length;
            ctrlF += c.ctrlFrontal;
        }
        long galN = 0;
        long galF = 0;
        for (Imp g : cv.imp.get(CV_BASES[0]).values()) {
            galN += g.r().length;
            galF += g.frontal();
        }
        t.append(String.format(Locale.ROOT, "Доля анфаса (|r| ≤ r0): пробы своих %s; пороговые чужие (П0, П1) %s; «чужие» галереи (П2, П3) %s; контрольные "
                + "чужие %s.%n", share(ownF, ownN), share(thrF, thrN), share(galF, galN), share(ctrlF, ctrlN)));

        t.append("\n=== Ошибка своих, FAR контроля, допуск ===\n");
        t.append("вариант | способ | ошибка своих FAR 1 % | FAR 5 % | анфас FAR 1 % | поворот FAR 1 % | FAR контроля 1 %: точка ↑95; КП | 5 % | "
                + "θ мед. 1 % / 5 % | «ноль недопустим» 1 % / 5 %: конфигураций (реплик×конфигураций) | допуск\n");
        for (Res r : res) {
            for (int m = 0; m < M; m++) {
                List<String> c = new ArrayList<>(List.of(r.label, METHODS[m]));
                for (int[] ks : new int[][] {{0, 0}, {1, 0}, {0, 1}, {0, 2}}) c.add(errCell(r, m, ks[0], ks[1], b));
                for (int k = 0; k < KS; k++) c.add(farCell(r, m, k, b));
                c.add(String.format(Locale.ROOT, "%.6f / %.6f", FeiMethods.median(r.theta[m][0]), FeiMethods.median(r.theta[m][1])));
                c.add(r.zeroFail[m][0] + " / " + r.zeroFail[m][1] + (b > 0 ? " (" + r.zeroFailRep[m][0] + " / " + r.zeroFailRep[m][1] + ")" : ""));
                c.add(admissible(r, m) ? "да" : "нет");
                t.append(String.join(" | ", c)).append('\n');
            }
        }

        t.append("\n=== Разложение ошибки своих ===\n");
        for (int k = 0; k < KS; k++) {
            t.append("-- ").append(farName[k]).append('\n');
            t.append("вариант | способ | деление | (i) | (ii) | (iii) | (iv) | (v-а)\n");
            for (Res r : res) {
                for (int m = 0; m < M; m++) {
                    for (int sp = 0; sp < SP; sp++) {
                        double[] cnt = r.counts[m][k][sp];
                        double n = Arrays.stream(cnt).sum();
                        List<String> c = new ArrayList<>(List.of(r.label, METHODS[m], FeiDecompose.SPLIT_NAMES[sp]));
                        for (int g = 0; g <= FeiDecompose.G_NO_SECOND; g++) c.add(FeiNorm.cell(cnt[g], n, b > 0 ? r.shareRep[m][k][sp][g] : null));
                        t.append(String.join(" | ", c)).append('\n');
                    }
                }
            }
        }

        if (b > 0) {
            t.append("\n=== Парные разности со способом П0, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("вариант | пара | ошибка FAR 1 % всего | анфас | поворот | FAR 5 % всего | анфас | поворот | FAR контроля 1 % | FAR контроля 5 %\n");
            for (Res r : res) {
                for (int m = 1; m < M; m++) {
                    List<String> c = new ArrayList<>(List.of(r.label, METHODS[m] + " − П0"));
                    for (int k = 0; k < KS; k++) {
                        for (int sp = 0; sp < SP; sp++) {
                            c.add(FeiBootstrap.diff(err(r, m, k, sp) - err(r, 0, k, sp), FeiNorm.diff(r.errRep[m][k][sp], r.errRep[0][k][sp]), 1));
                        }
                    }
                    for (int k = 0; k < KS; k++) {
                        c.add(FeiBootstrap.diff(farPct(r, m, k) - farPct(r, 0, k), FeiNorm.diff(r.farRep[m][k], r.farRep[0][k]), 2));
                    }
                    t.append(String.join(" | ", c)).append('\n');
                }
            }
        }

        t.append("\n=== Выбор способа (по каждому варианту отдельно) ===\n");
        for (Res r : res) {
            boolean[] adm = new boolean[M];
            double[] e = new double[M];
            double[][] reps = b > 0 ? new double[M][] : null;
            t.append("Допуск ").append(r.label).append(" (FAR контроля не выше 1,1 % / 5,5 %): ");
            for (int m = 0; m < M; m++) {
                adm[m] = admissible(r, m);
                e[m] = err(r, m, 0, 0);
                if (b > 0) reps[m] = r.errRep[m][0][0];
                t.append(m > 0 ? "; " : "").append(METHODS[m]).append(String.format(Locale.ROOT, " — FAR контроля %.2f %% / %.2f %% — %s", farPct(r, m, 0),
                        farPct(r, m, 1), adm[m] ? "да" : "нет"));
            }
            t.append('\n');
            int ch = choose(e, reps, adm);
            if (ch < 0) {
                t.append("  ").append(r.label).append(": ни один способ не укладывается в допуск FAR контроля — выбора нет (доложить, ничего не подбирать).\n");
            } else {
                t.append(String.format(Locale.ROOT, "  %s: выбран %s (ошибка своих при FAR 1 %% — %.1f %%).%n", r.label, METHODS[ch], e[ch]));
            }
        }
        return t;
    }

    static String share(long x, long n) {
        return String.format(Locale.ROOT, "%d/%d = %.1f %%", x, n, 100.0 * x / n);
    }

    static double farPct(Res r, int m, int k) {
        return 100.0 * r.far[m][k][0] / r.far[m][k][1];
    }

    static String errCell(Res r, int m, int k, int sp, int b) {
        double[] cnt = r.counts[m][k][sp];
        double n = Arrays.stream(cnt).sum();
        return FeiNorm.cell(n - cnt[FeiDecompose.G_OK], n, b > 0 ? r.errRep[m][k][sp] : null);
    }

    static String farCell(Res r, int m, int k, int b) {
        String s = String.format(Locale.ROOT, "%.2f %%", farPct(r, m, k));
        if (b > 0) s += String.format(Locale.ROOT, " ↑%.2f %%", FeiBootstrap.quantile(FeiBootstrap.sorted(r.farRep[m][k]), 0.95)) + FeiBootstrap.note(r.farRep[m][k]);
        int x = (int) r.far[m][k][0];
        int n = (int) r.far[m][k][1];
        return s + String.format(Locale.ROOT, "; КП %d/%d ↑%.2f %%", x, n, 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95));
    }
}
