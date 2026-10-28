package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Пачка 1 – оценка и порог (постановка Жорика №8 от 27.10.2026, план принят с поправками №9, подтверждён Хозяином). Режимы
 * pool-vec, feret-vec, gt-vec – один прогон принятой конфигурации (база (П2), косинус, обучение как у строки базы pool-abl2 /
 * feret-abl2 / gt-abl2) с полными векторами 1 − cos каждой пробы до центров всех людей галереи и до центров людей двух когорт FEI –
 * в обучении W (когорта MLDA, 20) и вне W (20 других «полных», seed faces.seed + 10500), плюс расстояния «центр человека галереи →
 * центры когорт» (статистика Z-нормировки); сверка – строки d₁/d₂, восстановленные по векторам, побайтно = строка базы файла
 * оценок abl2. Режимы pool-norm, feret-norm, gt-norm – по векторам, без обучения: нормировка по когорте в логарифмах (S-norm,
 * AS-norm K = 20, 10, 5, обе когорты), слияние кандидатов (w = 0,25; 0,5; 0,75), режим 1:1 с реалистичным FAR.
 *
 * @author ssv
 */
final class XBaseVec {

    /** Сдвиг seed когорты вне W. */
    static final long OUT_SEED_OFFSET = 10500;
    static final double BETA = 1;
    /**
     * Отображение оценки s в отношение строки: отношение d₁′ / 1e6 обязано лежать в [ρ_min, 1] (правило вырожденных случаев: меньше ρ_min
     * – ничья на ρ_min, больше 1 – исключение), а exp(s) / 1e6 при нормированных ℓ выходит за эти пределы (|s| бывает десятки). Поэтому
     * r = ρ_min^σ(−s/τ) – строго монотонно по s (все меры зависят только от порядка оценок – те же, что дало бы exp(s)); различимо при
     * |s| / τ < S_RES; τ = 25 (|s| < 900; в пробном режиме s доходило до −700). Число проб вне различимой области и пределы s – в отчёт.
     */
    static final double TAU = 25;
    static final double S_RES = 36;
    /** Прежнее окно (для отчёта): |s| ≤ ln 1e6. */
    static final double S_CLAMP = 13.8;
    /** ρ_min правила вырожденных случаев. */
    static final double RHO_MIN = 1e-12;
    static final int P2 = XBaseNap.P2;

    private XBaseVec() {
    }

    /** Пробный режим FACES_VEC_SMOKE: 0 – нет; 1 – малые данные; 2 – первая группа в полном размере – замер. */
    static int smoke() {
        String v = System.getenv("FACES_VEC_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    /** Когорта вне W: 20 «полных» людей FEI, не входящих в когорту MLDA (перестановка seed + OUT_SEED_OFFSET), по возрастанию. */
    static List<String> cohortOut(Map<String, Map<Integer, String>> feiKeys, long seed, List<String> cohortIn) {
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new HashSet<>(m.keySet())));
        List<String> full = new ArrayList<>(FeiProtocol.fullPersons(included));
        full.removeAll(cohortIn);
        Collections.shuffle(full, new Random(seed + OUT_SEED_OFFSET));
        List<String> out = new ArrayList<>(full.subList(0, XBaseEval.COHORT));
        Collections.sort(out);
        return out;
    }

    // ---------------------------------------------------------------- векторы одной модели

    /** Строка пробы: расстояния до галереи (+∞ – центра нет), до центров когорт в W и вне W. */
    record Row(String role, String person, int image, String key, int trueIdx, double[] dg, double[] dci, double[] dco) {}

    /** Модель: имя задачи (main, part q), метод, галерея, строки, Z – расстояния центров галереи до центров когорт [g][in…, out…]. */
    record Model(String task, String method, int s, List<String> gallery, List<Row> rows, double[][] z) {}

    static double[] center(LdaModel lm, XBaseEval x, List<String> keys, boolean mirror) {
        int d = x.vec.get(V.L1).get(keys.get(0)).length;
        double[] m = new double[d];
        int n = 0;
        for (String k : keys) {
            double[] v = x.vec.get(V.L1).get(k);
            for (int i = 0; i < d; i++) m[i] += v[i];
            n++;
            if (mirror) {
                double[] w = x.mir.get(V.L1).get(k);
                for (int i = 0; i < d; i++) m[i] += w[i];
                n++;
            }
        }
        for (int i = 0; i < d; i++) m[i] /= n;
        return lm.embed(m);
    }

    static double cos(double[] a, double[] b) {
        return XBaseRule.cos(a, b, a.length, null);
    }

    /** Строка d₁/d₂ по вектору расстояний до галереи (как XBaseReg.lines). */
    static String line(String method, int s, List<String> gallery, Row r) {
        return FeiScores.line(method, s, 0, r.role(), r.person(), r.image(), r.trueIdx(), gallery, r.dg(), false);
    }

    /** Модель (как задача XBaseLbp.job): обучение, векторы, сверка строк с XBaseReg.lines. */
    static Model model(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> cohort,
                       List<List<String>> cin, List<List<String>> cout, List<Probe> probes, String task) {
        XBaseRule.Fit f = XBaseRule.fit(x, mirror, gallery, cohort, null);
        double[][] c = f.lm.centers();
        double[][] ci = new double[cin.size()][];
        for (int i = 0; i < cin.size(); i++) ci[i] = center(f.lm, x, cin.get(i), mirror);
        double[][] co = new double[cout.size()][];
        for (int i = 0; i < cout.size(); i++) co[i] = center(f.lm, x, cout.get(i), mirror);
        List<Row> rows = new ArrayList<>();
        StringBuilder mine = new StringBuilder();
        for (Probe p : probes) {
            double[] y = f.lm.embed(x.vec.get(V.L1).get(p.key()));
            double[] dg = new double[c.length];
            for (int g = 0; g < c.length; g++) dg[g] = c[g] == null ? Double.POSITIVE_INFINITY : cos(y, c[g]);
            double[] dci = new double[ci.length];
            for (int i = 0; i < ci.length; i++) dci[i] = cos(y, ci[i]);
            double[] dco = new double[co.length];
            for (int i = 0; i < co.length; i++) dco[i] = cos(y, co[i]);
            Row r = new Row(p.role(), p.person(), p.image(), p.key(), p.trueIdx(), dg, dci, dco);
            rows.add(r);
            mine.append(line(method, s, own, r));
        }
        String ref = XBaseReg.lines(x, method, s, f.lm, own, probes)[1];
        if (!ref.equals(mine.toString())) throw new IllegalStateException("XBaseVec: строки по векторам ≠ XBaseReg.lines (" + task + ", " + method + ")");
        double[][] z = new double[c.length][];
        for (int g = 0; g < c.length; g++) {
            if (c[g] == null) continue;
            z[g] = new double[ci.length + co.length];
            for (int i = 0; i < ci.length; i++) z[g][i] = cos(c[g], ci[i]);
            for (int i = 0; i < co.length; i++) z[g][ci.length + i] = cos(c[g], co[i]);
        }
        return new Model(task, method, s, own, rows, z);
    }

    /** Задачи блока – как XBaseLbp.job (тот же порядок моделей и проб). */
    static List<Callable<Model>> job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts, int s,
                                     List<List<Probe>> extra, List<List<String>> cin, List<List<String>> cout) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<Model>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> model(x, m, method, s, own, fullGallery, cohort, cin, cout, main, "main"));
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                int qq = q;
                tasks.add(() -> model(x, m, method, s, g, gk, cohort, cin, cout, pr, "part " + qq));
            }
        }
        return tasks;
    }

    static List<List<Model>> run(List<List<Callable<Model>>> jobs, int threads) {
        List<Callable<Model>> flat = new ArrayList<>();
        for (List<Callable<Model>> j : jobs) flat.addAll(j);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<Model>> fs = ex.invokeAll(flat);
            List<List<Model>> out = new ArrayList<>();
            int i = 0;
            for (List<Callable<Model>> j : jobs) {
                List<Model> l = new ArrayList<>();
                for (int k = 0; k < j.size(); k++) l.add(fs.get(i++).get());
                out.add(l);
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseVec: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    /** Строки d₁/d₂ блока (все модели подряд, как файл оценок). */
    static String lines(List<Model> ms) {
        StringBuilder b = new StringBuilder();
        for (Model m : ms) for (Row r : m.rows()) b.append(line(m.method(), m.s(), m.gallery(), r));
        return b.toString();
    }

    // ---------------------------------------------------------------- файл векторов

    static String num(double v) {
        return Double.toString(v);
    }

    /** Запись блока: «#B имя», на модель – «#M задача метод s», «#G галерея», «#Z g значения», строки «R …». */
    static void write(Writer w, String block, List<Model> ms) throws IOException {
        w.write("#B\t" + block + "\n");
        for (Model m : ms) {
            w.write("#M\t" + m.task() + "\t" + m.method() + "\t" + m.s() + "\n");
            w.write("#G\t" + String.join("\t", m.gallery()) + "\n");
            for (int g = 0; g < m.z().length; g++) {
                if (m.z()[g] == null) continue;
                StringBuilder b = new StringBuilder("#Z\t").append(g);
                for (double v : m.z()[g]) b.append('\t').append(num(v));
                w.write(b.append('\n').toString());
            }
            for (Row r : m.rows()) {
                StringBuilder b = new StringBuilder("R\t").append(r.role()).append('\t').append(r.person()).append('\t').append(r.image()).append('\t')
                        .append(r.key()).append('\t').append(r.trueIdx()).append('\t').append(r.dg().length).append('\t').append(r.dci().length).append('\t')
                        .append(r.dco().length);
                for (double v : r.dg()) b.append('\t').append(num(v));
                for (double v : r.dci()) b.append('\t').append(num(v));
                for (double v : r.dco()) b.append('\t').append(num(v));
                w.write(b.append('\n').toString());
            }
        }
    }

    static Writer writer(Path p, String head) throws IOException {
        Path tmp = p.resolveSibling(p.getFileName() + ".tmp");
        Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16), StandardCharsets.UTF_8);
        w.write(head);
        return w;
    }

    static void finish(Writer w, Path p) throws IOException {
        w.close();
        Files.move(p.resolveSibling(p.getFileName() + ".tmp"), p, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    /** Чтение: блок → модели (порядок записи: #M, #G, #Z, R). */
    static Map<String, List<Model>> read(Path p) throws IOException {
        Map<String, List<Model>> out = new LinkedHashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(p), 1 << 16), StandardCharsets.UTF_8))) {
            List<Model> cur = null;
            String[] head = null;
            List<Row> rows = null;
            double[][] z = null;
            for (String l = in.readLine(); l != null; l = in.readLine()) {
                if (l.startsWith("#B\t")) {
                    cur = new ArrayList<>();
                    out.put(l.substring(3), cur);
                } else if (l.startsWith("#M\t")) {
                    head = l.split("\t");
                } else if (l.startsWith("#G")) {
                    List<String> gal = l.length() <= 3 ? List.of() : List.of(l.substring(3).split("\t"));
                    rows = new ArrayList<>();
                    z = new double[gal.size()][];
                    cur.add(new Model(head[1], head[2], Integer.parseInt(head[3]), gal, rows, z));
                } else if (l.startsWith("#Z\t")) {
                    String[] c = l.split("\t");
                    double[] v = new double[c.length - 2];
                    for (int i = 2; i < c.length; i++) v[i - 2] = Double.parseDouble(c[i]);
                    z[Integer.parseInt(c[1])] = v;
                } else if (l.startsWith("R\t")) {
                    String[] c = l.split("\t");
                    int ng = Integer.parseInt(c[6]);
                    int ni = Integer.parseInt(c[7]);
                    int no = Integer.parseInt(c[8]);
                    double[] dg = new double[ng];
                    double[] di = new double[ni];
                    double[] dn = new double[no];
                    int k = 9;
                    for (int i = 0; i < ng; i++) dg[i] = Double.parseDouble(c[k++]);
                    for (int i = 0; i < ni; i++) di[i] = Double.parseDouble(c[k++]);
                    for (int i = 0; i < no; i++) dn[i] = Double.parseDouble(c[k++]);
                    rows.add(new Row(c[1], c[2], Integer.parseInt(c[3]), c[4], Integer.parseInt(c[5]), dg, di, dn));
                }
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- нормировка, слияние, 1:1

    /** ℓ = log(1 − cos); 1 − cos ≤ 0 (в пределах округления) – ρ_min. */
    static double ell(double d) {
        if (Double.isInfinite(d)) return d;
        if (d < -1e-12 || Double.isNaN(d)) throw new IllegalStateException("XBaseVec: недопустимое расстояние " + d);
        return Math.log(Math.max(d, RHO_MIN));
    }

    /** Среднее и разброс (несмещённый) значений ℓ по K ближайшим (наименьшим) из v; K ≥ длины – по всем. */
    static double[] stats(double[] v, int k) {
        double[] l = new double[v.length];
        for (int i = 0; i < v.length; i++) l[i] = ell(v[i]);
        if (k < l.length) {
            Arrays.sort(l);
            l = Arrays.copyOf(l, k);
        }
        double m = 0;
        for (double a : l) m += a;
        m /= l.length;
        double s = 0;
        for (double a : l) s += (a - m) * (a - m);
        s = Math.sqrt(s / (l.length - 1));
        if (!(s > 0)) throw new IllegalStateException("XBaseVec: нулевой разброс когорты");
        return new double[] {m, s};
    }

    /** Вариант нормировки: когорта (0 – в W, 1 – вне W), K, as – адаптивная (μ, σ по K ближайшим). */
    record Norm(int cohort, int k, boolean as) {
        String name() {
            return (as ? "AS-norm K = " + k : "S-norm") + (cohort == 0 ? ", когорта в W" : ", когорта вне W");
        }
    }

    /**
     * Варианты по простоте: S-norm < AS-norm K = 20 < 10 < 5; при равенстве когорт – вне W (раньше в списке). AS-norm K = 20 при когорте
     * из 20 – те же значения, что S-norm (μ, σ по всем; отличие – только порядок суммирования).
     */
    static final List<Norm> NORMS = List.of(new Norm(1, 20, false), new Norm(0, 20, false), new Norm(1, 20, true), new Norm(0, 20, true),
            new Norm(1, 10, true), new Norm(0, 10, true), new Norm(1, 5, true), new Norm(0, 5, true));

    /**
     * Строка модели при нормировке n (null – без нормировки: d₁′ = d₁^(1+β) · d₂^(−β) по сырым расстояниям, побайтно XBaseAbl3.beta):
     * ℓₙ(g) = ((ℓ − μ_Z(g)) / σ_Z(g) + (ℓ − μ_T) / σ_T) / 2, лучший и второй – по ℓₙ, s = (1 + β) ℓ₁ₙ − β ℓ₂ₙ, d₁′ = exp(s), второе –
     * XBaseAbl.ABS_C (s ограничено окном S_CLAMP, см. там).
     */
    static String normLine(Model m, Row r, Norm n, int[] clamped) {
        if (n == null) return XBaseAbl3.beta(line(m.method(), m.s(), m.gallery(), r), BETA);
        int ng = r.dg().length;
        double[] cv = n.cohort() == 0 ? r.dci() : r.dco();
        int k = n.as() ? n.k() : Integer.MAX_VALUE;
        double[] t = stats(cv, k);
        double[] ln = new double[ng];
        for (int g = 0; g < ng; g++) {
            if (Double.isInfinite(r.dg()[g]) || m.z()[g] == null) {
                ln[g] = Double.POSITIVE_INFINITY;
                continue;
            }
            double[] zc = n.cohort() == 0 ? Arrays.copyOfRange(m.z()[g], 0, r.dci().length) : Arrays.copyOfRange(m.z()[g], r.dci().length, m.z()[g].length);
            double[] zs = stats(zc, k);
            double l = ell(r.dg()[g]);
            ln[g] = ((l - zs[0]) / zs[1] + (l - t[0]) / t[1]) / 2;
        }
        return decide(m, r, ln, clamped);
    }

    /** Лучший и второй по значениям ln (меньше – ближе), d₁′ = exp((1 + β) ℓ₁ − β ℓ₂), строка с постоянным вторым. */
    static String decide(Model m, Row r, double[] ln, int[] clamped) {
        int ng = ln.length;
        int b = -1;
        int c = -1;
        for (int g = 0; g < ng; g++) {
            if (b < 0 || ln[g] < ln[b]) {
                c = b;
                b = g;
            } else if (c < 0 || ln[g] < ln[c]) {
                c = g;
            }
        }
        double[] v = new double[ng];
        for (int g = 0; g < ng; g++) v[g] = XBaseAbl.ABS_C * 10;
        if (c < 0 || Double.isInfinite(ln[c]) || Double.isInfinite(ln[b])) {
            // Второго нет – отказ, как d₂ = +∞.
            double[] w = r.dg().clone();
            return FeiScores.line(m.method(), m.s(), 0, r.role(), r.person(), r.image(), r.trueIdx(), m.gallery(), w, false);
        }
        v[b] = XBaseAbl.ABS_C * squash((1 + BETA) * ln[b] - BETA * ln[c], clamped);
        v[c] = XBaseAbl.ABS_C;
        // Истинный (если не лучший и не второй) – 1e7: столбец «оценка истинного» в решении не участвует.
        return FeiScores.line(m.method(), m.s(), 0, r.role(), r.person(), r.image(), r.trueIdx(), m.gallery(), v, false);
    }

    /** Пределы s и счётчики: [0] – вне ±S_CLAMP, [1] – вне различимой области (|s| / τ ≥ S_RES); min и max s – в sRange. */
    static final double[] S_RANGE = {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};

    /** r = ρ_min^σ(−s/τ) ∈ (ρ_min, 1), σ(y) = 1 / (1 + e^(−y)), строго возрастает по s. */
    static double squash(double s, int[] counts) {
        if (counts != null) {
            if (Math.abs(s) > S_CLAMP) counts[0]++;
            if (Math.abs(s) / TAU >= S_RES) counts[1]++;
            synchronized (S_RANGE) {
                S_RANGE[0] = Math.min(S_RANGE[0], s);
                S_RANGE[1] = Math.max(S_RANGE[1], s);
            }
        }
        double y = -s / TAU;
        double sig = y >= 0 ? 1 / (1 + Math.exp(-y)) : Math.exp(y) / (1 + Math.exp(y));
        return Math.exp(Math.log(RHO_MIN) * sig);
    }

    static String normLines(List<Model> ms, Norm n, int[] clamped) {
        StringBuilder b = new StringBuilder();
        for (Model m : ms) for (Row r : m.rows()) b.append(normLine(m, r, n, clamped));
        return b.toString();
    }

    /**
     * Статистика слияния признака: μ и σ ℓ по всем расстояниям «чужой – человек галереи» у «чужих» порога (Т2) – роли cv q и od q моделей
     * частей; pool (моделей частей нет) – контрольные чужие основной модели.
     */
    static double[] fuseStats(List<Model> ms, boolean pool) {
        double s = 0;
        double ss = 0;
        long n = 0;
        for (Model m : ms) for (Row r : m.rows()) {
            boolean imp = r.trueIdx() < 0 && (pool ? r.role().equals(FeiScores.CTRL) : r.role().startsWith(FeiMethods.CV_ROLE) || r.role().startsWith("od"));
            if (!imp) continue;
            for (double d : r.dg()) {
                if (Double.isInfinite(d)) continue;
                double l = ell(d);
                s += l;
                ss += l * l;
                n++;
            }
        }
        if (n < 2) throw new IllegalStateException("XBaseVec: мало «чужих» для слияния");
        double m = s / n;
        return new double[] {m, Math.sqrt((ss - n * m * m) / (n - 1))};
    }

    /**
     * Слияние: модель L1-tt-xs-F – по позе пробы (|r| ≤ r0 – X), L1-xs-F – X+mirror; ℓ(g) = w · (ℓ_tt − μ_tt) / σ_tt + (1 − w) · (ℓ_xs −
     * μ_xs) / σ_xs; строка пишется под обоими методами (выбор модели при разборе не влияет). Модели tt и xs – одни задачи блока.
     */
    static String fuseLines(List<Model> tt, List<Model> xs, double w, double[] st, double[] sx, Map<String, Double> rOf, double r0, int[] clamped) {
        StringBuilder b = new StringBuilder();
        Map<String, Model> byTask = new LinkedHashMap<>();
        for (Model m : tt) byTask.putIfAbsent(m.task(), m);
        for (String task : byTask.keySet()) {
            Model tX = find(tt, task, false);
            Model tM = find(tt, task, true);
            Model xM = find(xs, task, true);
            if (!tX.gallery().equals(xM.gallery()) || tX.rows().size() != xM.rows().size()) throw new IllegalStateException("XBaseVec: слияние – разные задачи");
            for (int i = 0; i < tX.rows().size(); i++) {
                Row rx = xM.rows().get(i);
                Row r0r = tX.rows().get(i);
                if (!rx.key().equals(r0r.key()) || !rx.role().equals(r0r.role())) throw new IllegalStateException("XBaseVec: слияние – разные пробы");
                Row rt = rOf.get(r0r.key()) <= r0 ? r0r : tM.rows().get(i);
                int ng = rt.dg().length;
                double[] ln = new double[ng];
                for (int g = 0; g < ng; g++) {
                    if (Double.isInfinite(rt.dg()[g]) || Double.isInfinite(rx.dg()[g])) {
                        ln[g] = Double.POSITIVE_INFINITY;
                        continue;
                    }
                    ln[g] = w * (ell(rt.dg()[g]) - st[0]) / st[1] + (1 - w) * (ell(rx.dg()[g]) - sx[0]) / sx[1];
                }
                for (Model m : List.of(tX, tM)) b.append(decide(m, rt, ln, clamped));
            }
        }
        return b.toString();
    }

    static Model find(List<Model> ms, String task, boolean mirror) {
        for (Model m : ms) if (m.task().equals(task) && m.method().endsWith(FeiMethods.MIRROR) == mirror) return m;
        throw new IllegalStateException("XBaseVec: нет модели " + task + (mirror ? " X+mirror" : " X"));
    }

    /**
     * 1:1, случайно заявленный: у своего – расстояние до своего шаблона; у чужой попытки – до человека галереи, выбранного случайно
     * (Random(seed), по порядку проб); второе – постоянная.
     */
    static String oneRandom(List<Model> ms, long seed) {
        StringBuilder b = new StringBuilder();
        for (Model m : ms) {
            Random rnd = new Random(seed + m.task().hashCode() * 31L + m.method().hashCode());
            for (Row r : m.rows()) {
                String l = line(m.method(), m.s(), m.gallery(), r).stripTrailing();
                String[] c = l.split("\t", -1);
                if (r.trueIdx() >= 0) {
                    c[7] = c[6];
                    c[8] = c[11];
                    c[10] = Double.toString(XBaseAbl.ABS_C);
                } else {
                    List<Integer> ok = new ArrayList<>();
                    for (int g = 0; g < r.dg().length; g++) if (!Double.isInfinite(r.dg()[g])) ok.add(g);
                    int g = ok.get(rnd.nextInt(ok.size()));
                    c[7] = m.gallery().get(g);
                    c[8] = FeiScores.num(r.dg()[g]);
                    c[10] = Double.toString(XBaseAbl.ABS_C);
                }
                b.append(String.join("\t", c)).append('\n');
            }
        }
        return b.toString();
    }

    /**
     * 1:1 по всем парам, только основная модель (выбор модели пробы – chooseMirror): ошибка своих при FAR 1, 2, 5, 10 % по парам
     * «контрольный чужой – человек галереи» (порог – наибольший с долей принятых пар ≤ FAR) и EER; [ошибки…, EER, пар].
     */
    static double[] pairs(List<Model> ms, java.util.function.Predicate<Row> mirrorOf) {
        Model x = find(ms, "main", false);
        Model mm = find(ms, "main", true);
        List<Double> own = new ArrayList<>();
        List<Double> imp = new ArrayList<>();
        for (int i = 0; i < x.rows().size(); i++) {
            Row r = mirrorOf.test(x.rows().get(i)) ? mm.rows().get(i) : x.rows().get(i);
            if (r.role().equals(FeiScores.OWN)) own.add(r.dg()[r.trueIdx()]);
            else if (r.role().equals(FeiScores.CTRL)) for (double d : r.dg()) if (!Double.isInfinite(d)) imp.add(d);
        }
        double[] o = own.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double[] im = imp.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double[] fars = {0.01, 0.02, 0.05, 0.10};
        double[] out = new double[fars.length + 2];
        for (int f = 0; f < fars.length; f++) {
            // Наибольший порог с долей принятых пар ≤ FAR: принимаются d < im[k] (при равенствах – строже).
            int k = (int) Math.floor(fars[f] * im.length);
            double th = k < im.length ? Math.nextDown(im[k]) : Double.POSITIVE_INFINITY;
            int rej = 0;
            for (double d : o) if (!(d <= th)) rej++;
            out[f] = 100.0 * rej / o.length;
        }
        double best = Double.POSITIVE_INFINITY;
        int j = 0;
        for (double th : o) {
            while (j < im.length && im[j] <= th) j++;
            double far = (double) j / im.length;
            int rej = 0;
            for (double d : o) if (d > th) rej++;
            double frr = (double) rej / o.length;
            best = Math.min(best, Math.max(far, frr));
        }
        out[fars.length] = 100 * best;
        out[fars.length + 1] = im.length;
        return out;
    }

    // ---------------------------------------------------------------- постановки (как XBaseAbl2)

    /**
     * Блок: имя (файл векторов), заголовок блока в файле оценок abl2 без строки, Gt, галерея, пробы своих (null – пул), части (null –
     * пул), пробы основной модели, пробы (Т1), s, группа отчёта.
     */
    record Blk(String name, String abl2, XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts,
               List<Probe> probes, List<List<Probe>> extra, int s, String group) {}

    /** Общее: кадры, когорты, векторы признака. */
    static final class Setup {
        List<String> cohortIn;
        List<String> cohortOut;
        List<List<String>> cin = new ArrayList<>();
        List<List<String>> cout = new ArrayList<>();
        Map<String, Mat> fr;
        XBaseEval.Gt tpl;
        int persons;
        List<String> groups = new ArrayList<>();
        Map<String, Integer> setSize = new LinkedHashMap<>();
        List<Blk> blocks = new ArrayList<>();
        Path abl2Scores;
        Path out;
        String base;
        /** Снимки и детекции FERET (кадры k = 2 пачки 2). */
        XBaseA0.FeretSetup su;
    }

    static void cohorts(Setup u, Map<String, Map<Integer, String>> feiKeys, long seed, FarExport.Loaded export) throws IOException {
        u.cohortIn = XBaseEval.cohortPersons(feiKeys, seed);
        u.cohortOut = cohortOut(feiKeys, seed, u.cohortIn);
        Set<String> keys = new HashSet<>();
        for (String p : u.cohortIn) u.cin.add(new ArrayList<>(feiKeys.get(p).values()));
        for (String p : u.cohortOut) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            u.cout.add(ks);
            keys.addAll(ks);
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        u.fr.putAll(XBaseEval.frames(export, fei, keys));
    }

    static Setup poolSetup(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                           int sm) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        Setup u = new Setup();
        u.base = "pool";
        u.su = s.su;
        u.tpl = s.tpl;
        u.out = s.out;
        u.abl2Scores = s.out.resolve("xbase_pool_abl2_scores.tsv.gz");
        XBaseEval.Gt tpl = s.tpl;
        List<String> all = new ArrayList<>(s.pool.keySet());
        u.persons = all.size();
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(sm == 1 ? all.subList(0, 30) : all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        if (sm == 1) {
            sets.put("FERET-1", new TreeSet<>(new ArrayList<>(s.halves.get(1)).subList(0, 30)));
            sets.put("FERET-2", new TreeSet<>(new ArrayList<>(s.halves.get(0)).subList(0, 30)));
        }
        for (String set : sets.keySet()) {
            if (sm == 2 && !set.equals("GT")) break;
            u.groups.add(set);
            u.setSize.put(set, sets.get(set).size());
            Set<String> members = sets.get(set);
            List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
            int nb = 0;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                List<String> held = new ArrayList<>(parts.get(q));
                Set<String> train = new TreeSet<>(members);
                held.forEach(train::remove);
                Collections.shuffle(held, new Random(tpl.seed + XBaseRule.HALF_SEED_OFFSET + q));
                int h0 = (held.size() + 1) / 2;
                List<List<String>> hv = List.of(new ArrayList<>(new TreeSet<>(held.subList(0, h0))), new ArrayList<>(new TreeSet<>(held.subList(h0, held.size()))));
                Map<String, List<List<String>>> trainPool = new TreeMap<>();
                for (String p : train) trainPool.put(p, s.pool.get(p));
                List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), trainPool, Set.of(), Map.of()).extra();
                for (int h = 0; h < 2; h++) {
                    if (sm == 2 && nb >= XBaseNap.SMOKE2_BLOCKS) break;
                    List<String> own = hv.get(h);
                    List<String> ctrl = hv.get(1 - h);
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                    List<List<String>> aug = new ArrayList<>(s.cohort);
                    aug.addAll(extra);
                    g.cohort = aug;
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    List<Probe> probes = new ArrayList<>();
                    for (int i = 0; i < own.size(); i++) {
                        List<List<String>> ds = s.pool.get(own.get(i));
                        gal.put(own.get(i), ds.get(0));
                        for (int d = 1; d < ds.size(); d++) for (String k : ds.get(d)) probes.add(new Probe(FeiScores.OWN, own.get(i), g.img(k), k, i));
                    }
                    for (String p : ctrl) for (List<String> d : s.pool.get(p)) for (String k : d) probes.add(new Probe(FeiScores.CTRL, p, g.img(k), k, -1));
                    u.blocks.add(new Blk(String.format(Locale.ROOT, "%s_q%d_h%d", set, q, h), set + ", %s, часть " + q + ", смена " + h + ", ", g, gal, null, null,
                            probes, null, 2 * q + h, set));
                    nb++;
                }
            }
        }
        u.fr = XBaseLbp.frames(s, export, false);
        cohorts(u, feiKeys, tpl.seed, export);
        return u;
    }

    static Setup feretSetup(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                            int sm) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        Setup u = new Setup();
        u.base = "feret";
        u.su = s.su;
        u.tpl = s.tpl;
        u.out = s.out;
        u.abl2Scores = s.out.resolve("xbase_feret_abl2_scores.tsv.gz");
        XBaseEval.Gt tpl = s.tpl;
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        u.persons = persons.size();
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, List<List<String>>> kept = new TreeMap<>();
        for (String p : ro0.own1) {
            Map<String, List<String>> bd = new TreeMap<>();
            List<String> al = new ArrayList<>(ro0.gal1.get(p));
            al.addAll(ro0.pro1.get(p));
            for (String k : al) bd.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ks = new ArrayList<>();
            for (List<String> l : bd.values()) ks.add(XBaseFeret.sorted(l));
            kept.put(p, ks);
        }
        for (int post = 0; post < 2; post++) {
            if (sm == 2 && post > 0) break;
            String pname = post == 0 ? "межс." : "(ii) s3";
            u.groups.add(pname);
            int conf = post == 0 ? 0 : 71;
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (post == 0) {
                    own = XBaseVt.trimOwn(ro.own1.stream().filter(mine::contains).toList(), sm);
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else {
                    own = XBaseVt.trimOwn(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList(), sm);
                    for (String p : own) {
                        List<String> g = new ArrayList<>();
                        for (int j = 0; j < 3; j++) g.addAll(s.pool.get(p).get(j));
                        gal.put(p, XBaseFeret.sorted(g));
                        List<String> pr = new ArrayList<>();
                        for (int j = 3; j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                        pro.put(p, XBaseFeret.sorted(pr));
                    }
                }
                Map<String, List<List<String>>> projPool = new TreeMap<>();
                for (String p : s.halves.get(1 - ps)) if (sm != 1 || projPool.size() < XBaseDates.SMOKE_POOL) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + (post == 0 ? 0 : 71));
                List<Probe> probes = XBaseEval.probes(g, pro);
                u.blocks.add(new Blk(String.format(Locale.ROOT, "c%d_p%d", conf, ps), pname + ", %s, проход " + (ps + 1) + ", ", g, gal, pro, parts, probes,
                        XBaseThr.extra(g, pro, parts), conf, pname));
            }
        }
        u.fr = XBaseLbp.frames(s, export, true);
        cohorts(u, feiKeys, tpl.seed, export);
        return u;
    }

    static Setup gtSetup(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out,
                         int sm) throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-vec: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Setup u = new Setup();
        u.base = "gt";
        u.su = su;
        u.tpl = g;
        u.out = out;
        u.persons = g.persons.size();
        u.abl2Scores = out.resolve("xbase_gt_abl2_scores.tsv.gz");
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        u.fr = XBaseEval.frames(export, g.samples.values().stream().sorted((a, b) -> a.file().toString().compareTo(b.file().toString())).toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        u.fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) u.fr.put(k, FeretData.frame(su.dir, "a", k));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));
        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = XBaseVt.trimOwn(g.own, sm);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        for (int mi = 0; mi < modes.length; mi++) {
            if (sm == 2 && mi > 0) break;
            String mode = (String) modes[mi][0];
            u.groups.add(mode);
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
            List<Probe> probes = XBaseEval.probes(ga, pro);
            u.blocks.add(new Blk(mi == 0 ? "x" : "i", mode + ", %s, ", ga, gal, pro, parts, probes, mi == 0 ? XBaseThr.extra(ga, pro, parts) : null, 0, mode));
        }
        cohorts(u, feiKeys, g.seed, export);
        return u;
    }

    static Setup setup(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                       Path out, int sm) throws IOException {
        return switch (base) {
            case "pool" -> poolSetup(export, dir, feiKeys, selLines, settings, sm);
            case "feret" -> feretSetup(export, dir, feiKeys, selLines, settings, sm);
            default -> gtSetup(export, dir, feiKeys, selLines, settings, out, sm);
        };
    }

    static Path vecDir(Setup u, int sm) {
        return u.out.resolve("xbase_" + u.base + "_vec" + (sm == 0 ? "" : "_smoke" + sm));
    }

    static Path vecFile(Path dir, Blk b, int f) {
        return dir.resolve(b.name() + "_f" + f + ".tsv.gz");
    }

    // ---------------------------------------------------------------- pool-vec, feret-vec, gt-vec

    static void vec(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        int sm = smoke();
        Setup u = setup(base, export, dir, feiKeys, selLines, settings, out, sm);
        if (!Files.exists(u.abl2Scores)) throw new IllegalStateException(base + "-vec: для сверки нужен " + u.abl2Scores + " (" + base + "-abl2)");
        Map<String, String> abl2 = XBaseSubsp.blocks(u.abl2Scores);
        Path vd = vecDir(u, sm);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        String head = "# Векторы 1 − cos (XBaseVec " + base + "-vec): галерея, когорта в W (" + String.join(",", u.cohortIn) + "), когорта вне W ("
                + String.join(",", u.cohortOut) + ")\n# Код: коммит " + commit + "\n";
        StringBuilder log = new StringBuilder();
        long t0 = System.nanoTime();
        for (int f = 1; f >= 0; f--) {
            XBaseEval x = null;
            List<List<Callable<Model>>> jobs = new ArrayList<>();
            List<Blk> todo = new ArrayList<>();
            for (Blk b : u.blocks) {
                Path p = vecFile(vd, b, f);
                if (Files.exists(p)) {
                    String h = readHead(p);
                    if (!h.contains(commit)) throw new IllegalStateException(base + "-vec: " + p + " – другой коммит кода; удалите каталог " + vd);
                    continue;
                }
                if (x == null) {
                    long tv = System.nanoTime();
                    x = XBasePoem.vectors(u.fr, u.tpl.params, f, new int[] {P2})[P2];
                    System.out.printf(Locale.ROOT, "%s-vec: f%d vectors (%.0f s)%n", base, f, (System.nanoTime() - tv) / 1e9);
                }
                jobs.add(job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
                todo.add(b);
            }
            // По группам блоков – чтобы файлы появлялись по ходу (контрольная точка).
            int step = Math.max(1, Math.min(4, jobs.size()));
            for (int i = 0; i < jobs.size(); i += step) {
                List<List<Callable<Model>>> part = jobs.subList(i, Math.min(jobs.size(), i + step));
                long ts = System.nanoTime();
                List<List<Model>> done = run(part, u.tpl.threads);
                for (int k = 0; k < done.size(); k++) {
                    Blk b = todo.get(i + k);
                    String bn = String.format(b.abl2(), XBaseRule.FEATS[f]) + XBaseAbl2.rname(f, 0);
                    if (sm != 1) XBaseReg.check(bn, lines(done.get(k)), abl2, u.abl2Scores);
                    Path p = vecFile(vd, b, f);
                    Writer w = writer(p, head);
                    write(w, bn, done.get(k));
                    finish(w, p);
                    log.append(bn).append(": совпало с ").append(u.abl2Scores.getFileName()).append('\n');
                }
                System.out.printf(Locale.ROOT, "%s-vec: f%d blocks %d–%d ready (%.0f s)%n", base, f, i + 1, i + part.size(), (System.nanoTime() - ts) / 1e9);
            }
            x = null;
        }
        for (Mat m : u.fr.values()) m.release();
        double sec = (System.nanoTime() - t0) / 1e9;
        StringBuilder t = new StringBuilder("Векторы 1 − cos, " + base + " (XBaseEval " + base + "-vec)" + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\n");
        t.append("Код: коммит ").append(commit).append('\n').append("Каталог: ").append(vd).append('\n');
        t.append("Когорта в W: ").append(u.cohortIn).append("; когорта вне W (seed faces.seed + ").append(OUT_SEED_OFFSET).append("): ").append(u.cohortOut)
                .append('\n');
        t.append("=== Сверка строк d₁/d₂ по векторам с файлом оценок abl2 (строка базы), побайтно ===\n").append(log);
        t.append(String.format(Locale.ROOT, "Обучение и векторы: %.0f с (без блоков из каталога); потоков %d.%n", sec, u.tpl.threads));
        if (sm == 2) {
            // Доля замера в полном прогоне: пул – 2 блока из 10 набора GT (наборы FERET – по pool-poem); FERET и GT – «межс.» обоих признаков
            // (по feret-poem: 2143 + 1247 из 5739 с; gt-poem: 2843 + 2573 из 9577 с).
            double frac = base.equals("pool") ? 1.0 / (10.0 / XBaseNap.SMOKE2_BLOCKS * (1 + XBaseNap.FERET_TO_GT)) : base.equals("feret") ? (2143.0 + 1247)
                    / (2143 + 1365 + 1247 + 984) : (2843.0 + 2573) / (2843 + 2172 + 2573 + 1989);
            t.append(String.format(Locale.ROOT, "Замер (оба признака, первая группа): %.0f с; оценка полного прогона: %.0f с (%.1f ч).%n", sec, sec / frac, sec / frac
                    / 3600));
        }
        FarMethods.write(u.out.resolve("xbase_" + base + "_vec" + (sm == 0 ? "" : "_smoke" + sm) + ".txt"), t);
        System.out.print(t);
    }

    static String readHead(Path p) throws IOException {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(p)), StandardCharsets.UTF_8))) {
            return in.readLine() + "\n" + in.readLine();
        }
    }

    // ---------------------------------------------------------------- pool-norm, feret-norm, gt-norm

    /** Варианты разбора на признак: 0 – база β = 1, 1…8 – NORMS. */
    static String vname(int v) {
        return v == 0 ? "база (β = 1)" : NORMS.get(v - 1).name();
    }

    static final String[] WS = {"0,25", "0,5", "0,75"};
    static final double[] W = {0.25, 0.5, 0.75};

    static final String NOTE = "По векторам 1 − cos (режимы *-vec; принятая конфигурация: база (П2), косинус, L1-tt-xs-F – модель по позе, L1-xs-F – всегда "
            + "X+mirror), без обучения.\nНормировка (в логарифмах, ℓ = log(1 − cos), 1 − cos = 0 – ρ_min): ℓₙ = (z + t) / 2, z – по μ, σ ℓ «центр человека "
            + "галереи → центры когорты», t – по μ, σ ℓ «проба → центры когорты»;\nAS-norm – μ, σ по K ближайшим; лучший и второй – по ℓₙ; s = (1 + β) ℓ₁ₙ − "
            + "β ℓ₂ₙ, β = 1; d₁′ = exp(s), второе – 1e6; |s| ≤ 13,8 (иначе отношение выходит за [ρ_min, 1]; в постановке – ±700).\nКогорты: в W – 20 людей "
            + "FEI когорты MLDA; вне W – 20 других «полных» людей FEI (seed faces.seed + 10500). База – d₁²/d₂ по сырым расстояниям (= β = 1 comb).\n"
            + "Слияние: ℓ(g) = w · (ℓ_tt − μ_tt) / σ_tt + (1 − w) · (ℓ_xs − μ_xs) / σ_xs, μ, σ – по всем расстояниям «чужой (Т2) – человек галереи» "
            + "своей конфигурации\n(в пуле моделей частей нет – по контрольным чужим блока); дальше лучший, второй, β = 1.\n1:1: худший случай – заявлен "
            + "ближайший; случайный – каждая чужая попытка заявляет случайного человека галереи (Random, 5 повторов); по всем парам – только\nразделимость "
            + "и EER (основная модель, контрольные чужие × все люди галереи).\n";

    static void norm(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        int sm = smoke();
        Setup u = setup(base, export, dir, feiKeys, selLines, settings, out, sm);
        for (Mat m : u.fr.values()) m.release();
        Path vd = vecDir(u, sm);
        boolean pool = base.equals("pool");
        Map<String, String> abl2 = XBaseSubsp.blocks(u.abl2Scores);
        // [признак][блок] → модели
        List<List<List<Model>>> ms = new ArrayList<>();
        StringBuilder check = new StringBuilder("\n=== Проверка: строки d₁/d₂ по векторам = строка базы abl2, побайтно; база β = 1 = XBaseAbl3.beta ===\n");
        String vhead = null;
        for (int f = 0; f < 2; f++) {
            List<List<Model>> l = new ArrayList<>();
            for (Blk b : u.blocks) {
                Path p = vecFile(vd, b, f);
                if (vhead == null) vhead = readHead(p).split("\n")[1];
                List<Model> m = read(p).values().iterator().next();
                String bn = String.format(b.abl2(), XBaseRule.FEATS[f]) + XBaseAbl2.rname(f, 0);
                if (sm != 1) check.append(XBaseReg.check(bn, lines(m), abl2, u.abl2Scores));
                l.add(m);
            }
            ms.add(l);
        }
        int[] clamped = new int[2];
        int nv = 1 + NORMS.size();
        StringBuilder t = new StringBuilder();
        t.append("Пачка 1: нормировка по когорте, слияние кандидатов, 1:1 – ").append(base).append(" (XBaseEval ").append(base).append("-norm)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append("; векторы – ").append(vd).append(" (").append(vhead).append(")\n");
        t.append(NOTE);
        StringBuilder body = new StringBuilder();
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база β = 1») ===\n");
        StringBuilder pairsT = new StringBuilder("\n=== 1:1 по всем парам (основная модель): ошибка своих при FAR 1 / 2 / 5 / 10 % | EER | пар «чужой – человек» ===\n");
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseVec pool-norm): часть, набор, признак, номер, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        StringBuilder gtPart = new StringBuilder();
        for (String grp : u.groups) {
            int bodyStart = body.length();
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            // Строки вариантов: [признак][вариант] и слияние [w], 1:1 [признак][0 худший, 1…5 случайный].
            double[][] stT = new double[bi.size()][];
            double[][] stX = new double[bi.size()][];
            for (int k = 0; k < bi.size(); k++) {
                stT[k] = fuseStats(ms.get(1).get(bi.get(k)), pool);
                stX[k] = fuseStats(ms.get(0).get(bi.get(k)), pool);
            }
            if (pool) {
                body.append("\n\n######## Набор ").append(grp).append(" (").append(u.setSize.get(grp)).append(" человек) ########\n");
                body.append("вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | "
                        + "контрольные\n");
                double[][] crit = new double[2][nv];
                for (int f = 1; f >= 0; f--) {
                    for (int v = 0; v < nv; v++) {
                        List<FeiDecompose.Cfg> cf = new ArrayList<>();
                        for (int k : bi) {
                            Blk b = u.blocks.get(k);
                            String l = normLines(ms.get(f).get(k), v == 0 ? null : NORMS.get(v - 1), clamped);
                            cf.add(XBaseEval.parse(XBaseAbl3.mirrorGt(b.g(), f == 0), EnumSet.of(V.L1), b.gal(), b.probes(), noParts, l).cfg.get(V.L1));
                        }
                        XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cf)), FeiBootstrap.unit(u.persons), new int[0][]);
                        crit[f][v] = sp.crit();
                        body.append(XBaseAbl3.poolRow(vname(v), XBaseRule.FEATS[f], sp)).append('\n');
                    }
                    int c = XBaseNap.choose(crit[f]);
                    body.append(String.format(Locale.ROOT, "Нормировка, выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", grp, XBaseRule.FEATS[f],
                            vname(c), crit[f][c], crit[f][0], Arrays.stream(crit[f]).min().orElseThrow()));
                    sel.append("1\t").append(grp).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(vname(c)).append('\n');
                }
                double[] fc = new double[W.length];
                for (int wi = 0; wi < W.length; wi++) {
                    List<FeiDecompose.Cfg> cf = new ArrayList<>();
                    for (int k = 0; k < bi.size(); k++) {
                        Blk b = u.blocks.get(bi.get(k));
                        String l = fuseLines(ms.get(1).get(bi.get(k)), ms.get(0).get(bi.get(k)), W[wi], stT[k], stX[k], b.g().rOf, b.g().r0, clamped);
                        cf.add(XBaseEval.parse(b.g(), EnumSet.of(V.L1), b.gal(), b.probes(), noParts, l).cfg.get(V.L1));
                    }
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cf)), FeiBootstrap.unit(u.persons), new int[0][]);
                    fc[wi] = sp.crit();
                    body.append(XBaseAbl3.poolRow("слияние w = " + WS[wi], "оба", sp)).append('\n');
                }
                double single = Math.min(crit[0][0], crit[1][0]);
                double bestF = Arrays.stream(fc).min().orElseThrow();
                int wb = 0;
                for (int wi = 1; wi < W.length; wi++) if (fc[wi] < fc[wb]) wb = wi;
                String choice = single <= bestF + XBaseRule.NEAR ? "без слияния" : "w = " + WS[wb];
                body.append(String.format(Locale.ROOT, "Слияние, выбор %s: %s (лучший кандидат – %.1f %%; лучшее слияние – %.1f %%)%n", grp, choice, single, bestF));
                sel.append("2\t").append(grp).append("\tоба\t").append(single <= bestF + XBaseRule.NEAR ? -1 : wb).append('\t').append(choice).append('\n');
                for (int f = 1; f >= 0; f--) {
                    int ff = f;
                    double[] sum = new double[6];
                    for (int k : bi) {
                        XBaseEval.Gt g = u.blocks.get(k).g();
                        double[] pr = pairs(ms.get(f).get(k), r -> ff == 0 || g.rOf.get(r.key()) > g.r0);
                        for (int i = 0; i < sum.length; i++) sum[i] += i < 5 ? pr[i] / bi.size() : pr[i];
                    }
                    pairsT.append(String.format(Locale.ROOT, "%s %s | %.1f / %.1f / %.1f / %.1f %% | %.1f %% | %.0f (среднее по блокам)%n", grp, XBaseRule.FEATS[f], sum[0],
                            sum[1], sum[2], sum[3], sum[4], sum[5]));
                }
                if (grp.equals("GT")) gtPart.append(body.substring(bodyStart));
                continue;
            }
            // Свои: FERET (межс., (ii) s3) и GT (межс., внутри).
            boolean t2 = !grp.equals("внутри");
            boolean within = grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> prs = new ArrayList<>();
                String bl = grp + " " + feat + " " + vname(0);
                for (int v = 0; v < nv; v++) {
                    List<XBaseThr.Unit> us = new ArrayList<>();
                    for (int k : bi) {
                        Blk b = u.blocks.get(k);
                        String l = normLines(ms.get(f).get(k), v == 0 ? null : NORMS.get(v - 1), clamped);
                        if (v == 0) {
                            String ref = XBaseAbl3.beta(lines(ms.get(f).get(k)), BETA);
                            if (!ref.equals(l)) throw new IllegalStateException("XBaseVec: база β = 1 ≠ XBaseAbl3.beta");
                        }
                        us.add(new XBaseThr.Unit(XBaseAbl3.mirrorGt(b.g(), f == 0), b.gal(), b.pro(), b.parts(), l, null, t2));
                    }
                    String lb = grp + " " + feat + " " + vname(v);
                    XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                    outs.put(lb, o);
                    if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                    if (v > 0) prs.add(new String[] {lb, bl});
                }
                XBaseLbp.tables(body, prefix + ", " + feat + " – нормировка по когорте", outs, u.persons, u.tpl, prs, t2);
                if (within) for (String[] p : prs) {
                    double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], u.persons, u.tpl);
                    dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                            : "НЕ выполнено"));
                }
                // 1:1.
                Map<String, XBaseThr.Out> o1 = new LinkedHashMap<>();
                List<String[]> p1 = new ArrayList<>();
                String l1n = grp + " " + feat + " 1:N (β = 1)";
                o1.put(l1n, outs.get(bl));
                for (int rep = 0; rep <= 5; rep++) {
                    List<XBaseThr.Unit> us = new ArrayList<>();
                    for (int k : bi) {
                        Blk b = u.blocks.get(k);
                        String l = rep == 0 ? XBaseAbl3.oneLines(lines(ms.get(f).get(k))) : oneRandom(ms.get(f).get(k), u.tpl.seed + 10600 + rep);
                        us.add(new XBaseThr.Unit(XBaseAbl3.mirrorGt(b.g(), f == 0), b.gal(), b.pro(), b.parts(), l, null, t2));
                    }
                    String lb = grp + " " + feat + (rep == 0 ? " 1:1 худший случай" : " 1:1 случайный, повтор " + rep);
                    XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                    o1.put(lb, o);
                    if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                    p1.add(new String[] {lb, l1n});
                }
                XBaseLbp.tables(body, prefix + ", " + feat + " – 1:1", o1, u.persons, u.tpl, p1, t2);
                int ff = f;
                double[] sum = new double[6];
                for (int k : bi) {
                    XBaseEval.Gt g = u.blocks.get(k).g();
                    double[] pr = pairs(ms.get(f).get(k), r -> ff == 0 || g.rOf.get(r.key()) > g.r0);
                    for (int i = 0; i < sum.length; i++) sum[i] += i < 5 ? pr[i] / bi.size() : pr[i];
                }
                pairsT.append(String.format(Locale.ROOT, "%s %s | %.1f / %.1f / %.1f / %.1f %% | %.1f %% | %.0f (среднее по проходам)%n", grp, feat, sum[0], sum[1],
                        sum[2], sum[3], sum[4], sum[5]));
            }
            // Слияние.
            Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
            List<String[]> pf = new ArrayList<>();
            for (int f = 1; f >= 0; f--) {
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (int k : bi) {
                    Blk b = u.blocks.get(k);
                    us.add(new XBaseThr.Unit(XBaseAbl3.mirrorGt(b.g(), f == 0), b.gal(), b.pro(), b.parts(), normLines(ms.get(f).get(k), null, clamped), null, t2));
                }
                String lb = grp + " " + XBaseRule.FEATS[f] + " (β = 1)";
                of.put(lb, XBaseThr.evaluate(lb, us, u.persons, u.tpl));
            }
            String btt = grp + " " + XBaseRule.FEATS[1] + " (β = 1)";
            for (int wi = 0; wi < W.length; wi++) {
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (int k = 0; k < bi.size(); k++) {
                    Blk b = u.blocks.get(bi.get(k));
                    us.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), fuseLines(ms.get(1).get(bi.get(k)), ms.get(0).get(bi.get(k)), W[wi], stT[k], stX[k],
                            b.g().rOf, b.g().r0, clamped), null, t2));
                }
                String lb = grp + " слияние w = " + WS[wi];
                XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                of.put(lb, o);
                if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                pf.add(new String[] {lb, btt});
                pf.add(new String[] {lb, grp + " " + XBaseRule.FEATS[0] + " (β = 1)"});
            }
            XBaseLbp.tables(body, prefix + " – слияние кандидатов", of, u.persons, u.tpl, pf, t2);
            if (within) for (int wi = 0; wi < W.length; wi++) {
                String lb = grp + " слияние w = " + WS[wi];
                double[] d = XBaseVt.delta(of.get(lb), of.get(btt), XBaseThr.WAYS[0], u.persons, u.tpl);
                dl.append(String.format(Locale.ROOT, "%s − L1-tt-xs-F: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", lb, d[0], d[1], d[1] <= 3
                        ? "выполнено" : "НЕ выполнено"));
            }
        }
        t.append(check);
        t.append(String.format(Locale.ROOT, "%nОценка s (нормированные строки и слияние): от %.2f до %.2f; вне ±%.1f (exp(s) / 1e6 вышло бы за [ρ_min, 1]): %d; "
                + "вне различимой области сжатия (|s| ≥ %.0f): %d%n", S_RANGE[0], S_RANGE[1], S_CLAMP, clamped[0], TAU * S_RES, clamped[1]));
        if (!pool) t.append(hold).append(base.equals("gt") ? dl : "");
        t.append(pairsT).append(body);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        if (pool && sm == 0) Files.writeString(u.out.resolve("xbase_pool_norm_select.tsv"), sel.toString());
        FarMethods.write(u.out.resolve("xbase_" + base + "_norm" + sfx + ".txt"), t);
        if (pool) {
            Files.createDirectories(out);
            FarMethods.write(out.resolve("xbase_pool_norm" + sfx + "_gt.txt"), new StringBuilder(t.substring(0, t.indexOf(pairsT.toString()))).append(gtPart));
        }
        System.out.print(t);
    }
}
