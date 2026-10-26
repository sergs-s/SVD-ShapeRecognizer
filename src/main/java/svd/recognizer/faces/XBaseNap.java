package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * NAP / WCCN (п. 2.3; постановка Жорика №7 от 26.10.2026, план подтверждён Хозяином) поверх признака (П2) LBP + POEM · 1/√3
 * (решение Хозяина (28)): пространство – W текущей модели MLDA (пол c = 1; основная и 5 моделей частей, X / X+mirror). Помеха –
 * объединённая внутриклассовая ковариация S_n векторов y/‖y‖ (y = W-проекция) людей пула FERET этой модели (человек – один класс по
 * всем датам, у X+mirror – с отражениями, как в выборке W; галерея и когорта FEI в S_n не входят). (N) NAP: T = I − U_k U_kᵀ (k
 * главных направлений S_n), k ∈ {5, 10, 20, 40}; (W) WCCN: T = (S_n′ / m)^(−1/2), S_n′ = (1 − α) S_n + α m I, m = tr S_n / n, α ∈
 * {0,5; 0} (α = 1 – база). Оценка – 1 − cos(T y, T c_g), c_g – прежний центр класса галереи (нормировка длины после T косинус не
 * меняет), d₁/d₂; база (T = I) – строка XBaseReg.lines, путь тот же. Выбор по пулу – по схеме pool-rule: база в пределах 1 п. п. от
 * лучшего, иначе первый по простоте ((N) 5 < 10 < 20 < 40 < (W) 0,5 < 0) в пределах 1 п. п.
 *
 * @author ssv
 */
final class XBaseNap {

    static final int[] KS = {5, 10, 20, 40};
    static final double[] ALPHAS = {0.5, 0};
    /** Строки: 0 – база, 1…4 – (N) k, 5…6 – (W) α. */
    static final int R = 1 + KS.length + ALPHAS.length;
    static final int BASE = 0;
    /** Признак (П2) в XBasePoem. */
    static final int P2 = 2;
    /** Пол собственных значений S_n′ при отбеливании (доля наибольшего). */
    static final double FLOOR = 1e-12;
    static final String SELECT = "xbase_pool_nap_select.tsv";
    /** Доля наборов FERET-1 и FERET-2 к GT во времени pool-poem 26.10 (GT 2139 + 3810 с; FERET 605 + 683 + 755 + 904 с). */
    static final double FERET_TO_GT = (605.0 + 683 + 755 + 904) / (2139 + 3810);
    /** Доля первой группы (L1-tt-xs-F, «межс.») во времени feret-poem и gt-poem 26.10. */
    static final double FERET_FIRST = 2143.0 / (2143 + 1365 + 1247 + 984);
    static final double GT_FIRST = 2843.0 / (2843 + 2172 + 2573 + 1989);
    static final int SMOKE2_BLOCKS = 2;

    private XBaseNap() {
    }

    /** Пробный режим FACES_NAP_SMOKE: 0 – нет; 1 – малые данные, весь путь; 2 – первая группа в полном размере – замер. */
    static int smoke() {
        String v = System.getenv("FACES_NAP_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    static String rname(int r) {
        if (r == BASE) return "база (П2)";
        if (r <= KS.length) return "(N) k = " + KS[r - 1];
        return "(W) α = " + (ALPHAS[r - 1 - KS.length] == 0.5 ? "0,5" : "0");
    }

    // ---------------------------------------------------------------- линейная алгебра

    static double[] unit(double[] y) {
        double s = 0;
        for (double v : y) s += v * v;
        s = Math.sqrt(s);
        if (s == 0) throw new IllegalStateException("XBaseNap: нулевой вектор в W");
        double[] z = new double[y.length];
        for (int i = 0; i < y.length; i++) z[i] = y[i] / s;
        return z;
    }

    /** Объединённая внутриклассовая ковариация: Σ_c Σ_i (z − μ_c)(z − μ_c)ᵀ / (N − C); классы – не меньше 2 векторов. */
    record Sn(double[][] s, int n, int c) {}

    static Sn sn(List<List<double[]>> classes) {
        int d = classes.get(0).get(0).length;
        double[][] s = new double[d][d];
        int n = 0;
        for (List<double[]> cl : classes) {
            if (cl.size() < 2) throw new IllegalStateException("XBaseNap: класс помехи из " + cl.size() + " векторов");
            double[] mu = new double[d];
            for (double[] z : cl) for (int i = 0; i < d; i++) mu[i] += z[i];
            for (int i = 0; i < d; i++) mu[i] /= cl.size();
            for (double[] z : cl) {
                for (int i = 0; i < d; i++) {
                    double a = z[i] - mu[i];
                    for (int j = 0; j <= i; j++) s[i][j] += a * (z[j] - mu[j]);
                }
            }
            n += cl.size();
        }
        int c = classes.size();
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            s[i][j] /= n - c;
            s[j][i] = s[i][j];
        }
        return new Sn(s, n, c);
    }

    /** NAP: I − U_k U_kᵀ (U_k – собственные векторы с k наибольшими значениями). */
    static double[][] nap(LdaMath.Eigen e, int k) {
        int d = e.values().length;
        double[][] p = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
            double s = i == j ? 1 : 0;
            for (int q = 0; q < k; q++) s -= e.vectors()[i][q] * e.vectors()[j][q];
            p[i][j] = s;
        }
        return p;
    }

    /** Собственные значения S_n′ / m (m = tr S_n / n) с полом FLOOR · max; [значения, число поднятых до пола]. */
    static double[] scaled(LdaMath.Eigen e, double alpha, int[] raised) {
        int d = e.values().length;
        double m = 0;
        for (double v : e.values()) m += v;
        m /= d;
        double[] g = new double[d];
        double max = 0;
        for (int i = 0; i < d; i++) {
            g[i] = ((1 - alpha) * e.values()[i] + alpha * m) / m;
            max = Math.max(max, g[i]);
        }
        int up = 0;
        for (int i = 0; i < d; i++) if (!(g[i] > FLOOR * max)) {
            g[i] = FLOOR * max;
            up++;
        }
        if (raised != null) raised[0] = up;
        return g;
    }

    /** WCCN: (S_n′ / m)^(−1/2); α = 1 – null (база, T = I). */
    static double[][] wccn(LdaMath.Eigen e, double alpha) {
        if (alpha == 1) return null;
        double[] g = scaled(e, alpha, null);
        int d = g.length;
        double[][] t = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < d; k++) s += e.vectors()[i][k] * e.vectors()[j][k] / Math.sqrt(g[k]);
            t[i][j] = s;
            t[j][i] = s;
        }
        return t;
    }

    /** Строки d₁/d₂ при T (null – T = I, как XBaseReg.lines): 1 − cos(T y, T c_g). */
    static String lines(String method, int s, double[][] c, List<String> gallery, List<Probe> probes, Map<String, double[]> y, double[][] t) {
        int d = c.length == 0 ? 0 : Arrays.stream(c).filter(v -> v != null).findFirst().map(v -> v.length).orElse(0);
        double[][] tc = new double[c.length][];
        for (int g = 0; g < c.length; g++) if (c[g] != null) tc[g] = t == null ? c[g] : XBasePlda.mul(t, c[g]);
        StringBuilder out = new StringBuilder();
        for (Probe p : probes) {
            double[] yy = y.get(p.key());
            double[] ty = t == null ? yy : XBasePlda.mul(t, yy);
            double[] sc = new double[c.length];
            for (int g = 0; g < c.length; g++) sc[g] = tc[g] == null ? Double.POSITIVE_INFINITY : XBaseRule.cos(ty, tc[g], d, null);
            out.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc, false));
        }
        return out.toString();
    }

    // ---------------------------------------------------------------- одна модель

    /**
     * Строки модели: [0…R−1] – косинус по строкам (0 – XBaseReg.lines), [R] – сведения S_n (только при tag != null), [R + 1] – сведения
     * MLDA, [R + 2] – число осей. cohort[poolFrom…] – классы пула (помеха).
     */
    static String[] model(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> cohort, int poolFrom,
                          List<Probe> probes, String tag) {
        XBaseRule.Fit f = XBaseRule.fit(x, mirror, gallery, cohort, null);
        String[] out = new String[R + 3];
        out[0] = XBaseReg.lines(x, method, s, f.lm, own, probes)[1];
        List<List<double[]>> cls = new ArrayList<>();
        int vecs = 0;
        for (List<String> cl : cohort.subList(poolFrom, cohort.size())) {
            List<double[]> l = new ArrayList<>();
            for (String k : cl) {
                l.add(unit(f.lm.embed(x.vec.get(V.L1).get(k))));
                if (mirror) l.add(unit(f.lm.embed(x.mir.get(V.L1).get(k))));
            }
            cls.add(l);
            vecs += l.size();
        }
        Sn sn = sn(cls);
        LdaMath.Eigen e = LdaMath.eigen(sn.s());
        Map<String, double[]> y = new HashMap<>();
        for (Probe p : probes) y.computeIfAbsent(p.key(), k -> f.lm.embed(x.vec.get(V.L1).get(k)));
        double[][] c = f.lm.centers();
        for (int i = 0; i < KS.length; i++) out[1 + i] = lines(method, s, c, own, probes, y, nap(e, KS[i]));
        for (int i = 0; i < ALPHAS.length; i++) out[1 + KS.length + i] = lines(method, s, c, own, probes, y, wccn(e, ALPHAS[i]));
        if (tag != null) {
            double tr = 0;
            for (double v : e.values()) tr += v;
            StringBuilder sh = new StringBuilder();
            for (int k : KS) {
                double a = 0;
                for (int q = 0; q < k; q++) a += e.values()[q];
                sh.append(String.format(Locale.ROOT, "%s%.3f", sh.length() == 0 ? "" : " / ", a / tr));
            }
            StringBuilder cond = new StringBuilder();
            for (double al : ALPHAS) {
                int[] up = new int[1];
                double[] g = scaled(e, al, up);
                double mx = Arrays.stream(g).max().orElse(0);
                double mn = Arrays.stream(g).min().orElse(0);
                cond.append(String.format(Locale.ROOT, "%s%.3e (поднято %d)", cond.length() == 0 ? "" : " / ", mx / mn, up[0]));
            }
            int d = e.values().length;
            out[R] = String.format(Locale.ROOT, "%s | %d | %d | %d | %d | %s | %.3e | %.3e | %s%n", tag, d, sn.c(), vecs, sn.n() - sn.c(), sh, e.values()[0],
                    e.values()[d - 1], cond);
        } else {
            out[R] = "";
        }
        out[R + 1] = f.lm.info();
        out[R + 2] = Integer.toString(f.axes());
        return out;
    }

    static final String DIAG_HEAD = "\n=== Помеха S_n по основным моделям: блок | модель | размерность W | людей пула (C) | векторов (N) | N − C | доля следа S_n в "
            + "первых k = 5 / 10 / 20 / 40 направлениях | λ_max S_n | λ_min S_n | обусловленность S_n′ α = 0,5 / α = 0 ===\n";

    /** Задачи блока – как XBaseLbp.job (тот же порядок моделей и проб); parts == null – только основные модели. */
    record Job(List<Callable<String[]>> tasks) {}

    static Job job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts, int s, List<List<Probe>> extra,
                   int poolFrom, String block) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            String t0 = block + " | " + (m ? "X+mirror" : "X");
            tasks.add(() -> model(x, m, method, s, own, fullGallery, cohort, poolFrom, main, t0));
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                tasks.add(() -> model(x, m, method, s, g, gk, cohort, poolFrom, pr, null));
            }
        }
        return new Job(tasks);
    }

    /** Итог блока: строки [R], сведения S_n, сведения MLDA основных моделей X / X+mirror, оси. */
    record One(String[] rows, String diag, String[] info, int[] axes) {
        XBaseLbp.One row(int r) {
            return new XBaseLbp.One(new String[] {"", rows[r]}, info, axes);
        }
    }

    static List<One> run(List<Job> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (Job j : jobs) flat.addAll(j.tasks());
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<One> out = new ArrayList<>();
            int i = 0;
            for (Job j : jobs) {
                StringBuilder[] b = new StringBuilder[R];
                for (int r = 0; r < R; r++) b[r] = new StringBuilder();
                StringBuilder dg = new StringBuilder();
                String[] info = new String[2];
                int[] axes = new int[2];
                int mi = 0;
                for (int k = 0; k < j.tasks().size(); k++) {
                    String[] o = fs.get(i++).get();
                    for (int r = 0; r < R; r++) b[r].append(o[r]);
                    if (!o[R].isEmpty()) {
                        dg.append(o[R]);
                        info[mi] = o[R + 1];
                        axes[mi] = Integer.parseInt(o[R + 2]);
                        mi++;
                    }
                }
                String[] rows = new String[R];
                for (int r = 0; r < R; r++) rows[r] = b[r].toString();
                out.add(new One(rows, dg.toString(), info, axes));
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseNap: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    static One read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[] rows = new String[R];
        for (int r = 0; r < R; r++) {
            rows[r] = cp.read(tag + "_" + r);
            if (rows[r] == null) return null;
        }
        String dg = cp.read(tag + "_diag");
        String inf = cp.read(tag + "_info");
        if (dg == null || inf == null) return null;
        String[] p = inf.split("\n");
        String[] info = new String[2];
        int[] axes = new int[2];
        for (int m = 0; m < 2; m++) {
            String[] f = p[m].split("\t", 2);
            axes[m] = Integer.parseInt(f[0]);
            info[m] = f[1];
        }
        return new One(rows, dg, info, axes);
    }

    static void save(XBaseOther.Checkpoint cp, String tag, One l) throws IOException {
        for (int r = 0; r < R; r++) cp.save(tag + "_" + r, l.rows()[r]);
        cp.save(tag + "_diag", l.diag());
        cp.save(tag + "_info", l.axes()[0] + "\t" + l.info()[0] + "\n" + l.axes()[1] + "\t" + l.info()[1] + "\n");
    }

    /** База в пределах NEAR от лучшего – база; иначе первая по простоте (порядок строк) в пределах NEAR. */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        for (int r = 0; r < crit.length; r++) if (crit[r] <= best + XBaseRule.NEAR) return r;
        throw new IllegalStateException("XBaseNap: выбор не определён");
    }

    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseNap: нет файла выбора " + f + " (сначала pool-nap)");
        List<String> feats = List.of(XBaseRule.FEATS);
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int r = Integer.parseInt(p[2]);
            if (feats.indexOf(p[1]) < 0 || r < 0 || r >= R || !rname(r).equals(p[3])) throw new IllegalStateException("XBaseNap: строка «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], r);
        }
        return out;
    }

    /** Строки для своих: база и выбранные хоть в одном случае. */
    static int[] ownRows(Map<String, Map<String, Integer>> select) {
        TreeSet<Integer> rs = new TreeSet<>(List.of(BASE));
        for (Map<String, Integer> m : select.values()) rs.addAll(m.values());
        return rs.stream().mapToInt(Integer::intValue).toArray();
    }

    static String infoRow(String head, One l) {
        int[] a = XBaseLbp.hull(l.info()[0]);
        int[] b = XBaseLbp.hull(l.info()[1]);
        return String.format(Locale.ROOT, "%s | %d / %d | %d / %d | %d / %d%n", head, a[1], b[1], a[0], b[0], l.axes()[0], l.axes()[1]);
    }

    static final String NOTE = "Признак – (П2) LBP + POEM, часть POEM после корня · 1/√3 (решение (28); L1-xs-F 7×7 – 11 564, L1-tt-xs-F 9×9 – 19 116); MLDA с пулом "
            + "FERET, пол c = 1.\nПространство – W модели MLDA (основная и 5 моделей частей, X / X+mirror). Помеха S_n – объединённая внутриклассовая ковариация "
            + "векторов y/‖y‖\n(y – проекция в W) людей пула FERET этой модели (все с ≥ 2 датами; человек – один класс по всем датам; у X+mirror – с отражениями); "
            + "галерея и когорта FEI\nв S_n не входят. (N) NAP: T = I − U_k U_kᵀ, k = 5, 10, 20, 40; (W) WCCN: T = (S_n′ / m)^(−1/2), S_n′ = (1 − α) S_n + α m I, "
            + "m = tr S_n / n, α = 0,5; 0\n(α = 1 – база; пол собственных значений S_n′ – 1e-12 наибольшего). Оценка – 1 − cos(T y, T c_g), c_g – прежний центр "
            + "класса галереи (нормировка длины\nпосле T косинус не меняет), d₁/d₂, модель X / X+mirror по |r| пробы; база (T = I) – прежняя строка. Все строки – "
            + "на одной модели W.\n";

    // ---------------------------------------------------------------- pool-nap

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        int poolFrom = s.cohort.size();
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(sm == 1 ? all.subList(0, 30) : all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        if (sm == 1) {
            sets.put("FERET-1", new TreeSet<>(new ArrayList<>(s.halves.get(1)).subList(0, 30)));
            sets.put("FERET-2", new TreeSet<>(new ArrayList<>(s.halves.get(0)).subList(0, 30)));
        }
        List<String> setNames = new ArrayList<>(sets.keySet());
        Path poemScores = s.out.resolve("xbase_pool_poem_scores.tsv.gz");
        if (!Files.exists(poemScores)) throw new IllegalStateException("pool-nap: для сверки базы нужен " + poemScores + " (pool-poem)");
        Map<String, String> blocks = XBaseSubsp.blocks(poemScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        List<String> names = new ArrayList<>();
        for (int r = 0; r < R; r++) names.add(rname(r));
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-nap %s; seed %d; r0 %s; пул %d; половины %d/%d; строки %s; пробный %d",
                s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), String.join("; ", names), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_nap_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        StringBuilder head = new StringBuilder();
        head.append("П. 2.3, NAP / WCCN: выбор по пулу FERET (XBaseEval pool-nap)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-poem: GT – весь пул; FERET-1 – половина B (%d); FERET-2 – половина A (%d);%n5 частей "
                + "по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула; обучение W – галерея своих + "
                + "когорта FEI 20 + люди 4 частей (они же – помеха S_n).%n", s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET,
                XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база, если в пределах %.0f п. п. от лучшего,%nиначе "
                + "первая по простоте ((N) k = 5 < 10 < 20 < 40 < (W) α = 0,5 < 0) в пределах %.0f п. п.%n", XBaseRule.NEAR, XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки (П2) pool-poem, побайтно ===\n");
        StringBuilder diag = new StringBuilder(DIAG_HEAD);
        StringBuilder diagGt = new StringBuilder(DIAG_HEAD);
        List<List<List<List<FeiDecompose.Cfg>>>> cfgs = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            List<List<List<FeiDecompose.Cfg>>> a = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                List<List<FeiDecompose.Cfg>> b = new ArrayList<>();
                for (int r = 0; r < R; r++) b.add(new ArrayList<>());
                a.add(b);
            }
            cfgs.add(a);
        }
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        double[] smokeS = new double[2];
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        for (int f = 0; f < 2; f++) {
            XBaseEval x = null;
            for (int si = 0; si < setNames.size(); si++) {
                if (sm == 2 && si > 0) break;
                String set = setNames.get(si);
                Set<String> members = sets.get(set);
                List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
                List<Object[]> blk = new ArrayList<>();
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    if (sm == 2 && blk.size() >= SMOKE2_BLOCKS) break;
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
                        if (sm == 2 && blk.size() >= SMOKE2_BLOCKS) break;
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
                        String tag = String.format(Locale.ROOT, "%s_q%d_h%d_f%d", set, q, h, f);
                        blk.add(new Object[] {q, h, g, gal, probes, tag, read(cp, tag)});
                    }
                }
                List<Job> jobs = new ArrayList<>();
                List<Object[]> todo = new ArrayList<>();
                for (Object[] b : blk) {
                    if (b[6] != null) continue;
                    if (x == null) {
                        long t0 = System.nanoTime();
                        x = XBasePoem.vectors(fr, tpl.params, f, new int[] {P2})[P2];
                        System.out.printf(Locale.ROOT, "pool-nap: f%d vectors (%.0f s)%n", f, (System.nanoTime() - t0) / 1e9);
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    String bn = String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], (int) b[0], (int) b[1]);
                    jobs.add(job((XBaseEval.Gt) b[2], x, gal, probes, null, 2 * (int) b[0] + (int) b[1], null, poolFrom, bn));
                    todo.add(b);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<One> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    if (sm == 2) smokeS[f] = sec;
                    for (int i = 0; i < todo.size(); i++) {
                        todo.get(i)[6] = done.get(i);
                        save(cp, (String) todo.get(i)[5], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-nap: %s f%d ready, %d blocks (%.0f s)%n", set, f, jobs.size(), sec);
                }
                for (Object[] b : blk) {
                    int q = (int) b[0];
                    int h = (int) b[1];
                    XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    One ls = (One) b[6];
                    String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                    if (sm != 1) check.append(XBaseReg.check(bp + XBasePoem.vname(f, P2) + ", косинус", ls.rows()[BASE], blocks, poemScores));
                    diag.append(ls.diag());
                    if (set.equals("GT")) diagGt.append(ls.diag());
                    for (int r = 0; r < R; r++) {
                        scores.append("# ").append(bp).append(rname(r)).append(", косинус\n").append(ls.rows()[r]);
                        cfgs.get(si).get(f).get(r).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls.rows()[r]).cfg.get(V.L1));
                    }
                }
            }
            x = null;
        }
        for (Mat m : fr.values()) m.release();
        if (sm == 2) {
            double est = (smokeS[0] + smokeS[1]) * (10.0 / SMOKE2_BLOCKS) * (1 + FERET_TO_GT);
            StringBuilder t = new StringBuilder(head).append(check).append(diag).append(String.format(Locale.ROOT, "Замер: набор GT, %d блока из 10 на признак – "
                    + "L1-xs-F %.0f с, L1-tt-xs-F %.0f с (с векторами); потоков %d.%nОценка полного прогона: (сумма) · %d · (1 + %.3f) = %.0f с (%.1f ч); %.3f – доля "
                    + "наборов FERET-1 и FERET-2 к GT по pool-poem.%n", SMOKE2_BLOCKS, smokeS[0], smokeS[1], tpl.threads, 10 / SMOKE2_BLOCKS, FERET_TO_GT, est,
                    est / 3600, FERET_TO_GT));
            FarMethods.write(s.out.resolve("xbase_pool_nap" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseNap pool-nap): набор, признак, номер строки, строка\n# Код: коммит " + FarMethods.commit() + "\n");
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("строка | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | "
                    + "контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < 2; f++) {
                double[] crit = new double[R];
                for (int r = 0; r < R; r++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(r))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[r] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", rname(r), XBaseRule.FEATS[f], XBaseRule.fmt(sp.err[0]),
                            XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[r]), XBaseRule.fmt(sp.eer[3]),
                            sp.medOwn, sp.medCtrl));
                }
                int c = choose(crit);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], rname(c),
                        crit[c], crit[BASE], Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(rname(c)).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads))
                .append(body).append(diag);
        if (sm == 0) Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_nap" + sfx + "_scores.tsv.gz"), "# NAP / WCCN, выбор по пулу FERET (XBaseNap pool-nap), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_nap" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_nap" + sfx + "_gt.txt"), new StringBuilder(head).append(gtBody).append(diagGt));
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- прогоны на своих (feret-nap, gt-nap)

    static final String OWN_INFO_HEAD = "\n=== Модели (основная модель): ранг оболочки X / X+mirror | заменено собственных значений S_w в оболочке X / X+mirror | осей "
            + "MLDA X / X+mirror ===\n";

    static final String OWN_NOTE = "На своих – база и строки, выбранные pool-nap хоть в одном случае; все строки – на одной модели W (основные модели X / X+mirror и 5 "
            + "моделей частей,\nпорог П2/П3); помеха S_n – своя у каждой модели (люди пула этой модели).\n";

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        int[] rs = ownRows(select);
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        int poolFrom = s.cohort.size();
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
        Path poemScores = s.out.resolve("xbase_feret_poem_scores.tsv.gz");
        Path poemTxt = s.out.resolve("xbase_feret_poem.txt");
        for (Path f : List.of(poemScores, poemTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-nap: нужен " + f + " (feret-poem)");
        Map<String, String> poemBlocks = XBaseSubsp.blocks(poemScores);
        List<String> poemReport = Files.readAllLines(poemTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-nap %s; seed %d; r0 %s; пул %d; половины %d/%d; выбор %s; строки %s; "
                + "пробный %d", s.su.dir, tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), select, Arrays.toString(rs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_nap_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);

        StringBuilder t = new StringBuilder();
        t.append("П. 2.3, NAP / WCCN, FERET (XBaseEval feret-nap): Color FERET – только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-poem); (ii) s = 3 – как feret-dates (люди с ≥ 4 датами, первые 3 "
                + "даты, пробы с 4-й,\nчасти – номер 71), справочно. «Выбор по пулу» – строка из pool-nap (проход 1 – FERET-1, проход 2 – FERET-2). Контрольные – те "
                + "же 200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = строки (П2) feret-poem («межс.» и (ii) s3), побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("постановка, признак, проход | ранг оболочки | заменено | осей\n");
        StringBuilder diag = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        List<List<XBaseKiss.Pass>> passesBy = new ArrayList<>();
        for (int post = 0; post < 2; post++) {
            List<XBaseKiss.Pass> passes = new ArrayList<>();
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
                passes.add(new XBaseKiss.Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
            }
            passesBy.add(passes);
        }
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            String vn = XBasePoem.vname(f, P2);
            XBaseEval x = null;
            for (int post = 0; post < 2; post++) {
                String pname = post == 0 ? "межс." : "(ii) s3";
                int conf = post == 0 ? 0 : 71;
                List<XBaseKiss.Pass> passes = passesBy.get(post);
                One[] got = new One[2];
                List<Job> jobs = new ArrayList<>();
                List<Integer> todo = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int ps = 0; ps < 2; ps++) {
                    got[ps] = read(cp, String.format(Locale.ROOT, "c%d_p%d_f%d", conf, ps, f));
                    if (got[ps] != null) continue;
                    if (x == null) {
                        long tv = System.nanoTime();
                        x = XBasePoem.vectors(fr, tpl.params, f, new int[] {P2})[P2];
                        System.out.printf(Locale.ROOT, "feret-nap: f%d vectors (%.0f s)%n", f, (System.nanoTime() - tv) / 1e9);
                    }
                    XBaseKiss.Pass p = passes.get(ps);
                    jobs.add(job(p.g(), x, p.gal(), p.probes(), p.parts(), conf, p.extra(), poolFrom, pname + " " + feat + " проход " + (ps + 1)));
                    todo.add(ps);
                }
                if (!jobs.isEmpty()) {
                    List<One> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        got[todo.get(i)] = done.get(i);
                        save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d", conf, todo.get(i), f), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "feret-nap: %s %s ready, %d blocks (%.0f s)%n", pname, feat, jobs.size(), sec);
                }
                XBaseThr.Unit[][] units = new XBaseThr.Unit[R][2];
                for (int ps = 0; ps < 2; ps++) {
                    XBaseKiss.Pass p = passes.get(ps);
                    One ls = got[ps];
                    if (sm != 1) check.append(XBaseReg.check(pname + ", " + feat + ", проход " + (ps + 1) + ", " + vn + ", косинус", ls.rows()[BASE], poemBlocks,
                            poemScores));
                    info.append(infoRow(pname + ", " + feat + ", проход " + (ps + 1), ls));
                    diag.append(ls.diag());
                    for (int r = 0; r < R; r++) {
                        lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ").append(rname(r)).append(", косинус\n")
                                .append(ls.rows()[r]);
                        units[r][ps] = new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), ls.rows()[r], null, true);
                    }
                }
                if (sm >= 2) {
                    t.append(check).append(info).append(diag).append(String.format(Locale.ROOT, "Замер: «межс.» %s, 2 прохода (с векторами) – %.0f с; групп всего 4 "
                            + "(2 признака × 2 постановки).%nОценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по feret-poem.%n", feat,
                            fitS, fitS, FERET_FIRST, fitS / FERET_FIRST, fitS / FERET_FIRST / 3600, FERET_FIRST));
                    FarMethods.write(s.out.resolve("xbase_feret_nap" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                int[] chosen = {select.get("FERET-1").get(feat), select.get("FERET-2").get(feat)};
                if (post == 0) sel.append(String.format(Locale.ROOT, "%s: проход 1 – %s, проход 2 – %s%n", feat, rname(chosen[0]), rname(chosen[1])));
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = feat + " " + rname(BASE);
                for (int r : rs) {
                    String l = feat + " " + rname(r);
                    XBaseThr.Out o = XBaseThr.evaluate(l, List.of(units[r]), persons.size(), tpl);
                    if (r == BASE && sm != 1) {
                        for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(pname + " " + feat + " " + vn, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]),
                                relabel(poemReport, feat + " " + vn, pname + " " + feat + " " + vn), tpl.b, poemTxt));
                    }
                    outs.put(l, o);
                    if (r != BASE) pairs.add(new String[] {l, bl});
                }
                String ch = feat + " " + XBaseLbp.CHOICE;
                outs.put(ch, XBaseThr.evaluate(ch, List.of(units[chosen[0]][0], units[chosen[1]][1]), persons.size(), tpl));
                pairs.add(new String[] {ch, bl});
                XBaseLbp.tables(body, "FERET " + pname + ", " + feat + (post == 1 ? " (справочно; 25 человек)" : ""), outs, persons.size(), tpl, pairs, true);
            }
            x = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(sel).append(info).append(diag);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(body);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_nap" + sfx + "_scores.tsv.gz"), "# FERET, NAP / WCCN (XBaseNap feret-nap), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_nap" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    /**
     * Строки отчёта feret-poem с подписью from → to (в отчёте FERET строки подписаны без постановки; обе постановки различаются
     * числами, checkThr ищет строку целиком до столбца «держит»).
     */
    static List<String> relabel(List<String> report, String from, String to) {
        List<String> out = new ArrayList<>();
        for (String l : report) out.add(l.startsWith(from + " | ") ? to + l.substring(from.length()) : l);
        return out;
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-nap: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        int[] rs = ownRows(select);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new java.util.HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = XBaseVt.trimOwn(g.own, sm);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        int poolFrom = g.cohort.size();
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path poemScores = out.resolve("xbase_gt_poem_scores.tsv.gz");
        Path poemTxt = out.resolve("xbase_gt_poem.txt");
        if (!Files.exists(poemScores) || !Files.exists(poemTxt)) throw new IllegalStateException("gt-nap: нужны " + poemScores + " и " + poemTxt + " (gt-poem)");
        Map<String, String> poemBlocks = XBaseSubsp.blocks(poemScores);
        List<String> poemReport = Files.readAllLines(poemTxt, java.nio.charset.StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-nap; seed %d; r0 %s; пул FERET %d; выбор %s; строки %s; пробный %d", g.seed, g.r0,
                pool.size(), select, Arrays.toString(rs), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_nap" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("П. 2.3, NAP / WCCN, GT (XBaseEval gt-nap)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек (только обучение и помеха; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-poem: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «строка − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append(OWN_NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = строки (П2) gt-poem (межс. и внутри), побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder info = new StringBuilder(OWN_INFO_HEAD).append("режим, признак | ранг оболочки | заменено | осей\n");
        StringBuilder diag = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «строка − база») ===\n");
        double fitS = 0;
        for (int f = 1; f >= 0; f--) {
            String feat = XBaseRule.FEATS[f];
            String vn = XBasePoem.vname(f, P2);
            XBaseEval x = null;
            for (int mi = 0; mi < modes.length; mi++) {
                String mode = (String) modes[mi][0];
                boolean xm = mi == 0;
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
                List<Probe> probes = XBaseEval.probes(ga, pro);
                List<List<Probe>> extra = xm ? XBaseThr.extra(ga, pro, parts) : null;
                String tag = (xm ? "x" : "i") + "_f" + f;
                long t0 = System.nanoTime();
                One ls = read(cp, tag);
                if (ls == null) {
                    if (x == null) {
                        long tv = System.nanoTime();
                        x = XBasePoem.vectors(fr, g.params, f, new int[] {P2})[P2];
                        System.out.printf(Locale.ROOT, "gt-nap: f%d vectors (%.0f s)%n", f, (System.nanoTime() - tv) / 1e9);
                    }
                    ls = run(List.of(job(ga, x, gal, probes, parts, 0, extra, poolFrom, mode + " " + feat)), g.threads).get(0);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    save(cp, tag, ls);
                    System.out.printf(Locale.ROOT, "gt-nap: %s %s ready (%.0f s)%n", mode, feat, sec);
                }
                if (sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + vn + ", косинус", ls.rows()[BASE], poemBlocks, poemScores));
                info.append(infoRow(mode + ", " + feat, ls));
                diag.append(ls.diag());
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                String bl = mode + " " + feat + " " + rname(BASE);
                for (int r : rs) {
                    lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(rname(r)).append(", косинус\n").append(ls.rows()[r]);
                    String l = mode + " " + feat + " " + rname(r);
                    XBaseThr.Out o = XBaseThr.evaluate(l, List.of(new XBaseThr.Unit(ga, gal, pro, parts, ls.rows()[r], null, xm)), g.persons.size(), g);
                    if (r == BASE && sm != 1) {
                        for (int w : xm ? new int[] {0, 2} : new int[] {0}) check.append(XBaseReg.checkThr(mode + " " + feat + " " + vn, XBaseThr.WAYS[w],
                                o.res.get(XBaseThr.WAYS[w]), poemReport, g.b, poemTxt));
                    }
                    outs.put(l, o);
                    if (r != BASE) pairs.add(new String[] {l, bl});
                }
                if (sm >= 2) {
                    t.append(check).append(info).append(diag).append(String.format(Locale.ROOT, "Замер: межс. %s (с векторами) – %.0f с; групп всего 4 (2 признака × "
                            + "2 режима).%nОценка полного прогона: %.0f / %.3f = %.0f с (%.1f ч); %.3f – доля первой группы по gt-poem.%n", feat, fitS, fitS, GT_FIRST,
                            fitS / GT_FIRST, fitS / GT_FIRST / 3600, GT_FIRST));
                    FarMethods.write(out.resolve("xbase_gt_nap" + sfx + ".txt"), t);
                    System.out.print(t);
                    for (Mat m : fr.values()) m.release();
                    cp.delete();
                    return;
                }
                int c = select.get("GT").get(feat);
                String ch = mode + " " + feat + " " + XBaseLbp.CHOICE;
                outs.put(ch, XBaseThr.evaluate(ch, List.of(new XBaseThr.Unit(ga, gal, pro, parts, ls.rows()[c], null, xm)), g.persons.size(), g));
                pairs.add(new String[] {ch, bl});
                if (!xm) {
                    for (String[] p : pairs) {
                        double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], g.persons.size(), g);
                        boolean isChoice = p[0].equals(ch);
                        dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0] + (isChoice ? " (" + rname(c)
                                + ")" : ""), d[0], d[1], isChoice && c == BASE ? "не применяется (выбор – база)" : d[1] <= 3 ? "выполнено" : "НЕ выполнено"));
                    }
                }
                XBaseLbp.tables(body, "GT " + mode + ", " + feat + " (выбор по пулу – " + rname(c) + ")", outs, g.persons.size(), g, pairs, xm);
            }
            x = null;
        }
        for (Mat m : fr.values()) m.release();
        t.append(check).append(dl).append(info).append(diag);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (с векторами, без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_nap" + sfx + "_scores.tsv.gz"), "# GT, NAP / WCCN (XBaseNap gt-nap)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_nap" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
