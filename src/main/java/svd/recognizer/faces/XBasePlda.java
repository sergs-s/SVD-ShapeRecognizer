package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseKiss.Pass;
import svd.recognizer.storage.SettingsStore;

/**
 * Двухковариационная PLDA / Joint Bayesian – п. 2.2 ревизии (ответы Жорика 25.10.2026, план с поправками подтверждён Хозяином).
 * Пространство – W текущей модели MLDA; выборка обучения – та же, что у W (галерея, когорта FEI, пул FERET; у X+mirror – отражения
 * отдельными снимками). Предобработка: z = C^(−1/2)(y − m₀) по выборке W, затем z / ‖z‖ (справочно – без нормировки). S_w –
 * объединённая внутриклассовая, S_b = ковариация средних классов − (1/C) Σ S_w / n_c (пол 1e-6 от наибольшего). LLR в замкнутой
 * форме (апостериорное μ по n векторам шаблона); варианты: (Р2) – шаблон один вектор (среднее, n = 1), (Р1) – n векторов (у
 * X+mirror снимок и отражение усреднены в один); решение (-а) – по LLR₁, (-р) – по LLR₁ − LLR₂. В файл оценок – −LLR (меньше –
 * ближе); для разбора по d₁/d₂ – монотонные положительные формы (см. evalA, evalR). Режимы XBaseEval pool-plda, feret-plda, gt-plda.
 *
 * @author ssv
 */
final class XBasePlda {

    /** Варианты в порядке простоты. */
    static final String[] VAR = {"(Р2-а) шаблон-среднее, LLR₁", "(Р2-р) шаблон-среднее, LLR₁ − LLR₂", "(Р1-а) n снимков, LLR₁", "(Р1-р) n снимков, LLR₁ − LLR₂"};
    /** Масштаб S_w′ = κ·S_w (ответ Жорика на №4: переуверенность из-за S_w, сжатой обучением W). */
    static final double[] KS = {3, 10, 30, 100, 300};
    static final int K = KS.length;
    static final int NVAR = VAR.length;
    /** Строки модели: 0 – база; row(v, k) = 1 + v·K + k – вариант v при κ_k с нормировкой длины; + NONORM – без нормировки; FILE + … – файл
     *  (−LLR): (Р2, Р1) × κ с нормировкой, затем без; ROWS – сведения. */
    static final int NV = 1 + NVAR * K;
    static final int NONORM = NVAR * K;
    static final int FILE = 1 + 2 * NVAR * K;
    static final int ROWS = FILE + 4 * K;
    static final String[] VNAME = new String[NV];
    static final int BASE = 0;

    static {
        VNAME[0] = "база (косинус)";
        for (int v = 0; v < NVAR; v++) for (int k = 0; k < K; k++) VNAME[row(v, k)] = VAR[v] + ", κ = " + kappa(k);
    }

    static int row(int v, int k) {
        return 1 + v * K + k;
    }

    static int varOf(int r) {
        return (r - 1) / K;
    }

    static int kOf(int r) {
        return (r - 1) % K;
    }

    static String kappa(int k) {
        return String.format(Locale.ROOT, "%.0f", KS[k]);
    }

    /** Имя строки файла (−LLR): i = (нормировка ? 0 : 2K) + p·K + k. */
    static String fileName(int i) {
        boolean norm = i < 2 * K;
        int j = i % (2 * K);
        return (j / K == 0 ? "(Р2)" : "(Р1)") + " −LLR, κ = " + kappa(j % K) + (norm ? "" : ", без нормировки длины");
    }
    static final double CLAMP = 700;
    static final double SB_FLOOR = 1e-6;
    static final double PD_TOL = 1e-10;
    static final String SELECT = "xbase_pool_plda_select.tsv";

    private XBasePlda() {
    }

    static String rowName(int r) {
        return r < NV ? VNAME[r] : VNAME[r - NONORM] + ", без нормировки длины";
    }

    static int smoke() {
        String v = System.getenv("FACES_PLDA_SMOKE");
        return v == null ? 0 : Integer.parseInt(v.trim());
    }

    static List<String> trim(List<String> own) {
        return smoke() == 1 && own.size() > XBaseDates.SMOKE_N ? own.subList(0, XBaseDates.SMOKE_N) : own;
    }

    // ---------------------------------------------------------------- линейная алгебра

    /** Функция от симметричной матрицы по собственным значениям: V f(λ) Vᵀ. */
    static double[][] fn(LdaMath.Eigen e, java.util.function.DoubleUnaryOperator f) {
        int d = e.values().length;
        double[] g = new double[d];
        for (int k = 0; k < d; k++) g[k] = f.applyAsDouble(e.values()[k]);
        double[][] m = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int k = 0; k < d; k++) s += e.vectors()[i][k] * g[k] * e.vectors()[j][k];
            m[i][j] = s;
            m[j][i] = s;
        }
        return m;
    }

    static double[] mul(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < x.length; j++) s += a[i][j] * x[j];
            y[i] = s;
        }
        return y;
    }

    static double[][] mul(double[][] a, double[][] b) {
        int n = a.length;
        int k = b.length;
        int m = b[0].length;
        double[][] c = new double[n][m];
        for (int i = 0; i < n; i++) for (int t = 0; t < k; t++) {
            double v = a[i][t];
            if (v == 0) continue;
            for (int j = 0; j < m; j++) c[i][j] += v * b[t][j];
        }
        return c;
    }

    static double[][] add(double[][] a, double[][] b, double cb) {
        double[][] c = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a.length; j++) c[i][j] = a[i][j] + cb * b[i][j];
        return c;
    }

    // ---------------------------------------------------------------- модель

    /** Модель PLDA: m, S_b, S_w; сведения о спектрах. */
    record Model(double[] m, double[][] sb, double[][] sw, int raised, double[] swRange, double[] sbRange) {}

    /** Моменты: S_w – объединённая внутриклассовая (N − C), S_b – ковариация средних классов − (1/C) Σ S_w / n_c, пол 1e-6 от наибольшего. */
    static Model fit(List<List<double[]>> classes) {
        int d = -1;
        int nTot = 0;
        int c = 0;
        for (List<double[]> cl : classes) {
            if (cl.isEmpty()) continue;
            d = cl.get(0).length;
            nTot += cl.size();
            c++;
        }
        double[] m = new double[d];
        for (List<double[]> cl : classes) for (double[] v : cl) for (int i = 0; i < d; i++) m[i] += v[i];
        for (int i = 0; i < d; i++) m[i] /= nTot;
        double[][] sw = new double[d][d];
        double[][] sb = new double[d][d];
        double invN = 0;
        List<double[]> means = new ArrayList<>();
        for (List<double[]> cl : classes) {
            if (cl.isEmpty()) continue;
            double[] mc = new double[d];
            for (double[] v : cl) for (int i = 0; i < d; i++) mc[i] += v[i];
            for (int i = 0; i < d; i++) mc[i] /= cl.size();
            means.add(mc);
            invN += 1.0 / cl.size();
            for (double[] v : cl) for (int i = 0; i < d; i++) {
                double a = v[i] - mc[i];
                for (int j = 0; j <= i; j++) sw[i][j] += a * (v[j] - mc[j]);
            }
            for (int i = 0; i < d; i++) {
                double a = mc[i] - m[i];
                for (int j = 0; j <= i; j++) sb[i][j] += a * (mc[j] - m[j]);
            }
        }
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            sw[i][j] /= nTot - c;
            sw[j][i] = sw[i][j];
        }
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            sb[i][j] = sb[i][j] / c - sw[i][j] * invN / c;
            sb[j][i] = sb[i][j];
        }
        LdaMath.Eigen ew = LdaMath.eigen(sw);
        double wMax = Arrays.stream(ew.values()).max().orElse(0);
        double wMin = Arrays.stream(ew.values()).min().orElse(0);
        if (!(wMin > PD_TOL * wMax)) throw new IllegalStateException("XBasePlda: S_w не положительно определена (λ_min " + wMin + ", λ_max " + wMax + ")");
        org.apache.commons.math3.linear.EigenDecomposition eb = new org.apache.commons.math3.linear.EigenDecomposition(
                new org.apache.commons.math3.linear.Array2DRowRealMatrix(sb, false));
        double[] lb = eb.getRealEigenvalues();
        double bMax = Arrays.stream(lb).max().orElse(0);
        double bMin = Arrays.stream(lb).min().orElse(0);
        if (!(bMax > 0)) throw new IllegalStateException("XBasePlda: S_b после поправки без положительных собственных значений");
        int raised = 0;
        double fl = SB_FLOOR * bMax;
        double[][] sbF = new double[d][d];
        for (int k = 0; k < d; k++) {
            double v = lb[k];
            if (v <= fl) {
                v = fl;
                raised++;
            }
            double[] u = eb.getEigenvector(k).toArray();
            for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) sbF[i][j] += v * u[i] * u[j];
        }
        return new Model(m, sbF, sw, raised, new double[] {wMin, wMax}, new double[] {bMin, bMax});
    }

    /** Оценщик LLR: матрицы, общие для всех проб, и кэш по n. */
    static final class Scorer {
        final Model md;
        final double[][] swInv;
        final double[] sbInvM;
        final double[][] q0;
        final double ld0;
        final Map<Integer, double[][][]> byN = new HashMap<>();
        final Map<Integer, Double> ldN = new HashMap<>();

        Scorer(Model md) {
            this.md = md;
            LdaMath.Eigen ew = LdaMath.eigen(md.sw());
            swInv = fn(ew, v -> 1 / v);
            LdaMath.Eigen eb = LdaMath.eigen(md.sb());
            sbInvM = mul(fn(eb, v -> 1 / v), md.m());
            LdaMath.Eigen e0 = LdaMath.eigen(add(md.sb(), md.sw(), 1));
            q0 = fn(e0, v -> 1 / Math.sqrt(v));
            double l = 0;
            for (double v : e0.values()) l += Math.log(v);
            ld0 = l;
        }

        /** [K_n = Λ_n⁻¹, P_n = (S_w + K_n)^(−1/2)] и log det (S_w + K_n). */
        synchronized double[][][] forN(int n) {
            double[][][] r = byN.get(n);
            if (r != null) return r;
            LdaMath.Eigen eb = LdaMath.eigen(md.sb());
            double[][] lam = add(fn(eb, v -> 1 / v), swInv, n);
            LdaMath.Eigen el = LdaMath.eigen(lam);
            double[][] k = fn(el, v -> 1 / v);
            LdaMath.Eigen ep = LdaMath.eigen(add(md.sw(), k, 1));
            double[][] p = fn(ep, v -> 1 / Math.sqrt(v));
            double l = 0;
            for (double v : ep.values()) l += Math.log(v);
            r = new double[][][] {k, p};
            byN.put(n, r);
            ldN.put(n, l);
            return r;
        }

        /** Шаблон с суммой s по n векторам: P_n m_n. */
        double[] template(double[] s, int n) {
            double[][][] kp = forN(n);
            double[] a = mul(swInv, s);
            for (int i = 0; i < a.length; i++) a[i] += sbInvM[i];
            return mul(kp[1], mul(kp[0], a));
        }

        /** Часть LLR пробы, не зависящая от шаблона: ½(‖Q₀(y − m)‖² + log det Σ₀). */
        double prior(double[] y) {
            double[] d = new double[y.length];
            for (int i = 0; i < y.length; i++) d[i] = y[i] - md.m()[i];
            double[] q = mul(q0, d);
            double s = 0;
            for (double v : q) s += v * v;
            return 0.5 * (s + ld0);
        }

        /** LLR пробы y против шаблона (Py = P_n y, t = P_n m_n). */
        double llr(double[] py, double[] t, int n, double prior) {
            double s = 0;
            for (int i = 0; i < py.length; i++) {
                double a = py[i] - t[i];
                s += a * a;
            }
            return -0.5 * (s + ldN.get(n)) + prior;
        }

        /** LLR прямо по сумме s и n (для тестов). */
        double llr(double[] y, double[] s, int n) {
            double[] t = template(s, n);
            double[] py = mul(forN(n)[1], y);
            return llr(py, t, n, prior(y));
        }
    }

    // ---------------------------------------------------------------- формы для разбора

    /** (-а): d_g = 1 + 700 − clamp(LLR_g) (второе – постоянная 1e6 после строк, XBaseAbl.abs); cl[0] – проб с |LLR₁| > 700. */
    static double[] evalA(double[] llr, int[] cl) {
        int b = 0;
        for (int g = 1; g < llr.length; g++) if (llr[g] > llr[b]) b = g;
        if (Math.abs(llr[b]) > CLAMP) cl[0]++;
        double[] v = new double[llr.length];
        for (int g = 0; g < llr.length; g++) v[g] = Double.isInfinite(llr[g]) ? Double.POSITIVE_INFINITY : 1 + CLAMP - Math.max(-CLAMP, Math.min(CLAMP, llr[g]));
        return v;
    }

    /** (-р): d_g = 1 + min(LLR₁ − LLR_g, 700), d₁/d₂ = 1 / (1 + Δ); cl[0] – проб с Δ > 700. */
    static double[] evalR(double[] llr, int[] cl) {
        int b = 0;
        for (int g = 1; g < llr.length; g++) if (llr[g] > llr[b]) b = g;
        double sec = Double.NEGATIVE_INFINITY;
        for (int g = 0; g < llr.length; g++) if (g != b) sec = Math.max(sec, llr[g]);
        if (llr[b] - sec > CLAMP) cl[0]++;
        double[] v = new double[llr.length];
        for (int g = 0; g < llr.length; g++) v[g] = 1 + Math.min(llr[b] - llr[g], CLAMP);
        return v;
    }

    // ---------------------------------------------------------------- одна модель

    /** Предобработка: z = T(y − m₀), при norm – z / ‖z‖. */
    static double[] pre(double[] y, double[] m0, double[][] t, boolean norm) {
        double[] d = new double[y.length];
        for (int i = 0; i < y.length; i++) d[i] = y[i] - m0[i];
        double[] z = mul(t, d);
        if (!norm) return z;
        double s = 0;
        for (double v : z) s += v * v;
        s = Math.sqrt(s);
        if (s == 0) throw new IllegalStateException("XBasePlda: нулевой вектор после отбеливания");
        for (int i = 0; i < z.length; i++) z[i] /= s;
        return z;
    }

    /** Все строки модели (ROWS + сведения); gallery – классы галереи в порядке own, rest – когорта и пул. */
    static String[] model(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> rest,
                          List<Probe> probes, String tag) {
        XBaseRule.Fit f = XBaseRule.fit(x, mirror, gallery, rest, null);
        String[] out = new String[ROWS + 1];
        out[0] = XBaseReg.lines(x, method, s, f.lm, own, probes)[1];
        List<List<String>> classes = new ArrayList<>(gallery);
        classes.addAll(rest);
        List<List<double[]>> yc = new ArrayList<>();
        List<double[]> all = new ArrayList<>();
        for (List<String> cl : classes) {
            List<double[]> l = new ArrayList<>();
            for (String k : cl) {
                l.add(f.lm.embed(x.vec.get(V.L1).get(k)));
                if (mirror) l.add(f.lm.embed(x.mir.get(V.L1).get(k)));
            }
            yc.add(l);
            all.addAll(l);
        }
        int d = all.get(0).length;
        double[] m0 = new double[d];
        for (double[] v : all) for (int i = 0; i < d; i++) m0[i] += v[i];
        for (int i = 0; i < d; i++) m0[i] /= all.size();
        double[][] cv = new double[d][d];
        for (double[] v : all) for (int i = 0; i < d; i++) {
            double a = v[i] - m0[i];
            for (int j = 0; j <= i; j++) cv[i][j] += a * (v[j] - m0[j]);
        }
        for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
            cv[i][j] /= all.size() - 1;
            cv[j][i] = cv[i][j];
        }
        LdaMath.Eigen ec = LdaMath.eigen(cv);
        double cMax = Arrays.stream(ec.values()).max().orElse(0);
        int cFloor = 0;
        for (double v : ec.values()) if (v <= PD_TOL * cMax) cFloor++;
        double[][] t = fn(ec, v -> 1 / Math.sqrt(Math.max(v, PD_TOL * cMax)));
        Map<String, double[]> py = new HashMap<>();
        for (Probe p : probes) py.computeIfAbsent(p.key(), k -> f.lm.embed(x.vec.get(V.L1).get(k)));
        StringBuilder diag = new StringBuilder();
        for (boolean norm : new boolean[] {true, false}) {
            List<List<double[]>> zc = new ArrayList<>();
            for (List<double[]> l : yc) {
                List<double[]> z = new ArrayList<>();
                for (double[] v : l) z.add(pre(v, m0, t, norm));
                zc.add(z);
            }
            Model md = fit(zc);
            // Суммы шаблонов (от κ не зависят): Р2 – среднее всех векторов (с отражениями), n = 1; Р1 – по снимку (у X+mirror – среднее снимка и
            // отражения), n снимков.
            int g = gallery.size();
            double[][] sum2 = new double[g][];
            double[][] sum1 = new double[g][];
            int[] n1 = new int[g];
            for (int q = 0; q < g; q++) {
                List<String> keys = gallery.get(q);
                if (keys.isEmpty()) continue;
                double[] a2 = new double[d];
                double[] a1 = new double[d];
                int cnt = 0;
                for (String k : keys) {
                    double[] zo = pre(f.lm.embed(x.vec.get(V.L1).get(k)), m0, t, norm);
                    double[] snap = zo;
                    for (int i = 0; i < d; i++) a2[i] += zo[i];
                    cnt++;
                    if (mirror) {
                        double[] zm = pre(f.lm.embed(x.mir.get(V.L1).get(k)), m0, t, norm);
                        for (int i = 0; i < d; i++) a2[i] += zm[i];
                        cnt++;
                        snap = new double[d];
                        for (int i = 0; i < d; i++) snap[i] = (zo[i] + zm[i]) / 2;
                    }
                    for (int i = 0; i < d; i++) a1[i] += snap[i];
                }
                for (int i = 0; i < d; i++) a2[i] /= cnt;
                n1[q] = keys.size();
                sum2[q] = a2;
                sum1[q] = a1;
            }
            Map<String, double[]> zp = new HashMap<>();
            for (Probe p : probes) zp.computeIfAbsent(p.key(), k -> pre(py.get(k), m0, t, norm));
            StringBuilder clamps = new StringBuilder();
            for (int kk = 0; kk < K; kk++) {
                double kap = KS[kk];
                double[][] swK = new double[d][d];
                for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) swK[i][j] = kap * md.sw()[i][j];
                Scorer sc = new Scorer(new Model(md.m(), md.sb(), swK, md.raised(), md.swRange(), md.sbRange()));
                double[][] t2 = new double[g][];
                double[][] t1 = new double[g][];
                for (int q = 0; q < g; q++) {
                    if (sum2[q] == null) continue;
                    t2[q] = sc.template(sum2[q], 1);
                    t1[q] = sc.template(sum1[q], n1[q]);
                }
                StringBuilder[] ev = new StringBuilder[NVAR];
                for (int i = 0; i < NVAR; i++) ev[i] = new StringBuilder();
                StringBuilder[] fl = {new StringBuilder(), new StringBuilder()};
                int[][] cl = new int[NVAR][1];
                for (Probe p : probes) {
                    double[] z = zp.get(p.key());
                    double prior = sc.prior(z);
                    double[][] llr = new double[2][g];
                    Map<Integer, double[]> pz = new HashMap<>();
                    pz.put(1, mul(sc.forN(1)[1], z));
                    for (int q = 0; q < g; q++) {
                        if (t2[q] == null) {
                            llr[0][q] = Double.NEGATIVE_INFINITY;
                            llr[1][q] = Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        int nq = n1[q];
                        double[] zq = pz.computeIfAbsent(nq, n -> mul(sc.forN(n)[1], z));
                        llr[0][q] = sc.llr(pz.get(1), t2[q], 1, prior);
                        llr[1][q] = sc.llr(zq, t1[q], nq, prior);
                    }
                    for (int v = 0; v < 2; v++) {
                        double[] neg = new double[g];
                        for (int q = 0; q < g; q++) neg[q] = -llr[v][q];
                        fl[v].append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), own, neg, false));
                        ev[2 * v].append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), own, evalA(llr[v], cl[2 * v]), false));
                        ev[2 * v + 1].append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), own, evalR(llr[v], cl[2 * v + 1]), false));
                    }
                }
                for (int v = 0; v < NVAR; v++) out[row(v, kk) + (norm ? 0 : NONORM)] = v % 2 == 0 ? XBaseAbl.abs(ev[v].toString()) : ev[v].toString();
                for (int p = 0; p < 2; p++) out[FILE + (norm ? 0 : 2 * K) + p * K + kk] = fl[p].toString();
                clamps.append(String.format(Locale.ROOT, "%sκ=%s: %d/%d/%d/%d", kk == 0 ? "" : "; ", kappa(kk), cl[0][0], cl[1][0], cl[2][0], cl[3][0]));
            }
            double trB = 0;
            double trW = 0;
            for (int i = 0; i < d; i++) {
                trB += md.sb()[i][i];
                trW += md.sw()[i][i];
            }
            diag.append(String.format(Locale.ROOT, "%s\t%s\t%d\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%d\t%.4f\t%s%n", tag, norm ? "с нормировкой" : "без нормировки", d, cFloor,
                    md.swRange()[0], md.swRange()[1], md.sbRange()[0], md.sbRange()[1], md.raised(), trB / trW, clamps));
        }
        out[ROWS] = diag.toString();
        return out;
    }

    /** Задачи блока – как XBaseLbp.job (тот же порядок моделей и проб). */
    static List<Callable<String[]>> job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts, int s,
                                        List<List<Probe>> extra) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            String t0 = (m ? "X+mirror" : "X") + " основная";
            tasks.add(() -> model(x, m, method, s, own, fullGallery, cohort, main, t0));
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                String tq = (m ? "X+mirror" : "X") + " без части " + q;
                tasks.add(() -> model(x, m, method, s, g, gk, cohort, pr, tq));
            }
        }
        return tasks;
    }

    static List<String[]> run(List<List<Callable<String[]>>> jobs, int threads) {
        List<Callable<String[]>> flat = new ArrayList<>();
        for (List<Callable<String[]>> j : jobs) flat.addAll(j);
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            List<java.util.concurrent.Future<String[]>> fs = ex.invokeAll(flat);
            List<String[]> out = new ArrayList<>();
            int i = 0;
            for (List<Callable<String[]>> j : jobs) {
                StringBuilder[] b = new StringBuilder[ROWS + 1];
                for (int r = 0; r <= ROWS; r++) b[r] = new StringBuilder();
                for (int k = 0; k < j.size(); k++) {
                    String[] r = fs.get(i++).get();
                    for (int q = 0; q <= ROWS; q++) b[q].append(r[q]);
                }
                String[] o = new String[ROWS + 1];
                for (int r = 0; r <= ROWS; r++) o[r] = b[r].toString();
                out.add(o);
            }
            return out;
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBasePlda: " + e, e);
        } finally {
            ex.shutdown();
        }
    }

    static String[] read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        String[] o = new String[ROWS + 1];
        for (int r = 0; r <= ROWS; r++) {
            o[r] = cp.read(tag + "_" + r);
            if (o[r] == null) return null;
        }
        return o;
    }

    static void save(XBaseOther.Checkpoint cp, String tag, String[] o) throws IOException {
        for (int r = 0; r <= ROWS; r++) cp.save(tag + "_" + r, o[r]);
    }

    static final String DIAG_HEAD = "\n=== Сведения PLDA по моделям (блок | модель | нормировка | размерность W | поднято в C до пола | S_w λ_min | λ_max | S_b после "
            + "поправки λ_min | λ_max | поднято у S_b до 1e-6·λ_max | tr S_b / tr S_w | ограничено ±700 по κ: Р2-а / Р2-р / Р1-а / Р1-р) ===\n";

    static String diag(String block, String d) {
        StringBuilder t = new StringBuilder();
        for (String l : d.lines().toList()) t.append(block).append(" | ").append(l.replace("\t", " | ")).append('\n');
        return t.toString();
    }

    static final String NOTE = "PLDA (двухковариационная, Joint Bayesian) в пространстве W текущей модели MLDA (база и PLDA – одна модель W): выборка W "
            + "(галерея своих, когорта FEI 20, пул FERET – человек один класс по всем датам; у X+mirror отражения – отдельные снимки); z = C^(−1/2)(y − m₀) по "
            + "выборке W, затем z / ‖z‖ (без повторного центрирования); S_w – объединённая внутриклассовая (N − C), S_b – ковариация средних классов − (1/C) Σ "
            + "S_w / n_c, собственные значения ≤ 1e-6·λ_max подняты до 1e-6·λ_max; LLR = log N(y; m_n, S_w + Λ_n⁻¹) − log N(y; m, S_b + S_w), Λ_n = S_b⁻¹ + "
            + "n S_w⁻¹. (Р2) – шаблон один вектор (среднее снимков, у X+mirror – с отражениями), n = 1; (Р1) – n снимков шаблона (у X+mirror снимок и "
            + "отражение усреднены в один). (-а) – решение по LLR₁, (-р) – по LLR₁ − LLR₂; модель X / X+mirror по |r| пробы. Файл оценок – −LLR (меньше – "
            + "ближе). Для разбора по d₁/d₂ (FeiScores.ratio) – монотонные положительные формы без насыщения: (-а) d_g = 1 + 700 − LLR_g (LLR ограничен "
            + "±700), второе расстояние – постоянная 1e6 (как строка (4) в (4а)); (-р) d_g = 1 + min(LLR₁ − LLR_g, 700), d₁/d₂ = 1 / (1 + Δ). Формы "
            + "1 / (1 + e^LLR) и e^(−Δ) не взяты: при |LLR| или Δ больше ≈ 27 отношение уходит ниже 1e-12 и по правилу вырожденных случаев (п. 1) "
            + "склеивается, порядок проб теряется. Масштаб S_w′ = κ·S_w, κ ∈ {3, 10, 30, 100, 300} (S_w, оценённая на обучающей выборке W, сжата "
            + "обучением – PLDA переуверенна; ответ Жорика на №4), S_b – прежняя; одно κ на строку для X и X+mirror.\n";

    // ---------------------------------------------------------------- pool-plda

    /**
     * База в пределах NEAR от лучшего – база; иначе вариант – первый по простоте (Р2-а < Р2-р < Р1-а < Р1-р), у которого хоть одно κ в
     * пределах NEAR от лучшего, κ при нём – с лучшим критерием (κ – параметр калибровки, не сложности; ответ Жорика на №4).
     */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double v : crit) best = Math.min(best, v);
        if (crit[BASE] <= best + XBaseRule.NEAR) return BASE;
        for (int v = 0; v < NVAR; v++) {
            int bk = -1;
            for (int k = 0; k < K; k++) if (bk < 0 || crit[row(v, k)] < crit[row(v, bk)]) bk = k;
            if (crit[row(v, bk)] <= best + XBaseRule.NEAR) return row(v, bk);
        }
        throw new IllegalStateException("XBasePlda: выбор не определён");
    }

    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, false);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(sm == 1 ? all.subList(0, 30) : all));
        sets.put("FERET-1", sm == 1 ? new TreeSet<>(new ArrayList<>(s.halves.get(1)).subList(0, 30)) : s.halves.get(1));
        sets.put("FERET-2", sm == 1 ? new TreeSet<>(new ArrayList<>(s.halves.get(0)).subList(0, 30)) : s.halves.get(0));
        List<String> setNames = new ArrayList<>(sets.keySet());
        Path lbpScores = s.out.resolve("xbase_pool_lbp_scores.tsv.gz");
        if (!Files.exists(lbpScores)) throw new IllegalStateException("pool-plda: для сверки базы нужен " + lbpScores + " (pool-lbp)");
        Map<String, String> blocks = XBaseSubsp.blocks(lbpScores);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-plda %s; seed %d; r0 %s; пул %d; половины %d/%d; пробный %d", s.su.dir,
                tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_plda_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, false);
        XBaseEval[] xv = XBaseKiss.vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();
        StringBuilder head = new StringBuilder();
        head.append("П. 2.2 ревизии: PLDA / Joint Bayesian, выбор варианта по пулу FERET (XBaseEval pool-plda)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                .append('\n');
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы и схема – как pool-rule и pool-lbp: GT – весь пул; FERET-1 – половина B (%d; проход 1 feret-plda); FERET-2 – "
                + "половина A (%d; проход 2);%n5 частей по людям (Random(faces.seed + %d)), половины части – Random(faces.seed + %d + q); галерея своих – первая дата пула.%n",
                s.halves.get(1).size(), s.halves.get(0).size(), XBaseRule.PART_SEED_OFFSET, XBaseRule.HALF_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база, если в пределах %.0f п. п. от лучшего, иначе "
                + "первый по простоте (Р2-а < Р2-р < Р1-а < Р1-р), у которого хоть одно κ в пределах %.0f п. п., κ – с лучшим критерием.%n", XBaseRule.NEAR,
                XBaseRule.NEAR));
        head.append(NOTE);
        StringBuilder check = new StringBuilder("\n=== Проверка: база – строки косинуса = pool-lbp (L1-xs-F – база 7×7, L1-tt-xs-F – вариант (2) 9×9), побайтно ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        List<List<List<List<FeiDecompose.Cfg>>>> cfgs = new ArrayList<>();
        for (int si = 0; si < setNames.size(); si++) {
            List<List<List<FeiDecompose.Cfg>>> a = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                List<List<FeiDecompose.Cfg>> b = new ArrayList<>();
                for (int v = 0; v < NV; v++) b.add(new ArrayList<>());
                a.add(b);
            }
            cfgs.add(a);
        }
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        List<List<String>> noParts = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
        outer:
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            Set<String> members = sets.get(set);
            List<List<String>> parts = XBaseRule.parts5(members, tpl.seed + XBaseRule.PART_SEED_OFFSET);
            for (int f = 1; f >= 0; f--) {
                List<Object[]> blk = new ArrayList<>();
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
                List<List<Callable<String[]>>> jobs = new ArrayList<>();
                List<Object[]> todo = new ArrayList<>();
                for (Object[] b : blk) {
                    if (b[6] != null) continue;
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    jobs.add(job((XBaseEval.Gt) b[2], xv[f], gal, probes, null, 2 * (int) b[0] + (int) b[1], null));
                    todo.add(b);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        todo.get(i)[6] = done.get(i);
                        save(cp, (String) todo.get(i)[5], done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "pool-plda: %s f%d ready, %d blocks (%.0f s)%n", set, f, jobs.size(), sec);
                }
                for (Object[] b : blk) {
                    int q = (int) b[0];
                    int h = (int) b[1];
                    XBaseEval.Gt g = (XBaseEval.Gt) b[2];
                    @SuppressWarnings("unchecked")
                    Map<String, List<String>> gal = (Map<String, List<String>>) b[3];
                    @SuppressWarnings("unchecked")
                    List<Probe> probes = (List<Probe>) b[4];
                    String[] ls = (String[]) b[6];
                    String bp = set + ", " + XBaseRule.FEATS[f] + ", часть " + q + ", смена " + h + ", ";
                    if (sm != 1) check.append(XBaseReg.check(bp + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", ls[0], blocks, lbpScores));
                    dg.append(diag(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, XBaseRule.FEATS[f], q, h), ls[ROWS]));
                    scores.append("# ").append(bp).append(VNAME[0]).append('\n').append(ls[0]);
                    for (int i = 0; i < 2 * K; i++) scores.append("# ").append(bp).append(fileName(i)).append('\n').append(ls[FILE + i]);
                    for (int v = 0; v < NV; v++) cfgs.get(si).get(f).get(v).add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls[v]).cfg.get(V.L1));
                }
                if (sm >= 2) break outer;
            }
        }
        if (sm >= 2) {
            StringBuilder t = new StringBuilder(head).append(check).append(dg).append(String.format(Locale.ROOT, "Замер: первая группа (набор GT, L1-tt-xs-F, 10 блоков) – "
                    + "%.0f с; групп всего 6 (3 набора × 2 признака).%n", fitS));
            FarMethods.write(s.out.resolve("xbase_pool_plda" + sfx + ".txt"), t);
            System.out.print(t);
            cp.delete();
            return;
        }
        StringBuilder body = new StringBuilder();
        StringBuilder gtBody = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBasePlda pool-plda): набор, признак, номер варианта, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        for (int si = 0; si < setNames.size(); si++) {
            String set = setNames.get(si);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(sets.get(set).size()).append(" человек) ########\n");
            tb.append("вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < 2; f++) {
                double[] crit = new double[NV];
                for (int v = 0; v < NV; v++) {
                    XBaseRule.Sep sp = XBaseRule.sep(List.of(XBaseRule.merge(cfgs.get(si).get(f).get(v))), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[v] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", VNAME[v], XBaseRule.FEATS[f], XBaseRule.fmt(sp.err[0]),
                            XBaseRule.fmt(sp.err[1]), XBaseRule.fmt(sp.err[2]), XBaseRule.fmt(sp.err[3]), XBaseRule.fmt(crit[v]), XBaseRule.fmt(sp.eer[3]), sp.medOwn,
                            sp.medCtrl));
                }
                int c = choose(crit);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", set, XBaseRule.FEATS[f], VNAME[c], crit[c],
                        crit[BASE], Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(c).append('\t').append(VNAME[c]).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(check)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body).append(dg);
        if (sm == 0) Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_plda" + sfx + "_scores.tsv.gz"), "# PLDA, выбор по пулу FERET (XBasePlda pool-plda), только локально; "
                + "строки (Р2), (Р1) – −LLR (меньше – ближе)\n# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_plda" + sfx + ".txt"), t);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_plda" + sfx + "_gt.txt"), new StringBuilder(head).append(gtBody));
        System.out.print(t);
        cp.delete();
    }

    /** Выбор по пулу: набор → признак → номер варианта; в пробном режиме 1 – (Р1-р). */
    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (smoke() == 1) {
            Map<String, Map<String, Integer>> o = new LinkedHashMap<>();
            for (String set : List.of("GT", "FERET-1", "FERET-2")) for (String ft : XBaseRule.FEATS) o.computeIfAbsent(set, k -> new LinkedHashMap<>()).put(ft, row(3, 2));
            return o;
        }
        if (!Files.exists(f)) throw new IllegalStateException("XBasePlda: нет файла выбора " + f + " (сначала pool-plda)");
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int v = Integer.parseInt(p[2]);
            if (v < 0 || v >= NV || !VNAME[v].equals(p[3])) throw new IllegalStateException("XBasePlda: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    // ---------------------------------------------------------------- на своих

    /** Таблицы постановки: база, 4 варианта, выбор по пулу, выбор без нормировки длины (если не база); пары «строка − база». */
    static void post(StringBuilder t, String title, String feat, List<Pass> passes, List<String[]> rows, int[] chosen, int persons, XBaseEval.Gt tpl, boolean t2,
                     double[] delta, StringBuilder p3) {
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        String bl = feat + " " + VNAME[BASE];
        // 4 варианта при κ выбора (свой на проход; при выборе базы в проходе – κ другого прохода, иначе κ = 30); сетка κ для варианта выбора.
        int[] kk = new int[chosen.length];
        int vch = -1;
        int other = -1;
        for (int c : chosen) if (c != BASE) {
            other = c;
            if (vch < 0) vch = varOf(c);
        }
        for (int i = 0; i < chosen.length; i++) kk[i] = chosen[i] != BASE ? kOf(chosen[i]) : other >= 0 ? kOf(other) : 2;
        outs.put(bl, XBaseThr.evaluate(bl, XBaseKiss.units(passes, rows, new int[] {0, 0}, t2), persons, tpl));
        for (int v = 0; v < NVAR; v++) {
            int[] rr = new int[chosen.length];
            for (int i = 0; i < chosen.length; i++) rr[i] = row(v, kk[i]);
            String l = feat + " " + VAR[v] + ", κ выбора";
            outs.put(l, XBaseThr.evaluate(l, XBaseKiss.units(passes, rows, rr, t2), persons, tpl));
        }
        if (vch >= 0) for (int k = 0; k < K; k++) {
            int r = row(vch, k);
            String l = feat + " " + VNAME[r] + " (сетка κ)";
            outs.put(l, XBaseThr.evaluate(l, XBaseKiss.units(passes, rows, new int[] {r, r}, t2), persons, tpl));
        }
        String ch = feat + " выбор по пулу";
        outs.put(ch, XBaseThr.evaluate(ch, XBaseKiss.units(passes, rows, chosen, t2), persons, tpl));
        boolean nn = Arrays.stream(chosen).noneMatch(c -> c == BASE);
        if (nn) {
            String chN = feat + " выбор по пулу, без нормировки длины";
            outs.put(chN, XBaseThr.evaluate(chN, XBaseKiss.units(passes, rows, Arrays.stream(chosen).map(c -> c + NONORM).toArray(), t2), persons, tpl));
        }
        List<String[]> pairs = new ArrayList<>();
        for (String k : outs.keySet()) if (!k.equals(bl)) pairs.add(new String[] {k, bl});
        if (p3 != null) {
            String way = t2 ? XBaseThr.WAYS[2] : XBaseThr.WAYS[0];
            for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) {
                FeiThreshold.Res r = e.getValue().res.get(way);
                int x = (int) r.far[3][0][0];
                int n = (int) r.far[3][0][1];
                double up = 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95);
                p3.append(String.format(Locale.ROOT, "%s | %s | %.2f %% (%d/%d) | КП ↑%.2f %% | %s%n", title, e.getKey(), FeiThreshold.farPct(r, 3, 0), x, n, up,
                        up <= 1 ? "держит" : "не держит"));
            }
        }
        if (delta != null) {
            int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
            String way = t2 ? XBaseThr.WAYS[2] : XBaseThr.WAYS[0];
            XBaseRule.Sep a = XBaseRule.sep(outs.get(ch).res.get(way).cfgs, FeiBootstrap.unit(persons), w);
            XBaseRule.Sep z = XBaseRule.sep(outs.get(bl).res.get(way).cfgs, FeiBootstrap.unit(persons), w);
            double[] d = FeiNorm.diff(a.errRep[0], z.errRep[0]);
            delta[0] = a.err[0] - z.err[0];
            delta[1] = d.length == 0 ? Double.NaN : FeiBootstrap.quantile(FeiBootstrap.sorted(d), 0.975);
        }
        XBaseLbp.tables(t, title, outs, persons, tpl, pairs, t2);
    }

    /** FERET: постановка (0 – межс., 2 – (ii) s2, 3 – (ii) s3): проходы как feret-kiss / feret-dates. */
    static List<Pass> feretPasses(XBaseRule.FSetup s, XBaseFeret.Roles ro0, XBaseFeret.Roles ro, XBaseEval.Gt base, Map<String, List<List<String>>> kept, int post,
                                  int conf) {
        List<Pass> passes = new ArrayList<>();
        for (int ps = 0; ps < 2; ps++) {
            Set<String> mine = s.halves.get(ps);
            List<String> own;
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            if (post == 0) {
                own = trim(ro.own1.stream().filter(mine::contains).toList());
                for (String p : own) {
                    gal.put(p, ro.gal1.get(p));
                    pro.put(p, ro.pro1.get(p));
                }
            } else {
                own = trim(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList());
                for (String p : own) {
                    List<String> g = new ArrayList<>();
                    for (int j = 0; j < post; j++) g.addAll(s.pool.get(p).get(j));
                    gal.put(p, XBaseFeret.sorted(g));
                    List<String> pr = new ArrayList<>();
                    for (int j = 3; j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                    pro.put(p, XBaseFeret.sorted(pr));
                }
            }
            Map<String, List<List<String>>> projPool = new TreeMap<>();
            for (String p : s.halves.get(1 - ps)) if (smoke() != 1 || projPool.size() < XBaseDates.SMOKE_POOL) projPool.put(p, s.pool.get(p));
            XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
            List<List<String>> aug = new ArrayList<>(s.cohort);
            aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
            g.cohort = aug;
            List<List<String>> parts = XBaseOther.cvParts(own, s.tpl.seed, XBaseEval.CV_S + conf);
            List<Probe> probes = XBaseEval.probes(g, pro);
            passes.add(new Pass(g, gal, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
        }
        return passes;
    }

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        int sm = smoke();
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, List<List<String>>> kept = XBaseKissDiag.kept(ro0, s.su.byKey);
        Path lbpScores = s.out.resolve("xbase_feret_lbp_scores.tsv.gz");
        Path lbpTxt = s.out.resolve("xbase_feret_lbp.txt");
        Path datesScores = s.out.resolve("xbase_feret_dates_scores.tsv.gz");
        Path datesTxt = s.out.resolve("xbase_feret_dates.txt");
        for (Path f : List.of(lbpScores, lbpTxt, datesScores, datesTxt)) if (!Files.exists(f)) throw new IllegalStateException("feret-plda: нужен " + f);
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        Map<String, String> datesBlocks = XBaseSubsp.blocks(datesScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        List<String> datesReport = Files.readAllLines(datesTxt, StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-plda %s; seed %d; r0 %s; пул %d; половины %d/%d; выбор %s; пробный %d", s.su.dir,
                tpl.seed, tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), select, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_plda_eq" + sfx), fp);
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);
        XBaseEval[] xv = XBaseKiss.vectors(fr, tpl.params);
        for (Mat m : fr.values()) m.release();
        StringBuilder t = new StringBuilder();
        t.append("П. 2.2 ревизии: PLDA / Joint Bayesian, FERET (XBaseEval feret-plda): Color FERET — только локально").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                .append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append("Постановки: «межс.» – роль feret-xs2 (половины №2, части – номер 0, как feret-lbp); (ii) s = 2 и s = 3 – как feret-dates (люди с ≥ 4 датами, "
                + "первые 2 / 3 даты, пробы с 4-й,\nчасти – номера 73 и 71), справочно, 25 человек. «Выбор по пулу» – вариант из pool-plda (проход 1 – FERET-1, проход 2 – "
                + "FERET-2); «без нормировки длины» – тот же\nвариант без z / ‖z‖ (если выбор – не база). Контрольные – те же 200.\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса: B = %d, seed = %d.%n", persons.size(), tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = feret-lbp («межс.») и feret-dates ((ii) s2 (м), (i)=(ii) s3 (м)), строки побайтно; (Т0), (Т2) П3 FAR "
                + "1 % ===\n");
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder p3 = new StringBuilder("\n=== П3 (Т2) FAR 1 %: держит ли FAR контроля (КП – верхняя 95 % граница Клоппера – Пирсона по снимкам) ===\n"
                + "постановка | строка | FAR контроля | КП | итог\n");
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        int[][] postsDef = {{0, 0}, {2, 73}, {3, 71}};
        for (int[] pd : postsDef) {
            int post = pd[0];
            int conf = pd[1];
            String pname = post == 0 ? "межс." : "(ii) s" + post;
            List<Pass> passes = feretPasses(s, ro0, ro, base, kept, post, conf);
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                List<String[]> rows = new ArrayList<>();
                List<List<Callable<String[]>>> jobs = new ArrayList<>();
                List<Integer> todo = new ArrayList<>();
                for (int ps = 0; ps < 2; ps++) {
                    String tag = String.format(Locale.ROOT, "c%d_p%d_f%d", conf, ps, f);
                    String[] r = read(cp, tag);
                    rows.add(r);
                    if (r != null) continue;
                    Pass p = passes.get(ps);
                    jobs.add(job(p.g(), xv[f], p.gal(), p.probes(), p.parts(), conf, p.extra()));
                    todo.add(ps);
                }
                if (!jobs.isEmpty()) {
                    long t0 = System.nanoTime();
                    List<String[]> done = run(jobs, tpl.threads);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    for (int i = 0; i < todo.size(); i++) {
                        rows.set(todo.get(i), done.get(i));
                        save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d", conf, todo.get(i), f), done.get(i));
                    }
                    System.out.printf(Locale.ROOT, "feret-plda: %s %s ready, %d blocks (%.0f s)%n", pname, feat, jobs.size(), sec);
                }
                for (int ps = 0; ps < 2; ps++) {
                    String[] r = rows.get(ps);
                    if (sm != 1) {
                        if (post == 0) check.append(XBaseReg.check("проход " + (ps + 1) + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", r[0],
                                lbpBlocks, lbpScores));
                        else check.append(XBaseReg.check((post == 2 ? "(ii) s2" : "(i)=(ii) s3") + " (м) " + feat + ", проход " + (ps + 1), r[0], datesBlocks,
                                datesScores));
                    }
                    dg.append(diag(pname + " " + feat + " проход " + (ps + 1), r[ROWS]));
                    lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ").append(VNAME[0]).append('\n').append(r[0]);
                    for (int i = 0; i < 4 * K; i++) lines.append("# ").append(pname).append(", ").append(feat).append(", проход ").append(ps + 1).append(", ")
                            .append(fileName(i)).append('\n').append(r[FILE + i]);
                }
                if (sm == 2) {
                    t.append(check).append(dg).append(String.format(Locale.ROOT, "Замер: «межс.» L1-tt-xs-F, 2 прохода – %.0f с; блоков всего 12 (3 постановки × 2 признака "
                            + "× 2 прохода).%n", fitS));
                    FarMethods.write(s.out.resolve("xbase_feret_plda" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int[] chosen = {select.get("FERET-1").get(feat), select.get("FERET-2").get(feat)};
                if (post == 0) sel.append(String.format(Locale.ROOT, "%s: проход 1 – %s, проход 2 – %s%n", feat, VNAME[chosen[0]], VNAME[chosen[1]]));
                if (sm != 1) {
                    XBaseThr.Out o = XBaseThr.evaluate("база", XBaseKiss.units(passes, rows, new int[] {0, 0}, true), persons.size(), tpl);
                    String rl = post == 0 ? "межс. " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]] : (post == 2 ? "(ii) s2" : "(i)=(ii) s3") + " (м) " + feat;
                    for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), post == 0 ? lbpReport : datesReport,
                            tpl.b, post == 0 ? lbpTxt : datesTxt));
                }
                post(body, "FERET " + pname + ", " + feat + (post > 0 ? " (справочно; 25 человек)" : ""), feat, passes, rows, chosen, persons.size(), tpl, true, null,
                        post == 3 ? p3 : null);
            }
        }
        t.append(check).append(sel);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        t.append(p3).append(body).append(dg);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_plda" + sfx + "_scores.tsv.gz"), "# FERET, PLDA (XBasePlda feret-plda), только локально; строки (Р2), (Р1) – "
                + "−LLR (меньше – ближе)\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_plda" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-plda

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-plda: только равные галереи FERET (faces.feret.gallery = eq)");
        int sm = smoke();
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval[] xv = XBaseKiss.vectors(fr, g.params);
        for (Mat m : fr.values()) m.release();
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));
        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        List<String> own = trim(g.own);
        XBaseEval.Gt ga = XBaseDates.withOwn(g, own);
        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        ga.cohort = aug;
        List<List<String>> parts = sm == 1 ? XBaseOther.cvParts(own, g.seed, XBaseEval.CV_S) : FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        Path lbpTxt = out.resolve("xbase_gt_lbp.txt");
        if (!Files.exists(lbpScores) || !Files.exists(lbpTxt)) throw new IllegalStateException("gt-plda: нужны " + lbpScores + " и " + lbpTxt + " (gt-lbp)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-plda; seed %d; r0 %s; пул FERET %d; выбор %s; пробный %d", g.seed, g.r0, pool.size(),
                select, sm));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_plda" + sfx), fp);
        StringBuilder t = new StringBuilder();
        t.append("П. 2.2 ревизии: PLDA / Joint Bayesian, GT (XBaseEval gt-plda)").append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул – Color FERET, ")
                .append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол как gt-lbp: галерея по правилу №8 (seed faces.seed + 9800), межсессионно N = 6 (порог (Т2), рядом (Т0)); внутри сессии N = 3 – контроль, "
                + "порог (Т0);\nусловие Δ – верхняя граница 95 % интервала разности «выбор по пулу − база» разделимости при FAR 1 % внутри сессии ≤ 3 п. п.\n");
        t.append(NOTE);
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка: база = gt-lbp (межс. и внутри), строки побайтно; (Т0), (Т2) П3 FAR 1 % ===\n");
        StringBuilder dg = new StringBuilder(DIAG_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder lines = new StringBuilder();
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «выбор по пулу − база») ===\n");
        double fitS = 0;
        for (int mi = 0; mi < modes.length; mi++) {
            String mode = (String) modes[mi][0];
            boolean x = mi == 0;
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) modes[mi][1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) modes[mi][2];
            List<Probe> probes = XBaseEval.probes(ga, pro);
            Pass pass = new Pass(ga, gal, pro, parts, probes, x ? XBaseThr.extra(ga, pro, parts) : null);
            for (int f = 1; f >= 0; f--) {
                String feat = XBaseRule.FEATS[f];
                String tag = (x ? "x" : "i") + "_f" + f;
                String[] r = read(cp, tag);
                if (r == null) {
                    long t0 = System.nanoTime();
                    r = run(List.of(job(ga, xv[f], gal, probes, parts, 0, pass.extra())), g.threads).get(0);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    save(cp, tag, r);
                    System.out.printf(Locale.ROOT, "gt-plda: %s %s ready (%.0f s)%n", mode, feat, sec);
                }
                if (sm != 1) check.append(XBaseReg.check(mode + ", " + feat + ", " + XBaseLbp.VNAME[XBaseDates.VNO[f]] + ", косинус", r[0], lbpBlocks, lbpScores));
                dg.append(diag(mode + " " + feat, r[ROWS]));
                lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(VNAME[0]).append('\n').append(r[0]);
                for (int i = 0; i < 4 * K; i++) lines.append("# ").append(mode).append(", ").append(feat).append(", ").append(fileName(i)).append('\n').append(r[FILE + i]);
                if (sm == 2) {
                    t.append(check).append(dg).append(String.format(Locale.ROOT, "Замер: межс. L1-tt-xs-F – %.0f с; блоков всего 4 (2 режима × 2 признака).%n", fitS));
                    FarMethods.write(out.resolve("xbase_gt_plda" + sfx + ".txt"), t);
                    System.out.print(t);
                    cp.delete();
                    return;
                }
                int c = select.get("GT").get(feat);
                if (sm != 1) {
                    XBaseThr.Out o = XBaseThr.evaluate("база", List.of(new XBaseThr.Unit(ga, gal, pro, parts, r[0], null, x)), g.persons.size(), g);
                    String rl = mode + " " + feat + " " + XBaseLbp.VNAME[XBaseDates.VNO[f]];
                    for (int w : x ? new int[] {0, 2} : new int[] {0}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), lbpReport, g.b,
                            lbpTxt));
                }
                double[] delta = x ? null : new double[2];
                post(body, "GT " + mode + ", " + feat + " (выбор по пулу – " + VNAME[c] + ")", feat, List.of(pass), Collections.singletonList(r), new int[] {c},
                        g.persons.size(), g, x, delta, null);
                if (delta != null) dl.append(String.format(Locale.ROOT, "%s: выбор – %s; разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", feat, VNAME[c],
                        delta[0], delta[1], c == BASE ? "не применяется (выбор – база)" : delta[1] <= 3 ? "выполнено" : "НЕ выполнено"));
            }
        }
        t.append(check).append(dl);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body).append(dg);
        XBaseOther.writeScores(out.resolve("xbase_gt_plda" + sfx + "_scores.tsv.gz"), "# GT, PLDA (XBasePlda gt-plda); строки (Р2), (Р1) – −LLR (меньше – ближе)\n# Код: "
                + "коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_plda" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
