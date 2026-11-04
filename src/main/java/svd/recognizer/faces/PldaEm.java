package svd.recognizer.faces;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PLDA (El Shafey и др., TPAMI 2013): x = μ + F·h + G·w + ε, ε ~ N(0, diag Σ). Обучение EM на центрированных данных; маргинал набора из n снимков одной личности и LLR
 * «одна личность / разные» пробы против набора снимков регистрации – в замкнутой форме через β = (Σ + GGᵀ)⁻¹:
 * log p(набор) = −½(nD ln 2π + n ln|Σ + GGᵀ| + ln|I + nFᵀβF| + Σⱼ dⱼᵀβdⱼ − UᵀF_nU), U = Fᵀβ Σⱼ dⱼ, F_n = (I + nFᵀβF)⁻¹. Квадратичные члены dᵀβd в LLR сокращаются.
 *
 * @author ssv
 */
final class PldaEm {

    final int dim;
    final int df;
    final int dg;
    double[][] f;
    double[][] g;
    double[] sigma;
    /** Fᵀβ (df × dim). */
    double[][] ftb;
    double[][] ftbf;
    /** Fₙ и ln|I + nFᵀβF| по n. */
    private final Map<Integer, double[][]> fn = new HashMap<>();
    private final Map<Integer, Double> ld = new HashMap<>();
    int iterations;
    double logLik;
    double[] history;

    PldaEm(int dim, int df, int dg) {
        this.dim = dim;
        this.df = df;
        this.dg = dg;
    }

    /**
     * Обучение: classes – классы, снимки центрированы. Начало – по статье (стр. 6): F и G – главные направления разброса между классами и внутри классов, вектор умножен на √(собственное
     * значение), чтобы скрытые переменные имели единичную дисперсию; Σ – дисперсии данных по осям.
     */
    static PldaEm train(List<double[][]> classes, int df, int dg, int maxIter, double relTol) {
        int dim = classes.get(0)[0].length;
        PldaEm m = new PldaEm(dim, df, dg);
        double[] var = new double[dim];
        double[][] sb = new double[dim][dim];
        double[][] sw = new double[dim][dim];
        long n = 0;
        for (double[][] c : classes) {
            double[] mc = new double[dim];
            for (double[] x : c) {
                for (int i = 0; i < dim; i++) {
                    var[i] += x[i] * x[i];
                    mc[i] += x[i] / c.length;
                }
                n++;
            }
            for (int i = 0; i < dim; i++) for (int k = 0; k <= i; k++) sb[i][k] += mc[i] * mc[k];
            for (double[] x : c) {
                for (int i = 0; i < dim; i++) {
                    double di = x[i] - mc[i];
                    for (int k = 0; k <= i; k++) sw[i][k] += di * (x[k] - mc[k]);
                }
            }
        }
        double mean = 0;
        for (int i = 0; i < dim; i++) {
            var[i] /= n;
            mean += var[i] / dim;
            for (int k = 0; k <= i; k++) {
                sb[i][k] /= classes.size();
                sw[i][k] /= n;
                sb[k][i] = sb[i][k];
                sw[k][i] = sw[i][k];
            }
        }
        m.f = topScaled(sb, df);
        m.g = topScaled(sw, dg);
        m.sigma = new double[dim];
        double floor = 1e-10 * Math.max(mean, 1e-300);
        for (int i = 0; i < dim; i++) m.sigma[i] = Math.max(var[i], floor);
        double[] sumXx = new double[dim];
        for (double[][] c : classes) for (double[] x : c) for (int i = 0; i < dim; i++) sumXx[i] += x[i] * x[i];
        double[] hist = new double[maxIter];
        double prev = Double.NaN;
        int it = 0;
        while (it < maxIter) {
            double ll = m.emStep(classes, sumXx, n, floor);
            hist[it++] = ll;
            if (!Double.isNaN(prev) && Math.abs(ll - prev) <= relTol * Math.abs(prev)) {
                prev = ll;
                break;
            }
            prev = ll;
        }
        m.prepare();
        m.iterations = it;
        m.logLik = m.logLikelihood(classes);
        m.history = java.util.Arrays.copyOf(hist, it);
        return m;
    }

    /** Столбцы – k главных собственных векторов, умноженных на √(собственное значение). */
    private static double[][] topScaled(double[][] s, int k) {
        org.apache.commons.math3.linear.EigenDecomposition e = new org.apache.commons.math3.linear.EigenDecomposition(new org.apache.commons.math3.linear.Array2DRowRealMatrix(s, false));
        double[] v = e.getRealEigenvalues();
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(v[b], v[a]));
        double[][] out = new double[s.length][k];
        for (int j = 0; j < k && j < v.length; j++) {
            double l = Math.sqrt(Math.max(v[idx[j]], 0));
            double[] u = e.getEigenvector(idx[j]).toArray();
            for (int i = 0; i < s.length; i++) out[i][j] = l * u[i];
        }
        return out;
    }

    /** Промежуточные матрицы текущих F, G, Σ. */
    private double[][] alpha;
    private double[][] p;
    private double[][] a;
    private double logDetCov;

    private void prepare() {
        fn.clear();
        ld.clear();
        double[][] gtsi = new double[dg][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dg; j++) gtsi[j][i] = g[i][j] / sigma[i];
        double[][] gg = identityPlus(mul(gtsi, g));
        double[][] lg = chol(gg);
        alpha = invFromChol(lg);
        p = mul(alpha, gtsi);
        double logdetS = 0;
        for (double v : sigma) logdetS += Math.log(v);
        logDetCov = logdetS + logDet(lg);
        double[][] ftsi = new double[df][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < df; j++) ftsi[j][i] = f[i][j] / sigma[i];
        double[][] ftsig = mul(ftsi, g);
        double[][] c = mul(ftsig, p);
        ftb = new double[df][dim];
        for (int j = 0; j < df; j++) for (int i = 0; i < dim; i++) ftb[j][i] = ftsi[j][i] - c[j][i];
        ftbf = mul(ftb, f);
        a = mul(p, f);
    }

    /** Fₙ. */
    double[][] fn(int n) {
        return fn.computeIfAbsent(n, k -> {
            double[][] m = new double[df][df];
            for (int i = 0; i < df; i++) {
                for (int j = 0; j < df; j++) m[i][j] = k * ftbf[i][j];
                m[i][i] += 1;
            }
            double[][] l = chol(m);
            ld.put(k, logDet(l));
            return invFromChol(l);
        });
    }

    private double logDetN(int n) {
        fn(n);
        return ld.get(n);
    }

    /** Fᵀβ d. */
    double[] u(double[] d) {
        double[] r = new double[df];
        for (int j = 0; j < df; j++) {
            double[] row = ftb[j];
            double s = 0;
            for (int i = 0; i < dim; i++) s += row[i] * d[i];
            r[j] = s;
        }
        return r;
    }

    private static double quad(double[] v, double[][] m) {
        double s = 0;
        for (int i = 0; i < v.length; i++) {
            double t = 0;
            double[] r = m[i];
            for (int j = 0; j < v.length; j++) t += r[j] * v[j];
            s += v[i] * t;
        }
        return s;
    }

    /** LLR пробы (u – Fᵀβ d) против набора из n снимков (bigU – Fᵀβ Σ d). */
    double llr(double[] bigU, int n, double[] u) {
        double[] sum = new double[df];
        for (int i = 0; i < df; i++) sum[i] = bigU[i] + u[i];
        return 0.5 * (quad(sum, fn(n + 1)) - quad(bigU, fn(n)) - quad(u, fn(1))) - 0.5 * (logDetN(n + 1) - logDetN(n) - logDetN(1));
    }

    double logLikelihood(List<double[][]> classes) {
        double ll = 0;
        for (double[][] c : classes) ll += classLogLik(c);
        return ll;
    }

    /** log p набора одной личности. */
    double classLogLik(double[][] c) {
        int n = c.length;
        double[] sum = new double[dim];
        double q = 0;
        for (double[] d : c) {
            double t = 0;
            for (int i = 0; i < dim; i++) {
                sum[i] += d[i];
                t += d[i] * d[i] / sigma[i];
            }
            double[] gq = new double[dg];
            for (int j = 0; j < dg; j++) {
                double v = 0;
                for (int i = 0; i < dim; i++) v += g[i][j] * d[i] / sigma[i];
                gq[j] = v;
            }
            q += t - quad(gq, alpha);
        }
        double[] bigU = u(sum);
        return -0.5 * (n * dim * Math.log(2 * Math.PI) + n * logDetCov + logDetN(n) + q - quad(bigU, fn(n)));
    }

    /** Один шаг EM; возвращает log p при F, G, Σ до шага (неубывающая последовательность). */
    private double emStep(List<double[][]> classes, double[] sumXx, long nTotal, double floor) {
        prepare();
        int k = df + dg;
        double[][] c = new double[dim][k];
        double[][] r = new double[k][k];
        double ll = 0;
        for (double[][] cl : classes) {
            ll += classLogLik(cl);
            int n = cl.length;
            double[][] fnn = fn(n);
            double[] sum = new double[dim];
            for (double[] d : cl) for (int i = 0; i < dim; i++) sum[i] += d[i];
            double[] eh = new double[df];
            double[] bigU = u(sum);
            for (int i = 0; i < df; i++) {
                double t = 0;
                for (int j = 0; j < df; j++) t += fnn[i][j] * bigU[j];
                eh[i] = t;
            }
            double[] aeh = new double[dg];
            for (int i = 0; i < dg; i++) {
                double t = 0;
                for (int j = 0; j < df; j++) t += a[i][j] * eh[j];
                aeh[i] = t;
            }
            double[] sumW = new double[dg];
            for (double[] d : cl) {
                double[] w = new double[dg];
                for (int i = 0; i < dg; i++) {
                    double[] pi = p[i];
                    double t = 0;
                    for (int q = 0; q < dim; q++) t += pi[q] * d[q];
                    w[i] = t - aeh[i];
                    sumW[i] += w[i];
                }
                for (int i = 0; i < dim; i++) {
                    double di = d[i];
                    double[] ci = c[i];
                    for (int j = 0; j < dg; j++) ci[df + j] += di * w[j];
                }
                for (int i = 0; i < dg; i++) {
                    double[] ri = r[df + i];
                    double wi = w[i];
                    for (int j = 0; j < dg; j++) ri[df + j] += wi * w[j];
                }
            }
            for (int i = 0; i < dim; i++) {
                double si = sum[i];
                double[] ci = c[i];
                for (int j = 0; j < df; j++) ci[j] += si * eh[j];
            }
            double[][] afn = mul(a, fnn);
            double[][] afa = mulT(afn, a);
            for (int i = 0; i < df; i++) {
                for (int j = 0; j < df; j++) r[i][j] += n * (fnn[i][j] + eh[i] * eh[j]);
                for (int j = 0; j < dg; j++) {
                    double v = -n * afn[j][i] + eh[i] * sumW[j];
                    r[i][df + j] += v;
                    r[df + j][i] += v;
                }
            }
            for (int i = 0; i < dg; i++) for (int j = 0; j < dg; j++) r[df + i][df + j] += n * (alpha[i][j] + afa[i][j]);
        }
        double[][] l = chol(r);
        for (int i = 0; i < dim; i++) {
            double[] b = solveChol(l, c[i]);
            for (int j = 0; j < df; j++) f[i][j] = b[j];
            for (int j = 0; j < dg; j++) g[i][j] = b[df + j];
            double t = sumXx[i];
            for (int j = 0; j < k; j++) t -= b[j] * c[i][j];
            sigma[i] = Math.max(t / nTotal, floor);
        }
        return ll;
    }

    // ---------------------------------------------------------------- матрицы

    private static double[][] mul(double[][] a, double[][] b) {
        int n = a.length;
        int m = b.length;
        int w = b[0].length;
        double[][] r = new double[n][w];
        for (int i = 0; i < n; i++) {
            double[] ai = a[i];
            double[] ri = r[i];
            for (int k = 0; k < m; k++) {
                double v = ai[k];
                if (v == 0) continue;
                double[] bk = b[k];
                for (int j = 0; j < w; j++) ri[j] += v * bk[j];
            }
        }
        return r;
    }

    /** a · bᵀ. */
    private static double[][] mulT(double[][] a, double[][] b) {
        double[][] r = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double s = 0;
                for (int k = 0; k < b[j].length; k++) s += a[i][k] * b[j][k];
                r[i][j] = s;
            }
        }
        return r;
    }

    private static double[][] identityPlus(double[][] m) {
        for (int i = 0; i < m.length; i++) m[i][i] += 1;
        return m;
    }

    static double[][] chol(double[][] m) {
        int n = m.length;
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] li = l[i];
            for (int j = 0; j <= i; j++) {
                double[] lj = l[j];
                double s = m[i][j];
                for (int k = 0; k < j; k++) s -= li[k] * lj[k];
                if (i == j) {
                    if (s <= 0) throw new IllegalStateException("PLDA: матрица не положительно определена, " + s);
                    li[i] = Math.sqrt(s);
                } else {
                    li[j] = s / lj[j];
                }
            }
        }
        return l;
    }

    private static double logDet(double[][] l) {
        double s = 0;
        for (int i = 0; i < l.length; i++) s += 2 * Math.log(l[i][i]);
        return s;
    }

    private static double[] solveChol(double[][] l, double[] b) {
        int n = l.length;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            double[] li = l[i];
            double s = b[i];
            for (int k = 0; k < i; k++) s -= li[k] * z[k];
            z[i] = s / li[i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = z[i];
            for (int k = i + 1; k < n; k++) s -= l[k][i] * z[k];
            z[i] = s / l[i][i];
        }
        return z;
    }

    private static double[][] invFromChol(double[][] l) {
        int n = l.length;
        double[][] inv = new double[n][n];
        double[] e = new double[n];
        for (int j = 0; j < n; j++) {
            e[j] = 1;
            double[] x = solveChol(l, e);
            e[j] = 0;
            for (int i = 0; i < n; i++) inv[i][j] = x[i];
        }
        return inv;
    }
}
