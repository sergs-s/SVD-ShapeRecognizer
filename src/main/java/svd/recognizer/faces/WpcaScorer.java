package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Отбеливание PCA (этап 5, 4-wpca; запрет на whitening для лиц снят Хозяином 29.09.2026). PCA — на всех обучающих
 * классах (галерея и посторонние когорты), k — по доле энергии 95 % (сумма σᵢ² первых k от всей энергии центрированных
 * данных); λᵢ = σᵢ²/(N − 1); yᵢ = uᵢᵀ(x − μ)/√(λᵢ + δ), δ = c·λ_k. Посторонние участвуют только в PCA.
 *
 * <ul>
 *   <li>COS — эталон человека m — среднее y по его обучающим снимкам; оценка 1 − cos(y, m);</li>
 *   <li>EPS — подпространство человека в отбеленном пространстве (среднее, k по доле энергии 95 %, k ≤ N − 1), оценка —
 *       ошибка реконструкции ε.</li>
 * </ul>
 *
 * @author ssv
 */
final class WpcaScorer implements GalleryScorer {

    enum Mode { COS, EPS }

    static final double ENERGY = 0.95;

    private final Mode mode;
    private final double c;

    WpcaScorer(Mode mode, double c) {
        if (!(c >= 0)) throw new IllegalArgumentException("c < 0: " + c);
        this.mode = mode;
        this.c = c;
    }

    Mode mode() {
        return mode;
    }

    double c() {
        return c;
    }

    @Override
    public String label() {
        return String.format(Locale.ROOT, "отбеливание PCA (энергия %.0f %%, δ = %s·λ_k), %s", ENERGY * 100, fmt(c),
                mode == Mode.COS ? "1 − cos до среднего человека" : "подпространство человека (энергия 95 %, k ≤ N − 1), ε");
    }

    static String fmt(double c) {
        return c == Math.rint(c) ? String.valueOf((long) c) : String.valueOf(c);
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        return model(Pca.of(classes), classes, gallery, mode, c);
    }

    /** PCA обучающих данных: среднее, базис (строки), λᵢ и k по доле энергии. */
    record Pca(double[] mean, double[][] basis, double[] lambda, int k, int n, double energy) {

        static Pca of(List<List<double[]>> classes) {
            LdaMath.Data d = LdaMath.Data.of(classes);
            int nn = d.x().length;
            LdaMath.Span span = LdaMath.span(d, 1e-12, Integer.MAX_VALUE);
            double total = 0;
            for (double[] x : d.x()) for (int i = 0; i < x.length; i++) total += (x[i] - d.mean()[i]) * (x[i] - d.mean()[i]);
            double[] sigma = span.sigma();
            int k = 0;
            double acc = 0;
            while (k < sigma.length && acc < ENERGY * total) {
                acc += sigma[k] * sigma[k];
                k++;
            }
            double[] lambda = new double[k];
            double[][] basis = new double[k][];
            for (int i = 0; i < k; i++) {
                lambda[i] = sigma[i] * sigma[i] / (nn - 1);
                basis[i] = span.basis()[i];
            }
            return new Pca(d.mean(), basis, lambda, k, nn, acc / total);
        }

        /** Отбеленные координаты при δ. */
        double[] whiten(double[] x, double delta) {
            double[] y = new double[k];
            for (int i = 0; i < k; i++) {
                double[] u = basis[i];
                double s = 0;
                for (int t = 0; t < u.length; t++) s += u[t] * (x[t] - mean[t]);
                y[i] = s / Math.sqrt(lambda[i] + delta);
            }
            return y;
        }
    }

    /** Модель режима mode при c по готовой PCA (одна PCA на конфигурацию — для всех c и режимов). */
    static ScoreModel model(Pca pca, List<List<double[]>> classes, int gallery, Mode mode, double c) {
        double delta = c * pca.lambda()[pca.k() - 1];
        List<List<double[]>> y = new ArrayList<>();
        for (int p = 0; p < gallery; p++) {
            List<double[]> list = new ArrayList<>();
            for (double[] x : classes.get(p)) list.add(pca.whiten(x, delta));
            y.add(list);
        }
        String info = String.format(Locale.ROOT, "PCA: N %d, k %d (доля энергии %.4f), λ₁ %.4g, λ_k %.4g, δ %.4g", pca.n(), pca.k(),
                pca.energy(), pca.lambda()[0], pca.lambda()[pca.k() - 1], delta);
        if (mode == Mode.COS) {
            double[][] m = new double[gallery][];
            for (int p = 0; p < gallery; p++) m[p] = y.get(p).isEmpty() ? null : mean(y.get(p));
            return new ScoreModel() {
                @Override
                public double[] scores(double[] x) {
                    return cosScores(pca.whiten(x, delta), m);
                }

                @Override
                public String info() {
                    return info;
                }
            };
        }
        Subspace[] sub = new Subspace[gallery];
        int kMin = Integer.MAX_VALUE;
        int kMax = 0;
        for (int p = 0; p < gallery; p++) {
            sub[p] = Subspace.of(y.get(p));
            if (sub[p] == null) continue;
            kMin = Math.min(kMin, sub[p].basis().length);
            kMax = Math.max(kMax, sub[p].basis().length);
        }
        String infoEps = info + String.format(Locale.ROOT, "; подпространства людей: k от %d до %d", kMin, kMax);
        return new ScoreModel() {
            @Override
            public double[] scores(double[] x) {
                double[] w = pca.whiten(x, delta);
                double[] s = new double[gallery];
                for (int p = 0; p < gallery; p++) s[p] = sub[p] == null ? Double.POSITIVE_INFINITY : sub[p].error(w);
                return s;
            }

            @Override
            public String info() {
                return infoEps;
            }
        };
    }

    /** 1 − cos(y, mₚ) по людям; нет эталона — +∞. */
    static double[] cosScores(double[] y, double[][] m) {
        double ny = Math.sqrt(dot(y, y));
        double[] s = new double[m.length];
        for (int p = 0; p < m.length; p++) {
            s[p] = m[p] == null ? Double.POSITIVE_INFINITY : 1 - dot(y, m[p]) / (ny * Math.sqrt(dot(m[p], m[p])));
        }
        return s;
    }

    /** Подпространство человека: среднее и ортонормированный базис (строки), k по доле энергии, k ≤ N − 1. */
    record Subspace(double[] mean, double[][] basis) {

        static Subspace of(List<double[]> x) {
            if (x.size() < 2) return null;
            LdaMath.Data d = LdaMath.Data.of(List.of(x));
            LdaMath.Span span = LdaMath.span(d, 1e-12, x.size() - 1);
            double total = 0;
            for (double[] v : d.x()) for (int i = 0; i < v.length; i++) total += (v[i] - d.mean()[i]) * (v[i] - d.mean()[i]);
            double[] sigma = span.sigma();
            int k = 0;
            double acc = 0;
            while (k < sigma.length && acc < ENERGY * total) {
                acc += sigma[k] * sigma[k];
                k++;
            }
            double[][] basis = new double[k][];
            for (int i = 0; i < k; i++) basis[i] = span.basis()[i];
            return new Subspace(d.mean(), basis);
        }

        /** ε = ‖(y − μ) − B Bᵀ (y − μ)‖. */
        double error(double[] y) {
            double[] r = new double[y.length];
            for (int i = 0; i < y.length; i++) r[i] = y[i] - mean[i];
            for (double[] b : basis) {
                double s = dot(b, r);
                for (int i = 0; i < r.length; i++) r[i] -= s * b[i];
            }
            return Math.sqrt(dot(r, r));
        }
    }

    static double[] mean(List<double[]> x) {
        double[] m = new double[x.get(0).length];
        for (double[] v : x) for (int i = 0; i < m.length; i++) m[i] += v[i];
        for (int i = 0; i < m.length; i++) m[i] /= x.size();
        return m;
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
