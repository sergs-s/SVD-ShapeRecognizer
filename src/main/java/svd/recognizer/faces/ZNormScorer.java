package svd.recognizer.faces;

import java.util.List;
import java.util.Locale;

/**
 * Z-norm (шаг 5, этап 3): оценка каждого человека галереи i нормируется по когорте чужих — на модели каждой
 * конфигурации базовый метод даёт оценки sᵢ всех снимков когорты, по ним μᵢ и σᵢ (выборочное, N − 1);
 * zᵢ = (sᵢ − μᵢ)/σᵢ. Решение — i* = argmin zᵢ, итоговая оценка z_{i*}, порог — Нейман – Пирсон (как у прочих
 * методов). Базовая оценка — «сырая» (ε, расстояние LDA); ε₁/ε₂ по z не используется.
 *
 * @author ssv
 */
final class ZNormScorer implements GalleryScorer {

    private final GalleryScorer base;
    private final List<double[]> cohort;
    private final String cohortLabel;

    /**
     * @param base        базовый метод (оценка — больше хуже)
     * @param cohort      векторы снимков когорты (не участвуют ни в обучении базового метода, ни в пороге)
     * @param cohortLabel подпись когорты для отчёта
     */
    ZNormScorer(GalleryScorer base, List<double[]> cohort, String cohortLabel) {
        if (cohort.size() < 2) throw new IllegalArgumentException("Когорта Z-norm меньше 2 снимков");
        this.base = base;
        this.cohort = cohort;
        this.cohortLabel = cohortLabel;
    }

    @Override
    public String label() {
        return "Z-norm (когорта — " + cohortLabel + ", " + cohort.size() + " снимков) над: " + base.label();
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        ScoreModel m = base.fit(classes, gallery);
        double[][] s = cohort.parallelStream().map(m::scores).toArray(double[][]::new);
        double[] mu = new double[gallery];
        double[] sigma = new double[gallery];
        stats(s, mu, sigma);
        double smin = Double.POSITIVE_INFINITY;
        double smax = 0;
        for (double v : sigma) {
            if (!Double.isFinite(v)) continue;
            smin = Math.min(smin, v);
            smax = Math.max(smax, v);
        }
        String info = (m.info().isEmpty() ? "" : m.info() + "; ")
                + String.format(Locale.ROOT, "Z-norm: когорта %d снимков, σᵢ от %.4g до %.4g", cohort.size(), smin, smax);
        return new ScoreModel() {
            @Override
            public double[] scores(double[] x) {
                return normalize(m.scores(x), mu, sigma);
            }

            @Override
            public String info() {
                return info;
            }
        };
    }

    /** μᵢ и σᵢ (выборочное, N − 1) по столбцам s (строка — снимок когорты); бесконечные оценки дают μ = +∞. */
    static void stats(double[][] s, double[] mu, double[] sigma) {
        for (int i = 0; i < mu.length; i++) {
            double sum = 0;
            for (double[] row : s) sum += row[i];
            mu[i] = sum / s.length;
            double ss = 0;
            for (double[] row : s) ss += (row[i] - mu[i]) * (row[i] - mu[i]);
            sigma[i] = Math.sqrt(ss / (s.length - 1));
        }
    }

    /** zᵢ = (sᵢ − μᵢ)/σᵢ; нет модели или σᵢ = 0 — +∞. */
    static double[] normalize(double[] s, double[] mu, double[] sigma) {
        double[] z = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            z[i] = Double.isFinite(s[i]) && Double.isFinite(mu[i]) && sigma[i] > 0 ? (s[i] - mu[i]) / sigma[i] : Double.POSITIVE_INFINITY;
        }
        return z;
    }
}
