package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Объединение методов (шаг 5, п. 4.2 PLAN.md). Оценка каждой части по каждому человеку галереи i нормируется
 * по когорте чужих (снимки посторонних MUCT, на модели каждой конфигурации): zᵢ = (sᵢ − μᵢ)/σᵢ, μᵢ и σᵢ (N − 1) —
 * как в Z-norm (ZNormScorer). Контроль и набор порога в нормировке не участвуют.
 *
 * <ul>
 *   <li>SUM — итог по человеку i: сумма zᵢ частей; решение — argmin суммы, порог — Нейман – Пирсон.</li>
 *   <li>AGREE — правило согласия: argmin собственных (ненормированных) оценок у всех частей — один и тот же человек
 *       i*; тогда его итог — сумма z_{i*} частей, у прочих людей — Double.MAX_VALUE (не выбираются, но это не
 *       «нет модели»); при несогласии у всех +∞ (отказ при любом пороге).</li>
 * </ul>
 *
 * @author ssv
 */
final class FusionScorer implements GalleryScorer {

    enum Mode {
        SUM("сумма нормированных оценок, argmin суммы"),
        AGREE("правило согласия: argmin частей совпадает, итог — сумма нормированных оценок, иначе отказ");

        final String label;

        Mode(String label) {
            this.label = label;
        }
    }

    private final List<GalleryScorer> parts;
    private final List<double[]> cohort;
    private final String cohortLabel;
    private final Mode mode;

    FusionScorer(List<GalleryScorer> parts, List<double[]> cohort, String cohortLabel, Mode mode) {
        if (parts.size() < 2) throw new IllegalArgumentException("Объединение меньше двух частей");
        if (cohort.size() < 2) throw new IllegalArgumentException("Когорта нормировки меньше 2 снимков");
        this.parts = List.copyOf(parts);
        this.cohort = cohort;
        this.cohortLabel = cohortLabel;
        this.mode = mode;
    }

    Mode mode() {
        return mode;
    }

    @Override
    public String label() {
        return String.format(Locale.ROOT, "объединение (%s; нормировка — когорта %s, %d снимков) %d частей", mode.label, cohortLabel,
                cohort.size(), parts.size());
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        int n = parts.size();
        List<ScoreModel> models = new ArrayList<>();
        double[][] mu = new double[n][gallery];
        double[][] sigma = new double[n][gallery];
        for (int j = 0; j < n; j++) {
            ScoreModel m = parts.get(j).fit(classes, gallery);
            models.add(m);
            ZNormScorer.stats(cohort.parallelStream().map(m::scores).toArray(double[][]::new), mu[j], sigma[j]);
        }
        return x -> {
            double[][] raw = new double[n][];
            for (int j = 0; j < n; j++) raw[j] = models.get(j).scores(x);
            return combine(raw, mu, sigma, mode);
        };
    }

    /**
     * Итоговые оценки по людям галереи.
     *
     * @param raw   собственные оценки частей (строка — часть)
     * @param mu    средние по когорте (строка — часть)
     * @param sigma СКО по когорте (строка — часть)
     */
    static double[] combine(double[][] raw, double[][] mu, double[][] sigma, Mode mode) {
        int g = raw[0].length;
        double[] sum = new double[g];
        for (int j = 0; j < raw.length; j++) {
            double[] z = ZNormScorer.normalize(raw[j], mu[j], sigma[j]);
            for (int i = 0; i < g; i++) sum[i] += z[i];
        }
        if (mode == Mode.SUM) return sum;
        int agreed = (int) FarMethods.best(raw[0])[1];
        for (int j = 1; j < raw.length; j++) {
            if ((int) FarMethods.best(raw[j])[1] != agreed) {
                agreed = -1;
                break;
            }
        }
        double[] out = new double[g];
        for (int i = 0; i < g; i++) {
            out[i] = agreed < 0 ? Double.POSITIVE_INFINITY : i == agreed ? sum[i] : Double.MAX_VALUE;
        }
        return out;
    }
}
