package svd.recognizer.faces;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Fisherfaces (шаг 5, этап 2, варианты а и б): PCA (оболочка через матрицу Грама), затем LDA до C − 1 через
 * отбеливание S_w и разложение S_b в отбеленном пространстве; без регуляризации (после PCA S_w невырождена;
 * число обусловленности — в info). Оценка — евклидово расстояние до среднего класса галереи.
 *
 * Число компонент PCA: N − C (Fisherfaces по Belhumeur), заданное число или по доле энергии обучающих данных
 * (наименьшее k, при котором сумма σᵢ² первых k — не меньше доли от всей), всегда не больше N − C.
 *
 * @author ssv
 */
final class FisherScorer implements GalleryScorer {

    /** Число компонент PCA; 0 — N − C или по доле энергии. */
    private final int pcaDims;
    /** Доля энергии для числа компонент PCA; 0 — не используется. */
    private final double energy;

    FisherScorer() {
        this(0);
    }

    /** @param pcaDims число компонент PCA (не больше N − C); 0 — N − C */
    FisherScorer(int pcaDims) {
        this.pcaDims = pcaDims;
        this.energy = 0;
    }

    private FisherScorer(double energy) {
        if (!(energy > 0 && energy <= 1)) throw new IllegalArgumentException("Доля энергии PCA вне (0, 1]: " + energy);
        this.pcaDims = 0;
        this.energy = energy;
    }

    /** @param energy доля энергии (0, 1] для числа компонент PCA (не больше N − C) */
    static FisherScorer byEnergy(double energy) {
        return new FisherScorer(energy);
    }

    /** PCA до N − C (без доли энергии и заданного числа). */
    boolean fullPca() {
        return pcaDims == 0 && energy == 0;
    }

    @Override
    public String label() {
        String pca = energy > 0 ? String.format(Locale.ROOT, "по доле энергии %.0f %% (не больше N − C)", energy * 100)
                : pcaDims == 0 ? "до N − C" : "до " + pcaDims;
        return "Fisherfaces: PCA " + pca + ", LDA до C − 1, расстояние до среднего класса";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int n = d.x().length;
        int dims = pcaDims == 0 ? n - d.classes() : Math.min(pcaDims, n - d.classes());
        LdaMath.Span span = LdaMath.span(d, 1e-12, dims);
        String energyInfo = "";
        if (energy > 0) {
            // Вся энергия — сумма квадратов центрированных данных (след матрицы Грама), не только первых N − C.
            double total = 0;
            for (double[] x : d.x()) for (int i = 0; i < x.length; i++) total += (x[i] - d.mean()[i]) * (x[i] - d.mean()[i]);
            double[] sigma = span.sigma();
            int k = 0;
            double acc = 0;
            while (k < sigma.length && acc < energy * total) {
                acc += sigma[k] * sigma[k];
                k++;
            }
            span = truncate(span, k);
            energyInfo = String.format(Locale.ROOT, ", доля энергии %.4f", acc / total);
        }
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        LdaMath.Axes axes = LdaMath.fisherAxes(LdaMath.scatterWithin(span.y(), d.labels(), means), means, d.labels(),
                d.classes() - 1, 0);
        double[] sw = axes.swValues();
        String info = String.format(Locale.ROOT, "N %d, C %d, PCA %d (N − C = %d%s), осей LDA %d; S_w: λ max %.4g, min %.4g, "
                + "обусловленность %.3g", n, d.classes(), span.basis().length, n - d.classes(), energyInfo, axes.a()[0].length,
                sw[0], sw[sw.length - 1], sw[0] / sw[sw.length - 1]);
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.projection(axes.a(), span.basis()), info);
    }

    /** Первые k направлений оболочки. */
    private static LdaMath.Span truncate(LdaMath.Span s, int k) {
        double[][] y = new double[s.y().length][];
        for (int i = 0; i < y.length; i++) y[i] = Arrays.copyOf(s.y()[i], k);
        return new LdaMath.Span(Arrays.copyOf(s.basis(), k), y, Arrays.copyOf(s.sigma(), k));
    }
}
