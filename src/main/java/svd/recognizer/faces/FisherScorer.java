package svd.recognizer.faces;

import java.util.List;
import java.util.Locale;

/**
 * Fisherfaces (шаг 5, этап 2, варианты а и б): PCA до N − C (оболочка через матрицу Грама), затем LDA до
 * C − 1 через отбеливание S_w и разложение S_b в отбеленном пространстве; без регуляризации (после PCA S_w
 * невырождена; число обусловленности — в info). Оценка — евклидово расстояние до среднего класса галереи.
 *
 * @author ssv
 */
final class FisherScorer implements GalleryScorer {

    /** Число компонент PCA; 0 — N − C (Fisherfaces по Belhumeur). */
    private final int pcaDims;

    FisherScorer() {
        this(0);
    }

    /** @param pcaDims число компонент PCA (не больше N − C); 0 — N − C */
    FisherScorer(int pcaDims) {
        this.pcaDims = pcaDims;
    }

    @Override
    public String label() {
        return "Fisherfaces: PCA до " + (pcaDims == 0 ? "N − C" : Integer.toString(pcaDims)) + ", LDA до C − 1, расстояние до среднего класса";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int n = d.x().length;
        int dims = pcaDims == 0 ? n - d.classes() : Math.min(pcaDims, n - d.classes());
        LdaMath.Span span = LdaMath.span(d, 1e-12, dims);
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        LdaMath.Axes axes = LdaMath.fisherAxes(LdaMath.scatterWithin(span.y(), d.labels(), means), means, d.labels(),
                d.classes() - 1, 0);
        double[] sw = axes.swValues();
        String info = String.format(Locale.ROOT, "N %d, C %d, PCA %d (N − C = %d), осей LDA %d; S_w: λ max %.4g, min %.4g, "
                + "обусловленность %.3g", n, d.classes(), span.basis().length, n - d.classes(), axes.a()[0].length,
                sw[0], sw[sw.length - 1], sw[0] / sw[sw.length - 1]);
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.projection(axes.a(), span.basis()), info);
    }
}
