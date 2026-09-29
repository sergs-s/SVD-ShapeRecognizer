package svd.recognizer.faces;

import java.util.List;
import java.util.Locale;

/**
 * MLDA (Thomaz, Kitani, Gillies, 2006; шаг 5, этап 2, вариант в): без PCA; собственные значения λⱼ матрицы S_w
 * заменяются на max(λⱼ, λ̄), λ̄ = tr(S_w)/n, n — полное число пикселей кадра; S*_w — в критерий Фишера вместо S_w;
 * осей до C − 1. Расчёт — в линейной оболочке обучающих векторов: вне неё S_w и S_b нулевые, S*_w = λ̄·I, решение
 * лежит в оболочке (эквивалентность полному расчёту проверяет MldaCheck). Оценка — евклидово расстояние до среднего
 * класса галереи.
 *
 * @author ssv
 */
final class MldaScorer implements GalleryScorer {

    @Override
    public String label() {
        return "MLDA (Thomaz – Kitani – Gillies): S_w с заменой спектра max(λ, λ̄), LDA до C − 1, расстояние до среднего класса";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        return fit(classes, gallery, 1e-10);
    }

    /** relTol — порог ранга оболочки (σᵢ > relTol·σ₁). */
    ScoreModel fit(List<List<double[]>> classes, int gallery, double relTol) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int nFull = d.mean().length;
        LdaMath.Span span = LdaMath.span(d, relTol, Integer.MAX_VALUE);
        double[][] means = LdaMath.classMeans(span.y(), d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(span.y(), d.labels(), means);
        double trace = 0;
        for (int i = 0; i < sw.length; i++) trace += sw[i][i];
        double floor = trace / nFull;
        LdaMath.Axes axes = LdaMath.fisherAxes(sw, means, d.labels(), d.classes() - 1, floor);
        String info = String.format(Locale.ROOT, "N %d, C %d, n %d, ранг оболочки r %d, осей %d; λ̄ = tr(S_w)/n = %.6g; заменено "
                + "собственных значений в оболочке %d из %d (и все %d вне оболочки)", d.x().length, d.classes(), nFull,
                span.basis().length, axes.a()[0].length, floor, axes.replaced(), span.basis().length, nFull - span.basis().length);
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.projection(axes.a(), span.basis()), info);
    }
}
