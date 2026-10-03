package svd.recognizer.faces;

import java.util.List;

/**
 * Гистограммы LBP (п. 4.2, вариант L0): оценка человека — наименьшее расстояние χ² = Σ (a − b)²/(a + b) (бины с a + b = 0
 * пропускаются) от пробы до его снимков галереи; посторонние в обучении не участвуют.
 *
 * @author ssv
 */
final class ChiSquareScorer implements GalleryScorer {

    @Override
    public String label() {
        return "χ² гистограмм, наименьшее по снимкам человека";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        List<List<double[]>> refs = List.copyOf(classes.subList(0, gallery));
        return x -> {
            double[] s = new double[gallery];
            for (int p = 0; p < gallery; p++) {
                double best = Double.POSITIVE_INFINITY;
                for (double[] r : refs.get(p)) best = Math.min(best, chi2(x, r));
                s[p] = best;
            }
            return s;
        };
    }

    static double chi2(double[] a, double[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            double sum = a[i] + b[i];
            if (sum > 0) {
                double t = a[i] - b[i];
                d += t * t / sum;
            }
        }
        return d;
    }
}
