package svd.recognizer.faces;

import java.util.List;

/**
 * Гистограммы LBP (п. 4.2, вариант L0): оценка человека — наименьшее расстояние χ² = Σ (a − b)²/(a + b) (бины с a + b = 0
 * пропускаются) от пробы до его снимков галереи; посторонние в обучении не участвуют. Вариант mean (L0-mean, диагностика
 * GT, п. 4.2): χ² до средней гистограммы снимков галереи человека (при +mirror — вместе с отражениями).
 *
 * @author ssv
 */
final class ChiSquareScorer implements GalleryScorer {

    /** true — χ² до средней гистограммы человека (L0-mean); false — наименьшее χ² по снимкам (L0). */
    private final boolean mean;

    ChiSquareScorer() {
        this(false);
    }

    ChiSquareScorer(boolean mean) {
        this.mean = mean;
    }

    @Override
    public String label() {
        return mean ? "χ² гистограмм до средней гистограммы человека" : "χ² гистограмм, наименьшее по снимкам человека";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        List<List<double[]>> refs = List.copyOf(classes.subList(0, gallery));
        if (mean) {
            double[][] avg = new double[gallery][];
            for (int p = 0; p < gallery; p++) avg[p] = average(refs.get(p));
            return x -> {
                double[] s = new double[gallery];
                for (int p = 0; p < gallery; p++) s[p] = chi2(x, avg[p]);
                return s;
            };
        }
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

    /** Поэлементное среднее векторов. */
    static double[] average(List<double[]> v) {
        double[] a = new double[v.get(0).length];
        for (double[] x : v) for (int i = 0; i < a.length; i++) a[i] += x[i];
        for (int i = 0; i < a.length; i++) a[i] /= v.size();
        return a;
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
