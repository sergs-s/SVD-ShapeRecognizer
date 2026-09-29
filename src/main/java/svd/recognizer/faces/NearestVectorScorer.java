package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.List;

/**
 * Базовая линия «простой SVD по векторам» (этап 1а): каждый снимок галереи — вектор; оценка человека —
 * евклидово расстояние до его ближайшего снимка или (means) до его среднего вектора.
 *
 * @author ssv
 */
final class NearestVectorScorer implements GalleryScorer {

    private final boolean means;

    NearestVectorScorer(boolean means) {
        this.means = means;
    }

    @Override
    public String label() {
        return means ? "ближайший средний вектор человека, евклидово расстояние" : "ближайший снимок галереи, евклидово расстояние";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        List<List<double[]>> refs = new ArrayList<>();
        for (int p = 0; p < gallery; p++) {
            List<double[]> c = classes.get(p);
            if (means && !c.isEmpty()) {
                double[] m = new double[c.get(0).length];
                for (double[] v : c) for (int i = 0; i < m.length; i++) m[i] += v[i];
                for (int i = 0; i < m.length; i++) m[i] /= c.size();
                refs.add(List.of(m));
            } else {
                refs.add(c);
            }
        }
        return x -> {
            double[] s = new double[gallery];
            for (int p = 0; p < gallery; p++) {
                double best = Double.POSITIVE_INFINITY;
                for (double[] r : refs.get(p)) best = Math.min(best, distance2(x, r));
                s[p] = Math.sqrt(best);
            }
            return s;
        };
    }

    static double distance2(double[] a, double[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            double t = a[i] - b[i];
            d += t * t;
        }
        return d;
    }
}
