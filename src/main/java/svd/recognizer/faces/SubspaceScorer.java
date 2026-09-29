package svd.recognizer.faces;

import java.util.List;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;

/**
 * Подпространство на человека (как в GalleryEvaluation): k = 4 (не больше N − 1), оценка — ошибка
 * реконструкции εᵢ или (ratio) εᵢ / min_{j≠i} εⱼ — у argmin это ровно ε₁/ε₂.
 *
 * @author ssv
 */
final class SubspaceScorer implements GalleryScorer {

    private final boolean ratio;

    SubspaceScorer(boolean ratio) {
        this.ratio = ratio;
    }

    @Override
    public String label() {
        return ratio ? "подпространство на человека, ε₁/ε₂" : "подпространство на человека, ε";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        GalleryEvaluation.Model[] models = new GalleryEvaluation.Model[gallery];
        for (int p = 0; p < gallery; p++) models[p] = GalleryEvaluation.train(trainer, classes.get(p), false);
        return x -> {
            double[] eps = new double[gallery];
            for (int p = 0; p < gallery; p++) {
                GalleryEvaluation.Model m = models[p];
                eps[p] = m == null ? Double.POSITIVE_INFINITY : SubspaceRecognizer.reconstructionError(x, m.mean(), m.basis(), m.k());
            }
            return ratio ? ratio(eps) : eps;
        };
    }

    /** sᵢ = εᵢ / min_{j≠i} εⱼ. */
    static double[] ratio(double[] eps) {
        int best = -1;
        int second = -1;
        for (int p = 0; p < eps.length; p++) {
            if (best < 0 || eps[p] < eps[best]) {
                second = best;
                best = p;
            } else if (second < 0 || eps[p] < eps[second]) {
                second = p;
            }
        }
        double[] s = new double[eps.length];
        for (int p = 0; p < eps.length; p++) s[p] = eps[p] / eps[p == best ? second : best];
        return s;
    }
}
