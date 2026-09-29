package svd.recognizer.faces;

import java.util.List;

/**
 * SFace (1′) — справочная строка FarMethods, как в GalleryEvaluation: шаблон человека — нормированное среднее признаков
 * (не меньше двух снимков), оценка — 1 − cos с шаблоном.
 *
 * @author ssv
 */
final class SFaceScorer implements GalleryScorer {

    @Override
    public String label() {
        return "SFace (1′): шаблон — нормированное среднее, 1 − cos₁";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        double[][] templates = new double[gallery][];
        for (int p = 0; p < gallery; p++) {
            GalleryEvaluation.Model m = GalleryEvaluation.train(null, classes.get(p), true);
            templates[p] = m == null ? null : m.template();
        }
        return x -> {
            double[] s = new double[gallery];
            for (int p = 0; p < gallery; p++) {
                if (templates[p] == null) {
                    s[p] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double dot = 0;
                for (int i = 0; i < x.length; i++) dot += x[i] * templates[p][i];
                s[p] = 1 - dot;
            }
            return s;
        };
    }
}
