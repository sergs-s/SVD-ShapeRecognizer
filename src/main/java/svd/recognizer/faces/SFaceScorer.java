package svd.recognizer.faces;

import java.util.List;

/**
 * SFace (1′) — справочная строка FarMethods, как в GalleryEvaluation: шаблон человека — нормированное среднее признаков
 * (не меньше двух снимков), оценка — 1 − cos с шаблоном.
 *
 * @author ssv
 */
final class SFaceScorer implements GalleryScorer {

    /** Наименьшее число снимков шаблона (меньше — шаблона нет, оценка +∞). */
    private final int minImages;

    /** Как прежде: шаблон — из не меньше чем двух снимков. */
    SFaceScorer() {
        this(2);
    }

    /**
     * @param minImages наименьшее число снимков шаблона (1 или 2; FERET — 1, решение Хозяина 12.10.2026: в опоре внутри сессии и
     *                  в сессиях G2-ii у человека бывает один снимок)
     */
    SFaceScorer(int minImages) {
        if (minImages < 1 || minImages > 2) throw new IllegalArgumentException("SFace: минимум снимков шаблона " + minImages);
        this.minImages = minImages;
    }

    @Override
    public String label() {
        return "SFace (1′): шаблон — нормированное среднее, 1 − cos₁";
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        double[][] templates = new double[gallery][];
        for (int p = 0; p < gallery; p++) {
            List<double[]> c = classes.get(p);
            if (c.size() == 1 && minImages == 1) {
                // Один снимок: шаблон — его нормированный признак (как нормированное среднее одного снимка).
                templates[p] = c.get(0).clone();
                OwnEvaluation.normalize(templates[p]);
                continue;
            }
            GalleryEvaluation.Model m = GalleryEvaluation.train(null, c, true);
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
