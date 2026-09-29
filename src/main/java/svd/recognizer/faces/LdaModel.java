package svd.recognizer.faces;

import java.util.List;

/**
 * Обученная модель LDA (Fisherfaces, MLDA): проекция P (осей × n), общее среднее и центры классов галереи в
 * пространстве LDA; оценка человека — евклидово расстояние до центра его класса (нет снимков — +∞).
 *
 * @author ssv
 */
record LdaModel(double[][] projection, double[] mean, double[][] centers, String info) implements GalleryScorer.ScoreModel {

    static LdaModel of(List<List<double[]>> classes, int gallery, double[] mean, double[][] p, String info) {
        double[][] centers = new double[gallery][];
        for (int c = 0; c < gallery; c++) {
            List<double[]> list = classes.get(c);
            if (list.isEmpty()) continue;
            double[] m = new double[mean.length];
            for (double[] v : list) for (int i = 0; i < m.length; i++) m[i] += v[i];
            for (int i = 0; i < m.length; i++) m[i] /= list.size();
            centers[c] = LdaMath.project(p, m, mean);
        }
        return new LdaModel(p, mean, centers, info);
    }

    /** Координаты пробы в пространстве LDA. */
    double[] embed(double[] x) {
        return LdaMath.project(projection, x, mean);
    }

    @Override
    public double[] scores(double[] x) {
        double[] y = embed(x);
        double[] s = new double[centers.length];
        for (int c = 0; c < centers.length; c++) {
            s[c] = centers[c] == null ? Double.POSITIVE_INFINITY : Math.sqrt(NearestVectorScorer.distance2(y, centers[c]));
        }
        return s;
    }
}
