package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Проверка эквивалентности MLDA (шаг 5, этап 2, конфигурация 2c-check): расчёт в линейной оболочке обучающих
 * векторов (MldaScorer) против прямого расчёта в полном пространстве (S_w и S_b n × n, замена всего спектра,
 * обобщённая задача через отбеливание S*_w и разложение полной WᵀS_bW). Кадры а уменьшаются до 32×32 (n = 1024), конфигурация 0; два набора:
 * только галерея (N < n — сокращение работает) и галерея + 69 посторонних. Сравниваются подпространства решений
 * (наибольший синус главного угла) и расстояния до центров классов галереи по всем снимкам с лицом.
 *
 * @author ssv
 */
final class MldaCheck {

    static final int SIDE = 32;
    /** Допуск: наибольший синус главного угла и наибольшее относительное расхождение расстояний. */
    static final double TOLERANCE = 1e-6;

    /** Итог проверки: текст, прошла ли, наибольшие расхождения по обоим наборам. */
    record Outcome(String text, boolean passed, double maxSin, double maxRel) {}

    private MldaCheck() {
    }

    static Outcome run(FarMethods fm, IlluminationNorm norm) {
        Map<String, double[]> v = new HashMap<>();
        for (Map.Entry<String, Mat> e : fm.frames.entrySet()) {
            Mat small = new Mat();
            Imgproc.resize(e.getValue(), small, new Size(SIDE, SIDE), 0, 0, Imgproc.INTER_AREA);
            v.put(e.getKey(), norm.vector(small, fm.params));
            small.release();
        }
        List<double[]> probes = new ArrayList<>(v.values());
        StringBuilder t = new StringBuilder();
        double worstSin = 0;
        double worstRel = 0;
        t.append(String.format(Locale.ROOT, "Проверка эквивалентности MLDA: кадры а → %d×%d (n = %d), нормализация — %s, конфигурация 0, "
                + "снимков-проб %d.%n", SIDE, SIDE, SIDE * SIDE, norm.label, probes.size()));
        for (boolean outsiders : new boolean[] {false, true}) {
            List<List<double[]>> classes = fm.classes(0, v, outsiders);
            int gallery = fm.gallerySize();
            LdaModel span = (LdaModel) new MldaScorer().fit(classes, gallery);
            LdaModel full = full(classes, gallery);
            double sin = LdaMath.maxPrincipalSin(span.projection(), full.projection());
            double maxRel = 0;
            double maxAbs = 0;
            for (double[] x : probes) {
                double[] a = span.scores(x);
                double[] b = full.scores(x);
                for (int i = 0; i < a.length; i++) {
                    if (!Double.isFinite(b[i])) continue;
                    double d = Math.abs(a[i] - b[i]);
                    maxAbs = Math.max(maxAbs, d);
                    maxRel = Math.max(maxRel, d / Math.max(Math.abs(b[i]), Double.MIN_NORMAL));
                }
            }
            t.append(String.format(Locale.ROOT, "  %s: оболочка — %s; полное пространство — %s.%n    Наибольший синус главного угла между "
                    + "подпространствами решений %.3e; расстояния до центров галереи: наибольшее относительное расхождение %.3e, "
                    + "абсолютное %.3e.%n", outsiders ? "галерея + 69 посторонних" : "только галерея", span.info(), full.info(), sin,
                    maxRel, maxAbs));
            worstSin = Math.max(worstSin, sin);
            worstRel = Math.max(worstRel, maxRel);
        }
        boolean passed = worstSin <= TOLERANCE && worstRel <= TOLERANCE;
        return new Outcome(t.toString(), passed, worstSin, worstRel);
    }

    /** Прямой расчёт MLDA в полном пространстве n × n. */
    static LdaModel full(List<List<double[]>> classes, int gallery) {
        LdaMath.Data d = LdaMath.Data.of(classes);
        int n = d.mean().length;
        double[][] y = new double[d.x().length][n];
        for (int k = 0; k < y.length; k++) for (int i = 0; i < n; i++) y[k][i] = d.x()[k][i] - d.mean()[i];
        double[][] means = LdaMath.classMeans(y, d.labels(), d.classes());
        double[][] sw = LdaMath.scatterWithin(y, d.labels(), means);
        double trace = 0;
        for (int i = 0; i < n; i++) trace += sw[i][i];
        double floor = trace / n;
        LdaMath.Axes axes = LdaMath.fisherAxesFull(sw, LdaMath.scatterBetween(means, d.labels()), d.classes() - 1, floor);
        String info = String.format(Locale.ROOT, "N %d, C %d, λ̄ %.6g, заменено %d из %d", d.x().length, d.classes(), floor,
                axes.replaced(), n);
        return LdaModel.of(classes, gallery, d.mean(), LdaMath.transpose(axes.a()), info);
    }
}
