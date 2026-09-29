package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Диагностика оценок по людям галереи (шаг 5, FarMethods): AUC своих против чужих и хвосты.
 *
 * @author ssv
 */
final class ScoreDiagnostics {

    /** Сколько худших по AUC людей выводить подробно. */
    static final int WORST = 5;

    private ScoreDiagnostics() {
    }

    /**
     * AUC = P(чужой > свой) + ½ P(чужой = свой) (Манн – Уитни); оценки — «больше хуже».
     *
     * @return AUC; NaN — пустая выборка
     */
    static double auc(double[] genuine, double[] impostor) {
        if (genuine.length == 0 || impostor.length == 0) return Double.NaN;
        double[] g = genuine.clone();
        double[] im = impostor.clone();
        Arrays.sort(g);
        Arrays.sort(im);
        double sum = 0;
        int lo = 0;
        int hi = 0;
        for (double x : g) {
            while (lo < im.length && im[lo] < x) lo++;
            while (hi < im.length && im[hi] <= x) hi++;
            // чужих больше x: im.length − hi; равных: hi − lo.
            sum += (im.length - hi) + 0.5 * (hi - lo);
        }
        return sum / ((double) g.length * im.length);
    }

    /** Квантиль p (ближайший ранг) отсортированного массива. */
    static double quantile(double[] sorted, double p) {
        return sorted[(int) Math.round(p * (sorted.length - 1))];
    }

    /** Доля элементов, строго больших порога. */
    static double shareAbove(double[] v, double t) {
        int n = 0;
        for (double x : v) if (x > t) n++;
        return v.length == 0 ? Double.NaN : n / (double) v.length;
    }

    /**
     * Таблица по людям: медиана AUC и худшие {@link #WORST} подробно.
     *
     * @param names    подписи людей галереи
     * @param genuine  оценки своих проб против человека
     * @param impostor оценки чужих валидации против человека
     */
    static String perPerson(List<String> names, List<List<Double>> genuine, List<List<Double>> impostor) {
        int n = names.size();
        double[] auc = new double[n];
        List<Integer> order = new ArrayList<>();
        List<Double> all = new ArrayList<>();
        int tailQ = 0;
        int tailMin = 0;
        int genTotal = 0;
        for (int i = 0; i < n; i++) {
            double[] g = genuine.get(i).stream().filter(Double::isFinite).mapToDouble(Double::doubleValue).toArray();
            double[] im = impostor.get(i).stream().filter(Double::isFinite).mapToDouble(Double::doubleValue).toArray();
            auc[i] = auc(g, im);
            if (!Double.isNaN(auc[i])) {
                order.add(i);
                all.add(auc[i]);
                Arrays.sort(im);
                for (double x : g) {
                    if (x > quantile(im, 0.05)) tailQ++;
                    if (x > im[0]) tailMin++;
                }
                genTotal += g.length;
            }
        }
        order.sort((a, b) -> Double.compare(auc[a], auc[b]));
        double[] sorted = all.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        StringBuilder t = new StringBuilder();
        if (sorted.length == 0) return "нет данных\n";
        t.append(String.format(Locale.ROOT, "AUC по %d людям: медиана %.4f, минимум %.4f, 10 %% квантиль %.4f, людей с AUC < 0,99: %d, < 0,9: %d.%n",
                sorted.length, quantile(sorted, 0.5), sorted[0], quantile(sorted, 0.1),
                (int) Arrays.stream(sorted).filter(a -> a < 0.99).count(), (int) Arrays.stream(sorted).filter(a -> a < 0.9).count()));
        t.append(String.format(Locale.ROOT, "Своих проб %d: выше 5 %% квантиля чужих своего человека %.1f %%, выше минимума чужих %.1f %%.%n",
                genTotal, 100.0 * tailQ / genTotal, 100.0 * tailMin / genTotal));
        t.append("Худшие по AUC: человек | своих | AUC | свои макс. | чужие мин. / 5 % / медиана | своих выше 5 % чужих / выше мин.\n");
        for (int j = 0; j < Math.min(WORST, order.size()); j++) {
            int i = order.get(j);
            double[] g = genuine.get(i).stream().filter(Double::isFinite).mapToDouble(Double::doubleValue).sorted().toArray();
            double[] im = impostor.get(i).stream().filter(Double::isFinite).mapToDouble(Double::doubleValue).sorted().toArray();
            t.append(String.format(Locale.ROOT, "  %s | %d | %.4f | %.4f | %.4f / %.4f / %.4f | %.2f / %.2f%n", names.get(i), g.length,
                    auc[i], g[g.length - 1], im[0], quantile(im, 0.05), quantile(im, 0.5), shareAbove(g, quantile(im, 0.05)),
                    shareAbove(g, im[0])));
        }
        return t.toString();
    }
}
