package svd.recognizer.faces;

import java.util.List;

/**
 * Оценщик галереи (шаг 5, FarMethods): обучается на векторах классов и даёт оценку пробы
 * по каждому человеку галереи (больше — хуже). Решение — argmin по оценкам плюс порог.
 *
 * @author ssv
 */
interface GalleryScorer {

    /** @return подпись для отчёта */
    String label();

    /**
     * Обучение.
     *
     * @param classes векторы по классам: первые {@code gallery} — люди галереи, за ними (если есть) —
     *                посторонние, которые участвуют только в обучении (не кандидаты)
     * @param gallery число людей галереи
     * @return обученная модель
     */
    ScoreModel fit(List<List<double[]>> classes, int gallery);

    /** Обученная модель. */
    interface ScoreModel {

        /**
         * @param x вектор пробы
         * @return оценки по людям галереи (длина — число людей галереи; +∞ — человек без модели)
         */
        double[] scores(double[] x);

        /** @return сведения для отчёта (обусловленность, ранг и т. п.; пусто — нет) */
        default String info() {
            return "";
        }
    }
}
