package svd.recognizer.faces;

import java.util.List;
import java.util.Locale;

/**
 * Этап 5б: файл оценок faces-fei (reports/faces/fei/fei_scores.tsv.gz, TSV + gzip). Одна строка — одна попытка метода в
 * конфигурации (s, j): свой (контрольный снимок j-й из восьми), пороговый чужой или контрольный чужой (все отобранные
 * снимки). Оценка: меньше — ближе; решение — argmin по 100 своим галереи, принят, если оценка ≤ θ.
 *
 * Столбцы: method, s, j, role (own / thr / ctrl), person, image (номер снимка FEI), true_id (для своих — сам человек, у
 * чужих «-»), best_id, best_score, second_id, second_score, true_score (оценка по true_id; у чужих «-»), no_agreement
 * (для 4-agr-*: 1 — согласие не достигнуто, все оценки +∞, best_id «-»; 0 — достигнуто, best_score — итоговая сумма,
 * second_* «-»; у прочих методов «-»). Числа — Double.toString (точно восстанавливаются), бесконечность — Infinity.
 *
 * @author ssv
 */
final class FeiScores {

    static final String FILE = "fei_scores.tsv.gz";
    static final String OWN = "own";
    static final String THR = "thr";
    static final String CTRL = "ctrl";
    static final String NONE = "-";
    static final String COLUMNS = "method\ts\tj\trole\tperson\timage\ttrue_id\tbest_id\tbest_score\tsecond_id\tsecond_score\ttrue_score\t"
            + "no_agreement";

    private FeiScores() {
    }

    /** Строка попытки. */
    record Row(String method, int s, int j, String role, String person, int image, String trueId, String bestId, double best,
               String secondId, double second, double trueScore, int noAgreement) {

        boolean own() {
            return role.equals(OWN);
        }
    }

    /**
     * Строка файла по оценкам попытки (по людям галереи).
     *
     * @param trueIdx индекс своего в галерее; −1 — чужой
     * @param agree   метод согласия (4-agr-*)
     */
    static String line(String method, int s, int j, String role, String person, int image, int trueIdx, List<String> gallery, double[] v,
                       boolean agree) {
        int b = 0;
        for (int p = 1; p < v.length; p++) if (v[p] < v[b]) b = p;
        int c = -1;
        for (int p = 0; p < v.length; p++) if (p != b && (c < 0 || v[p] < v[c])) c = p;
        String bestId = gallery.get(b);
        String secondId = gallery.get(c);
        String second = num(v[c]);
        String flag = NONE;
        if (agree) {
            boolean no = v[b] == Double.POSITIVE_INFINITY;
            flag = no ? "1" : "0";
            if (no) bestId = NONE;
            secondId = NONE;
            second = NONE;
        }
        return String.join("\t", method, Integer.toString(s), Integer.toString(j), role, person, Integer.toString(image),
                trueIdx < 0 ? NONE : gallery.get(trueIdx), bestId, num(v[b]), secondId, second, trueIdx < 0 ? NONE : num(v[trueIdx]), flag)
                + "\n";
    }

    static String num(double x) {
        return Double.toString(x);
    }

    /** Разбор строки файла (не заголовка). */
    static Row parse(String line) {
        String[] c = line.split("\t", -1);
        if (c.length != 13) throw new IllegalArgumentException("Строка файла оценок: " + c.length + " столбцов вместо 13: " + line);
        return new Row(c[0], Integer.parseInt(c[1]), Integer.parseInt(c[2]), c[3], c[4], Integer.parseInt(c[5]), c[6], c[7], dbl(c[8]), c[9],
                dbl(c[10]), dbl(c[11]), c[12].equals(NONE) ? -1 : Integer.parseInt(c[12]));
    }

    private static double dbl(String s) {
        return s.equals(NONE) ? Double.NaN : Double.parseDouble(s);
    }

    /** Шапка файла (строки с «#») и строка столбцов. */
    static String header(String code, String data, int configs, List<String> methods) {
        StringBuilder t = new StringBuilder();
        t.append("# Этап 5б: оценки попыток faces-fei (FeiMethods), протокол FEI (PLAN.md, п. 3а)\n");
        t.append("# Код: коммит ").append(code).append('\n');
        t.append("# Данные: ").append(data).append('\n');
        t.append(String.format(Locale.ROOT, "# Конфигураций: %d из %d; s — разбиение (0…%d), j — индекс контрольного снимка своих в (4, 5, 6, 7, 11, 12, "
                + "13, 14)%n", configs, FeiProtocol.CONFIGS, FeiProtocol.SPLITS - 1));
        t.append("# Направление оценки: меньше — ближе; argmin по 100 своим галереи; попытка принята, если оценка лучшего ≤ θ\n");
        t.append("# Роли: own — свой (контрольный снимок), thr — пороговый чужой (40), ctrl — контрольный чужой (40); у чужих все отобранные "
                + "снимки\n");
        t.append("# true_score — оценка по истинному человеку (FeiMethods: свой отклонён, если она бесконечна или best_score > θ)\n");
        t.append("# 4-agr-*: no_agreement = 1 — согласие не достигнуто (все оценки +∞), 0 — достигнуто (best_score — сумма нормированных, "
                + "у прочих людей Double.MAX_VALUE, second_* «-»)\n");
        t.append("# Методы (").append(methods.size()).append("): ").append(String.join(", ", methods)).append('\n');
        t.append(COLUMNS).append('\n');
        return t.toString();
    }
}
