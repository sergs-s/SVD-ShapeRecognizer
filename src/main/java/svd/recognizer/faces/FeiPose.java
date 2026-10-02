package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Этап 5б: отобранные снимки FEI и их поза из fei_selection.tsv (экспорт SVD-faces-data): человек → номер снимка → r
 * (смещение носа от середины глаз в долях межглазья, точки YuNet). Только included = 1. Нужен бутстрепу (все люди FEI)
 * и режимам по позе: +mirror-turn (отражается обучающий снимок с |r| > r0) и +pose (выбор модели по |r| пробы).
 *
 * @author ssv
 */
final class FeiPose {

    /** Человек → номер → r; люди и номера по возрастанию. */
    final Map<String, Map<Integer, Double>> r;

    FeiPose(Map<String, Map<Integer, Double>> r) {
        this.r = r;
    }

    static FeiPose read(Path file) throws IOException {
        return parse(Files.readAllLines(file, StandardCharsets.UTF_8));
    }

    /** Строки fei_selection.tsv: «#» и заголовок пропускаются; столбцы person, number, name, found, r, reference, included. */
    static FeiPose parse(List<String> lines) {
        Map<String, Map<Integer, Double>> r = new TreeMap<>();
        for (String l : lines) {
            if (l.isEmpty() || l.startsWith("#") || l.startsWith("person\t")) continue;
            String[] c = l.split("\t", -1);
            if (c.length < 7) throw new IllegalArgumentException("fei_selection.tsv: " + c.length + " столбцов: " + l);
            if (!c[6].equals("1")) continue;
            r.computeIfAbsent(c[0], k -> new TreeMap<>()).put(Integer.parseInt(c[1]), Double.parseDouble(c[4]));
        }
        return new FeiPose(r);
    }

    /** Люди FEI (хотя бы один отобранный снимок), по возрастанию. */
    List<String> persons() {
        return Collections.unmodifiableList(new ArrayList<>(r.keySet()));
    }

    /** |r| снимка; нет такого отобранного снимка — ошибка. */
    double absR(String person, int number) {
        Map<Integer, Double> m = r.get(person);
        Double v = m == null ? null : m.get(number);
        if (v == null) throw new IllegalArgumentException("Нет позы снимка " + person + "-" + number + " в fei_selection.tsv");
        return Math.abs(v);
    }

    /**
     * Порог позы r0 из значения ключа faces.fei.pose.r0 (FACES_FEI_POSE_R0).
     *
     * @param raw  значение ключа или null
     * @param what кому нужен (для сообщения об ошибке)
     * @return r0 — конечное число ≥ 0
     */
    static double r0(String raw, String what) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalStateException(what + ": не задан порог позы faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        }
        double v = Double.parseDouble(raw.trim());
        if (!Double.isFinite(v) || v < 0) throw new IllegalStateException("faces.fei.pose.r0 должен быть конечным и ≥ 0: " + raw);
        return v;
    }

    /** Снимок «повёрнут» относительно r0: |r| > r0 (граница |r| = r0 — анфас). */
    boolean turned(String person, int number, double r0) {
        return absR(person, number) > r0;
    }
}
