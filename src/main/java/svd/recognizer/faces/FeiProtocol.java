package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

/**
 * Протокол «только FEI» (этап 5, решение Хозяина 30.09.2026): разбиение людей FEI на своих (галерея) и чужих
 * (когорта, пороговые, контроль). Снимки — только прошедшие отбор по позе (included = 1 в fei_selection.tsv).
 *
 * «Полные» люди — у кого прошли все 8 снимков №4, 5, 6, 7, 11, 12, 13, 14. Разбиение s (0…4): полные
 * перемешиваются (Random(seed + s)), первые 100 — свои; остальные 100 (оставшиеся полные + неполные, по порядку
 * номеров) перемешиваются тем же генератором: когорта 20, пороговые 40, контроль 40. Конфигурация (s, j): у каждого
 * своего на контроле снимок NUMBERS[j], на обучении остальные 7.
 *
 * @author ssv
 */
final class FeiProtocol {

    /** Снимки своих: в конфигурации j на контроле NUMBERS[j], на обучении остальные 7. */
    static final int[] NUMBERS = {4, 5, 6, 7, 11, 12, 13, 14};
    /** Контрольные снимки-анфасы (№11–14); остальные (№4–7) — повороты. */
    static final int FIRST_FRONTAL = 4;
    static final int SPLITS = 5;
    static final int GALLERY = 100;
    static final int COHORT = 20;
    static final int THRESHOLD = 40;
    static final int CONTROL = 40;
    static final int CONFIGS = SPLITS * NUMBERS.length;

    /** Разбиение людей: свои (галерея), когорта (посторонние 2b/2c/wpca и Z-norm), пороговые, контроль. */
    record Split(int index, List<String> gallery, List<String> cohort, List<String> threshold, List<String> control) {}

    private FeiProtocol() {
    }

    /** Полные люди (все 8 номеров NUMBERS прошли отбор), по порядку. */
    static List<String> fullPersons(Map<String, Set<Integer>> included) {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Set<Integer>> e : new TreeMap<>(included).entrySet()) {
            boolean full = true;
            for (int n : NUMBERS) full &= e.getValue().contains(n);
            if (full) out.add(e.getKey());
        }
        return out;
    }

    /** Разбиения 0…SPLITS − 1 (проверка непересечения и размеров — {@link #check}). */
    static List<Split> splits(Map<String, Set<Integer>> included, long seed) {
        List<String> full = fullPersons(included);
        List<String> all = new ArrayList<>(new TreeMap<>(included).keySet());
        if (full.size() < GALLERY || all.size() != GALLERY + COHORT + THRESHOLD + CONTROL) {
            throw new IllegalStateException("FEI: людей " + all.size() + ", полных " + full.size() + "; нужно " + (GALLERY + COHORT
                    + THRESHOLD + CONTROL) + " и не меньше " + GALLERY + " полных");
        }
        List<Split> out = new ArrayList<>();
        for (int s = 0; s < SPLITS; s++) {
            Random rnd = new Random(seed + s);
            List<String> f = new ArrayList<>(full);
            Collections.shuffle(f, rnd);
            List<String> gallery = List.copyOf(f.subList(0, GALLERY));
            Set<String> g = new HashSet<>(gallery);
            List<String> others = new ArrayList<>(all.stream().filter(p -> !g.contains(p)).toList());
            Collections.shuffle(others, rnd);
            Split sp = new Split(s, gallery, List.copyOf(others.subList(0, COHORT)),
                    List.copyOf(others.subList(COHORT, COHORT + THRESHOLD)), List.copyOf(others.subList(COHORT + THRESHOLD, others.size())));
            check(sp, included);
            out.add(sp);
        }
        return out;
    }

    /** Свои и роли чужих не пересекаются, размеры ролей точные, у каждого своего есть все 8 снимков NUMBERS. */
    static void check(Split sp, Map<String, Set<Integer>> included) {
        List<List<String>> roles = List.of(sp.gallery(), sp.cohort(), sp.threshold(), sp.control());
        int[] sizes = {GALLERY, COHORT, THRESHOLD, CONTROL};
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < roles.size(); i++) {
            if (roles.get(i).size() != sizes[i]) throw new IllegalStateException("Разбиение " + sp.index() + ": роль " + i + " не " + sizes[i]);
            for (String p : roles.get(i)) {
                if (!seen.add(p)) throw new IllegalStateException("Разбиение " + sp.index() + ": человек " + p + " в двух ролях");
            }
        }
        for (String p : sp.gallery()) {
            for (int n : NUMBERS) {
                if (!included.get(p).contains(n)) throw new IllegalStateException("Свой " + p + " без снимка №" + n);
            }
        }
    }

    /** Номера обучающих снимков своих в конфигурации j (7 из 8). */
    static List<Integer> trainNumbers(int j) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < NUMBERS.length; i++) if (i != j) out.add(NUMBERS[i]);
        return out;
    }

    /** Контрольный снимок-анфас (№11–14) в конфигурации j. */
    static boolean frontal(int j) {
        return j >= FIRST_FRONTAL;
    }
}
