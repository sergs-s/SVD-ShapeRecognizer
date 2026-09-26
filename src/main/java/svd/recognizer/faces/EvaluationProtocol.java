package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Разбиения протокола оценки на ORL (все случайности — от одного seed).
 *
 * - Люди перемешиваются; 4 фолда со сдвигом: в фолде f «чужие» — блок f
 *   из 10 человек, остальные 30 — «свои». Каждый человек ровно один раз чужой.
 * - Чужие фолда: первые 5 — валидация, остальные 5 — контроль (все 10 снимков).
 * - У каждого человека снимки перемешиваются один раз (одинаково во всех
 *   фолдах): N — обучение, 2 — валидация, остальные — контроль (при N = 5:
 *   5 / 2 / 3).
 *
 * @author ssv
 */
public final class EvaluationProtocol {

    public static final int FOLDS = 4;
    public static final int VALIDATION_PER_PERSON = 2;
    public static final int VALIDATION_IMPOSTORS = 5;

    private final int[] personOrder;
    private final int[][] imageOrder;
    private final int trainPerPerson;
    private final long seed;

    /**
     * @param seed           seed всех разбиений
     * @param trainPerPerson число обучающих снимков на человека (N ≥ 2,
     *                       N + 2 ≤ 9, чтобы на контроль остался хотя бы 1)
     */
    public EvaluationProtocol(long seed, int trainPerPerson) {
        int maxTrain = OrlDataset.IMAGES_PER_PERSON - VALIDATION_PER_PERSON - 1;
        if (trainPerPerson < 2 || trainPerPerson > maxTrain) {
            throw new IllegalArgumentException("Число обучающих снимков должно быть в [2, "
                    + maxTrain + "], получено " + trainPerPerson);
        }
        this.seed = seed;
        this.trainPerPerson = trainPerPerson;
        Random random = new Random(seed);
        this.personOrder = shuffled(OrlDataset.PERSONS, random);
        this.imageOrder = new int[OrlDataset.PERSONS][];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            imageOrder[p] = shuffled(OrlDataset.IMAGES_PER_PERSON, random);
        }
    }

    private static int[] shuffled(int n, Random random) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        Collections.shuffle(list, random);
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public long getSeed() {
        return seed;
    }

    public int getTrainPerPerson() {
        return trainPerPerson;
    }

    private int blockSize() {
        return OrlDataset.PERSONS / FOLDS;
    }

    /** @return «свои» фолда f (30 человек) в порядке перемешивания */
    public int[] knownPersons(int fold) {
        int[] known = new int[OrlDataset.PERSONS - blockSize()];
        int j = 0;
        for (int i = 0; i < OrlDataset.PERSONS; i++) {
            if (i / blockSize() != fold) {
                known[j++] = personOrder[i];
            }
        }
        return known;
    }

    /** @return «чужие» фолда f для валидации (5 человек) */
    public int[] validationImpostors(int fold) {
        return slice(personOrder, fold * blockSize(), VALIDATION_IMPOSTORS);
    }

    /** @return «чужие» фолда f для контроля (5 человек) */
    public int[] testImpostors(int fold) {
        return slice(personOrder, fold * blockSize() + VALIDATION_IMPOSTORS,
                blockSize() - VALIDATION_IMPOSTORS);
    }

    /** @return обучающие снимки человека */
    public int[] trainImages(int person) {
        return slice(imageOrder[person], 0, trainPerPerson);
    }

    /** @return валидационные снимки человека */
    public int[] validationImages(int person) {
        return slice(imageOrder[person], trainPerPerson, VALIDATION_PER_PERSON);
    }

    /** @return контрольные снимки человека */
    public int[] testImages(int person) {
        int from = trainPerPerson + VALIDATION_PER_PERSON;
        return slice(imageOrder[person], from, OrlDataset.IMAGES_PER_PERSON - from);
    }

    private static int[] slice(int[] source, int from, int length) {
        int[] result = new int[length];
        System.arraycopy(source, from, result, 0, length);
        return result;
    }
}
