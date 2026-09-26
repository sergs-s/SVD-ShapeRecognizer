package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * Проверка протокола и статистики оценки на лицах (без OpenCV и базы).
 *
 * @author ssv
 */
public class FaceEvaluationTest {

    @Test
    public void neymanPearsonThresholdKeepsFarWithinAlpha() {
        double[] impostors = {5, 1, 4, 2, 3, 6, 7, 8, 9, 10}; // n = 10
        // α = 0: ни один чужой не принят — θ чуть меньше минимума.
        assertTrue(FaceEvaluation.neymanPearsonThreshold(impostors, 0.0) < 1.0);
        // α = 0.2: можно принять двух — θ чуть меньше третьего по величине.
        double theta = FaceEvaluation.neymanPearsonThreshold(impostors, 0.2);
        assertTrue(theta >= 2.0 && theta < 3.0);
    }

    @Test
    public void binomialUpperBoundMatchesKnownValues() {
        // x = 0: 1 − 0.05^(1/n), близко к правилу трёх.
        assertEquals(1 - Math.pow(0.05, 1.0 / 50), FaceEvaluation.binomialUpperBound(0, 50, 0.95), 1e-9);
        // Клоппер – Пирсон, x = 1, n = 50, одностороннее 95 %: ≈ 0.0913.
        assertEquals(0.0913, FaceEvaluation.binomialUpperBound(1, 50, 0.95), 5e-4);
        assertEquals(1.0, FaceEvaluation.binomialUpperBound(5, 5, 0.95), 0.0);
    }

    @Test
    public void evaluateCountsPeopleGroupsAndPooledThreshold() {
        List<FaceEvaluation.Probe> probes = new ArrayList<>();
        for (int f = 0; f < EvaluationProtocol.FOLDS; f++) {
            // Валидация фолда: чужие с ε = 10 + f и 20.
            probes.add(probe(f, FaceEvaluation.Role.IMPOSTOR_VAL, 200 + f, 10 + f, 30));
            probes.add(probe(f, FaceEvaluation.Role.IMPOSTOR_VAL, 200 + f, 20, 30));
            // Контроль: у чужого 100+f один снимок близко (ε = 12), другой далеко.
            probes.add(probe(f, FaceEvaluation.Role.IMPOSTOR_TEST, 100 + f, 12, 13));
            probes.add(probe(f, FaceEvaluation.Role.IMPOSTOR_TEST, 100 + f, 50, 60));
            // Свой f: ε = 11, верный argmin.
            probes.add(probe(f, FaceEvaluation.Role.KNOWN_TEST, f, 11, 40));
        }
        Set<Integer> participants = Set.of(0, 1); // свои 0 и 1 — группа g1

        // per-fold, α = 0: θ_f чуть меньше 10 + f.
        FaceEvaluation.Counts[] perFold = FaceEvaluation.evaluate(probes,
                FaceEvaluation.Score.EPS, FaceEvaluation.ThresholdMode.PER_FOLD, 0.0, participants);
        FaceEvaluation.Counts total = perFold[EvaluationProtocol.FOLDS];
        assertEquals(4, total.impostorPeople);
        assertEquals(1, total.impostorPeopleAccepted); // только фолд 3: θ < 13, ε = 12 принят
        assertEquals(1, total.falseAccept);
        assertEquals(2, total.falseReject);             // ε = 11 отвергнут в фолдах 0 и 1
        assertEquals(2, total.knownTestG1);
        assertEquals(2, total.falseRejectG1);           // свои 0 и 1 — это и есть фолды 0 и 1
        assertEquals(0, total.falseRejectG2);

        // pooled, α = 0: общий θ чуть меньше min(10, 11, 12, 13) = 10 — всё отвергнуто.
        FaceEvaluation.Counts[] pooled = FaceEvaluation.evaluate(probes,
                FaceEvaluation.Score.EPS, FaceEvaluation.ThresholdMode.POOLED, 0.0, participants);
        for (FaceEvaluation.Counts c : pooled) {
            assertTrue(c.theta < 10.0 && c.theta > 9.99);
        }
        assertEquals(0, pooled[EvaluationProtocol.FOLDS].impostorPeopleAccepted);
        assertEquals(4, pooled[EvaluationProtocol.FOLDS].falseReject);

        // ratio = ε₁/ε₂.
        assertEquals(12.0 / 13.0, FaceEvaluation.Score.RATIO.of(probes.get(2)), 1e-12);
    }

    private static FaceEvaluation.Probe probe(int fold, FaceEvaluation.Role role, int person,
                                              double best, double second) {
        return new FaceEvaluation.Probe(fold, role, person, 0, person, best, second);
    }

    @Test
    public void protocolSplitsAreDisjointAndCoverEveryone() {
        EvaluationProtocol protocol = new EvaluationProtocol(42L, 5);
        Set<Integer> impostorsOverFolds = new HashSet<>();
        for (int fold = 0; fold < EvaluationProtocol.FOLDS; fold++) {
            Set<Integer> fold_ = new HashSet<>();
            for (int p : protocol.knownPersons(fold)) fold_.add(p);
            for (int p : protocol.validationImpostors(fold)) { fold_.add(p); impostorsOverFolds.add(p); }
            for (int p : protocol.testImpostors(fold)) { fold_.add(p); impostorsOverFolds.add(p); }
            assertEquals(OrlDataset.PERSONS, fold_.size());
            assertEquals(30, protocol.knownPersons(fold).length);
        }
        assertEquals(OrlDataset.PERSONS, impostorsOverFolds.size()); // каждый ровно раз чужой
        for (int person = 0; person < OrlDataset.PERSONS; person++) {
            Set<Integer> images = new HashSet<>();
            for (int i : protocol.trainImages(person)) images.add(i);
            for (int i : protocol.validationImages(person)) images.add(i);
            for (int i : protocol.testImages(person)) images.add(i);
            assertEquals(OrlDataset.IMAGES_PER_PERSON, images.size());
            assertEquals(3, protocol.testImages(person).length);
        }
    }
}
