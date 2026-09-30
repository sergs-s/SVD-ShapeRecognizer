package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.junit.Test;

/**
 * Протокол «только FEI»: свои и чужие не пересекаются, у каждого своего ровно 8 снимков (7 на обучении, 1 на контроле),
 * контроль не участвует в пороге и когорте; отбеливание PCA.
 *
 * @author ssv
 */
public class FeiProtocolTest {

    /** 200 человек как в FEI: 149 полных (все 8 номеров и ещё несколько), у остальных не хватает хотя бы одного из 8. */
    private static Map<String, Set<Integer>> included() {
        Map<String, Set<Integer>> m = new TreeMap<>();
        Random rnd = new Random(1);
        for (int p = 1; p <= 200; p++) {
            Set<Integer> s = new HashSet<>();
            for (int n = 1; n <= 14; n++) if (rnd.nextDouble() < 0.5) s.add(n);
            for (int n : FeiProtocol.NUMBERS) s.add(n);
            if (p > 149) s.remove(FeiProtocol.NUMBERS[p % FeiProtocol.NUMBERS.length]);
            m.put(String.format("%03d", p), s);
        }
        return m;
    }

    @Test
    public void rolesAreDisjointAndSized() {
        Map<String, Set<Integer>> inc = included();
        assertEquals(149, FeiProtocol.fullPersons(inc).size());
        List<FeiProtocol.Split> splits = FeiProtocol.splits(inc, 42);
        assertEquals(FeiProtocol.SPLITS, splits.size());
        Set<List<String>> galleries = new HashSet<>();
        for (FeiProtocol.Split sp : splits) {
            assertEquals(100, sp.gallery().size());
            assertEquals(20, sp.cohort().size());
            assertEquals(40, sp.threshold().size());
            assertEquals(40, sp.control().size());
            Set<String> all = new HashSet<>();
            all.addAll(sp.gallery());
            all.addAll(sp.cohort());
            all.addAll(sp.threshold());
            all.addAll(sp.control());
            assertEquals(200, all.size());
            // Контроль не пересекается ни с порогом, ни с когортой, ни со своими.
            for (String p : sp.control()) {
                assertFalse(sp.threshold().contains(p) || sp.cohort().contains(p) || sp.gallery().contains(p));
            }
            // Свои — только полные: все 8 снимков.
            for (String p : sp.gallery()) for (int n : FeiProtocol.NUMBERS) assertTrue(inc.get(p).contains(n));
            galleries.add(new ArrayList<>(sp.gallery()));
        }
        assertEquals("разбиения с разными ролями", FeiProtocol.SPLITS, galleries.size());
        // Воспроизводимость по seed.
        assertEquals(splits, FeiProtocol.splits(inc, 42));
    }

    @Test
    public void eightImagesPerOwnPerson() {
        for (int j = 0; j < FeiProtocol.NUMBERS.length; j++) {
            List<Integer> train = FeiProtocol.trainNumbers(j);
            assertEquals(7, train.size());
            assertFalse(train.contains(FeiProtocol.NUMBERS[j]));
            Set<Integer> all = new HashSet<>(train);
            all.add(FeiProtocol.NUMBERS[j]);
            assertEquals(8, all.size());
            assertEquals(FeiProtocol.NUMBERS[j] >= 11, FeiProtocol.frontal(j));
        }
    }

    @Test(expected = IllegalStateException.class)
    public void checkRejectsOverlap() {
        Map<String, Set<Integer>> inc = included();
        FeiProtocol.Split sp = FeiProtocol.splits(inc, 42).get(0);
        List<String> threshold = new ArrayList<>(sp.threshold());
        threshold.set(0, sp.control().get(0));
        FeiProtocol.check(new FeiProtocol.Split(0, sp.gallery(), sp.cohort(), threshold, sp.control()), inc);
    }

    @Test
    public void whiteningGivesUnitVarianceOnTrainingData() {
        // 2 класса по 6 векторов размерности 5; при δ = 0 у обучающих данных дисперсия каждой координаты y — 1.
        Random rnd = new Random(7);
        List<List<double[]>> classes = new ArrayList<>();
        for (int c = 0; c < 2; c++) {
            List<double[]> list = new ArrayList<>();
            for (int i = 0; i < 6; i++) {
                double[] x = new double[5];
                for (int t = 0; t < 5; t++) x[t] = rnd.nextGaussian() * (t + 1) + c;
                list.add(x);
            }
            classes.add(list);
        }
        WpcaScorer.Pca pca = WpcaScorer.Pca.of(classes);
        double[] var = new double[pca.k()];
        for (List<double[]> list : classes) {
            for (double[] x : list) {
                double[] y = pca.whiten(x, 0);
                for (int i = 0; i < y.length; i++) var[i] += y[i] * y[i] / (pca.n() - 1);
            }
        }
        for (double v : var) assertEquals(1, v, 1e-9);
        // 1 − cos: проба, совпадающая с эталоном, — 0; подпространство: вектор обучения — ε = 0 при полном ранге.
        GalleryScorer.ScoreModel cos = WpcaScorer.model(pca, classes, 2, WpcaScorer.Mode.COS, 0.1);
        double[] m0 = WpcaScorer.mean(classes.get(0));
        assertTrue(cos.scores(m0)[0] < 1e-9);
        WpcaScorer.Subspace s = WpcaScorer.Subspace.of(List.of(new double[] {1, 0, 0}, new double[] {0, 1, 0}, new double[] {-1, -1, 0}));
        assertEquals(0, s.error(new double[] {2, -1, 0}), 1e-9);
        assertEquals(3, s.error(new double[] {0, 0, 3}), 1e-9);
    }
}
