package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Проверка разбиения с ротацией контрольного снимка и бутстрепа (без OpenCV).
 *
 * @author ssv
 */
public class TrainingSizeCurveTest {

    private final int[] order = new EvaluationProtocol(42L, 5).imageOrder(7);

    @Test
    public void eachImageIsTestedOncePerFold() {
        Set<Integer> tested = new HashSet<>();
        for (int r = 0; r < TrainingSizeCurve.ROTATIONS; r++) {
            tested.add(TrainingSizeCurve.testImage(order, r));
        }
        assertEquals(OrlDataset.IMAGES_PER_PERSON, tested.size());
    }

    @Test
    public void trainingSetsAreNestedAndExcludeTestImage() {
        for (int r = 0; r < TrainingSizeCurve.ROTATIONS; r++) {
            int test = TrainingSizeCurve.testImage(order, r);
            for (int n = TrainingSizeCurve.MIN_N; n <= TrainingSizeCurve.MAX_N; n++) {
                int[] train = TrainingSizeCurve.trainImages(order, r, n);
                Set<Integer> distinct = new HashSet<>();
                for (int image : train) {
                    distinct.add(image);
                }
                assertEquals(n, distinct.size());
                assertFalse(distinct.contains(test));
                if (n > TrainingSizeCurve.MIN_N) {
                    int[] smaller = TrainingSizeCurve.trainImages(order, r, n - 1);
                    int[] prefix = new int[n - 1];
                    System.arraycopy(train, 0, prefix, 0, n - 1);
                    assertArrayEquals(smaller, prefix);
                }
            }
        }
    }

    @Test
    public void plateauRequiresAllLaterGainsInsignificant() {
        // Приросты: 3→4 незначим, 4→5 значим, 5→6 и 6→7 незначимы → плато с N = 5.
        java.util.List<TrainingSizeCurve.Point> points = java.util.List.of(
                point(3, -0.01), point(4, 0.02), point(5, -0.01), point(6, 0.0), point(7, Double.NaN));
        assertEquals(5, TrainingSizeCurve.plateau(points));
        // Последний прирост значим → плато нет.
        assertEquals(-1, TrainingSizeCurve.plateau(java.util.List.of(point(3, -0.01), point(4, 0.01), point(5, Double.NaN))));
    }

    private static TrainingSizeCurve.Point point(int n, double diffLo) {
        double diff = Double.isNaN(diffLo) ? Double.NaN : diffLo + 0.01;
        return new TrainingSizeCurve.Point(n, n - 1, 0, 0, 0, diff, diffLo, diffLo + 0.02,
                0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 20);
    }

    @Test
    public void bootstrapIsReproducibleAndPercentileIsOrdered() {
        int[][] a = TrainingSizeCurve.bootstrapSamples(40, 50, 42L);
        int[][] b = TrainingSizeCurve.bootstrapSamples(40, 50, 42L);
        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], b[i]);
        }
        double[] values = new double[1000];
        for (int i = 0; i < values.length; i++) {
            values[i] = i;
        }
        double[] ci = TrainingSizeCurve.percentile95(values);
        assertEquals(25.0, ci[0], 0.0);
        assertEquals(974.0, ci[1], 0.0);
        // Доля по людям: 2 человека, у первого 1/3, у второго 0/1 → 1/4.
        assertEquals(0.25, TrainingSizeCurve.bootRatio(new int[] {0, 1}, new int[] {1, 0}, new int[] {3, 1}), 1e-12);
        assertTrue(ci[0] < ci[1]);
    }
}
