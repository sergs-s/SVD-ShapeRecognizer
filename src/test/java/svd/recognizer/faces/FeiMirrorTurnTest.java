package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;

/**
 * +mirror-turn (этап 5б): значения ключа faces.fei.mirror; отражаются ровно обучающие снимки с |r| > r0, граница
 * |r| = r0 — без отражения; при true — все, как раньше.
 *
 * @author ssv
 */
public class FeiMirrorTurnTest {

    @Test
    public void modes() {
        assertEquals(EnumSet.noneOf(FeiMethods.Mirror.class), FeiMethods.mirrorModes(null));
        assertEquals(EnumSet.noneOf(FeiMethods.Mirror.class), FeiMethods.mirrorModes("false"));
        assertEquals(EnumSet.of(FeiMethods.Mirror.ALL), FeiMethods.mirrorModes("TRUE"));
        assertEquals(EnumSet.of(FeiMethods.Mirror.TURN), FeiMethods.mirrorModes("turn"));
        assertEquals(EnumSet.of(FeiMethods.Mirror.ALL, FeiMethods.Mirror.TURN), FeiMethods.mirrorModes("true, turn"));
        try {
            FeiMethods.mirrorModes("turns");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("turns"));
            return;
        }
        throw new AssertionError("ожидалась ошибка");
    }

    @Test
    public void r0Required() {
        assertEquals(0.12, FeiPose.r0("0.12", "x"), 0);
        for (String bad : new String[] {null, " ", "-0.1", "NaN"}) {
            try {
                FeiPose.r0(bad, "turn");
            } catch (IllegalStateException e) {
                continue;
            }
            throw new AssertionError("ожидалась ошибка для " + bad);
        }
    }

    @Test
    public void onlyTurnedAreMirrored() {
        FeiPose pose = FeiPose.parse(List.of(
                "person\tnumber\tname\tfound\tr\treference\tincluded\treason",
                "007\t4\t7-04\t1\t-0.30\t-\t1\t-",
                "007\t5\t7-05\t1\t0.05\t-\t1\t-",
                "007\t6\t7-06\t1\t0.12\t-\t1\t-",
                "007\t7\t7-07\t1\t-0.13\t-\t1\t-",
                "007\t11\t7-11\t1\t0.02\t-\t1\t-",
                "007\t12\t7-12\t1\t-0.19\t-\t1\t-"));
        Map<Integer, String> keys = new TreeMap<>();
        for (int n : new int[] {4, 5, 6, 7, 11, 12}) keys.put(n, "k" + n);
        List<Integer> train = List.of(5, 6, 7, 11, 12, 4);
        // |r|: 0,05 и 0,02 — анфас; 0,12 = r0 — граница, без отражения; 0,30, 0,13, 0,19 — отражаются (и №12, хоть это «анфас» по номеру).
        assertEquals(List.of("k7", "k12", "k4"), FeiMethods.mirrorKeys("007", keys, train, FeiMethods.Mirror.TURN, pose, 0.12));
        assertEquals(List.of("k5", "k6", "k7", "k11", "k12", "k4"), FeiMethods.mirrorKeys("007", keys, train, FeiMethods.Mirror.ALL, null,
                Double.NaN));
    }
}
