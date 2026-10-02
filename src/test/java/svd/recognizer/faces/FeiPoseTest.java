package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.List;
import org.junit.Test;

/**
 * Отбор FEI из fei_selection.tsv: только included = 1, люди по возрастанию, |r| и граница «повёрнут» (|r| > r0).
 *
 * @author ssv
 */
public class FeiPoseTest {

    private static final List<String> LINES = List.of(
            "# R\t0.38",
            "person\tnumber\tname\tfound\tr\treference\tincluded\treason",
            "002\t11\t2-11\t1\t0.05\t-\t1\tвсегда",
            "001\t4\t1-04\t1\t-0.12\t-\t1\tпоза",
            "001\t1\t1-01\t1\t-0.56\t-\t0\tисключён",
            "001\t5\t1-05\t1\t0.1200001\t-\t1\tпоза");

    @Test
    public void parse() {
        FeiPose p = FeiPose.parse(LINES);
        assertEquals(List.of("001", "002"), p.persons());
        assertEquals(0.12, p.absR("001", 4), 0);
        assertEquals(0.05, p.absR("002", 11), 0);
        assertFalse("|r| = r0 — без отражения", p.turned("001", 4, 0.12));
        assertTrue(p.turned("001", 5, 0.12));
        try {
            p.absR("001", 1);
        } catch (IllegalArgumentException e) {
            return;
        }
        throw new AssertionError("исключённый снимок должен давать ошибку");
    }
}
