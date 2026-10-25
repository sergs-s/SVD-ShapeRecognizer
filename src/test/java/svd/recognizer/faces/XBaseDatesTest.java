package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;

/**
 * Кривая «1 → 2 → 3 даты» (XBaseDates): границы интервалов, номера частей, оценка (с) по сессиям.
 *
 * @author ssv
 */
public class XBaseDatesTest {

    @Test
    public void bins() {
        assertEquals(0, XBaseDates.bin(0));
        assertEquals(0, XBaseDates.bin(30));
        assertEquals(1, XBaseDates.bin(31));
        assertEquals(1, XBaseDates.bin(365));
        assertEquals(2, XBaseDates.bin(366));
    }

    @Test
    public void posts() {
        Set<Integer> conf = new HashSet<>();
        for (XBaseDates.Post p : XBaseDates.POSTS) {
            assertTrue(conf.add(p.conf()));
            assertTrue(p.from() >= p.s());
            assertTrue(p.minDates() > p.from());
        }
        assertEquals(0, XBaseDates.POSTS.get(0).conf());
        assertEquals(List.of(0, 70, 71, 72, 73), XBaseDates.POSTS.stream().map(XBaseDates.Post::conf).toList());
    }

    /** (с): оценка человека – наименьшее 1 − cos по его сессиям; при одной сессии – как (м). */
    @Test
    public void sessions() {
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new java.util.HashMap<>();
        v.put("a1", new double[] {1, 0});
        v.put("a2", new double[] {0, 1});
        v.put("b1", new double[] {-1, 0});
        v.put("q", new double[] {0.9, 0.1});
        x.vec.put(V.L1, v);
        LdaModel lm = new LdaModel(new double[][] {{1, 0}, {0, 1}}, new double[] {0, 0}, null, "");
        List<String> gal = List.of("A", "B");
        Map<String, List<List<String>>> sess = Map.of("A", List.of(List.of("a1"), List.of("a2")), "B", List.of(List.of("b1")));
        List<Probe> pr = new ArrayList<>(List.of(new Probe(FeiScores.OWN, "A", 1, "q", 0)));
        String l = XBaseDates.sessLines(x, "L1", 0, lm, false, gal, sess, pr);
        FeiScores.Row r = FeiScores.parse(l.trim());
        assertEquals("A", r.bestId());
        assertEquals(XBaseRule.cos(new double[] {0.9, 0.1}, new double[] {1, 0}, 2, null), r.best(), 1e-15);
        assertEquals("B", r.secondId());
    }
}
