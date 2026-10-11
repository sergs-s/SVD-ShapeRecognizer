package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.Test;

/**
 * Межсессионная изменчивость (PLAN.md п. 4.2): MldaScorer с отдельными шаблонами, части своих, классы FEI для L1-xs0.
 *
 * @author ssv
 */
public class XBaseXsTest {

    static List<List<double[]>> classes(long seed, int c, int per, int dim) {
        Random rnd = new Random(seed);
        List<List<double[]>> out = new ArrayList<>();
        for (int k = 0; k < c; k++) {
            List<double[]> l = new ArrayList<>();
            for (int i = 0; i < per; i++) {
                double[] v = new double[dim];
                for (int j = 0; j < dim; j++) v[j] = k + rnd.nextGaussian();
                l.add(v);
            }
            out.add(l);
        }
        return out;
    }

    @Test
    public void mldaSeparateTemplates() {
        List<List<double[]>> cls = classes(1, 5, 4, 12);
        double[] x = new double[12];
        for (int j = 0; j < 12; j++) x[j] = 0.3 * j;
        MldaScorer s = new MldaScorer();
        // Шаблоны = обучающие классы — как прежний fit.
        assertArrayEquals(s.fit(cls, 3).scores(x), s.fit(cls, cls, 3, 1e-10).scores(x), 0);
        // Шаблон — среднее по классам шаблонов: проба, равная единственному снимку шаблона, на нулевом расстоянии.
        List<List<double[]>> templ = new ArrayList<>();
        for (int k = 0; k < 3; k++) templ.add(List.of(cls.get(k).get(0)));
        double[] d = s.fit(cls, templ, 3, 1e-10).scores(cls.get(1).get(0));
        assertEquals(0, d[1], 1e-9);
        assertTrue(d[0] > 0 && d[2] > 0);
    }

    @Test
    public void partsAreDisjointAndDeterministic() {
        List<String> own = new ArrayList<>();
        for (int i = 0; i < 30; i++) own.add(String.format("s%02d", i));
        List<List<String>> p = XBaseXs.parts(own, 42);
        assertEquals(p, XBaseXs.parts(own, 42));
        assertEquals(5, p.size());
        Set<String> all = new HashSet<>();
        for (List<String> q : p) {
            assertEquals(6, q.size());
            all.addAll(q);
        }
        assertEquals(30, all.size());
    }

    @Test
    public void feiClassesCount() {
        List<List<String>> pool = List.of(List.of("a1", "a2", "a3", "a4"), List.of("b1", "b2", "b3"), List.of("c1", "c2", "c3", "c4"));
        // n = 9: a (4) + b (3) + c (2).
        List<List<String>> c9 = XBaseXs.feiClasses(pool, 9);
        assertEquals(List.of(List.of("a1", "a2", "a3", "a4"), List.of("b1", "b2", "b3"), List.of("c1", "c2")), c9);
        // n = 8: последнему достался бы один снимок — у b убирается последний, c получает два.
        List<List<String>> c8 = XBaseXs.feiClasses(pool, 8);
        assertEquals(List.of(List.of("a1", "a2", "a3", "a4"), List.of("b1", "b2"), List.of("c1", "c2")), c8);
        assertEquals(8, c8.stream().mapToInt(List::size).sum());
        try {
            XBaseXs.feiClasses(pool, 12);
            fail();
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("не хватает"));
        }
    }
}
