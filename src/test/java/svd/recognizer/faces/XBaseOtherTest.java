package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.Test;

/**
 * Сверка на MUCT и ORL (PLAN.md п. 5.5): когорта MUCT без близнецов, близнецы в одной части, части покрывают людей без
 * повторов, деление снимков пополам (⌈n/2⌉, без пересечения, воспроизводимо), ошибка своих при фактическом FAR контроля по
 * нескольким конфигурациям (порог — своей конфигурации).
 *
 * @author ssv
 */
public class XBaseOtherTest {

    static List<String> muctLike() {
        List<String> p = new ArrayList<>();
        for (int i = 0; i < 276; i++) p.add(String.format("%03d", i));
        return p;
    }

    @Test
    public void cohortWithoutTwins() {
        for (long seed = 0; seed < 50; seed++) {
            List<String> c = XBaseOther.muctCohort(muctLike(), seed);
            assertEquals(20, c.size());
            assertFalse(c.contains("247") || c.contains("248"));
            assertEquals(c, c.stream().sorted().toList());
        }
        assertEquals(XBaseOther.muctCohort(muctLike(), 42), XBaseOther.muctCohort(muctLike(), 42));
    }

    @Test
    public void twinsInOnePartAndCover() {
        List<String> people = muctLike();
        people.removeAll(XBaseOther.muctCohort(people, 42));
        for (long seed = 0; seed < 50; seed++) {
            List<List<String>> parts = XBaseOther.parts(people, seed, XBaseOther.MUCT_PART_OFFSET);
            assertEquals(5, parts.size());
            Set<String> all = new HashSet<>();
            int n = 0;
            for (List<String> q : parts) {
                all.addAll(q);
                n += q.size();
                assertEquals(q.contains("247"), q.contains("248"));
                assertTrue(q.size() >= 50 && q.size() <= 53);
            }
            assertEquals(people.size(), n);
            assertEquals(new HashSet<>(people), all);
        }
        // ORL: 40 человек — по 8.
        List<String> orl = new ArrayList<>();
        for (int i = 1; i <= 40; i++) orl.add(String.format("s%02d", i));
        for (List<String> q : XBaseOther.parts(orl, 42, XBaseOther.ORL_PART_OFFSET)) assertEquals(8, q.size());
    }

    @Test
    public void halves() {
        Map<String, List<String>> keys = new TreeMap<>();
        keys.put("a", List.of("a1", "a2", "a3", "a4", "a5"));
        keys.put("b", List.of("b1", "b2"));
        List<Map<String, List<String>>> h = XBaseOther.halves(keys, 42, 9510);
        assertEquals(3, h.get(0).get("a").size());
        assertEquals(2, h.get(1).get("a").size());
        assertEquals(1, h.get(0).get("b").size());
        assertEquals(1, h.get(1).get("b").size());
        Set<String> u = new HashSet<>(h.get(0).get("a"));
        u.addAll(h.get(1).get("a"));
        assertEquals(5, u.size());
        assertEquals(h, XBaseOther.halves(keys, 42, 9510));
    }

    @Test
    public void cvPartsUneven() {
        List<String> p30 = muctLike().subList(0, 30);
        assertEquals(FeiMethods.cvParts(p30, 42, 100), XBaseOther.cvParts(p30, 42, 100));
        List<String> p32 = muctLike().subList(0, 32);
        List<List<String>> q = XBaseOther.cvParts(p32, 42, 101);
        assertEquals(List.of(7, 7, 6, 6, 6), q.stream().map(List::size).toList());
        Set<String> all = new HashSet<>();
        q.forEach(all::addAll);
        assertEquals(32, all.size());
    }

    static FeiDecompose.Cfg cfg(double[] ownR, boolean[] ok, int[] ownP, double[] ctrlR, int[] ctrlP) {
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        c.ownR = ownR;
        c.ownOk = ok;
        c.ownP = ownP;
        c.ctrlR = ctrlR;
        c.ctrlP = ctrlP;
        return c;
    }

    @Test
    public void errAtFarPerConfiguration() {
        // Конфигурация 1: контроль 0,5 / 0,6 / … (10 снимков), конфигурация 2: контроль 0,1 / 0,2 / … — пороги разные.
        double[] c1 = {0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95};
        double[] c2 = {0.1, 0.2, 0.3, 0.4, 0.45, 0.91, 0.92, 0.93, 0.94, 0.95};
        int[] cp = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        FeiDecompose.Cfg a = cfg(new double[] {0.4, 0.55}, new boolean[] {true, true}, new int[] {2, 2}, c1, cp);
        FeiDecompose.Cfg b = cfg(new double[] {0.05, 0.15}, new boolean[] {true, true}, new int[] {3, 3}, c2, cp);
        int[] unit = FeiBootstrap.unit(4);
        // FAR 10 %: допускается 1 снимок → θ₁ < 0,6 (0,4 и 0,55 приняты), θ₂ < 0,2 (0,05 и 0,15 приняты) → ошибка 0, FAR 10 %.
        double[] r = XBaseOther.errAtFar(List.of(a, b), unit, 0.10);
        assertEquals(0, r[0], 1e-12);
        assertEquals(10, r[1], 1e-12);
        // FAR 0: θ₁ < 0,5 (принят 0,4), θ₂ < 0,1 (принят 0,05) → ошибка 2/4.
        double[] r0 = XBaseOther.errAtFar(List.of(a, b), unit, 0.0);
        assertEquals(50, r0[0], 1e-12);
        assertEquals(0, r0[1], 1e-12);
    }
}
