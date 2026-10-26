package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.IntPredicate;
import org.junit.Test;

/**
 * Опыт «пул FERET + VidTIMIT» (VidTimitData, XBaseVt): а) цели фразы; б) ближайший годный кадр; в) выбор фраз по описи; г)
 * отбор детерминирован и в (В4) кадры фразы различны; д) состав классов – человек один класс по трём сессиям, 6 / 12 кадров; е)
 * правило выбора.
 *
 * @author ssv
 */
public class VidTimitDataTest {

    @Test
    public void targets() {
        assertArrayEquals(new int[] {5}, VidTimitData.targets(10, false));
        assertArrayEquals(new int[] {4, 7}, VidTimitData.targets(10, true));
        assertArrayEquals(new int[] {6}, VidTimitData.targets(11, false));
        assertArrayEquals(new int[] {4, 8}, VidTimitData.targets(11, true));
        assertArrayEquals(new int[] {4, 8}, VidTimitData.targets(12, true));
    }

    @Test
    public void nearest() {
        IntPredicate ok = i -> i == 3 || i == 7;
        assertEquals(3, VidTimitData.nearest(10, 5, ok, Set.of()));
        assertEquals(7, VidTimitData.nearest(10, 5, ok, Set.of(3)));
        assertEquals(7, VidTimitData.nearest(10, 7, ok, Set.of()));
        assertEquals(-1, VidTimitData.nearest(10, 5, i -> false, Set.of()));
        assertEquals(-1, VidTimitData.nearest(10, 5, ok, Set.of(3, 7)));
        assertEquals(1, VidTimitData.nearest(10, 10, i -> i == 1, Set.of()));
    }

    static List<String> inventory() {
        List<String> l = new ArrayList<>();
        l.add("person\tsession\tseq\tframes\tnot_jpeg\tgap\tfirst\tlast");
        for (String p : new String[] {"mbbb0", "faaa0", "fccc1"}) {
            l.add(p + "\t1\thead\t300\t0\t0\t1\t300");
            for (String s : new String[] {"sa1", "sa2", "si1279", "si1909", "si649", "sx109"}) l.add(p + "\t1\t" + s + "\t100\t0\t0\t1\t100");
            l.add(p + "\t2\thead2\t300\t0\t0\t1\t300");
            l.add(p + "\t2\tsx199\t90\t0\t0\t1\t90");
            l.add(p + "\t2\tsx19\t80\t0\t0\t1\t80");
            l.add(p + "\t3\tsx379\t70\t0\t0\t1\t70");
            l.add(p + "\t3\tsx289\t60\t0\t0\t1\t60");
            l.add(p + "\t0\t#audio:sa1\t10\t3\t0\t0\t0");
        }
        return l;
    }

    @Test
    public void phrases() {
        List<VidTimitData.Phrase> ps = VidTimitData.phrases(inventory());
        assertEquals(18, ps.size());
        List<String> first = new ArrayList<>();
        for (VidTimitData.Phrase p : ps.subList(0, 6)) first.add(p.person() + " " + p.session() + " " + p.name());
        assertEquals(List.of("faaa0 1 si649", "faaa0 1 sx109", "faaa0 2 sx19", "faaa0 2 sx199", "faaa0 3 sx289", "faaa0 3 sx379"), first);
        for (VidTimitData.Phrase p : ps) assertTrue(!p.name().startsWith("head") && !p.name().startsWith("#"));
    }

    /** Синтетические фразы: кадры 001…n, годность – случайная с seed. */
    static Object[] synth(long seed) {
        List<VidTimitData.Phrase> ps = VidTimitData.phrases(inventory());
        Map<VidTimitData.Phrase, List<String>> frames = new LinkedHashMap<>();
        Map<VidTimitData.Phrase, IntPredicate> ok = new HashMap<>();
        Map<VidTimitData.Phrase, double[]> r = new HashMap<>();
        Random rnd = new Random(seed);
        for (VidTimitData.Phrase p : ps) {
            int n = 10 + rnd.nextInt(30);
            List<String> names = new ArrayList<>();
            boolean[] good = new boolean[n + 1];
            double[] rr = new double[n];
            for (int i = 1; i <= n; i++) {
                names.add(String.format("%03d", i));
                good[i] = rnd.nextDouble() < 0.6;
                rr[i - 1] = good[i] ? rnd.nextDouble() * 0.3 : 0.5;
            }
            frames.put(p, names);
            ok.put(p, i -> good[i]);
            r.put(p, rr);
        }
        // Одна фраза без годных кадров – «нет кадра», человек остаётся.
        VidTimitData.Phrase bad = ps.get(3);
        ok.put(bad, i -> false);
        return new Object[] {ps, frames, ok, r, bad};
    }

    @SuppressWarnings("unchecked")
    static List<VidTimitData.Pick> pick(Object[] z, String v) {
        Map<VidTimitData.Phrase, double[]> r = (Map<VidTimitData.Phrase, double[]>) z[3];
        return VidTimitData.select(v, (List<VidTimitData.Phrase>) z[0], (Map<VidTimitData.Phrase, List<String>>) z[1], (Map<VidTimitData.Phrase, IntPredicate>) z[2],
                (p, i) -> r.get(p)[i - 1]);
    }

    @Test
    public void deterministicAndDistinct() {
        for (String v : VidTimitData.VARIANT) {
            String a = VidTimitData.table(0.387, pick(synth(5), v));
            String b = VidTimitData.table(0.387, pick(synth(5), v));
            assertEquals(a, b);
            assertNotEquals(a, VidTimitData.table(0.387, pick(synth(6), v)));
        }
        Object[] z = synth(5);
        List<VidTimitData.Pick> four = pick(z, VidTimitData.VARIANT[1]);
        assertEquals(36, four.size());
        for (int i = 0; i < four.size(); i += 2) {
            VidTimitData.Pick a = four.get(i);
            VidTimitData.Pick b = four.get(i + 1);
            assertEquals(a.phrase(), b.phrase());
            if (a.taken() > 0 && b.taken() > 0) assertNotEquals(a.taken(), b.taken());
            for (VidTimitData.Pick k : List.of(a, b)) {
                if (k.phrase().equals(z[4])) assertEquals(VidTimitData.NONE, k.reason());
                else assertTrue(k.taken() > 0);
                assertEquals(k.taken() == k.target() ? VidTimitData.OK : k.taken() < 0 ? VidTimitData.NONE : VidTimitData.NEAR, k.reason());
            }
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void classes() {
        Object[] z = synth(9);
        Map<VidTimitData.Phrase, List<String>> frames = (Map<VidTimitData.Phrase, List<String>>) z[1];
        VidTimitData.Phrase bad = (VidTimitData.Phrase) z[4];
        for (int v = 0; v < 2; v++) {
            List<List<String>> cls = VidTimitData.classes(pick(z, VidTimitData.VARIANT[v]), frames);
            assertEquals(3, cls.size());
            for (List<String> c : cls) {
                boolean hasBad = c.stream().anyMatch(k -> k.startsWith(VidTimitData.KEY_PREFIX + bad.person() + "/"));
                int want = 6 * (v + 1) - (hasBad && c.get(0).contains(bad.person()) ? v + 1 : 0);
                assertEquals(want, c.size());
                Set<String> persons = new HashSet<>();
                Set<String> sessionsSeen = new HashSet<>();
                for (String k : c) {
                    assertTrue(k.startsWith(VidTimitData.KEY_PREFIX));
                    String[] p = k.substring(VidTimitData.KEY_PREFIX.length()).split("/");
                    persons.add(p[0]);
                    sessionsSeen.add(p[1].startsWith("si") || p[1].equals("sx109") ? "1" : p[1].equals("sx19") || p[1].equals("sx199") ? "2" : "3");
                }
                assertEquals(1, persons.size());
                assertEquals(Set.of("1", "2", "3"), sessionsSeen);
                assertTrue(XBaseVt.isVt(c));
            }
        }
    }

    @Test
    public void choose() {
        assertEquals(XBaseVt.BASE, XBaseVt.choose(new double[] {20.0, 19.5, 19.2}));
        assertEquals(1, XBaseVt.choose(new double[] {20.0, 18.5, 18.0}));
        assertEquals(2, XBaseVt.choose(new double[] {20.0, 19.5, 18.0}));
        assertTrue(!XBaseVt.isVt(List.of("00001/x")));
    }

    @Test
    public void ownVariants() {
        Map<String, Map<String, Integer>> sel = new LinkedHashMap<>();
        sel.put("GT", Map.of("L1-xs-F", 0, "L1-tt-xs-F", 0));
        assertArrayEquals(new int[] {0}, XBaseVt.ownVariants(sel));
        sel.put("FERET-1", Map.of("L1-xs-F", 0, "L1-tt-xs-F", 1));
        assertArrayEquals(new int[] {0, 1}, XBaseVt.ownVariants(sel));
        sel.put("FERET-2", Map.of("L1-xs-F", 2, "L1-tt-xs-F", 1));
        assertArrayEquals(new int[] {0, 1, 2}, XBaseVt.ownVariants(sel));
    }
}
