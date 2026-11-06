package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import svd.recognizer.faces.XBaseChokeRun.Att;
import svd.recognizer.faces.XBaseChokeRun.Cand;
import svd.recognizer.faces.XBaseChokeRun.Res;

public class XBaseChokeRunTest {

    @Test
    public void orderPrefersLargestEyesWithinR0ThenSmallestR() {
        List<Cand> l = List.of(new Cand(5, 0.10, 20), new Cand(3, 0.05, 30), new Cand(4, 0.11, 30), new Cand(7, 0.30, 90), new Cand(6, 0.20, 10), new Cand(8, 0.50, 99));
        List<Integer> got = XBaseChokeRun.order(l, 0.12, 0.387).stream().map(Cand::frame).toList();
        assertEquals(List.of(3, 4, 5, 6, 7), got);
    }

    @Test
    public void orderTieGoesToSmallerFrame() {
        List<Cand> l = List.of(new Cand(9, 0.2, 10), new Cand(2, 0.2, 40));
        assertEquals(List.of(2, 9), XBaseChokeRun.order(l, 0.12, 0.387).stream().map(Cand::frame).toList());
    }

    @Test
    public void galleryOrderFollowsAnswerFour() {
        List<String[]> o = XBaseChokeRun.galleryOrder(3);
        assertArrayEquals(new String[] {"P1E", "3"}, o.get(0));
        assertArrayEquals(new String[] {"P1L", "3"}, o.get(1));
        assertArrayEquals(new String[] {"P1L", "4"}, o.get(2));
        assertArrayEquals(new String[] {"P1L", "1"}, o.get(3));
        assertArrayEquals(new String[] {"P1L", "2"}, o.get(4));
        assertEquals(5, o.size());
    }

    @Test
    public void partsUseSeedOfAnswerFive() {
        List<String> own = new ArrayList<>();
        for (int i = 0; i < 13; i++) own.add(String.format("%04d", i));
        List<String> perm = new ArrayList<>(own);
        java.util.Collections.shuffle(perm, new java.util.Random(42 + 12010 + 1));
        List<List<String>> p = XBaseChokeRun.parts(own, 42, 1);
        assertEquals(5, p.size());
        assertEquals(perm.subList(0, 3), p.get(0));
        assertEquals(perm.subList(11, 13), p.get(4));
    }

    @Test
    public void addFailIsErrorAtAnyThreshold() {
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        c.ownP = new int[] {0};
        c.ownR = new double[] {0.5};
        c.ownOk = new boolean[] {true};
        c.noSecond = new boolean[1];
        c.trueInf = new boolean[1];
        c.poseFrontal = new boolean[1];
        XBaseChokeRun.addFail(c, 7);
        assertEquals(2, c.ownP.length);
        assertEquals(7, c.ownP[1]);
        assertTrue(Double.isInfinite(c.ownR[1]));
        assertFalse(c.ownOk[1]);
        assertTrue(c.trueInf[1]);
    }

    @Test
    public void exifDateIsReadFromBigAndLittleEndian() {
        assertEquals("2010:04:16", XBaseChokeRun.exifDate(exif(true)));
        assertEquals("2010:04:16", XBaseChokeRun.exifDate(exif(false)));
        assertNull(XBaseChokeRun.exifDate(new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xDB, 0, 4, 0, 0, 0, 0}));
    }

    private static void put16(byte[] b, int p, int v, boolean le) {
        b[p + (le ? 0 : 1)] = (byte) v;
        b[p + (le ? 1 : 0)] = (byte) (v >> 8);
    }

    private static void put32(byte[] b, int p, int v, boolean le) {
        put16(b, p + (le ? 0 : 2), v & 0xFFFF, le);
        put16(b, p + (le ? 2 : 0), v >>> 16, le);
    }

    private static byte[] exif(boolean le) {
        byte[] b = new byte[200];
        b[0] = (byte) 0xFF;
        b[1] = (byte) 0xD8;
        b[2] = (byte) 0xFF;
        b[3] = (byte) 0xE1;
        b[4] = 0;
        b[5] = (byte) 190;
        System.arraycopy("Exif".getBytes(StandardCharsets.US_ASCII), 0, b, 6, 4);
        int base = 12;
        b[base] = (byte) (le ? 'I' : 'M');
        b[base + 1] = b[base];
        put32(b, base + 4, 8, le);
        put16(b, base + 8, 1, le);
        put16(b, base + 10, 0x8769, le);
        put32(b, base + 18, 40, le);
        put16(b, base + 40, 1, le);
        put16(b, base + 42, 0x9003, le);
        put32(b, base + 46, 20, le);
        put32(b, base + 50, 80, le);
        System.arraycopy("2010:04:16 10:00:00".getBytes(StandardCharsets.US_ASCII), 0, b, base + 80, 19);
        return b;
    }

    @Test
    public void eyesReturnsEmptyForOnePoint() {
        Map<Integer, int[]> e = XBaseChokeRun.eyes(List.of(" <frame number=\"00000001\">", "  <person id=\"0001\">", "   <leftEye x=\"10\" y=\"20\"/>", "   <rightEye x=\"30\" y=\"40\"/>", "  </person>", " </frame>",
                " <frame number=\"00000002\">", "  <person id=\"0001\">", "   <rightEye x=\"3\" y=\"4\"/>", "  </person>", " </frame>", " <frame number=\"00000003\"/>"));
        assertArrayEquals(new int[] {10, 20, 30, 40}, e.get(1));
        assertEquals(0, e.get(2).length);
        assertFalse(e.containsKey(3));
    }

    private static Att att(String person) {
        return new Att(person, "28", "P2E", 1, "1", "k", 30);
    }

    @Test
    public void errorsByPersonAndAttemptDiffer() {
        Map<String, Integer> idx = new HashMap<>();
        idx.put("a", 0);
        idx.put("b", 1);
        idx.put("c", 2);
        List<Res> l = new ArrayList<>();
        // a: 1 из 1 ошибка; b: 1 из 4 – принят, но чужое имя; c – чужой.
        l.add(new Res("a", 0, true, false, false, att("a")));
        l.add(new Res("b", 0, true, true, false, att("b")));
        for (int i = 0; i < 3; i++) l.add(new Res("b", 0, true, true, true, att("b")));
        l.add(new Res("c", 0, false, true, false, att("c")));
        l.add(new Res("c", 0, false, false, false, att("c")));
        int[] unit = {1, 1, 1};
        assertEquals(100 * (1 + 0.25) / 2, XBaseChokeRun.errPersons(l, idx, unit), 1e-12);
        assertEquals(100.0 * 2 / 5, XBaseChokeRun.errAtt(l, idx, unit), 1e-12);
        assertEquals(50.0, XBaseChokeRun.farAtt(l, idx, unit), 1e-12);
        assertEquals(100.0 * 3 / 5, XBaseChokeRun.rightShare(l), 1e-12);
        assertEquals(25.0, XBaseChokeRun.errPersons(l, idx, new int[] {0, 2, 1}), 1e-12);
    }

    @Test
    public void verdictWordsFollowSectionSix() {
        assertTrue(XBaseChokeRun.verdict(40, 30, 50).endsWith("хуже ожидаемого"));
        assertTrue(XBaseChokeRun.verdict(10, 5, 20).endsWith("лучше ожидаемого"));
        assertTrue(XBaseChokeRun.verdict(25, 15, 35).endsWith("согласуется"));
    }
}
