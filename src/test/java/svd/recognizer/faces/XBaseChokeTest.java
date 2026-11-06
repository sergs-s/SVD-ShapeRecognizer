package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import org.junit.Test;

public class XBaseChokeTest {

    @Test
    public void parseTakesPersonAndFrameOnly() {
        List<XBaseChoke.Rec> r = XBaseChoke.parse(List.of(
                "<dataset name=\"P2E_S1_C3\">",
                " <frame number=\"00000300\"/>",
                " <frame number=\"00000301\">",
                "  <person id=\"0014\">",
                "   <leftEye x=\"1\" y=\"2\"/>",
                "   <rightEye x=\"3\" y=\"4\"/>",
                "  </person>",
                " </frame>",
                " <frame number=\"00002119\">",
                "  <person id=\"0028\">",
                "   <rightEye x=\"3\" y=\"4\"/>",
                "  </person>",
                " </frame>"));
        assertEquals(2, r.size());
        assertEquals(new XBaseChoke.Rec(301, "0014", true), r.get(0));
        assertEquals(new XBaseChoke.Rec(2119, "0028", false), r.get(1));
        assertFalse(r.get(1).eyes());
    }

    @Test
    public void parseRejectsBrokenStructure() {
        assertThrows(IllegalStateException.class, () -> XBaseChoke.parse(List.of("<person id=\"0001\">")));
        assertThrows(IllegalStateException.class, () -> XBaseChoke.parse(List.of("<frame number=\"1\">", "<person id=\"0001\">")));
    }

    @Test
    public void splitIsStratifiedAndReproducible() {
        List<String> g28 = new ArrayList<>();
        for (int i = 0; i < 21; i++) g28.add(String.format("%04d", 100 + i));
        List<String> g45 = List.of("0003", "0015", "0021", "0022");
        List<List<String>> a = XBaseChoke.split(g28, g45, 12042);
        assertEquals(a, XBaseChoke.split(g28, g45, 12042));
        assertEquals(13, a.get(0).size());
        assertEquals(12, a.get(1).size());
        assertEquals(2, a.get(0).stream().filter(g45::contains).count());
        assertEquals(2, a.get(1).stream().filter(g45::contains).count());
        TreeSet<String> all = new TreeSet<>(a.get(0));
        all.addAll(a.get(1));
        assertEquals(25, all.size());
        assertThrows(IllegalStateException.class, () -> XBaseChoke.split(g28.subList(0, 20), g45, 1));
    }

    @Test
    public void frontalCamerasMatchReadme() {
        Map<String, int[]> f = XBaseChoke.FRONTAL;
        assertEquals("P1E_S1_C1", XBaseChoke.seqName("P1E", 1, f.get("P1E")[0], ""));
        assertEquals("P1L_S4_C1", XBaseChoke.seqName("P1L", 4, f.get("P1L")[3], ""));
        assertEquals("P2E_S1_C3.1", XBaseChoke.seqName("P2E", 1, f.get("P2E")[0], "1"));
        assertEquals("P2E_S3_C1.2", XBaseChoke.seqName("P2E", 3, f.get("P2E")[2], "2"));
        assertEquals("P2L_S3_C3.1", XBaseChoke.seqName("P2L", 3, f.get("P2L")[2], "1"));
        assertEquals("P2L_S4_C2.2", XBaseChoke.seqName("P2L", 4, f.get("P2L")[3], "2"));
    }

    @Test
    public void thresholdNodesForGallerySizes() {
        for (int n : XBaseChoke.GALLERY_N) {
            assertEquals(XBaseChoke.T_SFACE, XBaseFinalReport.sfaceT(n), 0);
            assertTrue(n > 10 && n < 50);
        }
    }
}
