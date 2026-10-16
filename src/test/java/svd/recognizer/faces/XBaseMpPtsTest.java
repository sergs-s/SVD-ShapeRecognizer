package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/**
 * Этап 1 MediaPipe (XBaseMpPts): ранги с равными, Спирмен, AUC, перевод в градусы на другой половине людей, чтение кэша точек.
 *
 * @author ssv
 */
public class XBaseMpPtsTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void ranksWithTies() {
        assertArrayEquals(new double[] {1, 2.5, 2.5, 4}, XBaseMpPts.ranks(new double[] {1, 5, 5, 9}), 0);
        assertArrayEquals(new double[] {3, 1, 2}, XBaseMpPts.ranks(new double[] {7, -1, 0}), 0);
    }

    @Test
    public void spearmanMonotone() {
        assertEquals(1, XBaseMpPts.spearman(new double[] {1, 2, 3, 4}, new double[] {1, 8, 27, 64}), 1e-12);
        assertEquals(-1, XBaseMpPts.spearman(new double[] {1, 2, 3, 4}, new double[] {4, 3, 2, 1}), 1e-12);
    }

    @Test
    public void aucCountsTiesAsHalf() {
        assertEquals(1, XBaseMpPts.auc(new double[] {0, 1}, new double[] {2, 3}), 0);
        assertEquals(0.5, XBaseMpPts.auc(new double[] {1}, new double[] {1}), 0);
        assertEquals(0.75, XBaseMpPts.auc(new double[] {0, 2}, new double[] {1, 3}), 0);
        assertEquals("1.000", XBaseMpPts.aucAbs(new double[] {0, 0, 22.5, -22.5}, new double[] {0.1, -0.1, -5, 6}, 22.5));
    }

    @Test
    public void outOfSampleUsesOtherHalf() {
        // Половина A: nom = 2x; половина B: nom = 2x + 10. Снимок A предсказывается прямой B и наоборот.
        double[] x = {1, 2, 1, 2};
        double[] nom = {2, 4, 12, 14};
        String[] person = {"a", "a", "b", "b"};
        double[] e = XBaseMpPts.outOfSample(x, nom, List.of(0, 1, 2, 3), person, Set.of("a"));
        assertArrayEquals(new double[] {10, 10, 10, 10}, e, 1e-12);
    }

    @Test
    public void readsCache() throws Exception {
        Path f = tmp.getRoot().toPath().resolve("mp_points.tsv");
        List<String> h = new ArrayList<>(List.of("key", "found", "faces", "w", "h"));
        for (String n : new String[] {"iris_re", "iris_le", "eye_re_out", "eye_re_in", "eye_le_in", "eye_le_out", "nose", "mouth_r", "mouth_l"})
            for (String c : new String[] {"_x", "_y", "_z"}) h.add(n + c);
        h.addAll(List.of("yaw", "pitch", "roll"));
        for (int i = 0; i < 16; i++) h.add("m" + i);
        h.addAll(List.of("eyeBlinkRight", "eyeBlinkLeft"));
        StringBuilder found = new StringBuilder("k1\t1\t1\t10\t20");
        for (int i = 0; i < 27; i++) found.append('\t').append(i);
        found.append("\t5.5\t1\t2");
        for (int i = 0; i < 16; i++) found.append("\t0");
        found.append("\t0.25\t0.75");
        StringBuilder none = new StringBuilder("k2\t0\t0\t10\t20");
        for (int i = 5; i < h.size(); i++) none.append('\t');
        Files.writeString(f, "# шапка\n" + String.join("\t", h) + "\n" + found + "\n" + none + "\n", StandardCharsets.UTF_8);
        List<String> head = new ArrayList<>();
        Map<String, XBaseMpPts.Mp> m = XBaseMpPts.read(f, head);
        assertEquals(List.of("# шапка"), head);
        XBaseMpPts.Mp a = m.get("k1");
        assertArrayEquals(new double[] {0, 1}, a.p().get("iris_re"), 0);
        assertArrayEquals(new double[] {18, 19}, a.p().get("nose"), 0);
        assertArrayEquals(new double[] {22.5, 23.5}, a.mid("mouth_r", "mouth_l"), 0);
        assertEquals(5.5, a.yaw(), 0);
        assertEquals(0.25, a.blinkR(), 0);
        assertEquals(0.75, a.blinkL(), 0);
        assertEquals(false, m.get("k2").found());
    }
}
