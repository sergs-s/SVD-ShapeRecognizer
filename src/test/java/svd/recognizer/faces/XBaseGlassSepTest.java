package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public class XBaseGlassSepTest {

    private static final double[] E1 = {1, 0};
    private static final double[] E2 = {0, 1};

    @Test
    public void personWithTwoCentersTakesNearest() {
        double[][][] centers = {{E1, E2}, {E2}};
        // Проба у E2: у двух центров – расстояние до E2, у одного – то же
        double[] d = XBaseGlassSep.minDistances(new double[] {0, 2}, centers);
        assertEquals(0, d[0], 1e-12);
        assertEquals(0, d[1], 1e-12);
        // Проба у E1: первый человек близок, второй (один центр E2) – ортогонален
        double[] e = XBaseGlassSep.minDistances(new double[] {3, 0}, centers);
        assertEquals(0, e[0], 1e-12);
        assertEquals(1, e[1], 1e-12);
    }

    @Test
    public void oneCenterIsPlainDistance() {
        double[] y = {1, 1};
        assertEquals(XBaseVec.cos(y, E1), XBaseGlassSep.minDistances(y, new double[][][] {{E1}})[0], 0);
    }

    @Test
    public void splitByPrevailingState() {
        Map<String, String> labels = Map.of("a", "1", "b", "0", "c", "1");
        assertEquals(List.of(List.of("a", "c"), List.of("b")), XBaseGlassSep.splitState("p", List.of("a", "b", "c"), labels, true));
        assertEquals(List.of(List.of("b"), List.of("a", "c")), XBaseGlassSep.splitState("p", List.of("a", "b", "c"), labels, false));
    }

    @Test
    public void splitRejectsOneStateAndUnknown() {
        assertThrows(IllegalStateException.class, () -> XBaseGlassSep.splitState("p", List.of("a", "c"), Map.of("a", "1", "c", "1"), true));
        assertThrows(IllegalStateException.class, () -> XBaseGlassSep.splitState("p", List.of("a", "c"), Map.of("a", "1", "c", "?"), true));
    }

    private static XBaseVec.Model model(double v) {
        return new XBaseVec.Model("main", "m", 0, List.of("p"), List.of(new XBaseVec.Row("own", "p", 1, "k", 0, new double[] {v}, new double[0], new double[0])), new double[1][]);
    }

    @Test
    public void checkSameStopsOnDifference() {
        XBaseGlassSep.checkSame(List.of(model(0.5)), List.of(model(0.5)), "t");
        assertThrows(IllegalStateException.class, () -> XBaseGlassSep.checkSame(List.of(model(0.5)), List.of(model(Math.nextUp(0.5))), "t"));
    }
}
