package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;

public class XBaseVarSelTest {

    private static XBaseVarSel.Pt p(String k, int no, double a, double b) {
        return new XBaseVarSel.Pt(k, no, a, b);
    }

    @Test
    public void firstNearestToMeanThenFarthest() {
        List<XBaseVarSel.Pt> pts = List.of(p("c", 3, 0.1, 0), p("a", 1, 5, 5), p("b", 2, -3, 0), p("d", 4, 0.2, 0.1));
        assertEquals(List.of("c", "a", "b"), XBaseVarSel.greedy(pts, 3));
    }

    @Test
    public void tieGoesToSmallerNumber() {
        List<XBaseVarSel.Pt> pts = List.of(p("z", 9, 1, 0), p("y", 2, -1, 0), p("x", 5, 0, 0));
        assertEquals(List.of("x", "y"), XBaseVarSel.greedy(pts, 2));
        List<XBaseVarSel.Pt> two = List.of(p("m", 7, 1, 0), p("n", 3, -1, 0));
        assertEquals(List.of("n"), XBaseVarSel.greedy(two, 1));
    }

    @Test
    public void fewerThanNReturnsAll() {
        List<XBaseVarSel.Pt> pts = List.of(p("b", 2, 1, 0), p("a", 1, 0, 0));
        assertEquals(List.of("a", "b"), XBaseVarSel.greedy(pts, 6));
    }
}
