package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Диагностика GT (PLAN.md п. 4.2, «Диагностика L0 и SFace»): оценка L0-mean = χ² до средней гистограммы, L0 = наименьшее
 * χ² по снимкам (при +mirror — и по отражениям), SFace — без модели +mirror, строка выбирается всегда X, разность падений.
 *
 * @author ssv
 */
public class XBaseDiagTest {

    static XBaseEval x() {
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new HashMap<>();
        Map<String, double[]> m = new HashMap<>();
        v.put("a1", new double[] {0.5, 0.5, 0, 0});
        v.put("a2", new double[] {0.3, 0.3, 0.4, 0});
        v.put("b1", new double[] {0, 0, 0.5, 0.5});
        m.put("a1", new double[] {0.6, 0.4, 0, 0});
        m.put("a2", new double[] {0.2, 0.2, 0.2, 0.4});
        m.put("b1", new double[] {0, 0.1, 0.4, 0.5});
        for (XBaseEval.V w : new XBaseEval.V[] {XBaseEval.V.L0, XBaseEval.V.L0MEAN}) {
            x.vec.put(w, v);
            x.mir.put(w, m);
        }
        return x;
    }

    @Test
    public void l0MeanIsChiSquareToMeanHistogram() {
        XBaseEval x = x();
        List<List<String>> gallery = List.of(List.of("a1", "a2"), List.of("b1"));
        double[] probe = {0.4, 0.2, 0.2, 0.2};
        double[] s = x.fit(XBaseEval.V.L0MEAN, false, gallery, List.of()).scores(probe);
        assertEquals(ChiSquareScorer.chi2(probe, new double[] {0.4, 0.4, 0.2, 0}), s[0], 1e-15);
        assertEquals(ChiSquareScorer.chi2(probe, x.vec.get(XBaseEval.V.L0).get("b1")), s[1], 1e-15);
        // +mirror: средняя по снимкам и их отражениям.
        double[] sm = x.fit(XBaseEval.V.L0MEAN, true, gallery, List.of()).scores(probe);
        assertEquals(ChiSquareScorer.chi2(probe, new double[] {0.4, 0.35, 0.15, 0.1}), sm[0], 1e-15);
    }

    @Test
    public void l0IsNearestSnapshotWithMirrors() {
        XBaseEval x = x();
        List<List<String>> gallery = List.of(List.of("a1", "a2"), List.of("b1"));
        double[] probe = {0.2, 0.2, 0.2, 0.4};
        double[] s = x.fit(XBaseEval.V.L0, false, gallery, List.of()).scores(probe);
        double c1 = ChiSquareScorer.chi2(probe, x.vec.get(XBaseEval.V.L0).get("a1"));
        double c2 = ChiSquareScorer.chi2(probe, x.vec.get(XBaseEval.V.L0).get("a2"));
        assertEquals(Math.min(c1, c2), s[0], 0);
        // Отражение a2 совпадает с пробой — у +mirror ближайший снимок на нулевом расстоянии.
        assertEquals(0, x.fit(XBaseEval.V.L0, true, gallery, List.of()).scores(probe)[0], 0);
    }

    @Test
    public void sfaceHasNoMirrorModel() {
        try {
            new XBaseEval().fit(XBaseEval.V.SFACE, true, List.of(), List.of());
            fail();
        } catch (IllegalArgumentException expected) {
            assertEquals("SFace: модели +mirror нет", expected.getMessage());
        }
        FeiScores.Row row = FeiScores.parse(FeiScores.line("0-sface", 0, 0, "own", "s01", 1, 0, List.of("s01", "s02"), new double[] {0.1, 0.4}, false).strip());
        Map<String, FeiScores.Row> rows = Map.of("0-sface\town\ts01\t1", row);
        assertSame(row, XBaseEval.choose(rows, XBaseEval.V.SFACE, "own", "s01", 1, false));
        assertSame(row, XBaseEval.choose(rows, XBaseEval.V.SFACE, "own", "s01", 1, true));
    }

    @Test
    public void dropDifference() {
        assertArrayEquals(new double[] {(5 - 1) - (7 - 2), (0 - 0) - (1 - 3)},
                LbpNormReport.dd(new double[] {5, 0}, new double[] {1, 0}, new double[] {7, 1}, new double[] {2, 3}), 0);
    }
}
