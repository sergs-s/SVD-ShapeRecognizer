package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import svd.recognizer.faces.FeiScores.Row;

/**
 * Разложение ошибки своих (FeiDecompose): группы (i)–(v-б) на ручном примере, сумма групп и ошибка как в FeiScores.Stat,
 * деление по позе, запас до порога, бутстреп при весах 1 совпадает с точкой.
 *
 * @author ssv
 */
public class FeiDecomposeTest {

    private static final double INF = Double.POSITIVE_INFINITY;

    private static Row own(String id, String best, double d1, double d2, double trueScore) {
        return new Row("2c-mlda+pose", 0, 4, FeiScores.OWN, id, 11, id, best, d1, "o8", d2, trueScore, -1);
    }

    /** Пороговые: 20 человек × 5 снимков, r = 0,50…0,99; свои o0–o5 — по одному в каждой группе. */
    private static List<Row> rows() {
        List<Row> out = new ArrayList<>();
        out.add(own("o0", "o0", 0.1, 1, 0.1));          // (i) верный, принят
        out.add(own("o1", "o1", 0.9, 1, 0.9));          // (ii) верный, отказ
        out.add(own("o2", "o9", 0.9, 1, 2));            // (iii) неверный, отказ
        out.add(own("o3", "o9", 0.1, 1, 2));            // (iv) неверный, принят
        out.add(own("o4", "o4", 0.1, INF, 0.1));        // (v-а) d₂ = +∞
        out.add(own("o5", "o5", 0.1, 1, INF));          // (v-б) оценка истинного +∞
        for (int i = 0; i < 100; i++) {
            out.add(new Row("2c-mlda+pose", 0, 4, FeiScores.THR, "t" + (10 + i / 5), i % 5 + 1, FeiScores.NONE, "o0", 0.5 + i * 0.005, "o1", 1,
                    Double.NaN, -1));
        }
        return out;
    }

    private static List<String> persons() {
        List<String> p = new ArrayList<>();
        for (int i = 0; i < 6; i++) p.add("o" + i);
        for (int i = 10; i < 30; i++) p.add("t" + i);
        p.sort(null);
        return p;
    }

    /** Поза: o0–o2 анфас (|r| = 0,05), o3–o5 поворот (0,3). */
    private static FeiPose pose() {
        List<String> l = new ArrayList<>(List.of("person\tnumber\tname\tfound\tr\treference\tincluded\treason"));
        for (int i = 0; i < 6; i++) l.add("o" + i + "\t11\t-\t1\t" + (i < 3 ? "0.05" : "-0.3") + "\t-\t1\t-");
        return FeiPose.parse(l);
    }

    private static FeiDecompose decompose() {
        FeiDecompose d = new FeiDecompose(List.of("A"), pose(), 0.12, new FeiBootstrap(persons()));
        d.part("A", rows());
        return d;
    }

    @Test
    public void groupsMatchHandExample() {
        FeiDecompose d = decompose();
        d.check();
        double[][][] t = FeiDecompose.counts(d.cfgs.get(0), null);
        for (int k = 0; k < FeiDecompose.KS; k++) {
            assertArrayEquals(new double[] {1, 1, 1, 1, 1, 1}, t[k][0], 0);
            assertArrayEquals(new double[] {1, 1, 1, 0, 0, 0}, t[k][1], 0);
            assertArrayEquals(new double[] {0, 0, 0, 1, 1, 1}, t[k][2], 0);
            assertArrayEquals(t[k][0], t[k][3], 0);
            assertEquals(5, d.stats[0].errors(k == 0 ? FeiScores.T1 : FeiScores.T5));
        }
    }

    @Test
    public void marginOfRejectedCorrect() {
        FeiDecompose d = decompose();
        FeiDecompose.Cfg c = d.cfgs.get(0).get(0);
        double th = FeiDecompose.threshold(c, null, 1);
        assertEquals(Math.nextDown(0.5 + 5 * 0.005), th, 0);
        assertArrayEquals(new double[] {Math.log(0.9) - Math.log(th)}, FeiDecompose.margins(d.cfgs.get(0), 1, 0), 1e-15);
        assertEquals(0, FeiDecompose.margins(d.cfgs.get(0), 1, 2).length);
    }

    @Test
    public void unitWeightsEqualPoint() {
        FeiDecompose d = decompose();
        double[][][] a = FeiDecompose.counts(d.cfgs.get(0), null);
        double[][][] b = FeiDecompose.counts(d.cfgs.get(0), FeiBootstrap.unit(persons().size()));
        for (int k = 0; k < FeiDecompose.KS; k++) for (int sp = 0; sp < FeiDecompose.SPLITS; sp++) assertArrayEquals(a[k][sp], b[k][sp], 0);
    }
}
