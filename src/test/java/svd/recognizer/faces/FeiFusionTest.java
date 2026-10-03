package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;
import org.junit.Test;
import svd.recognizer.faces.FeiScores.Row;

/**
 * Шаг (а1), объединение по решениям (FeiFusion) и правило вырожденных случаев отношения d₁/d₂ (FeiScores.ratio):
 * случаи 1–6; при одной части объединение совпадает с частью (и в бутстрепе); при несовпадении кандидатов — отказ;
 * m и s — только по пороговым и по всем их снимкам.
 *
 * @author ssv
 */
public class FeiFusionTest {

    private static final double INF = Double.POSITIVE_INFINITY;

    // ---------------------------------------------------------------- правило отношения

    private static Row row(String method, double d1, double d2) {
        return new Row(method, 1, 2, FeiScores.THR, "p7", 5, FeiScores.NONE, "o0", d1, "o1", d2, Double.NaN, -1);
    }

    private static void assertDataError(String method, double d1, double d2) {
        try {
            FeiScores.ratio(row(method, d1, d2), new int[FeiScores.CASES]);
            throw new AssertionError("ожидалось исключение: d₁ = " + d1 + ", d₂ = " + d2);
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("случай 3"));
            assertTrue(e.getMessage(), e.getMessage().contains(method + " s=1 j=2 thr p7 №5"));
        }
    }

    @Test
    public void case1ZeroBestIsBoundedBelow() {
        int[] c = new int[FeiScores.CASES];
        assertEquals(FeiScores.RHO_MIN, FeiScores.ratio(row("x", 0, 2), c), 0);
        assertEquals(FeiScores.RHO_MIN, FeiScores.ratio(row("x", 1e-20, 1), c), 0);
        assertEquals(2, c[FeiScores.CASE_TINY]);
        assertEquals(0.5, FeiScores.ratio(row("x", 1, 2), c), 0);
        assertEquals(2, c[FeiScores.CASE_TINY]);
    }

    @Test
    public void case2BothZeroIsWorst() {
        int[] c = new int[FeiScores.CASES];
        assertEquals(1, FeiScores.ratio(row("x", 0, 0), c), 0);
        assertEquals(1, c[FeiScores.CASE_BOTH_ZERO]);
    }

    @Test
    public void case3DataErrors() {
        assertDataError("x", 1, 0);
        assertDataError("x", 3, 2);
        assertDataError("x", -1, 2);
        assertDataError("x", 1, -2);
        assertDataError("x", Double.NaN, 2);
        assertDataError("x", 1, Double.NaN);
        assertDataError("x", INF, 2);
        // Допуск округления — только у 1 − cos.
        assertDataError("x", -1e-13, 2);
    }

    @Test
    public void case4NoSecondIsRejection() {
        int[] c = new int[FeiScores.CASES];
        assertEquals(INF, FeiScores.ratio(row("x", 1, INF), c), 0);
        assertEquals(INF, FeiScores.ratio(row("x", INF, INF), c), 0);
        assertEquals(2, c[FeiScores.CASE_NO_SECOND]);
        // Строки *-ratio разбора — та же функция.
        assertEquals(INF, FeiScores.Transform.RATIO.apply(row("x", 1, INF)), 0);
    }

    @Test
    public void case5CosineRounding() {
        String cos = "4-wpca-cos-0+pose";
        int[] c = new int[FeiScores.CASES];
        assertEquals(FeiScores.RHO_MIN, FeiScores.ratio(row(cos, -1e-13, 0.5), c), 0);
        assertEquals(1, c[FeiScores.CASE_COS]);
        assertEquals(1, c[FeiScores.CASE_TINY]);
        assertEquals(1, FeiScores.ratio(row(cos, -1e-13, -5e-13), c), 0);
        assertEquals(2, c[FeiScores.CASE_COS]);
        assertEquals(1, c[FeiScores.CASE_BOTH_ZERO]);
        assertDataError(cos, -1e-11, 0.5);
    }

    @Test
    public void case6ZeroSpreadIsError() {
        FeiFusion f = new FeiFusion(List.of("pA"), new int[][] {{0}}, null);
        try {
            f.part("pA", rows("2c-mlda+pose", 0, 0, new Random(1), r -> r.role().equals(FeiScores.THR) ? withRatio(r, 0.5) : r));
            throw new AssertionError("ожидалось исключение: s = 0");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("случай 6"));
        }
    }

    // ---------------------------------------------------------------- объединение

    private static final String[] OWN = {"o0", "o1", "o2", "o3", "o4", "o5"};

    /** Строка с заданным отношением (d₂ прежний). */
    private static Row withRatio(Row r, double ratio) {
        return new Row(r.method(), r.s(), r.j(), r.role(), r.person(), r.image(), r.trueId(), r.bestId(), ratio * r.second(), r.secondId(),
                r.second(), r.trueScore(), r.noAgreement());
    }

    private static Row withBest(Row r, String best) {
        return new Row(r.method(), r.s(), r.j(), r.role(), r.person(), r.image(), r.trueId(), best, r.best(), r.secondId(), r.second(),
                r.trueScore(), r.noAgreement());
    }

    /** Конфигурация (s, j): свои 6 (истинный — лучший у 3 из 4), пороговые 8 × 3 снимка, контроль 4 × 3. */
    private static List<Row> rows(String method, int s, int j, Random rnd, UnaryOperator<Row> edit) {
        List<Row> out = new ArrayList<>();
        for (String id : OWN) {
            double d1 = rnd.nextDouble();
            out.add(new Row(method, s, j, FeiScores.OWN, id, FeiProtocol.NUMBERS[j], id, rnd.nextInt(4) == 0 ? "o9" : id, d1, "o8",
                    d1 + 0.01 + rnd.nextDouble(), d1, -1));
        }
        for (String role : List.of(FeiScores.THR, FeiScores.CTRL)) {
            for (int p = 0; p < (role.equals(FeiScores.THR) ? 8 : 4); p++) {
                for (int q = 1; q <= 3; q++) {
                    double d1 = rnd.nextDouble();
                    out.add(new Row(method, s, j, role, role + p, q, FeiScores.NONE, OWN[rnd.nextInt(3)], d1, "o7", d1 + 0.01 + rnd.nextDouble(),
                            Double.NaN, -1));
                }
            }
        }
        return out.stream().map(edit).toList();
    }

    private static final int[][] CFGS = {{0, 0}, {0, 4}, {1, 5}};

    private static List<String> persons() {
        List<String> p = new ArrayList<>(List.of(OWN));
        for (int i = 0; i < 8; i++) p.add("thr" + i);
        for (int i = 0; i < 4; i++) p.add("ctrl" + i);
        p.sort(null);
        return p;
    }

    private static void assertSameDecisions(FeiScores.Stat x, FeiScores.Stat y) {
        assertEquals(x.att, y.att);
        for (int f = 0; f < 2; f++) {
            assertArrayEquals(x.rej[f], y.rej[f]);
            assertArrayEquals(x.mis[f], y.mis[f]);
        }
        for (int k = 0; k < FeiScores.THRESHOLDS; k++) {
            for (int c = 0; c < x.farCtrl.get(k).size(); c++) {
                assertArrayEquals(x.farCtrl.get(k).get(c), y.farCtrl.get(k).get(c));
                assertArrayEquals(x.farThr.get(k).get(c), y.farThr.get(k).get(c));
            }
        }
    }

    @Test
    public void singlePartEqualsPart() {
        FeiFusion f = new FeiFusion(List.of("pA"), new int[][] {{0}}, new FeiBootstrap(persons()));
        Random rnd = new Random(3);
        for (int[] c : CFGS) f.part("pA", rows("2c-mlda+pose", c[0], c[1], rnd, UnaryOperator.identity()));
        f.finish();
        assertSameDecisions(f.partStats[0], f.fusionStats[0]);
        FeiBootstrap.Result r = f.bootstrap(50, 11);
        assertArrayEquals(r.point.get("pA"), r.point.get("A"), 1e-12);
        for (int k = 0; k < FeiBootstrap.METRICS; k++) assertArrayEquals(r.reps.get("pA")[k], r.reps.get("A")[k], 1e-12);
        assertArrayEquals(r.cp.get("pA")[0], r.cp.get("A")[0]);
    }

    @Test
    public void disagreementIsRejection() {
        FeiFusion f = new FeiFusion(List.of("pA", "pB"), new int[][] {{0, 1}}, null);
        // Свои уверенно приняты обеими частями (отношение 0,01), у o2 лучший кандидат части B — другой человек.
        UnaryOperator<Row> sure = r -> r.own() ? withRatio(withBest(r, r.trueId()), 0.01) : r;
        List<Row> a = rows("2c-mlda+pose", 0, 0, new Random(5), sure);
        List<Row> b = a.stream().map(r -> r.own() && r.person().equals("o2") ? withBest(r, "o9") : r).toList();
        f.part("pA", a);
        f.part("pB", b);
        f.finish();
        FeiFusion.Att o2 = f.confs.get(0).own.stream().filter(x -> x.person.equals("o2")).findFirst().orElseThrow();
        assertNull(FeiFusion.candidate(o2, new int[] {0, 1}));
        assertEquals(INF, FeiFusion.score(o2, new int[] {0, 1}, new double[][] {{0, 1}, {0, 1}}), 0);
        FeiScores.Stat s = f.fusionStats[0];
        for (int k = 0; k < FeiScores.THRESHOLDS; k++) {
            assertEquals(1, s.frr(k));
            assertEquals(0, s.misaccepted(k));
        }
        assertEquals(1, f.disagreement(new int[] {0, 1})[0][0]);
    }

    @Test
    public void normalizationUsesAllThresholdSnapshotsOnly() {
        // Часть B: у половины пороговых другой лучший кандидат — на m и s части A это не влияет.
        List<Row> a = rows("2c-mlda+pose", 0, 4, new Random(7), UnaryOperator.identity());
        List<Row> b = a.stream().map(r -> r.role().equals(FeiScores.THR) && r.image() % 2 == 0 ? withBest(r, "o5") : r).toList();
        FeiFusion f = new FeiFusion(List.of("pA", "pB"), new int[][] {{0, 1}}, null);
        f.part("pA", a);
        f.part("pB", b);
        f.finish();
        double[] l = a.stream().filter(r -> r.role().equals(FeiScores.THR)).mapToDouble(r -> Math.log(r.best() / r.second())).toArray();
        assertEquals(24, l.length);
        double m = 0;
        for (double v : l) m += v;
        m /= l.length;
        double ss = 0;
        for (double v : l) ss += (v - m) * (v - m);
        double s = Math.sqrt(ss / (l.length - 1));
        assertEquals(m, f.norms.get(0).get(0)[0], 1e-12);
        assertEquals(s, f.norms.get(0).get(0)[1], 1e-12);
        // Свои и контроль не участвуют: другие их оценки — те же m и s.
        List<Row> a2 = a.stream().map(r -> r.role().equals(FeiScores.THR) ? r : withRatio(r, 0.3)).toList();
        FeiFusion g = new FeiFusion(List.of("pA", "pB"), new int[][] {{0, 1}}, null);
        g.part("pA", a2);
        g.part("pB", b);
        g.finish();
        assertArrayEquals(f.norms.get(0).get(0), g.norms.get(0).get(0), 0);
        assertArrayEquals(f.norms.get(1).get(0), g.norms.get(1).get(0), 0);
    }
}
