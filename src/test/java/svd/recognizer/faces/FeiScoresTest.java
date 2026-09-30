package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * Файл оценок faces-fei (этап 5б): строка попытки восстанавливается точно; лучший и второй кандидаты, признак согласия.
 * Разбор на маленьком искусственном наборе: порог по снимкам и по людям, отношение и разность.
 *
 * @author ssv
 */
public class FeiScoresTest {

    private static final List<String> GALLERY = List.of("001", "002", "003", "004");

    @Test
    public void ownLineRoundTrip() {
        double[] v = {0.3, 0.1 + 0.2, 1e-17, 7.0 / 3};
        FeiScores.Row r = FeiScores.parse(line(FeiScores.OWN, 1, v, false));
        assertEquals("m", r.method());
        assertEquals(2, r.s());
        assertEquals(5, r.j());
        assertEquals(FeiScores.OWN, r.role());
        assertEquals(12, r.image());
        assertEquals("002", r.trueId());
        assertEquals("003", r.bestId());
        assertEquals(1e-17, r.best(), 0);
        assertEquals("001", r.secondId());
        assertEquals(0.3, r.second(), 0);
        assertEquals(0.1 + 0.2, r.trueScore(), 0);
        assertEquals(-1, r.noAgreement());
    }

    @Test
    public void impostorAndTies() {
        double[] v = {2, 1, 1, Double.POSITIVE_INFINITY};
        FeiScores.Row r = FeiScores.parse(line(FeiScores.THR, -1, v, false));
        assertEquals(FeiScores.NONE, r.trueId());
        assertEquals("002", r.bestId());
        assertEquals("003", r.secondId());
        assertEquals(1, r.second(), 0);
        assertEquals(Double.NaN, r.trueScore(), 0);
    }

    @Test
    public void agreement() {
        double m = Double.MAX_VALUE;
        FeiScores.Row yes = FeiScores.parse(line(FeiScores.OWN, 0, new double[] {m, -3.5, m, m}, true));
        assertEquals(0, yes.noAgreement());
        assertEquals("002", yes.bestId());
        assertEquals(-3.5, yes.best(), 0);
        assertEquals(FeiScores.NONE, yes.secondId());
        assertEquals(m, yes.trueScore(), 0);
        double inf = Double.POSITIVE_INFINITY;
        FeiScores.Row no = FeiScores.parse(line(FeiScores.CTRL, -1, new double[] {inf, inf, inf, inf}, true));
        assertEquals(1, no.noAgreement());
        assertEquals(FeiScores.NONE, no.bestId());
        assertEquals(inf, no.best(), 0);
    }

    /**
     * Пороговые: 20 человек по 5 снимков, оценка 1 + i + 20·q (i — человек, q — снимок); минимумы людей 1…20.
     * По снимкам (n = 100): FAR 1 % — допускается 1, θ < 2; FAR 5 % — 5, θ < 6. По людям (20): α = 0,05 — 1, θ < 2;
     * α = 0 — θ < 1. Свои (конфигурация j = 4 — анфас): A 1,5 верно; B 3 верно; C 0,5 под чужим; D 10 верно.
     * Контроль: c0 {1,5; 100}, c1 {5; 7}.
     */
    private static List<FeiScores.Row> group(String method) {
        return group(method, 0);
    }

    private static List<FeiScores.Row> group(String method, int split) {
        List<FeiScores.Row> rows = new ArrayList<>();
        double[] own = {1.5, 3, 0.5, 10};
        for (int p = 0; p < own.length; p++) {
            String id = GALLERY.get(p);
            rows.add(new FeiScores.Row(method, split, 4, FeiScores.OWN, id, 11, id, p == 2 ? GALLERY.get(0) : id, own[p], "x", own[p] + 100,
                    own[p], -1));
        }
        for (int i = 0; i < 20; i++) {
            for (int q = 0; q < 5; q++) {
                double v = 1 + i + 20 * q;
                rows.add(new FeiScores.Row(method, split, 4, FeiScores.THR, "t" + i, q, FeiScores.NONE, "001", v, "002", v + 100, Double.NaN, -1));
            }
        }
        double[][] ctrl = {{1.5, 100}, {5, 7}};
        for (int c = 0; c < ctrl.length; c++) {
            for (double v : ctrl[c]) {
                rows.add(new FeiScores.Row(method, split, 4, FeiScores.CTRL, "c" + c, 1, FeiScores.NONE, "001", v, "002", v + 100, Double.NaN, -1));
            }
        }
        return rows;
    }

    @Test
    public void thresholdsBySnapshotsAndPersons() {
        FeiScores.Stat s = new FeiScores.Stat("m", FeiScores.Transform.NONE);
        s.add(group("m"));
        assertEquals(Math.nextDown(2.0), s.theta.get(FeiScores.T1).get(0), 0);
        assertEquals(Math.nextDown(6.0), s.theta.get(FeiScores.T5).get(0), 0);
        assertEquals(Math.nextDown(2.0), s.theta.get(FeiScores.P05).get(0), 0);
        assertEquals(Math.nextDown(1.0), s.theta.get(FeiScores.P0).get(0), 0);
        assertEquals(4, s.att);
        assertEquals(4, s.attBy[1]);
        int[] rej = {2, 1, 2, 3};
        for (int k = 0; k < FeiScores.THRESHOLDS; k++) {
            assertEquals("отказы, порог " + k, rej[k], s.rej[1][k]);
            assertEquals("под чужим, порог " + k, 1, s.mis[1][k]);
            assertEquals(0, s.rej[0][k] + s.mis[0][k]);
        }
        assertArrayEquals(new int[] {1, 100, 1, 20}, s.farThr.get(FeiScores.T1).get(0));
        assertArrayEquals(new int[] {5, 100, 5, 20}, s.farThr.get(FeiScores.T5).get(0));
        assertArrayEquals(new int[] {1, 100, 1, 20}, s.farThr.get(FeiScores.P05).get(0));
        assertArrayEquals(new int[] {1, 4, 1, 2}, s.farCtrl.get(FeiScores.T1).get(0));
        assertArrayEquals(new int[] {2, 4, 2, 2}, s.farCtrl.get(FeiScores.T5).get(0));
        assertArrayEquals(new int[] {0, 4, 0, 2}, s.farCtrl.get(FeiScores.P0).get(0));
    }

    @Test
    public void ratioAndMargin() {
        assertEquals(0.5, FeiScores.Transform.RATIO.apply(2, 4), 0);
        assertEquals(1, FeiScores.Transform.RATIO.apply(0, 0), 0);
        assertEquals(Double.POSITIVE_INFINITY, FeiScores.Transform.RATIO.apply(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY), 0);
        assertEquals(-2, FeiScores.Transform.MARGIN.apply(-3, -1), 0);
        FeiScores.Analysis a = new FeiScores.Analysis();
        for (int split = 0; split < 2; split++) {
            for (String m : List.of("2c-mlda", "3-znorm-lda")) for (FeiScores.Row r : group(m, split)) a.add(r);
        }
        a.finish();
        FeiScores.Stat ratio = a.stats.get("2c-mlda-ratio");
        FeiScores.Stat margin = a.stats.get("3-znorm-lda-margin");
        assertNotNull(ratio);
        assertNotNull(margin);
        assertEquals(List.of("2c-mlda", "2c-mlda-ratio", "3-znorm-lda", "3-znorm-lda-margin"), new ArrayList<>(a.stats.keySet()));
        assertEquals(2, a.configs.size());
        // Порог по людям у исходной строки — как в thresholdsBySnapshotsAndPersons (оба разбиения одинаковы).
        assertEquals(4, a.stats.get("2c-mlda").rej[1][FeiScores.P05]);
        assertEquals(2, ratio.configs);
        // Отношение v / (v + 100) растёт с v: порядок чужих прежний, 6-й по величине снимок — 6/106.
        assertEquals(Math.nextDown(6.0 / 106), ratio.theta.get(FeiScores.T5).get(0), 0);
        // Разность v − (v + 100) у всех −100: ⌊0,05·100⌋ = 5 < 100 одинаковых — θ ниже −100, все отказаны и чужие не приняты.
        assertEquals(Math.nextDown(-100.0), margin.theta.get(FeiScores.T5).get(0), 0);
        assertEquals(8, margin.rej[1][FeiScores.T5]);
        assertEquals(0, margin.farCtrl.get(FeiScores.T5).get(0)[0]);
    }

    @Test
    public void groupsMustBeContiguous() {
        FeiScores.Analysis a = new FeiScores.Analysis();
        a.add(group("m").get(0));
        a.add(group("n").get(0));
        try {
            a.add(group("m").get(1));
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("не подряд"));
            return;
        }
        throw new AssertionError("ожидалась ошибка");
    }

    private static String line(String role, int trueIdx, double[] v, boolean agree) {
        String l = FeiScores.line("m", 2, 5, role, trueIdx < 0 ? "150" : GALLERY.get(trueIdx), 12, trueIdx, GALLERY, v, agree);
        assertEquals('\n', l.charAt(l.length() - 1));
        return l.substring(0, l.length() - 1);
    }
}
