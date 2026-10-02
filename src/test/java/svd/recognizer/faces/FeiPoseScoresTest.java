package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

/**
 * +pose (этап 5б): выбор модели по позе пробы в разборе файла оценок. r0 выше всех |r| — ровно X; r0 ниже всех — ровно
 * X+mirror; граница |r| = r0 — X; выбор по (роль, человек, снимок) у всех проб; сетка r0 — вне основных таблиц;
 * r0 файла и ключа согласованы.
 *
 * @author ssv
 */
public class FeiPoseScoresTest {

    private static final double[] R = {0.05, 0.12, 0.2, -0.3};

    /** Поза: |r| снимка (человек, номер) — из R по кругу. */
    private static FeiPose pose(List<FeiScores.Row> rows) {
        List<String> lines = new ArrayList<>();
        lines.add("person\tnumber\tname\tfound\tr\treference\tincluded\treason");
        java.util.Set<String> seen = new java.util.HashSet<>();
        int i = 0;
        for (FeiScores.Row r : rows) {
            if (seen.add(r.person() + "-" + r.image())) {
                lines.add(r.person() + "\t" + r.image() + "\t-\t1\t" + R[i++ % R.length] + "\t-\t1\t-");
            }
        }
        return FeiPose.parse(lines);
    }

    /** Группы метода: конфигурации (0, 0) и (0, 4); свои 6, пороговые 8 по 3 снимка, контроль 4 по 3. */
    private static List<FeiScores.Row> rows(String method, long seed) {
        Random rnd = new Random(seed);
        List<FeiScores.Row> out = new ArrayList<>();
        for (int j : new int[] {0, 4}) {
            for (int p = 0; p < 6; p++) {
                String id = "o" + p;
                double v = rnd.nextDouble();
                out.add(new FeiScores.Row(method, 0, j, FeiScores.OWN, id, FeiProtocol.NUMBERS[j], id, rnd.nextInt(4) == 0 ? "o9" : id, v, "o8",
                        v + rnd.nextDouble(), v, -1));
            }
            for (String role : List.of(FeiScores.THR, FeiScores.CTRL)) {
                for (int p = 0; p < (role.equals(FeiScores.THR) ? 8 : 4); p++) {
                    for (int q = 1; q <= 3; q++) {
                        double v = rnd.nextDouble() * 1.2;
                        out.add(new FeiScores.Row(method, 0, j, role, role + p, q, FeiScores.NONE, "o0", v, "o1", v + rnd.nextDouble(), Double.NaN, -1));
                    }
                }
            }
        }
        return out;
    }

    private static FeiScores.Analysis analysis(double r0) {
        List<FeiScores.Row> base = rows("2c-mlda", 1);
        List<FeiScores.Row> mir = rows("2c-mlda+mirror", 2);
        FeiScores.Analysis a = new FeiScores.Analysis();
        a.pose = pose(base);
        a.r0 = r0;
        // Порядок файла: в каждой конфигурации база, затем отражённая строка.
        for (int j : new int[] {0, 4}) {
            for (FeiScores.Row r : base) if (r.j() == j) a.add(r);
            for (FeiScores.Row r : mir) if (r.j() == j) a.add(r);
        }
        a.finish();
        return a;
    }

    private static void assertSameStat(FeiScores.Stat x, FeiScores.Stat y) {
        assertEquals(x.att, y.att);
        for (int f = 0; f < 2; f++) {
            assertArrayEquals(x.rej[f], y.rej[f]);
            assertArrayEquals(x.mis[f], y.mis[f]);
        }
        for (int k = 0; k < FeiScores.THRESHOLDS; k++) {
            assertEquals(x.theta.get(k), y.theta.get(k));
            for (int c = 0; c < x.farCtrl.get(k).size(); c++) {
                assertArrayEquals(x.farCtrl.get(k).get(c), y.farCtrl.get(k).get(c));
                assertArrayEquals(x.farThr.get(k).get(c), y.farThr.get(k).get(c));
            }
        }
    }

    @Test
    public void extremesGiveBaseOrMirror() {
        FeiScores.Analysis all = analysis(1.0);
        assertSameStat(all.stats.get("2c-mlda"), all.stats.get("2c-mlda+pose"));
        assertSameStat(all.stats.get("2c-mlda-ratio"), all.stats.get("2c-mlda-ratio+pose"));
        FeiScores.Analysis none = analysis(0.0);
        assertSameStat(none.stats.get("2c-mlda+mirror"), none.stats.get("2c-mlda+pose"));
        assertSameStat(none.stats.get("2c-mlda-ratio+mirror"), none.stats.get("2c-mlda-ratio+pose"));
    }

    @Test
    public void choosesByPoseOfEachProbe() {
        List<FeiScores.Row> base = rows("x", 1).stream().filter(r -> r.j() == 4).toList();
        List<FeiScores.Row> mir = rows("x+mirror", 2).stream().filter(r -> r.j() == 4).toList();
        FeiPose pose = pose(base);
        List<FeiScores.Row> c = FeiScores.choose(base, mir, pose, 0.12, "x+pose");
        assertEquals(mir.size(), c.size());
        int fromBase = 0;
        for (int i = 0; i < c.size(); i++) {
            FeiScores.Row m = mir.get(i);
            FeiScores.Row b = base.stream().filter(r -> r.role().equals(m.role()) && r.person().equals(m.person()) && r.image() == m.image())
                    .findFirst().orElseThrow();
            boolean useBase = pose.absR(m.person(), m.image()) <= 0.12;
            FeiScores.Row src = useBase ? b : m;
            if (useBase) fromBase++;
            assertEquals("x+pose", c.get(i).method());
            assertEquals(src.best(), c.get(i).best(), 0);
            assertEquals(src.second(), c.get(i).second(), 0);
            assertEquals(src.bestId(), c.get(i).bestId());
            assertEquals(src.trueScore(), c.get(i).trueScore(), 0);
        }
        // |r| = 0,05 и 0,12 (граница) — база, 0,2 и 0,3 — отражённая модель; есть и те, и другие во всех ролях.
        assertTrue(fromBase > 0 && fromBase < c.size());
    }

    @Test
    public void gridIsReferenceOnly() {
        FeiScores.Analysis a = analysis(0.12);
        assertEquals(FeiScores.POSE_GRID.length, a.grid.size());
        assertTrue(a.grid.contains(FeiScores.gridId(0.12)));
        assertFalse(a.main().stream().anyMatch(s -> a.grid.contains(s.id)));
        // Сетка при r0 = 0,12 — та же строка, что основная 2c-mlda-ratio+pose.
        assertSameStat(a.stats.get("2c-mlda-ratio+pose"), a.stats.get(FeiScores.gridId(0.12)));
        String text = FeiScores.report(a, "test").toString();
        assertTrue(text.contains("НЕ ДЛЯ ВЫБОРА"));
        assertTrue(text.contains("справочно: шкалы X и X+mirror разные"));
    }

    @Test
    public void r0OfFileMustMatchKey() throws Exception {
        String file = FeiScores.header("c", "d", 1, List.of("2c-mlda"), List.of(FeiScores.R0_HEADER + 0.12))
                + String.join("\t", "2c-mlda", "0", "4", "own", "o0", "11", "o0", "o0", "0.1", "o1", "0.2", "0.1", "-") + "\n";
        FeiScores.Analysis noKey = new FeiScores.Analysis();
        try {
            noKey.read(new BufferedReader(new StringReader(file)));
            throw new AssertionError("ожидалась ошибка: r0 в файле, ключа нет");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("faces.fei.pose.r0"));
        }
        FeiScores.Analysis other = new FeiScores.Analysis();
        other.pose = FeiPose.parse(List.of("o0\t11\t-\t1\t0.0\t-\t1\t-"));
        other.r0 = 0.15;
        try {
            other.read(new BufferedReader(new StringReader(file)));
            throw new AssertionError("ожидалась ошибка: r0 файла ≠ ключа");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("≠"));
        }
        FeiScores.Analysis same = new FeiScores.Analysis();
        same.pose = other.pose;
        same.r0 = 0.12;
        same.read(new BufferedReader(new StringReader(file)));
        assertSame(1L, same.lines);
    }
}
