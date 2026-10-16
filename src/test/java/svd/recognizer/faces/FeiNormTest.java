package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import org.junit.Test;

/**
 * Полные векторы (FeiVectors, части в FeiCheckpoint) и нормировка оценки (FeiNorm): строка читается без потерь, лучший,
 * второй и ранг; сборка файлов по методам; log d по правилу; T, Z, ZT на ручном примере; Z-статистики с весами;
 * расхождение вектора с файлом оценок — исключение; критерий выбора варианта.
 *
 * @author ssv
 */
public class FeiNormTest {

    @Test
    public void lineRoundTripBestTwoAndRank() {
        double[] v = {0.5, 0.1, 0.30000000000000004, 0.1, 7e-300};
        FeiVectors.Row r = FeiVectors.parse(FeiVectors.line("2c-mlda", 1, 3, FeiScores.OWN, "007", 13, "007", v).trim());
        assertArrayEquals(v, r.d(), 0);
        assertEquals("007", r.trueId());
        assertEquals(13, r.image());
        assertArrayEquals(new int[] {4, 1}, FeiVectors.bestTwo(v));
        // Равные — по индексу, как первый минимум.
        assertArrayEquals(new int[] {1, 3}, FeiVectors.bestTwo(new double[] {0.5, 0.1, 0.3, 0.1}));
        assertEquals(1, FeiVectors.rank(v, 4));
        assertEquals(2, FeiVectors.rank(v, 1));
        assertEquals(3, FeiVectors.rank(v, 3));
        assertEquals(5, FeiVectors.rank(v, 0));
    }

    @Test
    public void vectorPartsAssembledByMethod() throws Exception {
        Path dir = Files.createTempDirectory("feivec");
        FeiCheckpoint cp = new FeiCheckpoint(dir.resolve("checkpoint"));
        cp.vectors = true;
        cp.saveVectors(0, FeiVectors.line("m1", 0, 0, "own", "a", 4, "a", new double[] {1, 2}) + FeiVectors.line("m2", 0, 0, "own", "a", 4, "a",
                new double[] {3, 4}));
        cp.saveVectors(1, FeiVectors.line("m1", 0, 1, "thr", "b", 5, "-", new double[] {5, 6}));
        cp.assembleVectors(dir, List.of("m1", "m2"), m -> "# шапка " + m + "\n", 2);
        assertEquals(List.of("# шапка m1", "m1\t0\t0\town\ta\t4\ta\t1.0\t2.0", "m1\t0\t1\tthr\tb\t5\t-\t5.0\t6.0"), lines(dir.resolve(FeiVectors.file("m1"))));
        assertEquals(List.of("# шапка m2", "m2\t0\t0\town\ta\t4\ta\t3.0\t4.0"), lines(dir.resolve(FeiVectors.file("m2"))));
        CheckpointCleanup.delete(cp);
    }

    private static List<String> lines(Path f) throws Exception {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(f)), StandardCharsets.UTF_8))) {
            return in.lines().toList();
        }
    }

    @Test
    public void logdRule() {
        int[] c = new int[2];
        assertEquals(Math.log(2), FeiNorm.logd(2, false, c, "x"), 0);
        assertEquals(Math.log(FeiNorm.D_MIN), FeiNorm.logd(0, false, c, "x"), 0);
        assertEquals(1, c[FeiNorm.L_FLOOR]);
        assertEquals(Math.log(FeiNorm.D_MIN), FeiNorm.logd(-1e-13, true, c, "x"), 0);
        assertEquals(1, c[FeiNorm.L_COS]);
        assertEquals(2, c[FeiNorm.L_FLOOR]);
        for (double bad : new double[] {-1e-13, Double.NaN, Double.POSITIVE_INFINITY}) {
            try {
                FeiNorm.logd(bad, false, c, "x");
                throw new AssertionError("ожидалось исключение: " + bad);
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("попытка x"));
            }
        }
    }

    @Test
    public void tNormExcludesCandidate() {
        assertEquals(-2, FeiNorm.tNorm(new double[] {0, 1, 2, 3}, 0), 1e-15);
        assertEquals(0, FeiNorm.tNorm(new double[] {1, 0, 2, 3}, 2) - (2 - 4.0 / 3) / Math.sqrt(7.0 / 3), 1e-15);
    }

    private static FeiNorm.Att att(String role, int p, int mode, double... l) {
        FeiNorm.Att a = new FeiNorm.Att();
        a.role = role;
        a.person = "x" + p;
        a.p = p;
        a.mode = mode;
        a.l = l;
        return a;
    }

    /** Конфигурация: пороговые люди 0 (2 снимка) и 1 (1 снимок) в режиме X, человек 2 (2 снимка) в режиме X+mirror; галерея из 3. */
    private static FeiNorm.Conf conf() {
        FeiNorm.Conf c = new FeiNorm.Conf();
        c.all.add(att(FeiScores.THR, 0, 0, 1, 2, 3));
        c.all.add(att(FeiScores.THR, 0, 0, 2, 4, 3));
        c.all.add(att(FeiScores.THR, 1, 0, 3, 6, 6));
        c.all.add(att(FeiScores.THR, 2, 1, 5, 5, 5.5));
        c.all.add(att(FeiScores.THR, 2, 1, 7, 6, 6.5));
        FeiNorm.sums(c);
        return c;
    }

    @Test
    public void zStatsPerModeAndWeighted() {
        FeiNorm.Conf c = conf();
        double[][][] z = FeiNorm.zStats(c, null);
        // Режим X, человек галереи 0: значения 1, 2, 3 → μ = 2, σ = 1.
        assertEquals(2, z[0][0][0], 1e-12);
        assertEquals(1, z[1][0][0], 1e-12);
        // Режим X+mirror, человек галереи 1: 5, 6 → μ = 5,5, σ = √0,5.
        assertEquals(5.5, z[0][1][1], 1e-12);
        assertEquals(Math.sqrt(0.5), z[1][1][1], 1e-12);
        // Вес 2 у человека 1 — как его снимок дважды: 1, 2, 3, 3 → μ = 2,25, σ² = 2,75 / 3.
        double[][][] w = FeiNorm.zStats(c, new int[] {1, 2, 1});
        assertEquals(2.25, w[0][0][0], 1e-12);
        assertEquals(Math.sqrt(2.75 / 3), w[1][0][0], 1e-12);
        // Z и ZT пробы режима X с кандидатом 0.
        FeiNorm.Att a = att(FeiScores.OWN, 0, 0, 0, 2, 3);
        a.cand = 0;
        assertEquals(-2, FeiNorm.score(a, FeiNorm.V_Z, z, "x"), 1e-12);
        double[] zs = {(0 - z[0][0][0]) / z[1][0][0], (2 - z[0][0][1]) / z[1][0][1], (3 - z[0][0][2]) / z[1][0][2]};
        assertEquals(FeiNorm.tNorm(zs, 0), FeiNorm.score(a, FeiNorm.V_ZT, z, "x"), 1e-12);
    }

    @Test
    public void vectorMustMatchScoresFile() {
        FeiVectors.Row v = new FeiVectors.Row("m", 0, 0, FeiScores.OWN, "a", 4, "a", new double[] {0.3, 0.1, 0.2});
        List<String> g = List.of("a", "b", "c");
        FeiScores.Row ok = new FeiScores.Row("m", 0, 0, FeiScores.OWN, "a", 4, "a", "b", 0.1, "c", 0.2, 0.3, -1);
        FeiNorm.check(v, g, ok);
        FeiScores.Row bad = new FeiScores.Row("m", 0, 0, FeiScores.OWN, "a", 4, "a", "b", 0.1, "c", 0.20000000000000004, 0.3, -1);
        try {
            FeiNorm.check(v, g, bad);
            throw new AssertionError("ожидалось исключение");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("не совпало"));
        }
    }

    /** Реплики: константа плюс шум ±0,1, у каждого вызова свой сдвиг (разность значима, если константы различаются больше чем на 0,2). */
    private static int phase;

    private static double[] reps(double c) {
        double[] r = new double[200];
        int s = phase++ * 7;
        for (int q = 0; q < r.length; q++) r[q] = c + 0.1 * Math.sin(q * 1.3 + s);
        return r;
    }

    @Test
    public void choiceRule() {
        // Нет значимых улучшений — база.
        assertEquals(FeiNorm.V_BASE, FeiNorm.choose(new double[] {20, 21, 20.05, 22}, new double[][] {reps(20), reps(21), reps(20.05), reps(22)}));
        // ZT лучше всех, но T значим против базы и от ZT значимо не отличается — T.
        List<double[]> r = new ArrayList<>(List.of(reps(20), reps(18), reps(19), reps(17.95)));
        assertEquals(FeiNorm.V_T, FeiNorm.choose(new double[] {20, 18, 19, 17.95}, r.toArray(double[][]::new)));
        // ZT значимо лучше T — ZT.
        assertEquals(FeiNorm.V_ZT, FeiNorm.choose(new double[] {20, 18, 19, 16}, new double[][] {reps(20), reps(18), reps(19), reps(16)}));
    }
}
