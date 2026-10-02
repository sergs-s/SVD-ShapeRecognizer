package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/**
 * Контрольные точки faces-fei: состояние сохраняется и читается без потерь; чужой отпечаток и неполные части — ошибка;
 * итоговый файл — шапка и части подряд; временных файлов не остаётся; каталог удаляется.
 *
 * @author ssv
 */
public class FeiCheckpointTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static FeiCheckpoint.State state(String fp) {
        FeiCheckpoint.State s = new FeiCheckpoint.State(fp, "abc (есть незакоммиченные изменения)", "экспорт X");
        FeiMethods.Result r = new FeiMethods.Result("2c-mlda");
        r.configs = 2;
        r.att = 200;
        r.rej[1][0] = 7;
        r.info.add("модель 0");
        r.farCtrl[0][0] = new int[] {1, 350, 1, 40};
        r.theta[0][1] = Math.nextDown(0.25);
        s.results.put(r.id, r);
        s.done = 2;
        s.timeText = "Конфигурация 0: 1 с.\nКонфигурация 1: 2 с.\n";
        return s;
    }

    @Test
    public void roundTrip() throws Exception {
        FeiCheckpoint cp = new FeiCheckpoint(tmp.getRoot().toPath().resolve("cp"));
        assertNull("нет контрольной точки — начать заново", cp.load("fp"));
        cp.savePart(0, "a\tb\n");
        cp.savePart(1, "c\td\n");
        cp.save(state("fp"));
        FeiCheckpoint.State s = cp.load("fp");
        assertEquals(2, s.done);
        assertEquals("abc (есть незакоммиченные изменения)", s.code);
        assertEquals("экспорт X", s.source);
        assertEquals("Конфигурация 0: 1 с.\nКонфигурация 1: 2 с.\n", s.timeText);
        FeiMethods.Result r = s.results.get("2c-mlda");
        assertEquals(200, r.att);
        assertEquals(7, r.rej[1][0]);
        assertEquals("модель 0", r.info.get(0));
        assertArrayEquals(new int[] {1, 350, 1, 40}, r.farCtrl[0][0]);
        assertEquals(Math.nextDown(0.25), r.theta[0][1], 0);
        try (var files = Files.list(cp.dir)) {
            assertFalse("временных файлов нет", files.anyMatch(p -> p.toString().endsWith(".tmp")));
        }

        Path out = tmp.getRoot().toPath().resolve("fei_scores.tsv.gz");
        cp.assemble(out, "# шапка\n", 2);
        try (InputStream in = new GZIPInputStream(Files.newInputStream(out))) {
            assertEquals("# шапка\na\tb\nc\td\n", new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
        cp.delete();
        assertFalse(Files.exists(cp.dir));
    }

    @Test
    public void otherFingerprintIsError() throws Exception {
        FeiCheckpoint cp = new FeiCheckpoint(tmp.getRoot().toPath().resolve("cp"));
        cp.savePart(0, "x\n");
        cp.savePart(1, "y\n");
        cp.save(state("код: src 1\nключи: seed 42"));
        try {
            cp.load("код: src 2\nключи: seed 42");
            throw new AssertionError("ожидалась ошибка");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("от другого прогона"));
            assertTrue(e.getMessage().contains("src 1") && e.getMessage().contains("src 2"));
        }
    }

    @Test
    public void missingPartIsError() throws Exception {
        FeiCheckpoint cp = new FeiCheckpoint(tmp.getRoot().toPath().resolve("cp"));
        cp.savePart(0, "x\n");
        cp.save(state("fp"));
        try {
            cp.load("fp");
            throw new AssertionError("ожидалась ошибка");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("conf_01"));
        }
    }
}
