package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * CaltechLabeler: файл разметки (запись, чтение, резервная копия), номер нового человека, люди, порядок показа.
 *
 * @author ssv
 */
public class CaltechLabelerTest {

    private static CaltechLabeler.Label ok(String p) {
        return new CaltechLabeler.Label(p, CaltechLabeler.OK, "2026-10-04T12:00:00");
    }

    private static CaltechLabeler.Det det(double... f) {
        double[] v = f.clone();
        CaltechLabeler.normalize(v);
        return new CaltechLabeler.Det(new double[] {0, 0, 10, 10}, v);
    }

    @Test
    public void csvRoundTripAndBackup() throws Exception {
        Path dir = Files.createTempDirectory("caltech");
        Path csv = dir.resolve(CaltechLabeler.CSV_NAME);
        List<String> files = List.of("a.jpg", "b.jpg", "c.jpg");
        Map<String, CaltechLabeler.Label> labels = new LinkedHashMap<>();
        labels.put("c.jpg", ok("p01"));
        labels.put("a.jpg", new CaltechLabeler.Label(CaltechLabeler.NONE, CaltechLabeler.EXCLUDED, "2026-10-04T12:00:01"));
        CaltechLabeler.writeCsv(csv, files, labels);
        assertTrue(!Files.exists(dir.resolve(CaltechLabeler.CSV_NAME + ".bak")));
        Map<String, CaltechLabeler.Label> back = CaltechLabeler.readCsv(csv);
        assertEquals(labels, back);
        // Порядок строк — порядок файлов.
        assertEquals(List.of("a.jpg", "c.jpg"), List.copyOf(back.keySet()));
        labels.put("b.jpg", ok("p02"));
        CaltechLabeler.writeCsv(csv, files, labels);
        assertEquals(2, CaltechLabeler.readCsv(dir.resolve(CaltechLabeler.CSV_NAME + ".bak")).size());
        assertEquals(3, CaltechLabeler.readCsv(csv).size());
    }

    @Test
    public void nextPersonAndPeople() {
        Map<String, CaltechLabeler.Label> labels = new LinkedHashMap<>();
        assertEquals("p01", CaltechLabeler.nextPerson(labels.values()));
        labels.put("b.jpg", ok("p03"));
        labels.put("a.jpg", ok("p01"));
        labels.put("c.jpg", new CaltechLabeler.Label(CaltechLabeler.NONE, CaltechLabeler.DOUBTFUL, "t"));
        assertEquals("p04", CaltechLabeler.nextPerson(labels.values()));
        Map<String, List<String>> ppl = CaltechLabeler.people(List.of("a.jpg", "b.jpg", "c.jpg"), labels);
        assertEquals(List.of("p01", "p03"), List.copyOf(ppl.keySet()));
        assertEquals(List.of("a.jpg"), ppl.get("p01"));
    }

    @Test
    public void nextOrder() {
        List<String> files = List.of("1.jpg", "2.jpg", "3.jpg", "4.jpg");
        Map<String, CaltechLabeler.Det> dets = new HashMap<>();
        dets.put("1.jpg", det(1, 0));
        dets.put("2.jpg", null);
        dets.put("3.jpg", det(0, 1));
        dets.put("4.jpg", det(1, 0.1));
        Map<String, CaltechLabeler.Label> labels = new LinkedHashMap<>();
        // Людей нет — по порядку файлов.
        assertEquals("1.jpg", CaltechLabeler.next(files, labels, dets));
        labels.put("1.jpg", ok("p01"));
        // Ближайший к p01 — 4.jpg; снимок без лица — последним.
        assertEquals("4.jpg", CaltechLabeler.next(files, labels, dets));
        labels.put("4.jpg", ok("p01"));
        assertEquals("3.jpg", CaltechLabeler.next(files, labels, dets));
        labels.put("3.jpg", ok("p02"));
        assertEquals("2.jpg", CaltechLabeler.next(files, labels, dets));
        labels.put("2.jpg", new CaltechLabeler.Label(CaltechLabeler.NONE, CaltechLabeler.EXCLUDED, "t"));
        assertNull(CaltechLabeler.next(files, labels, dets));
        Object[] n = CaltechLabeler.nearest(det(0.1, 1), CaltechLabeler.centroids(CaltechLabeler.people(files, labels), dets));
        assertEquals("p02", n[0]);
    }
}
