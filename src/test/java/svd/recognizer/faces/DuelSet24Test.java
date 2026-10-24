package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Набор 24.10 (DuelSet.run24): номера файлов формата коллеги и описание папки обучения.
 *
 * @author ssv
 */
public class DuelSet24Test {

    @Test
    public void numbering() {
        assertEquals("обучение/001.jpg", DuelSet.file24("P01_1", 8));
        assertEquals("обучение/008.jpg", DuelSet.file24("P01_8", 8));
        assertEquals("обучение/073.jpg", DuelSet.file24("P10_1", 8));
        assertEquals("обучение/080.jpg", DuelSet.file24("P10_8", 8));
        assertEquals("обучение/081.jpg", DuelSet.file24("G01", 8));
        assertEquals("обучение/090.jpg", DuelSet.file24("G10", 8));
        assertEquals("пробы/001.jpg", DuelSet.file24("Q001", 8));
        assertEquals("пробы/070.jpg", DuelSet.file24("Q070", 8));
        // Все 90 файлов обучения различны и подряд.
        Set<String> seen = new HashSet<>();
        for (int i = 1; i <= 10; i++) for (int j = 1; j <= 8; j++) seen.add(DuelSet.file24(String.format("P%02d_%d", i, j), 8));
        for (int g = 1; g <= 10; g++) seen.add(DuelSet.file24(String.format("G%02d", g), 8));
        assertEquals(90, seen.size());
        for (int n = 1; n <= 90; n++) assertEquals(true, seen.contains(String.format("обучение/%03d.jpg", n)));
    }

    @Test
    public void trainText() {
        String t = DuelSet.trainText(8).substring(1).replace("\r", "");
        String[] l = t.split("\n");
        assertEquals(11, l.length);
        assertEquals("1 – 001–008", l[0]);
        assertEquals("10 – 073–080", l[9]);
        assertEquals("гости – 081–090", l[10]);
    }
}
