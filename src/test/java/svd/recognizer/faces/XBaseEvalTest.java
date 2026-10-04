package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;

/**
 * XBaseEval: дата галереи своего, чтение сессий GT, встроенная когорта FEI.
 *
 * @author ssv
 */
public class XBaseEvalTest {

    @Test
    public void galleryDate() {
        Map<Integer, List<String>> bd = new TreeMap<>();
        bd.put(1, List.of("a"));
        bd.put(2, List.of("b", "c", "d"));
        bd.put(3, List.of("e", "f", "g"));
        // Наибольшее число снимков, при равенстве — более ранняя дата.
        assertEquals(2, XBaseEval.galleryDate(bd));
        bd.put(0, List.of("x", "y", "z", "w"));
        // Снимки без даты не учитываются.
        assertEquals(2, XBaseEval.galleryDate(bd));
        bd.put(4, List.of("h", "i", "j", "k"));
        assertEquals(4, XBaseEval.galleryDate(bd));
    }

    @Test
    public void sessions() throws Exception {
        Path f = Files.createTempFile("gt_sessions", ".tsv");
        Files.writeString(f, "# шапка\nperson\timage\texif_datetime\tsession\ns01\t01\t1999:07:15 13:46:02\t1\ns37\t15\tNONE\t-\n");
        Map<String, Integer> s = XBaseEval.sessions(f);
        assertEquals(Integer.valueOf(1), s.get("s01/01"));
        assertEquals(Integer.valueOf(0), s.get("s37/15"));
    }

    @Test
    public void cohortPersons() {
        Map<String, Map<Integer, String>> keys = new TreeMap<>();
        for (int p = 0; p < 30; p++) {
            Map<Integer, String> m = new TreeMap<>();
            for (int n : FeiProtocol.NUMBERS) m.put(n, p + "-" + n);
            keys.put(String.format("%03d", p), m);
        }
        List<String> c = XBaseEval.cohortPersons(keys, 42);
        assertEquals(XBaseEval.COHORT, c.size());
        assertEquals(c, XBaseEval.cohortPersons(keys, 42));
        assertEquals(c.stream().sorted().toList(), c);
    }
}
