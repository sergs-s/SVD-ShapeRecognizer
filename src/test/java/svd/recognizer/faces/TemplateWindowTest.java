package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import java.util.List;
import org.junit.Test;

/**
 * Окно шаблона (задание №4 от 15.10.2026): вытеснение старой сессии, без окна, чужая сессия.
 *
 * @author ssv
 */
public class TemplateWindowTest {

    private static TemplateWindow.Session s(String date, String owner) {
        return new TemplateWindow.Session(date, owner, List.of(owner + "_" + date + "_fa", owner + "_" + date + "_fb"));
    }

    @Test
    public void windowOfTwoEvictsOldestSession() {
        TemplateWindow w = new TemplateWindow(2);
        assertNull(w.add(s("930831", "00001")));
        assertNull(w.add(s("940128", "00001")));
        TemplateWindow.Session out = w.add(s("940519", "00001"));
        assertEquals("930831", out.date());
        assertEquals(2, w.size());
        assertEquals(List.of("00001_940128_fa", "00001_940128_fb", "00001_940519_fa", "00001_940519_fb"), w.keys());
        assertEquals("940128", w.oldestDate());
    }

    @Test
    public void windowOfThreeKeepsThreeAndUnboundedKeepsAll() {
        TemplateWindow w3 = new TemplateWindow(3);
        TemplateWindow wInf = new TemplateWindow(TemplateWindow.UNBOUNDED);
        for (String d : new String[] {"930831", "940128", "940519"}) {
            assertNull(w3.add(s(d, "00001")));
            assertNull(wInf.add(s(d, "00001")));
        }
        assertEquals(3, w3.size());
        assertEquals("930831", w3.add(s("950101", "00001")).date());
        assertNull(wInf.add(s("950101", "00001")));
        assertEquals(4, wInf.size());
    }

    @Test
    public void foreignSessionEvictsLikeOwnAndIsCounted() {
        TemplateWindow w = new TemplateWindow(2);
        w.add(s("930831", "00001"));
        w.add(s("940128", "00001"));
        TemplateWindow c = w.copy();
        assertEquals("930831", w.add(s("940519", "00777")).date());
        assertEquals(1, w.foreign("00001"));
        assertEquals(0, c.foreign("00001"));
        assertEquals(2, c.size());
    }
}
