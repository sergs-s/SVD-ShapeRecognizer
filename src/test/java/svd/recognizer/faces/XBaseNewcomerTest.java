package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Новички (задание №7 от 15.10.2026): правило (н3) – ширина оценки на шаге – наименьшее число сессий в галерее (не больше M),
 * берутся последние сессии; синтетическая сессия (н2) выбывает первой.
 *
 * @author ssv
 */
public class XBaseNewcomerTest {

    private static TemplateWindow.Session s(String date, String p) {
        return new TemplateWindow.Session(date, p, List.of(p + "_" + date + "_fa", p + "_" + date + "_fb"));
    }

    @Test
    public void widthIsSmallestSessionCountCappedByM() {
        TemplateWindow nw = new TemplateWindow(XBaseNewcomer.M);
        nw.add(s("940128", "00010"));
        TemplateWindow old = new TemplateWindow(XBaseNewcomer.M);
        old.add(s("930831", "00001"));
        old.add(s("940128", "00001"));
        Map<String, TemplateWindow> tpl = Map.of("00010", nw, "00001", old);
        // Шаг 1: у новичка 1 сессия – ширина 1, у старожила берётся только последняя.
        assertEquals(1, XBaseNewcomer.width(tpl, List.of("00001", "00010")));
        assertEquals(List.of("00001_940128_fa", "00001_940128_fb"), XBaseNewcomer.keys(old, 1));
        // Шаг 2: после обновления у новичка 2, у старожила 3 – ширина 2.
        nw.add(s("940519", "00010"));
        old.add(s("940519", "00001"));
        assertEquals(2, XBaseNewcomer.width(tpl, List.of("00001", "00010")));
        assertEquals(List.of("00001_940128_fa", "00001_940128_fb", "00001_940519_fa", "00001_940519_fb"), XBaseNewcomer.keys(old, 2));
        // Больше M не бывает; ширина 0 – все сессии.
        TemplateWindow a = new TemplateWindow(XBaseNewcomer.M);
        TemplateWindow b = new TemplateWindow(XBaseNewcomer.M);
        for (String d : new String[] {"930831", "940128", "940519", "950101"}) {
            a.add(s(d, "00002"));
            b.add(s(d, "00003"));
        }
        assertEquals(XBaseNewcomer.M, XBaseNewcomer.width(Map.of("00002", a, "00003", b), List.of("00002", "00003")));
        assertEquals(6, XBaseNewcomer.keys(a, 0).size());
    }

    @Test
    public void synthSessionLeavesFirst() {
        TemplateWindow w = new TemplateWindow(XBaseNewcomer.M);
        w.add(new TemplateWindow.Session("940128", "00010", List.of("00010_940128_fa#n0_0", "00010_940128_fb#n0_1")));
        w.add(s("940128", "00010"));
        w.add(s("940519", "00010"));
        TemplateWindow.Session out = w.add(s("950101", "00010"));
        assertEquals("00010_940128_fa#n0_0", out.keys().get(0));
        assertEquals(3, w.size());
    }
}
