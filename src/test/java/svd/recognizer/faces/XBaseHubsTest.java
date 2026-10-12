package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import svd.recognizer.faces.XBaseExcl.Data;
import svd.recognizer.faces.XBaseExcl.G;
import svd.recognizer.faces.XBaseExcl.Model;

/**
 * Симметричные галереи и защита от хабов (PLAN.md п. 5.5, (а2в)): (с3) — не больше 3 последних сессий; в схеме (а2) с
 * шаблонами по сессиям у пробы нет её даты; часть 1 — первые s дат и пробы после; мера хабовости. Данные синтетические
 * (XBaseExclTest.data()).
 *
 * @author ssv
 */
public class XBaseHubsTest {

    static int dateOf(String key) {
        return Integer.parseInt(key.split("/")[1]);
    }

    @Test
    public void lastSessions() {
        assertEquals(List.of(2, 3, 5), XBaseHubs.last(List.of(0, 1, 2, 3, 5), 3));
        assertEquals(List.of(0, 1), XBaseHubs.last(List.of(0, 1), 3));
        assertEquals(List.of(0, 1, 2, 3), XBaseHubs.last(List.of(0, 1, 2, 3), 0));
    }

    @Test
    public void modelSessions() {
        Data d = XBaseExclTest.data();
        List<List<String>> parts = XBaseOther.cvParts(d.own(), 42, XBaseEval.CV_S);
        for (int cap : new int[] {0, XBaseHubs.M}) {
            for (Model md : XBaseExcl.models(d, parts, G.ALL)) {
                Map<String, List<List<String>>> s = XBaseHubs.modelSessions(d, md, cap);
                for (String p : d.own()) {
                    List<List<String>> ss = s.get(p);
                    int n = d.sess().get(p).size();
                    assertTrue(cap == 0 || ss.size() <= cap);
                    List<Integer> dates = new ArrayList<>();
                    for (List<String> one : ss) {
                        int dt = dateOf(one.get(0));
                        for (String k : one) assertEquals(dt, dateOf(k));
                        dates.add(dt);
                    }
                    if (md.pro().containsKey(p)) {
                        assertTrue("дата пробы в галерее", !dates.contains(md.k() - 1));
                        assertEquals(cap == 0 ? n - 1 : Math.min(n - 1, cap), dates.size());
                    } else {
                        assertEquals(cap == 0 ? n : Math.min(n, cap), dates.size());
                    }
                    // Последние по времени: пропущенные даты (кроме даты пробы) — только раньше взятых.
                    for (int j = 0; j < n; j++) {
                        if (dates.contains(j) || (md.pro().containsKey(p) && j == md.k() - 1)) continue;
                        assertTrue(j < dates.get(0));
                    }
                }
            }
        }
    }

    @Test
    public void firstAndProbes() {
        Data d = XBaseExclTest.data();
        List<String> own = d.own().stream().filter(p -> d.sess().get(p).size() >= 4).toList();
        Map<String, List<List<String>>> fs = XBaseHubs.firstSessions(d, own, 2);
        Map<String, List<String>> pr = XBaseHubs.probesFrom(d, own, 3);
        for (String p : own) {
            assertEquals(2, fs.get(p).size());
            for (String k : XBaseHubs.keys(fs).get(p)) assertTrue(dateOf(k) < 2);
            assertTrue(!pr.get(p).isEmpty());
            for (String k : pr.get(p)) assertTrue(dateOf(k) >= 3);
        }
    }

    @Test
    public void hubs() {
        XBaseHubs.Hubs h = XBaseHubs.hubs(List.of("a", "b", "a", "c", "a", "d", "e", "f", "b", "g"));
        assertEquals(10, h.wrong());
        assertEquals(7, h.hubs());
        assertEquals(0.3, h.top1(), 1e-12);
        assertEquals(0.8, h.top5(), 1e-12); // a 3, b 2, c, d, e — по 1
        assertEquals("a", h.top().get(0).getKey());
        assertEquals("b", h.top().get(1).getKey());
        assertEquals(0, XBaseHubs.hubs(List.of()).wrong());
    }
}
