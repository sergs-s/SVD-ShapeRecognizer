package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import svd.recognizer.faces.XBaseExcl.Data;
import svd.recognizer.faces.XBaseExcl.G;
import svd.recognizer.faces.XBaseExcl.Model;

/**
 * Исключение даты пробы (PLAN.md п. 5.5, (а2)): схема моделей по частям и раундам — у пробы в галерее (она же обучение MLDA)
 * нет снимков её человека с её даты (Г-все) и с её и более поздних дат (Г-прошлые); каждая проба — ровно в одной модели;
 * у людей вне проб модели — все даты; разбивки. Данные синтетические.
 *
 * @author ssv
 */
public class XBaseExclTest {

    /** 23 человека с 2–6 датами, по 1–3 снимка на дату; ключ «p/дата/снимок». */
    static Data data() {
        List<String> own = new ArrayList<>();
        Map<String, List<List<String>>> sess = new LinkedHashMap<>();
        Map<String, List<LocalDate>> when = new LinkedHashMap<>();
        for (int i = 0; i < 23; i++) {
            String p = String.format("p%02d", i);
            own.add(p);
            int n = 2 + i % 5;
            List<List<String>> ds = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                List<String> ks = new ArrayList<>();
                for (int m = 0; m <= (i + j) % 3; m++) ks.add(p + "/" + j + "/" + m);
                ds.add(ks);
                ws.add(LocalDate.of(1993, 1, 1).plusDays(40L * j * j));
            }
            sess.put(p, ds);
            when.put(p, ws);
        }
        return new Data(own, sess, when);
    }

    static int dateOf(String key) {
        return Integer.parseInt(key.split("/")[1]);
    }

    static String person(String key) {
        return key.split("/")[0];
    }

    void checkScheme(G v) {
        Data d = data();
        List<List<String>> parts = XBaseOther.cvParts(d.own(), 42, XBaseEval.CV_S);
        List<Model> models = XBaseExcl.models(d, parts, v);
        Map<String, Integer> seen = new HashMap<>();
        for (Model m : models) {
            Set<String> part = new HashSet<>(parts.get(m.q()));
            assertEquals(d.own(), new ArrayList<>(m.gal().keySet()));
            for (String p : d.own()) assertFalse("галерея " + p + " пуста", m.gal().get(p).isEmpty());
            for (Map.Entry<String, List<String>> e : m.pro().entrySet()) {
                String p = e.getKey();
                assertTrue(part.contains(p));
                for (String k : e.getValue()) {
                    assertEquals(p, person(k));
                    assertEquals(m.k() - 1, dateOf(k));
                    seen.merge(k, 1, Integer::sum);
                    for (String gk : m.gal().get(p)) {
                        assertEquals(p, person(gk));
                        if (v == G.ALL) assertTrue("в галерее дата пробы: " + gk, dateOf(gk) != dateOf(k));
                        else assertTrue("в галерее дата не раньше пробы: " + gk, dateOf(gk) < dateOf(k));
                    }
                }
            }
            // Люди вне проб модели — все даты.
            for (String p : d.own()) {
                if (m.pro().containsKey(p)) continue;
                List<String> all = new ArrayList<>();
                d.sess().get(p).forEach(all::addAll);
                assertEquals(XBaseExcl.sorted(all), m.gal().get(p));
            }
        }
        // Каждая проба варианта — ровно в одной модели.
        Map<String, List<String>> pk = XBaseExcl.probeKeys(d, v);
        int total = 0;
        for (List<String> ks : pk.values()) {
            for (String k : ks) assertEquals("проба " + k, Integer.valueOf(1), seen.get(k));
            total += ks.size();
        }
        assertEquals(total, seen.size());
    }

    @Test
    public void schemeAll() {
        checkScheme(G.ALL);
    }

    @Test
    public void schemePast() {
        checkScheme(G.PAST);
        // Первая дата пробой не бывает.
        for (List<String> ks : XBaseExcl.probeKeys(data(), G.PAST).values()) for (String k : ks) assertTrue(dateOf(k) >= 1);
    }

    @Test
    public void galleryDates() {
        assertEquals(List.of(0, 2, 3), XBaseExcl.galleryDates(G.ALL, 1, 4));
        assertEquals(List.of(0, 1), XBaseExcl.galleryDates(G.PAST, 2, 4));
        assertEquals(List.of(0), XBaseExcl.galleryDates(G.FIRST, 3, 4));
        assertEquals(List.of(), XBaseExcl.galleryDates(G.PAST, 0, 4));
    }

    @Test
    public void bins() {
        Data d = data();
        // p04: 6 дат, снимков на дату (4 + j) % 3 + 1 = 2, 3, 1, 2, 3, 1; даты — 1993-01-01 + 40 j² дн.
        assertArrayEquals(new int[] {3, 3, 1}, XBaseExcl.bins(d, "p04", 0, G.ALL)); // 5 дат, 10 снимков, ближайшая — 40 дн.
        assertArrayEquals(new int[] {0, 0, 1}, XBaseExcl.bins(d, "p04", 3, G.FIRST)); // 360 дн. — 31–365
        assertArrayEquals(new int[] {2, 2, 1}, XBaseExcl.bins(d, "p04", 3, G.PAST)); // 2+3+1 = 6; 360 − 160 = 200 дн.
        assertArrayEquals(new int[] {3, 3, 1}, XBaseExcl.bins(d, "p04", 5, G.PAST)); // 5 дат, 11 снимков; 1000 − 640 = 360 дн.
        assertArrayEquals(new int[] {0, 0, 2}, XBaseExcl.bins(d, "p04", 4, G.FIRST)); // 640 дн. — > 365
        assertArrayEquals(new int[] {0, 0, 1}, XBaseExcl.bins(d, "p00", 1, G.ALL)); // p00: 2 даты, 1 снимок на дату 0; 40 дн.
    }
}
