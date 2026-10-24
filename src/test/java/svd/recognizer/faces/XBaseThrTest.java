package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Ревизия 1.14 (XBaseThr): строки роли od при сверке с прежними отбрасываются, а прочие – нет; (Т2) – объединение (Т0) и
 * (Т1); пробы (Т1н) не пересекаются с оцениваемыми пробами сокращённого набора и вместе дают все пробы.
 *
 * @author ssv
 */
public class XBaseThrTest {

    static String line(String role, String person, int image) {
        return "5-lbp-mlda\t0\t0\t" + role + "\t" + person + "\t" + image + "\tx";
    }

    @Test
    public void sortedDropsOnlyT1() {
        String lines = line("own", "a", 1) + "\n" + line("od0", "b", 2) + "\n" + line("cv0", "b", 3) + "\n" + line("ctrl", "c", 4) + "\n";
        List<String> kept = XBaseThr.sorted(lines, true);
        assertEquals(3, kept.size());
        for (String l : kept) assertTrue(!l.split("\t")[3].startsWith(XBaseThr.T1_ROLE));
        assertEquals(4, XBaseThr.sorted(lines, false).size());
        // Роль od не начинается с cv: разбор «чужих» (Т0) её не захватывает.
        assertTrue(!XBaseThr.T1_ROLE.startsWith(FeiMethods.CV_ROLE));
    }

    @Test
    public void unionIsT0PlusT1() {
        FeiThreshold.Imp a = FeiThreshold.Imp.of(new ArrayList<>(List.of(new double[] {0, 0.5}, new double[] {1, 0.9})), 1);
        FeiThreshold.Imp b = FeiThreshold.Imp.of(new ArrayList<>(List.of(new double[] {2, 0.7})), 0);
        FeiThreshold.Imp u = XBaseCalib.union(a, b);
        assertEquals(3, u.r().length);
        assertEquals(0.5, u.r()[0], 0);
        assertEquals(0.7, u.r()[1], 0);
        assertEquals(0.9, u.r()[2], 0);
        assertEquals(1, u.frontal());
    }

    @Test
    public void droppedDisjoint() {
        Map<String, List<String>> pro = new LinkedHashMap<>();
        pro.put("p1", List.of("k1", "k2", "k3", "k4"));
        pro.put("p2", List.of("k5", "k6"));
        Map<String, List<String>> red = new LinkedHashMap<>();
        red.put("p1", List.of("k1", "k4"));
        red.put("p2", List.of("k5", "k6"));
        XBaseThr.Unit u = new XBaseThr.Unit(null, null, pro, null, "", red, true);
        Map<String, List<String>> d = XBaseThr.dropped(u);
        assertEquals(List.of("k2", "k3"), d.get("p1"));
        assertTrue(!d.containsKey("p2"));
        for (String p : pro.keySet()) {
            Set<String> all = new HashSet<>(red.get(p));
            for (String k : d.getOrDefault(p, List.of())) assertTrue("пересечение " + k, all.add(k));
            assertEquals(new HashSet<>(pro.get(p)), all);
        }
    }
}
