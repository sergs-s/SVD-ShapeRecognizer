package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public class XBaseChinTest {

    @Test
    public void mcnemarExact() {
        assertEquals(1, XBaseChin.mcnemar(0, 0), 0);
        assertEquals(0.0625, XBaseChin.mcnemar(0, 5), 1e-12);
        assertEquals(22.0 / 1024, XBaseChin.mcnemar(1, 9), 1e-12);
        assertEquals(22.0 / 1024, XBaseChin.mcnemar(9, 1), 1e-12);
        assertEquals(1, XBaseChin.mcnemar(3, 3), 0);
    }

    @Test
    public void ratesWeighted() {
        List<XBaseChin.PO> l = List.of(new XBaseChin.PO("0|a", "p1", true, true, true), new XBaseChin.PO("0|b", "p1", true, false, true),
                new XBaseChin.PO("0|c", "p2", true, true, false), new XBaseChin.PO("0|d", "c1", false, true, false), new XBaseChin.PO("0|e", "c1", false, false, false),
                new XBaseChin.PO("0|f", "c2", false, false, false));
        Map<String, Integer> idx = Map.of("p1", 0, "p2", 1, "c1", 2, "c2", 3);
        double[] r = XBaseChin.rates(l, idx, null);
        assertEquals(200.0 / 3, r[0], 1e-12);
        assertEquals(100.0 / 3, r[1], 1e-12);
        assertEquals(50, r[2], 1e-12);
        double[] rw = XBaseChin.rates(l, idx, new int[] {2, 0, 0, 1});
        assertEquals(50, rw[0], 1e-12);
        assertEquals(0, rw[1], 1e-12);
        assertEquals(0, rw[2], 1e-12);
    }
}
