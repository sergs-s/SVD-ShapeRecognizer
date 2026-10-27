package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Test;

/**
 * Сочетание и совмещение d₁ и d₁/d₂ (XBaseAbl3): а) β = ∞ – строки без изменений, β = 0 – побайтно XBaseAbl.abs; б) d₁′ монотонно по d₁
 * при постоянном d₂ и по d₁/d₂ при постоянном d₁; в) d₁ = 0 и d₂ = +∞; г) (Т+М) и (М) – разбор с r0 = −1; д) правила выбора и файл.
 *
 * @author ssv
 */
public class XBaseAbl3Test {

    static String line(String d1, String d2) {
        return "L1\t0\t0\town\ta\t1\t0\ta\t" + d1 + "\tb\t" + d2 + "\t" + d1 + "\n";
    }

    static double d1p(String l) {
        return Double.parseDouble(l.split("\t", -1)[8]);
    }

    @Test
    public void betaForms() {
        String l = line("0.25", "0.5") + line("0.125", "Infinity") + line("0.0", "0.3");
        assertSame(l, XBaseAbl3.beta(l, Double.POSITIVE_INFINITY));
        assertEquals(XBaseAbl.abs(l), XBaseAbl3.beta(l, 0));
        String[] b1 = XBaseAbl3.beta(l, 1).split("\n");
        assertEquals(0.25 * 0.25 / 0.5, d1p(b1[0]), 1e-15);
        assertEquals(Double.toString(XBaseAbl.ABS_C), b1[0].split("\t", -1)[10]);
        assertEquals(line("0.125", "Infinity").trim(), b1[1]);
        assertEquals(0, d1p(b1[2]), 0);
        assertEquals(Math.pow(0.25, 1.5) * Math.pow(0.5, -0.5), d1p(XBaseAbl3.beta(line("0.25", "0.5"), 0.5)), 1e-15);
        for (double b : new double[] {0.5, 1, 2}) {
            // по d₁ при постоянном d₂
            assertTrue(d1p(XBaseAbl3.beta(line("0.2", "0.6"), b)) < d1p(XBaseAbl3.beta(line("0.3", "0.6"), b)));
            // по d₁/d₂ при постоянном d₁
            assertTrue(d1p(XBaseAbl3.beta(line("0.2", "0.8"), b)) < d1p(XBaseAbl3.beta(line("0.2", "0.4"), b)));
        }
    }

    @Test
    public void mirrorAndChoice() {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.r0 = 0.12;
        assertSame(g, XBaseAbl3.mirrorGt(g, false));
        assertEquals(-1, XBaseAbl3.mirrorGt(g, true).r0, 0);
        assertEquals(0.12, g.r0, 0);
        assertEquals(List.of(0, 0, 3, 3), java.util.Arrays.stream(XBaseAbl3.PART1_ROW).boxed().toList());
        assertEquals(0, XBaseAbl3.chooseBeta(new double[] {10.5, 9.6, 12, 12, 12}));
        assertEquals(2, XBaseAbl3.chooseBeta(new double[] {12, 10.8, 10.5, 11, 12}));
        assertEquals(0, XBaseAbl3.choose1(new double[] {10.9, 10.2, 9.95, 12}));
        assertEquals(1, XBaseAbl3.choose1(new double[] {12, 10.9, 10, 9.95}));
        assertEquals(2, XBaseAbl3.choose1(new double[] {12, 12, 10.5, 9.6}));
        assertEquals("β = ∞ (d₁/d₂, база)", XBaseAbl3.bname(0));
        assertEquals("β = 0 (d₁)", XBaseAbl3.bname(1));
        assertEquals("β = 0,5", XBaseAbl3.bname(2));
        assertEquals("β = 2", XBaseAbl3.bname(4));
    }

    @Test
    public void selectFile() throws Exception {
        Path f = Files.createTempFile("comb_select", ".tsv");
        try {
            Files.writeString(f, "# x\n2\tGT\tL1-xs-F\t0\tβ = ∞ (d₁/d₂, база)\n1\tGT\tL1-xs-F\t0\tбаза (П2)\n1\tFERET-1\tL1-xs-F\t0\tбаза (П2)\n");
            assertFalse(XBaseAbl3.part1Own(f));
            Files.writeString(f, "1\tGT\tL1-xs-F\t2\t(Т) с Тан – Триггс\n");
            assertTrue(XBaseAbl3.part1Own(f));
            Files.writeString(f, "1\tGT\tL1-xs-F\t2\t(М) всегда X+mirror\n");
            try {
                XBaseAbl3.part1Own(f);
                fail("не из сетки");
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("не из сетки"));
            }
        } finally {
            Files.deleteIfExists(f);
        }
    }
}
