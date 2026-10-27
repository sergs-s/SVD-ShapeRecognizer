package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import svd.recognizer.OpenCvLoader;

/**
 * Повтор (4а) на новой базе (XBaseAbl2): каждая строка отличается от базы только своим элементом – а) база набора 0 побайтно =
 * XBasePoem.feature (вариант 2); б) (2) без корня: часть LBP = (часть LBP базы)², часть POEM = √3 · (часть POEM базы)² (поправка
 * Жорика №2); в) (8) без множителя: часть LBP побайтно та же, часть POEM = √3 · часть POEM базы; г) (3): у L1-tt-xs-F – вход без
 * нормировки при той же сетке 9×9, у L1-xs-F – Тан – Триггс при 7×7; д) (1) – когорта без классов пула, прочее то же; е) (4), (5),
 * (6), (7) – из строк базы: d₁ с постоянной, r0, евклид; ж) метки и файл выбора.
 *
 * @author ssv
 */
public class XBaseAbl2Test {

    static final IlluminationNorm.NormParams P = new IlluminationNorm.NormParams(2.0, 8, 0.2, 1.0, 2.0, 0.1, 10.0);

    @BeforeClass
    public static void load() {
        OpenCvLoader.load();
    }

    static Mat frame(long seed) {
        Random r = new Random(seed);
        Mat m = new Mat(112, 92, CvType.CV_8UC1);
        byte[] b = new byte[92 * 112];
        for (int y = 0; y < 112; y++) for (int x = 0; x < 92; x++) b[y * 92 + x] = (byte) Math.min(255, Math.max(0, 128 + 60 * Math.sin(x / 7.0 + y / 11.0)
                + r.nextGaussian() * 20));
        m.put(0, 0, b);
        return m;
    }

    @Test
    public void featuresDifferOnlyInOwnElement() {
        Mat m = frame(5);
        for (int f = 0; f < 2; f++) {
            int lbp = XBaseDates.PARAMS[f].length();
            double[] base = XBaseAbl2.feature(m, f, 0, P);
            assertArrayEquals(XBasePoem.feature(m, f, 2, P), base, 0);
            double[] v2 = XBaseAbl2.feature(m, f, 1, P);
            double[] v8 = XBaseAbl2.feature(m, f, 2, P);
            assertEquals(base.length, v2.length);
            assertEquals(base.length, v8.length);
            for (int i = 0; i < base.length; i++) {
                if (i < lbp) {
                    assertEquals(base[i] * base[i], v2[i], 1e-15);
                    assertEquals(base[i], v8[i], 0);
                } else {
                    assertEquals(Math.sqrt(3) * base[i] * base[i], v2[i], 1e-14);
                    assertEquals(Math.sqrt(3) * base[i], v8[i], 1e-14);
                }
            }
            double[] v3 = XBaseAbl2.feature(m, f, 3, P);
            IlluminationNorm other = f == 1 ? IlluminationNorm.NONE : IlluminationNorm.TAN_TRIGGS;
            assertEquals(other, XBaseAbl2.norm(f, 3));
            assertEquals(XBaseDates.NORMS[f], XBaseAbl2.norm(f, 0));
            double[] lh = LbpFeature.sqrt(LbpFeature.histogram(m, other, P, XBaseDates.PARAMS[f]));
            double[] img = PoemFeature.input(m, other, P);
            double[] ph = LbpFeature.sqrt(PoemFeature.histogram(img, 92, 112, XBasePoem.grid(f)));
            assertArrayEquals(PoemFeature.join(lh, ph), v3, 0);
            assertEquals(base.length, v3.length);
        }
        m.release();
    }

    @Test
    public void rowsFromBaseModel() {
        String l = "L1\t0\t0\town\ta\t1\t0\ta\t0.25\tb\t0.5\t0.25\n" + "L1\t0\t0\tctrl\tz\t2\t-1\tb\t0.125\ta\tInfinity\tNaN\n";
        XBaseLbp.One base = new XBaseLbp.One(new String[] {"евклид\n", l}, new String[] {"i", "i"}, new int[] {1, 1});
        XBaseLbp.One other = new XBaseLbp.One(new String[] {"e2\n", "c2\n"}, new String[] {"i", "i"}, new int[] {1, 1});
        XBaseLbp.One[] fits = {base, other, other, other, other};
        assertEquals(l, XBaseAbl2.rowLines(0, fits));
        assertEquals("евклид\n", XBaseAbl2.rowLines(7, fits));
        assertEquals(l, XBaseAbl2.rowLines(5, fits));
        assertEquals(l, XBaseAbl2.rowLines(6, fits));
        assertEquals("c2\n", XBaseAbl2.rowLines(1, fits));
        assertEquals("c2\n", XBaseAbl2.rowLines(8, fits));
        String[] d1 = XBaseAbl2.rowLines(4, fits).split("\n");
        assertEquals(Double.toString(XBaseAbl.ABS_C), d1[0].split("\t", -1)[10]);
        assertEquals("Infinity", d1[1].split("\t", -1)[10]);
        assertEquals(XBaseAbl2.ROWS, XBaseAbl2.ROW_FIT.length);
        assertEquals(List.of(0, 1, 2, 3, 4), java.util.Arrays.stream(XBaseAbl2.fitsFor(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8})).boxed().toList());
        assertEquals(List.of(0), java.util.Arrays.stream(XBaseAbl2.fitsFor(new int[] {0, 4, 5, 6, 7})).boxed().toList());
    }

    @Test
    public void gtCopies() {
        XBaseEval.Gt g = new XBaseEval.Gt();
        g.r0 = 0.12;
        g.cohort = new ArrayList<>(List.of(List.of("fei1"), List.of("fei2"), List.of("pool1", "pool1b"), List.of("pool2", "pool2b")));
        g.own = List.of("a", "b");
        g.rOf = Map.of("k", 0.1);
        assertSame(g, XBaseAbl2.fitGt(g, 0, 2));
        assertSame(g, XBaseAbl2.fitGt(g, 2, 2));
        XBaseEval.Gt n = XBaseAbl2.fitGt(g, 1, 2);
        assertEquals(List.of(List.of("fei1"), List.of("fei2")), n.cohort);
        assertEquals(g.own, n.own);
        assertEquals(g.r0, n.r0, 0);
        assertEquals(4, g.cohort.size());
        assertEquals(Double.POSITIVE_INFINITY, XBaseAbl2.rowGt(g, 5).r0, 0);
        assertEquals(-1, XBaseAbl2.rowGt(g, 6).r0, 0);
        assertSame(g.cohort, XBaseAbl2.rowGt(g, 5).cohort);
        assertSame(g, XBaseAbl2.rowGt(g, 4));
    }

    @Test
    public void labelsAndSelect() throws Exception {
        assertEquals(XBaseAbl2.NEED, XBaseAbl2.label(1.01));
        assertEquals(XBaseAbl2.SPARE, XBaseAbl2.label(1.0));
        assertEquals(XBaseAbl2.SPARE, XBaseAbl2.label(-1.0));
        assertEquals(XBaseAbl2.HARM, XBaseAbl2.label(-1.01));
        assertEquals("(3) без Тан – Триггс", XBaseAbl2.rname(1, 3));
        assertEquals("(3) с Тан – Триггс (справочно)", XBaseAbl2.rname(0, 3));
        Path f = Files.createTempFile("abl2_select", ".tsv");
        try {
            Files.writeString(f, "# x\nGT\tL1-xs-F\t1\t(1) без пула\t+5.0\tнужен\nGT\tL1-tt-xs-F\t5\t(5) r0 = +∞ (всегда X)\t+0.2\tлишний\n"
                    + "FERET-1\tL1-xs-F\t8\t(8) без 1/√3\t-1.5\tмешает\n");
            assertEquals(List.of(0, 5, 8), java.util.Arrays.stream(XBaseAbl2.ownRows(f)).boxed().toList());
            Files.writeString(f, "GT\tL1-xs-F\t3\t(3) без Тан – Триггс\t+0.0\tлишний\n");
            try {
                XBaseAbl2.ownRows(f);
                fail("строка не из сетки");
            } catch (IllegalStateException e) {
                assertTrue(e.getMessage().contains("не из сетки"));
            }
        } finally {
            Files.deleteIfExists(f);
        }
    }
}
