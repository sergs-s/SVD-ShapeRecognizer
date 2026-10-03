package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Test;

/**
 * LBP (п. 4.2, L0, L1): таблица равномерных кодов, коды на ручных картинках, гистограммы блоков, χ² и наименьшее по снимкам
 * человека; совпадение ошибок FeiCompare.
 *
 * @author ssv
 */
public class LbpFeatureTest {

    @Test
    public void uniformLabels() {
        int uniform = 0;
        for (int c = 0; c < 256; c++) if (LbpFeature.uniform(c)) uniform++;
        assertEquals(58, uniform);
        assertEquals(0, LbpFeature.LABEL[0]);
        assertEquals(57, LbpFeature.LABEL[255]);
        assertEquals(58, LbpFeature.LABEL[0b01010101]);
        assertTrue(LbpFeature.uniform(0b11000111));
        assertEquals(LbpFeature.GRID * LbpFeature.GRID * LbpFeature.BINS, LbpFeature.LENGTH);
        assertEquals(2891, LbpFeature.LENGTH);
    }

    @Test
    public void codesOnHandImages() {
        int w = 12;
        int h = 10;
        double[] flat = new double[w * h];
        java.util.Arrays.fill(flat, 7);
        for (int c : LbpFeature.codes(flat, w, h)) assertEquals(255, c);
        // Ступенька: столбцы < 5 — 0, ≥ 5 — 100; центр (5, 5) — первый светлый столбец: светлее или равны соседи 0, 1, 2, 6, 7.
        double[] step = new double[w * h];
        for (int y = 0; y < h; y++) for (int x = 5; x < w; x++) step[y * w + x] = 100;
        int[] codes = LbpFeature.codes(step, w, h);
        assertEquals(1 + 2 + 4 + 64 + 128, codes[(5 - 2) * (w - 4) + (5 - 2)]);
        // Слева от ступеньки (4, 5): все соседи ≥ 0.
        assertEquals(255, codes[(5 - 2) * (w - 4) + (4 - 2)]);
    }

    @Test
    public void blockHistogramsSumToOne() {
        Random rnd = new Random(3);
        double[] img = new double[92 * 112];
        for (int i = 0; i < img.length; i++) img[i] = rnd.nextInt(256);
        double[] v = LbpFeature.histogram(img, 92, 112);
        assertEquals(2891, v.length);
        for (int b = 0; b < 49; b++) {
            double s = 0;
            for (int k = 0; k < 59; k++) s += v[b * 59 + k];
            assertEquals(1, s, 1e-12);
        }
        double[] r = LbpFeature.sqrt(v);
        assertEquals(Math.sqrt(v[100]), r[100], 0);
    }

    @Test
    public void chiSquareNearestSnapshot() {
        double[] a = {0.5, 0.5, 0};
        double[] b = {0.25, 0.75, 0};
        assertEquals(0.0625 / 0.75 + 0.0625 / 1.25, ChiSquareScorer.chi2(a, b), 1e-15);
        assertEquals(0, ChiSquareScorer.chi2(a, a), 0);
        // Человек 0 — два снимка (ближе второй), человек 1 — один; третий класс — посторонний, в оценки не входит.
        GalleryScorer.ScoreModel m = new ChiSquareScorer().fit(List.of(List.of(new double[] {1, 0, 0}, b), List.of(new double[] {0, 0, 1}),
                List.of(a)), 2);
        double[] s = m.scores(a);
        assertEquals(2, s.length);
        assertEquals(ChiSquareScorer.chi2(a, b), s[0], 0);
        assertEquals(2, s[1], 1e-15);
    }

    @Test
    public void agreementTable() {
        Map<String, boolean[]> x = new LinkedHashMap<>();
        Map<String, boolean[]> y = new LinkedHashMap<>();
        x.put("1", new boolean[] {true, true, true});
        y.put("1", new boolean[] {true, false, true});
        x.put("2", new boolean[] {false, true, false});
        y.put("2", new boolean[] {true, false, false});
        int[][] t = FeiCompare.agreement(x, y);
        assertArrayEquals(new int[] {1, 0, 1, 0}, t[0]);
        assertArrayEquals(new int[] {0, 2, 0, 0}, t[1]);
        assertArrayEquals(new int[] {1, 0, 0, 1}, t[2]);
    }

    @Test
    public void masks() {
        LbpFeature.Mask in = LbpFeature.Mask.INNER;
        LbpFeature.Mask out = LbpFeature.Mask.RING;
        assertEquals(25, in.blocks());
        assertEquals(24, out.blocks());
        for (int by = 0; by < 7; by++) {
            for (int bx = 0; bx < 7; bx++) {
                assertTrue(in.has(by, bx) != out.has(by, bx));
                assertEquals(by == 0 || by == 6 || bx == 0 || bx == 6, out.has(by, bx));
            }
        }
        // Вектор: значение = номер блока; выборка — блоки маски по строкам, гистограммы целиком.
        double[] v = new double[LbpFeature.LENGTH];
        for (int i = 0; i < v.length; i++) v[i] = i / LbpFeature.BINS + (i % LbpFeature.BINS) / 1000.0;
        double[] vi = LbpFeature.select(v, in);
        double[] vo = LbpFeature.select(v, out);
        assertEquals(1475, vi.length);
        assertEquals(1416, vo.length);
        assertEquals(8, vi[0], 0);
        assertEquals(9 + 0.001, vi[LbpFeature.BINS + 1], 0);
        assertEquals(40, vi[vi.length - 1] - 0.058, 1e-12);
        assertEquals(0, vo[0], 0);
        assertEquals(7, vo[7 * LbpFeature.BINS], 0);
        assertEquals(13, vo[8 * LbpFeature.BINS], 0);
        assertEquals(48 + 0.058, vo[vo.length - 1], 1e-12);
    }

    @Test
    public void lbpModes() {
        assertTrue(FeiMethods.lbpModes(null).isEmpty());
        assertTrue(FeiMethods.lbpModes("false").isEmpty());
        assertEquals(java.util.EnumSet.of(FeiMethods.Lbp.ON), FeiMethods.lbpModes("TRUE"));
        assertEquals(java.util.EnumSet.of(FeiMethods.Lbp.MASK, FeiMethods.Lbp.ONLY), FeiMethods.lbpModes("mask, only"));
        try {
            FeiMethods.lbpModes("yes");
            org.junit.Assert.fail();
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("yes"));
        }
    }
}
