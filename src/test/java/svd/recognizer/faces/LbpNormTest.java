package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.zip.GZIPOutputStream;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Нормировка освещения перед LBP (п. 4.2): ключ faces.lbp.norm, режим faces.fei.lbp = in, путь «кадр → нормировка → LBP»
 * (none — прежний, tt — без квантования, clahe — по 8-битному CLAHE), порядок «отражение → нормировка → LBP», нормировка
 * в шапке файла, FAR контроля по людям и граница по n_eff (LbpNormReport).
 *
 * @author ssv
 */
public class LbpNormTest {

    static final IlluminationNorm.NormParams P = new IlluminationNorm.NormParams(2.0, 8, 0.2, 1.0, 2.0, 0.1, 10.0);

    @BeforeClass
    public static void load() {
        svd.recognizer.OpenCvLoader.load();
    }

    static Mat frame(long seed) {
        Random rnd = new Random(seed);
        byte[] b = new byte[92 * 112];
        for (int y = 0; y < 112; y++) {
            for (int x = 0; x < 92; x++) {
                // Плавный градиент освещения плюс шум: у Tan – Triggs нецелые значения, у CLAHE — заметное изменение.
                int v = (int) Math.round(40 + 1.2 * x + 0.5 * y + 30 * rnd.nextDouble());
                b[y * 92 + x] = (byte) Math.min(255, v);
            }
        }
        Mat m = new Mat(112, 92, CvType.CV_8UC1);
        m.put(0, 0, b);
        return m;
    }

    @Test
    public void parseNorms() {
        assertEquals(List.of(IlluminationNorm.NONE), IlluminationNorm.lbpNorms(null));
        assertEquals(List.of(IlluminationNorm.NONE), IlluminationNorm.lbpNorms(" "));
        assertEquals(List.of(IlluminationNorm.NONE, IlluminationNorm.TAN_TRIGGS, IlluminationNorm.CLAHE), IlluminationNorm.lbpNorms("none, TT,clahe"));
        assertEquals(IlluminationNorm.TAN_TRIGGS, IlluminationNorm.lbpNorm("tt"));
        for (String bad : new String[] {"gamma", "tt,tt", "none,x"}) {
            try {
                IlluminationNorm.lbpNorms(bad);
                fail(bad);
            } catch (IllegalStateException expected) {
                assertTrue(expected.getMessage().startsWith("faces.lbp.norm"));
            }
        }
        try {
            IlluminationNorm.lbpNorm("tt,clahe");
            fail();
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("одно значение"));
        }
    }

    @Test
    public void lbpModeIn() {
        assertEquals(java.util.EnumSet.of(FeiMethods.Lbp.IN, FeiMethods.Lbp.ONLY), FeiMethods.lbpModes("in, only"));
    }

    @Test
    public void noneIsUnchanged() {
        Mat f = frame(1);
        assertArrayEquals(LbpFeature.histogram(f), LbpFeature.histogram(f, IlluminationNorm.NONE, P), 0);
        f.release();
    }

    @Test
    public void tanTriggsWithoutQuantization() {
        Mat f = frame(2);
        double[] tt = IlluminationNorm.TAN_TRIGGS.vector(f, P);
        boolean fractional = false;
        for (double v : tt) {
            assertTrue(Math.abs(v) <= P.tau());
            if (v != Math.rint(v)) fractional = true;
        }
        assertTrue("Tan – Triggs даёт нецелые значения", fractional);
        double[] h = LbpFeature.histogram(f, IlluminationNorm.TAN_TRIGGS, P);
        assertArrayEquals(LbpFeature.histogram(tt, 92, 112), h, 0);
        assertFalse(java.util.Arrays.equals(LbpFeature.histogram(f), h));
        f.release();
    }

    @Test
    public void claheOnEightBits() {
        Mat f = frame(3);
        Mat c = IlluminationNorm.clahe8(f, P);
        assertEquals(CvType.CV_8UC1, c.type());
        assertArrayEquals(LbpFeature.histogram(c), LbpFeature.histogram(f, IlluminationNorm.CLAHE, P), 0);
        // Тот же CLAHE, что у 1-clahe-eps (IlluminationNorm.CLAHE.vector — яркости /255).
        double[] v = IlluminationNorm.CLAHE.vector(f, P);
        byte[] b = new byte[(int) c.total()];
        c.get(0, 0, b);
        for (int i = 0; i < b.length; i++) assertEquals((b[i] & 0xFF) / 255.0, v[i], 1e-12);
        c.release();
        f.release();
    }

    @Test
    public void mirrorThenNorm() {
        Mat f = frame(4);
        XBaseEval x = new XBaseEval();
        x.vectors(Map.of("k", f), Set.of(XBaseEval.V.L1, XBaseEval.V.L1IN), P, IlluminationNorm.TAN_TRIGGS);
        Mat m = FaceMirror.mirror(f);
        double[] s = LbpFeature.sqrt(LbpFeature.histogram(f, IlluminationNorm.TAN_TRIGGS, P));
        double[] sm = LbpFeature.sqrt(LbpFeature.histogram(m, IlluminationNorm.TAN_TRIGGS, P));
        assertArrayEquals(s, x.vec.get(XBaseEval.V.L1).get("k"), 0);
        assertArrayEquals(sm, x.mir.get(XBaseEval.V.L1).get("k"), 0);
        assertArrayEquals(LbpFeature.select(sm, LbpFeature.Mask.INNER), x.mir.get(XBaseEval.V.L1IN).get("k"), 0);
        m.release();
        f.release();
    }

    @Test
    public void normInHeader() throws Exception {
        Path dir = Files.createTempDirectory("lbpnorm");
        Path a = dir.resolve("a.tsv.gz");
        Path b = dir.resolve("b.tsv.gz");
        try (Writer w = new java.io.OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(a)), StandardCharsets.UTF_8)) {
            w.write("# Код: x\n# " + FeiMethods.lbpNormText(IlluminationNorm.TAN_TRIGGS, P) + "\n" + FeiScores.COLUMNS + "\n");
        }
        try (Writer w = new java.io.OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(b)), StandardCharsets.UTF_8)) {
            w.write("# Код: x\n" + FeiScores.COLUMNS + "\n");
        }
        assertEquals(IlluminationNorm.TAN_TRIGGS, FeiThreshold.normOf(a));
        assertEquals(IlluminationNorm.NONE, FeiThreshold.normOf(b));
        assertTrue(("# " + FeiMethods.lbpNormText(IlluminationNorm.CLAHE, P)).startsWith(FeiThreshold.NORM_HEADER + "clahe "));
        Files.delete(a);
        Files.delete(b);
        Files.delete(dir);
    }

    /** Итог с одной конфигурацией: контрольные чужие (человек, оценка), порог θ у способа П2 при FAR 1 %. */
    static FeiThreshold.Res res(int[] p, double[] r, double theta) {
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        c.ctrlP = p;
        c.ctrlR = r;
        FeiThreshold.Res res = new FeiThreshold.Res();
        res.label = "L1";
        res.cfgs = List.of(c);
        res.theta[2][0] = new double[] {theta};
        for (double v : r) {
            res.far[2][0][1]++;
            if (v <= theta) res.far[2][0][0]++;
        }
        return res;
    }

    @Test
    public void farByPeopleAndEffectiveBound() {
        // Люди 0, 1, 2 по 10 снимков; принято 2, 0 и 1 (оценка ≤ 0,5).
        int[] p = new int[30];
        double[] r = new double[30];
        for (int i = 0; i < 30; i++) {
            p[i] = i / 10;
            r[i] = 1;
        }
        r[0] = 0.1;
        r[1] = 0.2;
        r[20] = 0.3;
        FeiThreshold.Res res = res(p, r, 0.5);
        assertArrayEquals(new int[] {2, 3}, LbpNormReport.farPeople(res, 2, 0));
        double[] e = LbpNormReport.farEff(res, 2, 0);
        double ph = 3.0 / 30;
        double v = 3.0 / 2 * ((2 - ph * 10) * (2 - ph * 10) + (0 - ph * 10) * (0 - ph * 10) + (1 - ph * 10) * (1 - ph * 10)) / (30.0 * 30);
        double d = Math.max(1, v / (ph * (1 - ph) / 30));
        double ne = 30 / d;
        assertEquals(d, e[2], 1e-12);
        assertEquals(ne, e[3], 1e-12);
        assertEquals(new BetaDistribution(ph * ne + 1, ne - ph * ne).inverseCumulativeProbability(0.95), e[4], 1e-12);
        // Ноль принятых: deff = 1, граница 1 − 0,05^(1/n).
        FeiThreshold.Res z = res(p, r, 0.05);
        double[] ez = LbpNormReport.farEff(z, 2, 0);
        assertEquals(1, ez[2], 0);
        assertEquals(1 - Math.pow(0.05, 1 / 30.0), ez[4], 1e-12);
        assertArrayEquals(new int[] {0, 3}, LbpNormReport.farPeople(z, 2, 0));
    }
}
