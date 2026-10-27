package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Test;

/**
 * Пачка 1 (XBaseVec): а) S-norm и AS-norm по формуле на синтетике с известными распределениями; б) без нормировки – побайтно
 * XBaseAbl3.beta (β = 1); в) запись и чтение векторов – без потерь; г) слияние при w = 1 и w = 0 – те же решения (лучший, второй) и тот
 * же порядок оценок, что у кандидата отдельно; д) 1:1 по всем парам – порог по доле чужих пар.
 *
 * @author ssv
 */
public class XBaseVecTest {

    static XBaseVec.Model model(String method, int ng, List<XBaseVec.Row> rows, long seed) {
        Random r = new Random(seed);
        List<String> gal = new ArrayList<>();
        for (int g = 0; g < ng; g++) gal.add("p" + g);
        double[][] z = new double[ng][40];
        for (double[] v : z) for (int i = 0; i < v.length; i++) v[i] = Math.exp(r.nextGaussian() * 0.3 - 0.5);
        return new XBaseVec.Model("main", method, 0, gal, rows, z);
    }

    static XBaseVec.Row row(String role, int trueIdx, int ng, long seed) {
        Random r = new Random(seed);
        double[] dg = new double[ng];
        for (int g = 0; g < ng; g++) dg[g] = Math.exp(r.nextGaussian() * 0.4 - 0.4);
        double[] ci = new double[20];
        double[] co = new double[20];
        for (int i = 0; i < 20; i++) {
            ci[i] = Math.exp(r.nextGaussian() * 0.2 - 0.3);
            co[i] = Math.exp(r.nextGaussian() * 0.25 - 0.2);
        }
        return new XBaseVec.Row(role, trueIdx < 0 ? "z" + seed : "p" + trueIdx, (int) seed, "k" + seed, trueIdx, dg, ci, co);
    }

    static String[] cols(String line) {
        return line.trim().split("\t", -1);
    }

    @Test
    public void snormFormula() {
        int ng = 6;
        XBaseVec.Row r = row(FeiScores.OWN, 2, ng, 11);
        XBaseVec.Model m = model("L1", ng, List.of(r), 5);
        for (XBaseVec.Norm n : XBaseVec.NORMS) {
            double[] cv = n.cohort() == 0 ? r.dci() : r.dco();
            int k = n.as() ? n.k() : 20;
            double[] t = muSigma(cv, k);
            double[] ln = new double[ng];
            for (int g = 0; g < ng; g++) {
                double[] zc = new double[20];
                System.arraycopy(m.z()[g], n.cohort() == 0 ? 0 : 20, zc, 0, 20);
                double[] zs = muSigma(zc, k);
                double l = Math.log(r.dg()[g]);
                ln[g] = ((l - zs[0]) / zs[1] + (l - t[0]) / t[1]) / 2;
            }
            int b = 0;
            for (int g = 1; g < ng; g++) if (ln[g] < ln[b]) b = g;
            int c = b == 0 ? 1 : 0;
            for (int g = 0; g < ng; g++) if (g != b && ln[g] < ln[c]) c = g;
            String[] l = cols(XBaseVec.normLine(m, r, n, new int[2]));
            assertEquals("p" + b, l[7]);
            assertEquals("p" + c, l[9]);
            double want = XBaseAbl.ABS_C * XBaseVec.squash(2 * ln[b] - ln[c], null);
            assertEquals(want, Double.parseDouble(l[8]), 1e-12 * want);
            assertEquals(XBaseAbl.ABS_C, Double.parseDouble(l[10]), 0);
        }
        // S-norm и AS-norm K = 20 – те же значения (с точностью до порядка суммирования).
        double a = Double.parseDouble(cols(XBaseVec.normLine(m, r, XBaseVec.NORMS.get(0), new int[2]))[8]);
        double bb = Double.parseDouble(cols(XBaseVec.normLine(m, r, XBaseVec.NORMS.get(2), new int[2]))[8]);
        assertEquals(a, bb, 1e-12 * a);
    }

    /** μ и σ (несмещённый) log по k наименьшим. */
    static double[] muSigma(double[] v, int k) {
        double[] l = new double[v.length];
        for (int i = 0; i < v.length; i++) l[i] = Math.log(v[i]);
        java.util.Arrays.sort(l);
        int n = Math.min(k, l.length);
        double m = 0;
        for (int i = 0; i < n; i++) m += l[i];
        m /= n;
        double s = 0;
        for (int i = 0; i < n; i++) s += (l[i] - m) * (l[i] - m);
        return new double[] {m, Math.sqrt(s / (n - 1))};
    }

    @Test
    public void squashMonotone() {
        double prev = 0;
        for (double s = -250; s <= 250; s += 0.5) {
            double r = XBaseVec.squash(s, null);
            assertTrue(r > XBaseVec.RHO_MIN && r < 1);
            assertTrue("s = " + s, r > prev);
            prev = r;
        }
        assertEquals(Math.pow(XBaseVec.RHO_MIN, 0.5), XBaseVec.squash(0, null), 1e-20);
    }

    @Test
    public void baseIsBeta() {
        XBaseVec.Row r = row(FeiScores.CTRL, -1, 5, 3);
        XBaseVec.Model m = model("L1", 5, List.of(r), 1);
        String line = XBaseVec.line("L1", 0, m.gallery(), r);
        assertEquals(XBaseAbl3.beta(line, 1), XBaseVec.normLine(m, r, null, new int[2]));
        assertEquals(-Math.log(XBaseVec.RHO_MIN), -XBaseVec.ell(0), 0);
        assertEquals(Math.log(XBaseVec.RHO_MIN), XBaseVec.ell(-1e-13), 0);
    }

    @Test
    public void roundTrip() throws Exception {
        List<XBaseVec.Row> rows = new ArrayList<>();
        for (int i = 0; i < 4; i++) rows.add(row(i < 2 ? FeiScores.OWN : FeiScores.CTRL, i < 2 ? i : -1, 3, 20 + i));
        rows.get(3).dg()[1] = Double.POSITIVE_INFINITY;
        XBaseVec.Model m = model("L1+mirror", 3, rows, 7);
        m.z()[1] = null;
        Path p = Files.createTempFile("vec", ".tsv.gz");
        try {
            Writer w = XBaseVec.writer(p, "# h\n# Код: коммит x\n");
            XBaseVec.write(w, "блок 1", List.of(m, model("L1", 3, rows, 8)));
            XBaseVec.finish(w, p);
            Map<String, List<XBaseVec.Model>> back = XBaseVec.read(p);
            List<XBaseVec.Model> ms = back.get("блок 1");
            assertEquals(2, ms.size());
            XBaseVec.Model a = ms.get(0);
            assertEquals(m.method(), a.method());
            assertEquals(m.gallery(), a.gallery());
            assertEquals(null, a.z()[1]);
            assertArrayEquals(m.z()[0], a.z()[0], 0);
            for (int i = 0; i < rows.size(); i++) {
                assertEquals(rows.get(i), new XBaseVec.Row(a.rows().get(i).role(), a.rows().get(i).person(), a.rows().get(i).image(), a.rows().get(i).key(),
                        a.rows().get(i).trueIdx(), rows.get(i).dg(), rows.get(i).dci(), rows.get(i).dco()));
                assertArrayEquals(rows.get(i).dg(), a.rows().get(i).dg(), 0);
                assertArrayEquals(rows.get(i).dco(), a.rows().get(i).dco(), 0);
            }
            assertEquals("# Код: коммит x", XBaseVec.readHead(p).split("\n")[1]);
        } finally {
            Files.deleteIfExists(p);
        }
    }

    @Test
    public void fusionExtremes() {
        int ng = 5;
        List<XBaseVec.Row> rt = new ArrayList<>();
        List<XBaseVec.Row> rx = new ArrayList<>();
        Map<String, Double> rOf = new java.util.HashMap<>();
        for (int i = 0; i < 12; i++) {
            XBaseVec.Row a = row(i % 3 == 0 ? FeiScores.CTRL : FeiScores.OWN, i % 3 == 0 ? -1 : i % ng, ng, 100 + i);
            XBaseVec.Row b = row(a.role(), a.trueIdx(), ng, 500 + i);
            b = new XBaseVec.Row(a.role(), a.person(), a.image(), a.key(), a.trueIdx(), b.dg(), b.dci(), b.dco());
            rt.add(a);
            rx.add(b);
            rOf.put(a.key(), 0.0);
        }
        List<XBaseVec.Model> tt = List.of(model("L1", ng, rt, 1), model("L1+mirror", ng, rt, 2));
        List<XBaseVec.Model> xs = List.of(model("L1", ng, rx, 3), model("L1+mirror", ng, rx, 4));
        double[] st = {-0.5, 0.4};
        double[] sx = {-0.3, 0.7};
        for (int[] wc : new int[][] {{1, 0}, {0, 1}}) {
            double w = wc[0];
            List<XBaseVec.Row> src = w == 1 ? rt : rx;
            XBaseVec.Model ref = model("L1", ng, src, 9);
            String[] fused = XBaseVec.fuseLines(tt, xs, w, st, sx, rOf, 0.12, new int[2]).split("\n");
            List<double[]> pairs = new ArrayList<>();
            for (int i = 0; i < src.size(); i++) {
                String[] f = cols(fused[2 * i]);
                String[] b = cols(XBaseAbl3.beta(XBaseVec.line("L1", 0, ref.gallery(), src.get(i)), 1));
                assertEquals(b[7], f[7]);
                assertEquals(b[9], f[9]);
                pairs.add(new double[] {Double.parseDouble(b[8]), Double.parseDouble(f[8])});
            }
            // Порядок оценок – тот же (монотонное преобразование).
            for (double[] p : pairs) for (double[] q : pairs) if (p[0] < q[0]) assertTrue(p[1] < q[1]);
        }
    }

    @Test
    public void allPairs() {
        int ng = 4;
        List<XBaseVec.Row> rows = new ArrayList<>();
        for (int i = 0; i < 4; i++) rows.add(new XBaseVec.Row(FeiScores.OWN, "p" + i, i, "o" + i, i, new double[] {0.5, 0.5, 0.5, 0.5}, new double[20],
                new double[20]));
        rows.set(0, new XBaseVec.Row(FeiScores.OWN, "p0", 0, "o0", 0, new double[] {0.1, 0.5, 0.5, 0.5}, new double[20], new double[20]));
        for (int i = 0; i < 25; i++) {
            double[] d = new double[ng];
            for (int g = 0; g < ng; g++) d[g] = 0.2 + 0.01 * (4 * i + g);
            rows.add(new XBaseVec.Row(FeiScores.CTRL, "z" + i, 0, "c" + i, -1, d, new double[20], new double[20]));
        }
        XBaseVec.Model x = model("L1", ng, rows, 1);
        XBaseVec.Model mm = model("L1+mirror", ng, rows, 2);
        double[] r = XBaseVec.pairs(List.of(x, mm), row -> false);
        assertEquals(100, r[5], 0);
        // FAR 1 %: k = 1 – принимается d < 0,21 – из своих только p0 (0,1); ошибка 3 из 4.
        assertEquals(75, r[0], 1e-9);
        // FAR 10 %: k = 10 – порог < 0,30; свои 0,5 отвергнуты.
        assertEquals(75, r[3], 1e-9);
    }
}
