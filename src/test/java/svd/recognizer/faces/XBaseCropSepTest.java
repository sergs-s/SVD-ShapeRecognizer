package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Проверка выравнивания и разделимости на GT (PLAN.md п. 4.2): L1-crop-92 — столбцы 10–101 alignCrop; поправка отражения
 * alignCrop способом а по шаблону ArcFace; d₂ своих при неверном кандидате = лучший; точка равных ошибок.
 *
 * @author ssv
 */
public class XBaseCropSepTest {

    @BeforeClass
    public static void load() {
        nu.pattern.OpenCV.loadLocally();
    }

    @Test
    public void crop92IsCentralColumns() {
        Mat m = new Mat(112, 112, CvType.CV_8UC1);
        byte[] b = new byte[112 * 112];
        for (int y = 0; y < 112; y++) for (int x = 0; x < 112; x++) b[y * 112 + x] = (byte) (x + y);
        m.put(0, 0, b);
        Mat c = XBaseEval.crop92(m);
        assertEquals(92, c.cols());
        assertEquals(112, c.rows());
        byte[] cb = new byte[92 * 112];
        c.get(0, 0, cb);
        for (int y = 0; y < 112; y++) for (int x = 0; x < 92; x++) assertEquals(b[y * 112 + x + 10], cb[y * 92 + x]);
        // Шаблон кадра «а» — шаблон ArcFace со сдвигом −10 px.
        double[][] t = FaceAlignment.ownTemplate(92, 112);
        for (int i = 0; i < 5; i++) assertArrayEquals(new double[] {FaceAlignment.ARCFACE_112[i][0] - 10, FaceAlignment.ARCFACE_112[i][1]}, t[i], 1e-12);
        m.release();
        c.release();
    }

    @Test
    public void mirrorCorrectionForAlignCrop() {
        double[][] t = FaceAlignment.ARCFACE_112;
        double[][] mt = FaceMirror.mirroredTemplate(t, 112);
        double[][] c = FaceMirror.correction(t, 112);
        double before = 0;
        double after = 0;
        for (int i = 0; i < 5; i++) {
            double[] p = FaceMirror.apply(c, mt[i]);
            after = Math.max(after, Math.hypot(p[0] - t[i][0], p[1] - t[i][1]));
            before = Math.max(before, Math.hypot(mt[i][0] - t[i][0], mt[i][1] - t[i][1]));
        }
        assertTrue("простое отражение сдвигает точки на ≥ 0,5 px: " + before, before >= 0.5);
        assertTrue("после поправки остаток ≤ 0,3 px: " + after, after <= 0.3);
    }

    @Test
    public void ownSecondDistanceAndEer() {
        Map<String, FeiScores.Row> rows = new HashMap<>();
        List<String> g = List.of("s01", "s02");
        // Свой s01: верный кандидат (d₂ — второй), свой s02: неверный кандидат (d₂ — лучший).
        String[] lines = {
            FeiScores.line("m", 0, 0, "own", "s01", 1, 0, g, new double[] {0.2, 0.8}, false),
            FeiScores.line("m", 0, 0, "own", "s02", 1, 1, g, new double[] {0.3, 0.6}, false),
            FeiScores.line("m", 0, 0, "ctrl", "s03", 1, -1, g, new double[] {0.5, 0.6}, false),
            FeiScores.line("m", 0, 0, "ctrl", "s04", 1, -1, g, new double[] {0.9, 0.95}, false),
            FeiScores.line("m", 0, 0, "cv0", "s01", 2, -1, List.of("s02", "s03"), new double[] {0.5, 0.7}, false)};
        for (String l : lines) {
            FeiScores.Row r = FeiScores.parse(l.strip());
            rows.put(r.method() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        Map<String, Double> rOf = Map.of("s01/1", 0.0, "s02/1", 0.0, "s03/1", 0.0, "s04/1", 0.0, "s01/2", 0.0);
        Map<String, Integer> p = Map.of("s01", 0, "s02", 1, "s03", 2, "s04", 3);
        XBaseSep.Data d = XBaseSep.data(rows, "m", false, rOf, 0.12, p);
        for (int i = 0; i < d.ownR.length; i++) {
            if (d.ownOk[i]) {
                assertEquals(0.2, d.ownD1[i], 0);
                assertEquals(0.8, d.ownD2[i], 0);
                assertEquals(0.25, d.ownR[i], 1e-15);
            } else {
                assertEquals(0.6, d.ownD1[i], 0);
                assertEquals(0.3, d.ownD2[i], 0);
            }
        }
        // Свои: r = 0,25 (верный), 0,5 (неверный — всегда ошибка); чужие r = 5/6, 0,9/0,95.
        // θ = 0,25: ошибка своих 50 %, FAR 0 % — разрыв 50; θ = 5/6: 50 % и 50 % — разрыв 0.
        double[] e = XBaseSep.eer(d, new int[] {1, 1, 1, 1});
        assertEquals(0.5 / 0.6, e[0], 1e-12);
        assertEquals(50, e[1], 1e-12);
        assertEquals(50, e[2], 1e-12);
        assertEquals(50, e[3], 1e-12);
        assertEquals(50, XBaseSep.ownErr(d, new int[] {1, 1, 1, 1}, 1.0), 1e-12);
        assertEquals(100, XBaseSep.ctrlFar(d, new int[] {1, 1, 1, 1}, 1.0), 1e-12);
    }
}
