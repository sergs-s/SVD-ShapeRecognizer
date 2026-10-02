package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Зеркальные копии кадра «а» (FaceMirror, способ а): отражение, поправка подобием, выравнивание отражённого кадра.
 *
 * @author ssv
 */
public class FaceMirrorTest {

    @BeforeClass
    public static void loadOpenCv() {
        nu.pattern.OpenCV.loadLocally();
    }

    private static Mat random(int w, int h, long seed) {
        Random rnd = new Random(seed);
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        byte[] b = new byte[w * h];
        rnd.nextBytes(b);
        m.put(0, 0, b);
        return m;
    }

    private static byte[] bytes(Mat m) {
        byte[] b = new byte[(int) m.total()];
        m.get(0, 0, b);
        return b;
    }

    /** Кадр с яркими точками 3×3 в точках шаблона t (центры округлены). */
    private static Mat dots(double[][] t, int w, int h) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC1);
        for (double[] p : t) {
            int x = (int) Math.round(p[0]);
            int y = (int) Math.round(p[1]);
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) m.put(y + dy, x + dx, 255.0);
        }
        return m;
    }

    /** Центр яркости в окне 9×9 около точки p. */
    private static double[] centroid(Mat m, double[] p) {
        double sx = 0, sy = 0, s = 0;
        int cx = (int) Math.round(p[0]);
        int cy = (int) Math.round(p[1]);
        for (int y = cy - 4; y <= cy + 4; y++) {
            for (int x = cx - 4; x <= cx + 4; x++) {
                double v = m.get(y, x)[0];
                sx += v * x;
                sy += v * y;
                s += v;
            }
        }
        return new double[] {sx / s, sy / s};
    }

    @Test
    public void flipTwiceIsIdentity() {
        Mat m = random(92, 112, 1);
        Mat f = FaceMirror.flip(FaceMirror.flip(m));
        assertArrayEquals(bytes(m), bytes(f));
    }

    @Test
    public void mirroredTemplateSwapsEyesAndMouthCorners() {
        double[][] t = FaceAlignment.ownTemplate(92, 112);
        double[][] m = FaceMirror.mirroredTemplate(t, 92);
        assertEquals(91 - t[1][0], m[0][0], 0.0); // левый глаз отражённого кадра — отражённый правый
        assertEquals(t[1][1], m[0][1], 0.0);
        assertEquals(91 - t[2][0], m[2][0], 0.0); // нос на месте
        assertEquals(91 - t[4][0], m[3][0], 0.0);
        // Шаблон кадра «а» несимметричен: без поправки глаза отражённого кадра левее эталона на 0,83 px.
        assertEquals(-0.826, m[0][0] - t[0][0], 0.001);
    }

    @Test
    public void symmetricTemplateNeedsNoCorrectionAndMirroredFrameIsAligned() {
        int w = 91; // нечётная ширина: ось (w − 1)/2 = 45 — целый столбец, точки симметричны и после округления
        int h = 112;
        // Симметричный шаблон: глаза на одной высоте, нос и середина рта на оси (w − 1)/2.
        double ax = (w - 1) / 2.0;
        double[][] t = {{ax - 18, 52}, {ax + 18, 52}, {ax, 72}, {ax - 15, 92}, {ax + 15, 92}};
        double[][] c = FaceMirror.correction(t, w);
        assertArrayEquals(new double[] {1, 0, 0}, c[0], 1e-12);
        assertArrayEquals(new double[] {0, 1, 0}, c[1], 1e-12);
        // Кадр с точками в симметричном шаблоне после отражения совпадает с собой побайтно.
        Mat d = dots(t, w, h);
        assertArrayEquals(bytes(d), bytes(FaceMirror.mirror(d, t)));
    }

    @Test
    public void correctionMapsMirroredTemplateOntoTemplate() {
        double[][] t = FaceAlignment.ownTemplate(92, 112);
        double[][] mt = FaceMirror.mirroredTemplate(t, 92);
        double[][] c = FaceMirror.correction(t, 92);
        for (int i = 0; i < t.length; i++) {
            double[] p = FaceMirror.apply(c, mt[i]);
            assertEquals(t[i][0], p[0], 0.3);
            assertEquals(t[i][1], p[1], 0.3);
        }
        // Глаза (по ним в первую очередь выравнивание) — точнее.
        for (int i = 0; i < 2; i++) assertEquals(0, Math.hypot(FaceMirror.apply(c, mt[i])[0] - t[i][0], FaceMirror.apply(c, mt[i])[1] - t[i][1]), 0.15);
    }

    @Test
    public void mirroringTwiceIsIdentityTransform() {
        double[][] c = FaceMirror.correction(FaceAlignment.ownTemplate(92, 112), 92);
        double[][] pts = {{0, 0}, {91, 0}, {0, 111}, {45.5, 60}, {20, 90}};
        for (double[] p : pts) {
            double[] q = FaceMirror.apply(c, new double[] {91 - p[0], p[1]});
            double[] r = FaceMirror.apply(c, new double[] {91 - q[0], q[1]});
            assertEquals(p[0], r[0], 1e-6);
            assertEquals(p[1], r[1], 1e-6);
        }
    }

    @Test
    public void mirroredFrameOfTemplateDotsIsAlignedToTemplate() {
        double[][] t = FaceAlignment.ownTemplate(92, 112);
        // Кадр «а» с точками ровно в точках шаблона (лицо выровнено); отражение с поправкой — точки снова у шаблона.
        Mat d = dots(t, 92, 112);
        Mat m = FaceMirror.mirror(d);
        for (int i = 0; i < 2; i++) {
            double[] p = centroid(m, t[i]);
            assertTrue("глаз " + i + ": " + p[0] + ", " + p[1], Math.hypot(p[0] - Math.round(t[i][0]), p[1] - Math.round(t[i][1])) < 0.6);
        }
    }
}
