package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.FeretData.Img;

/**
 * Уточнение точек глаз (PLAN.md п. 5.5, (а2б)): центр по градиентам на синтетическом «зрачке», обрезка области по краю
 * снимка, расхождение по явным точкам, взвешенные квантиль и доля. Данные синтетические.
 *
 * @author ssv
 */
public class EyeRefineTest {

    @BeforeClass
    public static void load() {
        svd.recognizer.OpenCvLoader.load();
    }

    /** Светлый фон, тёмный круг радиуса rad с центром (cx, cy) — рисуется с субпиксельной точностью (shift 4). */
    static Mat pupil(double cx, double cy, double rad) {
        Mat m = new Mat(240, 320, CvType.CV_8UC1, new Scalar(200));
        Imgproc.circle(m, new Point(cx * 16, cy * 16), (int) Math.round(rad * 16), new Scalar(30), -1, Imgproc.LINE_AA, 4);
        return m;
    }

    @Test
    public void gradientFindsPupil() {
        double[][] cases = {{150.0, 110.0}, {157.3, 104.6}, {141.8, 121.2}};
        for (double[] c : cases) {
            Mat m = pupil(c[0], c[1], 9);
            // Грубая точка смещена на 6 px, межглазье 80 px → область 40 px.
            double[] p = EyeRefine.gradient(m, new double[] {c[0] + 6, c[1] - 4}, 80);
            assertTrue("x " + p[0] + " против " + c[0], Math.abs(p[0] - c[0]) <= 1.0);
            assertTrue("y " + p[1] + " против " + c[1], Math.abs(p[1] - c[1]) <= 1.0);
            m.release();
        }
    }

    @Test
    public void squareClipped() {
        Mat m = new Mat(100, 100, CvType.CV_8UC1, new Scalar(0));
        Rect r = EyeRefine.square(m, new double[] {2, 50}, 20);
        assertEquals(0, r.x);
        assertEquals(12, r.width);
        assertNull(EyeRefine.square(m, new double[] {-30, 50}, 20));
    }

    @Test
    public void eyeErrorPoints() {
        Img im = new Img("00001_930831_fa", "00001", "930831", "fa", "-", null, new double[] {140, 100}, new double[] {60, 100}, null, null);
        double[] row = new double[15];
        row[4] = 64;
        row[5] = 103;
        row[6] = 140;
        row[7] = 108;
        assertArrayEquals(FeretData.eyeError(im, row), FeretData.eyeError(im, new double[] {64, 103}, new double[] {140, 108}), 0);
        assertEquals(0.0625, FeretData.eyeError(im, row)[1], 1e-12); // правый: 5 / 80
        assertNull(FeretData.eyeError(im, null, new double[] {1, 1}));
    }

    @Test
    public void weighted() {
        double[] v = {5, 1, 3, 4, 2};
        assertEquals(3, XBaseEyePts.weightedQuantile(v, new int[] {1, 1, 1, 1, 1}, 0.5), 0);
        assertEquals(5, XBaseEyePts.weightedQuantile(v, new int[] {5, 0, 0, 0, 0}, 0.5), 0);
        assertEquals(0.4, XBaseEyePts.weightedShare(v, new int[] {1, 1, 1, 1, 1}, 3), 1e-12);
        assertEquals(1.0 / 3, XBaseEyePts.weightedShare(v, new int[] {1, 2, 0, 0, 0}, 3), 1e-12);
    }
}
