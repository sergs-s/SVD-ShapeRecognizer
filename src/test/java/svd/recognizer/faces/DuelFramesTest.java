package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.OpenCvLoader;

/**
 * Д-№2 (дуэль): цветной кадр набора – то же окно, что у серого кадра метода Прокаева. Снимок с тремя равными каналами: серый от цветного
 * кадра побайтно равен прежнему серому кадру (К-лицо – crop, К-обычный – plain); окна – как у прежней геометрии (копия кода до выноса).
 *
 * @author ssv
 */
public class DuelFramesTest {

    @BeforeClass
    public static void load() {
        OpenCvLoader.load();
    }

    /** Строки YuNet (рамка x, y, w, h) – случайные в снимке cols × rows, в том числе у краёв и крупнее снимка. */
    static List<double[]> rows(int cols, int rows, long seed) {
        Random rnd = new Random(seed);
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            double w = 20 + rnd.nextDouble() * cols * 0.9;
            double h = w * (1.0 + rnd.nextDouble() * 0.4);
            double x = -w * 0.2 + rnd.nextDouble() * (cols - w * 0.6);
            double y = -h * 0.2 + rnd.nextDouble() * (rows - h * 0.6);
            out.add(new double[] {x, y, w, h});
        }
        return out;
    }

    static Mat noise(int cols, int rows, long seed) {
        Mat g = new Mat(rows, cols, CvType.CV_8UC1);
        byte[] b = new byte[cols * rows];
        new Random(seed).nextBytes(b);
        g.put(0, 0, b);
        return g;
    }

    static boolean same(Mat a, Mat b) {
        if (!a.size().equals(b.size()) || a.type() != b.type()) return false;
        Mat d = new Mat();
        Core.absdiff(a, b, d);
        boolean eq = Core.countNonZero(d) == 0;
        d.release();
        return eq;
    }

    @Test
    public void grayOfColorFrameEqualsMethodFrame() {
        int[][] sizes = {{640, 480}, {512, 768}, {300, 600}};
        for (int[] sz : sizes) {
            Mat gray = noise(sz[0], sz[1], 7 + sz[0]);
            Mat color = new Mat();
            Core.merge(List.of(gray, gray, gray), color);
            for (double[] r : rows(sz[0], sz[1], 11 + sz[1])) {
                Mat cf = ProkaevFrames.colorFrame(color, r, false, null);
                Mat cg = new Mat();
                Imgproc.cvtColor(cf, cg, Imgproc.COLOR_BGR2GRAY);
                Mat old = ProkaevFrames.crop(color, r, null);
                assertEquals("К-лицо " + sz[0] + "×" + sz[1], true, same(cg, old));
                Mat pf = ProkaevFrames.colorFrame(color, r, true, null);
                Mat pg = new Mat();
                Imgproc.cvtColor(pf, pg, Imgproc.COLOR_BGR2GRAY);
                Mat g2 = gray.clone();
                Mat oldP = ProkaevFrames.plain(g2, r, null, 0);
                assertEquals("К-обычный " + sz[0] + "×" + sz[1], true, same(pg, oldP));
                for (Mat m : List.of(cf, cg, old, pf, pg, oldP)) m.release();
            }
        }
    }

    /** Прежняя геометрия К-лицо (копия кода ProkaevFrames.crop до выноса в faceWindow, 22.10.2026). */
    static Rect oldFace(int cols, int rows, double[] row, int[] clipped) {
        double hh = row[3] / ProkaevFrames.BOX_SHARE;
        double ww = hh * ProkaevFrames.W / ProkaevFrames.H;
        if (ww > cols || hh > rows) {
            double f = Math.min(cols / ww, rows / hh);
            ww *= f;
            hh *= f;
            clipped[0]++;
        }
        int w = (int) Math.min(cols, Math.round(ww));
        int h = (int) Math.min(rows, Math.round(hh));
        int x0 = (int) Math.round(row[0] + row[2] / 2 - w / 2.0);
        int y0 = (int) Math.round(row[1] + row[3] / 2 - h / 2.0);
        x0 = Math.max(0, Math.min(x0, cols - w));
        y0 = Math.max(0, Math.min(y0, rows - h));
        return new Rect(x0, y0, w, h);
    }

    /** Прежняя геометрия К-обычный (копия кода ProkaevFrames.plain до выноса в plainWindow). */
    static Rect oldPlain(int cols, int rows, double[] row, int[] clipped) {
        int w;
        int h;
        if (cols * 3L >= rows * 2L) {
            h = rows;
            w = (int) Math.min(cols, Math.round(h * 2.0 / 3));
        } else {
            w = cols;
            h = (int) Math.min(rows, Math.round(w * 1.5));
        }
        if (row[2] > w) clipped[0]++;
        int x0 = (int) Math.round(row[0] + row[2] / 2 - w / 2.0);
        int y0 = (int) Math.round(row[1] + row[3] / 2 - h / 2.0);
        x0 = Math.max(0, Math.min(x0, cols - w));
        y0 = Math.max(0, Math.min(y0, rows - h));
        return new Rect(x0, y0, w, h);
    }

    @Test
    public void windowsAsBefore() {
        int[][] sizes = {{640, 480}, {512, 768}, {92, 112}, {4000, 3000}};
        for (int[] sz : sizes) {
            for (double[] r : rows(sz[0], sz[1], 13 + sz[0])) {
                int[] a = new int[1];
                int[] b = new int[1];
                assertEquals(oldFace(sz[0], sz[1], r, a), ProkaevFrames.faceWindow(sz[0], sz[1], r, b));
                assertEquals(a[0], b[0]);
                a[0] = 0;
                b[0] = 0;
                assertEquals(oldPlain(sz[0], sz[1], r, a), ProkaevFrames.plainWindow(sz[0], sz[1], r, b));
                assertEquals(a[0], b[0]);
            }
        }
    }
}
