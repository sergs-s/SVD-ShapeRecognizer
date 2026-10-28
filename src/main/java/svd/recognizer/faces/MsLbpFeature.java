package svd.recognizer.faces;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Многомасштабный LBP вокруг опорных точек (Chen, Cao, Wen, Sun, «Blessing of dimensionality», CVPR 2013; пачка 2, задание
 * Жорика №2–№3 от 28.10.2026). У Chen: 27 точек, пирамида выровненного лица из 5 масштабов (стороны 300, 212, 150, 106, 75 –
 * шаг 1/√2), окно 40×40 вокруг точки на каждом масштабе, 4×4 ячейки. У нас: 5 точек YuNet (глаза, нос, углы рта), перенесённые
 * в выровненный кадр k = 2 (184×224) тем же подобием; масштаб s = 0…4 – кадр, уменьшенный INTER_AREA до round(w·2^(−s/2)) ×
 * round(h·2^(−s/2)) (s = 0 – как есть), точка – по центрам пикселей ((p + 0,5)·w_s/w − 0,5); окно 40×40 – столбцы
 * round(p) − 20 … round(p) + 19 (так же строки), значения за краем кадра – повтор крайнего пикселя; LBP u2 (8, 2) (LbpFeature,
 * радиус в статье не указан – как у базы, решение Жорика №3) по окну с полями 2 (44×44 → коды 40×40); 4×4 ячейки 10×10,
 * гистограмма ячейки на сумму 1 (59 меток), корень. Порядок: точка, масштаб, ячейка по строкам; длина 5·5·16·59 = 23 600.
 *
 * @author ssv
 */
final class MsLbpFeature {

    static final int POINTS = 5;
    static final int SCALES = 5;
    static final int PATCH = 40;
    static final int CELLS = 4;
    static final int CELL = PATCH / CELLS;
    static final int MARGIN = (int) LbpFeature.RADIUS;
    static final int LENGTH = POINTS * SCALES * CELLS * CELLS * LbpFeature.BINS;

    private MsLbpFeature() {
    }

    /** Размер масштаба s: round(n·2^(−s/2)). */
    static int size(int n, int s) {
        return (int) Math.round(n * Math.pow(2, -s / 2.0));
    }

    /** Уменьшенный кадр масштаба s (INTER_AREA; s = 0 – копия). */
    static double[] scaled(double[] img, int w, int h, int s) {
        if (s == 0) return img.clone();
        Mat m = new Mat(h, w, CvType.CV_64F);
        m.put(0, 0, img);
        Mat o = new Mat();
        Imgproc.resize(m, o, new Size(size(w, s), size(h, s)), 0, 0, Imgproc.INTER_AREA);
        double[] v = new double[(int) o.total()];
        o.get(0, 0, v);
        m.release();
        o.release();
        return v;
    }

    /** Окно (PATCH + 2·MARGIN)² с центром у пикселя (cx, cy) (столбцы cx − 22 … cx + 21), за краем – повтор крайнего пикселя. */
    static double[] patch(double[] img, int w, int h, int cx, int cy) {
        int n = PATCH + 2 * MARGIN;
        int x0 = cx - PATCH / 2 - MARGIN;
        int y0 = cy - PATCH / 2 - MARGIN;
        double[] p = new double[n * n];
        for (int y = 0; y < n; y++) {
            int yy = Math.max(0, Math.min(h - 1, y0 + y));
            for (int x = 0; x < n; x++) {
                int xx = Math.max(0, Math.min(w - 1, x0 + x));
                p[y * n + x] = img[yy * w + xx];
            }
        }
        return p;
    }

    /** Корни гистограмм 4×4 ячеек окна (CELLS²·59). */
    static double[] cells(double[] patch) {
        int n = PATCH + 2 * MARGIN;
        int[] codes = LbpFeature.codes(patch, n, n, LbpFeature.RADIUS);
        double[] v = new double[CELLS * CELLS * LbpFeature.BINS];
        for (int cy = 0; cy < CELLS; cy++) {
            for (int cx = 0; cx < CELLS; cx++) {
                int off = (cy * CELLS + cx) * LbpFeature.BINS;
                for (int y = cy * CELL; y < (cy + 1) * CELL; y++) for (int x = cx * CELL; x < (cx + 1) * CELL; x++) v[off + LbpFeature.LABEL[codes[y * PATCH + x]]]++;
            }
        }
        for (int i = 0; i < v.length; i++) v[i] = Math.sqrt(v[i] / (CELL * CELL));
        return v;
    }

    /** Признак (LENGTH, корни) по кадру img (w × h, по строкам) и точкам pts [POINTS][x, y] в его координатах. */
    static double[] feature(double[] img, int w, int h, double[][] pts) {
        if (pts.length != POINTS) throw new IllegalArgumentException("MS-LBP: точек " + pts.length);
        double[] v = new double[LENGTH];
        int k = 0;
        double[][] sc = new double[SCALES][];
        for (int s = 0; s < SCALES; s++) sc[s] = scaled(img, w, h, s);
        for (double[] p : pts) {
            for (int s = 0; s < SCALES; s++) {
                int ws = s == 0 ? w : size(w, s);
                int hs = s == 0 ? h : size(h, s);
                int cx = (int) Math.round((p[0] + 0.5) * ws / w - 0.5);
                int cy = (int) Math.round((p[1] + 0.5) * hs / h - 0.5);
                double[] c = cells(patch(sc[s], ws, hs, cx, cy));
                System.arraycopy(c, 0, v, k, c.length);
                k += c.length;
            }
        }
        return v;
    }

    /**
     * Множитель части MS-LBP в сцеплении (б) с (П2) сетки grid: квадрат полной нормы части MS-LBP (POINTS·SCALES·16 ячеек по 1)
     * приравнивается квадрату нормы части (П2) – 2·grid² (по блоку: LBP 1 + POEM 3 · 1/3; решение Жорика №3 от 28.10).
     */
    static double joinScale(int grid) {
        return Math.sqrt(2.0 * grid * grid / (POINTS * SCALES * CELLS * CELLS));
    }
}
