package svd.recognizer.faces;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Зеркальные копии кадра «а» для обучения (этап 5б, строки +mirror в faces-fei; PLAN.md, п. 4.2). Шаблон кадра
 * ({@link FaceAlignment#ownTemplate}) несимметричен относительно вертикальной оси кадра (середина глаз 45,91 при оси
 * 45,5 у кадра 92 px; линия глаз чуть наклонена), поэтому простое отражение сдвигает лицо относительно эталона на
 * 0,8–1,3 px — одинаково у всех отражённых кадров. Способ (а), решение Хозяина 02.10.2026: отражённый кадр выравнивается
 * заново преобразованием подобия, которое переводит отражённый шаблон (левый и правый глаз, углы рта меняются местами) в
 * шаблон; warpAffine, INTER_LINEAR, BORDER_REPLICATE — как у кадра «а». Подобие — по методу наименьших квадратов по
 * 5 точкам (явная формула, без случайного отбора).
 *
 * @author ssv
 */
final class FaceMirror {

    /** Порядок точек шаблона после отражения: левый и правый глаз, углы рта меняются местами, нос на месте. */
    static final int[] SWAP = {1, 0, 2, 4, 3};

    private FaceMirror() {
    }

    /** Отражение по горизонтали без поправки (столбец i → width − 1 − i). */
    static Mat flip(Mat frame) {
        Mat out = new Mat();
        Core.flip(frame, out, 1);
        return out;
    }

    /** Точки шаблона t на отражённом кадре ширины width, в порядке шаблона (глаза и углы рта переставлены). */
    static double[][] mirroredTemplate(double[][] t, int width) {
        double[][] m = new double[t.length][2];
        for (int i = 0; i < t.length; i++) {
            m[i][0] = width - 1 - t[SWAP[i]][0];
            m[i][1] = t[SWAP[i]][1];
        }
        return m;
    }

    /**
     * Подобие (2×3: [a −b tx; b a ty]), переводящее точки from в точки to по наименьшим квадратам (центрирование, затем
     * a = Σ(x·x′ + y·y′)/Σ(x² + y²), b = Σ(x·y′ − y·x′)/Σ(x² + y²)).
     */
    static double[][] similarityLsq(double[][] from, double[][] to) {
        int n = from.length;
        double fx = 0, fy = 0, tx = 0, ty = 0;
        for (int i = 0; i < n; i++) {
            fx += from[i][0];
            fy += from[i][1];
            tx += to[i][0];
            ty += to[i][1];
        }
        fx /= n;
        fy /= n;
        tx /= n;
        ty /= n;
        double sa = 0, sb = 0, ss = 0;
        for (int i = 0; i < n; i++) {
            double x = from[i][0] - fx;
            double y = from[i][1] - fy;
            double u = to[i][0] - tx;
            double v = to[i][1] - ty;
            sa += x * u + y * v;
            sb += x * v - y * u;
            ss += x * x + y * y;
        }
        double a = sa / ss;
        double b = sb / ss;
        return new double[][] {{a, -b, tx - (a * fx - b * fy)}, {b, a, ty - (b * fx + a * fy)}};
    }

    /** Поправка для отражённого кадра ширины width с шаблоном t: отражённый шаблон → шаблон. */
    static double[][] correction(double[][] t, int width) {
        return similarityLsq(mirroredTemplate(t, width), t);
    }

    /** Отражённый и заново выровненный кадр (способ а) с шаблоном кадра «а» ({@link FaceAlignment#ownTemplate}). */
    static Mat mirror(Mat frame) {
        return mirror(frame, FaceAlignment.ownTemplate(frame.cols(), frame.rows()));
    }

    /** Отражённый и заново выровненный кадр по шаблону t. */
    static Mat mirror(Mat frame, double[][] t) {
        int w = frame.cols();
        int h = frame.rows();
        double[][] c = correction(t, w);
        Mat m = new Mat(2, 3, CvType.CV_64F);
        for (int r = 0; r < 2; r++) m.put(r, 0, c[r]);
        Mat f = flip(frame);
        Mat out = new Mat();
        Imgproc.warpAffine(f, out, m, new Size(w, h), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        f.release();
        m.release();
        return out;
    }

    /** Применить подобие c к точке p. */
    static double[] apply(double[][] c, double[] p) {
        return new double[] {c[0][0] * p[0] + c[0][1] * p[1] + c[0][2], c[1][0] * p[0] + c[1][1] * p[1] + c[1][2]};
    }
}
