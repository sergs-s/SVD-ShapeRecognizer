package svd.recognizer.faces;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * Уточнение центра глаза в малой области вокруг грубой точки (PLAN.md п. 5.5, (а2б), кандидат 0; параметры заданы заранее и
 * не подбираются):
 * <ul>
 *   <li>0а — центр по градиентам (Timm, Barth, 2011, «Accurate eye centre localisation by means of gradients»): квадрат со
 *       стороной GRAD_SIDE межглазья вокруг точки, масштаб к ширине GRAD_WIDTH, сглаживание Гаусса σ = 1, градиенты Собеля с
 *       модулем больше среднего + 0,3·СКО, центр — максимум w(c)·Σ max(0, dᵀg)² с весом «темноты» w = 255 − сглаженная
 *       яркость, поиск без края BORDER;</li>
 *   <li>0б — каскад глаз в квадрате со стороной CASCADE_SIDE межглазья: ближайший к точке найденный глаз, центр — центр
 *       прямоугольника; не нашёл — null;</li>
 *   <li>0в — 0б, затем 0а внутри найденного прямоугольника; каскад не нашёл — 0а.</li>
 * </ul>
 * Вход — серый снимок (CV_8U) в координатах снимка.
 *
 * @author ssv
 */
final class EyeRefine {

    static final double GRAD_SIDE = 0.5;
    static final int GRAD_WIDTH = 50;
    static final double GRAD_THRESHOLD = 0.3;
    static final int BORDER = 2;
    static final double CASCADE_SIDE = 0.6;

    private EyeRefine() {
    }

    /** Квадрат со стороной side вокруг точки p, обрезанный по снимку; null — пусто. */
    static Rect square(Mat gray, double[] p, double side) {
        int x0 = (int) Math.round(p[0] - side / 2);
        int y0 = (int) Math.round(p[1] - side / 2);
        int x1 = (int) Math.round(p[0] + side / 2);
        int y1 = (int) Math.round(p[1] + side / 2);
        x0 = Math.max(0, x0);
        y0 = Math.max(0, y0);
        x1 = Math.min(gray.cols(), x1);
        y1 = Math.min(gray.rows(), y1);
        return x1 - x0 < 4 || y1 - y0 < 4 ? null : new Rect(x0, y0, x1 - x0, y1 - y0);
    }

    /** 0а: центр по градиентам вокруг точки p при межглазье iod; null — область пуста. */
    static double[] gradient(Mat gray, double[] p, double iod) {
        Rect r = square(gray, p, GRAD_SIDE * iod);
        return r == null ? null : gradient(gray, r);
    }

    /** Центр по градиентам в прямоугольнике r снимка (координаты снимка). */
    static double[] gradient(Mat gray, Rect r) {
        Mat roi = new Mat(gray, r);
        double scale = (double) GRAD_WIDTH / r.width;
        int h = Math.max(4, (int) Math.round(r.height * scale));
        Mat small = new Mat();
        Imgproc.resize(roi, small, new Size(GRAD_WIDTH, h), 0, 0, Imgproc.INTER_AREA);
        Mat f = new Mat();
        small.convertTo(f, CvType.CV_64F);
        Mat blur = new Mat();
        Imgproc.GaussianBlur(f, blur, new Size(0, 0), 1.0);
        Mat gx = new Mat();
        Mat gy = new Mat();
        Imgproc.Sobel(blur, gx, CvType.CV_64F, 1, 0, 3);
        Imgproc.Sobel(blur, gy, CvType.CV_64F, 0, 1, 3);
        int w = GRAD_WIDTH;
        double[] ax = new double[w * h];
        double[] ay = new double[w * h];
        double[] b = new double[w * h];
        gx.get(0, 0, ax);
        gy.get(0, 0, ay);
        blur.get(0, 0, b);
        for (Mat m : new Mat[] {small, f, blur, gx, gy}) m.release();
        double[] mag = new double[w * h];
        double sum = 0;
        for (int i = 0; i < mag.length; i++) {
            mag[i] = Math.hypot(ax[i], ay[i]);
            sum += mag[i];
        }
        double mean = sum / mag.length;
        double var = 0;
        for (double m : mag) var += (m - mean) * (m - mean);
        double thr = mean + GRAD_THRESHOLD * Math.sqrt(var / mag.length);
        int n = 0;
        int[] px = new int[mag.length];
        int[] py = new int[mag.length];
        double[] ux = new double[mag.length];
        double[] uy = new double[mag.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                if (mag[i] <= thr || mag[i] == 0) continue;
                px[n] = x;
                py[n] = y;
                ux[n] = ax[i] / mag[i];
                uy[n] = ay[i] / mag[i];
                n++;
            }
        }
        double best = -1;
        int bx = w / 2;
        int by = h / 2;
        for (int cy = BORDER; cy < h - BORDER; cy++) {
            for (int cx = BORDER; cx < w - BORDER; cx++) {
                double s = 0;
                for (int k = 0; k < n; k++) {
                    double dx = px[k] - cx;
                    double dy = py[k] - cy;
                    double d = Math.hypot(dx, dy);
                    if (d == 0) continue;
                    double dot = (dx * ux[k] + dy * uy[k]) / d;
                    if (dot > 0) s += dot * dot;
                }
                s *= 255 - b[cy * w + cx];
                if (s > best) {
                    best = s;
                    bx = cx;
                    by = cy;
                }
            }
        }
        // Центр пикселя уменьшенной области → координаты снимка.
        return new double[] {r.x + (bx + 0.5) / scale - 0.5, r.y + (by + 0.5) * r.height / (double) h - 0.5};
    }

    /** 0б: прямоугольник найденного каскадом глаза, ближайшего к p (координаты снимка); null — не найден. */
    static Rect cascadeRect(Mat gray, double[] p, double iod, CascadeClassifier cc) {
        Rect r = square(gray, p, CASCADE_SIDE * iod);
        if (r == null) return null;
        Mat roi = new Mat(gray, r);
        MatOfRect found = new MatOfRect();
        int side = Math.min(r.width, r.height);
        cc.detectMultiScale(roi, found, 1.1, 3, 0, new Size(side / 4.0, side / 4.0), new Size(side, side));
        Rect best = null;
        double bestD = Double.MAX_VALUE;
        for (Rect e : found.toArray()) {
            double d = Math.hypot(r.x + e.x + e.width / 2.0 - p[0], r.y + e.y + e.height / 2.0 - p[1]);
            if (d < bestD) {
                bestD = d;
                best = new Rect(r.x + e.x, r.y + e.y, e.width, e.height);
            }
        }
        found.release();
        return best;
    }

    /** 0б: центр прямоугольника каскада; null — не найден. */
    static double[] cascade(Mat gray, double[] p, double iod, CascadeClassifier cc) {
        Rect e = cascadeRect(gray, p, iod, cc);
        return e == null ? null : new double[] {e.x + e.width / 2.0 - 0.5, e.y + e.height / 2.0 - 0.5};
    }

    /** 0в: каскад, затем градиенты внутри прямоугольника; каскад не нашёл — 0а. */
    static double[] combined(Mat gray, double[] p, double iod, CascadeClassifier cc) {
        Rect e = cascadeRect(gray, p, iod, cc);
        return e == null ? gradient(gray, p, iod) : gradient(gray, e);
    }

    /** Серый CV_8U из цветного BGR. */
    static Mat gray(Mat color) {
        Mat g = new Mat();
        if (color.channels() == 1) color.copyTo(g);
        else Imgproc.cvtColor(color, g, Imgproc.COLOR_BGR2GRAY);
        if (g.depth() != CvType.CV_8U) {
            Mat u = new Mat();
            Core.normalize(g, u, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
            g.release();
            return u;
        }
        return g;
    }
}
