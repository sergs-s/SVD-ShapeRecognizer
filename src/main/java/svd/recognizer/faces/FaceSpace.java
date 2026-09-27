package svd.recognizer.faces;

import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Подпространство образов фиксированного размера (лица 92×112, глаза) для
 * поиска по расстоянию до подпространства (DFFS, Turk и Pentland, 1991).
 *
 * Каждое окно нормируется: вычитается среднее окна, делится на норму
 * (‖x̂‖ = 1) — поиск не зависит от яркости и контраста окна. Модель:
 * среднее μ нормированных образцов и первые k левых сингулярных векторов
 * центрированной матрицы; k — по доле энергии (Σσᵢ² первых k ≥ η), не больше
 * kMax и N − 1.
 *
 * DFFS окна: ε² = ‖x̂ − μ‖² − Σⱼ⟨x̂ − μ, eⱼ⟩². Карта по всем положениям окна
 * считается корреляциями (Imgproc.filter2D, при больших ядрах — через ДПФ)
 * и суммами по окну (boxFilter): ⟨x̂, v⟩ = (corr(x, v) − m·Σv)/σ.
 *
 * @author ssv
 */
public final class FaceSpace {

    final int width;
    final int height;
    final double[] mean;
    /** basis[j] — j-я компонента (длина width·height, построчно). */
    final double[][] basis;
    final double energy;

    private FaceSpace(int width, int height, double[] mean, double[][] basis, double energy) {
        this.width = width;
        this.height = height;
        this.mean = mean;
        this.basis = basis;
        this.energy = energy;
    }

    public int k() {
        return basis.length;
    }

    /** Нормированный вектор окна: минус среднее, делённое на норму; null для плоского окна. */
    static double[] normalize(Mat patch8) {
        int n = (int) patch8.total();
        double[] x = new double[n];
        Mat d = new Mat();
        patch8.convertTo(d, CvType.CV_64F);
        d.get(0, 0, x);
        double m = 0;
        for (double v : x) m += v;
        m /= n;
        double ss = 0;
        for (int i = 0; i < n; i++) {
            x[i] -= m;
            ss += x[i] * x[i];
        }
        if (ss < 1e-9) {
            return null;
        }
        double s = Math.sqrt(ss);
        for (int i = 0; i < n; i++) x[i] /= s;
        return x;
    }

    /** Обучение по образцам одного размера (8 бит, 1 канал). */
    static FaceSpace train(List<Mat> patches, double eta, int kMax) {
        int w = patches.get(0).cols();
        int h = patches.get(0).rows();
        int n = w * h;
        double[][] rows = patches.stream().map(FaceSpace::normalize).filter(v -> v != null).toArray(double[][]::new);
        int count = rows.length;
        double[] mean = new double[n];
        for (double[] r : rows) for (int i = 0; i < n; i++) mean[i] += r[i] / count;
        double[][] centered = new double[n][count];
        for (int c = 0; c < count; c++) for (int i = 0; i < n; i++) centered[i][c] = rows[c][i] - mean[i];
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(centered, false));
        double[] sigma = svd.getSingularValues();
        double total = 0;
        for (double s : sigma) total += s * s;
        int k = 0;
        double acc = 0;
        while (k < Math.min(kMax, count - 1) && acc < eta * total) {
            acc += sigma[k] * sigma[k];
            k++;
        }
        RealMatrix u = svd.getU();
        double[][] basis = new double[k][];
        for (int j = 0; j < k; j++) basis[j] = u.getColumn(j);
        return new FaceSpace(w, h, mean, basis, acc / total);
    }

    /** DFFS одного окна (8 бит); плоское окно — +∞. */
    double dffs(Mat patch8) {
        double[] x = normalize(patch8);
        if (x == null) return Double.POSITIVE_INFINITY;
        double ss = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] -= mean[i];
            ss += x[i] * x[i];
        }
        for (double[] e : basis) {
            double p = 0;
            for (int i = 0; i < x.length; i++) p += x[i] * e[i];
            ss -= p * p;
        }
        return ss;
    }

    /**
     * Карта DFFS по всем положениям окна в изображении (8 бит, 1 канал):
     * элемент (y, x) — окно с левым верхним углом (x, y); размер
     * (rows − height + 1) × (cols − width + 1), CV_64F. Плоские окна — +∞.
     */
    Mat map(Mat gray8) {
        Mat img = new Mat();
        gray8.convertTo(img, CvType.CV_64F);
        int outW = img.cols() - width + 1;
        int outH = img.rows() - height + 1;
        double n = width * height;
        Point anchor = new Point(0, 0);
        Mat sum = new Mat();
        Imgproc.boxFilter(img, sum, CvType.CV_64F, new Size(width, height), anchor, false, Core.BORDER_CONSTANT);
        Mat sq = new Mat();
        Core.multiply(img, img, sq);
        Mat sumSq = new Mat();
        Imgproc.boxFilter(sq, sumSq, CvType.CV_64F, new Size(width, height), anchor, false, Core.BORDER_CONSTANT);
        double[] s1 = new double[outW * outH];
        double[] s2 = new double[outW * outH];
        sum.submat(0, outH, 0, outW).clone().get(0, 0, s1);
        sumSq.submat(0, outH, 0, outW).clone().get(0, 0, s2);
        double[] m = new double[s1.length];
        double[] sd = new double[s1.length];
        for (int i = 0; i < s1.length; i++) {
            m[i] = s1[i] / n;
            double var = s2[i] - n * m[i] * m[i];
            sd[i] = var > 1e-6 ? Math.sqrt(var) : 0;
        }
        // ‖x̂ − μ‖² = 1 − 2⟨x̂, μ⟩ + ‖μ‖²; ⟨x̂, eⱼ⟩ − ⟨μ, eⱼ⟩ — проекция на компоненту.
        double[] res = new double[s1.length];
        double muNorm = dot(mean, mean);
        double[] corrMu = correlate(img, mean, outW, outH);
        double muSum = sum(mean);
        for (int i = 0; i < res.length; i++) {
            res[i] = sd[i] == 0 ? Double.POSITIVE_INFINITY : 1 - 2 * (corrMu[i] - m[i] * muSum) / sd[i] + muNorm;
        }
        for (double[] e : basis) {
            double[] c = correlate(img, e, outW, outH);
            double eSum = sum(e);
            double muE = dot(mean, e);
            for (int i = 0; i < res.length; i++) {
                if (sd[i] == 0) continue;
                double p = (c[i] - m[i] * eSum) / sd[i] - muE;
                res[i] -= p * p;
            }
        }
        Mat out = new Mat(outH, outW, CvType.CV_64F);
        out.put(0, 0, res);
        return out;
    }

    private double[] correlate(Mat img, double[] v, int outW, int outH) {
        Mat kernel = new Mat(height, width, CvType.CV_64F);
        kernel.put(0, 0, v);
        Mat c = new Mat();
        Imgproc.filter2D(img, c, CvType.CV_64F, kernel, new Point(0, 0), 0, Core.BORDER_CONSTANT);
        double[] r = new double[outW * outH];
        c.submat(0, outH, 0, outW).clone().get(0, 0, r);
        return r;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static double sum(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s;
    }
}
