package svd.recognizer.faces;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Ретинальный фильтр Vu (N.-S. Vu, «Towards unconstrained face recognition from one sample», дисс., Гренобль, 2010, гл. 3,
 * разд. 3.3, уравнения 3.3.1–3.3.10; пачка 4, строка Б1′ – задание Жорика №4 от 01.11.2026). Шаги: 1) две адаптивные
 * логарифмические компрессии: F₁(p) = (I ∗ G_σ₁)(p) + Ī/2, I₁(p) = (max I + F₁(p)) · I(p) / (I(p) + F₁(p)); вторая – так же
 * по I₁ с σ₂; 2) DoG: I_bip = G_σPh ∗ I₂ − G_σH ∗ I₂; 3) нормировка I_nor = I_bip / √E(I_bip²) (среднее не вычитается, как у
 * автора); 4) усечение sign(I_nor) · min(Th, |I_nor|) – в уравнении 3.3.10 напечатано max, но по тексту («подавить 2–4 %
 * крайних значений») это усечение. Ядра Гаусса – окно [−3σ, 3σ] (как у автора), край – BORDER_REFLECT_101 (как у Тан –
 * Триггс). Параметры автора (разд. 3.4.1.3): σ₁ = σ₂ = 1, σ_Ph = 0,5, σ_H = 3, Th = 3,5; кадры автора – Yale B 192×168.
 * |DoG| ≤ IlluminationNorm.DOG_EPS · max |I₂| обнуляется (как у Тан – Триггс: шум округления ровных участков).
 *
 * @author ssv
 */
final class RetinaFilter {

    /** Параметры: σ компрессий, σ_Ph, σ_H, порог усечения. */
    record Params(double sigma1, double sigma2, double sigmaPh, double sigmaH, double th) {
        /** σ × k (кадр k = 2 MS-LBP); порог – прежний. */
        Params scaled(double k) {
            return new Params(sigma1 * k, sigma2 * k, sigmaPh * k, sigmaH * k, th);
        }
    }

    /** Параметры автора (разд. 3.4.1.3). */
    static final Params AUTHOR = new Params(1, 1, 0.5, 3, 3.5);

    private RetinaFilter() {
    }

    /** Гаусс с окном [−3σ, 3σ]. */
    static Mat blur(Mat x, double sigma) {
        int r = (int) Math.ceil(3 * sigma);
        Mat out = new Mat();
        Imgproc.GaussianBlur(x, out, new Size(2 * r + 1, 2 * r + 1), sigma, sigma, Core.BORDER_REFLECT_101);
        return out;
    }

    /** Адаптивная компрессия: (max + F) · I / (I + F), F = I ∗ G_σ + Ī/2; I + F = 0 (только у нулевого пикселя нулевого кадра) – 0. */
    static double[] compress(double[] img, int w, int h, double sigma) {
        double mean = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : img) {
            mean += v;
            max = Math.max(max, v);
        }
        mean /= img.length;
        Mat x = new Mat(h, w, CvType.CV_64F);
        x.put(0, 0, img);
        Mat b = blur(x, sigma);
        double[] f = new double[img.length];
        b.get(0, 0, f);
        x.release();
        b.release();
        double[] out = new double[img.length];
        for (int i = 0; i < img.length; i++) {
            double fi = f[i] + mean / 2;
            double den = img[i] + fi;
            out[i] = den == 0 ? 0 : (max + fi) * img[i] / den;
        }
        return out;
    }

    /** Фильтр по кадру img (w × h, по строкам, яркости ≥ 0). */
    static double[] filter(double[] img, int w, int h, Params p) {
        double[] i2 = compress(compress(img, w, h, p.sigma1()), w, h, p.sigma2());
        double maxAbs = 0;
        for (double v : i2) maxAbs = Math.max(maxAbs, Math.abs(v));
        Mat x = new Mat(h, w, CvType.CV_64F);
        x.put(0, 0, i2);
        Mat ph = blur(x, p.sigmaPh());
        Mat hz = blur(x, p.sigmaH());
        Core.subtract(ph, hz, x);
        double[] v = new double[img.length];
        x.get(0, 0, v);
        x.release();
        ph.release();
        hz.release();
        IlluminationNorm.zeroNegligible(v, IlluminationNorm.DOG_EPS * maxAbs);
        double e = 0;
        for (double a : v) e += a * a;
        e = Math.sqrt(e / v.length);
        if (!(e > 0)) return new double[v.length];
        for (int i = 0; i < v.length; i++) {
            double n = v[i] / e;
            v[i] = Math.signum(n) * Math.min(p.th(), Math.abs(n));
        }
        return v;
    }

    /** Фильтр по серому кадру 8 бит (яркости 0..255). */
    static double[] filter(Mat gray8, Params p) {
        if (gray8.type() != CvType.CV_8UC1) throw new IllegalArgumentException("RetinaFilter: кадр не 8 бит одноканальный");
        byte[] b = new byte[gray8.cols() * gray8.rows()];
        gray8.get(0, 0, b);
        double[] img = new double[b.length];
        for (int i = 0; i < b.length; i++) img[i] = b[i] & 0xFF;
        return filter(img, gray8.cols(), gray8.rows(), p);
    }

    /** Доля усечённых значений (|I_nor| > Th) – для сверки с «2–4 %» автора. */
    static double clippedShare(double[] out, Params p) {
        int n = 0;
        for (double v : out) if (Math.abs(v) >= p.th()) n++;
        return (double) n / out.length;
    }
}
