package svd.recognizer.faces;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Нормализация освещения кадра лица перед векторизацией (шаг 5, FarMethods): одна на прогон (не каскадом),
 * одинаково для галереи, посторонних и проб.
 *
 * @author ssv
 */
enum IlluminationNorm {

    /** Без нормализации: яркости /255. */
    NONE("без нормализации", "none"),

    /** CLAHE (faces.clahe.clip, faces.clahe.tile), затем яркости /255. */
    CLAHE("CLAHE", "clahe") {
        @Override
        double[] vector(Mat gray8, NormParams p) {
            org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(p.claheClip(), new Size(p.claheTile(), p.claheTile()));
            Mat out = new Mat();
            clahe.apply(gray8, out);
            double[] v = FaceAlignment.toVector(out);
            out.release();
            return v;
        }
    },

    /**
     * Tan – Triggs (2010): гамма-коррекция I^γ (яркости 0..255), разность гауссианов σ₀ − σ₁, выравнивание
     * контраста в два шага (α, τ) и сжатие τ·tanh(I/τ).
     */
    TAN_TRIGGS("Tan – Triggs", "tt") {
        @Override
        double[] vector(Mat gray8, NormParams p) {
            Mat x = new Mat();
            gray8.convertTo(x, CvType.CV_64F);
            org.opencv.core.Core.pow(x, p.gamma(), x);
            Mat b0 = new Mat();
            Mat b1 = new Mat();
            Imgproc.GaussianBlur(x, b0, new Size(0, 0), p.sigma0());
            Imgproc.GaussianBlur(x, b1, new Size(0, 0), p.sigma1());
            org.opencv.core.Core.subtract(b0, b1, x);
            double[] v = new double[(int) x.total()];
            x.get(0, 0, v);
            x.release();
            b0.release();
            b1.release();
            return tanTriggsContrast(v, p.alpha(), p.tau());
        }
    };

    /** Порог ничтожного знаменателя выравнивания контраста Tan – Triggs. */
    static final double NEGLIGIBLE = 1e-10;

    final String label;
    final String id;

    IlluminationNorm(String label, String id) {
        this.label = label;
        this.id = id;
    }

    /** Параметры нормализаций (SettingsStore: faces.clahe.*, faces.tt.*). */
    record NormParams(double claheClip, int claheTile, double gamma, double sigma0, double sigma1, double alpha, double tau) {
        static NormParams of(SettingsStore s) {
            return new NormParams(s.loadFacesClaheClip(), s.loadFacesClaheTile(), s.loadFacesTtGamma(), s.loadFacesTtSigma0(),
                    s.loadFacesTtSigma1(), s.loadFacesTtAlpha(), s.loadFacesTtTau());
        }
    }

    /**
     * @param gray8 серый кадр 8 бит
     * @param p     параметры
     * @return вектор кадра построчно
     */
    double[] vector(Mat gray8, NormParams p) {
        return FaceAlignment.toVector(gray8);
    }

    /**
     * Выравнивание контраста Tan – Triggs: I / mean(|I|^α)^{1/α}, затем I / mean(min(τ, |I|)^α)^{1/α}, затем
     * τ·tanh(I/τ). Ничтожный знаменатель (≤ NEGLIGIBLE: постоянный кадр, DoG — погрешность округления) — шаг
     * пропускается.
     */
    static double[] tanTriggsContrast(double[] v, double alpha, double tau) {
        double m = 0;
        for (double x : v) m += Math.pow(Math.abs(x), alpha);
        m = Math.pow(m / v.length, 1 / alpha);
        if (m > NEGLIGIBLE) for (int i = 0; i < v.length; i++) v[i] /= m;
        m = 0;
        for (double x : v) m += Math.pow(Math.min(tau, Math.abs(x)), alpha);
        m = Math.pow(m / v.length, 1 / alpha);
        if (m > NEGLIGIBLE) for (int i = 0; i < v.length; i++) v[i] /= m;
        for (int i = 0; i < v.length; i++) v[i] = tau * Math.tanh(v[i] / tau);
        return v;
    }
}
