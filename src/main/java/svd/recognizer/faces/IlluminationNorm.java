package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
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
            return tanTriggs(gray8, p, true);
        }
    };

    /** CLAHE (faces.clahe.clip, faces.clahe.tile) по серому кадру 8 бит: результат — 8 бит. */
    static Mat clahe8(Mat gray8, NormParams p) {
        org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(p.claheClip(), new Size(p.claheTile(), p.claheTile()));
        Mat out = new Mat();
        clahe.apply(gray8, out);
        return out;
    }

    /**
     * Нормировки перед LBP из значения ключа faces.lbp.norm (п. 4.2, «Нормировка освещения перед LBP»): список через запятую
     * из none, tt, clahe без учёта регистра, без повторов, в порядке значения; null или пусто — [NONE]; прочее — ошибка.
     */
    static List<IlluminationNorm> lbpNorms(String raw) {
        List<IlluminationNorm> out = new ArrayList<>();
        if (raw == null || raw.isBlank()) return List.of(NONE);
        for (String v : raw.split(",")) {
            String id = v.trim().toLowerCase(Locale.ROOT);
            IlluminationNorm n = null;
            for (IlluminationNorm c : values()) if (c.id.equals(id)) n = c;
            if (n == null) throw new IllegalStateException("faces.lbp.norm: неизвестное значение «" + v.trim() + "» (none, tt, clahe)");
            if (out.contains(n)) throw new IllegalStateException("faces.lbp.norm: повтор «" + id + "»");
            out.add(n);
        }
        return out;
    }

    /** Одна нормировка перед LBP (faces-fei): значение ключа faces.lbp.norm — ровно одно из none, tt, clahe. */
    static IlluminationNorm lbpNorm(String raw) {
        List<IlluminationNorm> l = lbpNorms(raw);
        if (l.size() != 1) throw new IllegalStateException("faces.lbp.norm: в faces-fei — одно значение, задано «" + raw + "»");
        return l.get(0);
    }

    /** Порог ничтожного знаменателя выравнивания контраста Tan – Triggs. */
    static final double NEGLIGIBLE = 1e-10;

    /**
     * Относительный порог ничтожной разности гауссианов Tan – Triggs (решение Хозяина 13.10.2026, №3): |DoG| ≤ DOG_EPS ·
     * max |I^γ| кадра обнуляется перед выравниванием контраста. Причина: в ровных (например, пересвеченных) участках DoG
     * математически равна 0, но после свёрток остаётся шум последних разрядов (около 1e-16), разный у разных сборок OpenCV;
     * степень |x|^α (α = 0,1) превращает его в величину около 0,025 и сдвигает знаменатель выравнивания — весь выход кадра
     * смещался до 4e-3 (FERET, openpnp против bytedeco; TASKS.md, «Переход на bytedeco»). Поправка возвращает реализацию к
     * определению метода; на кадрах без ровных участков результат не меняется.
     */
    static final double DOG_EPS = 1e-12;

    /**
     * Tan – Triggs: гамма-коррекция I^γ (яркости 0..255), разность гауссианов σ₀ − σ₁, при correct — обнуление |DoG| ≤
     * DOG_EPS · max |I^γ|, затем выравнивание контраста (tanTriggsContrast). Вектор кадра построчно.
     */
    static double[] tanTriggs(Mat gray8, NormParams p, boolean correct) {
        Mat x = new Mat();
        gray8.convertTo(x, CvType.CV_64F);
        org.opencv.core.Core.pow(x, p.gamma(), x);
        double maxGamma = org.opencv.core.Core.minMaxLoc(x).maxVal;
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
        if (correct) zeroNegligible(v, DOG_EPS * maxGamma);
        return tanTriggsContrast(v, p.alpha(), p.tau());
    }

    /** Обнуляет |v[i]| ≤ eps; возвращает число обнулённых. */
    static int zeroNegligible(double[] v, double eps) {
        int n = 0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] != 0 && Math.abs(v[i]) <= eps) {
                v[i] = 0;
                n++;
            }
        }
        return n;
    }

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
