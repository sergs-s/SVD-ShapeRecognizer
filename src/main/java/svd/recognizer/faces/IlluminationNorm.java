package svd.recognizer.faces;

import org.opencv.core.Mat;
import svd.recognizer.storage.SettingsStore;

/**
 * Нормализация освещения кадра лица перед векторизацией (шаг 5, FarMethods): одна на прогон,
 * одинаково для галереи, посторонних и проб.
 *
 * @author ssv
 */
enum IlluminationNorm {

    /** Без нормализации: яркости /255. */
    NONE("без нормализации", "none");

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
}
