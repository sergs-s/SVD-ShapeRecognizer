package svd.recognizer;

/**
 * Загрузка нативной OpenCV (org.bytedeco:opencv 4.9.0-1.5.10, Java-API org.opencv.* с модулями contrib, в том числе face —
 * Facemark LBF; решение Хозяина 12.10.2026 о переходе с org.openpnp:opencv). Повторный вызов ничего не делает.
 *
 * @author ssv
 */
public final class OpenCvLoader {

    private static boolean loaded;

    private OpenCvLoader() {
    }

    public static synchronized void load() {
        if (loaded) return;
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
        loaded = true;
    }
}
