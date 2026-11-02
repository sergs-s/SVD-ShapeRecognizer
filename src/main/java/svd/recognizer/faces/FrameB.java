package svd.recognizer.faces;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Кадры Б – итоговый прогон перед ПО (решение Хозяина 24.10.2026 (25); постановка итогового прогона, разд. 1). Кадр 92×112 – способ Б
 * ревизии 1.12 (Ахмед, tools/verify VerifyRev.frame, проверен 23.10): подобие по 5 точкам YuNet полного снимка (LMEDS, как кадр «а»),
 * warpAffine в кадр k·(92×112), k = ⌈1 / масштаб подобия⌉, матрица k·M со сдвигом (k − 1)/2 (центры блоков k×k совпадают с центрами
 * пикселей кадра), затем INTER_AREA до 92×112; при k = 1 кадр Б совпадает с кадром «а». Кадр k = 2 (184×224) – прежний кадр «в»
 * (FrameK2.fromFull, уже с передискретизацией при уменьшении) – не меняется.
 * <p>
 * Переключатель FACES_FRAMES = b: кадр «а» читается из каталога a_b (FeretData.frame, FarExport.Loaded, VidTimitData.png), каталог
 * отчётов FERET – eq_b (XBaseEval.eqDir); без ключа всё прежнее.
 *
 * @author ssv
 */
final class FrameB {

    static final String DIR = "a_b";
    static final int W = 92;
    static final int H = 112;

    private FrameB() {
    }

    /** Включены ли кадры Б (FACES_FRAMES = b). */
    static boolean on() {
        String s = System.getenv("FACES_FRAMES");
        if (s == null || s.isBlank() || s.equals("a")) return false;
        if (s.equals("b")) return true;
        throw new IllegalStateException("FACES_FRAMES: a или b, получено " + s);
    }

    /** Каталог кадра «а» с учётом переключателя: "a" → a_b при FACES_FRAMES = b; прочие варианты – как есть. */
    static String dir(String variant) {
        return variant.equals("a") && on() ? DIR : variant;
    }

    static double scale(Mat m) {
        return Math.hypot(m.get(0, 0)[0], m.get(1, 0)[0]);
    }

    static int k(Mat m) {
        return Math.max(1, (int) Math.ceil(1.0 / scale(m) - 1e-9));
    }

    /** Кадр 92×112 серый по полному цветному снимку и строке YuNet в его координатах: smooth – Б, иначе – как кадр «а». */
    static Mat frame(Mat color, double[] row, boolean smooth) {
        double[][] lm = FrameK2.lm(row);
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat m = FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(W, H));
        Mat a = new Mat();
        int k = smooth ? k(m) : 1;
        if (k == 1) {
            Imgproc.warpAffine(gray, a, m, new Size(W, H), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        } else {
            Mat mk = new Mat(2, 3, CvType.CV_64F);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) mk.put(i, j, k * m.get(i, j)[0] + (j == 2 ? (k - 1) / 2.0 : 0));
            }
            Mat big = new Mat();
            Imgproc.warpAffine(gray, big, mk, new Size(W * k, H * k), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
            Imgproc.resize(big, a, new Size(W, H), 0, 0, Imgproc.INTER_AREA);
            big.release();
            mk.release();
        }
        gray.release();
        m.release();
        return a;
    }

    /** SHA-256 файла, hex. */
    static String sha256(Path f) throws IOException {
        try (InputStream in = Files.newInputStream(f)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0;) md.update(buf, 0, n);
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    static String sha256(String s) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Опись каталога: относительный путь (через /) → SHA-256, по порядку путей. */
    static Map<String, String> manifest(Path root) throws IOException {
        Map<String, String> m = new TreeMap<>();
        if (!Files.exists(root)) return m;
        List<Path> files;
        try (Stream<Path> s = Files.walk(root)) {
            files = s.filter(Files::isRegularFile).toList();
        }
        for (Path f : files) m.put(root.relativize(f).toString().replace('\\', '/'), sha256(f));
        return m;
    }

    /** Текст описи (путь, табуляция, SHA-256) и его SHA-256 – одна строка на файл, LF. */
    static String manifestText(Map<String, String> m) {
        StringBuilder b = new StringBuilder();
        m.forEach((k, v) -> b.append(k).append('\t').append(v).append('\n'));
        return b.toString();
    }

    static String fmt(String base, int n, String sha) {
        return String.format(Locale.ROOT, "%s\t%d\t%s%n", base, n, sha);
    }
}
