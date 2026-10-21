package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.storage.SettingsStore;

/**
 * Кадр метода Прокаева (Д-№1, п. 2): вырезка 2:3 по рамке YuNet, высота H = h_рамки / {@value #BOX_SHARE} (рамка – около 87,5 %
 * высоты кадра; одно значение на все базы), ширина 2H/3, центр – центр рамки, без поворота; выход за кадр – сдвиг внутрь, не влезает
 * – наибольшая вырезка 2:3 в пределах снимка (считается); серый (COLOR_BGR2GRAY), затем 400 × 600 (INTER_AREA при уменьшении,
 * INTER_LINEAR при увеличении). Строка пикселей – float, без нормировки.
 * <p>
 * Режим prokaev-sheet – лист 24 кадров GT (reports/faces/xbase/prokaev/gt_frames.png) и 24 кадров FERET (только
 * D:\data\Feret\reports\eq\prokaev\feret_frames.png) для проверки глазами (подгонка значения – только по листу).
 *
 * @author ssv
 */
final class ProkaevFrames {

    /**
     * Доля рамки в высоте кадра К-лицо: по умолчанию 0,70 (решение Хозяина 20.10.2026, Д9; до того – 0,875, теперь справочно);
     * FACES_PROKAEV_BOX – другое значение.
     */
    static final double BOX_SHARE = System.getenv("FACES_PROKAEV_BOX") == null ? 0.70 : Double.parseDouble(System.getenv("FACES_PROKAEV_BOX"));
    /**
     * Комплект кадров (Д10, Д-№3): face – К-лицо (по умолчанию), plain – К-обычный, plain-t1 – К-обычный «только лицо» (вне эллипса лица –
     * ровный серый), plain-t2 – К-обычный «лицо закрашено» (внутри эллипса – ровный серый); ключ FACES_PROKAEV_FRAME.
     */
    static final String FRAME = System.getenv("FACES_PROKAEV_FRAME") == null ? "face" : System.getenv("FACES_PROKAEV_FRAME");
    static final boolean PLAIN = FRAME.startsWith("plain");
    /** Маска Д-№3: 0 – нет, 1 – Т1 «только лицо», 2 – Т2 «лицо закрашено». */
    static final int MASK = FRAME.equals("plain-t1") ? 1 : FRAME.equals("plain-t2") ? 2 : 0;
    /** Эллипс лица Д-№3: вписан в рамку YuNet, расширенную на 10 % по каждой оси. */
    static final double ELLIPSE_GROW = 1.10;
    static final int W = 400;
    static final int H = 600;

    private ProkaevFrames() {
    }

    /** Добавка к именам отчётов, оценок и контрольных точек: К-лицо 0,875 – без добавки (прежние имена), иначе «_face0.700», «_plain». */
    static String tag() {
        // Только основная строка (FACES_PROKAEV_ONLY=main) – добавка «_main», чтобы не затирать полные отчёты того же комплекта.
        String only = "main".equals(System.getenv("FACES_PROKAEV_ONLY")) ? "_main" : "";
        if (PLAIN) return (MASK == 0 ? "_plain" : "_plain_t" + MASK) + only;
        return (BOX_SHARE == 0.875 ? "" : String.format(Locale.ROOT, "_face%.3f", BOX_SHARE)) + only;
    }

    /** Добавка к имени листа: «plain» или доля рамки («0.700», «0.875» – прежние имена). */
    static String sheetTag() {
        return PLAIN ? (MASK == 0 ? "plain" : "plain_t" + MASK) : String.format(Locale.ROOT, "%.3f", BOX_SHARE);
    }

    /** Описание кадра для шапок отчётов и отпечатка контрольной точки. */
    static String describe(int clipped) {
        if (PLAIN) return String.format(Locale.ROOT, "К-обычный – наибольшее окно 2:3 в снимке (на всю высоту, а у снимка уже 2:3 – на всю ширину), по "
                + "центру рамки YuNet со сдвигом внутрь у края, 400 × 600, серый; рамка не влезла по ширине окна – %d", clipped) + maskText();
        return String.format(Locale.ROOT, "вырезка 2:3 по рамке YuNet, H = h_рамки / %.3f, 400 × 600, серый; не влезло в снимок – %d", BOX_SHARE, clipped);
    }

    /** Кадр 400 × 600 (серый) по цветному снимку и строке YuNet; clipped[0] увеличивается, если вырезка не влезла в снимок. */
    static Mat crop(Mat color, double[] row, int[] clipped) {
        Mat gray = new Mat();
        if (color.channels() == 1) color.copyTo(gray);
        else Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        if (PLAIN) return plain(gray, row, clipped, MASK);
        double hh = row[3] / BOX_SHARE;
        double ww = hh * W / H;
        if (ww > gray.cols() || hh > gray.rows()) {
            double f = Math.min(gray.cols() / ww, gray.rows() / hh);
            ww *= f;
            hh *= f;
            if (clipped != null) clipped[0]++;
        }
        int w = (int) Math.min(gray.cols(), Math.round(ww));
        int h = (int) Math.min(gray.rows(), Math.round(hh));
        int x0 = (int) Math.round(row[0] + row[2] / 2 - w / 2.0);
        int y0 = (int) Math.round(row[1] + row[3] / 2 - h / 2.0);
        x0 = Math.max(0, Math.min(x0, gray.cols() - w));
        y0 = Math.max(0, Math.min(y0, gray.rows() - h));
        Mat out = new Mat();
        Imgproc.resize(gray.submat(new Rect(x0, y0, w, h)), out, new Size(W, H), 0, 0, w > W ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        gray.release();
        return out;
    }

    /**
     * К-обычный (Д10): наибольшее окно 2:3 – на всю высоту снимка (ширина 2·h/3), если снимок шире 2:3; иначе на всю ширину (высота
     * 3·w/2). Центр окна – центр рамки YuNet, у края – сдвиг внутрь; clipped[0] – рамка шире окна. Затем 400 × 600, как К-лицо.
     */
    static Mat plain(Mat gray, double[] row, int[] clipped, int mask) {
        int w;
        int h;
        if (gray.cols() * 3L >= gray.rows() * 2L) {
            h = gray.rows();
            w = (int) Math.min(gray.cols(), Math.round(h * 2.0 / 3));
        } else {
            w = gray.cols();
            h = (int) Math.min(gray.rows(), Math.round(w * 1.5));
        }
        if (clipped != null && row[2] > w) clipped[0]++;
        int x0 = (int) Math.round(row[0] + row[2] / 2 - w / 2.0);
        int y0 = (int) Math.round(row[1] + row[3] / 2 - h / 2.0);
        x0 = Math.max(0, Math.min(x0, gray.cols() - w));
        y0 = Math.max(0, Math.min(y0, gray.rows() - h));
        Mat out = new Mat();
        Imgproc.resize(gray.submat(new Rect(x0, y0, w, h)), out, new Size(W, H), 0, 0, w > W ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        gray.release();
        if (mask != 0) mask(out, row, x0, y0, (double) W / w, (double) H / h, mask);
        return out;
    }

    static String maskText() {
        return MASK == 0 ? "" : String.format(Locale.ROOT, "; Д-№3 %s: эллипс, вписанный в рамку YuNet, расширенную в %.2f раза по каждой оси; %s – ровный "
                + "серый (среднее кадра, округлённое)", MASK == 1 ? "Т1 «только лицо»" : "Т2 «лицо закрашено»", ELLIPSE_GROW, MASK == 1 ? "вне эллипса" : "внутри эллипса");
    }

    /**
     * Маска Д-№3 на кадре 400 × 600: эллипс – центр рамки YuNet, полуоси – половины ширины и высоты рамки × ELLIPSE_GROW (в координатах кадра:
     * окно x0, y0, масштаб sx, sy); заливка – среднее кадра до маски (округлённое): mask 1 – вне эллипса, 2 – внутри.
     */
    static void mask(Mat frame, double[] row, int x0, int y0, double sx, double sy, int mask) {
        double mean = Math.round(Core.mean(frame).val[0]);
        Mat m = Mat.zeros(frame.size(), CvType.CV_8UC1);
        Point c = new Point((row[0] + row[2] / 2 - x0) * sx, (row[1] + row[3] / 2 - y0) * sy);
        Size ax = new Size(row[2] * ELLIPSE_GROW / 2 * sx, row[3] * ELLIPSE_GROW / 2 * sy);
        Imgproc.ellipse(m, c, ax, 0, 0, 360, new Scalar(255), -1);
        if (mask == 1) Core.bitwise_not(m, m);
        frame.setTo(new Scalar(mean), m);
        m.release();
    }

    /** Строка пикселей кадра (float, без нормировки). */
    static float[] row(Mat frame) {
        Mat f = new Mat();
        frame.convertTo(f, CvType.CV_32F);
        float[] v = new float[(int) f.total()];
        f.get(0, 0, v);
        f.release();
        return v;
    }

    // ---------------------------------------------------------------- лист

    static void sheet(Path exportDir, SettingsStore settings, Path out) throws IOException {
        long seed = settings.loadFacesSeed();
        int[] clipped = new int[1];
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> gt = XBaseMpPts.exportRows(exportDir, "gt");
        List<String> gk = new ArrayList<>(gt.keySet().stream().filter(k -> gt.get(k) != null).sorted().toList());
        Collections.shuffle(gk, new Random(seed + 9976));
        List<Mat> tiles = new ArrayList<>();
        for (String k : gk.subList(0, 24)) {
            Mat c = Imgcodecs.imread(gtDir.resolve(k + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            tiles.add(tile(crop(c, gt.get(k), clipped), k));
            c.release();
        }
        Path g = out.resolve("prokaev").resolve("gt_frames_" + sheetTag() + ".png");
        write(g, tiles);
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("prokaev-sheet: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        List<Img> imgs = FeretData.frontal(dir);
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), ""));
        List<Img> fk = new ArrayList<>(imgs.stream().filter(im -> det.get(im.key()) != null).toList());
        Collections.shuffle(fk, new Random(seed + 9977));
        tiles = new ArrayList<>();
        for (Img im : fk.subList(0, 24)) {
            Mat c = FeretData.decode(im.file());
            tiles.add(tile(crop(c, det.get(im.key()), clipped), im.key()));
            c.release();
        }
        Path f = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve("prokaev").resolve("feret_frames_" + sheetTag() + ".png");
        write(f, tiles);
        System.out.printf(Locale.ROOT, "prokaev-sheet: %s, %s; %s (из 48)%n", g, f, describe(clipped[0]));
    }

    /**
     * Д-№3: лист 12 снимков, у каждого рядом Т1 и Т2 (К-обычный с маской): GT – reports/faces/xbase/prokaev/gt_frames_t1_t2.png, FERET – только
     * D:\data\Feret\reports\eq\prokaev\feret_frames_t1_t2.png; снимки – первые 12 той же перестановки, что у листа prokaev-sheet.
     */
    static void sheetMask(Path exportDir, SettingsStore settings, Path out) throws IOException {
        long seed = settings.loadFacesSeed();
        int[] clipped = new int[1];
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> gt = XBaseMpPts.exportRows(exportDir, "gt");
        List<String> gk = new ArrayList<>(gt.keySet().stream().filter(k -> gt.get(k) != null).sorted().toList());
        Collections.shuffle(gk, new Random(seed + 9976));
        List<Mat> tiles = new ArrayList<>();
        for (String k : gk.subList(0, 12)) {
            for (int m = 1; m <= 2; m++) {
                Mat c = Imgcodecs.imread(gtDir.resolve(k + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
                Mat gray = new Mat();
                Imgproc.cvtColor(c, gray, Imgproc.COLOR_BGR2GRAY);
                c.release();
                tiles.add(tile(plain(gray, gt.get(k), clipped, m), "T" + m + " " + k));
            }
        }
        Path g = out.resolve("prokaev").resolve("gt_frames_t1_t2.png");
        write(g, tiles);
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("prokaev-sheet-mask: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        List<Img> imgs = FeretData.frontal(dir);
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), ""));
        List<Img> fk = new ArrayList<>(imgs.stream().filter(im -> det.get(im.key()) != null).toList());
        Collections.shuffle(fk, new Random(seed + 9977));
        tiles = new ArrayList<>();
        for (Img im : fk.subList(0, 12)) {
            for (int m = 1; m <= 2; m++) {
                Mat c = FeretData.decode(im.file());
                Mat gray = new Mat();
                Imgproc.cvtColor(c, gray, Imgproc.COLOR_BGR2GRAY);
                c.release();
                tiles.add(tile(plain(gray, det.get(im.key()), clipped, m), "T" + m + " " + im.key()));
            }
        }
        Path f = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve("prokaev").resolve("feret_frames_t1_t2.png");
        write(f, tiles);
        System.out.printf(Locale.ROOT, "prokaev-sheet-mask: %s, %s%n", g, f);
    }

    static Mat tile(Mat frame, String caption) {
        Mat t = new Mat();
        Imgproc.resize(frame, t, new Size(W / 2.0, H / 2.0), 0, 0, Imgproc.INTER_AREA);
        Imgproc.cvtColor(t, t, Imgproc.COLOR_GRAY2BGR);
        Mat big = new Mat(t.rows() + 14, t.cols(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        t.copyTo(big.submat(new Rect(0, 0, t.cols(), t.rows())));
        Imgproc.putText(big, caption.replaceAll("[^\\x20-\\x7E]", "?"), new Point(2, t.rows() + 11), Imgproc.FONT_HERSHEY_PLAIN, 0.8, new Scalar(0, 0, 0), 1);
        frame.release();
        t.release();
        return big;
    }

    static void write(Path f, List<Mat> tiles) throws IOException {
        List<Mat> rows = new ArrayList<>();
        for (int i = 0; i < tiles.size(); i += 8) {
            Mat r = new Mat();
            Core.hconcat(tiles.subList(i, Math.min(tiles.size(), i + 8)), r);
            rows.add(r);
        }
        Mat all = new Mat();
        Core.vconcat(rows, all);
        Files.createDirectories(f.getParent());
        FeretData.writePng(f, all);
    }
}
