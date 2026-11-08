package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Проба «кадр с подбородком и овальная маска» (постановка Жорика 08.11, SHA-256 fb7aaa58…). chin-sheet – п. 2.1: положения глаз, рта и низа
 * кадра (Ч) в шаблоне выравнивания и лист 10 людей GT (база, (Ч), (Ч+О)); вывод – только D:\data\GeorgiaTech\chin\.
 *
 * @author ssv
 */
final class XBaseChin {

    private XBaseChin() {
    }

    /** Множитель удлинения: низ кадра на 1,0 · (y_рта − y_глаз) ниже линии рта шаблона. */
    static final double CHIN = 1.0;

    /** Высота кадра (Ч): низ = y_рта + CHIN · (y_рта − y_глаз) по шаблону кадра Б, округление до ближайшего чётного. */
    static int height() {
        double[][] t = FaceAlignment.ownTemplate(FrameB.W, FrameB.H);
        double ye = (t[0][1] + t[1][1]) / 2;
        double ym = (t[3][1] + t[4][1]) / 2;
        return 2 * (int) Math.round((ym + CHIN * (ym - ye)) / 2);
    }

    /** Маска (Ч+О): эллипс с центром посередине кадра, полуоси – половины ширины и высоты; 255 – внутри. */
    static Mat mask(int w, int h) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC1);
        Imgproc.ellipse(m, new Point((w - 1) / 2.0, (h - 1) / 2.0), new Size(w / 2.0, h / 2.0), 0, 0, 360, new Scalar(255), -1);
        return m;
    }

    /** Кадр Б высоты h: то же подобие на шаблон 92×112 и то же сглаживание, что FrameB.frame, выход – 92×h. */
    static Mat frame(Mat color, double[] row, int h) {
        double[][] lm = FrameK2.lm(row);
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat m = FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(FrameB.W, FrameB.H));
        Mat a = new Mat();
        int k = FrameB.k(m);
        if (k == 1) {
            Imgproc.warpAffine(gray, a, m, new Size(FrameB.W, h), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        } else {
            Mat mk = new Mat(2, 3, CvType.CV_64F);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) mk.put(i, j, k * m.get(i, j)[0] + (j == 2 ? (k - 1) / 2.0 : 0));
            Mat big = new Mat();
            Imgproc.warpAffine(gray, big, mk, new Size(FrameB.W * k, h * k), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
            Imgproc.resize(big, a, new Size(FrameB.W, h), 0, 0, Imgproc.INTER_AREA);
            big.release();
            mk.release();
        }
        gray.release();
        m.release();
        return a;
    }

    static void sheet(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("chin-sheet: FACES_FRAMES=b");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Path od = gtDir.getParent().resolve("chin");
        Files.createDirectories(od);
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        Map<String, double[]> rows = FrameK2.exportRows(dir);
        double[][] t = FaceAlignment.ownTemplate(FrameB.W, FrameB.H);
        double ye = (t[0][1] + t[1][1]) / 2;
        double ym = (t[3][1] + t[4][1]) / 2;
        int h = height();
        StringBuilder r = new StringBuilder("Кадр с подбородком (постановка 08.11, п. 2.1) – геометрия и лист (XBaseEval chin-sheet)\nКод: коммит " + FarMethods.commit() + "\n\n");
        r.append(String.format(Locale.ROOT, "Шаблон выравнивания кадра Б (ArcFace 112, масштаб %.4f, сдвиг x %.2f), кадр %d×%d, px (x, y):%n", FrameB.H / 112.0, (FrameB.W - 112.0 * FrameB.H / 112) / 2, FrameB.W, FrameB.H));
        String[] nm = {"глаз (слева на кадре)", "глаз (справа на кадре)", "нос", "рот (угол слева)", "рот (угол справа)"};
        for (int i = 0; i < 5; i++) r.append(String.format(Locale.ROOT, "  %s: %.2f, %.2f%n", nm[i], t[i][0], t[i][1]));
        r.append(String.format(Locale.ROOT, "Линия глаз y_глаз = %.3f; линия рта y_рта = %.3f; y_рта − y_глаз = %.3f; низ базы – %d (%.3f под линией рта).%n", ye, ym, ym - ye, FrameB.H, FrameB.H - ym));
        r.append(String.format(Locale.ROOT, "(Ч): низ = y_рта + 1,0 · (y_рта − y_глаз) = %.3f → высота кадра %d (ближайшее чётное); кадр %d×%d, верх и бока – как в базе, масштаб тот же.%n", ym + CHIN * (ym - ye), h, FrameB.W, h));
        r.append(String.format(Locale.ROOT, "(Ч+О): эллипс – центр (%.1f; %.1f), полуоси %.1f и %.1f px; вне эллипса пикселей %d из %d (%.1f %%).%n", (FrameB.W - 1) / 2.0, (h - 1) / 2.0, FrameB.W / 2.0, h / 2.0,
                FrameB.W * h - Core.countNonZero(mask(FrameB.W, h)), FrameB.W * h, 100.0 * (FrameB.W * h - Core.countNonZero(mask(FrameB.W, h))) / (FrameB.W * h)));
        r.append(String.format(Locale.ROOT, "Ячейки: L1-tt-xs-F 9×9 – %.2f × %.2f px (база %.2f × %.2f); L1-xs-F 7×7 – %.2f × %.2f (база %.2f × %.2f).%n%n", FrameB.W / 9.0, h / 9.0, FrameB.W / 9.0, FrameB.H / 9.0,
                FrameB.W / 7.0, h / 7.0, FrameB.W / 7.0, FrameB.H / 7.0));
        // Лист: 10 первых своих GT (по возрастанию), первая межсессионная проба.
        List<String> own = new ArrayList<>(g.own);
        java.util.Collections.sort(own);
        own = own.subList(0, Math.min(10, own.size()));
        int sc = 3;
        int cw = FrameB.W * sc, chh = h * sc, gap = 12, lab = 150, head = 50, cap = 26;
        int W = lab + 3 * (cw + gap) + gap, H = head + own.size() * (chh + cap + gap) + gap;
        Mat sheet = new Mat(H, W, CvType.CV_8UC1, new Scalar(255));
        Imgproc.putText(sheet, "GT, 10 persons, cross-session probe: base 92x112 | chin 92x" + h + " | chin + oval (outside: black)", new Point(gap, 32),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0), 1);
        Mat mk = mask(FrameB.W, h);
        int y = head;
        int same = 0;
        r.append("Лист: ").append(od.resolve("chin_sheet.png")).append("; люди и снимки:\n");
        for (String p : own) {
            String key = g.probeKeys.get(p).get(0);
            Path kp = Paths.get(key);
            Path raw = gtDir.resolve(kp.getName(1).toString()).resolve(kp.getName(2) + ".jpg");
            Mat color = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(raw)), Imgcodecs.IMREAD_COLOR);
            double[] row = rows.get(key);
            Mat base = FrameB.frame(color, row, true);
            Mat chin = frame(color, row, h);
            boolean eq = Core.countNonZero(diff(base, chin.submat(new Rect(0, 0, FrameB.W, FrameB.H)))) == 0;
            if (eq) same++;
            Mat oval = Mat.zeros(chin.size(), chin.type());
            chin.copyTo(oval, mk);
            r.append(String.format(Locale.ROOT, "  %s – %s (верх кадра (Ч) = кадр Б: %s)%n", p, key, eq ? "да" : "НЕТ"));
            Imgproc.putText(sheet, p, new Point(gap, y + chh / 2), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0), 2);
            Imgproc.putText(sheet, kp.getName(2).toString(), new Point(gap, y + chh / 2 + 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, new Scalar(0), 1);
            int x = lab;
            for (Mat v : List.of(base, chin, oval)) {
                Mat z = new Mat();
                Imgproc.resize(v, z, new Size(v.cols() * sc, v.rows() * sc), 0, 0, Imgproc.INTER_NEAREST);
                z.copyTo(sheet.submat(new Rect(x, y, z.cols(), z.rows())));
                x += cw + gap;
                z.release();
            }
            Imgproc.putText(sheet, "base", new Point(lab, y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            Imgproc.putText(sheet, "chin", new Point(lab + cw + gap, y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            Imgproc.putText(sheet, "chin + oval", new Point(lab + 2 * (cw + gap), y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            y += chh + cap + gap;
            color.release();
            base.release();
            chin.release();
            oval.release();
        }
        r.append(String.format(Locale.ROOT, "Верх кадра (Ч) совпал с кадром Б побайтно у %d из %d (то же выравнивание и масштаб).%n", same, own.size()));
        r.append("На листе: увеличение ×3 без сглаживания (INTER_NEAREST); в (Ч+О) пиксели вне эллипса показаны чёрными только для просмотра – в расчёте они не входят в гистограммы.\n");
        if (!Imgcodecs.imwrite(od.resolve("chin_sheet.png").toString(), sheet)) throw new IOException("chin-sheet: не записан лист");
        r.append("SHA-256 листа: ").append(FrameB.sha256(od.resolve("chin_sheet.png"))).append('\n');
        Files.writeString(od.resolve("chin_geometry.txt"), r, StandardCharsets.UTF_8);
        System.out.print(r);
    }

    static Mat diff(Mat a, Mat b) {
        Mat d = new Mat();
        Core.absdiff(a, b, d);
        return d;
    }
}
