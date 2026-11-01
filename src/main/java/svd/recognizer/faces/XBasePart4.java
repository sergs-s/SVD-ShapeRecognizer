package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Пачка 4 – подстройка параметров конфигурации (32) (задание Жорика №1 от 01.11.2026, дополнение №4). Режим p4-retina-sheet –
 * лист кадров «а» 8 снимков GT: оригинал, Тан – Триггс (параметры конфигурации), ретинальный фильтр Vu (RetinaFilter, параметры
 * автора) – для проверки глазами до прогона; рядом – доля усечённых значений (у автора 2–4 %).
 *
 * @author ssv
 */
final class XBasePart4 {

    private XBasePart4() {
    }

    /** Вещественный кадр → 8 бит по min..max кадра. */
    static Mat toGray(double[] v, int w, int h) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) b[i] = (byte) (hi > lo ? Math.round(255 * (v[i] - lo) / (hi - lo)) : 128);
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        m.put(0, 0, b);
        return m;
    }

    /** Режим p4-retina-sheet: кадры GT из экспорта (a/gt/sNN/MM.png), человек 1 + 7i, снимок i + 1, i = 0…7. */
    static void retinaSheet(SettingsStore settings, Path out) throws IOException {
        String ed = System.getenv("FACES_EXPORT_DIR");
        if (ed == null) throw new IllegalStateException("p4-retina-sheet: задайте FACES_EXPORT_DIR");
        IlluminationNorm.NormParams p = IlluminationNorm.NormParams.of(settings);
        Path o = out.resolve("part4");
        Files.createDirectories(o);
        StringBuilder rep = new StringBuilder("Пачка 4, Б1′: лист кадров – оригинал | Тан – Триггс | ретинальный фильтр Vu (XBaseEval p4-retina-sheet)\nКод: коммит "
                + FarMethods.commit() + "\n");
        rep.append(String.format(Locale.ROOT, "Тан – Триггс: γ = %s, σ₀ = %s, σ₁ = %s, α = %s, τ = %s; ретинальный: σ₁ = σ₂ = %s, σ_Ph = %s, σ_H = %s, Th = %s "
                + "(дисс. Vu, 2010, разд. 3.4.1.3; кадры автора – Yale B 192×168, у нас 92×112, σ не масштабируются)%n", p.gamma(), p.sigma0(), p.sigma1(),
                p.alpha(), p.tau(), RetinaFilter.AUTHOR.sigma1(), RetinaFilter.AUTHOR.sigmaPh(), RetinaFilter.AUTHOR.sigmaH(), RetinaFilter.AUTHOR.th()));
        rep.append("снимок | доля усечённых (|I_nor| ≥ Th)\n");
        List<Mat> rows = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            String key = String.format(Locale.ROOT, "gt/s%02d/%02d.png", 1 + 7 * i, i + 1);
            Path f = Paths.get(ed, "a").resolve(key);
            Mat a = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(f)), Imgcodecs.IMREAD_GRAYSCALE);
            double[] tt = IlluminationNorm.TAN_TRIGGS.vector(a, p);
            double[] rf = RetinaFilter.filter(a, RetinaFilter.AUTHOR);
            rep.append(String.format(Locale.ROOT, "%s | %.2f %%%n", key, 100 * RetinaFilter.clippedShare(rf, RetinaFilter.AUTHOR)));
            List<Mat> tiles = List.of(a, toGray(tt, a.cols(), a.rows()), toGray(rf, a.cols(), a.rows()));
            Mat row = new Mat();
            Core.hconcat(tiles, row);
            rows.add(row);
        }
        Mat sheet = new Mat();
        Core.vconcat(rows, sheet);
        Mat big = new Mat();
        Imgproc.resize(sheet, big, new Size(sheet.cols() * 3, sheet.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2BGR);
        String[] cap = {"orig", "Tan-Triggs", "retina Vu"};
        for (int c = 0; c < 3; c++) Imgproc.putText(big, cap[c], new Point(c * 92 * 3 + 6, 22), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 255), 2);
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".png", big, buf);
        Files.write(o.resolve("p4_retina_sheet.png"), buf.toArray());
        FarMethods.write(o.resolve("p4_retina_sheet.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }
}
