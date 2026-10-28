package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Кадр k = 2 (184×224) и 5 точек YuNet в его координатах – вход MsLbpFeature (пачка 2). Кадр – как вариант «в» экспорта
 * (GalleryEvaluation.fromFull: полный серый снимок уменьшается INTER_AREA до масштаба подобия, затем подобие по 5 точкам в шаблон
 * ownTemplate(184, 224) = 2 · ownTemplate(92, 112), INTER_LINEAR, BORDER_REPLICATE – передискретизация Б, решение (25)); точки –
 * то же второе подобие, применённое к точкам YuNet уменьшенного снимка. GT и FEI: кадр из исходного снимка сверяется с кадром «в»
 * экспорта побайтно (иначе остановка). FERET: кадры – D:\data\Feret\derived\c\, точки – derived\c_points.tsv (кэш).
 * Отражённый кадр – FaceMirror.mirror (способ а), точки – отражение, перестановка FaceMirror.SWAP и та же поправка подобием.
 *
 * @author ssv
 */
final class FrameK2 {

    static final int K = 2;
    static final int W = 92 * K;
    static final int H = 112 * K;
    static final String DIR = "c";
    static final String POINTS = "c_points.tsv";

    /** Кадр и точки [5][x, y] (порядок YuNet: глаз на снимке слева, справа, нос, угол рта слева, справа). */
    record F(Mat frame, double[][] pts) {}

    private FrameK2() {
    }

    static double[][] template() {
        return FaceAlignment.ownTemplate(W, H);
    }

    static double[][] lm(double[] row) {
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = row[4 + 2 * j];
            lm[j][1] = row[5 + 2 * j];
        }
        return lm;
    }

    static double[] apply(Mat m, double[] p) {
        return new double[] {m.get(0, 0)[0] * p[0] + m.get(0, 1)[0] * p[1] + m.get(0, 2)[0], m.get(1, 0)[0] * p[0] + m.get(1, 1)[0] * p[1] + m.get(1, 2)[0]};
    }

    /** Кадр и точки по полному серому снимку (шаги GalleryEvaluation.fromFull). */
    static F fromFull(Mat fullGray, double[][] lmFull) {
        double[][] t = template();
        Mat m1 = FaceAlignment.similarity(lmFull, t);
        double s = Math.hypot(m1.get(0, 0)[0], m1.get(1, 0)[0]);
        m1.release();
        Mat scaled = new Mat();
        Imgproc.resize(fullGray, scaled, new Size(Math.max(1, Math.round(fullGray.cols() * s)), Math.max(1, Math.round(fullGray.rows() * s))), 0, 0,
                s < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        double sx = scaled.cols() / (double) fullGray.cols();
        double sy = scaled.rows() / (double) fullGray.rows();
        double[][] lm = new double[5][2];
        for (int j = 0; j < 5; j++) {
            lm[j][0] = lmFull[j][0] * sx;
            lm[j][1] = lmFull[j][1] * sy;
        }
        Mat m2 = FaceAlignment.similarity(lm, t);
        Mat out = new Mat();
        Imgproc.warpAffine(scaled, out, m2, new Size(W, H), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        double[][] pts = new double[5][];
        for (int j = 0; j < 5; j++) pts[j] = apply(m2, lm[j]);
        m2.release();
        scaled.release();
        return new F(out, pts);
    }

    /** Точки на отражённом кадре (FaceMirror.mirror) в порядке YuNet. */
    static double[][] mirrorPts(double[][] p) {
        double[][] c = FaceMirror.correction(template(), W);
        double[][] out = new double[p.length][];
        for (int i = 0; i < p.length; i++) {
            double[] q = p[FaceMirror.SWAP[i]];
            out[i] = FaceMirror.apply(c, new double[] {W - 1 - q[0], q[1]});
        }
        return out;
    }

    static boolean same(Mat a, Mat b) {
        if (a.rows() != b.rows() || a.cols() != b.cols() || a.type() != b.type()) return false;
        Mat d = new Mat();
        Core.absdiff(a, b, d);
        boolean z = Core.countNonZero(d) == 0;
        d.release();
        return z;
    }

    /**
     * Кадры и точки для ключей keys: ключи FERET (без разделителя пути) – из кэша derived (недостающие – из исходного снимка, с
     * записью); ключи экспорта (gt\…, fei\…) – кадр «в» экспорта, точки – по исходному снимку со сверкой кадра.
     */
    static Map<String, F> load(Iterable<String> keys, XBaseA0.FeretSetup su, Path feretDir, FarExport.Loaded export, Path exportDir, Path gtDir, Path feiDir)
            throws IOException {
        Map<String, F> out = new LinkedHashMap<>();
        Map<String, double[]> expRows = null;
        Map<String, Path> feiRaw = null;
        Map<String, GalleryEvaluation.Sample> bySample = null;
        Map<String, double[][]> fPts = null;
        Writer fw = null;
        int made = 0;
        try {
            for (String k : keys) {
                if (!k.contains("\\") && !k.contains("/")) {
                    if (fPts == null) {
                        fPts = readPoints(feretDir.resolve(FeretData.DERIVED).resolve(POINTS));
                        Files.createDirectories(feretDir.resolve(FeretData.DERIVED).resolve(DIR));
                    }
                    Path png = feretDir.resolve(FeretData.DERIVED).resolve(DIR).resolve(k + ".png");
                    if (fPts.containsKey(k) && Files.exists(png)) {
                        Mat m = Imgcodecs.imread(png.toString(), Imgcodecs.IMREAD_GRAYSCALE);
                        if (m.empty()) throw new IOException("Нет кадра " + png);
                        out.put(k, new F(m, fPts.get(k)));
                        continue;
                    }
                    double[] row = su.det.get(k);
                    if (row == null) throw new IllegalStateException("FrameK2: нет детекции FERET " + k);
                    Mat color = FeretData.decode(su.byKey.get(k).file());
                    Mat gray = new Mat();
                    Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
                    color.release();
                    F f = fromFull(gray, lm(row));
                    gray.release();
                    FeretData.writePng(png, f.frame());
                    if (fw == null) fw = Files.newBufferedWriter(feretDir.resolve(FeretData.DERIVED).resolve(POINTS), StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                    fw.write(pointsLine(k, f.pts()));
                    fw.flush();
                    out.put(k, f);
                    if (++made % 200 == 0) System.out.printf(Locale.ROOT, "FrameK2: FERET кадров k = 2 записано %d%n", made);
                    continue;
                }
                if (expRows == null) {
                    expRows = exportRows(exportDir);
                    feiRaw = feiRaw(feiDir);
                    bySample = new HashMap<>();
                    for (Map<String, List<GalleryEvaluation.Sample>> m : List.of(export.data().gt(), export.data().fei())) {
                        for (List<GalleryEvaluation.Sample> l : m.values()) for (GalleryEvaluation.Sample s : l) bySample.put(s.file().toString(), s);
                    }
                }
                GalleryEvaluation.Sample s = bySample.get(k);
                double[] row = expRows.get(k);
                if (s == null || row == null) throw new IllegalStateException("FrameK2: нет снимка или детекции экспорта " + k);
                Path p = Paths.get(k);
                Path raw = p.getName(0).toString().equals("gt") ? gtDir.resolve(p.getName(1).toString()).resolve(p.getName(2) + ".jpg")
                        : feiRaw.get(p.getName(2).toString());
                if (raw == null || !Files.exists(raw)) throw new IllegalStateException("FrameK2: нет исходного снимка " + k);
                Mat color = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(raw)), Imgcodecs.IMREAD_COLOR);
                Mat gray = new Mat();
                Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
                color.release();
                F f = fromFull(gray, lm(row));
                gray.release();
                Mat e = export.frame(s, GalleryEvaluation.Variant.C);
                if (e == null || !same(e, f.frame())) throw new IllegalStateException("FrameK2: кадр k = 2 из исходного снимка ≠ кадр «в» экспорта: " + k);
                f.frame().release();
                out.put(k, new F(e, f.pts()));
            }
        } finally {
            if (fw != null) fw.close();
        }
        return out;
    }

    static String pointsLine(String k, double[][] p) {
        StringBuilder b = new StringBuilder(k);
        for (double[] q : p) b.append('\t').append(q[0]).append('\t').append(q[1]);
        return b.append('\n').toString();
    }

    static Map<String, double[][]> readPoints(Path f) throws IOException {
        Map<String, double[][]> m = new HashMap<>();
        if (!Files.exists(f)) return m;
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            String[] c = l.split("\t");
            if (c.length != 11) continue; // оборванная строка
            double[][] p = new double[5][2];
            for (int j = 0; j < 5; j++) {
                p[j][0] = Double.parseDouble(c[1 + 2 * j]);
                p[j][1] = Double.parseDouble(c[2 + 2 * j]);
            }
            m.put(c[0], p);
        }
        return m;
    }

    /** Строки YuNet экспорта (detections.tsv): ключ base\person\name → 15 чисел (лицо найдено). */
    static Map<String, double[]> exportRows(Path exportDir) throws IOException {
        Map<String, double[]> m = new HashMap<>();
        List<String> ls = Files.readAllLines(exportDir.resolve(FarExport.DETECTIONS), StandardCharsets.UTF_8);
        for (String l : ls.subList(1, ls.size())) {
            String[] c = l.split("\t", -1);
            if (!c[4].equals("1")) continue;
            double[] row = new double[15];
            for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[5 + i]);
            m.put(Paths.get(c[0], c[1], c[2]).toString(), row);
        }
        return m;
    }

    /** Исходные снимки FEI по имени без расширения («1-05»). */
    static Map<String, Path> feiRaw(Path feiDir) throws IOException {
        Map<String, Path> m = new HashMap<>();
        try (Stream<Path> s = Files.walk(feiDir.resolve(FeiDataset.ORIGINALS))) {
            for (Path f : s.filter(Files::isRegularFile).toList()) {
                String n = f.getFileName().toString();
                if (n.toLowerCase(Locale.ROOT).endsWith(".jpg")) m.put(n.substring(0, n.length() - 4), f);
            }
        }
        return m;
    }
}
