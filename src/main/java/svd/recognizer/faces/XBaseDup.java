package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.storage.SettingsStore;

/**
 * Поиск пар-дублей людей в FEI и MUCT (Жорик №4 от 04.11.2026; режим dup-scan, без обучения): признаки SFace по кадрам alignCrop экспорта
 * (FEI – отобранные снимки); для каждой пары разных людей – наименьшее 1 − cos по парам снимков («лучший к лучшему») и медиана по всем
 * парам снимков; распределение «свой – свой» – 1 − cos всех пар разных снимков одного человека. Кандидаты – пары с наименьшим 1 − cos ниже
 * 99-го процентиля распределения своих. Отчёт, списки и листы кадров – reports/faces/duplicates/ (вне git). Решение о дубле – только глазами.
 *
 * @author ssv
 */
final class XBaseDup {

    private XBaseDup() {
    }

    /** Листы – пары с наименьшим 1 − cos, по порядку. */
    static final int SHEET_PAIRS = 60;
    static final int SHEET_ROWS = 20;

    record Img(String person, String name, String date, java.util.concurrent.Callable<Mat> frame, double[] v) {}

    record Pair(String a, String b, double min, String ia, String ib, Img xa, Img xb, double median) {}

    static double dist(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) s += x[i] * y[i];
        return 1 - s;
    }

    static void scan(FarExport.Loaded export, Path exportDir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        Path d = out.getParent().resolve("duplicates");
        Files.createDirectories(d);
        FaceRecognizerSF rec = FaceRecognizerSF.create(Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE).toString(), "");
        StringBuilder rep = new StringBuilder("Поиск пар-дублей (XBaseEval dup-scan; без обучения)\nКод: коммит " + FarMethods.commit() + "\n1 − cos признаков SFace, кадры alignCrop экспорта; пара людей "
                + "– наименьшее 1 − cos по парам снимков; кандидаты – ниже 99-го процентиля «свой – свой». Решение о дубле – только глазами.\n");
        for (Map.Entry<String, Map<String, List<Sample>>> e : List.of(Map.entry("gt", export.data().gt()), Map.entry("fei", export.data().fei()), Map.entry("muct", export.data().muct()))) {
            String base = e.getKey();
            Map<String, List<Img>> imgs = new TreeMap<>();
            for (Map.Entry<String, List<Sample>> p : e.getValue().entrySet()) {
                for (Sample s : p.getValue()) {
                    Mat c = export.frame(s, GalleryEvaluation.Variant.SFACE);
                    if (c == null) continue;
                    double[] v = GalleryEvaluation.vector(c, GalleryEvaluation.Variant.SFACE, rec);
                    c.release();
                    imgs.computeIfAbsent(p.getKey(), k -> new ArrayList<>()).add(new Img(p.getKey(), FarExport.stem(s.file()), "", () -> export.frame(s, GalleryEvaluation.Variant.SFACE), v));
                }
            }
            rep.append(one(base, imgs, d));
        }
        rep.append(String.format(Locale.ROOT, "%nВремя: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(d.resolve("dup_scan.txt"), rep);
        System.out.print(rep);
        if (System.getenv("FACES_FERET_DIR") != null) feret(exportDir, selLines, settings, rec);
        System.out.println("DONE");
    }

    /** FERET: fa/fb всех людей с лицом YuNet; «свой – свой» – только снимки разных дат. Списки и листы – только D:\data\Feret\reports\duplicates\. */
    static void feret(Path exportDir, List<String> selLines, SettingsStore settings, FaceRecognizerSF rec) throws IOException {
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        Path d = su.dir.resolve("reports").resolve("duplicates");
        Files.createDirectories(d);
        Map<String, List<Img>> imgs = new TreeMap<>();
        for (FeretData.Img im : su.imgs) {
            if (su.det.get(im.key()) == null) continue;
            Mat c = FeretData.frame(su.dir, "sface", im.key());
            double[] v = GalleryEvaluation.vector(c, GalleryEvaluation.Variant.SFACE, rec);
            c.release();
            String k = im.key();
            imgs.computeIfAbsent(im.person(), x -> new ArrayList<>()).add(new Img(im.person(), k, im.date(), () -> FeretData.frame(su.dir, "sface", k), v));
        }
        StringBuilder rep = new StringBuilder("Поиск пар-дублей FERET (XBaseEval dup-scan; без обучения)\nКод: коммит " + FarMethods.commit() + "\nfa/fb всех людей с лицом YuNet; 1 − cos SFace; «свой – свой» – пары снимков разных дат одного человека.\n");
        rep.append(one("feret", imgs, d));
        rep.append(String.format(Locale.ROOT, "%nВремя: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(d.resolve("dup_scan_feret.txt"), rep);
        System.out.print(rep);
    }

    /** Одна база: распределение своих, пары людей, список и листы; строка отчёта. */
    static String one(String base, Map<String, List<Img>> imgs, Path d) throws IOException {
        StringBuilder rep = new StringBuilder();
        {
            int n = imgs.values().stream().mapToInt(List::size).sum();
            List<String> persons = new ArrayList<>(imgs.keySet());
            List<Double> own = new ArrayList<>();
            for (List<Img> l : imgs.values()) for (int i = 0; i < l.size(); i++) for (int j = i + 1; j < l.size(); j++) if (base.equals("feret") ? !l.get(i).date().equals(l.get(j).date()) : true) own.add(dist(l.get(i).v(), l.get(j).v()));
            double[] so = own.stream().mapToDouble(Double::doubleValue).sorted().toArray();
            double p99 = FeiBootstrap.quantile(so, 0.99);
            List<Pair> pairs = new ArrayList<>();
            for (int a = 0; a < persons.size(); a++) {
                List<Img> la = imgs.get(persons.get(a));
                for (int b = a + 1; b < persons.size(); b++) {
                    List<Img> lb = imgs.get(persons.get(b));
                    double[] all = new double[la.size() * lb.size()];
                    int k = 0;
                    double min = Double.POSITIVE_INFINITY;
                    Img xa = null;
                    Img xb = null;
                    for (Img x : la) {
                        for (Img y : lb) {
                            double v = dist(x.v(), y.v());
                            all[k++] = v;
                            if (v < min) {
                                min = v;
                                xa = x;
                                xb = y;
                            }
                        }
                    }
                    Arrays.sort(all);
                    pairs.add(new Pair(persons.get(a), persons.get(b), min, xa.name(), xb.name(), xa, xb, FeiBootstrap.quantile(all, 0.5)));
                }
            }
            pairs.sort((x, y) -> Double.compare(x.min(), y.min()));
            long cand = pairs.stream().filter(q -> q.min() < p99).count();
            rep.append(String.format(Locale.ROOT, "%n=== %s: людей %d, снимков %d; пар людей %d ===%n«свой – свой» (%d пар снимков): медиана %.3f, 90-й %.3f, 95-й %.3f, 99-й %.3f, наибольшее %.3f%n",
                    base.toUpperCase(Locale.ROOT), persons.size(), n, pairs.size(), so.length, FeiBootstrap.quantile(so, 0.5), FeiBootstrap.quantile(so, 0.9), FeiBootstrap.quantile(so, 0.95), p99,
                    so[so.length - 1]));
            rep.append(String.format(Locale.ROOT, "Кандидатов (наименьшее 1 − cos < 99-го процентиля %.3f): %d; ниже 95-го: %d; ниже 90-го: %d; ниже медианы своих: %d%n", p99, cand, pairs.stream().filter(q -> q
                    .min() < FeiBootstrap.quantile(so, 0.95)).count(), pairs.stream().filter(q -> q.min() < FeiBootstrap.quantile(so, 0.9)).count(), pairs.stream().filter(q -> q.min() < FeiBootstrap.quantile(so,
                            0.5)).count()));
            rep.append("Первые 15: человек 1 | человек 2 | наименьшее 1 − cos | снимки | медиана по парам снимков\n");
            StringBuilder tsv = new StringBuilder("# " + base + ": пары людей с наименьшим 1 − cos < 99-го процентиля «свой – свой» (" + p99 + ")\nperson1\tperson2\tmin_1mcos\timage1\timage2\tmedian_1mcos\n");
            for (int i = 0; i < pairs.size(); i++) {
                Pair q = pairs.get(i);
                if (q.min() < p99) tsv.append(String.format(Locale.ROOT, "%s\t%s\t%.4f\t%s\t%s\t%.4f%n", q.a(), q.b(), q.min(), q.ia(), q.ib(), q.median()));
                if (i < 15) rep.append(String.format(Locale.ROOT, "%s | %s | %.4f | %s, %s | %.4f%n", q.a(), q.b(), q.min(), q.ia(), q.ib(), q.median()));
            }
            Files.writeString(d.resolve(base + "_pairs.tsv"), tsv.toString(), StandardCharsets.UTF_8);
            sheets(base, pairs.subList(0, Math.min(SHEET_PAIRS, pairs.size())), d);
        }
        return rep.toString();
    }

    /** Листы: строка – лучшая пара снимков (кадры alignCrop рядом) с подписью; по SHEET_ROWS строк, в двух столбцах. */
    static void sheets(String base, List<Pair> pairs, Path d) throws IOException {
        int perSheet = SHEET_ROWS * 2;
        for (int s = 0; s * perSheet < pairs.size(); s++) {
            List<Mat> cols = new ArrayList<>();
            for (int c = 0; c < 2; c++) {
                List<Mat> rows = new ArrayList<>();
                for (int r = 0; r < SHEET_ROWS; r++) {
                    int i = s * perSheet + c * SHEET_ROWS + r;
                    if (i >= pairs.size()) break;
                    rows.add(row(pairs.get(i), i + 1));
                }
                if (rows.isEmpty()) continue;
                Mat col = new Mat();
                Core.vconcat(rows, col);
                cols.add(col);
            }
            if (cols.size() == 2 && cols.get(1).rows() < cols.get(0).rows()) {
                Mat pad = new Mat(cols.get(0).rows() - cols.get(1).rows(), cols.get(1).cols(), cols.get(1).type(), new Scalar(255, 255, 255));
                Mat c2 = new Mat();
                Core.vconcat(List.of(cols.get(1), pad), c2);
                cols.set(1, c2);
            }
            Mat sheet = new Mat();
            Core.hconcat(cols, sheet);
            MatOfByte buf = new MatOfByte();
            Imgcodecs.imencode(".png", sheet, buf);
            Files.write(d.resolve(String.format(Locale.ROOT, "%s_sheet_%02d.png", base, s + 1)), buf.toArray());
        }
    }

    /** Строка листа: подпись (номер, люди, снимки, расстояния – латиницей) сверху, под ней два кадра. */
    static Mat row(Pair q, int no) throws IOException {
        Mat a;
        Mat b;
        try {
            a = q.xa().frame().call();
            b = q.xb().frame().call();
        } catch (Exception x) {
            throw new IOException(x);
        }
        Mat pair = new Mat();
        Core.hconcat(List.of(a, new Mat(a.rows(), 8, a.type(), new Scalar(255, 255, 255)), b), pair);
        Mat pad = new Mat(a.rows(), 300, a.type(), new Scalar(255, 255, 255));
        Mat line = new Mat();
        Core.hconcat(List.of(pair, pad), line);
        int x = pair.cols() + 6;
        Scalar ink = new Scalar(0, 0, 0);
        Imgproc.putText(line, String.format(Locale.ROOT, "#%d  %s / %s", no, q.a(), q.b()), new Point(x, 24), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1);
        Imgproc.putText(line, q.ia() + "  " + q.ib(), new Point(x, 50), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1);
        Imgproc.putText(line, String.format(Locale.ROOT, "min %.3f  med %.3f", q.min(), q.median()), new Point(x, 76), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1);
        Mat out = new Mat();
        Core.copyMakeBorder(line, out, 0, 4, 0, 0, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));
        return out;
    }
}
