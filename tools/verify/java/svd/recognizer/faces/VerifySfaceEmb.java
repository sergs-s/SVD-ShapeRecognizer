package svd.recognizer.faces;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.FaceRecognizerSF;

/**
 * Признаки SFace по готовым кадрам alignCrop 112×112 (tools/verify, ветка cloud-verify; Жорик №6 от 22.10.2026): порог SFace по
 * 1 − cos. Новых детекций нет. Кадр читается как FeretData.frame (IMREAD_COLOR), признак – GalleryEvaluation.vector(…, SFACE, …)
 * (нормированный). Код проекта не меняется.
 * <p>
 * Аргументы: модель SFace, выходной TSV, затем пары «метка каталог». Строка вывода: метка, относительный путь кадра без .png
 * (разделитель /), 128 чисел (Double.toString).
 */
public final class VerifySfaceEmb {

    private VerifySfaceEmb() {
    }

    public static void main(String[] args) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        FaceRecognizerSF sface = FaceRecognizerSF.create(args[0], "");
        Path out = Paths.get(args[1]);
        Path tmp = Paths.get(args[1] + ".tmp");
        int n = 0;
        try (BufferedWriter w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
            for (int a = 2; a + 1 < args.length; a += 2) {
                Path dir = Paths.get(args[a + 1]);
                List<Path> files;
                try (Stream<Path> s = Files.walk(dir)) {
                    files = s.filter(p -> p.toString().endsWith(".png")).sorted().toList();
                }
                for (Path f : files) {
                    Mat m = Imgcodecs.imread(f.toString(), Imgcodecs.IMREAD_COLOR);
                    if (m.empty()) throw new IOException("Нет кадра " + f);
                    double[] x = GalleryEvaluation.vector(m, GalleryEvaluation.Variant.SFACE, sface);
                    m.release();
                    String rel = dir.relativize(f).toString().replace('\\', '/');
                    StringBuilder b = new StringBuilder(args[a]).append('\t').append(rel, 0, rel.length() - 4);
                    for (double v : x) b.append('\t').append(v);
                    w.write(b.append('\n').toString());
                    n++;
                }
                System.out.println("VerifySfaceEmb: " + args[a] + " " + files.size());
            }
        }
        Files.move(tmp, out, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        System.out.println("VerifySfaceEmb: всего " + n + "\nDONE");
    }
}
