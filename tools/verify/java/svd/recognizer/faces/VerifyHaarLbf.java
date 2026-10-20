package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.face.Facemark;
import org.opencv.objdetect.CascadeClassifier;

/**
 * Запускалка проверки (tools/verify, ветка cloud-verify; Жорик №12 от 20.10.2026): Хаар (рамка, глаза) и Facemark LBF на снимках
 * FERET из списка ключей – теми же методами кода проекта, без его правки: HaarFrames.input / detect, LbfFrames.load / fit,
 * HaarDebug.load, FeretData.decode; каскады и модель – из каталога моделей (имена – HaarDebug.FACE_FILE, EYE_GLASSES_FILE,
 * LbfFrames.MODEL_FILE). Масштаб входа 1,0 (как FERET в HaarFrames.bases). Вывод – строки в формате кэшей haar.tsv и lbf.tsv
 * (HaarFrames.cached, LbfFrames.cached) с теми же подписями.
 *
 * Аргументы: каталог моделей, корень FERET (dvd1, dvd2), файл ключей (по одному в строке), выход haar, выход lbf.
 */
public final class VerifyHaarLbf {

    private VerifyHaarLbf() {
    }

    static Path image(Path feret, String key) throws IOException {
        String person = key.split("_")[0];
        for (String dvd : new String[] {"dvd1", "dvd2"}) {
            Path p = feret.resolve(dvd).resolve("data").resolve("images").resolve(person).resolve(key + ".ppm.bz2");
            if (Files.exists(p)) return p;
        }
        throw new IOException("нет снимка " + key);
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        Path models = Paths.get(a[0]);
        Path feret = Paths.get(a[1]);
        List<String> keys;
        try (Stream<String> s = Files.lines(Paths.get(a[2]), StandardCharsets.UTF_8)) {
            keys = s.map(String::trim).filter(x -> !x.isEmpty()).toList();
        }
        CascadeClassifier face = HaarDebug.load(models.resolve(HaarDebug.FACE_FILE));
        CascadeClassifier eye = HaarDebug.load(models.resolve(HaarDebug.EYE_GLASSES_FILE));
        Facemark fm = LbfFrames.load(models);
        String lbfSig = LbfFrames.signature(models);
        int n = 0;
        try (Writer wh = Files.newBufferedWriter(Paths.get(a[3]), StandardCharsets.UTF_8);
             Writer wl = Files.newBufferedWriter(Paths.get(a[4]), StandardCharsets.UTF_8)) {
            wh.write("# " + HaarFrames.SIGNATURE + "\n");
            wl.write("# " + lbfSig + "\n");
            for (String k : keys) {
                Path f = image(feret, k);
                if (Files.size(f) == 0) throw new IOException("пустой файл снимка (опись, не снимок): " + f);
                Mat color = FeretData.decode(f);
                Mat in = HaarFrames.input(color, 1.0);
                color.release();
                HaarFrames.Det d = HaarFrames.detect(in, 1.0, face, eye);
                StringBuilder l = new StringBuilder(k).append('\t').append(d.rects());
                for (int i = 0; i < 4; i++) l.append('\t').append(d.face() == null ? "" : Double.toString(d.face()[i]));
                l.append("\t\t");
                for (double[] p : new double[][] {d.left(), d.right()})
                    l.append('\t').append(p == null ? "" : Double.toString(p[0])).append('\t').append(p == null ? "" : Double.toString(p[1]));
                wh.write(l.append('\n').toString());
                LbfFrames.Lm lm = d.found() ? LbfFrames.fit(fm, in, 1.0, d) : null;
                in.release();
                StringBuilder m = new StringBuilder(k).append('\t').append(lm == null ? "0" : "1");
                for (int i = 0; i < LbfFrames.POINTS; i++)
                    m.append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][0])).append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][1]));
                wl.write(m.append('\n').toString());
                if (++n % 250 == 0) {
                    wh.flush();
                    wl.flush();
                    System.out.printf(Locale.ROOT, "VerifyHaarLbf: %d из %d%n", n, keys.size());
                }
            }
        }
        System.out.printf(Locale.ROOT, "VerifyHaarLbf: готово, %d снимков%n", n);
    }
}
