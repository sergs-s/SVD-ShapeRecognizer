package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;

/**
 * Протокол FERET (PLAN.md п. 5.5): разбор имён, дни между датами, расхождение точек глаз, выбор снимка G2', контрольные чужие,
 * интервалы, «до ближайшего шаблона сессии», распаковка .ppm.bz2 в памяти. Данные — синтетические (база FERET в тестах не
 * используется).
 *
 * @author ssv
 */
public class FeretTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void load() {
        nu.pattern.OpenCV.loadLocally();
    }

    static Img img(String key) {
        return FeretData.parse(key, null, null, null);
    }

    @Test
    public void parseNames() {
        Img a = img("00012_930831_fb_a");
        assertEquals("00012", a.person());
        assertEquals("930831", a.date());
        assertEquals("fb", a.pose());
        assertEquals("a", a.flag());
        assertTrue(a.frontal());
        Img b = img("00012_930831_pl");
        assertEquals("-", b.flag());
        assertFalse(b.frontal());
        for (String bad : new String[] {"0012_930831_fa", "00012_9308_fa", "00012_930831"}) {
            try {
                img(bad);
                fail("принято «" + bad + "»");
            } catch (IllegalArgumentException expected) {
                // ожидаемо
            }
        }
    }

    @Test
    public void daysAndCoords() {
        assertEquals(188, FeretData.days("930831", "940307"));
        assertEquals(843, FeretData.days("940307", "960627"));
        assertArrayEquals(new double[] {348, 350}, FeretData.coords(" 348 350 "), 0);
    }

    @Test
    public void eyeError() {
        Img im = FeretData.parse("00012_930831_fa", null, new double[] {348, 350}, new double[] {242, 344});
        double[] row = new double[15];
        row[4] = 245;
        row[5] = 344;
        row[6] = 348;
        row[7] = 346;
        double iod = Math.hypot(106, 6);
        double[] e = FeretData.eyeError(im, row);
        assertEquals(4 / iod, e[0], 1e-12);
        assertEquals(3 / iod, e[1], 1e-12);
        assertEquals(3.5 / iod, e[2], 1e-12);
        assertEquals(iod, e[3], 1e-12);
        assertNull(FeretData.eyeError(im, null));
        assertNull(FeretData.eyeError(img("00012_930831_fa"), row));
    }

    @Test
    public void g2pChoice() {
        assertEquals("00001_930831_fa", XBaseFeret.g2pChoice(List.of(img("00001_930831_fb"), img("00001_930831_fa_a"), img("00001_930831_fa"))));
        assertEquals("00001_930831_fa_a", XBaseFeret.g2pChoice(List.of(img("00001_930831_fb"), img("00001_930831_fa_a"))));
        assertEquals("00001_930831_fb", XBaseFeret.g2pChoice(List.of(img("00001_930831_fb_a"), img("00001_930831_fb"))));
        assertNull(XBaseFeret.g2pChoice(List.of()));
    }

    @Test
    public void controls() {
        List<String> one = new ArrayList<>();
        for (int i = 1; i <= 300; i++) one.add(String.format("%05d", i));
        Set<String> kept = new HashSet<>(one);
        kept.remove("00005");
        List<String> c = XBaseFeret.controls(one, kept, 42);
        assertEquals(200, c.size());
        assertEquals(c, c.stream().sorted().toList());
        assertFalse(c.contains("00005"));
        assertEquals(c, XBaseFeret.controls(one, kept, 42));
        assertNotEquals(c, XBaseFeret.controls(one, kept, 43));
        try {
            XBaseFeret.controls(one, new HashSet<>(one.subList(0, 150)), 42);
            fail("принято 150 < 200");
        } catch (IllegalStateException expected) {
            // ожидаемо
        }
    }

    @Test
    public void bins() {
        assertEquals(0, XBaseFeret.bin(0));
        assertEquals(0, XBaseFeret.bin(30));
        assertEquals(1, XBaseFeret.bin(31));
        assertEquals(1, XBaseFeret.bin(365));
        assertEquals(2, XBaseFeret.bin(366));
    }

    @Test
    public void nearestSession() {
        GalleryScorer.ScoreModel inner = x -> new double[] {5, 1, 3, 2};
        assertArrayEquals(new double[] {1, 2}, XBaseFeret.nearestSession(inner, new int[] {0, 0, 1, 1}, 2).scores(new double[1]), 0);
        GalleryScorer.ScoreModel two = x -> new double[] {5, 1};
        double[] s = XBaseFeret.nearestSession(two, new int[] {0, 0}, 2).scores(new double[1]);
        assertEquals(1, s[0], 0);
        assertTrue(Double.isInfinite(s[1]));
    }

    @Test
    public void decodeBz2InMemory() throws Exception {
        Path f = tmp.newFile("x.ppm.bz2").toPath();
        byte[] head = "P6\n4 3\n255\n".getBytes(StandardCharsets.US_ASCII);
        byte[] px = new byte[4 * 3 * 3];
        for (int i = 0; i < px.length; i++) px[i] = (byte) (i * 7);
        try (OutputStream out = new BZip2CompressorOutputStream(Files.newOutputStream(f))) {
            out.write(head);
            out.write(px);
        }
        Mat m = FeretData.decode(f);
        assertEquals(3, m.rows());
        assertEquals(4, m.cols());
        assertEquals(3, m.channels());
        // Пиксель (строка 1, столбец 2): RGB из файла — BGR в Mat.
        int o = (1 * 4 + 2) * 3;
        double[] bgr = m.get(1, 2);
        assertEquals(px[o + 2] & 0xFF, bgr[0], 0);
        assertEquals(px[o + 1] & 0xFF, bgr[1], 0);
        assertEquals(px[o] & 0xFF, bgr[2], 0);
        m.release();
    }
}
