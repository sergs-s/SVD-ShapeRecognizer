package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import svd.recognizer.OpenCvLoader;

/**
 * К-обычный (Д10, ProkaevFrames.plain), выбор рамки Хаара при равной площади и перезапись кэша без повторов (№11 Жорика).
 *
 * @author ssv
 */
public class ProkaevPlainHaarTest {

    @BeforeClass
    public static void load() {
        OpenCvLoader.load();
    }

    /** Снимок с номером столбца в пикселе (столбец x – значение x), чтобы по кадру узнать окно. */
    static Mat columns(int w, int h) {
        Mat m = new Mat(h, w, CvType.CV_8UC1, new Scalar(0));
        for (int x = 0; x < w; x++) m.submat(new Rect(x, 0, 1, h)).setTo(new Scalar(x % 256));
        return m;
    }

    @Test
    public void plainWindowGt() {
        // 640 × 480: окно 320 × 480 по центру рамки; рамка у левого края – сдвиг внутрь (x0 = 0).
        Mat out = ProkaevFrames.plain(columns(640, 480), new double[] {300, 100, 100, 120}, new int[1]);
        assertEquals(400, out.cols());
        assertEquals(600, out.rows());
        double[] first = out.get(300, 0);
        assertEquals(190, first[0], 1.0); // x0 = 350 − 160 = 190
        Mat edge = ProkaevFrames.plain(columns(640, 480), new double[] {10, 100, 100, 120}, new int[1]);
        assertEquals(0, edge.get(300, 0)[0], 1.0);
    }

    @Test
    public void plainWindowNarrow() {
        // 92 × 112 (уже 2:3 по ширине? 92/112 > 2/3 – окно на всю высоту: 75 × 112).
        int[] cl = new int[1];
        Mat out = ProkaevFrames.plain(columns(92, 112), new double[] {10, 20, 80, 80}, cl);
        assertEquals(400, out.cols());
        assertEquals(1, cl[0]); // рамка 80 шире окна 75
        // 300 × 600 (уже 2:3): окно на всю ширину, высота 450.
        Mat tall = ProkaevFrames.plain(columns(300, 600), new double[] {100, 0, 100, 100}, new int[1]);
        assertEquals(0, tall.get(0, 0)[0], 1.0);
    }

    @Test
    public void haarTieByPosition() {
        Mat in = new Mat(100, 100, CvType.CV_8UC1);
        Rect centre = new Rect(40, 40, 20, 20);
        Rect corner = new Rect(0, 0, 20, 20);
        assertTrue(HaarFrames.before(centre, corner, in));
        assertFalse(HaarFrames.before(corner, centre, in));
        // Равное удаление – меньший y, затем x.
        Rect up = new Rect(40, 20, 20, 20);
        Rect down = new Rect(40, 60, 20, 20);
        assertTrue(HaarFrames.before(up, down, in));
    }

    @Test
    public void dedupeKeepsLast() throws Exception {
        Path f = Files.createTempFile("haar", ".tsv");
        Map<String, String> last = new LinkedHashMap<>();
        last.put("a", "a\t2");
        last.put("b", "b\t1");
        HaarFrames.dedupe(f, "# sig", last);
        assertEquals(List.of("# sig", "a\t2", "b\t1"), Files.readAllLines(f, StandardCharsets.UTF_8));
        Files.delete(f);
    }
}
