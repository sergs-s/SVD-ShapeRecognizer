package svd.recognizer.faces;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Проверка критерия ложной рамки для ORL (без модели и базы).
 *
 * @author ssv
 */
public class DetectorDiagnosticsTest {

    @BeforeClass
    public static void loadOpenCv() {
        nu.pattern.OpenCV.loadLocally();
    }

    private static DetectorDiagnostics.Box box(double x, double y, double w, double h) {
        return new DetectorDiagnostics.Box(x, y, w, h, new double[5][2], 0.9);
    }

    @Test
    public void falseBoxCriterionForOrlFrame() {
        Mat image = new Mat(112, 92, CvType.CV_8UC1);
        // Лицо почти на весь кадр, центр в середине — нормальная рамка.
        assertFalse(DetectorDiagnostics.isFalse(box(10, 15, 70, 85), image));
        // Центр в углу — ложная.
        assertTrue(DetectorDiagnostics.isFalse(box(0, 0, 30, 30), image));
        // Узкая рамка в центре (ширина < 30 % от 92) — ложная.
        assertTrue(DetectorDiagnostics.isFalse(box(35, 45, 20, 20), image));
    }
}
