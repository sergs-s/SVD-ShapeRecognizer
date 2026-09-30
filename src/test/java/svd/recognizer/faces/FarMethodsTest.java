package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.Test;
import svd.recognizer.faces.GalleryEvaluation.Impostors;

/**
 * Порог Неймана – Пирсона по людям: чужой принят, если принят хотя бы один его снимок.
 *
 * @author ssv
 */
public class FarMethodsTest {

    @Test
    public void personThresholdLimitsAcceptedPersons() {
        // 69 чужих по 3 снимка; у человека p оценки p + 1, p + 100, NaN (отказ детектора); у последнего — одни NaN.
        Map<String, double[]> map = new LinkedHashMap<>();
        for (int p = 0; p < 69; p++) map.put("p" + p, new double[] {p + 1, p + 100, Double.NaN});
        map.put("none", new double[] {Double.NaN});
        Impostors imp = new Impostors(map);
        // α = 0,05: ⌊0,05 · 69⌋ = 3 человека.
        double theta = FarMethods.personThreshold(imp, 0.05);
        assertTrue(theta >= 3 && theta < 4);
        assertEquals(3, GalleryEvaluation.far(imp, theta)[2]);
        // α = 0: совпадает с порогом по попыткам.
        assertEquals(FaceEvaluation.neymanPearsonThreshold(imp.detected(), 0.0), FarMethods.personThreshold(imp, 0.0), 0.0);
    }
}
