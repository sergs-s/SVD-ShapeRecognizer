package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Mat;
import svd.recognizer.faces.FaceEvaluation.Role;

/**
 * Проверка выравнивания по 5 точкам и политики неудачных детекций
 * (OpenCV нужен только для подобия; модели и база не нужны).
 *
 * @author ssv
 */
public class AlignmentEvaluationTest {

    @BeforeClass
    public static void loadOpenCv() {
        nu.pattern.OpenCV.loadLocally();
    }

    @Test
    public void ownTemplateIsArcFaceShiftedIntoNarrowFrame() {
        double[][] t = FaceAlignment.ownTemplate();
        assertEquals(38.2946 - 10, t[0][0], 1e-9);
        assertEquals(51.6963, t[0][1], 1e-9);
        // Середина между глазами — около центра кадра 92 по ширине.
        assertEquals(46, (t[0][0] + t[1][0]) / 2, 0.2);
    }

    @Test
    public void similarityRecoversScaleRotationAndShift() {
        double[][] to = FaceAlignment.ownTemplate();
        // from = to, повёрнутые на 10°, увеличенные в 1,5 раза и сдвинутые.
        double a = Math.toRadians(10);
        double[][] from = new double[5][2];
        for (int i = 0; i < 5; i++) {
            from[i][0] = 1.5 * (Math.cos(a) * to[i][0] - Math.sin(a) * to[i][1]) + 7;
            from[i][1] = 1.5 * (Math.sin(a) * to[i][0] + Math.cos(a) * to[i][1]) - 3;
        }
        Mat m = FaceAlignment.similarity(from, to);
        for (int i = 0; i < 5; i++) {
            double x = m.get(0, 0)[0] * from[i][0] + m.get(0, 1)[0] * from[i][1] + m.get(0, 2)[0];
            double y = m.get(1, 0)[0] * from[i][0] + m.get(1, 1)[0] * from[i][1] + m.get(1, 2)[0];
            assertEquals(to[i][0], x, 1e-4);
            assertEquals(to[i][1], y, 1e-4);
        }
    }

    @Test
    public void failedDetectionIsRejectedAndExcludedFromDetectedFar() {
        List<AlignmentEvaluation.Probe> probes = new ArrayList<>();
        // Валидация чужих (с детекцией): оценки 0,5 и 0,9; α = 0 → θ чуть меньше 0,5.
        probes.add(new AlignmentEvaluation.Probe(0, Role.IMPOSTOR_VAL, 30, true, 0.5, 1.0, 1));
        probes.add(new AlignmentEvaluation.Probe(0, Role.IMPOSTOR_VAL, 30, true, 0.9, 1.0, 1));
        probes.add(new AlignmentEvaluation.Probe(0, Role.IMPOSTOR_VAL, 30, false, Double.NaN, Double.NaN, -1));
        // Свои контроля: один принят, один отказ детектора, один отказ по порогу.
        probes.add(new AlignmentEvaluation.Probe(0, Role.KNOWN_TEST, 1, true, 0.1, 1.0, 1));
        probes.add(new AlignmentEvaluation.Probe(0, Role.KNOWN_TEST, 1, false, Double.NaN, Double.NaN, -1));
        probes.add(new AlignmentEvaluation.Probe(0, Role.KNOWN_TEST, 2, true, 0.8, 1.0, 2));
        // Чужие контроля: неудачная детекция и принятый.
        probes.add(new AlignmentEvaluation.Probe(0, Role.IMPOSTOR_TEST, 31, false, Double.NaN, Double.NaN, -1));
        probes.add(new AlignmentEvaluation.Probe(0, Role.IMPOSTOR_TEST, 31, true, 0.2, 1.0, 1));
        AlignmentEvaluation.Counts c = AlignmentEvaluation.evaluate(probes, AlignmentEvaluation.Score.BEST, 0.0);
        assertTrue(c.theta < 0.5 && c.theta > 0.49);
        assertEquals(2, c.sum(c.rejects));
        assertEquals(1, c.sum(c.detectorRejects));
        assertEquals(1, c.falseAccept[0]);
        assertEquals(2, c.impostorAttempts[0]);
        assertEquals(1, c.impostorDetected[0]);
        assertEquals(1, c.peopleAccepted[0]);
    }

    @Test
    public void normalizeGivesUnitVector() {
        double[] v = AlignmentEvaluation.normalize(new double[] {3, 4});
        assertEquals(0.6, v[0], 1e-12);
        assertEquals(0.8, v[1], 1e-12);
    }
}
