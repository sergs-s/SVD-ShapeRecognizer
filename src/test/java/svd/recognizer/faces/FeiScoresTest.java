package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;

/**
 * Файл оценок faces-fei (этап 5б): строка попытки восстанавливается точно; лучший и второй кандидаты, признак согласия.
 *
 * @author ssv
 */
public class FeiScoresTest {

    private static final List<String> GALLERY = List.of("001", "002", "003", "004");

    @Test
    public void ownLineRoundTrip() {
        double[] v = {0.3, 0.1 + 0.2, 1e-17, 7.0 / 3};
        FeiScores.Row r = FeiScores.parse(line(FeiScores.OWN, 1, v, false));
        assertEquals("m", r.method());
        assertEquals(2, r.s());
        assertEquals(5, r.j());
        assertEquals(FeiScores.OWN, r.role());
        assertEquals(12, r.image());
        assertEquals("002", r.trueId());
        assertEquals("003", r.bestId());
        assertEquals(1e-17, r.best(), 0);
        assertEquals("001", r.secondId());
        assertEquals(0.3, r.second(), 0);
        assertEquals(0.1 + 0.2, r.trueScore(), 0);
        assertEquals(-1, r.noAgreement());
    }

    @Test
    public void impostorAndTies() {
        double[] v = {2, 1, 1, Double.POSITIVE_INFINITY};
        FeiScores.Row r = FeiScores.parse(line(FeiScores.THR, -1, v, false));
        assertEquals(FeiScores.NONE, r.trueId());
        assertEquals("002", r.bestId());
        assertEquals("003", r.secondId());
        assertEquals(1, r.second(), 0);
        assertEquals(Double.NaN, r.trueScore(), 0);
    }

    @Test
    public void agreement() {
        double m = Double.MAX_VALUE;
        FeiScores.Row yes = FeiScores.parse(line(FeiScores.OWN, 0, new double[] {m, -3.5, m, m}, true));
        assertEquals(0, yes.noAgreement());
        assertEquals("002", yes.bestId());
        assertEquals(-3.5, yes.best(), 0);
        assertEquals(FeiScores.NONE, yes.secondId());
        assertEquals(m, yes.trueScore(), 0);
        double inf = Double.POSITIVE_INFINITY;
        FeiScores.Row no = FeiScores.parse(line(FeiScores.CTRL, -1, new double[] {inf, inf, inf, inf}, true));
        assertEquals(1, no.noAgreement());
        assertEquals(FeiScores.NONE, no.bestId());
        assertEquals(inf, no.best(), 0);
    }

    private static String line(String role, int trueIdx, double[] v, boolean agree) {
        String l = FeiScores.line("m", 2, 5, role, trueIdx < 0 ? "150" : GALLERY.get(trueIdx), 12, trueIdx, GALLERY, v, agree);
        assertEquals('\n', l.charAt(l.length() - 1));
        return l.substring(0, l.length() - 1);
    }
}
