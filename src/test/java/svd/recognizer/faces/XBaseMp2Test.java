package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.Set;
import org.junit.Test;

/**
 * Этап 2 MediaPipe (XBaseMp2): отказ отбором (d₂ = +∞ у отвергнутых проб – FeiScores.ratio даёт отказ), отбор строк проб и
 * частей cv для сверки с прежними файлами оценок.
 *
 * @author ssv
 */
public class XBaseMp2Test {

    static final String OWN = "m\t0\t0\town\tp1\t3\tp1\tp1\t0.5\tp2\t0.9\t0.5\t-\n";
    static final String CTRL = "m\t0\t0\tctrl\tc1\t2\t-\tp1\t0.7\tp2\t0.8\tInfinity\t-\n";
    static final String CV = "m\t0\t0\tcv0\tp3\t1\tp3\tp3\t0.1\tp1\t0.6\t0.1\t-\n";

    @Test
    public void refuseSetsSecondToInfinity() {
        String out = XBaseMp2.refuse(OWN + CTRL + CV, Set.of("own\tp1\t3"));
        String[] l = out.split("\n");
        assertEquals("Infinity", l[0].split("\t", -1)[10]);
        assertEquals(CTRL.trim(), l[1]);
        assertEquals(CV.trim(), l[2]);
        assertEquals(Double.POSITIVE_INFINITY, FeiScores.ratio(FeiScores.parse(l[0]), null), 0);
        assertEquals(0.5 / 0.9, FeiScores.ratio(FeiScores.parse(OWN.trim()), null), 1e-15);
    }

    @Test
    public void onlyKeepsChosenProbesAndCvRows() {
        assertEquals(OWN + CV, XBaseMp2.only(OWN + CTRL + CV, Set.of("own\tp1\t3")));
        assertEquals(CTRL + CV, XBaseMp2.only(OWN + CTRL + CV, Set.of("ctrl\tc1\t2")));
    }
}
