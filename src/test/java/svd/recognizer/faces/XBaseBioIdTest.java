package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;
import svd.recognizer.faces.XBaseBioId.Run;

public class XBaseBioIdTest {

    @Test
    public void runsAreConsecutiveNumbersOfOneGroup() {
        Map<String, List<Run>> r = XBaseBioId.runs(new int[] {2, 2, 1, 2, 2, 2});
        assertEquals(List.of(new Run(0, 1), new Run(3, 5)), r.get("A02"));
        assertEquals(List.of(new Run(2, 2)), r.get("A01"));
    }

    @Test
    public void registrationIsFirstRunOfTenOrMore() {
        assertEquals(1, XBaseBioId.regRun(List.of(new Run(0, 7), new Run(20, 29), new Run(40, 80))));
        assertEquals(-1, XBaseBioId.regRun(List.of(new Run(0, 8))));
    }

    @Test
    public void galleryIsEvenAndReplacedByNextNumber() {
        Set<Integer> all = new TreeSet<>();
        for (int i = 100; i <= 150; i++) all.add(i);
        assertEquals(List.of(100, 110, 120, 130, 140, 150), XBaseBioId.gallery(new Run(100, 150), all));
        all.remove(110);
        all.remove(111);
        assertEquals(List.of(100, 112, 120, 130, 140, 150), XBaseBioId.gallery(new Run(100, 150), all));
        all.remove(150);
        assertNull(XBaseBioId.gallery(new Run(100, 150), all));
    }

    @Test
    public void splitFirstHalfIsCeilAfterShuffle() {
        List<String> own = new ArrayList<>();
        for (int i = 15; i >= 1; i--) own.add(XBaseBioId.person(i));
        List<String> p = new ArrayList<>(new TreeSet<>(own));
        Collections.shuffle(p, new Random(42 + 13000));
        List<List<String>> h = XBaseBioId.split(own, 42 + 13000);
        assertEquals(8, h.get(0).size());
        assertEquals(new TreeSet<>(p.subList(0, 8)), new TreeSet<>(h.get(0)));
        assertEquals(new TreeSet<>(p.subList(8, 15)), new TreeSet<>(h.get(1)));
    }

    @Test
    public void runLabelIsRelativeToRegistration() {
        assertEquals("до", XBaseBioId.runLabel(0, 1));
        assertEquals("1", XBaseBioId.runLabel(1, 1));
        assertEquals("2", XBaseBioId.runLabel(2, 1));
        assertEquals("3+", XBaseBioId.runLabel(4, 1));
        assertEquals("-", XBaseBioId.runLabel(0, -1));
    }

    @Test
    public void verdictFollowsSectionFour() {
        assertEquals(true, XBaseBioId.verdict(30, 20, 40).endsWith(": ближе к межсессионному"));
        assertEquals(true, XBaseBioId.verdict(3, 1, 8).endsWith(": ближе к внутрисессионному"));
        assertEquals(true, XBaseBioId.verdict(12, 6, 20).endsWith(": между"));
        assertEquals(true, XBaseBioId.verdict(15, 2, 30).endsWith("не различается"));
        assertEquals(true, XBaseBioId.verdict(30, 26, 40).endsWith(": ближе к межсессионному"));
    }

    @Test
    public void partsUseSeedOfSection1() {
        List<String> own = new ArrayList<>();
        for (int i = 0; i < 8; i++) own.add(XBaseBioId.person(i + 1));
        List<String> perm = new ArrayList<>(own);
        Collections.shuffle(perm, new Random(42 + 13010 + 2));
        List<List<String>> p = XBaseBioId.parts(own, 42, 2);
        assertEquals(5, p.size());
        assertEquals(perm.subList(0, 2), p.get(0));
    }
}
