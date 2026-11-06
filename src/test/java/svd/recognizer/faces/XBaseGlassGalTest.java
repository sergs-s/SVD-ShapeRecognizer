package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.List;
import java.util.Random;
import org.junit.Test;

public class XBaseGlassGalTest {

    @Test
    public void chooseLargestNWithEnoughParticipants() {
        int[] m = {6, 6, 5, 5, 4, 3};
        int[] o = {1, 2, 1, 0, 1, 3};
        assertEquals(6, XBaseGlassGal.chooseN(m, o, 2));
        assertEquals(5, XBaseGlassGal.chooseN(m, o, 3));
        assertEquals(4, XBaseGlassGal.chooseN(m, o, 4));
        assertEquals(3, XBaseGlassGal.chooseN(m, o, 5));
        assertEquals(-1, XBaseGlassGal.chooseN(m, o, 6));
    }

    @Test
    public void pickVariants() {
        List<String> main = List.of("m1", "m2", "m3", "m4", "m5", "m6");
        List<String> other = List.of("o1", "o2", "o3", "o4");
        assertEquals(List.of("m1", "m2", "m3", "m4", "m5", "m6"), XBaseGlassGal.pick(main, other, 6, false));
        assertEquals(List.of("m1", "m2", "m3", "o1", "o2", "o3"), XBaseGlassGal.pick(main, other, 6, true));
        // другого меньше ⌊N/2⌋ – сколько есть, остальные – преобладающее
        assertEquals(List.of("m1", "m2", "m3", "m4", "m5", "o1"), XBaseGlassGal.pick(main, List.of("o1"), 6, true));
        // нечётное N: ⌊5/2⌋ = 2
        assertEquals(List.of("m1", "m2", "m3", "o1", "o2"), XBaseGlassGal.pick(main, other, 5, true));
    }

    @Test
    public void bothIsPrefixOfSingleWherePossible() {
        Random rnd = new Random(1);
        List<String> main = XBaseGlassGal.shuffled(List.of("a", "b", "c", "d", "e", "f", "g"), rnd);
        List<String> other = XBaseGlassGal.shuffled(List.of("x", "y", "z"), rnd);
        List<String> a = XBaseGlassGal.pick(main, other, 6, false);
        List<String> b = XBaseGlassGal.pick(main, other, 6, true);
        assertEquals(6, a.size());
        assertEquals(6, b.size());
        long common = b.stream().filter(a::contains).count();
        assertEquals(3, common);
    }

    @Test
    public void shuffleDoesNotDependOnInputOrder() {
        assertEquals(XBaseGlassGal.shuffled(List.of("a", "b", "c", "d"), new Random(5)), XBaseGlassGal.shuffled(List.of("d", "c", "b", "a"), new Random(5)));
        assertTrue(XBaseGlassGal.personNo("s07") == 7);
    }
}
