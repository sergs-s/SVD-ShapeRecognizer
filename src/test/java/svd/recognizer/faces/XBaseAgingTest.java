package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import java.util.Set;
import org.junit.Test;

/**
 * Кривая старения (часть 5, задание №4 от 16.10.2026): корзины интервала и «ближайшая дата шаблона».
 *
 * @author ssv
 */
public class XBaseAgingTest {

    @Test
    public void binsSplitAtThirtyOneEightyAndYear() {
        assertEquals(0, XBaseAging.bin(0));
        assertEquals(0, XBaseAging.bin(30));
        assertEquals(1, XBaseAging.bin(31));
        assertEquals(1, XBaseAging.bin(180));
        assertEquals(2, XBaseAging.bin(181));
        assertEquals(2, XBaseAging.bin(365));
        assertEquals(3, XBaseAging.bin(366));
    }

    @Test
    public void intervalIsToNearestTemplateDate() {
        // Шаблон 31.08.1993 и 28.01.1994; проба 19.05.1994 – ближе ко второй (111 дней).
        assertEquals(111, XBaseAging.interval("940519", Set.of("930831", "940128")));
        assertEquals(0, XBaseAging.interval("930831", Set.of("930831")));
        // Проба раньше шаблона – по модулю.
        assertEquals(30, XBaseAging.interval("930801", Set.of("930831")));
    }
}
