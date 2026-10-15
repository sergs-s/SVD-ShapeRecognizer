package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Схемы W (задание №8 от 15.10.2026): классы обучения MLDA – (Wа) все люди галереи по стартовым сессиям (новичок – 1 дата,
 * старожил – 2), (н0) – все по 2 датам, (Wб) – своих в обучении нет.
 *
 * @author ssv
 */
public class XBaseWStatusTest {

    private static final Map<String, List<List<String>>> POOL = Map.of(
            "00001", List.of(List.of("00001_930831_fa", "00001_930831_fb"), List.of("00001_940128_fa", "00001_940128_fb"), List.of("00001_940519_fa")),
            "00002", List.of(List.of("00002_930831_fa", "00002_930831_fb"), List.of("00002_940307_fa", "00002_940307_fb")));

    @Test
    public void schemeWaPutsEveryoneInTrainingWithStartSessions() {
        Map<String, List<String>> t = XBaseWStatus.train(List.of("00001", "00002"), POOL, Set.of("00002"), false);
        assertEquals(List.of("00001", "00002"), List.copyOf(t.keySet()));
        assertEquals(List.of("00001_930831_fa", "00001_930831_fb", "00001_940128_fa", "00001_940128_fb"), t.get("00001"));
        // Новичок – только 1-я дата.
        assertEquals(List.of("00002_930831_fa", "00002_930831_fb"), t.get("00002"));
    }

    @Test
    public void schemeN0UsesTwoDatesForAllAndWbNobody() {
        Map<String, List<String>> n0 = XBaseWStatus.train(List.of("00001", "00002"), POOL, Set.of(), false);
        assertEquals(4, n0.get("00002").size());
        assertTrue(XBaseWStatus.train(List.of("00001", "00002"), POOL, Set.of("00002"), true).isEmpty());
    }
}
