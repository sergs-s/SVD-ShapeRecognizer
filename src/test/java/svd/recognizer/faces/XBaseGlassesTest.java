package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public class XBaseGlassesTest {

    @Test
    public void groupRules() {
        assertEquals(XBaseGlasses.MATCH, XBaseGlasses.group("1", List.of("0", "1")));
        assertEquals(XBaseGlasses.CHANGE, XBaseGlasses.group("1", List.of("0", "0")));
        assertEquals(XBaseGlasses.EXCLUDED, XBaseGlasses.group("?", List.of("0", "1")));
        assertEquals(XBaseGlasses.EXCLUDED, XBaseGlasses.group("1", List.of("?", "?")));
        assertEquals(XBaseGlasses.CHANGE, XBaseGlasses.group("1", Arrays.asList("?", "0")));
    }

    private static Path tsv(String text) throws IOException {
        Path f = Files.createTempFile("gt_glasses", ".tsv");
        Files.writeString(f, text, StandardCharsets.UTF_8);
        return f;
    }

    @Test
    public void readsValues() throws IOException {
        Path f = tsv("человек\tфайл\tдата_exif\tочки\ns01\t01.jpg\t1999-07-15\t1\ns01\t02.jpg\t1999-08-05\t?\n");
        Map<String, String> m = XBaseGlasses.readGt(f);
        assertEquals("1", m.get(Paths.get("gt", "s01", "01").toString()));
        assertEquals("?", m.get(Paths.get("gt", "s01", "02").toString()));
        Files.delete(f);
    }

    @Test
    public void emptyValueNamesTheLine() throws IOException {
        Path f = tsv("человек\tфайл\tдата_exif\tочки\ns01\t01.jpg\t1999-07-15\t1\ns01\t02.jpg\t1999-08-05\t\n");
        try {
            XBaseGlasses.readGt(f);
            fail("пустое значение должно быть ошибкой");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("строка 3"));
        }
        Files.delete(f);
    }

    @Test
    public void wrongValueIsError() throws IOException {
        Path f = tsv("человек\tфайл\tдата_exif\tочки\ns01\t01.jpg\t1999-07-15\tда\n");
        try {
            XBaseGlasses.readGt(f);
            fail("недопустимое значение должно быть ошибкой");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("строка 2"));
        }
        Files.delete(f);
    }
}
