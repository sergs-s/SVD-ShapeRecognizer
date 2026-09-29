package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * Очистка каталога экспорта и сборка README.md (FarExport) на временном каталоге (без OpenCV и баз).
 *
 * @author ssv
 */
public class FarExportTest {

    @Test
    public void cleanKeepsReferenceAndOtherFiles() throws Exception {
        Path dir = Files.createTempDirectory("farexport");
        try {
            Files.createDirectories(dir.resolve("a/own/X"));
            Files.writeString(dir.resolve("a/own/X/1.png"), "png");
            Files.createDirectories(dir.resolve("reference/FEI/set"));
            Files.writeString(dir.resolve("reference/FEI/set/1.jpg"), "jpg");
            Files.writeString(dir.resolve("reference/far_eval_32dc1fa.txt"), "эталон\r\n");
            Files.writeString(dir.resolve("MANIFEST.txt"), "m");
            Files.writeString(dir.resolve("README.md"), "старый", StandardCharsets.UTF_8);
            Files.writeString(dir.resolve("other.txt"), "o");
            String old = FarExport.clean(dir);
            assertEquals("старый", old);
            assertFalse(Files.exists(dir.resolve("a")));
            assertFalse(Files.exists(dir.resolve("MANIFEST.txt")));
            assertFalse(Files.exists(dir.resolve("README.md")));
            assertEquals("jpg", Files.readString(dir.resolve("reference/FEI/set/1.jpg")));
            assertEquals("эталон\r\n", Files.readString(dir.resolve("reference/far_eval_32dc1fa.txt")));
            assertTrue(Files.exists(dir.resolve("other.txt")));
        } finally {
            try (Stream<Path> s = Files.walk(dir)) {
                for (Path p : s.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
            }
        }
    }

    @Test
    public void readmeWithoutMarkersKeepsReferenceParagraphs() {
        String gen = "# SVD-faces-data\n\n- `reference/FEI/` — справочные наборы.\n";
        String old = "# SVD-faces-data\n\nстарый текст\n\n- `reference/FEI/` — справочные наборы.\n\n"
                + "`reference/far_eval_32dc1fa.txt` — эталон far_eval.txt.\n";
        String merged = FarExport.mergeReadme(old, gen);
        assertEquals(FarExport.README_BEGIN + "\n" + gen + FarExport.README_END + "\n"
                + "\n`reference/far_eval_32dc1fa.txt` — эталон far_eval.txt.\n", merged);
        // Повторный экспорт: заменяется только генерируемая часть, ручной текст — как был.
        String gen2 = "# SVD-faces-data\n\nновый\n";
        assertEquals(FarExport.README_BEGIN + "\n" + gen2 + FarExport.README_END + "\n"
                + "\n`reference/far_eval_32dc1fa.txt` — эталон far_eval.txt.\n", FarExport.mergeReadme(merged, gen2));
        assertEquals(FarExport.README_BEGIN + "\n" + gen2 + FarExport.README_END + "\n", FarExport.mergeReadme(null, gen2));
    }
}
