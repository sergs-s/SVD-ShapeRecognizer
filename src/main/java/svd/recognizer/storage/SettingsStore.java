package svd.recognizer.storage;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumMap;
import java.util.Map;
import java.util.Properties;
import svd.recognizer.model.ShapeClass;

/**
 * Хранилище пользовательских настроек в файле settings.properties в корне
 * проекта (рядом с каталогами templates и learningData).
 *
 * Сейчас хранит три порога распознавания — по одному на класс фигур. Пороги
 * не обнуляются между запусками: загружаются при старте, сохраняются при
 * каждом изменении (классический подход через java.util.Properties).
 *
 * @author ssv
 */
public class SettingsStore {

    private static final String FILE_NAME = "settings.properties";
    private static final String KEY_PREFIX = "threshold.";
    private static final double DEFAULT_THRESHOLD = 0.35;

    /** @return путь к файлу settings.properties в корне проекта */
    private Path getPath() {
        return Paths.get(System.getProperty("user.dir"), FILE_NAME);
    }

    /**
     * Загружает пороги всех классов из settings.properties. Для отсутствующих
     * или некорректных значений подставляется порог по умолчанию.
     *
     * @return карта «класс → порог»
     */
    public Map<ShapeClass, Double> loadThresholds() {
        Map<ShapeClass, Double> map = new EnumMap<>(ShapeClass.class);
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        for (ShapeClass sc : ShapeClass.values()) {
            double value = DEFAULT_THRESHOLD;
            String raw = props.getProperty(KEY_PREFIX + sc.name());
            if (raw != null) {
                try {
                    value = Double.parseDouble(raw);
                } catch (NumberFormatException ignored) {
                }
            }
            map.put(sc, value);
        }
        return map;
    }

    /**
     * Сохраняет пороги всех классов в settings.properties в корне проекта.
     *
     * @param thresholds карта «класс → порог»
     */
    public void saveThresholds(Map<ShapeClass, Double> thresholds) {
        Properties props = new Properties();
        for (Map.Entry<ShapeClass, Double> e : thresholds.entrySet()) {
            props.setProperty(KEY_PREFIX + e.getKey().name(),
                    Double.toString(e.getValue()));
        }
        try (FileOutputStream out = new FileOutputStream(getPath().toFile())) {
            props.store(out, "SVD Shape Recognizer settings");
        } catch (IOException ignored) {
        }
    }
}
