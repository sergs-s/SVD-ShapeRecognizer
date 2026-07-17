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
import svd.recognizer.model.RecognitionMode;
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

    // НОВЫЕ КОНСТАНТЫ ДЛЯ SUBSPACE
    private static final String KEY_SUBSPACE_THRESHOLD = "subspace.threshold";
    private static final String KEY_SUBSPACE_K = "subspace.k";
    private static final String KEY_RECOGNITION_MODE = "recognition.mode";
    private static final double DEFAULT_SUBSPACE_THRESHOLD = 13.0;
    private static final int DEFAULT_SUBSPACE_K = 4;

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

    // ========================================================================
    // НОВЫЕ МЕТОДЫ ДЛЯ SUBSPACE
    // ========================================================================

    /**
     * Загружает порог отвержения для subspace-режима.
     *
     * @return значение порога (по умолчанию 13.0)
     */
    public double loadSubspaceThreshold() {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        String raw = props.getProperty(KEY_SUBSPACE_THRESHOLD);
        if (raw != null) {
            try {
                return Double.parseDouble(raw);
            } catch (NumberFormatException ignored) {
            }
        }
        return DEFAULT_SUBSPACE_THRESHOLD;
    }

    /**
     * Сохраняет порог отвержения для subspace-режима.
     *
     * @param theta значение порога
     */
    public void saveSubspaceThreshold(double theta) {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        props.setProperty(KEY_SUBSPACE_THRESHOLD, Double.toString(theta));
        try (FileOutputStream out = new FileOutputStream(path.toFile())) {
            props.store(out, "SVD Shape Recognizer settings");
        } catch (IOException ignored) {
        }
    }

    /**
     * Загружает размерность подпространства k для subspace-режима.
     *
     * @return значение k (по умолчанию 4)
     */
    public int loadSubspaceK() {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        String raw = props.getProperty(KEY_SUBSPACE_K);
        if (raw != null) {
            try {
                return Integer.parseInt(raw);
            } catch (NumberFormatException ignored) {
            }
        }
        return DEFAULT_SUBSPACE_K;
    }

    /**
     * Сохраняет размерность подпространства k для subspace-режима.
     *
     * @param k значение размерности
     */
    public void saveSubspaceK(int k) {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        props.setProperty(KEY_SUBSPACE_K, Integer.toString(k));
        try (FileOutputStream out = new FileOutputStream(path.toFile())) {
            props.store(out, "SVD Shape Recognizer settings");
        } catch (IOException ignored) {
        }
    }

    /**
     * Загружает текущий режим распознавания.
     *
     * @return режим распознавания (по умолчанию SIGMA_VECTOR)
     */
    public RecognitionMode loadRecognitionMode() {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        String raw = props.getProperty(KEY_RECOGNITION_MODE);
        if (raw != null) {
            try {
                return RecognitionMode.valueOf(raw);
            } catch (IllegalArgumentException ignored) {
            }
        }
        return RecognitionMode.SIGMA_VECTOR;
    }

    /**
     * Сохраняет текущий режим распознавания.
     *
     * @param mode режим распознавания
     */
    public void saveRecognitionMode(RecognitionMode mode) {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        props.setProperty(KEY_RECOGNITION_MODE, mode.name());
        try (FileOutputStream out = new FileOutputStream(path.toFile())) {
            props.store(out, "SVD Shape Recognizer settings");
        } catch (IOException ignored) {
        }
    }
}