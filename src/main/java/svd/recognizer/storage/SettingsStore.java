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
 * Хранит:
 * 1. Три порога распознавания для σ-режима (по одному на класс)
 * 2. Единый порог отвержения θ для subspace-режима
 * 3. Размерность подпространства k
 * 4. Текущий режим распознавания (SIGMA_VECTOR / SUBSPACE)
 *
 * Пороги не обнуляются между запусками: загружаются при старте,
 * сохраняются при каждом изменении (классический подход через
 * java.util.Properties).
 *
 * @author ssv
 */
public class SettingsStore {

    private static final String FILE_NAME = "settings.properties";
    private static final String KEY_PREFIX = "threshold.";
    private static final double DEFAULT_THRESHOLD = 0.35;

    // Константы для подпространств
    private static final String KEY_SUBSPACE_THRESHOLD = "subspace.threshold";
    private static final String KEY_SUBSPACE_K = "subspace.k";
    private static final String KEY_RECOGNITION_MODE = "recognition.mode";
    private static final double DEFAULT_SUBSPACE_THRESHOLD = 15.0;
    private static final int DEFAULT_SUBSPACE_K = 4;

    // Константы для лиц (оценка на базе ORL)
    private static final String KEY_FACES_DATASET_DIR = "faces.dataset.dir";
    private static final String KEY_FACES_SEED = "faces.seed";
    private static final String KEY_FACES_TRAIN_PER_PERSON = "faces.train.per.person";
    private static final String DEFAULT_FACES_DATASET_DIR = "D:\\data\\ORL";
    private static final long DEFAULT_FACES_SEED = 42L;
    private static final int DEFAULT_FACES_TRAIN_PER_PERSON = 5;

    /** @return путь к файлу settings.properties в корне проекта */
    private Path getPath() {
        return Paths.get(System.getProperty("user.dir"), FILE_NAME);
    }

    /**
     * Загружает Properties из файла. Если файл не существует,
     * возвращает пустой Properties.
     *
     * @return объект Properties
     */
    private Properties loadProperties() {
        Properties props = new Properties();
        Path path = getPath();
        if (Files.exists(path)) {
            try (FileInputStream in = new FileInputStream(path.toFile())) {
                props.load(in);
            } catch (IOException ignored) {
            }
        }
        return props;
    }

    /**
     * Сохраняет Properties в файл.
     *
     * @param props объект Properties для сохранения
     */
    private void saveProperties(Properties props) {
        try (FileOutputStream out = new FileOutputStream(getPath().toFile())) {
            props.store(out, "SVD Shape Recognizer settings");
        } catch (IOException ignored) {
        }
    }

    /**
     * Загружает пороги всех классов из settings.properties. Для отсутствующих
     * или некорректных значений подставляется порог по умолчанию.
     *
     * @return карта «класс → порог»
     */
    public Map<ShapeClass, Double> loadThresholds() {
        Map<ShapeClass, Double> map = new EnumMap<>(ShapeClass.class);
        Properties props = loadProperties();
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
        Properties props = loadProperties();
        for (Map.Entry<ShapeClass, Double> e : thresholds.entrySet()) {
            props.setProperty(KEY_PREFIX + e.getKey().name(),
                    Double.toString(e.getValue()));
        }
        saveProperties(props);
    }

    /**
     * Загружает порог отвержения для subspace-режима.
     *
     * @return значение порога (по умолчанию 15.0); единственный источник
     *         порога θ по умолчанию в проекте
     */
    public double loadSubspaceThreshold() {
        String raw = loadProperties().getProperty(KEY_SUBSPACE_THRESHOLD);
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
        Properties props = loadProperties();
        props.setProperty(KEY_SUBSPACE_THRESHOLD, Double.toString(theta));
        saveProperties(props);
    }

    /**
     * Загружает размерность подпространства k для subspace-режима.
     *
     * @return значение k (по умолчанию 4); единственный источник k по
     *         умолчанию в проекте
     */
    public int loadSubspaceK() {
        String raw = loadProperties().getProperty(KEY_SUBSPACE_K);
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
        Properties props = loadProperties();
        props.setProperty(KEY_SUBSPACE_K, Integer.toString(k));
        saveProperties(props);
    }

    /**
     * Загружает текущий режим распознавания.
     *
     * @return режим распознавания (по умолчанию SIGMA_VECTOR)
     */
    public RecognitionMode loadRecognitionMode() {
        String raw = loadProperties().getProperty(KEY_RECOGNITION_MODE);
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
        Properties props = loadProperties();
        props.setProperty(KEY_RECOGNITION_MODE, mode.name());
        saveProperties(props);
    }

    /**
     * Загружает путь к папке базы лиц ORL (структура sX/Y.pgm).
     *
     * @return путь к базе (по умолчанию D:\data\ORL)
     */
    public String loadFacesDatasetDir() {
        String raw = loadProperties().getProperty(KEY_FACES_DATASET_DIR);
        return (raw != null && !raw.isBlank()) ? raw : DEFAULT_FACES_DATASET_DIR;
    }

    /**
     * Загружает seed для воспроизводимых разбиений при оценке на лицах.
     *
     * @return seed (по умолчанию 42)
     */
    public long loadFacesSeed() {
        String raw = loadProperties().getProperty(KEY_FACES_SEED);
        if (raw != null) {
            try {
                return Long.parseLong(raw.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        return DEFAULT_FACES_SEED;
    }

    /**
     * Загружает число обучающих снимков на человека при оценке на лицах.
     *
     * @return число снимков (по умолчанию 5)
     */
    public int loadFacesTrainPerPerson() {
        String raw = loadProperties().getProperty(KEY_FACES_TRAIN_PER_PERSON);
        if (raw != null) {
            try {
                return Integer.parseInt(raw.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        return DEFAULT_FACES_TRAIN_PER_PERSON;
    }
}