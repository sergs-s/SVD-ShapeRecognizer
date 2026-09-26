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
    private static final String KEY_FACES_FRAME_WIDTH = "faces.frame.width";
    private static final String KEY_FACES_FRAME_HEIGHT = "faces.frame.height";
    private static final int DEFAULT_FACES_FRAME_WIDTH = 92;
    private static final int DEFAULT_FACES_FRAME_HEIGHT = 112;
    private static final String KEY_FACES_DETECTOR_SCORE = "faces.detector.score";
    private static final float DEFAULT_FACES_DETECTOR_SCORE = 0.8f;
    private static final String KEY_FACES_MODELS_DIR = "faces.models.dir";
    private static final String DEFAULT_FACES_MODELS_DIR = "D:\\data\\models\\opencv_zoo";
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
     * Загружает ширину кадра лица (загрузка ORL и кадр своего аффинного
     * выравнивания).
     *
     * @return ширина в пикселях (по умолчанию 92 — родная ширина ORL)
     * @throws IllegalArgumentException если значение не положительное
     */
    public int loadFacesFrameWidth() {
        return positiveInt(KEY_FACES_FRAME_WIDTH, DEFAULT_FACES_FRAME_WIDTH);
    }

    /**
     * Загружает высоту кадра лица (загрузка ORL и кадр своего аффинного
     * выравнивания).
     *
     * @return высота в пикселях (по умолчанию 112 — родная высота ORL)
     * @throws IllegalArgumentException если значение не положительное
     */
    public int loadFacesFrameHeight() {
        return positiveInt(KEY_FACES_FRAME_HEIGHT, DEFAULT_FACES_FRAME_HEIGHT);
    }

    /**
     * Загружает порог уверенности детектора лиц YuNet (FaceDetectorYN).
     * Выбран на ORL (0,8; до шага 5 было 0,9); для своих фото выбирается заново.
     *
     * @return порог в интервале (0, 1) (по умолчанию 0,8)
     * @throws IllegalArgumentException если значение не число или вне (0, 1)
     */
    public float loadFacesDetectorScore() {
        String raw = loadProperties().getProperty(KEY_FACES_DETECTOR_SCORE);
        if (raw == null || raw.isBlank()) {
            return DEFAULT_FACES_DETECTOR_SCORE;
        }
        float value;
        try {
            value = Float.parseFloat(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(KEY_FACES_DETECTOR_SCORE + " должно быть числом, получено «" + raw + "»", e);
        }
        if (!(value > 0f && value < 1f)) {
            throw new IllegalArgumentException(KEY_FACES_DETECTOR_SCORE + " должно быть в интервале (0, 1), получено " + value);
        }
        return value;
    }

    private int positiveInt(String key, int defaultValue) {
        String raw = loadProperties().getProperty(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        int value;
        try {
            value = Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " должно быть целым числом, получено «" + raw + "»", e);
        }
        if (value <= 0) {
            throw new IllegalArgumentException(key + " должно быть положительным, получено " + value);
        }
        return value;
    }

    /**
     * Загружает путь к папке моделей opencv_zoo (YuNet, SFace; ONNX, вне git).
     *
     * @return путь к моделям (по умолчанию D:\data\models\opencv_zoo)
     */
    public String loadFacesModelsDir() {
        String raw = loadProperties().getProperty(KEY_FACES_MODELS_DIR);
        return (raw != null && !raw.isBlank()) ? raw : DEFAULT_FACES_MODELS_DIR;
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