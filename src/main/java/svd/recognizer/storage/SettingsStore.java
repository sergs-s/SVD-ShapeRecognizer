package svd.recognizer.storage;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
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
    private static final String KEY_FACES_FRAME_WIDTH = "faces.frame.width";
    private static final String KEY_FACES_FRAME_HEIGHT = "faces.frame.height";
    private static final int DEFAULT_FACES_FRAME_WIDTH = 92;
    private static final int DEFAULT_FACES_FRAME_HEIGHT = 112;
    private static final String KEY_FACES_DETECTOR_SCORE = "faces.detector.score";
    private static final float DEFAULT_FACES_DETECTOR_SCORE = 0.8f;
    private static final String KEY_FACES_OWN_DETECTOR_SCORE = "faces.own.detector.score";
    private static final float DEFAULT_FACES_OWN_DETECTOR_SCORE = 0.7f;
    private static final String KEY_FACES_GT_DIR = "faces.gt.dir";
    private static final String KEY_FACES_MUCT_DIR = "faces.muct.dir";
    private static final String KEY_FACES_OWN_DIR = "faces.own.dir";
    private static final String KEY_FACES_MODELS_DIR = "faces.models.dir";
    private static final String KEY_FACES_EXPORT_DIR = "faces.export.dir";
    private static final String KEY_FACES_FEI_DIR = "faces.fei.dir";
    private static final String KEY_FACES_FEI_MIRROR = "faces.fei.mirror";
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
     * @return путь к базе (переменная окружения FACES_DATASET_DIR или settings.properties)
     * @throws IllegalStateException если путь не задан
     */
    public String loadFacesDatasetDir() {
        return requiredPath(KEY_FACES_DATASET_DIR);
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
        return unitInterval(KEY_FACES_DETECTOR_SCORE, DEFAULT_FACES_DETECTOR_SCORE);
    }

    /**
     * Загружает порог уверенности YuNet для своей базы (отдельно от ORL).
     * Выбран на своей базе (сетка 0,5…0,9, шаг 5): 0,7.
     *
     * @return порог в интервале (0, 1) (по умолчанию 0,7)
     * @throws IllegalArgumentException если значение не число или вне (0, 1)
     */
    public float loadFacesOwnDetectorScore() {
        return unitInterval(KEY_FACES_OWN_DETECTOR_SCORE, DEFAULT_FACES_OWN_DETECTOR_SCORE);
    }

    private float unitInterval(String key, float defaultValue) {
        String raw = loadProperties().getProperty(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        float value;
        try {
            value = Float.parseFloat(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " должно быть числом, получено «" + raw + "»", e);
        }
        if (!(value > 0f && value < 1f)) {
            throw new IllegalArgumentException(key + " должно быть в интервале (0, 1), получено " + value);
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
     * Загружает путь к Georgia Tech Face Database (папки s01…s50, вне git).
     *
     * @return путь (FACES_GT_DIR или settings.properties)
     * @throws IllegalStateException если путь не задан
     */
    public String loadFacesGtDir() {
        return requiredPath(KEY_FACES_GT_DIR);
    }

    /**
     * Загружает путь к снимкам MUCT (папка jpg, вне git).
     *
     * @return путь (FACES_MUCT_DIR или settings.properties)
     * @throws IllegalStateException если путь не задан
     */
    public String loadFacesMuctDir() {
        return requiredPath(KEY_FACES_MUCT_DIR);
    }

    /**
     * Загружает путь к своей базе лиц (папки людей с JPEG, вне git).
     *
     * @return путь к базе (FACES_OWN_DIR или settings.properties)
     * @throws IllegalStateException если путь не задан
     */
    public String loadFacesOwnDir() {
        return requiredPath(KEY_FACES_OWN_DIR);
    }

    /**
     * Загружает путь к папке моделей opencv_zoo (YuNet, SFace; ONNX, вне git).
     *
     * @return путь к моделям (FACES_MODELS_DIR или settings.properties)
     * @throws IllegalStateException если путь не задан
     */
    public String loadFacesModelsDir() {
        return requiredPath(KEY_FACES_MODELS_DIR);
    }

    /**
     * Загружает путь к экспорту выровненных кадров (репозиторий данных SVD-faces-data): FarExport пишет туда,
     * GalleryEvaluation при заданном ключе читает все базы оттуда. Ключ необязательный.
     *
     * @return путь (FACES_EXPORT_DIR или settings.properties) или null, если не задан
     */
    public String loadFacesExportDir() {
        return path(KEY_FACES_EXPORT_DIR);
    }

    /**
     * Загружает путь к FEI Face Database (originalimages, справочные наборы; вне git). Ключ необязательный: без него
     * FEI в faces-far не подключается.
     *
     * @return путь (FACES_FEI_DIR или settings.properties) или null, если не задан
     */
    public String loadFacesFeiDir() {
        return path(KEY_FACES_FEI_DIR);
    }

    /**
     * Режим зеркальных копий в обучении faces-fei (строки +mirror, этап 5б): ключ faces.fei.mirror или переменная
     * окружения FACES_FEI_MIRROR. По умолчанию выключен — набор строк и отчёты прежние.
     *
     * @return true, если значение — «true» (без учёта регистра)
     */
    public boolean loadFacesFeiMirror() {
        String raw = path(KEY_FACES_FEI_MIRROR);
        return raw != null && Boolean.parseBoolean(raw.trim());
    }

    /**
     * Путь по ключу: переменная окружения (ключ заглавными, точки — подчёркивания: faces.gt.dir →
     * FACES_GT_DIR), иначе settings.properties; null — не задан нигде.
     */
    private String path(String key) {
        String env = System.getenv(envName(key));
        if (env != null && !env.isBlank()) {
            return env;
        }
        String raw = loadProperties().getProperty(key);
        return (raw != null && !raw.isBlank()) ? raw : null;
    }

    /** Обязательный путь; незаданный — ошибка, путей по умолчанию нет. */
    private String requiredPath(String key) {
        String value = path(key);
        if (value == null) {
            throw new IllegalStateException("Не задан путь " + key + " (settings.properties) и переменная окружения "
                    + envName(key));
        }
        return value;
    }

    /** Имя переменной окружения для ключа: faces.gt.dir → FACES_GT_DIR. */
    static String envName(String key) {
        return key.toUpperCase(Locale.ROOT).replace('.', '_');
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

    // ---------------------------------------------------------------- шаг 5, отсечение чужих (FarMethods)

    private static final String KEY_FACES_FAR_METHODS = "faces.far.methods";
    private static final String KEY_FACES_CLAHE_CLIP = "faces.clahe.clip";
    private static final double DEFAULT_FACES_CLAHE_CLIP = 2.0;
    private static final String KEY_FACES_CLAHE_TILE = "faces.clahe.tile";
    private static final int DEFAULT_FACES_CLAHE_TILE = 8;
    private static final String KEY_FACES_TT_GAMMA = "faces.tt.gamma";
    private static final double DEFAULT_FACES_TT_GAMMA = 0.2;
    private static final String KEY_FACES_TT_SIGMA0 = "faces.tt.sigma0";
    private static final double DEFAULT_FACES_TT_SIGMA0 = 1.0;
    private static final String KEY_FACES_TT_SIGMA1 = "faces.tt.sigma1";
    private static final double DEFAULT_FACES_TT_SIGMA1 = 2.0;
    private static final String KEY_FACES_TT_ALPHA = "faces.tt.alpha";
    private static final double DEFAULT_FACES_TT_ALPHA = 0.1;
    private static final String KEY_FACES_TT_TAU = "faces.tt.tau";
    private static final double DEFAULT_FACES_TT_TAU = 10.0;

    /**
     * Значение по ключу: переменная окружения (правило {@link #envName}), иначе settings.properties;
     * null — не задано нигде.
     *
     * @param key ключ settings.properties
     * @return значение или null
     */
    public String setting(String key) {
        return path(key);
    }

    /**
     * Загружает список конфигураций FarMethods (faces.far.methods / FACES_FAR_METHODS, через запятую).
     *
     * @return идентификаторы; пустой список — все конфигурации
     */
    public List<String> loadFacesFarMethods() {
        String raw = setting(KEY_FACES_FAR_METHODS);
        List<String> list = new ArrayList<>();
        if (raw != null) {
            for (String s : raw.split(",")) {
                if (!s.isBlank()) list.add(s.trim());
            }
        }
        return list;
    }

    /** @return предел усиления контраста CLAHE (по умолчанию 2,0) */
    public double loadFacesClaheClip() {
        return positiveDouble(KEY_FACES_CLAHE_CLIP, DEFAULT_FACES_CLAHE_CLIP);
    }

    /** @return число клеток CLAHE по стороне (по умолчанию 8) */
    public int loadFacesClaheTile() {
        String raw = setting(KEY_FACES_CLAHE_TILE);
        if (raw == null) return DEFAULT_FACES_CLAHE_TILE;
        int value;
        try {
            value = Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(KEY_FACES_CLAHE_TILE + " должно быть целым числом, получено «" + raw + "»", e);
        }
        if (value <= 0) {
            throw new IllegalArgumentException(KEY_FACES_CLAHE_TILE + " должно быть положительным, получено " + value);
        }
        return value;
    }

    /** @return показатель гамма-коррекции Tan – Triggs (по умолчанию 0,2) */
    public double loadFacesTtGamma() {
        return positiveDouble(KEY_FACES_TT_GAMMA, DEFAULT_FACES_TT_GAMMA);
    }

    /** @return σ₀ разности гауссианов Tan – Triggs (по умолчанию 1,0) */
    public double loadFacesTtSigma0() {
        return positiveDouble(KEY_FACES_TT_SIGMA0, DEFAULT_FACES_TT_SIGMA0);
    }

    /** @return σ₁ разности гауссианов Tan – Triggs (по умолчанию 2,0) */
    public double loadFacesTtSigma1() {
        return positiveDouble(KEY_FACES_TT_SIGMA1, DEFAULT_FACES_TT_SIGMA1);
    }

    /** @return показатель α выравнивания контраста Tan – Triggs (по умолчанию 0,1) */
    public double loadFacesTtAlpha() {
        return positiveDouble(KEY_FACES_TT_ALPHA, DEFAULT_FACES_TT_ALPHA);
    }

    /** @return порог τ выравнивания контраста Tan – Triggs (по умолчанию 10) */
    public double loadFacesTtTau() {
        return positiveDouble(KEY_FACES_TT_TAU, DEFAULT_FACES_TT_TAU);
    }

    private double positiveDouble(String key, double defaultValue) {
        String raw = setting(key);
        if (raw == null) return defaultValue;
        double value;
        try {
            value = Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " должно быть числом, получено «" + raw + "»", e);
        }
        if (!(value > 0)) {
            throw new IllegalArgumentException(key + " должно быть положительным, получено " + value);
        }
        return value;
    }
}