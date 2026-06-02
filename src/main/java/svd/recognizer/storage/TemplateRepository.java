package svd.recognizer.storage;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumMap;
import java.util.Map;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;

/**
 * Репозиторий эталонов в каталоге templates корня проекта.
 *
 * Правила хранения:
 * 1. Для каждого класса фигур используется отдельный бинарный dat-файл
 * 2. Если файл отсутствует или пуст, создаётся новый пустой TemplateStore
 * 3. Верхний уровень не зависит от конкретной SVD-библиотеки, потому что
 *    здесь хранятся уже вычисленные эталоны и усреднённые данные
 *
 * Используемые файлы:
 * - circle.dat
 * - triangle.dat
 * - rectangle.dat
 *
 * @author ssv
 */
public class TemplateRepository {
    private static final String TEMPLATES_DIR = "templates";

    public void save(TemplateStore store) throws Exception {
        Path path = getPath(store.getShapeClass());
        Files.createDirectories(path.getParent());
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path.toFile()))) {
            oos.writeObject(store);
        }
    }

    public TemplateStore load(ShapeClass shapeClass) {
        Path path = getPath(shapeClass);
        try {
            if (!Files.exists(path) || Files.size(path) == 0) {
                return new TemplateStore(shapeClass);
            }
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path.toFile()))) {
                Object object = ois.readObject();
                if (object instanceof TemplateStore store) {
                    return store;
                }
            }
        } catch (EOFException ignored) {
            return new TemplateStore(shapeClass);
        } catch (Exception ignored) {
            return new TemplateStore(shapeClass);
        }
        return new TemplateStore(shapeClass);
    }

    public Map<ShapeClass, TemplateStore> loadAll() {
        Map<ShapeClass, TemplateStore> map = new EnumMap<>(ShapeClass.class);
        for (ShapeClass shapeClass : ShapeClass.values()) {
            map.put(shapeClass, load(shapeClass));
        }
        return map;
    }

    public void reset(ShapeClass shapeClass) throws Exception {
        Path path = getPath(shapeClass);
        Files.createDirectories(path.getParent());
        if (Files.exists(path)) {
            Files.delete(path);
        }
        Files.createFile(path);
    }

    private Path getPath(ShapeClass shapeClass) {
        return Paths.get(TEMPLATES_DIR, shapeClass.getFileName());
    }
}