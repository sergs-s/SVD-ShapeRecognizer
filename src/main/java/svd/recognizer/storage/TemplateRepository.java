package svd.recognizer.storage;

import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Сохранение и загрузка эталонов на диск.
 *
 * Каждый класс фигуры хранится в отдельном бинарном файле .dat
 * в папке templates/ рядом с JAR-файлом приложения.
 *
 * Формат: Java Object Serialization (ObjectOutputStream / ObjectInputStream)
 * Файлы: triangle.dat, square.dat, circle.dat
 */
public class TemplateRepository {

    /** Папка для хранения .dat файлов */
    private static final String TEMPLATES_DIR = "templates";

    /**
     * Сохранить TemplateStore для одного класса фигур.
     *
     * @param store  хранилище эталонов
     * @throws IOException при ошибке записи
     */
    public void save(TemplateStore store) throws IOException {
        // TODO:
        //   Path path = getFilePath(store.getShapeClass());
        //   Files.createDirectories(path.getParent());
        //   try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path.toFile()))) {
        //       oos.writeObject(store);
        //   }
    }

    /**
     * Загрузить TemplateStore для одного класса фигур.
     * Если файл не существует — вернуть пустой TemplateStore.
     *
     * @param shapeClass  класс фигуры
     * @return            загруженное или новое хранилище
     */
    public TemplateStore load(ShapeClass shapeClass) {
        // TODO:
        //   Path path = getFilePath(shapeClass);
        //   if (!Files.exists(path)) return new TemplateStore(shapeClass);
        //   try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path.toFile()))) {
        //       return (TemplateStore) ois.readObject();
        //   }
        return new TemplateStore(shapeClass);
    }

    /**
     * Загрузить эталоны для всех трёх классов.
     *
     * @return  массив из трёх TemplateStore (TRIANGLE, SQUARE, CIRCLE)
     */
    public TemplateStore[] loadAll() {
        // TODO: for (ShapeClass sc : ShapeClass.values()) load(sc)
        return new TemplateStore[]{
            new TemplateStore(ShapeClass.TRIANGLE),
            new TemplateStore(ShapeClass.SQUARE),
            new TemplateStore(ShapeClass.CIRCLE)
        };
    }

    /**
     * Удалить .dat файл для указанного класса (сброс эталонов).
     *
     * @param shapeClass  класс фигуры
     */
    public void delete(ShapeClass shapeClass) {
        // TODO: Files.deleteIfExists(getFilePath(shapeClass))
    }

    /** Вернуть путь к .dat файлу для данного класса */
    private Path getFilePath(ShapeClass shapeClass) {
        return Paths.get(TEMPLATES_DIR, shapeClass.getFileName());
    }
}
