package svd.recognizer.model;

/**
 * Перечисление поддерживаемых классов геометрических фигур.
 * Каждый класс содержит русское название для отображения в интерфейсе
 * и имя файла для сохранения эталонов на диск.
 */
public enum ShapeClass {

    TRIANGLE("Треугольник", "triangle.dat"),
    SQUARE("Квадрат",       "square.dat"),
    CIRCLE("Круг",          "circle.dat");

    private final String displayName;  // Название для отображения в UI
    private final String fileName;     // Имя .dat файла для хранения эталонов

    ShapeClass(String displayName, String fileName) {
        this.displayName = displayName;
        this.fileName    = fileName;
    }

    public String getDisplayName() { return displayName; }
    public String getFileName()    { return fileName;    }
}
