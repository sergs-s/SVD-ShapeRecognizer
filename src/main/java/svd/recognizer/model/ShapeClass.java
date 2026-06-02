package svd.recognizer.model;

/**
 * Поддерживаемые классы геометрических фигур.
 *
 * Для каждого класса задаются:
 * 1. Человекочитаемое имя для GUI
 * 2. Имя dat-файла в корневом каталоге templates
 *
 * В проекте используются три класса фигур, указанные в ТЗ:
 * - Circle
 * - Triangle
 * - Rectangle
 *
 * @author ssv
 */
public enum ShapeClass {
    CIRCLE("Circle", "circle.dat"),
    TRIANGLE("Triangle", "triangle.dat"),
    RECTANGLE("Rectangle", "rectangle.dat");

    private final String displayName;
    private final String fileName;

    ShapeClass(String displayName, String fileName) {
        this.displayName = displayName;
        this.fileName = fileName;
    }

    public String getDisplayName() {
        return displayName;
    }

    public String getFileName() {
        return fileName;
    }
}