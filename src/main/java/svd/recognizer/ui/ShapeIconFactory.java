package svd.recognizer.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import svd.recognizer.model.ShapeClass;

/**
 * Рисует «идеальные» контурные эталоны фигур (чёрная линия на белом фоне) и
 * значок «не распознано» (красный наклонный крест) для панели Template.
 *
 * Изображения рисуются программно при каждом запросе — средние эталонные
 * картинки не хранятся: усреднение реальных контуров даёт визуальный мусор
 * (наложение разных фигур) и на распознавание не влияет, поэтому для индикации
 * класса достаточно нарисовать канонический контур.
 *
 * @author ssv
 */
public final class ShapeIconFactory {

    private static final int SIZE = 180;
    private static final int MARGIN = 30;
    private static final float STROKE = 3f;

    private ShapeIconFactory() {
    }

    /**
     * Контурная идеальная фигура заданного класса: круг, квадрат или
     * равнобедренный треугольник на широком основании.
     *
     * @param shapeClass класс фигуры для отрисовки
     * @return контурное изображение идеальной фигуры (чёрным на белом)
     */
    public static BufferedImage createShapeIcon(ShapeClass shapeClass) {
        BufferedImage img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, SIZE, SIZE);
        g.setColor(Color.BLACK);
        g.setStroke(new BasicStroke(STROKE));

        int min = MARGIN;
        int max = SIZE - MARGIN;
        int side = max - min;

        switch (shapeClass) {
            case CIRCLE:
                g.drawOval(min, min, side, side);
                break;
            case RECTANGLE:
                g.drawRect(min, min, side, side);
                break;
            case TRIANGLE:
                int[] xs = {SIZE / 2, min, max};
                int[] ys = {min, max, max};
                g.drawPolygon(xs, ys, 3);
                break;
            default:
                break;
        }
        g.dispose();
        return img;
    }

    /**
     * Значок «не распознано»: красный наклонный крест среднего размера по
     * центру белого поля.
     *
     * @return изображение с красным наклонным крестом (значок «не распознано»)
     */
    public static BufferedImage createNotRecognizedIcon() {
        BufferedImage img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, SIZE, SIZE);
        g.setColor(Color.RED);
        g.setStroke(new BasicStroke(6f));

        int a = SIZE / 3;
        int b = SIZE - a;
        g.drawLine(a, a, b, b);
        g.drawLine(b, a, a, b);
        g.dispose();
        return img;
    }
}
