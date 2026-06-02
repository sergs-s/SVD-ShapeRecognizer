package svd.recognizer.model;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Хранилище эталонов одного класса фигур.
 *
 * Обязанности класса: 1. Хранить список загруженных шаблонов выбранного класса
 * 2. Автоматически пересчитывать усреднённый σ-вектор 3. Автоматически
 * пересчитывать усреднённое изображение 4. Поддерживать признак готовности
 * набора при количестве 5 и более
 *
 * После каждого добавления или сброса вызывается пересчёт усреднённых данных,
 * чтобы TemplatesPanel сразу обновляла счётчики и предпросмотр эталона.
 *
 * @author ssv
 */
public class TemplateStore implements Serializable {

    private static final long serialVersionUID = 1L;
    public static final int MIN_TEMPLATES = 5;

    private final ShapeClass shapeClass;
    private final List<Template> templates = new ArrayList<>();
    private double[] averageSingularValues;
    private transient BufferedImage averageImage;

    public TemplateStore(ShapeClass shapeClass) {
        this.shapeClass = shapeClass;
    }

    public void addTemplate(Template template) {
        templates.add(template);
        recalculateAverage();
    }

    public void clear() {
        templates.clear();
        averageSingularValues = null;
        averageImage = null;
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        recalculateAverage();
    }

    private void recalculateAverage() {
        if (templates.isEmpty()) {
            averageSingularValues = null;
            averageImage = null;
            return;
        }

        int featureCount = templates.get(0).getSingularValues().length;
        averageSingularValues = new double[featureCount];
        for (Template template : templates) {
            double[] values = template.getSingularValues();
            for (int i = 0; i < featureCount; i++) {
                averageSingularValues[i] += values[i];
            }
        }
        for (int i = 0; i < featureCount; i++) {
            averageSingularValues[i] /= templates.size();
        }

        BufferedImage sample = templates.get(0).getNormalizedImage();
        int width = sample.getWidth();
        int height = sample.getHeight();
        double[] acc = new double[width * height];
        for (Template template : templates) {
            BufferedImage image = scale(template.getNormalizedImage(), width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = image.getRGB(x, y) & 0xFF;
                    acc[y * width + x] += rgb;
                }
            }
        }

        averageImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (int) Math.round(acc[y * width + x] / templates.size());
                int rgb = (gray << 16) | (gray << 8) | gray;
                averageImage.setRGB(x, y, rgb);
            }
        }
    }

    private BufferedImage scale(BufferedImage source, int width, int height) {
        if (source.getWidth() == width && source.getHeight() == height) {
            return source;
        }
        BufferedImage target = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = target.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(source, 0, 0, width, height, null);
        g.dispose();
        return target;
    }

    public boolean isReady() {
        return templates.size() >= MIN_TEMPLATES;
    }

    public int getCount() {
        return templates.size();
    }

    public ShapeClass getShapeClass() {
        return shapeClass;
    }

    public double[] getAverageSingularValues() {
        return averageSingularValues;
    }

    public BufferedImage getAverageImage() {
        return averageImage;
    }

    public List<Template> getTemplates() {
        return Collections.unmodifiableList(templates);
    }
}
