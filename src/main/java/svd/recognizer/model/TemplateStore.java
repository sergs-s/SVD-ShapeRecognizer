package svd.recognizer.model;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Хранилище эталонов одного класса фигур.
 *
 * Хранит список загруженных шаблонов и кэшированный усреднённый σ-вектор.
 * Усреднённый σ-вектор пересчитывается автоматически при каждом добавлении
 * или сбросе эталонов и используется при распознавании.
 *
 * Усреднённое изображение НЕ хранится и НЕ строится: для отладки в UI
 * отображаются все загруженные образцы (SampleStripPanel в TemplatesPanel).
 *
 * @author ssv
 */
public class TemplateStore implements Serializable {

    private static final long serialVersionUID = 1L;
    public static final int MIN_TEMPLATES = 5;

    private final ShapeClass shapeClass;
    private final List<Template> templates = new ArrayList<>();
    private double[] averageSingularValues;

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
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        recalculateAverage();
    }

    private void recalculateAverage() {
        if (templates.isEmpty()) {
            averageSingularValues = null;
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

    public List<Template> getTemplates() {
        return Collections.unmodifiableList(templates);
    }
}
