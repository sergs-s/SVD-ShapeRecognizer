package svd.recognizer.model;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Хранилище эталонов для одного класса фигур.
 * Содержит список загруженных эталонов и кэшированный усреднённый σ-вектор.
 * При добавлении или удалении эталона усреднение пересчитывается автоматически.
 */
public class TemplateStore implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Минимальное количество эталонов для активации распознавания */
    public static final int MIN_TEMPLATES = 5;

    private final ShapeClass shapeClass;
    private final List<Template> templates = new ArrayList<>();

    /** Кэшированный усреднённый σ-вектор (пересчитывается при изменении списка) */
    private double[] averageSingularValues;

    /** Кэшированное усреднённое изображение (пересчитывается при изменении списка) */
    private transient BufferedImage averageImage;

    public TemplateStore(ShapeClass shapeClass) {
        this.shapeClass = shapeClass;
    }

    /** Добавить эталон и пересчитать среднее */
    public void addTemplate(Template template) {
        // TODO: templates.add(template); recalculateAverage();
    }

    /** Удалить все эталоны данного класса */
    public void clear() {
        // TODO: templates.clear(); averageSingularValues = null; averageImage = null;
    }

    /** Пересчитать усреднённый σ-вектор и усреднённое изображение */
    private void recalculateAverage() {
        // TODO: поэлементное среднее по всем templates
    }

    /** Проверка: достаточно ли эталонов для распознавания */
    public boolean isReady() {
        return templates.size() >= MIN_TEMPLATES;
    }

    public int getCount()                      { return templates.size();        }
    public ShapeClass getShapeClass()          { return shapeClass;              }
    public double[] getAverageSingularValues() { return averageSingularValues;   }
    public BufferedImage getAverageImage()     { return averageImage;            }
    public List<Template> getTemplates()       { return templates;               }
}
