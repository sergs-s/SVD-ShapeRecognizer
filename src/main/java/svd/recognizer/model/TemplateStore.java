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

    private static final long serialVersionUID = 2L;
    public static final int MIN_TEMPLATES = 5;

    private final ShapeClass shapeClass;
    private final List<Template> templates = new ArrayList<>();
    private double[] averageSingularValues;

    // модель подпространства класса (null, если класс не обучен)
    private SubspaceModel subspaceModel;

    public TemplateStore(ShapeClass shapeClass) {
        this.shapeClass = shapeClass;
    }

    /**
     * Добавляет эталон в класс и немедленно пересчитывает усреднённый σ-вектор.
     * Пересчёт «на лету» означает, что усреднённый портрет класса (по сути
     * результат обучения) всегда актуален и не требует отдельной операции.
     *
     * @param template новый эталонный образец
     */
    public void addTemplate(Template template) {
        templates.add(template);
        recalculateAverage();
        clearSubspaceModel(); // Сброс подпространства при изменении эталонов
    }

    /** Удаляет все эталоны класса и сбрасывает усреднённый σ-вектор. */
    public void clear() {
        templates.clear();
        averageSingularValues = null;
        clearSubspaceModel(); // Сброс подпространства при очистке
    }

    /**
     * При десериализации хранилища кэш усреднённого σ-вектора не пишется в
     * поток, поэтому пересчитывается заново после чтения списка эталонов.
     * Поле subspaceModel сериализуется стандартно (не transient).
     */
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        recalculateAverage();
    }

    /**
     * Пересчитывает усреднённый σ-вектор класса — покомпонентное среднее
     * арифметическое σ-векторов всех эталонов. Этот средний вектор служит
     * «портретом класса»: при распознавании расстояние неизвестной фигуры
     * измеряется именно до него. Длина берётся по первому эталону (у всех
     * эталонов одинаковое число признаков).
     */
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

    /**
     * @return true, если эталонов не меньше минимума (MIN_TEMPLATES) — класс
     *         пригоден для устойчивого усреднения и распознавания
     */
    public boolean isReady() {
        return templates.size() >= MIN_TEMPLATES;
    }

    public int getCount() {
        return templates.size();
    }

    public ShapeClass getShapeClass() {
        return shapeClass;
    }

    /**
     * @return усреднённый σ-вектор класса (портрет класса) или null, если
     *         эталонов нет
     */
    public double[] getAverageSingularValues() {
        return averageSingularValues;
    }

    public List<Template> getTemplates() {
        return Collections.unmodifiableList(templates);
    }

    /**
     * Устанавливает модель подпространства для класса.
     *
     * @param model обученная модель подпространства (может быть null)
     */
    public void setSubspaceModel(SubspaceModel model) {
        this.subspaceModel = model;
    }

    /**
     * @return текущая модель подпространства или null, если класс не обучен
     */
    public SubspaceModel getSubspaceModel() {
        return subspaceModel;
    }

    /**
     * Проверяет, обучен ли класс (имеет ли подпространство).
     *
     * @return true, если subspaceModel != null
     */
    public boolean isTrained() {
        return subspaceModel != null;
    }

    /**
     * Сбрасывает обученное подпространство — вызывается при изменении набора эталонов.
     */
    public void clearSubspaceModel() {
        this.subspaceModel = null;
    }
}