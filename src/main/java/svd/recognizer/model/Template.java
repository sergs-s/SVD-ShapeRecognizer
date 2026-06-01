package svd.recognizer.model;

import java.awt.image.BufferedImage;
import java.io.Serializable;

/**
 * Один эталонный образец фигуры.
 * Содержит сингулярный вектор (σ-вектор), полученный после предобработки
 * и SVD-разложения изображения, а также само нормализованное изображение 64×64
 * для отображения в интерфейсе.
 */
public class Template implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Сингулярные числа (первые N значений после SVD) */
    private double[] singularValues;

    /** Нормализованное бинарное изображение 64×64 пикселя */
    private transient BufferedImage normalizedImage;

    /** Исходный путь к файлу (для информации) */
    private String sourceFilePath;

    public Template() { }

    public Template(double[] singularValues, BufferedImage normalizedImage, String sourceFilePath) {
        // TODO: присвоить поля
    }

    // TODO: геттеры и сеттеры
    public double[] getSingularValues()       { return singularValues;  }
    public BufferedImage getNormalizedImage() { return normalizedImage; }
    public String getSourceFilePath()        { return sourceFilePath;  }

    public void setSingularValues(double[] sv)      { this.singularValues  = sv;  }
    public void setNormalizedImage(BufferedImage img){ this.normalizedImage = img; }
    public void setSourceFilePath(String path)      { this.sourceFilePath  = path;}
}
