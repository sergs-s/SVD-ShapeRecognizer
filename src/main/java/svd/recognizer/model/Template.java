package svd.recognizer.model;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import javax.imageio.ImageIO;

/**
 * Один эталонный образец фигуры.
 *
 * Каждый эталон хранит:
 * 1. Нормализованный σ-вектор признаков
 * 2. Нормализованное изображение 64x64 для интерфейса и усреднения
 * 3. Путь к исходному файлу для трассировки происхождения данных
 * 4. Флаг низкого качества препроцессинга + причина
 *
 * Изображение сериализуется вручную через PNG-представление, чтобы dat-файл
 * содержал не только вектор признаков, но и сам нормализованный эталон.
 *
 * @author ssv
 */
public class Template implements Serializable {
    private static final long serialVersionUID = 2L;

    private double[] singularValues;
    private transient BufferedImage normalizedImage;
    private String sourceFilePath;
    private boolean lowQuality;
    private String qualityReason;

    public Template() {
    }

    public Template(double[] singularValues, BufferedImage normalizedImage,
                    String sourceFilePath) {
        this(singularValues, normalizedImage, sourceFilePath, false, "");
    }

    public Template(double[] singularValues, BufferedImage normalizedImage,
                    String sourceFilePath, boolean lowQuality, String qualityReason) {
        this.singularValues  = singularValues;
        this.normalizedImage = normalizedImage;
        this.sourceFilePath  = sourceFilePath;
        this.lowQuality      = lowQuality;
        this.qualityReason   = qualityReason;
    }

    public double[] getSingularValues()          { return singularValues; }
    public BufferedImage getNormalizedImage()     { return normalizedImage; }
    public String getSourceFilePath()            { return sourceFilePath; }
    public boolean isLowQuality()                { return lowQuality; }
    public String getQualityReason()             { return qualityReason; }

    public void setSingularValues(double[] singularValues)      { this.singularValues = singularValues; }
    public void setNormalizedImage(BufferedImage normalizedImage){ this.normalizedImage = normalizedImage; }
    public void setSourceFilePath(String sourceFilePath)        { this.sourceFilePath = sourceFilePath; }
    public void setLowQuality(boolean lowQuality)               { this.lowQuality = lowQuality; }
    public void setQualityReason(String qualityReason)          { this.qualityReason = qualityReason; }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        if (normalizedImage == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            ImageIO.write(normalizedImage, "png", out);
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        boolean hasImage = in.readBoolean();
        if (hasImage) {
            normalizedImage = ImageIO.read(in);
        }
    }
}
