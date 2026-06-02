package svd.recognizer.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

/**
 * Универсальная панель отображения изображений.
 *
 * При отсутствии картинки выводится подпись "нет данных", чтобы
 * пользователь видел, что область работает, но пока не заполнена.
 *
 * @author ssv
 */
public class ImageView extends javax.swing.JPanel {

    private BufferedImage image;

    public ImageView() {
        setPreferredSize(new Dimension(180, 180));
        setBackground(Color.WHITE);
        setBorder(javax.swing.BorderFactory.createLineBorder(Color.LIGHT_GRAY));
    }

    public void setImage(BufferedImage image) {
        this.image = image;
        repaint();
    }

    public BufferedImage getImage() {
        return image;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image == null) {
            g.setColor(Color.GRAY);
            g.drawString("нет данных", getWidth() / 2 - 30, getHeight() / 2);
        } else {
            g.drawImage(image, 0, 0, getWidth(), getHeight(), this);
        }
    }
}
