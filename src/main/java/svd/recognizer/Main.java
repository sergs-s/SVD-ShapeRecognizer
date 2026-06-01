package svd.recognizer;

import svd.recognizer.ui.MainFrame;
import javax.swing.SwingUtilities;

/**
 * Точка входа приложения.
 * Распознавание геометрических фигур методом SVD.
 *
 * @author SVD-ShapeRecognizer
 * @version 1.0
 */
public class Main {

    public static void main(String[] args) {
        // TODO: Инициализация OpenCV нативных библиотек
        // TODO: Запуск главного окна в потоке Swing (EDT)
        SwingUtilities.invokeLater(() -> {
            new MainFrame().setVisible(true);
        });
    }
}
