package svd.recognizer;

import javax.swing.SwingUtilities;
import nu.pattern.OpenCV;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.math.SvdEngine;
import svd.recognizer.processing.SVDComputer;
import svd.recognizer.storage.TemplateRepository;
import svd.recognizer.ui.MainFrame;

/**
 * Точка входа приложения распознавания геометрических фигур методом SVD.
 *
 * Последовательность запуска:
 * 1. Инициализация нативной библиотеки OpenCV
 * 2. Создание математического движка SVD через интерфейс SvdEngine
 * 3. Создание вычислителя SVD-признаков без жёсткой привязки к библиотеке
 * 4. Создание репозитория эталонов в каталоге templates корня проекта
 * 5. Запуск главного окна приложения в EDT потоке Swing
 *
 * Архитектурное решение зафиксировано окончательно:
 * - сейчас используется CommonsMathSvdEngine;
 * - SVDComputer работает через SvdEngine;
 * - GUI, TemplateRepository, ShapeRecognizer и ImagePreprocessor не зависят от конкретной SVD-библиотеки;
 * - в будущем можно добавить EjmlSvdEngine без переписывания верхнего уровня.
 *
 * @author ssv
 */
public class Main {
    public static void main(String[] args) {
        OpenCV.loadLocally();
        SvdEngine svdEngine = new CommonsMathSvdEngine();
        SVDComputer svdComputer = new SVDComputer(svdEngine);
        TemplateRepository repository = new TemplateRepository();
        SwingUtilities.invokeLater(() -> new MainFrame(svdComputer, repository).setVisible(true));
    }
}