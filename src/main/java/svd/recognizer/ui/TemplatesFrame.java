package svd.recognizer.ui;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileNameExtensionFilter;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;
import svd.recognizer.processing.ImagePreprocessor;
import svd.recognizer.processing.SVDComputer;
import svd.recognizer.storage.TemplateRepository;

/**
 * Отдельная форма управления эталонами.
 *
 * Назначение формы: 1. Загружать по одному файлу в выбранный класс фигур 2.
 * Загружать все JPEG из выбранной папки в выбранный класс фигур 3.
 * Сбрасывать шаблоны по каждому классу отдельно 4. Отображать все загруженные образцы
 * фигур и количество шаблонов 5. Обновлять интерфейс после каждого изменения данных
 *
 * Кнопка Templates на MainFrame открывает именно эту форму, а вся работа с
 * эталонами сосредоточена только здесь.
 *
 * @author ssv
 */
public class TemplatesFrame extends javax.swing.JFrame {

    private static final String LEARNING_DATA_DIR = "learningData";

    private final MainFrame owner;
    private final SVDComputer svdComputer;
    private final TemplateRepository repository;
    private final ImagePreprocessor preprocessor = new ImagePreprocessor();
    private Map<ShapeClass, TemplateStore> stores;

    public TemplatesFrame(MainFrame owner, SVDComputer svdComputer, TemplateRepository repository) {
        this.owner = owner;
        this.svdComputer = svdComputer;
        this.repository = repository;
        this.stores = repository.loadAll();
        initComponents();
        setLocationRelativeTo(owner);
        bindActions();
        refreshPanel();
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        lblTitle = new javax.swing.JLabel();
        templatesPanel = new svd.recognizer.ui.TemplatesPanel();
        btnClose = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Templates");

        lblTitle.setFont(new java.awt.Font("Segoe UI", 1, 16)); // NOI18N
        lblTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTitle.setText("Shape Templates");

        btnClose.setText("Close");
        btnClose.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnCloseActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(lblTitle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(templatesPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap(730, Short.MAX_VALUE)
                .addComponent(btnClose)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(lblTitle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(templatesPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, Short.MAX_VALUE)
                .addComponent(btnClose)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnCloseActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnCloseActionPerformed
        owner.reloadStores();
        dispose();
    }//GEN-LAST:event_btnCloseActionPerformed

    /**
     * Привязывает кнопки панели к действиям: добавление эталона (файл/папка) и
     * сброс — отдельно для каждого из трёх классов фигур.
     */
    private void bindActions() {
        templatesPanel.getBtnAddCircle().addActionListener(e -> addTemplate(ShapeClass.CIRCLE));
        templatesPanel.getBtnAddTriangle().addActionListener(e -> addTemplate(ShapeClass.TRIANGLE));
        templatesPanel.getBtnAddRectangle().addActionListener(e -> addTemplate(ShapeClass.RECTANGLE));
        templatesPanel.getBtnResetCircle().addActionListener(e -> resetStore(ShapeClass.CIRCLE));
        templatesPanel.getBtnResetTriangle().addActionListener(e -> resetStore(ShapeClass.TRIANGLE));
        templatesPanel.getBtnResetRectangle().addActionListener(e -> resetStore(ShapeClass.RECTANGLE));
        templatesPanel.getBtnAddFolderCircle().addActionListener(e -> addTemplatesFromFolder(ShapeClass.CIRCLE));
        templatesPanel.getBtnAddFolderTriangle().addActionListener(e -> addTemplatesFromFolder(ShapeClass.TRIANGLE));
        templatesPanel.getBtnAddFolderRectangle().addActionListener(e -> addTemplatesFromFolder(ShapeClass.RECTANGLE));
    }

    // -------------------------------------------------------------------------
    // Загрузка одного файла
    // -------------------------------------------------------------------------

    /** @return диалог выбора одного JPEG-файла, открытый в каталоге learningData */
    private JFileChooser createJpegChooser() {
        JFileChooser chooser = new JFileChooser();
        File dir = new File(System.getProperty("user.dir"), LEARNING_DATA_DIR);
        if (!dir.exists()) dir = new File(System.getProperty("user.dir"));
        chooser.setCurrentDirectory(dir);
        chooser.setAcceptAllFileFilterUsed(false);
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setMultiSelectionEnabled(false);
        chooser.setFileFilter(new FileNameExtensionFilter("JPEG images (*.jpg, *.jpeg)", "jpg", "jpeg"));
        return chooser;
    }

    /**
     * Добавляет один эталон в указанный класс: препроцессинг файла → при низком
     * качестве запрос подтверждения у пользователя → вычисление σ-вектора →
     * сохранение Template в хранилище класса и на диск.
     * @param shapeClass класс, в который добавляется эталон
     */
    private void addTemplate(ShapeClass shapeClass) {
        JFileChooser chooser = createJpegChooser();
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File file = chooser.getSelectedFile();
        try {
            ImagePreprocessor.PreprocessResult result = preprocessor.preprocess(file, shapeClass);

            if (result.isLowQuality()) {
                String shapeRu = shapeClassToRussian(shapeClass);
                String msg = "Не удалось качественно распознать шаблон (" + shapeRu + "):\n"
                        + result.getQualityReason() + "\n\n"
                        + "Добавить шаблон всё равно?";
                int choice = JOptionPane.showConfirmDialog(
                        this, msg, "Низкое качество изображения",
                        JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
                if (choice != JOptionPane.YES_OPTION) {
                    return;
                }
            }

            double[] signature = svdComputer.computeFeatures(result.getMatrix());
            Template template = new Template(
                    signature, result.getImage(), file.getAbsolutePath(),
                    result.isLowQuality(), result.getQualityReason());

            template.setNormalizedMatrix(result.getMatrix());

            stores.get(shapeClass).addTemplate(template);
            repository.save(stores.get(shapeClass));
            refreshPanel();
        } catch (Exception ex) {
            ex.printStackTrace();
            String shortMsg = ex.getMessage() != null
                    ? ex.getMessage().split("\n")[0] : ex.getClass().getSimpleName();
            JOptionPane.showMessageDialog(this, shortMsg, "Ошибка", JOptionPane.ERROR_MESSAGE);
        }
    }

    // -------------------------------------------------------------------------
    // Загрузка папки
    // -------------------------------------------------------------------------

    /** @return диалог выбора папки с JPEG-файлами для пакетной загрузки */
    private JFileChooser createFolderChooser() {
        JFileChooser chooser = new JFileChooser();
        File dir = new File(System.getProperty("user.dir"), LEARNING_DATA_DIR);
        if (!dir.exists()) dir = new File(System.getProperty("user.dir"));
        chooser.setCurrentDirectory(dir);
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        chooser.setAcceptAllFileFilterUsed(false);
        return chooser;
    }

    /**
     * Пакетно добавляет все JPEG из выбранной папки в класс. Файлы берутся в
     * воспроизводимом порядке (по имени), каждый препроцессится и превращается в
     * эталон; в конце один раз сохраняется хранилище и показывается сводка
     * (добавлено / ошибки / образцы низкого качества).
     * @param shapeClass класс, в который добавляются эталоны
     */
    private void addTemplatesFromFolder(ShapeClass shapeClass) {
        JFileChooser chooser = createFolderChooser();
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File folder = chooser.getSelectedFile();
        File[] files = folder.listFiles(f ->
            f.isFile() && (f.getName().toLowerCase().endsWith(".jpg")
                        || f.getName().toLowerCase().endsWith(".jpeg")));

        if (files == null || files.length == 0) {
            JOptionPane.showMessageDialog(this,
                "В папке не найдено JPEG-файлов.",
                "Нет файлов", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        Arrays.sort(files);   // воспроизводимый порядок (по имени)

        int added = 0;
        int errors = 0;
        List<String> lowQualityNames = new ArrayList<>();

        for (File file : files) {
            try {
                ImagePreprocessor.PreprocessResult result =
                    preprocessor.preprocess(file, shapeClass);

                if (result.isLowQuality()) {
                    lowQualityNames.add(file.getName() + " — " + result.getQualityReason());
                }

                double[] signature = svdComputer.computeFeatures(result.getMatrix());
                Template template = new Template(
                    signature, result.getImage(), file.getAbsolutePath(),
                    result.isLowQuality(), result.getQualityReason());

                template.setNormalizedMatrix(result.getMatrix());

                stores.get(shapeClass).addTemplate(template);
                added++;

            } catch (Exception ex) {
                ex.printStackTrace();
                errors++;
            }
        }

        // Один save и один refresh после всего цикла
        try {
            repository.save(stores.get(shapeClass));
        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(this,
                "Ошибка сохранения: " + ex.getMessage(),
                "Ошибка", JOptionPane.ERROR_MESSAGE);
        }
        refreshPanel();

        // Сводный диалог
        StringBuilder msg = new StringBuilder();
        msg.append("Добавлено: ").append(added).append("\n");
        if (errors > 0)
            msg.append("Ошибок:    ").append(errors).append("\n");
        if (!lowQualityNames.isEmpty()) {
            msg.append("\nНизкое качество (").append(lowQualityNames.size()).append("):\n");
            for (String s : lowQualityNames)
                msg.append("  \u2022 ").append(s).append("\n");
        }
        JOptionPane.showMessageDialog(this,
            msg.toString().trim(),
            "Загрузка папки завершена",
            lowQualityNames.isEmpty() && errors == 0
                ? JOptionPane.INFORMATION_MESSAGE
                : JOptionPane.WARNING_MESSAGE);
    }

    // -------------------------------------------------------------------------
    // Сброс и вспомогательные методы
    // -------------------------------------------------------------------------

    /**
     * Очищает класс (удаляет все эталоны) и сохраняет пустое хранилище.
     *
     * @param shapeClass класс для очистки
     */
    private void resetStore(ShapeClass shapeClass) {
        try {
            stores.get(shapeClass).clear();
            repository.save(stores.get(shapeClass));
            refreshPanel();
        } catch (Exception ex) {
            ex.printStackTrace();
            String shortMsg = ex.getMessage() != null
                ? ex.getMessage().split("\n")[0] : ex.getClass().getSimpleName();
            JOptionPane.showMessageDialog(this, shortMsg, "Ошибка", JOptionPane.ERROR_MESSAGE);
        }
    }

    /** Перерисовывает панель эталонов согласно текущему содержимому хранилищ. */
    private void refreshPanel() {
        templatesPanel.refresh(stores);
    }

    /**
     * @param sc класс фигуры
     * @return русское название класса для сообщений пользователю
     */
    private static String shapeClassToRussian(ShapeClass sc) {
        if (sc == null) return "неизвестная фигура";
        switch (sc) {
            case CIRCLE:    return "круг";
            case TRIANGLE:  return "треугольник";
            case RECTANGLE: return "прямоугольник";
            default:        return sc.name();
        }
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnClose;
    private javax.swing.JLabel lblTitle;
    private svd.recognizer.ui.TemplatesPanel templatesPanel;
    // End of variables declaration//GEN-END:variables
}
