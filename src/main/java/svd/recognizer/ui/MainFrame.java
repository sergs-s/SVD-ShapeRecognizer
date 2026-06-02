package svd.recognizer.ui;

import java.io.File;
import java.util.Map;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.SpinnerNumberModel;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;
import svd.recognizer.processing.ImagePreprocessor;
import svd.recognizer.processing.SVDComputer;
import svd.recognizer.processing.ShapeRecognizer;
import svd.recognizer.processing.ShapeRecognizer.RecognitionResult;
import svd.recognizer.storage.TemplateRepository;

/**
 * Главное окно приложения.
 *
 * Назначение формы: 1. Загружать тестовое изображение фигуры 2. Запускать
 * preprocessing и распознавание 3. Отображать исходную фигуру, эталон и
 * результат preprocessing 4. Управлять порогом распознавания через JSpinner 5.
 * Открывать отдельную форму TemplatesFrame
 *
 * Вся работа с эталонами вынесена в TemplatesFrame, а MainFrame только
 * использует уже сохранённые эталоны для распознавания.
 *
 * @author ssv
 */
public class MainFrame extends javax.swing.JFrame {

    private final SVDComputer svdComputer;
    private final TemplateRepository repository;
    private final ImagePreprocessor preprocessor = new ImagePreprocessor();
    private final ShapeRecognizer recognizer = new ShapeRecognizer();
    private final RecognitionPanel recognitionPanel = new RecognitionPanel();
    private Map<ShapeClass, TemplateStore> stores;
    private File selectedImageFile;

    public MainFrame(SVDComputer svdComputer, TemplateRepository repository) {
        this.svdComputer = svdComputer;
        this.repository = repository;
        this.stores = repository.loadAll();
        initComponents();
        setLocationRelativeTo(null);
        recognizer.setThreshold(((Number) spinnerThreshold.getValue()).doubleValue());
        recognitionPanel.appendLog("Приложение запущено.");
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        lblTitle = new javax.swing.JLabel();
        panelHost = new javax.swing.JPanel();
        btnLoadImage = new javax.swing.JButton();
        btnRecognize = new javax.swing.JButton();
        btnTemplates = new javax.swing.JButton();
        btnExit = new javax.swing.JButton();
        lblThreshold = new javax.swing.JLabel();
        spinnerThreshold = new javax.swing.JSpinner();
        btnAutoThreshold = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("SVD Shape Recognizer");

        lblTitle.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        lblTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTitle.setText("SVD Shape Recognizer");

        btnLoadImage.setText("Load Image");
        btnLoadImage.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnLoadImageActionPerformed(evt);
            }
        });

        btnRecognize.setText("Recognize");
        btnRecognize.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnRecognizeActionPerformed(evt);
            }
        });

        btnTemplates.setText("Templates");
        btnTemplates.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnTemplatesActionPerformed(evt);
            }
        });

        btnExit.setText("Exit");
        btnExit.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnExitActionPerformed(evt);
            }
        });

        lblThreshold.setText("Threshold:");

        btnAutoThreshold.setText("Auto Threshold");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(lblTitle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(panelHost, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(btnLoadImage)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnRecognize)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnTemplates)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnAutoThreshold)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(lblThreshold)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(spinnerThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, 90, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 130, Short.MAX_VALUE)
                .addComponent(btnExit)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(lblTitle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(panelHost, javax.swing.GroupLayout.DEFAULT_SIZE, 295, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(btnLoadImage)
                    .addComponent(btnRecognize)
                    .addComponent(btnTemplates)
                    .addComponent(btnExit)
                    .addComponent(lblThreshold)
                    .addComponent(spinnerThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnAutoThreshold))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnLoadImageActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnLoadImageActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_btnLoadImageActionPerformed

    private void btnRecognizeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnRecognizeActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_btnRecognizeActionPerformed

    private void btnExitActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnExitActionPerformed
        System.exit(0);
    }//GEN-LAST:event_btnExitActionPerformed

    private void btnTemplatesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnTemplatesActionPerformed
        TemplatesFrame templatesFrame = new TemplatesFrame(this, svdComputer, repository);
        templatesFrame.setVisible(true);
    }//GEN-LAST:event_btnTemplatesActionPerformed

    private void onLoadImage() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            selectedImageFile = chooser.getSelectedFile();
            try {
                ImagePreprocessor.PreprocessResult result = preprocessor.preprocess(selectedImageFile);
                recognitionPanel.getSourceView().setImage(javax.imageio.ImageIO.read(selectedImageFile));
                recognitionPanel.getProcessedView().setImage(result.getImage());
                recognitionPanel.appendLog("Загружен файл: " + selectedImageFile.getName());
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, ex.getMessage(), "Ошибка", JOptionPane.ERROR_MESSAGE);
                recognitionPanel.appendLog("Ошибка загрузки: " + ex.getMessage());
            }
        }
    }

    private void onRecognize() {
        if (selectedImageFile == null) {
            JOptionPane.showMessageDialog(this, "Сначала загрузите изображение.", "Информация", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        try {
            recognitionPanel.getProgressBar().setIndeterminate(true);
            stores = repository.loadAll();
            ImagePreprocessor.PreprocessResult prep = preprocessor.preprocess(selectedImageFile);
            double[] signature = svdComputer.computeFeatures(prep.getMatrix());
            recognizer.setThreshold(((Number) spinnerThreshold.getValue()).doubleValue());
            RecognitionResult result = recognizer.recognize(signature, stores);

            if (result.isRecognized()) {
                TemplateStore store = stores.get(result.getShapeClass());
                recognitionPanel.getTemplateView().setImage(store.getAverageImage());
                recognitionPanel.appendLog("Результат: " + result.getShapeClass().getDisplayName());
                recognitionPanel.appendLog(String.format("Distance = %.6f, Threshold = %.6f", result.getDistance(), result.getThreshold()));
            } else {
                recognitionPanel.getTemplateView().setImage(null);
                recognitionPanel.appendLog("Фигура не распознана");
                recognitionPanel.appendLog(String.format("Distance = %.6f, Threshold = %.6f", result.getDistance(), result.getThreshold()));
            }
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, ex.getMessage(), "Ошибка", JOptionPane.ERROR_MESSAGE);
            recognitionPanel.appendLog("Ошибка распознавания: " + ex.getMessage());
        } finally {
            recognitionPanel.getProgressBar().setIndeterminate(false);
        }
    }

    private void onTemplates() {
        new TemplatesFrame(this, svdComputer, repository).setVisible(true);
    }

    private void onAutoThreshold() {
        stores = repository.loadAll();
        double threshold = recognizer.calculateAutoThreshold(stores);
        spinnerThreshold.setValue(threshold);
        recognizer.setThreshold(threshold);
        recognitionPanel.appendLog(String.format("Автоматический порог установлен: %.6f", threshold));
    }

    public void reloadStores() {
        stores = repository.loadAll();
        recognitionPanel.appendLog("Эталоны перечитаны из каталога templates.");
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnAutoThreshold;
    private javax.swing.JButton btnExit;
    private javax.swing.JButton btnLoadImage;
    private javax.swing.JButton btnRecognize;
    private javax.swing.JButton btnTemplates;
    private javax.swing.JLabel lblThreshold;
    private javax.swing.JLabel lblTitle;
    private javax.swing.JPanel panelHost;
    private javax.swing.JSpinner spinnerThreshold;
    // End of variables declaration//GEN-END:variables
}
