package svd.recognizer.ui;

import java.io.File;
import java.util.EnumMap;
import java.util.Map;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.SpinnerNumberModel;
import javax.swing.filechooser.FileNameExtensionFilter;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;
import svd.recognizer.processing.ImagePreprocessor;
import svd.recognizer.processing.SVDComputer;
import svd.recognizer.processing.ShapeRecognizer;
import svd.recognizer.processing.ShapeRecognizer.RecognitionResult;
import svd.recognizer.storage.SettingsStore;
import svd.recognizer.storage.TemplateRepository;

/**
 * Главное окно приложения.
 *
 * Назначение:
 * 1. Загрузка тестового изображения фигуры.
 * 2. Распознавание по схеме «Путь B»: фигура прогоняется через все три ветки
 *    обработки (круг, треугольник, прямоугольник) и сравнивается с эталоном
 *    каждого класса в его системе координат; выбирается ближайший класс,
 *    прошедший свой порог.
 * 3. Отображение исходной фигуры, идеального эталона распознанного класса и
 *    результата preprocessing 64x64 (картинка той ветки, что победила).
 * 4. Три независимых порога (по классу) через три JSpinner; кнопки ×1.0/×1.5/×2.0
 *    выставляют пороги от внутриклассовой статистики. Значения сохраняются в
 *    settings.properties и не обнуляются между запусками.
 * 5. Открытие формы управления эталонами (TemplatesFrame).
 *
 * @author ssv
 */
public class MainFrame extends javax.swing.JFrame {

    private static final String LEARNING_DATA_DIR = "learningData";

    private final SVDComputer svdComputer;
    private final TemplateRepository repository;
    private final SettingsStore settingsStore = new SettingsStore();
    private final ImagePreprocessor preprocessor = new ImagePreprocessor();
    private final ShapeRecognizer recognizer = new ShapeRecognizer();
    private Map<ShapeClass, TemplateStore> stores;
    private File selectedImageFile;

    public MainFrame(SVDComputer svdComputer, TemplateRepository repository) {
        this.svdComputer = svdComputer;
        this.repository = repository;
        this.stores = repository.loadAll();
        initComponents();
        setLocationRelativeTo(null);
        loadThresholdsIntoUi();
        recognitionPanel.appendLog("Приложение запущено.");
    }

    /** Загружает сохранённые пороги из settings.properties в спиннеры и recognizer. */
    private void loadThresholdsIntoUi() {
        Map<ShapeClass, Double> saved = settingsStore.loadThresholds();
        spinnerCircle.setValue(saved.get(ShapeClass.CIRCLE));
        spinnerTriangle.setValue(saved.get(ShapeClass.TRIANGLE));
        spinnerRectangle.setValue(saved.get(ShapeClass.RECTANGLE));
        applySpinnersToRecognizer();
    }

    /** Переносит значения трёх спиннеров в recognizer. */
    private void applySpinnersToRecognizer() {
        recognizer.setThreshold(ShapeClass.CIRCLE, doubleValue(spinnerCircle));
        recognizer.setThreshold(ShapeClass.TRIANGLE, doubleValue(spinnerTriangle));
        recognizer.setThreshold(ShapeClass.RECTANGLE, doubleValue(spinnerRectangle));
    }

    /** Сохраняет текущие три порога в settings.properties. */
    private void persistThresholds() {
        Map<ShapeClass, Double> map = new EnumMap<>(ShapeClass.class);
        map.put(ShapeClass.CIRCLE, doubleValue(spinnerCircle));
        map.put(ShapeClass.TRIANGLE, doubleValue(spinnerTriangle));
        map.put(ShapeClass.RECTANGLE, doubleValue(spinnerRectangle));
        settingsStore.saveThresholds(map);
    }

    private double doubleValue(javax.swing.JSpinner spinner) {
        return ((Number) spinner.getValue()).doubleValue();
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        lblTitle = new javax.swing.JLabel();
        recognitionPanel = new RecognitionPanel();
        btnLoadImage = new javax.swing.JButton();
        btnRecognize = new javax.swing.JButton();
        btnTemplates = new javax.swing.JButton();
        btnExit = new javax.swing.JButton();
        lblCircle = new javax.swing.JLabel();
        spinnerCircle = new javax.swing.JSpinner();
        lblTriangle = new javax.swing.JLabel();
        spinnerTriangle = new javax.swing.JSpinner();
        lblRectangle = new javax.swing.JLabel();
        spinnerRectangle = new javax.swing.JSpinner();
        btnMul10 = new javax.swing.JButton();
        btnMul15 = new javax.swing.JButton();
        btnMul20 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("SVD Shape Recognizer");

        lblTitle.setFont(new java.awt.Font("Segoe UI", 1, 18));
        lblTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTitle.setText("SVD Shape Recognizer");

        btnLoadImage.setText("Load Image");
        btnRecognize.setText("Recognize");
        btnTemplates.setText("Templates");
        btnExit.setText("Exit");

        lblCircle.setText("Circle:");
        spinnerCircle.setModel(new SpinnerNumberModel(0.35d, 0.01d, 10.0d, 0.01d));
        lblTriangle.setText("Triangle:");
        spinnerTriangle.setModel(new SpinnerNumberModel(0.35d, 0.01d, 10.0d, 0.01d));
        lblRectangle.setText("Rectangle:");
        spinnerRectangle.setModel(new SpinnerNumberModel(0.35d, 0.01d, 10.0d, 0.01d));

        btnMul10.setText("\u00d71.0");
        btnMul15.setText("\u00d71.5");
        btnMul20.setText("\u00d72.0");

        btnLoadImage.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnLoadImageActionPerformed(evt);
            }
        });
        btnRecognize.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnRecognizeActionPerformed(evt);
            }
        });
        btnTemplates.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnTemplatesActionPerformed(evt);
            }
        });
        btnExit.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnExitActionPerformed(evt);
            }
        });
        btnMul10.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnMul10ActionPerformed(evt);
            }
        });
        btnMul15.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnMul15ActionPerformed(evt);
            }
        });
        btnMul20.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnMul20ActionPerformed(evt);
            }
        });
        spinnerCircle.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                spinnerThresholdStateChanged(evt);
            }
        });
        spinnerTriangle.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                spinnerThresholdStateChanged(evt);
            }
        });
        spinnerRectangle.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                spinnerThresholdStateChanged(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(lblTitle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(recognitionPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(btnLoadImage)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnRecognize)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnTemplates)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnExit)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(lblCircle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(spinnerCircle, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(lblTriangle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(spinnerTriangle, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(lblRectangle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(spinnerRectangle, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(btnMul10)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnMul15)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnMul20)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(lblTitle)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(recognitionPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(btnLoadImage)
                    .addComponent(btnRecognize)
                    .addComponent(btnTemplates)
                    .addComponent(btnExit))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblCircle)
                    .addComponent(spinnerCircle, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(lblTriangle)
                    .addComponent(spinnerTriangle, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(lblRectangle)
                    .addComponent(spinnerRectangle, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnMul10)
                    .addComponent(btnMul15)
                    .addComponent(btnMul20))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private JFileChooser createJpegChooser() {
        JFileChooser chooser = new JFileChooser();
        File dir = new File(System.getProperty("user.dir"), LEARNING_DATA_DIR);
        if (!dir.exists()) {
            dir = new File(System.getProperty("user.dir"));
        }
        chooser.setCurrentDirectory(dir);
        chooser.setAcceptAllFileFilterUsed(false);
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setMultiSelectionEnabled(false);
        chooser.setFileFilter(new FileNameExtensionFilter("JPEG images (*.jpg, *.jpeg)", "jpg", "jpeg"));
        return chooser;
    }

    private void btnLoadImageActionPerformed(java.awt.event.ActionEvent evt) {
        JFileChooser chooser = createJpegChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            selectedImageFile = chooser.getSelectedFile();
            try {
                recognitionPanel.getSourceView().setImage(javax.imageio.ImageIO.read(selectedImageFile));
                recognitionPanel.getProcessedView().setImage(null);
                recognitionPanel.getTemplateView().setImage(null);
                recognitionPanel.appendLog("Загружен файл: " + selectedImageFile.getName());
            } catch (Exception ex) {
                ex.printStackTrace();
                String shortMsg = ex.getMessage() != null
                        ? ex.getMessage().split("\n")[0] : ex.getClass().getSimpleName();
                JOptionPane.showMessageDialog(this, shortMsg, "Ошибка", JOptionPane.ERROR_MESSAGE);
                recognitionPanel.appendLog("Ошибка загрузки: " + shortMsg);
            }
        }
    }

    private void btnRecognizeActionPerformed(java.awt.event.ActionEvent evt) {
        if (selectedImageFile == null) {
            JOptionPane.showMessageDialog(this, "Сначала загрузите изображение.", "Информация", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        // Распознавание выполняется в фоновом потоке (SwingWorker), чтобы интерфейс
        // не зависал и прогресс-бар реально анимировался во время обработки.
        final File fileToRecognize = selectedImageFile;
        setControlsEnabled(false);
        recognitionPanel.getProgressBar().setIndeterminate(true);
        recognitionPanel.setResult(" ", java.awt.Color.BLACK);
        recognitionPanel.appendLog("Распознавание...");

        javax.swing.SwingWorker<RecognitionResult, Void> worker =
                new javax.swing.SwingWorker<RecognitionResult, Void>() {

            private final Map<ShapeClass, java.awt.image.BufferedImage> processedImages =
                    new EnumMap<>(ShapeClass.class);

            @Override
            protected RecognitionResult doInBackground() throws Exception {
                stores = repository.loadAll();
                applySpinnersToRecognizer();

                // Путь B: обрабатываем фигуру каждой веткой и считаем σ-вектор гипотезы.
                Map<ShapeClass, double[]> hypotheses = new EnumMap<>(ShapeClass.class);
                for (ShapeClass sc : ShapeClass.values()) {
                    ImagePreprocessor.PreprocessResult prep =
                            preprocessor.preprocess(fileToRecognize, sc);
                    hypotheses.put(sc, svdComputer.computeFeatures(prep.getMatrix()));
                    processedImages.put(sc, prep.getImage());
                }
                return recognizer.recognize(hypotheses, stores);
            }

            @Override
            protected void done() {
                try {
                    RecognitionResult result = get();
                    if (result.isRecognized()) {
                        ShapeClass winner = result.getShapeClass();
                        // Processed 64x64 — картинка ветки-победителя (правильная ориентация).
                        recognitionPanel.getProcessedView().setImage(processedImages.get(winner));
                        // Template — идеальный контурный эталон победившего класса.
                        recognitionPanel.getTemplateView().setImage(ShapeIconFactory.createShapeIcon(winner));
                        // Подпись результата под изображениями.
                        recognitionPanel.setResult(winner.getDisplayName(), new java.awt.Color(0, 128, 0));
                        recognitionPanel.appendLog("Результат: " + winner.getDisplayName());
                        recognitionPanel.appendLog(String.format("Distance = %.6f, Threshold = %.6f",
                                result.getDistance(), result.getThreshold()));
                    } else {
                        recognitionPanel.getProcessedView().setImage(null);
                        recognitionPanel.getTemplateView().setImage(ShapeIconFactory.createNotRecognizedIcon());
                        recognitionPanel.setResult("Не распознано", java.awt.Color.RED);
                        recognitionPanel.appendLog("Фигура не распознана");
                        recognitionPanel.appendLog(String.format(
                                "Ближайшее расстояние = %.6f, порог класса = %.6f",
                                result.getDistance(), result.getThreshold()));
                    }
                } catch (Exception ex) {
                    Throwable cause = ex.getCause() != null ? ex.getCause() : ex;
                    cause.printStackTrace();
                    String shortMsg = cause.getMessage() != null
                            ? cause.getMessage().split("\n")[0] : cause.getClass().getSimpleName();
                    recognitionPanel.setResult("Ошибка", java.awt.Color.RED);
                    JOptionPane.showMessageDialog(MainFrame.this, shortMsg, "Ошибка", JOptionPane.ERROR_MESSAGE);
                    recognitionPanel.appendLog("Ошибка распознавания: " + shortMsg);
                } finally {
                    recognitionPanel.getProgressBar().setIndeterminate(false);
                    setControlsEnabled(true);
                }
            }
        };
        worker.execute();
    }

    private void btnTemplatesActionPerformed(java.awt.event.ActionEvent evt) {
        new TemplatesFrame(this, svdComputer, repository).setVisible(true);
    }

    /**
     * Кнопки ×1.0 / ×1.5 / ×2.0: для каждого класса выставляет порог =
     * (среднее внутриклассовое расстояние) × множитель. У каждого класса своя
     * статистика, поэтому три спиннера получают разные значения.
     */
    private void onAutoThreshold(double multiplier) {
        stores = repository.loadAll();
        spinnerCircle.setValue(recognizer.calculateAutoThreshold(stores.get(ShapeClass.CIRCLE), multiplier));
        spinnerTriangle.setValue(recognizer.calculateAutoThreshold(stores.get(ShapeClass.TRIANGLE), multiplier));
        spinnerRectangle.setValue(recognizer.calculateAutoThreshold(stores.get(ShapeClass.RECTANGLE), multiplier));
        applySpinnersToRecognizer();
        persistThresholds();
        recognitionPanel.appendLog(String.format(
                "Пороги установлены (×%.1f): Circle=%.4f, Triangle=%.4f, Rectangle=%.4f",
                multiplier, doubleValue(spinnerCircle), doubleValue(spinnerTriangle), doubleValue(spinnerRectangle)));
    }

    private void btnExitActionPerformed(java.awt.event.ActionEvent evt) {
        dispose();
    }

    private void btnMul10ActionPerformed(java.awt.event.ActionEvent evt) {
        onAutoThreshold(1.0);
    }

    private void btnMul15ActionPerformed(java.awt.event.ActionEvent evt) {
        onAutoThreshold(1.5);
    }

    private void btnMul20ActionPerformed(java.awt.event.ActionEvent evt) {
        onAutoThreshold(2.0);
    }

    private void spinnerThresholdStateChanged(javax.swing.event.ChangeEvent evt) {
        applySpinnersToRecognizer();
        persistThresholds();
    }

    /** Блокирует/разблокирует кнопки на время фонового распознавания. */
    private void setControlsEnabled(boolean enabled) {
        btnLoadImage.setEnabled(enabled);
        btnRecognize.setEnabled(enabled);
        btnTemplates.setEnabled(enabled);
        btnMul10.setEnabled(enabled);
        btnMul15.setEnabled(enabled);
        btnMul20.setEnabled(enabled);
    }

    public void reloadStores() {
        stores = repository.loadAll();
        recognitionPanel.appendLog("Эталоны перечитаны из каталога templates.");
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnExit;
    private javax.swing.JButton btnLoadImage;
    private javax.swing.JButton btnMul10;
    private javax.swing.JButton btnMul15;
    private javax.swing.JButton btnMul20;
    private javax.swing.JButton btnRecognize;
    private javax.swing.JButton btnTemplates;
    private javax.swing.JLabel lblCircle;
    private javax.swing.JLabel lblRectangle;
    private javax.swing.JLabel lblTitle;
    private javax.swing.JLabel lblTriangle;
    private RecognitionPanel recognitionPanel;
    private javax.swing.JSpinner spinnerCircle;
    private javax.swing.JSpinner spinnerRectangle;
    private javax.swing.JSpinner spinnerTriangle;
    // End of variables declaration//GEN-END:variables
}
