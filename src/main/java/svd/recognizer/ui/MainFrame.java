package svd.recognizer.ui;

import java.io.File;
import java.util.EnumMap;
import java.util.Map;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.SpinnerNumberModel;
import javax.swing.filechooser.FileNameExtensionFilter;

import svd.recognizer.model.*;
import svd.recognizer.processing.ImagePreprocessor;
import svd.recognizer.processing.ImageVectorizer;
import svd.recognizer.processing.SVDComputer;
import svd.recognizer.processing.ShapeRecognizer;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
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

    private final SubspaceRecognizer subspaceRecognizer;
    private final SubspaceTrainer subspaceTrainer;
    private RecognitionMode currentMode = RecognitionMode.SIGMA_VECTOR;

    private Map<ShapeClass, TemplateStore> stores;
    private File selectedImageFile;

    public MainFrame(SVDComputer svdComputer, TemplateRepository repository) {
        this.svdComputer = svdComputer;
        this.repository = repository;

        this.subspaceRecognizer = new SubspaceRecognizer();
        this.subspaceTrainer = new SubspaceTrainer(svdComputer.getSvdEngine());

        this.stores = repository.loadAll();
        initComponents();
        setLocationRelativeTo(null);
        loadThresholdsIntoUi();
        loadSubspaceSettings();
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

    /** Загружает настройки subspace-режима. */
    private void loadSubspaceSettings() {
        currentMode = settingsStore.loadRecognitionMode();
        double theta = settingsStore.loadSubspaceThreshold();
        int k = settingsStore.loadSubspaceK();

        subspaceRecognizer.setThreshold(theta);
        spinnerTheta.setValue(theta);
        comboMode.setSelectedItem(currentMode == RecognitionMode.SUBSPACE ? "Subspace" : "Sigma-vector");

        recognitionPanel.appendLog("Subspace режим: k=" + k + ", theta=" + theta);
        updateTrainingStatus();
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

        btnTrain = new javax.swing.JButton();
        lblTheta = new javax.swing.JLabel();
        spinnerTheta = new javax.swing.JSpinner();
        comboMode = new javax.swing.JComboBox<>();
        lblTrainStatus = new javax.swing.JLabel();

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

        btnTrain.setText("Обучение");
        btnTrain.setToolTipText("Построить подпространства для всех классов");

        lblTheta.setText("Порог θ:");

        spinnerTheta.setModel(new SpinnerNumberModel(13.0d, 1.0d, 50.0d, 0.5d));
        spinnerTheta.setPreferredSize(new java.awt.Dimension(60, 20));

        comboMode.setModel(new javax.swing.DefaultComboBoxModel<>(
                new String[] { "Sigma-vector", "Subspace" }
        ));
        comboMode.setPreferredSize(new java.awt.Dimension(120, 25));

        lblTrainStatus.setText("Обучено: Circle — Triangle — Rectangle —");
        lblTrainStatus.setFont(new java.awt.Font("Segoe UI", 0, 11));

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

        btnTrain.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnTrainActionPerformed(evt);
            }
        });

        spinnerTheta.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                spinnerThetaStateChanged(evt);
            }
        });

        comboMode.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                comboModeActionPerformed(evt);
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
                                .addComponent(btnTrain)
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
                        .addGroup(layout.createSequentialGroup()
                                .addContainerGap()
                                .addComponent(comboMode, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(lblTheta)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(spinnerTheta, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(lblTrainStatus)
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
                                        .addComponent(btnTrain)
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
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                        .addComponent(comboMode, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(lblTheta)
                                        .addComponent(spinnerTheta, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(lblTrainStatus))
                                .addContainerGap())
        );

        pack();
    }

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

        final File fileToRecognize = selectedImageFile;
        setControlsEnabled(false);
        recognitionPanel.getProgressBar().setIndeterminate(true);
        recognitionPanel.setResult(" ", java.awt.Color.BLACK);
        recognitionPanel.appendLog("Распознавание...");

        final RecognitionMode mode = currentMode;
        stores = repository.loadAll();
        applySpinnersToRecognizer();

        javax.swing.SwingWorker<RecognitionResult, Void> worker =
                new javax.swing.SwingWorker<RecognitionResult, Void>() {

                    private final Map<ShapeClass, java.awt.image.BufferedImage> processedImages =
                            new EnumMap<>(ShapeClass.class);

                    @Override
                    protected RecognitionResult doInBackground() throws Exception {
                        if (mode == RecognitionMode.SIGMA_VECTOR) {
                            return doSigmaRecognition(fileToRecognize);
                        } else {
                            return doSubspaceRecognition(fileToRecognize);
                        }
                    }

                    private RecognitionResult doSigmaRecognition(File file) throws Exception {
                        Map<ShapeClass, double[]> hypotheses = new EnumMap<>(ShapeClass.class);
                        for (ShapeClass sc : ShapeClass.values()) {
                            ImagePreprocessor.PreprocessResult prep =
                                    preprocessor.preprocess(file, sc);
                            hypotheses.put(sc, svdComputer.computeFeatures(prep.getMatrix()));
                            processedImages.put(sc, prep.getImage());
                        }
                        return recognizer.recognize(hypotheses, stores);
                    }

                    private RecognitionResult doSubspaceRecognition(File file) throws Exception {
                        Map<ShapeClass, double[]> hypothesisVectors = new EnumMap<>(ShapeClass.class);
                        for (ShapeClass sc : ShapeClass.values()) {
                            ImagePreprocessor.PreprocessResult prep =
                                    preprocessor.preprocess(file, sc);
                            double[] vector = ImageVectorizer.toVector(prep.getMatrix());
                            hypothesisVectors.put(sc, vector);
                            processedImages.put(sc, prep.getImage());
                        }
                        return subspaceRecognizer.recognize(hypothesisVectors, stores);
                    }

                    @Override
                    protected void done() {
                        try {
                            RecognitionResult result = get();
                            if (result.isRecognized()) {
                                ShapeClass winner = result.getShapeClass();
                                recognitionPanel.getProcessedView().setImage(processedImages.get(winner));
                                recognitionPanel.getTemplateView().setImage(ShapeIconFactory.createShapeIcon(winner));
                                recognitionPanel.setResult(winner.getDisplayName(), new java.awt.Color(0, 128, 0));
                                recognitionPanel.appendLog("Результат: " + winner.getDisplayName());
                                if (mode == RecognitionMode.SIGMA_VECTOR) {
                                    recognitionPanel.appendLog(String.format("Distance = %.6f, Threshold = %.6f",
                                            result.getDistance(), result.getThreshold()));
                                } else {
                                    recognitionPanel.appendLog(String.format("Reconstruction error = %.6f, Threshold = %.6f",
                                            result.getScore(), result.getThreshold()));
                                    if (result.getClassScores() != null) {
                                        recognitionPanel.appendLog("Оценки по классам: " + result.getClassScores());
                                    }
                                }
                            } else {
                                recognitionPanel.getProcessedView().setImage(null);
                                recognitionPanel.getTemplateView().setImage(ShapeIconFactory.createNotRecognizedIcon());
                                recognitionPanel.setResult("Не распознано", java.awt.Color.RED);
                                recognitionPanel.appendLog("Фигура не распознана");
                                if (mode == RecognitionMode.SIGMA_VECTOR) {
                                    recognitionPanel.appendLog(String.format(
                                            "Ближайшее расстояние = %.6f, порог класса = %.6f",
                                            result.getDistance(), result.getThreshold()));
                                } else {
                                    recognitionPanel.appendLog(String.format(
                                            "Минимальная ошибка = %.6f, порог θ = %.6f",
                                            result.getScore(), result.getThreshold()));
                                    if (result.getClassScores() != null) {
                                        recognitionPanel.appendLog("Оценки по классам: " + result.getClassScores());
                                    }
                                }
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

    private void setControlsEnabled(boolean enabled) {
        btnLoadImage.setEnabled(enabled);
        btnRecognize.setEnabled(enabled);
        btnTemplates.setEnabled(enabled);
        btnTrain.setEnabled(enabled);
        btnMul10.setEnabled(enabled);
        btnMul15.setEnabled(enabled);
        btnMul20.setEnabled(enabled);
        comboMode.setEnabled(enabled);
        spinnerTheta.setEnabled(enabled);
    }

    public void reloadStores() {
        stores = repository.loadAll();
        recognitionPanel.appendLog("Эталоны перечитаны из каталога templates.");
        updateTrainingStatus();
    }

    /**
     * Обработчик кнопки «Обучение».
     * Строит подпространства для всех классов по текущим эталонам.
     */
    private void btnTrainActionPerformed(java.awt.event.ActionEvent evt) {
        stores = repository.loadAll();

        // Проверяем, что все классы имеют эталоны с normalizedMatrix
        boolean allReady = true;
        StringBuilder missing = new StringBuilder();
        for (ShapeClass sc : ShapeClass.values()) {
            TemplateStore store = stores.get(sc);
            if (store == null || store.getTemplates().isEmpty()) {
                allReady = false;
                missing.append(sc.getDisplayName()).append(" (нет эталонов)\n");
                continue;
            }
            boolean hasMatrix = true;
            for (Template template : store.getTemplates()) {
                if (template.getNormalizedMatrix() == null) {
                    hasMatrix = false;
                    break;
                }
            }
            if (!hasMatrix) {
                allReady = false;
                missing.append(sc.getDisplayName()).append(" (нет normalizedMatrix)\n");
            }
        }

        if (!allReady) {
            JOptionPane.showMessageDialog(this,
                    "Невозможно выполнить обучение:\n" +
                            missing.toString() + "\n\n" +
                            "Удалите и загрузите эталоны заново через Templates.",
                    "Ошибка обучения",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }

        int confirm = JOptionPane.showConfirmDialog(this,
                "Будет выполнено обучение подпространств для всех трёх классов.\n" +
                        "Это может занять некоторое время. Продолжить?",
                "Подтверждение обучения",
                JOptionPane.YES_NO_OPTION);
        if (confirm != JOptionPane.YES_OPTION) {
            return;
        }

        setControlsEnabled(false);
        recognitionPanel.getProgressBar().setIndeterminate(true);
        recognitionPanel.appendLog("Начало обучения подпространств...");

        final int k = settingsStore.loadSubspaceK();

        javax.swing.SwingWorker<Void, Void> worker =
                new javax.swing.SwingWorker<Void, Void>() {

                    private final Map<ShapeClass, String> errors = new EnumMap<>(ShapeClass.class);

                    @Override
                    protected Void doInBackground() throws Exception {
                        stores = repository.loadAll();

                        for (ShapeClass sc : ShapeClass.values()) {
                            try {
                                recognitionPanel.appendLog("Обучение класса " + sc.getDisplayName() + "...");
                                TemplateStore store = stores.get(sc);

                                SubspaceModel model = subspaceTrainer.train(store, k);
                                store.setSubspaceModel(model);
                                repository.save(store);

                                recognitionPanel.appendLog("  ✅ " + sc.getDisplayName() + " обучен (k=" + model.getK() + ")");
                            } catch (Exception e) {
                                errors.put(sc, e.getMessage());
                                recognitionPanel.appendLog("  ❌ Ошибка обучения " + sc.getDisplayName() + ": " + e.getMessage());
                            }
                        }
                        return null;
                    }

                    @Override
                    protected void done() {
                        recognitionPanel.getProgressBar().setIndeterminate(false);
                        setControlsEnabled(true);

                        if (errors.isEmpty()) {
                            recognitionPanel.appendLog("✅ Обучение завершено успешно для всех классов!");
                            JOptionPane.showMessageDialog(MainFrame.this,
                                    "Обучение завершено успешно!\n\n" +
                                            "Все три класса обучены и готовы к распознаванию в subspace-режиме.",
                                    "Обучение завершено",
                                    JOptionPane.INFORMATION_MESSAGE);
                        } else {
                            StringBuilder msg = new StringBuilder("Обучение завершено с ошибками:\n");
                            for (Map.Entry<ShapeClass, String> entry : errors.entrySet()) {
                                msg.append("  • ").append(entry.getKey().getDisplayName())
                                        .append(": ").append(entry.getValue()).append("\n");
                            }
                            recognitionPanel.appendLog("❌ Обучение завершено с ошибками");
                            JOptionPane.showMessageDialog(MainFrame.this,
                                    msg.toString(),
                                    "Ошибки обучения",
                                    JOptionPane.ERROR_MESSAGE);
                        }
                        updateTrainingStatus();
                    }
                };
        worker.execute();
    }

    /**
     * Обработчик изменения порога θ.
     */
    private void spinnerThetaStateChanged(javax.swing.event.ChangeEvent evt) {
        double theta = doubleValue(spinnerTheta);
        subspaceRecognizer.setThreshold(theta);
        settingsStore.saveSubspaceThreshold(theta);
        recognitionPanel.appendLog("Порог θ установлен: " + theta);
    }

    /**
     * Обработчик переключения режима распознавания.
     */
    private void comboModeActionPerformed(java.awt.event.ActionEvent evt) {
        String selected = (String) comboMode.getSelectedItem();
        RecognitionMode newMode = "Subspace".equals(selected)
                ? RecognitionMode.SUBSPACE
                : RecognitionMode.SIGMA_VECTOR;

        if (newMode == RecognitionMode.SUBSPACE) {
            stores = repository.loadAll();
            boolean allTrained = true;
            StringBuilder notTrained = new StringBuilder();
            for (ShapeClass sc : ShapeClass.values()) {
                TemplateStore store = stores.get(sc);
                if (store == null || !store.isTrained()) {
                    allTrained = false;
                    notTrained.append("  • ").append(sc.getDisplayName()).append("\n");
                }
            }
            if (!allTrained) {
                JOptionPane.showMessageDialog(this,
                        "Subspace-режим недоступен: не все классы обучены.\n" +
                                "Не обучены:\n" + notTrained.toString() +
                                "\nНажмите кнопку «Обучение» для построения подпространств.",
                        "Режим недоступен",
                        JOptionPane.WARNING_MESSAGE);
                comboMode.setSelectedItem("Sigma-vector");
                return;
            }
        }

        currentMode = newMode;
        settingsStore.saveRecognitionMode(newMode);
        recognitionPanel.appendLog("Режим распознавания: " + selected);
    }

    /**
     * Обновляет индикацию обученности классов.
     */
    private void updateTrainingStatus() {
        stores = repository.loadAll();
        StringBuilder status = new StringBuilder("Обучено: ");
        for (ShapeClass sc : ShapeClass.values()) {
            TemplateStore store = stores.get(sc);
            if (store != null && store.isTrained()) {
                status.append(sc.getDisplayName()).append(" ✓ ");
            } else {
                status.append(sc.getDisplayName()).append(" — ");
            }
        }
        lblTrainStatus.setText(status.toString());
    }

    // Variables declaration
    private javax.swing.JButton btnExit;
    private javax.swing.JButton btnLoadImage;
    private javax.swing.JButton btnMul10;
    private javax.swing.JButton btnMul15;
    private javax.swing.JButton btnMul20;
    private javax.swing.JButton btnRecognize;
    private javax.swing.JButton btnTemplates;
    private javax.swing.JButton btnTrain;
    private javax.swing.JComboBox<String> comboMode;
    private javax.swing.JLabel lblCircle;
    private javax.swing.JLabel lblRectangle;
    private javax.swing.JLabel lblTheta;
    private javax.swing.JLabel lblTitle;
    private javax.swing.JLabel lblTrainStatus;
    private javax.swing.JLabel lblTriangle;
    private RecognitionPanel recognitionPanel;
    private javax.swing.JSpinner spinnerCircle;
    private javax.swing.JSpinner spinnerRectangle;
    private javax.swing.JSpinner spinnerTheta;
    private javax.swing.JSpinner spinnerTriangle;
    // End of variables declaration
}