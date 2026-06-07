package svd.recognizer.ui;

/**
 * Панель распознавания, встраиваемая в MainFrame.
 *
 * Содержит:
 * 1. Область исходного изображения
 * 2. Область усреднённого эталона распознанного класса
 * 3. Область изображения после preprocessing 64x64
 * 4. Progress bar выполнения операций
 * 5. Текстовый журнал сообщений
 *
 * Таким образом MainFrame содержит только RecognitionPanel, а вся конкретная
 * визуальная логика сосредоточена внутри панели.
 *
 * @author ssv
 */
public class RecognitionPanel extends javax.swing.JPanel {

    public RecognitionPanel() {
        initComponents();
        txtLog.setEditable(false);
    }

    public ImageView getSourceView() {
        return pnlSource;
    }

    public ImageView getTemplateView() {
        return pnlTemplate;
    }

    public ImageView getProcessedView() {
        return pnlProcessed;
    }

    public javax.swing.JProgressBar getProgressBar() {
        return progressBar;
    }

    /**
     * Крупная метка под изображениями для вывода названия распознанного класса
     * (или сообщения «Не распознано»). Цвет задаётся вызывающей стороной.
     */
    public javax.swing.JLabel getResultLabel() {
        return lblResult;
    }

    /** Устанавливает текст и цвет метки результата. */
    public void setResult(String text, java.awt.Color color) {
        lblResult.setText(text);
        lblResult.setForeground(color);
    }

    public void appendLog(String message) {
        txtLog.append(message + System.lineSeparator());
        // Автопрокрутка к последней добавленной строке: ставим курсор в конец
        // текста, и область прокручивается так, чтобы он был виден.
        txtLog.setCaretPosition(txtLog.getDocument().getLength());
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        lblSource = new javax.swing.JLabel();
        lblTemplate = new javax.swing.JLabel();
        lblProcessed = new javax.swing.JLabel();
        pnlSource = new ImageView();
        pnlTemplate = new ImageView();
        pnlProcessed = new ImageView();
        lblResult = new javax.swing.JLabel();
        progressBar = new javax.swing.JProgressBar();
        scrollLog = new javax.swing.JScrollPane();
        txtLog = new javax.swing.JTextArea();

        lblSource.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblSource.setText("Source");

        lblTemplate.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTemplate.setText("Template");

        lblProcessed.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblProcessed.setText("Processed 64x64");

        lblResult.setFont(new java.awt.Font("Segoe UI", 1, 20));
        lblResult.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblResult.setText(" ");

        txtLog.setColumns(20);
        txtLog.setRows(8);
        scrollLog.setViewportView(txtLog);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lblResult, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(progressBar, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(scrollLog)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblSource, javax.swing.GroupLayout.DEFAULT_SIZE, 190, Short.MAX_VALUE)
                            .addComponent(pnlSource, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblTemplate, javax.swing.GroupLayout.DEFAULT_SIZE, 190, Short.MAX_VALUE)
                            .addComponent(pnlTemplate, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblProcessed, javax.swing.GroupLayout.DEFAULT_SIZE, 190, Short.MAX_VALUE)
                            .addComponent(pnlProcessed, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblSource)
                    .addComponent(lblTemplate)
                    .addComponent(lblProcessed))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(pnlSource, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlTemplate, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlProcessed, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(lblResult, javax.swing.GroupLayout.PREFERRED_SIZE, 32, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(progressBar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(scrollLog, javax.swing.GroupLayout.PREFERRED_SIZE, 180, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel lblProcessed;
    private javax.swing.JLabel lblResult;
    private javax.swing.JLabel lblSource;
    private javax.swing.JLabel lblTemplate;
    private svd.recognizer.ui.ImageView pnlProcessed;
    private svd.recognizer.ui.ImageView pnlSource;
    private svd.recognizer.ui.ImageView pnlTemplate;
    private javax.swing.JProgressBar progressBar;
    private javax.swing.JScrollPane scrollLog;
    private javax.swing.JTextArea txtLog;
    // End of variables declaration//GEN-END:variables
}