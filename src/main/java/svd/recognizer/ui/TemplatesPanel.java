package svd.recognizer.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Map;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.TemplateStore;

/**
 * Панель управления эталонами, встраиваемая в TemplatesFrame.
 *
 * Структура соответствует ТЗ:
 * 1. Три независимые секции для Circle, Triangle и Rectangle
 * 2. Для каждой секции есть отдельные кнопки добавления и сброса
 * 3. Для каждой секции отображается усреднённое изображение класса
 * 4. Под каждой секцией выводится подпись с количеством загруженных и усреднённых шаблонов
 * 5. Цвет подписи красный при количестве < 5 и зелёный при количестве >= 5
 * 6. Панель рассчитана на автоперерисовку после каждого изменения данных
 *
 * @author ssv
 */
public class TemplatesPanel extends javax.swing.JPanel {
    public TemplatesPanel() {
        initComponents();
    }

    public JButton getBtnAddCircle() {
        return btnAddCircle;
    }

    public JButton getBtnAddTriangle() {
        return btnAddTriangle;
    }

    public JButton getBtnAddRectangle() {
        return btnAddRectangle;
    }

    public JButton getBtnResetCircle() {
        return btnResetCircle;
    }

    public JButton getBtnResetTriangle() {
        return btnResetTriangle;
    }

    public JButton getBtnResetRectangle() {
        return btnResetRectangle;
    }

    public void refresh(Map<ShapeClass, TemplateStore> stores) {
        updateSection(stores.get(ShapeClass.CIRCLE), pnlCircleImage, lblCircleCount);
        updateSection(stores.get(ShapeClass.TRIANGLE), pnlTriangleImage, lblTriangleCount);
        updateSection(stores.get(ShapeClass.RECTANGLE), pnlRectangleImage, lblRectangleCount);
    }

    private void updateSection(TemplateStore store, PreviewPanel panel, JLabel label) {
        if (store == null) {
            panel.setImage(null);
            label.setText("0 загружено, 0 усреднено");
            label.setForeground(Color.RED);
            return;
        }
        int count = store.getCount();
        panel.setImage(store.getAverageImage());
        label.setText(count + " загружено, " + count + " усреднено");
        label.setForeground(count >= TemplateStore.MIN_TEMPLATES ? new Color(0, 128, 0) : Color.RED);
    }

    /**
     * Панель предпросмотра усреднённого эталона.
     *
     * Если эталоны ещё не загружены, рисуется служебная подпись "нет данных".
     */
    public static class PreviewPanel extends JPanel {
        private BufferedImage image;

        public PreviewPanel() {
            setPreferredSize(new Dimension(180, 180));
            setBackground(Color.WHITE);
            setBorder(javax.swing.BorderFactory.createLineBorder(Color.LIGHT_GRAY));
        }

        public void setImage(BufferedImage image) {
            this.image = image;
            repaint();
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

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        pnlCircle = new javax.swing.JPanel();
        pnlTriangle = new javax.swing.JPanel();
        pnlRectangle = new javax.swing.JPanel();
        lblCircleTitle = new javax.swing.JLabel();
        lblTriangleTitle = new javax.swing.JLabel();
        lblRectangleTitle = new javax.swing.JLabel();
        pnlCircleImage = new PreviewPanel();
        pnlTriangleImage = new PreviewPanel();
        pnlRectangleImage = new PreviewPanel();
        lblCircleCount = new javax.swing.JLabel();
        lblTriangleCount = new javax.swing.JLabel();
        lblRectangleCount = new javax.swing.JLabel();
        btnAddCircle = new javax.swing.JButton();
        btnAddTriangle = new javax.swing.JButton();
        btnAddRectangle = new javax.swing.JButton();
        btnResetCircle = new javax.swing.JButton();
        btnResetTriangle = new javax.swing.JButton();
        btnResetRectangle = new javax.swing.JButton();

        lblCircleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblCircleTitle.setText("Circle");
        lblTriangleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTriangleTitle.setText("Triangle");
        lblRectangleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblRectangleTitle.setText("Rectangle");

        lblCircleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblCircleCount.setText("0 загружено, 0 усреднено");
        lblTriangleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTriangleCount.setText("0 загружено, 0 усреднено");
        lblRectangleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblRectangleCount.setText("0 загружено, 0 усреднено");

        btnAddCircle.setText("Add Circle");
        btnAddTriangle.setText("Add Triangle");
        btnAddRectangle.setText("Add Rectangle");
        btnResetCircle.setText("Reset Circle");
        btnResetTriangle.setText("Reset Triangle");
        btnResetRectangle.setText("Reset Rectangle");

        buildSection(pnlCircle, lblCircleTitle, pnlCircleImage, lblCircleCount, btnAddCircle, btnResetCircle);
        buildSection(pnlTriangle, lblTriangleTitle, pnlTriangleImage, lblTriangleCount, btnAddTriangle, btnResetTriangle);
        buildSection(pnlRectangle, lblRectangleTitle, pnlRectangleImage, lblRectangleCount, btnAddRectangle, btnResetRectangle);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(pnlTriangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(pnlRectangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(pnlCircle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(pnlTriangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlRectangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlCircle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void buildSection(JPanel panel, JLabel title, PreviewPanel preview, JLabel count, JButton add, JButton reset) {
        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(panel);
        panel.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(title, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(preview, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(count, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addComponent(add, javax.swing.GroupLayout.DEFAULT_SIZE, 120, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(reset, javax.swing.GroupLayout.DEFAULT_SIZE, 120, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(title)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(preview, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(count)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(add)
                    .addComponent(reset))
                .addGap(0, 0, Short.MAX_VALUE))
        );
    }

    private javax.swing.JButton btnAddCircle;
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnAddRectangle;
    private javax.swing.JButton btnAddTriangle;
    private javax.swing.JButton btnResetCircle;
    private javax.swing.JButton btnResetRectangle;
    private javax.swing.JButton btnResetTriangle;
    private javax.swing.JLabel lblCircleCount;
    private javax.swing.JLabel lblCircleTitle;
    private javax.swing.JLabel lblRectangleCount;
    private javax.swing.JLabel lblRectangleTitle;
    private javax.swing.JLabel lblTriangleCount;
    private javax.swing.JLabel lblTriangleTitle;
    private javax.swing.JPanel pnlCircle;
    private PreviewPanel pnlCircleImage;
    private javax.swing.JPanel pnlRectangle;
    private PreviewPanel pnlRectangleImage;
    private javax.swing.JPanel pnlTriangle;
    private PreviewPanel pnlTriangleImage;
}    // End of variables declaration//GEN-END:variables
