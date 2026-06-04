package svd.recognizer.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

/**
 * Панель управления эталонами.
 *
 * Для каждого класса фигур отображаются ВСЕ загруженные нормализованные
 * изображения в виде горизонтальной прокручиваемой полосы — для отладки
 * препроцессинга.
 *
 * @author ssv
 */
public class TemplatesPanel extends javax.swing.JPanel {

    public TemplatesPanel() {
        initComponents();
    }

    public JButton getBtnAddCircle()     { return btnAddCircle; }
    public JButton getBtnAddTriangle()   { return btnAddTriangle; }
    public JButton getBtnAddRectangle()  { return btnAddRectangle; }
    public JButton getBtnResetCircle()   { return btnResetCircle; }
    public JButton getBtnResetTriangle() { return btnResetTriangle; }
    public JButton getBtnResetRectangle(){ return btnResetRectangle; }

    public void refresh(Map<ShapeClass, TemplateStore> stores) {
        updateSection(stores.get(ShapeClass.CIRCLE),    stripCircle,    lblCircleCount);
        updateSection(stores.get(ShapeClass.TRIANGLE),  stripTriangle,  lblTriangleCount);
        updateSection(stores.get(ShapeClass.RECTANGLE), stripRectangle, lblRectangleCount);
    }

    private void updateSection(TemplateStore store, SampleStripPanel strip, JLabel label) {
        if (store == null) {
            strip.setSamples(List.of());
            label.setText("0 загружено");
            label.setForeground(Color.RED);
            return;
        }
        int count = store.getCount();
        strip.setSamples(store.getTemplates());
        label.setText(count + " загружено");
        label.setForeground(count >= TemplateStore.MIN_TEMPLATES ? new Color(0, 128, 0) : Color.RED);
    }

    // -------------------------------------------------------------------------
    // SampleStripPanel — горизонтальная полоса из миниатюр всех образцов
    // -------------------------------------------------------------------------
    public static class SampleStripPanel extends JPanel {
        private static final int THUMB = 128;
        private static final int GAP   = 4;
        private List<Template> samples = List.of();

        public SampleStripPanel() {
            setBackground(Color.WHITE);
            setBorder(javax.swing.BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            setPreferredSize(new Dimension(THUMB, THUMB));
        }

        public void setSamples(List<Template> templates) {
            this.samples = templates;
            int w = samples.isEmpty() ? THUMB : samples.size() * (THUMB + GAP) + GAP;
            setPreferredSize(new Dimension(w, THUMB + 2 * GAP));
            revalidate();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (samples.isEmpty()) {
                g.setColor(Color.GRAY);
                g.drawString("нет данных", GAP, getHeight() / 2 + 5);
                return;
            }
            int x = GAP;
            for (Template t : samples) {
                BufferedImage img = t.getNormalizedImage();
                if (img != null) {
                    g.drawImage(img, x, GAP, THUMB, THUMB, this);
                } else {
                    g.setColor(Color.LIGHT_GRAY);
                    g.fillRect(x, GAP, THUMB, THUMB);
                }
                x += THUMB + GAP;
            }
        }
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        pnlCircle    = new javax.swing.JPanel();
        pnlTriangle  = new javax.swing.JPanel();
        pnlRectangle = new javax.swing.JPanel();

        lblCircleTitle    = new javax.swing.JLabel();
        lblTriangleTitle  = new javax.swing.JLabel();
        lblRectangleTitle = new javax.swing.JLabel();

        stripCircle    = new SampleStripPanel();
        stripTriangle  = new SampleStripPanel();
        stripRectangle = new SampleStripPanel();

        scrollCircle    = new JScrollPane(stripCircle,
            JScrollPane.VERTICAL_SCROLLBAR_NEVER, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollTriangle  = new JScrollPane(stripTriangle,
            JScrollPane.VERTICAL_SCROLLBAR_NEVER, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollRectangle = new JScrollPane(stripRectangle,
            JScrollPane.VERTICAL_SCROLLBAR_NEVER, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

        scrollCircle.setPreferredSize(new Dimension(180, 144));
        scrollTriangle.setPreferredSize(new Dimension(180, 144));
        scrollRectangle.setPreferredSize(new Dimension(180, 144));

        lblCircleCount    = new javax.swing.JLabel();
        lblTriangleCount  = new javax.swing.JLabel();
        lblRectangleCount = new javax.swing.JLabel();

        btnAddCircle     = new javax.swing.JButton();
        btnAddTriangle   = new javax.swing.JButton();
        btnAddRectangle  = new javax.swing.JButton();
        btnResetCircle   = new javax.swing.JButton();
        btnResetTriangle = new javax.swing.JButton();
        btnResetRectangle = new javax.swing.JButton();

        lblCircleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblCircleTitle.setText("Circle");
        lblTriangleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTriangleTitle.setText("Triangle");
        lblRectangleTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblRectangleTitle.setText("Rectangle");

        lblCircleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblCircleCount.setText("0 загружено");
        lblTriangleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTriangleCount.setText("0 загружено");
        lblRectangleCount.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblRectangleCount.setText("0 загружено");

        btnAddCircle.setText("Add Circle");
        btnAddTriangle.setText("Add Triangle");
        btnAddRectangle.setText("Add Rectangle");
        btnResetCircle.setText("Reset Circle");
        btnResetTriangle.setText("Reset Triangle");
        btnResetRectangle.setText("Reset Rectangle");

        buildSection(pnlCircle,    lblCircleTitle,    scrollCircle,    lblCircleCount,    btnAddCircle,    btnResetCircle);
        buildSection(pnlTriangle,  lblTriangleTitle,  scrollTriangle,  lblTriangleCount,  btnAddTriangle,  btnResetTriangle);
        buildSection(pnlRectangle, lblRectangleTitle, scrollRectangle, lblRectangleCount, btnAddRectangle, btnResetRectangle);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(pnlTriangle,  javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(pnlRectangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(pnlCircle,    javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(pnlTriangle,  javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlRectangle, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(pnlCircle,    javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void buildSection(JPanel panel, JLabel title, JScrollPane scroll,
                              JLabel count, JButton add, JButton reset) {
        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(panel);
        panel.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(title,  javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(scroll, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(count,  javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addComponent(add,   javax.swing.GroupLayout.DEFAULT_SIZE, 120, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(reset, javax.swing.GroupLayout.DEFAULT_SIZE, 120, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(title)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(scroll, javax.swing.GroupLayout.PREFERRED_SIZE, 144, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(count)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(add)
                    .addComponent(reset))
                .addGap(0, 0, Short.MAX_VALUE))
        );
    }

    // Variables declaration
    private javax.swing.JButton btnAddCircle;
    private javax.swing.JButton btnAddRectangle;
    private javax.swing.JButton btnAddTriangle;
    private javax.swing.JButton btnResetCircle;
    private javax.swing.JButton btnResetRectangle;
    private javax.swing.JButton btnResetTriangle;
    private javax.swing.JLabel  lblCircleCount;
    private javax.swing.JLabel  lblCircleTitle;
    private javax.swing.JLabel  lblRectangleCount;
    private javax.swing.JLabel  lblRectangleTitle;
    private javax.swing.JLabel  lblTriangleCount;
    private javax.swing.JLabel  lblTriangleTitle;
    private javax.swing.JPanel  pnlCircle;
    private javax.swing.JPanel  pnlRectangle;
    private javax.swing.JPanel  pnlTriangle;
    private SampleStripPanel    stripCircle;
    private SampleStripPanel    stripTriangle;
    private SampleStripPanel    stripRectangle;
    private JScrollPane         scrollCircle;
    private JScrollPane         scrollTriangle;
    private JScrollPane         scrollRectangle;
}
