package svd.recognizer.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.nio.file.Paths;
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

    // Поля для кнопок «Add Folder» объявлены вне GEN-блока (добавлены вручную)
    private javax.swing.JButton btnAddFolderCircle;
    private javax.swing.JButton btnAddFolderTriangle;
    private javax.swing.JButton btnAddFolderRectangle;

    public TemplatesPanel() {
        initComponents();
    }

    public JButton getBtnAddCircle()          { return btnAddCircle; }
    public JButton getBtnAddTriangle()        { return btnAddTriangle; }
    public JButton getBtnAddRectangle()       { return btnAddRectangle; }
    public JButton getBtnResetCircle()        { return btnResetCircle; }
    public JButton getBtnResetTriangle()      { return btnResetTriangle; }
    public JButton getBtnResetRectangle()     { return btnResetRectangle; }
    public JButton getBtnAddFolderCircle()    { return btnAddFolderCircle; }
    public JButton getBtnAddFolderTriangle()  { return btnAddFolderTriangle; }
    public JButton getBtnAddFolderRectangle() { return btnAddFolderRectangle; }

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
        private static final int THUMB   = 128;
        private static final int GAP     = 4;
        private static final int LABEL_H = 16;   // высота строки с именем файла

        private static final Color BAD_BORDER    = new Color(210, 0, 0);
        private static final Color BAD_LABEL_BG  = new Color(210, 0, 0, 200);
        private static final Color BAD_LABEL_FG  = Color.WHITE;
        private static final Color FILE_LABEL_FG  = new Color(80, 80, 80);
        private static final Color FILE_LABEL_BAD = new Color(180, 0, 0);

        private List<Template> samples = List.of();

        public SampleStripPanel() {
            setBackground(Color.WHITE);
            setBorder(javax.swing.BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            setPreferredSize(new Dimension(THUMB, THUMB + LABEL_H));
        }

        public void setSamples(List<Template> templates) {
            this.samples = templates;
            int w = samples.isEmpty() ? THUMB : samples.size() * (THUMB + GAP) + GAP;
            setPreferredSize(new Dimension(w, THUMB + LABEL_H + 2 * GAP));
            revalidate();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (samples.isEmpty()) {
                g.setColor(Color.GRAY);
                g.drawString("нет данных", GAP, (getHeight() + LABEL_H) / 2 + 5);
                return;
            }

            Font fileFont = new Font("SansSerif", Font.PLAIN, 10);
            g.setFont(fileFont);
            FontMetrics fileFm = g.getFontMetrics();

            int x = GAP;
            for (Template t : samples) {
                // --- подпись с именем файла над миниатюрой ---
                String path = t.getSourceFilePath();
                if (path != null && !path.isEmpty()) {
                    String name = Paths.get(path).getFileName().toString();
                    name = ellipsize(name, THUMB, fileFm);
                    g.setColor(t.isLowQuality() ? FILE_LABEL_BAD : FILE_LABEL_FG);
                    g.drawString(name, x, GAP + fileFm.getAscent());
                }

                BufferedImage img = t.getNormalizedImage();
                if (img != null) {
                    g.drawImage(img, x, GAP + LABEL_H, THUMB, THUMB, this);
                } else {
                    g.setColor(Color.LIGHT_GRAY);
                    g.fillRect(x, GAP + LABEL_H, THUMB, THUMB);
                }

                if (t.isLowQuality()) {
                    // Двойная красная рамка
                    g.setColor(BAD_BORDER);
                    g.drawRect(x,     GAP + LABEL_H,     THUMB - 1, THUMB - 1);
                    g.drawRect(x + 1, GAP + LABEL_H + 1, THUMB - 3, THUMB - 3);

                    // Подпись «BAD» внизу миниатюры
                    Font badFont = new Font("SansSerif", Font.BOLD, 11);
                    g.setFont(badFont);
                    FontMetrics fm = g.getFontMetrics();
                    String label = "BAD";
                    int lw = fm.stringWidth(label);
                    int lh = fm.getHeight();
                    int lx = x + (THUMB - lw) / 2;
                    int ly = GAP + LABEL_H + THUMB - 2;
                    // фон подписи
                    g.setColor(BAD_LABEL_BG);
                    g.fillRect(lx - 2, ly - lh + 2, lw + 4, lh);
                    // текст
                    g.setColor(BAD_LABEL_FG);
                    g.drawString(label, lx, ly);

                    // Восстановить шрифт для следующей итерации
                    g.setFont(fileFont);
                }

                x += THUMB + GAP;
            }
        }

        /** Обрезает строку до maxWidth пикселей, добавляя «…» если не влезает. */
        private String ellipsize(String text, int maxWidth, FontMetrics fm) {
            if (fm.stringWidth(text) <= maxWidth) return text;
            String ellipsis = "\u2026";
            int ellipsisW = fm.stringWidth(ellipsis);
            StringBuilder sb = new StringBuilder(text);
            while (sb.length() > 0 && fm.stringWidth(sb.toString()) + ellipsisW > maxWidth) {
                sb.deleteCharAt(sb.length() - 1);
            }
            return sb + ellipsis;
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

        lblCircleCount    = new javax.swing.JLabel();
        lblTriangleCount  = new javax.swing.JLabel();
        lblRectangleCount = new javax.swing.JLabel();

        btnAddCircle      = new javax.swing.JButton();
        btnAddTriangle    = new javax.swing.JButton();
        btnAddRectangle   = new javax.swing.JButton();
        btnResetCircle    = new javax.swing.JButton();
        btnResetTriangle  = new javax.swing.JButton();
        btnResetRectangle = new javax.swing.JButton();

        // Кнопки «Add Folder» — инициализируются вручную (вне GEN-блока переменных)
        btnAddFolderCircle    = new javax.swing.JButton();
        btnAddFolderTriangle  = new javax.swing.JButton();
        btnAddFolderRectangle = new javax.swing.JButton();

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
        btnAddFolderCircle.setText("Add Folder");
        btnAddFolderTriangle.setText("Add Folder");
        btnAddFolderRectangle.setText("Add Folder");

        buildSection(pnlCircle,    lblCircleTitle,    scrollCircle,    lblCircleCount,    btnAddCircle,    btnAddFolderCircle,    btnResetCircle);
        buildSection(pnlTriangle,  lblTriangleTitle,  scrollTriangle,  lblTriangleCount,  btnAddTriangle,  btnAddFolderTriangle,  btnResetTriangle);
        buildSection(pnlRectangle, lblRectangleTitle, scrollRectangle, lblRectangleCount, btnAddRectangle, btnAddFolderRectangle, btnResetRectangle);

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
                              JLabel count, JButton add, JButton addFolder, JButton reset) {
        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(panel);
        panel.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(title,     javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(scroll,    javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(count,     javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(layout.createSequentialGroup()
                .addComponent(add,       javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(addFolder, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(reset,     javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createSequentialGroup()
                .addComponent(title)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(scroll, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(count)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(add)
                    .addComponent(addFolder)
                    .addComponent(reset))
        );
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnAddCircle;
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
    private javax.swing.JPanel pnlRectangle;
    private javax.swing.JPanel pnlTriangle;
    private JScrollPane scrollCircle;
    private JScrollPane scrollRectangle;
    private JScrollPane scrollTriangle;
    private SampleStripPanel stripCircle;
    private SampleStripPanel stripRectangle;
    private SampleStripPanel stripTriangle;
    // End of variables declaration//GEN-END:variables
}
