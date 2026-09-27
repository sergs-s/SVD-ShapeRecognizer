package svd.recognizer.faces;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import javax.swing.AbstractAction;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import svd.recognizer.storage.SettingsStore;

/**
 * Ручная разметка центров зрачков на своей базе: представитель каждого пригодного момента
 * (OwnDataset.representatives()). Эталон для оценки точности глаз у способов детекции.
 *
 * Показ: область лица крупно по рамке YuNet с полями (рамка — только для
 * навигации, эталоном не является); масштаб такой, чтобы межзрачковое
 * расстояние было не меньше {@value #MIN_IPD_SCREEN} пикселей экрана. Если
 * YuNet лицо не нашёл — весь кадр, первый щелчок указывает лицо.
 *
 * Щелчки: сначала глаз, левый на снимке (правый глаз человека), затем правый
 * на снимке. Клавиши: Enter — сохранить (ok), D — сохранить «сомнительно»,
 * S — пропустить кадр («не виден»), Backspace — отменить последний щелчок,
 * F — весь кадр, указать лицо щелчком (если YuNet показал не ту область).
 * Другой кадр для перезаписи — выбор из списка вверху окна.
 *
 * CSV — eyes.csv в корне своей базы (вне git): file, lx, ly, rx, ry, flag
 * (ok / doubtful / skipped); координаты — в полном снимке после поворота по
 * EXIF. Запись после каждого кадра; при повторном запуске — продолжение с
 * первого неразмеченного.
 *
 * Режим проверки (аргумент sheet): листы с увеличенными лицами и точками,
 * reports/faces/own/eyes/ (вне git).
 *
 * Запуск: mvn compile exec:java@faces-eyes; проверка:
 * mvn compile exec:java@faces-eyes -Dexec.args=sheet
 *
 * @author ssv
 */
public final class EyeLabeler {

    static final String CSV_NAME = "eyes.csv";
    static final int MIN_IPD_SCREEN = 300;
    /** Масштаб входа YuNet для навигации и порог его уверенности (только навигация). */
    private static final double NAV_SCALE = 0.25;
    private static final float NAV_SCORE = 0.5f;
    /** Поле вокруг рамки лица, доля её размера с каждой стороны. */
    private static final double MARGIN = 0.15;

    /** Разметка кадра; для skipped координаты NaN. */
    record Label(double lx, double ly, double rx, double ry, String flag) {}

    private final OwnDataset dataset;
    private final List<OwnDataset.Frame> frames;
    private final Path csv;
    private final Map<String, Label> labels;
    private final FaceDetectorYN detector;

    // Состояние текущего кадра.
    private int current = -1;
    private Mat image;
    private boolean facePick;
    private Rect crop;
    private double scale;
    private final List<double[]> clicks = new ArrayList<>();

    private JFrame window;
    private JLabel hint;
    private JComboBox<String> chooser;
    private ImagePanel panel;
    private JScrollPane scroll;
    private boolean updatingChooser;

    private EyeLabeler(OwnDataset dataset, Path modelsDir) throws IOException {
        this.dataset = dataset;
        this.frames = dataset.representatives();
        this.csv = dataset.root().resolve(CSV_NAME);
        this.labels = migrate(dataset, frames, readCsv(csv), csv);
        if (Files.exists(csv) && !labels.keySet().equals(readCsv(csv).keySet())) {
            writeCsv();
        }
        long missing = frames.stream().filter(f -> !labels.containsKey(dataset.relative(f.file()))).count();
        System.out.println("Представителей пригодных моментов: " + frames.size() + ", без разметки: " + missing);
        this.detector = modelsDir == null ? null : FaceDetectorYN.create(
                modelsDir.resolve(FaceAlignment.YUNET_FILE).toString(), "", new Size(320, 320), NAV_SCORE, 0.3f, 5000);
    }

    public static void main(String[] args) throws Exception {
        nu.pattern.OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        OwnDataset dataset = OwnDataset.load(Paths.get(settings.loadFacesOwnDir()));
        boolean sheet = args.length > 0 && args[0].equals("sheet");
        EyeLabeler labeler = new EyeLabeler(dataset, sheet ? null : Paths.get(settings.loadFacesModelsDir()));
        if (sheet) {
            labeler.writeSheets(Paths.get(System.getProperty("user.dir"), "reports", "faces", "own", "eyes"));
        } else {
            SwingUtilities.invokeLater(labeler::showWindow);
        }
    }

    // ---------------------------------------------------------------- CSV

    static Map<String, Label> readCsv(Path file) throws IOException {
        Map<String, Label> map = new LinkedHashMap<>();
        if (!Files.exists(file)) {
            return map;
        }
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#") || line.startsWith("file,")) {
                continue;
            }
            String[] f = line.split(",", -1);
            map.put(f[0], new Label(num(f[1]), num(f[2]), num(f[3]), num(f[4]), f[5].trim()));
        }
        return map;
    }

    /**
     * Перенос разметки на текущие имена файлов (после пометки качества в именах):
     * строка сопоставляется кадру по человеку, моменту (секунда в имени) и номеру
     * в имени; сохраняется, только если этот кадр — представитель пригодного
     * момента. Остальные строки (сменившийся представитель, непригодный момент)
     * не используются. При изменениях прежний файл копируется в eyes_before_rename.csv.
     */
    static Map<String, Label> migrate(OwnDataset dataset, List<OwnDataset.Frame> representatives,
                                      Map<String, Label> old, Path csv) throws IOException {
        Map<String, OwnDataset.Frame> byId = new LinkedHashMap<>();
        for (OwnDataset.Frame f : representatives) {
            byId.put(id(dataset.relative(f.file())), f);
        }
        Map<String, Label> result = new LinkedHashMap<>();
        for (Map.Entry<String, Label> e : old.entrySet()) {
            OwnDataset.Frame f = byId.get(id(e.getKey()));
            if (f != null) {
                result.put(dataset.relative(f.file()), e.getValue());
            }
        }
        if (!result.keySet().equals(old.keySet()) && Files.exists(csv)) {
            Path backup = csv.resolveSibling("eyes_before_rename.csv");
            if (!Files.exists(backup)) {
                Files.copy(csv, backup);
            }
            System.out.println("eyes.csv: перенесено " + result.size() + " из " + old.size() + " строк; прежний файл — " + backup);
        }
        return result;
    }

    /** Ключ кадра без пометки: «человек/секунда/номер». */
    private static String id(String relative) {
        int slash = relative.indexOf('/');
        java.util.regex.Matcher m = OwnDataset.NAME.matcher(relative.substring(slash + 1));
        if (!m.matches()) return relative;
        return relative.substring(0, slash) + "/" + m.group(2) + m.group(3) + m.group(4) + "/" + (m.group(5) == null ? "-1" : m.group(5));
    }

    private static double num(String s) {
        return s.isBlank() ? Double.NaN : Double.parseDouble(s);
    }

    private static String fmt(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.ROOT, "%.1f", v);
    }

    /** Переписывает CSV целиком (через временный файл) в порядке кадров. */
    private void writeCsv() throws IOException {
        Path tmp = csv.resolveSibling(CSV_NAME + ".tmp");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(tmp, StandardCharsets.UTF_8))) {
            out.println("# Центры зрачков, ручная разметка (EyeLabeler). l — левый на снимке (правый глаз человека),");
            out.println("# r — правый на снимке (левый глаз человека). Координаты в пикселях полного снимка после поворота");
            out.println("# по EXIF (3000x4000). flag: ok / doubtful (блик, прикрытый глаз) / skipped (не виден, координат нет).");
            out.println("file,lx,ly,rx,ry,flag");
            for (OwnDataset.Frame f : frames) {
                String key = dataset.relative(f.file());
                Label l = labels.get(key);
                if (l != null) {
                    out.println(key + "," + fmt(l.lx()) + "," + fmt(l.ly()) + "," + fmt(l.rx()) + "," + fmt(l.ry()) + "," + l.flag());
                }
            }
        }
        Files.move(tmp, csv, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    // ---------------------------------------------------------------- окно

    private void showWindow() {
        window = new JFrame("Eye labeling — " + dataset.root());
        window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        hint = new JLabel(" ");
        hint.setFont(hint.getFont().deriveFont(Font.BOLD, 16f));
        chooser = new JComboBox<>();
        chooser.setFocusable(false);
        chooser.addActionListener(this::chooserActionPerformed);
        JPanel top = new JPanel(new BorderLayout(8, 0));
        top.add(hint, BorderLayout.CENTER);
        top.add(chooser, BorderLayout.EAST);
        panel = new ImagePanel();
        scroll = new JScrollPane(panel);
        scroll.getVerticalScrollBar().setUnitIncrement(24);
        scroll.getHorizontalScrollBar().setUnitIncrement(24);
        window.getContentPane().add(top, BorderLayout.NORTH);
        window.getContentPane().add(scroll, BorderLayout.CENTER);
        bindKey(KeyEvent.VK_ENTER, "save-ok", () -> save("ok"));
        bindKey(KeyEvent.VK_D, "save-doubtful", () -> save("doubtful"));
        bindKey(KeyEvent.VK_S, "skip", this::skip);
        bindKey(KeyEvent.VK_BACK_SPACE, "undo", this::undo);
        bindKey(KeyEvent.VK_F, "whole", this::showWhole);
        window.setExtendedState(JFrame.MAXIMIZED_BOTH);
        window.setSize(1400, 1000);
        window.setVisible(true);
        refreshChooser();
        int next = nextUnlabeled(0);
        if (next < 0) {
            JOptionPane.showMessageDialog(window, "Все " + frames.size() + " кадров размечены. Кадр для перезаписи — в списке вверху.");
            next = 0;
        }
        load(next);
    }

    private void bindKey(int key, String name, Runnable action) {
        JComponent root = window.getRootPane();
        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(key, 0), name);
        root.getActionMap().put(name, new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                action.run();
            }
        });
    }

    private void chooserActionPerformed(ActionEvent e) {
        if (!updatingChooser && chooser.getSelectedIndex() >= 0 && chooser.getSelectedIndex() != current) {
            load(chooser.getSelectedIndex());
        }
    }

    private void refreshChooser() {
        updatingChooser = true;
        DefaultComboBoxModel<String> model = new DefaultComboBoxModel<>();
        for (OwnDataset.Frame f : frames) {
            String key = dataset.relative(f.file());
            Label l = labels.get(key);
            model.addElement((l == null ? "[ ]  " : "[" + l.flag() + "]  ") + key);
        }
        chooser.setModel(model);
        if (current >= 0) {
            chooser.setSelectedIndex(current);
        }
        updatingChooser = false;
    }

    private int nextUnlabeled(int from) {
        for (int k = 0; k < frames.size(); k++) {
            int i = (from + k) % frames.size();
            if (!labels.containsKey(dataset.relative(frames.get(i).file()))) {
                return i;
            }
        }
        return -1;
    }

    // ---------------------------------------------------------------- кадр

    private void load(int index) {
        try {
            current = index;
            clicks.clear();
            image = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(frames.get(index).file())), Imgcodecs.IMREAD_COLOR);
            double[] face = detect(image);
            if (face == null) {
                showWhole();
            } else {
                showFace(face);
            }
            updatingChooser = true;
            chooser.setSelectedIndex(index);
            updatingChooser = false;
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(window, "Не удалось прочитать " + frames.get(index).file() + ": " + ex.getMessage());
        }
    }

    /** Весь кадр по размеру окна; следующий щелчок указывает лицо (YuNet не нашёл или нашёл не то — клавиша F). */
    private void showWhole() {
        if (image == null) {
            return;
        }
        facePick = true;
        clicks.clear();
        crop = new Rect(0, 0, image.cols(), image.rows());
        Dimension view = scroll.getViewport().getExtentSize();
        scale = Math.min((view.width - 4) / (double) image.cols(), (view.height - 4) / (double) image.rows());
        if (scale <= 0) {
            scale = 0.2;
        }
        show(null);
    }

    /** {x, y, w, h, межзрачковое расстояние, центр глаз x, y} в полном снимке или null. */
    private double[] detect(Mat full) {
        Mat small = new Mat();
        Imgproc.resize(full, small, new Size(Math.round(full.cols() * NAV_SCALE), Math.round(full.rows() * NAV_SCALE)),
                0, 0, Imgproc.INTER_AREA);
        detector.setInputSize(small.size());
        Mat faces = new Mat();
        detector.detect(small, faces);
        if (faces.rows() == 0) {
            return null;
        }
        int best = 0;
        for (int i = 1; i < faces.rows(); i++) {
            if (faces.get(i, 14)[0] > faces.get(best, 14)[0]) {
                best = i;
            }
        }
        double[] v = new double[15];
        for (int j = 0; j < 15; j++) {
            v[j] = faces.get(best, j)[0] / NAV_SCALE;
        }
        double ipd = Math.hypot(v[6] - v[4], v[7] - v[5]);
        return new double[] {v[0], v[1], v[2], v[3], ipd, (v[4] + v[6]) / 2, (v[5] + v[7]) / 2};
    }

    /** Показ области лица крупно; face — {x, y, w, h, ipd, cx, cy}. */
    private void showFace(double[] face) {
        facePick = false;
        double mx = face[2] * MARGIN;
        double my = face[3] * MARGIN;
        int x0 = (int) Math.max(0, Math.floor(face[0] - mx));
        int y0 = (int) Math.max(0, Math.floor(face[1] - my));
        int x1 = (int) Math.min(image.cols(), Math.ceil(face[0] + face[2] + mx));
        int y1 = (int) Math.min(image.rows(), Math.ceil(face[1] + face[3] + my));
        crop = new Rect(x0, y0, x1 - x0, y1 - y0);
        double ipd = Math.max(face[4], 10);
        scale = Math.max(1.0, (MIN_IPD_SCREEN + 20) / ipd);
        show(new double[] {face[5], face[6]});
    }

    /** Строит увеличенный фрагмент и прокручивает к точке центра (в полном снимке) или к началу. */
    private void show(double[] center) {
        Mat part = new Mat(image, crop);
        Mat big = new Mat();
        Imgproc.resize(part, big, new Size(Math.round(crop.width * scale), Math.round(crop.height * scale)), 0, 0,
                scale > 1 ? Imgproc.INTER_CUBIC : Imgproc.INTER_AREA);
        panel.setImage(toImage(big));
        updateHint();
        SwingUtilities.invokeLater(() -> {
            Dimension view = scroll.getViewport().getExtentSize();
            int cx = center == null ? 0 : (int) ((center[0] - crop.x) * scale - view.width / 2.0);
            int cy = center == null ? 0 : (int) ((center[1] - crop.y) * scale - view.height / 2.0);
            panel.scrollRectToVisible(new Rectangle(Math.max(0, cx), Math.max(0, cy), view.width, view.height));
        });
    }

    private void updateHint() {
        String key = dataset.relative(frames.get(current).file());
        long done = labels.size();
        String state;
        if (facePick) {
            state = "Щёлкните по лицу";
        } else if (clicks.isEmpty()) {
            state = "Щелчок 1: зрачок ЛЕВЫЙ НА СНИМКЕ (правый глаз человека)";
        } else if (clicks.size() == 1) {
            state = "Щелчок 2: зрачок ПРАВЫЙ НА СНИМКЕ (левый глаз человека)";
        } else {
            state = "Enter — сохранить, D — сомнительно, Backspace — отменить щелчок";
        }
        hint.setText(String.format(Locale.ROOT, "  %s   (%d/%d размечено)   %s   [S — не виден, F — весь кадр]",
                key, done, frames.size(), state));
        panel.repaint();
    }

    private void click(double px, double py) {
        double x = crop.x + px / scale;
        double y = crop.y + py / scale;
        if (facePick) {
            double w = 0.15 * image.cols();
            double h = 1.25 * w;
            showFace(new double[] {x - w / 2, y - h / 2, w, h, 0.4 * w, x, y - 0.1 * h});
            return;
        }
        if (clicks.size() < 2) {
            clicks.add(new double[] {x, y});
            updateHint();
        }
    }

    private void undo() {
        if (!clicks.isEmpty()) {
            clicks.remove(clicks.size() - 1);
            updateHint();
        }
    }

    private void save(String flag) {
        if (facePick || clicks.size() < 2) {
            return;
        }
        double[] l = clicks.get(0);
        double[] r = clicks.get(1);
        store(new Label(l[0], l[1], r[0], r[1], flag));
    }

    private void skip() {
        store(new Label(Double.NaN, Double.NaN, Double.NaN, Double.NaN, "skipped"));
    }

    private void store(Label label) {
        labels.put(dataset.relative(frames.get(current).file()), label);
        try {
            writeCsv();
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(window, "Не удалось записать " + csv + ": " + ex.getMessage());
            return;
        }
        refreshChooser();
        int next = nextUnlabeled(current + 1);
        if (next < 0) {
            updateHint();
            JOptionPane.showMessageDialog(window, "Все " + frames.size() + " кадров размечены. Проверка: "
                    + "mvn compile exec:java@faces-eyes -Dexec.args=sheet");
            return;
        }
        load(next);
    }

    static BufferedImage toImage(Mat bgr) {
        BufferedImage img = new BufferedImage(bgr.cols(), bgr.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] data = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        bgr.get(0, 0, data);
        return img;
    }

    /** Увеличенный фрагмент с точками; щелчки — в пикселях фрагмента. */
    private final class ImagePanel extends JPanel {

        private BufferedImage img;

        ImagePanel() {
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    if (img != null && e.getButton() == MouseEvent.BUTTON1
                            && e.getX() < img.getWidth() && e.getY() < img.getHeight()) {
                        click(e.getX() + 0.5, e.getY() + 0.5);
                    }
                }
            });
        }

        void setImage(BufferedImage img) {
            this.img = img;
            setPreferredSize(new Dimension(img.getWidth(), img.getHeight()));
            revalidate();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (img == null) {
                return;
            }
            Graphics2D g2 = (Graphics2D) g;
            g2.drawImage(img, 0, 0, null);
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setStroke(new BasicStroke(1.5f));
            Label old = current < 0 ? null : labels.get(dataset.relative(frames.get(current).file()));
            if (old != null && !Double.isNaN(old.lx()) && !facePick) {
                g2.setColor(Color.YELLOW);
                cross(g2, old.lx(), old.ly(), "old L");
                cross(g2, old.rx(), old.ry(), "old R");
            }
            Color[] colors = {Color.RED, Color.GREEN};
            String[] names = {"L", "R"};
            for (int i = 0; i < clicks.size(); i++) {
                g2.setColor(colors[i]);
                cross(g2, clicks.get(i)[0], clicks.get(i)[1], names[i]);
            }
        }

        private void cross(Graphics2D g2, double x, double y, String text) {
            int px = (int) Math.round((x - crop.x) * scale);
            int py = (int) Math.round((y - crop.y) * scale);
            g2.drawLine(px - 12, py, px - 3, py);
            g2.drawLine(px + 3, py, px + 12, py);
            g2.drawLine(px, py - 12, px, py - 3);
            g2.drawLine(px, py + 3, px, py + 12);
            g2.drawString(text, px + 8, py - 8);
        }
    }

    // ---------------------------------------------------------------- листы проверки

    private static final int TILE = 240;
    private static final int COLS = 5;
    private static final int PER_SHEET = 20;

    /** Листы проверки: область глаз крупно, точки L (красная) и R (зелёная), подписи латиницей. */
    private void writeSheets(Path outDir) throws IOException {
        Files.createDirectories(outDir);
        int sheets = (frames.size() + PER_SHEET - 1) / PER_SHEET;
        int counted = 0;
        for (int s = 0; s < sheets; s++) {
            int n = Math.min(PER_SHEET, frames.size() - s * PER_SHEET);
            int rows = (n + COLS - 1) / COLS;
            Mat sheet = new Mat(rows * (TILE + 18), COLS * TILE, org.opencv.core.CvType.CV_8UC3, new Scalar(255, 255, 255));
            for (int k = 0; k < n; k++) {
                OwnDataset.Frame f = frames.get(s * PER_SHEET + k);
                String key = dataset.relative(f.file());
                Label l = labels.get(key);
                int x0 = (k % COLS) * TILE;
                int y0 = (k / COLS) * (TILE + 18);
                Mat tile = new Mat(TILE, TILE, org.opencv.core.CvType.CV_8UC3, new Scalar(200, 200, 200));
                String flag = l == null ? "none" : l.flag();
                if (l != null && !Double.isNaN(l.lx())) {
                    counted++;
                    Mat full = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(f.file())), Imgcodecs.IMREAD_COLOR);
                    double ipd = Math.hypot(l.rx() - l.lx(), l.ry() - l.ly());
                    double side = Math.max(3 * ipd, 60);
                    double cx = (l.lx() + l.rx()) / 2;
                    double cy = (l.ly() + l.ry()) / 2;
                    int rx0 = (int) Math.max(0, Math.round(cx - side / 2));
                    int ry0 = (int) Math.max(0, Math.round(cy - side / 2));
                    int rw = (int) Math.min(full.cols() - rx0, Math.round(side));
                    int rh = (int) Math.min(full.rows() - ry0, Math.round(side));
                    double sc = TILE / side;
                    Imgproc.resize(new Mat(full, new Rect(rx0, ry0, rw, rh)), tile,
                            new Size(Math.round(rw * sc), Math.round(rh * sc)), 0, 0, Imgproc.INTER_AREA);
                    Mat padded = new Mat(TILE, TILE, org.opencv.core.CvType.CV_8UC3, new Scalar(200, 200, 200));
                    Mat roi = padded.submat(0, Math.min(TILE, tile.rows()), 0, Math.min(TILE, tile.cols()));
                    tile.submat(0, roi.rows(), 0, roi.cols()).copyTo(roi);
                    tile = padded;
                    Point pl = new Point((l.lx() - rx0) * sc, (l.ly() - ry0) * sc);
                    Point pr = new Point((l.rx() - rx0) * sc, (l.ry() - ry0) * sc);
                    Imgproc.drawMarker(tile, pl, new Scalar(0, 0, 255), Imgproc.MARKER_CROSS, 20, 2);
                    Imgproc.drawMarker(tile, pr, new Scalar(0, 200, 0), Imgproc.MARKER_CROSS, 20, 2);
                }
                tile.copyTo(sheet.submat(y0, y0 + TILE, x0, x0 + TILE));
                String name = f.file().getFileName().toString().replace(".jpg", "");
                String caption = f.file().getParent().getFileName() + " " + name.substring(name.indexOf('_') + 1) + " " + flag;
                Imgproc.putText(sheet, caption, new Point(x0 + 3, y0 + TILE + 13), Imgproc.FONT_HERSHEY_PLAIN, 0.9,
                        new Scalar(0, 0, 0), 1);
                Imgproc.rectangle(sheet, new Point(x0, y0), new Point(x0 + TILE - 1, y0 + TILE - 1), new Scalar(128, 128, 128), 1);
            }
            Path out = outDir.resolve(String.format(Locale.ROOT, "eyes_sheet_%d.png", s + 1));
            Imgcodecs.imwrite(out.toString(), sheet);
            System.out.println("Лист: " + out);
        }
        System.out.println("Размечено с точками: " + counted + " из " + frames.size() + "; CSV: " + csv);
    }
}
