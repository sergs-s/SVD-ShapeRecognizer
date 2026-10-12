package svd.recognizer.faces;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
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
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.storage.SettingsStore;

/**
 * Ручная разметка личностей Caltech Face Dataset 1999 (450 снимков, около 27 человек без разметки; PLAN.md п. 5.5,
 * сверка): какой снимок к какому человеку относится. По образцу EyeLabeler.
 *
 * Экран: слева — текущий снимок, лицо крупно по рамке YuNet с полями (не найдено — весь кадр; F — весь кадр / лицо);
 * справа — заведённые люди: миниатюра-представитель (первый в порядке файлов снимок со статусом ok), номер и число
 * снимков. Подсказка SFace (1 − cos к среднему признаку человека) — рамка вокруг миниатюры; сама ничего не относит.
 *
 * Действия: щелчок по миниатюре — отнести к человеку; Enter — отнести к подсвеченному (подсказка); N — новый человек;
 * D — «сомнительно» (в разметку не входит); X — «не лицо / исключить»; Backspace — отменить последнее действие;
 * ← — предыдущий показанный снимок (можно переразметить).
 *
 * Порядок показа: пока людей нет — по порядку файлов; затем неразмеченный снимок с наибольшим сходством SFace с одним
 * из размеченных людей (только подсказка), снимки без лица — в конце по порядку файлов.
 *
 * Файл identities.csv в каталоге базы (вне git): file;person;status;time, статус ok / doubtful / excluded, у doubtful и
 * excluded человек «-»; одна строка на снимок (по порядку файлов); запись после каждого действия через временный файл,
 * перед записью прежний копируется в identities.csv.bak; при запуске — продолжение. Детекции и признаки SFace —
 * кэш caltech_yunet_sface.tsv там же.
 *
 * Режим проверки (аргумент sheet): листы «человек — все его снимки» с именами файлов и лист сомнительных и исключённых,
 * reports/faces/caltech/ (вне git).
 *
 * Запуск: mvn compile exec:java@faces-caltech (каталог — faces.caltech.dir / FACES_CALTECH_DIR); проверка:
 * mvn compile exec:java@faces-caltech -Dexec.args=sheet
 *
 * @author ssv
 */
public final class CaltechLabeler {

    static final String CSV_NAME = "identities.csv";
    static final String CACHE_NAME = "caltech_yunet_sface.tsv";
    static final String OK = "ok";
    static final String DOUBTFUL = "doubtful";
    static final String EXCLUDED = "excluded";
    static final String NONE = "-";
    /** Порог уверенности YuNet для показа и подсказки. */
    private static final float NAV_SCORE = 0.5f;
    /** Поле вокруг рамки лица, доля её размера с каждой стороны. */
    private static final double MARGIN = 0.5;
    private static final int THUMB = 96;
    private static final int VIEW = 640;

    /** Разметка снимка. */
    record Label(String person, String status, String time) {}

    /** Детекция снимка: рамка {x, y, w, h} и нормированный признак SFace; null — лицо не найдено. */
    record Det(double[] box, double[] feature) {}

    private final Path dir;
    private final List<String> files;
    private final Map<String, Label> labels;
    private final Map<String, Det> dets;
    private final Path csv;

    /** История действий: снимок и его разметка до действия (null — не был размечен). */
    private final Deque<Object[]> history = new ArrayDeque<>();
    /** Показанные снимки (для «предыдущий»). */
    private final Deque<String> shown = new ArrayDeque<>();
    private String current;
    private boolean whole;
    private String suggestion;

    private JFrame window;
    private JLabel hint;
    private JLabel picture;
    private JPanel people;
    private final Map<String, BufferedImage> thumbs = new HashMap<>();

    private CaltechLabeler(Path dir, List<String> files, Map<String, Det> dets) throws IOException {
        this.dir = dir;
        this.files = files;
        this.dets = dets;
        this.csv = dir.resolve(CSV_NAME);
        this.labels = readCsv(csv);
    }

    public static void main(String[] args) throws Exception {
        svd.recognizer.OpenCvLoader.load();
        SettingsStore settings = new SettingsStore();
        String raw = settings.loadFacesCaltechDir();
        if (raw == null) throw new IllegalStateException("faces-caltech: не задан faces.caltech.dir / FACES_CALTECH_DIR");
        Path dir = Paths.get(raw);
        List<String> files;
        try (Stream<Path> s = Files.list(dir)) {
            files = s.map(p -> p.getFileName().toString()).filter(n -> n.toLowerCase(Locale.ROOT).endsWith(".jpg")).sorted().toList();
        }
        Map<String, Det> dets = detections(dir, files, Paths.get(settings.loadFacesModelsDir()));
        CaltechLabeler labeler = new CaltechLabeler(dir, files, dets);
        long found = dets.values().stream().filter(d -> d != null).count();
        System.out.println("Снимков: " + files.size() + ", лицо найдено: " + found + ", размечено: " + labeler.labels.size());
        if (args.length > 0 && args[0].equals("sheet")) {
            labeler.writeSheets(Paths.get(System.getProperty("user.dir"), "reports", "faces", "caltech"));
        } else {
            SwingUtilities.invokeLater(labeler::showWindow);
        }
    }

    // ---------------------------------------------------------------- детекция и признаки

    /** Детекции YuNet и признаки SFace всех снимков: из кэша или заново (тогда кэш записывается). */
    static Map<String, Det> detections(Path dir, List<String> files, Path modelsDir) throws IOException {
        Path cache = dir.resolve(CACHE_NAME);
        Map<String, Det> out = new LinkedHashMap<>();
        if (Files.exists(cache)) {
            for (String l : Files.readAllLines(cache, StandardCharsets.UTF_8)) {
                if (l.startsWith("#") || l.isBlank()) continue;
                String[] c = l.split("\t", -1);
                if (c[1].equals("0")) {
                    out.put(c[0], null);
                    continue;
                }
                double[] box = new double[4];
                for (int i = 0; i < 4; i++) box[i] = Double.parseDouble(c[2 + i]);
                double[] f = new double[c.length - 6];
                for (int i = 0; i < f.length; i++) f[i] = Double.parseDouble(c[6 + i]);
                out.put(c[0], new Det(box, f));
            }
            if (out.keySet().containsAll(files)) return out;
            out.clear();
        }
        FaceDetectorYN detector = FaceDetectorYN.create(modelsDir.resolve(FaceAlignment.YUNET_FILE).toString(), "", new Size(320, 320),
                NAV_SCORE, 0.3f, 5000);
        FaceRecognizerSF recognizer = FaceRecognizerSF.create(modelsDir.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        StringBuilder t = new StringBuilder("# Caltech1999: YuNet (порог " + NAV_SCORE + ", лицо с наибольшей уверенностью) и признак SFace "
                + "(alignCrop, L2-норма 1); file, found, x, y, w, h, признак\n");
        for (String f : files) {
            Mat img = read(dir.resolve(f));
            detector.setInputSize(img.size());
            Mat faces = new Mat();
            detector.detect(img, faces);
            if (faces.rows() == 0) {
                out.put(f, null);
                t.append(f).append("\t0\n");
                continue;
            }
            int best = 0;
            for (int i = 1; i < faces.rows(); i++) if (faces.get(i, 14)[0] > faces.get(best, 14)[0]) best = i;
            Mat face = faces.row(best);
            Mat crop = new Mat();
            recognizer.alignCrop(img, face, crop);
            Mat feat = new Mat();
            recognizer.feature(crop, feat);
            double[] v = new double[(int) feat.total()];
            for (int i = 0; i < v.length; i++) v[i] = feat.get(0, i)[0];
            normalize(v);
            double[] box = {face.get(0, 0)[0], face.get(0, 1)[0], face.get(0, 2)[0], face.get(0, 3)[0]};
            out.put(f, new Det(box, v));
            t.append(f).append("\t1");
            for (double b : box) t.append('\t').append(b);
            for (double x : v) t.append('\t').append(x);
            t.append('\n');
        }
        Files.writeString(cache, t, StandardCharsets.UTF_8);
        return out;
    }

    static Mat read(Path file) throws IOException {
        Mat img = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(file)), Imgcodecs.IMREAD_COLOR);
        if (img.empty()) throw new IOException("Не удалось декодировать " + file);
        return img;
    }

    static void normalize(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        s = Math.sqrt(s);
        if (s > 0) for (int i = 0; i < v.length; i++) v[i] /= s;
    }

    // ---------------------------------------------------------------- файл разметки

    static Map<String, Label> readCsv(Path file) throws IOException {
        Map<String, Label> map = new LinkedHashMap<>();
        if (!Files.exists(file)) return map;
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#") || line.startsWith("file;")) continue;
            String[] f = line.split(";", -1);
            map.put(f[0], new Label(f[1], f[2], f[3]));
        }
        return map;
    }

    /** Переписывает файл целиком в порядке files: перед записью — копия .bak, запись через временный файл. */
    static void writeCsv(Path file, List<String> files, Map<String, Label> labels) throws IOException {
        if (Files.exists(file)) Files.copy(file, file.resolveSibling(file.getFileName() + ".bak"), StandardCopyOption.REPLACE_EXISTING);
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(tmp, StandardCharsets.UTF_8))) {
            out.println("# Caltech1999: ручная разметка личностей (CaltechLabeler). status: ok — снимок человека person; doubtful — сомнительно");
            out.println("# (в разметку не входит); excluded — не лицо / исключён. У doubtful и excluded person = -.");
            out.println("file;person;status;time");
            for (String f : files) {
                Label l = labels.get(f);
                if (l != null) out.println(f + ";" + l.person() + ";" + l.status() + ";" + l.time());
            }
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /** Новый номер человека: p01, p02, … — на единицу больше наибольшего заведённого. */
    static String nextPerson(Collection<Label> labels) {
        int max = 0;
        for (Label l : labels) if (l.person().startsWith("p")) max = Math.max(max, Integer.parseInt(l.person().substring(1)));
        return String.format(Locale.ROOT, "p%02d", max + 1);
    }

    /** Люди со снимками ok по порядку номеров: человек → снимки в порядке files. */
    static Map<String, List<String>> people(List<String> files, Map<String, Label> labels) {
        Map<String, List<String>> out = new TreeMap<>();
        for (String f : files) {
            Label l = labels.get(f);
            if (l != null && l.status().equals(OK)) out.computeIfAbsent(l.person(), k -> new ArrayList<>()).add(f);
        }
        return out;
    }

    /** Средние нормированные признаки людей (по снимкам ok с лицом). */
    static Map<String, double[]> centroids(Map<String, List<String>> people, Map<String, Det> dets) {
        Map<String, double[]> out = new TreeMap<>();
        for (Map.Entry<String, List<String>> e : people.entrySet()) {
            double[] c = null;
            for (String f : e.getValue()) {
                Det d = dets.get(f);
                if (d == null) continue;
                if (c == null) c = new double[d.feature().length];
                for (int i = 0; i < c.length; i++) c[i] += d.feature()[i];
            }
            if (c != null) {
                normalize(c);
                out.put(e.getKey(), c);
            }
        }
        return out;
    }

    /** Ближайший человек снимка по косинусу {номер человека, 1 − cos}; нет признака или людей — null. */
    static Object[] nearest(Det d, Map<String, double[]> centroids) {
        if (d == null || centroids.isEmpty()) return null;
        String best = null;
        double bestCos = -2;
        for (Map.Entry<String, double[]> e : centroids.entrySet()) {
            double cos = 0;
            for (int i = 0; i < d.feature().length; i++) cos += d.feature()[i] * e.getValue()[i];
            if (cos > bestCos) {
                bestCos = cos;
                best = e.getKey();
            }
        }
        return new Object[] {best, 1 - bestCos};
    }

    /**
     * Следующий снимок: людей нет — первый неразмеченный по порядку файлов; иначе неразмеченный с наименьшим 1 − cos к
     * среднему одного из людей (при равенстве — раньше по порядку), снимки без лица — после всех, по порядку. Все
     * размечены — null.
     */
    static String next(List<String> files, Map<String, Label> labels, Map<String, Det> dets) {
        Map<String, double[]> c = centroids(people(files, labels), dets);
        String best = null;
        double bestD = Double.POSITIVE_INFINITY;
        String firstNoFace = null;
        for (String f : files) {
            if (labels.containsKey(f)) continue;
            if (c.isEmpty()) return f;
            Object[] n = nearest(dets.get(f), c);
            if (n == null) {
                if (firstNoFace == null) firstNoFace = f;
                continue;
            }
            if ((double) n[1] < bestD) {
                bestD = (double) n[1];
                best = f;
            }
        }
        return best != null ? best : firstNoFace;
    }

    // ---------------------------------------------------------------- окно

    private void showWindow() {
        window = new JFrame("Caltech1999 — разметка личностей — " + dir);
        window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        hint = new JLabel(" ");
        hint.setFont(hint.getFont().deriveFont(Font.BOLD, 15f));
        JLabel keys = new JLabel("  Щелчок по человеку — отнести; Enter — к подсвеченному; N — новый человек; D — сомнительно; "
                + "X — не лицо / исключить; Backspace — отменить; ← — предыдущий; F — весь кадр / лицо");
        JPanel top = new JPanel(new BorderLayout());
        top.add(hint, BorderLayout.NORTH);
        top.add(keys, BorderLayout.SOUTH);
        picture = new JLabel();
        picture.setHorizontalAlignment(SwingConstants.CENTER);
        picture.setPreferredSize(new Dimension(VIEW + 20, VIEW + 20));
        people = new JPanel(new GridLayout(0, 4, 6, 6));
        JPanel peopleWrap = new JPanel(new BorderLayout());
        peopleWrap.add(people, BorderLayout.NORTH);
        JScrollPane right = new JScrollPane(peopleWrap);
        right.setPreferredSize(new Dimension(4 * (THUMB + 16) + 30, VIEW));
        right.getVerticalScrollBar().setUnitIncrement(24);
        window.getContentPane().add(top, BorderLayout.NORTH);
        window.getContentPane().add(picture, BorderLayout.CENTER);
        window.getContentPane().add(right, BorderLayout.EAST);
        bindKey(KeyEvent.VK_ENTER, "accept", this::acceptSuggestion);
        bindKey(KeyEvent.VK_N, "new", this::newPerson);
        bindKey(KeyEvent.VK_D, "doubtful", () -> store(NONE, DOUBTFUL));
        bindKey(KeyEvent.VK_X, "excluded", () -> store(NONE, EXCLUDED));
        bindKey(KeyEvent.VK_BACK_SPACE, "undo", this::undo);
        bindKey(KeyEvent.VK_LEFT, "previous", this::previous);
        bindKey(KeyEvent.VK_F, "whole", this::toggleWhole);
        window.pack();
        window.setExtendedState(JFrame.MAXIMIZED_BOTH);
        window.setVisible(true);
        String first = next(files, labels, dets);
        if (first == null) {
            JOptionPane.showMessageDialog(window, "Все " + files.size() + " снимков размечены. Проверка: mvn compile exec:java@faces-caltech "
                    + "-Dexec.args=sheet");
            first = files.get(0);
        }
        show(first, true);
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

    /** Показ снимка f; remember — запомнить предыдущий для «←». */
    private void show(String f, boolean remember) {
        if (remember && current != null && !current.equals(f)) shown.push(current);
        current = f;
        whole = false;
        render();
    }

    private void render() {
        try {
            Mat img = read(dir.resolve(current));
            Det d = dets.get(current);
            Mat part = whole || d == null ? img : new Mat(img, faceRect(d, img));
            double s = Math.min(VIEW / (double) part.cols(), VIEW / (double) part.rows());
            Mat big = new Mat();
            Imgproc.resize(part, big, new Size(Math.round(part.cols() * s), Math.round(part.rows() * s)), 0, 0,
                    s > 1 ? Imgproc.INTER_CUBIC : Imgproc.INTER_AREA);
            picture.setIcon(new ImageIcon(EyeLabeler.toImage(big)));
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(window, "Не удалось прочитать " + current + ": " + ex.getMessage());
        }
        Map<String, List<String>> ppl = people(files, labels);
        Object[] n = nearest(dets.get(current), centroids(ppl, dets));
        suggestion = n == null ? null : (String) n[0];
        Label l = labels.get(current);
        long done = labels.size();
        hint.setText(String.format(Locale.ROOT, "  %s   (%d из %d; размечено %d, людей %d)   %s   %s%s", current, files.indexOf(current) + 1,
                files.size(), done, ppl.size(), l == null ? "не размечен" : "разметка: " + l.person() + " " + l.status(),
                n == null ? "подсказки нет" : String.format(Locale.ROOT, "подсказка SFace: %s, 1 − cos = %.3f", n[0], (double) n[1]),
                dets.get(current) == null ? "   [лицо не найдено — весь кадр]" : ""));
        refreshPeople(ppl);
    }

    private static Rect faceRect(Det d, Mat img) {
        double[] b = d.box();
        double mx = b[2] * MARGIN;
        double my = b[3] * MARGIN;
        int x0 = (int) Math.max(0, Math.floor(b[0] - mx));
        int y0 = (int) Math.max(0, Math.floor(b[1] - my));
        int x1 = (int) Math.min(img.cols(), Math.ceil(b[0] + b[2] + mx));
        int y1 = (int) Math.min(img.rows(), Math.ceil(b[1] + b[3] + my));
        return new Rect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    }

    private void refreshPeople(Map<String, List<String>> ppl) {
        people.removeAll();
        for (Map.Entry<String, List<String>> e : ppl.entrySet()) {
            JLabel t = new JLabel(e.getKey() + " (" + e.getValue().size() + ")", new ImageIcon(thumb(e.getValue().get(0))), SwingConstants.CENTER);
            t.setVerticalTextPosition(SwingConstants.BOTTOM);
            t.setHorizontalTextPosition(SwingConstants.CENTER);
            boolean hl = e.getKey().equals(suggestion);
            t.setBorder(BorderFactory.createLineBorder(hl ? Color.RED : Color.LIGHT_GRAY, hl ? 4 : 1));
            t.addMouseListener(new PersonClick(e.getKey()));
            people.add(t);
        }
        people.revalidate();
        people.repaint();
    }

    /** Щелчок по миниатюре человека. */
    private final class PersonClick extends MouseAdapter {
        private final String person;

        PersonClick(String person) {
            this.person = person;
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            store(person, OK);
        }
    }

    private BufferedImage thumb(String f) {
        return thumbs.computeIfAbsent(f, k -> {
            try {
                Mat img = read(dir.resolve(k));
                Det d = dets.get(k);
                Mat part = d == null ? img : new Mat(img, faceRect(d, img));
                double s = Math.min(THUMB / (double) part.cols(), THUMB / (double) part.rows());
                Mat small = new Mat();
                Imgproc.resize(part, small, new Size(Math.max(1, Math.round(part.cols() * s)), Math.max(1, Math.round(part.rows() * s))), 0, 0,
                        Imgproc.INTER_AREA);
                return EyeLabeler.toImage(small);
            } catch (IOException ex) {
                return new BufferedImage(THUMB, THUMB, BufferedImage.TYPE_3BYTE_BGR);
            }
        });
    }

    private void acceptSuggestion() {
        if (suggestion != null) store(suggestion, OK);
    }

    private void newPerson() {
        store(nextPerson(labels.values()), OK);
    }

    private void toggleWhole() {
        whole = !whole;
        render();
    }

    private void previous() {
        if (!shown.isEmpty()) show(shown.pop(), false);
    }

    /** Действие над текущим снимком: запись, затем следующий снимок. */
    private void store(String person, String status) {
        history.push(new Object[] {current, labels.get(current)});
        labels.put(current, new Label(person, status, LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME).substring(0, 19)));
        if (!save()) return;
        String n = next(files, labels, dets);
        if (n == null) {
            render();
            JOptionPane.showMessageDialog(window, "Все " + files.size() + " снимков размечены. Проверка: mvn compile exec:java@faces-caltech "
                    + "-Dexec.args=sheet");
            return;
        }
        show(n, true);
    }

    /** Отмена последнего действия: прежняя разметка снимка восстанавливается, снимок показывается. */
    private void undo() {
        if (history.isEmpty()) return;
        Object[] h = history.pop();
        String f = (String) h[0];
        if (h[1] == null) labels.remove(f);
        else labels.put(f, (Label) h[1]);
        if (!save()) return;
        show(f, true);
    }

    private boolean save() {
        try {
            writeCsv(csv, files, labels);
            return true;
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(window, "Не удалось записать " + csv + ": " + ex.getMessage());
            return false;
        }
    }

    // ---------------------------------------------------------------- листы проверки

    private static final int TILE = 160;
    private static final int COLS = 6;

    /** Листы: на каждого человека — все его снимки (лицо по рамке, имя файла); отдельно — сомнительные и исключённые. */
    private void writeSheets(Path outDir) throws IOException {
        Files.createDirectories(outDir);
        Map<String, List<String>> groups = new LinkedHashMap<>(people(files, labels));
        List<String> other = new ArrayList<>();
        for (String f : files) {
            Label l = labels.get(f);
            if (l != null && !l.status().equals(OK)) other.add(f);
        }
        if (!other.isEmpty()) groups.put("doubtful_excluded", other);
        for (Map.Entry<String, List<String>> g : groups.entrySet()) {
            List<String> fs = g.getValue();
            int rows = (fs.size() + COLS - 1) / COLS;
            Mat sheet = new Mat(rows * (TILE + 18) + 24, COLS * TILE, org.opencv.core.CvType.CV_8UC3, new Scalar(255, 255, 255));
            Imgproc.putText(sheet, g.getKey() + ": " + fs.size() + " images", new Point(4, 17), Imgproc.FONT_HERSHEY_PLAIN, 1.2, new Scalar(0, 0, 0), 1);
            for (int k = 0; k < fs.size(); k++) {
                String f = fs.get(k);
                Mat img = read(dir.resolve(f));
                Det d = dets.get(f);
                Mat part = d == null ? img : new Mat(img, faceRect(d, img));
                double s = Math.min(TILE / (double) part.cols(), TILE / (double) part.rows());
                Mat small = new Mat();
                Imgproc.resize(part, small, new Size(Math.max(1, Math.round(part.cols() * s)), Math.max(1, Math.round(part.rows() * s))), 0, 0,
                        Imgproc.INTER_AREA);
                int x0 = (k % COLS) * TILE;
                int y0 = 24 + (k / COLS) * (TILE + 18);
                small.copyTo(sheet.submat(y0, y0 + small.rows(), x0, x0 + small.cols()));
                Label l = labels.get(f);
                String cap = f.replace(".jpg", "") + (l.status().equals(OK) ? "" : " " + l.status());
                Imgproc.putText(sheet, cap, new Point(x0 + 3, y0 + TILE + 13), Imgproc.FONT_HERSHEY_PLAIN, 0.9, new Scalar(0, 0, 0), 1);
            }
            Path out = outDir.resolve("caltech_" + g.getKey() + ".png");
            Imgcodecs.imwrite(out.toString(), sheet);
            System.out.println("Лист: " + out);
        }
        long ok = labels.values().stream().filter(l -> l.status().equals(OK)).count();
        System.out.println(String.format(Locale.ROOT, "Людей %d, снимков ok %d, сомнительных и исключённых %d, не размечено %d; файл %s",
                people(files, labels).size(), ok, other.size(), files.size() - labels.size(), csv));
    }
}
