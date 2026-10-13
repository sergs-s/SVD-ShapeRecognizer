package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Листы кадров у ошибок (PLAN.md п. 5.5, (а0); задание №6 от 13.10.2026): по имеющимся файлам оценок, без распознавания.
 * Решения — модель по позе пробы (X / X+mirror, XBaseEval.choose), порог П2 при FAR 1 % по «чужим» галереи
 * (FeiBootstrap.threshold, веса 1). Выборки: отказы своим (ii)+(iii) — до 50 с наибольшим r − θ; (iv) — все; принятые
 * контрольные чужие — до 30 с r, ближайшим к θ снизу; контроль — 30 случайных из (i) (Random(faces.seed + 9700), пробы по
 * ключу). Строка листа: исходный снимок с рамкой и 5 точками YuNet | кадр «а» | он же после Tan – Triggs (шкала [−τ, τ] →
 * 0–255, одна для всех) | ближайший снимок галереи (наименьший χ² гистограмм LBP того же признака без MLDA; для отказа и
 * контроля — свой, для ошибочного приёма — принятый) «а» и tt | подпись. Плюс листы галереи: все снимки галереи людей,
 * попавших в выборки, человек на строку. Режимы XBaseEval gt-a0 (GT межсессионно: L1, L1-tt; reports/faces/xbase/a0/) и
 * feret-a0 (FERET G1, L1-tt; только D:\data\Feret\reports\a0\). Индекс строк — a0_index.tsv, сводка — a0_summary.txt.
 *
 * @author ssv
 */
final class XBaseA0 {

    static final long OK_SEED_OFFSET = 9700;
    static final double FAR = 0.01;
    static final int MAX_REJ = 50;
    static final int MAX_CTRL = 30;
    static final int N_OK = 30;
    /** Строк на листе проб: 8 — высота листа 1884 px, смотрится в полном разрешении (≤ 2000). */
    static final int ROWS = 8;
    /** Рядов на листе галереи (≤ 2000 px по высоте). */
    static final int GAL_ROWS = 7;
    static final int SC = 2;
    static final int TW = 92 * SC;
    static final int TH = 112 * SC;
    static final int RAW_W = 300;
    static final int CAP_W = 360;
    static final int GAP = 8;
    static final int HEAD = 48;
    static final String[] GROUPS = {"rej", "iv", "ctrl-acc", "ok"};
    static final String[] GROUP_NAMES = {"(ii)+(iii) rejected own", "(iv) own accepted as other", "control accepted", "(i) correct accept"};

    private XBaseA0() {
    }

    /** Источник снимков базы: исходный снимок, строка YuNet, кадр «а», подпись даты. */
    interface Source {
        Mat raw(String key) throws IOException;

        double[] det(String key);

        Mat frameA(String key) throws IOException;

        String date(String key);

        String name(String key);

        /** |r| снимка (Gt.rOf). */
        double absR(String key);
    }

    /** Строка листа: проба, группа, оценка, кого показать из галереи. */
    record Case(Probe probe, int group, double ratio, String bestId, String galPerson) {}

    /** Пара «база – метод»: решения и выборки. */
    static final class Pair {
        String base;
        String label;
        IlluminationNorm norm;
        double theta;
        int[] groups = new int[4];
        int ownN;
        int ctrlN;
        int ctrlAcc;
        List<List<Case>> sel = new ArrayList<>();
    }

    // ---------------------------------------------------------------- решения и выборки

    /** Строки файла оценок методов base и base+mirror; block — заголовок блока «# конфигурация …» или null (весь файл). */
    static Map<String, Row> read(Path file, String base, String block) throws IOException {
        Map<String, Row> rows = new HashMap<>();
        boolean in = block == null;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8))) {
            String l;
            while ((l = r.readLine()) != null) {
                if (l.startsWith("# конфигурация")) {
                    in = block == null || l.equals(block);
                    continue;
                }
                if (!in || l.startsWith("#") || l.isEmpty() || l.equals(FeiScores.COLUMNS)) continue;
                Row row = FeiScores.parse(l);
                if (!row.method().equals(base) && !row.method().equals(base + FeiMethods.MIRROR)) continue;
                String k = row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image();
                if (rows.put(k, row) != null) throw new IllegalStateException("Повтор строки оценок " + k);
            }
        }
        if (rows.isEmpty()) throw new IllegalStateException("Нет строк " + base + " в " + file + (block != null ? " (" + block + ")" : ""));
        return rows;
    }

    /** Решения по позе пробы, порог П2 FAR 1 %, группы и выборки. */
    static Pair decide(String base, String label, IlluminationNorm norm, Map<String, Row> rows, XBaseEval.Gt g, Map<String, List<String>> galleryKeys,
                       List<Probe> probes, List<List<String>> parts) {
        Pair pr = new Pair();
        pr.base = base;
        pr.label = label;
        pr.norm = norm;
        List<double[]> cv = new ArrayList<>();
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
            for (String p : parts.get(q)) {
                for (String k : galleryKeys.get(p)) {
                    Row r = XBaseEval.choose(rows, V.L1, FeiMethods.CV_ROLE + q, p, g.img(k), g.rOf.get(k) <= g.r0);
                    cv.add(new double[] {g.pIdx.get(p), FeiScores.ratio(r, null)});
                }
            }
        }
        cv.sort((a, b) -> Double.compare(a[1], b[1]));
        double[] cvR = cv.stream().mapToDouble(a -> a[1]).toArray();
        int[] cvP = cv.stream().mapToInt(a -> (int) a[0]).toArray();
        pr.theta = FeiBootstrap.threshold(cvR, cvP, FeiBootstrap.unit(g.persons.size()), FAR);

        List<List<Case>> all = new ArrayList<>();
        for (int i = 0; i < 4; i++) all.add(new ArrayList<>());
        for (Probe p : probes) {
            Row r = XBaseEval.choose(rows, V.L1, p.role(), p.person(), p.image(), g.rOf.get(p.key()) <= g.r0);
            double ratio = FeiScores.ratio(r, null);
            boolean acc = ratio <= pr.theta;
            if (p.role().equals(FeiScores.OWN)) {
                pr.ownN++;
                boolean ok = r.bestId().equals(r.trueId());
                int grp = ok ? (acc ? 0 : 1) : (acc ? 3 : 2);
                pr.groups[grp]++;
                if (grp == 1 || grp == 2) all.get(0).add(new Case(p, 0, ratio, r.bestId(), p.person()));
                else if (grp == 3) all.get(1).add(new Case(p, 1, ratio, r.bestId(), r.bestId()));
                else all.get(3).add(new Case(p, 3, ratio, r.bestId(), p.person()));
            } else {
                pr.ctrlN++;
                if (acc) {
                    pr.ctrlAcc++;
                    all.get(2).add(new Case(p, 2, ratio, r.bestId(), r.bestId()));
                }
            }
        }
        List<Case> rej = new ArrayList<>(all.get(0));
        rej.sort((a, b) -> b.ratio() != a.ratio() ? Double.compare(b.ratio(), a.ratio()) : a.probe().key().compareTo(b.probe().key()));
        pr.sel.add(rej.subList(0, Math.min(MAX_REJ, rej.size())));
        List<Case> iv = new ArrayList<>(all.get(1));
        iv.sort((a, b) -> a.probe().key().compareTo(b.probe().key()));
        pr.sel.add(iv);
        List<Case> ca = new ArrayList<>(all.get(2));
        ca.sort((a, b) -> b.ratio() != a.ratio() ? Double.compare(b.ratio(), a.ratio()) : a.probe().key().compareTo(b.probe().key()));
        pr.sel.add(ca.subList(0, Math.min(MAX_CTRL, ca.size())));
        List<Case> ok = new ArrayList<>(all.get(3));
        ok.sort((a, b) -> a.probe().key().compareTo(b.probe().key()));
        Collections.shuffle(ok, new Random(g.seed + OK_SEED_OFFSET));
        List<Case> okSel = new ArrayList<>(ok.subList(0, Math.min(N_OK, ok.size())));
        okSel.sort((a, b) -> a.probe().key().compareTo(b.probe().key()));
        pr.sel.add(okSel);
        return pr;
    }

    static String summary(Pair p) {
        int err = p.groups[1] + p.groups[2] + p.groups[3];
        return String.format(Locale.ROOT, "%s %s: θ (П2, FAR 1 %%) = %.6f; свои %d, ошибка %d/%d = %.1f %%; группы (i) %d, (ii) %d, (iii) %d, (iv) %d; "
                + "контрольные чужие приняты %d/%d; в листах: отказы %d, (iv) %d, принятые чужие %d, контроль %d%n", p.base, p.label, p.theta, p.ownN, err,
                p.ownN, 100.0 * err / p.ownN, p.groups[0], p.groups[1], p.groups[2], p.groups[3], p.ctrlAcc, p.ctrlN, p.sel.get(0).size(),
                p.sel.get(1).size(), p.sel.get(2).size(), p.sel.get(3).size());
    }

    // ---------------------------------------------------------------- рисование

    /** Выход Tan – Triggs кадра «а» на общей шкале [−τ, τ] → 0–255, увеличенный в SC раз. */
    static Mat ttTile(Mat a, IlluminationNorm.NormParams prm) {
        double[] v = IlluminationNorm.tanTriggs(a, prm, true);
        double tau = prm.tau();
        Mat m = new Mat(a.rows(), a.cols(), CvType.CV_8UC1);
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) b[i] = (byte) (int) Math.round(Math.max(0, Math.min(255, (v[i] + tau) / (2 * tau) * 255)));
        m.put(0, 0, b);
        Mat t = tile(m);
        m.release();
        return t;
    }

    /** Серый кадр, увеличенный в SC раз (ближайший сосед), BGR. */
    static Mat tile(Mat gray) {
        Mat big = new Mat();
        Imgproc.resize(gray, big, new Size(gray.cols() * SC, gray.rows() * SC), 0, 0, Imgproc.INTER_NEAREST);
        Mat bgr = new Mat();
        Imgproc.cvtColor(big, bgr, Imgproc.COLOR_GRAY2BGR);
        big.release();
        return bgr;
    }

    /** Исходный снимок с рамкой и 5 точками YuNet, вписанный в RAW_W × TH. */
    static Mat rawTile(Mat color, double[] row) {
        Mat c = color.clone();
        int th = Math.max(2, (int) Math.round(color.rows() / 200.0));
        Imgproc.rectangle(c, new Rect((int) Math.round(row[0]), (int) Math.round(row[1]), (int) Math.round(row[2]), (int) Math.round(row[3])),
                new Scalar(0, 255, 0), th);
        Scalar[] col = {new Scalar(0, 0, 255), new Scalar(255, 0, 0), new Scalar(0, 255, 255), new Scalar(255, 0, 255), new Scalar(255, 255, 0)};
        for (int i = 0; i < 5; i++) Imgproc.circle(c, new Point(row[4 + 2 * i], row[5 + 2 * i]), th * 2, col[i], -1);
        double s = Math.min((double) RAW_W / c.cols(), (double) TH / c.rows());
        Mat out = new Mat();
        Imgproc.resize(c, out, new Size(Math.max(1, Math.round(c.cols() * s)), Math.max(1, Math.round(c.rows() * s))), 0, 0, Imgproc.INTER_AREA);
        c.release();
        return out;
    }

    static void put(Mat sheet, Mat m, int x0, int y0) {
        m.copyTo(sheet.submat(y0, y0 + m.rows(), x0, x0 + m.cols()));
    }

    static void text(Mat sheet, String s, int x, int y, double scale) {
        Imgproc.putText(sheet, s, new Point(x, y), Imgproc.FONT_HERSHEY_PLAIN, scale, new Scalar(0, 0, 0), 1);
    }

    /** Ближайший снимок галереи человека по χ² гистограмм LBP признака (norm) без MLDA. */
    static String nearest(Source src, String probeKey, List<String> gal, IlluminationNorm norm, IlluminationNorm.NormParams prm,
                          Map<String, double[]> cache) throws IOException {
        String best = null;
        double bd = Double.POSITIVE_INFINITY;
        double[] hp = hist(src, probeKey, norm, prm, cache);
        for (String k : gal) {
            double d = ChiSquareScorer.chi2(hp, hist(src, k, norm, prm, cache));
            if (d < bd) {
                bd = d;
                best = k;
            }
        }
        return best;
    }

    static double[] hist(Source src, String key, IlluminationNorm norm, IlluminationNorm.NormParams prm, Map<String, double[]> cache) throws IOException {
        double[] h = cache.get(key);
        if (h == null) {
            Mat a = src.frameA(key);
            h = LbpFeature.histogram(a, norm, prm);
            a.release();
            cache.put(key, h);
        }
        return h;
    }

    /** Листы проб пары и строки индекса; возвращает людей, чьи галереи нужны. */
    static Set<String> sheets(Pair p, Source src, Map<String, List<String>> galleryKeys, IlluminationNorm.NormParams prm, Path out, String prefix,
                              StringBuilder index) throws IOException {
        Set<String> people = new TreeSet<>();
        Map<String, double[]> cache = new HashMap<>();
        int w = RAW_W + 4 * TW + CAP_W + 6 * GAP;
        for (int gi = 0; gi < 4; gi++) {
            List<Case> cs = p.sel.get(gi);
            for (int from = 0, sh = 1; from < cs.size(); from += ROWS, sh++) {
                List<Case> part = cs.subList(from, Math.min(cs.size(), from + ROWS));
                Mat sheet = new Mat(HEAD + part.size() * (TH + GAP), w, CvType.CV_8UC3, new Scalar(255, 255, 255));
                String file = String.format(Locale.ROOT, "%s_%s_%02d.png", prefix, GROUPS[gi], sh);
                text(sheet, String.format(Locale.ROOT, "%s %s | %s | theta P2 FAR1%% = %.4f | sheet %d", p.base, p.label, GROUP_NAMES[gi], p.theta, sh),
                        GAP, 18, 1.0);
                text(sheet, String.format(Locale.ROOT, "cols: raw+YuNet, probe a, probe tt | gallery a, gallery tt | gallery: nearest by chi2 LBP | "
                        + "tt scale fixed [-tau, tau], tau = %s", fmt(prm.tau())), GAP, 38, 1.0);
                for (int i = 0; i < part.size(); i++) {
                    Case c = part.get(i);
                    String key = c.probe().key();
                    people.add(c.galPerson());
                    if (c.probe().role().equals(FeiScores.OWN)) people.add(c.probe().person());
                    String gk = nearest(src, key, galleryKeys.get(c.galPerson()), p.norm, prm, cache);
                    int y0 = HEAD + i * (TH + GAP);
                    int x = GAP;
                    Mat raw = src.raw(key);
                    Mat rt = rawTile(raw, src.det(key));
                    put(sheet, rt, x + (RAW_W - rt.cols()) / 2, y0 + (TH - rt.rows()) / 2);
                    raw.release();
                    rt.release();
                    x += RAW_W + GAP;
                    for (String k : new String[] {key, gk}) {
                        Mat a = src.frameA(k);
                        Mat t1 = tile(a);
                        Mat t2 = ttTile(a, prm);
                        put(sheet, t1, x, y0);
                        put(sheet, t2, x + TW + GAP, y0);
                        x += 2 * (TW + GAP);
                        for (Mat m : new Mat[] {a, t1, t2}) m.release();
                    }
                    Imgproc.line(sheet, new Point(RAW_W + 2 * TW + 3.5 * GAP, y0), new Point(RAW_W + 2 * TW + 3.5 * GAP, y0 + TH), new Scalar(0, 0, 255), 2);
                    String[] lines = {
                        String.format(Locale.ROOT, "#%d %s", from + i + 1, c.probe().role().equals(FeiScores.OWN) ? "own" : "control (stranger)"),
                        "id " + c.probe().person() + "  img " + src.name(key),
                        "date " + src.date(key) + "  |r| = " + fmt(src.absR(key)),
                        "group " + GROUP_NAMES[gi],
                        String.format(Locale.ROOT, "d1/d2 = %.4f  theta = %.4f", c.ratio(), p.theta),
                        "best " + c.bestId() + (c.probe().role().equals(FeiScores.OWN) ? "  true " + c.probe().person() : ""),
                        "gallery " + c.galPerson() + "  img " + src.name(gk),
                        "gallery date " + src.date(gk)};
                    for (int l = 0; l < lines.length; l++) text(sheet, lines[l], x + GAP, y0 + 18 + l * 22, 1.1);
                    index.append(String.join("\t", file, Integer.toString(i + 1), p.base, p.label, GROUPS[gi], c.probe().role(), c.probe().person(),
                            src.name(key), fmt(src.absR(key)), String.format(Locale.ROOT, "%.6f", c.ratio()), String.format(Locale.ROOT, "%.6f", p.theta),
                            c.bestId(), c.galPerson(), src.name(gk))).append('\n');
                }
                Imgcodecs.imwrite(out.resolve(file).toString(), sheet);
                sheet.release();
            }
        }
        return people;
    }

    static String fmt(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }

    /** Листы галереи: человек на строку (кадр «а» и tt каждого снимка галереи), до 4 снимков в ряду, до GAL_ROWS рядов на лист. */
    static void gallerySheets(Set<String> people, Source src, Map<String, List<String>> galleryKeys, IlluminationNorm.NormParams prm, Path out,
                              String prefix, StringBuilder index) throws IOException {
        int perLine = 4;
        int lab = 150;
        int w = lab + perLine * 2 * (TW + GAP) + GAP;
        List<String[]> lines = new ArrayList<>();
        for (String p : people) {
            List<String> ks = galleryKeys.get(p);
            for (int i = 0; i < ks.size(); i += perLine) {
                List<String> chunk = ks.subList(i, Math.min(ks.size(), i + perLine));
                lines.add(new String[] {p, i == 0 ? "" : "(cont.)", String.join(" ", chunk)});
            }
        }
        for (int from = 0, sh = 1; from < lines.size(); from += GAL_ROWS, sh++) {
            List<String[]> part = lines.subList(from, Math.min(lines.size(), from + GAL_ROWS));
            int cap = 18;
            Mat sheet = new Mat(HEAD + part.size() * (TH + cap + GAP), w, CvType.CV_8UC3, new Scalar(255, 255, 255));
            String file = String.format(Locale.ROOT, "%s_%02d.png", prefix, sh);
            text(sheet, String.format(Locale.ROOT, "gallery of persons in samples | sheet %d | per image: a, tt | tt scale fixed [-tau, tau], tau = %s", sh,
                    fmt(prm.tau())), GAP, 18, 1.0);
            for (int i = 0; i < part.size(); i++) {
                String[] l = part.get(i);
                int y0 = HEAD + i * (TH + cap + GAP);
                text(sheet, l[0], GAP, y0 + 20, 1.4);
                if (!l[1].isEmpty()) text(sheet, l[1], GAP, y0 + 44, 1.0);
                int x = lab;
                for (String k : l[2].split(" ")) {
                    Mat a = src.frameA(k);
                    Mat t1 = tile(a);
                    Mat t2 = ttTile(a, prm);
                    put(sheet, t1, x, y0);
                    put(sheet, t2, x + TW, y0);
                    text(sheet, src.name(k) + " " + src.date(k) + " |r|=" + fmt(src.absR(k)), x + 2, y0 + TH + 14, 1.0);
                    x += 2 * (TW + GAP);
                    for (Mat m : new Mat[] {a, t1, t2}) m.release();
                }
                index.append(String.join("\t", file, Integer.toString(i + 1), "gallery", "-", "gallery", "-", l[0], l[2].replace(' ', ','), "-", "-", "-",
                        "-", l[0], "-")).append('\n');
            }
            Imgcodecs.imwrite(out.resolve(file).toString(), sheet);
            sheet.release();
        }
    }

    static final String INDEX_COLUMNS = "sheet\trow\tbase\tmethod\tgroup\trole\tperson\timage\tabs_r\tratio\ttheta\tbest_id\tgallery_person\tgallery_image\n";

    // ---------------------------------------------------------------- GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                   Path xbaseOut) throws IOException {
        String rawGt = settings.loadFacesGtDir();
        if (rawGt == null) throw new IllegalStateException("XBaseEval gt-a0: задайте faces.gt.dir / FACES_GT_DIR (исходные снимки)");
        Path gtDir = Paths.get(rawGt);
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        List<Probe> probes = XBaseEval.probes(g, g.probeKeys);
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Map<String, double[]> det = new HashMap<>();
        for (String l : Files.readAllLines(dir.resolve("detections.tsv"), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (!c[0].equals("gt") || !c[4].equals("1")) continue;
            double[] row = new double[15];
            for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[5 + i]);
            det.put(c[1] + "/" + c[2], row);
        }
        Map<String, Integer> sess = XBaseEval.sessions(dir.resolve(XBaseEval.SESSIONS));
        Source src = new Source() {
            public Mat raw(String key) throws IOException {
                Sample s = g.samples.get(key);
                Path f = gtDir.resolve(s.person()).resolve(s.file().getFileName().toString() + ".jpg");
                Mat m = Imgcodecs.imread(f.toString(), Imgcodecs.IMREAD_COLOR);
                if (m.empty()) throw new IOException("Нет снимка " + f);
                return m;
            }

            public double[] det(String key) {
                Sample s = g.samples.get(key);
                return det.get(s.person() + "/" + s.file().getFileName());
            }

            public Mat frameA(String key) throws IOException {
                Mat f = export.frame(g.samples.get(key), Variant.A);
                if (f == null) throw new IOException("Нет кадра " + key);
                return f;
            }

            public String date(String key) {
                Sample s = g.samples.get(key);
                return "s" + sess.get(s.person() + "/" + s.file().getFileName());
            }

            public String name(String key) {
                return g.samples.get(key).file().getFileName().toString();
            }

            public double absR(String key) {
                return g.rOf.get(key);
            }
        };
        Path out = xbaseOut.resolve("a0");
        Files.createDirectories(out);
        StringBuilder sum = new StringBuilder();
        sum.append("(а0) листы кадров у ошибок, GT межсессионно (XBaseEval gt-a0; PLAN.md п. 5.5)\n");
        sum.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        sum.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append('\n');
        sum.append(String.format(Locale.ROOT, "Контроль (i): Random(faces.seed + %d = %d)%n", OK_SEED_OFFSET, g.seed + OK_SEED_OFFSET));
        StringBuilder index = new StringBuilder(INDEX_COLUMNS);
        Set<String> people = new TreeSet<>();
        Object[][] pairs = {{"xbase_gt_scores.tsv.gz", "L1", IlluminationNorm.NONE}, {"xbase_gt_ltp_scores.tsv.gz", "L1-tt", IlluminationNorm.TAN_TRIGGS}};
        for (Object[] pp : pairs) {
            Path f = xbaseOut.resolve((String) pp[0]);
            sum.append("Файл оценок: ").append(f).append(" (").append(FeiThreshold.sha(f)).append(")\n");
            Map<String, Row> rows = read(f, FeiMethods.LBP_MLDA, null);
            Pair p = decide("GT", (String) pp[1], (IlluminationNorm) pp[2], rows, g, g.galleryKeys, probes, parts);
            sum.append(summary(p));
            people.addAll(sheets(p, src, g.galleryKeys, g.params, out, "gt_" + pp[1], index));
        }
        gallerySheets(people, src, g.galleryKeys, g.params, out, "gallery", index);
        FarMethods.write(out.resolve("a0_index.tsv"), index);
        FarMethods.write(out.resolve("a0_summary.txt"), sum);
        System.out.print(sum);
    }

    // ---------------------------------------------------------------- FERET

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-a0: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval feret-a0: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        tpl.r0 = FeiPose.r0(rawR0, "+pose");
        tpl.b = 0;
        tpl.threads = 1;
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);

        // Снимки, детекции, роли — как XBaseFeret.run.
        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        Map<String, Img> byKey = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> kept = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        for (Img im : imgs) {
            byKey.put(im.key(), im);
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) continue;
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) continue;
            kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), d -> new ArrayList<>()).add(im);
        }
        XBaseFeret.Roles ro = XBaseFeret.roles(XBaseFeret.dates(imgs), kept, tpl.seed);
        TreeSet<String> ps = new TreeSet<>(ro.own2);
        ps.addAll(ro.ctrl);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, new ArrayList<>(ps), ro.ctrl, ro.ctrlKeys, new ArrayList<>(), rOf, imageNo);
        XBaseEval.Gt g = XBaseFeret.withOwn(base, ro.own2);
        List<Probe> probes = XBaseEval.probes(g, ro.pro2);
        int s = 2; // G1 — номер конфигурации в XBaseFeret
        List<List<String>> parts = XBaseOther.cvParts(ro.own2, tpl.seed, XBaseEval.CV_S + s);
        Source src = new Source() {
            public Mat raw(String key) throws IOException {
                return FeretData.decode(byKey.get(key).file());
            }

            public double[] det(String key) {
                return det.get(key);
            }

            public Mat frameA(String key) throws IOException {
                return FeretData.frame(dir, "a", key);
            }

            public String date(String key) {
                return byKey.get(key).date();
            }

            public String name(String key) {
                return key;
            }

            public double absR(String key) {
                return rOf.get(key);
            }
        };
        Path out = dir.resolve("reports").resolve("a0");
        Files.createDirectories(out);
        Path f = dir.resolve("reports").resolve("xbase_feret_scores.tsv.gz");
        StringBuilder sum = new StringBuilder();
        sum.append("(а0) листы кадров у ошибок, FERET G1 (XBaseEval feret-a0; PLAN.md п. 5.5); только локально\n");
        sum.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        sum.append("Файл оценок: ").append(f).append(" (").append(FeiThreshold.sha(f)).append("), блок «G1, набор tt»\n");
        sum.append(String.format(Locale.ROOT, "Контроль (i): Random(faces.seed + %d = %d)%n", OK_SEED_OFFSET, tpl.seed + OK_SEED_OFFSET));
        StringBuilder index = new StringBuilder(INDEX_COLUMNS);
        Map<String, Row> rows = read(f, FeiMethods.LBP_MLDA, "# конфигурация G1, набор tt");
        Pair p = decide("FERET-G1", "L1-tt", IlluminationNorm.TAN_TRIGGS, rows, g, ro.g1, probes, parts);
        sum.append(summary(p));
        Set<String> people = sheets(p, src, ro.g1, tpl.params, out, "feret_G1_L1-tt", index);
        gallerySheets(people, src, ro.g1, tpl.params, out, "gallery", index);
        FarMethods.write(out.resolve("a0_index.tsv"), index);
        FarMethods.write(out.resolve("a0_summary.txt"), sum);
        System.out.print(sum);
    }
}
