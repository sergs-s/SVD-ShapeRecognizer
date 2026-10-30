package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseVec.Blk;
import svd.recognizer.faces.XBaseVec.Model;
import svd.recognizer.faces.XBaseVec.Row;
import svd.recognizer.faces.XBaseVec.Setup;
import svd.recognizer.storage.SettingsStore;

/**
 * Пачка 3 – MLDA по областям лица (задание Жорика №6 от 30.10.2026). Признак – (б) (П2) + MS-LBP (решение (32)); области Г (глаза с
 * бровями), Н (нос), Р (рот), О (остальное) – прямоугольники по шаблону 5 точек кадра «а» 92×112 (FaceAlignment.ownTemplate),
 * симметричные относительно середины кадра, заданы до прогона: Г – |x − x_c| ≤ d, y от y_глаз − 0,6 d до середины глаза – нос; Н –
 * |x − x_c| ≤ 0,4 d, от середины глаза – нос до середины нос – рот; Р – |x − x_c| ≤ полуширина рта + 0,3 d, от середины нос – рот до
 * y_рта + 0,4 d; О – остальное (d – межглазье шаблона, x_c = (92 − 1) / 2). Ячейка сетки LBP и POEM – к области своего номинального центра
 * (без округления границ ячеек – разбиение симметрично по номерам столбцов); части MS-LBP – глаза → Г, нос → Н, углы рта → Р. Признак
 * области – срез полного вектора (П2) + MS-LBP (те же веса). Модель области – своя MLDA с пулом FERET (XBaseVec.job), выбор X / X+mirror
 * – как у кандидата. Общая модель – векторы pool / feret / gt-feat2 (вариант 8), не пересчитываются. Слияние внутри кандидата: ẑ_m =
 * (ℓ_m − μ_m) / σ_m по «чужим» (Т2) модели (XBaseVec.fuseStats), ℓ_канд = Σ a_m ẑ_m, при нескольких моделях – повторное приведение по тем
 * же «чужим»; затем слияние кандидатов w = 0,75, β = 1. Строки базы и одной модели – XBaseVec.normLines / fuseLines (побайтно как прежде).
 *
 * @author ssv
 */
final class XBasePart3 {

    static final String[] REG = {"Г", "Н", "Р", "О"};
    static final String[] REG_LAT = {"G", "N", "R", "O"};
    static final int NR = REG.length;
    static final int AW = 92;
    static final int AH = 112;
    /** Компоненты: 0 – общая модель, 1…4 – Г, Н, Р, О. */
    static final String[] VN = {"база (П2) + MS-LBP", "(Ч1) общая + Г, Н, Р по 1/4", "(Ч1′) общая 1/2 + Г, Н, Р по 1/6", "(Ч2) Г, Н, Р, О по 1/4",
        "справочно: только Г", "справочно: только Н", "справочно: только Р", "справочно: только О"};
    static final int[][] COMP = {{0}, {0, 1, 2, 3}, {0, 1, 2, 3}, {1, 2, 3, 4}, {1}, {2}, {3}, {4}};
    static final double[][] A = {{1}, {0.25, 0.25, 0.25, 0.25}, {0.5, 1.0 / 6, 1.0 / 6, 1.0 / 6}, {0.25, 0.25, 0.25, 0.25}, {1}, {1}, {1}, {1}};
    /** Выбор – среди первых NSEL (порядок простоты: база, (Ч1), (Ч1′), (Ч2)). */
    static final int NSEL = 4;
    static final String SELECT = "xbase_pool_part3_select.tsv";

    private XBasePart3() {
    }

    /** Пробный режим FACES_PART3_SMOKE: 0 – нет; 2 – первая группа в полном размере – замер (только обучение и оценка времени). */
    static int smoke() {
        String v = System.getenv("FACES_PART3_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    // ---------------------------------------------------------------- области

    /** Прямоугольники Г, Н, Р {x0, x1, y0, y1} в координатах пикселей кадра «а» (x – включительно с обеих сторон, y – [y0, y1)). */
    static double[][] rects() {
        double[][] t = FaceAlignment.ownTemplate(AW, AH);
        double xc = (AW - 1) / 2.0;
        double eyeY = (t[0][1] + t[1][1]) / 2;
        double d = Math.hypot(t[1][0] - t[0][0], t[1][1] - t[0][1]);
        double noseY = t[2][1];
        double mouthY = (t[3][1] + t[4][1]) / 2;
        double mouthHalf = Math.abs(t[4][0] - t[3][0]) / 2;
        double y1 = (eyeY + noseY) / 2;
        double y2 = (noseY + mouthY) / 2;
        return new double[][] {{xc - d, xc + d, eyeY - 0.6 * d, y1}, {xc - 0.4 * d, xc + 0.4 * d, y1, y2},
            {xc - mouthHalf - 0.3 * d, xc + mouthHalf + 0.3 * d, y2, mouthY + 0.4 * d}};
    }

    static int region(double[][] rc, double x, double y) {
        for (int r = 0; r < rc.length; r++) if (x >= rc[r][0] && x <= rc[r][1] && y >= rc[r][2] && y < rc[r][3]) return r;
        return NR - 1;
    }

    /** Номинальный центр ячейки (by, bx) сетки g на кадре «а» с краем b (как LbpFeature и PoemFeature: область (AW − 2b) × (AH − 2b)). */
    static double[] center(int g, int b, int by, int bx) {
        double cw = AW - 2 * b;
        double ch = AH - 2 * b;
        return new double[] {b + (bx + 0.5) * cw / g - 0.5, b + (by + 0.5) * ch / g - 0.5};
    }

    static int lbpBorder(int f) {
        LbpFeature.Params lp = XBaseDates.PARAMS[f];
        if (lp.radii().length != 1) throw new IllegalStateException("part3: LBP с несколькими радиусами");
        return (int) lp.radii()[0];
    }

    /**
     * Области ячеек [часть 0 – по центру ячейки LBP, 1 – по центру ячейки POEM, справочно][by][bx]. Ячейка (П2) (LBP и POEM с теми же
     * номерами) относится к области целиком – по центру ячейки LBP (у POEM край больше, центры крайних ячеек сдвинуты внутрь).
     */
    static int[][][] cellRegions(int f) {
        int g = XBaseFeat2.grid(f);
        double[][] rc = rects();
        int[] border = {lbpBorder(f), PoemFeature.BORDER};
        int[][][] out = new int[2][g][g];
        for (int p = 0; p < 2; p++) for (int by = 0; by < g; by++) for (int bx = 0; bx < g; bx++) {
            double[] c = center(g, border[p], by, bx);
            out[p][by][bx] = region(rc, c[0], c[1]);
        }
        return out;
    }

    /** Область точки MS-LBP (порядок YuNet: два глаза, нос, два угла рта; у О нет). */
    static final int[] MS_REG = {0, 0, 1, 2, 2};

    /** Номера координат вектора (б) (П2) + MS-LBP по областям: LBP (ячейка × BINS), POEM (ячейка × ORIENT × BINS), MS-LBP (точка). */
    static int[][] masks(int f) {
        int g = XBaseFeat2.grid(f);
        int bins = LbpFeature.BINS;
        int lbpLen = XBaseDates.PARAMS[f].length();
        int poemLen = PoemFeature.length(g);
        int msLen = MsLbpFeature.LENGTH;
        if (lbpLen + poemLen + msLen != XBaseFeat2.dims(f)[XBaseFeat2.MS_JOIN]) throw new IllegalStateException("part3: размерность (П2) + MS-LBP");
        if (lbpLen != g * g * bins || poemLen != g * g * PoemFeature.ORIENT * bins) throw new IllegalStateException("part3: раскладка (П2)");
        int[][][] cr = cellRegions(f);
        List<List<Integer>> ix = new ArrayList<>();
        for (int r = 0; r < NR; r++) ix.add(new ArrayList<>());
        for (int by = 0; by < g; by++) for (int bx = 0; bx < g; bx++) {
            int c = by * g + bx;
            for (int k = 0; k < bins; k++) ix.get(cr[0][by][bx]).add(c * bins + k);
        }
        int pb = PoemFeature.ORIENT * bins;
        for (int by = 0; by < g; by++) for (int bx = 0; bx < g; bx++) {
            int c = by * g + bx;
            for (int k = 0; k < pb; k++) ix.get(cr[0][by][bx]).add(lbpLen + c * pb + k);
        }
        int per = msLen / MsLbpFeature.POINTS;
        for (int p = 0; p < MsLbpFeature.POINTS; p++) for (int k = 0; k < per; k++) ix.get(MS_REG[p]).add(lbpLen + poemLen + p * per + k);
        int[][] out = new int[NR][];
        int total = 0;
        for (int r = 0; r < NR; r++) {
            out[r] = ix.get(r).stream().mapToInt(Integer::intValue).toArray();
            total += out[r].length;
        }
        if (total != lbpLen + poemLen + msLen) throw new IllegalStateException("part3: области не покрывают вектор");
        return out;
    }

    static double[] slice(double[] v, int[] ix) {
        double[] o = new double[ix.length];
        for (int i = 0; i < ix.length; i++) o[i] = v[ix[i]];
        return o;
    }

    /** Таблица ячеек по областям и размерности (текст для плана и отчётов). */
    static String regionTable() {
        StringBuilder t = new StringBuilder();
        double[][] rc = rects();
        double[][] tp = FaceAlignment.ownTemplate(AW, AH);
        t.append("Шаблон 5 точек кадра «а» 92×112:");
        for (double[] p : tp) t.append(String.format(Locale.ROOT, " (%.2f; %.2f)", p[0], p[1]));
        t.append("\nПрямоугольники (x включительно, y – [y0; y1)):\n");
        for (int r = 0; r < rc.length; r++) t.append(String.format(Locale.ROOT, "%s: x %.2f … %.2f, y %.2f … %.2f%n", REG[r], rc[r][0], rc[r][1], rc[r][2], rc[r][3]));
        t.append("О – остальное.\n");
        for (int f = 1; f >= 0; f--) {
            int g = XBaseFeat2.grid(f);
            int[][][] cr = cellRegions(f);
            int[][] mk = masks(f);
            t.append(String.format(Locale.ROOT, "%n%s, сетка %d×%d:%n", XBaseRule.FEATS[f], g, g));
            String[] pn = {"ячейки (П2) – по центру ячейки LBP (край " + lbpBorder(f) + "), принято", "справочно: по центру ячейки POEM (край " + PoemFeature.BORDER
                + "), не используется"};
            boolean same = true;
            boolean sym = true;
            for (int p = 0; p < 2; p++) {
                t.append(pn[p]).append(":\n");
                int[] cnt = new int[NR];
                for (int by = 0; by < g; by++) {
                    StringBuilder row = new StringBuilder("  ");
                    for (int bx = 0; bx < g; bx++) {
                        int r = cr[p][by][bx];
                        cnt[r]++;
                        row.append(REG[r]).append(' ');
                        if (cr[p][by][bx] != cr[0][by][bx]) same = false;
                        if (r != cr[p][by][g - 1 - bx]) sym = false;
                    }
                    t.append(row.toString().stripTrailing()).append('\n');
                }
                t.append("  ячеек:");
                for (int r = 0; r < NR; r++) t.append(' ').append(REG[r]).append(' ').append(cnt[r]);
                t.append('\n');
            }
            t.append(same ? "По центрам POEM разбиение то же.\n" : "По центрам POEM разбиение было бы другим (крайние ячейки); принято – по LBP.\n");
            t.append(sym ? "Разбиение симметрично по столбцам (bx ↔ g − 1 − bx).\n" : "ВНИМАНИЕ: разбиение несимметрично.\n");
            t.append("Размерности признака области:");
            for (int r = 0; r < NR; r++) t.append(' ').append(REG[r]).append(' ').append(mk[r].length);
            t.append(String.format(Locale.ROOT, " (всего %d)%n", XBaseFeat2.dims(f)[XBaseFeat2.MS_JOIN]));
        }
        t.append("\nMS-LBP: глаза → Г, нос → Н, углы рта → Р (у О нет).\n");
        return t.toString();
    }

    // ---------------------------------------------------------------- лист кадров

    static Scalar color(int r) {
        return switch (r) {
            case 0 -> new Scalar(0, 200, 0);
            case 1 -> new Scalar(255, 120, 0);
            case 2 -> new Scalar(0, 0, 255);
            default -> new Scalar(160, 160, 160);
        };
    }

    /** Плитка кадра «а» ×4 с прямоугольниками и центрами ячеек LBP сетки признака f. */
    static Mat tileA(Mat a, int f) {
        int s = 4;
        Mat c = new Mat();
        Imgproc.cvtColor(a, c, Imgproc.COLOR_GRAY2BGR);
        Mat big = new Mat();
        Imgproc.resize(c, big, new Size(AW * s, AH * s), 0, 0, Imgproc.INTER_NEAREST);
        c.release();
        double[][] rc = rects();
        for (int r = 0; r < rc.length; r++) {
            Imgproc.rectangle(big, new Point((rc[r][0] + 0.5) * s, (rc[r][2] + 0.5) * s), new Point((rc[r][1] + 0.5) * s, (rc[r][3] + 0.5) * s), color(r), 2);
        }
        int g = XBaseFeat2.grid(f);
        int[][][] cr = cellRegions(f);
        for (int by = 0; by < g; by++) for (int bx = 0; bx < g; bx++) {
            double[] p = center(g, lbpBorder(f), by, bx);
            Imgproc.circle(big, new Point((p[0] + 0.5) * s, (p[1] + 0.5) * s), 5, color(cr[0][by][bx]), -1);
        }
        Imgproc.putText(big, g + "x" + g, new Point(6, 22), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 255), 2);
        return big;
    }

    /** Плитка кадра k = 2 ×2 с окнами MS-LBP масштаба 0 по областям. */
    static Mat tileK2(FrameK2.F k) {
        int s = 2;
        Mat c = new Mat();
        Imgproc.cvtColor(k.frame(), c, Imgproc.COLOR_GRAY2BGR);
        Mat big = new Mat();
        Imgproc.resize(c, big, new Size(FrameK2.W * s, FrameK2.H * s), 0, 0, Imgproc.INTER_NEAREST);
        c.release();
        int h = MsLbpFeature.PATCH / 2;
        for (int p = 0; p < MsLbpFeature.POINTS; p++) {
            double[] q = k.pts()[p];
            Imgproc.rectangle(big, new Point((q[0] - h + 0.5) * s, (q[1] - h + 0.5) * s), new Point((q[0] + h + 0.5) * s, (q[1] + h + 0.5) * s), color(MS_REG[p]), 2);
            Imgproc.circle(big, new Point((q[0] + 0.5) * s, (q[1] + 0.5) * s), 4, color(MS_REG[p]), -1);
        }
        Imgproc.putText(big, "k=2 MS-LBP", new Point(6, 22), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 255), 2);
        return big;
    }

    /** part3-regions: таблица ячеек и лист кадров GT (8 снимков: «а» 9×9, «а» 7×7, отражение «а» 9×9, k = 2 с окнами MS-LBP). */
    static void sheet(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        Setup u = XBaseVec.setup("gt", export, dir, feiKeys, selLines, settings, out, 0);
        // Только снимки GT: кадры FERET – только в D:\data\Feret.
        List<String> keys = new ArrayList<>(new TreeSet<>(u.fr.keySet().stream().filter(k -> k.startsWith("gt\\") || k.startsWith("gt/")).toList()));
        if (keys.size() < 8) throw new IllegalStateException("part3: мало снимков GT для листа");
        List<String> pick = new ArrayList<>();
        int n = 8;
        for (int i = 0; i < n; i++) pick.add(keys.get((int) ((i + 0.5) * keys.size() / n)));
        Map<String, FrameK2.F> k2 = FrameK2.load(pick, u.su, u.su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        List<Mat> rows = new ArrayList<>();
        for (String key : pick) {
            Mat a = u.fr.get(key);
            Mat am = FaceMirror.mirror(a);
            FrameK2.F k = k2.get(key);
            List<Mat> tiles = List.of(tileA(a, 1), tileA(a, 0), tileA(am, 1), tileK2(k));
            Mat row = new Mat();
            Core.hconcat(tiles, row);
            for (Mat m : tiles) m.release();
            am.release();
            rows.add(row);
        }
        Mat sheet = new Mat();
        Core.vconcat(rows, sheet);
        Path pd = out.resolve("part3");
        Files.createDirectories(pd);
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".png", sheet, buf);
        Files.write(pd.resolve("part3_regions.png"), buf.toArray());
        String t = "Пачка 3: области лица Г, Н, Р, О – таблица ячеек (XBaseEval part3-regions)\nКод: коммит " + FarMethods.commit() + "\n" + regionTable()
                + "\nЛист part3_regions.png (GT, снимки: " + String.join(", ", pick) + "): столбцы – «а» с центрами ячеек 9×9, «а» 7×7, отражённый «а» 9×9, "
                + "кадр k = 2 с окнами MS-LBP масштаба 0;\nцвета – Г зелёный, Н синий, Р красный, О серый.\n";
        FarMethods.write(pd.resolve("xbase_part3_regions.txt"), new StringBuilder(t));
        System.out.print(t);
    }

    // ---------------------------------------------------------------- признаки и модели областей

    /** Признаки областей признака f: срезы (б) (П2) + MS-LBP (XBaseFeat2.feature, вариант 8) и их отражений. */
    static XBaseEval[] vectors(Map<String, Mat> fr, XBaseFeat2.Feats fs, int f, int[][] mk) {
        List<Map<String, double[]>> vec = new ArrayList<>();
        List<Map<String, double[]>> mir = new ArrayList<>();
        for (int r = 0; r < NR; r++) {
            vec.add(new ConcurrentHashMap<>());
            mir.add(new ConcurrentHashMap<>());
        }
        int v = XBaseFeat2.MS_JOIN;
        ForkJoinPool pool = new ForkJoinPool(fs.threads);
        try {
            pool.submit(() -> fr.entrySet().parallelStream().forEach(e -> {
                Mat a = e.getValue();
                Mat am = FaceMirror.mirror(a);
                FrameK2.F k = fs.k2.get(e.getKey());
                if (k == null) throw new IllegalStateException("part3: нет кадра k = 2 " + e.getKey());
                Mat cm = FaceMirror.mirror(k.frame());
                double[] x = XBaseFeat2.feature(fs, f, v, a, k.frame(), k.pts());
                double[] y = XBaseFeat2.feature(fs, f, v, am, cm, FrameK2.mirrorPts(k.pts()));
                for (int r = 0; r < NR; r++) {
                    vec.get(r).put(e.getKey(), slice(x, mk[r]));
                    mir.get(r).put(e.getKey(), slice(y, mk[r]));
                }
                am.release();
                cm.release();
            })).get();
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("part3: признаки " + e, e);
        } finally {
            pool.shutdown();
        }
        XBaseEval[] xs = new XBaseEval[NR];
        for (int r = 0; r < NR; r++) {
            xs[r] = new XBaseEval();
            xs[r].vec.put(V.L1, new HashMap<>(vec.get(r)));
            xs[r].mir.put(V.L1, new HashMap<>(mir.get(r)));
        }
        return xs;
    }

    static Path dir(Setup u, int sm) {
        return u.out.resolve("xbase_" + u.base + "_part3" + (sm == 0 ? "" : "_smoke" + sm));
    }

    static Path file(Path d, Blk b, int f, int r) {
        return d.resolve(b.name() + "_f" + f + "_r" + r + ".tsv.gz");
    }

    /** Модели областей по блокам (готовые файлы с тем же коммитом – пропускаются); sec[f][r] – время. */
    static void train(Setup u, XBaseFeat2.Feats fs, int sm, double[][] sec) throws IOException {
        Path vd = dir(u, sm);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        String head = "# Векторы 1 − cos моделей областей (XBasePart3 " + u.base + "-part3): галерея, когорта в W (" + String.join(",", u.cohortIn) + "), когорта вне W ("
                + String.join(",", u.cohortOut) + ")\n# Код: коммит " + commit + "\n";
        for (int f = 1; f >= 0; f--) {
            List<List<Blk>> todo = new ArrayList<>();
            boolean any = false;
            for (int r = 0; r < NR; r++) {
                List<Blk> l = new ArrayList<>();
                for (Blk b : u.blocks) {
                    Path p = file(vd, b, f, r);
                    if (Files.exists(p)) {
                        if (!XBaseFeat2.reusable(XBaseVec.readHead(p), commit)) throw new IllegalStateException("part3: " + p + " – другой коммит кода; удалите каталог " + vd);
                        continue;
                    }
                    l.add(b);
                }
                todo.add(l);
                any |= !l.isEmpty();
            }
            if (!any) continue;
            long t0 = System.nanoTime();
            XBaseEval[] x = vectors(u.fr, fs, f, masks(f));
            double tf = (System.nanoTime() - t0) / 1e9;
            System.out.printf(Locale.ROOT, "%s-part3: f%d признаки областей (%.0f с)%n", u.base, f, tf);
            for (int r = 0; r < NR; r++) {
                List<Blk> l = todo.get(r);
                int step = Math.max(1, Math.min(4, l.size()));
                for (int i = 0; i < l.size(); i += step) {
                    List<List<java.util.concurrent.Callable<Model>>> jobs = new ArrayList<>();
                    for (Blk b : l.subList(i, Math.min(l.size(), i + step))) {
                        jobs.add(XBaseVec.job(b.g(), x[r], b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
                    }
                    long ts = System.nanoTime();
                    List<List<Model>> done = XBaseVec.run(jobs, u.tpl.threads);
                    for (int k = 0; k < done.size(); k++) {
                        Blk b = l.get(i + k);
                        Path p = file(vd, b, f, r);
                        Writer w = XBaseVec.writer(p, head);
                        XBaseVec.write(w, String.format(b.abl2(), XBaseRule.FEATS[f]) + "область " + REG[r], done.get(k));
                        XBaseVec.finish(w, p);
                    }
                    double tb = (System.nanoTime() - ts) / 1e9;
                    sec[f][r] += tb;
                    System.out.printf(Locale.ROOT, "%s-part3: f%d %s блоки %d–%d (%.0f с)%n", u.base, f, REG[r], i + 1, i + jobs.size(), tb);
                }
                x[r] = null;
                sec[f][r] += tf / NR;
            }
        }
    }

    /** Модели компонента c (0 – общая, 1…4 – области) блока b признака f. */
    static List<Model> comp(Setup u, int sm, Blk b, int f, int c) throws IOException {
        if (c == 0) return XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, f, XBaseFeat2.MS_JOIN);
        return XBaseVec.read(file(dir(u, sm), b, f, c - 1)).values().iterator().next();
    }

    // ---------------------------------------------------------------- слияние

    static boolean impostor(Row r, boolean pool) {
        return r.trueIdx() < 0 && (pool ? r.role().equals(FeiScores.CTRL) : r.role().startsWith(FeiMethods.CV_ROLE) || r.role().startsWith("od"));
    }

    /**
     * ℓ_канд по моделям кандидата [номер модели в списке][строка][g]: ẑ компонентов (μ, σ – XBaseVec.fuseStats), сумма с весами a; при
     * нескольких компонентах – повторное приведение по «чужим» (Т2) (pool – контрольные чужие).
     */
    static double[][][] cand(List<List<Model>> comps, double[] a, boolean pool) {
        List<Model> m0 = comps.get(0);
        double[][] st = new double[comps.size()][];
        for (int c = 0; c < comps.size(); c++) {
            st[c] = XBaseVec.fuseStats(comps.get(c), pool);
            List<Model> mc = comps.get(c);
            if (mc.size() != m0.size()) throw new IllegalStateException("part3: разное число моделей компонентов");
            for (int m = 0; m < m0.size(); m++) {
                Model x = mc.get(m);
                Model y = m0.get(m);
                if (!x.task().equals(y.task()) || !x.method().equals(y.method()) || !x.gallery().equals(y.gallery()) || x.rows().size() != y.rows().size()) {
                    throw new IllegalStateException("part3: компоненты – разные задачи");
                }
                for (int i = 0; i < x.rows().size(); i++) {
                    if (!x.rows().get(i).key().equals(y.rows().get(i).key()) || !x.rows().get(i).role().equals(y.rows().get(i).role())) {
                        throw new IllegalStateException("part3: компоненты – разные пробы");
                    }
                }
            }
        }
        double[][][] out = new double[m0.size()][][];
        double s = 0;
        double ss = 0;
        long n = 0;
        for (int m = 0; m < m0.size(); m++) {
            List<Row> rows = m0.get(m).rows();
            out[m] = new double[rows.size()][];
            for (int i = 0; i < rows.size(); i++) {
                int ng = rows.get(i).dg().length;
                double[] l = new double[ng];
                for (int g = 0; g < ng; g++) {
                    double v = 0;
                    for (int c = 0; c < comps.size(); c++) {
                        double d = comps.get(c).get(m).rows().get(i).dg()[g];
                        if (Double.isInfinite(d)) {
                            v = Double.POSITIVE_INFINITY;
                            break;
                        }
                        v += a[c] * (XBaseVec.ell(d) - st[c][0]) / st[c][1];
                    }
                    l[g] = v;
                    if (impostor(rows.get(i), pool) && !Double.isInfinite(v)) {
                        s += v;
                        ss += v * v;
                        n++;
                    }
                }
                out[m][i] = l;
            }
        }
        if (comps.size() > 1) {
            if (n < 2) throw new IllegalStateException("part3: мало «чужих» для приведения");
            double mu = s / n;
            double sd = Math.sqrt((ss - n * mu * mu) / (n - 1));
            for (double[][] mm : out) for (double[] l : mm) for (int g = 0; g < l.length; g++) l[g] = (l[g] - mu) / sd;
        }
        return out;
    }

    /** Строки кандидата (все модели подряд, как normLines): лучший и второй по ℓ_канд, β = 1. */
    static String candLines(List<Model> m0, double[][][] l, int[] clamped) {
        StringBuilder b = new StringBuilder();
        for (int m = 0; m < m0.size(); m++) {
            List<Row> rows = m0.get(m).rows();
            for (int i = 0; i < rows.size(); i++) b.append(XBaseVec.decide(m0.get(m), rows.get(i), l[m][i], clamped));
        }
        return b.toString();
    }

    /** Слияние кандидатов по ℓ_канд (как XBaseVec.fuseLines: tt – модель по позе пробы, xs – X+mirror). */
    static String fuse(List<Model> tt, double[][][] lt, List<Model> xs, double[][][] lx, double w, Map<String, Double> rOf, double r0, int[] clamped) {
        StringBuilder b = new StringBuilder();
        Map<String, Model> byTask = new LinkedHashMap<>();
        for (Model m : tt) byTask.putIfAbsent(m.task(), m);
        for (String task : byTask.keySet()) {
            Model tX = XBaseVec.find(tt, task, false);
            Model tM = XBaseVec.find(tt, task, true);
            Model xM = XBaseVec.find(xs, task, true);
            int iX = tt.indexOf(tX);
            int iM = tt.indexOf(tM);
            int jM = xs.indexOf(xM);
            if (!tX.gallery().equals(xM.gallery()) || tX.rows().size() != xM.rows().size()) throw new IllegalStateException("part3: слияние – разные задачи");
            for (int i = 0; i < tX.rows().size(); i++) {
                Row rx = xM.rows().get(i);
                Row r0r = tX.rows().get(i);
                if (!rx.key().equals(r0r.key()) || !rx.role().equals(r0r.role())) throw new IllegalStateException("part3: слияние – разные пробы");
                boolean front = rOf.get(r0r.key()) <= r0;
                Row rt = front ? r0r : tM.rows().get(i);
                double[] a = front ? lt[iX][i] : lt[iM][i];
                double[] c = lx[jM][i];
                double[] ln = new double[a.length];
                for (int g = 0; g < a.length; g++) ln[g] = Double.isInfinite(a[g]) || Double.isInfinite(c[g]) ? Double.POSITIVE_INFINITY : w * a[g] + (1 - w) * c[g];
                for (Model m : List.of(tX, tM)) b.append(XBaseVec.decide(m, rt, ln, clamped));
            }
        }
        return b.toString();
    }

    /** Строки варианта v: [0] – L1-xs-F, [1] – L1-tt-xs-F, [2] – слияние w = 0,75 (база и одна модель – normLines / fuseLines). */
    static String[] lines(Setup u, int sm, Blk b, int v, boolean pool, int[] clamped) throws IOException {
        int[] cs = COMP[v];
        List<List<List<Model>>> ms = new ArrayList<>();
        for (int f = 0; f < 2; f++) {
            List<List<Model>> l = new ArrayList<>();
            for (int c : cs) l.add(comp(u, sm, b, f, c));
            ms.add(l);
        }
        String[] out = new String[3];
        if (cs.length == 1) {
            for (int f = 0; f < 2; f++) out[f] = XBaseVec.normLines(ms.get(f).get(0), null, clamped);
            List<Model> tt = ms.get(1).get(0);
            List<Model> xs = ms.get(0).get(0);
            out[2] = XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, pool), XBaseVec.fuseStats(xs, pool), b.g().rOf, b.g().r0, clamped);
            return out;
        }
        double[][][][] l = new double[2][][][];
        for (int f = 0; f < 2; f++) {
            l[f] = cand(ms.get(f), A[v], pool);
            out[f] = candLines(ms.get(f).get(0), l[f], clamped);
        }
        out[2] = fuse(ms.get(1).get(0), l[1], ms.get(0).get(0), l[0], XBaseFeat2.W_FUSE, b.g().rOf, b.g().r0, clamped);
        return out;
    }

    static String rowName(int part, int v) {
        return (part == 2 ? "слияние w = 0,75, " : XBaseRule.FEATS[part] + " ") + VN[v];
    }

    // ---------------------------------------------------------------- режимы

    static final String NOTE = "Пачка 3 – MLDA по областям лица. База – решение (32): (б) (П2) + MS-LBP у обоих кандидатов, MLDA с пулом FERET, пол c = 1, "
            + "1 − cos по всем осям, β = 1;\nL1-tt-xs-F – модель X / X+mirror по |r| пробы (r0 = 0,12), L1-xs-F – всегда X+mirror; слияние кандидатов w = 0,75 "
            + "(приведение по «чужим» (Т2), в пуле – по контрольным чужим блока).\nОбласти Г, Н, Р, О – прямоугольники по шаблону кадра «а» (таблица – "
            + "xbase_part3_regions.txt); модель области – своя MLDA на срезе вектора (П2) + MS-LBP;\nслияние внутри кандидата – Σ a_m ẑ_m (ẑ по «чужим» (Т2) "
            + "модели), при нескольких моделях – повторное приведение по тем же «чужим».\n";

    static double frac(String base) {
        return base.equals("pool") ? 1.0 / (10.0 / XBaseNap.SMOKE2_BLOCKS * (1 + XBaseNap.FERET_TO_GT)) : base.equals("feret") ? (2143.0 + 1247) / (2143 + 1365
                + 1247 + 984) : (2843.0 + 2573) / (2843 + 2172 + 2573 + 1989);
    }

    /** pool-part3, feret-part3, gt-part3: обучение моделей областей, затем выбор по пулу или сравнение на своих. */
    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        int sm = smoke();
        if (sm != 0 && sm != 2) throw new IllegalStateException("part3: пробный режим – только 2");
        Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, sm);
        for (Blk b : u.blocks) for (int f = 0; f < 2; f++) {
            Path p = XBaseFeat2.file(XBaseFeat2.dir(u, 0), b, f, XBaseFeat2.MS_JOIN);
            if (!Files.exists(p)) throw new IllegalStateException("part3: нет векторов общей модели " + p + " (" + base + "-feat2 / feat2b)");
        }
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        double[][] sec = new double[2][NR];
        long t0 = System.nanoTime();
        train(u, fs, sm, sec);
        double total = (System.nanoTime() - t0) / 1e9;
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        StringBuilder t = new StringBuilder("Пачка 3: MLDA по областям лица – " + base + " (XBaseEval " + base + "-part3)" + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append("; модели областей – ").append(dir(u, sm)).append("; общая модель – ").append(XBaseFeat2.dir(u, 0))
                .append('\n').append(NOTE).append('\n').append(regionTable());
        t.append("\n=== Время обучения моделей областей (без файлов из каталога; потоков ").append(u.tpl.threads).append(") ===\n");
        for (int f = 1; f >= 0; f--) for (int r = 0; r < NR; r++) t.append(String.format(Locale.ROOT, "%s %s: %.0f с%n", XBaseRule.FEATS[f], REG[r], sec[f][r]));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с%n", total));
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        if (sm == 2) {
            double fr = frac(base);
            t.append(String.format(Locale.ROOT, "Замер: %.0f с; оценка полного прогона: %.0f с (%.1f ч).%n", total, total / fr, total / fr / 3600));
            FarMethods.write(u.out.resolve("xbase_" + base + "_part3" + sfx + ".txt"), t);
            System.out.print(t);
            return;
        }
        if (base.equals("pool")) poolReport(u, t, out);
        else ownReport(u, base, t);
    }

    static void poolReport(Setup u, StringBuilder t, Path out) throws IOException {
        int[] clamped = new int[2];
        List<String> sets = u.groups;
        int nv = VN.length;
        double[][][] crit = new double[3][sets.size()][nv];
        for (int s = 0; s < sets.size(); s++) {
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(sets.get(s))) bi.add(i);
            for (int v = 0; v < nv; v++) {
                List<List<String>> ls = List.of(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
                for (int k : bi) {
                    String[] l = lines(u, 0, u.blocks.get(k), v, true, clamped);
                    for (int p = 0; p < 3; p++) ls.get(p).add(l[p]);
                }
                for (int p = 0; p < 3; p++) crit[p][s][v] = XBaseFeat2.poolSep(u, bi, ls.get(p), p == 2 ? 1 : p, p == 2).crit();
            }
        }
        t.append("\nПравило (PLAN.md, уточнённое): база остаётся, если в пределах 1 п. п. от лучшего во всех трёх наборах; иначе – первый в пределах в порядке "
                + "(Ч1), (Ч1′), (Ч2);\nесли таких нет – наименьшее наибольшее отставание (условие Δ – на своих). Решающая – таблица слияния; справочные строки в "
                + "выборе не участвуют.\n");
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBasePart3 pool-part3): часть, номер, вариант, основание\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder gt = new StringBuilder();
        String[] title = {XBaseRule.FEATS[0], XBaseRule.FEATS[1], "Слияние кандидатов w = 0,75 (решающая)"};
        for (int p : new int[] {1, 0, 2}) {
            double[][] c = new double[sets.size()][];
            for (int s = 0; s < sets.size(); s++) c[s] = java.util.Arrays.copyOf(crit[p][s], NSEL);
            int ch = XBaseFeat2.chooseAll(c);
            String why = "в пределах 1 п. п. во всех трёх";
            if (ch < 0) {
                ch = 0;
                for (int v = 1; v < NSEL; v++) if (XBaseFeat2.maxLag(c, v) < XBaseFeat2.maxLag(c, ch)) ch = v;
                why = "в пределах – нет; наименьшее наибольшее отставание (условие Δ – на своих)";
            }
            String[] names = new String[nv];
            for (int v = 0; v < nv; v++) names[v] = VN[v];
            t.append(XBaseFeat2.lagTable(title[p] + ": вариант × набор (критерий, в скобках – отставание от лучшего набора среди выбираемых, п. п.)",
                    java.util.Arrays.copyOf(names, NSEL), c, sets, ch));
            t.append("справочно (без выбора):\n");
            for (int v = NSEL; v < nv; v++) {
                t.append(VN[v]);
                for (int s = 0; s < sets.size(); s++) t.append(String.format(Locale.ROOT, " | %.1f %%", crit[p][s][v]));
                t.append('\n');
            }
            t.append(String.format(Locale.ROOT, "Выбор %s: %s (%s)%n", title[p], VN[ch], why));
            sel.append(p == 2 ? "слияние" : XBaseRule.FEATS[p]).append('\t').append(ch).append('\t').append(VN[ch]).append('\t').append(why).append('\n');
            gt.append(title[p]).append(" (набор GT):\n");
            for (int v = 0; v < nv; v++) gt.append(String.format(Locale.ROOT, "%s | %.1f %%%n", VN[v], crit[p][0][v]));
        }
        Files.writeString(u.out.resolve(SELECT), sel.toString());
        FarMethods.write(u.out.resolve("xbase_pool_part3.txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_part3_gt.txt"), new StringBuilder(t.substring(0, t.indexOf("\n=== Время"))).append(
                "\n=== Набор GT (наборы FERET – только в D:\\data\\Feret\\reports\\eq\\) ===\n").append(gt));
        System.out.print(t);
    }

    /** Вариант выбора по пулу (строка «слияние» файла выбора). */
    static int readSelect(Setup u, String base) throws IOException {
        Path sf = (base.equals("gt") ? u.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : u.out).resolve(SELECT);
        if (!Files.exists(sf)) throw new IllegalStateException("part3: нет файла выбора " + sf + " (сначала pool-part3)");
        for (String l : Files.readAllLines(sf, java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            if (p[0].equals("слияние")) {
                int v = Integer.parseInt(p[1]);
                if (v < 0 || v >= NSEL || !VN[v].equals(p[2])) throw new IllegalStateException("part3: вариант «" + l + "» не из сетки");
                return v;
            }
        }
        throw new IllegalStateException("part3: в файле выбора нет строки слияния");
    }

    static void ownReport(Setup u, String base, StringBuilder t) throws IOException {
        int ch = readSelect(u, base);
        List<Integer> vs = new ArrayList<>(List.of(0));
        vs.add(ch == 0 ? 1 : ch);
        t.append("\nВыбор по пулу (слияние): ").append(VN[ch]).append(ch == 0 ? "; (Ч1) – справочно" : "").append('\n');
        int[] clamped = new int[2];
        StringBuilder body = new StringBuilder();
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            String[][][] ls = new String[vs.size()][bi.size()][];
            for (int j = 0; j < vs.size(); j++) for (int k = 0; k < bi.size(); k++) ls[j][k] = lines(u, 0, u.blocks.get(bi.get(k)), vs.get(j), false, clamped);
            for (int p : new int[] {1, 0, 2}) {
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> prs = new ArrayList<>();
                String bl = grp + " " + rowName(p, 0);
                for (int j = 0; j < vs.size(); j++) {
                    List<XBaseThr.Unit> us = new ArrayList<>();
                    for (int k = 0; k < bi.size(); k++) {
                        Blk b = u.blocks.get(bi.get(k));
                        XBaseEval.Gt g = p == 2 ? b.g() : XBaseAbl3.mirrorGt(b.g(), p == 0);
                        us.add(new XBaseThr.Unit(g, b.gal(), b.pro(), b.parts(), ls[j][k][p], null, t2));
                    }
                    String lb = grp + " " + rowName(p, vs.get(j)) + (vs.get(j) == ch && ch > 0 ? " (выбор по пулу)" : "");
                    XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                    outs.put(lb, o);
                    if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                    if (j > 0) prs.add(new String[] {lb, bl});
                }
                XBaseLbp.tables(body, prefix + ", " + (p == 2 ? "слияние кандидатов" : XBaseRule.FEATS[p]), outs, u.persons, u.tpl, prs, t2);
                if (!t2) for (String[] pr : prs) {
                    double[] d = XBaseVt.delta(outs.get(pr[0]), outs.get(bl), XBaseThr.WAYS[0], u.persons, u.tpl);
                    dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", pr[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                            : "НЕ выполнено"));
                }
            }
        }
        t.append(hold).append(base.equals("gt") ? dl : "").append(body);
        FarMethods.write(u.out.resolve("xbase_" + base + "_part3.txt"), t);
        System.out.print(t);
    }
}
