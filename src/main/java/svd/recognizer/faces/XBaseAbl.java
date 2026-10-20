package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.Facemark;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * (4а) поочерёдное отключение (PLAN.md п. 5.5; план Васи от 17.10.2026 – TASKS.md, «(4а) – план»; уточнения №2 Жорика от 17.10 и
 * №1–№2 от 19.10). Кандидаты – L1-xs-F и L1-tt-xs-F; база – роли, галереи, пул и когорта feret-xs2 / feret-subsp (FERET eq:
 * межс., внутри, G2; половины №2, два прохода) и gt-subsp (GT межс. N = 6, внутри N = 3, пул FERET), строки базы сверяются с их
 * файлами оценок побайтно.
 * <p>
 * Отключения: (1) пул FERET (L1, L1-tt); (2) корень гистограмм; (3) MLDA → χ² (L0-mean – основная, L0 – рядом; снимает MLDA вместе с
 * пулом); (4) d₁/d₂ → d₁ (по тем же строкам: второе расстояние заменяется постоянной, порядок проб – по d₁); (5) без зеркал (r0 =
 * +∞, всегда X); (6) без выбора модели (всегда X+mirror); (7) без отбора по позе (GT: пробы, отвергнутые отбором, распознаются;
 * пара – база (а), где они – отказ; FERET – отбор ничего не отвергает); (8) кадр по рамке YuNet без выравнивания (k и d – медианы
 * по своей базе; отражение – простое зеркало); (9) YuNet → Хаар: глаза по (ii) в галерее и обучении, без отбора и выбора модели
 * (всегда X+mirror); (i) – проба без двух глаз – отказ, (ii) – глаза по геометрии рамки; справочно – отбор и выбор модели по r_H;
 * таблица (б) – пробы, где Хаар нашёл лицо и два глаза; (9б) Хаар + Facemark LBF: кадр по 5 точкам LBF, отбор |r_LBF| ≤ R_LBF
 * (своя база без ложных рамок) и выбор модели по r0_LBF. Пропуск лица – отказ своему и непринятый чужой; в галерее и обучении –
 * выпадение (остановка при выпадении > 5 % галереи).
 * <p>
 * feret-abl – отчёт, оценки и сводка – только reports/eq/xbase_feret_abl*; gt-abl – reports/faces/xbase/xbase_gt_abl*;
 * abl-summary – сводная таблица «отключено − база» и правило «убрать» (только reports/eq/xbase_abl_summary.txt в каталоге FERET).
 * Пробный режим FACES_ABL_SMOKE=1 – только межс., L1-xs-F, проход 1.
 *
 * @author ssv
 */
final class XBaseAbl {

    static final String[] CAND = {"L1-xs-F", "L1-tt-xs-F"};
    static final String[] NOPOOL = {"L1", "L1-tt"};
    static final IlluminationNorm[] NORM = {IlluminationNorm.NONE, IlluminationNorm.TAN_TRIGGS};
    /** (4): постоянное второе расстояние; d₁ должно быть меньше и не меньше 1e-12 · C. */
    static final double ABS_C = 1e6;
    static final double STOP_SHARE = 0.05;
    static final String SUMMARY = "_summary.tsv";

    enum Sch {
        BASE("база"), NOPOOL("(1) без пула"), NOROOT("(2) без корня"), CHI("(3) χ²"), BOX("(8) кадр по рамке"), HAAR("(9) Хаар"), LBF("(9б) Хаар + LBF");

        final String label;

        Sch(String label) {
            this.label = label;
        }
    }

    /** Строки таблиц по порядку: имя и пара (с чем сравнивается). */
    static final String[][] ROWS = {
        {"база", null},
        {"(1) без пула", "база"},
        {"(2) без корня", "база"},
        {"(3) χ² L0-mean", "база"},
        {"(3) χ² L0 (справочно)", "база"},
        {"(4) d₁ вместо d₁/d₂", "база"},
        {"(5) без зеркал", "база"},
        {"(6) без выбора модели", "база"},
        {"(8) кадр по рамке YuNet", "база"},
        {"(9)(i) Хаар", "база"},
        {"(9)(ii) Хаар, глаза по рамке", "база"},
        {"(9)(i) Хаар + отбор и модель по r_H (справочно)", "база"},
        {"(9б) Хаар + LBF", "база"},
        {"база [Хаар: лицо и два глаза]", null},
        {"(9)(ii) Хаар [Хаар: лицо и два глаза]", "база [Хаар: лицо и два глаза]"},
        {"база (а): отвергнутые отбором – отказ", null},
        {"(7) без отбора по позе", "база (а): отвергнутые отбором – отказ"}};

    private XBaseAbl() {
    }

    // ---------------------------------------------------------------- пороги и кадры

    /** Пороги схем: r_H (справочно), LBF, кадр по рамке (k, d). */
    record Thr(double rH, double r0H, double rLbf, double rLbfAll, double r0Lbf, int ownLbf, int ownLbfAll, int feiLbf, double k, double d, int ownBox) {}

    /** Источник кадров (8), (9), (9б): ключ протокола → строка YuNet, итог Хаара, точки LBF, цветной снимок. */
    static final class Src {
        final Map<String, double[]> row = new HashMap<>();
        final Map<String, HaarFrames.Det> haar = new HashMap<>();
        final Map<String, LbfFrames.Lm> lbf = new HashMap<>();
        final Map<String, HaarFrames.Loader> load = new HashMap<>();
        /** FERET: кэш кадров в derived (a_box, a_haar, a_lbf); null – в памяти. */
        Path cache;

        void putAll(Src o) {
            row.putAll(o.row);
            haar.putAll(o.haar);
            lbf.putAll(o.lbf);
            load.putAll(o.load);
        }
    }

    /** Общие данные (4а): базы Хаара и LBF (кэши), пороги. */
    static final class Common {
        HaarFrames.Bases b;
        Map<String, Map<String, LbfFrames.Lm>> lbf = new HashMap<>();
        Thr th;
        String lbfSig;
    }

    static Common common(Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings) throws IOException {
        Common c = new Common();
        double rLimit = XBaseEval.poseLimit(selLines);
        Path models = Paths.get(settings.loadFacesModelsDir());
        CascadeClassifier face = HaarDebug.load(models.resolve(HaarDebug.FACE_FILE));
        CascadeClassifier eye = HaarDebug.load(models.resolve(HaarDebug.EYE_GLASSES_FILE));
        c.b = HaarFrames.bases(exportDir, feiKeys, rLimit, settings, face, eye);
        c.lbfSig = LbfFrames.signature(models);
        Facemark fm = LbfFrames.load(models);
        for (HaarFrames.Set s : List.of(c.b.own(), c.b.fei(), c.b.gt(), c.b.feret())) c.lbf.put(s.name(), LbfFrames.cached(LbfFrames.cacheFile(c.b, s.name()), c.lbfSig, s, fm, s.name()));
        HaarFrames.Bases b = c.b;
        Map<String, LbfFrames.Lm> ownL = c.lbf.get(b.own().name());
        Map<String, LbfFrames.Lm> feiL = c.lbf.get(b.fei().name());
        double[] all = b.ownProto().stream().filter(k -> ownL.get(k) != null).mapToDouble(k -> Math.abs(ownL.get(k).pose())).sorted().toArray();
        double[] tru = b.ownProto().stream().filter(k -> ownL.get(k) != null && b.own().yunet().get(k) != null
                && HaarFrames.iou(b.own().haar().get(k).face(), b.own().yunet().get(k)) >= HaarFrames.IOU_FALSE).mapToDouble(k -> Math.abs(ownL.get(k).pose())).sorted()
                .toArray();
        double[] fei = b.fei1114().stream().filter(k -> feiL.get(k) != null).mapToDouble(k -> Math.abs(feiL.get(k).pose())).sorted().toArray();
        // Кадр по рамке: k = медиана 112 / (масштаб подобия кадра «а» · h рамки), d – медиана смещения центра кадра «а» от центра рамки (доли h).
        List<Double> ks = new ArrayList<>();
        List<Double> ds = new ArrayList<>();
        for (String k : b.ownProto()) {
            double[] row = b.own().yunet().get(k);
            if (row == null) continue;
            double[][] lm = new double[5][2];
            for (int j = 0; j < 5; j++) {
                lm[j][0] = row[4 + 2 * j];
                lm[j][1] = row[5 + 2 * j];
            }
            Mat m = FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(HaarFrames.W, HaarFrames.H));
            double a = m.get(0, 0)[0];
            double bb = m.get(1, 0)[0];
            double tx = m.get(0, 2)[0];
            double ty = m.get(1, 2)[0];
            double s2 = a * a + bb * bb;
            // Обратное подобие: p = Rᵀ (c − t) / s².
            double cx = HaarFrames.W / 2.0 - tx;
            double cy = HaarFrames.H / 2.0 - ty;
            double py = (-bb * cx + a * cy) / s2;
            ks.add(HaarFrames.H / (Math.sqrt(s2) * row[3]));
            ds.add((py - (row[1] + row[3] / 2)) / row[3]);
            m.release();
        }
        double[] kk = FeiBootstrap.sorted(ks.stream().mapToDouble(Double::doubleValue).toArray());
        double[] dd = FeiBootstrap.sorted(ds.stream().mapToDouble(Double::doubleValue).toArray());
        c.th = new Thr(HaarFrames.percentileAbs(b.ownProto(), b.own().haar()), HaarFrames.percentileAbs(b.fei1114(), b.fei().haar()),
                FeiDataset.percentile(tru, FeiDataset.POSE_PERCENTILE), FeiDataset.percentile(all, FeiDataset.POSE_PERCENTILE),
                FeiDataset.percentile(fei, FeiDataset.POSE_PERCENTILE), tru.length, all.length, fei.length, FeiBootstrap.quantile(kk, 0.5),
                FeiBootstrap.quantile(dd, 0.5), kk.length);
        return c;
    }

    /** Кадр (8): прямоугольник 92:112 высотой k·h с центром в центре рамки, сдвиг по вертикали d·h, без поворота; за краем – повтор края. */
    static Mat boxFrame(Mat gray, double[] row, double k, double d) {
        double h = row[3];
        double hh = k * h;
        double ww = hh * HaarFrames.W / HaarFrames.H;
        double cx = row[0] + row[2] / 2;
        double cy = row[1] + row[3] / 2 + d * h;
        int x0 = (int) Math.round(cx - ww / 2);
        int y0 = (int) Math.round(cy - hh / 2);
        int w = (int) Math.round(ww);
        int ht = (int) Math.round(hh);
        int l = Math.max(0, -x0);
        int t = Math.max(0, -y0);
        int r = Math.max(0, x0 + w - gray.cols());
        int bt = Math.max(0, y0 + ht - gray.rows());
        Mat src = gray;
        if (l + t + r + bt > 0) {
            src = new Mat();
            Core.copyMakeBorder(gray, src, t, bt, l, r, Core.BORDER_REPLICATE);
        }
        Mat roi = src.submat(new Rect(x0 + l, y0 + t, w, ht));
        Mat out = new Mat();
        Imgproc.resize(roi, out, new Size(HaarFrames.W, HaarFrames.H), 0, 0, Imgproc.INTER_AREA);
        if (src != gray) src.release();
        return out;
    }

    /** Кадры (8), (9)(ii), (9б) ключей keys: [box, haar, lbf] (null – нет рамки YuNet / лица Хаара / точек). FERET – кэш PNG в derived. */
    static Map<String, Mat[]> frames3(Src s, Collection<String> keys, Thr th) throws IOException {
        Map<String, Mat[]> out = new LinkedHashMap<>();
        String[] dirs = {"a_box", "a_haar", "a_lbf"};
        if (s.cache != null) for (String d : dirs) Files.createDirectories(s.cache.resolve(d));
        int n = 0;
        for (String k : keys) {
            boolean[] need = {s.row.get(k) != null, s.haar.get(k).found(), s.lbf.get(k) != null};
            Mat[] f = new Mat[3];
            boolean cached = s.cache != null;
            if (cached) for (int j = 0; j < 3; j++) if (need[j] && !Files.exists(s.cache.resolve(dirs[j]).resolve(k + ".png"))) cached = false;
            if (cached) {
                for (int j = 0; j < 3; j++) if (need[j]) f[j] = FeretData.frame(s.cache.getParent(), dirs[j], k);
            } else {
                Mat color = s.load.get(k).color(k);
                Mat in = HaarFrames.input(color, 1.0);
                color.release();
                if (need[0]) f[0] = boxFrame(in, s.row.get(k), th.k(), th.d());
                if (need[1]) f[1] = HaarFrames.frame(in, 1.0, s.haar.get(k), true);
                if (need[2]) f[2] = LbfFrames.frame(in, 1.0, s.lbf.get(k).five());
                in.release();
                if (s.cache != null) for (int j = 0; j < 3; j++) if (need[j]) FeretData.writePng(s.cache.resolve(dirs[j]).resolve(k + ".png"), f[j]);
                if (++n % 500 == 0) System.out.printf(Locale.ROOT, "abl: кадры (8), (9), (9б) – %d новых%n", n);
            }
            out.put(k, f);
        }
        return out;
    }

    /** Векторы L1 (кандидат): корень гистограмм LBP или без корня; отражение – FaceMirror (способ а) или простое зеркало. */
    static XBaseEval l1(Map<String, Mat> fr, IlluminationNorm.NormParams p, IlluminationNorm n, boolean root, boolean flip) {
        XBaseEval x = new XBaseEval();
        Map<String, double[]> v = new HashMap<>();
        Map<String, double[]> m = new HashMap<>();
        for (Map.Entry<String, Mat> e : fr.entrySet()) {
            Mat mm;
            if (flip) {
                mm = new Mat();
                Core.flip(e.getValue(), mm, 1);
            } else {
                mm = FaceMirror.mirror(e.getValue());
            }
            double[] h = LbpFeature.histogram(e.getValue(), n, p);
            double[] hm = LbpFeature.histogram(mm, n, p);
            v.put(e.getKey(), root ? LbpFeature.sqrt(h) : h);
            m.put(e.getKey(), root ? LbpFeature.sqrt(hm) : hm);
            mm.release();
        }
        x.vec.put(V.L1, v);
        x.mir.put(V.L1, m);
        return x;
    }

    /** Векторы схемы sc и признака f по кадрам fr. */
    static XBaseEval vectors(Sch sc, int f, Map<String, Mat> fr, IlluminationNorm.NormParams p) {
        XBaseEval x = new XBaseEval();
        switch (sc) {
            case NOROOT -> {
                return l1(fr, p, NORM[f], false, false);
            }
            case BOX -> {
                return l1(fr, p, NORM[f], true, true);
            }
            case CHI -> x.vectors(fr, EnumSet.of(V.L0, V.L0MEAN), p, NORM[f]);
            default -> x.vectors(fr, EnumSet.of(V.L1), p, NORM[f]);
        }
        return x;
    }

    static Mat dummy() {
        return new Mat(HaarFrames.H, HaarFrames.W, CvType.CV_8UC1, new Scalar(128));
    }

    // ---------------------------------------------------------------- строки и разбор

    /** (4): второе расстояние – постоянная ABS_C (у отказов +∞ остаётся); порядок проб – по d₁. */
    static String abs(String lines) {
        StringBuilder b = new StringBuilder();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (!c[10].equals("Infinity")) {
                double d1 = Double.parseDouble(c[8]);
                if (d1 >= ABS_C || (d1 > 0 && d1 / ABS_C < 1e-12)) throw new IllegalStateException("(4): d₁ вне диапазона постоянной: " + l);
                c[10] = Double.toString(ABS_C);
            }
            b.append(String.join("\t", c)).append('\n');
        }
        return b.toString();
    }

    /** Копия Gt с другой когортой (обучение), позами и r0. */
    static XBaseEval.Gt copy(XBaseEval.Gt g, List<List<String>> cohort, Map<String, Double> rOf, double r0) {
        XBaseEval.Gt c = new XBaseEval.Gt();
        c.seed = g.seed;
        c.b = g.b;
        c.threads = g.threads;
        c.rLimit = g.rLimit;
        c.dir = g.dir;
        c.params = g.params;
        c.imageNo = g.imageNo;
        c.samples = g.samples;
        c.persons = g.persons;
        c.pIdx = g.pIdx;
        c.ctrl = g.ctrl;
        c.byPerson = g.byPerson;
        c.own = g.own;
        c.cohort = cohort;
        c.rOf = rOf;
        c.r0 = r0;
        return c;
    }

    /** Накопитель строки таблицы: вариант (V), конфигурации и «чужие» по проходам. */
    static final class Acc {
        final V v;
        final List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        final List<FeiThreshold.Imp> imps = new ArrayList<>();

        Acc(V v) {
            this.v = v;
        }

        void add(XBaseEval.Parsed p) {
            cfgs.add(p.cfg.get(v));
            imps.add(p.imp.get(v));
        }
    }

    /** Задание: постановка, признак, проход; роли базы. */
    record Job(String conf, int f, int pass, XBaseEval.Gt g, Map<String, List<String>> gal, List<Probe> probes, XBaseEval.Gt gU, List<Probe> probesU,
               List<List<String>> parts, int s, List<List<String>> cohort, List<List<String>> poolCls, String oldBase, String oldNoPool) {

        String name() {
            return conf + "_" + f + "_" + pass;
        }
    }

    /** Ход счёта: накопители, строки оценок, сверка, время. */
    static final class Run {
        final Map<String, Acc> acc = new LinkedHashMap<>();
        final StringBuilder lines = new StringBuilder();
        final StringBuilder check = new StringBuilder();
        final Map<String, String> baseLines = new HashMap<>();
        double fitS;

        void add(Job j, String row, V v, XBaseEval.Parsed p) {
            acc.computeIfAbsent(j.conf() + " " + CAND[j.f()] + " " + row, x -> new Acc(v)).add(p);
        }
    }

    static List<List<String>> filter(List<List<String>> cls, Set<String> ok) {
        List<List<String>> out = new ArrayList<>();
        for (List<String> c : cls) {
            List<String> k = c.stream().filter(ok::contains).toList();
            if (!k.isEmpty()) out.add(k);
        }
        return out;
    }

    static Set<String> ids(List<Probe> probes, java.util.function.Predicate<Probe> pick) {
        Set<String> out = new HashSet<>();
        for (Probe q : probes) if (pick.test(q)) out.add(q.role() + "\t" + q.person() + "\t" + q.image());
        return out;
    }

    /**
     * Счёт схемы sc по заданиям jobs: векторы x[f]; для (9) и (9б) – кадры есть только у ключей okKeys (прочие – пустой кадр,
     * пробы – отказ, в галерее и обучении – выпадение); src – Хаар и LBF по ключам протокола; probeKey – ключ протокола пробы.
     */
    static void scheme(Sch sc, List<Job> jobs, XBaseEval[] x, Set<String> okKeys, Src src, Thr th, double r0, XBaseOther.Checkpoint cp, Run run, String mode)
            throws IOException {
        boolean drop = sc == Sch.HAAR || sc == Sch.LBF;
        for (Job j : jobs) {
            if (x[j.f()] == null) continue;
            Map<String, List<String>> gal = j.gal();
            List<List<String>> cohort = j.cohort();
            List<List<String>> pool = j.poolCls();
            if (drop) {
                gal = new LinkedHashMap<>();
                for (Map.Entry<String, List<String>> e : j.gal().entrySet()) gal.put(e.getKey(), e.getValue().stream().filter(okKeys::contains).toList());
                cohort = filter(cohort, okKeys);
                pool = filter(pool, okKeys);
            }
            List<List<String>> train = new ArrayList<>(cohort);
            if (sc != Sch.NOPOOL && sc != Sch.CHI) train.addAll(pool);
            boolean base = sc == Sch.BASE;
            XBaseEval.Gt gFit = base && j.gU() != null ? j.gU() : j.g();
            List<Probe> pFit = base && j.probesU() != null ? j.probesU() : j.probes();
            Set<V> vs = sc == Sch.CHI ? EnumSet.of(V.L0, V.L0MEAN) : EnumSet.of(V.L1);
            String name = sc.ordinal() + "_" + j.name();
            String ls = cp.read(name);
            if (ls == null) {
                long t1 = System.nanoTime();
                ls = XBaseEval.fitLines(copy(gFit, train, gFit.rOf, gFit.r0), x[j.f()], vs, gal, pFit, j.parts(), j.s());
                double s = (System.nanoTime() - t1) / 1e9;
                run.fitS += s;
                cp.save(name, ls);
                System.out.printf(Locale.ROOT, "%s: %s, %s, %s, проход %d ready (%.0f s)%n", mode, sc.label, j.conf(), CAND[j.f()], j.pass() + 1, s);
            }
            run.lines.append("# схема ").append(sc.label).append(", ").append(j.conf()).append(", кандидат ").append(CAND[j.f()]).append(", проход ")
                    .append(j.pass() + 1).append('\n').append(ls);
            XBaseEval.Gt g = copy(j.g(), train, j.g().rOf, j.g().r0);
            switch (sc) {
                case BASE -> {
                    checkOld(run, j, ls, j.oldBase(), "база");
                    run.baseLines.put(j.name(), ls);
                    run.add(j, "база", V.L1, XBaseEval.parse(g, EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                    run.add(j, "(4) d₁ вместо d₁/d₂", V.L1, XBaseEval.parse(g, EnumSet.of(V.L1), gal, j.probes(), j.parts(), abs(ls)));
                    run.add(j, "(5) без зеркал", V.L1, XBaseEval.parse(copy(g, train, g.rOf, Double.POSITIVE_INFINITY), EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                    run.add(j, "(6) без выбора модели", V.L1, XBaseEval.parse(copy(g, train, g.rOf, -1), EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                    if (j.gU() != null) {
                        XBaseEval.Gt gU = copy(j.gU(), train, j.gU().rOf, j.gU().r0);
                        Set<String> inI = ids(j.probes(), q -> true);
                        Set<String> rej = ids(j.probesU(), q -> !inI.contains(q.role() + "\t" + q.person() + "\t" + q.image()));
                        run.add(j, "база (а): отвергнутые отбором – отказ", V.L1,
                                XBaseEval.parse(gU, EnumSet.of(V.L1), gal, j.probesU(), j.parts(), XBaseMp2.refuse(ls, rej)));
                        run.add(j, "(7) без отбора по позе", V.L1, XBaseEval.parse(gU, EnumSet.of(V.L1), gal, j.probesU(), j.parts(), ls));
                    }
                }
                case NOPOOL -> {
                    checkOld(run, j, ls, j.oldNoPool(), "(1) без пула");
                    run.add(j, "(1) без пула", V.L1, XBaseEval.parse(g, EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                }
                case NOROOT -> run.add(j, "(2) без корня", V.L1, XBaseEval.parse(g, EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                case CHI -> {
                    XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L0, V.L0MEAN), gal, j.probes(), j.parts(), ls);
                    run.add(j, "(3) χ² L0-mean", V.L0MEAN, p);
                    run.add(j, "(3) χ² L0 (справочно)", V.L0, p);
                }
                case BOX -> run.add(j, "(8) кадр по рамке YuNet", V.L1, XBaseEval.parse(g, EnumSet.of(V.L1), gal, j.probes(), j.parts(), ls));
                case HAAR -> {
                    Set<String> noFace = ids(j.probes(), q -> !src.haar.get(q.key()).found());
                    Set<String> noEyes = ids(j.probes(), q -> !src.haar.get(q.key()).both());
                    Map<String, Double> rH = new HashMap<>(j.g().rOf);
                    Set<String> poseRej = new HashSet<>();
                    for (Probe q : j.probes()) {
                        HaarFrames.Det d = src.haar.get(q.key());
                        double r = d.both() ? Math.abs(d.pose()) : 0;
                        rH.put(q.key(), r);
                        if (r > th.rH()) poseRej.add(q.role() + "\t" + q.person() + "\t" + q.image());
                    }
                    for (List<String> ks : gal.values()) for (String k : ks) rH.put(k, src.haar.get(k).both() ? Math.abs(src.haar.get(k).pose()) : 0);
                    XBaseEval.Gt gm = copy(g, train, g.rOf, -1);
                    run.add(j, "(9)(i) Хаар", V.L1, XBaseEval.parse(gm, EnumSet.of(V.L1), gal, j.probes(), j.parts(), XBaseMp2.refuse(ls, noEyes)));
                    run.add(j, "(9)(ii) Хаар, глаза по рамке", V.L1, XBaseEval.parse(gm, EnumSet.of(V.L1), gal, j.probes(), j.parts(), XBaseMp2.refuse(ls, noFace)));
                    Set<String> rejH = new HashSet<>(noEyes);
                    rejH.addAll(poseRej);
                    run.add(j, "(9)(i) Хаар + отбор и модель по r_H (справочно)", V.L1,
                            XBaseEval.parse(copy(g, train, rH, th.r0H()), EnumSet.of(V.L1), gal, j.probes(), j.parts(), XBaseMp2.refuse(ls, rejH)));
                    List<Probe> both = j.probes().stream().filter(q -> src.haar.get(q.key()).both()).toList();
                    String bl = run.baseLines.get(j.name());
                    if (bl != null) {
                        XBaseEval.Gt gb = copy(j.g(), j.cohort(), j.g().rOf, j.g().r0);
                        run.add(j, "база [Хаар: лицо и два глаза]", V.L1, XBaseEval.parse(gb, EnumSet.of(V.L1), j.gal(), both, j.parts(), bl));
                        run.add(j, "(9)(ii) Хаар [Хаар: лицо и два глаза]", V.L1, XBaseEval.parse(gm, EnumSet.of(V.L1), gal, both, j.parts(), ls));
                    }
                }
                case LBF -> {
                    Map<String, Double> rL = new HashMap<>(j.g().rOf);
                    Set<String> rej = new HashSet<>();
                    for (Probe q : j.probes()) {
                        LbfFrames.Lm lm = src.lbf.get(q.key());
                        double r = lm == null ? 0 : Math.abs(lm.pose());
                        rL.put(q.key(), r);
                        if (lm == null || r > th.rLbf()) rej.add(q.role() + "\t" + q.person() + "\t" + q.image());
                    }
                    for (List<String> ks : gal.values()) for (String k : ks) rL.put(k, Math.abs(src.lbf.get(k).pose()));
                    run.add(j, "(9б) Хаар + LBF", V.L1, XBaseEval.parse(copy(g, train, rL, th.r0Lbf()), EnumSet.of(V.L1), gal, j.probes(), j.parts(),
                            XBaseMp2.refuse(ls, rej)));
                }
            }
        }
    }

    /** Сверка строк L1 с прежним файлом оценок (блок old; null – сверять не с чем). */
    static void checkOld(Run run, Job j, String ls, String old, String what) {
        String row = String.format(Locale.ROOT, "%s, %s, %s, проход %d", what, j.conf(), what.equals("база") ? CAND[j.f()] : NOPOOL[j.f()], j.pass() + 1);
        if (old == null) {
            run.check.append(row).append(": прежнего блока нет – новая строка, сверять не с чем\n");
            return;
        }
        List<String> mine = XBaseSubsp.l1Rows(XBaseMp2.only(ls, ids(j.probes(), q -> true)));
        boolean same = XBaseSubsp.l1Rows(old).equals(mine);
        run.check.append(String.format(Locale.ROOT, "%s: %s (%d строк)%n", row, same ? "совпало" : "НЕ СОВПАЛО", mine.size()));
        if (!same) throw new IllegalStateException("abl: " + row + " не совпала с прежним файлом оценок");
    }

    // ---------------------------------------------------------------- таблицы и сводка

    static int order(String label) {
        for (int i = 0; i < ROWS.length; i++) if (label.endsWith(" " + ROWS[i][0])) return i;
        throw new IllegalStateException("abl: нет строки " + label);
    }

    static String rowOf(String label) {
        return ROWS[order(label)][0];
    }

    /** Таблицы групп постановок; сводка – строки TSV «постановка, кандидат, строка, пара, разделимость 1 % и EER: точка и интервал разности, П3». */
    static void tables(StringBuilder r, StringBuilder sum, Run run, int persons, XBaseEval.Gt tpl, List<List<String>> groups, String base) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons);
        for (List<String> grp : groups) {
            List<String> labels = new ArrayList<>();
            for (String l : run.acc.keySet()) if (grp.stream().anyMatch(g -> l.startsWith(g + " "))) labels.add(l);
            labels.sort((a, b) -> {
                int ca = a.contains(" " + CAND[1] + " ") ? 1 : 0;
                int cb = b.contains(" " + CAND[1] + " ") ? 1 : 0;
                if (ca != cb) return ca - cb;
                int ga = grp.indexOf(a.substring(0, a.indexOf(' ')));
                int gb = grp.indexOf(b.substring(0, b.indexOf(' ')));
                if (ga != gb) return ga - gb;
                return order(a) - order(b);
            });
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            for (String l : labels) {
                Acc a = run.acc.get(l);
                res.put(l, FeiThreshold.evaluate(l, a.v.ratioId(), a.cfgs, a.imps, persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                sep.put(l, XBaseOther.sep(a.cfgs, unit, w));
            }
            for (String l : labels) {
                String pair = ROWS[order(l)][1];
                if (pair == null) continue;
                String p = l.substring(0, l.length() - rowOf(l).length()) + pair;
                if (res.containsKey(p)) pairs.add(new String[] {l, p});
            }
            r.append("\n\n######## ").append(base).append(": ").append(String.join(", ", grp)).append(" — оба прохода ########\n");
            XBaseOther.tables(r, res, sep, tpl.b, pairs);
            for (String[] p : pairs) {
                XBaseOther.Sep a = sep.get(p[0]);
                XBaseOther.Sep z = sep.get(p[1]);
                String conf = p[0].substring(0, p[0].indexOf(' '));
                String cand = p[0].contains(" " + CAND[1] + " ") ? CAND[1] : CAND[0];
                List<String> cells = new ArrayList<>(List.of(base, conf, cand, rowOf(p[0]), rowOf(p[1])));
                cells.addAll(diffCells(a.err[0] - z.err[0], a.errRep[0], z.errRep[0]));
                cells.addAll(diffCells(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                cells.add(FeiThreshold.errCell(res.get(p[0]), 3, 0, 0, tpl.b));
                cells.add(FeiThreshold.errCell(res.get(p[1]), 3, 0, 0, tpl.b));
                cells.add(String.format(Locale.ROOT, "%.2f", FeiThreshold.farPct(res.get(p[0]), 3, 0)));
                cells.add(String.format(Locale.ROOT, "%.2f", FeiThreshold.farPct(res.get(p[1]), 3, 0)));
                cells.add(String.format(Locale.ROOT, "%.1f", a.err[0]));
                cells.add(String.format(Locale.ROOT, "%.1f", z.err[0]));
                sum.append(String.join("\t", cells)).append('\n');
            }
        }
    }

    /** Точка и границы 2,5–97,5 % парной разности (те же реплики). */
    static List<String> diffCells(double point, double[] a, double[] z) {
        if (a.length == 0) return List.of(String.format(Locale.ROOT, "%.2f", point), "", "");
        double[] d = new double[a.length];
        for (int q = 0; q < a.length; q++) d[q] = a[q] - z[q];
        d = FeiBootstrap.sorted(d);
        return List.of(String.format(Locale.ROOT, "%.2f", point), String.format(Locale.ROOT, "%.2f", FeiBootstrap.quantile(d, 0.025)),
                String.format(Locale.ROOT, "%.2f", FeiBootstrap.quantile(d, 0.975)));
    }

    static String sumHead() {
        return "# база\tпостановка\tкандидат\tстрока\tпара\tразд. 1 %: разность\tниз\tверх\tEER: разность\tниз\tверх\tП3 FAR 1 %: строка\tпара\tFAR контроля П3 %: "
                + "строка\tпара\tразд. 1 %: строка\tпара\n";
    }

    static void header(StringBuilder r, Thr th, XBaseEval.Gt tpl, Common c) {
        r.append(String.format(Locale.ROOT, "Кандидаты: %s, %s (MLDA (Wа) + когорта FEI + пул FERET, LBP u2 8/2 7×7 с корнем, кадр «а», d₁/d₂, зеркала и выбор "
                + "модели по |r| ≤ r0 = %s, отбор |r| ≤ R = %.4f).%n", CAND[0], CAND[1], tpl.r0, tpl.rLimit));
        r.append(String.format(Locale.ROOT, "(8): k = %.4f, d = %+.4f (медианы по %d снимкам протокола своей базы с рамкой YuNet: k = 112 / (масштаб подобия кадра «а» · h), "
                + "d – смещение центра кадра «а» от центра рамки по вертикали, доли h); отражение – простое зеркало кадра.%n", th.k(), th.d(), th.ownBox()));
        r.append(String.format(Locale.ROOT, "(9): Хаар – кэши haar.tsv (abl-haar); кадр – подобие по двум глазам на шаблон кадра «а»; в галерее и обучении – глаза по "
                + "(ii); всегда X+mirror. Справочно r_H: R_H = %.4f, r0_H = %.4f.%n", th.rH(), th.r0H()));
        r.append(String.format(Locale.ROOT, "(9б): LBF (%s); R_LBF = %.4f – своя база без ложных рамок (%d снимков; с ложными – %.4f по %d), r0_LBF = %.4f (%d снимков №11–14 "
                + "FEI).%n", c.lbfSig.substring(0, c.lbfSig.indexOf('|', 8)), th.rLbf(), th.ownLbf(), th.rLbfAll(), th.ownLbfAll(), th.r0Lbf(), th.feiLbf()));
        r.append(String.format(Locale.ROOT, "(4): второе расстояние заменено постоянной %.0e (порядок проб – по d₁; отказы остаются).%n", ABS_C));
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d; «*» — интервал разности не содержит 0.%n", tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
    }

    static boolean smoke() {
        return "1".equals(System.getenv("FACES_ABL_SMOKE"));
    }

    /** Выпадение для (9): доли снимков галереи своих без лица Хаара по постановкам; true – больше STOP_SHARE. */
    static boolean drops(StringBuilder r, List<Job> jobs, Set<String> ok, Src src, List<List<String>> cohort, List<List<String>> pool) {
        r.append("\n=== (9), (9б): выпадение (нет лица Хаара) и отказы проб ===\n");
        r.append("постановка | проход | снимков галереи своих | выпало | проб своих и контроля | нет лица | глаз < 2 | вне R_LBF или нет точек\n");
        boolean stop = false;
        Set<String> seen = new HashSet<>();
        for (Job j : jobs) {
            if (j.f() != 0 || !seen.add(j.conf() + j.pass())) continue;
            int gn = 0;
            int gd = 0;
            for (List<String> ks : j.gal().values()) {
                for (String k : ks) {
                    gn++;
                    if (!ok.contains(k)) gd++;
                }
            }
            boolean emptyOwn = j.gal().values().stream().anyMatch(ks -> ks.stream().noneMatch(ok::contains));
            if ((double) gd / gn > STOP_SHARE || emptyOwn) stop = true;
            long nf = j.probes().stream().filter(q -> !src.haar.get(q.key()).found()).count();
            long ne = j.probes().stream().filter(q -> !src.haar.get(q.key()).both()).count();
            r.append(String.format(Locale.ROOT, "%s | %d | %d | %d (%.1f %%)%s | %d | %d | %d | –%n", j.conf(), j.pass() + 1, gn, gd, 100.0 * gd / gn,
                    emptyOwn ? " – есть свой без галереи" : "", j.probes().size(), nf, ne));
        }
        int cn = cohort.stream().mapToInt(List::size).sum();
        long cd = cohort.stream().flatMap(List::stream).filter(k -> !ok.contains(k)).count();
        int pn = pool.stream().mapToInt(List::size).sum();
        long pd = pool.stream().flatMap(List::stream).filter(k -> !ok.contains(k)).count();
        r.append(String.format(Locale.ROOT, "Когорта FEI: снимков %d, выпало %d; пул FERET: снимков %d, выпало %d.%n", cn, cd, pn, pd));
        return stop;
    }

    // ---------------------------------------------------------------- FERET

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-abl: только равные галереи (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        boolean smoke = smoke();
        Common cm = common(exportDir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(out);
        XBaseFeret.Roles ro = XBaseFeret.equalize(su.ro, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        java.util.Collections.shuffle(perm, new java.util.Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        Map<String, String> xs2 = XBaseSubsp.blocks(out.resolve("xbase_feret_xs2_scores.tsv.gz"));
        Map<String, String> subsp = XBaseSubsp.blocks(out.resolve("xbase_feret_subsp_scores.tsv.gz"));
        List<Job> jobs = new ArrayList<>();
        Set<String> used = new TreeSet<>();
        for (XBaseXsF.Conf c : XBaseMp2.confs(ro)) {
            if (smoke && !c.name().equals("межс.")) continue;
            for (String p : c.own()) {
                used.addAll(c.gal().get(p));
                used.addAll(c.pro().get(p));
            }
            for (int f = 0; f < 2; f++) {
                if (smoke && f > 0) continue;
                for (int ps = 0; ps < 2; ps++) {
                    if (smoke && ps > 0) continue;
                    List<String> own = c.own().stream().filter(halves.get(ps)::contains).toList();
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Map<String, List<String>> pro = new LinkedHashMap<>();
                    for (String p : own) {
                        gal.put(p, c.gal().get(p));
                        pro.put(p, c.pro().get(p));
                    }
                    Map<String, List<List<String>>> pp = new TreeMap<>();
                    for (String p : halves.get(1 - ps)) pp.put(p, pool.get(p));
                    List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp, Set.of(), Map.of()).extra();
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                    String oldB = c.name().equals("G2") ? subsp.get("G2, проекция " + CAND[f] + ", проход " + (ps + 1) + ", k = 0")
                            : xs2.get("проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + CAND[f]);
                    String oldN = c.name().equals("G2") ? subsp.get("G2, проекция " + NOPOOL[f] + ", проход " + (ps + 1) + ", k = 0")
                            : xs2.get("проход " + (ps + 1) + ", конфигурация " + c.name() + ", строка " + NOPOOL[f]);
                    jobs.add(new Job(c.name(), f, ps, g, gal, XBaseEval.probes(g, pro), null, null, XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s()),
                            c.s(), cohort, poolCls, oldB, oldN));
                }
            }
        }
        ro.ctrlKeys.values().forEach(used::addAll);
        pool.values().forEach(ds -> ds.forEach(used::addAll));

        // Источник кадров (8), (9), (9б): FERET (кэш derived) и когорта FEI (исходные снимки).
        Src frSrc = feretSrc(su, cm, used);
        Src feiSrc = feiSrc(export, exportDir, cohortKeys, cm, settings);
        // Справочник Хаара и LBF по всем ключам; кадры – отдельно (FERET – кэш derived, FEI – в памяти).
        Src src = new Src();
        src.putAll(frSrc);
        src.putAll(feiSrc);
        Set<String> ok = new HashSet<>();
        for (String k : src.haar.keySet()) if (src.haar.get(k).found()) ok.add(k);
        Thr th = cm.th;

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (4а) поочерёдное отключение, FERET eq (XBaseEval feret-abl): Color FERET — только локально").append(smoke ? " — ПРОБНЫЙ (FACES_ABL_SMOKE)" : "")
                .append('\n');
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet, кадры «а» и Хаар/LBF – derived/; кадры (8), (9), (9б) – derived/a_box, a_haar, a_lbf); когорта FEI — "
                + "экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        header(r, th, tpl, cm);
        r.append(String.format(Locale.ROOT, "Роли – feret-xs2 / feret-subsp (половины №2, seed faces.seed + %d; два прохода; пул – другая половина, люди – один класс); люди "
                + "бутстрепа – пул и контроль (%d). (7) на FERET пусто: отбор базы fa/fb не отвергает ни одного снимка.%n", XBaseXsF.SPLIT_SEED_OFFSET, persons.size()));
        boolean stop = drops(r, jobs, ok, src, cohort, jobs.isEmpty() ? List.of() : jobs.get(0).poolCls());
        System.out.print(r);
        System.out.flush();
        String suffix = smoke ? "_smoke" : "";
        if (stop) {
            r.append(String.format(Locale.ROOT, "%nОСТАНОВКА: выпадение галереи своих в (9) больше %.0f %% или свой без галереи – счёт не выполнялся.%n", 100 * STOP_SHARE));
            FarMethods.write(out.resolve("xbase_feret_abl" + suffix + ".txt"), r);
            return;
        }
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-abl%s %s; seed %d; r0 %s; R %s; пул %d; k %s; d %s; R_H %s; r0_H %s; R_LBF %s; "
                + "r0_LBF %s; %s", suffix, dir, tpl.seed, tpl.r0, tpl.rLimit, pool.size(), th.k(), th.d(), th.rH(), th.r0H(), th.rLbf(), th.r0Lbf(), cm.lbfSig));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_abl_eq" + suffix), fp);
        Run run = new Run();
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        for (Sch sc : Sch.values()) {
            Map<String, Mat> fr = new LinkedHashMap<>();
            if (sc == Sch.BOX || sc == Sch.HAAR || sc == Sch.LBF) {
                int j = sc == Sch.BOX ? 0 : sc == Sch.HAAR ? 1 : 2;
                Map<String, Mat[]> f3 = new LinkedHashMap<>(frames3(frSrc, used, th));
                f3.putAll(frames3(feiSrc, new TreeSet<>(cohortKeys), th));
                for (Map.Entry<String, Mat[]> e : f3.entrySet()) {
                    Mat m = e.getValue()[j];
                    fr.put(e.getKey(), m != null ? m : dummy());
                    for (int q = 0; q < 3; q++) if (q != j && e.getValue()[q] != null) e.getValue()[q].release();
                }
            } else {
                for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
                fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            }
            XBaseEval[] x = {vectors(sc, 0, fr, tpl.params), smoke ? null : vectors(sc, 1, fr, tpl.params)};
            fr.values().forEach(Mat::release);
            Set<String> okSc = sc == Sch.LBF ? lbfOk(src) : ok;
            scheme(sc, jobs, x, okSc, src, th, tpl.r0, cp, run, "feret-abl");
        }
        r.append("\n=== Проверка базы: строки L1 и +mirror – побайтно с файлами оценок feret-xs2 (межс., внутри) / feret-subsp (G2, k = 0) ===\n").append(run.check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", run.fitS, tpl.threads));
        StringBuilder sum = new StringBuilder(sumHead());
        tables(r, sum, run, persons.size(), tpl, smoke ? List.of(List.of("межс.")) : List.of(List.of("межс."), List.of("G2"), List.of("внутри")), "FERET");
        XBaseOther.writeScores(out.resolve("xbase_feret_abl" + suffix + "_scores.tsv.gz"), "# FERET, (4а) поочерёдное отключение, XBaseAbl feret-abl, только локально; блоки — "
                + "схема, постановка, кандидат, проход\n# Код: коммит " + FarMethods.commit() + "\n", run.lines.toString());
        FarMethods.write(out.resolve("xbase_feret_abl" + suffix + SUMMARY), sum);
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; память JVM (наибольшая занятая куча за прогон не отслеживается; max %d МБ).%n", (System.nanoTime() - t0) / 1e9,
                Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_feret_abl" + suffix + ".txt"), r);
        System.out.print(r);
        cp.delete();
    }

    static Set<String> lbfOk(Src s) {
        Set<String> out = new HashSet<>();
        for (Map.Entry<String, LbfFrames.Lm> e : s.lbf.entrySet()) if (e.getValue() != null) out.add(e.getKey());
        return out;
    }

    /** Источник FERET: ключ снимка → YuNet, Хаар, LBF, раскодирование; кадры – кэш derived. */
    static Src feretSrc(XBaseA0.FeretSetup su, Common cm, Set<String> keys) {
        Src s = new Src();
        s.cache = su.dir.resolve(FeretData.DERIVED);
        Map<String, LbfFrames.Lm> l = cm.lbf.get(cm.b.feret().name());
        for (String k : keys) {
            HaarFrames.Det d = cm.b.feret().haar().get(k);
            if (d == null) throw new IllegalStateException("abl: нет Хаара FERET для " + k);
            s.row.put(k, su.det.get(k));
            s.haar.put(k, d);
            s.lbf.put(k, l.get(k));
            Img im = su.byKey.get(k);
            s.load.put(k, x -> FeretData.decode(im.file()));
        }
        return s;
    }

    /** Источник когорты FEI: ключ экспорта → YuNet экспорта, Хаар и LBF исходного снимка (ключ «человек-NN»), исходный снимок. */
    static Src feiSrc(FarExport.Loaded export, Path exportDir, Set<String> keys, Common cm, SettingsStore settings) throws IOException {
        Src s = new Src();
        Path feiDir = Paths.get(settings.loadFacesFeiDir());
        Map<String, Path> raw = new HashMap<>();
        Map<String, String> hk = new HashMap<>();
        for (List<Sample> l : FeiDataset.originals(feiDir).values()) {
            for (Sample x : l) {
                String st = FarExport.stem(x.file());
                raw.put(st, x.file());
                hk.put(st, x.person() + "-" + String.format(Locale.ROOT, "%02d", FeiDataset.number(x)));
            }
        }
        Map<String, double[]> rows = XBaseMpPts.exportRows(exportDir, "fei");
        Map<String, LbfFrames.Lm> l = cm.lbf.get(cm.b.fei().name());
        for (List<Sample> list : export.data().fei().values()) {
            for (Sample x : list) {
                String k = x.file().toString();
                if (!keys.contains(k)) continue;
                String st = FarExport.stem(x.file());
                String h = hk.get(st);
                if (h == null) throw new IllegalStateException("abl: нет исходного снимка FEI для " + k);
                s.row.put(k, rows.get(x.person() + "/" + x.file().getFileName().toString()));
                s.haar.put(k, cm.b.fei().haar().get(h));
                s.lbf.put(k, l.get(h));
                Path f = raw.get(st);
                s.load.put(k, y -> Imgcodecs.imread(f.toString(), Imgcodecs.IMREAD_COLOR));
            }
        }
        return s;
    }

    // ---------------------------------------------------------------- GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-abl: только равные галереи FERET (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        boolean smoke = smoke();
        Common cm = common(dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("gt-abl: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rOld = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        g0.rLimit = rLimit;
        long eqSeed = g0.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g0.own, g0.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g0.own, g0.galleryAll, g0.seed);
        Map<String, List<String>> galIn = XBaseEval.equalize(g0.own, split.get(0), eqSeed);

        // (7): пробы U – все снимки с рамкой YuNet (отвергнутые отбором – тоже); роли и галереи – базовые.
        XBaseEval.Gt gUx = XBaseMp2.gtRoles(export, dir, feiKeys, settings, rOld::containsKey, rOld, r0);
        XBaseEval.Gt gU = copy(g0, g0.cohort, gUx.rOf, r0);
        gU.samples = gUx.samples;
        gU.byPerson = new TreeMap<>(g0.byPerson);
        for (String p : g0.ctrl) gU.byPerson.put(p, gUx.byPerson.get(p));
        for (String p : g0.own) gU.byPerson.put(p, gUx.byPerson.get(p));
        Map<String, String> keyOf = new HashMap<>();
        Map<String, Sample> sampleOf = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) {
            for (Sample s : l) {
                keyOf.put(s.person() + "/" + s.file().getFileName().toString(), s.file().toString());
                sampleOf.put(s.file().toString(), s);
            }
        }
        Map<String, Integer> sess = XBaseEval.sessions(dir.resolve(XBaseEval.SESSIONS));
        Map<String, List<String>> proXU = new LinkedHashMap<>();
        Map<String, List<String>> proInU = new LinkedHashMap<>();
        for (String p : g0.own) {
            List<String> x = new ArrayList<>(g0.probeKeys.get(p));
            List<String> in = new ArrayList<>(split.get(1).get(p));
            for (String pk : rOld.keySet()) {
                if (!pk.startsWith(p + "/") || rOld.get(pk) <= rLimit) continue;
                (sess.get(pk).equals(g0.galleryDateOf.get(p)) ? in : x).add(keyOf.get(pk));
            }
            x.sort(null);
            in.sort(null);
            proXU.put(p, x);
            proInU.put(p, in);
        }

        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        List<List<String>> cohort = new ArrayList<>(g0.cohort);
        Set<String> cohortKeys = new HashSet<>();
        cohort.forEach(cohortKeys::addAll);
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));
        List<List<String>> parts = FeiMethods.cvParts(g0.own, g0.seed, XBaseEval.CV_S);
        Map<String, String> old = XBaseSubsp.blocks(out.resolve("xbase_gt_subsp_scores.tsv.gz"));
        Object[][] modes = {{"межс.", galX, g0.probeKeys, proXU}, {"внутри", galIn, split.get(1), proInU}};
        List<Job> jobs = new ArrayList<>();
        for (Object[] md : modes) {
            String mode = (String) md[0];
            if (smoke && !mode.equals("межс.")) continue;
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> proU = (Map<String, List<String>>) md[3];
            for (int f = 0; f < 2; f++) {
                if (smoke && f > 0) continue;
                jobs.add(new Job(mode, f, 0, g0, gal, XBaseEval.probes(g0, pro), gU, XBaseEval.probes(gU, proU), parts, 0, cohort, poolCls,
                        old.get(mode + ", проекция " + CAND[f] + ", k = 0"), old.get(mode + ", проекция " + NOPOOL[f] + ", k = 0")));
            }
        }

        // Источник кадров (8), (9), (9б): GT (исходные снимки gt_db), когорта FEI, пул FERET (кэш derived).
        Src src = new Src();
        Map<String, LbfFrames.Lm> gl = cm.lbf.get(cm.b.gt().name());
        for (String k : gU.samples.keySet()) {
            Sample s = sampleOf.get(k);
            String pk = s.person() + "/" + s.file().getFileName().toString();
            src.row.put(k, det.get(pk));
            src.haar.put(k, cm.b.gt().haar().get(pk));
            src.lbf.put(k, gl.get(pk));
            src.load.put(k, y -> Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR));
        }
        Src fsrc = feretSrc(su, cm, poolKeys);
        Src feiSrc = feiSrc(export, dir, cohortKeys, cm, settings);
        Set<String> ok = new HashSet<>();
        Thr th = cm.th;

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (4а) поочерёдное отключение, GT (XBaseEval gt-abl)").append(smoke ? " — ПРОБНЫЙ (FACES_ABL_SMOKE)" : "").append('\n');
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, YuNet – detections.tsv; когорта – a/fei); снимки "
                + "GT – ").append(gtDir).append(" (кадры (8), (9), (9б) – в памяти); пул L1-xs-F – Color FERET (только обучение; имён снимков FERET нет)\n");
        header(r, th, g0, cm);
        r.append(String.format(Locale.ROOT, "Роли – gt-subsp (межс. N = 6, внутри N = 3, правило №8, seed faces.seed + %d); пул FERET – %d человек; люди бутстрепа – %d. (7): "
                + "пробы U – все снимки с рамкой YuNet: своих межс. %d, внутри %d (из них отвергнуто отбором базы %d / %d); контрольных %d (отвергнуто %d).%n",
                XBaseEval.EQ_SEED_OFFSET, pool.size(), g0.persons.size(), count(proXU), count(proInU), count(proXU) - count(g0.probeKeys),
                count(proInU) - count(split.get(1)), ctrlCount(gU), ctrlCount(gU) - ctrlCount(g0)));
        Src all = new Src();
        all.putAll(src);
        all.putAll(fsrc);
        all.putAll(feiSrc);
        for (String k : all.haar.keySet()) if (all.haar.get(k).found()) ok.add(k);
        boolean stop = drops(r, jobs, ok, all, cohort, poolCls);
        System.out.print(r);
        System.out.flush();
        String suffix = smoke ? "_smoke" : "";
        if (stop) {
            r.append(String.format(Locale.ROOT, "%nОСТАНОВКА: выпадение галереи своих в (9) больше %.0f %% или свой без галереи – счёт не выполнялся.%n", 100 * STOP_SHARE));
            FarMethods.write(out.resolve("xbase_gt_abl" + suffix + ".txt"), r);
            return;
        }
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-abl%s; seed %d; r0 %s; R %s; пул %d; k %s; d %s; R_H %s; r0_H %s; R_LBF %s; r0_LBF %s; %s",
                suffix, g0.seed, r0, rLimit, pool.size(), th.k(), th.d(), th.rH(), th.r0H(), th.rLbf(), th.r0Lbf(), cm.lbfSig));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_abl" + suffix), fp);
        Run run = new Run();
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        for (Sch sc : Sch.values()) {
            Map<String, Mat> fr = new LinkedHashMap<>();
            if (sc == Sch.BOX || sc == Sch.HAAR || sc == Sch.LBF) {
                int j = sc == Sch.BOX ? 0 : sc == Sch.HAAR ? 1 : 2;
                Map<String, Mat[]> f3 = new LinkedHashMap<>(frames3(src, new TreeSet<>(gU.samples.keySet()), th));
                f3.putAll(frames3(feiSrc, new TreeSet<>(cohortKeys), th));
                f3.putAll(frames3(fsrc, poolKeys, th));
                for (Map.Entry<String, Mat[]> e : f3.entrySet()) {
                    Mat m = e.getValue()[j];
                    fr.put(e.getKey(), m != null ? m : dummy());
                    for (int q = 0; q < 3; q++) if (q != j && e.getValue()[q] != null) e.getValue()[q].release();
                }
            } else {
                fr.putAll(XBaseEval.frames(export, gU.samples.keySet().stream().sorted().map(sampleOf::get).toList(), null));
                fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
                for (String k : poolKeys) fr.put(k, FeretData.frame(su.dir, "a", k));
            }
            XBaseEval[] x = {vectors(sc, 0, fr, g0.params), smoke ? null : vectors(sc, 1, fr, g0.params)};
            fr.values().forEach(Mat::release);
            scheme(sc, jobs, x, sc == Sch.LBF ? lbfOk(all) : ok, all, th, r0, cp, run, "gt-abl");
        }
        r.append("\n=== Проверка базы: строки L1 и +mirror – побайтно с файлом оценок gt-subsp (k = 0; пробы базы) ===\n").append(run.check);
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", run.fitS, g0.threads));
        StringBuilder sum = new StringBuilder(sumHead());
        tables(r, sum, run, g0.persons.size(), copyB(g0), smoke ? List.of(List.of("межс.")) : List.of(List.of("межс."), List.of("внутри")), "GT");
        XBaseOther.writeScores(out.resolve("xbase_gt_abl" + suffix + "_scores.tsv.gz"), "# GT, (4а) поочерёдное отключение (XBaseAbl gt-abl); блоки — схема, постановка, "
                + "кандидат, проход; пробы базы – U (с отвергнутыми отбором)\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных "
                + FarMethods.dataCommit(dir) + "\n", run.lines.toString());
        FarMethods.write(out.resolve("xbase_gt_abl" + suffix + SUMMARY), sum);
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; max куча JVM %d МБ.%n", (System.nanoTime() - t0) / 1e9, Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_gt_abl" + suffix + ".txt"), r);
        System.out.print(r);
        cp.delete();
    }

    static XBaseEval.Gt copyB(XBaseEval.Gt g) {
        return g;
    }

    static int count(Map<String, List<String>> m) {
        return m.values().stream().mapToInt(List::size).sum();
    }

    static int ctrlCount(XBaseEval.Gt g) {
        int n = 0;
        for (String p : g.ctrl) for (List<String> ks : g.byPerson.get(p).values()) n += ks.size();
        return n;
    }

    // ---------------------------------------------------------------- сводка

    /** Строки правила (основная строка каждого отключения) по порядку. */
    static final String[][] RULE = {
        {"(1)", "(1) без пула"}, {"(2)", "(2) без корня"}, {"(3)", "(3) χ² L0-mean"}, {"(4)", "(4) d₁ вместо d₁/d₂"}, {"(5)", "(5) без зеркал"},
        {"(6)", "(6) без выбора модели"}, {"(7)", "(7) без отбора по позе"}, {"(8)", "(8) кадр по рамке YuNet"}, {"(9)", "(9)(i) Хаар"}, {"(9б)", "(9б) Хаар + LBF"}};
    static final String[][] PROTOS = {{"FERET", "межс."}, {"FERET", "G2"}, {"GT", "межс."}, {"FERET", "внутри"}, {"GT", "внутри"}};

    /**
     * abl-summary: сводная таблица «отключено − база» (разделимость при FAR контроля 1 %, разность с интервалом) по кандидатам и
     * протоколам; правило «убрать» (заранее, PLAN.md п. 5.5): на основных протоколах (FERET 1 и 2 даты, GT межс.) точка ≤ 0 или
     * верхняя граница ≤ +3 п. п., внутри сессии (FERET, GT) верхняя граница ≤ +3 п. п.; справочные строки – без вывода.
     */
    static void summary(SettingsStore settings, Path out) throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("abl-summary: задайте faces.feret.dir / FACES_FERET_DIR");
        Path fdir = Paths.get(rawDir).resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Map<String, String[]> cells = new HashMap<>();
        List<String> heads = new ArrayList<>();
        for (Path f : new Path[] {fdir.resolve("xbase_feret_abl" + SUMMARY), out.resolve("xbase_gt_abl" + SUMMARY)}) {
            if (!Files.exists(f)) throw new IllegalStateException("abl-summary: нет " + f);
            for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
                if (l.startsWith("#")) continue;
                String[] c = l.split("\t", -1);
                cells.put(c[0] + "|" + c[1] + "|" + c[2] + "|" + c[3], c);
            }
            heads.add(f + " (" + FeiThreshold.sha(f) + ")");
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, (4а) поочерёдное отключение: сводная таблица «отключено − база» и правило «убрать» (XBaseEval abl-summary); "
                + "содержит сводные числа FERET – только локально\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        for (String h : heads) r.append("Сводка прогона: ").append(h).append('\n');
        r.append("Мера правила – ошибка своих при фактическом FAR контроля 1 % (разделимость), разность «отключено − база», п. п., [2,5; 97,5] – бутстреп по людям. "
                + "Правило (заранее): кандидат на удаление, если на FERET 1 и 2 даты и GT межс. точка ≤ 0 или верхняя граница ≤ +3, и внутри сессии (FERET, GT) "
                + "верхняя граница ≤ +3; иначе приём остаётся. Окончательно – Хозяин с учётом цены приёма.\n");
        for (String cand : CAND) {
            r.append("\n=== ").append(cand).append(" ===\n");
            r.append("приём | строка | FERET межс. (1 дата) | FERET G2 (2 даты) | GT межс. | FERET внутри | GT внутри | вывод\n");
            for (String[] rule : RULE) {
                List<String> row = new ArrayList<>(List.of(rule[0], rule[1]));
                boolean all = true;
                boolean any = false;
                for (int p = 0; p < PROTOS.length; p++) {
                    String[] c = cells.get(PROTOS[p][0] + "|" + PROTOS[p][1] + "|" + cand + "|" + rule[1]);
                    if (c == null) {
                        row.add("–");
                        continue;
                    }
                    any = true;
                    double pt = Double.parseDouble(c[5]);
                    double hi = c[7].isEmpty() ? pt : Double.parseDouble(c[7]);
                    boolean okP = p < 3 ? (pt <= 0 || hi <= 3) : hi <= 3;
                    if (!okP) all = false;
                    boolean sig = !c[6].isEmpty() && (Double.parseDouble(c[6]) > 0 || hi < 0);
                    row.add(String.format(Locale.ROOT, "%s [%s; %s]%s (%s → %s)", fmt(c[5]), fmt(c[6]), fmt(c[7]), sig ? " *" : "", c[16], c[15]));
                }
                if (rule[0].equals("(7)")) {
                    row.add("данными не проверяется (FERET – отбор ничего не отвергает, GT – только отвергнутые пробы); в ПО остаётся как часть контроля качества пробы (3)");
                } else {
                    row.add(!any ? "не проверяется (на наших протоколах пусто)" : all ? "кандидат на удаление" : "остаётся");
                }
                r.append(String.join(" | ", row)).append('\n');
            }
            r.append("Справочно (без вывода):\n");
            for (String ref : new String[] {"(3) χ² L0 (справочно)", "(9)(ii) Хаар, глаза по рамке", "(9)(i) Хаар + отбор и модель по r_H (справочно)",
                "(9)(ii) Хаар [Хаар: лицо и два глаза]"}) {
                List<String> row = new ArrayList<>(List.of("", ref));
                for (String[] pr : PROTOS) {
                    String[] c = cells.get(pr[0] + "|" + pr[1] + "|" + cand + "|" + ref);
                    row.add(c == null ? "–" : String.format(Locale.ROOT, "%s [%s; %s] (%s → %s)", fmt(c[5]), fmt(c[6]), fmt(c[7]), c[16], c[15]));
                }
                r.append(String.join(" | ", row)).append('\n');
            }
        }
        r.append("\nВ скобках – ошибка своих при FAR контроля 1 %: база → отключено; «*» – интервал разности не содержит 0. Строки (9)(i) и (9б) – пропуск лица и (для (i)) "
                + "отсутствие двух глаз – отказ; (9б) – основная строка режима «только классика».\n");
        r.append("(7): пара – «база (а)» на пробах U (все снимки с рамкой YuNet, включая отвергнутые отбором по позе: такая проба своего – отказ, "
                + "контрольная – не принята); на GT в U контрольных снимков 300 вместо 295 (порог разделимости по контрольным чужим – другой), внутри "
                + "сессии ещё 3 пробы своих – поэтому ошибка базы в скобках строки (7) отличается от основной базы (пробы I).\n");
        FarMethods.write(fdir.resolve("xbase_abl_summary.txt"), r);
        System.out.print(r);
    }

    static String fmt(String v) {
        if (v.isEmpty()) return "–";
        double d = Double.parseDouble(v);
        return String.format(Locale.ROOT, "%+.1f", d);
    }
}
