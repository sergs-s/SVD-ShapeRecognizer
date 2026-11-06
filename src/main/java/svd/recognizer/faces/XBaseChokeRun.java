package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.XBaseChoke.Key;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.storage.SettingsStore;

/**
 * ChokePoint, итоговая проверка (постановка Жорика 06.11, редакция 2; уточнения №8 – TASKS.md). cp-run: выбор кадра прохода (YuNet в два
 * прохода, наибольшая рамка), кадры Б, W и модели частей заново в каждом проходе, строки 1–5, меры разд. 5, итог разд. 6. Всё производное –
 * только D:\data\ChokePoint\ (derived\, reports\). FACES_CP_TRIAL – пробный режим на одном проходе без оценок; FACES_CP_DRY=1 – сухой прогон
 * всего пути и отчёта на GT, ChokePoint не читается.
 *
 * @author ssv
 */
final class XBaseChokeRun {

    private XBaseChokeRun() {
    }

    static final String KEY = "cp:";
    static final String CACHE = "cp_detections.tsv";
    static final String MULTI = "cp_multi.tsv";
    static final double REF_ALL = 25.9;
    static final double REF_28 = 18.3;
    static final double REF_45 = 45.3;
    static final XBaseFinal.Cfg XST = new XBaseFinal.Cfg("xst", "L1-xs-F (Т+М)", 0, 1.0);

    private static final Pattern FRAME = Pattern.compile("<frame number=\"(\\d+)\"");
    private static final Pattern PERSON = Pattern.compile("<person id=\"(\\d+)\"");
    private static final Pattern XY = Pattern.compile("x=\"(-?\\d+)\" y=\"(-?\\d+)\"");

    /** Кандидат прохода: номер кадра, |r|, межглазье YuNet. */
    record Cand(int frame, double absR, double iod) {}

    /** Порядок выбора (постановка, разд. 3): |r| ≤ r0 – по убыванию межглазья; затем r0 &lt; |r| ≤ R – по возрастанию |r|; при равенстве – меньший номер кадра. */
    static List<Cand> order(List<Cand> c, double r0, double rLimit) {
        List<Cand> a = new ArrayList<>();
        List<Cand> b = new ArrayList<>();
        for (Cand x : c) {
            if (x.absR() <= r0) a.add(x);
            else if (x.absR() <= rLimit) b.add(x);
        }
        a.sort(java.util.Comparator.comparingDouble((Cand x) -> -x.iod()).thenComparingInt(Cand::frame));
        b.sort(java.util.Comparator.comparingDouble(Cand::absR).thenComparingInt(Cand::frame));
        a.addAll(b);
        return a;
    }

    /** Порядок источников кадра галереи (Жорик №8, п. 4): P1E своей S, P1L той же S, затем P1L следующих S по кругу. */
    static List<String[]> galleryOrder(int s) {
        List<String[]> out = new ArrayList<>();
        out.add(new String[] {"P1E", Integer.toString(s)});
        for (int i = 0; i < 4; i++) out.add(new String[] {"P1L", Integer.toString((s - 1 + i) % 4 + 1)});
        return out;
    }

    /** Части «чужих» (Т2) прохода pass (1, 2): Random(faces.seed + 12010 + pass), деление – XBaseOther.cvParts. */
    static List<List<String>> parts(List<String> own, long seed, int pass) {
        return XBaseOther.cvParts(own, seed + 12010 + pass - FeiMethods.CV_SEED_OFFSET, 0);
    }

    /** Отказ захвата своего: попытка с +∞ и неверным кандидатом (ошибка при любом пороге). */
    static void addFail(FeiDecompose.Cfg c, int person) {
        int n = c.ownP.length;
        c.ownP = Arrays.copyOf(c.ownP, n + 1);
        c.ownR = Arrays.copyOf(c.ownR, n + 1);
        c.ownOk = Arrays.copyOf(c.ownOk, n + 1);
        c.noSecond = Arrays.copyOf(c.noSecond, n + 1);
        c.trueInf = Arrays.copyOf(c.trueInf, n + 1);
        c.poseFrontal = Arrays.copyOf(c.poseFrontal, n + 1);
        c.ownP[n] = person;
        c.ownR[n] = Double.POSITIVE_INFINITY;
        c.trueInf[n] = true;
    }

    /** Дата DateTimeOriginal из EXIF (ГГГГ:ММ:ДД) или null. */
    static String exifDate(byte[] b) {
        int pos = 2;
        while (pos + 4 < b.length && (b[pos] & 0xFF) == 0xFF) {
            int marker = b[pos + 1] & 0xFF;
            int len = ((b[pos + 2] & 0xFF) << 8) | (b[pos + 3] & 0xFF);
            if (marker == 0xE1 && new String(b, pos + 4, 4, StandardCharsets.US_ASCII).equals("Exif")) {
                int base = pos + 10;
                boolean le = b[base] == 'I';
                int ifd0 = base + u32(b, base + 4, le);
                int exif = -1;
                int n = u16(b, ifd0, le);
                for (int i = 0; i < n; i++) {
                    int e = ifd0 + 2 + 12 * i;
                    if (u16(b, e, le) == 0x8769) exif = base + u32(b, e + 8, le);
                }
                if (exif < 0) return null;
                n = u16(b, exif, le);
                for (int i = 0; i < n; i++) {
                    int e = exif + 2 + 12 * i;
                    if (u16(b, e, le) == 0x9003) return new String(b, base + u32(b, e + 8, le), 10, StandardCharsets.US_ASCII);
                }
                return null;
            }
            pos += 2 + len;
        }
        return null;
    }

    private static int u16(byte[] b, int p, boolean le) {
        return le ? (b[p] & 0xFF) | (b[p + 1] & 0xFF) << 8 : (b[p] & 0xFF) << 8 | (b[p + 1] & 0xFF);
    }

    private static int u32(byte[] b, int p, boolean le) {
        return le ? u16(b, p, true) | u16(b, p + 2, true) << 16 : u16(b, p, false) << 16 | u16(b, p + 2, false);
    }

    /** Наибольшая рамка (постановка, разд. 3); при равенстве – первая. */
    private static int largest(Mat faces) {
        int best = 0;
        for (int i = 1; i < faces.rows(); i++) if (faces.get(i, 2)[0] * faces.get(i, 3)[0] > faces.get(best, 2)[0] * faces.get(best, 3)[0]) best = i;
        return best;
    }

    /** Два прохода YuNet, как YunetTwoPass, но при нескольких лицах – наибольшая рамка; multi[0] – несколько лиц в первом проходе. */
    static double[] detect(FaceDetectorYN d, Mat color, Mat input, double scale, boolean[] multi) {
        d.setInputSize(input.size());
        Mat faces = new Mat();
        d.detect(input, faces);
        if (faces.rows() == 0) return null;
        multi[0] = faces.rows() > 1;
        int best = largest(faces);
        double[] first = new double[15];
        for (int j = 0; j < 15; j++) first[j] = j == 14 ? faces.get(best, j)[0] : faces.get(best, j)[0] / scale;
        faces.release();
        int x0 = (int) Math.max(0, Math.floor(first[0] - YunetTwoPass.MARGIN * first[2]));
        int y0 = (int) Math.max(0, Math.floor(first[1] - YunetTwoPass.MARGIN * first[3]));
        int x1 = (int) Math.min(color.cols(), Math.ceil(first[0] + first[2] + YunetTwoPass.MARGIN * first[2]));
        int y1 = (int) Math.min(color.rows(), Math.ceil(first[1] + first[3] + YunetTwoPass.MARGIN * first[3]));
        double f = YunetTwoPass.FACE / first[2];
        Mat crop = new Mat();
        Mat roi = new Mat(color, new Rect(x0, y0, x1 - x0, y1 - y0));
        Imgproc.resize(roi, crop, new Size(Math.round((x1 - x0) * f), Math.round((y1 - y0) * f)), 0, 0, f < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        d.setInputSize(crop.size());
        Mat fc = new Mat();
        d.detect(crop, fc);
        double[] out = first;
        if (fc.rows() > 0) {
            int b2 = largest(fc);
            out = new double[15];
            for (int j = 0; j < 14; j++) out[j] = j == 2 || j == 3 ? fc.get(b2, j)[0] / f : (j % 2 == 0 ? x0 : y0) + fc.get(b2, j)[0] / f;
            out[14] = fc.get(b2, 14)[0];
        }
        fc.release();
        crop.release();
        roi.release();
        return out;
    }

    static boolean alignable(double[] row) {
        try {
            double[][] lm = FrameK2.lm(row);
            Mat m = FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(FrameB.W, FrameB.H));
            boolean ok = !m.empty();
            m.release();
            if (!ok) return false;
            Mat k = FaceAlignment.similarity(lm, FrameK2.template());
            ok = !k.empty();
            k.release();
            return ok;
        } catch (org.opencv.core.CvException e) {
            return false;
        }
    }

    static double iod(double[] row) {
        return Math.hypot(row[6] - row[4], row[7] - row[5]);
    }

    /** Попытка портала 2: key == null – отказ захвата. */
    record Att(String person, String group, String dir, int s, String part, String key, double iod) {}

    static final class Plan {
        List<String> all = new ArrayList<>();
        List<List<String>> halves;
        List<String> only2;
        Map<String, List<String>> gal = new TreeMap<>();
        Map<String, List<String>> still = new TreeMap<>();
        /** Дата самого позднего снимка шаблона still (ГГГГ:ММ:ДД); нет даты – нет записи. */
        Map<String, String> stillDate = new TreeMap<>();
        List<Att> atts = new ArrayList<>();
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, FrameK2.F> k2 = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        StringBuilder stats = new StringBuilder();
        Path dir;
    }

    /** Чтение кадров и детекции с кэшем; кадры выбранных – в derived. */
    static final class Frames {
        final Path root;
        final Path der;
        final Map<String, double[]> rows;
        final Set<String> multi = new HashSet<>();
        final FaceDetectorYN det;
        final FaceRecognizerSF sface;
        final Writer cache;
        final Writer multiOut;
        final StringBuilder pts = new StringBuilder();
        long computed;
        long found;
        long total;

        Frames(Path root, SettingsStore settings) throws IOException {
            this.root = root;
            der = root.resolve("derived");
            Files.createDirectories(der);
            Path cf = der.resolve(CACHE);
            rows = Files.exists(cf) ? VidTimitData.readDetections(cf) : new HashMap<>();
            Path mf = der.resolve(MULTI);
            if (Files.exists(mf)) multi.addAll(Files.readAllLines(mf, StandardCharsets.UTF_8));
            Path models = Paths.get(settings.loadFacesModelsDir());
            det = OwnDetectorThreshold.create(models, FeretData.SCORE);
            sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
            cache = Files.newBufferedWriter(cf, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            multiOut = Files.newBufferedWriter(mf, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            for (Path p : List.of(der.resolve(FrameB.DIR), der.resolve(FrameK2.DIR), der.resolve("sface"))) Files.createDirectories(p);
        }

        double[] row(String key, Path file, double scale) throws IOException {
            total++;
            if (rows.containsKey(key)) {
                double[] r = rows.get(key);
                if (r != null) found++;
                return r;
            }
            Mat color = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_COLOR);
            if (color.empty()) throw new IOException("ChokePoint: нет кадра " + file);
            Mat input = color;
            if (scale != 1.0) {
                input = new Mat();
                Imgproc.resize(color, input, new Size(Math.round(color.cols() * scale), Math.round(color.rows() * scale)), 0, 0, Imgproc.INTER_AREA);
            }
            boolean[] m = new boolean[1];
            double[] r = detect(det, color, input, scale, m);
            if (input != color) input.release();
            color.release();
            StringBuilder l = new StringBuilder(key).append('\t').append(r == null ? "0" : "1");
            for (int j = 0; j < 15; j++) l.append('\t').append(r == null ? "" : Double.toString(r[j]));
            cache.write(l.append('\n').toString());
            cache.flush();
            if (m[0]) {
                multi.add(key);
                multiOut.write(key + "\n");
                multiOut.flush();
            }
            rows.put(key, r);
            computed++;
            if (r != null) found++;
            return r;
        }

        void make(Plan p, String key, Path file, double[] row) throws IOException {
            Mat color = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_COLOR);
            if (color.empty()) throw new IOException("ChokePoint: нет кадра " + file);
            String name = key.substring(KEY.length()).replace('/', '_').replace('.', '_') + ".png";
            Mat b = FrameB.frame(color, row, true);
            Mat gray = new Mat();
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
            FrameK2.F f = FrameK2.fromFull(gray, FrameK2.lm(row));
            Mat s = FeretData.frameSface(color, row, sface);
            FeretData.writePng(der.resolve(FrameB.DIR).resolve(name), b);
            FeretData.writePng(der.resolve(FrameK2.DIR).resolve(name), f.frame());
            FeretData.writePng(der.resolve("sface").resolve(name), s);
            pts.append(FrameK2.pointsLine(key, f.pts()));
            p.fr.put(key, b);
            p.k2.put(key, f);
            p.sf.put(key, s);
            p.rOf.put(key, Math.abs(FeiDataset.pose(row)));
            gray.release();
            color.release();
        }

        void close() throws IOException {
            cache.close();
            multiOut.close();
        }
    }

    static String detKey(Key k, int cam, int frame) {
        return XBaseChoke.seqName(k.dir(), k.s(), cam, k.part()) + "/" + String.format(Locale.ROOT, "%08d", frame);
    }

    static Path framePath(Path root, Key k, int cam, int frame) {
        return root.resolve("frames").resolve(XBaseChoke.seqName(k.dir(), k.s(), cam, k.part())).resolve(String.format(Locale.ROOT, "%08d.jpg", frame));
    }

    /** Выбор кадра прохода: все кадры камеры анфаса через кэш YuNet; годные к выравниванию и не использованные. Возвращает ключ или null. */
    static String pick(Frames fx, Plan p, Key k, List<Integer> frames, double r0, double rLimit, Set<String> used, int[] noAlign) throws IOException {
        int cam = XBaseChoke.FRONTAL.get(k.dir())[k.s() - 1];
        List<Cand> cs = new ArrayList<>();
        Map<Integer, double[]> rows = new HashMap<>();
        for (int f : frames) {
            double[] r = fx.row(detKey(k, cam, f), framePath(fx.root, k, cam, f), 1.0);
            if (r == null) continue;
            rows.put(f, r);
            cs.add(new Cand(f, Math.abs(FeiDataset.pose(r)), iod(r)));
        }
        for (Cand c : order(cs, r0, rLimit)) {
            String key = KEY + detKey(k, cam, c.frame());
            if (used.contains(key)) continue;
            if (!alignable(rows.get(c.frame()))) {
                noAlign[0]++;
                continue;
            }
            used.add(key);
            fx.make(p, key, framePath(fx.root, k, cam, c.frame()), rows.get(c.frame()));
            return key;
        }
        return null;
    }

    /** Общая часть обучения: когорта FEI и пул FERET; кадры Б в p.fr, кадры k = 2 в p.k2. */
    static List<List<String>> train(Plan p, XBaseA0.FeretSetup su, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, SettingsStore settings, long seed, int poolMax)
            throws IOException {
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        if (pool.containsKey("00700")) throw new IllegalStateException("ChokePoint: FERET 00700 в пуле – решение (38)");
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (poolMax > 0) poolCls = poolCls.subList(0, Math.min(poolMax, poolCls.size()));
        List<String> cp = XBaseEval.cohortPersons(feiKeys, seed);
        if (cp.contains("071")) throw new IllegalStateException("ChokePoint: FEI 071 в когорте – решение (38)");
        List<List<String>> cohort = new ArrayList<>();
        for (String q : cp) cohort.add(new ArrayList<>(feiKeys.get(q).values()));
        List<List<String>> train = new ArrayList<>(cohort);
        train.addAll(poolCls);
        List<GalleryEvaluation.Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        Set<String> ck = new HashSet<>();
        cohort.forEach(ck::addAll);
        Map<String, Mat> tf = new LinkedHashMap<>(XBaseEval.frames(export, fei, ck));
        for (List<String> c : poolCls) for (String k : c) if (!tf.containsKey(k)) tf.put(k, FeretData.frame(su.dir, "a", k));
        Map<String, FrameK2.F> k2 = FrameK2.load(tf.keySet(), su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        p.fr.putAll(tf);
        p.k2.putAll(k2);
        return train;
    }

    /** Настройка блоков: основная галерея (проход 1, 2) и галерея still; W и модели частей заново в каждом блоке. */
    static XBaseFinal.Run setup(Plan p, XBaseA0.FeretSetup su, List<List<String>> train, SettingsStore settings, double r0, double rLimit) {
        long seed = settings.loadFacesSeed();
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < p.all.size(); i++) pIdx.put(p.all.get(i), i);
        Map<String, Integer> numOf = new HashMap<>();
        for (String k : new TreeSet<>(p.rOf.keySet())) numOf.put(k, numOf.size());
        XBaseVec.Setup u = new XBaseVec.Setup();
        u.base = "cp";
        u.su = su;
        u.persons = p.all.size();
        u.fr = p.fr;
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.seed = seed;
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.params = IlluminationNorm.NormParams.of(settings);
        tpl.r0 = r0;
        tpl.rLimit = rLimit;
        tpl.persons = p.all;
        tpl.pIdx = pIdx;
        tpl.imageNo = numOf;
        tpl.rOf = p.rOf;
        Map<String, List<String>> proAll = new TreeMap<>();
        for (String q : p.all) proAll.put(q, new ArrayList<>());
        for (Att a : p.atts) if (a.key() != null) proAll.get(a.person()).add(a.key());
        proAll.forEach((q, ks) -> tpl.byPerson.put(q, new TreeMap<>(Map.of(1, ks))));
        u.tpl = tpl;
        u.groups.add("основная");
        u.groups.add("still");
        for (int kind = 0; kind < 2; kind++) {
            for (int pass = 0; pass < 2; pass++) {
                List<String> own = new ArrayList<>(p.halves.get(pass));
                List<String> ctrl = new ArrayList<>(p.halves.get(1 - pass));
                ctrl.addAll(p.only2);
                Map<String, List<String>> gs = kind == 0 ? p.gal : p.still;
                own.removeIf(q -> gs.get(q) == null || gs.get(q).isEmpty());
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String q : own) {
                    gal.put(q, gs.get(q));
                    List<String> l = new ArrayList<>();
                    for (Att a : p.atts) if (a.person().equals(q) && a.key() != null && (kind == 0 || laterThanStill(p, a))) l.add(a.key());
                    pro.put(q, l);
                }
                XBaseEval.Gt g = XBaseAbl.copy(tpl, train, p.rOf, r0);
                g.own = own;
                g.ctrl = ctrl;
                g.persons = p.all;
                g.pIdx = pIdx;
                g.imageNo = numOf;
                List<List<String>> ps = parts(own, seed, pass + 1);
                List<Probe> probes = XBaseEval.probes(g, pro);
                String name = (kind == 0 ? "p" : "still_p") + pass;
                String grp = kind == 0 ? "основная" : "still";
                u.blocks.add(new XBaseVec.Blk(name, grp + ", %s, проход " + (pass + 1) + ", ", g, gal, pro, ps, probes, XBaseThr.extra(g, pro, ps), kind * 10 + pass + 1, grp));
            }
        }
        return new XBaseFinal.Run(u, p.k2, Map.of(), p.dir, false, 0);
    }

    static String attDate(Att a) {
        return a.part().equals("1") ? "2010:05:14" : "2010:05:31";
    }

    /** Проба строки 5: позже даты портрета; нет даты EXIF – засчитывается (Жорик №8, п. 3). */
    static boolean laterThanStill(Plan p, Att a) {
        String d = p.stillDate.get(a.person());
        return d == null || attDate(a).compareTo(d) > 0;
    }

    /** Исход попытки при пороге: человек, проход, свой ли, принят ли, верный ли лучший, попытка. */
    record Res(String person, int pass, boolean own, boolean acc, boolean right, Att att) {}

    static List<Res> atT(XBaseVec.Blk b, int pass, XBaseFinal.Run r, double t, Plan p, boolean stillKind) throws IOException {
        List<XBaseVec.Model> mt = XBaseFinal.models(r, b, XBaseFinal.TT1, false);
        List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
        double[] st = XBaseVec.fuseStats(mt, false);
        double[] sx = XBaseVec.fuseStats(mx, false);
        List<XBaseFinalReport.Out1> o = XBaseFinalReport.atT(mt, mx, st, sx, b.g().rOf, b.g().r0, t, ro -> ro.equals(FeiScores.OWN) || ro.equals(FeiScores.CTRL));
        Map<String, Att> byKey = new HashMap<>();
        for (Att a : p.atts) if (a.key() != null) byKey.put(a.key(), a);
        List<Res> out = new ArrayList<>();
        int i = 0;
        for (XBaseVec.Row row : XBaseVec.find(mt, "main", false).rows()) {
            if (!(row.role().equals(FeiScores.OWN) || row.role().equals(FeiScores.CTRL))) continue;
            XBaseFinalReport.Out1 x = o.get(i++);
            out.add(new Res(x.person(), pass, x.own(), x.accepted(), x.right(), byKey.get(row.key())));
        }
        addFails(out, b, pass, p, stillKind);
        return out;
    }

    /** Отказы своих: нет кадра прохода; в строке 5 – ещё люди без шаблона (все их засчитанные пробы). */
    static void addFails(List<Res> out, XBaseVec.Blk b, int pass, Plan p, boolean stillKind) {
        for (String q : p.halves.get(pass)) {
            boolean noTpl = stillKind && !b.g().own.contains(q);
            for (Att a : p.atts) {
                if (!a.person().equals(q) || stillKind && !laterThanStill(p, a)) continue;
                if (a.key() == null || noTpl) out.add(new Res(q, pass, true, false, false, a));
            }
        }
    }

    /** Ошибка своих по людям (среднее долей ошибочных проходов человека) при весах w. */
    static double errPersons(List<Res> l, Map<String, Integer> pIdx, int[] w) {
        Map<String, double[]> by = new TreeMap<>();
        for (Res x : l) {
            if (!x.own()) continue;
            double[] c = by.computeIfAbsent(x.person(), k -> new double[2]);
            c[1]++;
            if (!(x.acc() && x.right())) c[0]++;
        }
        double s = 0;
        double n = 0;
        for (Map.Entry<String, double[]> e : by.entrySet()) {
            int wt = w[pIdx.get(e.getKey())];
            s += wt * e.getValue()[0] / e.getValue()[1];
            n += wt;
        }
        return n == 0 ? Double.NaN : 100 * s / n;
    }

    static double errAtt(List<Res> l, Map<String, Integer> pIdx, int[] w) {
        double e = 0;
        double n = 0;
        for (Res x : l) {
            if (!x.own()) continue;
            n += w[pIdx.get(x.person())];
            if (!(x.acc() && x.right())) e += w[pIdx.get(x.person())];
        }
        return n == 0 ? Double.NaN : 100 * e / n;
    }

    static double farAtt(List<Res> l, Map<String, Integer> pIdx, int[] w) {
        double e = 0;
        double n = 0;
        for (Res x : l) {
            if (x.own()) continue;
            n += w[pIdx.get(x.person())];
            if (x.acc()) e += w[pIdx.get(x.person())];
        }
        return n == 0 ? Double.NaN : 100 * e / n;
    }

    static double rightShare(List<Res> l) {
        long n = l.stream().filter(Res::own).count();
        return n == 0 ? Double.NaN : 100.0 * l.stream().filter(x -> x.own() && x.right()).count() / n;
    }

    static double[] reps(java.util.function.ToDoubleFunction<int[]> f, int[][] w) {
        double[] r = new double[w.length];
        for (int q = 0; q < w.length; q++) r[q] = f.applyAsDouble(w[q]);
        return r;
    }

    static String interval(double[] reps) {
        double[] v = Arrays.stream(reps).filter(x -> !Double.isNaN(x)).toArray();
        if (v.length == 0) return "–";
        double[] d = FeiBootstrap.sorted(v);
        return String.format(Locale.ROOT, "[%.1f; %.1f]", FeiBootstrap.quantile(d, 0.025), FeiBootstrap.quantile(d, 0.975));
    }

    /** Сводка попыток без столбца «держит»: по постановке (разд. 6) порог держит при FAR по попыткам ≤ 1 % в точке – в итоге отчёта. */
    static String summaryNoHold(List<XBaseFinalReport.Out1> o) {
        String s = XBaseFinalReport.summary(o);
        return s.substring(0, s.lastIndexOf(" | "));
    }

    /** Строка главных мер при рабочем пороге; чужой по людям – человек в своём проходе (33 единицы за два прохода). */
    static String mainRow(String name, List<Res> l, Map<String, Integer> pIdx, int[][] w, int persons) {
        List<XBaseFinalReport.Out1> o = new ArrayList<>();
        for (Res x : l) o.add(new XBaseFinalReport.Out1(x.own() ? FeiScores.OWN : FeiScores.CTRL, x.person() + (x.own() ? "" : "#" + (x.pass() + 1)), x.own(), x.acc(), x.right()));
        int[] unit = FeiBootstrap.unit(persons);
        double far = farAtt(l, pIdx, unit);
        return String.format(Locale.ROOT, "%s | %s | свои по людям %.1f %% %s | свои по попыткам %.1f %% %s | FAR по попыткам %s %s | верный лучший %.1f %%%n", name, summaryNoHold(o), errPersons(l, pIdx, unit),
                interval(reps(a -> errPersons(l, pIdx, a), w)), errAtt(l, pIdx, unit), interval(reps(a -> errAtt(l, pIdx, a), w)), Double.isNaN(far) ? "–" : String.format(Locale.ROOT, "%.2f %%", far),
                interval(reps(a -> farAtt(l, pIdx, a), w)), rightShare(l));
    }

    static void breakdown(StringBuilder t, String name, List<Res> l, Map<String, Integer> pIdx, int[][] w, int persons) {
        Map<String, java.util.function.Predicate<Res>> by = new LinkedHashMap<>();
        by.put("свои 28 дней; все чужие", x -> !x.own() || x.att().group().equals("28"));
        by.put("свои 45 дней; все чужие", x -> !x.own() || x.att().group().equals("45"));
        by.put("проба E (P2E, улица)", x -> x.att().dir().equals("P2E"));
        by.put("проба L (P2L, помещение)", x -> x.att().dir().equals("P2L"));
        by.put("межглазье < 35 px", x -> x.att().key() != null && x.att().iod() < 35);
        by.put("межглазье ≥ 35 px", x -> x.att().key() != null && x.att().iod() >= 35);
        for (Map.Entry<String, java.util.function.Predicate<Res>> e : by.entrySet()) t.append(mainRow(name + " | " + e.getKey(), l.stream().filter(e.getValue()).toList(), pIdx, w, persons));
    }

    /** Итог разд. 6: интервал ошибки своих по людям против GT межс. */
    static String verdict(double point, double lo, double hi) {
        String word = lo > REF_ALL ? "хуже ожидаемого" : hi < REF_ALL ? "лучше ожидаемого" : "согласуется";
        return String.format(Locale.ROOT, "ошибка своих по людям %.1f %% [%.1f; %.1f] против GT межс. %.1f %%: %s", point, lo, hi, REF_ALL, word);
    }

    /** Строка мер без порога: разделимость при фактическом FAR 1 % и 5 %, EER, pAUC. */
    static String sepRow(String name, XBaseFinalReport.M m) {
        XBaseRule.Sep s = m.sep;
        return String.format(Locale.ROOT, "%s | FAR 1 %%: %s (факт. %.2f %%) | FAR 5 %%: %s (факт. %.2f %%) | EER %s | pAUC %s%n", name, XBaseSepLtp.cell(s.err[0], s.errRep[0]), s.far[0], XBaseSepLtp.cell(s.err[2], s.errRep[2]), s.far[2],
                XBaseSepLtp.cell(s.eer[3], s.eerRep), XBaseSepLtp.cell(m.pauc, m.paucRep));
    }

    static double[] diff(double[] a, double[] b) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) d[i] = a[i] - b[i];
        return d;
    }

    /** Конфигурации с отказами своих. */
    static List<FeiDecompose.Cfg> withFails(List<FeiDecompose.Cfg> cs, List<XBaseVec.Blk> bs, Plan p, boolean stillKind, Map<String, Integer> pIdx) {
        for (int i = 0; i < cs.size(); i++) {
            List<Res> l = new ArrayList<>();
            addFails(l, bs.get(i), i, p, stillKind);
            for (Res x : l) addFail(cs.get(i), pIdx.get(x.person()));
        }
        return cs;
    }

    /** Модели конфигурации Т+М основной галереи (вход L1-xs-F – Тан – Триггс, как вариант 19). */
    static double computeXst(XBaseFinal.Run r, List<XBaseVec.Blk> main) throws IOException {
        XBaseVec.Setup u = r.u();
        String commit = FarMethods.commit();
        List<XBaseVec.Blk> todo = new ArrayList<>();
        for (XBaseVec.Blk b : main) {
            Path f = XBaseFinal.vfile(r, b, XST);
            if (Files.exists(f)) {
                if (!XBaseVec.readHead(f).contains(commit)) throw new IllegalStateException("ChokePoint: " + f + " – другой коммит кода");
                continue;
            }
            todo.add(b);
        }
        if (todo.isEmpty()) return 0;
        long t0 = System.nanoTime();
        FeatP4.P b0 = XBaseFinal.param(XBaseFinal.TT1, u.tpl.params);
        FeatP4.P pv = new FeatP4.P(b0.inTt(), FeatP4.In.TT, b0.tt(), b0.orient(), b0.cell(), b0.radius(), b0.scales(), b0.patch(), b0.msWeight());
        XBaseEval x = XBasePart4.vectorsB(u.fr, r.k2(), pv, 0, u.tpl.threads);
        for (XBaseVec.Blk b : todo) {
            List<List<java.util.concurrent.Callable<XBaseVec.Model>>> jobs = List.of(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
            List<XBaseVec.Model> done = XBaseVec.run(jobs, u.tpl.threads).get(0);
            Path f = XBaseFinal.vfile(r, b, XST);
            Writer w = XBaseVec.writer(f, "# Векторы 1 − cos (XBaseChokeRun): " + XST.name() + ", кадры Б\n# Код: коммит " + commit + "\n");
            XBaseVec.write(w, String.format(b.abl2(), XBaseRule.FEATS[0]) + XST.name(), done);
            XBaseVec.finish(w, f);
            System.out.println("cp: xst " + b.name());
        }
        return (System.nanoTime() - t0) / 1e9;
    }

    /** Полный путь по плану: модели, строки 1–5, отчёт. Пишет только в p.dir. */
    static void evaluate(Plan p, XBaseFinal.Run r, Map<Integer, XBasePart4.TRow> tb37, Map<Integer, XBasePart4.TRow> tbOld, SettingsStore settings, String head) throws IOException {
        long t0 = System.nanoTime();
        XBaseVec.Setup u = r.u();
        Map<String, Double> sec = XBaseFinal.compute(r);
        List<XBaseVec.Blk> main = u.blocks.stream().filter(b -> b.group().equals("основная")).toList();
        List<XBaseVec.Blk> still = u.blocks.stream().filter(b -> b.group().equals("still")).toList();
        sec.put("xst", computeXst(r, main));
        for (Mat m : p.fr.values()) m.release();
        for (FrameK2.F k : p.k2.values()) k.frame().release();
        long ts = System.nanoTime();
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        xsf.sfaceVectors(p.sf, FaceRecognizerSF.create(Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE).toString(), ""));
        for (Mat m : p.sf.values()) m.release();
        sec.put("SFace", (System.nanoTime() - ts) / 1e9);
        Map<String, Integer> pIdx = u.tpl.pIdx;
        int persons = u.persons;
        int[] unit = FeiBootstrap.unit(persons);
        int[][] w = u.tpl.b > 0 ? FeiBootstrap.weights(persons, u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] clamped = new int[2];
        StringBuilder sc = new StringBuilder("# Файл оценок ChokePoint (XBaseChokeRun); строки – формат FeiScores\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder t = new StringBuilder(head);
        t.append("Код: коммит ").append(FarMethods.commit()).append("; модели – ").append(r.dir().resolve("vec")).append(String.format(Locale.ROOT, "; бутстреп по людям B = %d, seed %d; людей %d%n", u.tpl.b, u.tpl.seed + FeiBootstrap.SEED_OFFSET, persons));
        t.append(p.stats);
        List<Res> r37 = new ArrayList<>();
        List<Res> rOld = new ArrayList<>();
        List<Res> rSf = new ArrayList<>();
        List<Res> rStill = new ArrayList<>();
        StringBuilder nodes = new StringBuilder();
        List<FeiDecompose.Cfg> csSf = new ArrayList<>();
        Map<Integer, Att> byImg = new HashMap<>();
        for (Att a : p.atts) if (a.key() != null) byImg.put(u.tpl.imageNo.get(a.key()), a);
        List<XBaseThr.Unit> uFus = new ArrayList<>();
        List<XBaseThr.Unit> uM = new ArrayList<>();
        List<XBaseThr.Unit> uT = new ArrayList<>();
        long failCtrl = 0;
        for (int i = 0; i < 2; i++) {
            XBaseVec.Blk b = main.get(i);
            int n = b.g().own.size();
            double a = XBaseFinalReport.tOf(tb37, n)[0];
            double c = XBaseFinalReport.tOf(tbOld, n)[0];
            double sfT = XBaseFinalReport.sfaceT(n);
            nodes.append(String.format(Locale.ROOT, "Проход %d: N = %d; t (37) %.2f, прежняя %.2f, SFace %.2f%n", i + 1, n, a, c, sfT));
            r37.addAll(atT(b, i, r, a, p, false));
            rOld.addAll(atT(b, i, r, c, p, false));
            String sl = XBaseFinal.sfaceLines(r, b, xsf);
            sc.append("# SFace | ").append(b.name()).append('\n').append(sl);
            List<Res> one = new ArrayList<>();
            for (String l : sl.split("\n")) {
                if (l.isEmpty()) continue;
                String[] q = l.split("\t", -1);
                boolean own = q[3].equals(FeiScores.OWN);
                if (!own && !q[3].equals(FeiScores.CTRL)) continue;
                one.add(new Res(q[4], i, own, Double.parseDouble(q[8]) <= sfT, own && q[6].equals(q[7]), byImg.get(Integer.parseInt(q[5]))));
            }
            addFails(one, b, i, p, false);
            rSf.addAll(one);
            csSf.add(XBaseEval.parse(b.g(), EnumSet.of(XBaseEval.V.SFACE), b.gal(), b.probes(), b.parts(), sl).cfg.get(XBaseEval.V.SFACE));
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, XBaseFinal.TT1, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            List<XBaseVec.Model> ms = XBaseFinal.models(r, b, XST, false);
            String fl = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, false), XBaseVec.fuseStats(mx, false), b.g().rOf, b.g().r0, clamped);
            sc.append("# слияние | ").append(b.name()).append('\n').append(fl);
            uFus.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), fl, null, true));
            double[] sxm = XBaseVec.fuseStats(mx, false);
            double[] sxt = XBaseVec.fuseStats(ms, false);
            uM.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), XBaseVec.fuseLines(mx, mx, 1.0, sxm, sxm, b.g().rOf, -1, clamped), null, true));
            uT.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), XBaseVec.fuseLines(ms, ms, 1.0, sxt, sxt, b.g().rOf, -1, clamped), null, true));
            for (Att at : p.atts) if (at.key() == null && !p.halves.get(i).contains(at.person())) failCtrl++;
        }
        List<XBaseThr.Unit> uStill = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            XBaseVec.Blk b = still.get(i);
            rStill.addAll(atT(b, i, r, XBaseFinalReport.tOf(tb37, main.get(i).g().own.size())[0], p, true));
            List<XBaseVec.Model> mt = XBaseFinal.models(r, b, XBaseFinal.TT1, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(r, b, XBaseFinal.XS, false);
            String fl = XBaseVec.fuseLines(mt, mx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(mt, false), XBaseVec.fuseStats(mx, false), b.g().rOf, b.g().r0, clamped);
            sc.append("# still | ").append(b.name()).append('\n').append(fl);
            uStill.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), fl, null, true));
        }
        Files.createDirectories(r.dir());
        Path scf = r.dir().resolve("cp_scores.tsv.gz");
        Writer sw = XBaseVec.writer(scf, sc.toString());
        XBaseVec.finish(sw, scf);
        t.append("\n=== Пороги ===\n").append(nodes);
        t.append("\n=== Главные меры при рабочем пороге ===\nстрока | ").append("ошибка своих | «чужое имя» | FAR по попыткам | FAR по людям").append(" | свои по людям и по попыткам, 95 % интервал | FAR | верный лучший\n");
        t.append(mainRow("1 слияние (32), t по (37)", r37, pIdx, w, persons));
        t.append(mainRow("2 слияние (32), прежняя таблица", rOld, pIdx, w, persons));
        t.append(mainRow("3 SFace (24)", rSf, pIdx, w, persons));
        t.append(mainRow("5 слияние (32), галерея still, вне (21)", rStill, pIdx, w, persons));
        t.append("\n=== Итог по разд. 6 постановки ===\n");
        double far1 = farAtt(r37, pIdx, unit);
        double far2 = farAtt(rOld, pIdx, unit);
        t.append(String.format(Locale.ROOT, "FAR строки 1 по попыткам с захватом %.2f %%: %s; строки 2 %.2f %%: %s (порог держит при ≤ 1 %% в точке).%n", far1, far1 <= 1 ? "держит" : "не держит", far2, far2 <= 1 ? "держит" : "не держит"));
        if (w.length > 0) {
            double[] d = FeiBootstrap.sorted(Arrays.stream(reps(a -> errPersons(r37, pIdx, a), w)).filter(x -> !Double.isNaN(x)).toArray());
            t.append(verdict(errPersons(r37, pIdx, unit), FeiBootstrap.quantile(d, 0.025), FeiBootstrap.quantile(d, 0.975))).append('\n');
        }
        t.append(String.format(Locale.ROOT, "Справочно: GT межс. 8–30 дней %.1f %% (группа 28 дней), > 30 дней %.1f %% (группа 45 дней).%n", REF_28, REF_45));
        t.append("\n=== Разбивки (направление E / L совпадает с делением улица / помещение) ===\n");
        breakdown(t, "1", r37, pIdx, w, persons);
        breakdown(t, "3 SFace", rSf, pIdx, w, persons);
        t.append(String.format(Locale.ROOT, "%n=== Отказы захвата ===%nСвои: %d из %d попыток. Чужие: %d попыток вне FAR (по проходам).%n", r37.stream().filter(x -> x.own() && x.att().key() == null).count(), r37.stream().filter(Res::own).count(), failCtrl));
        t.append("\n=== Меры без порога (бутстреп по людям; отказы своих – ошибка) ===\n");
        XBaseThr.Out oFus = XBaseThr.evaluate("слияние", uFus, persons, u.tpl);
        XBaseThr.Out oM = XBaseThr.evaluate("L1-xs-F (М)", uM, persons, u.tpl);
        XBaseThr.Out oT = XBaseThr.evaluate("L1-xs-F (Т+М)", uT, persons, u.tpl);
        XBaseThr.Out oS = XBaseThr.evaluate("still", uStill, persons, u.tpl);
        XBaseFinalReport.M mFus = XBaseFinalReport.measure(withFails(oFus.res.get(XBaseThr.WAYS[0]).cfgs, main, p, false, pIdx), persons, w);
        XBaseFinalReport.M mSf = XBaseFinalReport.measure(withFails(csSf, main, p, false, pIdx), persons, w);
        XBaseFinalReport.M mM = XBaseFinalReport.measure(withFails(oM.res.get(XBaseThr.WAYS[0]).cfgs, main, p, false, pIdx), persons, w);
        XBaseFinalReport.M mT = XBaseFinalReport.measure(withFails(oT.res.get(XBaseThr.WAYS[0]).cfgs, main, p, false, pIdx), persons, w);
        XBaseFinalReport.M mS = XBaseFinalReport.measure(withFails(oS.res.get(XBaseThr.WAYS[0]).cfgs, still, p, true, pIdx), persons, w);
        t.append(sepRow("слияние (32)", mFus)).append(sepRow("SFace", mSf)).append(sepRow("слияние, галерея still", mS));
        t.append("\n=== Строка 4 (справочно): L1-xs-F (Т+М) против (М), β = 1, всегда X+mirror ===\n");
        t.append(sepRow("(М)", mM)).append(sepRow("(Т+М)", mT));
        if (w.length > 0) {
            t.append("(Т+М) − (М): FAR 1 % ").append(XBaseFinalReport.ci(mT.sep.err[0] - mM.sep.err[0], diff(mT.sep.errRep[0], mM.sep.errRep[0])));
            t.append("; FAR 5 % ").append(XBaseFinalReport.ci(mT.sep.err[2] - mM.sep.err[2], diff(mT.sep.errRep[2], mM.sep.errRep[2])));
            t.append("; EER ").append(XBaseFinalReport.ci(mT.sep.eer[3] - mM.sep.eer[3], diff(mT.sep.eerRep, mM.sep.eerRep)));
            t.append("; pAUC ").append(XBaseFinalReport.ci(mT.pauc - mM.pauc, diff(mT.paucRep, mM.paucRep))).append('\n');
        }
        t.append(XBaseAbl3.holds("(М)", oM, XBaseThr.WAYS[2])).append(XBaseAbl3.holds("(Т+М)", oT, XBaseThr.WAYS[2]));
        Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
        of.put("(М)", oM);
        of.put("(Т+М)", oT);
        XBaseLbp.tables(t, "\nСтрока 4: П3 по (Т2) (отказы захвата в таблицы П3 не входят)", of, persons, u.tpl, new ArrayList<>(), true);
        t.append("\n=== Время, с ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f%n", k, v)));
        t.append(String.format(Locale.ROOT, "модели, оценка и отчёт: %.0f%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(r.dir().resolve("cp_report.txt"), t);
        System.out.print(t);
    }

    // ---------------------------------------------------------------- план прогона

    /** Просмотр разметки: кадры человека на камере анфаса и на всех камерах. */
    static final class Scan {
        final Map<Key, List<Integer>> front = new TreeMap<>();
        final Map<Key, Integer> anyCam = new TreeMap<>();
        final TreeSet<String> p1 = new TreeSet<>();
        final Map<String, TreeSet<String>> partsOf = new TreeMap<>();
        final List<String> g28 = new ArrayList<>();
        final List<String> g45 = new ArrayList<>();
        TreeSet<String> only2;
    }

    static Scan scan(Path gt) throws IOException {
        Scan x = new Scan();
        for (String dir : XBaseChoke.DIRS) {
            for (String part : dir.startsWith("P2") ? List.of("1", "2") : List.of("")) {
                for (int s = 1; s <= 4; s++) {
                    int fc = XBaseChoke.FRONTAL.get(dir)[s - 1];
                    for (int c = 1; c <= 3; c++) {
                        for (XBaseChoke.Rec r : XBaseChoke.parse(Files.readAllLines(gt.resolve(XBaseChoke.seqName(dir, s, c, part) + ".xml"), StandardCharsets.UTF_8))) {
                            Key k = new Key(dir, s, part, r.person());
                            x.anyCam.merge(k, 1, Integer::sum);
                            if (c == fc) x.front.computeIfAbsent(k, q -> new ArrayList<>()).add(r.frame());
                        }
                    }
                }
            }
        }
        for (Key k : x.anyCam.keySet()) {
            if (k.dir().startsWith("P1")) x.p1.add(k.person());
            else x.partsOf.computeIfAbsent(k.person(), q -> new TreeSet<>()).add(k.part());
        }
        for (String p : x.p1) (x.partsOf.get(p).first().equals("1") ? x.g28 : x.g45).add(p);
        x.only2 = new TreeSet<>(x.partsOf.keySet());
        x.only2.removeAll(x.p1);
        return x;
    }

    /** Точки глаз (только справочная проверка полного прогона): кадр → {xл, yл, xп, yп}; запись не с двумя точками – пустой массив. */
    static Map<Integer, int[]> eyes(List<String> xml) {
        Map<Integer, int[]> out = new HashMap<>();
        int frame = -1;
        boolean inside = false;
        int[] v = null;
        int got = 0;
        for (String l : xml) {
            Matcher f = FRAME.matcher(l);
            if (f.find()) frame = Integer.parseInt(f.group(1));
            if (PERSON.matcher(l).find()) {
                inside = true;
                v = new int[4];
                got = 0;
            }
            if (inside && l.contains("Eye ")) {
                Matcher m = XY.matcher(l);
                if (m.find()) {
                    int i = l.contains("<leftEye") ? 0 : 2;
                    v[i] = Integer.parseInt(m.group(1));
                    v[i + 1] = Integer.parseInt(m.group(2));
                    got++;
                }
            }
            if (l.contains("</person>")) {
                out.put(frame, got == 2 ? v : new int[0]);
                inside = false;
            }
        }
        return out;
    }

    /** {кадров с лицом и записью с двумя точками, из них середина глаз вне рамки, записей с одной точкой} по камерам анфаса. */
    static long[] eyeCheck(Frames fx, Path gt) throws IOException {
        long[] out = new long[3];
        for (String dir : XBaseChoke.DIRS) {
            for (String part : dir.startsWith("P2") ? List.of("1", "2") : List.of("")) {
                for (int s = 1; s <= 4; s++) {
                    String seq = XBaseChoke.seqName(dir, s, XBaseChoke.FRONTAL.get(dir)[s - 1], part);
                    for (Map.Entry<Integer, int[]> x : eyes(Files.readAllLines(gt.resolve(seq + ".xml"), StandardCharsets.UTF_8)).entrySet()) {
                        if (x.getValue().length == 0) {
                            out[2]++;
                            continue;
                        }
                        double[] row = fx.rows.get(seq + "/" + String.format(Locale.ROOT, "%08d", x.getKey()));
                        if (row == null) continue;
                        out[0]++;
                        double mx = (x.getValue()[0] + x.getValue()[2]) / 2.0;
                        double my = (x.getValue()[1] + x.getValue()[3]) / 2.0;
                        if (mx < row[0] || my < row[1] || mx > row[0] + row[2] || my > row[1] + row[3]) out[1]++;
                    }
                }
            }
        }
        return out;
    }

    /** План полного прогона; trial != null – один проход (пробный режим). */
    static Plan realPlan(Frames fx, SettingsStore settings, double r0, double rLimit, Key trial) throws IOException {
        Path gt = fx.root.resolve("groundtruth");
        Scan sc = scan(gt);
        Plan p = new Plan();
        p.dir = fx.root.resolve("reports").resolve("cp");
        p.halves = XBaseChoke.split(sc.g28, sc.g45, settings.loadFacesSeed() + XBaseChoke.SPLIT_SEED_OFFSET);
        p.only2 = new ArrayList<>(sc.only2);
        p.all.addAll(sc.p1);
        p.all.addAll(sc.only2);
        Collections.sort(p.all);
        int[] noAlign = new int[1];
        if (trial != null) {
            String key = pick(fx, p, trial, sc.front.get(trial), r0, rLimit, new HashSet<>(), noAlign);
            p.atts.add(new Att(trial.person(), "проба", trial.dir(), trial.s(), trial.part(), key, key == null ? Double.NaN : iod(fx.rows.get(key.substring(KEY.length())))));
            return p;
        }
        for (Key k : sc.anyCam.keySet()) if (k.dir().startsWith("P2") && !sc.front.containsKey(k)) throw new IllegalStateException("ChokePoint: проход без кадров камеры анфаса " + k + " – по описи таких нет");
        int replaced = 0;
        int shortTpl = 0;
        for (String q : sc.p1) {
            Set<String> used = new HashSet<>();
            List<String> keys = new ArrayList<>();
            for (int s = 1; s <= 4; s++) {
                String got = null;
                boolean first = true;
                for (String[] src : galleryOrder(s)) {
                    Key k = new Key(src[0], Integer.parseInt(src[1]), "", q);
                    List<Integer> fr = sc.front.get(k);
                    if (fr != null) got = pick(fx, p, k, fr, r0, rLimit, used, noAlign);
                    if (got != null) break;
                    first = false;
                }
                if (got != null) {
                    keys.add(got);
                    if (!first) replaced++;
                }
            }
            if (keys.size() != 4) shortTpl++;
            p.gal.put(q, keys);
        }
        long failCommon = 0;
        long failOnly2 = 0;
        for (Map.Entry<Key, List<Integer>> e : sc.front.entrySet()) {
            Key k = e.getKey();
            if (!k.dir().startsWith("P2")) continue;
            String key = pick(fx, p, k, e.getValue(), r0, rLimit, new HashSet<>(), noAlign);
            String group = sc.g45.contains(k.person()) ? "45" : sc.g28.contains(k.person()) ? "28" : "P2";
            if (key == null) {
                if (group.equals("P2")) failOnly2++;
                else failCommon++;
            }
            p.atts.add(new Att(k.person(), group, k.dir(), k.s(), k.part(), key, key == null ? Double.NaN : iod(fx.rows.get(key.substring(KEY.length())))));
        }
        if (GalleryEvaluation.OWN_INPUT_SCALE != 0.25) throw new IllegalStateException("ChokePoint: масштаб своей базы не 0,25");
        long stillNoFace = 0;
        long stillPose = 0;
        long stillNoAlign = 0;
        long stillNoExif = 0;
        TreeSet<String> dates = new TreeSet<>();
        for (String q : sc.p1) {
            List<String> keys = new ArrayList<>();
            String latest = null;
            for (String kind : List.of("Neutral", "Smile")) {
                Path f = fx.root.resolve("still").resolve(kind).resolve("ID" + q + ".JPG");
                if (!Files.exists(f)) throw new IOException("ChokePoint: нет портрета " + f);
                String dk = "still/" + kind + "/ID" + q;
                double[] row = fx.row(dk, f, GalleryEvaluation.OWN_INPUT_SCALE);
                if (row == null) {
                    stillNoFace++;
                    continue;
                }
                if (Math.abs(FeiDataset.pose(row)) > rLimit) {
                    stillPose++;
                    continue;
                }
                if (!alignable(row)) {
                    stillNoAlign++;
                    continue;
                }
                String key = KEY + dk;
                fx.make(p, key, f, row);
                keys.add(key);
                String d;
                try (java.io.InputStream in = Files.newInputStream(f)) {
                    d = exifDate(in.readNBytes(256 * 1024));
                }
                if (d == null) {
                    stillNoExif++;
                } else {
                    dates.add(d);
                    if (latest == null || d.compareTo(latest) > 0) latest = d;
                }
            }
            p.still.put(q, keys);
            if (latest != null) p.stillDate.put(q, latest);
        }
        long dropped = p.atts.stream().filter(a -> sc.p1.contains(a.person()) && !laterThanStill(p, a)).count();
        long noTpl = p.still.values().stream().filter(List::isEmpty).count();
        long[] eye = eyeCheck(fx, gt);
        double[] iods = p.atts.stream().filter(a -> a.key() != null).mapToDouble(Att::iod).sorted().toArray();
        p.stats.append(String.format(Locale.ROOT, "%nОбработка: кадров с записью разметки обработано %d, YuNet нашёл лицо %d, новых детекций в этом запуске %d; кадров с несколькими лицами (первый проход) %d; отказ выравнивания кандидата %d.%n", fx.total, fx.found, fx.computed, fx.multi.size(), noAlign[0]));
        p.stats.append(String.format(Locale.ROOT, "Галерея P1: замен кадра %d; людей с шаблоном меньше 4 кадров %d.%n", replaced, shortTpl));
        p.stats.append(String.format(Locale.ROOT, "Пробы портала 2: попыток %d, отказ захвата – общие люди %d, только портал 2 %d; межглазье выбранного кадра: мин %.1f, медиана %.1f, макс %.1f px.%n", p.atts.size(), failCommon, failOnly2, iods.length == 0 ? 0 : iods[0], iods.length == 0 ? 0 : iods[iods.length / 2], iods.length == 0 ? 0 : iods[iods.length - 1]));
        p.stats.append(String.format(Locale.ROOT, "Проверка «середина глаз разметки внутри рамки YuNet» (справочно, на выбор не влияет): кадров с лицом %d, отбраковала бы %d, нельзя проверить %d (запись с одной точкой глаза).%n", eye[0], eye[1], eye[2]));
        p.stats.append(String.format(Locale.ROOT, "Портреты still: нет лица %d, |r| > R %d, отказ выравнивания %d, снимков без даты EXIF %d; людей без шаблона %d; проб не позже портрета (не засчитаны) %d; даты EXIF: %s.%n", stillNoFace, stillPose, stillNoAlign, stillNoExif, noTpl, dropped, dates));
        Files.writeString(fx.der.resolve("cp_c_points.tsv"), fx.pts.toString(), StandardCharsets.UTF_8);
        StringBuilder sel = new StringBuilder("person\trole\tdir\ts\tpart\tkey\tiod\tr\n");
        p.gal.forEach((q, ks) -> ks.forEach(k -> sel.append(q).append("\tgallery\t-\t-\t-\t").append(k).append("\t-\t").append(p.rOf.get(k)).append('\n')));
        p.still.forEach((q, ks) -> ks.forEach(k -> sel.append(q).append("\tstill\t-\t-\t-\t").append(k).append("\t-\t").append(p.rOf.get(k)).append('\n')));
        for (Att a : p.atts) sel.append(a.person()).append("\tprobe\t").append(a.dir()).append('\t').append(a.s()).append('\t').append(a.part()).append('\t').append(a.key()).append('\t').append(a.iod()).append('\t').append(a.key() == null ? "-" : p.rOf.get(a.key())).append('\n');
        Files.writeString(fx.der.resolve("cp_selected.tsv"), sel.toString(), StandardCharsets.UTF_8);
        return p;
    }

    /**
     * Сухой прогон пути на GT (ChokePoint не читается): 29 человек GT, галерея 4 снимка, still 2, пробы 8, один искусственный отказ захвата
     * общего человека и один – чужого, один человек без шаблона still. Только для проверки кода отчёта; числа смысла не имеют.
     */
    static Plan dryPlan(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out, XBaseA0.FeretSetup su) throws IOException {
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        Map<String, double[]> rows = FrameK2.exportRows(dir);
        Map<String, List<String>> keys = new TreeMap<>();
        for (String q : g.persons) {
            List<String> l = new ArrayList<>();
            g.byPerson.get(q).values().forEach(l::addAll);
            Collections.sort(l);
            if (l.size() >= 14 && keys.size() < 29) keys.put(q, l);
        }
        if (keys.size() < 29) throw new IllegalStateException("cp dry: мало людей GT " + keys.size());
        Plan p = new Plan();
        p.dir = out.resolve("cp_dry");
        p.all.addAll(keys.keySet());
        List<String> common = new ArrayList<>(p.all.subList(0, 25));
        p.only2 = new ArrayList<>(p.all.subList(25, 29));
        List<String> g45 = common.subList(21, 25);
        p.halves = XBaseChoke.split(common.subList(0, 21), g45, settings.loadFacesSeed() + XBaseChoke.SPLIT_SEED_OFFSET);
        Set<String> used = new HashSet<>();
        for (String q : p.all) {
            List<String> l = keys.get(q);
            if (common.contains(q)) {
                p.gal.put(q, new ArrayList<>(l.subList(0, 4)));
                p.still.put(q, new ArrayList<>(l.subList(4, 6)));
                used.addAll(l.subList(0, 6));
            }
            for (int i = 0; i < 8; i++) {
                String k = l.get(6 + i);
                used.add(k);
                p.atts.add(new Att(q, g45.contains(q) ? "45" : common.contains(q) ? "28" : "P2", i < 4 ? "P2E" : "P2L", i % 4 + 1, "1", k, iod(rows.get(k))));
            }
        }
        String ownQ = p.halves.get(0).get(0);
        for (int i = 0; i < p.atts.size(); i++) {
            Att a = p.atts.get(i);
            if (a.person().equals(ownQ)) {
                p.atts.set(i, new Att(a.person(), a.group(), a.dir(), a.s(), a.part(), null, Double.NaN));
                break;
            }
        }
        for (int i = p.atts.size() - 1; i >= 0; i--) {
            Att a = p.atts.get(i);
            if (p.only2.contains(a.person())) {
                p.atts.set(i, new Att(a.person(), a.group(), a.dir(), a.s(), a.part(), null, Double.NaN));
                break;
            }
        }
        p.still.put(p.halves.get(1).get(0), new ArrayList<>());
        for (String k : used) p.rOf.put(k, Math.abs(FeiDataset.pose(rows.get(k))));
        p.fr.putAll(XBaseEval.frames(export, g.samples.values(), used));
        p.k2.putAll(FrameK2.load(used, su, su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir())));
        for (GalleryEvaluation.Sample s : g.samples.values()) {
            String k = s.file().toString();
            if (used.contains(k)) p.sf.put(k, export.frame(s, GalleryEvaluation.Variant.SFACE));
        }
        p.stats.append("\nСУХОЙ ПРОГОН НА GT: ChokePoint не читался; проверка пути и отчёта, числа смысла не имеют.\n");
        return p;
    }

    /** cp-run: FACES_CP_DRY=1 – сухой прогон на GT; FACES_CP_TRIAL=P2L,1,1,0001 – пробный режим; иначе полный прогон. */
    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("cp-run: FACES_FRAMES=b");
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("cp-run: задайте FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        double rLimit = XBaseEval.poseLimit(selLines);
        if (r0 != 0.12 || Math.abs(rLimit - 0.387) > 5e-4) throw new IllegalStateException("cp-run: r0 = " + r0 + ", R = " + rLimit + " не совпали с постановкой");
        long seed = settings.loadFacesSeed();
        long t0 = System.nanoTime();
        Map<Integer, XBasePart4.TRow> tb37 = XBasePart4.readTable(Paths.get(System.getenv("FACES_P4_TABLE")));
        Map<Integer, XBasePart4.TRow> tbOld = XBasePart4.readTable(Paths.get(System.getenv("FACES_CP_TABLE_OLD")));
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        if ("1".equals(System.getenv("FACES_CP_DRY"))) {
            Plan p = dryPlan(export, dir, feiKeys, selLines, settings, out, su);
            List<List<String>> train = train(p, su, export, dir, feiKeys, settings, seed, 15);
            evaluate(p, setup(p, su, train, settings, r0, rLimit), tb37, tbOld, settings, "ChokePoint – СУХОЙ ПРОГОН НА GT\n");
            System.out.println("DONE");
            return;
        }
        String root = System.getenv("FACES_CHOKEPOINT_DIR");
        if (root == null) throw new IllegalStateException("cp-run: задайте FACES_CHOKEPOINT_DIR");
        Frames fx = new Frames(Paths.get(root), settings);
        String tr = System.getenv("FACES_CP_TRIAL");
        if (tr != null) {
            String[] c = tr.split(",");
            trial(fx, new Key(c[0], Integer.parseInt(c[1]), c[2], c[3]), su, export, dir, feiKeys, settings, r0, rLimit, seed, t0);
            fx.close();
            System.out.println("DONE");
            return;
        }
        Plan p = realPlan(fx, settings, r0, rLimit, null);
        fx.close();
        System.out.printf(Locale.ROOT, "cp: кадры выбраны за %.0f с%n", (System.nanoTime() - t0) / 1e9);
        List<List<String>> train = train(p, su, export, dir, feiKeys, settings, seed, 0);
        String head = "ChokePoint – однократная итоговая проверка (постановка 06.11, редакция 2, SHA-256 994a3538…8aadc; XBaseEval cp-run); кадры изменены: выровнены и уменьшены\nДанные: ChokePoint (Copyright (C) 2011 NICTA); Y. Wong, S. Chen, S. Mau, C. Sanderson, B.C. Lovell, CVPR Workshops 2011, pp. 81–88\n";
        evaluate(p, setup(p, su, train, settings, r0, rLimit), tb37, tbOld, settings, head);
        System.out.printf(Locale.ROOT, "Всего: %.0f с%n", (System.nanoTime() - t0) / 1e9);
        System.out.println("DONE");
    }

    /** Пробный режим: один проход одного человека; оценка s и решение не считаются и не выводятся. */
    static void trial(Frames fx, Key k, XBaseA0.FeretSetup su, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, SettingsStore settings, double r0, double rLimit, long seed, long t0)
            throws IOException {
        long t00 = System.nanoTime();
        Plan p = realPlan(fx, settings, r0, rLimit, k);
        double tDet = (System.nanoTime() - t00) / 1e9;
        Att a = p.atts.get(0);
        if (a.key() == null) throw new IllegalStateException("cp-trial: отказ захвата на пробном проходе");
        StringBuilder t = new StringBuilder("ChokePoint – пробный режим, проверка пути; оценка s и решение не считаются\nКод: коммит " + FarMethods.commit() + "\n");
        t.append(String.format(Locale.ROOT, "Проход %s; кадров разметки на камере анфаса %d; лицо найдено %d; новых детекций %d; выбор и кадры за %.1f с (%.1f мс на кадр при новых детекциях)%n", k, fx.total, fx.found, fx.computed, tDet, fx.computed == 0 ? 0 : 1000 * tDet / fx.computed));
        t.append(String.format(Locale.ROOT, "Выбран %s, |r| %.4f, межглазье %.1f px; кадр Б %dx%d, кадр k = 2 %dx%d, вход SFace %dx%d%n", a.key(), p.rOf.get(a.key()), a.iod(), p.fr.get(a.key()).cols(), p.fr.get(a.key()).rows(), p.k2.get(a.key()).frame().cols(), p.k2.get(a.key()).frame().rows(), p.sf.get(a.key()).cols(), p.sf.get(a.key()).rows()));
        long t1 = System.nanoTime();
        List<List<String>> train = train(p, su, export, dir, feiKeys, settings, seed, 0);
        t.append(String.format(Locale.ROOT, "Загрузка пула и когорты: %.0f с; кадров всего %d%n", (System.nanoTime() - t1) / 1e9, p.fr.size()));
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        xsf.sfaceVectors(Map.of(a.key(), p.sf.get(a.key())), fx.sface);
        t.append("Признак SFace: размерность ").append(xsf.vec.get(XBaseEval.V.SFACE).get(a.key()).length).append('\n');
        int threads = settings.loadFacesFeiCvThreads();
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            long t2 = System.nanoTime();
            XBaseEval x = XBasePart4.vectorsB(p.fr, p.k2, XBaseFinal.param(c, IlluminationNorm.NormParams.of(settings)), c.f(), threads);
            double tf = (System.nanoTime() - t2) / 1e9;
            t2 = System.nanoTime();
            // Путь обучения W и вложения пробы; расстояния не считаются. Галерея – сам проход (класс из одного снимка).
            XBaseRule.Fit f = XBaseRule.fit(x, true, List.of(List.of(a.key())), train, null);
            int dim = f.lm.embed(x.vec.get(XBaseEval.V.L1).get(a.key())).length;
            t.append(String.format(Locale.ROOT, "%s: признаки %.0f с; размерность %d; одна модель W (X+mirror) %.0f с, размерность вложения %d%n", c.name(), tf, x.vec.get(XBaseEval.V.L1).get(a.key()).length, (System.nanoTime() - t2) / 1e9, dim));
        }
        t.append(String.format(Locale.ROOT, "Всего %.0f с; куча %.1f ГБ%n", (System.nanoTime() - t0) / 1e9, (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1e9));
        Files.createDirectories(fx.root.resolve("reports"));
        FarMethods.write(fx.root.resolve("reports").resolve("cp_trial.txt"), t);
        System.out.print(t);
    }
}
