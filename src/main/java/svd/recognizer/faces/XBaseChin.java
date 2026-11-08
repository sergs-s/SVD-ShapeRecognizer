package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Проба «кадр с подбородком и овальная маска» (постановка Жорика 08.11, SHA-256 fb7aaa58…). chin-sheet – п. 2.1: положения глаз, рта и низа
 * кадра (Ч) в шаблоне выравнивания и лист 10 людей GT (база, (Ч), (Ч+О)); вывод – только D:\data\GeorgiaTech\chin\.
 *
 * @author ssv
 */
final class XBaseChin {

    private XBaseChin() {
    }

    /** Множитель удлинения: низ кадра на 1,0 · (y_рта − y_глаз) ниже линии рта шаблона. */
    static final double CHIN = 1.0;

    /** Высота кадра (Ч): низ = y_рта + CHIN · (y_рта − y_глаз) по шаблону кадра Б, округление до ближайшего чётного. */
    static int height() {
        double[][] t = FaceAlignment.ownTemplate(FrameB.W, FrameB.H);
        double ye = (t[0][1] + t[1][1]) / 2;
        double ym = (t[3][1] + t[4][1]) / 2;
        return 2 * (int) Math.round((ym + CHIN * (ym - ye)) / 2);
    }

    /** Маска (Ч+О): эллипс с центром посередине кадра, полуоси – половины ширины и высоты; 255 – внутри. */
    static Mat mask(int w, int h) {
        // Точно по формуле (центр пикселя внутри или на эллипсе): Imgproc.ellipse округляет центр до целого и даёт несимметричную маску.
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0, a = w / 2.0, b = h / 2.0;
        byte[] v = new byte[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            double u = (x - cx) / a, q = (y - cy) / b;
            if (u * u + q * q <= 1) v[y * w + x] = (byte) 255;
        }
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        m.put(0, 0, v);
        return m;
    }

    /** Кадр Б высоты h: то же подобие на шаблон 92×112 и то же сглаживание, что FrameB.frame, выход – 92×h. */
    static Mat frame(Mat color, double[] row, int h) {
        double[][] lm = FrameK2.lm(row);
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        Mat m = FaceAlignment.similarity(lm, FaceAlignment.ownTemplate(FrameB.W, FrameB.H));
        Mat a = new Mat();
        int k = FrameB.k(m);
        if (k == 1) {
            Imgproc.warpAffine(gray, a, m, new Size(FrameB.W, h), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        } else {
            Mat mk = new Mat(2, 3, CvType.CV_64F);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) mk.put(i, j, k * m.get(i, j)[0] + (j == 2 ? (k - 1) / 2.0 : 0));
            Mat big = new Mat();
            Imgproc.warpAffine(gray, big, mk, new Size(FrameB.W * k, h * k), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
            Imgproc.resize(big, a, new Size(FrameB.W, h), 0, 0, Imgproc.INTER_AREA);
            big.release();
            mk.release();
        }
        gray.release();
        m.release();
        return a;
    }

    static void sheet(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("chin-sheet: FACES_FRAMES=b");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Path od = gtDir.getParent().resolve("chin");
        Files.createDirectories(od);
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        Map<String, double[]> rows = FrameK2.exportRows(dir);
        double[][] t = FaceAlignment.ownTemplate(FrameB.W, FrameB.H);
        double ye = (t[0][1] + t[1][1]) / 2;
        double ym = (t[3][1] + t[4][1]) / 2;
        int h = height();
        StringBuilder r = new StringBuilder("Кадр с подбородком (постановка 08.11, п. 2.1) – геометрия и лист (XBaseEval chin-sheet)\nКод: коммит " + FarMethods.commit() + "\n\n");
        r.append(String.format(Locale.ROOT, "Шаблон выравнивания кадра Б (ArcFace 112, масштаб %.4f, сдвиг x %.2f), кадр %d×%d, px (x, y):%n", FrameB.H / 112.0, (FrameB.W - 112.0 * FrameB.H / 112) / 2, FrameB.W, FrameB.H));
        String[] nm = {"глаз (слева на кадре)", "глаз (справа на кадре)", "нос", "рот (угол слева)", "рот (угол справа)"};
        for (int i = 0; i < 5; i++) r.append(String.format(Locale.ROOT, "  %s: %.2f, %.2f%n", nm[i], t[i][0], t[i][1]));
        r.append(String.format(Locale.ROOT, "Линия глаз y_глаз = %.3f; линия рта y_рта = %.3f; y_рта − y_глаз = %.3f; низ базы – %d (%.3f под линией рта).%n", ye, ym, ym - ye, FrameB.H, FrameB.H - ym));
        r.append(String.format(Locale.ROOT, "(Ч): низ = y_рта + 1,0 · (y_рта − y_глаз) = %.3f → высота кадра %d (ближайшее чётное); кадр %d×%d, верх и бока – как в базе, масштаб тот же.%n", ym + CHIN * (ym - ye), h, FrameB.W, h));
        r.append(String.format(Locale.ROOT, "(Ч+О): эллипс – центр (%.1f; %.1f), полуоси %.1f и %.1f px; вне эллипса пикселей %d из %d (%.1f %%).%n", (FrameB.W - 1) / 2.0, (h - 1) / 2.0, FrameB.W / 2.0, h / 2.0,
                FrameB.W * h - Core.countNonZero(mask(FrameB.W, h)), FrameB.W * h, 100.0 * (FrameB.W * h - Core.countNonZero(mask(FrameB.W, h))) / (FrameB.W * h)));
        r.append(String.format(Locale.ROOT, "Ячейки: L1-tt-xs-F 9×9 – %.2f × %.2f px (база %.2f × %.2f); L1-xs-F 7×7 – %.2f × %.2f (база %.2f × %.2f).%n%n", FrameB.W / 9.0, h / 9.0, FrameB.W / 9.0, FrameB.H / 9.0,
                FrameB.W / 7.0, h / 7.0, FrameB.W / 7.0, FrameB.H / 7.0));
        // Лист: 10 первых своих GT (по возрастанию), первая межсессионная проба.
        List<String> own = new ArrayList<>(g.own);
        java.util.Collections.sort(own);
        own = own.subList(0, Math.min(10, own.size()));
        int sc = 3;
        int cw = FrameB.W * sc, chh = h * sc, gap = 12, lab = 150, head = 50, cap = 26;
        int W = lab + 3 * (cw + gap) + gap, H = head + own.size() * (chh + cap + gap) + gap;
        Mat sheet = new Mat(H, W, CvType.CV_8UC1, new Scalar(255));
        Imgproc.putText(sheet, "GT, 10 persons, cross-session probe: base 92x112 | chin 92x" + h + " | chin + oval (outside: black)", new Point(gap, 32),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0), 1);
        Mat mk = mask(FrameB.W, h);
        int y = head;
        int same = 0;
        r.append("Лист: ").append(od.resolve("chin_sheet.png")).append("; люди и снимки:\n");
        for (String p : own) {
            String key = g.probeKeys.get(p).get(0);
            Path kp = Paths.get(key);
            Path raw = gtDir.resolve(kp.getName(1).toString()).resolve(kp.getName(2) + ".jpg");
            Mat color = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(raw)), Imgcodecs.IMREAD_COLOR);
            double[] row = rows.get(key);
            Mat base = FrameB.frame(color, row, true);
            Mat chin = frame(color, row, h);
            boolean eq = Core.countNonZero(diff(base, chin.submat(new Rect(0, 0, FrameB.W, FrameB.H)))) == 0;
            if (eq) same++;
            Mat oval = Mat.zeros(chin.size(), chin.type());
            chin.copyTo(oval, mk);
            r.append(String.format(Locale.ROOT, "  %s – %s (верх кадра (Ч) = кадр Б: %s)%n", p, key, eq ? "да" : "НЕТ"));
            Imgproc.putText(sheet, p, new Point(gap, y + chh / 2), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0), 2);
            Imgproc.putText(sheet, kp.getName(2).toString(), new Point(gap, y + chh / 2 + 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, new Scalar(0), 1);
            int x = lab;
            for (Mat v : List.of(base, chin, oval)) {
                Mat z = new Mat();
                Imgproc.resize(v, z, new Size(v.cols() * sc, v.rows() * sc), 0, 0, Imgproc.INTER_NEAREST);
                z.copyTo(sheet.submat(new Rect(x, y, z.cols(), z.rows())));
                x += cw + gap;
                z.release();
            }
            Imgproc.putText(sheet, "base", new Point(lab, y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            Imgproc.putText(sheet, "chin", new Point(lab + cw + gap, y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            Imgproc.putText(sheet, "chin + oval", new Point(lab + 2 * (cw + gap), y + chh + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0), 1);
            y += chh + cap + gap;
            color.release();
            base.release();
            chin.release();
            oval.release();
        }
        r.append(String.format(Locale.ROOT, "Верх кадра (Ч) совпал с кадром Б побайтно у %d из %d (то же выравнивание и масштаб).%n", same, own.size()));
        r.append("На листе: увеличение ×3 без сглаживания (INTER_NEAREST); в (Ч+О) пиксели вне эллипса показаны чёрными только для просмотра – в расчёте они не входят в гистограммы.\n");
        if (!Imgcodecs.imwrite(od.resolve("chin_sheet.png").toString(), sheet)) throw new IOException("chin-sheet: не записан лист");
        r.append("SHA-256 листа: ").append(FrameB.sha256(od.resolve("chin_sheet.png"))).append('\n');
        Files.writeString(od.resolve("chin_geometry.txt"), r, StandardCharsets.UTF_8);
        System.out.print(r);
    }

    /** Кадр (Ч) по снимку: верх 92×112 обязан совпасть с готовым кадром Б b побайтно; запись через временный файл. */
    static boolean make(Mat color, double[] row, Path bFile, Path out) throws IOException {
        if (Files.exists(out)) return false;
        Mat c = frame(color, row, height());
        Mat b = Imgcodecs.imread(bFile.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        if (b.empty()) throw new IOException("frames-chin: нет кадра Б " + bFile);
        Mat top = c.submat(new Rect(0, 0, FrameB.W, FrameB.H));
        if (Core.countNonZero(diff(top, b)) != 0) throw new IllegalStateException("frames-chin: верх кадра (Ч) ≠ кадр Б " + bFile);
        Files.createDirectories(out.getParent());
        FeretData.writePng(out, c);
        c.release();
        b.release();
        return true;
    }

    /**
     * frames-chin: кадры (Ч) 92×132 для всех снимков с кадром Б – FERET (derived\a_b → derived\a_bc), экспорт (a_b → a_bc по базам),
     * VidTIMIT (derived\a_b → derived\a_bc); сверка верха с кадром Б побайтно; готовые пропускаются (продолжение). BioID – строится в bioid-run.
     */
    static void frames(FarExport.Loaded export, Path dir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (FrameB.chin() == null) throw new IllegalStateException("frames-chin: задайте FACES_CHIN=c");
        long t0 = System.nanoTime();
        StringBuilder rep = new StringBuilder("Кадры (Ч) 92×" + height() + " (XBaseEval frames-chin)\nКод: коммит " + FarMethods.commit() + "\n");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Path der = su.dir.resolve(FeretData.DERIVED);
        int nf = 0, mf = 0;
        for (String k : new java.util.TreeSet<>(su.det.keySet())) {
            Path b = der.resolve(FrameB.DIR).resolve(k + ".png");
            if (!Files.exists(b)) continue;
            nf++;
            Path o = der.resolve(FrameB.dirName()).resolve(k + ".png");
            if (Files.exists(o)) continue;
            Mat color = FeretData.decode(su.byKey.get(k).file());
            if (make(color, su.det.get(k), b, o)) mf++;
            color.release();
            if (mf % 250 == 1) System.out.println("frames-chin: FERET " + mf);
        }
        rep.append(String.format(Locale.ROOT, "FERET: кадров Б %d, построено (Ч) в этом запуске %d%n", nf, mf));
        GalleryEvaluation.Raw raw = GalleryEvaluation.raw(settings, settings.loadFacesOwnDetectorScore(), Paths.get(settings.loadFacesModelsDir()),
                Paths.get(System.getProperty("user.dir"), "reports", "faces", "far"), false);
        Map<String, int[]> nb = new java.util.TreeMap<>();
        for (GalleryEvaluation.Sample s : raw.data().samples()) {
            double[] row = raw.dets().get(s.file().toString()).row();
            if (row == null) continue;
            Path id = FarExport.id(s);
            Path b = dir.resolve(FrameB.DIR).resolve(id + ".png");
            if (!Files.exists(b)) continue;
            int[] c = nb.computeIfAbsent(FarExport.dirName(s.base()), x -> new int[2]);
            c[0]++;
            Path o = dir.resolve(FrameB.dirName()).resolve(id + ".png");
            if (Files.exists(o)) continue;
            Mat[] m = GalleryEvaluation.load(s);
            try {
                if (make(m[0], row, b, o)) c[1]++;
            } finally {
                GalleryEvaluation.release(m);
            }
        }
        nb.forEach((k, c) -> rep.append(String.format(Locale.ROOT, "экспорт %s: кадров Б %d, построено (Ч) %d%n", k, c[0], c[1])));
        Path vdir = VidTimitData.dir(settings);
        Path vder = vdir.resolve(VidTimitData.DERIVED);
        Map<String, double[]> vrows = VidTimitData.readDetections(vder.resolve(VidTimitData.DETECTIONS));
        vrows.putAll(VidTimitData.readDetections(vder.resolve(XBaseFinal.VT_HEAD_DET)));
        int nv = 0, mv = 0;
        try (var st = Files.list(vder.resolve(FrameB.DIR))) {
            for (Path b : st.sorted().toList()) {
                String name = b.getFileName().toString();
                if (!name.endsWith(".png")) continue;
                nv++;
                Path o = vder.resolve(FrameB.dirName()).resolve(name);
                if (Files.exists(o)) continue;
                String[] q = name.substring(0, name.length() - 4).split("_");
                if (q.length != 3) throw new IllegalStateException("frames-chin: имя кадра VidTIMIT " + name);
                String dk = q[0] + "/" + q[1] + "/" + q[2];
                double[] row = vrows.get(dk);
                if (row == null) throw new IllegalStateException("frames-chin: нет детекции VidTIMIT " + dk);
                Mat color = Imgcodecs.imread(vdir.resolve(q[0]).resolve("video").resolve(q[1]).resolve(q[2]).toString(), Imgcodecs.IMREAD_COLOR);
                if (color.empty()) throw new IOException("frames-chin: нет кадра VidTIMIT " + dk);
                if (make(color, row, b, o)) mv++;
                color.release();
            }
        }
        rep.append(String.format(Locale.ROOT, "VidTIMIT: кадров Б %d, построено (Ч) %d%n", nv, mv));
        rep.append("Сверка: верх 92×112 каждого построенного кадра (Ч) совпал с кадром Б побайтно.\n");
        rep.append(String.format(Locale.ROOT, "Время: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Files.createDirectories(out);
        FarMethods.write(out.resolve("frames_chin.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- пул (п. 3, правило (36))

    /** Вариант модели пула: номер 90, признак – (32) по умолчанию; кадр и маска – по FACES_CHIN (база Б – без ключа). */
    static XBasePart4.BV poolVariant(IlluminationNorm.NormParams tt) {
        String c = FrameB.chin();
        String name = c == null ? "база (32), кадр Б 92×112" : c.equals("c") ? "(Ч) кадр 92×132" : "(Ч+О) кадр 92×132, овальная маска";
        return new XBasePart4.BV(90, "chin", name, "постановка 08.11, кадр с подбородком", FeatP4.base(tt), new int[] {1, 0});
    }

    /** pool-chin: модели пула (5 частей, оба признака) для варианта по FACES_CHIN; каталог – &lt;eq_b{метка}&gt;\xbase_pool_p4b, файлы _b90. */
    static void pool(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (!FrameB.on()) throw new IllegalStateException("pool-chin: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0);
        XBasePart4.BV v = poolVariant(u.tpl.params);
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        Map<Integer, Double> sec = XBasePart4.computeB(u, fs, List.of(v), 0);
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        System.out.printf(Locale.ROOT, "pool-chin %s: %s, модели – %s, %.0f с%n", FrameB.tag(), v.name(), XBasePart4.dirB(u, 0), (System.nanoTime() - t0) / 1e9);
        System.out.println("DONE");
    }

    /**
     * pool-chin-cmp: критерий пула (среднее ошибок своих при FAR 1, 2, 5 %; слияние (32)) у базы на кадрах Б, (Ч) и (Ч+О) по моделям pool-chin
     * и парные разности с бутстрепом по людям пула (как pool-noise: B = 10 000, seed faces.seed + 11 000). Запуск – FACES_FRAMES=b без
     * FACES_CHIN; отчёт – только D:\data\Feret\reports\eq_b\xbase_pool_chin.txt.
     */
    static void poolCmp(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (!FrameB.on() || FrameB.chin() != null) throw new IllegalStateException("pool-chin-cmp: FACES_FRAMES=b без FACES_CHIN");
        long t0 = System.nanoTime();
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0);
        for (Mat m : u.fr.values()) m.release();
        Path feret = u.out.getParent();
        String[] tags = {"", "_c", "_co"};
        String[] rows = {"база (32), кадр Б 92×112", "(Ч) кадр 92×132", "(Ч+О) кадр 92×132, маска"};
        List<String> sets = u.groups;
        int[] clamped = new int[2];
        FeiDecompose.Cfg[][] cfg = new FeiDecompose.Cfg[3][sets.size()];
        StringBuilder files = new StringBuilder();
        for (int s = 0; s < sets.size(); s++) {
            List<List<FeiDecompose.Cfg>> acc = new ArrayList<>();
            for (int r = 0; r < 3; r++) acc.add(new ArrayList<>());
            for (XBaseVec.Blk b : u.blocks) {
                if (!b.group().equals(sets.get(s))) continue;
                for (int r = 0; r < 3; r++) {
                    Path d = feret.resolve("eq_b" + tags[r]).resolve("xbase_pool_p4b");
                    Path ft = XBasePart4.fileB(d, b, 1, 90);
                    Path fx = XBasePart4.fileB(d, b, 0, 90);
                    if (s == 0 && files.indexOf(d.toString()) < 0) files.append("  ").append(rows[r]).append(" – ").append(d).append('\n');
                    List<XBaseVec.Model> tt = XBaseVec.read(ft).values().iterator().next();
                    List<XBaseVec.Model> xs = XBaseVec.read(fx).values().iterator().next();
                    String l = XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, true), XBaseVec.fuseStats(xs, true), b.g().rOf, b.g().r0, clamped);
                    acc.get(r).add(XBaseEval.parse(b.g(), java.util.EnumSet.of(XBaseEval.V.L1), b.gal(), b.probes(), XBaseFeat2.noParts(), l).cfg.get(XBaseEval.V.L1));
                }
            }
            for (int r = 0; r < 3; r++) cfg[r][s] = XBaseRule.merge(acc.get(r));
        }
        long seed = u.tpl.seed + XBasePart4.NOISE_SEED_OFFSET;
        int n = u.persons;
        int[][] w = FeiBootstrap.weights(n, XBasePart4.NOISE_B, seed);
        int[] unit = FeiBootstrap.unit(n);
        double[][] pt = new double[3][sets.size()];
        double[][][] rep = new double[3][sets.size()][XBasePart4.NOISE_B];
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(u.tpl.threads);
        try {
            for (int r = 0; r < 3; r++) for (int s = 0; s < sets.size(); s++) {
                FeiDecompose.Cfg c = cfg[r][s];
                pt[r][s] = XBasePart4.critW(c, unit);
                double[] o = rep[r][s];
                pool.submit(() -> java.util.stream.IntStream.range(0, XBasePart4.NOISE_B).parallel().forEach(q -> o[q] = XBasePart4.critW(c, w[q]))).get();
            }
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("pool-chin-cmp: " + e, e);
        } finally {
            pool.shutdown();
        }
        StringBuilder t = new StringBuilder("Кадр с подбородком – выбор по пулу, правило (36) (постановка 08.11; XBaseEval pool-chin-cmp)\nКод: коммит " + FarMethods.commit() + "\n");
        t.append("Конфигурация (32): (П2) + MS-LBP, σ₀ = 1, слияние w = 0,75, β = 1; все три строки – на кадрах Б (база (32) прежде считалась по пулу на кадрах «а» – 4,90 / 2,00 / 7,82 %, pool-noise).\n");
        t.append("Модели (pool-chin, вариант 90):\n").append(files);
        t.append(String.format(Locale.ROOT, "Критерий – среднее ошибок своих при FAR 1, 2, 5 %%. Бутстреп по людям пула: B = %d, seed %d, людей %d (как pool-noise).%n", XBasePart4.NOISE_B, seed, n));
        t.append("\n=== Критерий, точка [95 %] ===\nстрока");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        for (int r = 0; r < 3; r++) {
            t.append(rows[r]);
            for (int s = 0; s < sets.size(); s++) {
                double[] a = FeiBootstrap.sorted(rep[r][s].clone());
                t.append(String.format(Locale.ROOT, " | %.2f %% [%.2f; %.2f]", pt[r][s], FeiBootstrap.quantile(a, 0.025), FeiBootstrap.quantile(a, 0.975)));
            }
            t.append('\n');
        }
        t.append("\n=== Парная разность, п. п., точка [95 %]; * – значимо (меньше – лучше) ===\nпара");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        String[] pn = {"(Ч) − база", "(Ч+О) − база", "(Ч+О) − (Ч)"};
        int[][] pairs = {{1, 0}, {2, 0}, {2, 1}};
        boolean[] better = new boolean[3];
        for (int pi = 0; pi < 3; pi++) {
            t.append(pn[pi]);
            for (int s = 0; s < sets.size(); s++) {
                double[] d = new double[XBasePart4.NOISE_B];
                for (int q = 0; q < d.length; q++) d[q] = rep[pairs[pi][0]][s][q] - rep[pairs[pi][1]][s][q];
                double[] ds = FeiBootstrap.sorted(d);
                double lo = FeiBootstrap.quantile(ds, 0.025);
                double hi = FeiBootstrap.quantile(ds, 0.975);
                if (hi < 0) better[pi] = true;
                t.append(String.format(Locale.ROOT, " | %+.2f [%+.2f; %+.2f]%s", pt[pairs[pi][0]][s] - pt[pairs[pi][1]][s], lo, hi, lo > 0 || hi < 0 ? " *" : ""));
            }
            t.append('\n');
        }
        t.append(String.format(Locale.ROOT, "По пулу (правило (36)): (Ч) – %s; (Ч+О) – %s.%n", better[0] ? "значимо лучше базы хотя бы на одном наборе" : "значимо лучше ни на одном наборе",
                better[1] ? "значимо лучше базы хотя бы на одном наборе" : "значимо лучше ни на одном наборе"));
        t.append(String.format(Locale.ROOT, "Время: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Path f = feret.resolve("eq_b").resolve("xbase_pool_chin.txt");
        FarMethods.write(f, t);
        System.out.print(t);
        System.out.println("DONE");
    }

    static Mat diff(Mat a, Mat b) {
        Mat d = new Mat();
        Core.absdiff(a, b, d);
        return d;
    }
}
