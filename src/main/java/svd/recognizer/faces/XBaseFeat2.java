package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseVec.Blk;
import svd.recognizer.faces.XBaseVec.Model;
import svd.recognizer.faces.XBaseVec.Setup;
import svd.recognizer.storage.SettingsStore;

/**
 * Пачка 2 – признаки (задание Жорика №2 от 28.10.2026, план принят №3 с ответами, подтверждён Хозяином). База – принятая
 * конфигурация (П2), косинус, β = 1, оба кандидата и их слияние w = 0,75 (как пачка 1, XBaseVec); варианты признака, каждый (а)
 * вместо (П2) и (б) в сцеплении с (П2): LPQ (LpqFeature), BSIF 7×7 и 9×9 на 8 бит (BsifFeature), многомасштабный LBP вокруг 5 точек
 * YuNet (MsLbpFeature, кадр k = 2 – FrameK2). Вход – как у кандидата: L1-xs-F – кадр, L1-tt-xs-F – Тан – Триггс (у MS-LBP – на
 * кадре k = 2 с σ₀, σ₁ × k, γ, α, τ – прежние; решение Жорика №3). Вес частей в (б): LPQ и BSIF – 1 (квадрат нормы на блок, как
 * у частей (П2)), MS-LBP – MsLbpFeature.joinScale (полные нормы частей равны). Режимы pool-feat2, feret-feat2, gt-feat2: векторы
 * 1 − cos (XBaseVec.job – те же блоки, модели и пробы, что пачка 1) по вариантам в каталог xbase_&lt;база&gt;_feat2 (файл на блок,
 * признак и вариант – контрольная точка); база сверяется побайтно с векторами пачки 1 и со строкой базы abl2. pool-feat2 – выбор по
 * пулу (pool-rule: база в пределах 1 п. п. – база, иначе первый по простоте – порядок номеров вариантов). feret-feat2, gt-feat2 –
 * база и выбранные варианты: β = 1 по кандидатам, слияние w = 0,75, П3 при (Т2), условие Δ (GT внутри).
 *
 * @author ssv
 */
final class XBaseFeat2 {

    static final int NV = 9;
    static final String SELECT = "xbase_pool_feat2_select.tsv";
    static final double W_FUSE = 0.75;
    /** Порядок по простоте: (а) раньше (б), меньшая размерность раньше; при равной – LPQ, BSIF 7×7, BSIF 9×9. */
    static final String[] KIND = {"база", "LPQ", "BSIF окно 7×7", "BSIF окно 9×9", "MS-LBP", "LPQ", "BSIF окно 7×7", "BSIF окно 9×9", "MS-LBP"};
    static final int BSIF_BITS = 8;

    private XBaseFeat2() {
    }

    /** Пробный режим FACES_FEAT2_SMOKE: 0 – нет; 1 – малые данные; 2 – первая группа в полном размере – замер. */
    static int smoke() {
        String v = System.getenv("FACES_FEAT2_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    static int grid(int f) {
        return XBasePoem.grid(f);
    }

    static boolean joined(int v) {
        return v >= 5;
    }

    static String vname(int f, int v) {
        int g = grid(f);
        if (v == 0) return String.format(Locale.ROOT, "база (П2) %d×%d", g, g);
        String part = v == 4 || v == 8 ? "MS-LBP 5 точек × 5 масштабов" : String.format(Locale.ROOT, "%s, сетка %d×%d", KIND[v], g, g);
        return joined(v) ? "(б) (П2) + " + part : "(а) " + part;
    }

    static int[] dims(int f) {
        int g = grid(f);
        int p2 = XBasePoem.dims(f)[XBasePoem.NV - 2];
        int lpq = LpqFeature.length(g);
        int bsif = g * g * (1 << BSIF_BITS);
        int ms = MsLbpFeature.LENGTH;
        return new int[] {p2, lpq, bsif, bsif, ms, p2 + lpq, p2 + bsif, p2 + bsif, p2 + ms};
    }

    // ---------------------------------------------------------------- признаки

    /** Общее для признаков: фильтры BSIF, кадры k = 2, параметры Тан – Триггса (кадр «а» и k = 2). */
    static final class Feats {
        BsifFeature b7;
        BsifFeature b9;
        Map<String, FrameK2.F> k2;
        IlluminationNorm.NormParams params;
        IlluminationNorm.NormParams paramsK2;
        int threads;
    }

    static IlluminationNorm.NormParams scaledTt(IlluminationNorm.NormParams p, int k) {
        return new IlluminationNorm.NormParams(p.claheClip(), p.claheTile(), p.gamma(), p.sigma0() * k, p.sigma1() * k, p.alpha(), p.tau());
    }

    /** Часть варианта v (без (П2)) по кадру «а» a и кадру k = 2 c (уже отражённым при mirror), корни гистограмм. */
    static double[] part(Feats fs, int f, int v, Mat a, Mat c, double[][] pts) {
        int kind = joined(v) ? v - 4 : v;
        IlluminationNorm norm = XBaseDates.NORMS[f];
        if (kind == 4) {
            double[] img = PoemFeature.input(c, norm, fs.paramsK2);
            return MsLbpFeature.feature(img, c.cols(), c.rows(), pts);
        }
        double[] img = PoemFeature.input(a, norm, fs.params);
        int w = a.cols();
        int h = a.rows();
        return LbpFeature.sqrt(switch (kind) {
            case 1 -> LpqFeature.histogram(img, w, h, grid(f));
            case 2 -> fs.b7.histogram(img, w, h, grid(f));
            case 3 -> fs.b9.histogram(img, w, h, grid(f));
            default -> throw new IllegalArgumentException("feat2: вариант " + v);
        });
    }

    /** Признак варианта v ≥ 1: (а) – часть; (б) – (П2), затем часть · вес. */
    static double[] feature(Feats fs, int f, int v, Mat a, Mat c, double[][] pts) {
        double[] p = part(fs, f, v, a, c, pts);
        if (!joined(v)) return p;
        double[] p2 = XBasePoem.feature(a, f, XBaseVec.P2, fs.params);
        double sc = v == 8 ? MsLbpFeature.joinScale(grid(f)) : 1;
        double[] out = Arrays.copyOf(p2, p2.length + p.length);
        for (int i = 0; i < p.length; i++) out[p2.length + i] = p[i] * sc;
        return out;
    }

    /** Признаки варианта v (v = 0 – как XBaseVec: XBasePoem.vectors (П2)); отражения – FaceMirror (кадр «а» и k = 2). */
    static XBaseEval vectors(Map<String, Mat> fr, Feats fs, int f, int v) {
        if (v == 0) return XBasePoem.vectors(fr, fs.params, f, new int[] {XBaseVec.P2})[XBaseVec.P2];
        Map<String, double[]> vec = new ConcurrentHashMap<>();
        Map<String, double[]> mir = new ConcurrentHashMap<>();
        boolean ms = v == 4 || v == 8;
        ForkJoinPool pool = new ForkJoinPool(fs.threads);
        try {
            pool.submit(() -> fr.entrySet().parallelStream().forEach(e -> {
                Mat a = e.getValue();
                Mat am = FaceMirror.mirror(a);
                FrameK2.F k = ms ? fs.k2.get(e.getKey()) : null;
                if (ms && k == null) throw new IllegalStateException("feat2: нет кадра k = 2 " + e.getKey());
                Mat cm = ms ? FaceMirror.mirror(k.frame()) : null;
                vec.put(e.getKey(), feature(fs, f, v, a, ms ? k.frame() : null, ms ? k.pts() : null));
                mir.put(e.getKey(), feature(fs, f, v, am, cm, ms ? FrameK2.mirrorPts(k.pts()) : null));
                am.release();
                if (cm != null) cm.release();
            })).get();
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("feat2: признаки " + e, e);
        } finally {
            pool.shutdown();
        }
        XBaseEval x = new XBaseEval();
        x.vec.put(V.L1, new HashMap<>(vec));
        x.mir.put(V.L1, new HashMap<>(mir));
        return x;
    }

    static Feats feats(Setup u, FarExport.Loaded export, Path exportDir, SettingsStore settings, boolean needK2) throws IOException {
        Feats fs = new Feats();
        String bd = System.getenv("FACES_BSIF_DIR");
        if (bd == null || bd.isBlank()) bd = System.getProperty("faces.bsif.dir");
        if (bd == null || bd.isBlank()) throw new IllegalStateException("feat2: задайте faces.bsif.dir / FACES_BSIF_DIR (архив фильтров BSIF)");
        fs.b7 = BsifFeature.load(Paths.get(bd), 7, BSIF_BITS);
        fs.b9 = BsifFeature.load(Paths.get(bd), 9, BSIF_BITS);
        fs.params = u.tpl.params;
        fs.paramsK2 = scaledTt(u.tpl.params, FrameK2.K);
        fs.threads = u.tpl.threads;
        if (needK2) {
            long t = System.nanoTime();
            fs.k2 = FrameK2.load(u.fr.keySet(), u.su, u.su.dir, export, exportDir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
            System.out.printf(Locale.ROOT, "feat2: кадры k = 2 – %d (%.0f с)%n", fs.k2.size(), (System.nanoTime() - t) / 1e9);
        }
        return fs;
    }

    // ---------------------------------------------------------------- векторы

    static Path dir(Setup u, int sm) {
        return u.out.resolve("xbase_" + u.base + "_feat2" + (sm == 0 ? "" : "_smoke" + sm));
    }

    static Path file(Path dir, Blk b, int f, int v) {
        return dir.resolve(b.name() + "_f" + f + "_v" + v + ".tsv.gz");
    }

    /** Тело файла векторов без двух строк шапки. */
    static String body(Path p) throws IOException {
        StringBuilder b = new StringBuilder();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(p), 1 << 16), StandardCharsets.UTF_8))) {
            in.readLine();
            in.readLine();
            for (String l = in.readLine(); l != null; l = in.readLine()) b.append(l).append('\n');
        }
        return b.toString();
    }

    /**
     * Векторы вариантов vs обоих признаков по всем блокам (готовые файлы с тем же коммитом пропускаются); база – сверка с abl2 и с
     * векторами пачки 1 побайтно. Возвращает журнал сверок и время по вариантам.
     */
    static String vectors(Setup u, Feats fs, int[] vs, int sm, double[][] sec) throws IOException {
        Path vd = dir(u, sm);
        Files.createDirectories(vd);
        String commit = FarMethods.commit();
        String head = "# Векторы 1 − cos (XBaseFeat2 " + u.base + "-feat2): галерея, когорта в W (" + String.join(",", u.cohortIn) + "), когорта вне W ("
                + String.join(",", u.cohortOut) + ")\n# Код: коммит " + commit + "\n";
        Map<String, String> abl2 = sm == 1 ? Map.of() : XBaseSubsp.blocks(u.abl2Scores);
        // Векторы пачки 1 – полного прогона (блоки пробного режима 2 – те же).
        Path p1 = XBaseVec.vecDir(u, 0);
        StringBuilder log = new StringBuilder();
        for (int f = 1; f >= 0; f--) {
            for (int v : vs) {
                List<Blk> todo = new ArrayList<>();
                for (Blk b : u.blocks) {
                    Path p = file(vd, b, f, v);
                    if (Files.exists(p)) {
                        if (!XBaseVec.readHead(p).contains(commit)) throw new IllegalStateException(u.base + "-feat2: " + p + " – другой коммит кода; удалите каталог " + vd);
                        continue;
                    }
                    todo.add(b);
                }
                if (todo.isEmpty()) continue;
                long t0 = System.nanoTime();
                XBaseEval x = vectors(u.fr, fs, f, v);
                double tf = (System.nanoTime() - t0) / 1e9;
                System.out.printf(Locale.ROOT, "%s-feat2: f%d v%d признаки (%.0f с)%n", u.base, f, v, tf);
                int step = Math.max(1, Math.min(4, todo.size()));
                for (int i = 0; i < todo.size(); i += step) {
                    List<List<Callable<Model>>> jobs = new ArrayList<>();
                    for (Blk b : todo.subList(i, Math.min(todo.size(), i + step))) {
                        jobs.add(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
                    }
                    long ts = System.nanoTime();
                    List<List<Model>> done = XBaseVec.run(jobs, u.tpl.threads);
                    for (int k = 0; k < done.size(); k++) {
                        Blk b = todo.get(i + k);
                        String bn = String.format(b.abl2(), XBaseRule.FEATS[f]) + XBaseAbl2.rname(f, 0);
                        String name = v == 0 ? bn : String.format(b.abl2(), XBaseRule.FEATS[f]) + vname(f, v);
                        if (v == 0 && sm != 1) {
                            XBaseReg.check(bn, XBaseVec.lines(done.get(k)), abl2, u.abl2Scores);
                            StringWriter sw = new StringWriter();
                            XBaseVec.write(sw, bn, done.get(k));
                            Path ref = XBaseVec.vecFile(p1, b, f);
                            if (!Files.exists(ref)) throw new IllegalStateException(u.base + "-feat2: для сверки базы нужен " + ref + " (" + u.base + "-vec)");
                            if (!sw.toString().equals(body(ref))) throw new IllegalStateException(u.base + "-feat2: векторы базы ≠ " + ref);
                            log.append(bn).append(": совпало с ").append(u.abl2Scores.getFileName()).append(" и ").append(p1.getFileName()).append('/')
                                    .append(ref.getFileName()).append('\n');
                        }
                        Path p = file(vd, b, f, v);
                        Writer w = XBaseVec.writer(p, head);
                        XBaseVec.write(w, name, done.get(k));
                        XBaseVec.finish(w, p);
                    }
                    double tb = (System.nanoTime() - ts) / 1e9;
                    sec[f][v] += tb;
                    System.out.printf(Locale.ROOT, "%s-feat2: f%d v%d блоки %d–%d (%.0f с)%n", u.base, f, v, i + 1, i + jobs.size(), tb);
                }
                sec[f][v] += tf;
            }
        }
        return log.toString();
    }

    static List<Model> read(Path dir, Blk b, int f, int v) throws IOException {
        return XBaseVec.read(file(dir, b, f, v)).values().iterator().next();
    }

    // ---------------------------------------------------------------- режимы

    static final String NOTE = "Пачка 2 (признаки). Конвейер – принятая конфигурация: MLDA с пулом FERET, пол c = 1, 1 − cos по всем осям, β = 1 (d₁²/d₂); "
            + "L1-tt-xs-F – модель X / X+mirror по |r| пробы (r0 = 0,12),\nL1-xs-F – всегда X+mirror; слияние ℓ(g) = 0,75 · ℓ_tt + 0,25 · ℓ_xs (приведение по «чужим» "
            + "(Т2), в пуле – по контрольным чужим блока). Вход признака – как у кандидата (L1-xs-F – кадр,\nL1-tt-xs-F – Тан – Триггс; у MS-LBP – кадр k = 2, Тан – "
            + "Триггс с σ₀ = 2, σ₁ = 4). LPQ: M = 7, a = 1/7, ρ = 0,9, декорреляция (Ahonen и др., ICPR 2008); BSIF: фильтры авторов\n(ICA, естественные изображения), "
            + "8 бит, коды только у пикселей с окном целиком в кадре (у авторов – циклический край); MS-LBP: 5 точек YuNet в кадре k = 2, 5 масштабов\n(шаг 1/√2, "
            + "INTER_AREA), окно 40×40, 4×4 ячейки, LBP u2 (8, 2). Сетка LPQ и BSIF – как у кандидата. (б): (П2), затем часть; вес LPQ и BSIF – 1, MS-LBP – √(2·G² / "
            + "400).\n";

    static void run(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        int sm = smoke();
        Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, sm);
        boolean pool = base.equals("pool");
        Map<String, Map<String, Integer>> select = null;
        int[] vs;
        if (pool) {
            vs = new int[NV];
            for (int v = 0; v < NV; v++) vs[v] = v;
        } else {
            Path sf = (base.equals("gt") ? u.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : u.out).resolve(selectName(sm));
            select = readSelect(sf);
            TreeSet<Integer> need = new TreeSet<>(List.of(0));
            for (String set : base.equals("gt") ? List.of("GT") : List.of("FERET-1", "FERET-2")) need.addAll(select.get(set).values());
            vs = need.stream().mapToInt(Integer::intValue).toArray();
        }
        boolean needK2 = Arrays.stream(vs).anyMatch(v -> v == 4 || v == 8);
        Feats fs = feats(u, export, dir, settings, needK2);
        double[][] sec = new double[2][NV];
        long t0 = System.nanoTime();
        String log = vectors(u, fs, vs, sm, sec);
        double total = (System.nanoTime() - t0) / 1e9;
        for (Mat m : u.fr.values()) m.release();
        if (fs.k2 != null) for (FrameK2.F k : fs.k2.values()) k.frame().release();
        StringBuilder t = new StringBuilder();
        t.append("Пачка 2: признаки LPQ, BSIF, многомасштабный LBP – ").append(base).append(" (XBaseEval ").append(base).append("-feat2)")
                .append(sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm).append('\n');
        t.append("Код: коммит ").append(FarMethods.commit()).append("; векторы – ").append(dir(u, sm)).append('\n').append(NOTE);
        t.append("\n=== Размерности (L1-xs-F / L1-tt-xs-F) ===\n");
        for (int v = 0; v < NV; v++) t.append(String.format(Locale.ROOT, "%d | %s / %s | %d / %d%n", v, vname(0, v), vname(1, v), dims(0)[v], dims(1)[v]));
        t.append("\n=== Сверка базы: строки d₁/d₂ = строка базы abl2, векторы = векторы пачки 1 (побайтно) ===\n").append(sm == 1 ? "(в пробном режиме 1 не делается)\n" : log.isEmpty() ? "(блоки базы – из каталога, сверены при записи)\n" : log);
        t.append("\n=== Время (признаки и модели, без блоков из каталога; потоков ").append(u.tpl.threads).append(") ===\n");
        for (int f = 1; f >= 0; f--) for (int v : vs) t.append(String.format(Locale.ROOT, "%s %s: %.0f с%n", XBaseRule.FEATS[f], vname(f, v), sec[f][v]));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с%n", total));
        if (sm == 2) {
            // Доля замера: как XBaseVec (пул – 2 блока из 10 набора GT; FERET и GT – первая группа).
            double frac = pool ? 1.0 / (10.0 / XBaseNap.SMOKE2_BLOCKS * (1 + XBaseNap.FERET_TO_GT)) : base.equals("feret") ? (2143.0 + 1247) / (2143 + 1365 + 1247 + 984)
                    : (2843.0 + 2573) / (2843 + 2172 + 2573 + 1989);
            t.append(String.format(Locale.ROOT, "Замер: %.0f с; оценка полного прогона: %.0f с (%.1f ч).%n", total, total / frac, total / frac / 3600));
        }
        if (pool) poolReport(u, t, sm, out);
        else ownReport(u, base, select, vs, t, sm);
    }

    static String selectName(int sm) {
        return sm == 0 ? SELECT : SELECT.replace(".tsv", "_smoke" + sm + ".tsv");
    }

    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("feat2: нет файла выбора " + f + " (сначала pool-feat2)");
        List<String> feats = List.of(XBaseRule.FEATS);
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] p = l.split("\t");
            int fi = feats.indexOf(p[1]);
            int v = Integer.parseInt(p[2]);
            if (fi < 0 || v < 0 || v >= NV || !vname(fi, v).equals(p[3])) throw new IllegalStateException("feat2: вариант «" + l + "» не из сетки");
            out.computeIfAbsent(p[0], k -> new LinkedHashMap<>()).put(p[1], v);
        }
        return out;
    }

    static List<List<String>> noParts() {
        List<List<String>> n = new ArrayList<>();
        for (int p = 0; p < FeiMethods.CV_PARTS; p++) n.add(List.of());
        return n;
    }

    /** Критерий пула строки (β = 1) по блокам bi: модели признака f. */
    static XBaseRule.Sep poolSep(Setup u, List<Integer> bi, List<String> lines, int f, boolean fuse) {
        List<FeiDecompose.Cfg> cf = new ArrayList<>();
        for (int k = 0; k < bi.size(); k++) {
            Blk b = u.blocks.get(bi.get(k));
            XBaseEval.Gt g = fuse ? b.g() : XBaseAbl3.mirrorGt(b.g(), f == 0);
            cf.add(XBaseEval.parse(g, EnumSet.of(V.L1), b.gal(), b.probes(), noParts(), lines.get(k)).cfg.get(V.L1));
        }
        return XBaseRule.sep(List.of(XBaseRule.merge(cf)), FeiBootstrap.unit(u.persons), new int[0][]);
    }

    static void poolReport(Setup u, StringBuilder t, int sm, Path out) throws IOException {
        Path vd = dir(u, sm);
        int[] clamped = new int[2];
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseFeat2 pool-feat2): набор, признак, номер, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder body = new StringBuilder();
        StringBuilder gtPart = new StringBuilder();
        StringBuilder summary = new StringBuilder("\n=== Выбор по пулу (критерий – среднее ошибок своих при FAR 1, 2, 5 %; база в пределах 1 п. п. – база) ===\n");
        for (String grp : u.groups) {
            int start = body.length();
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            body.append("\n\n######## Набор ").append(grp).append(" (").append(u.setSize.get(grp)).append(" человек) ########\n");
            body.append("вариант | признак | ошибка своих при FAR 1 % | 2 % | 5 % | 10 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | контрольные\n");
            int[] ch = new int[2];
            List<List<List<Model>>> chosen = new ArrayList<>(List.of(new ArrayList<>(), new ArrayList<>()));
            List<List<List<Model>>> baseM = new ArrayList<>(List.of(new ArrayList<>(), new ArrayList<>()));
            for (int f = 1; f >= 0; f--) {
                double[] crit = new double[NV];
                for (int v = 0; v < NV; v++) {
                    List<String> ls = new ArrayList<>();
                    for (int k : bi) ls.add(XBaseVec.normLines(read(vd, u.blocks.get(k), f, v), null, clamped));
                    XBaseRule.Sep sp = poolSep(u, bi, ls, f, false);
                    crit[v] = sp.crit();
                    body.append(XBaseAbl3.poolRow(vname(f, v), XBaseRule.FEATS[f], sp)).append('\n');
                }
                ch[f] = XBaseNap.choose(crit);
                body.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база – %.1f %%; лучший %.1f %%)%n", grp, XBaseRule.FEATS[f], vname(f, ch[f]),
                        crit[ch[f]], crit[0], Arrays.stream(crit).min().orElseThrow()));
                summary.append(String.format(Locale.ROOT, "%s %s: база %.1f %% → %s %.1f %%%n", grp, XBaseRule.FEATS[f], crit[0], vname(f, ch[f]), crit[ch[f]]));
                sel.append(grp).append('\t').append(XBaseRule.FEATS[f]).append('\t').append(ch[f]).append('\t').append(vname(f, ch[f])).append('\n');
                for (int k : bi) {
                    chosen.get(f).add(read(vd, u.blocks.get(k), f, ch[f]));
                    baseM.get(f).add(read(vd, u.blocks.get(k), f, 0));
                }
            }
            // Слияние w = 0,75: база и выбранные (справочно).
            for (int which = 0; which < 2; which++) {
                List<List<List<Model>>> ms = which == 0 ? baseM : chosen;
                List<String> ls = new ArrayList<>();
                for (int k = 0; k < bi.size(); k++) {
                    Blk b = u.blocks.get(bi.get(k));
                    ls.add(XBaseVec.fuseLines(ms.get(1).get(k), ms.get(0).get(k), W_FUSE, XBaseVec.fuseStats(ms.get(1).get(k), true), XBaseVec.fuseStats(ms.get(0)
                            .get(k), true), b.g().rOf, b.g().r0, clamped));
                }
                XBaseRule.Sep sp = poolSep(u, bi, ls, 1, true);
                body.append(XBaseAbl3.poolRow("слияние w = 0,75, " + (which == 0 ? "база" : "выбранные"), "оба", sp)).append('\n');
                summary.append(String.format(Locale.ROOT, "%s слияние w = 0,75, %s: %.1f %%%n", grp, which == 0 ? "база" : "выбранные", sp.crit()));
            }
            if (grp.equals("GT")) gtPart.append(body.substring(start));
        }
        t.append(summary).append(body);
        if (sm == 0) Files.writeString(u.out.resolve(SELECT), sel.toString());
        else Files.writeString(u.out.resolve(selectName(sm)), sel.toString());
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        FarMethods.write(u.out.resolve("xbase_pool_feat2" + sfx + ".txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_feat2" + sfx + "_gt.txt"), new StringBuilder(t.substring(0, t.indexOf(summary.toString()))).append(summary).append(gtPart));
        System.out.print(t);
    }

    /** Набор пула, по которому выбирается вариант блока: FERET – проход 1 → FERET-1, проход 2 → FERET-2; GT – GT. */
    static String setOf(String base, Blk b) {
        return base.equals("gt") ? "GT" : b.name().endsWith("_p0") ? "FERET-1" : "FERET-2";
    }

    static void ownReport(Setup u, String base, Map<String, Map<String, Integer>> select, int[] vs, StringBuilder t, int sm) throws IOException {
        Path vd = dir(u, sm);
        int[] clamped = new int[2];
        StringBuilder body = new StringBuilder();
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        StringBuilder sl = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        for (Blk b : u.blocks) {
            if (!b.group().equals(u.groups.get(0))) continue;
            for (int f = 1; f >= 0; f--) sl.append(String.format(Locale.ROOT, "%s, %s: %s%n", b.name(), XBaseRule.FEATS[f], vname(f, select.get(setOf(base, b)).get(XBaseRule.FEATS[f]))));
        }
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            boolean within = grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            // [f][v][k] модели.
            Map<Integer, List<List<Model>>>[] ms = new Map[2];
            for (int f = 0; f < 2; f++) {
                ms[f] = new LinkedHashMap<>();
                for (int v : vs) {
                    List<List<Model>> l = new ArrayList<>();
                    for (int k : bi) l.add(read(vd, u.blocks.get(k), f, v));
                    ms[f].put(v, l);
                }
            }
            int[][] chosen = new int[2][bi.size()];
            for (int f = 0; f < 2; f++) for (int k = 0; k < bi.size(); k++) chosen[f][k] = select.get(setOf(base, u.blocks.get(bi.get(k)))).get(XBaseRule.FEATS[f]);
            for (int f = 1; f >= 0; f--) {
                Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
                List<String[]> prs = new ArrayList<>();
                String bl = grp + " " + XBaseRule.FEATS[f] + " " + vname(f, 0);
                List<String> labels = new ArrayList<>();
                List<int[]> rows = new ArrayList<>();
                for (int v : vs) {
                    int[] vv = new int[bi.size()];
                    Arrays.fill(vv, v);
                    labels.add(grp + " " + XBaseRule.FEATS[f] + " " + vname(f, v));
                    rows.add(vv);
                }
                if (base.equals("feret") && chosen[f][0] != chosen[f][bi.size() - 1]) {
                    labels.add(grp + " " + XBaseRule.FEATS[f] + " выбор по пулу (проход 1 – " + vname(f, chosen[f][0]) + ", проход 2 – " + vname(f, chosen[f][bi.size() - 1]) + ")");
                    rows.add(chosen[f].clone());
                }
                for (int r = 0; r < labels.size(); r++) {
                    List<XBaseThr.Unit> us = new ArrayList<>();
                    for (int k = 0; k < bi.size(); k++) {
                        Blk b = u.blocks.get(bi.get(k));
                        String l = XBaseVec.normLines(ms[f].get(rows.get(r)[k]).get(k), null, clamped);
                        us.add(new XBaseThr.Unit(XBaseAbl3.mirrorGt(b.g(), f == 0), b.gal(), b.pro(), b.parts(), l, null, t2));
                    }
                    String lb = labels.get(r);
                    XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                    outs.put(lb, o);
                    if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                    if (r > 0) prs.add(new String[] {lb, bl});
                }
                XBaseLbp.tables(body, prefix + ", " + XBaseRule.FEATS[f], outs, u.persons, u.tpl, prs, t2);
                if (within) for (String[] p : prs) {
                    double[] d = XBaseVt.delta(outs.get(p[0]), outs.get(bl), XBaseThr.WAYS[0], u.persons, u.tpl);
                    dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                            : "НЕ выполнено"));
                }
            }
            // Слияние w = 0,75: база и выбор по пулу.
            Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
            List<String[]> pf = new ArrayList<>();
            String fb = grp + " слияние w = 0,75, база";
            for (int which = 0; which < 2; which++) {
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (int k = 0; k < bi.size(); k++) {
                    Blk b = u.blocks.get(bi.get(k));
                    List<Model> tt = ms[1].get(which == 0 ? 0 : chosen[1][k]).get(k);
                    List<Model> xs = ms[0].get(which == 0 ? 0 : chosen[0][k]).get(k);
                    us.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), XBaseVec.fuseLines(tt, xs, W_FUSE, XBaseVec.fuseStats(tt, false), XBaseVec.fuseStats(xs,
                            false), b.g().rOf, b.g().r0, clamped), null, t2));
                }
                String lb = which == 0 ? fb : grp + " слияние w = 0,75, выбор по пулу";
                XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                of.put(lb, o);
                if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                if (which == 1) pf.add(new String[] {lb, fb});
            }
            XBaseLbp.tables(body, prefix + " – слияние кандидатов w = 0,75", of, u.persons, u.tpl, pf, t2);
            if (within) {
                double[] d = XBaseVt.delta(of.get(grp + " слияние w = 0,75, выбор по пулу"), of.get(fb), XBaseThr.WAYS[0], u.persons, u.tpl);
                dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", grp + " слияние, выбор по пулу − база", d[0],
                        d[1], d[1] <= 3 ? "выполнено" : "НЕ выполнено"));
            }
        }
        t.append(sl).append(hold).append(base.equals("gt") ? dl : "").append(body);
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        FarMethods.write(u.out.resolve("xbase_" + base + "_feat2" + sfx + ".txt"), t);
        System.out.print(t);
    }
}
