package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.zip.GZIPInputStream;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Подпространство человека в пространстве L1 (PLAN.md п. 5.5, гипотеза и очередь п. (3); задание №3 от 15.10.2026, план
 * подтверждён Хозяином). Модель MLDA обучается как сейчас; поверх неё человек c — аффинное подпространство: μ_c — центр
 * класса (как сейчас, с отражёнными снимками шаблона у модели +mirror), B_c — первые k правых сингулярных векторов SVD матрицы
 * проекций реальных снимков шаблона (без отражений), центрированных на μ_c; расстояние пробы — ‖r − B_c B_cᵀ r‖, r = y − μ_c
 * (k = 0 — прежнее расстояние до центра). Справочно «k = 1м» — B_c по снимкам шаблона вместе с отражениями. Решение — d₁/d₂,
 * модель по позе пробы, порог П2/П3 по «чужим» галереи (модели частей — с тем же k), свой на проход. Режимы feret-subsp
 * (половины №2: G2 – основная, 3 даты, G2' и G1 – проверка k = 0, 1) и gt-subsp (GT межсессионно N = 6 и внутри сессии N = 3,
 * правило №8, seed faces.seed + 9800; пул FERET).
 *
 * @author ssv
 */
final class XBaseSubsp {

    /** Вариант подпространства: подпись, k, базис с отражениями. */
    record Kv(String label, int k, boolean mirrorBasis) {}

    private XBaseSubsp() {
    }

    /** Модель: прежняя LDA (центры) и базисы подпространств (строки — векторы базиса) по людям галереи. */
    static final class SubModel implements GalleryScorer.ScoreModel {
        final LdaModel lm;
        final double[][][] basis;

        SubModel(LdaModel lm, double[][][] basis) {
            this.lm = lm;
            this.basis = basis;
        }

        @Override
        public double[] scores(double[] x) {
            double[] y = lm.embed(x);
            double[] s = new double[lm.centers().length];
            for (int c = 0; c < s.length; c++) {
                double[] mu = lm.centers()[c];
                if (mu == null) {
                    s[c] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double[] r = new double[y.length];
                for (int i = 0; i < r.length; i++) r[i] = y[i] - mu[i];
                for (double[] b : basis[c]) {
                    double dot = 0;
                    for (int i = 0; i < r.length; i++) dot += b[i] * r[i];
                    for (int i = 0; i < r.length; i++) r[i] -= dot * b[i];
                }
                double d2 = 0;
                for (double v : r) d2 += v * v;
                s[c] = Math.sqrt(d2);
            }
            return s;
        }

        @Override
        public String info() {
            return lm.info();
        }
    }

    /** Базис k векторов по точкам pts (центрированы на mu) и сингулярные числа; при k = 0 — пусто. */
    static double[][] basis(List<double[]> pts, double[] mu, int k, double[] sigmaOut) {
        if (k == 0) return new double[0][];
        double[][] m = new double[pts.size()][];
        for (int i = 0; i < m.length; i++) {
            m[i] = new double[mu.length];
            for (int j = 0; j < mu.length; j++) m[i][j] = pts.get(i)[j] - mu[j];
        }
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(m, false));
        double[] sv = svd.getSingularValues();
        if (sigmaOut != null) System.arraycopy(sv, 0, sigmaOut, 0, Math.min(sv.length, sigmaOut.length));
        double[][] v = svd.getV().getData();
        int kk = Math.min(k, sv.length);
        double[][] out = new double[kk][mu.length];
        for (int a = 0; a < kk; a++) for (int j = 0; j < mu.length; j++) out[a][j] = v[j][a];
        return out;
    }

    /** Модель MLDA (как XBaseEval.fit для V.L1) и модели подпространств вариантов kvs. */
    static Map<String, GalleryScorer.ScoreModel> models(XBaseEval x, boolean mirror, List<List<String>> gallery, List<List<String>> cohort, List<Kv> kvs,
                                                        double[] diag) {
        List<List<double[]>> cls = new ArrayList<>();
        for (List<List<String>> group : List.of(gallery, cohort)) {
            for (List<String> keys : group) {
                List<double[]> list = new ArrayList<>();
                for (String k : keys) list.add(x.vec.get(V.L1).get(k));
                if (mirror) for (String k : keys) list.add(x.mir.get(V.L1).get(k));
                cls.add(list);
            }
        }
        LdaModel lm = (LdaModel) new MldaScorer().fit(cls, gallery.size());
        Map<String, GalleryScorer.ScoreModel> out = new LinkedHashMap<>();
        for (Kv kv : kvs) {
            if (kv.k() == 0) {
                out.put(kv.label(), lm);
                continue;
            }
            double[][][] b = new double[gallery.size()][][];
            for (int c = 0; c < gallery.size(); c++) {
                List<double[]> pts = new ArrayList<>();
                for (String k : gallery.get(c)) pts.add(lm.embed(x.vec.get(V.L1).get(k)));
                if (kv.mirrorBasis()) for (String k : gallery.get(c)) pts.add(lm.embed(x.mir.get(V.L1).get(k)));
                double[] sig = new double[pts.size()];
                b[c] = basis(pts, lm.centers()[c], kv.k(), sig);
                // Диагностика k = 1 (без отражений в базисе): доля энергии первого вектора и |cos| с «снимок − отражение».
                if (diag != null && kv.k() == 1 && !kv.mirrorBasis()) {
                    double tot = 0;
                    for (double s : sig) tot += s * s;
                    double cos = 0;
                    for (String k : gallery.get(c)) {
                        double[] d = lm.embed(x.vec.get(V.L1).get(k));
                        double[] e = lm.embed(x.mir.get(V.L1).get(k));
                        double dot = 0;
                        double n = 0;
                        for (int j = 0; j < d.length; j++) {
                            double t = d[j] - e[j];
                            dot += t * b[c][0][j];
                            n += t * t;
                        }
                        cos += n > 0 ? Math.abs(dot) / Math.sqrt(n) : 0;
                    }
                    synchronized (diag) {
                        diag[0] += tot > 0 ? sig[0] * sig[0] / tot : 0;
                        diag[1] += cos / gallery.get(c).size();
                        diag[2]++;
                    }
                }
            }
            out.put(kv.label(), new SubModel(lm, b));
        }
        return out;
    }

    static String lines(XBaseEval x, String method, int s, GalleryScorer.ScoreModel m, List<String> gallery, List<Probe> probes) {
        StringBuilder t = new StringBuilder();
        for (Probe p : probes) {
            t.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, m.scores(x.vec.get(V.L1).get(p.key())), false));
        }
        return t.toString();
    }

    /**
     * Строки оценок вариантов (как XBaseEval.fitLines для V.L1, тот же порядок): основная модель и модели без части q, оба
     * зеркала; один раз обученная MLDA — для всех k.
     */
    static Map<String, String> fitLines(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts,
                                        int s, List<Kv> kvs, double[] diag) {
        List<String> own = gt.own;
        List<List<String>> cohort = gt.cohort;
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        List<Callable<Map<String, String>>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> {
                Map<String, String> r = new LinkedHashMap<>();
                models(x, m, fullGallery, cohort, kvs, diag).forEach((k, md) -> r.put(k, lines(x, method, s, md, own, mainProbes)));
                return r;
            });
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                tasks.add(() -> {
                    Map<String, String> r = new LinkedHashMap<>();
                    models(x, m, gk, cohort, kvs, null).forEach((k, md) -> r.put(k, lines(x, method, s, md, g, pr)));
                    return r;
                });
            }
        }
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(gt.threads);
        Map<String, StringBuilder> all = new LinkedHashMap<>();
        for (Kv kv : kvs) all.put(kv.label(), new StringBuilder());
        try {
            for (java.util.concurrent.Future<Map<String, String>> f : ex.invokeAll(tasks)) f.get().forEach((k, v) -> all.get(k).append(v));
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseSubsp: " + e, e);
        } finally {
            ex.shutdown();
        }
        Map<String, String> out = new LinkedHashMap<>();
        all.forEach((k, v) -> out.put(k, v.toString()));
        return out;
    }

    /** Строки L1 (5-lbp-mlda и +mirror) блока, отсортированные (сверка k = 0 с прежним файлом оценок). */
    static List<String> l1Rows(String lines) {
        List<String> out = new ArrayList<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            String m = l.substring(0, l.indexOf('\t'));
            if (m.equals(V.L1.id) || m.equals(V.L1.id + FeiMethods.MIRROR)) out.add(l);
        }
        Collections.sort(out);
        return out;
    }

    static Map<String, String> blocks(Path scores) throws IOException {
        Map<String, StringBuilder> b = new LinkedHashMap<>();
        String cur = null;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(scores), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("# ")) {
                    cur = l.substring(2);
                    b.putIfAbsent(cur, new StringBuilder());
                } else if (!l.startsWith("#") && !l.startsWith("method\t") && cur != null) {
                    b.get(cur).append(l).append('\n');
                }
            }
        }
        Map<String, String> out = new LinkedHashMap<>();
        b.forEach((k, v) -> out.put(k, v.toString()));
        return out;
    }

    /** Итог: проходы (Gt, галерея, пробы, части, строки) → Res, сосредоточенность контроля (два самых частых лучших). */
    record Pass(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, String lines) {}

    static FeiThreshold.Res result(String label, List<Pass> passes, int persons, XBaseEval.Gt tpl, int[] conc) {
        List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        List<FeiThreshold.Imp> imps = new ArrayList<>();
        for (Pass p : passes) {
            List<Probe> probes = XBaseEval.probes(p.g(), p.pro());
            XBaseEval.Parsed ps = XBaseEval.parse(p.g(), EnumSet.of(V.L1), p.gal(), probes, p.parts(), p.lines());
            cfgs.add(ps.cfg.get(V.L1));
            imps.add(ps.imp.get(V.L1));
            Map<String, Row> rows = new HashMap<>();
            for (String l : p.lines().lines().toList()) {
                Row row = FeiScores.parse(l);
                rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
            }
            Map<String, Integer> best = new HashMap<>();
            for (Probe q : probes) {
                if (q.role().equals(FeiScores.OWN)) continue;
                Row row = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), p.g().rOf.get(q.key()) <= p.g().r0);
                best.merge(row.bestId(), 1, Integer::sum);
                conc[1]++;
            }
            conc[0] += best.values().stream().sorted(Collections.reverseOrder()).limit(2).mapToInt(Integer::intValue).sum();
        }
        return FeiThreshold.evaluate(label, V.L1.ratioId(), cfgs, imps, persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
    }

    static List<Kv> kvs(int kmax) {
        List<Kv> out = new ArrayList<>();
        for (int k = 0; k <= kmax; k++) out.add(new Kv("k = " + k, k, false));
        out.add(new Kv("k = 1м", 1, true));
        return out;
    }

    /** Таблицы группы: Res по подписям «постановка проекция k», пары k − (k = 0), сосредоточенность. */
    static void tables(StringBuilder t, String title, Map<String, FeiThreshold.Res> res, Map<String, int[]> conc, Map<String, double[]> diag, int persons,
                       XBaseEval.Gt tpl, List<String[]> pairs) {
        t.append("\n\n######## ").append(title).append(" ########\n");
        t.append("\n=== Сосредоточенность контроля (два самых частых лучших кандидата, по проходам) и диагностика k = 1 ===\n"
                + "строка | контрольных проб с лучшим из двух самых частых | k = 1: доля энергии первого вектора | |cos| первого вектора с «снимок − отражение»\n");
        for (String l : res.keySet()) {
            int[] c = conc.get(l);
            double[] d = diag.get(l);
            t.append(String.format(Locale.ROOT, "%s | %d/%d = %.1f %% | %s | %s%n", l, c[0], c[1], 100.0 * c[0] / c[1],
                    d == null || d[2] == 0 ? "–" : String.format(Locale.ROOT, "%.3f", d[0] / d[2]), d == null || d[2] == 0 ? "–" : String.format(Locale.ROOT, "%.3f",
                            d[1] / d[2])));
        }
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(persons), w));
        XBaseOther.tables(t, res, sep, tpl.b, pairs);
    }

    // ---------------------------------------------------------------- feret-subsp

    /** Постановка FERET: подпись, номер s (части «чужих»), свои, галерея, пробы, наибольшее k. */
    record Setting(String name, int s, List<String> own, Map<String, List<String>> gal, Map<String, List<String>> pro, int kmax) {}

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-subsp: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.g2, roAll.g2p, roAll.pro2, roAll.ctrlKeys)) m.values().forEach(used::addAll);
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();

        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);

        // 3 даты: люди с ≥ 4 датами, галерея – fa, fb первых 3 дат, пробы – даты с 4-й.
        List<String> own3 = new ArrayList<>();
        Map<String, List<String>> gal3 = new LinkedHashMap<>();
        Map<String, List<String>> pro3 = new LinkedHashMap<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) {
            if (e.getValue().size() < 4) continue;
            own3.add(e.getKey());
            List<String> g = new ArrayList<>();
            for (int j = 0; j < 3; j++) g.addAll(e.getValue().get(j));
            gal3.put(e.getKey(), XBaseFeret.sorted(g));
            List<String> pr = new ArrayList<>();
            for (int j = 3; j < e.getValue().size(); j++) pr.addAll(e.getValue().get(j));
            pro3.put(e.getKey(), XBaseFeret.sorted(pr));
        }
        List<Setting> settingsList = List.of(new Setting("G2", 3, ro.own2, ro.g2, ro.pro2, 3), new Setting("3 даты", 6, own3, gal3, pro3, 3),
                new Setting("G2'", 5, ro.own2, ro.g2p, ro.pro2, 1), new Setting("G1", 2, ro.own2, ro.g1, ro.pro2, 1));
        String[][] projs = {{"L1-xs-F", "0", "1"}, {"L1-tt-xs-F", "1", "1"}, {"L1", "0", "0"}};
        Path xs2 = out.resolve("xbase_feret_xs2_scores.tsv.gz");
        Map<String, String> xs2b = blocks(xs2);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-subsp %s; снимков %d; seed %d; r0 %s; пул %d; №8 галерея fa/fb", dir,
                su.imgs.size(), tpl.seed, tpl.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_subsp_eq"), fp);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, подпространство человека в пространстве L1 (XBaseEval feret-subsp): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("Модель: MLDA как сейчас; человек c – аффинное подпространство: μ_c – центр класса (как сейчас; у модели +mirror – с отражёнными\n"
                + "снимками шаблона), B_c – первые k правых сингулярных векторов SVD проекций реальных снимков шаблона (без отражений), центрированных\n"
                + "на μ_c; расстояние – ‖r − B_c B_cᵀ r‖, r = y − μ_c; k = 0 – прежнее расстояние до центра; «k = 1м» (справочно) – B_c по снимкам\n"
                + "шаблона вместе с отражениями. Решение – d₁/d₂, модель по позе пробы; порог П2/П3 по «чужим» галереи (модели частей – с тем же k),\n"
                + "свой на проход. Половины №2 (seed faces.seed + 9910): свои – половина, пул проекции L1-xs-F – другая половина целиком.\n");
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder lines = new StringBuilder();
        StringBuilder check = new StringBuilder("\n=== Проверка k = 0 = строка feret-xs2 (строки L1 блока, побайтно) ===\n");
        StringBuilder vol = new StringBuilder("\n=== Объём ===\nпостановка | проход | своих | снимков шаблона | проб\n");
        double fitS = 0;
        for (Setting st : settingsList) {
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            Map<String, int[]> conc = new LinkedHashMap<>();
            Map<String, double[]> diag = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            List<Kv> kvs = kvs(st.kmax());
            for (String[] pj : projs) {
                int f = Integer.parseInt(pj[1]);
                boolean usePool = pj[2].equals("1");
                Map<String, List<Pass>> passes = new LinkedHashMap<>();
                for (Kv kv : kvs) passes.put(kv.label(), new ArrayList<>());
                double[] dg = new double[3];
                for (int pp = 0; pp < 2; pp++) {
                    Set<String> mine = halves.get(pp);
                    List<String> own = st.own().stream().filter(mine::contains).toList();
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Map<String, List<String>> pro = new LinkedHashMap<>();
                    for (String p : own) {
                        gal.put(p, st.gal().get(p));
                        pro.put(p, st.pro().get(p));
                    }
                    Map<String, List<List<String>>> projPool = new TreeMap<>();
                    for (String p : halves.get(1 - pp)) projPool.put(p, pool.get(p));
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                    List<List<String>> aug = new ArrayList<>(cohort);
                    if (usePool) aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                    g.cohort = aug;
                    List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + st.s());
                    List<Probe> probes = XBaseEval.probes(g, pro);
                    Map<String, String> ls = new LinkedHashMap<>();
                    boolean all = true;
                    for (Kv kv : kvs) {
                        String r = cp.read(name(st, pj[0], pp, kv));
                        if (r == null) all = false;
                        ls.put(kv.label(), r);
                    }
                    if (!all) {
                        long t0 = System.nanoTime();
                        ls = fitLines(g, xs[f], gal, probes, parts, st.s(), kvs, dg);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        for (Kv kv : kvs) cp.save(name(st, pj[0], pp, kv), ls.get(kv.label()));
                        cp.save(name(st, pj[0], pp, null), String.format(Locale.ROOT, "%.17g\t%.17g\t%.17g%n", dg[0], dg[1], dg[2]));
                        System.out.printf(Locale.ROOT, "feret-subsp: %s %s pass %d ready (%.0f s)%n", st.s(), pj[0], pp + 1, sec);
                    } else {
                        String[] d = cp.read(name(st, pj[0], pp, null)).trim().split("\t");
                        for (int i = 0; i < 3; i++) dg[i] = Double.parseDouble(d[i]);
                    }
                    for (Kv kv : kvs) {
                        lines.append("# ").append(st.name()).append(", проекция ").append(pj[0]).append(", проход ").append(pp + 1).append(", ").append(kv.label())
                                .append('\n').append(ls.get(kv.label()));
                        passes.get(kv.label()).add(new Pass(g, gal, pro, parts, ls.get(kv.label())));
                    }
                    if (pj[0].equals("L1-xs-F")) vol.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d%n", st.name(), pp + 1, own.size(),
                            gal.values().stream().mapToInt(List::size).sum(), pro.values().stream().mapToInt(List::size).sum()));
                    // Проверка k = 0 против feret-xs2 (G1, G2' – там есть те же блоки).
                    if (st.name().equals("G1") || st.name().equals("G2'")) {
                        String bn = "проход " + (pp + 1) + ", конфигурация " + st.name() + ", строка " + pj[0];
                        String old = xs2b.get(bn);
                        if (old == null) throw new IllegalStateException("feret-subsp: нет блока «" + bn + "» в " + xs2);
                        boolean same = l1Rows(old).equals(l1Rows(ls.get("k = 0")));
                        check.append(String.format(Locale.ROOT, "%s %s проход %d: %s (%d строк)%n", st.name(), pj[0], pp + 1, same ? "совпало" : "НЕ СОВПАЛО",
                                l1Rows(old).size()));
                        if (!same) throw new IllegalStateException("feret-subsp: k = 0 не совпало с feret-xs2: " + bn);
                    }
                }
                for (Kv kv : kvs) {
                    String l = st.name() + " " + pj[0] + " " + kv.label();
                    int[] c = new int[2];
                    res.put(l, result(l, passes.get(kv.label()), persons.size(), tpl, c));
                    conc.put(l, c);
                    if (kv.k() == 1 && !kv.mirrorBasis()) diag.put(l, dg);
                    if (kv.k() > 0 || kv.mirrorBasis()) pairs.add(new String[] {l, st.name() + " " + pj[0] + " k = 0"});
                }
            }
            tables(t, "Постановка " + st.name() + (st.kmax() == 1 ? " (проверка вырожденного случая, k = 0, 1)" : ""), res, conc, diag, persons.size(), tpl, pairs);
        }
        t.insert(t.indexOf("\n\n######## "), vol + check.toString() + String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной "
                + "точки); потоков %d.%n", fitS, tpl.threads));
        XBaseOther.writeScores(out.resolve("xbase_feret_subsp_scores.tsv.gz"), "# FERET, подпространство человека (XBaseSubsp feret-subsp), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_subsp.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    static String name(Setting st, String proj, int pp, Kv kv) {
        return String.format(Locale.ROOT, "s%d_%s_p%d_%s", st.s(), proj, pp, kv == null ? "diag" : kv.label().replace(" = ", "").replace("м", "m"));
    }

    // ---------------------------------------------------------------- gt-subsp

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-subsp: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        xs[1].vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        Map<String, Mat> pf = new LinkedHashMap<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) pf.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval p0 = new XBaseEval();
        p0.vectors(pf, EnumSet.of(V.L1), g.params);
        XBaseEval p1 = new XBaseEval();
        p1.vectors(pf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : pf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(p0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(p0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(p1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(p1.mir.get(V.L1));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();

        // Галереи по правилу №8 (seed faces.seed + 9800): межс. N = 6, внутри N = 3.
        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        Object[][] modes = {{"межс.", galX, g.probeKeys, 3, "xbase_gt_scores.tsv.gz"}, {"внутри", galI, split.get(1), 2, "xbase_gt_in_scores.tsv.gz"}};
        String[][] projs = {{"L1-xs-F", "0", "1"}, {"L1-tt-xs-F", "1", "1"}, {"L1", "0", "0"}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Path eqDir = out.resolve(XBaseEval.eqDir(0, false));
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-subsp; seed %d; r0 %s; пул FERET %d", g.seed, g.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_subsp"), fp);
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, подпространство человека в пространстве L1 – GT, предварительно (XBaseEval gt-subsp)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F – Color "
                + "FERET, 173 человека (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол gt; галерея по правилу №8 (seed faces.seed + 9800): межсессионно N = 6, внутри сессии N = 3 (k ≤ 2). Модель – как feret-subsp.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET,
                g.persons.size()));
        StringBuilder lines = new StringBuilder();
        StringBuilder check = new StringBuilder("\n=== Проверка k = 0 у L1 = строка gt при faces.gallery.equal = 0 (eq9800, строки L1, побайтно) ===\n");
        double fitS = 0;
        List<List<String>> cohort0 = g.cohort;
        StringBuilder body = new StringBuilder();
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Kv> kvs = kvs((Integer) md[3]);
                List<Probe> probes = XBaseEval.probes(g, pro);
                Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
                Map<String, int[]> conc = new LinkedHashMap<>();
                Map<String, double[]> diag = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                for (String[] pj : projs) {
                    int f = Integer.parseInt(pj[1]);
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    if (pj[2].equals("1")) aug.addAll(poolCls);
                    g.cohort = aug;
                    String tag = (mode.equals("межс.") ? "x" : "i") + "_" + pj[0];
                    Map<String, String> ls = new LinkedHashMap<>();
                    boolean all = true;
                    for (Kv kv : kvs) {
                        String r = cp.read(tag + "_" + kv.label().replace(" = ", "").replace("м", "m"));
                        if (r == null) all = false;
                        ls.put(kv.label(), r);
                    }
                    double[] dg = new double[3];
                    if (!all) {
                        long t0 = System.nanoTime();
                        ls = fitLines(g, xs[f], gal, probes, parts, 0, kvs, dg);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        for (Kv kv : kvs) cp.save(tag + "_" + kv.label().replace(" = ", "").replace("м", "m"), ls.get(kv.label()));
                        cp.save(tag + "_diag", String.format(Locale.ROOT, "%.17g\t%.17g\t%.17g%n", dg[0], dg[1], dg[2]));
                        System.out.printf(Locale.ROOT, "gt-subsp: %s ready (%.0f s)%n", tag, sec);
                    } else {
                        String[] d = cp.read(tag + "_diag").trim().split("\t");
                        for (int i = 0; i < 3; i++) dg[i] = Double.parseDouble(d[i]);
                    }
                    if (pj[0].equals("L1")) {
                        Path old = eqDir.resolve((String) md[4]);
                        if (Files.exists(old)) {
                            StringBuilder ol = new StringBuilder();
                            blocks(old).values().forEach(ol::append);
                            boolean same = l1Rows(ol.toString()).equals(l1Rows(ls.get("k = 0")));
                            check.append(String.format(Locale.ROOT, "%s L1: %s (%s, %d строк)%n", mode, same ? "совпало" : "НЕ СОВПАЛО", old.getFileName(),
                                    l1Rows(ls.get("k = 0")).size()));
                            if (!same) throw new IllegalStateException("gt-subsp: k = 0 не совпало с " + old);
                        } else {
                            check.append(mode).append(" L1: файла ").append(old).append(" нет – проверка не делалась\n");
                        }
                    }
                    for (Kv kv : kvs) {
                        lines.append("# ").append(mode).append(", проекция ").append(pj[0]).append(", ").append(kv.label()).append('\n').append(ls.get(kv.label()));
                        String l = mode + " " + pj[0] + " " + kv.label();
                        int[] c = new int[2];
                        res.put(l, result(l, List.of(new Pass(g, gal, pro, parts, ls.get(kv.label()))), g.persons.size(), g, c));
                        conc.put(l, c);
                        if (kv.k() == 1 && !kv.mirrorBasis()) diag.put(l, dg);
                        if (kv.k() > 0 || kv.mirrorBasis()) pairs.add(new String[] {l, mode + " " + pj[0] + " k = 0"});
                    }
                }
                tables(body, "GT " + mode, res, conc, diag, g.persons.size(), g, pairs);
            }
        } finally {
            g.cohort = cohort0;
        }
        t.append(check).append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_subsp_scores.tsv.gz"), "# GT, подпространство человека (XBaseSubsp gt-subsp)\n# Код: коммит "
                + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_subsp.txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
