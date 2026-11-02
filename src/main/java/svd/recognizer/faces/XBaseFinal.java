package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.storage.SettingsStore;

/**
 * Итоговый прогон перед ПО (постановка – PLAN.md п. 5.5, «Постановка итогового прогона перед ПО», утверждена Хозяином 02.11.2026;
 * задание Жорика №7 от 02.11). Режим frames-b – кадры Б (FrameB) по всем базам из исходных снимков и опись SHA-256 для разд. 7
 * постановки.
 *
 * @author ssv
 */
final class XBaseFinal {

    private XBaseFinal() {
    }

    static final String MANIFEST = "frames_b_manifest.tsv";

    /**
     * frames-b: FERET (все снимки с кадром «а» в derived\a) → derived\a_b; базы экспорта (GT, FEI, MUCT, ORL, своя) → &lt;экспорт&gt;\a_b.
     * Сверка: тот же код без сглаживания повторяет кадр «а» побайтно (своя база – нет: её «а» строится по входу 0,25). Готовые файлы
     * пропускаются (продолжение после обрыва; запись – через временный файл). В конце – описи: FERET – derived\frames_b_manifest.tsv,
     * экспорт – &lt;экспорт&gt;\frames_b_manifest.tsv (кадры Б и кадры k = 2 «c», путь и SHA-256), сводка – отчёт frames_b.txt.
     */
    static void framesB(FarExport.Loaded export, Path dir, List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (FrameB.on()) throw new IllegalStateException("frames-b: без FACES_FRAMES (сверка идёт с кадрами «а»)");
        long t0 = System.nanoTime();
        StringBuilder rep = new StringBuilder("Итоговый прогон: кадры Б (XBaseEval frames-b)\nКод: коммит " + FarMethods.commit() + "\n");
        // FERET
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Path der = su.dir.resolve(FeretData.DERIVED);
        Files.createDirectories(der.resolve(FrameB.DIR));
        int nF = 0;
        int kMax = 1;
        for (String k : new TreeSet<>(su.det.keySet())) {
            double[] row = su.det.get(k);
            if (row == null || !Files.exists(der.resolve("a").resolve(k + ".png"))) continue;
            Path f = der.resolve(FrameB.DIR).resolve(k + ".png");
            nF++;
            if (Files.exists(f)) continue;
            Mat color = FeretData.decode(su.byKey.get(k).file());
            Mat plain = FrameB.frame(color, row, false);
            Mat stored = FeretData.frame(su.dir, "a", k);
            if (!FrameK2.same(plain, stored)) throw new IllegalStateException("frames-b: кадр «а» FERET не повторён " + k);
            Mat b = FrameB.frame(color, row, true);
            Mat m = FaceAlignment.similarity(FrameK2.lm(row), FaceAlignment.ownTemplate(FrameB.W, FrameB.H));
            kMax = Math.max(kMax, FrameB.k(m));
            m.release();
            FeretData.writePng(f, b);
            for (Mat x : new Mat[] {color, plain, stored, b}) x.release();
            if (nF % 250 == 0) System.out.println("frames-b: FERET " + nF);
        }
        // базы экспорта
        GalleryEvaluation.Raw raw = GalleryEvaluation.raw(settings, settings.loadFacesOwnDetectorScore(), Paths.get(settings.loadFacesModelsDir()),
                Paths.get(System.getProperty("user.dir"), "reports", "faces", "far"), false);
        Map<String, Integer> nBase = new LinkedHashMap<>();
        Map<String, Sample> expSample = new java.util.HashMap<>();
        for (Sample s : export.data().samples()) expSample.put(s.file().toString(), s);
        for (Sample s : raw.data().samples()) {
            double[] row = raw.dets().get(s.file().toString()).row();
            if (row == null) continue;
            Path id = FarExport.id(s);
            Path fa = dir.resolve("a").resolve(id + ".png");
            if (!Files.exists(fa)) throw new IllegalStateException("frames-b: нет кадра «а» экспорта " + id);
            Path f = dir.resolve(FrameB.DIR).resolve(id + ".png");
            String base = FarExport.dirName(s.base());
            nBase.merge(base, 1, Integer::sum);
            if (Files.exists(f)) continue;
            Files.createDirectories(f.getParent());
            Mat[] m = GalleryEvaluation.load(s);
            try {
                if (s.base() != GalleryEvaluation.Base.OWN) {
                    Mat plain = FrameB.frame(m[0], row, false);
                    Sample es = expSample.get(id.toString());
                    Mat stored = es == null ? null : export.frame(es, GalleryEvaluation.Variant.A);
                    if (stored == null || !FrameK2.same(plain, stored)) throw new IllegalStateException("frames-b: кадр «а» экспорта не повторён " + id);
                    plain.release();
                    stored.release();
                }
                Mat b = FrameB.frame(m[0], row, true);
                FeretData.writePng(f, b);
                b.release();
            } finally {
                GalleryEvaluation.release(m);
            }
            int n = nBase.values().stream().mapToInt(Integer::intValue).sum();
            if (n % 250 == 0) System.out.println("frames-b: экспорт " + n);
        }
        rep.append(String.format(Locale.ROOT, "Кадров Б: FERET %d (наибольшее k у построенных в этом запуске – %d); экспорт %s%n", nF, kMax, nBase));
        rep.append("Сверка «тот же код без сглаживания = кадр «а»» – побайтно у всех построенных (кроме своей базы).\n");
        rep.append(manifests(su.dir, dir));
        rep.append(String.format(Locale.ROOT, "Время: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Files.createDirectories(out);
        FarMethods.write(out.resolve("frames_b.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    /** Описи кадров Б и k = 2: FERET (a_b, c, c_points.tsv) и экспорт по базам (a_b/&lt;база&gt;, c/&lt;база&gt;); строки «опись | файлов | SHA-256 описи». */
    static String manifests(Path feretDir, Path exportDir) throws IOException {
        StringBuilder rep = new StringBuilder("\n=== Описи (SHA-256 текста описи: строки «путь<TAB>SHA-256 файла», LF, по порядку путей) ===\n");
        Path der = feretDir.resolve(FeretData.DERIVED);
        StringBuilder fm = new StringBuilder();
        for (String sub : List.of(FrameB.DIR, FrameK2.DIR)) {
            Map<String, String> m = FrameB.manifest(der.resolve(sub));
            String t = FrameB.manifestText(m);
            fm.append("# ").append(sub).append('\n').append(t);
            rep.append(FrameB.fmt("FERET derived\\" + sub, m.size(), FrameB.sha256(t)));
        }
        String cp = FrameB.sha256(der.resolve(FrameK2.POINTS));
        fm.append("# ").append(FrameK2.POINTS).append('\n').append(FrameK2.POINTS).append('\t').append(cp).append('\n');
        rep.append(FrameB.fmt("FERET derived\\" + FrameK2.POINTS, 1, cp));
        Files.writeString(der.resolve(MANIFEST), fm.toString(), StandardCharsets.UTF_8);
        StringBuilder em = new StringBuilder();
        List<String> bases = new ArrayList<>();
        try (var s = Files.list(exportDir.resolve(FrameB.DIR))) {
            s.filter(Files::isDirectory).map(p -> p.getFileName().toString()).sorted().forEach(bases::add);
        }
        for (String b : bases) {
            for (String sub : List.of(FrameB.DIR, FrameK2.DIR)) {
                Map<String, String> m = FrameB.manifest(exportDir.resolve(sub).resolve(b));
                String t = FrameB.manifestText(m);
                em.append("# ").append(sub).append('/').append(b).append('\n').append(t);
                rep.append(FrameB.fmt("экспорт " + sub + "/" + b, m.size(), FrameB.sha256(t)));
            }
        }
        Files.writeString(exportDir.resolve(MANIFEST), em.toString(), StandardCharsets.UTF_8);
        rep.append(FrameB.fmt("опись FERET " + der.resolve(MANIFEST), 1, FrameB.sha256(der.resolve(MANIFEST))));
        rep.append(FrameB.fmt("опись экспорта " + exportDir.resolve(MANIFEST), 1, FrameB.sha256(exportDir.resolve(MANIFEST))));
        return rep.toString();
    }

    // ---------------------------------------------------------------- Ф1: постановки FERET и GT

    /** Пробный режим FACES_FINAL_SMOKE: 0 – полный; 1 – малые данные (XBaseVt.trimOwn, пул урезан), весь путь. */
    static int smoke() {
        String s = System.getenv("FACES_FINAL_SMOKE");
        return s == null || s.isBlank() ? 0 : Integer.parseInt(s.trim());
    }

    /** Конфигурация признака: имя файла, подпись, признак f (1 – L1-tt-xs-F, 0 – L1-xs-F), σ₀ Тан – Триггса (у L1-xs-F не влияет). */
    record Cfg(String id, String name, int f, double sigma0) {}

    static final Cfg TT1 = new Cfg("tt1", "L1-tt-xs-F σ₀ = 1", 1, 1.0);
    static final Cfg TT05 = new Cfg("tt05", "L1-tt-xs-F σ₀ = 0,5", 1, 0.5);
    static final Cfg XS = new Cfg("xs", "L1-xs-F", 0, 1.0);

    static FeatP4.P param(Cfg c, IlluminationNorm.NormParams tt) {
        return FeatP4.base(new IlluminationNorm.NormParams(tt.claheClip(), tt.claheTile(), tt.gamma(), c.sigma0(), tt.sigma1(), tt.alpha(), tt.tau()));
    }

    /** Роль контрольных чужих Ф2 (FEI и MUCT вне W, когорт и порога) – добавочные пробы основной модели GT межс. */
    static final String CTRL_X = "ctrlx";

    /**
     * FERET, половины №2 (как XBaseVec.feretSetup): межс. (s 0), внутри (fa → fb даты 1, s 1, порог по (Т0)), (ii) s1, s2, s3 (люди с ≥ 4
     * датами, галерея – первые s дат, пробы – с 4-й даты; s 72, 73, 71). groups – какие роли строить.
     */
    static XBaseVec.Setup feretSetup(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines,
                                     SettingsStore settings, int sm, List<String> groups) throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseVec.Setup u = new XBaseVec.Setup();
        u.base = "feret";
        u.su = s.su;
        u.tpl = s.tpl;
        u.out = s.out;
        XBaseEval.Gt tpl = s.tpl;
        Map<String, FeretData.Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        u.persons = persons.size();
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, List<List<String>>> kept = new TreeMap<>();
        for (String p : ro0.own1) {
            Map<String, List<String>> bd = new TreeMap<>();
            List<String> al = new ArrayList<>(ro0.gal1.get(p));
            al.addAll(ro0.pro1.get(p));
            for (String k : al) bd.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ks = new ArrayList<>();
            for (List<String> l : bd.values()) ks.add(XBaseFeret.sorted(l));
            kept.put(p, ks);
        }
        Object[][] posts = {{"межс.", 0}, {"внутри", 1}, {"(ii) s1", 72}, {"(ii) s2", 73}, {"(ii) s3", 71}};
        for (Object[] pt : posts) {
            String pname = (String) pt[0];
            if (!groups.contains(pname)) continue;
            int conf = (Integer) pt[1];
            int sDates = pname.startsWith("(ii)") ? pname.charAt(pname.length() - 1) - '0' : 0;
            u.groups.add(pname);
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                if (conf == 0) {
                    own = XBaseVt.trimOwn(ro.own1.stream().filter(mine::contains).toList(), sm);
                    for (String p : own) {
                        gal.put(p, ro.gal1.get(p));
                        pro.put(p, ro.pro1.get(p));
                    }
                } else if (conf == 1) {
                    own = XBaseVt.trimOwn(ro.own1in.stream().filter(mine::contains).toList(), sm);
                    for (String p : own) {
                        gal.put(p, ro.gal1in.get(p));
                        pro.put(p, ro.pro1in.get(p));
                    }
                } else {
                    own = XBaseVt.trimOwn(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= 4).toList(), sm);
                    for (String p : own) {
                        List<String> g = new ArrayList<>();
                        for (int j = 0; j < sDates; j++) g.addAll(s.pool.get(p).get(j));
                        gal.put(p, XBaseFeret.sorted(g));
                        List<String> pr = new ArrayList<>();
                        for (int j = 3; j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                        pro.put(p, XBaseFeret.sorted(pr));
                    }
                }
                Map<String, List<List<String>>> projPool = new TreeMap<>();
                for (String p : s.halves.get(1 - ps)) if (sm != 1 || projPool.size() < XBaseDates.SMOKE_POOL) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + conf);
                List<Probe> probes = XBaseEval.probes(g, pro);
                u.blocks.add(new XBaseVec.Blk(String.format(Locale.ROOT, "c%d_p%d", conf, ps), pname + ", %s, проход " + (ps + 1) + ", ", g, gal, pro, parts, probes,
                        conf == 1 ? null : XBaseThr.extra(g, pro, parts), conf, pname));
            }
        }
        u.fr = XBaseLbp.frames(s, export, true);
        for (XBaseVec.Blk b : u.blocks) {
            for (Map<String, List<String>> m : List.of(b.gal(), b.pro())) {
                for (List<String> l : m.values()) for (String k : l) if (!u.fr.containsKey(k)) u.fr.put(k, FeretData.frame(u.su.dir, "a", k));
            }
        }
        XBaseVec.cohorts(u, feiKeys, tpl.seed, export);
        return u;
    }

    /** Ф2: контрольные чужие FEI (кроме когорт в W и вне W) и MUCT, отбор |r| ≤ R по строке YuNet экспорта; ключ → «источник:человек». */
    static Map<String, String> extraCtrl(FarExport.Loaded export, Path exportDir, XBaseVec.Setup u, Map<String, Map<Integer, String>> feiKeys, double rLimit,
                                         Map<String, Double> rOut) throws IOException {
        Map<String, double[]> rows = FrameK2.exportRows(exportDir);
        Set<String> cohort = new HashSet<>(u.cohortIn);
        cohort.addAll(u.cohortOut);
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, Map<Integer, String>> e : new TreeMap<>(feiKeys).entrySet()) {
            if (cohort.contains(e.getKey())) continue;
            for (String k : e.getValue().values()) {
                double[] row = rows.get(k);
                if (row == null) continue;
                double r = Math.abs(FeiDataset.pose(row));
                if (r > rLimit) continue;
                out.put(k, "fei:" + e.getKey());
                rOut.put(k, r);
            }
        }
        for (Map.Entry<String, List<Sample>> e : new TreeMap<>(export.data().muct()).entrySet()) {
            for (Sample s : e.getValue()) {
                String k = s.file().toString();
                double[] row = rows.get(k);
                if (row == null) continue;
                double r = Math.abs(FeiDataset.pose(row));
                if (r > rLimit) continue;
                out.put(k, "muct:" + e.getKey());
                rOut.put(k, r);
            }
        }
        return out;
    }

    /** Кадры k = 2 ключей экспорта MUCT и своей базы: кадр «в» экспорта, точки – по исходному снимку со сверкой кадра (как FrameK2.load для GT и FEI). */
    static Map<String, FrameK2.F> k2Muct(Iterable<String> keys, FarExport.Loaded export, Path exportDir, SettingsStore settings) throws IOException {
        Map<String, FrameK2.F> out = new LinkedHashMap<>();
        Map<String, double[]> rows = FrameK2.exportRows(exportDir);
        Map<String, Sample> raw = new HashMap<>();
        for (Sample s : GalleryEvaluation.rawData(settings).samples()) raw.put(FarExport.id(s).toString(), s);
        Map<String, Sample> exp = new HashMap<>();
        for (Sample s : export.data().samples()) exp.put(s.file().toString(), s);
        for (String k : keys) {
            Sample rs = raw.get(k);
            Sample es = exp.get(k);
            double[] row = rows.get(k);
            if (rs == null || es == null || row == null) throw new IllegalStateException("final: нет снимка или детекции MUCT " + k);
            Mat[] m = GalleryEvaluation.load(rs);
            Mat gray = new Mat();
            Imgproc.cvtColor(m[0], gray, Imgproc.COLOR_BGR2GRAY);
            GalleryEvaluation.release(m);
            FrameK2.F f = FrameK2.fromFull(gray, FrameK2.lm(row));
            gray.release();
            Mat e = export.frame(es, GalleryEvaluation.Variant.C);
            if (e == null || !FrameK2.same(e, f.frame())) throw new IllegalStateException("final: кадр k = 2 MUCT из исходного снимка ≠ кадр «в» экспорта: " + k);
            f.frame().release();
            out.put(k, new FrameK2.F(e, f.pts()));
        }
        return out;
    }

    /** Настройка Ф1 и контроля: setup базы, кадры (а или Б – по FACES_FRAMES), кадры k = 2; GT межс. – добавочные пробы Ф2 (кроме контроля). */
    record Run(XBaseVec.Setup u, Map<String, FrameK2.F> k2, Map<String, String> extra, Path dir, boolean freeze, int sm) {}

    static Run setup(String base, boolean freeze, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines,
                     SettingsStore settings, Path out) throws IOException {
        int sm = smoke();
        if (freeze == FrameB.on()) throw new IllegalStateException("final: контроль замораживания – без FACES_FRAMES, Ф1 – FACES_FRAMES=b");
        XBaseVec.Setup u;
        Map<String, String> extra = new LinkedHashMap<>();
        if (base.equals("fei")) {
            u = feiSetup(export, dir, feiKeys, selLines, settings, sm);
        } else if (base.equals("own")) {
            u = ownSetup(export, dir, feiKeys, selLines, settings, sm);
        } else if (base.equals("feret")) {
            u = feretSetup(export, dir, feiKeys, selLines, settings, sm, freeze ? List.of("межс.", "(ii) s3") : List.of("межс.", "внутри", "(ii) s1", "(ii) s2", "(ii) s3"));
        } else {
            u = XBaseVec.setup("gt", export, dir, feiKeys, selLines, settings, out, sm);
            if (!freeze) {
                Map<String, Double> r = new HashMap<>();
                extra = extraCtrl(export, dir, u, feiKeys, u.tpl.rLimit, r);
                List<Sample> all = new ArrayList<>();
                export.data().fei().values().forEach(all::addAll);
                export.data().muct().values().forEach(all::addAll);
                u.fr.putAll(XBaseEval.frames(export, all, extra.keySet()));
                XBaseVec.Blk x = u.blocks.get(0);
                if (!x.group().equals("межс.")) throw new IllegalStateException("final: первый блок GT – не межс.");
                x.g().rOf.putAll(r);
                int i = 0;
                for (Map.Entry<String, String> e : extra.entrySet()) x.probes().add(new Probe(CTRL_X, e.getValue(), i++, e.getKey(), -1));
            }
        }
        List<String> std = u.fr.keySet().stream().filter(k -> !k.startsWith("muct") && !k.startsWith("own")).toList();
        List<String> muct = u.fr.keySet().stream().filter(k -> k.startsWith("muct") || k.startsWith("own")).toList();
        Map<String, FrameK2.F> k2 = FrameK2.load(std, u.su, u.su.dir, export, dir, Paths.get(settings.loadFacesGtDir()), Paths.get(settings.loadFacesFeiDir()));
        if (!muct.isEmpty()) k2.putAll(k2Muct(muct, export, dir, settings));
        Path fd = base.equals("feret") ? u.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(freeze ? "final_a" : "final") : out.resolve(freeze
                ? "final_a" : "final");
        if (sm > 0) fd = fd.resolveSibling(fd.getFileName() + "_smoke" + sm);
        return new Run(u, k2, extra, fd, freeze, sm);
    }

    /** FERET и GT (Ф1) – σ₀ = 1 и 0,5; кадры «а», FEI и своя база – σ₀ = 1 (σ₀ = 0,5 – при FACES_FINAL_TT05=1, досчёт после Ф1б). */
    static List<Cfg> cfgs(boolean freeze, String base) {
        if (freeze) return List.of(TT1, XS);
        if (base.equals("feret") || base.equals("gt")) return List.of(TT1, TT05, XS);
        return "1".equals(System.getenv("FACES_FINAL_TT05")) ? List.of(TT1, TT05, XS) : List.of(TT1, XS);
    }

    static Path vfile(Run r, XBaseVec.Blk b, Cfg c) {
        return r.dir().resolve("vec").resolve(b.name() + "_" + c.id() + ".tsv.gz");
    }

    /** Модели всех блоков и конфигураций (готовые файлы с тем же коммитом кода пропускаются – продолжение после обрыва). */
    static Map<String, Double> compute(Run r) throws IOException {
        XBaseVec.Setup u = r.u();
        Files.createDirectories(r.dir().resolve("vec"));
        String commit = FarMethods.commit();
        Map<String, Double> sec = new LinkedHashMap<>();
        for (Cfg c : cfgs(r.freeze(), r.u().base)) {
            List<XBaseVec.Blk> todo = new ArrayList<>();
            for (XBaseVec.Blk b : u.blocks) {
                Path p = vfile(r, b, c);
                if (Files.exists(p)) {
                    if (!XBaseVec.readHead(p).contains(commit)) throw new IllegalStateException("final: " + p + " – другой коммит кода; удалите каталог " + r.dir());
                    continue;
                }
                todo.add(b);
            }
            if (todo.isEmpty()) continue;
            long t0 = System.nanoTime();
            XBaseEval x = XBasePart4.vectorsB(u.fr, r.k2(), param(c, u.tpl.params), c.f(), u.tpl.threads);
            System.out.printf(Locale.ROOT, "final %s: %s признаки (%.0f с)%n", u.base, c.id(), (System.nanoTime() - t0) / 1e9);
            int step = Math.max(1, Math.min(4, todo.size()));
            for (int i = 0; i < todo.size(); i += step) {
                List<List<java.util.concurrent.Callable<XBaseVec.Model>>> jobs = new ArrayList<>();
                for (XBaseVec.Blk b : todo.subList(i, Math.min(todo.size(), i + step))) {
                    jobs.add(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
                }
                List<List<XBaseVec.Model>> done = XBaseVec.run(jobs, u.tpl.threads);
                for (int k = 0; k < done.size(); k++) {
                    XBaseVec.Blk b = todo.get(i + k);
                    Path p = vfile(r, b, c);
                    java.io.Writer w = XBaseVec.writer(p, "# Векторы 1 − cos (XBaseFinal " + u.base + "): " + c.name() + ", кадры " + (r.freeze() ? "«а»" : "Б")
                            + "\n# Код: коммит " + commit + "\n");
                    XBaseVec.write(w, String.format(b.abl2(), XBaseRule.FEATS[c.f()]) + c.name(), done.get(k));
                    XBaseVec.finish(w, p);
                }
                System.out.printf(Locale.ROOT, "final %s: %s блоки %d–%d из %d%n", u.base, c.id(), i + 1, i + jobs.size(), todo.size());
            }
            sec.merge(c.id(), (System.nanoTime() - t0) / 1e9, Double::sum);
        }
        return sec;
    }

    static List<XBaseVec.Model> models(Run r, XBaseVec.Blk b, Cfg c, boolean ext) throws IOException {
        List<XBaseVec.Model> ms = XBaseVec.read(vfile(r, b, c)).values().iterator().next();
        return XBasePart4.filtered(ms, row -> row.role().equals(CTRL_X) == ext);
    }

    /** Строки оценок блока: подпись → строки (слияние по каждой σ₀, кандидаты, Ф2 – строки ctrlx основной модели). */
    static Map<String, String> blockLines(Run r, XBaseVec.Blk b, int[] clamped) throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        List<XBaseVec.Model> xs = models(r, b, XS, false);
        double[] sx = XBaseVec.fuseStats(xs, false);
        for (Cfg c : cfgs(r.freeze(), r.u().base)) {
            if (c.f() != 1) continue;
            List<XBaseVec.Model> tt = models(r, b, c, false);
            double[] st = XBaseVec.fuseStats(tt, false);
            out.put(fusionLabel(r, c), XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, st, sx, b.g().rOf, b.g().r0, clamped));
            if (!r.extra().isEmpty() && b.group().equals("межс.")) {
                out.put("Ф2 " + fusionLabel(r, c), XBaseVec.fuseLines(models(r, b, c, true), models(r, b, XS, true), XBaseFeat2.W_FUSE, st, sx, b.g().rOf, b.g().r0,
                        clamped));
            }
        }
        for (Cfg c : cfgs(r.freeze(), r.u().base)) {
            out.put(c.name(), XBaseVec.normLines(models(r, b, c, false), null, clamped));
            if (!r.extra().isEmpty() && b.group().equals("межс.")) out.put("Ф2 " + c.name(), XBaseVec.normLines(models(r, b, c, true), null, clamped));
        }
        return out;
    }

    static final String FREEZE_LABEL = "слияние (б) (П2) + MS-LBP, w = 0,75";

    static String fusionLabel(Run r, Cfg c) {
        return r.freeze() && !r.u().base.equals("fei") && !r.u().base.equals("own") ? FREEZE_LABEL : "слияние σ₀ = " + (c.sigma0() == 1 ? "1" : "0,5");
    }

    /** SFace блока (минимум снимков шаблона 1, порог П2/П3 – по «чужим» галереи без когорты), строки d₁/d₂ по 1 − cos. */
    static String sfaceLines(Run r, XBaseVec.Blk b, XBaseEval xsf) {
        XBaseEval.Gt gs = r.u().base.equals("feret") ? XBaseFeret.withOwn(b.g(), b.g().own) : XBaseDates.withOwn(b.g(), b.g().own);
        gs.cohort = List.of();
        List<Probe> pr = b.probes().stream().filter(p -> !p.role().equals(CTRL_X)).toList();
        return XBaseEval.fitLines(gs, xsf, java.util.EnumSet.of(XBaseEval.V.SFACE), b.gal(), pr, b.parts(), b.s());
    }

    static XBaseEval sfaceVectors(Run r, FarExport.Loaded export, SettingsStore settings) throws IOException {
        XBaseVec.Setup u = r.u();
        Set<String> keys = new TreeSet<>();
        for (XBaseVec.Blk b : u.blocks) {
            for (Map<String, List<String>> m : List.of(b.gal(), b.pro())) for (List<String> l : m.values()) keys.addAll(l);
            for (Probe p : b.probes()) if (!p.role().equals(CTRL_X)) keys.add(p.key());
        }
        Map<String, Mat> sf = new LinkedHashMap<>();
        if (u.base.equals("feret")) {
            for (String k : keys) sf.put(k, FeretData.frame(u.su.dir, "sface", k));
        } else {
            for (Sample s : u.tpl.samples.values()) {
                String k = s.file().toString();
                if (!keys.contains(k)) continue;
                Mat c = export.frame(s, GalleryEvaluation.Variant.SFACE);
                if (c == null) throw new IllegalStateException("final: нет кадра SFace " + k);
                sf.put(k, c);
            }
        }
        Path model = Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        xsf.sfaceVectors(sf, org.opencv.objdetect.FaceRecognizerSF.create(model.toString(), ""));
        sf.values().forEach(Mat::release);
        return xsf;
    }

    /** final-feret, final-gt (Ф1) и final-freeze-feret, final-freeze-gt (контроль замораживания на кадрах «а»). */
    static void phase1(String base, boolean freeze, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines,
                       SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        Run r = setup(base, freeze, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        Map<String, Double> sec = compute(r);
        XBaseEval xsf = freeze ? null : sfaceVectors(r, export, settings);
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        int[] clamped = new int[2];
        // файл оценок: «# группа | подпись | блок», затем строки
        StringBuilder sc = new StringBuilder("# Файл оценок итогового прогона (XBaseFinal " + base + (freeze ? ", контроль замораживания, кадры «а»" : ", Ф1, кадры Б")
                + "); строки – формат FeiScores (d₁/d₂; у слияния – монотонное сжатие s = 2ℓ₁ − ℓ₂, XBaseVec.fuseLines)\n# Код: коммит " + FarMethods.commit() + "\n");
        Map<String, Map<String, List<String>>> byGroup = new LinkedHashMap<>();
        for (XBaseVec.Blk b : u.blocks) {
            Map<String, String> ls = blockLines(r, b, clamped);
            if (xsf != null) ls.put("SFace", sfaceLines(r, b, xsf));
            for (Map.Entry<String, String> e : ls.entrySet()) {
                sc.append("# ").append(b.group()).append(" | ").append(e.getKey()).append(" | ").append(b.name()).append('\n').append(e.getValue());
                byGroup.computeIfAbsent(b.group(), g -> new LinkedHashMap<>()).computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(e.getValue());
            }
        }
        Files.createDirectories(r.dir());
        Path scf = r.dir().resolve("final_" + base + "_scores.tsv.gz");
        java.io.Writer sw = XBaseVec.writer(scf, sc.toString());
        XBaseVec.finish(sw, scf);
        // отчёт: таблицы как feat2c (XBaseLbp.tables), группы по очереди
        StringBuilder t = new StringBuilder("Итоговый прогон – " + (freeze ? "контроль замораживания (кадры «а»)" : "Ф1 (кадры Б)") + " – " + base + " (XBaseEval "
                + (freeze ? "final-freeze-" : "final-") + base + ")" + (r.sm() == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + r.sm()) + "\nКод: коммит " + FarMethods.commit()
                + "; модели – " + r.dir().resolve("vec") + "; файл оценок – " + scf + "\n");
        t.append("\n=== Время ===\n");
        sec.forEach((k, v) -> t.append(String.format(Locale.ROOT, "%s: %.0f с%n", k, v)));
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) ===\n");
        StringBuilder body = new StringBuilder();
        StringBuilder selT = new StringBuilder(SIGMA_HEAD);
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            List<XBaseVec.Blk> bs = u.blocks.stream().filter(b -> b.group().equals(grp)).toList();
            Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            for (Map.Entry<String, List<String>> e : byGroup.get(grp).entrySet()) {
                String lab = e.getKey();
                if (lab.startsWith("Ф2 ") || lab.equals("SFace")) continue;
                Cfg c = cfgs(freeze, base).stream().filter(x -> x.name().equals(lab)).findFirst().orElse(null);
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (int i = 0; i < bs.size(); i++) {
                    XBaseVec.Blk b = bs.get(i);
                    XBaseEval.Gt g = c == null ? b.g() : XBaseAbl3.mirrorGt(b.g(), c.f() == 0);
                    us.add(new XBaseThr.Unit(g, b.gal(), b.pro(), b.parts(), e.getValue().get(i), null, t2));
                }
                String rl = grp + " " + lab;
                XBaseThr.Out o = XBaseThr.evaluate(rl, us, u.persons, u.tpl);
                of.put(rl, o);
                if (t2) hold.append(XBaseAbl3.holds(rl, o, XBaseThr.WAYS[2]));
            }
            if (cfgs(freeze, base).contains(TT05)) {
                pairs.add(new String[] {grp + " слияние σ₀ = 0,5", grp + " слияние σ₀ = 1"});
                pairs.add(new String[] {grp + " " + TT05.name(), grp + " " + TT1.name()});
                pairs.add(new String[] {grp + " слияние σ₀ = 1", grp + " " + TT1.name()});
            }
            if (cfgs(freeze, base).contains(TT05)) selT.append(sigmaRows(base, grp, of.get(grp + " слияние σ₀ = 0,5"), of.get(grp + " слияние σ₀ = 1"), u.persons, u.tpl));
            XBaseLbp.tables(body, prefix + " – слияние кандидатов", of, u.persons, u.tpl, pairs, t2);
            if (byGroup.get(grp).containsKey("SFace")) {
                List<XBaseDates.Pass> passes = new ArrayList<>();
                for (XBaseVec.Blk b : bs) passes.add(new XBaseDates.Pass(b.g(), b.gal(), null, b.pro(), b.parts(), b.probes(), b.extra()));
                XBaseDates.sface(body, prefix, List.of(new XBaseDates.Line(grp + " SFace", passes, byGroup.get(grp).get("SFace"))), u.persons, u.tpl, List.of());
            }
        }
        t.append(hold).append(body);
        if (selT.length() > SIGMA_HEAD.length()) Files.writeString(r.dir().resolve("final_" + base + "_sigma.tsv"), selT.toString(), StandardCharsets.UTF_8);
        String rep = t.toString();
        if (freeze && (base.equals("feret") || base.equals("gt"))) rep += freezeCheck(base, rep, u.su.dir, out);
        Path rf = r.dir().resolve("final_" + base + ".txt");
        FarMethods.write(rf, new StringBuilder(rep));
        System.out.print(rep);
        System.out.printf(Locale.ROOT, "Всего: %.0f с%n", (System.nanoTime() - t0) / 1e9);
        if (rep.contains("НЕ СОВПАЛО")) throw new IllegalStateException("final: контроль замораживания не прошёл – остановка");
        System.out.println("DONE");
    }

    /** Сверка строк «слияние (б) (П2) + MS-LBP, w = 0,75 | …» с отчётом feat2c (FERET – reports\eq\, GT – reports/faces/xbase/). */
    static String freezeCheck(String base, String rep, Path feretDir, Path out) throws IOException {
        Path f = base.equals("feret") ? feretDir.resolve("reports").resolve("eq").resolve("xbase_feret_feat2c.txt") : out.resolve("xbase_gt_feat2c.txt");
        String key = FREEZE_LABEL + " |";
        List<String> want = Files.readAllLines(f, StandardCharsets.UTF_8).stream().map(l -> l.replace("\r", "")).filter(l -> l.contains(key)).toList();
        Set<String> got = new HashSet<>(List.of(rep.replace("\r", "").split("\n")));
        int miss = 0;
        StringBuilder b = new StringBuilder("\n=== Контроль замораживания: строки feat2c («" + key + "»), " + f + " ===\n");
        for (String l : want) {
            if (got.contains(l)) continue;
            miss++;
            b.append("нет: ").append(l).append('\n');
        }
        b.append(String.format(Locale.ROOT, "Строк в feat2c: %d, совпало побайтно: %d – %s%n", want.size(), want.size() - miss, miss == 0 ? "совпало" : "НЕ СОВПАЛО"));
        return b.toString();
    }

    // ---------------------------------------------------------------- Ф1: FEI внутри

    /**
     * FEI внутри (протокол п. 3а, как fei-xs XBaseC1): 5 разбиений, свои – галерея разбиения (7 снимков, проба №11), контроль – контрольные
     * люди разбиения (все отобранные снимки); пороговые люди разбиения не нужны (порог – П3 по «чужим» (Т0) галереи, 5 частей seed
     * faces.seed + 8000 + 100 + s; рабочий t(N) – в отчёте); когорта разбиения + пул FERET – в обучении W; группа «внутри».
     */
    static XBaseVec.Setup feiSetup(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                                   int sm) throws IOException {
        long seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("final-fei: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new TreeSet<>(m.keySet())));
        List<FeiProtocol.Split> splits = FeiProtocol.splits(included, seed);
        int jn = -1;
        for (int i = 0; i < FeiProtocol.NUMBERS.length; i++) if (FeiProtocol.NUMBERS[i] == XBaseC1.PROBE_NO) jn = i;
        List<String> persons = new ArrayList<>(new TreeSet<>(feiKeys.keySet()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        Map<String, Integer> numOf = new HashMap<>();
        feiKeys.forEach((p, m) -> m.forEach((num, k) -> numOf.put(k, num)));
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "fei");
        List<Sample> feiS = new ArrayList<>();
        export.data().fei().values().forEach(feiS::addAll);
        Map<String, Double> rOf = new HashMap<>();
        for (Sample s : feiS) {
            double[] row = det.get(s.person() + "/" + s.file().getFileName().toString());
            if (row == null) throw new IllegalStateException("final-fei: нет строки YuNet для " + s.file());
            rOf.put(s.file().toString(), Math.abs(FeiDataset.pose(row)));
        }
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));
        XBaseVec.Setup u = new XBaseVec.Setup();
        u.base = "fei";
        u.su = su;
        u.persons = persons.size();
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.seed = seed;
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.params = IlluminationNorm.NormParams.of(settings);
        tpl.r0 = r0;
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.persons = persons;
        tpl.pIdx = pIdx;
        tpl.imageNo = numOf;
        tpl.rOf = rOf;
        u.tpl = tpl;
        u.groups.add("внутри");
        u.fr = new LinkedHashMap<>(XBaseEval.frames(export, feiS, null));
        Set<String> used = new HashSet<>();
        for (FeiProtocol.Split sp : splits) {
            if (sm == 1 && sp.index() > 0) break;
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            List<Probe> probes = new ArrayList<>();
            List<String> own = XBaseVt.trimOwn(sp.gallery(), sm);
            for (int i = 0; i < own.size(); i++) {
                String p = own.get(i);
                gal.put(p, FeiProtocol.trainNumbers(jn).stream().map(n -> feiKeys.get(p).get(n)).toList());
                pro.put(p, List.of(feiKeys.get(p).get(XBaseC1.PROBE_NO)));
                probes.add(new Probe(FeiScores.OWN, p, XBaseC1.PROBE_NO, feiKeys.get(p).get(XBaseC1.PROBE_NO), i));
            }
            for (String p : sp.control()) for (Map.Entry<Integer, String> e : new TreeMap<>(feiKeys.get(p)).entrySet()) probes.add(new Probe(FeiScores.CTRL, p, e.getKey(),
                    e.getValue(), -1));
            List<List<String>> cohort = new ArrayList<>();
            for (String p : sp.cohort()) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
            List<List<String>> train = new ArrayList<>(cohort);
            train.addAll(poolCls);
            poolCls.forEach(used::addAll);
            XBaseEval.Gt g = XBaseAbl.copy(tpl, train, rOf, r0);
            g.own = own;
            g.ctrl = sp.control();
            g.persons = persons;
            g.pIdx = pIdx;
            g.imageNo = numOf;
            List<List<String>> parts = XBaseOther.cvParts(own, seed, XBaseEval.CV_S + sp.index());
            u.blocks.add(new XBaseVec.Blk("s" + sp.index(), "внутри, %s, разбиение " + sp.index() + ", ", g, gal, pro, parts, probes, null, sp.index(), "внутри"));
        }
        for (String k : used) if (!u.fr.containsKey(k)) u.fr.put(k, FeretData.frame(su.dir, "a", k));
        return u;
    }

    // ---------------------------------------------------------------- Ф1: своя база внутри

    /**
     * Своя база внутри сессии (как own-xs XBaseC1, оптимистично): ротации OwnEvaluation.split (FACES_OWN_ROTATIONS, по умолчанию 3 – как
     * С1, решение Хозяина 22.10.2026), галерея – снимки обучающих моментов качества ≥ FAIR, проба – представитель контрольного момента;
     * контроль – односессионные люди GT (|r| ≤ R), когорта FEI 20 + пул FERET в W; группа «внутри».
     */
    static XBaseVec.Setup ownSetup(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                                   int sm) throws IOException {
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("final-own: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Map<String, double[]> gtDet = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rOld = new HashMap<>();
        gtDet.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        List<OwnDataset.Person> ownP = export.data().own().persons();
        List<List<OwnDataset.Moment>> moments = export.data().moments();
        Map<String, double[]> ownDet = XBaseMpPts.exportRows(dir, "own");
        Map<String, Integer> numOf = new HashMap<>();
        Map<String, Double> rOf = new HashMap<>(g0.rOf);
        List<Sample> ownFound = new ArrayList<>();
        for (int i = 0; i < ownP.size(); i++) {
            int n = 0;
            for (OwnDataset.Moment m : moments.get(i)) {
                for (OwnDataset.Frame f : m.frames()) {
                    String k = f.file().toString();
                    if (export.found().get(k) == null) continue;
                    numOf.put(k, n++);
                    double[] row = ownDet.get(ownP.get(i).name() + "/" + f.file().getFileName().toString());
                    if (row != null) rOf.put(k, Math.abs(FeiDataset.pose(row)));
                    if (Boolean.TRUE.equals(export.found().get(k))) ownFound.add(new Sample(GalleryEvaluation.Base.OWN, ownP.get(i).name(), f.file(), ' '));
                }
            }
        }
        for (Sample s : g0.samples.values()) numOf.put(s.file().toString(), XBaseEval.image(s));
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (sm == 1) poolCls = poolCls.subList(0, Math.min(XBaseDates.SMOKE_POOL, poolCls.size()));
        Set<String> cohortKeys = new HashSet<>();
        g0.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        XBaseVec.Setup u = new XBaseVec.Setup();
        u.base = "own";
        u.su = su;
        u.fr = new LinkedHashMap<>(XBaseEval.frames(export, ownFound, null));
        u.fr.putAll(XBaseEval.frames(export, g0.samples.values().stream().sorted((a, z) -> a.file().toString().compareTo(z.file().toString())).toList(), null));
        u.fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<String> c : poolCls) for (String k : c) if (!u.fr.containsKey(k)) u.fr.put(k, FeretData.frame(su.dir, "a", k));
        List<String> own = ownP.stream().map(OwnDataset.Person::name).toList();
        List<String> persons = new ArrayList<>(own);
        persons.addAll(g0.ctrl);
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        XBaseEval.Gt tpl = XBaseAbl.copy(g0, g0.cohort, rOf, r0);
        tpl.own = own;
        tpl.persons = persons;
        tpl.pIdx = pIdx;
        tpl.imageNo = numOf;
        u.tpl = tpl;
        u.persons = persons.size();
        u.groups.add("внутри");
        String env = System.getenv("FACES_OWN_ROTATIONS");
        int rotations = env == null ? 3 : Integer.parseInt(env);
        if (sm == 1) rotations = 1;
        List<List<String>> train = new ArrayList<>(g0.cohort);
        train.addAll(poolCls);
        for (int r = 0; r < rotations; r++) {
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<String>> pro = new LinkedHashMap<>();
            List<Probe> probes = new ArrayList<>();
            for (int i = 0; i < own.size(); i++) {
                List<OwnDataset.Moment> ms = moments.get(i);
                OwnEvaluation.Split spl = OwnEvaluation.split(ms, r % ms.size(), false);
                List<String> ks = new ArrayList<>();
                for (int j : spl.train()) for (OwnDataset.Frame f : ms.get(j).frames()) {
                    String k = f.file().toString();
                    if (f.quality() >= OwnDataset.FAIR && Boolean.TRUE.equals(export.found().get(k))) ks.add(k);
                }
                gal.put(own.get(i), ks);
                String k = ms.get(spl.control()).representative().file().toString();
                if (Boolean.TRUE.equals(export.found().get(k))) {
                    probes.add(new Probe(FeiScores.OWN, own.get(i), numOf.get(k), k, i));
                    pro.put(own.get(i), List.of(k));
                } else {
                    pro.put(own.get(i), List.of());
                }
            }
            for (String p : g0.ctrl) for (List<String> ks : g0.byPerson.get(p).values()) for (String k : ks) probes.add(new Probe(FeiScores.CTRL, p, numOf.get(k), k, -1));
            XBaseEval.Gt g = XBaseAbl.copy(tpl, train, rOf, r0);
            g.own = own;
            g.ctrl = g0.ctrl;
            g.persons = persons;
            g.pIdx = pIdx;
            g.imageNo = numOf;
            List<List<String>> parts = XBaseOther.cvParts(own, g0.seed, XBaseEval.CV_S + r);
            u.blocks.add(new XBaseVec.Blk("r" + r, "внутри, %s, ротация " + r + ", ", g, gal, pro, parts, probes, null, r, "внутри"));
        }
        return u;
    }

    // ---------------------------------------------------------------- Ф1б: выбор σ₀ по правилу (36)

    static final String SIGMA_HEAD = "# база\tгруппа\tмера\tразность σ₀ = 0,5 − σ₀ = 1, п. п.\t2,5 %\t97,5 %\t0,25 %\t99,75 %\n";

    /** Парные разности «σ₀ = 0,5 − σ₀ = 1» слияния: разделимость при FAR 1 % и EER, бутстреп по людям (веса как в таблицах). */
    static String sigmaRows(String base, String grp, XBaseThr.Out a, XBaseThr.Out z, int persons, XBaseEval.Gt tpl) {
        if (tpl.b <= 0) return "";
        int[][] w = FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        XBaseRule.Sep sa = XBaseRule.sep(a.res.get(XBaseThr.WAYS[0]).cfgs, FeiBootstrap.unit(persons), w);
        XBaseRule.Sep sz = XBaseRule.sep(z.res.get(XBaseThr.WAYS[0]).cfgs, FeiBootstrap.unit(persons), w);
        StringBuilder b = new StringBuilder();
        String[] names = {"FAR 1 %", "EER"};
        double[] pt = {sa.err[0] - sz.err[0], sa.eer[3] - sz.eer[3]};
        double[][] reps = {FeiNorm.diff(sa.errRep[0], sz.errRep[0]), FeiNorm.diff(sa.eerRep, sz.eerRep)};
        for (int i = 0; i < 2; i++) {
            double[] d = FeiBootstrap.sorted(reps[i]);
            b.append(String.format(Locale.ROOT, "%s\t%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f%n", base, grp, names[i], pt[i], FeiBootstrap.quantile(d, 0.025),
                    FeiBootstrap.quantile(d, 0.975), FeiBootstrap.quantile(d, 0.0025), FeiBootstrap.quantile(d, 0.9975)));
        }
        return b.toString();
    }

    /**
     * final-select (Ф1б): правило (36) механически по файлам final_feret_sigma.tsv и final_gt_sigma.tsv Ф1. σ₀ = 0,5 принимается, если хотя
     * бы на одной из строк FERET межс., GT межс. разность «0,5 − 1» значимо меньше нуля (верхняя граница 95 % интервала бутстрепа по людям
     * &lt; 0) при FAR 1 % или по EER и выполнено условие Δ (GT внутри, FAR 1 %: верхняя граница 95 % ≤ +3 п. п.); иначе σ₀ = 1. Итог –
     * final_sigma.tsv рядом с GT и отчёт final_select.txt.
     */
    static void select(Path feretDir, Path out) throws IOException {
        Path ff = feretDir.resolve("reports").resolve("eq_b").resolve("final").resolve("final_feret_sigma.tsv");
        Path fg = out.resolve("final").resolve("final_gt_sigma.tsv");
        List<String[]> rows = new ArrayList<>();
        for (Path f : List.of(ff, fg)) {
            if (!Files.exists(f)) throw new IllegalStateException("final-select: нет " + f + " (сначала final-feret, final-gt)");
            for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) if (!l.startsWith("#") && !l.isBlank()) rows.add(l.split("\t"));
        }
        StringBuilder t = new StringBuilder("Итоговый прогон, Ф1б: выбор σ₀ по правилу (36) (XBaseEval final-select)\nКод: коммит " + FarMethods.commit() + "\nФайлы: " + ff
                + ", " + fg + "\n");
        t.append("Правило (предрегистрация, TASKS.md «Итоговый прогон перед ПО – план»): σ₀ = 0,5, если хотя бы на одной из строк FERET межс., GT межс. разность "
                + "слияния «0,5 − 1» значимо меньше нуля\n(верхняя граница 95 % интервала < 0) при FAR 1 % или по EER, и условие Δ (GT внутри, FAR 1 %, верхняя "
                + "граница 95 % ≤ +3 п. п.) выполнено; иначе σ₀ = 1.\n\nбаза | группа | мера | разность | 95 % | 99,5 % | значимо лучше\n");
        boolean better = false;
        Double deltaUp = null;
        for (String[] c : rows) {
            double up = Double.parseDouble(c[5]);
            boolean sig = up < 0;
            boolean x = c[1].equals("межс.");
            if (x && sig) better = true;
            if (c[0].equals("gt") && c[1].equals("внутри") && c[2].equals("FAR 1 %")) deltaUp = up;
            t.append(String.format(Locale.ROOT, "%s | %s | %s | %+.1f | [%+.1f; %+.1f] | [%+.1f; %+.1f] | %s%n", c[0], c[1], c[2], Double.parseDouble(c[3]),
                    Double.parseDouble(c[4]), up, Double.parseDouble(c[6]), Double.parseDouble(c[7]), x ? (sig ? "да" : "нет") : "–"));
        }
        if (deltaUp == null) throw new IllegalStateException("final-select: нет строки Δ (GT внутри, FAR 1 %)");
        boolean delta = deltaUp <= 3;
        double sigma = better && delta ? 0.5 : 1.0;
        t.append(String.format(Locale.ROOT, "%nЗначимо лучше межсессионно: %s; условие Δ: верхняя граница %+.1f п. п. – %s.%nВыбор: σ₀ = %s%n", better ? "да" : "нет",
                deltaUp, delta ? "выполнено" : "НЕ выполнено", sigma == 1 ? "1" : "0,5"));
        Files.writeString(out.resolve("final").resolve("final_sigma.tsv"), "# Выбор σ₀ (XBaseEval final-select), коммит " + FarMethods.commit() + "\nsigma0\t" + sigma
                + "\n", StandardCharsets.UTF_8);
        FarMethods.write(out.resolve("final").resolve("final_select.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- Ф3: кадры VidTIMIT

    static final String VT_LIST = "final_vt.tsv";
    static final String VT_HEAD_DET = "final_vt_head3_detections.tsv";
    static final String VT_POINTS = "final_vt_c_points.tsv";
    static final long VT_HEAD_SEED = 11110;

    /**
     * frames-b-vt: кадры Ф3. Галерея и пробы фраз – кадры (В2) описи vt_selection.tsv (сессии 1, 2 – галерея M = 1 / M = 2, сессия 3 – по
     * кадру на фразу); поворот – один кадр head3 с лицом YuNet и |r| ≤ R: кадры каждого человека (по имени) в случайном порядке
     * Random(faces.seed + 11110), один генератор на всех людей по порядку имён, берётся первый годный (детекции – кэш
     * derived\final_vt_head3_detections.tsv, vt_detections.tsv не меняется). Кадры Б – derived\a_b\, k = 2 – derived\c\ и
     * final_vt_c_points.tsv; опись final_vt.tsv (человек, сессия, вид, фраза, ключ, r); описи SHA-256 – отчёт.
     */
    static void vtFrames(List<String> selLines, SettingsStore settings, Path out) throws IOException {
        if (FrameB.on()) throw new IllegalStateException("frames-b-vt: без FACES_FRAMES");
        long t0 = System.nanoTime();
        Path dir = VidTimitData.dir(settings);
        Path der = dir.resolve(VidTimitData.DERIVED);
        double rLimit = XBaseEval.poseLimit(selLines);
        long seed = settings.loadFacesSeed();
        Map<String, String> inv = new TreeMap<>();
        for (String l : Files.readAllLines(dir.resolve(VidTimitData.INVENTORY), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (c.length > 2 && c[2].equals("head3") && c[1].equals("3")) inv.put(c[0], c[2]);
        }
        List<String> persons = new ArrayList<>();
        for (String l : Files.readAllLines(der.resolve(VidTimitData.SELECTION), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (c.length > 1 && c[0].equals("В2") && !persons.contains(c[1])) persons.add(c[1]);
        }
        java.util.Collections.sort(persons);
        List<VidTimitData.Phrase> heads = new ArrayList<>();
        for (String p : persons) {
            if (!inv.containsKey(p)) throw new IllegalStateException("frames-b-vt: нет head3 сессии 3 у " + p);
            heads.add(new VidTimitData.Phrase(p, 3, "head3"));
        }
        Path hf = der.resolve(VT_HEAD_DET);
        StringBuilder list = new StringBuilder("# Кадры Ф3 итогового прогона (XBaseFinal frames-b-vt): R = " + rLimit + "; поворот – seed faces.seed + " + VT_HEAD_SEED
                + "\nperson\tsession\tkind\tphrase\tkey\tr\n");
        List<String[]> keys = new ArrayList<>();
        try (java.io.Writer w = Files.newBufferedWriter(hf, StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND)) {
            org.opencv.objdetect.FaceDetectorYN det = OwnDetectorThreshold.create(Paths.get(settings.loadFacesModelsDir()), FeretData.SCORE);
            VidTimitData.Dets d = new VidTimitData.Dets(dir, heads, rLimit, det, w);
            d.rows.putAll(VidTimitData.readDetections(hf));
            java.util.Random rnd = new java.util.Random(seed + VT_HEAD_SEED);
            Map<String, String> headKey = new HashMap<>();
            Map<String, Double> headR = new HashMap<>();
            for (VidTimitData.Phrase p : heads) {
                List<Integer> idx = new ArrayList<>();
                for (int i = 1; i <= d.frames.get(p).size(); i++) idx.add(i);
                java.util.Collections.shuffle(idx, rnd);
                for (int i : idx) {
                    if (!d.ok(p, i)) continue;
                    headKey.put(p.person(), VidTimitData.KEY_PREFIX + VidTimitData.detKey(p, d.frames.get(p).get(i - 1)));
                    headR.put(p.person(), d.r(p, i));
                    break;
                }
            }
            for (String l : Files.readAllLines(der.resolve(VidTimitData.SELECTION), StandardCharsets.UTF_8)) {
                String[] c = l.split("\t", -1);
                if (c.length < 9 || !c[0].equals("В2") || Integer.parseInt(c[6]) < 1) continue;
                VidTimitData.Phrase p = new VidTimitData.Phrase(c[1], Integer.parseInt(c[2]), c[3]);
                String k = VidTimitData.KEY_PREFIX + VidTimitData.detKey(p, VidTimitData.frameNames(dir, p).get(Integer.parseInt(c[6]) - 1));
                keys.add(new String[] {c[1], c[2], "фраза", c[3], k, c[7]});
            }
            for (String p : persons) {
                if (headKey.containsKey(p)) keys.add(new String[] {p, "3", "поворот", "head3", headKey.get(p), String.format(Locale.ROOT, "%.4f", headR.get(p))});
            }
        }
        keys.sort((a, b) -> (a[0] + a[1] + a[2] + a[3]).compareTo(b[0] + b[1] + b[2] + b[3]));
        Map<String, double[]> rows = VidTimitData.readDetections(der.resolve(VidTimitData.DETECTIONS));
        rows.putAll(VidTimitData.readDetections(hf));
        Files.createDirectories(der.resolve(FrameB.DIR));
        Files.createDirectories(der.resolve(FrameK2.DIR));
        StringBuilder pts = new StringBuilder();
        for (String[] k : keys) {
            list.append(String.join("\t", k)).append('\n');
            String dk = k[4].substring(VidTimitData.KEY_PREFIX.length());
            double[] row = rows.get(dk);
            if (row == null) throw new IllegalStateException("frames-b-vt: нет детекции " + dk);
            String[] q = dk.split("/");
            Mat color = Imgcodecs.imread(dir.resolve(q[0]).resolve("video").resolve(q[1]).resolve(q[2]).toString(), Imgcodecs.IMREAD_COLOR);
            if (color.empty()) throw new IOException("frames-b-vt: нет кадра " + dk);
            String name = dk.replace('/', '_') + ".png";
            Mat b = FrameB.frame(color, row, true);
            FeretData.writePng(der.resolve(FrameB.DIR).resolve(name), b);
            Mat gray = new Mat();
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
            FrameK2.F f = FrameK2.fromFull(gray, FrameK2.lm(row));
            FeretData.writePng(der.resolve(FrameK2.DIR).resolve(name), f.frame());
            pts.append(FrameK2.pointsLine(k[4], f.pts()));
            for (Mat m : new Mat[] {color, b, gray, f.frame()}) m.release();
        }
        Files.writeString(der.resolve(VT_POINTS), pts.toString(), StandardCharsets.UTF_8);
        Files.writeString(der.resolve(VT_LIST), list.toString(), StandardCharsets.UTF_8);
        StringBuilder rep = new StringBuilder("Итоговый прогон: кадры Ф3 VidTIMIT (XBaseEval frames-b-vt)\nКод: коммит " + FarMethods.commit() + "\n");
        long nHead = keys.stream().filter(k -> k[2].equals("поворот")).count();
        rep.append(String.format(Locale.ROOT, "Людей %d; кадров фраз (В2) %d; поворот head3 – найден у %d из %d.%n", persons.size(), keys.size() - nHead, nHead,
                persons.size()));
        StringBuilder fm = new StringBuilder();
        for (String sub : List.of(FrameB.DIR, FrameK2.DIR)) {
            Map<String, String> m = new TreeMap<>(FrameB.manifest(der.resolve(sub)));
            m.keySet().removeIf(n -> keys.stream().noneMatch(k -> (k[4].substring(VidTimitData.KEY_PREFIX.length()).replace('/', '_') + ".png").equals(n)));
            String t = FrameB.manifestText(m);
            fm.append("# ").append(sub).append('\n').append(t);
            rep.append(FrameB.fmt("VidTIMIT derived\\" + sub + " (ключи Ф3)", m.size(), FrameB.sha256(t)));
        }
        for (String f : List.of(VT_POINTS, VT_LIST, VidTimitData.SELECTION)) {
            String s = FrameB.sha256(der.resolve(f));
            fm.append("# ").append(f).append('\n').append(f).append('\t').append(s).append('\n');
            rep.append(FrameB.fmt("VidTIMIT derived\\" + f, 1, s));
        }
        Files.writeString(der.resolve(MANIFEST), fm.toString(), StandardCharsets.UTF_8);
        rep.append(FrameB.fmt("опись VidTIMIT " + der.resolve(MANIFEST), 1, FrameB.sha256(der.resolve(MANIFEST))));
        rep.append(String.format(Locale.ROOT, "Время: %.0f с%n", (System.nanoTime() - t0) / 1e9));
        Files.createDirectories(out);
        FarMethods.write(out.resolve("frames_b_vt.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }
}
