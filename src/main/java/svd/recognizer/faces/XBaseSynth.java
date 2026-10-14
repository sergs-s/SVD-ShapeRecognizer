package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * (а3) Синтез снимков галереи для новичков поверх L1-xs-F на смешанной галерее FERET (PLAN.md п. 5.5; задание №3 от
 * 14.10.2026, план подтверждён Хозяином; этап 1 — диапазоны по всем сессиям, оптимистично). Варианты D = 3 и D = 2:
 * подходящие люди — с ≥ D + 1 датами (по правилу равных галерей), пул проекции — остальные люди с ≥ 2 датами (вариант D = 2 —
 * только люди ровно с 2 датами), контроль — 200 прежних. Подходящие делятся пополам (java.util.Random(faces.seed + 9920));
 * проход 1 — новички половина N, старожилы O, проход 2 — наоборот. Старожилы — fa, fb первых D дат (2D снимков); новички:
 * (а) fa, fb даты 1; (б) они же + дубли до 2D; (в) они же + синтез до 2D (FaceSynth; синтез — и в шаблоне, и в классе
 * обучения MLDA). Пробы — даты с (D + 1)-й, одни и те же для новичка и старожила. Проекции: L1-xs-F (пул — новыми классами),
 * L1-tt-xs-F, L1 (без пула). Порог П2/П3 — по «чужим» галереи прохода (5 частей, seed faces.seed + 8000 + 100 + s), свой
 * на проход; итог — объединение проходов, новички и старожилы — подмножества проб при одном пороге.
 *
 * @author ssv
 */
final class XBaseSynth {

    static final long SPLIT_SEED_OFFSET = 9920;
    static final long SYNTH_SEED_OFFSET = 9930;
    /** Синтезов на человека для листов (и запас для вариантов). */
    static final int SYNTH_N = 6;
    static final int SHEET_PERSONS = 4;

    enum Kind {
        A("(а) реальные"), B("(б) + дубли"), C("(в) + синтез");

        final String label;

        Kind(String label) {
            this.label = label;
        }
    }

    /** Проекция: подпись, набор векторов (0 — none, 1 — tt), с пулом. */
    record Proj(String label, int feat, boolean pool) {}

    static final List<Proj> PROJS = List.of(new Proj("L1-xs-F", 0, true), new Proj("L1-tt-xs-F", 1, true), new Proj("L1", 0, false));

    private XBaseSynth() {
    }

    static final String OPT = "ЭТАП 1 – ОПТИМИСТИЧНО: диапазоны синтеза по всем сессиям базы, включая даты проб.";

    static void run(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-synth: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Path sheets = out.resolve("synth");
        Files.createDirectories(sheets);
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);

        // ---- Диапазоны этапа 1: шум и A по кадрам «а» всех отобранных снимков, подобия YuNet → ручная разметка.
        Set<String> all = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.ctrlKeys)) m.values().forEach(all::addAll);
        List<Double> noise = new ArrayList<>();
        List<Double> asym = new ArrayList<>();
        Map<String, double[]> cache = new HashMap<>();
        XBaseA0.Source src = su.src;
        for (String k : all) {
            double[] m = XBaseA0Check.measures(src, k, cache);
            asym.add(m[0]);
            noise.add(m[1]);
        }
        List<double[]> align = new ArrayList<>();
        List<Double> shifts = new ArrayList<>();
        for (Img im : su.imgs) {
            double[][] man = FeretData.manual4(im);
            double[] row = su.det.get(im.key());
            if (man == null || row == null || su.rOf.get(im.key()) > tpl.rLimit) continue;
            double[] d = FaceSynth.alignDelta(FeretData.yunet4(row), man);
            align.add(d);
            shifts.add(FaceSynth.eyeShift(d));
        }
        FaceSynth.Ranges ranges = new FaceSynth.Ranges(sortedArr(noise), sortedArr(asym), align);

        // ---- Подходящие люди (≥ 3 дат – D = 2, ≥ 4 – D = 3) и их синтез (один набор на человека).
        List<String> elig2 = new ArrayList<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= 3) elig2.add(e.getKey());
        Random rnd = new Random(tpl.seed + SYNTH_SEED_OFFSET);
        Map<String, List<String>> synthKeys = new LinkedHashMap<>();
        Map<String, Mat> synthFr = new LinkedHashMap<>();
        Map<String, List<FaceSynth.Params>> synthPar = new LinkedHashMap<>();
        for (String p : elig2) {
            List<String> real = pool.get(p).get(0);
            List<String> ks = new ArrayList<>();
            List<FaceSynth.Params> ps = new ArrayList<>();
            for (int i = 0; i < SYNTH_N; i++) {
                String base = real.get(i % real.size());
                Img im = su.byKey.get(base);
                Mat color = FeretData.decode(im.file());
                FaceSynth.Params prm = FaceSynth.sample(ranges, rnd);
                Mat a = FaceSynth.synth(color, su.det.get(base), prm, rnd);
                color.release();
                String key = base + "#s" + i;
                synthFr.put(key, a);
                ks.add(key);
                ps.add(prm);
                su.rOf.put(key, su.rOf.get(base));
                su.imageNo.put(key, 1000 + 10 * su.imageNo.get(base) + i);
            }
            synthKeys.put(p, ks);
            synthPar.put(p, ps);
        }

        // ---- Векторы: все снимки ролей и пула, когорта, синтез.
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : all) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        fr.putAll(synthFr);
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);

        // ---- Листы проверки синтеза (первые SHEET_PERSONS подходящих D = 3).
        List<String> elig3 = new ArrayList<>();
        for (String p : elig2) if (pool.get(p).size() >= 4) elig3.add(p);
        sheets(sheets, elig3.subList(0, Math.min(SHEET_PERSONS, elig3.size())), pool, synthKeys, synthPar, fr, tpl);
        for (Mat m : fr.values()) m.release();

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а3) синтез снимков галереи для новичков поверх L1-xs-F (XBaseEval feret-synth): Color FERET — только локально\n");
        r.append(OPT).append('\n');
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры — как feret, кэш derived/; синтез — из исходных снимков, в памяти); когорта FEI — "
                + "экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(XBaseFeret.EQ_NOTE);
        r.append(String.format(Locale.ROOT, "Диапазоны синтеза (этап 1, по %d кадрам «а» всех сессий; выравнивание — по %d снимкам с ручной разметкой):%n"
                + "  σ шума (Immerkær): медиана %.2f, 5–95 %% [%.2f; %.2f]; A = |ln(m_L/m_R)|: медиана %.3f, 75 %% %.3f, 95 %% %.3f;%n"
                + "  выравнивание – подобие «4 точки YuNet → ручные 4 точки», смещение глаз: медиана %.1f %% межглазья (сверка: измерено 5,5 %%);%n"
                + "  тень – в %.0f %% синтезов (граница 46 ± %.0f px, сглаживание %.0f px, глубина 1 − e^(−A), A из верхней четверти), иначе –%n"
                + "  плавный перепад exp(±A·(2x/(W−1) − 1)/2); шум – гауссов до σ цели. Яркость, контраст, гамма – не синтезируются.%n",
                all.size(), align.size(), FaceSynth.median(ranges.noise()), XBaseA0Check.q(ranges.noise(), 0.05), XBaseA0Check.q(ranges.noise(), 0.95),
                FaceSynth.median(ranges.asym()), XBaseA0Check.q(ranges.asym(), 0.75), XBaseA0Check.q(ranges.asym(), 0.95),
                100 * FaceSynth.median(sortedArr(shifts)), 100 * FaceSynth.SHADOW, FaceSynth.SHADOW_X, FaceSynth.SHADOW_S));
        r.append(String.format(Locale.ROOT, "Синтез: java.util.Random(seed %d + %d), по %d на человека (люди с ≥ 3 датами, %d), основа – fa и fb даты 1 по "
                + "очереди; в варианте D – первые 2D − 2. Листы – %s (%d человек: кадр «а» и tt, реальные и синтезы).%n", tpl.seed, SYNTH_SEED_OFFSET,
                SYNTH_N, elig2.size(), sheets, Math.min(SHEET_PERSONS, elig3.size())));
        r.append("Проекции: L1-xs-F – LBP u2 (8, 2) 7×7 + MLDA, none, пул новыми классами (человек – класс); L1-tt-xs-F – то же на выходе Tan – Triggs;\n"
                + "L1 – без пула; когорта FEI 20; зеркала и выбор X / X+mirror по |r| пробы, r0 = " + tpl.r0 + ". Порог П2/П3 – по «чужим» галереи\n"
                + "прохода (5 частей), свой на проход; итог – объединение двух проходов; новички и старожилы – части проб при одном пороге.\n");
        r.append("Оговорка: людей мало (D = 3 – 25, D = 2 – 61); различимы только крупные эффекты. Числа вариантов D = 3 и D = 2 между собой не\n"
                + "сравнивать – только строки внутри варианта.\n");
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям (подходящие и контроль): B = %d, seed = %d; «*» — интервал разности не содержит "
                + "0.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-synth %s; снимков %d; seed %d; r0 %s; пул %d; синтез %d×%d; №8 "
                + "галерея fa/fb", dir, su.imgs.size(), tpl.seed, tpl.r0, pool.size(), elig2.size(), SYNTH_N));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_synth_eq"), fp);
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        for (int dd : new int[] {3, 2}) {
            int need = dd + 1;
            List<String> elig = new ArrayList<>();
            for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= need) elig.add(e.getKey());
            Map<String, List<List<String>>> projPool = new TreeMap<>();
            for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) {
                if (elig.contains(e.getKey())) continue;
                if (dd == 2 && e.getValue().size() != 2) continue;
                projPool.put(e.getKey(), e.getValue());
            }
            List<String> perm = new ArrayList<>(elig);
            Collections.shuffle(perm, new Random(tpl.seed + SPLIT_SEED_OFFSET));
            int half = (perm.size() + 1) / 2;
            List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
            TreeSet<String> ps0 = new TreeSet<>(elig);
            ps0.addAll(ro.ctrl);
            List<String> persons = new ArrayList<>(ps0);
            XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
            List<String> own = new ArrayList<>(new TreeSet<>(elig));
            Map<String, List<String>> pro = new LinkedHashMap<>();
            Map<String, List<String>> oldGal = new LinkedHashMap<>();
            for (String p : own) {
                List<List<String>> ds = pool.get(p);
                List<String> g = new ArrayList<>();
                for (int j = 0; j < dd; j++) g.addAll(ds.get(j));
                oldGal.put(p, XBaseFeret.sorted(g));
                List<String> pr = new ArrayList<>();
                for (int j = dd; j < ds.size(); j++) pr.addAll(ds.get(j));
                pro.put(p, XBaseFeret.sorted(pr));
            }
            int extraN = 2 * dd - 2;
            // Дубли: ключ #d, вектор — копия реального снимка.
            for (String p : own) {
                List<String> real = pool.get(p).get(0);
                for (int i = 0; i < extraN; i++) {
                    String b = real.get(i % real.size());
                    String key = b + "#d" + i;
                    for (XBaseEval x : xs) {
                        x.vec.get(V.L1).put(key, x.vec.get(V.L1).get(b));
                        x.mir.get(V.L1).put(key, x.mir.get(V.L1).get(b));
                    }
                    su.rOf.put(key, su.rOf.get(b));
                    su.imageNo.put(key, 2000 + 10 * su.imageNo.get(b) + i);
                }
            }
            r.append(String.format(Locale.ROOT, "%n%n######## Вариант D = %d: старожил – %d дат (%d снимков), новичок – 1 дата (2 снимка) + %d; %s ########%n", dd, dd,
                    2 * dd, extraN, OPT));
            r.append(String.format(Locale.ROOT, "Подходящих (≥ %d дат): %d; половины (seed + %d): %d / %d; пул проекции: %d людей (%d снимков); проб у подходящих %d.%n",
                    need, elig.size(), SPLIT_SEED_OFFSET, halves.get(0).size(), halves.get(1).size(), projPool.size(),
                    projPool.values().stream().mapToInt(ds -> ds.stream().mapToInt(List::size).sum()).sum(), pro.values().stream().mapToInt(List::size).sum()));
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            StringBuilder hubT = new StringBuilder("\n=== Перехват новичками (оба прохода; модель по позе пробы; " + OPT + ") ===\n"
                    + "строка | контроль: лучший – новичок | пробы других своих: неверный лучший – новичок | пробы старожилов: неверный лучший – новичок\n");
            StringBuilder synthT = new StringBuilder("\n=== Синтетические снимки в обучении MLDA (без отражений; с отражениями – вдвое) ===\n");
            for (Proj pj : PROJS) {
                List<List<String>> extra = pj.pool() ? XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra() : List.of();
                for (Kind kind : Kind.values()) {
                    List<FeiDecompose.Cfg> nc = new ArrayList<>();
                    List<FeiDecompose.Cfg> oc = new ArrayList<>();
                    List<FeiThreshold.Imp> imps = new ArrayList<>();
                    int[] hub = new int[6];
                    for (int ps = 0; ps < 2; ps++) {
                        Set<String> newbies = halves.get(ps);
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        int synthUsed = 0;
                        for (String p : own) {
                            if (!newbies.contains(p)) {
                                gal.put(p, oldGal.get(p));
                                continue;
                            }
                            List<String> g = new ArrayList<>(pool.get(p).get(0));
                            if (kind == Kind.B) {
                                List<String> real = pool.get(p).get(0);
                                for (int i = 0; i < extraN; i++) g.add(real.get(i % real.size()) + "#d" + i);
                            } else if (kind == Kind.C) {
                                g.addAll(synthKeys.get(p).subList(0, extraN));
                                synthUsed += extraN;
                            }
                            gal.put(p, g);
                        }
                        if (pj == PROJS.get(0)) synthT.append(String.format(Locale.ROOT, "%s, проход %d: %d%n", kind.label, ps + 1, synthUsed));
                        XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                        List<List<String>> aug = new ArrayList<>(cohort);
                        aug.addAll(extra);
                        g.cohort = aug;
                        List<Probe> probes = XBaseEval.probes(g, pro);
                        List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + 30 + dd);
                        String name = String.format(Locale.ROOT, "d%d_p%d_%s_%s", dd, ps, pj.label(), kind.name());
                        String ls = cp.read(name);
                        if (ls == null) {
                            long t0 = System.nanoTime();
                            ls = XBaseEval.fitLines(g, xs[pj.feat()], EnumSet.of(V.L1), gal, probes, parts, 30 + dd);
                            double s = (System.nanoTime() - t0) / 1e9;
                            fitS += s;
                            cp.save(name, ls);
                            System.out.printf(Locale.ROOT, "feret-synth: D %d, pass %d, %s, %s ready (%.0f s)%n", dd, ps + 1, pj.label(), kind.name(), s);
                        }
                        lines.append("# D ").append(dd).append(", проход ").append(ps + 1).append(", проекция ").append(pj.label()).append(", строка ")
                                .append(kind.name()).append('\n').append(ls);
                        XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, ls);
                        FeiDecompose.Cfg c = p.cfg.get(V.L1);
                        List<Integer> ni = new ArrayList<>();
                        List<Integer> oi = new ArrayList<>();
                        int idx = 0;
                        for (Probe q : probes) {
                            if (!q.role().equals(FeiScores.OWN)) continue;
                            (newbies.contains(q.person()) ? ni : oi).add(idx++);
                        }
                        nc.add(XBaseFeret.subset(c, ni));
                        oc.add(XBaseFeret.subset(c, oi));
                        imps.add(p.imp.get(V.L1));
                        Map<String, Row> rows = new HashMap<>();
                        for (String l : ls.lines().toList()) {
                            Row row = FeiScores.parse(l);
                            rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
                        }
                        for (Probe q : probes) {
                            Row row = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), g.rOf.get(q.key()) <= g.r0);
                            boolean toNew = newbies.contains(row.bestId());
                            if (!q.role().equals(FeiScores.OWN)) {
                                hub[0]++;
                                if (toNew) hub[1]++;
                            } else {
                                hub[2]++;
                                if (toNew && !row.bestId().equals(row.trueId())) hub[3]++;
                                if (!newbies.contains(q.person())) {
                                    hub[4]++;
                                    if (toNew) hub[5]++;
                                }
                            }
                        }
                    }
                    String ln = pj.label() + " новички " + kind.label;
                    String lo = pj.label() + " старожилы при " + kind.label;
                    res.put(ln, FeiThreshold.evaluate(ln, V.L1.ratioId(), nc, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    res.put(lo, FeiThreshold.evaluate(lo, V.L1.ratioId(), oc, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    hubT.append(String.format(Locale.ROOT, "%s %s | %d/%d = %.2f %% | %d/%d = %.2f %% | %d/%d = %.2f %%%n", pj.label(), kind.label, hub[1], hub[0],
                            100.0 * hub[1] / hub[0], hub[3], hub[2], 100.0 * hub[3] / hub[2], hub[5], hub[4], 100.0 * hub[5] / hub[4]));
                }
                String n = pj.label() + " новички ";
                String o = pj.label() + " старожилы при ";
                pairs.add(new String[] {n + Kind.C.label, n + Kind.A.label});
                pairs.add(new String[] {n + Kind.C.label, n + Kind.B.label});
                pairs.add(new String[] {n + Kind.B.label, n + Kind.A.label});
                pairs.add(new String[] {n + Kind.C.label, o + Kind.C.label});
                pairs.add(new String[] {n + Kind.A.label, o + Kind.A.label});
                pairs.add(new String[] {o + Kind.C.label, o + Kind.A.label});
            }
            r.append(synthT).append(hubT);
            XBaseXsF.flush(r, res, pairs, persons.size(), tpl, String.format(Locale.ROOT, "Вариант D = %d — %s", dd, OPT));
        }
        r.append(String.format(Locale.ROOT, "%nОбучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        XBaseOther.writeScores(out.resolve("xbase_feret_synth_scores.tsv.gz"), "# FERET (а3), XBaseSynth feret-synth, только локально; " + OPT
                + "; блоки — D, проход, проекция, строка\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_synth.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    static double[] sortedArr(List<Double> l) {
        double[] a = l.stream().mapToDouble(Double::doubleValue).toArray();
        java.util.Arrays.sort(a);
        return a;
    }

    /** Листы: на человека две строки — кадр «а» и tt: реальный fa, реальный fb, затем синтезы (подпись — параметры). */
    static void sheets(Path dir, List<String> persons, Map<String, List<List<String>>> pool, Map<String, List<String>> synthKeys,
                       Map<String, List<FaceSynth.Params>> synthPar, Map<String, Mat> fr, XBaseEval.Gt tpl) throws IOException {
        int n = 0;
        for (String p : persons) {
            List<String> keys = new ArrayList<>(pool.get(p).get(0));
            keys.addAll(synthKeys.get(p));
            List<Mat> top = new ArrayList<>();
            List<Mat> bottom = new ArrayList<>();
            for (int i = 0; i < keys.size(); i++) {
                Mat a = fr.get(keys.get(i));
                Mat t1 = XBaseA0.tile(a);
                Mat t2 = XBaseA0.ttTile(a, tpl.params);
                String cap;
                if (i < pool.get(p).get(0).size()) {
                    cap = "real";
                } else {
                    FaceSynth.Params q = synthPar.get(p).get(i - pool.get(p).get(0).size());
                    cap = String.format(Locale.ROOT, "%s A%.2f s%.1f", q.shadow() ? "sh" : "gr", q.sign() * q.asym(), q.noise());
                }
                Imgproc.putText(t1, cap, new Point(2, 10), Imgproc.FONT_HERSHEY_PLAIN, 0.7, new Scalar(0, 255, 0), 1);
                top.add(t1);
                bottom.add(t2);
            }
            Mat r1 = new Mat();
            Mat r2 = new Mat();
            Core.hconcat(top, r1);
            Core.hconcat(bottom, r2);
            Mat sheet = new Mat();
            Core.vconcat(List.of(r1, r2), sheet);
            Imgproc.putText(sheet, "FERET " + p + " date1: real fa/fb + synth (top - frame a, bottom - tt); stage 1", new Point(4, sheet.rows() - 4),
                    Imgproc.FONT_HERSHEY_PLAIN, 0.8, new Scalar(0, 0, 255), 1);
            FeretData.writePng(dir.resolve(String.format(Locale.ROOT, "feret_synth_%02d.png", ++n)), sheet);
            for (Mat m : top) m.release();
            for (Mat m : bottom) m.release();
            r1.release();
            r2.release();
            sheet.release();
        }
    }
}
