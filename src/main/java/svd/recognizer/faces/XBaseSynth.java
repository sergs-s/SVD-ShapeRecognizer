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

    // ---------------------------------------------------------------- feret-synth2 (этап 2)

    static final long SYNTH2_SEED_OFFSET = 9940;
    /** Меньше стольких снимков на распределение этапа 2 — берётся распределение этапа 1 (строка в отчёте). */
    static final int RANGE_MIN = 100;

    /** Строки этапа 2: (а), (б), (в) этапа 1 — из файла оценок feret-synth, (в2) — синтез этапа 2 (считается). */
    enum K2 {
        A("(а) реальные"), B("(б) + дубли"), C("(в) + синтез"), C2("(в2) + синтез, этап 2");

        final String label;

        K2(String label) {
            this.label = label;
        }
    }

    /** Блоки файла оценок feret-synth: «D d, проход p, проекция X, строка K» → строки. */
    static Map<String, String> synthBlocks(Path scores) throws IOException {
        Map<String, StringBuilder> b = new LinkedHashMap<>();
        String cur = null;
        try (java.io.BufferedReader in = new java.io.BufferedReader(new java.io.InputStreamReader(new java.util.zip.GZIPInputStream(Files.newInputStream(scores),
                1 << 16), java.nio.charset.StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("# D ")) {
                    cur = l.substring(2);
                    b.put(cur, new StringBuilder());
                } else if (!l.startsWith("#") && !l.startsWith("method\t") && cur != null) {
                    b.get(cur).append(l).append('\n');
                }
            }
        }
        Map<String, String> out = new LinkedHashMap<>();
        b.forEach((k, v) -> out.put(k, v.toString()));
        return out;
    }

    /**
     * (а3) этап 2 на FERET (задание №4 от 14.10.2026, план подтверждён Хозяином): тот же протокол, что feret-synth (D = 3 и
     * D = 2, те же половины и пробы); диапазоны синтеза — только по людям вне подходящих и вне контроля, без снимков дат проб
     * подходящих; строки (а), (б), (в) этапа 1 — из файла оценок feret-synth (сводка сверяется с его отчётом побайтно), (в2) —
     * синтез этапа 2 (java.util.Random(faces.seed + 9940 + D)).
     */
    static void run2(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-synth2: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Path oldScores = out.resolve("xbase_feret_synth_scores.tsv.gz");
        Path oldReport = out.resolve("xbase_feret_synth.txt");
        if (!Files.exists(oldScores) || !Files.exists(oldReport)) throw new IllegalStateException("feret-synth2: нужны отчёт и файл оценок feret-synth в " + out);
        Map<String, String> oldBlocks = synthBlocks(oldScores);
        Set<String> oldLines = new HashSet<>(Files.readAllLines(oldReport, java.nio.charset.StandardCharsets.UTF_8).stream().map(s -> s.replace("\r", "")).toList());
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);

        // Ключи синтеза этапа 1 (только ключи, номера и позы — для разбора строк из файла).
        List<String> elig2 = new ArrayList<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= 3) elig2.add(e.getKey());
        Map<String, List<String>> synth1 = new LinkedHashMap<>();
        for (String p : elig2) {
            List<String> real = pool.get(p).get(0);
            List<String> ks = new ArrayList<>();
            for (int i = 0; i < SYNTH_N; i++) {
                String b = real.get(i % real.size());
                String key = b + "#s" + i;
                ks.add(key);
                su.rOf.put(key, su.rOf.get(b));
                su.imageNo.put(key, 1000 + 10 * su.imageNo.get(b) + i);
            }
            synth1.put(p, ks);
        }

        // Векторы снимков ролей, пула, контроля и когорты.
        Set<String> all = new TreeSet<>();
        for (Map<String, List<String>> m : List.of(roAll.gal1, roAll.pro1, roAll.ctrlKeys)) m.values().forEach(all::addAll);
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
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();

        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а3) этап 2 синтеза для новичков поверх L1-xs-F (XBaseEval feret-synth2): Color FERET — только локально\n");
        r.append("Этап 2: диапазоны синтеза – только по людям вне подходящих и вне контроля, без снимков дат проб подходящих; строки (а), (б), (в) – этап 1\n"
                + "(" + OPT + ") – из файла оценок feret-synth ").append(oldScores.getFileName()).append(" (").append(FeiThreshold.sha(oldScores)).append("),\n"
                + "сводка сверена с ").append(oldReport.getFileName()).append(" побайтно.\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        r.append(XBaseFeret.EQ_NOTE);
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям (подходящие и контроль): B = %d, seed = %d; «*» — интервал разности не содержит "
                + "0.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-synth2 %s; снимков %d; seed %d; r0 %s; пул %d; этап 1 %s; №8 "
                + "галерея fa/fb", dir, su.imgs.size(), tpl.seed, tpl.r0, pool.size(), FeiThreshold.sha(oldScores)));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_synth2_eq"), fp);
        StringBuilder lines = new StringBuilder();
        Set<String> checkLabels = new HashSet<>();
        double fitS = 0;
        Map<String, double[]> cache = new HashMap<>();
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
            Set<String> probeDates = new TreeSet<>();
            for (String p : own) {
                List<List<String>> ds = pool.get(p);
                List<String> g = new ArrayList<>();
                for (int j = 0; j < dd; j++) g.addAll(ds.get(j));
                oldGal.put(p, XBaseFeret.sorted(g));
                List<String> pr = new ArrayList<>();
                for (int j = dd; j < ds.size(); j++) pr.addAll(ds.get(j));
                pro.put(p, XBaseFeret.sorted(pr));
                for (String k : pr) probeDates.add(su.byKey.get(k).date());
            }
            int extraN = 2 * dd - 2;
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

            // ---- Диапазоны этапа 2: люди вне подходящих и вне контроля, даты не из дат проб.
            Set<String> exclP = new HashSet<>(elig);
            exclP.addAll(ro.ctrl);
            List<Double> noise = new ArrayList<>();
            List<Double> asym = new ArrayList<>();
            List<double[]> align = new ArrayList<>();
            List<Double> shifts = new ArrayList<>();
            Set<String> rangePersons = new TreeSet<>();
            for (Img im : su.imgs) {
                double[] row = su.det.get(im.key());
                if (row == null || su.rOf.get(im.key()) > tpl.rLimit) continue;
                if (exclP.contains(im.person()) || probeDates.contains(im.date())) continue;
                rangePersons.add(im.person());
                double[] m = XBaseA0Check.measures(su.src, im.key(), cache);
                asym.add(m[0]);
                noise.add(m[1]);
                double[][] man = FeretData.manual4(im);
                if (man != null) {
                    double[] d = FaceSynth.alignDelta(FeretData.yunet4(row), man);
                    align.add(d);
                    shifts.add(FaceSynth.eyeShift(d));
                }
            }
            // Мало размеченных снимков вне дат проб – подобия выравнивания у тех же людей без исключения дат (ошибка
            // выравнивания – свойство детектора, а не сессии).
            String alignNote = null;
            if (align.size() < RANGE_MIN) {
                int before = align.size();
                align.clear();
                shifts.clear();
                for (Img im : su.imgs) {
                    double[] row = su.det.get(im.key());
                    if (row == null || su.rOf.get(im.key()) > tpl.rLimit || exclP.contains(im.person())) continue;
                    double[][] man = FeretData.manual4(im);
                    if (man == null) continue;
                    double[] d = FaceSynth.alignDelta(FeretData.yunet4(row), man);
                    align.add(d);
                    shifts.add(FaceSynth.eyeShift(d));
                }
                alignNote = String.format(Locale.ROOT, "Выравнивание: вне дат проб размеченных снимков %d (< %d) – подобия взяты у тех же людей вне подходящих и "
                        + "контроля без исключения дат (%d снимков).%n", before, RANGE_MIN, align.size());
            }
            if (noise.size() < RANGE_MIN || align.size() < RANGE_MIN) {
                throw new IllegalStateException(String.format(Locale.ROOT, "feret-synth2 D = %d: мало снимков для диапазонов этапа 2 (σ и A %d, выравнивание %d)", dd,
                        noise.size(), align.size()));
            }
            FaceSynth.Ranges ranges = new FaceSynth.Ranges(sortedArr(noise), sortedArr(asym), align);

            // ---- Синтез этапа 2 для подходящих (основа – fa и fb даты 1).
            Random rnd = new Random(tpl.seed + SYNTH2_SEED_OFFSET + dd);
            Map<String, List<String>> synth2 = new LinkedHashMap<>();
            Map<String, Mat> sf = new LinkedHashMap<>();
            for (String p : own) {
                List<String> real = pool.get(p).get(0);
                List<String> ks = new ArrayList<>();
                for (int i = 0; i < extraN; i++) {
                    String b = real.get(i % real.size());
                    Mat color = FeretData.decode(su.byKey.get(b).file());
                    FaceSynth.Params prm = FaceSynth.sample(ranges, rnd);
                    Mat a = FaceSynth.synth(color, su.det.get(b), prm, rnd);
                    color.release();
                    String key = b + "#t" + dd + "_" + i;
                    sf.put(key, a);
                    ks.add(key);
                    su.rOf.put(key, su.rOf.get(b));
                    su.imageNo.put(key, 3000 + 100 * dd + 10 * su.imageNo.get(b) + i);
                }
                synth2.put(p, ks);
            }
            XBaseEval h0 = new XBaseEval();
            h0.vectors(sf, EnumSet.of(V.L1), tpl.params);
            XBaseEval h1 = new XBaseEval();
            h1.vectors(sf, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
            for (Mat m : sf.values()) m.release();
            xs[0].vec.get(V.L1).putAll(h0.vec.get(V.L1));
            xs[0].mir.get(V.L1).putAll(h0.mir.get(V.L1));
            xs[1].vec.get(V.L1).putAll(h1.vec.get(V.L1));
            xs[1].mir.get(V.L1).putAll(h1.mir.get(V.L1));

            r.append(String.format(Locale.ROOT, "%n%n######## Вариант D = %d (этап 2; строки (а), (б), (в) – этап 1, %s) ########%n", dd, OPT));
            r.append(String.format(Locale.ROOT, "Подходящих: %d; половины %d / %d; пул проекции %d людей; проб у подходящих %d; дат проб (исключены из диапазонов) %d.%n",
                    elig.size(), halves.get(0).size(), halves.get(1).size(), projPool.size(), pro.values().stream().mapToInt(List::size).sum(), probeDates.size()));
            r.append(String.format(Locale.ROOT, "Диапазоны этапа 2: людей %d, снимков %d (σ, A), с ручной разметкой %d (выравнивание); σ шума медиана %.2f, 5–95 %% "
                    + "[%.2f; %.2f]; A медиана %.3f, 75 %% %.3f, 95 %% %.3f; смещение глаз медиана %.1f %% межглазья. Синтез этапа 2 – seed %d + %d + %d, по %d на "
                    + "человека.%n", rangePersons.size(), noise.size(), align.size(), FaceSynth.median(ranges.noise()), XBaseA0Check.q(ranges.noise(), 0.05),
                    XBaseA0Check.q(ranges.noise(), 0.95), FaceSynth.median(ranges.asym()), XBaseA0Check.q(ranges.asym(), 0.75), XBaseA0Check.q(ranges.asym(), 0.95),
                    100 * FaceSynth.median(sortedArr(shifts)), tpl.seed, SYNTH2_SEED_OFFSET, dd, extraN));
            if (alignNote != null) r.append(alignNote);
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            StringBuilder hubT = new StringBuilder("\n=== Перехват новичками (оба прохода; модель по позе пробы) ===\n"
                    + "строка | контроль: лучший – новичок | справедливая доля (новички среди людей галереи) | пробы других своих: неверный лучший – новичок\n");
            for (Proj pj : PROJS) {
                List<List<String>> extra = pj.pool() ? XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra() : List.of();
                for (K2 kind : K2.values()) {
                    List<FeiDecompose.Cfg> nc = new ArrayList<>();
                    List<FeiDecompose.Cfg> oc = new ArrayList<>();
                    List<FeiThreshold.Imp> imps = new ArrayList<>();
                    int[] hub = new int[4];
                    double fair = 0;
                    for (int ps = 0; ps < 2; ps++) {
                        Set<String> newbies = halves.get(ps);
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        for (String p : own) {
                            if (!newbies.contains(p)) {
                                gal.put(p, oldGal.get(p));
                                continue;
                            }
                            List<String> g = new ArrayList<>(pool.get(p).get(0));
                            List<String> real = pool.get(p).get(0);
                            switch (kind) {
                                case B -> {
                                    for (int i = 0; i < extraN; i++) g.add(real.get(i % real.size()) + "#d" + i);
                                }
                                case C -> g.addAll(synth1.get(p).subList(0, extraN));
                                case C2 -> g.addAll(synth2.get(p));
                                default -> {
                                }
                            }
                            gal.put(p, g);
                        }
                        fair += newbies.size() / (double) own.size() / 2;
                        XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                        List<List<String>> aug = new ArrayList<>(cohort);
                        aug.addAll(extra);
                        g.cohort = aug;
                        List<Probe> probes = XBaseEval.probes(g, pro);
                        List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + 30 + dd);
                        String ls;
                        if (kind != K2.C2) {
                            String bn = String.format(Locale.ROOT, "D %d, проход %d, проекция %s, строка %s", dd, ps + 1, pj.label(), kind.name());
                            ls = oldBlocks.get(bn);
                            if (ls == null) throw new IllegalStateException("feret-synth2: нет блока «" + bn + "» в " + oldScores);
                        } else {
                            String name = String.format(Locale.ROOT, "d%d_p%d_%s_C2", dd, ps, pj.label());
                            ls = cp.read(name);
                            if (ls == null) {
                                long t0 = System.nanoTime();
                                ls = XBaseEval.fitLines(g, xs[pj.feat()], EnumSet.of(V.L1), gal, probes, parts, 30 + dd);
                                double s = (System.nanoTime() - t0) / 1e9;
                                fitS += s;
                                cp.save(name, ls);
                                System.out.printf(Locale.ROOT, "feret-synth2: D %d, pass %d, %s, C2 ready (%.0f s)%n", dd, ps + 1, pj.label(), s);
                            }
                            lines.append("# D ").append(dd).append(", проход ").append(ps + 1).append(", проекция ").append(pj.label()).append(", строка C2\n")
                                    .append(ls);
                        }
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
                            }
                        }
                    }
                    String ln = pj.label() + " новички " + kind.label;
                    String lo = pj.label() + " старожилы при " + kind.label;
                    res.put(ln, FeiThreshold.evaluate(ln, V.L1.ratioId(), nc, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    res.put(lo, FeiThreshold.evaluate(lo, V.L1.ratioId(), oc, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                    if (kind != K2.C2) {
                        checkLabels.add(ln);
                        checkLabels.add(lo);
                    }
                    hubT.append(String.format(Locale.ROOT, "%s %s | %d/%d = %.2f %% | %.1f %% | %d/%d = %.2f %%%n", pj.label(), kind.label, hub[1], hub[0],
                            100.0 * hub[1] / hub[0], 100 * fair, hub[3], hub[2], 100.0 * hub[3] / hub[2]));
                }
                String n = pj.label() + " новички ";
                String o = pj.label() + " старожилы при ";
                pairs.add(new String[] {n + K2.C2.label, n + K2.A.label});
                pairs.add(new String[] {n + K2.C2.label, n + K2.B.label});
                pairs.add(new String[] {n + K2.C2.label, n + K2.C.label});
                pairs.add(new String[] {n + K2.C.label, n + K2.A.label});
                pairs.add(new String[] {n + K2.C2.label, o + K2.C2.label});
                pairs.add(new String[] {o + K2.C2.label, o + K2.A.label});
            }
            r.append(hubT);
            XBaseXsF.flush(r, res, pairs, persons.size(), tpl, String.format(Locale.ROOT, "Вариант D = %d, этап 2 рядом с этапом 1", dd));
        }
        // Сверка: строки таблиц с подписями (а), (б), (в) этапа 1 – как в отчёте feret-synth.
        int matched = 0;
        for (String l : r.toString().replace("\r", "").split("\n")) {
            int bar = l.indexOf(" | ");
            if (bar < 0 || !checkLabels.contains(l.substring(0, bar))) continue;
            if (!oldLines.contains(l)) throw new IllegalStateException("feret-synth2: строка этапа 1 не совпала с " + oldReport.getFileName() + ":\n" + l);
            matched++;
        }
        r.append(String.format(Locale.ROOT, "%nСверка с %s: строк этапа 1 совпало %d. Обучение и оценки этапа 2: %.0f с (без блоков из контрольной точки); потоков %d.%n",
                oldReport.getFileName(), matched, fitS, tpl.threads));
        XBaseOther.writeScores(out.resolve("xbase_feret_synth2_scores.tsv.gz"), "# FERET (а3) этап 2, XBaseSynth feret-synth2, только локально; блоки — D, "
                + "проход, проекция, строка C2 (строки A, B, C – в файле оценок feret-synth)\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_synth2.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-synth

    static final int GT_PART_S = 1950;
    static final long GT_SYNTH_SEED_OFFSET = 9960;
    static final int GT_EXTRA = 4;

    /** {A, σ шума} кадра «а» (как XBaseA0Check.measures). */
    static double[] measure(Mat a) {
        int w = a.cols();
        double l = 0;
        double r = 0;
        int nl = 0;
        int nr = 0;
        byte[] px = new byte[(int) a.total()];
        a.get(0, 0, px);
        for (int y = 20; y <= 100; y++) {
            for (int x = 10; x <= 81; x++) {
                int v = px[y * w + x] & 0xff;
                if (x <= 45) {
                    l += v;
                    nl++;
                } else {
                    r += v;
                    nr++;
                }
            }
        }
        return new double[] {Math.abs(Math.log(Math.max(l / nl, 1e-9) / Math.max(r / nr, 1e-9))), XBaseQuality.frameMeasures(a)[2]};
    }

    /** Строки YuNet GT из detections.tsv экспорта: «человек/номер» → строка (15 чисел). */
    static Map<String, double[]> gtRows(Path exportDir) throws IOException {
        Map<String, double[]> out = new HashMap<>();
        for (String l : Files.readAllLines(exportDir.resolve(FarExport.DETECTIONS), java.nio.charset.StandardCharsets.UTF_8)) {
            String[] f = l.split("\t");
            if (f.length < 20 || !f[0].equals("gt") || !f[4].equals("1")) continue;
            double[] row = new double[15];
            for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(f[5 + i]);
            out.put(f[1] + "/" + Integer.parseInt(f[2]), row);
        }
        return out;
    }

    /** Календарные даты снимков GT (reports/gt_sessions.tsv экспорта): «человек/номер» → ГГГГ:ММ:ДД. */
    static Map<String, String> gtDates(Path exportDir) throws IOException {
        Map<String, String> out = new HashMap<>();
        for (String l : Files.readAllLines(exportDir.resolve("reports").resolve("gt_sessions.tsv"), java.nio.charset.StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t")) continue;
            String[] f = l.split("\t");
            if (f.length < 3 || f[2].length() < 10) continue;
            out.put(f[0] + "/" + Integer.parseInt(f[1]), f[2].substring(0, 10));
        }
        return out;
    }

    /**
     * (а3) этап 2 на GT (задание №4 от 14.10.2026): протокол gt (межсессионно и внутри сессии), все свои — «новички» (одна дата
     * галереи); строки (а) без добавок, (б) + GT_EXTRA дублей, (в) + GT_EXTRA синтезов (FaceSynth из исходных снимков gt_db);
     * диапазоны: σ шума и A — по кадрам «а» GT людей вне части q своего (5 частей, seed faces.seed + 9950; свои других частей и
     * контрольные односессионные), без снимков дат проб части q; выравнивание — подобия FERET (ручная разметка есть только
     * там). Проекции L1-xs-F (пул FERET, 173 человека), L1-tt-xs-F, L1.
     */
    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-synth: только равные галереи FERET (faces.feret.gallery = eq)");
        String gtDir = settings.loadFacesGtDir();
        if (gtDir == null) throw new IllegalStateException("gt-synth: задайте faces.gt.dir / FACES_GT_DIR (исходные снимки gt_db)");
        Path gtDb = Path.of(gtDir);
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Map<String, double[]> rowsGt = gtRows(dir);
        Map<String, String> dateGt = gtDates(dir);

        // Выравнивание – подобия FERET по всем снимкам с ручной разметкой.
        List<double[]> align = new ArrayList<>();
        for (Img im : su.imgs) {
            double[][] man = FeretData.manual4(im);
            double[] row = su.det.get(im.key());
            if (man == null || row == null || su.rOf.get(im.key()) > su.tpl.rLimit) continue;
            align.add(FaceSynth.alignDelta(FeretData.yunet4(row), man));
        }

        // Кадры «а» GT (все отобранные), меры σ и A.
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        Map<String, double[]> meas = new HashMap<>();
        for (Map.Entry<String, Mat> e : fr.entrySet()) if (g.samples.containsKey(e.getKey())) meas.put(e.getKey(), measure(e.getValue()));
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {g.x, new XBaseEval()};
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

        // Номера снимков: для всех ключей (реальные – как XBaseEval.image), синтез и дубли – новые.
        Map<String, Integer> imageNo = new HashMap<>();
        for (Map.Entry<String, Sample> e : g.samples.entrySet()) imageNo.put(e.getKey(), XBaseEval.image(e.getValue()));
        g.imageNo = imageNo;

        List<List<String>> parts = XBaseOther.cvParts(g.own, g.seed, GT_PART_S);
        Map<String, Integer> partOf = new HashMap<>();
        for (int q = 0; q < parts.size(); q++) for (String p : parts.get(q)) partOf.put(p, q);
        List<Map<String, List<String>>> split = XBaseEval.within(g);
        Object[][] modes = {{"межс.", g.galleryKeys, g.probeKeys}, {"внутри", split.get(0), split.get(1)}};
        Random rnd = new Random(g.seed + GT_SYNTH_SEED_OFFSET);
        StringBuilder rangeT = new StringBuilder("\n=== Диапазоны этапа 2 по частям (σ и A – GT вне части; выравнивание – FERET, " + align.size()
                + " подобий) ===\nрежим | часть | людей части | снимков для σ и A | σ медиана | A медиана | A 75 %\n");
        Map<String, Map<String, List<String>>> synthOf = new LinkedHashMap<>();
        Map<String, Map<String, List<String>>> dupOf = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (Object[] md : modes) {
            String mode = (String) md[0];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
            Map<String, List<String>> syn = new LinkedHashMap<>();
            Map<String, List<String>> dup = new LinkedHashMap<>();
            for (int q = 0; q < parts.size(); q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                Set<String> probeDates = new HashSet<>();
                for (String p : parts.get(q)) for (String k : pro.get(p)) probeDates.add(dateGt.get(p + "/" + XBaseEval.image(g.samples.get(k))));
                List<Double> noise = new ArrayList<>();
                List<Double> asym = new ArrayList<>();
                for (Map.Entry<String, double[]> e : meas.entrySet()) {
                    Sample s = g.samples.get(e.getKey());
                    if (part.contains(s.person())) continue;
                    String d = dateGt.get(s.person() + "/" + XBaseEval.image(s));
                    if (d != null && probeDates.contains(d)) continue;
                    asym.add(e.getValue()[0]);
                    noise.add(e.getValue()[1]);
                }
                FaceSynth.Ranges rg = new FaceSynth.Ranges(sortedArr(noise), sortedArr(asym), align);
                rangeT.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %.2f | %.3f | %.3f%n", mode, q, parts.get(q).size(), noise.size(),
                        FaceSynth.median(rg.noise()), FaceSynth.median(rg.asym()), XBaseA0Check.q(rg.asym(), 0.75)));
                for (String p : parts.get(q)) {
                    List<String> real = gal.get(p);
                    List<String> ks = new ArrayList<>();
                    List<String> ds = new ArrayList<>();
                    for (int i = 0; i < GT_EXTRA; i++) {
                        String b = real.get(i % real.size());
                        Sample s = g.samples.get(b);
                        int no = XBaseEval.image(s);
                        double[] row = rowsGt.get(s.person() + "/" + no);
                        if (row == null) throw new IllegalStateException("gt-synth: нет строки YuNet для " + s.person() + " №" + no);
                        Mat color = org.opencv.imgcodecs.Imgcodecs.imread(gtDb.resolve(s.person()).resolve(String.format(Locale.ROOT, "%02d.jpg", no)).toString());
                        if (color.empty()) throw new IOException("gt-synth: не читается исходный снимок " + s.person() + " №" + no);
                        FaceSynth.Params prm = FaceSynth.sample(rg, rnd);
                        Mat a = FaceSynth.synth(color, row, prm, rnd);
                        color.release();
                        String key = b + "#t" + (mode.equals("межс.") ? "x" : "i") + i;
                        sf.put(key, a);
                        ks.add(key);
                        g.rOf.put(key, g.rOf.get(b));
                        imageNo.put(key, 1000 + 10 * no + i + (mode.equals("межс.") ? 0 : 500));
                        String dk = b + "#d" + (mode.equals("межс.") ? "x" : "i") + i;
                        ds.add(dk);
                        g.rOf.put(dk, g.rOf.get(b));
                        imageNo.put(dk, 3000 + 10 * no + i + (mode.equals("межс.") ? 0 : 500));
                        for (XBaseEval x : xs) {
                            x.vec.get(V.L1).put(dk, x.vec.get(V.L1).get(b));
                            x.mir.get(V.L1).put(dk, x.mir.get(V.L1).get(b));
                        }
                    }
                    syn.put(p, ks);
                    dup.put(p, ds);
                }
            }
            synthOf.put(mode, syn);
            dupOf.put(mode, dup);
        }
        XBaseEval h0 = new XBaseEval();
        h0.vectors(sf, EnumSet.of(V.L1), g.params);
        XBaseEval h1 = new XBaseEval();
        h1.vectors(sf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : sf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(h0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(h0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(h1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(h1.mir.get(V.L1));

        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-synth; seed %d; r0 %s; пул FERET %d; синтез %d; части %d", g.seed, g.r0,
                pool.size(), GT_EXTRA, GT_PART_S));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_synth"), fp);
        List<List<String>> cvParts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        List<List<String>> cohort0 = g.cohort;
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        StringBuilder conc = new StringBuilder("\n=== Сосредоточенность: контрольные пробы с лучшим кандидатом – одним из двух самых частых (модель по позе пробы) ===\n"
                + "строка | контрольных проб | два самых частых лучших (проб) | доля\n");
        StringBuilder lines = new StringBuilder();
        double fitS = 0;
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal0 = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Probe> probes = XBaseEval.probes(g, pro);
                for (Proj pj : PROJS) {
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    if (pj.pool()) aug.addAll(poolCls);
                    for (Kind kind : Kind.values()) {
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        for (String p : g.own) {
                            List<String> l = new ArrayList<>(gal0.get(p));
                            if (kind == Kind.B) l.addAll(dupOf.get(mode).get(p));
                            if (kind == Kind.C) l.addAll(synthOf.get(mode).get(p));
                            gal.put(p, l);
                        }
                        g.cohort = aug;
                        String name = String.format(Locale.ROOT, "%s_%s_%s", mode.equals("межс.") ? "x" : "i", pj.label(), kind.name());
                        String ls = cp.read(name);
                        if (ls == null) {
                            long t0 = System.nanoTime();
                            ls = XBaseEval.fitLines(g, xs[pj.feat()], EnumSet.of(V.L1), gal, probes, cvParts, 0);
                            double s = (System.nanoTime() - t0) / 1e9;
                            fitS += s;
                            cp.save(name, ls);
                            System.out.printf(Locale.ROOT, "gt-synth: %s, %s, %s ready (%.0f s)%n", mode.equals("межс.") ? "cross" : "in", pj.label(), kind.name(), s);
                        }
                        lines.append("# режим ").append(mode).append(", проекция ").append(pj.label()).append(", строка ").append(kind.name()).append('\n')
                                .append(ls);
                        XBaseEval.Parsed p = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, cvParts, ls);
                        String label = mode + " " + pj.label() + " " + kind.label;
                        res.put(label, FeiThreshold.evaluate(label, V.L1.ratioId(), List.of(p.cfg.get(V.L1)), List.of(p.imp.get(V.L1)), g.persons.size(), g.b,
                                g.seed + FeiBootstrap.SEED_OFFSET));
                        if (kind != Kind.B) {
                            Map<String, Row> rows = new HashMap<>();
                            for (String l : ls.lines().toList()) {
                                Row row = FeiScores.parse(l);
                                rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
                            }
                            Map<String, Integer> best = new TreeMap<>();
                            int n = 0;
                            for (Probe q : probes) {
                                if (q.role().equals(FeiScores.OWN)) continue;
                                Row row = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), g.rOf.get(q.key()) <= g.r0);
                                best.merge(row.bestId(), 1, Integer::sum);
                                n++;
                            }
                            List<Map.Entry<String, Integer>> top = new ArrayList<>(best.entrySet());
                            top.sort((u, w) -> w.getValue() - u.getValue());
                            int t2 = top.stream().limit(2).mapToInt(Map.Entry::getValue).sum();
                            conc.append(String.format(Locale.ROOT, "%s | %d | %s | %.1f %%%n", label, n, top.stream().limit(2).map(e -> e.getKey() + " (" + e.getValue()
                                    + ")").toList(), 100.0 * t2 / n));
                        }
                    }
                    String b = mode + " " + pj.label() + " ";
                    pairs.add(new String[] {b + Kind.C.label, b + Kind.A.label});
                    pairs.add(new String[] {b + Kind.C.label, b + Kind.B.label});
                    pairs.add(new String[] {b + Kind.B.label, b + Kind.A.label});
                }
            }
        } finally {
            g.cohort = cohort0;
        }
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, (а3) этап 2 на GT: синтез галереи, все свои – «новички» (XBaseEval gt-synth)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, detections.tsv, gt_sessions.tsv; "
                + "когорта – a/fei); исходные снимки – ").append(gtDb).append("; пул и подобия выравнивания – Color FERET (только обучение и диапазоны; имён "
                + "снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол gt прежний (межсессионно и внутри сессии, когорта FEI 20, порог П2/П3 по «чужим» галереи, 5 частей seed faces.seed + 8100).\n"
                + "Строки: (а) галерея как в gt; (б) + " + GT_EXTRA + " дубля (ключи #d, тот же вектор); (в) + " + GT_EXTRA + " синтеза (FaceSynth из исходных\n"
                + "снимков gt_db, основа – снимки галереи по очереди; синтез – и в шаблоне, и в обучении MLDA). Проекции: L1-xs-F – пул FERET (173\n"
                + "человека новыми классами), L1-tt-xs-F, L1 (без пула).\n");
        t.append(String.format(Locale.ROOT, "Диапазоны этапа 2: свои – 5 частей (XBaseOther.cvParts, seed faces.seed + %d); синтез человека части q – σ шума и A по "
                + "кадрам «а» GT людей вне части q (свои других частей и контрольные односессионные) без снимков дат проб части q; выравнивание – %d "
                + "подобий FERET (ручная разметка есть только там); синтез – java.util.Random(seed %d + %d).%n", 8000 + GT_PART_S, align.size(), g.seed,
                GT_SYNTH_SEED_OFFSET));
        t.append("σ и A по контрольным односессионным людям, которые же служат контрольными чужими: только распределения σ и A, не изображения; утечка в FAR "
                + "ничтожна.\n");
        t.append("На GT людей с ≥ 3 датами – 3 (gt_sessions.tsv, до отбора по позе); смешанная галерея на GT не делалась.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d; «*» — интервал разности не содержит 0.%n", g.b,
                g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки), потоков %d.%n", fitS, g.threads));
        t.append(rangeT).append(conc);
        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(g.persons.size());
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
        XBaseOther.tables(t, res, sep, g.b, pairs);
        XBaseOther.writeScores(out.resolve("xbase_gt_synth_scores.tsv.gz"), "# GT (а3) этап 2, XBaseSynth gt-synth; блоки — режим, проекция, строка; роли own, "
                + "ctrl, cv<q>\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n",
                lines.toString());
        FarMethods.write(out.resolve("xbase_gt_synth.txt"), t);
        System.out.print(t);
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
