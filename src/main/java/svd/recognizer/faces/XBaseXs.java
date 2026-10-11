package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.storage.SettingsStore;

/**
 * Межсессионная изменчивость, выученная на других людях (PLAN.md, п. 4.2; план подтверждён Хозяином 11.10.2026). 30 своих
 * GT делятся по людям на 5 частей (java.util.Random(faces.seed + 9200), подряд по 6). Для части q:
 * <ul>
 *   <li>L1 — MLDA на галерее всех 30 и когорте FEI 20 (как L1, от части не зависит);</li>
 *   <li>L1-xs — то же, но в класс каждого своего вне части q добавлены его снимки остальных дат;</li>
 *   <li>L1-xs0 — вместо них столько же (N_q) снимков людей FEI вне когорты — новыми классами (по классу на человека):
 *       перестановка людей java.util.Random(faces.seed + 9300), снимки по номеру, последний — частично; если последнему
 *       достаётся один снимок, у предыдущего убирается его последний, а последнему даётся два.</li>
 * </ul>
 * Шаблоны (средние в пространстве MLDA) — только по снимкам галереи (MldaScorer.fit(train, templates, …)). Признак — как
 * L1 (корень гистограмм LBP u2 (8, 2), 7×7, кадр «а»); зеркала — как L1 (модель X+mirror: к обучающим снимкам и шаблонам
 * — их отражения, способ а); выбор X / X+mirror по |r| пробы. Проверяются пробы людей части q (объединение — все пробы 30
 * своих). Порог П2/П3: «чужие» части q — снимки галереи её людей против модели без части q (классы остальных 24 со своими
 * добавками, те же классы FEI у L1-xs0, когорта), объединение по 5 частям — один порог. Контрольные чужие оцениваются всеми 5
 * моделями (5 × 295 попыток при общем пороге). Межсессионно — галерея = дата галереи; внутри сессии — половина даты (деление
 * gt-in), добавки те же. Отчёт xbase_gt_xs.txt, оценки xbase_gt_xs_scores.tsv.gz и xbase_gt_in_xs_scores.tsv.gz.
 *
 * @author ssv
 */
final class XBaseXs {

    static final long PART_SEED_OFFSET = 9200;
    static final long FEI_SEED_OFFSET = 9300;
    static final int PARTS = 5;

    enum Kind {
        L1(FeiMethods.LBP_MLDA, "L1"), XS(FeiMethods.LBP_MLDA + "-xs", "L1-xs"), XS0(FeiMethods.LBP_MLDA + "-xs0", "L1-xs0");

        final String id;
        final String label;

        Kind(String id, String label) {
            this.id = id;
            this.label = label;
        }
    }

    private XBaseXs() {
    }

    /** Части своих: перестановка java.util.Random(seed + PART_SEED_OFFSET), подряд по own.size() / PARTS. */
    static List<List<String>> parts(List<String> own, long seed) {
        if (own.size() % PARTS != 0) throw new IllegalArgumentException("Своих " + own.size() + " не делится на " + PARTS);
        List<String> perm = new ArrayList<>(own);
        Collections.shuffle(perm, new Random(seed + PART_SEED_OFFSET));
        int size = own.size() / PARTS;
        List<List<String>> out = new ArrayList<>();
        for (int q = 0; q < PARTS; q++) out.add(List.copyOf(perm.subList(q * size, (q + 1) * size)));
        return out;
    }

    /**
     * Классы FEI для L1-xs0: люди пула по порядку (у каждого — снимки по номеру), пока не наберётся n; последний — первые
     * нужные снимки; если ему достаётся один — у предыдущего убирается последний снимок, последнему даётся два.
     */
    static List<List<String>> feiClasses(List<List<String>> pool, int n) {
        if (n < 2) throw new IllegalArgumentException("L1-xs0: снимков FEI " + n + " (нужно ≥ 2)");
        List<List<String>> out = new ArrayList<>();
        int left = n;
        for (List<String> ks : pool) {
            if (left == 0) break;
            int take = Math.min(left, ks.size());
            if (take == 1) {
                List<String> prev = out.get(out.size() - 1);
                if (prev.size() < 3 || ks.size() < 2) throw new IllegalStateException("L1-xs0: нельзя перераспределить один снимок");
                out.set(out.size() - 1, new ArrayList<>(prev.subList(0, prev.size() - 1)));
                take = 2;
            }
            out.add(new ArrayList<>(ks.subList(0, take)));
            left -= Math.min(left, ks.size());
        }
        if (left > 0) throw new IllegalStateException("L1-xs0: в пуле FEI не хватает снимков (" + left + ")");
        return out;
    }

    /** Векторы L1 (корень гистограмм LBP) и их отражения: GT и когорта — из Gt, пул FEI — досчитываются. */
    final Map<String, double[]> vec = new HashMap<>();
    final Map<String, double[]> mir = new HashMap<>();

    List<double[]> vectors(List<String> keys, boolean mirror) {
        List<double[]> out = new ArrayList<>();
        for (String k : keys) out.add(vec.get(k));
        if (mirror) for (String k : keys) out.add(mir.get(k));
        return out;
    }

    /** Модель MLDA: обучение — классы train (галерея с добавками, классы FEI, когорта), шаблоны — templ (первые классы). */
    GalleryScorer.ScoreModel fit(boolean mirror, List<List<String>> train, List<List<String>> templ) {
        List<List<double[]>> tr = new ArrayList<>();
        for (List<String> ks : train) tr.add(vectors(ks, mirror));
        List<List<double[]>> te = new ArrayList<>();
        for (List<String> ks : templ) te.add(vectors(ks, mirror));
        return new MldaScorer().fit(tr, te, templ.size(), 1e-10);
    }

    /** Попытка: роль, человек, снимок, ключ, индекс истинного в галерее (−1 — чужой). */
    record Probe(String role, String person, int image, String key, int trueIdx) {}

    String lines(String method, int s, GalleryScorer.ScoreModel m, List<String> gallery, List<Probe> probes) {
        StringBuilder t = new StringBuilder();
        for (Probe p : probes) t.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, m.scores(vec.get(p.key())), false));
        return t.toString();
    }

    /** Итоги режима: итоги вариантов, строки оценок, число добавок по частям. */
    static final class Mode {
        final List<FeiThreshold.Res> results = new ArrayList<>();
        String lines;
        double fitS;
        int ctrlAttempts;
        int cvImages;
        int ownProbes;
    }

    Mode evaluate(XBaseEval.Gt g, List<List<String>> parts, Map<String, List<String>> gal, Map<String, List<String>> pro,
                  Map<Integer, List<List<String>>> feiByPart) {
        List<String> own = g.own;
        List<String> ctrlKeys = new ArrayList<>();
        for (String p : g.ctrl) for (List<String> ks : g.byPerson.get(p).values()) ctrlKeys.addAll(ks);
        ctrlKeys.sort(null);
        List<Callable<String>> tasks = new ArrayList<>();
        for (Kind kind : Kind.values()) {
            for (int q = 0; q < PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                int qq = q;
                // Классы своих для обучения: галерея, у L1-xs — плюс остальные даты людей вне части.
                List<List<String>> ownTrain = new ArrayList<>();
                List<List<String>> ownTempl = new ArrayList<>();
                List<List<String>> restTrain = new ArrayList<>();
                List<List<String>> restTempl = new ArrayList<>();
                List<String> rest = new ArrayList<>();
                for (String p : own) {
                    List<String> tr = new ArrayList<>(gal.get(p));
                    if (kind == Kind.XS && !part.contains(p)) tr.addAll(g.probeKeys.get(p));
                    ownTrain.add(tr);
                    ownTempl.add(gal.get(p));
                    if (!part.contains(p)) {
                        restTrain.add(tr);
                        restTempl.add(gal.get(p));
                        rest.add(p);
                    }
                }
                List<List<String>> extra = kind == Kind.XS0 ? feiByPart.get(q) : List.of();
                List<List<String>> mainTrain = new ArrayList<>(ownTrain);
                mainTrain.addAll(extra);
                mainTrain.addAll(g.cohort);
                List<List<String>> calTrain = new ArrayList<>(restTrain);
                calTrain.addAll(extra);
                calTrain.addAll(g.cohort);
                List<Probe> mainProbes = new ArrayList<>();
                for (String p : parts.get(q)) {
                    for (String k : pro.get(p)) mainProbes.add(new Probe(FeiScores.OWN, p, XBaseEval.image(g.samples.get(k)), k, own.indexOf(p)));
                }
                for (String k : ctrlKeys) mainProbes.add(new Probe(FeiScores.CTRL, g.samples.get(k).person(), XBaseEval.image(g.samples.get(k)), k, -1));
                List<Probe> calProbes = new ArrayList<>();
                for (String p : parts.get(q)) {
                    for (String k : gal.get(p)) calProbes.add(new Probe(FeiMethods.CV_ROLE + q, p, XBaseEval.image(g.samples.get(k)), k, -1));
                }
                for (boolean m : new boolean[] {false, true}) {
                    String method = kind.id + (m ? FeiMethods.MIRROR : "");
                    tasks.add(() -> lines(method, qq, fit(m, mainTrain, ownTempl), own, mainProbes));
                    tasks.add(() -> lines(method, qq, fit(m, calTrain, restTempl), rest, calProbes));
                }
            }
        }
        Mode md = new Mode();
        long t0 = System.nanoTime();
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, g.threads).forEach(all::append);
        md.fitS = (System.nanoTime() - t0) / 1e9;
        md.lines = all.toString();
        Map<String, Row> rows = new HashMap<>();
        for (String l : md.lines.lines().toList()) {
            Row r = FeiScores.parse(l);
            rows.put(r.method() + "\t" + r.s() + "\t" + r.role() + "\t" + r.person() + "\t" + r.image(), r);
        }
        for (Kind kind : Kind.values()) {
            FeiDecompose.Cfg c = new FeiDecompose.Cfg();
            List<Row> ownRows = new ArrayList<>();
            List<Boolean> front = new ArrayList<>();
            List<double[]> ctrl = new ArrayList<>();
            List<double[]> cv = new ArrayList<>();
            int cvFront = 0;
            for (int q = 0; q < PARTS; q++) {
                for (String p : parts.get(q)) {
                    for (String k : pro.get(p)) {
                        boolean f = g.rOf.get(k) <= g.r0;
                        ownRows.add(choose(rows, kind, q, FeiScores.OWN, p, g.samples.get(k), f));
                        front.add(f);
                    }
                    for (String k : gal.get(p)) {
                        boolean f = g.rOf.get(k) <= g.r0;
                        if (f) cvFront++;
                        Row r = choose(rows, kind, q, FeiMethods.CV_ROLE + q, p, g.samples.get(k), f);
                        cv.add(new double[] {g.pIdx.get(p), FeiScores.ratio(r, null)});
                    }
                }
                for (String k : ctrlKeys) {
                    Sample s = g.samples.get(k);
                    Row r = choose(rows, kind, q, FeiScores.CTRL, s.person(), s, g.rOf.get(k) <= g.r0);
                    ctrl.add(new double[] {g.pIdx.get(s.person()), FeiScores.ratio(r, null)});
                }
            }
            int n = ownRows.size();
            c.ownP = new int[n];
            c.ownR = new double[n];
            c.ownOk = new boolean[n];
            c.noSecond = new boolean[n];
            c.trueInf = new boolean[n];
            c.poseFrontal = new boolean[n];
            for (int i = 0; i < n; i++) {
                Row r = ownRows.get(i);
                int[] cs = new int[FeiScores.CASES];
                c.ownP[i] = g.pIdx.get(r.person());
                c.ownR[i] = FeiScores.ratio(r, cs);
                c.noSecond[i] = cs[FeiScores.CASE_NO_SECOND] > 0;
                c.trueInf[i] = Double.isInfinite(r.trueScore());
                c.ownOk[i] = r.bestId().equals(r.trueId());
                c.poseFrontal[i] = front.get(i);
            }
            c.thrP = new int[0];
            c.thrR = new double[0];
            c.ctrlP = ctrl.stream().mapToInt(d -> (int) d[0]).toArray();
            c.ctrlR = ctrl.stream().mapToDouble(d -> d[1]).toArray();
            md.ctrlAttempts = c.ctrlR.length;
            md.cvImages = cv.size();
            md.ownProbes = n;
            md.results.add(FeiThreshold.evaluate(kind.label, kind.id + "-ratio+pose", List.of(c), List.of(FeiThreshold.Imp.of(cv, cvFront)),
                    g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET));
        }
        return md;
    }

    /** Строка модели части q по позе: анфас — X, иначе X+mirror. */
    static Row choose(Map<String, Row> rows, Kind kind, int q, String role, String person, Sample s, boolean frontal) {
        String method = kind.id + (frontal ? "" : FeiMethods.MIRROR);
        Row r = rows.get(method + "\t" + q + "\t" + role + "\t" + person + "\t" + XBaseEval.image(s));
        if (r == null) throw new IllegalStateException("Нет строки " + method + " s=" + q + " " + role + " " + person + " №" + XBaseEval.image(s));
        return r;
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseXs x = new XBaseXs();
        x.vec.putAll(g.x.vec.get(XBaseEval.V.L1));
        x.mir.putAll(g.x.mir.get(XBaseEval.V.L1));
        List<List<String>> parts = parts(g.own, g.seed);

        // Пул FEI для L1-xs0: люди вне встроенной когорты, перестановка seed + FEI_SEED_OFFSET, снимки по номеру.
        Set<String> cohort = new HashSet<>(g.cohortP);
        List<String> poolP = new ArrayList<>();
        for (String p : new TreeMap<>(feiKeys).keySet()) if (!cohort.contains(p)) poolP.add(p);
        Collections.shuffle(poolP, new Random(g.seed + FEI_SEED_OFFSET));
        List<List<String>> pool = new ArrayList<>();
        for (String p : poolP) pool.add(new ArrayList<>(new TreeMap<>(feiKeys.get(p)).values()));
        Map<Integer, List<List<String>>> feiByPart = new LinkedHashMap<>();
        Map<Integer, Integer> nq = new LinkedHashMap<>();
        Set<String> needed = new HashSet<>();
        for (int q = 0; q < PARTS; q++) {
            Set<String> part = new HashSet<>(parts.get(q));
            int n = 0;
            for (String p : g.own) if (!part.contains(p)) n += g.probeKeys.get(p).size();
            nq.put(q, n);
            List<List<String>> cls = feiClasses(pool, n);
            feiByPart.put(q, cls);
            for (List<String> ks : cls) needed.addAll(ks);
        }
        // Векторы L1 нужных снимков пула (кадр «а», корень гистограмм LBP, отражения — способ а).
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, fei, needed);
        XBaseEval h = new XBaseEval();
        h.vectors(fr, java.util.EnumSet.of(XBaseEval.V.L1), g.params);
        fr.values().forEach(Mat::release);
        x.vec.putAll(h.vec.get(XBaseEval.V.L1));
        x.mir.putAll(h.mir.get(XBaseEval.V.L1));

        Mode cross = x.evaluate(g, parts, g.galleryKeys, g.probeKeys, feiByPart);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryKeys, g.seed);
        Mode in = x.evaluate(g, parts, split.get(0), split.get(1), feiByPart);
        for (Object[] f : new Object[][] {{"xbase_gt_xs_scores.tsv.gz", " — межсессионная изменчивость (L1-xs), межсессионно", cross},
            {"xbase_gt_in_xs_scores.tsv.gz", " — межсессионная изменчивость (L1-xs), внутри сессии", in}}) {
            XBaseEval.Eval ev = new XBaseEval.Eval();
            ev.lines = ((Mode) f[2]).lines;
            XBaseEval.writeScores(out.resolve((String) f[0]), (String) f[1], g, ev);
        }
        Map<String, FeiThreshold.Res> cm = new LinkedHashMap<>();
        Map<String, FeiThreshold.Res> wm = new LinkedHashMap<>();
        for (FeiThreshold.Res r : cross.results) cm.put(r.label, r);
        for (FeiThreshold.Res r : in.results) wm.put(r.label, r);

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: межсессионная изменчивость, выученная на других людях — GT (XBaseEval gt-xs)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта и пул L1-xs0 — a/fei)\n");
        t.append(String.format(Locale.ROOT, "Части своих: java.util.Random(seed %d + %d), %d частей по %d; пул FEI для L1-xs0 — люди вне когорты (%d), "
                + "перестановка java.util.Random(seed %d + %d).%n", g.seed, PART_SEED_OFFSET, PARTS, g.own.size() / PARTS, poolP.size(), g.seed,
                FEI_SEED_OFFSET));
        t.append("Варианты (признак — как L1: корень гистограмм LBP u2 (8, 2), 7×7, кадр «а», нормировка none; MLDA; d₁/d₂ по 30 своим;\n"
                + "зеркала способом а и выбор X / X+mirror по |r| пробы, r0 = ").append(g.r0).append("; шаблоны — только по снимкам галереи):\n"
                        + "  L1 — классы: галерея 30 своих + когорта FEI 20;\n"
                        + "  L1-xs — в класс каждого своего вне проверяемой части добавлены его снимки остальных дат;\n"
                        + "  L1-xs0 — вместо них столько же снимков людей FEI вне когорты, новыми классами (по человеку).\n"
                        + "Порог П2/П3 — по «чужим» галереи: снимки галереи людей части q против модели без части q (классы остальных 24 с\n"
                        + "добавками, те же классы FEI, когорта), объединение по 5 частям, один порог. Контрольные чужие — оцениваются всеми 5\n"
                        + "моделями частей при общем пороге. Внутри сессии — галерея и пробы — половины даты (как gt-in), добавки те же.\n");
        t.append("\n=== Части ===\nчасть | люди | L1-xs: добавлено снимков других дат (людей) | L1-xs0: снимков FEI (классов: людей и снимков на человека)\n");
        for (int q = 0; q < PARTS; q++) {
            Set<String> part = new HashSet<>(parts.get(q));
            int people = 0;
            for (String p : g.own) if (!part.contains(p) && !g.probeKeys.get(p).isEmpty()) people++;
            List<String> sizes = new ArrayList<>();
            for (List<String> ks : feiByPart.get(q)) sizes.add(Integer.toString(ks.size()));
            t.append(String.format(Locale.ROOT, "%d | %s | %d (%d) | %d (%d: %s)%n", q, String.join(", ", parts.get(q)), nq.get(q), people,
                    feiByPart.get(q).stream().mapToInt(List::size).sum(), feiByPart.get(q).size(), String.join(", ", sizes)));
        }
        t.append(String.format(Locale.ROOT, "Объём: межсессионно пробы своих %d, «чужие» П2/П3 %d снимков; внутри сессии пробы %d, «чужие» %d; контроль — "
                + "%d попыток (5 моделей × %d снимков %d людей).%n", cross.ownProbes, cross.cvImages, in.ownProbes, in.cvImages, cross.ctrlAttempts,
                cross.ctrlAttempts / PARTS, g.ctrl.size()));
        if (g.b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d; одни веса для всех вариантов и обоих "
                    + "режимов.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        t.append(String.format(Locale.ROOT, "Обучение и оценки: межсессионно %.0f с, внутри сессии %.0f с, потоков %d.%n", cross.fitS, in.fitS, g.threads));
        t.append("Оговорка: 30 своих — различимы только крупные эффекты; внутри сессии снимки почти дубли.\n");
        LbpNormReport.diag(t, cm, wm, List.of("L1-xs", "L1-xs0", "L1"), "L1", List.of("L1-xs", "L1-xs0"), g.b);
        if (g.b > 0) LbpNormReport.dropLevel(t, cm, wm, "L1-xs0", List.of("L1-xs"));
        FarMethods.write(out.resolve("xbase_gt_xs.txt"), t);
        System.out.print(t);
    }
}
