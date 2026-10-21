package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * П. 5 дуэли (справочно; урезанный – решение Хозяина 20.10.2026, Д9, Д10): зависимость от числа людей галереи N. Метод Прокаева (гости 10, l = 3)
 * при k = 6 и k = 20 на всей сетке N; L1-xs-F – только при N = 10 на FERET межс. и GT межс. (только в прогоне с кадром К-лицо).
 * <p>
 * FERET межс. (свои половины №2, два прохода): N = 5, 10, 20, 40, все; GT межс.: N = 5, 10, 20, 30 (все); FEI (протокол п. 3а, 5 разбиений ×
 * 8 конфигураций): N = 5, 10, 20, 40, 100 (все). На каждое N меньше всех – 5 выборок людей из своих (перестановка отсортированных своих
 * Random(faces.seed + 9979 + 10·N + s), первые N; на FERET – из своих прохода, на FEI – из галереи разбиения). Гости и контроль – те же, что в
 * основной строке (ProkaevEval / ProkaevOther). Меры: верный первый кандидат (без порога), ошибка своих при фактическом FAR контроля 1 %
 * (порог по контрольным чужим каждой конфигурации – XBaseOther.errAtFar), правило 0,95 (своих принято верно / за другого / отвергнуто, FAR
 * контроля). По выборкам – среднее и [наименьшее; наибольшее].
 *
 * @author ssv
 */
final class ProkaevN {

    static final long N_SEED = 9979;
    static final int SAMPLES = 5;
    static final int L1_N = 10;
    /** Строки cfgs(): основная (k = 6) и k = 20. */
    static final int[] CFG = {0, 13};
    static final String[] ROWS = {"метод Прокаева k = 6", "метод Прокаева k = 20"};
    static final String L1_ROW = "L1-xs-F";

    private ProkaevN() {
    }

    /** Выборка N своих (s – номер выборки): перестановка отсортированных Random(seed + 9979 + 10·N + s), первые N, по порядку own. */
    static List<String> sample(List<String> own, int n, int s, long seed) {
        if (n >= own.size()) return own;
        List<String> p = new ArrayList<>(new TreeSet<>(own));
        Collections.shuffle(p, new Random(seed + N_SEED + 10L * n + s));
        Set<String> pick = new HashSet<>(p.subList(0, n));
        return own.stream().filter(pick::contains).toList();
    }

    /** Сетка N: заданные значения меньше всех и «все» (одна выборка). */
    static List<Integer> grid(int[] ns, int all) {
        List<Integer> g = new ArrayList<>();
        for (int n : ns) if (n < all) g.add(n);
        g.add(all);
        return g;
    }

    /** Итог (N, выборки, строки): конфигурации разбора и правило 0,95 {своих, отвергнуто, не тот, контрольных, принято контрольных}. */
    static final class Cell {
        final List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        final long[] rule = new long[5];
    }

    /** Накопитель: ключ «N|s|строка» → итог. */
    static final class Acc {
        final Map<String, Cell> cells = new LinkedHashMap<>();
        final Map<String, Integer> pIdx;

        Acc(Map<String, Integer> pIdx) {
            this.pIdx = pIdx;
        }

        Cell cell(int n, int s, String row) {
            return cells.computeIfAbsent(n + "|" + s + "|" + row, x -> new Cell());
        }
    }

    /** Конфигурация разбора по строкам оценок (оценка – отношение первых двух; у строк метода Прокаева – после XBaseAbl.abs). */
    static FeiDecompose.Cfg cfg(String lines, Map<String, Integer> pIdx) {
        List<double[]> own = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            FeiScores.Row r = FeiScores.parse(l);
            double x = FeiScores.ratio(r, null);
            int p = pIdx.get(r.person());
            if (r.own()) own.add(new double[] {p, x, r.bestId().equals(r.trueId()) ? 1 : 0});
            else if (r.role().equals(FeiScores.CTRL)) ctrl.add(new double[] {p, x});
        }
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        c.ownP = own.stream().mapToInt(a -> (int) a[0]).toArray();
        c.ownR = own.stream().mapToDouble(a -> a[1]).toArray();
        c.ownOk = new boolean[own.size()];
        for (int i = 0; i < own.size(); i++) c.ownOk[i] = own.get(i)[2] == 1;
        c.thrP = new int[0];
        c.thrR = new double[0];
        c.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
        c.ctrlR = ctrl.stream().mapToDouble(a -> a[1]).toArray();
        return c;
    }

    /** Счёт заданий метода Прокаева (n, s – метка) с контрольной точкой; разбор в acc. */
    static void prokaev(List<ProkaevEval.Job> jobs, int n, int s, ProkaevEval.Vecs v, XBaseOther.Checkpoint cp, Acc acc, String mode) throws IOException {
        ProkaevEval.Run run = new ProkaevEval.Run();
        Map<String, String> rowOf = new HashMap<>();
        for (int i = 0; i < CFG.length; i++) rowOf.put(ProkaevEval.cfgs().get(CFG[i]).row(), ROWS[i]);
        run.parse = o -> {
            ProkaevEval.Job j = (ProkaevEval.Job) o[0];
            String label = (String) o[1];
            String row = rowOf.get(label.substring(j.conf.length() + 1));
            acc.cell(n, s, row).cfgs.add(cfg((String) o[2], acc.pIdx));
        };
        for (ProkaevEval.Job j : jobs) {
            String name = String.format(Locale.ROOT, "p_%03d_%d_%02d", n, s, j.pass);
            String text = cp.read(name);
            if (text == null) {
                text = ProkaevEval.compute(j, v);
                cp.save(name, text);
            }
            ProkaevEval.absorb(j, text, run);
        }
        for (Map.Entry<String, long[]> e : run.rule.entrySet()) {
            String row = null;
            for (Map.Entry<String, String> r : rowOf.entrySet()) if (e.getKey().endsWith(" " + r.getKey())) row = r.getValue();
            long[] t = e.getValue();
            long[] c = acc.cell(n, s, row).rule;
            c[0] += t[0];
            c[1] += t[1];
            c[2] += t[2];
            c[3] += t[3];
            c[4] += t[4];
        }
        System.out.printf(Locale.ROOT, "%s: N = %d, выборка %d ready%n", mode, n, s);
    }

    /** Задание метода Прокаева по готовому заданию основной строки: свои – выборка sub (галерея и пробы своих – их же). */
    static ProkaevEval.Job subJob(ProkaevEval.Job b, List<String> sub, XBaseEval.Gt g, Map<String, List<String>> pro) {
        ProkaevEval.Job j = new ProkaevEval.Job();
        j.conf = b.conf;
        j.pass = b.pass;
        j.proto = b.proto;
        j.s = b.s;
        j.cfgJ = b.cfgJ;
        j.g = g;
        j.gal = new LinkedHashMap<>();
        Map<String, List<String>> pr = new LinkedHashMap<>();
        for (String p : sub) {
            j.gal.put(p, b.gal.get(p));
            pr.put(p, pro.get(p));
        }
        j.probes = pro == null ? null : XBaseEval.probes(g, pr);
        j.parts = null;
        // Только 10 гостей основной строки (матрица Грама собирается из гостей всех наборов; наборы 20 и 30 здесь не считаются).
        j.g30 = new ArrayList<>(b.guests(ProkaevEval.Gs.G10));
        j.only = Set.of(CFG[0], CFG[1]);
        return j;
    }

    // ---------------------------------------------------------------- отчёт

    static String stat(double[] x, String fmt) {
        double m = 0;
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            m += v;
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        m /= x.length;
        return x.length == 1 ? String.format(Locale.ROOT, fmt, m) : String.format(Locale.ROOT, fmt + " [" + fmt + "; " + fmt + "]", m, lo, hi);
    }

    /** Таблица «N × метод» и строка «до какого N верный первый ≥ 90 %». */
    static void table(StringBuilder r, String title, Acc acc, List<Integer> grid, int persons) {
        int[] unit = FeiBootstrap.unit(persons);
        r.append("\n=== ").append(title).append(" ===\n");
        r.append("N | строка | выборок | верный первый кандидат, % | ошибка своих при FAR контроля 1 %, % (фактический FAR, %) | правило 0,95: принято верно / "
                + "за другого / отвергнуто, % | FAR контроля по правилу, %\n");
        Map<String, Integer> hold = new LinkedHashMap<>();
        for (String row : new String[] {ROWS[0], ROWS[1], L1_ROW}) {
            for (int n : grid) {
                List<Cell> cs = new ArrayList<>();
                for (int s = 0; s < SAMPLES; s++) {
                    Cell c = acc.cells.get(n + "|" + s + "|" + row);
                    if (c != null) cs.add(c);
                }
                if (cs.isEmpty()) continue;
                double[] first = new double[cs.size()];
                double[] err = new double[cs.size()];
                double[] far = new double[cs.size()];
                double[] ok = new double[cs.size()];
                double[] wrong = new double[cs.size()];
                double[] rej = new double[cs.size()];
                double[] rfar = new double[cs.size()];
                boolean rule = cs.get(0).rule[0] > 0;
                for (int i = 0; i < cs.size(); i++) {
                    Cell c = cs.get(i);
                    long good = 0;
                    long all = 0;
                    for (FeiDecompose.Cfg f : c.cfgs) {
                        all += f.ownOk.length;
                        for (boolean b : f.ownOk) if (b) good++;
                    }
                    first[i] = 100.0 * good / all;
                    double[] e = XBaseOther.errAtFar(c.cfgs, unit, 0.01);
                    err[i] = e[0];
                    far[i] = e[1];
                    long[] t = c.rule;
                    if (rule) {
                        rej[i] = 100.0 * t[1] / t[0];
                        wrong[i] = 100.0 * t[2] / t[0];
                        ok[i] = 100.0 * (t[0] - t[1] - t[2]) / t[0];
                        rfar[i] = 100.0 * t[4] / t[3];
                    }
                }
                String ruleCells = rule ? stat(ok, "%.1f") + " / " + stat(wrong, "%.1f") + " / " + stat(rej, "%.1f") + " | " + stat(rfar, "%.2f") : "– | –";
                r.append(String.format(Locale.ROOT, "%d | %s | %d | %s | %s (%s) | %s%n", n, row, cs.size(), stat(first, "%.1f"), stat(err, "%.1f"), stat(far, "%.2f"),
                        ruleCells));
                double mean = 0;
                for (double x : first) mean += x / first.length;
                if (mean >= 90 && !row.equals(L1_ROW)) hold.merge(row, n, Math::max);
            }
        }
        for (String row : ROWS) {
            Integer n = hold.get(row);
            r.append(String.format(Locale.ROOT, "%s: верный первый кандидат ≥ 90 %% (среднее по выборкам) – %s.%n", row,
                    n == null ? "ни при одном N сетки" : "до N = " + n + " включительно (наибольшее такое N сетки)"));
        }
    }

    static String head(String what, int[] ns) {
        return String.format(Locale.ROOT, "Д-№1, п. 5 (справочно; урезанный, решение Хозяина 20.10.2026): зависимость от числа людей галереи, %s; N = %s; выборок %d на N меньше "
                + "всех (seed faces.seed + %d + 10·N + s); метод Прокаева – гости 10, l = 3, k = 6 и k = 20; L1-xs-F – только N = %d, межс., кадр «а» (в прогоне К-лицо)%n",
                what, java.util.Arrays.toString(ns), SAMPLES, N_SEED, L1_N);
    }

    // ---------------------------------------------------------------- FERET межс.

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles ro = XBaseFeret.equalize(su.ro, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
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
        Map<String, List<String>> keysOf = new HashMap<>();
        pool.forEach((p, ds) -> {
            List<String> ks = new ArrayList<>();
            ds.forEach(ks::addAll);
            keysOf.put(p, ks);
        });
        XBaseXsF.Conf c = XBaseMp2.confs(ro).get(0);
        if (!c.name().equals("межс.")) throw new IllegalStateException("prokaev-n: ожидалась постановка межс.");
        List<ProkaevEval.Job> full = new ArrayList<>();
        Set<String> used = new TreeSet<>();
        for (int ps = 0; ps < 2; ps++) {
            ProkaevEval.Job j = new ProkaevEval.Job();
            j.conf = c.name();
            j.pass = ps;
            j.proto = 1;
            List<String> own = c.own().stream().filter(halves.get(ps)::contains).toList();
            j.gal = new LinkedHashMap<>();
            for (String p : own) j.gal.put(p, c.gal().get(p));
            j.g = XBaseFeret.withOwn(base, own);
            j.s = c.s();
            ProkaevEval.pickGuests(j, new ArrayList<>(halves.get(1 - ps)), keysOf, tpl.seed);
            full.add(j);
            for (String p : own) {
                used.addAll(c.gal().get(p));
                used.addAll(c.pro().get(p));
            }
            used.addAll(j.guests(ProkaevEval.Gs.G10));
        }
        ro.ctrlKeys.values().forEach(used::addAll);
        int all = Math.min(full.get(0).g.own.size(), full.get(1).g.own.size());
        int[] ns = {5, 10, 20, 40};
        List<Integer> grid = grid(ns, Integer.MAX_VALUE);
        ProkaevEval.Vecs v = new ProkaevEval.Vecs();
        int k0 = 0;
        for (String k : used) {
            Mat m = FeretData.decode(su.byKey.get(k).file());
            ProkaevEval.addBig(v, k, m, su.det.get(k));
            m.release();
            if (++k0 % 500 == 0) System.out.printf(Locale.ROOT, "feret-prokaev-n: кадры – %d / %d%n", k0, used.size());
        }
        Map<String, Integer> pIdx = base.pIdx;
        Acc acc = new Acc(pIdx);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-prokaev-n %s; seed %d; кадр %s", dir, tpl.seed, ProkaevFrames.sheetTag()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_prokaev_n" + ProkaevFrames.tag()), fp);
        for (int n : grid) {
            for (int s = 0; s < (n == Integer.MAX_VALUE ? 1 : SAMPLES); s++) {
                List<ProkaevEval.Job> jobs = new ArrayList<>();
                for (ProkaevEval.Job b : full) {
                    List<String> sub = n == Integer.MAX_VALUE ? b.g.own : sample(b.g.own, n, s, tpl.seed);
                    jobs.add(subJob(b, sub, XBaseFeret.withOwn(base, sub), c.pro()));
                }
                prokaev(jobs, n, s, v, cp, acc, "feret-prokaev-n");
            }
        }
        // L1-xs-F при N = 10 (только К-лицо): W на выборке своих, пуле другой половины и когорте FEI (схема (Wа)), как база (4а).
        if (!ProkaevFrames.PLAIN) {
            Map<String, Mat> fr = new LinkedHashMap<>();
            Set<String> keys = new TreeSet<>(used);
            pool.values().forEach(ds -> ds.forEach(keys::addAll));
            for (String k : keys) fr.put(k, FeretData.frame(dir, "a", k));
            List<Sample> fei = new ArrayList<>();
            export.data().fei().values().forEach(fei::addAll);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            XBaseEval x = XBaseAbl.vectors(XBaseAbl.Sch.BASE, 0, fr, tpl.params);
            fr.values().forEach(Mat::release);
            for (int s = 0; s < SAMPLES; s++) {
                for (int ps = 0; ps < 2; ps++) {
                    List<String> sub = sample(full.get(ps).g.own, L1_N, s, tpl.seed);
                    Map<String, List<List<String>>> pp = new TreeMap<>();
                    for (String p : halves.get(1 - ps)) pp.put(p, pool.get(p));
                    List<List<String>> train = new ArrayList<>(cohort);
                    train.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp, Set.of(), Map.of()).extra());
                    XBaseEval.Gt g = XBaseFeret.withOwn(base, sub);
                    Map<String, List<String>> gal = new LinkedHashMap<>();
                    Map<String, List<String>> pro = new LinkedHashMap<>();
                    for (String p : sub) {
                        gal.put(p, c.gal().get(p));
                        pro.put(p, c.pro().get(p));
                    }
                    List<Probe> probes = XBaseEval.probes(g, pro);
                    List<List<String>> parts = XBaseOther.cvParts(sub, tpl.seed, XBaseEval.CV_S + c.s());
                    String name = String.format(Locale.ROOT, "l1_%d_%d", s, ps);
                    String ls = cp.read(name);
                    XBaseEval.Gt gt = XBaseAbl.copy(g, train, g.rOf, g.r0);
                    if (ls == null) {
                        ls = XBaseEval.fitLines(gt, x, EnumSet.of(V.L1), gal, probes, parts, c.s());
                        cp.save(name, ls);
                        System.out.printf(Locale.ROOT, "feret-prokaev-n: L1-xs-F, выборка %d, проход %d ready%n", s, ps + 1);
                    }
                    acc.cell(L1_N, s, L1_ROW).cfgs.add(XBaseEval.parse(gt, EnumSet.of(V.L1), gal, probes, parts, ls).cfg.get(V.L1));
                }
            }
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 п. 5: метод Прокаева – зависимость от N, FERET eq межс. (XBaseEval feret-prokaev-n): Color FERET – только "
                + "локально; справочно\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb, YuNet – derived/detections.tsv; кадр «а» – derived/a); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(head("FERET межс. (свои половины №2, два прохода; «все» – все свои прохода: " + full.get(0).g.own.size() + " / " + full.get(1).g.own.size() + ")", ns));
        r.append(String.format(Locale.ROOT, "Кадр метода Прокаева: %s. Гости – 10 первых гостей основной строки (люди другой половины №2); контроль – %d человек.%n",
                ProkaevFrames.describe(v.clipped), ro.ctrl.size()));
        if (ProkaevFrames.PLAIN) r.append("L1-xs-F – в отчёте К-лицо (xbase_feret_prokaev_n_face0.700.txt).\n");
        table(r, "FERET межс.: N × строка (оба прохода вместе; N = " + Integer.MAX_VALUE + " – все)", acc, grid, persons.size());
        finish(r, t0, out.resolve("xbase_feret_prokaev_n" + ProkaevFrames.tag() + ".txt"), cp);
    }

    static void finish(StringBuilder r, long t0, Path file, XBaseOther.Checkpoint cp) throws IOException {
        String t = r.toString().replace("N = " + Integer.MAX_VALUE + " – все", "«все» – последняя строка каждой группы").replace(Integer.MAX_VALUE + " | ", "все | ");
        StringBuilder w = new StringBuilder(t);
        w.append(String.format(Locale.ROOT, "%nВремя: %.0f с; max куча JVM %d МБ.%n", (System.nanoTime() - t0) / 1e9, Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(file, w);
        System.out.print(w);
        cp.delete();
    }

    // ---------------------------------------------------------------- GT межс.

    /** GT: межсессионно (within = false; gt-prokaev-n) или внутри сессии (within = true; gt-prokaev-n-in, Д-№3 – без L1-xs-F, отдельный отчёт «_in»). */
    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out,
            boolean within) throws IOException {
        String mode = within ? "gt-prokaev-n-in" : "gt-prokaev-n";
        String in = within ? "_in" : "";
        String conf = within ? "внутри" : "межс.";
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("gt-prokaev-n: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rOld = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        g0.rLimit = rLimit;
        Map<String, List<String>> galX = XBaseEval.equalize(g0.own, g0.galleryAll, g0.seed + XBaseEval.EQ_SEED_OFFSET);
        // Внутри сессии – как gt-prokaev (протокол 5): снимки даты галереи пополам, галерея по правилу №8 из первой половины.
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g0.own, g0.galleryAll, g0.seed);
        Map<String, List<String>> galP = within ? XBaseEval.equalize(g0.own, split.get(0), g0.seed + XBaseEval.EQ_SEED_OFFSET) : galX;
        Map<String, List<String>> proAll = within ? split.get(1) : g0.probeKeys;
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Map<String, List<String>> keysOf = new HashMap<>();
        pool.forEach((p, ds) -> {
            List<String> ks = new ArrayList<>();
            ds.forEach(ks::addAll);
            keysOf.put(p, ks);
        });
        Map<String, Sample> sampleOf = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) for (Sample s : l) sampleOf.put(s.file().toString(), s);
        ProkaevEval.Job b = new ProkaevEval.Job();
        b.conf = conf;
        b.pass = 0;
        b.proto = within ? 5 : 4;
        b.g = g0;
        b.gal = galP;
        b.s = 0;
        ProkaevEval.pickGuests(b, new ArrayList<>(pool.keySet()), keysOf, g0.seed);
        int[] ns = {5, 10, 20};
        List<Integer> grid = grid(ns, g0.own.size());
        // Кадры: GT – исходные снимки gt_db и YuNet экспорта; гости FERET – исходные снимки.
        ProkaevEval.Vecs v = new ProkaevEval.Vecs();
        Set<String> gtKeys = new TreeSet<>(g0.samples.keySet());
        for (String k : gtKeys) {
            Sample s = sampleOf.get(k);
            String pk = s.person() + "/" + s.file().getFileName().toString();
            Mat m = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            ProkaevEval.addBig(v, k, m, det.get(pk));
            m.release();
        }
        for (String k : b.guests(ProkaevEval.Gs.G10)) {
            Mat m = FeretData.decode(su.byKey.get(k).file());
            ProkaevEval.addBig(v, k, m, su.det.get(k));
            m.release();
        }
        Acc acc = new Acc(g0.pIdx);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "%s; seed %d; r0 %s; R %s; кадр %s", mode, g0.seed, r0, rLimit, ProkaevFrames.sheetTag()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_prokaev_n" + in + ProkaevFrames.tag()), fp);
        for (int n : grid) {
            for (int s = 0; s < (n == g0.own.size() ? 1 : SAMPLES); s++) {
                List<String> sub = sample(g0.own, n, s, g0.seed);
                prokaev(List.of(subJob(b, sub, withOwn(g0, sub), proAll)), n, s, v, cp, acc, mode);
            }
        }
        if (!ProkaevFrames.PLAIN && !within) {
            List<List<String>> train = new ArrayList<>(g0.cohort);
            train.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra());
            Set<String> cohortKeys = new HashSet<>();
            g0.cohort.forEach(cohortKeys::addAll);
            Map<String, Mat> fr = new LinkedHashMap<>(XBaseEval.frames(export, gtKeys.stream().map(sampleOf::get).toList(), null));
            List<Sample> fei = new ArrayList<>();
            export.data().fei().values().forEach(fei::addAll);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
            XBaseEval x = XBaseAbl.vectors(XBaseAbl.Sch.BASE, 0, fr, g0.params);
            fr.values().forEach(Mat::release);
            for (int s = 0; s < SAMPLES; s++) {
                List<String> sub = sample(g0.own, L1_N, s, g0.seed);
                XBaseEval.Gt g = XBaseAbl.copy(withOwn(g0, sub), train, g0.rOf, g0.r0);
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : sub) {
                    gal.put(p, galX.get(p));
                    pro.put(p, g0.probeKeys.get(p));
                }
                List<Probe> probes = XBaseEval.probes(g, pro);
                List<List<String>> parts = XBaseOther.cvParts(sub, g0.seed, XBaseEval.CV_S);
                String name = String.format(Locale.ROOT, "l1_%d", s);
                String ls = cp.read(name);
                if (ls == null) {
                    ls = XBaseEval.fitLines(g, x, EnumSet.of(V.L1), gal, probes, parts, 0);
                    cp.save(name, ls);
                    System.out.printf(Locale.ROOT, "gt-prokaev-n: L1-xs-F, выборка %d ready%n", s);
                }
                acc.cell(L1_N, s, L1_ROW).cfgs.add(XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, parts, ls).cfg.get(V.L1));
            }
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 п. 5" + (within ? " и Д-№3" : "") + ": метод Прокаева – зависимость от N, GT " + conf + " (XBaseEval " + mode + "); справочно\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, YuNet – detections.tsv); снимки GT – ")
                .append(gtDir).append("; гости и пул L1-xs-F – Color FERET (имён снимков FERET нет)\n");
        r.append(head("GT " + conf + " (галерея – правило №8, N = " + (within ? 3 : 6) + " снимков" + (within ? ", снимки даты галереи пополам" : "") + "; «все» – " + g0.own.size() + ")", ns));
        r.append(String.format(Locale.ROOT, "Кадр метода Прокаева: %s. Гости – 10 первых гостей основной строки (люди FERET); контроль – %d человек GT.%n",
                ProkaevFrames.describe(v.clipped), g0.ctrl.size()));
        if (within) r.append("L1-xs-F внутри сессии не считается (п. 5 – только межс.).\n");
        else if (ProkaevFrames.PLAIN) r.append("L1-xs-F – в отчёте К-лицо (xbase_gt_prokaev_n_face0.700.txt).\n");
        table(r, "GT " + conf + ": N × строка", acc, grid, g0.persons.size());
        finish(r, t0, out.resolve("xbase_gt_prokaev_n" + in + ProkaevFrames.tag() + ".txt"), cp);
    }

    /** Копия Gt GT со своими sub (порядок own – индексы проб). */
    static XBaseEval.Gt withOwn(XBaseEval.Gt g, List<String> sub) {
        XBaseEval.Gt c = XBaseAbl.copy(g, g.cohort, g.rOf, g.r0);
        c.own = sub;
        return c;
    }

    // ---------------------------------------------------------------- FEI

    static void fei(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, SettingsStore settings, Path out) throws IOException {
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new TreeSet<>(m.keySet())));
        List<FeiProtocol.Split> splits = FeiProtocol.splits(included, seed);
        List<String> persons = new ArrayList<>(new TreeSet<>(feiKeys.keySet()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        Map<String, Integer> numOf = new HashMap<>();
        feiKeys.forEach((p, m) -> m.forEach((num, k) -> numOf.put(k, num)));
        Map<String, List<String>> cohortKeysOf = new HashMap<>();
        List<ProkaevEval.Job> full = new ArrayList<>();
        List<Map<String, List<String>>> proOf = new ArrayList<>();
        Set<String> keys = new TreeSet<>();
        for (FeiProtocol.Split sp : splits) {
            for (String p : sp.cohort()) cohortKeysOf.put(p, new ArrayList<>(feiKeys.get(p).values()));
            for (int jn = 0; jn < FeiProtocol.NUMBERS.length; jn++) {
                ProkaevEval.Job j = new ProkaevEval.Job();
                j.conf = "FEI";
                j.pass = sp.index() * FeiProtocol.NUMBERS.length + jn;
                j.proto = 6;
                j.s = sp.index();
                j.cfgJ = jn;
                XBaseEval.Gt g = new XBaseEval.Gt();
                g.own = sp.gallery();
                g.ctrl = sp.control();
                g.imageNo = numOf;
                g.persons = persons;
                g.pIdx = pIdx;
                g.byPerson = new HashMap<>();
                for (String p : sp.control()) g.byPerson.put(p, Map.of(0, new ArrayList<>(new TreeMap<>(feiKeys.get(p)).values())));
                j.g = g;
                j.gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                int nj = FeiProtocol.NUMBERS[jn];
                for (String p : sp.gallery()) {
                    List<String> gk = new ArrayList<>();
                    for (int num : FeiProtocol.trainNumbers(jn)) gk.add(feiKeys.get(p).get(num));
                    j.gal.put(p, gk);
                    pro.put(p, List.of(feiKeys.get(p).get(nj)));
                    keys.addAll(gk);
                    keys.add(feiKeys.get(p).get(nj));
                }
                for (String p : sp.control()) keys.addAll(feiKeys.get(p).values());
                ProkaevEval.pickGuests(j, sp.cohort(), cohortKeysOf, seed + sp.index());
                keys.addAll(j.guests(ProkaevEval.Gs.G10));
                full.add(j);
                proOf.add(pro);
            }
        }
        ProkaevEval.Vecs v = new ProkaevEval.Vecs();
        ProkaevEval.feiFrames(export, dir, keys, settings, v);
        int[] ns = {5, 10, 20, 40};
        int all = full.get(0).g.own.size();
        List<Integer> grid = grid(ns, all);
        Acc acc = new Acc(pIdx);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "fei-prokaev-n; seed %d; конфигураций %d; кадр %s", seed, full.size(), ProkaevFrames.sheetTag()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_fei_prokaev_n" + ProkaevFrames.tag()), fp);
        for (int n : grid) {
            for (int s = 0; s < (n == all ? 1 : SAMPLES); s++) {
                List<ProkaevEval.Job> jobs = new ArrayList<>();
                for (int i = 0; i < full.size(); i++) {
                    ProkaevEval.Job b = full.get(i);
                    List<String> sub = sample(b.g.own, n, s, seed);
                    XBaseEval.Gt g = withOwnFei(b.g, sub);
                    jobs.add(subJob(b, sub, g, proOf.get(i)));
                }
                prokaev(jobs, n, s, v, cp, acc, "fei-prokaev-n");
            }
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 п. 5: метод Прокаева – зависимость от N, FEI (XBaseEval fei-prokaev-n); справочно\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (fei_selection.tsv, YuNet – detections.tsv); снимки FEI – ")
                .append(settings.loadFacesFeiDir()).append('\n');
        r.append(head("FEI (протокол п. 3а: 5 разбиений × 8 конфигураций, выборка – из галереи разбиения; «все» – " + all + "; пороговые не используются)", ns));
        r.append(String.format(Locale.ROOT, "Кадр метода Прокаева: %s. Гости – 10 первых гостей основной строки (люди когорты разбиения); контроль – 40 человек разбиения.%n",
                ProkaevFrames.describe(v.clipped)));
        table(r, "FEI: N × строка", acc, grid, persons.size());
        finish(r, t0, out.resolve("xbase_fei_prokaev_n" + ProkaevFrames.tag() + ".txt"), cp);
    }

    static XBaseEval.Gt withOwnFei(XBaseEval.Gt g, List<String> sub) {
        XBaseEval.Gt c = new XBaseEval.Gt();
        c.own = sub;
        c.ctrl = g.ctrl;
        c.imageNo = g.imageNo;
        c.persons = g.persons;
        c.pIdx = g.pIdx;
        c.byPerson = g.byPerson;
        return c;
    }
}
