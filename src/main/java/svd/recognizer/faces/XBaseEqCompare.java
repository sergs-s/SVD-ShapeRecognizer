package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.XBaseA0.Att;
import svd.recognizer.faces.XBaseA0.Pair;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * №8 (13.10.2026): сравнение «прежняя галерея → одинаковое число снимков» по файлам оценок, без распознавания. Прежние
 * оценки — reports/faces/xbase/ и D:\data\Feret\reports\, новые — eq9800…eq9802/ и reports\eq\ (faces.gallery.equal).
 * Решения — XBaseA0.decide (модель по позе, порог П2 FAR 1 % по «чужим» галереи своей галереи). По каждой конфигурации и
 * методу: ошибка своих П2 и П3 при FAR 1 %, FAR контроля по снимкам и людям (порог П2), EER (порог — значения d₁/d₂ своих
 * и контрольных; наименьшая |ошибка своих − FAR контроля|, EER — их среднее), разделимость (ошибка своих при пороге по
 * контрольным чужим, FAR контроля 1 %); у L1 и L1-tt — парная разность ошибки своих П2 «новое − прежнее», бутстреп по
 * людям (B = faces.fei.bootstrap, seed faces.seed + 7000; порог заново в каждой реплике). FERET — ещё люди спорного
 * случая (А) (на дате галереи нет чистого fa или fb) и хаб (лучший кандидат у принятых контрольных чужих).
 * Режимы XBaseEval gt-eq-compare (reports/faces/xbase/xbase_gt_eq_compare.txt) и feret-eq-compare (только
 * D:\data\Feret\reports\xbase_feret_eq_compare.txt).
 *
 * @author ssv
 */
final class XBaseEqCompare {

    static final double FAR = 0.01;

    private XBaseEqCompare() {
    }

    /** Метрики пары: {ошибка своих П2, П3, FAR контроля по снимкам, по людям, EER, разделимость}, проценты. */
    static double[] metrics(Pair p, int persons) {
        double th3 = FeiThreshold.margin(p.cvR, p.cvP, FeiBootstrap.unit(persons), FAR).theta();
        List<double[]> own = new ArrayList<>();
        List<double[]> ctrl = new ArrayList<>();
        Map<String, Boolean> ctrlPerson = new HashMap<>();
        for (Att a : p.atts) {
            if (a.probe().role().equals(FeiScores.OWN)) {
                own.add(new double[] {a.ratio(), a.row().bestId().equals(a.row().trueId()) ? 1 : 0});
            } else {
                ctrl.add(new double[] {a.ratio()});
                ctrlPerson.merge(a.probe().person(), a.ratio() <= p.theta, Boolean::logicalOr);
            }
        }
        double e2 = ownErr(own, p.theta);
        double e3 = ownErr(own, th3);
        double far = 100.0 * ctrl.stream().filter(c -> c[0] <= p.theta).count() / ctrl.size();
        double farP = 100.0 * ctrlPerson.values().stream().filter(b -> b).count() / ctrlPerson.size();
        // EER.
        TreeSet<Double> cand = new TreeSet<>();
        for (double[] o : own) cand.add(o[0]);
        for (double[] c : ctrl) cand.add(c[0]);
        double best = Double.POSITIVE_INFINITY;
        double eer = Double.NaN;
        for (double th : cand) {
            double eo = ownErr(own, th);
            double fc = 100.0 * ctrl.stream().filter(c -> c[0] <= th).count() / ctrl.size();
            if (Math.abs(eo - fc) < best) {
                best = Math.abs(eo - fc);
                eer = (eo + fc) / 2;
            }
        }
        // Разделимость: порог по контрольным чужим, FAR 1 % по снимкам.
        double[] cs = ctrl.stream().mapToDouble(c -> c[0]).sorted().toArray();
        long allowed = (long) Math.floor(FAR * cs.length + 1e-9);
        double thc = allowed >= cs.length ? Double.POSITIVE_INFINITY : Math.nextDown(cs[(int) allowed]);
        double sep = ownErr(own, thc);
        return new double[] {e2, e3, far, farP, eer, sep};
    }

    static double ownErr(List<double[]> own, double th) {
        long e = own.stream().filter(o -> !(o[1] == 1 && o[0] <= th)).count();
        return 100.0 * e / own.size();
    }

    /** Ошибка своих П2 при весах людей w (порог заново по «чужим» с теми же весами). */
    static double ownErrW(Pair p, Map<String, Integer> pIdx, int[] w) {
        double th = FeiBootstrap.threshold(p.cvR, p.cvP, w, FAR);
        if (Double.isNaN(th)) return Double.NaN; // порог не определён (решение Хозяина 20.10.2026, №9 Жорика)
        double e = 0;
        double n = 0;
        for (Att a : p.atts) {
            if (!a.probe().role().equals(FeiScores.OWN)) continue;
            int wt = w[pIdx.get(a.probe().person())];
            n += wt;
            if (!(a.row().bestId().equals(a.row().trueId()) && a.ratio() <= th)) e += wt;
        }
        return 100 * e / n;
    }

    static final String HEAD = "конфигурация | метод | П2 FAR 1 % было → стало | П3 FAR 1 % | FAR контроля по снимкам | по людям | EER | разделимость (FAR "
            + "контроля 1 %) | разность П2 «новое − прежнее» [2,5; 97,5] (L1, L1-tt)\n";

    /** Строка сравнения пары; diff — считать парную разность (бутстреп). */
    static String line(String conf, String method, Pair old, Pair neu, Map<String, Integer> pIdx, int b, long seed, boolean diff) {
        int n = pIdx.size();
        double[] mo = metrics(old, n);
        double[] mn = metrics(neu, n);
        StringBuilder s = new StringBuilder(conf + " | " + method);
        for (int i = 0; i < mo.length; i++) s.append(String.format(Locale.ROOT, " | %.1f → %.1f", mo[i], mn[i]));
        if (diff && b > 0) {
            int[][] w = FeiBootstrap.weights(n, b, seed);
            double[] d = new double[b];
            for (int q = 0; q < b; q++) d[q] = ownErrW(neu, pIdx, w[q]) - ownErrW(old, pIdx, w[q]);
            Arrays.sort(d);
            double point = mn[0] - mo[0];
            double lo = XBaseA0Check.q(d, 0.025);
            double hi = XBaseA0Check.q(d, 0.975);
            s.append(String.format(Locale.ROOT, " | %+.1f [%+.1f; %+.1f]%s", point, lo, hi, lo > 0 || hi < 0 ? " *" : ""));
        } else {
            s.append(" | –");
        }
        s.append(String.format(Locale.ROOT, " (θ %.4f → %.4f)%n", old.theta, neu.theta));
        return s.toString();
    }

    /** Ошибка своих П2 только по пробам людей set (каждая пара — своим порогом). */
    static String subset(String conf, String method, Pair old, Pair neu, Set<String> set) {
        int[] e = new int[2];
        int[] n = new int[2];
        Pair[] ps = {old, neu};
        for (int i = 0; i < 2; i++) {
            for (Att a : ps[i].atts) {
                if (!a.probe().role().equals(FeiScores.OWN) || !set.contains(a.probe().person())) continue;
                n[i]++;
                if (a.group() != 0) e[i]++;
            }
        }
        return String.format(Locale.ROOT, "%s | %s | люди спорного случая (%d) | ошибка своих П2: %d/%d → %d/%d%n", conf, method, set.size(), e[0], n[0], e[1], n[1]);
    }

    /** Хаб: лучший кандидат с наибольшим числом принятых контрольных чужих (П2). */
    static String hub(Pair p) {
        Map<String, Integer> c = new TreeMap<>();
        int all = 0;
        for (Att a : p.atts) {
            if (a.group() != 5) continue;
            all++;
            c.merge(a.row().bestId(), 1, Integer::sum);
        }
        if (all == 0) return "принятых 0";
        Map.Entry<String, Integer> top = c.entrySet().stream().max(Map.Entry.comparingByValue()).get();
        return String.format(Locale.ROOT, "принятых %d, людей %d, наибольший – %s: %d", all, c.size(), top.getKey(), top.getValue());
    }

    // ---------------------------------------------------------------- FERET

    record Conf(String name, int s, List<String> own, Map<String, List<String>> galOld, Map<String, List<String>> galNew, Map<String, List<String>> pro,
                boolean ex1) {}

    record M(String label, String set, String id, V v) {}

    static final List<M> FERET_METHODS = List.of(new M("L1", "none", FeiMethods.LBP_MLDA, V.L1), new M("A", "none", "2c-mlda", V.A),
            new M("L1-tt", "tt", FeiMethods.LBP_MLDA, V.L1), new M("LTP-in-tt", "tt", "5-ltp-mlda-in", V.LTPIN), new M("SFace", "SFace", "0-sface", V.SFACE));

    static void feret(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseFeret.Roles ro = su.ro;
        XBaseFeret.Roles rn = XBaseFeret.equalize(ro, su.byKey);
        Path rep = su.dir.resolve("reports");
        Path fOld = rep.resolve(XBaseA0.FERET_SCORES);
        Path fNew = rep.resolve("eq").resolve(XBaseA0.FERET_SCORES);
        int b = su.tpl.b;
        long seed = su.tpl.seed + FeiBootstrap.SEED_OFFSET;
        StringBuilder t = new StringBuilder();
        t.append("№8: FERET – прежняя галерея → правило fa/fb (XBaseEval feret-eq-compare; PLAN.md п. 5.5); только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Прежние оценки: ").append(fOld).append(" (").append(FeiThreshold.sha(fOld)).append(")\n");
        t.append("Новые оценки: ").append(fNew).append(" (").append(FeiThreshold.sha(fNew)).append(")\n");
        t.append(XBaseFeret.EQ_NOTE);
        t.append(String.format(Locale.ROOT, "Бутстреп по людям роли и контроля: B = %d, seed = %d; * – интервал без нуля.%n", b, seed));
        List<Conf> confs = List.of(new Conf("межс.", 0, ro.own1, ro.gal1, rn.gal1, ro.pro1, false), new Conf("внутри", 1, ro.own1in, ro.gal1in, rn.gal1in, ro.pro1in, false),
                new Conf("G1", 2, ro.own2, ro.g1, rn.g1, ro.pro2, true), new Conf("G2-i", 3, ro.own2, ro.g2, rn.g2, ro.pro2, true),
                new Conf("G2-ii", 4, ro.own2, ro.g2, rn.g2, ro.pro2, true), new Conf("G2'", 5, ro.own2, ro.g2p, rn.g2p, ro.pro2, true));
        StringBuilder tab = new StringBuilder("\n=== Прежнее → новое ===\n" + HEAD);
        StringBuilder sub = new StringBuilder("\n=== Люди спорного случая (А): на дате галереи нет чистого fa или fb ===\n");
        StringBuilder hubs = new StringBuilder("\n=== Хабы: лучший кандидат принятых контрольных чужих (П2), прежнее | новое ===\n");
        for (Conf c : confs) {
            TreeSet<String> ps = new TreeSet<>(c.own());
            ps.addAll(ro.ctrl);
            List<String> persons = new ArrayList<>(ps);
            XBaseEval.Gt base = XBaseFeret.gt(su.tpl, persons, ro.ctrl, ro.ctrlKeys, new ArrayList<>(), su.rOf, su.imageNo);
            XBaseEval.Gt g = XBaseFeret.withOwn(base, c.own());
            List<Probe> probes = XBaseEval.probes(g, c.pro());
            List<List<String>> parts = XBaseOther.cvParts(c.own(), su.tpl.seed, XBaseEval.CV_S + c.s());
            // G2' — по одному снимку на дату (прежний выбор), спорного случая нет по определению.
            Set<String> disputed = c.s() == 5 ? Set.of() : disputed(c.galOld(), su.byKey, c.s() == 1);
            for (M m : FERET_METHODS) {
                String block = "# конфигурация " + c.name() + ", набор " + m.set();
                Pair old = XBaseA0.decide("FERET", m.label(), IlluminationNorm.NONE, m.v(), XBaseA0.read(fOld, m.id(), block), g, c.galOld(), probes, parts);
                Pair neu = XBaseA0.decide("FERET", m.label(), IlluminationNorm.NONE, m.v(), XBaseA0.read(fNew, m.id(), block), g, c.galNew(), probes, parts);
                tab.append(line(c.name(), m.label(), old, neu, g.pIdx, b, seed, m.label().equals("L1") || m.label().equals("L1-tt")));
                if (!disputed.isEmpty() && (m.label().equals("L1") || m.label().equals("L1-tt"))) sub.append(subset(c.name(), m.label(), old, neu, disputed));
                if (m.label().equals("L1") || m.label().equals("L1-tt")) hubs.append(c.name()).append(" | ").append(m.label()).append(" | ").append(hub(old))
                        .append(" | ").append(hub(neu)).append('\n');
            }
        }
        t.append(tab).append(sub).append(hubs);
        FarMethods.write(rep.resolve("xbase_feret_eq_compare.txt"), t);
        System.out.print(t);
    }

    /** Люди, у которых на какой-либо дате галереи нет чистого fa (или fb; faOnly — только fa). */
    static Set<String> disputed(Map<String, List<String>> gal, Map<String, Img> byKey, boolean faOnly) {
        Set<String> out = new TreeSet<>();
        for (Map.Entry<String, List<String>> e : gal.entrySet()) {
            Map<String, Set<String>> byDate = new TreeMap<>();
            for (String k : e.getValue()) {
                Img im = byKey.get(k);
                if (im.flag().equals("-")) byDate.computeIfAbsent(im.date(), d -> new HashSet<>()).add(im.pose());
                else byDate.computeIfAbsent(im.date(), d -> new HashSet<>());
            }
            for (Set<String> poses : byDate.values()) {
                if (!poses.contains("fa") || !faOnly && !poses.contains("fb")) out.add(e.getKey());
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        if (g.eq != null) throw new IllegalStateException("gt-eq-compare: faces.gallery.equal не задавать (прежняя галерея читается как есть)");
        int b = g.b;
        long seed = g.seed + FeiBootstrap.SEED_OFFSET;
        List<Probe> cross = XBaseEval.probes(g, g.probeKeys);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        List<Probe> in = XBaseEval.probes(g, split.get(1));
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        // {подпись режима, файл оценок, метод, вариант, внутри сессии}.
        Object[][] rows = {{"межс.", "xbase_gt_scores.tsv.gz", FeiMethods.LBP_MLDA, V.L1, "L1", false}, {"межс.", "xbase_gt_scores.tsv.gz", "2c-mlda", V.A, "A", false},
            {"межс.", "xbase_gt_ltp_scores.tsv.gz", FeiMethods.LBP_MLDA, V.L1, "L1-tt", false}, {"межс.", "xbase_gt_diag_scores.tsv.gz", "0-sface", V.SFACE, "SFace", false},
            {"внутри", "xbase_gt_in_scores.tsv.gz", FeiMethods.LBP_MLDA, V.L1, "L1", true}, {"внутри", "xbase_gt_in_scores.tsv.gz", "2c-mlda", V.A, "A", true},
            {"внутри", "xbase_gt_in_ltp_scores.tsv.gz", FeiMethods.LBP_MLDA, V.L1, "L1-tt", true},
            {"внутри", "xbase_gt_in_diag_scores.tsv.gz", "0-sface", V.SFACE, "SFace", true}};
        StringBuilder t = new StringBuilder();
        t.append("№8: GT – прежняя галерея → N снимков у всех своих (XBaseEval gt-eq-compare; PLAN.md п. 5.5)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append('\n');
        t.append("Правило: N = наименьшее число снимков галереи среди своих роли (межс. – дата галереи; внутри – половина даты), N случайных снимков, "
                + "seed faces.seed + 9800 + k (k = 0 – основной, 1, 2 – справочно); пробы прежние; «чужие» П2/П3 – из новой галереи.\n");
        t.append(String.format(Locale.ROOT, "Галерея: межс. – прежде %s снимков на человека, N = %d; внутри – прежде %s, N = %d. Сравнение на GT смешивает "
                + "«поровну» и «меньше снимков» (решение Жорика 13.10: главный ответ – по FERET).%n", range(g.galleryAll), min(g.galleryAll), range(split.get(0)),
                min(split.get(0))));
        t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d; * – интервал без нуля.%n", b, seed));
        for (int k = 0; k < 3; k++) {
            Path nd = out.resolve(XBaseEval.eqDir(k, false));
            Map<String, List<String>> crossNew = XBaseEval.equalize(g.own, g.galleryAll, g.seed + XBaseEval.EQ_SEED_OFFSET + k);
            Map<String, List<String>> inNew = XBaseEval.equalize(g.own, split.get(0), g.seed + XBaseEval.EQ_SEED_OFFSET + k);
            t.append(String.format(Locale.ROOT, "%n=== seed faces.seed + %d%s (новые оценки – %s) ===%n", XBaseEval.EQ_SEED_OFFSET + k, k == 0 ? ", основной" : ", справочно", nd));
            t.append(HEAD);
            for (Object[] r : rows) {
                boolean within = (boolean) r[5];
                Map<String, List<String>> galOld = within ? split.get(0) : g.galleryAll;
                Map<String, List<String>> galNew = within ? inNew : crossNew;
                List<Probe> probes = within ? in : cross;
                Path fo = out.resolve((String) r[1]);
                Path fn = nd.resolve((String) r[1]);
                Pair old = XBaseA0.decide("GT", (String) r[4], IlluminationNorm.NONE, (V) r[3], XBaseA0.read(fo, (String) r[2], null), g, galOld, probes, parts);
                Pair neu = XBaseA0.decide("GT", (String) r[4], IlluminationNorm.NONE, (V) r[3], XBaseA0.read(fn, (String) r[2], null), g, galNew, probes, parts);
                String lab = (String) r[4];
                t.append(line((String) r[0], lab, old, neu, g.pIdx, b, seed, lab.equals("L1") || lab.equals("L1-tt")));
            }
        }
        FarMethods.write(out.resolve("xbase_gt_eq_compare.txt"), t);
        System.out.print(t);
    }

    static int min(Map<String, List<String>> gal) {
        return gal.values().stream().mapToInt(List::size).min().orElse(0);
    }

    static String range(Map<String, List<String>> gal) {
        int[] s = gal.values().stream().mapToInt(List::size).sorted().toArray();
        return String.format(Locale.ROOT, "%d–%d (медиана %d)", s[0], s[s.length - 1], s[s.length / 2]);
    }
}
