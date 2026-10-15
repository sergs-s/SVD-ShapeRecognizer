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
import java.util.zip.GZIPInputStream;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Пессимизм контроля FERET (PLAN.md п. 5.5, решение (13); задание №2 от 15.10.2026, план подтверждён Хозяином): по готовым
 * файлам оценок, без распознавания. Для каждого контрольного снимка — модель по позе, d₁/d₂, лучший кандидат, принят ли при
 * П2 и П3 (FAR 1 %, порог — FeiThreshold.evaluate, свой на проход; FAR «всего» сверяется с ним). Приписывание: доля принятых,
 * у которых дата снимка совпадает с одной из дат галереи шаблона, к которому приняли. Расслоение: S — контрольные снимки,
 * дата которых есть среди дат галереи своих прохода, N — остальные; FAR по снимкам и по людям с верхней 95 % границей КП.
 * Режим XBaseEval feret-sameday; отчёт — только D:\data\Feret\reports\eq\xbase_feret_sameday.txt.
 *
 * @author ssv
 */
final class XBaseSameDay {

    private XBaseSameDay() {
    }

    /** Проход сценария: Gt со своими, галерея, пробы, части «чужих», строки блока. */
    record Pass(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, String lines) {}

    /** Блоки файла оценок: заголовок «# …» (без «# ») → строки до следующего заголовка. */
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

    static String need(Map<String, String> blocks, String name, Path file) {
        String s = blocks.get(name);
        if (s == null || s.isEmpty()) throw new IllegalStateException("feret-sameday: нет блока «" + name + "» в " + file);
        return s;
    }

    /** Ключ без суффикса синтеза или дубля (#…). */
    static String base(String k) {
        int i = k.indexOf('#');
        return i < 0 ? k : k.substring(0, i);
    }

    /** Строка таблицы по сценарию × методу: для П2 и П3. */
    static void evaluate(StringBuilder t, String label, List<Pass> passes, int persons, XBaseA0.FeretSetup su) {
        List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        List<FeiThreshold.Imp> imps = new ArrayList<>();
        List<List<Probe>> probesOf = new ArrayList<>();
        for (Pass p : passes) {
            List<Probe> probes = XBaseEval.probes(p.g(), p.pro());
            XBaseEval.Parsed ps = XBaseEval.parse(p.g(), EnumSet.of(V.L1), p.gal(), probes, p.parts(), p.lines());
            cfgs.add(ps.cfg.get(V.L1));
            imps.add(ps.imp.get(V.L1));
            probesOf.add(probes);
        }
        FeiThreshold.Res r = FeiThreshold.evaluate(label, V.L1.ratioId(), cfgs, imps, persons, 0, 0);
        for (int m : new int[] {2, 3}) {
            int[] c = new int[8];
            // c: 0 — контроль всего, 1 — принято, 2 — принято к шаблону той же даты, 3 — снимков S, 4 — принято в S, 5 — снимков N, 6 — принято в N
            Map<String, boolean[]> people = new TreeMap<>();
            for (int ci = 0; ci < passes.size(); ci++) {
                Pass p = passes.get(ci);
                double th = r.theta[m][0][ci];
                Map<String, Set<String>> galDates = new HashMap<>();
                Set<String> allDates = new HashSet<>();
                for (Map.Entry<String, List<String>> e : p.gal().entrySet()) {
                    Set<String> ds = new HashSet<>();
                    for (String k : e.getValue()) ds.add(su.byKey.get(base(k)).date());
                    galDates.put(e.getKey(), ds);
                    allDates.addAll(ds);
                }
                Map<String, Row> rows = new HashMap<>();
                for (String l : p.lines().lines().toList()) {
                    Row row = FeiScores.parse(l);
                    rows.put(row.method() + "\t" + row.role() + "\t" + row.person() + "\t" + row.image(), row);
                }
                for (Probe q : probesOf.get(ci)) {
                    if (q.role().equals(FeiScores.OWN)) continue;
                    Row row = XBaseEval.choose(rows, V.L1, q.role(), q.person(), q.image(), p.g().rOf.get(q.key()) <= p.g().r0);
                    boolean acc = FeiScores.ratio(row, null) <= th;
                    String date = su.byKey.get(q.key()).date();
                    boolean inS = allDates.contains(date);
                    c[0]++;
                    c[inS ? 3 : 5]++;
                    String pk = q.person() + (inS ? "\tS" : "\tN");
                    boolean[] pa = people.computeIfAbsent(pk, x -> new boolean[1]);
                    if (acc) {
                        c[1]++;
                        c[inS ? 4 : 6]++;
                        pa[0] = true;
                        if (galDates.get(row.bestId()).contains(date)) c[2]++;
                    }
                }
            }
            if (c[1] != (int) r.far[m][0][0] || c[0] != (int) r.far[m][0][1]) {
                throw new IllegalStateException(String.format(Locale.ROOT, "feret-sameday %s m=%d: FAR не совпал с FeiThreshold (%d/%d против %.0f/%.0f)", label, m,
                        c[1], c[0], r.far[m][0][0], r.far[m][0][1]));
            }
            int[] ps = new int[4];
            for (Map.Entry<String, boolean[]> e : people.entrySet()) {
                int s = e.getKey().endsWith("\tS") ? 0 : 2;
                ps[s]++;
                if (e.getValue()[0]) ps[s + 1]++;
            }
            t.append(String.join(" | ", label, m == 2 ? "П2" : "П3", cell(c[1], c[0]),
                    c[1] == 0 ? "–" : String.format(Locale.ROOT, "%d/%d = %.0f %%", c[2], c[1], 100.0 * c[2] / c[1]),
                    String.format(Locale.ROOT, "%d/%d = %.1f %%", c[3], c[0], 100.0 * c[3] / c[0]), cell(c[4], c[3]), cell(ps[1], ps[0]), cell(c[6], c[5]),
                    cell(ps[3], ps[2]))).append('\n');
        }
    }

    static String cell(int x, int n) {
        if (n == 0) return "–";
        return String.format(Locale.ROOT, "%d/%d = %.2f %% (↑%.2f %%)", x, n, 100.0 * x / n, 100 * FaceEvaluation.binomialUpperBound(x, n, 0.95));
    }

    static void run(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-sameday: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path out = su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles roAll = su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(roAll, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);
        List<List<String>> none = List.of();

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, решение (13): пессимизм контроля FERET – чужие той же сессии, что галерея (XBaseEval feret-sameday): только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("По готовым файлам оценок D:\\data\\Feret\\reports\\eq\\ (без распознавания); порог П2 / П3 при FAR 1 % – FeiThreshold.evaluate, свой на проход;\n"
                + "FAR «всего» сверен с ним (иначе остановка). «Та же дата» – доля принятых контрольных снимков, дата которых совпадает с одной из дат\n"
                + "галереи шаблона, к которому приняли. S – контрольные снимки с датой из дат галереи своих прохода, N – остальные («чужой в день, когда\n"
                + "своих не регистрировали»); FAR по снимкам и по людям (человек слоя принят, если принят хоть один его снимок слоя), ↑ – верхняя 95 %\n"
                + "граница Клоппера – Пирсона.\n");
        StringBuilder tab = new StringBuilder("\n=== Сценарий × метод × порог ===\nсценарий и метод | порог | FAR всего | принятых к шаблону той же даты | доля снимков S | "
                + "S: FAR по снимкам | S: по людям | N: FAR по снимкам | N: по людям\n");
        Map<String, String> files = new LinkedHashMap<>();

        // ---- feret (eq): межс., внутри, G1, G2'; L1, L1-tt.
        Path f1 = out.resolve("xbase_feret_scores.tsv.gz");
        Map<String, String> b1 = blocks(f1);
        files.put("feret", f1.getFileName() + " (" + FeiThreshold.sha(f1) + ")");
        List<XBaseXsF.Conf> confs = List.of(new XBaseXsF.Conf("межс.", 0, ro.own1, ro.gal1, ro.pro1), new XBaseXsF.Conf("внутри", 1, ro.own1in, ro.gal1in, ro.pro1in),
                new XBaseXsF.Conf("G1", 2, ro.own2, ro.g1, ro.pro2), new XBaseXsF.Conf("G2'", 5, ro.own2, ro.g2p, ro.pro2));
        for (XBaseXsF.Conf c : confs) {
            TreeSet<String> ps = new TreeSet<>(c.s() < 2 ? ro.own1 : ro.own2);
            ps.addAll(ro.ctrl);
            List<String> persons = new ArrayList<>(ps);
            XBaseEval.Gt g = XBaseFeret.withOwn(XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, none, su.rOf, su.imageNo), c.own());
            List<List<String>> parts = XBaseOther.cvParts(c.own(), tpl.seed, XBaseEval.CV_S + c.s());
            for (String[] mt : new String[][] {{"L1", "none"}, {"L1-tt", "tt"}}) {
                String ls = need(b1, "конфигурация " + c.name() + ", набор " + mt[1], f1);
                evaluate(tab, "feret " + c.name() + " " + mt[0], List.of(new Pass(g, c.gal(), c.pro(), parts, ls)), persons.size(), su);
            }
        }

        // ---- feret-xs2: половины, оба прохода.
        Path f2 = out.resolve("xbase_feret_xs2_scores.tsv.gz");
        Map<String, String> b2 = blocks(f2);
        files.put("feret-xs2", f2.getFileName() + " (" + FeiThreshold.sha(f2) + ")");
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> psH = new TreeSet<>(pool.keySet());
        psH.addAll(ro.ctrl);
        List<String> personsH = new ArrayList<>(psH);
        XBaseEval.Gt baseH = XBaseFeret.gt(tpl, personsH, ro.ctrl, ro.ctrlKeys, none, su.rOf, su.imageNo);
        for (XBaseXsF.Conf c : confs) {
            for (String mt : new String[] {"L1", "L1-xs-F", "L1-tt", "L1-tt-xs-F"}) {
                List<Pass> passes = new ArrayList<>();
                for (int pp = 0; pp < 2; pp++) {
                    passes.add(halfPass(baseH, c, halves.get(pp), tpl, need(b2, "проход " + (pp + 1) + ", конфигурация " + c.name() + ", строка " + mt, f2)));
                }
                evaluate(tab, "feret-xs2 " + c.name() + " " + mt, passes, personsH.size(), su);
            }
        }

        // ---- feret-calib, сценарий 1 (К0 и К0' – те же блоки).
        Path f3 = out.resolve("xbase_feret_calib_scores.tsv.gz");
        Map<String, String> b3 = blocks(f3);
        files.put("feret-calib", f3.getFileName() + " (" + FeiThreshold.sha(f3) + ")");
        for (XBaseXsF.Conf c : confs) {
            for (String mt : new String[] {"L1", "L1-tt", "L1-xs-F", "L1-tt-xs-F"}) {
                List<Pass> passes = new ArrayList<>();
                for (int pp = 0; pp < 2; pp++) {
                    passes.add(halfPass(baseH, c, halves.get(pp), tpl, need(b3, String.format(Locale.ROOT, "s1_%d_%s_p%d", c.s(), mt, pp), f3)));
                }
                evaluate(tab, "feret-calib-1 " + c.name() + " " + mt, passes, personsH.size(), su);
            }
        }

        // ---- Смешанная галерея D = 2: feret-synth (а), feret-synth2 (в2), feret-calib сценарий 2 ((а), (в2)).
        int dd = 2;
        List<String> elig = new ArrayList<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= dd + 1) elig.add(e.getKey());
        List<String> permE = new ArrayList<>(elig);
        Collections.shuffle(permE, new Random(tpl.seed + XBaseSynth.SPLIT_SEED_OFFSET));
        int hE = (permE.size() + 1) / 2;
        List<Set<String>> halvesE = List.of(new TreeSet<>(permE.subList(0, hE)), new TreeSet<>(permE.subList(hE, permE.size())));
        List<String> own = new ArrayList<>(new TreeSet<>(elig));
        Map<String, List<String>> pro = new LinkedHashMap<>();
        Map<String, List<String>> oldGal = new LinkedHashMap<>();
        Map<String, List<String>> synth2 = new LinkedHashMap<>();
        for (String p : own) {
            List<List<String>> ds = pool.get(p);
            List<String> gl = new ArrayList<>();
            for (int j = 0; j < dd; j++) gl.addAll(ds.get(j));
            oldGal.put(p, XBaseFeret.sorted(gl));
            List<String> pr = new ArrayList<>();
            for (int j = dd; j < ds.size(); j++) pr.addAll(ds.get(j));
            pro.put(p, XBaseFeret.sorted(pr));
            List<String> real = ds.get(0);
            List<String> ks = new ArrayList<>();
            for (int i = 0; i < 2 * dd - 2; i++) {
                String b = real.get(i % real.size());
                String key = b + "#t" + dd + "_" + i;
                ks.add(key);
                su.rOf.put(key, su.rOf.get(b));
                su.imageNo.put(key, 3000 + 100 * dd + 10 * su.imageNo.get(b) + i);
            }
            synth2.put(p, ks);
        }
        Path f4 = out.resolve("xbase_feret_synth_scores.tsv.gz");
        Path f5 = out.resolve("xbase_feret_synth2_scores.tsv.gz");
        Map<String, String> b4 = blocks(f4);
        Map<String, String> b5 = blocks(f5);
        files.put("feret-synth", f4.getFileName() + " (" + FeiThreshold.sha(f4) + ")");
        files.put("feret-synth2", f5.getFileName() + " (" + FeiThreshold.sha(f5) + ")");
        TreeSet<String> psE = new TreeSet<>(elig);
        psE.addAll(ro.ctrl);
        List<String> personsE = new ArrayList<>(psE);
        XBaseEval.Gt baseE = XBaseFeret.gt(tpl, personsE, ro.ctrl, ro.ctrlKeys, none, su.rOf, su.imageNo);
        // feret-calib сценарий 2: в людях бутстрепа ещё калибровочные (на порог П2 / П3 не влияют – веса единичные).
        Set<String> two = new TreeSet<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (!elig.contains(e.getKey()) && e.getValue().size() == 2) two.add(e.getKey());
        TreeSet<String> psC = new TreeSet<>(psE);
        psC.addAll(XBaseCalib.split(two, tpl.seed).get(1));
        List<String> personsC = new ArrayList<>(psC);
        XBaseEval.Gt baseC = XBaseFeret.gt(tpl, personsC, ro.ctrl, ro.ctrlKeys, none, su.rOf, su.imageNo);
        List<List<String>> partsE = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + 30 + dd);
        for (String pj : new String[] {"L1-xs-F", "L1-tt-xs-F", "L1"}) {
            for (String row : new String[] {"A", "C2"}) {
                for (int src = 0; src < 2; src++) {
                    List<Pass> passes = new ArrayList<>();
                    for (int pp = 0; pp < 2; pp++) {
                        Set<String> newbies = halvesE.get(pp);
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        for (String p : own) {
                            List<String> gl = newbies.contains(p) ? new ArrayList<>(pool.get(p).get(0)) : new ArrayList<>(oldGal.get(p));
                            if (newbies.contains(p) && row.equals("C2")) gl.addAll(synth2.get(p));
                            gal.put(p, gl);
                        }
                        String ls;
                        if (src == 0) {
                            ls = row.equals("A") ? need(b4, String.format(Locale.ROOT, "D %d, проход %d, проекция %s, строка A", dd, pp + 1, pj), f4)
                                    : need(b5, String.format(Locale.ROOT, "D %d, проход %d, проекция %s, строка C2", dd, pp + 1, pj), f5);
                        } else {
                            ls = need(b3, String.format(Locale.ROOT, "s2_%s_%s_p%d", pj, row.equals("A") ? "(а) " : "(в2)", pp), f3);
                        }
                        passes.add(new Pass(XBaseFeret.withOwn(src == 0 ? baseE : baseC, own), gal, pro, partsE, ls));
                    }
                    String lbl = (src == 0 ? "смешанная D = 2 " : "feret-calib-2 ") + pj + (row.equals("A") ? " (а)" : " (в2)");
                    evaluate(tab, lbl, passes, src == 0 ? personsE.size() : personsC.size(), su);
                }
            }
        }
        t.append("Файлы оценок: ");
        files.forEach((k, v) -> t.append(k).append(" – ").append(v).append("; "));
        t.append('\n');
        t.append("Примечание: feret-calib сценарий 1 – те же половины, но пул проекции B1 (L1-xs-F) вдвое меньше, чем в feret-xs2; строки L1 и L1-tt без пула\n"
                + "совпадают с feret-xs2 по модели (проверка воспроизводимости).\n");
        t.append(tab);
        FarMethods.write(out.resolve("xbase_feret_sameday.txt"), t);
        System.out.print(t);
    }

    /** Проход схемы половин: свои конфигурации — люди половины mine. */
    static Pass halfPass(XBaseEval.Gt base, XBaseXsF.Conf c, Set<String> mine, XBaseEval.Gt tpl, String lines) {
        List<String> own = c.own().stream().filter(mine::contains).toList();
        Map<String, List<String>> gal = new LinkedHashMap<>();
        Map<String, List<String>> pro = new LinkedHashMap<>();
        for (String p : own) {
            gal.put(p, c.gal().get(p));
            pro.put(p, c.pro().get(p));
        }
        return new Pass(XBaseFeret.withOwn(base, own), gal, pro, XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s()), lines);
    }
}
