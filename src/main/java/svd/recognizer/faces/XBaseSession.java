package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import svd.recognizer.storage.SettingsStore;

/**
 * Гипотеза общей сессии (задание Жорика №6 от 15.10.2026, п. 1; PLAN.md п. 5.5, очередь (4)) – по готовому файлу оценок
 * feret-update (№4), без распознавания: шаблоны галереи по шагам восстанавливаются из строк своих и порогов (правила
 * XBaseUpdate), затем для контрольных проб считается доля случаев, когда дата пробы чужого совпадает с одной из дат шаблона
 * его лучшего кандидата, против ожидаемой при случайном кандидате (доля людей галереи прохода, у которых в шаблоне есть эта
 * дата). Бутстреп по контрольным людям. Режим XBaseEval feret-session; отчёт – только D:\data\Feret\reports\eq\.
 *
 * @author ssv
 */
final class XBaseSession {

    private XBaseSession() {
    }

    static final String SCORES = "xbase_feret_update_scores.tsv.gz";
    static final String GATE = "П3";
    /** Меньше проб – интервал не даётся. */
    static final int MIN_N = 10;
    static final List<String> VARS = List.of(XBaseUpdate.Var.A.label, XBaseUpdate.Var.B.label, XBaseUpdate.Var.C.label, XBaseUpdate.Var.E.label,
            XBaseUpdate.Var.F.label);

    record Row(String role, String person, String key, String best, double r) {}

    static String date(String key) {
        return key.split("_")[1];
    }

    /** Окно M для подписи варианта (а – без обновлений). */
    static int windowOf(String var) {
        for (XBaseUpdate.Var v : XBaseUpdate.Var.values()) if (v.label.equals(var)) return v == XBaseUpdate.Var.A ? TemplateWindow.UNBOUNDED : v.m;
        throw new IllegalArgumentException(var);
    }

    static boolean oracle(String var) {
        for (XBaseUpdate.Var v : XBaseUpdate.Var.values()) if (v.label.equals(var)) return v.oracle;
        throw new IllegalArgumentException(var);
    }

    /** Совпадения одной группы: на контрольного человека – {наблюдаемых, ожидаемых (сумма долей), проб}. */
    static final class Acc {
        final Map<String, double[]> byCtrl = new TreeMap<>();

        void add(String ctrl, boolean same, double expected) {
            double[] a = byCtrl.computeIfAbsent(ctrl, k -> new double[3]);
            if (same) a[0]++;
            a[1] += expected;
            a[2]++;
        }

        double[] total(Map<String, Integer> w) {
            double[] t = new double[3];
            for (Map.Entry<String, double[]> e : byCtrl.entrySet()) {
                int q = w == null ? 1 : w.getOrDefault(e.getKey(), 0);
                for (int i = 0; i < 3; i++) t[i] += q * e.getValue()[i];
            }
            return t;
        }
    }

    static void run(List<String> selLines, Path exportDir, SettingsStore settings) throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-session: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path out = su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));

        // Строки файла оценок: проекция → проход → вариант → шаг → строки; пороги П3 шага.
        Map<String, List<Row>> rows = new HashMap<>();
        Map<String, Double> th3 = new HashMap<>();
        Path sf = out.resolve(SCORES);
        String code = null;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(sf), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("# Код:")) code = l.substring(2);
                if (l.startsWith("#") || l.startsWith("proj\t")) continue;
                String[] c = l.split("\t", -1);
                if (!c[2].equals(GATE)) continue;
                String k = c[0] + "\t" + c[1] + "\t" + c[3] + "\t" + c[4];
                if (c[5].equals("threshold")) {
                    th3.put(k, Double.parseDouble(c[10].substring(c[10].indexOf("П3 ") + 3)));
                } else {
                    rows.computeIfAbsent(k, z -> new ArrayList<>()).add(new Row(c[5], c[6], c[8], c[9], Double.parseDouble(c[10])));
                }
            }
        }
        int maxStep = 0;
        for (String k : th3.keySet()) maxStep = Math.max(maxStep, Integer.parseInt(k.split("\t")[3]));

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4): гипотеза общей сессии по оценкам feret-update (XBaseEval feret-session): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл оценок: ").append(sf).append(" (").append(code).append("; ").append(FeiThreshold.sha(sf)).append(")\n");
        t.append("""
                Шаблоны по шагам восстановлены из строк своих (порог приёма П3, правила XBaseUpdate: старт – 2 даты; (б), (в) – принятая дата
                в шаблон лучшего кандидата, окно M; (д), (е) – дата в шаблон истинного, окно M; (а) – без обновлений). Совпадение – дата
                решающего снимка контрольного чужого (сессия = дата FERET) есть среди дат шаблона его лучшего кандидата. Ожидаемое при
                случайном кандидате – доля людей галереи прохода, у которых в шаблоне есть эта дата. Разность «наблюдаемое − ожидаемое» –
                бутстреп по контрольным людям (B = %d, seed faces.seed + 7000 = %d; человек – оба прохода вместе), «*» – интервал
                [2,5; 97,5] не содержит 0. Приём – d₁/d₂ ≤ П3 своего шага. «Шаги ≥ 2» – попытки всех шагов вместе (у (а) шаги
                одинаковы – одни и те же пробы повторяются, это не новые данные; у приёмов на разных шагах может быть один и тот же человек).
                """.formatted(tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        t.append("\n=== Проверка восстановления: обновлений после шага 1 (верных / неверных), сравнить с ходом по шагам xbase_feret_update.txt ===\n");
        StringBuilder tab = new StringBuilder("\n=== Совпадение даты чужого с датой шаблона лучшего кандидата ===\n"
                + "проекция | вариант | шаги | все контрольные: наблюдаемое | ожидаемое | разность [95 %] | принятые при П3: наблюдаемое (x/n) | "
                + "ожидаемое | разность [95 %]\n");
        List<String> ctrlP = new ArrayList<>(new TreeSet<>(su.ro.ctrl));
        int[][] w = FeiBootstrap.weights(ctrlP.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        for (XBaseUpdate.Proj pj : XBaseUpdate.PROJS) {
            for (String var : VARS) {
                Acc[][] acc = new Acc[2][2]; // [шаг 2 / шаги ≥ 2][все / принятые]
                for (Acc[] a : acc) for (int i = 0; i < 2; i++) a[i] = new Acc();
                int[] upd1 = new int[2];
                for (int pp = 0; pp < 2; pp++) {
                    List<String> gal = new ArrayList<>(halves.get(pp));
                    Map<String, TemplateWindow> tw = new HashMap<>();
                    for (String p : gal) {
                        TemplateWindow win = new TemplateWindow(windowOf(var));
                        for (int d = 0; d < XBaseUpdate.START; d++) {
                            List<String> ks = pool.get(p).get(d);
                            win.add(new TemplateWindow.Session(date(ks.get(0)), p, ks));
                        }
                        tw.put(p, win);
                    }
                    for (int j = 1; j <= maxStep; j++) {
                        String k = pj.label() + "\t" + (pp + 1) + "\t" + var + "\t" + j;
                        List<Row> rs = rows.get(k);
                        if (rs == null) throw new IllegalStateException("feret-session: нет строк " + k.replace('\t', ' '));
                        double th = th3.get(k);
                        if (j >= 2) {
                            // Даты шаблонов галереи на этом шаге.
                            Map<String, Set<String>> dates = new HashMap<>();
                            for (String p : gal) {
                                Set<String> ds = new HashSet<>();
                                for (TemplateWindow.Session s : tw.get(p).sessions()) ds.add(s.date());
                                dates.put(p, ds);
                            }
                            for (Row r : rs) {
                                if (!r.role().equals(FeiScores.CTRL)) continue;
                                String d = date(r.key());
                                boolean same = dates.get(r.best()).contains(d);
                                int n = 0;
                                for (String p : gal) if (dates.get(p).contains(d)) n++;
                                double exp = (double) n / gal.size();
                                boolean acc3 = r.r() <= th;
                                for (int g = 0; g < 2; g++) {
                                    if (g == 0 && j != 2) continue;
                                    acc[g][0].add(r.person(), same, exp);
                                    if (acc3) acc[g][1].add(r.person(), same, exp);
                                }
                            }
                        }
                        // Обновления после шага j (только свои; порядок – порядок строк).
                        if (!var.equals(XBaseUpdate.Var.A.label)) {
                            for (Row r : rs) {
                                if (!r.role().equals(FeiScores.OWN)) continue;
                                String to;
                                if (oracle(var)) to = r.person();
                                else if (r.r() <= th) to = r.best();
                                else continue;
                                List<String> ks = pool.get(r.person()).get(XBaseUpdate.START + j - 1);
                                if (!ks.contains(r.key())) throw new IllegalStateException("feret-session: ключ пробы не из даты шага: " + r.key());
                                tw.get(to).add(new TemplateWindow.Session(date(ks.get(0)), r.person(), ks));
                                if (j == 1) upd1[to.equals(r.person()) ? 0 : 1]++;
                            }
                        }
                    }
                }
                t.append(String.format(Locale.ROOT, "%s %s: %d / %d%n", pj.label(), var, upd1[0], upd1[1]));
                for (int g = 0; g < 2; g++) {
                    tab.append(pj.label()).append(" | ").append(var).append(" | ").append(g == 0 ? "2" : "≥ 2");
                    for (int a = 0; a < 2; a++) tab.append(" | ").append(cell(acc[g][a], ctrlP, w));
                    tab.append('\n');
                }
            }
        }
        t.append(tab);
        FarMethods.write(out.resolve("xbase_feret_session.txt"), t);
        System.out.print(t);
    }

    /** Наблюдаемое (x/n), ожидаемое, разность с интервалом по репликам весов контрольных людей. */
    static String cell(Acc a, List<String> ctrlP, int[][] w) {
        double[] p = a.total(null);
        if (p[2] == 0) return "– | – | –";
        if (p[2] < MIN_N) {
            return String.format(Locale.ROOT, "%.1f %% (%d/%d) | %.1f %% | мало проб (< %d), без интервала", 100.0 * p[0] / p[2], (int) p[0], (int) p[2],
                    100.0 * p[1] / p[2], MIN_N);
        }
        double[] reps = new double[w.length];
        for (int q = 0; q < w.length; q++) {
            Map<String, Integer> wq = new HashMap<>();
            for (int i = 0; i < ctrlP.size(); i++) wq.put(ctrlP.get(i), w[q][i]);
            double[] r = a.total(wq);
            reps[q] = r[2] == 0 ? Double.NaN : 100.0 * (r[0] - r[1]) / r[2];
        }
        double[] ok = java.util.Arrays.stream(reps).filter(x -> !Double.isNaN(x)).sorted().toArray();
        double lo = ok.length == 0 ? Double.NaN : ok[(int) Math.floor(0.025 * (ok.length - 1))];
        double hi = ok.length == 0 ? Double.NaN : ok[(int) Math.ceil(0.975 * (ok.length - 1))];
        double diff = 100.0 * (p[0] - p[1]) / p[2];
        boolean sig = lo > 0 || hi < 0;
        return String.format(Locale.ROOT, "%.1f %% (%d/%d) | %.1f %% | %+.1f [%+.1f; %+.1f]%s", 100.0 * p[0] / p[2], (int) p[0], (int) p[2], 100.0 * p[1] / p[2], diff,
                lo, hi, sig ? " *" : "");
    }
}
