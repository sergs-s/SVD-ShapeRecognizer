package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.TreeSet;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.storage.SettingsStore;

/**
 * Кривая старения (PLAN.md п. 5.5, очередь (4), часть 5; задание Жорика №4 от 16.10.2026, план №5): по готовым файлам оценок
 * FERET eq, без распознавания – растёт ли ошибка своих с интервалом «дата пробы − ближайшая дата шаблона» при фиксированной
 * ширине шаблона. Даты восстанавливаются по номеру снимка из описи (XBaseA0.feretSetup – тот же сквозной номер, что при
 * прогонах); до счёта – проверки сопоставления и структуры ролей, затем сверка общих чисел с отчётами feret-xs2 и feret-subsp
 * побайтно. Роли: «межс.» feret-xs2 (1 дата шаблона), «G2» feret-subsp k = 0 (2 даты); SFace – «межс.» файла feret (eq),
 * справочно. Режим XBaseEval feret-aging; отчёт – только D:\data\Feret\reports\eq\xbase_feret_aging.txt.
 *
 * @author ssv
 */
final class XBaseAging {

    private XBaseAging() {
    }

    /** Верхние границы корзин интервала, дней (последняя – без границы). */
    static final int[] BINS = {30, 180, 365};
    static final String[] BIN_NAMES = {"≤ 30 дн.", "31–180", "181–365", "> 365"};

    static int bin(long days) {
        for (int i = 0; i < BINS.length; i++) if (days <= BINS[i]) return i;
        return BINS.length;
    }

    static LocalDate day(String yymmdd) {
        return XBaseUpdate.day(yymmdd);
    }

    /** Интервал пробы до ближайшей даты шаблона, дней. */
    static long interval(String probeDate, Set<String> galleryDates) {
        long best = Long.MAX_VALUE;
        for (String g : galleryDates) best = Math.min(best, Math.abs(ChronoUnit.DAYS.between(day(g), day(probeDate))));
        return best;
    }

    /** Проба своего: человек (индекс), d₁/d₂, верный, нет второго, оценка истинного +∞, анфас, корзина, год пробы. */
    record Own(int p, double r, boolean ok, boolean noSecond, boolean trueInf, boolean front, int bin, int year) {}

    /** Проход: свои, контроль (человек, d₁/d₂), «чужие» П2/П3. */
    static final class Pass {
        final List<Own> own = new ArrayList<>();
        final List<double[]> ctrl = new ArrayList<>();
        final List<double[]> imp = new ArrayList<>();
        int impFrontal;
    }

    /** Роль: подпись, файл, шаблон блока, галерея и пробы по людям. */
    record Role(String name, String file, String block, Map<String, List<String>> gal, Map<String, List<String>> pro, List<String> own, boolean halves) {}

    // ---------------------------------------------------------------- запуск

    static void run(Path exportDir, List<String> selLines, SettingsStore settings) throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-aging: только равные галереи (faces.feret.gallery = eq)");
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path out = su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles ro = XBaseFeret.equalize(su.ro, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        // Номер снимка → ключ (как при прогонах: сквозной номер человека в описи).
        Map<String, String> keyOf = new HashMap<>();
        for (Map.Entry<String, Integer> e : su.imageNo.entrySet()) {
            String k = e.getKey();
            if (keyOf.put(su.byKey.get(k).person() + "\t" + e.getValue(), k) != null) throw new IllegalStateException("feret-aging: повтор номера снимка " + k);
        }
        Set<String> ctrlSet = new HashSet<>(ro.ctrl);

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4), часть 5: кривая старения по готовым файлам оценок (XBaseEval feret-aging): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(su.dir).append("; кэш детекций derived/detections.tsv – ").append(FeiThreshold.sha(su.dir.resolve(FeretData.DERIVED)
                .resolve("detections.tsv"))).append('\n');
        t.append(String.format(Locale.ROOT, """
                Схема (задание №4 от 16.10, план №5): даты проб и шаблонов восстановлены по номеру снимка из описи (FeretData.frontal, сквозной
                номер человека – как в XBaseA0.feretSetup и XBaseFeret при прогонах). Интервал – дата пробы − ближайшая дата шаблона, корзины
                %s. П3 – по «чужим» (роли cv*) файла, свой на проход; «разд.» – один порог по контрольным чужим своего прохода (FAR 1 %%) на все
                корзины. Модель по позе пробы (|r| ≤ r0 – X, иначе X+mirror). Бутстреп по людям: B = %d, seed = %d (пороги – заново в каждой
                реплике); «*» – интервал [2,5; 97,5] не содержит 0. Ближняя корзина – первая непустая, дальняя – последняя непустая; парно –
                только люди с пробами и в ближней, и в дальней; тренд – наклон ошибки по номеру корзины (МНК по пробам), п. п. на корзину.
                Проба в ролях «межс.» и «G2» – одиночный снимок (не пара fa/fb): проверка «пары одной даты» заменена на «проба – с не галерейной
                даты своего человека».
                """, String.join(" / ", BIN_NAMES), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        List<Role> roles = new ArrayList<>();
        for (String pj : new String[] {"L1-xs-F", "L1-tt-xs-F", "L1", "L1-tt"}) {
            roles.add(new Role("межс. " + pj, "xbase_feret_xs2_scores.tsv.gz", "проход %d, конфигурация межс., строка " + pj, ro.gal1, ro.pro1, ro.own1, true));
        }
        for (String pj : new String[] {"L1-xs-F", "L1-tt-xs-F", "L1"}) {
            roles.add(new Role("G2 " + pj + " k = 0", "xbase_feret_subsp_scores.tsv.gz", "G2, проекция " + pj + ", проход %d, k = 0", ro.g2, ro.pro2, ro.own2, true));
        }
        roles.add(new Role("межс. SFace (feret, без половин, справочно)", "xbase_feret_scores.tsv.gz", "конфигурация межс., набор SFace", ro.gal1, ro.pro1, ro.own1,
                false));

        StringBuilder checks = new StringBuilder("\n=== Проверки сопоставления и структуры ролей (до счёта; при любом промахе – остановка) ===\n");
        StringBuilder check3 = new StringBuilder("\n=== Сверка: общие числа (все корзины вместе) = строка сводки отчёта прогона (побайтно) ===\n");
        StringBuilder body = new StringBuilder();
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, Map<String, String>> fileBlocks = new HashMap<>();
        for (Role rl : roles) {
            Path f = out.resolve(rl.file());
            if (!Files.exists(f)) {
                checks.append(rl.name()).append(": файла ").append(f.getFileName()).append(" нет – строка пропущена\n");
                continue;
            }
            Map<String, String> blocks = fileBlocks.computeIfAbsent(rl.file(), z -> {
                try {
                    return XBaseSubsp.blocks(f);
                } catch (IOException e) {
                    throw new IllegalStateException(e);
                }
            });
            List<Pass> passes = new ArrayList<>();
            List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
            List<FeiThreshold.Imp> imps = new ArrayList<>();
            int[] cnt = new int[4]; // строк сопоставлено: свои, контроль, «чужие», всего
            int nPass = rl.halves() ? 2 : 1;
            for (int pp = 0; pp < nPass; pp++) {
                String bn = rl.halves() ? String.format(Locale.ROOT, rl.block(), pp + 1) : rl.block();
                String ls = blocks.get(bn);
                if (ls == null) throw new IllegalStateException("feret-aging: нет блока «" + bn + "» в " + rl.file());
                Set<String> mine = rl.halves() ? halves.get(pp) : null;
                List<String> own = rl.own().stream().filter(p -> mine == null || mine.contains(p)).toList();
                Pass ps = pass(ls, rl, own, keyOf, su, ctrlSet, pIdx, tpl.r0, cnt);
                passes.add(ps);
                cfgs.add(cfg(ps));
                imps.add(FeiThreshold.Imp.of(new ArrayList<>(ps.imp), ps.impFrontal));
            }
            checks.append(String.format(Locale.ROOT, "%s: сопоставлено строк %d (свои %d, контроль %d, «чужие» %d); все проверки прошли%n", rl.name(), cnt[3], cnt[0],
                    cnt[1], cnt[2]));
            // Сверка общих чисел со строкой сводки отчёта прогона.
            String reportFile = rl.file().replace("_scores.tsv.gz", ".txt");
            Path rf = out.resolve(reportFile);
            FeiThreshold.Res res = FeiThreshold.evaluate(rl.name(), XBaseEval.V.L1.ratioId(), cfgs, imps, persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
            if (rl.halves()) {
                Map<String, FeiThreshold.Res> m = new LinkedHashMap<>();
                String label = rl.name();
                m.put(label, res);
                Map<String, XBaseOther.Sep> sep = Map.of(label, XBaseOther.sep(res.cfgs, FeiBootstrap.unit(persons.size()), w));
                StringBuilder tt = new StringBuilder();
                XBaseOther.tables(tt, m, sep, tpl.b, new ArrayList<>());
                String mine2 = tt.toString().lines().filter(l -> l.startsWith(label + " | ")).findFirst().orElse(null);
                String old = Files.readAllLines(rf).stream().map(l -> l.replace("\r", "")).filter(l -> l.startsWith(label + " | ")).filter(l -> !l.startsWith(label
                        + " | П")).filter(l -> l.contains("(") && l.contains("%")).findFirst().orElse(null);
                boolean same = mine2 != null && mine2.equals(old);
                check3.append(String.format(Locale.ROOT, "%s: %s%n", label, same ? "совпало (" + reportFile + ")" : "НЕ СОВПАЛО"));
                if (!same) throw new IllegalStateException("feret-aging: сводка не совпала с " + reportFile + ":\nбыло  " + old + "\nсейчас " + mine2);
            } else {
                check3.append(rl.name()).append(": сверка не делалась (справочная строка, другой протокол)\n");
            }
            body.append(tables(rl.name(), passes, persons.size(), w));
        }
        t.append(checks).append(check3).append(body);
        FarMethods.write(out.resolve("xbase_feret_aging.txt"), t);
        System.out.print(t);
    }

    /** Строки блока → проход; проверки сопоставления и структуры (любой промах – исключение). */
    static Pass pass(String lines, Role rl, List<String> own, Map<String, String> keyOf, XBaseA0.FeretSetup su, Set<String> ctrlSet, Map<String, Integer> pIdx,
                     double r0, int[] cnt) {
        Set<String> ownSet = new HashSet<>(own);
        // Структура галереи: межс. – одна, самая ранняя дата; G2 – две даты; на дате не больше одного fa и одного fb.
        for (String p : own) {
            List<String> g = rl.gal().get(p);
            Map<String, List<String>> byDate = new TreeMap<>();
            for (String k : g) byDate.computeIfAbsent(su.byKey.get(k).date(), d -> new ArrayList<>()).add(su.byKey.get(k).pose());
            int need = rl.name().startsWith("G2") ? 2 : 1;
            if (byDate.size() != need) throw new IllegalStateException("feret-aging: " + rl.name() + ", " + p + " – дат галереи " + byDate.size() + " вместо " + need);
            for (List<String> poses : byDate.values()) {
                if (poses.stream().filter("fa"::equals).count() > 1 || poses.stream().filter("fb"::equals).count() > 1 || poses.size() > 2) {
                    throw new IllegalStateException("feret-aging: " + rl.name() + ", " + p + " – на дате галереи не fa / fb: " + poses);
                }
            }
            TreeSet<String> allDates = new TreeSet<>();
            // Даты отобранных снимков (лицо найдено, |r| ≤ R) – как в XBaseFeret.roles.
            for (Img im : su.imgs) if (im.person().equals(p) && su.rOf.containsKey(im.key()) && su.rOf.get(im.key()) <= su.tpl.rLimit) allDates.add(im.date());
            List<String> gd = new ArrayList<>(byDate.keySet());
            List<String> first = new ArrayList<>(allDates).subList(0, Math.min(need, allDates.size()));
            if (!gd.equals(first)) throw new IllegalStateException("feret-aging: " + rl.name() + ", " + p + " – даты галереи " + gd + " не самые ранние " + first);
        }
        Map<String, Row[]> pairs = new LinkedHashMap<>();
        for (String l : lines.split("\n")) {
            if (l.isEmpty()) continue;
            Row r = FeiScores.parse(l);
            String key = keyOf.get(r.person() + "\t" + r.image());
            if (key == null) throw new IllegalStateException("feret-aging: нет снимка для " + r.person() + " №" + r.image());
            boolean mir = r.method().endsWith(FeiMethods.MIRROR);
            Row[] pr = pairs.computeIfAbsent(r.role() + "\t" + r.person() + "\t" + r.image(), z -> new Row[2]);
            if (pr[mir ? 1 : 0] != null) throw new IllegalStateException("feret-aging: повтор строки " + l);
            pr[mir ? 1 : 0] = r;
        }
        Pass ps = new Pass();
        for (Map.Entry<String, Row[]> e : pairs.entrySet()) {
            Row[] pr = e.getValue();
            Row any = pr[0] != null ? pr[0] : pr[1];
            String key = keyOf.get(any.person() + "\t" + any.image());
            Img im = su.byKey.get(key);
            boolean front = su.rOf.get(key) <= r0;
            Row r = pr[1] == null ? pr[0] : pr[0] == null ? pr[1] : front ? pr[0] : pr[1];
            String role = any.role();
            if (role.equals(FeiScores.OWN)) {
                String p = any.person();
                if (!ownSet.contains(p)) throw new IllegalStateException("feret-aging: проба своего " + p + " вне своих прохода (" + rl.name() + ")");
                if (!rl.pro().get(p).contains(key)) throw new IllegalStateException("feret-aging: проба " + key + " не из проб " + p + " (" + rl.name() + ")");
                Set<String> gd = new HashSet<>();
                for (String k : rl.gal().get(p)) gd.add(su.byKey.get(k).date());
                if (gd.contains(im.date())) throw new IllegalStateException("feret-aging: проба " + key + " – с даты галереи (" + rl.name() + ")");
                long days = interval(im.date(), gd);
                int[] cs = new int[FeiScores.CASES];
                ps.own.add(new Own(pIdx.get(p), FeiScores.ratio(r, cs), r.bestId().equals(r.trueId()), cs[FeiScores.CASE_NO_SECOND] > 0, Double.isInfinite(r.trueScore()),
                        front, bin(days), 1900 + Integer.parseInt(im.date().substring(0, 2))));
                cnt[0]++;
            } else if (role.equals(FeiScores.CTRL)) {
                if (!ctrlSet.contains(any.person())) throw new IllegalStateException("feret-aging: контрольный " + any.person() + " не из контроля");
                ps.ctrl.add(new double[] {pIdx.get(any.person()), FeiScores.ratio(r, null)});
                cnt[1]++;
            } else if (role.startsWith(FeiMethods.CV_ROLE)) {
                String p = any.person();
                if (!rl.gal().get(p).contains(key)) throw new IllegalStateException("feret-aging: «чужой» " + key + " не из шаблона " + p + " (" + rl.name() + ")");
                ps.imp.add(new double[] {pIdx.get(p), FeiScores.ratio(r, null)});
                if (front) ps.impFrontal++;
                cnt[2]++;
            } else {
                throw new IllegalStateException("feret-aging: роль " + role);
            }
            cnt[3] += (pr[0] != null ? 1 : 0) + (pr[1] != null ? 1 : 0);
        }
        return ps;
    }

    static FeiDecompose.Cfg cfg(Pass ps) {
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        int n = ps.own.size();
        c.ownP = new int[n];
        c.ownR = new double[n];
        c.ownOk = new boolean[n];
        c.noSecond = new boolean[n];
        c.trueInf = new boolean[n];
        c.poseFrontal = new boolean[n];
        for (int i = 0; i < n; i++) {
            Own o = ps.own.get(i);
            c.ownP[i] = o.p();
            c.ownR[i] = o.r();
            c.ownOk[i] = o.ok();
            c.noSecond[i] = o.noSecond();
            c.trueInf[i] = o.trueInf();
            c.poseFrontal[i] = o.front();
        }
        c.thrP = new int[0];
        c.thrR = new double[0];
        c.ctrlP = ps.ctrl.stream().mapToInt(d -> (int) d[0]).toArray();
        c.ctrlR = ps.ctrl.stream().mapToDouble(d -> d[1]).toArray();
        return c;
    }

    // ---------------------------------------------------------------- меры по корзинам

    /** Меры при весах w: [порог][корзина] {ошибок, проб}; порог 0 – П3, 1 – разд. */
    static double[][][] counts(List<Pass> passes, int[] w, Set<Integer> only) {
        double[][][] c = new double[2][BIN_NAMES.length][2];
        for (Pass ps : passes) {
            List<double[]> imp = new ArrayList<>(ps.imp);
            imp.sort((a, b) -> Double.compare(a[1], b[1]));
            double th3 = FeiThreshold.margin(imp.stream().mapToDouble(a -> a[1]).toArray(), imp.stream().mapToInt(a -> (int) a[0]).toArray(), w, XBaseUpdate.FAR).theta();
            List<double[]> ct = new ArrayList<>(ps.ctrl);
            ct.sort((a, b) -> Double.compare(a[1], b[1]));
            double ths = FeiBootstrap.threshold(ct.stream().mapToDouble(a -> a[1]).toArray(), ct.stream().mapToInt(a -> (int) a[0]).toArray(), w, XBaseUpdate.FAR);
            for (Own o : ps.own) {
                if (only != null && !only.contains(o.p())) continue;
                int q = w[o.p()];
                if (q == 0) continue;
                double[] th = {th3, ths};
                for (int k = 0; k < 2; k++) {
                    c[k][o.bin()][1] += q;
                    if (!(o.ok() && o.r() <= th[k])) c[k][o.bin()][0] += q;
                }
            }
        }
        return c;
    }

    static double rate(double[] a) {
        return a[1] == 0 ? Double.NaN : 100 * a[0] / a[1];
    }

    /** Наклон ошибки по номеру корзины (МНК по пробам), п. п. на корзину. */
    static double slope(double[][] c) {
        double n = 0;
        double sx = 0;
        double sy = 0;
        double sxx = 0;
        double sxy = 0;
        for (int b = 0; b < c.length; b++) {
            n += c[b][1];
            sx += b * c[b][1];
            sxx += (double) b * b * c[b][1];
            sy += c[b][0];
            sxy += b * c[b][0];
        }
        double d = n * sxx - sx * sx;
        return d == 0 ? Double.NaN : 100 * (n * sxy - sx * sy) / d;
    }

    static String ci(double point, double[] reps) {
        double[] ok = Arrays.stream(reps).filter(v -> !Double.isNaN(v)).toArray();
        if (Double.isNaN(point)) return "–";
        if (ok.length == 0) return String.format(Locale.ROOT, "%+.1f", point);
        double[] s = FeiBootstrap.sorted(ok);
        double lo = FeiBootstrap.quantile(s, 0.025);
        double hi = FeiBootstrap.quantile(s, 0.975);
        return String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", FeiBootstrap.nz(point, "%+.1f"), FeiBootstrap.nz(lo, "%+.1f"), FeiBootstrap.nz(hi, "%+.1f"),
                FeiBootstrap.sig(lo, hi) ? " *" : "");
    }

    static String tables(String name, List<Pass> passes, int persons, int[][] w) {
        StringBuilder t = new StringBuilder("\n\n######## " + name + " ########\n");
        int[] unit = FeiBootstrap.unit(persons);
        double[][][] c = counts(passes, unit, null);
        // Люди и пробы по корзинам.
        List<Set<Integer>> people = new ArrayList<>();
        for (int b = 0; b < BIN_NAMES.length; b++) people.add(new TreeSet<>());
        for (Pass ps : passes) for (Own o : ps.own) people.get(o.bin()).add(o.p());
        int near = -1;
        int far = -1;
        for (int b = 0; b < BIN_NAMES.length; b++) {
            if (c[0][b][1] == 0) continue;
            if (near < 0) near = b;
            far = b;
        }
        Set<Integer> both = new TreeSet<>(people.get(Math.max(near, 0)));
        both.retainAll(people.get(Math.max(far, 0)));
        double[][][] cb = counts(passes, unit, both);
        // Реплики.
        double[][][] reps = new double[2][BIN_NAMES.length][w.length];
        double[][] dFar = new double[2][w.length];
        double[][] dPair = new double[2][w.length];
        double[][] sl = new double[2][w.length];
        int nf = near;
        int ff = far;
        java.util.stream.IntStream.range(0, w.length).parallel().forEach(q -> {
            double[][][] cq = counts(passes, w[q], null);
            double[][][] bq = counts(passes, w[q], both);
            for (int k = 0; k < 2; k++) {
                for (int b = 0; b < BIN_NAMES.length; b++) reps[k][b][q] = rate(cq[k][b]);
                dFar[k][q] = nf < 0 ? Double.NaN : rate(cq[k][ff]) - rate(cq[k][nf]);
                dPair[k][q] = nf < 0 ? Double.NaN : rate(bq[k][ff]) - rate(bq[k][nf]);
                sl[k][q] = slope(cq[k]);
            }
        });
        String[] th = {"П3", "разд. (FAR контроля 1 %)"};
        t.append("\n=== Ошибка своих по корзинам интервала (точка [95 %]) ===\nкорзина | людей | проб | ошибка при П3 | при пороге по контролю\n");
        for (int b = 0; b < BIN_NAMES.length; b++) {
            t.append(String.format(Locale.ROOT, "%s | %d | %d | %s | %s%n", BIN_NAMES[b], people.get(b).size(), (int) c[0][b][1], cell(c[0][b], reps[0][b]), cell(c[1][b],
                    reps[1][b])));
        }
        if (near < 0) return t.toString();
        t.append(String.format(Locale.ROOT, "%n=== Сравнения (ближняя – %s, дальняя – %s; парно – %d людей с пробами в обеих) ===%n", BIN_NAMES[near], BIN_NAMES[far],
                both.size()));
        t.append("порог | дальняя − ближняя, п. п. | то же парно (только люди в обеих) | тренд, п. п. на корзину\n");
        for (int k = 0; k < 2; k++) {
            t.append(String.format(Locale.ROOT, "%s | %s | %s (ближняя %.1f %%, дальняя %.1f %%) | %s%n", th[k], ci(rate(c[k][far]) - rate(c[k][near]), dFar[k]),
                    ci(rate(cb[k][far]) - rate(cb[k][near]), dPair[k]), rate(cb[k][near]), rate(cb[k][far]), ci(slope(c[k]), sl[k])));
        }
        // Год пробы × корзина (точка, без интервала).
        Map<Integer, double[][]> byYear = new TreeMap<>();
        double[] th3 = new double[passes.size()];
        double[] ths = new double[passes.size()];
        for (int i = 0; i < passes.size(); i++) {
            Pass ps = passes.get(i);
            List<double[]> imp = new ArrayList<>(ps.imp);
            imp.sort((a, b) -> Double.compare(a[1], b[1]));
            th3[i] = FeiThreshold.margin(imp.stream().mapToDouble(a -> a[1]).toArray(), imp.stream().mapToInt(a -> (int) a[0]).toArray(), unit, XBaseUpdate.FAR).theta();
            List<double[]> ct = new ArrayList<>(ps.ctrl);
            ct.sort((a, b) -> Double.compare(a[1], b[1]));
            ths[i] = FeiBootstrap.threshold(ct.stream().mapToDouble(a -> a[1]).toArray(), ct.stream().mapToInt(a -> (int) a[0]).toArray(), unit, XBaseUpdate.FAR);
            for (Own o : ps.own) {
                double[][] y = byYear.computeIfAbsent(o.year(), z -> new double[BIN_NAMES.length][3]);
                y[o.bin()][2]++;
                if (!(o.ok() && o.r() <= th3[i])) y[o.bin()][0]++;
                if (!(o.ok() && o.r() <= ths[i])) y[o.bin()][1]++;
            }
        }
        t.append("\n=== Диагностика смешения: год пробы × корзина (ошибка при П3 / при пороге по контролю, проб; точка) ===\nгод пробы | ")
                .append(String.join(" | ", BIN_NAMES)).append('\n');
        for (Map.Entry<Integer, double[][]> e : byYear.entrySet()) {
            List<String> row = new ArrayList<>(List.of(Integer.toString(e.getKey())));
            for (double[] v : e.getValue()) row.add(v[2] == 0 ? "–" : String.format(Locale.ROOT, "%.0f %% / %.0f %% (%d)", 100 * v[0] / v[2], 100 * v[1] / v[2], (int) v[2]));
            t.append(String.join(" | ", row)).append('\n');
        }
        return t.toString();
    }

    static String cell(double[] a, double[] reps) {
        if (a[1] == 0) return "–";
        double[] ok = Arrays.stream(reps).filter(v -> !Double.isNaN(v)).toArray();
        String p = String.format(Locale.ROOT, "%d/%d = %.1f %%", (int) a[0], (int) a[1], rate(a));
        if (ok.length == 0) return p;
        double[] s = FeiBootstrap.sorted(ok);
        return p + String.format(Locale.ROOT, " [%.1f; %.1f]", FeiBootstrap.quantile(s, 0.025), FeiBootstrap.quantile(s, 0.975));
    }
}
