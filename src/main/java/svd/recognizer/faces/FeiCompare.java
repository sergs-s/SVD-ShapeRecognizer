package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.storage.SettingsStore;

/**
 * PLAN.md, п. 4.2, «LBP-гистограммы»: сравнение лидера A = 2c-mlda-ratio+pose с L0 = 5-lbp-chi-ratio+pose и
 * L1 = 5-lbp-mlda-ratio+pose по полным векторам расстояний (FeiVectors) в протоколе fei_scores.txt: у каждой пробы вектор
 * модели X или X+mirror по позе (|r| ≤ r0), оценка d₁/d₂ (FeiScores.ratio), порог Нейман – Пирсон по снимкам 40
 * пороговых, бутстреп по людям FEI целиком (одни веса для всех строк — разности парные). Контроль векторов каждой части
 * — по её файлу оценок (FeiNorm.load: лучший, второй, d₁, d₂ и ошибки базы).
 *
 * Запуск: {@code mvn compile exec:java@faces-fei-compare "-Dexec.args=<векторы A> <оценки A> <векторы LBP> <оценки LBP>"} при
 * заданных faces.fei.pose.r0 и faces.export.dir; бутстреп — faces.fei.bootstrap. Отчёт reports/faces/fei/fei_lbp.txt.
 * Режим mask (п. 4.2, «LBP: внутренние и краевые блоки»): {@code "-Dexec.args=mask <векторы A> <оценки A> <векторы L> <оценки L>
 * <векторы L1 прежние>"} — A против L1, L1-in, L1-out (FeiMethods, faces.fei.lbp = mask,only), отчёт fei_lbp_mask.txt.
 *
 * @author ssv
 */
public final class FeiCompare {

    static final String FILE = "fei_lbp.txt";
    static final String COMMAND = "FACES_FEI_MIRROR=true FACES_FEI_LBP=true FACES_FEI_VECTORS=2c-mlda,5-lbp-chi,5-lbp-mlda "
            + "FACES_EXPORT_DIR=<экспорт> java … svd.recognizer.faces.FeiMethods (как exec:exec@faces-fei)";

    private FeiCompare() {
    }

    /** Попытки своих части: «s j человек» → {верно при FAR 1 %, верно при FAR 5 %, верный argmin}. */
    @SuppressWarnings("unchecked")
    static Map<String, boolean[]> own(FeiNorm.Part p) {
        List<FeiDecompose.Cfg> dc = (List<FeiDecompose.Cfg>) FeiNorm.cfgs(p.confs, FeiNorm.V_BASE, null)[0];
        Map<String, boolean[]> out = new LinkedHashMap<>();
        for (int c = 0; c < dc.size(); c++) {
            FeiNorm.Conf conf = p.confs.get(c);
            FeiDecompose.Cfg d = dc.get(c);
            double[] th = {FeiDecompose.threshold(d, null, 0), FeiDecompose.threshold(d, null, 1)};
            List<FeiNorm.Att> own = conf.all.stream().filter(a -> a.role.equals(FeiScores.OWN)).toList();
            for (int q = 0; q < own.size(); q++) {
                FeiNorm.Att a = own.get(q);
                boolean[] ok = {FeiDecompose.group(d, q, th[0]) == FeiDecompose.G_OK, FeiDecompose.group(d, q, th[1]) == FeiDecompose.G_OK,
                    a.cand == a.t};
                if (out.put(conf.s + " " + conf.j + " " + a.person, ok) != null) throw new IllegalStateException("Повтор попытки своего " + a.person);
            }
        }
        return out;
    }

    /** Совпадение ошибок пары: [порог 0, 1, argmin 2] → {оба верно, только первый, только второй, оба ошиблись}. */
    static int[][] agreement(Map<String, boolean[]> x, Map<String, boolean[]> y) {
        if (!x.keySet().equals(y.keySet())) throw new IllegalStateException("Разные попытки своих у частей");
        int[][] t = new int[3][4];
        for (Map.Entry<String, boolean[]> e : x.entrySet()) {
            boolean[] b = y.get(e.getKey());
            for (int k = 0; k < 3; k++) {
                boolean cx = e.getValue()[k];
                boolean cy = b[k];
                t[k][cx ? (cy ? 0 : 1) : (cy ? 2 : 3)]++;
            }
        }
        return t;
    }

    public static void main(String[] args) throws IOException {
        String[] a = args.length == 1 ? args[0].trim().split("\\s+") : args;
        if (a.length > 0 && a[0].equals("mask")) {
            maskMain(Arrays.copyOfRange(a, 1, a.length));
            return;
        }
        if (a.length < 4) throw new IllegalArgumentException("faces-fei-compare: -Dexec.args=\"<векторы A> <оценки A> <векторы LBP> <оценки LBP>\"");
        SettingsStore settings = new SettingsStore();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("faces-fei-compare: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        FeiPose sel = FeiScores.selection(settings);
        List<String> people = sel.persons();
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < people.size(); i++) idx.put(people.get(i), i);
        int b = settings.loadFacesFeiBootstrap();
        long seed = settings.loadFacesSeed() + FeiBootstrap.SEED_OFFSET;
        long t0 = System.nanoTime();
        String[][] parts = {{"2c-mlda", "2c-mlda-ratio+pose", "A"}, {FeiMethods.LBP_CHI, "5-lbp-chi-ratio+pose", "L0"},
            {FeiMethods.LBP_MLDA, "5-lbp-mlda-ratio+pose", "L1"}};
        List<FeiNorm.Part> ps = new ArrayList<>();
        Map<String, List<Row>> groupsA = new HashMap<>();
        Map<String, List<Row>> groupsL = new HashMap<>();
        for (String[] p : parts) {
            boolean isA = p[2].equals("A");
            List<String> methods = List.of(p[0], p[0] + FeiMethods.MIRROR);
            Map<String, List<Row>> groups = isA ? groupsA : groupsL;
            Map<String, Row> ref = FeiNorm.reference(Paths.get(isA ? a[1] : a[3]), methods, groups);
            FeiNorm.Part part = FeiNorm.load(p[0], p[1], p[2], Paths.get(isA ? a[0] : a[2]), ref, groups, sel, r0, idx);
            FeiNorm.evaluate(part, people.size(), b, seed, 1);
            ps.add(part);
            groups.clear();
        }
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(out);
        FarMethods.write(out.resolve(FILE), report(ps, a, r0, b, seed, people.size()));
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "Сравнение LBP: %.0f с; отчёт %s",
                (System.nanoTime() - t0) / 1e9, out.resolve(FILE)));
    }

    static StringBuilder report(List<FeiNorm.Part> ps, String[] a, double r0, int b, long seed, int persons) {
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: LBP-гистограммы как вход — сравнение с лидером A (FeiCompare, faces-fei-compare)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Векторы A: ").append(a[0]).append(" (вне git; получены: ").append(FeiNorm.COMMAND).append("); контроль по ").append(a[1]).append('\n');
        t.append("Векторы L0, L1: ").append(a[2]).append(" (вне git; получены: ").append(COMMAND).append("); контроль по ").append(a[3]).append('\n');
        for (FeiNorm.Part p : ps) for (String f : p.files) t.append("  ").append(p.label).append(": ").append(f).append('\n');
        t.append("Строки: A = 2c-mlda-ratio+pose; L0 = 5-lbp-chi-ratio+pose (LBP u2 (8, 2), 7×7, кадр а без нормализации освещения, χ²,\n"
                + "ближайший снимок человека); L1 = 5-lbp-mlda-ratio+pose (корень гистограмм + MLDA, как A). Выбор модели по позе: r0 = ")
                .append(r0).append(" (|r| пробы ≤ r0 — X, иначе X+mirror); оценка d₁/d₂ (FeiScores.ratio); порог — Нейман – Пирсон\n"
                + "по снимкам 40 пороговых (FAR 1 % и 5 %), как в fei_scores.txt. Группы (i) верный принят, (ii) верный отказ, (iii)\n"
                + "неверный отказ, (iv) неверный принят — как в fei_decompose.txt; анфас/поворот — по |r| ≤ r0.\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям FEI целиком: B = %d, seed = %d, людей %d, одни веса для всех строк (как "
                    + "fei_bootstrap.txt); порог — заново в\nкаждой реплике и конфигурации. [2,5; 97,5] — процентили; «*» — интервал разности не "
                    + "содержит 0.\n", b, seed, persons));
        }
        t.append("\n=== Контроль ===\n");
        control(t, ps);
        tables(t, ps, b, new int[][] {{1, 0}, {2, 0}, {2, 1}}, new String[] {"", "", " (справочно)"});

        t.append("\n=== Совпадение ошибок своих (верно — принят под своим именем; argmin — лучший кандидат истинный, без порога) ===\n");
        Map<String, boolean[]> oa = own(ps.get(0));
        String[] what = {"FAR 1 %", "FAR 5 %", "argmin"};
        t.append("пара, порог | оба верно | верно только первый | верно только второй | оба ошиблись\n");
        for (int i = 1; i < ps.size(); i++) {
            int[][] g = agreement(oa, own(ps.get(i)));
            int n = oa.size();
            for (int k = 0; k < 3; k++) {
                t.append(String.join(" | ", "A–" + ps.get(i).label + ", " + what[k], FeiMethods.pctOf(g[k][0], n), FeiMethods.pctOf(g[k][1], n),
                        FeiMethods.pctOf(g[k][2], n), FeiMethods.pctOf(g[k][3], n))).append('\n');
            }
        }
        return t;
    }

    /** Строки контроля частей: векторы против файла оценок, правило d₁/d₂. */
    static void control(StringBuilder t, List<FeiNorm.Part> ps) {
        for (FeiNorm.Part p : ps) {
            t.append(String.format(Locale.ROOT, "%s: векторы X и X+mirror — %d и %d попыток, лучший, второй, d₁, d₂ — ровно как в файле оценок; "
                    + "база — те же ошибки своих и FAR контроля, что FeiScores.Stat %s; правило d₁/d₂: ", p.label, p.checked[0], p.checked[1], p.id));
            for (int k = 0; k < FeiScores.CASES; k++) t.append(k > 0 ? ", " : "").append(k + 1).append(": ").append(p.rCases[k]);
            t.append('\n');
        }
    }

    /**
     * Таблицы: ошибка своих и FAR контроля, разложение (i)–(iv), парные разности pairs (индексы частей; к подписи пары —
     * notes), ранг истинного у неверного кандидата.
     */
    static void tables(StringBuilder t, List<FeiNorm.Part> ps, int b, int[][] pairs, String[] notes) {
        t.append("\n=== Ошибка своих и FAR контроля ===\n");
        t.append("строка | ошибка своих FAR 1 % | FAR 5 % | анфас FAR 1 % | поворот FAR 1 % | анфас FAR 5 % | поворот FAR 5 % | "
                + "FAR контроля снимки 1 %: точка ↑95; КП | 5 %\n");
        for (FeiNorm.Part p : ps) {
            List<String> c = new ArrayList<>(List.of(p.label + " = " + p.id));
            for (int[] ks : new int[][] {{0, 0}, {1, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}}) {
                double[][][] cnt = p.pointCounts[0];
                double n = Arrays.stream(cnt[ks[0]][ks[1]]).sum();
                c.add(FeiNorm.cell(n - cnt[ks[0]][ks[1]][FeiDecompose.G_OK], n, b > 0 ? FeiNorm.errReps(p, 0, ks[0], ks[1]) : null));
            }
            for (int k = 0; k < 2; k++) {
                int m = k == 0 ? FeiBootstrap.FAR1 : FeiBootstrap.FAR5;
                String s = String.format(Locale.ROOT, "%.2f %%", p.pointFar[0][m]);
                if (b > 0) s += String.format(Locale.ROOT, " ↑%.2f %%", FeiBootstrap.quantile(FeiBootstrap.sorted(p.repFar[0][m]), 0.95));
                int[] xn = p.cp[0][k];
                s += String.format(Locale.ROOT, "; КП %d/%d ↑%.2f %%", xn[0], xn[1], 100 * FaceEvaluation.binomialUpperBound(xn[0], xn[1], 0.95));
                c.add(s);
            }
            t.append(String.join(" | ", c)).append('\n');
        }

        String[] farName = {"FAR 1 %", "FAR 5 %"};
        t.append("\n=== Разложение ошибки своих ===\n");
        for (int k = 0; k < 2; k++) {
            t.append("-- ").append(farName[k]).append('\n');
            t.append("строка | деление | (i) | (ii) | (iii) | (iv) | (v-а)\n");
            for (FeiNorm.Part p : ps) {
                for (int sp = 0; sp < 3; sp++) {
                    double[][][] cnt = p.pointCounts[0];
                    double n = Arrays.stream(cnt[k][sp]).sum();
                    List<String> c = new ArrayList<>(List.of(p.label, FeiDecompose.SPLIT_NAMES[sp]));
                    for (int g = 0; g <= FeiDecompose.G_NO_SECOND; g++) {
                        double[] r = null;
                        if (b > 0) {
                            r = new double[b];
                            for (int q = 0; q < b; q++) r[q] = p.repShares[0][q][k][sp][g];
                        }
                        c.add(FeiNorm.cell(cnt[k][sp][g], n, r));
                    }
                    t.append(String.join(" | ", c)).append('\n');
                }
            }
        }

        if (b > 0) {
            t.append("\n=== Парные разности ошибки своих, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("пара | FAR 1 % всего | анфас | поворот | FAR 5 % всего | анфас | поворот | FAR контроля снимки 1 %\n");
            for (int q = 0; q < pairs.length; q++) {
                int[] pr = pairs[q];
                FeiNorm.Part x = ps.get(pr[0]);
                FeiNorm.Part y = ps.get(pr[1]);
                List<String> c = new ArrayList<>(List.of(x.label + " − " + y.label + notes[q]));
                for (int[] ks : new int[][] {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}) {
                    double pt = FeiNorm.err(FeiDecompose.shares(x.pointCounts[0]), ks[0], ks[1]) - FeiNorm.err(FeiDecompose.shares(y.pointCounts[0]), ks[0], ks[1]);
                    c.add(FeiBootstrap.diff(pt, FeiNorm.diff(FeiNorm.errReps(x, 0, ks[0], ks[1]), FeiNorm.errReps(y, 0, ks[0], ks[1])), 1));
                }
                c.add(FeiBootstrap.diff(x.pointFar[0][FeiBootstrap.FAR1] - y.pointFar[0][FeiBootstrap.FAR1],
                        FeiNorm.diff(x.repFar[0][FeiBootstrap.FAR1], y.repFar[0][FeiBootstrap.FAR1]), 2));
                t.append(String.join(" | ", c)).append('\n');
            }
        }

        t.append("\n=== Ранг истинного в сыром векторе (выбранном по позе) у неверного кандидата: 2 / 3–5 / >5 ===\n");
        t.append("строка | все неверные argmin | группа (iii) FAR 1 % | группа (iii) FAR 5 %\n");
        for (FeiNorm.Part p : ps) {
            int n = p.rankWrong[0] + p.rankWrong[1] + p.rankWrong[2];
            int[] r1 = p.rankIii[0][0];
            int[] r5 = p.rankIii[0][1];
            t.append(String.join(" | ", p.label, FeiNorm.bins(p.rankWrong, n), FeiNorm.bins(r1, r1[0] + r1[1] + r1[2]),
                    FeiNorm.bins(r5, r5[0] + r5[1] + r5[2]))).append('\n');
        }
    }

    // ---------------------------------------------------------------- маски LBP

    /** Отчёт режима mask (п. 4.2, «LBP: внутренние и краевые блоки»). */
    static final String MASK_FILE = "fei_lbp_mask.txt";
    static final String MASK_COMMAND = "FACES_FEI_MIRROR=true FACES_FEI_LBP=mask,only FACES_FEI_VECTORS=5-lbp-mlda,5-lbp-mlda-in,5-lbp-mlda-out "
            + "FACES_EXPORT_DIR=<экспорт> java … svd.recognizer.faces.FeiMethods (как exec:exec@faces-fei)";

    /**
     * Данные (без шапки) двух файлов векторов совпадают: строки попыток и галереи разбиений; иначе исключение. Возвращает
     * число строк попыток.
     */
    static long sameContent(Path x, Path y) throws IOException {
        List<String> a = content(x);
        List<String> b = content(y);
        if (!a.equals(b)) {
            int i = 0;
            while (i < Math.min(a.size(), b.size()) && a.get(i).equals(b.get(i))) i++;
            throw new IllegalStateException("Контроль L1: векторы " + x + " и " + y + " различаются (строк " + a.size() + " и " + b.size()
                    + ", первое различие — строка данных " + i + ")");
        }
        return a.stream().filter(l -> !l.startsWith("#")).count();
    }

    /** Строки данных файла векторов: галереи разбиений и попытки, без прочей шапки. */
    static List<String> content(Path file) throws IOException {
        List<String> out = new ArrayList<>();
        try (java.io.BufferedReader in = new java.io.BufferedReader(new java.io.InputStreamReader(
                new java.util.zip.GZIPInputStream(Files.newInputStream(file), 1 << 16), StandardCharsets.UTF_8), 1 << 16)) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") && !l.startsWith(FeiVectors.GALLERY) || l.isEmpty() || l.equals(FeiVectors.COLUMNS)) continue;
                out.add(l);
            }
        }
        return out;
    }

    /**
     * Режим mask: A (векторы и оценки прежние), L1, L1-in, L1-out (векторы и оценки прогона mask,only); контроль — векторы L1
     * этого прогона = прежние (каталог oldL1) по содержимому.
     */
    static void maskMain(String[] a) throws IOException {
        if (a.length < 5) {
            throw new IllegalArgumentException("faces-fei-compare mask: -Dexec.args=\"mask <векторы A> <оценки A> <векторы L> <оценки L> "
                    + "<векторы L1 прежние>\"");
        }
        SettingsStore settings = new SettingsStore();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("faces-fei-compare: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        FeiPose sel = FeiScores.selection(settings);
        List<String> people = sel.persons();
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < people.size(); i++) idx.put(people.get(i), i);
        int b = settings.loadFacesFeiBootstrap();
        long seed = settings.loadFacesSeed() + FeiBootstrap.SEED_OFFSET;
        long t0 = System.nanoTime();
        List<String> sameL1 = new ArrayList<>();
        for (String m : new String[] {FeiMethods.LBP_MLDA, FeiMethods.LBP_MLDA + FeiMethods.MIRROR}) {
            Path x = Paths.get(a[2]).resolve(FeiVectors.file(m));
            Path y = Paths.get(a[4]).resolve(FeiVectors.file(m));
            sameL1.add(FeiVectors.file(m) + " — строк попыток " + sameContent(x, y));
        }
        String[][] parts = {{"2c-mlda", "2c-mlda-ratio+pose", "A"}, {FeiMethods.LBP_MLDA, "5-lbp-mlda-ratio+pose", "L1"},
            {FeiMethods.LBP_MLDA_IN, "5-lbp-mlda-in-ratio+pose", "L1-in"}, {FeiMethods.LBP_MLDA_OUT, "5-lbp-mlda-out-ratio+pose", "L1-out"}};
        List<FeiNorm.Part> ps = new ArrayList<>();
        for (String[] p : parts) {
            boolean isA = p[2].equals("A");
            Map<String, List<Row>> groups = new HashMap<>();
            Map<String, Row> ref = FeiNorm.reference(Paths.get(isA ? a[1] : a[3]), List.of(p[0], p[0] + FeiMethods.MIRROR), groups);
            FeiNorm.Part part = FeiNorm.load(p[0], p[1], p[2], Paths.get(isA ? a[0] : a[2]), ref, groups, sel, r0, idx);
            FeiNorm.evaluate(part, people.size(), b, seed, 1);
            ps.add(part);
        }
        Path out = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Files.createDirectories(out);
        FarMethods.write(out.resolve(MASK_FILE), maskReport(ps, a, sameL1, r0, b, seed, people.size()));
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println(String.format(Locale.ROOT, "Маски LBP: %.0f с; отчёт %s",
                (System.nanoTime() - t0) / 1e9, out.resolve(MASK_FILE)));
    }

    static StringBuilder maskReport(List<FeiNorm.Part> ps, String[] a, List<String> sameL1, double r0, int b, long seed, int persons) {
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: LBP — внутренние и краевые блоки, проверка гипотезы «LBP различает сессию» (FeiCompare mask, faces-fei-compare)\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Векторы A: ").append(a[0]).append(" (вне git; получены: ").append(FeiNorm.COMMAND).append("); контроль по ").append(a[1]).append('\n');
        t.append("Векторы L1, L1-in, L1-out: ").append(a[2]).append(" (вне git; получены: ").append(MASK_COMMAND).append("); контроль по ")
                .append(a[3]).append('\n');
        for (FeiNorm.Part p : ps) for (String f : p.files) t.append("  ").append(p.label).append(": ").append(f).append('\n');
        t.append("Строки: A = 2c-mlda-ratio+pose; L1 = 5-lbp-mlda-ratio+pose (LBP u2 (8, 2), сетка 7×7, кадр а без нормализации освещения,\n"
                + "корень гистограмм + MLDA, как A); L1-in = 5-lbp-mlda-in-ratio+pose — то же, только внутренние блоки (строки и столбцы\n"
                + "1–5, 25 блоков, вектор 1475); L1-out = 5-lbp-mlda-out-ratio+pose — только краевое кольцо (строка или столбец 0 или 6,\n"
                + "24 блока, вектор 1416). Маски заданы заранее, на FEI не подбирались. Выбор модели по позе: r0 = ").append(r0)
                .append(" (|r| пробы ≤ r0 — X,\nиначе X+mirror); оценка d₁/d₂ (FeiScores.ratio); порог — Нейман – Пирсон по снимкам 40 пороговых "
                + "(FAR 1 % и 5 %),\nкак в fei_scores.txt. Группы (i) верный принят, (ii) верный отказ, (iii) неверный отказ, (iv) неверный принят — "
                + "как в\nfei_decompose.txt; анфас/поворот — по |r| ≤ r0.\n");
        t.append("Блоки по шаблону выравнивания кадра 92×112 (ArcFace −10 по x; коды 88×108, границы блоков x: 0, 12, 25, 37, 50, 62, 75, 88;\n"
                + "y: 0, 15, 30, 46, 61, 77, 92, 108): глаза — блоки (3, 2) и (3, 4), нос — (4, 3), углы рта — (5, 2) и (5, 4), все во\n"
                + "внутренних. Кольцо содержит линию лба и волос, контур щёк, уши, фон, подбородок и, вероятно, низ нижней губы (углы рта\n"
                + "в 1,6 пкс от строки 6); поэтому L1-out — верхняя оценка вклада края кадра вместе с краем лица, L1-in — оценка лица без\n"
                + "низа губы и подбородка. Для повёрнутых снимков положение блоков не проверялось.\n");
        if (b > 0) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям FEI целиком: B = %d, seed = %d, людей %d, одни веса для всех строк (как "
                    + "fei_bootstrap.txt); порог — заново в\nкаждой реплике и конфигурации. [2,5; 97,5] — процентили; «*» — интервал разности не "
                    + "содержит 0.\n", b, seed, persons));
        }
        t.append("\n=== Контроль ===\n");
        t.append("L1: векторы этого прогона = ").append(a[4]).append(" по содержимому (галереи и строки попыток, без шапки): ")
                .append(String.join("; ", sameL1)).append('\n');
        control(t, ps);
        tables(t, ps, b, new int[][] {{2, 1}, {3, 1}, {2, 0}}, new String[] {"", "", ""});
        return t;
    }
}
