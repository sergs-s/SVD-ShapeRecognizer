package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.IntStream;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeiScores.Stat;
import svd.recognizer.faces.FeiScores.Transform;

/**
 * PLAN.md, п. 4.2, «d₁/d₂ в объединениях», шаг (а1): объединение на уровне решений из файла оценок faces-fei, без
 * переобучения. Части — строки выбора модели по позе пробы (+pose, r0 = faces.fei.pose.r0: |r| пробы ≤ r0 — строка X,
 * иначе X+mirror): A = 2c-mlda-ratio+pose, B = 4-wpca-cos-0-ratio+pose, C = 2a95-fisher-ratio+pose; объединения A+B,
 * A+C, A+B+C.
 *
 * Оценка части — ℓ = log r, r = d₁/d₂ по правилу вырожденных случаев ({@link FeiScores#ratio(Row, int[])}). Нормировка
 * части в каждой конфигурации: z = (ℓ − m)/s, m и s — среднее и выборочное стандартное отклонение (n − 1) ℓ по всем
 * снимкам пороговых чужих конфигурации, независимо от совпадения кандидатов частей; свои и контроль в нормировке не
 * участвуют. Каждая часть нормируется один раз (части — отношения над сырыми расстояниями, когорты в файле нет), поэтому
 * тождества «повторная нормировка той же оценки», как у дублей 4-sum, нет. Правило: лучшие кандидаты всех частей
 * совпали — оценка Σ z, кандидат общий; иначе отказ (+∞). Свой отклонён и при бесконечной оценке истинного хотя бы у
 * одной части (как в FeiScores.Stat). Порог — как везде в разборе (FeiScores.Stat: по снимкам пороговых, FAR 1 % и 5 %;
 * по людям справочно).
 *
 * Бутстреп (faces.fei.bootstrap): люди FEI целиком, те же веса, что в fei_bootstrap.txt (FeiBootstrap.weights, тот же
 * seed), поэтому разности парные; m и s пересчитываются по взвешенным пороговым в каждой реплике и конфигурации (это
 * калибровка, как порог), порог — тоже заново.
 *
 * Включается ключом faces.fei.fusion (FACES_FEI_FUSION = true) при заданном faces.fei.pose.r0; отчёт fei_fusion.txt.
 * fei_scores.txt и fei_bootstrap.txt не меняются.
 *
 * @author ssv
 */
final class FeiFusion {

    static final String FILE = "fei_fusion.txt";
    /** Части A, B, C. */
    static final List<String> PARTS = List.of("2c-mlda-ratio+pose", "4-wpca-cos-0-ratio+pose", "2a95-fisher-ratio+pose");
    /** Объединения (индексы частей): A+B, A+C, A+B+C. */
    static final int[][] FUSIONS = {{0, 1}, {0, 2}, {0, 1, 2}};
    /** Пары частей для таблиц совпадения ошибок. */
    static final int[][] PAIRS = {{0, 1}, {0, 2}, {1, 2}};
    /** Парные разности бутстрепа (метки строк). */
    static final String[][] DIFFS = {{"A+B", "A"}, {"A+C", "A"}, {"A+B+C", "A"}, {"A+B", "A+C"}};
    private static final String LETTERS = "ABCDEFGH";

    /** Попытка конфигурации: по частям — r, ℓ = log r, лучший кандидат, бесконечна ли оценка истинного. */
    static final class Att {
        final String role;
        final String person;
        final int image;
        final String trueId;
        /** Индекс человека FEI (бутстреп) или −1. */
        final int p;
        final double[] r;
        final double[] l;
        final String[] best;
        final boolean[] trueInf;

        Att(Row row, int parts, int p) {
            role = row.role();
            person = row.person();
            image = row.image();
            trueId = row.trueId();
            this.p = p;
            r = new double[parts];
            l = new double[parts];
            best = new String[parts];
            trueInf = new boolean[parts];
        }
    }

    /** Попытки одной конфигурации (s, j) в порядке файла. */
    static final class Conf {
        final int s;
        final int j;
        final boolean frontal;
        final List<Att> all = new ArrayList<>();
        final List<Att> own = new ArrayList<>();
        final List<Att> thr = new ArrayList<>();

        Conf(int s, int j) {
            this.s = s;
            this.j = j;
            frontal = FeiProtocol.frontal(j);
        }
    }

    private final List<String> parts;
    private final int[][] fusions;
    private final FeiBootstrap boot;
    final Stat[] partStats;
    final Stat[] fusionStats;
    /** Счётчики случаев правила отношения [часть][случай] по всем попыткам части. */
    final int[][] cases;
    final List<Conf> confs = new ArrayList<>();
    /** {m, s} частей по конфигурациям (исходная выборка). */
    final List<List<double[]>> norms = new ArrayList<>();
    private final Map<String, List<List<Row>>> pending = new HashMap<>();

    /** @param boot бутстреп по людям FEI (части добавляются в него как строки RATIO) или null */
    FeiFusion(FeiBootstrap boot) {
        this(PARTS, FUSIONS, boot);
    }

    FeiFusion(List<String> parts, int[][] fusions, FeiBootstrap boot) {
        if (parts.size() > LETTERS.length()) throw new IllegalArgumentException("Частей больше " + LETTERS.length());
        this.parts = List.copyOf(parts);
        this.fusions = fusions;
        this.boot = boot;
        partStats = new Stat[parts.size()];
        cases = new int[parts.size()][FeiScores.CASES];
        for (int i = 0; i < parts.size(); i++) {
            partStats[i] = new Stat(parts.get(i), Transform.RATIO);
            norms.add(new ArrayList<>());
        }
        fusionStats = new Stat[fusions.length];
        for (int f = 0; f < fusions.length; f++) fusionStats[f] = new Stat(label(fusions[f]), Transform.NONE);
    }

    /** Метка части: A, B, C… */
    static String letter(int i) {
        return String.valueOf(LETTERS.charAt(i));
    }

    /** Метка объединения: «A+B». */
    static String label(int[] f) {
        StringBuilder b = new StringBuilder();
        for (int i : f) b.append(b.length() > 0 ? "+" : "").append(letter(i));
        return b.toString();
    }

    /**
     * Попытки строки id в одной конфигурации (строки +pose разбора); не часть — пропускается. Когда собраны все части
     * конфигурации, строятся объединения.
     */
    void part(String id, List<Row> rows) {
        int i = parts.indexOf(id);
        if (i < 0) return;
        Row first = rows.get(0);
        String key = first.s() + " " + first.j();
        List<List<Row>> got = pending.computeIfAbsent(key, k -> new ArrayList<>(Collections.nCopies(parts.size(), null)));
        if (got.get(i) != null) throw new IllegalStateException("Часть " + id + " повторяется в конфигурации " + key);
        got.set(i, rows);
        partStats[i].add(rows);
        if (boot != null) boot.add(id, Transform.RATIO, rows);
        if (got.contains(null)) return;
        pending.remove(key);
        build(got);
    }

    /** Все конфигурации полны; хотя бы одна есть. */
    void finish() {
        if (!pending.isEmpty()) throw new IllegalStateException("Объединение: в конфигурациях " + pending.keySet() + " есть не все части " + parts);
        if (confs.isEmpty()) {
            throw new IllegalStateException("Объединение: нет строк частей " + parts + " (нужен файл оценок со строками +mirror и faces.fei.pose.r0)");
        }
    }

    private static String key(Row r) {
        return r.role() + "\t" + r.person() + "\t" + r.image();
    }

    private void build(List<List<Row>> got) {
        Row first = got.get(0).get(0);
        Conf c = new Conf(first.s(), first.j());
        int n = parts.size();
        List<Map<String, Row>> by = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (got.get(i).size() != got.get(0).size()) {
                throw new IllegalStateException("Разное число попыток у частей " + parts.get(0) + " и " + parts.get(i) + " в конфигурации " + c.s + " " + c.j);
            }
            Map<String, Row> m = new HashMap<>();
            for (Row r : got.get(i)) if (m.put(key(r), r) != null) throw new IllegalStateException("Повтор попытки: " + r);
            by.add(m);
        }
        for (Row r0 : got.get(0)) {
            Att a = new Att(r0, n, boot == null ? -1 : boot.person(r0.person()));
            for (int i = 0; i < n; i++) {
                Row r = by.get(i).get(key(r0));
                if (r == null) throw new IllegalStateException("Нет попытки части " + parts.get(i) + " для " + r0);
                a.r[i] = FeiScores.ratio(r, cases[i]);
                a.l[i] = Math.log(a.r[i]);
                a.best[i] = r.bestId();
                a.trueInf[i] = Double.isInfinite(r.trueScore());
            }
            c.all.add(a);
            if (a.role.equals(FeiScores.OWN)) c.own.add(a);
            if (a.role.equals(FeiScores.THR)) c.thr.add(a);
        }
        double[][] ms = new double[n][];
        for (int i = 0; i < n; i++) {
            ms[i] = norm(c, i, null);
            norms.get(i).add(ms[i]);
        }
        confs.add(c);
        for (int f = 0; f < fusions.length; f++) fusionStats[f].add(rows(c, fusions[f], ms));
    }

    /**
     * {m, s} части i: среднее и выборочное стандартное отклонение ℓ по всем снимкам пороговых конфигурации, с весами
     * людей w (null — веса 1). s = 0 — случай 6, исключение.
     */
    double[] norm(Conf c, int i, int[] w) {
        double sw = 0;
        double sum = 0;
        for (Att a : c.thr) {
            int q = w == null ? 1 : w[a.p];
            sw += q;
            sum += q * a.l[i];
        }
        String where = parts.get(i) + " s=" + c.s + " j=" + c.j;
        if (sw < 2) throw new IllegalStateException("Нормировка части " + where + ": меньше двух снимков пороговых (" + sw + ")");
        double m = sum / sw;
        if (!Double.isFinite(m)) throw new IllegalStateException("Нормировка части " + where + ": среднее ℓ пороговых не конечно (" + m + ")");
        double v = 0;
        for (Att a : c.thr) {
            int q = w == null ? 1 : w[a.p];
            double d = a.l[i] - m;
            v += q * d * d;
        }
        double s = Math.sqrt(v / (sw - 1));
        if (!(s > 0)) throw new IllegalStateException("Отношение d₁/d₂ (случай 6): s = 0 у части " + where + " — нормировка не определена");
        return new double[] {m, s};
    }

    /** Общий лучший кандидат частей f или null (кандидаты не совпали). */
    static String candidate(Att a, int[] f) {
        String c = a.best[f[0]];
        for (int i : f) if (!a.best[i].equals(c)) return null;
        return c;
    }

    /** Оценка объединения: Σ z частей при общем кандидате, иначе +∞. */
    static double score(Att a, int[] f, double[][] ms) {
        if (candidate(a, f) == null) return Double.POSITIVE_INFINITY;
        double z = 0;
        for (int i : f) z += (a.l[i] - ms[i][0]) / ms[i][1];
        return z;
    }

    static boolean trueInf(Att a, int[] f) {
        for (int i : f) if (a.trueInf[i]) return true;
        return false;
    }

    /** Строки объединения f в конфигурации (для FeiScores.Stat, оценка — best_score, Transform.NONE). */
    List<Row> rows(Conf c, int[] f, double[][] ms) {
        String id = label(f);
        List<Row> out = new ArrayList<>(c.all.size());
        for (Att a : c.all) {
            String cand = candidate(a, f);
            out.add(new Row(id, c.s, c.j, a.role, a.person, a.image, a.trueId, cand == null ? FeiScores.NONE : cand, score(a, f, ms), FeiScores.NONE,
                    Double.NaN, trueInf(a, f) ? Double.POSITIVE_INFINITY : 0.0, -1));
        }
        return out;
    }

    // ---------------------------------------------------------------- бутстреп

    /** Конфигурации объединения f для FeiBootstrap.metrics при весах людей w (m, s и порог — по взвешенным пороговым). */
    List<FeiBootstrap.Cfg> cfgs(int[] f, int[] w) {
        List<FeiBootstrap.Cfg> out = new ArrayList<>(confs.size());
        for (Conf c : confs) {
            double[][] ms = new double[parts.size()][];
            for (int i : f) ms[i] = norm(c, i, w);
            FeiBootstrap.Cfg g = new FeiBootstrap.Cfg(c.frontal);
            g.ownP = new int[c.own.size()];
            g.ownS = new double[c.own.size()];
            g.ownOk = new boolean[c.own.size()];
            List<double[]> thr = new ArrayList<>();
            List<double[]> ctrl = new ArrayList<>();
            int k = 0;
            for (Att a : c.all) {
                double sc = score(a, f, ms);
                switch (a.role) {
                    case FeiScores.OWN -> {
                        String cand = candidate(a, f);
                        g.ownP[k] = a.p;
                        g.ownS[k] = trueInf(a, f) ? Double.POSITIVE_INFINITY : sc;
                        g.ownOk[k] = cand != null && cand.equals(a.trueId);
                        k++;
                    }
                    case FeiScores.THR -> thr.add(new double[] {a.p, sc});
                    case FeiScores.CTRL -> ctrl.add(new double[] {a.p, sc});
                    default -> throw new IllegalArgumentException("Роль: " + a.role);
                }
            }
            thr.sort((x, y) -> Double.compare(x[1], y[1]));
            ctrl.sort((x, y) -> Double.compare(x[1], y[1]));
            g.thrP = thr.stream().mapToInt(x -> (int) x[0]).toArray();
            g.thrS = thr.stream().mapToDouble(x -> x[1]).toArray();
            g.ctrlP = ctrl.stream().mapToInt(x -> (int) x[0]).toArray();
            g.ctrlS = ctrl.stream().mapToDouble(x -> x[1]).toArray();
            g.ctrlPeople = Arrays.stream(g.ctrlP).distinct().sorted().toArray();
            out.add(g);
        }
        return out;
    }

    /** Бутстреп: части — FeiBootstrap.run (как fei_bootstrap.txt), объединения — те же веса, m, s и порог в каждой реплике. */
    FeiBootstrap.Result bootstrap(int b, long seed) {
        if (boot == null) throw new IllegalStateException("Объединение: бутстреп не задан");
        FeiBootstrap.Result res = boot.run(b, seed);
        int n = boot.size();
        int[][] w = FeiBootstrap.weights(n, b, seed);
        int[] u = FeiBootstrap.unit(n);
        for (int[] f : fusions) {
            double[][] v = new double[FeiBootstrap.METRICS][b];
            IntStream.range(0, b).parallel().forEach(i -> {
                double[] m = FeiBootstrap.metrics(cfgs(f, w[i]), w[i]);
                for (int k = 0; k < FeiBootstrap.METRICS; k++) v[k][i] = m[k];
            });
            List<FeiBootstrap.Cfg> cu = cfgs(f, u);
            String id = label(f);
            res.point.put(id, FeiBootstrap.metrics(cu, u));
            res.reps.put(id, v);
            res.cp.put(id, new int[][] {FeiBootstrap.ctrlCounts(cu, u, 0), FeiBootstrap.ctrlCounts(cu, u, 1)});
        }
        return res;
    }

    // ---------------------------------------------------------------- совпадение ошибок

    /** Верно ли свой a принят частью i при пороге θ (как FeiScores.Stat: не отклонён и лучший — истинный). */
    static boolean correct(Att a, int i, double theta) {
        double s = a.trueInf[i] ? Double.POSITIVE_INFINITY : a.r[i];
        return !(s > theta) && a.best[i].equals(a.trueId);
    }

    /**
     * Таблица 2×2 по своим для пары частей (x, y): {оба верно, верно только x, верно только y, оба ошиблись, лучший
     * кандидат совпал}; k — индекс FeiScores.FARS (порог части по снимкам пороговых в каждой конфигурации) или −1 —
     * argmin без порога.
     */
    int[] agreement(int x, int y, int k) {
        int[] t = new int[5];
        for (Conf c : confs) {
            double tx = Double.POSITIVE_INFINITY;
            double ty = Double.POSITIVE_INFINITY;
            if (k >= 0) {
                tx = FaceEvaluation.neymanPearsonThreshold(c.thr.stream().mapToDouble(a -> a.r[x]).toArray(), FeiScores.FARS[k]);
                ty = FaceEvaluation.neymanPearsonThreshold(c.thr.stream().mapToDouble(a -> a.r[y]).toArray(), FeiScores.FARS[k]);
            }
            for (Att a : c.own) {
                boolean cx = k >= 0 ? correct(a, x, tx) : a.best[x].equals(a.trueId);
                boolean cy = k >= 0 ? correct(a, y, ty) : a.best[y].equals(a.trueId);
                t[cx ? (cy ? 0 : 1) : (cy ? 2 : 3)]++;
                if (a.best[x].equals(a.best[y])) t[4]++;
            }
        }
        return t;
    }

    /** Несовпадение кандидатов объединения f по ролям own, thr, ctrl: {x, n}. */
    int[][] disagreement(int[] f) {
        int[][] t = new int[3][2];
        for (Conf c : confs) {
            for (Att a : c.all) {
                int r = switch (a.role) {
                    case FeiScores.OWN -> 0;
                    case FeiScores.THR -> 1;
                    default -> 2;
                };
                t[r][1]++;
                if (candidate(a, f) == null) t[r][0]++;
            }
        }
        return t;
    }

    // ---------------------------------------------------------------- отчёт

    private String partLabel(int i) {
        return letter(i) + " = " + parts.get(i);
    }

    /** Ключ строки в результате бутстрепа по метке («A» — id части, «A+B» — объединение). */
    private String resultKey(String label) {
        if (label.length() == 1) {
            int i = LETTERS.indexOf(label.charAt(0));
            return i >= 0 && i < parts.size() ? parts.get(i) : null;
        }
        return label;
    }

    StringBuilder report(FeiBootstrap.Result r, List<String> header, String source, double r0) {
        StringBuilder t = new StringBuilder();
        t.append("Шаг (а1), PLAN.md п. 4.2: объединение по решениям из файла оценок faces-fei (FeiFusion, faces-fei-scores с faces.fei.fusion), "
                + "без переобучения\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл: ").append(source).append('\n');
        t.append("Шапка файла:\n");
        for (String h : header) t.append("  ").append(h).append('\n');
        t.append(String.format(Locale.ROOT, "Конфигураций: %d из %d.\n", confs.size(), FeiProtocol.CONFIGS));
        t.append("Части (выбор модели по позе пробы, r0 = ").append(r0).append(": |r| пробы ≤ r0 — строка X, иначе X+mirror): ");
        for (int i = 0; i < parts.size(); i++) t.append(i > 0 ? "; " : "").append(partLabel(i));
        t.append(".\nОбъединения: ");
        for (int f = 0; f < fusions.length; f++) t.append(f > 0 ? ", " : "").append(label(fusions[f]));
        t.append(".\n");
        t.append("Оценка части ℓ = log(d₁/d₂) (лучший / второй кандидат; правило вырожденных случаев — ниже). Нормировка части в каждой\n"
                + "конфигурации: z = (ℓ − m)/s, m и s — среднее и выборочное стандартное отклонение (n − 1) ℓ по всем снимкам пороговых\n"
                + "чужих конфигурации (лучший кандидат — как в файле), независимо от совпадения кандидатов; свои и контроль в нормировке\n"
                + "не участвуют, когорты в файле нет. Каждая часть нормируется один раз (части — отношения над сырыми расстояниями),\n"
                + "повторной нормировки одной оценки нет (не как у дублей 4-sum: там Z-norm по той же когорте дважды).\n"
                + "Правило: лучшие кандидаты всех частей совпали — оценка Σ z, кандидат общий; не совпали — отказ (оценка +∞; у своего —\n"
                + "ошибка «отказ»). Порог — как в fei_scores.txt: по снимкам пороговых чужих в каждой конфигурации (FAR 1 % и 5 %), по\n"
                + "людям — справочно. Ошибка своих = FRR + принят под чужим именем. Строки A, B, C — те же, что в fei_scores.txt\n"
                + "(монотонное преобразование одной оценки порог не меняет). Оговорка: снимки одного человека коррелируют, FAR по\n"
                + "снимкам оптимистичен; интервалы — бутстреп по людям.\n");
        t.append("Правило вырожденных случаев (FeiScores.ratio, r = d₁/d₂, по определению d₁ ≤ d₂): 1 — d₁ = 0 (и любое r < 1e-12):\n"
                + "r = ρ_min = 1e-12; 2 — d₁ = d₂ = 0: r = 1; 3 — d₂ = 0 при d₁ > 0, d₁ > d₂, отрицательное расстояние, NaN: исключение;\n"
                + "4 — d₂ = +∞: отказ (r = +∞); 5 — 1 − cos ≥ −1e-12 и < 0: приводится к 0, дальше 1–2; 6 — s = 0 у части в\n"
                + "конфигурации: исключение.\n");

        List<Stat> all = new ArrayList<>(Arrays.asList(partStats));
        all.addAll(Arrays.asList(fusionStats));
        t.append("\n=== Основная таблица: порог по снимкам пороговых чужих ===\n");
        t.append("строка | ошибка своих FAR 1 % (FRR + под чужим) | FAR 5 % (FRR + под чужим) | анфас / поворот FAR 5 % | "
                + "FAR контроля снимки FAR 1 %: сумма; худш. (↑95) / мед. | FAR 5 %: сумма; худш. (↑95) / мед. | "
                + "справочно, порог по людям α=0,05: ошибка своих | FAR контроля люди худш. (↑95) / мед.\n");
        for (int q = 0; q < all.size(); q++) {
            Stat s = all.get(q);
            String name = q < parts.size() ? partLabel(q) : s.id;
            t.append(String.join(" | ", name, FeiScores.errText(s, FeiScores.T1), FeiScores.errText(s, FeiScores.T5),
                    FeiMethods.pctOf(s.errors(FeiScores.T5, 1), s.attBy[1]) + " / " + FeiMethods.pctOf(s.errors(FeiScores.T5, 0), s.attBy[0]),
                    FeiMethods.attemptsSum(s.far(s.farCtrl, FeiScores.T1)) + "; " + FarMethods.attempts(s.far(s.farCtrl, FeiScores.T1)),
                    FeiMethods.attemptsSum(s.far(s.farCtrl, FeiScores.T5)) + "; " + FarMethods.attempts(s.far(s.farCtrl, FeiScores.T5)),
                    FeiMethods.pctOf(s.errors(FeiScores.P05), s.att), FarMethods.persons(s.far(s.farCtrl, FeiScores.P05)))).append('\n');
        }

        if (r == null) {
            t.append("\nБутстреп выключен (faces.fei.bootstrap не задан).\n");
        } else {
            t.append(String.format(Locale.ROOT, "\n=== Бутстреп по людям FEI: B = %d, seed = %d (faces.seed + %d), людей %d, в файле %d ===\n", r.b,
                    r.seed, FeiBootstrap.SEED_OFFSET, r.persons, r.inFile));
            t.append("Как fei_bootstrap.txt: люди FEI вытягиваются целиком, один вес на человека — общий для всех строк (разности парные;\n"
                    + "части A, B, C — те же числа, что в fei_bootstrap.txt); m и s частей и порог — заново в каждой реплике и конфигурации\n"
                    + "по взвешенным пороговым. Точка — веса 1; [2,5; 97,5] — процентили; ↑95 — процентиль 95 %; КП — Клоппер – Пирсон\n"
                    + "по снимкам (оптимистична); FAR контроля по людям при пороге по снимкам 1 % — справочно.\n");
            t.append("строка | ошибка своих FAR 1 % [2,5; 97,5] | FAR 5 % [2,5; 97,5] | анфас FAR 5 % [..] | поворот FAR 5 % [..] | "
                    + "FAR контроля снимки 1 %: точка [2,5; 97,5] ↑95; КП | 5 %: точка [..] ↑95; КП | FAR контроля люди 1 % [..]\n");
            List<String> labels = new ArrayList<>();
            for (int i = 0; i < parts.size(); i++) labels.add(letter(i));
            for (int[] f : fusions) labels.add(label(f));
            for (String lb : labels) {
                String id = resultKey(lb);
                double[] p = r.point.get(id);
                double[][] v = r.reps.get(id);
                int[][] cp = r.cp.get(id);
                t.append(String.join(" | ", lb.length() == 1 ? partLabel(LETTERS.indexOf(lb.charAt(0))) : lb,
                        FeiBootstrap.ci(p[FeiBootstrap.ERR1], v[FeiBootstrap.ERR1], 1), FeiBootstrap.ci(p[FeiBootstrap.ERR5], v[FeiBootstrap.ERR5], 1),
                        FeiBootstrap.ci(p[FeiBootstrap.ERR5_FRONTAL], v[FeiBootstrap.ERR5_FRONTAL], 1),
                        FeiBootstrap.ci(p[FeiBootstrap.ERR5_TURN], v[FeiBootstrap.ERR5_TURN], 1),
                        FeiBootstrap.far(p[FeiBootstrap.FAR1], v[FeiBootstrap.FAR1], cp[0]), FeiBootstrap.far(p[FeiBootstrap.FAR5], v[FeiBootstrap.FAR5], cp[1]),
                        FeiBootstrap.ci(p[FeiBootstrap.FARP1], v[FeiBootstrap.FARP1], 1))).append('\n');
            }
            t.append("\n=== Парные разности A − B, п. п. (те же реплики; «*» — интервал не содержит 0) ===\n");
            t.append("A − B | ошибка своих FAR 1 % [2,5; 97,5] | FAR 5 % | анфас FAR 5 % | поворот FAR 5 % | FAR контроля снимки 1 % | 5 %\n");
            int[] ks = {FeiBootstrap.ERR1, FeiBootstrap.ERR5, FeiBootstrap.ERR5_FRONTAL, FeiBootstrap.ERR5_TURN, FeiBootstrap.FAR1, FeiBootstrap.FAR5};
            int[] dec = {1, 1, 1, 1, 2, 2};
            for (String[] d : DIFFS) {
                String a = resultKey(d[0]);
                String b = resultKey(d[1]);
                if (a == null || b == null || !r.point.containsKey(a) || !r.point.containsKey(b)) continue;
                List<String> cells = new ArrayList<>();
                cells.add(d[0] + " − " + d[1]);
                for (int i = 0; i < ks.length; i++) {
                    int k = ks[i];
                    double[] diff = new double[r.b];
                    for (int q = 0; q < r.b; q++) diff[q] = r.reps.get(a)[k][q] - r.reps.get(b)[k][q];
                    cells.add(FeiBootstrap.diff(r.point.get(a)[k] - r.point.get(b)[k], diff, dec[i]));
                }
                t.append(String.join(" | ", cells)).append('\n');
            }
        }

        int nOwn = confs.stream().mapToInt(c -> c.own.size()).sum();
        t.append("\n=== Совпадение ошибок на своих (").append(nOwn).append(" попыток) ===\n");
        t.append("«Верно» при пороге — принят под своим именем (порог части по снимкам пороговых в каждой конфигурации, как в основной\n"
                + "таблице); argmin — лучший кандидат истинный, без порога. Последний столбец argmin — лучшие кандидаты пары совпали\n"
                + "(верно или нет).\n");
        String[] what = {"FAR 5 %", "FAR 1 %", "argmin без порога"};
        int[] kk = {1, 0, -1};
        for (int q = 0; q < what.length; q++) {
            t.append("-- ").append(what[q]).append('\n');
            t.append("пара | оба верно | верно только первый | верно только второй | оба ошиблись").append(kk[q] < 0 ? " | лучший кандидат совпал" : "")
                    .append('\n');
            for (int[] pr : PAIRS) {
                if (pr[0] >= parts.size() || pr[1] >= parts.size()) continue;
                int[] a = agreement(pr[0], pr[1], kk[q]);
                t.append(String.join(" | ", letter(pr[0]) + "–" + letter(pr[1]), FeiMethods.pctOf(a[0], nOwn), FeiMethods.pctOf(a[1], nOwn),
                        FeiMethods.pctOf(a[2], nOwn), FeiMethods.pctOf(a[3], nOwn))).append(kk[q] < 0 ? " | " + FeiMethods.pctOf(a[4], nOwn) : "")
                        .append('\n');
            }
        }

        t.append("\n=== Отказы из-за несовпадения лучших кандидатов частей (без порога) ===\n");
        t.append("объединение | свои | пороговые чужие (снимки) | контрольные чужие (снимки)\n");
        for (int[] f : fusions) {
            int[][] d = disagreement(f);
            t.append(String.join(" | ", label(f), FeiMethods.pctOf(d[0][0], d[0][1]), FeiMethods.pctOf(d[1][0], d[1][1]),
                    FeiMethods.pctOf(d[2][0], d[2][1]))).append('\n');
        }

        t.append("\n=== Случаи правила отношения по частям (все попытки части: свои, пороговые, контроль; 3 и 6 — исключение, при\n"
                + "завершённом разборе 0) ===\n");
        t.append("часть | попыток | 1: r < ρ_min (d₁ = 0) | 2: d₁ = d₂ = 0 | 3: ошибка данных | 4: d₂ = +∞ (отказ) | 5: 1 − cos округлено к 0 | "
                + "6: s = 0\n");
        for (int i = 0; i < parts.size(); i++) {
            int n = confs.stream().mapToInt(c -> c.all.size()).sum();
            StringBuilder row = new StringBuilder(partLabel(i)).append(" | ").append(n);
            for (int k = 0; k < FeiScores.CASES; k++) row.append(" | ").append(cases[i][k]);
            t.append(row).append('\n');
        }

        t.append("\n=== Нормировка частей по пороговым (исходная выборка), по конфигурациям: медиана [мин; макс] ===\n");
        t.append("часть | m | s\n");
        for (int i = 0; i < parts.size(); i++) {
            double[] m = norms.get(i).stream().mapToDouble(x -> x[0]).sorted().toArray();
            double[] s = norms.get(i).stream().mapToDouble(x -> x[1]).sorted().toArray();
            t.append(String.format(Locale.ROOT, "%s | %.4f [%.4f; %.4f] | %.4f [%.4f; %.4f]\n", partLabel(i), FeiMethods.median(m), m[0], m[m.length - 1],
                    FeiMethods.median(s), s[0], s[s.length - 1]));
        }
        return t;
    }
}
