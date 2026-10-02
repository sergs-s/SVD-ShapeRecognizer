package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import svd.recognizer.faces.GalleryEvaluation.Impostors;

/**
 * Этап 5б: файл оценок faces-fei (reports/faces/fei/fei_scores.tsv.gz, TSV + gzip). Одна строка — одна попытка метода в
 * конфигурации (s, j): свой (контрольный снимок j-й из восьми), пороговый чужой или контрольный чужой (все отобранные
 * снимки). Оценка: меньше — ближе; решение — argmin по 100 своим галереи, принят, если оценка ≤ θ.
 *
 * Столбцы: method, s, j, role (own / thr / ctrl), person, image (номер снимка FEI), true_id (для своих — сам человек, у
 * чужих «-»), best_id, best_score, second_id, second_score, true_score (оценка по true_id; у чужих «-»), no_agreement
 * (для 4-agr-*: 1 — согласие не достигнуто, все оценки +∞, best_id «-»; 0 — достигнуто, best_score — итоговая сумма,
 * second_* «-»; у прочих методов «-»). Числа — Double.toString (точно восстанавливаются), бесконечность — Infinity.
 *
 * Разбор файла без обучения — {@code mvn compile exec:java@faces-fei-scores} (путь к файлу — -Dexec.args, по умолчанию
 * reports/faces/fei/fei_scores.tsv.gz), отчёт reports/faces/fei/fei_scores.txt. Порог в каждой конфигурации: основной —
 * по снимкам пороговых чужих (наибольший θ, при котором принято не больше ⌊FAR·n⌋ снимков; FAR 1 % и 5 %); справочно —
 * по людям (α = 0,05 и 0), как в FeiMethods. Новые строки из файла: оценка лучшего / второго (-ratio) и z₁ − z₂
 * (3-znorm-lda-margin; отношение для z не определено); argmin прежний.
 *
 * @author ssv
 */
public final class FeiScores {

    static final String FILE = "fei_scores.tsv.gz";
    static final String OWN = "own";
    static final String THR = "thr";
    static final String CTRL = "ctrl";
    static final String NONE = "-";
    static final String COLUMNS = "method\ts\tj\trole\tperson\timage\ttrue_id\tbest_id\tbest_score\tsecond_id\tsecond_score\ttrue_score\t"
            + "no_agreement";

    private FeiScores() {
    }

    /** Строка попытки. */
    record Row(String method, int s, int j, String role, String person, int image, String trueId, String bestId, double best,
               String secondId, double second, double trueScore, int noAgreement) {

        boolean own() {
            return role.equals(OWN);
        }
    }

    /**
     * Строка файла по оценкам попытки (по людям галереи).
     *
     * @param trueIdx индекс своего в галерее; −1 — чужой
     * @param agree   метод согласия (4-agr-*)
     */
    static String line(String method, int s, int j, String role, String person, int image, int trueIdx, List<String> gallery, double[] v,
                       boolean agree) {
        int b = 0;
        for (int p = 1; p < v.length; p++) if (v[p] < v[b]) b = p;
        int c = -1;
        for (int p = 0; p < v.length; p++) if (p != b && (c < 0 || v[p] < v[c])) c = p;
        String bestId = gallery.get(b);
        String secondId = gallery.get(c);
        String second = num(v[c]);
        String flag = NONE;
        if (agree) {
            boolean no = v[b] == Double.POSITIVE_INFINITY;
            flag = no ? "1" : "0";
            if (no) bestId = NONE;
            secondId = NONE;
            second = NONE;
        }
        return String.join("\t", method, Integer.toString(s), Integer.toString(j), role, person, Integer.toString(image),
                trueIdx < 0 ? NONE : gallery.get(trueIdx), bestId, num(v[b]), secondId, second, trueIdx < 0 ? NONE : num(v[trueIdx]), flag)
                + "\n";
    }

    static String num(double x) {
        return Double.toString(x);
    }

    /** Разбор строки файла (не заголовка). */
    static Row parse(String line) {
        String[] c = line.split("\t", -1);
        if (c.length != 13) throw new IllegalArgumentException("Строка файла оценок: " + c.length + " столбцов вместо 13: " + line);
        return new Row(c[0], Integer.parseInt(c[1]), Integer.parseInt(c[2]), c[3], c[4], Integer.parseInt(c[5]), c[6], c[7], dbl(c[8]), c[9],
                dbl(c[10]), dbl(c[11]), c[12].equals(NONE) ? -1 : Integer.parseInt(c[12]));
    }

    private static double dbl(String s) {
        return s.equals(NONE) ? Double.NaN : Double.parseDouble(s);
    }

    // ---------------------------------------------------------------- разбор

    /** FAR по снимкам пороговых чужих (основной порог). */
    static final double[] FARS = {0.01, 0.05};
    /** Пороги: снимки FAR 1 %, 5 %; люди α = 0,05, 0. */
    static final int T1 = 0;
    static final int T5 = 1;
    static final int P05 = 2;
    static final int P0 = 3;
    static final int THRESHOLDS = 4;

    /** Оценка строки по лучшему и второму кандидатам. */
    enum Transform {
        NONE, RATIO, MARGIN;

        double apply(double best, double second) {
            double v = switch (this) {
                case NONE -> best;
                case RATIO -> second == 0 && best == 0 ? 1 : best / second;
                case MARGIN -> best - second;
            };
            return Double.isNaN(v) ? Double.POSITIVE_INFINITY : v;
        }
    }

    /** Новая строка из файла: идентификатор, исходный метод, оценка. */
    record NewRow(String id, String base, Transform t, String about) {}

    static final List<NewRow> NEW_ROWS = List.of(
            new NewRow("2a95-fisher-ratio", "2a95-fisher", Transform.RATIO, "d₁/d₂ над 2a95-fisher"),
            new NewRow("2c-mlda-ratio", "2c-mlda", Transform.RATIO, "d₁/d₂ над 2c-mlda"),
            new NewRow("4-wpca-cos-0-ratio", "4-wpca-cos-0", Transform.RATIO, "d₁/d₂ над 4-wpca-cos-0"),
            new NewRow("3-znorm-lda-margin", "3-znorm-lda", Transform.MARGIN, "z₁ − z₂ над 3-znorm-lda"));

    /** То же для строк +mirror (этап 5б, faces.fei.mirror); появляются, только если основа есть в файле. */
    static final List<NewRow> MIRROR_ROWS = List.of(
            new NewRow("2a95-fisher-ratio+mirror", "2a95-fisher+mirror", Transform.RATIO, "d₁/d₂ над 2a95-fisher+mirror"),
            new NewRow("2c-mlda-ratio+mirror", "2c-mlda+mirror", Transform.RATIO, "d₁/d₂ над 2c-mlda+mirror"),
            new NewRow("4-wpca-cos-0-ratio+mirror", "4-wpca-cos-0+mirror", Transform.RATIO, "d₁/d₂ над 4-wpca-cos-0+mirror"));

    /** Метрики строки, накопленные по конфигурациям. */
    static final class Stat {
        final String id;
        final Transform t;
        int configs;
        int att;
        /** [анфас 1 / поворот 0]. */
        final int[] attBy = new int[2];
        /** [анфас 1 / поворот 0][порог]: отказы и приняты под чужим именем. */
        final int[][] rej = new int[2][THRESHOLDS];
        final int[][] mis = new int[2][THRESHOLDS];
        /** [порог] → по конфигурациям {x, n, xP, nP}: контроль 40, пороговые 40; θ. */
        final List<List<int[]>> farCtrl = new ArrayList<>();
        final List<List<int[]>> farThr = new ArrayList<>();
        final List<List<Double>> theta = new ArrayList<>();

        Stat(String id, Transform t) {
            this.id = id;
            this.t = t;
            for (int k = 0; k < THRESHOLDS; k++) {
                farCtrl.add(new ArrayList<>());
                farThr.add(new ArrayList<>());
                theta.add(new ArrayList<>());
            }
        }

        /** Попытки одной конфигурации (все строки — одного метода, s, j). */
        void add(List<Row> rows) {
            configs++;
            Map<String, List<Double>> thr = new LinkedHashMap<>();
            Map<String, List<Double>> ctrl = new LinkedHashMap<>();
            List<Row> own = new ArrayList<>();
            for (Row r : rows) {
                switch (r.role()) {
                    case OWN -> own.add(r);
                    case THR -> thr.computeIfAbsent(r.person(), k -> new ArrayList<>()).add(score(r));
                    case CTRL -> ctrl.computeIfAbsent(r.person(), k -> new ArrayList<>()).add(score(r));
                    default -> throw new IllegalArgumentException("Роль: " + r.role());
                }
            }
            Impostors thrImp = impostors(thr);
            Impostors ctrlImp = impostors(ctrl);
            double[] thrAll = thr.values().stream().flatMap(List::stream).mapToDouble(Double::doubleValue).toArray();
            double[] th = new double[THRESHOLDS];
            th[T1] = FaceEvaluation.neymanPearsonThreshold(thrAll, FARS[0]);
            th[T5] = FaceEvaluation.neymanPearsonThreshold(thrAll, FARS[1]);
            th[P05] = FarMethods.personThreshold(thrImp, FeiMethods.ALPHAS[FeiMethods.A05]);
            th[P0] = FarMethods.personThreshold(thrImp, FeiMethods.ALPHAS[FeiMethods.A0]);
            for (Row r : own) {
                int fr = FeiProtocol.frontal(r.j()) ? 1 : 0;
                att++;
                attBy[fr]++;
                boolean ok = r.bestId().equals(r.trueId());
                double s = Double.isInfinite(r.trueScore()) ? Double.POSITIVE_INFINITY : score(r);
                for (int k = 0; k < THRESHOLDS; k++) {
                    if (s > th[k]) rej[fr][k]++;
                    else if (!ok) mis[fr][k]++;
                }
            }
            for (int k = 0; k < THRESHOLDS; k++) {
                theta.get(k).add(th[k]);
                farCtrl.get(k).add(GalleryEvaluation.far(ctrlImp, th[k]));
                farThr.get(k).add(GalleryEvaluation.far(thrImp, th[k]));
            }
        }

        double score(Row r) {
            return t.apply(r.best(), r.second());
        }

        int frr(int k) {
            return rej[0][k] + rej[1][k];
        }

        int misaccepted(int k) {
            return mis[0][k] + mis[1][k];
        }

        int errors(int k) {
            return frr(k) + misaccepted(k);
        }

        int errors(int k, int frontal) {
            return rej[frontal][k] + mis[frontal][k];
        }

        int[][] far(List<List<int[]>> f, int k) {
            return f.get(k).toArray(int[][]::new);
        }
    }

    static Impostors impostors(Map<String, List<Double>> byPerson) {
        Map<String, double[]> m = new LinkedHashMap<>();
        byPerson.forEach((p, v) -> m.put(p, v.stream().mapToDouble(Double::doubleValue).toArray()));
        return new Impostors(m);
    }

    /** Разбор файла: строки → метрики по методам (порядок — первое появление; новые строки — сразу после исходных). */
    static final class Analysis {
        final List<String> header = new ArrayList<>();
        final Map<String, Stat> stats = new LinkedHashMap<>();
        final Set<String> configs = new TreeSet<>();
        long lines;
        private final Set<String> done = new HashSet<>();
        private final List<Row> group = new ArrayList<>();
        private String key;

        void read(BufferedReader in) throws IOException {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#")) {
                    header.add(l);
                    continue;
                }
                if (l.equals(COLUMNS)) continue;
                if (l.isEmpty()) continue;
                add(parse(l));
            }
            finish();
        }

        /** Завершить последнюю группу. */
        void finish() {
            flush();
        }

        void add(Row r) {
            lines++;
            String k = r.method() + "	" + r.s() + "	" + r.j();
            if (!k.equals(key)) {
                flush();
                if (!done.add(k)) throw new IllegalStateException("Группа (метод, s, j) встречается не подряд: " + k.replace('	', ' '));
                key = k;
            }
            group.add(r);
        }

        private void flush() {
            if (group.isEmpty()) return;
            Row first = group.get(0);
            configs.add(first.s() + "	" + first.j());
            stat(first.method(), Transform.NONE).add(group);
            for (NewRow n : NEW_ROWS) if (n.base().equals(first.method())) stat(n.id(), n.t()).add(group);
            for (NewRow n : MIRROR_ROWS) if (n.base().equals(first.method())) stat(n.id(), n.t()).add(group);
            group.clear();
        }

        private Stat stat(String id, Transform t) {
            return stats.computeIfAbsent(id, k -> new Stat(id, t));
        }
    }

    public static void main(String[] args) throws IOException {
        Path dir = Paths.get(System.getProperty("user.dir"), "reports", "faces", "fei");
        Path file = args.length > 0 && !args[0].isBlank() ? Paths.get(args[0].trim()) : dir.resolve(FILE);
        Analysis a = new Analysis();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16),
                StandardCharsets.UTF_8))) {
            a.read(in);
        }
        Files.createDirectories(dir);
        FarMethods.write(dir.resolve("fei_scores.txt"), report(a, file.toString()));
        new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8).println("Строк " + a.lines + ", конфигураций " + a.configs.size() + ", строк отчёта " + a.stats.size() + "; отчёт "
                + dir.resolve("fei_scores.txt"));
    }

    static StringBuilder report(Analysis a, String source) {
        StringBuilder t = new StringBuilder();
        t.append("Этап 5б: разбор файла оценок faces-fei (FeiScores, faces-fei-scores), без обучения\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл: ").append(source).append(" (строк попыток ").append(a.lines).append(")\n");
        t.append("Шапка файла:\n");
        for (String h : a.header) t.append("  ").append(h).append('\n');
        t.append(String.format(Locale.ROOT, "Конфигураций в файле: %d из %d.%n", a.configs.size(), FeiProtocol.CONFIGS));
        t.append("Порог (в каждой конфигурации, только по пороговым 40 чужим): основной — по снимкам: наибольший θ, при котором\n"
                + "принято (оценка ≤ θ) не больше ⌊FAR·n⌋ снимков пороговых чужих, FAR = 1 % (цель) и 5 %; справочно — по людям\n"
                + "(чужой принят, если принят хотя бы один его снимок; α = 0,05 — 2 человека, α = 0 — ни одного), как в FeiMethods.\n"
                + "Ошибка своих = FRR (отказ) + принят под чужим именем. Контроль 40 чужих — только FAR. Оговорка: снимки одного\n"
                + "человека коррелируют, FAR по снимкам оптимистичен (граница ↑95 по снимкам тоже).\n");
        t.append("Новые строки (argmin прежний, меняется только оценка для порога): ");
        for (NewRow n : NEW_ROWS) t.append(n.id()).append(" — ").append(n.about()).append("; ");
        t.append("для z отношение не определено (знак и ноль произвольны), поэтому разность.\n");
        t.append("Строки 4-sum-blk-both+* в файле есть все (в таблице fei_methods.txt — только условная строка).\n");
        if (a.stats.keySet().stream().anyMatch(FeiMethods::isMirror)) {
            t.append("Строки +mirror (режим faces.fei.mirror): обучение + отражённые кадры (FaceMirror, способ а), пробы без отражения; ");
            for (NewRow n : MIRROR_ROWS) t.append(n.id()).append(" — ").append(n.about()).append("; ");
            t.append("в конце таблиц.\n");
        }

        t.append("\n=== Основная таблица: порог по снимкам пороговых чужих ===\n");
        t.append("метод | ошибка своих FAR 1 % (FRR + под чужим) | FAR 5 % (FRR + под чужим) | анфас / поворот FAR 5 % | "
                + "FAR контроля снимки FAR 1 %: сумма; худш. (↑95) / мед. | FAR 5 %: сумма; худш. (↑95) / мед. | "
                + "справочно: ошибка своих, порог по людям α=0,05\n");
        for (Stat s : a.stats.values()) {
            t.append(String.join(" | ", s.id, errText(s, T1), errText(s, T5),
                    FeiMethods.pctOf(s.errors(T5, 1), s.attBy[1]) + " / " + FeiMethods.pctOf(s.errors(T5, 0), s.attBy[0]),
                    FeiMethods.attemptsSum(s.far(s.farCtrl, T1)) + "; " + FarMethods.attempts(s.far(s.farCtrl, T1)),
                    FeiMethods.attemptsSum(s.far(s.farCtrl, T5)) + "; " + FarMethods.attempts(s.far(s.farCtrl, T5)),
                    FeiMethods.pctOf(s.errors(P05), s.att))).append('\n');
        }

        t.append("\n=== Справочно: порог по людям (как fei_methods.txt; для сверки) ===\n");
        t.append("метод | ошибки своих α=0,05 (FRR + под чужим) | α=0 | анфас / поворот α=0,05 | анфас / поворот α=0 | "
                + "FAR контроля люди α=0,05: худш. (↑95) / мед. | FAR контроля попытки α=0,05, сумма | FAR контроля люди α=0: худш. / мед.\n");
        for (Stat s : a.stats.values()) {
            t.append(String.join(" | ", s.id,
                    String.format(Locale.ROOT, "%d + %d = %s", s.frr(P05), s.misaccepted(P05), FeiMethods.pctOf(s.errors(P05), s.att)),
                    FeiMethods.pctOf(s.errors(P0), s.att),
                    FeiMethods.pctOf(s.errors(P05, 1), s.attBy[1]) + " / " + FeiMethods.pctOf(s.errors(P05, 0), s.attBy[0]),
                    FeiMethods.pctOf(s.errors(P0, 1), s.attBy[1]) + " / " + FeiMethods.pctOf(s.errors(P0, 0), s.attBy[0]),
                    FarMethods.persons(s.far(s.farCtrl, P05)), FeiMethods.attemptsSum(s.far(s.farCtrl, P05)),
                    FeiMethods.personsNoUb(s.far(s.farCtrl, P0)))).append('\n');
        }

        t.append("\n=== Пороги и FAR на пороговых (набор порога) ===\n");
        t.append("метод | θ мед.: снимки 1 % / 5 % / люди α=0,05 / α=0 | FAR пороговых снимки, сумма: 1 % / 5 % | "
                + "FAR пороговых люди худш. / мед.: 1 % / 5 %\n");
        for (Stat s : a.stats.values()) {
            StringBuilder th = new StringBuilder();
            for (int k = 0; k < THRESHOLDS; k++) {
                if (k > 0) th.append(" / ");
                th.append(String.format(Locale.ROOT, "%.6f", FeiMethods.median(s.theta.get(k).stream().mapToDouble(Double::doubleValue).toArray())));
            }
            t.append(String.join(" | ", s.id, th, FeiMethods.attemptsSum(s.far(s.farThr, T1)) + " / " + FeiMethods.attemptsSum(s.far(s.farThr, T5)),
                    FeiMethods.personsNoUb(s.far(s.farThr, T1)) + " / " + FeiMethods.personsNoUb(s.far(s.farThr, T5)))).append('\n');
        }
        return t;
    }

    /** «FRR + под чужим = x/n = p %». */
    static String errText(Stat s, int k) {
        return String.format(Locale.ROOT, "%d + %d = %s", s.frr(k), s.misaccepted(k), FeiMethods.pctOf(s.errors(k), s.att));
    }

    /** Шапка файла (строки с «#») и строка столбцов. */
    static String header(String code, String data, int configs, List<String> methods) {
        StringBuilder t = new StringBuilder();
        t.append("# Этап 5б: оценки попыток faces-fei (FeiMethods), протокол FEI (PLAN.md, п. 3а)\n");
        t.append("# Код: коммит ").append(code).append('\n');
        t.append("# Данные: ").append(data).append('\n');
        t.append(String.format(Locale.ROOT, "# Конфигураций: %d из %d; s — разбиение (0…%d), j — индекс контрольного снимка своих в (4, 5, 6, 7, 11, 12, "
                + "13, 14)%n", configs, FeiProtocol.CONFIGS, FeiProtocol.SPLITS - 1));
        t.append("# Направление оценки: меньше — ближе; argmin по 100 своим галереи; попытка принята, если оценка лучшего ≤ θ\n");
        t.append("# Роли: own — свой (контрольный снимок), thr — пороговый чужой (40), ctrl — контрольный чужой (40); у чужих все отобранные "
                + "снимки\n");
        t.append("# true_score — оценка по истинному человеку (FeiMethods: свой отклонён, если она бесконечна или best_score > θ)\n");
        t.append("# 4-agr-*: no_agreement = 1 — согласие не достигнуто (все оценки +∞), 0 — достигнуто (best_score — сумма нормированных, "
                + "у прочих людей Double.MAX_VALUE, second_* «-»)\n");
        t.append("# Методы (").append(methods.size()).append("): ").append(String.join(", ", methods)).append('\n');
        t.append(COLUMNS).append('\n');
        return t.toString();
    }
}
