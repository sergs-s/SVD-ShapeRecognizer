package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeiScores.Stat;
import svd.recognizer.faces.FeiScores.Transform;

/**
 * PLAN.md, п. 4.2, «Разложение ошибки своих»: попытки своих частей A = 2c-mlda-ratio+pose, B = 4-wpca-cos-0-ratio+pose,
 * C = 2a95-fisher-ratio+pose по группам при порогах fei_scores.txt (Нейман – Пирсон по снимкам пороговых, FAR 1 % и
 * 5 %), оценка r = d₁/d₂ ({@link FeiScores#ratio(Row, int[])}):
 * (i) верный кандидат, принят; (ii) верный кандидат, отказ по порогу; (iii) неверный кандидат, отказ; (iv) неверный
 * кандидат, принят (ложное опознание); (v-а) отказ по правилу вырожденных случаев, случай 4 (d₂ = +∞); (v-б) бесконечная
 * оценка истинного (в FeiScores.Stat такой свой отклоняется). (v-а) проверяется раньше (v-б), обе — раньше (i)–(iv).
 * Ошибка своих = (ii) + (iii) + (iv) + (v-а) + (v-б); сверяется с FeiScores.Stat (иначе исключение).
 *
 * Деление: анфас |r| ≤ r0 / поворот (поза пробы, fei_selection.tsv) и справочно по номеру снимка (№11–14 / №4–7).
 * Запас до порога в группе (ii): log r − log θ. Бутстреп (faces.fei.bootstrap): люди FEI целиком, те же веса, что в
 * fei_bootstrap.txt, порог заново в каждой реплике и конфигурации (FeiBootstrap.threshold).
 *
 * Включается ключом faces.fei.decompose (FACES_FEI_DECOMPOSE = true) при заданном faces.fei.pose.r0; отчёт
 * fei_decompose.txt; прочие отчёты не меняются.
 *
 * @author ssv
 */
final class FeiDecompose {

    static final String FILE = "fei_decompose.txt";
    static final int G_OK = 0;
    static final int G_REJ = 1;
    static final int G_WRONG_REJ = 2;
    static final int G_WRONG_ACC = 3;
    static final int G_NO_SECOND = 4;
    static final int G_TRUE_INF = 5;
    static final int GROUPS = 6;
    static final String[] GROUP_NAMES = {"(i) верный, принят", "(ii) верный, отказ по порогу", "(iii) неверный, отказ",
        "(iv) неверный, принят", "(v-а) отказ: случай 4 (d₂ = +∞)", "(v-б) отказ: оценка истинного +∞"};
    /** Деления: все; анфас и поворот по позе (|r| ≤ r0); справочно по номеру снимка. */
    static final int SPLITS = 5;
    static final String[] SPLIT_NAMES = {"всего", "анфас |r| ≤ r0", "поворот |r| > r0", "справочно №11–14", "справочно №4–7"};
    /** Пороги: индексы FeiScores.FARS (FAR 1 %, 5 %). */
    static final int KS = 2;

    /** Свои и пороговые одной части в одной конфигурации. */
    static final class Cfg {
        int[] ownP;
        double[] ownR;
        boolean[] ownOk;
        boolean[] noSecond;
        boolean[] trueInf;
        boolean[] poseFrontal;
        boolean numberFrontal;
        /** Пороговые: r по возрастанию и человек каждого снимка. */
        int[] thrP;
        double[] thrR;
    }

    private final List<String> parts;
    private final FeiPose pose;
    private final double r0;
    private final FeiBootstrap boot;
    final Stat[] stats;
    final List<List<Cfg>> cfgs = new ArrayList<>();

    /** @param boot люди FEI для бутстрепа (строки в него не добавляются) или null */
    FeiDecompose(FeiPose pose, double r0, FeiBootstrap boot) {
        this(FeiFusion.PARTS, pose, r0, boot);
    }

    FeiDecompose(List<String> parts, FeiPose pose, double r0, FeiBootstrap boot) {
        this.parts = List.copyOf(parts);
        this.pose = pose;
        this.r0 = r0;
        this.boot = boot;
        stats = new Stat[parts.size()];
        for (int i = 0; i < parts.size(); i++) {
            stats[i] = new Stat(parts.get(i), Transform.RATIO);
            cfgs.add(new ArrayList<>());
        }
    }

    /** Попытки строки id в одной конфигурации (строки +pose разбора); не часть — пропускается. */
    void part(String id, List<Row> rows) {
        int i = parts.indexOf(id);
        if (i < 0) return;
        stats[i].add(rows);
        Cfg c = new Cfg();
        c.numberFrontal = FeiProtocol.frontal(rows.get(0).j());
        List<Row> own = rows.stream().filter(Row::own).toList();
        int n = own.size();
        c.ownP = new int[n];
        c.ownR = new double[n];
        c.ownOk = new boolean[n];
        c.noSecond = new boolean[n];
        c.trueInf = new boolean[n];
        c.poseFrontal = new boolean[n];
        for (int k = 0; k < n; k++) {
            Row r = own.get(k);
            int[] cs = new int[FeiScores.CASES];
            c.ownP[k] = boot == null ? -1 : boot.person(r.person());
            c.ownR[k] = FeiScores.ratio(r, cs);
            c.noSecond[k] = cs[FeiScores.CASE_NO_SECOND] > 0;
            c.trueInf[k] = Double.isInfinite(r.trueScore());
            c.ownOk[k] = r.bestId().equals(r.trueId());
            c.poseFrontal[k] = pose.absR(r.person(), r.image()) <= r0;
        }
        List<double[]> thr = new ArrayList<>();
        for (Row r : rows) {
            if (r.role().equals(FeiScores.THR)) thr.add(new double[] {boot == null ? -1 : boot.person(r.person()), FeiScores.ratio(r, null)});
        }
        thr.sort((x, y) -> Double.compare(x[1], y[1]));
        c.thrP = thr.stream().mapToInt(x -> (int) x[0]).toArray();
        c.thrR = thr.stream().mapToDouble(x -> x[1]).toArray();
        cfgs.get(i).add(c);
    }

    /** Группа своего k конфигурации c при пороге θ. */
    static int group(Cfg c, int k, double theta) {
        if (c.noSecond[k]) return G_NO_SECOND;
        if (c.trueInf[k]) return G_TRUE_INF;
        boolean acc = c.ownR[k] <= theta;
        if (c.ownOk[k]) return acc ? G_OK : G_REJ;
        return acc ? G_WRONG_ACC : G_WRONG_REJ;
    }

    /** Попадает ли свой k в деление sp. */
    static boolean in(Cfg c, int k, int sp) {
        return switch (sp) {
            case 0 -> true;
            case 1 -> c.poseFrontal[k];
            case 2 -> !c.poseFrontal[k];
            case 3 -> c.numberFrontal;
            default -> !c.numberFrontal;
        };
    }

    /** Порог конфигурации: веса null — ровно FaceEvaluation.neymanPearsonThreshold (как FeiScores.Stat), иначе с весами людей. */
    static double threshold(Cfg c, int[] w, int k) {
        return w == null ? FaceEvaluation.neymanPearsonThreshold(c.thrR, FeiScores.FARS[k]) : FeiBootstrap.threshold(c.thrR, c.thrP, w, FeiScores.FARS[k]);
    }

    /** Взвешенные числа попыток [порог][деление][группа]; w null — веса 1. */
    static double[][][] counts(List<Cfg> cs, int[] w) {
        double[][][] t = new double[KS][SPLITS][GROUPS];
        for (Cfg c : cs) {
            for (int k = 0; k < KS; k++) {
                double th = threshold(c, w, k);
                for (int q = 0; q < c.ownR.length; q++) {
                    int wt = w == null ? 1 : w[c.ownP[q]];
                    if (wt == 0) continue;
                    int g = group(c, q, th);
                    for (int sp = 0; sp < SPLITS; sp++) if (in(c, q, sp)) t[k][sp][g] += wt;
                }
            }
        }
        return t;
    }

    /** Доли групп, % [порог][деление][группа]. */
    static double[][][] shares(double[][][] t) {
        double[][][] p = new double[KS][SPLITS][GROUPS];
        for (int k = 0; k < KS; k++) {
            for (int sp = 0; sp < SPLITS; sp++) {
                double n = Arrays.stream(t[k][sp]).sum();
                for (int g = 0; g < GROUPS; g++) p[k][sp][g] = 100.0 * t[k][sp][g] / n;
            }
        }
        return p;
    }

    /** Запасы log r − log θ группы (ii) при пороге k в делении sp (веса 1), по возрастанию. */
    static double[] margins(List<Cfg> cs, int k, int sp) {
        List<Double> m = new ArrayList<>();
        for (Cfg c : cs) {
            double th = threshold(c, null, k);
            for (int q = 0; q < c.ownR.length; q++) {
                if (in(c, q, sp) && group(c, q, th) == G_REJ) m.add(Math.log(c.ownR[q]) - Math.log(th));
            }
        }
        return m.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    }

    /** Сверка с FeiScores.Stat: ошибка (ii)…(v-б) и число попыток; не совпало — исключение. */
    void check() {
        for (int i = 0; i < parts.size(); i++) {
            double[][][] t = counts(cfgs.get(i), null);
            int[] ks = {FeiScores.T1, FeiScores.T5};
            for (int k = 0; k < KS; k++) {
                double err = Arrays.stream(t[k][0]).sum() - t[k][0][G_OK];
                if (err != stats[i].errors(ks[k]) || Arrays.stream(t[k][0]).sum() != stats[i].att) {
                    throw new IllegalStateException("Разложение " + parts.get(i) + ": ошибка " + err + " ≠ FeiScores " + stats[i].errors(ks[k]));
                }
            }
        }
    }

    /** Реплики бутстрепа [часть][b] → доли [порог][деление][группа]. */
    double[][][][][] bootstrap(int b, long seed) {
        int[][] w = FeiBootstrap.weights(boot.size(), b, seed);
        double[][][][][] reps = new double[parts.size()][b][][][];
        for (int i = 0; i < parts.size(); i++) {
            List<Cfg> cs = cfgs.get(i);
            double[][][][] r = reps[i];
            IntStream.range(0, b).parallel().forEach(q -> r[q] = shares(counts(cs, w[q])));
        }
        return reps;
    }

    // ---------------------------------------------------------------- отчёт

    StringBuilder report(double[][][][][] reps, int b, long seed, List<String> header, String source) {
        check();
        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: разложение ошибки своих (FeiDecompose, faces-fei-scores с faces.fei.decompose), из файла оценок, без "
                + "переобучения\n");
        t.append("Код разбора: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Файл: ").append(source).append('\n');
        t.append("Шапка файла:\n");
        for (String h : header) t.append("  ").append(h).append('\n');
        t.append("Части (выбор модели по позе пробы, r0 = ").append(r0).append("): ");
        for (int i = 0; i < parts.size(); i++) t.append(i > 0 ? "; " : "").append(FeiFusion.letter(i)).append(" = ").append(parts.get(i));
        t.append(".\n");
        t.append("Пороги — ровно как в fei_scores.txt: в каждой конфигурации Нейман – Пирсон по снимкам 40 пороговых чужих, FAR 1 % и 5 %;\n"
                + "оценка r = d₁/d₂ (FeiScores.ratio, правило вырожденных случаев), принят — r ≤ θ. Группы своих: (i) верный кандидат\n"
                + "(лучший = истинный), принят; (ii) верный, отказ по порогу; (iii) неверный, отказ; (iv) неверный, принят (ложное\n"
                + "опознание); (v-а) отказ по правилу вырожденных случаев, случай 4 (d₂ = +∞, отношение не определено); (v-б) оценка\n"
                + "истинного (true_score) бесконечна. (v-а) проверяется раньше (v-б), обе — раньше (i)–(iv).\n"
                + "(v-б): true_score = +∞ в файле оценок пишет FeiMethods, когда оценка истинного в векторе попытки бесконечна; в\n"
                + "faces-fei так бывает только у 4-agr-* (FusionScorer: согласие не достигнуто — все оценки +∞), у сырых расстояний\n"
                + "частей A, B, C — нет. В fei_scores.txt (FeiScores.Stat) такой свой отклоняется (оценка +∞ > θ) и входит в FRR.\n"
                + "Ошибка своих = (ii) + (iii) + (iv) + (v-а) + (v-б) — сверена с FeiScores.Stat (то же, что fei_scores.txt) по обоим\n"
                + "порогам: совпало (иначе разбор остановился бы с ошибкой). Ошибка argmin = (iii) + (iv) + (v-а, v-б при неверном\n"
                + "кандидате) — без порога, то же множество 4000 попыток, что в fei_fusion.txt.\n"
                + "Деления: анфас |r| ≤ r0 / поворот |r| > r0 по позе пробы (fei_selection.tsv), как выбор модели в +pose; справочно —\n"
                + "по номеру снимка (№11–14 / №4–7), как в прежних таблицах (это другое деление).\n");
        if (reps != null) {
            t.append(String.format(Locale.ROOT, "Интервалы [2,5; 97,5] — бутстреп по людям FEI целиком (B = %d, seed = %d, те же веса, что в "
                    + "fei_bootstrap.txt); порог\nзаново в каждой реплике и конфигурации; доля — от взвешенного числа попыток деления.\n", b, seed));
        } else {
            t.append("Бутстреп выключен (faces.fei.bootstrap не задан).\n");
        }
        String[] farName = {"FAR 1 %", "FAR 5 %"};
        for (int i = 0; i < parts.size(); i++) {
            List<Cfg> cs = cfgs.get(i);
            double[][][] cnt = counts(cs, null);
            double[][][] pt = shares(cnt);
            for (int k = 0; k < KS; k++) {
                t.append("\n=== ").append(FeiFusion.letter(i)).append(" = ").append(parts.get(i)).append(", ").append(farName[k]).append(" ===\n");
                t.append("группа");
                for (String s : SPLIT_NAMES) t.append(" | ").append(s);
                t.append('\n');
                for (int g = 0; g <= GROUPS; g++) {
                    t.append(g < GROUPS ? GROUP_NAMES[g] : "ошибка своих (ii)+(iii)+(iv)+(v)");
                    for (int sp = 0; sp < SPLITS; sp++) {
                        int n = (int) Arrays.stream(cnt[k][sp]).sum();
                        int x = g < GROUPS ? (int) cnt[k][sp][g] : n - (int) cnt[k][sp][G_OK];
                        t.append(" | ").append(FeiMethods.pctOf(x, n));
                        if (reps != null) {
                            double[] v = new double[b];
                            for (int q = 0; q < b; q++) v[q] = g < GROUPS ? reps[i][q][k][sp][g] : 100 - reps[i][q][k][sp][G_OK];
                            double[] s = FeiBootstrap.sorted(v);
                            t.append(String.format(Locale.ROOT, " [%.1f; %.1f]", FeiBootstrap.quantile(s, 0.025), FeiBootstrap.quantile(s, 0.975)));
                        }
                    }
                    t.append('\n');
                }
            }
            int wrong = 0;
            int n = 0;
            for (Cfg c : cs) {
                for (int q = 0; q < c.ownR.length; q++) {
                    n++;
                    if (!c.ownOk[q]) wrong++;
                }
            }
            t.append("Ошибка argmin (неверный кандидат первого ранга, без порога): ").append(FeiMethods.pctOf(wrong, n))
                    .append("; ошибка своих по FeiScores.Stat: FAR 1 % ").append(stats[i].errors(FeiScores.T1)).append(", FAR 5 % ")
                    .append(stats[i].errors(FeiScores.T5)).append(" — совпадает с суммой групп.\n");
        }

        t.append("\n=== Запас до порога в группе (ii): log r − log θ (> 0), квартили [25 %; медиана; 75 %] (элемент ⌈p·n⌉), n ===\n");
        t.append("часть, порог");
        for (int sp = 0; sp < 3; sp++) t.append(" | ").append(SPLIT_NAMES[sp]);
        t.append('\n');
        for (int i = 0; i < parts.size(); i++) {
            for (int k = 0; k < KS; k++) {
                t.append(FeiFusion.letter(i)).append(", ").append(farName[k]);
                for (int sp = 0; sp < 3; sp++) {
                    double[] m = margins(cfgs.get(i), k, sp);
                    t.append(m.length == 0 ? " | n = 0" : String.format(Locale.ROOT, " | [%.4f; %.4f; %.4f], n = %d", FeiBootstrap.quantile(m, 0.25),
                            FeiBootstrap.quantile(m, 0.5), FeiBootstrap.quantile(m, 0.75), m.length));
                }
                t.append('\n');
            }
        }
        return t;
    }
}
