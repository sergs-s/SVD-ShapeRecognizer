package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.IntStream;
import nu.pattern.OpenCV;
import svd.recognizer.faces.FaceEvaluation.Role;
import svd.recognizer.faces.PpcaEvaluation.Gallery;
import svd.recognizer.faces.PpcaEvaluation.Outcome;
import svd.recognizer.faces.PpcaEvaluation.ProbeData;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * Двумерное правило допуска на ORL (шаг 4б, пункт «д»; одна команда:
 * {@code mvn compile exec:java@faces-2d}).
 *
 * Допуск, если ε₁ ≤ θ₁ и ε₁/ε₂ ≤ θ₂; класс — argmin ε. θ₁ = ∞ — база
 * (только ε₁/ε₂). Правило лишь добавляет отказы, поэтому выигрыш возможен
 * только за счёт ослабления θ₂ при жёстком θ₁.
 *
 * Выбор пары — только на объединённой валидации (все фолды, для N = 8 — и все
 * ротации): θ₁ — из сетки {∞; квантили 90, 95, 99 % ε₁ всех валидационных
 * снимков своих}; для каждого θ₁ порог θ₂ по Нейману – Пирсону: наибольший,
 * при котором чужих валидации с ε₁ ≤ θ₁ и ε₁/ε₂ ≤ θ₂ не больше ⌊α·n⌋ (n —
 * все валидационные попытки чужих); затем θ₁ с наименьшим FRR своих на
 * валидации, при равенстве — θ₁ = ∞.
 *
 * Разбиения и ε — те же, что в {@link PpcaEvaluation}: основной протокол
 * (N = 5) и протокол кривой при N = 8.
 *
 * @author ssv
 */
public final class TwoThresholdEvaluation {

    static final double[] ALPHAS = PpcaEvaluation.ALPHAS;
    /** Квантили ε₁ своих на валидации; NaN — без ограничения (θ₁ = ∞). */
    static final double[] QUANTILES = {Double.NaN, 0.90, 0.95, 0.99};

    /** ε₁, ε₁/ε₂ и argmin для каждого снимка: [галерея][снимок]. */
    record Scores(double[][] e1, double[][] ratio, int[][] predicted) {}

    /** Пара порогов и её показатели на валидации. */
    record Pair(double quantile, double theta1, double theta2, double valFrr, double impostorCutByTheta1) {}

    private TwoThresholdEvaluation() {}

    public static void main(String[] args) throws IOException {
        long start = System.currentTimeMillis();
        OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        long seed = settings.loadFacesSeed();
        EvaluationProtocol protocol = new EvaluationProtocol(seed, settings.loadFacesTrainPerPerson());
        OrlDataset dataset = OrlDataset.load(datasetDir, false,
                settings.loadFacesFrameWidth(), settings.loadFacesFrameHeight());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces");
        Files.createDirectories(outDir);
        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        int[][] boot = TrainingSizeCurve.bootstrapSamples(OrlDataset.PERSONS, PpcaEvaluation.BOOTSTRAP, seed);

        System.out.println("Основной протокол (N = " + protocol.getTrainPerPerson() + ")...");
        List<Gallery> main = IntStream.range(0, EvaluationProtocol.FOLDS).parallel()
                .mapToObj(f -> PpcaEvaluation.mainGallery(dataset, protocol, trainer, f)).toList();
        System.out.println("Протокол кривой (N = " + PpcaEvaluation.CURVE_N + ")...");
        List<Gallery> curve = IntStream.range(0, EvaluationProtocol.FOLDS * TrainingSizeCurve.ROTATIONS).parallel()
                .mapToObj(t -> PpcaEvaluation.curveGallery(dataset, protocol, trainer,
                        t / TrainingSizeCurve.ROTATIONS, t % TrainingSizeCurve.ROTATIONS))
                .toList();

        StringBuilder text = new StringBuilder();
        List<String> csv = new ArrayList<>();
        report(text, csv, "основной, N = " + protocol.getTrainPerPerson(), "main_n5", main, boot);
        report(text, csv, "кривая, N = " + PpcaEvaluation.CURVE_N, "curve_n" + PpcaEvaluation.CURVE_N, curve, boot);
        writeFiles(outDir, text, csv, protocol, datasetDir);
        System.out.printf(Locale.ROOT, "Готово: %s (%.0f с)%n", outDir,
                (System.currentTimeMillis() - start) / 1000.0);
    }

    static Scores scores(List<Gallery> galleries) {
        double[][] e1 = new double[galleries.size()][];
        double[][] ratio = new double[galleries.size()][];
        int[][] predicted = new int[galleries.size()][];
        for (int gi = 0; gi < galleries.size(); gi++) {
            Gallery g = galleries.get(gi);
            int size = g.probes().size();
            e1[gi] = new double[size];
            ratio[gi] = new double[size];
            predicted[gi] = new int[size];
            for (int pi = 0; pi < size; pi++) {
                double[] eps = g.probes().get(pi).eps();
                double best = Double.MAX_VALUE;
                double second = Double.MAX_VALUE;
                int bestPerson = -1;
                for (int m = 0; m < eps.length; m++) {
                    if (eps[m] < best) {
                        second = best;
                        best = eps[m];
                        bestPerson = g.persons()[m];
                    } else if (eps[m] < second) {
                        second = eps[m];
                    }
                }
                e1[gi][pi] = best;
                ratio[gi][pi] = best / second;
                predicted[gi][pi] = bestPerson;
            }
        }
        return new Scores(e1, ratio, predicted);
    }

    /** Квантиль по ближайшему рангу: наименьшее значение, не меньше которого доля q выборки. */
    static double quantile(double[] values, double q) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(q * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
    }

    /**
     * θ₂ по Нейману – Пирсону при фиксированном θ₁: наибольший порог, при котором
     * чужих с ε₁ ≤ θ₁ и отношением ≤ θ₂ не больше ⌊α·n⌋, n — все попытки чужих.
     */
    static double theta2(double[] impostorE1, double[] impostorRatio, double theta1, double alpha) {
        List<Double> passed = new ArrayList<>();
        for (int i = 0; i < impostorE1.length; i++) {
            if (impostorE1[i] <= theta1) {
                passed.add(impostorRatio[i]);
            }
        }
        int allowed = (int) Math.floor(alpha * impostorE1.length + 1e-9);
        if (passed.size() <= allowed) {
            return Double.POSITIVE_INFINITY;
        }
        double[] sorted = passed.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return Math.nextDown(sorted[allowed]);
    }

    /** Вся сетка пар на валидации; первый элемент — база (θ₁ = ∞). */
    static List<Pair> grid(double[] ownE1, double[] ownRatio, double[] impostorE1, double[] impostorRatio,
                           double alpha) {
        List<Pair> pairs = new ArrayList<>();
        for (double q : QUANTILES) {
            double theta1 = Double.isNaN(q) ? Double.POSITIVE_INFINITY : quantile(ownE1, q);
            double theta2 = theta2(impostorE1, impostorRatio, theta1, alpha);
            int rejected = 0;
            for (int i = 0; i < ownE1.length; i++) {
                if (ownE1[i] > theta1 || ownRatio[i] > theta2) {
                    rejected++;
                }
            }
            int cut = 0;
            for (double e : impostorE1) {
                if (e > theta1) cut++;
            }
            pairs.add(new Pair(q, theta1, theta2, rejected / (double) ownE1.length, cut / (double) impostorE1.length));
        }
        return pairs;
    }

    /** Выбор: наименьший FRR_вал; при равенстве — более ранний элемент сетки (база первой). */
    static Pair select(List<Pair> pairs) {
        Pair best = pairs.get(0);
        for (Pair p : pairs) {
            if (p.valFrr() < best.valFrr()) {
                best = p;
            }
        }
        return best;
    }

    /** Значения валидации: {ε₁ своих, отношение своих, ε₁ чужих, отношение чужих}. */
    private static double[][] validation(List<Gallery> galleries, Scores s) {
        List<double[]> own = new ArrayList<>();
        List<double[]> imp = new ArrayList<>();
        for (int gi = 0; gi < galleries.size(); gi++) {
            List<ProbeData> probes = galleries.get(gi).probes();
            for (int pi = 0; pi < probes.size(); pi++) {
                double[] v = {s.e1()[gi][pi], s.ratio()[gi][pi]};
                if (probes.get(pi).role() == Role.KNOWN_VAL) own.add(v);
                else if (probes.get(pi).role() == Role.IMPOSTOR_VAL) imp.add(v);
            }
        }
        return new double[][] {
            own.stream().mapToDouble(v -> v[0]).toArray(), own.stream().mapToDouble(v -> v[1]).toArray(),
            imp.stream().mapToDouble(v -> v[0]).toArray(), imp.stream().mapToDouble(v -> v[1]).toArray()
        };
    }

    /** Метрики на контроле для пары порогов. */
    static Outcome evaluate(List<Gallery> galleries, Scores s, Pair pair) {
        int groups = galleries.stream().mapToInt(Gallery::group).max().orElse(0) + 1;
        int[] attempts = new int[OrlDataset.PERSONS];
        int[] rejects = new int[OrlDataset.PERSONS];
        int[] correct = new int[OrlDataset.PERSONS];
        int[] farPerGroup = new int[groups];
        int[] impostorAttempts = new int[groups];
        List<Set<Integer>> accepted = new ArrayList<>();
        List<Set<Integer>> impostors = new ArrayList<>();
        for (int grp = 0; grp < groups; grp++) {
            accepted.add(new HashSet<>());
            impostors.add(new HashSet<>());
        }
        int valKnown = 0;
        int valRejected = 0;
        for (int gi = 0; gi < galleries.size(); gi++) {
            Gallery g = galleries.get(gi);
            for (int pi = 0; pi < g.probes().size(); pi++) {
                ProbeData p = g.probes().get(pi);
                boolean ok = s.e1()[gi][pi] <= pair.theta1() && s.ratio()[gi][pi] <= pair.theta2();
                switch (p.role()) {
                    case KNOWN_VAL -> {
                        valKnown++;
                        if (!ok) valRejected++;
                    }
                    case KNOWN_TEST -> {
                        attempts[p.person()]++;
                        if (!ok) rejects[p.person()]++;
                        if (s.predicted()[gi][pi] == p.person()) correct[p.person()]++;
                    }
                    case IMPOSTOR_TEST -> {
                        impostorAttempts[g.group()]++;
                        impostors.get(g.group()).add(p.person());
                        if (ok) {
                            farPerGroup[g.group()]++;
                            accepted.get(g.group()).add(p.person());
                        }
                    }
                    default -> { }
                }
            }
        }
        Set<Integer> any = new HashSet<>();
        int[] peoplePerGroup = new int[groups];
        for (int grp = 0; grp < groups; grp++) {
            any.addAll(accepted.get(grp));
            peoplePerGroup[grp] = accepted.get(grp).size();
        }
        return new Outcome(valRejected / (double) valKnown, attempts, rejects, correct, farPerGroup,
                peoplePerGroup, any.size(), impostorAttempts[0], impostors.get(0).size(), groups);
    }

    private static void report(StringBuilder text, List<String> csv, String name, String id,
                               List<Gallery> galleries, int[][] boot) {
        Scores s = scores(galleries);
        double[][] val = validation(galleries, s);
        text.append(String.format(Locale.ROOT, "%n==================== Протокол: %s ====================%n", name));
        text.append(String.format(Locale.ROOT, "Валидация: своих %d снимков, чужих %d попыток.%n", val[0].length, val[2].length));
        for (double alpha : ALPHAS) {
            List<Pair> pairs = grid(val[0], val[1], val[2], val[3], alpha);
            Pair chosen = select(pairs);
            Outcome base = evaluate(galleries, s, pairs.get(0));
            text.append(String.format(Locale.ROOT, "%n--- α = %.2f ---%n", alpha));
            text.append(String.format(Locale.ROOT, "Регрессия: база (θ₁ = ∞) — FRR %s %% (%d/%d); должно совпасть с отчётом (б).%n",
                    pct(base.frr()), Arrays.stream(base.rejects()).sum(), Arrays.stream(base.attempts()).sum()));
            text.append("Сетка (выбор только по FRR_вал; FRR на контроле — для прозрачности):\n");
            text.append("  θ₁ (квантиль)   θ₁        θ₂        FRR_вал  чужих вал. отсечено θ₁  FRR контроль\n");
            for (Pair p : pairs) {
                Outcome o = evaluate(galleries, s, p);
                text.append(String.format(Locale.ROOT, "  %-15s %-9s %-9s %5s    %5s %%                  %5s%s%n",
                        Double.isNaN(p.quantile()) ? "∞ (база)" : String.format(Locale.ROOT, "%.0f %%", 100 * p.quantile()),
                        num(p.theta1()), num(p.theta2()), pct(p.valFrr()), pct(p.impostorCutByTheta1()), pct(o.frr()),
                        p == chosen ? "   ← выбрано" : ""));
                csv.add(String.format(Locale.ROOT, "%s,%.2f,%s,%s,%s,%.4f,%.4f,%.4f,%b",
                        id, alpha, Double.isNaN(p.quantile()) ? "inf" : String.format(Locale.ROOT, "%.2f", p.quantile()),
                        num(p.theta1()), num(p.theta2()), p.valFrr(), p.impostorCutByTheta1(), o.frr(), p == chosen));
            }
            Outcome o = evaluate(galleries, s, chosen);
            text.append("  правило           FRR, % [95 % ДИ]      Δ к базе, п.п. [ДИ]    argmin,%  FAR ср/макс,%  ↑95 худш.  люди (хоть раз/худш.)  ↑95 люди\n");
            row(text, "база (θ₁ = ∞)", base, null, boot);
            row(text, chosen == pairs.get(0) ? "выбрано = база" : "двумерное", o, base, boot);
        }
    }

    private static void row(StringBuilder text, String label, Outcome o, Outcome base, int[][] boot) {
        double[] ci = PpcaEvaluation.frrCi(o, boot);
        String diff = "—";
        if (base != null) {
            double[] d = PpcaEvaluation.diffCi(o, base, boot);
            String mark = d[1] < 0 ? " *" : d[0] > 0 ? " !" : "";
            diff = String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", 100 * (o.frr() - base.frr()), 100 * d[0], 100 * d[1], mark);
        }
        int farWorst = Arrays.stream(o.farPerGroup()).max().orElse(0);
        int peopleWorst = Arrays.stream(o.peoplePerGroup()).max().orElse(0);
        double farMean = Arrays.stream(o.farPerGroup()).average().orElse(0) / o.impostorAttemptsPerGroup();
        text.append(String.format(Locale.ROOT,
                "  %-17s %5s [%5s; %5s]   %-22s %5s     %4s/%4s      %5s      %d/%d / %d/%d           %5s%n",
                label, pct(o.frr()), pct(ci[0]), pct(ci[1]), diff, pct(o.accuracy()),
                pct(farMean), pct(farWorst / (double) o.impostorAttemptsPerGroup()),
                pct(FaceEvaluation.binomialUpperBound(farWorst, o.impostorAttemptsPerGroup(), FaceEvaluation.CONFIDENCE)),
                o.peopleAny(), o.impostorPeoplePerGroup(), peopleWorst, o.impostorPeoplePerGroup(),
                pct(FaceEvaluation.binomialUpperBound(peopleWorst, o.impostorPeoplePerGroup(), FaceEvaluation.CONFIDENCE))));
    }

    private static String num(double v) {
        return Double.isInfinite(v) ? "∞" : String.format(Locale.ROOT, "%.4f", v);
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.ROOT, "%.1f", 100 * v);
    }

    private static void writeFiles(Path outDir, StringBuilder text, List<String> csv,
                                   EvaluationProtocol protocol, Path datasetDir) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("two_threshold.txt"), StandardCharsets.UTF_8))) {
            out.println("Двумерное правило допуска на ORL (шаг 4б, пункт «д»)");
            out.println("База: " + datasetDir + " (The Database of Faces, AT&T Laboratories Cambridge)");
            out.printf(Locale.ROOT, "seed = %d; k = N − 1; без нормализации; бутстреп по людям: %d повторов.%n",
                    protocol.getSeed(), PpcaEvaluation.BOOTSTRAP);
            out.println("Допуск, если ε₁ ≤ θ₁ и ε₁/ε₂ ≤ θ₂; класс — argmin ε; θ₁ = ∞ — база (только ε₁/ε₂).");
            out.println("θ₁ — из {∞; квантили 90, 95, 99 % ε₁ всех валидационных снимков своих}; θ₂ — Нейман – Пирсон:");
            out.println("чужих валидации с ε₁ ≤ θ₁ и ε₁/ε₂ ≤ θ₂ не больше ⌊α·n⌋ (n — все попытки чужих валидации);");
            out.println("выбор θ₁ — наименьший FRR своих на валидации, при равенстве — база. Всё по объединённой валидации");
            out.println("(все фолды; для N = 8 — и все ротации), на контроле ничего не подбирается.");
            out.println("Δ — парная разность FRR на тех же бутстреп-выборках людей: * — значимо лучше, ! — значимо хуже.");
            out.print(text);
        }
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("two_threshold.csv"), StandardCharsets.UTF_8))) {
            out.println("protocol,alpha,theta1_quantile,theta1,theta2,val_frr,impostor_val_cut_by_theta1,test_frr,selected");
            for (String row : csv) {
                out.println(row);
            }
        }
    }
}
