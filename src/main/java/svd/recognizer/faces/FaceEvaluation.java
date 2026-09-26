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
import nu.pattern.OpenCV;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * Пакетная оценка метода подпространств на базе лиц ORL по протоколу плана
 * работ (одна команда: {@code mvn compile exec:java@faces-eval}).
 *
 * Подпространство на каждого «своего» человека, класс — argmin ошибки
 * реконструкции ε. Решение «принять / отвергнуть» — по порогу на скоринг:
 * <ul>
 *   <li>{@code eps} — ε лучшего подпространства;</li>
 *   <li>{@code ratio} — ε₁/ε₂ (лучшее к второму).</li>
 * </ul>
 * Порог выбирается на валидации по критерию Неймана – Пирсона (наибольший
 * порог, при котором доля принятых чужих ≤ α), на контроле не подбирается:
 * <ul>
 *   <li>{@code per-fold} — по валидации своего фолда (5 чужих людей);</li>
 *   <li>{@code pooled} — один общий порог по объединённой валидации всех
 *       фолдов (20 чужих людей).</li>
 * </ul>
 * Разбиения — см. {@link EvaluationProtocol}. Варианты модели:
 * {без нормализации, эквализация гистограммы} × {единый k = N − 1,
 * k_c по энергии η = 0.90, η = 0.95}.
 *
 * Результат — в reports/faces: scores_&lt;вариант&gt;.csv (ε по каждому
 * снимку), metrics.csv (все метрики по фолдам и суммарно для сетки α),
 * report.txt (сводка при рабочем α).
 *
 * @author ssv
 */
public final class FaceEvaluation {

    /** Сетка ограничений FAR ≤ α на валидации. */
    static final double[] ALPHAS = {0.0, 0.02, 0.05, 0.10};
    /** Рабочее α для сравнения вариантов на ORL. */
    static final double WORKING_ALPHA = 0.05;
    /** Доли энергии для выбора k_c; 0 означает единый k = N − 1. */
    static final double[] ETAS = {0.0, 0.90, 0.95};
    /** Уровень доверия верхних границ FAR. */
    static final double CONFIDENCE = 0.95;

    enum Role { KNOWN_VAL, KNOWN_TEST, IMPOSTOR_VAL, IMPOSTOR_TEST }

    /** Скоринг для решения «принять / отвергнуть» (меньше — увереннее). */
    enum Score {
        EPS("eps"), RATIO("ratio");

        final String label;

        Score(String label) {
            this.label = label;
        }

        double of(Probe p) {
            return this == EPS ? p.bestError() : p.bestError() / p.secondError();
        }
    }

    /** Способ выбора порога на валидации. */
    enum ThresholdMode {
        PER_FOLD("per-fold"), POOLED("pooled");

        final String label;

        ThresholdMode(String label) {
            this.label = label;
        }
    }

    /** Результат сравнения одного снимка со всеми подпространствами фолда. */
    record Probe(int fold, Role role, int person, int image,
                 int bestPerson, double bestError, double secondError) {
        boolean correct() {
            return bestPerson == person;
        }
    }

    /** Итог одного варианта модели: снимки и выбранные k_c. */
    record VariantResult(String name, List<Probe> probes, List<Integer> ks) {}

    private FaceEvaluation() {}

    public static void main(String[] args) throws IOException {
        OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        EvaluationProtocol protocol = new EvaluationProtocol(
                settings.loadFacesSeed(), settings.loadFacesTrainPerPerson());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces");
        Files.createDirectories(outDir);

        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        List<VariantResult> results = new ArrayList<>();
        for (boolean equalize : new boolean[] {false, true}) {
            OrlDataset dataset = OrlDataset.load(datasetDir, equalize,
                    settings.loadFacesFrameWidth(), settings.loadFacesFrameHeight());
            for (double eta : ETAS) {
                String name = (equalize ? "eqhist" : "none") + "_"
                        + (eta == 0.0 ? "k" + (protocol.getTrainPerPerson() - 1)
                                      : String.format(Locale.ROOT, "eta%03d", Math.round(eta * 100)));
                System.out.println("Вариант " + name + "...");
                VariantResult result = runVariant(name, dataset, protocol, trainer, eta);
                writeScores(outDir.resolve("scores_" + name + ".csv"), result);
                results.add(result);
            }
        }
        Set<Integer> pooledParticipants = pooledParticipants(protocol);
        writeMetrics(outDir.resolve("metrics.csv"), results, pooledParticipants);
        writeReport(outDir.resolve("report.txt"), results, protocol, datasetDir, pooledParticipants);
        System.out.println("Готово: " + outDir);
    }

    static VariantResult runVariant(String name, OrlDataset dataset, EvaluationProtocol protocol,
                                    SubspaceTrainer trainer, double eta) {
        List<Probe> probes = new ArrayList<>();
        List<Integer> ks = new ArrayList<>();
        for (int fold = 0; fold < EvaluationProtocol.FOLDS; fold++) {
            int[] known = protocol.knownPersons(fold);
            double[][] means = new double[known.length][];
            double[][][] bases = new double[known.length][][];
            int[] kByModel = new int[known.length];
            for (int m = 0; m < known.length; m++) {
                int[] train = protocol.trainImages(known[m]);
                double[][] vectors = new double[train.length][];
                for (int j = 0; j < train.length; j++) {
                    vectors[j] = dataset.vector(known[m], train[j]);
                }
                SubspaceModel model = eta == 0.0
                        ? trainer.train(vectors, train.length - 1)
                        : trainer.trainByEnergy(vectors, eta);
                means[m] = model.getMeanVector();
                bases[m] = model.getBasisMatrix();
                kByModel[m] = model.getK();
                ks.add(model.getK());
            }
            for (int person : known) {
                for (int image : protocol.validationImages(person)) {
                    probes.add(score(fold, Role.KNOWN_VAL, person, image, dataset, known, means, bases, kByModel));
                }
                for (int image : protocol.testImages(person)) {
                    probes.add(score(fold, Role.KNOWN_TEST, person, image, dataset, known, means, bases, kByModel));
                }
            }
            for (int person : protocol.validationImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    probes.add(score(fold, Role.IMPOSTOR_VAL, person, image, dataset, known, means, bases, kByModel));
                }
            }
            for (int person : protocol.testImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    probes.add(score(fold, Role.IMPOSTOR_TEST, person, image, dataset, known, means, bases, kByModel));
                }
            }
        }
        return new VariantResult(name, probes, ks);
    }

    static Probe score(int fold, Role role, int person, int image, OrlDataset dataset,
                               int[] known, double[][] means, double[][][] bases, int[] ks) {
        double[] x = dataset.vector(person, image);
        int best = -1;
        double bestError = Double.MAX_VALUE;
        double secondError = Double.MAX_VALUE;
        for (int m = 0; m < known.length; m++) {
            double error = SubspaceRecognizer.reconstructionError(x, means[m], bases[m], ks[m]);
            if (error < bestError) {
                secondError = bestError;
                bestError = error;
                best = known[m];
            } else if (error < secondError) {
                secondError = error;
            }
        }
        return new Probe(fold, role, person, image, best, bestError, secondError);
    }

    /** Люди, входящие в объединённую валидацию как чужие (20 человек). */
    static Set<Integer> pooledParticipants(EvaluationProtocol protocol) {
        Set<Integer> people = new HashSet<>();
        for (int fold = 0; fold < EvaluationProtocol.FOLDS; fold++) {
            for (int p : protocol.validationImpostors(fold)) {
                people.add(p);
            }
        }
        return people;
    }

    /**
     * Порог по Нейману – Пирсону: наибольший θ, при котором среди чужих
     * валидации принято (скоринг ≤ θ) не больше ⌊α·n⌋.
     */
    static double neymanPearsonThreshold(double[] impostorScores, double alpha) {
        double[] sorted = impostorScores.clone();
        Arrays.sort(sorted);
        int allowed = (int) Math.floor(alpha * sorted.length + 1e-9);
        if (allowed >= sorted.length) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.nextDown(sorted[allowed]);
    }

    /**
     * Односторонняя верхняя граница доли по точному биномиальному методу
     * (Клоппер – Пирсон): p, при котором P(X ≤ x | n, p) = 1 − confidence.
     * При x = 0 равна 1 − (1 − confidence)^(1/n) ≈ 3/n («правило трёх»).
     */
    static double binomialUpperBound(int x, int n, double confidence) {
        if (n == 0) {
            return Double.NaN;
        }
        if (x >= n) {
            return 1.0;
        }
        double target = 1.0 - confidence;
        double lo = x / (double) n;
        double hi = 1.0;
        for (int iter = 0; iter < 200; iter++) {
            double mid = (lo + hi) / 2;
            if (binomialCdf(x, n, mid) > target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2;
    }

    private static double binomialCdf(int x, int n, double p) {
        if (p <= 0.0) {
            return 1.0;
        }
        if (p >= 1.0) {
            return x >= n ? 1.0 : 0.0;
        }
        double logP = Math.log(p);
        double logQ = Math.log1p(-p);
        double logCoef = 0.0; // log C(n, 0)
        double sum = 0.0;
        for (int i = 0; i <= x; i++) {
            if (i > 0) {
                logCoef += Math.log(n - i + 1) - Math.log(i);
            }
            sum += Math.exp(logCoef + i * logP + (n - i) * logQ);
        }
        return Math.min(1.0, sum);
    }

    /** Счётчики метрик одного фолда (или суммы фолдов). */
    static final class Counts {
        double theta = Double.NaN;
        int knownTest, correctAccept, falseReject, confusion, argminCorrect;
        int impostorTest, falseAccept;
        int impostorPeople, impostorPeopleAccepted;
        int knownVal, valFalseReject, impostorVal, valFalseAccept;
        /** Свои контроля: участники объединённой валидации (g1) и остальные (g2). */
        int knownTestG1, falseRejectG1, knownTestG2, falseRejectG2;

        void add(Counts o) {
            knownTest += o.knownTest; correctAccept += o.correctAccept;
            falseReject += o.falseReject; confusion += o.confusion;
            argminCorrect += o.argminCorrect;
            impostorTest += o.impostorTest; falseAccept += o.falseAccept;
            impostorPeople += o.impostorPeople; impostorPeopleAccepted += o.impostorPeopleAccepted;
            knownVal += o.knownVal; valFalseReject += o.valFalseReject;
            impostorVal += o.impostorVal; valFalseAccept += o.valFalseAccept;
            knownTestG1 += o.knownTestG1; falseRejectG1 += o.falseRejectG1;
            knownTestG2 += o.knownTestG2; falseRejectG2 += o.falseRejectG2;
        }
    }

    /**
     * Метрики по фолдам и суммарно (последний элемент массива) для одного
     * скоринга, способа выбора порога и α.
     */
    static Counts[] evaluate(List<Probe> probes, Score score, ThresholdMode mode, double alpha,
                             Set<Integer> pooledParticipants) {
        double pooledTheta = neymanPearsonThreshold(probes.stream()
                .filter(p -> p.role() == Role.IMPOSTOR_VAL)
                .mapToDouble(score::of).toArray(), alpha);
        Counts[] result = new Counts[EvaluationProtocol.FOLDS + 1];
        Counts total = new Counts();
        for (int fold = 0; fold < EvaluationProtocol.FOLDS; fold++) {
            final int f = fold;
            Counts c = new Counts();
            c.theta = mode == ThresholdMode.POOLED ? pooledTheta
                    : neymanPearsonThreshold(probes.stream()
                            .filter(p -> p.fold() == f && p.role() == Role.IMPOSTOR_VAL)
                            .mapToDouble(score::of).toArray(), alpha);
            Set<Integer> people = new HashSet<>();
            Set<Integer> acceptedPeople = new HashSet<>();
            for (Probe p : probes) {
                if (p.fold() != fold) {
                    continue;
                }
                boolean accepted = score.of(p) <= c.theta;
                switch (p.role()) {
                    case KNOWN_TEST -> {
                        c.knownTest++;
                        if (p.correct()) c.argminCorrect++;
                        if (!accepted) c.falseReject++;
                        else if (p.correct()) c.correctAccept++;
                        else c.confusion++;
                        if (pooledParticipants.contains(p.person())) {
                            c.knownTestG1++;
                            if (!accepted) c.falseRejectG1++;
                        } else {
                            c.knownTestG2++;
                            if (!accepted) c.falseRejectG2++;
                        }
                    }
                    case IMPOSTOR_TEST -> {
                        c.impostorTest++;
                        people.add(p.person());
                        if (accepted) {
                            c.falseAccept++;
                            acceptedPeople.add(p.person());
                        }
                    }
                    case KNOWN_VAL -> {
                        c.knownVal++;
                        if (!accepted) c.valFalseReject++;
                    }
                    case IMPOSTOR_VAL -> {
                        c.impostorVal++;
                        if (accepted) c.valFalseAccept++;
                    }
                }
            }
            c.impostorPeople = people.size();
            c.impostorPeopleAccepted = acceptedPeople.size();
            result[fold] = c;
            total.add(c);
        }
        if (mode == ThresholdMode.POOLED) {
            total.theta = pooledTheta;
        }
        result[EvaluationProtocol.FOLDS] = total;
        return result;
    }

    private static double ratio(int a, int b) {
        return b == 0 ? Double.NaN : a / (double) b;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static void writeScores(Path file, VariantResult result) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.println("fold,role,person,image,best_person,best_error,second_error,argmin_correct,ratio");
            for (Probe p : result.probes()) {
                out.printf(Locale.ROOT, "%d,%s,s%d,%d,s%d,%.10f,%.10f,%b,%.10f%n",
                        p.fold() + 1, p.role(), p.person() + 1, p.image() + 1,
                        p.bestPerson() + 1, p.bestError(), p.secondError(),
                        p.role() == Role.KNOWN_VAL || p.role() == Role.KNOWN_TEST ? p.correct() : false,
                        Score.RATIO.of(p));
            }
        }
    }

    private static void writeMetrics(Path file, List<VariantResult> results,
                                     Set<Integer> pooledParticipants) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.println("variant,score,threshold_mode,alpha,fold,theta,known_test,correct_accept,frr,"
                    + "confusion,argmin_accuracy,impostor_test,false_accept,far,far_upper95,far_rule_of_three,"
                    + "val_frr,val_far,impostor_people,impostor_people_accepted,far_people,far_people_upper95,"
                    + "frr_val_participants,frr_non_participants");
            for (VariantResult r : results) {
                for (Score score : Score.values()) {
                    for (ThresholdMode mode : ThresholdMode.values()) {
                        for (double alpha : ALPHAS) {
                            Counts[] cs = evaluate(r.probes(), score, mode, alpha, pooledParticipants);
                            for (int i = 0; i < cs.length; i++) {
                                String fold = i < EvaluationProtocol.FOLDS ? String.valueOf(i + 1) : "total";
                                metricsRow(out, r.name(), score, mode, alpha, fold, cs[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void metricsRow(PrintWriter out, String variant, Score score, ThresholdMode mode,
                                   double alpha, String fold, Counts c) {
        out.printf(Locale.ROOT, "%s,%s,%s,%.2f,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.4f,%s,%.4f,%.4f,"
                        + "%d,%d,%.4f,%.4f,%.4f,%.4f%n",
                variant, score.label, mode.label, alpha, fold,
                Double.isNaN(c.theta) ? "" : String.format(Locale.ROOT, "%.6f", c.theta),
                c.knownTest, ratio(c.correctAccept, c.knownTest), ratio(c.falseReject, c.knownTest),
                ratio(c.confusion, c.knownTest), ratio(c.argminCorrect, c.knownTest),
                c.impostorTest, c.falseAccept, ratio(c.falseAccept, c.impostorTest),
                binomialUpperBound(c.falseAccept, c.impostorTest, CONFIDENCE),
                c.falseAccept == 0 ? String.format(Locale.ROOT, "%.4f", 3.0 / c.impostorTest) : "",
                ratio(c.valFalseReject, c.knownVal), ratio(c.valFalseAccept, c.impostorVal),
                c.impostorPeople, c.impostorPeopleAccepted, ratio(c.impostorPeopleAccepted, c.impostorPeople),
                binomialUpperBound(c.impostorPeopleAccepted, c.impostorPeople, CONFIDENCE),
                ratio(c.falseRejectG1, c.knownTestG1), ratio(c.falseRejectG2, c.knownTestG2));
    }

    private static String upperWithRule3(int x, int n) {
        String s = fmt(binomialUpperBound(x, n, CONFIDENCE));
        return x == 0 ? s + " (3/n=" + fmt(3.0 / n) + ")" : s;
    }

    private static void writeReport(Path file, List<VariantResult> results, EvaluationProtocol protocol,
                                     Path datasetDir, Set<Integer> pooledParticipants) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.println("Оценка метода подпространств на базе лиц ORL");
            out.println("База: " + datasetDir + " (The Database of Faces, AT&T Laboratories Cambridge)");
            out.printf(Locale.ROOT, "seed = %d; фолдов %d; свои: 30 человек, снимки %d обучение / %d валидация / %d контроль;%n",
                    protocol.getSeed(), EvaluationProtocol.FOLDS, protocol.getTrainPerPerson(),
                    EvaluationProtocol.VALIDATION_PER_PERSON,
                    OrlDataset.IMAGES_PER_PERSON - protocol.getTrainPerPerson() - EvaluationProtocol.VALIDATION_PER_PERSON);
            out.printf(Locale.ROOT, "чужие: 10 человек, %d валидация / %d контроль (по 10 снимков).%n",
                    EvaluationProtocol.VALIDATION_IMPOSTORS, 10 - EvaluationProtocol.VALIDATION_IMPOSTORS);
            out.println();
            out.println("Скоринг: eps — ε лучшего подпространства; ratio — ε₁/ε₂ (лучшее к второму). Класс — argmin ε.");
            out.println("Порог — по Нейману – Пирсону на валидации (наибольший порог с FAR_вал ≤ α), на контроле не подбирается:");
            out.println("  per-fold — по валидации своего фолда (5 чужих людей, 50 попыток);");
            out.println("  pooled   — один порог по объединённой валидации 4 фолдов (20 чужих людей, 200 попыток).");
            out.println("FAR↑95 — точная (Клоппер – Пирсон) односторонняя верхняя 95 % граница; при 0 ошибок также 3/n.");
            out.println("  по попыткам — снимки одного человека коррелируют, граница оптимистична;");
            out.println("  по людям — чужой считается принятым, если принят хотя бы один его снимок.");
            out.println();
            out.println("Нюанс pooled: 15 из 30 своих в каждом фолде входят в объединённую валидацию как чужие (их снимки");
            out.println("сравнивались с галереями других фолдов). Чужие контроля с валидацией не пересекаются, оценка FAR");
            out.println("чистая. Влияние на FRR — по сравнению групп: g1 — свои, участвовавшие в объединённой валидации,");
            out.println("g2 — не участвовавшие. Если FRR групп близки, влиянием можно пренебречь.");
            out.println();

            out.printf(Locale.ROOT, "=== Сводка при α = %.2f (контроль, сумма 4 фолдов) ===%n", WORKING_ALPHA);
            out.println("вариант         скоринг  порог     argmin  FRR     путан.  FAR (x/n)        FAR↑95   FAR люди  люди↑95  FRR_вал");
            for (VariantResult r : results) {
                for (Score score : Score.values()) {
                    for (ThresholdMode mode : ThresholdMode.values()) {
                        Counts t = evaluate(r.probes(), score, mode, WORKING_ALPHA, pooledParticipants)[EvaluationProtocol.FOLDS];
                        out.printf(Locale.ROOT, "%-15s %-8s %-9s %.4f  %.4f  %.4f  %.4f (%d/%d)  %.4f   %d/%-6d  %.4f   %.4f%n",
                                r.name(), score.label, mode.label,
                                ratio(t.argminCorrect, t.knownTest), ratio(t.falseReject, t.knownTest),
                                ratio(t.confusion, t.knownTest), ratio(t.falseAccept, t.impostorTest),
                                t.falseAccept, t.impostorTest,
                                binomialUpperBound(t.falseAccept, t.impostorTest, CONFIDENCE),
                                t.impostorPeopleAccepted, t.impostorPeople,
                                binomialUpperBound(t.impostorPeopleAccepted, t.impostorPeople, CONFIDENCE),
                                ratio(t.valFalseReject, t.knownVal));
                    }
                }
            }
            out.println();

            out.printf(Locale.ROOT, "=== По фолдам при α = %.2f ===%n", WORKING_ALPHA);
            for (VariantResult r : results) {
                int kMin = r.ks().stream().mapToInt(Integer::intValue).min().orElse(0);
                int kMax = r.ks().stream().mapToInt(Integer::intValue).max().orElse(0);
                double kMean = r.ks().stream().mapToInt(Integer::intValue).average().orElse(0);
                out.printf(Locale.ROOT, "%n--- %s: k_c min %d / среднее %.2f / max %d%n", r.name(), kMin, kMean, kMax);
                for (Score score : Score.values()) {
                    for (ThresholdMode mode : ThresholdMode.values()) {
                        Counts[] cs = evaluate(r.probes(), score, mode, WORKING_ALPHA, pooledParticipants);
                        out.printf("  [%s, %s]%n", score.label, mode.label);
                        out.print("  фолд  θ          верно   FRR     путан.  FAR (x/n)       FAR↑95                  люди x/n  люди↑95   FRR_вал");
                        out.println(mode == ThresholdMode.POOLED ? "  FRR g1 (x/n)      FRR g2 (x/n)" : "");
                        for (int i = 0; i < cs.length; i++) {
                            Counts c = cs[i];
                            String fold = i < EvaluationProtocol.FOLDS ? String.valueOf(i + 1) : "всего";
                            String theta = Double.isNaN(c.theta) ? "—" : String.format(Locale.ROOT, "%.4f", c.theta);
                            out.printf(Locale.ROOT, "  %-5s %-10s %.4f  %.4f  %.4f  %.4f (%d/%d)  %-22s  %d/%-6d  %.4f    %.4f",
                                    fold, theta,
                                    ratio(c.correctAccept, c.knownTest), ratio(c.falseReject, c.knownTest),
                                    ratio(c.confusion, c.knownTest), ratio(c.falseAccept, c.impostorTest),
                                    c.falseAccept, c.impostorTest, upperWithRule3(c.falseAccept, c.impostorTest),
                                    c.impostorPeopleAccepted, c.impostorPeople,
                                    binomialUpperBound(c.impostorPeopleAccepted, c.impostorPeople, CONFIDENCE),
                                    ratio(c.valFalseReject, c.knownVal));
                            if (mode == ThresholdMode.POOLED) {
                                out.printf(Locale.ROOT, "  %.4f (%d/%d)  %.4f (%d/%d)",
                                        ratio(c.falseRejectG1, c.knownTestG1), c.falseRejectG1, c.knownTestG1,
                                        ratio(c.falseRejectG2, c.knownTestG2), c.falseRejectG2, c.knownTestG2);
                            }
                            out.println();
                        }
                    }
                }
            }
        }
    }
}
