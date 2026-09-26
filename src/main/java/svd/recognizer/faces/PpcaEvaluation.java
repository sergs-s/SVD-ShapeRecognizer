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
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * PPCA-скоринг на ORL (шаг 4б, пункт «б»; пункт «а» входит сюда как w = 0;
 * одна команда: {@code mvn compile exec:java@faces-ppca}).
 *
 * Для человека c (k = N − 1, без нормализации): y = Bᵀ(x − μ),
 * ε² = ‖x − μ − B y‖², λᵢ = σᵢ²/(N − 1) — из проекций обучающих снимков,
 * λ̃ᵢ = (1 − γ)·λᵢ + γ·λ̄ (сжатие к среднему), s² — скользящим контролем внутри
 * обучающего набора (подпространство по N − 1 снимкам, k′ = N − 2, остаток на
 * отложенном снимке, среднее εⱼ² / (D − k)), свой для каждого человека или
 * общий — среднее по галерее.
 *
 * Скоринг d_w = ε²/s² + w·Σ yᵢ²/λ̃ᵢ, w ∈ {0; 1; 10; 100}, γ ∈ {0; 0,5; 1};
 * класс — argmin d_w; варианты — d₁ и d₁/d₂. Базовый вариант — ε₁/ε₂.
 * Отдельно — полное отрицательное логарифмическое правдоподобие PPCA
 * (w = 1, γ = 0, свой s²): d + (D − k)·ln s² + Σ ln λᵢ.
 *
 * Параметры (w, γ) и порог выбираются только на объединённой валидации:
 * для каждого α — сочетание с наименьшим FRR своих на валидации при пороге
 * Неймана – Пирсона (FAR_вал ≤ α); при равенстве — меньший w, затем меньший γ.
 *
 * Протоколы: основной (N = 5, свои 5/2/3) и протокол кривой при N = 8
 * (ротация контрольного снимка o[r], обучение o[r+1…r+8], валидация своих —
 * оставшийся снимок o[r+9]); порог и параметры — по валидации всех фолдов
 * (и ротаций).
 *
 * @author ssv
 */
public final class PpcaEvaluation {

    static final double[] ALPHAS = {0.02, 0.05};
    static final double[] WEIGHTS = {0, 1, 10, 100};
    static final double[] GAMMAS = {0, 0.5, 1};
    static final int CURVE_N = 8;
    static final int BOOTSTRAP = TrainingSizeCurve.BOOTSTRAP;

    /** Вариант скоринга. */
    enum Variant {
        BASE("ε₁/ε₂ (база)", false, false),
        D_OWN("d, s² свой", true, false),
        D_GLOBAL("d, s² общий", true, false),
        RATIO_OWN("d₁/d₂, s² свой", true, true),
        RATIO_GLOBAL("d₁/d₂, s² общий", true, true),
        NLL_OWN("−ln p, s² свой (w=1, γ=0)", false, false);

        final String label;
        final boolean tuned;
        final boolean ratio;

        Variant(String label, boolean tuned, boolean ratio) {
            this.label = label;
            this.tuned = tuned;
            this.ratio = ratio;
        }

        boolean ownS2() {
            return this == D_OWN || this == RATIO_OWN || this == NLL_OWN;
        }
    }

    /** Снимок, сравненный со всеми моделями галереи. */
    record ProbeData(Role role, int person, double[] eps, double[][] y) {}

    /**
     * Галерея одного фолда (и ротации): люди, их λ и s², снимки.
     * group — номер ротации (для основного протокола 0).
     */
    record Gallery(int group, int[] persons, double[][] lambda, double[] s2own, double s2global,
                   int dim, int k, List<ProbeData> probes) {}

    /** Итог одного сочетания на одном протоколе. */
    record Outcome(double valFrr, int[] attempts, int[] rejects, int[] correct,
                   int[] farPerGroup, int[] peoplePerGroup, int peopleAny,
                   int impostorAttemptsPerGroup, int impostorPeoplePerGroup, int groups) {
        double frr() {
            return Arrays.stream(rejects).sum() / (double) Arrays.stream(attempts).sum();
        }

        double accuracy() {
            return Arrays.stream(correct).sum() / (double) Arrays.stream(attempts).sum();
        }
    }

    private PpcaEvaluation() {}

    public static void main(String[] args) throws IOException {
        long start = System.currentTimeMillis();
        OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        long seed = settings.loadFacesSeed();
        EvaluationProtocol protocol = new EvaluationProtocol(seed, settings.loadFacesTrainPerPerson());
        OrlDataset dataset = OrlDataset.load(datasetDir, false);
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces");
        Files.createDirectories(outDir);
        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        int[][] boot = TrainingSizeCurve.bootstrapSamples(OrlDataset.PERSONS, BOOTSTRAP, seed);

        System.out.println("Основной протокол (N = " + protocol.getTrainPerPerson() + ")...");
        List<Gallery> main = new ArrayList<>(IntStream.range(0, EvaluationProtocol.FOLDS).parallel()
                .mapToObj(f -> mainGallery(dataset, protocol, trainer, f)).toList());
        System.out.println("Протокол кривой (N = " + CURVE_N + ")...");
        List<Gallery> curve = new ArrayList<>(IntStream.range(0, EvaluationProtocol.FOLDS * TrainingSizeCurve.ROTATIONS)
                .parallel()
                .mapToObj(t -> curveGallery(dataset, protocol, trainer,
                        t / TrainingSizeCurve.ROTATIONS, t % TrainingSizeCurve.ROTATIONS))
                .toList());

        StringBuilder text = new StringBuilder();
        List<String> csv = new ArrayList<>();
        report(text, csv, "основной, N = " + protocol.getTrainPerPerson(), main, boot, false);
        report(text, csv, "кривая, N = " + CURVE_N, curve, boot, true);
        writeFiles(outDir, text, csv, protocol, datasetDir);
        System.out.printf(Locale.ROOT, "Готово: %s (%.0f с)%n", outDir,
                (System.currentTimeMillis() - start) / 1000.0);
    }

    static Gallery mainGallery(OrlDataset dataset, EvaluationProtocol protocol, SubspaceTrainer trainer, int fold) {
        int[] known = protocol.knownPersons(fold);
        int[][] train = new int[known.length][];
        for (int m = 0; m < known.length; m++) {
            train[m] = protocol.trainImages(known[m]);
        }
        List<int[]> probeSpec = new ArrayList<>();
        for (int person : known) {
            for (int image : protocol.validationImages(person)) {
                probeSpec.add(new int[] {Role.KNOWN_VAL.ordinal(), person, image});
            }
            for (int image : protocol.testImages(person)) {
                probeSpec.add(new int[] {Role.KNOWN_TEST.ordinal(), person, image});
            }
        }
        addImpostors(probeSpec, protocol, fold);
        return buildGallery(dataset, trainer, 0, known, train, probeSpec);
    }

    static Gallery curveGallery(OrlDataset dataset, EvaluationProtocol protocol, SubspaceTrainer trainer,
                                int fold, int r) {
        int[] known = protocol.knownPersons(fold);
        int[][] train = new int[known.length][];
        List<int[]> probeSpec = new ArrayList<>();
        for (int m = 0; m < known.length; m++) {
            int[] order = protocol.imageOrder(known[m]);
            train[m] = TrainingSizeCurve.trainImages(order, r, CURVE_N);
            probeSpec.add(new int[] {Role.KNOWN_VAL.ordinal(), known[m], validationImage(order, r)});
            probeSpec.add(new int[] {Role.KNOWN_TEST.ordinal(), known[m], TrainingSizeCurve.testImage(order, r)});
        }
        addImpostors(probeSpec, protocol, fold);
        return buildGallery(dataset, trainer, r, known, train, probeSpec);
    }

    /** Валидация своих в протоколе кривой при N = 8: снимок вне обучения и контроля. */
    static int validationImage(int[] order, int r) {
        return order[(r + CURVE_N + 1) % order.length];
    }

    private static void addImpostors(List<int[]> probeSpec, EvaluationProtocol protocol, int fold) {
        for (int person : protocol.validationImpostors(fold)) {
            for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                probeSpec.add(new int[] {Role.IMPOSTOR_VAL.ordinal(), person, image});
            }
        }
        for (int person : protocol.testImpostors(fold)) {
            for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                probeSpec.add(new int[] {Role.IMPOSTOR_TEST.ordinal(), person, image});
            }
        }
    }

    private static Gallery buildGallery(OrlDataset dataset, SubspaceTrainer trainer, int group,
                                        int[] known, int[][] train, List<int[]> probeSpec) {
        int models = known.length;
        int n = train[0].length;
        int k = n - 1;
        int dim = dataset.vector(0, 0).length;
        double[][] means = new double[models][];
        double[][][] bases = new double[models][][];
        double[][] lambda = new double[models][];
        double[] s2own = new double[models];
        for (int m = 0; m < models; m++) {
            double[][] vectors = new double[n][];
            for (int j = 0; j < n; j++) {
                vectors[j] = dataset.vector(known[m], train[m][j]);
            }
            SubspaceModel model = trainer.train(vectors, k);
            means[m] = model.getMeanVector();
            bases[m] = model.getBasisMatrix();
            lambda[m] = eigenvalues(vectors, means[m], bases[m], model.getK());
            s2own[m] = looNoiseVariance(trainer, vectors, k, dim);
        }
        double s2global = Arrays.stream(s2own).average().orElse(Double.NaN);
        List<ProbeData> probes = new ArrayList<>();
        for (int[] spec : probeSpec) {
            double[] x = dataset.vector(spec[1], spec[2]);
            double[] eps = new double[models];
            double[][] y = new double[models][];
            for (int m = 0; m < models; m++) {
                eps[m] = SubspaceRecognizer.reconstructionError(x, means[m], bases[m], k);
                y[m] = project(x, means[m], bases[m], k);
            }
            probes.add(new ProbeData(Role.values()[spec[0]], spec[1], eps, y));
        }
        return new Gallery(group, known, lambda, s2own, s2global, dim, k, probes);
    }

    /** Проекции y = Bᵀ(x − μ). */
    static double[] project(double[] x, double[] mean, double[][] basis, int k) {
        double[] y = new double[k];
        for (int i = 0; i < x.length; i++) {
            double c = x[i] - mean[i];
            for (int j = 0; j < k; j++) {
                y[j] += basis[i][j] * c;
            }
        }
        return y;
    }

    /** λⱼ = Σ по обучающим снимкам yⱼ² / (n − 1) — равно σⱼ²/(n − 1). */
    static double[] eigenvalues(double[][] vectors, double[] mean, double[][] basis, int k) {
        double[] lambda = new double[k];
        for (double[] v : vectors) {
            double[] y = project(v, mean, basis, k);
            for (int j = 0; j < k; j++) {
                lambda[j] += y[j] * y[j];
            }
        }
        for (int j = 0; j < k; j++) {
            lambda[j] /= vectors.length - 1;
        }
        return lambda;
    }

    /**
     * Дисперсия шума на одно измерение скользящим контролем: для каждого
     * снимка — подпространство по остальным (k′ = n − 2), остаток на нём;
     * среднее εⱼ² / (dim − k).
     */
    static double looNoiseVariance(SubspaceTrainer trainer, double[][] vectors, int k, int dim) {
        int n = vectors.length;
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double[][] others = new double[n - 1][];
            for (int i = 0, t = 0; i < n; i++) {
                if (i != j) {
                    others[t++] = vectors[i];
                }
            }
            SubspaceModel loo = trainer.train(others, n - 2);
            double e = SubspaceRecognizer.reconstructionError(vectors[j], loo.getMeanVector(),
                    loo.getBasisMatrix(), loo.getK());
            sum += e * e;
        }
        return sum / n / (dim - k);
    }

    /** Регуляризованные λ̃ = (1 − γ)·λ + γ·λ̄. */
    static double[] shrink(double[] lambda, double gamma) {
        double mean = Arrays.stream(lambda).average().orElse(0);
        double[] result = new double[lambda.length];
        for (int i = 0; i < lambda.length; i++) {
            result[i] = (1 - gamma) * lambda[i] + gamma * mean;
        }
        return result;
    }

    /** Оценка модели m для снимка p в варианте v. Меньше — ближе. */
    static double modelScore(Gallery g, ProbeData p, int m, Variant v, double w, double[] lambdaTilde) {
        if (v == Variant.BASE) {
            return p.eps()[m];
        }
        double s2 = v.ownS2() ? g.s2own()[m] : g.s2global();
        double eps2 = p.eps()[m] * p.eps()[m];
        double inner = 0.0;
        double[] y = p.y()[m];
        for (int i = 0; i < y.length; i++) {
            inner += y[i] * y[i] / lambdaTilde[i];
        }
        if (v == Variant.NLL_OWN) {
            double logDet = (g.dim() - g.k()) * Math.log(s2);
            for (double l : lambdaTilde) {
                logDet += Math.log(l);
            }
            return eps2 / s2 + inner + logDet;
        }
        return eps2 / s2 + w * inner;
    }

    /** {оценка снимка, предсказанный человек}. */
    static double[] probeScore(Gallery g, ProbeData p, Variant v, double w, double[][] lambdaTilde) {
        double best = Double.MAX_VALUE;
        double second = Double.MAX_VALUE;
        int bestPerson = -1;
        for (int m = 0; m < g.persons().length; m++) {
            double s = modelScore(g, p, m, v, w, lambdaTilde[m]);
            if (s < best) {
                second = best;
                best = s;
                bestPerson = g.persons()[m];
            } else if (s < second) {
                second = s;
            }
        }
        double value = (v == Variant.BASE || v.ratio) ? best / second : best;
        return new double[] {value, bestPerson};
    }

    /** Оценки всех снимков всех галерей для одного сочетания: [галерея][снимок] → {оценка, человек}. */
    static double[][][] scoreAll(List<Gallery> galleries, Variant v, double w, double gamma) {
        double[][][] result = new double[galleries.size()][][];
        for (int gi = 0; gi < galleries.size(); gi++) {
            Gallery g = galleries.get(gi);
            double[][] lambdaTilde = new double[g.lambda().length][];
            for (int m = 0; m < lambdaTilde.length; m++) {
                lambdaTilde[m] = shrink(g.lambda()[m], gamma);
            }
            result[gi] = new double[g.probes().size()][];
            for (int pi = 0; pi < g.probes().size(); pi++) {
                result[gi][pi] = probeScore(g, g.probes().get(pi), v, w, lambdaTilde);
            }
        }
        return result;
    }

    /**
     * Метрики при пороге Неймана – Пирсона на объединённой валидации.
     * perGroupThreshold — порог отдельно для каждой группы (ротации),
     * объединяя фолды (схема кривой «в»; только для проверки регрессии).
     */
    static Outcome evaluate(List<Gallery> galleries, double[][][] scores, double alpha, boolean perGroupThreshold) {
        int groups = galleries.stream().mapToInt(Gallery::group).max().orElse(0) + 1;
        double[] thetaByGroup = new double[groups];
        for (int grp = 0; grp < groups; grp++) {
            List<Double> impostorVal = new ArrayList<>();
            for (int gi = 0; gi < galleries.size(); gi++) {
                if (perGroupThreshold && galleries.get(gi).group() != grp) {
                    continue;
                }
                List<ProbeData> probes = galleries.get(gi).probes();
                for (int pi = 0; pi < probes.size(); pi++) {
                    if (probes.get(pi).role() == Role.IMPOSTOR_VAL) {
                        impostorVal.add(scores[gi][pi][0]);
                    }
                }
            }
            thetaByGroup[grp] = FaceEvaluation.neymanPearsonThreshold(
                    impostorVal.stream().mapToDouble(Double::doubleValue).toArray(), alpha);
        }
        int[] attempts = new int[OrlDataset.PERSONS];
        int[] rejects = new int[OrlDataset.PERSONS];
        int[] correct = new int[OrlDataset.PERSONS];
        int[] farPerGroup = new int[groups];
        List<Set<Integer>> acceptedPeople = new ArrayList<>();
        List<Set<Integer>> impostorPeople = new ArrayList<>();
        int[] impostorAttempts = new int[groups];
        for (int grp = 0; grp < groups; grp++) {
            acceptedPeople.add(new HashSet<>());
            impostorPeople.add(new HashSet<>());
        }
        int valKnown = 0;
        int valRejected = 0;
        for (int gi = 0; gi < galleries.size(); gi++) {
            Gallery g = galleries.get(gi);
            double theta = thetaByGroup[g.group()];
            for (int pi = 0; pi < g.probes().size(); pi++) {
                ProbeData p = g.probes().get(pi);
                boolean accepted = scores[gi][pi][0] <= theta;
                switch (p.role()) {
                    case KNOWN_VAL -> {
                        valKnown++;
                        if (!accepted) valRejected++;
                    }
                    case KNOWN_TEST -> {
                        attempts[p.person()]++;
                        if (!accepted) rejects[p.person()]++;
                        if ((int) scores[gi][pi][1] == p.person()) correct[p.person()]++;
                    }
                    case IMPOSTOR_TEST -> {
                        impostorAttempts[g.group()]++;
                        impostorPeople.get(g.group()).add(p.person());
                        if (accepted) {
                            farPerGroup[g.group()]++;
                            acceptedPeople.get(g.group()).add(p.person());
                        }
                    }
                    default -> { }
                }
            }
        }
        Set<Integer> any = new HashSet<>();
        int[] peoplePerGroup = new int[groups];
        for (int grp = 0; grp < groups; grp++) {
            any.addAll(acceptedPeople.get(grp));
            peoplePerGroup[grp] = acceptedPeople.get(grp).size();
        }
        return new Outcome(valRejected / (double) valKnown, attempts, rejects, correct, farPerGroup,
                peoplePerGroup, any.size(), impostorAttempts[0], impostorPeople.get(0).size(), groups);
    }

    /** Выбранное сочетание (w, γ) и его итог. */
    record Choice(double w, double gamma, Outcome outcome) {}

    static Choice select(List<Gallery> galleries, Variant v, double alpha, List<String> grid, String protocolName) {
        if (!v.tuned) {
            double w = v == Variant.NLL_OWN ? 1 : 0;
            return new Choice(w, 0, evaluate(galleries, scoreAll(galleries, v, w, 0), alpha, false));
        }
        Choice best = null;
        for (double w : WEIGHTS) {
            for (double gamma : GAMMAS) {
                Outcome o = evaluate(galleries, scoreAll(galleries, v, w, gamma), alpha, false);
                grid.add(String.format(Locale.ROOT, "%s,%.2f,%s,%s,%s,%.4f,%.4f,,,grid,%.4f,,,,",
                        protocolName, alpha, v.name(), fmtNum(w), fmtNum(gamma), o.valFrr(), o.frr(), o.accuracy()));
                if (best == null || o.valFrr() < best.outcome().valFrr()) {
                    best = new Choice(w, gamma, o);
                }
            }
        }
        return best;
    }

    /** Бутстреп-ДИ FRR и парной разности FRR(a) − FRR(b) по людям. */
    static double[] frrCi(Outcome o, int[][] boot) {
        double[] values = new double[boot.length];
        for (int b = 0; b < boot.length; b++) {
            values[b] = TrainingSizeCurve.bootRatio(boot[b], o.rejects(), o.attempts());
        }
        return TrainingSizeCurve.percentile95(values);
    }

    static double[] diffCi(Outcome a, Outcome b, int[][] boot) {
        double[] values = new double[boot.length];
        for (int i = 0; i < boot.length; i++) {
            values[i] = TrainingSizeCurve.bootRatio(boot[i], a.rejects(), a.attempts())
                    - TrainingSizeCurve.bootRatio(boot[i], b.rejects(), b.attempts());
        }
        return TrainingSizeCurve.percentile95(values);
    }

    private static void report(StringBuilder text, List<String> csv, String name, List<Gallery> galleries,
                               int[][] boot, boolean curve) {
        text.append(String.format(Locale.ROOT, "%n==================== Протокол: %s ====================%n", name));
        int galleriesCount = galleries.size();
        double[] s2 = galleries.stream().flatMapToDouble(g -> Arrays.stream(g.s2own())).sorted().toArray();
        text.append(String.format(Locale.ROOT, "Галерей %d; s² свой (на одно измерение): медиана %.3e, мин %.3e, макс %.3e (разброс ×%.1f)%n",
                galleriesCount, s2[s2.length / 2], s2[0], s2[s2.length - 1], s2[s2.length - 1] / s2[0]));
        appendTermMedians(text, galleries);

        for (double alpha : ALPHAS) {
            text.append(String.format(Locale.ROOT, "%n--- α = %.2f ---%n", alpha));
            Choice base = select(galleries, Variant.BASE, alpha, csv, csvId(curve));
            if (curve) {
                Outcome perRotation = evaluate(galleries, scoreAll(galleries, Variant.BASE, 0, 0), alpha, true);
                text.append(String.format(Locale.ROOT,
                        "Регрессия: ε₁/ε₂ с порогом по ротации (схема кривой «в») — FRR %s %% (%d/%d).%n",
                        pct(perRotation.frr()), Arrays.stream(perRotation.rejects()).sum(),
                        Arrays.stream(perRotation.attempts()).sum()));
            } else {
                text.append(String.format(Locale.ROOT,
                        "Регрессия: ε₁/ε₂ — FRR %s %% (%d/%d), должно совпасть с основным отчётом (ratio, pooled).%n",
                        pct(base.outcome().frr()), Arrays.stream(base.outcome().rejects()).sum(),
                        Arrays.stream(base.outcome().attempts()).sum()));
            }
            text.append("вариант                     w     γ     FRR_вал  FRR, % [95 % ДИ]      Δ к ε₁/ε₂, п.п. [ДИ]    Δ к w=0, п.п. [ДИ]      argmin,%  FAR ср/макс,%  ↑95 худш.  люди (хоть раз/худш.)  ↑95 люди\n");
            for (Variant v : Variant.values()) {
                Choice c = select(galleries, v, alpha, csv, csvId(curve));
                Outcome o = c.outcome();
                double[] ci = frrCi(o, boot);
                String vsBase = v == Variant.BASE ? "—" : diffText(o, base.outcome(), boot);
                String vsW0 = "—";
                if (v.tuned) {
                    Outcome w0 = evaluate(galleries, scoreAll(galleries, v, 0, 0), alpha, false);
                    vsW0 = c.w() == 0 ? "выбран w=0" : diffText(o, w0, boot);
                }
                int farWorst = Arrays.stream(o.farPerGroup()).max().orElse(0);
                int peopleWorst = Arrays.stream(o.peoplePerGroup()).max().orElse(0);
                double farMean = Arrays.stream(o.farPerGroup()).average().orElse(0) / o.impostorAttemptsPerGroup();
                text.append(String.format(Locale.ROOT,
                        "%-27s %-5s %-5s %5s    %5s [%5s; %5s]   %-22s  %-22s  %5s     %4s/%4s      %5s      %d/%d / %d/%d           %5s%n",
                        v.label, v.tuned || v == Variant.NLL_OWN ? fmtNum(c.w()) : "—",
                        v.tuned || v == Variant.NLL_OWN ? fmtNum(c.gamma()) : "—",
                        pct(o.valFrr()), pct(o.frr()), pct(ci[0]), pct(ci[1]), vsBase, vsW0, pct(o.accuracy()),
                        pct(farMean), pct(farWorst / (double) o.impostorAttemptsPerGroup()),
                        pct(FaceEvaluation.binomialUpperBound(farWorst, o.impostorAttemptsPerGroup(), FaceEvaluation.CONFIDENCE)),
                        o.peopleAny(), o.impostorPeoplePerGroup(), peopleWorst, o.impostorPeoplePerGroup(),
                        pct(FaceEvaluation.binomialUpperBound(peopleWorst, o.impostorPeoplePerGroup(), FaceEvaluation.CONFIDENCE))));
                csv.add(String.format(Locale.ROOT, "%s,%.2f,%s,%s,%s,%.4f,%.4f,%.4f,%.4f,selected,%.4f,%d,%d,%d,%d",
                        csvId(curve), alpha, v.name(), fmtNum(c.w()), fmtNum(c.gamma()), o.valFrr(), o.frr(), ci[0], ci[1],
                        o.accuracy(), farWorst, o.impostorAttemptsPerGroup(), peopleWorst, o.peopleAny()));
            }
            if (!curve) {
                text.append("  (основной протокол: одна группа — FAR по сумме 4 фолдов, 200 попыток / 20 людей)\n");
            }
        }
        appendNllDiagnostic(text, galleries);
    }

    /** Идентификатор протокола для CSV (без запятых). */
    private static String csvId(boolean curve) {
        return curve ? "curve_n" + CURVE_N : "main_n5";
    }

    private static String diffText(Outcome a, Outcome b, int[][] boot) {
        double[] ci = diffCi(a, b, boot);
        double d = a.frr() - b.frr();
        String mark = ci[1] < 0 ? " *" : ci[0] > 0 ? " !" : "";
        return String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", 100 * d, 100 * ci[0], 100 * ci[1], mark);
    }

    /** Медианы слагаемых d у своих (модель своего человека) и чужих (ближайшая по ε модель), s² свой. */
    private static void appendTermMedians(StringBuilder text, List<Gallery> galleries) {
        List<double[]> own = new ArrayList<>();
        List<double[]> imp = new ArrayList<>();
        for (Gallery g : galleries) {
            double[][][] lt = new double[GAMMAS.length][g.lambda().length][];
            for (int gi = 0; gi < GAMMAS.length; gi++) {
                for (int m = 0; m < g.lambda().length; m++) {
                    lt[gi][m] = shrink(g.lambda()[m], GAMMAS[gi]);
                }
            }
            for (ProbeData p : g.probes()) {
                int m;
                if (p.role() == Role.KNOWN_TEST) {
                    m = indexOf(g.persons(), p.person());
                } else if (p.role() == Role.IMPOSTOR_TEST) {
                    m = 0;
                    for (int i = 1; i < p.eps().length; i++) {
                        if (p.eps()[i] < p.eps()[m]) m = i;
                    }
                } else {
                    continue;
                }
                double[] terms = new double[1 + GAMMAS.length];
                terms[0] = p.eps()[m] * p.eps()[m] / g.s2own()[m];
                for (int gi = 0; gi < GAMMAS.length; gi++) {
                    double inner = 0;
                    for (int i = 0; i < p.y()[m].length; i++) {
                        inner += p.y()[m][i] * p.y()[m][i] / lt[gi][m][i];
                    }
                    terms[1 + gi] = inner;
                }
                (p.role() == Role.KNOWN_TEST ? own : imp).add(terms);
            }
        }
        text.append("Медианы слагаемых (s² свой): ε²/s²; Σ y²/λ̃ при γ = 0 / 0,5 / 1\n");
        text.append(String.format(Locale.ROOT, "  свои (модель своего человека):  %s%n", medians(own)));
        text.append(String.format(Locale.ROOT, "  чужие (ближайшая по ε модель):  %s%n", medians(imp)));
    }

    private static String medians(List<double[]> rows) {
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < rows.get(0).length; c++) {
            final int col = c;
            double[] values = rows.stream().mapToDouble(r -> r[col]).sorted().toArray();
            sb.append(String.format(Locale.ROOT, c == 0 ? "%.0f;" : " %.1f", values[values.length / 2]));
        }
        return sb.toString();
    }

    /** Проверка ожидаемого вырождения −ln p: как часто argmin — человек с наименьшим s² галереи. */
    private static void appendNllDiagnostic(StringBuilder text, List<Gallery> galleries) {
        double[][][] scores = scoreAll(galleries, Variant.NLL_OWN, 1, 0);
        int total = 0;
        int minS2 = 0;
        for (int gi = 0; gi < galleries.size(); gi++) {
            Gallery g = galleries.get(gi);
            int argMinS2 = 0;
            for (int m = 1; m < g.s2own().length; m++) {
                if (g.s2own()[m] < g.s2own()[argMinS2]) argMinS2 = m;
            }
            for (int pi = 0; pi < g.probes().size(); pi++) {
                if (g.probes().get(pi).role() == Role.IMPOSTOR_VAL) {
                    continue;
                }
                total++;
                if ((int) scores[gi][pi][1] == g.persons()[argMinS2]) minS2++;
            }
        }
        text.append(String.format(Locale.ROOT,
                "%nДиагностика −ln p: argmin совпадает с человеком наименьшего s² галереи в %d из %d снимков (%s %%;"
                        + " при случайном выборе ≈ %s %%).%n",
                minS2, total, pct(minS2 / (double) total), pct(1.0 / galleries.get(0).persons().length)));
    }

    private static int indexOf(int[] array, int value) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == value) return i;
        }
        return -1;
    }

    private static String fmtNum(double v) {
        return v == Math.rint(v) ? String.valueOf((long) v) : String.format(Locale.ROOT, "%.1f", v);
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.ROOT, "%.1f", 100 * v);
    }

    private static void writeFiles(Path outDir, StringBuilder text, List<String> csv,
                                   EvaluationProtocol protocol, Path datasetDir) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("ppca.txt"), StandardCharsets.UTF_8))) {
            out.println("PPCA-скоринг на ORL (шаг 4б, пункт «б»; пункт «а» — это w = 0)");
            out.println("База: " + datasetDir + " (The Database of Faces, AT&T Laboratories Cambridge)");
            out.printf(Locale.ROOT, "seed = %d; k = N − 1; без нормализации; бутстреп по людям: %d повторов.%n",
                    protocol.getSeed(), BOOTSTRAP);
            out.println("d_w = ε²/s² + w·Σ yᵢ²/λ̃ᵢ; λ̃ = (1 − γ)·λ + γ·λ̄; w ∈ {0; 1; 10; 100}, γ ∈ {0; 0,5; 1}; класс — argmin d_w.");
            out.println("s² — скользящим контролем внутри обучающего набора (k′ = N − 2), среднее εⱼ²/(D − k); общий — среднее по галерее.");
            out.println("(w, γ) и порог — по объединённой валидации всех фолдов (и ротаций): наименьший FRR_вал при FAR_вал ≤ α (Нейман – Пирсон);");
            out.println("при равенстве — меньший w, затем меньший γ. На контроле ничего не подбирается.");
            out.println("Δ — парная разность FRR на тех же бутстреп-выборках людей: * — значимо лучше, ! — значимо хуже.");
            out.println("FAR — по группам (ротациям; у основного протокола одна группа), граница Клоппера – Пирсона — для худшей группы.");
            out.print(text);
        }
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("ppca.csv"), StandardCharsets.UTF_8))) {
            out.println("protocol,alpha,variant,w,gamma,val_frr,test_frr,frr_lo95,frr_hi95,kind,argmin_accuracy,"
                    + "far_worst,impostor_attempts,people_worst,people_any");
            for (String row : csv) {
                out.println(row);
            }
        }
    }
}
