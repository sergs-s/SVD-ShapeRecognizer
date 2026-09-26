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
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import nu.pattern.OpenCV;
import svd.recognizer.faces.FaceEvaluation.Probe;
import svd.recognizer.faces.FaceEvaluation.Role;
import svd.recognizer.faces.FaceEvaluation.Score;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * Кривая FRR от числа обучающих снимков на человека N = 3…9 на ORL
 * (шаг 4б, пункт «в»; одна команда: {@code mvn compile exec:java@faces-curve}).
 *
 * Разбиение — ротация контрольного снимка: для ротации r = 0…9 контроль —
 * снимок o[r] (порядок o — общий с основным протоколом), обучение — первые
 * N снимков из остальных девяти по кругу начиная с o[r+1]. Обучающие наборы
 * вложены по N, контрольные снимки не зависят от N. Фолды и чужие — как в
 * {@link EvaluationProtocol}; валидация своих не нужна (порог строится по
 * чужим).
 *
 * Порог — по Нейману – Пирсону на объединённой валидации всех фолдов,
 * отдельно для каждой пары (N, r). Без нормализации освещения.
 *
 * Доверительные интервалы FRR и точности без порога — бутстреп по людям
 * (ресэмплинг 40 человек с возвращением), разность соседних N — парный
 * бутстреп на тех же выборках. FAR — по ротациям (среднее, минимум, максимум),
 * по людям — принят хотя бы в одной ротации; граница Клоппера – Пирсона —
 * для худшей ротации.
 *
 * @author ssv
 */
public final class TrainingSizeCurve {

    static final int MIN_N = 3;
    static final int MAX_N = 9;
    static final int ROTATIONS = OrlDataset.IMAGES_PER_PERSON;
    static final double[] ALPHAS = {0.05, 0.02};
    static final double ETA = 0.95;
    static final int BOOTSTRAP = 1000;

    /** Выбор размерности подпространства. */
    enum KMode {
        FIXED("k=N-1"), ENERGY("eta095");

        final String label;

        KMode(String label) {
            this.label = label;
        }
    }

    /** Снимки одной пары (N, r) по всем фолдам и сумма выбранных k. */
    record Cell(List<Probe> probes, int kSum, int kCount) {}

    /** Строка кривой для одного N. */
    record Point(int n, double kMean,
                 double frr, double frrLo, double frrHi,
                 double diffNext, double diffLo, double diffHi,
                 double acc, double accLo, double accHi, double confusion,
                 double farMean, double farMin, double farMax, int farWorst, int impostorAttempts,
                 int peopleAny, int peopleWorst, int impostorPeople) {}

    private TrainingSizeCurve() {}

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
        int[][] boot = bootstrapSamples(OrlDataset.PERSONS, BOOTSTRAP, seed);

        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        List<String[]> csv = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        for (KMode mode : KMode.values()) {
            System.out.println("Режим " + mode.label + "...");
            Cell[][] cells = new Cell[MAX_N - MIN_N + 1][ROTATIONS];
            IntStream.range(0, cells.length * ROTATIONS).parallel().forEach(t -> {
                int n = MIN_N + t / ROTATIONS;
                int r = t % ROTATIONS;
                cells[n - MIN_N][r] = runCell(dataset, protocol, trainer, mode, n, r);
            });
            for (Score score : new Score[] {Score.RATIO, Score.EPS}) {
                for (double alpha : ALPHAS) {
                    List<Point> points = curve(cells, score, alpha, boot);
                    for (Point p : points) {
                        csv.add(csvRow(mode, score, alpha, p));
                    }
                    appendText(text, mode, score, alpha, points);
                }
            }
        }
        writeCsv(outDir.resolve("curve_n.csv"), csv);
        writeText(outDir.resolve("curve_n.txt"), text, protocol, datasetDir);
        System.out.printf(Locale.ROOT, "Готово: %s (%.0f с)%n", outDir,
                (System.currentTimeMillis() - start) / 1000.0);
    }

    /** Контрольный снимок ротации r. */
    static int testImage(int[] order, int r) {
        return order[r];
    }

    /** Обучающие снимки ротации r: первые n из остальных по кругу начиная с order[r+1]. */
    static int[] trainImages(int[] order, int r, int n) {
        if (n < 2 || n > order.length - 1) {
            throw new IllegalArgumentException("N должно быть в [2, " + (order.length - 1) + "], получено " + n);
        }
        int[] train = new int[n];
        for (int j = 0; j < n; j++) {
            train[j] = order[(r + 1 + j) % order.length];
        }
        return train;
    }

    static Cell runCell(OrlDataset dataset, EvaluationProtocol protocol, SubspaceTrainer trainer,
                        KMode mode, int n, int r) {
        List<Probe> probes = new ArrayList<>();
        int kSum = 0;
        int kCount = 0;
        for (int fold = 0; fold < EvaluationProtocol.FOLDS; fold++) {
            int[] known = protocol.knownPersons(fold);
            double[][] means = new double[known.length][];
            double[][][] bases = new double[known.length][][];
            int[] ks = new int[known.length];
            for (int m = 0; m < known.length; m++) {
                int[] train = trainImages(protocol.imageOrder(known[m]), r, n);
                double[][] vectors = new double[n][];
                for (int j = 0; j < n; j++) {
                    vectors[j] = dataset.vector(known[m], train[j]);
                }
                SubspaceModel model = mode == KMode.FIXED
                        ? trainer.train(vectors, n - 1)
                        : trainer.trainByEnergy(vectors, ETA);
                means[m] = model.getMeanVector();
                bases[m] = model.getBasisMatrix();
                ks[m] = model.getK();
                kSum += ks[m];
                kCount++;
            }
            for (int person : known) {
                int image = testImage(protocol.imageOrder(person), r);
                probes.add(FaceEvaluation.score(fold, Role.KNOWN_TEST, person, image,
                        dataset, known, means, bases, ks));
            }
            for (int person : protocol.validationImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    probes.add(FaceEvaluation.score(fold, Role.IMPOSTOR_VAL, person, image,
                            dataset, known, means, bases, ks));
                }
            }
            for (int person : protocol.testImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    probes.add(FaceEvaluation.score(fold, Role.IMPOSTOR_TEST, person, image,
                            dataset, known, means, bases, ks));
                }
            }
        }
        return new Cell(probes, kSum, kCount);
    }

    /** Выборки бутстрепа: samples[b] — индексы людей 0…persons−1 с возвращением. */
    static int[][] bootstrapSamples(int persons, int count, long seed) {
        Random random = new Random(seed);
        int[][] samples = new int[count][persons];
        for (int b = 0; b < count; b++) {
            for (int i = 0; i < persons; i++) {
                samples[b][i] = random.nextInt(persons);
            }
        }
        return samples;
    }

    /** Доля Σnum/Σden по людям одной бутстреп-выборки. */
    static double bootRatio(int[] sample, int[] num, int[] den) {
        long a = 0;
        long b = 0;
        for (int p : sample) {
            a += num[p];
            b += den[p];
        }
        return b == 0 ? Double.NaN : a / (double) b;
    }

    /** Перцентильный 95 % интервал: {нижняя, верхняя} граница. */
    static double[] percentile95(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int lo = (int) Math.floor(0.025 * sorted.length);
        int hi = (int) Math.ceil(0.975 * sorted.length) - 1;
        return new double[] {sorted[lo], sorted[hi]};
    }

    /** Попытки, ложные отказы и верные argmin по людям для одного N. */
    private record PersonCounts(int[] attempts, int[] rejects, int[] correct, int confusion,
                                int[] farPerRotation, int[] peoplePerRotation, int peopleAny,
                                int impostorAttempts, int impostorPeople) {}

    private static PersonCounts count(Cell[] rotations, Score score, double alpha) {
        int[] attempts = new int[OrlDataset.PERSONS];
        int[] rejects = new int[OrlDataset.PERSONS];
        int[] correct = new int[OrlDataset.PERSONS];
        int confusion = 0;
        int[] farPerRotation = new int[rotations.length];
        int[] peoplePerRotation = new int[rotations.length];
        Set<Integer> peopleAny = new HashSet<>();
        int impostorAttempts = 0;
        Set<Integer> impostorPeople = new HashSet<>();
        for (int r = 0; r < rotations.length; r++) {
            List<Probe> probes = rotations[r].probes();
            double theta = FaceEvaluation.neymanPearsonThreshold(probes.stream()
                    .filter(p -> p.role() == Role.IMPOSTOR_VAL)
                    .mapToDouble(score::of).toArray(), alpha);
            Set<Integer> acceptedPeople = new HashSet<>();
            int attemptsThisRotation = 0;
            for (Probe p : probes) {
                boolean accepted = score.of(p) <= theta;
                if (p.role() == Role.KNOWN_TEST) {
                    attempts[p.person()]++;
                    if (p.correct()) correct[p.person()]++;
                    if (!accepted) rejects[p.person()]++;
                    else if (!p.correct()) confusion++;
                } else if (p.role() == Role.IMPOSTOR_TEST) {
                    attemptsThisRotation++;
                    impostorPeople.add(p.person());
                    if (accepted) {
                        farPerRotation[r]++;
                        acceptedPeople.add(p.person());
                    }
                }
            }
            impostorAttempts = attemptsThisRotation;
            peoplePerRotation[r] = acceptedPeople.size();
            peopleAny.addAll(acceptedPeople);
        }
        return new PersonCounts(attempts, rejects, correct, confusion, farPerRotation,
                peoplePerRotation, peopleAny.size(), impostorAttempts, impostorPeople.size());
    }

    static List<Point> curve(Cell[][] cells, Score score, double alpha, int[][] boot) {
        PersonCounts[] counts = new PersonCounts[cells.length];
        double[][] frrBoot = new double[cells.length][boot.length];
        for (int i = 0; i < cells.length; i++) {
            counts[i] = count(cells[i], score, alpha);
            for (int b = 0; b < boot.length; b++) {
                frrBoot[i][b] = bootRatio(boot[b], counts[i].rejects(), counts[i].attempts());
            }
        }
        List<Point> points = new ArrayList<>();
        for (int i = 0; i < cells.length; i++) {
            PersonCounts c = counts[i];
            int totalAttempts = Arrays.stream(c.attempts()).sum();
            double frr = Arrays.stream(c.rejects()).sum() / (double) totalAttempts;
            double[] frrCi = percentile95(frrBoot[i]);
            double acc = Arrays.stream(c.correct()).sum() / (double) totalAttempts;
            double[] accBoot = new double[boot.length];
            for (int b = 0; b < boot.length; b++) {
                accBoot[b] = bootRatio(boot[b], c.correct(), c.attempts());
            }
            double[] accCi = percentile95(accBoot);
            double diff = Double.NaN;
            double[] diffCi = {Double.NaN, Double.NaN};
            if (i + 1 < cells.length) {
                double nextFrr = Arrays.stream(counts[i + 1].rejects()).sum()
                        / (double) Arrays.stream(counts[i + 1].attempts()).sum();
                diff = frr - nextFrr;
                double[] diffBoot = new double[boot.length];
                for (int b = 0; b < boot.length; b++) {
                    diffBoot[b] = frrBoot[i][b] - frrBoot[i + 1][b];
                }
                diffCi = percentile95(diffBoot);
            }
            int kSum = 0;
            int kCount = 0;
            for (Cell cell : cells[i]) {
                kSum += cell.kSum();
                kCount += cell.kCount();
            }
            int[] far = c.farPerRotation();
            int farWorst = Arrays.stream(far).max().orElse(0);
            points.add(new Point(MIN_N + i, kSum / (double) kCount,
                    frr, frrCi[0], frrCi[1], diff, diffCi[0], diffCi[1],
                    acc, accCi[0], accCi[1], c.confusion() / (double) totalAttempts,
                    Arrays.stream(far).average().orElse(0) / c.impostorAttempts(),
                    Arrays.stream(far).min().orElse(0) / (double) c.impostorAttempts(),
                    farWorst / (double) c.impostorAttempts(), farWorst, c.impostorAttempts(),
                    c.peopleAny(), Arrays.stream(c.peoplePerRotation()).max().orElse(0), c.impostorPeople()));
        }
        return points;
    }

    /**
     * Наименьшее N, начиная с которого все приросты FRR(N') − FRR(N'+1),
     * N' ≥ N, незначимы (нижняя граница 95 % бутстреп-интервала разности ≤ 0);
     * −1, если прирост от MAX_N − 1 к MAX_N ещё значим.
     */
    static int plateau(List<Point> points) {
        int plateau = -1;
        for (int i = points.size() - 1; i >= 0; i--) {
            Point p = points.get(i);
            if (Double.isNaN(p.diffNext())) {
                continue;
            }
            if (p.diffLo() > 0.0) {
                break;
            }
            plateau = p.n();
        }
        return plateau;
    }

    private static String[] csvRow(KMode mode, Score score, double alpha, Point p) {
        return new String[] {
            mode.label, score.label, String.format(Locale.ROOT, "%.2f", alpha), String.valueOf(p.n()),
            f(p.kMean()), f(p.frr()), f(p.frrLo()), f(p.frrHi()),
            f(p.diffNext()), f(p.diffLo()), f(p.diffHi()),
            f(p.acc()), f(p.accLo()), f(p.accHi()), f(p.confusion()),
            f(p.farMean()), f(p.farMin()), f(p.farMax()),
            f(FaceEvaluation.binomialUpperBound(p.farWorst(), p.impostorAttempts(), FaceEvaluation.CONFIDENCE)),
            p.peopleAny() + "/" + p.impostorPeople(),
            String.valueOf(p.peopleWorst()),
            f(FaceEvaluation.binomialUpperBound(p.peopleWorst(), p.impostorPeople(), FaceEvaluation.CONFIDENCE))
        };
    }

    private static String f(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.ROOT, "%.4f", v);
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.ROOT, "%.1f", 100 * v);
    }

    private static void writeCsv(Path file, List<String[]> rows) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.println("k_mode,score,alpha,n,k_mean,frr,frr_lo95,frr_hi95,frr_minus_next,diff_lo95,diff_hi95,"
                    + "argmin_accuracy,acc_lo95,acc_hi95,confusion,far_mean,far_min,far_max,far_worst_upper95,"
                    + "far_people_any_rotation,far_people_worst_rotation,far_people_worst_upper95");
            for (String[] row : rows) {
                out.println(String.join(",", row));
            }
        }
    }

    private static void appendText(StringBuilder text, KMode mode, Score score, double alpha, List<Point> points) {
        text.append(String.format(Locale.ROOT, "%n=== k: %s; скоринг: %s; α = %.2f ===%n", mode.label, score.label, alpha));
        text.append("  N  k_ср   FRR, % [95 % ДИ]      ΔFRR N→N+1, п.п. [95 % ДИ]   argmin, % [95 % ДИ]   путан.,%  FAR ср/мин/макс, %   FAR↑95 худш.  люди (хоть раз / худш.)  люди↑95 худш.\n");
        for (Point p : points) {
            String diff = Double.isNaN(p.diffNext()) ? "—" : String.format(Locale.ROOT, "%+5.1f [%+5.1f; %+5.1f]%s",
                    100 * p.diffNext(), 100 * p.diffLo(), 100 * p.diffHi(), p.diffLo() > 0 ? " *" : "  ");
            text.append(String.format(Locale.ROOT,
                    "  %d  %4.2f   %5s [%5s; %5s]   %-29s  %5s [%5s; %5s]   %5s     %4s/%4s/%4s      %5s         %d/%d / %d/%d              %5s%n",
                    p.n(), p.kMean(), pct(p.frr()), pct(p.frrLo()), pct(p.frrHi()), diff,
                    pct(p.acc()), pct(p.accLo()), pct(p.accHi()), pct(p.confusion()),
                    pct(p.farMean()), pct(p.farMin()), pct(p.farMax()),
                    pct(FaceEvaluation.binomialUpperBound(p.farWorst(), p.impostorAttempts(), FaceEvaluation.CONFIDENCE)),
                    p.peopleAny(), p.impostorPeople(), p.peopleWorst(), p.impostorPeople(),
                    pct(FaceEvaluation.binomialUpperBound(p.peopleWorst(), p.impostorPeople(), FaceEvaluation.CONFIDENCE))));
        }
        int plateau = plateau(points);
        text.append(plateau < 0
                ? String.format(Locale.ROOT, "  Плато не достигнуто до N = %d: прирост FRR от %d к %d ещё значим.%n",
                        MAX_N, MAX_N - 1, MAX_N)
                : String.format(Locale.ROOT, "  Плато: с N = %d все дальнейшие приросты FRR незначимы (нижняя граница ДИ ≤ 0).%n", plateau));
    }

    private static void writeText(Path file, StringBuilder text, EvaluationProtocol protocol, Path datasetDir)
            throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.println("Кривая FRR от числа обучающих снимков на человека N (ORL)");
            out.println("База: " + datasetDir + " (The Database of Faces, AT&T Laboratories Cambridge)");
            out.printf(Locale.ROOT, "seed = %d; фолдов %d; ротация контрольного снимка r = 0…%d; бутстреп по людям: %d повторов.%n",
                    protocol.getSeed(), EvaluationProtocol.FOLDS, ROTATIONS - 1, BOOTSTRAP);
            out.println("Контроль — снимок o[r], обучение — первые N из остальных девяти по кругу (наборы вложены по N).");
            out.println("На каждое N: 1200 попыток своих (40 человек × 10 снимков × 3 фолда), чужие — 200 попыток / 20 людей в каждой ротации.");
            out.println("Порог — Нейман – Пирсон на объединённой валидации 4 фолдов, отдельно для каждой пары (N, r). Без нормализации.");
            out.println("ДИ FRR и argmin — бутстреп по людям; ΔFRR — парный бутстреп на тех же выборках, * — прирост значим.");
            out.println("FAR — по ротациям (среднее / минимум / максимум); люди — приняты хотя бы в одной ротации / в худшей ротации;");
            out.println("границы Клоппера – Пирсона — для худшей ротации.");
            out.print(text);
        }
    }
}
