package svd.recognizer.faces;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.IntStream;
import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.FaceEvaluation.Role;
import svd.recognizer.math.CommonsMathSvdEngine;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.processing.SubspaceRecognizer;
import svd.recognizer.processing.SubspaceTrainer;
import svd.recognizer.storage.SettingsStore;

/**
 * Выравнивание лиц на ORL и эталон SFace (подготовка к шагу 5; одна команда:
 * {@code mvn compile exec:java@faces-align}).
 *
 * Источники признаков: сырые кадры ORL (база), (1) FaceRecognizerSF.alignCrop,
 * (2) собственное аффинное выравнивание по 5 точкам YuNet — все с нашим
 * распознаванием (подпространство на человека, k = n − 1, ε₁/ε₂); SFace —
 * признак 128, шаблон человека — нормированное среднее нормированных признаков
 * обучающих снимков, оценка 1 − cos (лучший и отношение лучший/второй).
 *
 * Неудачная детекция: обучающий снимок исключается; запрос — отказ (свой —
 * «отказ детектора», чужой — не принят). Порог — Нейман – Пирсон на
 * объединённой валидации по попыткам чужих с успешной детекцией. FAR — по всем
 * попыткам и отдельно по попыткам с успешной детекцией.
 *
 * Протоколы: основной (N = 5) и протокол кривой при N = 8 — те же разбиения,
 * что в {@link PpcaEvaluation}.
 *
 * @author ssv
 */
public final class AlignmentEvaluation {

    static final double[] PADDINGS = {0, 0.25, 0.5};
    static final double[] ALPHAS = {0.02, 0.05};
    static final int MOSAIC_COLS = 5;
    static final int MOSAIC_ROWS = 4;
    static final int TILE = 112;
    static final int LABEL = 16;

    /** Источник признаков. */
    enum Source {
        BASE("база: сырые кадры ORL + SVD", false),
        ALIGN_CROP("(1) alignCrop + SVD", false),
        OWN_AFFINE("(2) своё аффинное + SVD", false),
        SFACE("SFace", true);

        final String label;
        final boolean sface;

        Source(String label, boolean sface) {
            this.label = label;
            this.sface = sface;
        }
    }

    /** Скоринг: лучший или отношение лучший/второй. */
    enum Score { BEST, RATIO }

    /**
     * Сравнение одного снимка с галереей. group — ротация (0 для основного протокола);
     * ownModel — у своего человека есть модель в этой галерее (для чужих всегда true).
     */
    record Probe(int group, Role role, int person, boolean detected, double best, double second, int predicted,
                 boolean ownModel) {
        double value(Score s) {
            if (!detected) {
                return Double.POSITIVE_INFINITY;
            }
            return s == Score.BEST ? best : best / second;
        }
    }

    /** Итоги по протоколу. */
    record Result(List<Probe> probes, int modelsMissing) {}

    /** Метрики одного сочетания (источник, скоринг, α). */
    static final class Counts {
        double theta;
        int[] attempts = new int[OrlDataset.PERSONS];
        int[] rejects = new int[OrlDataset.PERSONS];
        int[] detectorRejects = new int[OrlDataset.PERSONS];
        int[] correct = new int[OrlDataset.PERSONS];
        int valKnown, valRejected;
        int groups;
        int[] impostorAttempts, impostorDetected, falseAccept;
        int[] peopleAccepted;
        int peopleAny, impostorPeople;

        int sum(int[] a) {
            return Arrays.stream(a).sum();
        }

        double frr() {
            return sum(rejects) / (double) sum(attempts);
        }
    }

    private AlignmentEvaluation() {}

    public static void main(String[] args) throws IOException {
        long start = System.currentTimeMillis();
        OpenCV.loadLocally();
        SettingsStore settings = new SettingsStore();
        Path datasetDir = Paths.get(settings.loadFacesDatasetDir());
        Path modelsDir = Paths.get(settings.loadFacesModelsDir());
        long seed = settings.loadFacesSeed();
        EvaluationProtocol protocol = new EvaluationProtocol(seed, settings.loadFacesTrainPerPerson());
        Path outDir = Paths.get(System.getProperty("user.dir"), "reports", "faces");
        Files.createDirectories(outDir);
        int frameWidth = settings.loadFacesFrameWidth();
        int frameHeight = settings.loadFacesFrameHeight();
        FaceAlignment alignment = new FaceAlignment(modelsDir, frameWidth, frameHeight);

        // Детекция при разных полях; выбор полей по числу удачных детекций (метки людей не используются).
        Mat[][] originals = new Mat[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                originals[p][i] = FaceAlignment.readGray(datasetDir.resolve("s" + (p + 1)).resolve((i + 1) + ".pgm"));
            }
        }
        int[] detectedByPadding = new int[PADDINGS.length];
        FaceAlignment.Result[][][] byPadding = new FaceAlignment.Result[PADDINGS.length][][];
        for (int k = 0; k < PADDINGS.length; k++) {
            byPadding[k] = new FaceAlignment.Result[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON];
            for (int p = 0; p < OrlDataset.PERSONS; p++) {
                for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                    byPadding[k][p][i] = alignment.process(originals[p][i], PADDINGS[k]);
                    if (byPadding[k][p][i].detected()) detectedByPadding[k]++;
                }
            }
        }
        int chosen = 0;
        for (int k = 1; k < PADDINGS.length; k++) {
            if (detectedByPadding[k] > detectedByPadding[chosen]) chosen = k;
        }
        FaceAlignment.Result[][] results = byPadding[chosen];

        OrlDataset raw = OrlDataset.load(datasetDir, false,
                settings.loadFacesFrameWidth(), settings.loadFacesFrameHeight());
        double[][][][] features = new double[Source.values().length][OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON][];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                FaceAlignment.Result r = results[p][i];
                features[Source.BASE.ordinal()][p][i] = raw.vector(p, i);
                features[Source.ALIGN_CROP.ordinal()][p][i] = r.detected() ? r.alignCropVector() : null;
                features[Source.OWN_AFFINE.ordinal()][p][i] = r.detected() ? r.ownAffineVector() : null;
                features[Source.SFACE.ordinal()][p][i] = r.detected() ? normalize(r.sfaceFeature()) : null;
            }
        }
        saveAligned(outDir.resolve("aligned"), results);
        saveMosaics(outDir.resolve("aligned"), results);

        SubspaceTrainer trainer = new SubspaceTrainer(new CommonsMathSvdEngine());
        int[][] boot = TrainingSizeCurve.bootstrapSamples(OrlDataset.PERSONS, PpcaEvaluation.BOOTSTRAP, seed);
        StringBuilder text = new StringBuilder();
        List<String> csv = new ArrayList<>();
        appendDetection(text, detectedByPadding, chosen, results);
        // База, ограниченная снимками с успешной детекцией (задача 1).
        double[][][] maskedBase = new double[OrlDataset.PERSONS][OrlDataset.IMAGES_PER_PERSON][];
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                maskedBase[p][i] = results[p][i].detected() ? raw.vector(p, i) : null;
            }
        }
        StringBuilder subsetText = new StringBuilder();
        StringBuilder alphaText = new StringBuilder();
        for (boolean curve : new boolean[] {false, true}) {
            String name = curve ? "кривая, N = " + PpcaEvaluation.CURVE_N : "основной, N = " + protocol.getTrainPerPerson();
            String id = curve ? "curve_n" + PpcaEvaluation.CURVE_N : "main_n5";
            System.out.println("Протокол " + id + "...");
            Result[] bySource = new Result[Source.values().length];
            for (Source s : Source.values()) {
                bySource[s.ordinal()] = run(features[s.ordinal()], s.sface, protocol, trainer, curve);
            }
            report(text, csv, name, id, bySource, boot);
            Result baseSubset = run(maskedBase, false, protocol, trainer, curve);
            reportSubset(subsetText, name, baseSubset, bySource[Source.OWN_AFFINE.ordinal()], boot);
            reportSfaceAlpha(alphaText, name, bySource[Source.SFACE.ordinal()], curve);
        }
        writeFiles(outDir, text, csv, protocol, datasetDir, modelsDir, PADDINGS[chosen], frameWidth, frameHeight);
        writeExtra(outDir.resolve("alignment_subset.txt"), subsetHeader(protocol), subsetText);
        writeExtra(outDir.resolve("sface_alpha.txt"), alphaHeader(protocol), alphaText);
        System.out.printf(Locale.ROOT, "Готово: %s (%.0f с)%n", outDir, (System.currentTimeMillis() - start) / 1000.0);
    }

    static double[] normalize(double[] v) {
        double norm = 0;
        for (double x : v) norm += x * x;
        norm = Math.sqrt(norm);
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = v[i] / norm;
        return r;
    }

    /** Все галереи протокола для одного источника признаков. */
    static Result run(double[][][] feat, boolean sface, EvaluationProtocol protocol, SubspaceTrainer trainer,
                      boolean curve) {
        int galleries = curve ? EvaluationProtocol.FOLDS * TrainingSizeCurve.ROTATIONS : EvaluationProtocol.FOLDS;
        List<Result> parts = IntStream.range(0, galleries).parallel().mapToObj(t -> {
            int fold = curve ? t / TrainingSizeCurve.ROTATIONS : t;
            int r = curve ? t % TrainingSizeCurve.ROTATIONS : 0;
            int[] known = protocol.knownPersons(fold);
            int[][] train = new int[known.length][];
            List<int[]> spec = new ArrayList<>();
            for (int m = 0; m < known.length; m++) {
                int person = known[m];
                if (curve) {
                    int[] order = protocol.imageOrder(person);
                    train[m] = TrainingSizeCurve.trainImages(order, r, PpcaEvaluation.CURVE_N);
                    spec.add(new int[] {Role.KNOWN_VAL.ordinal(), person, PpcaEvaluation.validationImage(order, r)});
                    spec.add(new int[] {Role.KNOWN_TEST.ordinal(), person, TrainingSizeCurve.testImage(order, r)});
                } else {
                    train[m] = protocol.trainImages(person);
                    for (int image : protocol.validationImages(person)) {
                        spec.add(new int[] {Role.KNOWN_VAL.ordinal(), person, image});
                    }
                    for (int image : protocol.testImages(person)) {
                        spec.add(new int[] {Role.KNOWN_TEST.ordinal(), person, image});
                    }
                }
            }
            for (int person : protocol.validationImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    spec.add(new int[] {Role.IMPOSTOR_VAL.ordinal(), person, image});
                }
            }
            for (int person : protocol.testImpostors(fold)) {
                for (int image = 0; image < OrlDataset.IMAGES_PER_PERSON; image++) {
                    spec.add(new int[] {Role.IMPOSTOR_TEST.ordinal(), person, image});
                }
            }
            return gallery(feat, sface, trainer, r, known, train, spec);
        }).toList();
        List<Probe> probes = new ArrayList<>();
        int missing = 0;
        for (Result part : parts) {
            probes.addAll(part.probes());
            missing += part.modelsMissing();
        }
        return new Result(probes, missing);
    }

    private static Result gallery(double[][][] feat, boolean sface, SubspaceTrainer trainer, int group,
                                  int[] known, int[][] train, List<int[]> spec) {
        List<Integer> persons = new ArrayList<>();
        List<double[]> means = new ArrayList<>();
        List<double[][]> bases = new ArrayList<>();
        List<Integer> ks = new ArrayList<>();
        int missing = 0;
        for (int m = 0; m < known.length; m++) {
            List<double[]> vectors = new ArrayList<>();
            for (int image : train[m]) {
                if (feat[known[m]][image] != null) vectors.add(feat[known[m]][image]);
            }
            if (sface) {
                if (vectors.isEmpty()) {
                    missing++;
                    continue;
                }
                double[] mean = new double[vectors.get(0).length];
                for (double[] v : vectors) {
                    for (int i = 0; i < mean.length; i++) mean[i] += v[i];
                }
                persons.add(known[m]);
                means.add(normalize(mean));
            } else {
                if (vectors.size() < 2) {
                    missing++;
                    continue;
                }
                SubspaceModel model = trainer.train(vectors.toArray(new double[0][]), vectors.size() - 1);
                persons.add(known[m]);
                means.add(model.getMeanVector());
                bases.add(model.getBasisMatrix());
                ks.add(model.getK());
            }
        }
        List<Probe> probes = new ArrayList<>();
        for (int[] s : spec) {
            Role role = Role.values()[s[0]];
            double[] x = feat[s[1]][s[2]];
            if (x == null) {
                probes.add(new Probe(group, role, s[1], false, Double.NaN, Double.NaN, -1, hasModel(role, s[1], persons)));
                continue;
            }
            double best = Double.MAX_VALUE;
            double second = Double.MAX_VALUE;
            int predicted = -1;
            for (int m = 0; m < persons.size(); m++) {
                double d;
                if (sface) {
                    double dot = 0;
                    double[] t = means.get(m);
                    for (int i = 0; i < t.length; i++) dot += x[i] * t[i];
                    d = 1 - dot;
                } else {
                    d = SubspaceRecognizer.reconstructionError(x, means.get(m), bases.get(m), ks.get(m));
                }
                if (d < best) {
                    second = best;
                    best = d;
                    predicted = persons.get(m);
                } else if (d < second) {
                    second = d;
                }
            }
            probes.add(new Probe(group, role, s[1], true, best, second, predicted, hasModel(role, s[1], persons)));
        }
        return new Result(probes, missing);
    }

    /** Метрики при пороге Неймана – Пирсона по чужим валидации с успешной детекцией. */
    static Counts evaluate(List<Probe> probes, Score score, double alpha) {
        Counts c = new Counts();
        c.theta = FaceEvaluation.neymanPearsonThreshold(probes.stream()
                .filter(p -> p.role() == Role.IMPOSTOR_VAL && p.detected())
                .mapToDouble(p -> p.value(score)).toArray(), alpha);
        c.groups = probes.stream().mapToInt(Probe::group).max().orElse(0) + 1;
        c.impostorAttempts = new int[c.groups];
        c.impostorDetected = new int[c.groups];
        c.falseAccept = new int[c.groups];
        c.peopleAccepted = new int[c.groups];
        List<Set<Integer>> accepted = new ArrayList<>();
        Set<Integer> impostorPeople = new HashSet<>();
        for (int g = 0; g < c.groups; g++) accepted.add(new HashSet<>());
        for (Probe p : probes) {
            boolean ok = p.value(score) <= c.theta;
            switch (p.role()) {
                case KNOWN_VAL -> {
                    c.valKnown++;
                    if (!ok) c.valRejected++;
                }
                case KNOWN_TEST -> {
                    c.attempts[p.person()]++;
                    if (!p.detected()) {
                        c.detectorRejects[p.person()]++;
                        c.rejects[p.person()]++;
                    } else if (!ok) {
                        c.rejects[p.person()]++;
                    }
                    if (p.detected() && p.predicted() == p.person()) c.correct[p.person()]++;
                }
                case IMPOSTOR_TEST -> {
                    c.impostorAttempts[p.group()]++;
                    impostorPeople.add(p.person());
                    if (p.detected()) c.impostorDetected[p.group()]++;
                    if (ok) {
                        c.falseAccept[p.group()]++;
                        accepted.get(p.group()).add(p.person());
                    }
                }
                default -> { }
            }
        }
        Set<Integer> any = new HashSet<>();
        for (int g = 0; g < c.groups; g++) {
            c.peopleAccepted[g] = accepted.get(g).size();
            any.addAll(accepted.get(g));
        }
        c.peopleAny = any.size();
        c.impostorPeople = impostorPeople.size();
        return c;
    }

    private static void report(StringBuilder text, List<String> csv, String name, String id, Result[] bySource,
                               int[][] boot) {
        text.append(String.format(Locale.ROOT, "%n==================== Протокол: %s ====================%n", name));
        for (Source s : Source.values()) {
            Result r = bySource[s.ordinal()];
            long failedOwn = r.probes().stream().filter(p -> p.role() == Role.KNOWN_TEST && !p.detected()).count();
            long own = r.probes().stream().filter(p -> p.role() == Role.KNOWN_TEST).count();
            long failedImp = r.probes().stream().filter(p -> p.role() == Role.IMPOSTOR_TEST && !p.detected()).count();
            long imp = r.probes().stream().filter(p -> p.role() == Role.IMPOSTOR_TEST).count();
            text.append(String.format(Locale.ROOT,
                    "%-30s неудачных детекций: свои контроля %d/%d, чужие контроля %d/%d; людей без модели (галерея×человек): %d%n",
                    s.label, failedOwn, own, failedImp, imp, r.modelsMissing()));
        }
        for (double alpha : ALPHAS) {
            text.append(String.format(Locale.ROOT, "%n--- α = %.2f ---%n", alpha));
            Counts base = evaluate(bySource[Source.BASE.ordinal()].probes(), Score.RATIO, alpha);
            text.append(String.format(Locale.ROOT, "Регрессия: база — FRR %s %% (%d/%d); должно совпасть с (б) и (д).%n",
                    pct(base.frr()), base.sum(base.rejects), base.sum(base.attempts)));
            text.append("источник, скоринг                   FRR, % [95 % ДИ]     из них: детектор / порог  Δ к базе, п.п. [ДИ]    argmin,%  "
                    + "FAR все (ср/худш.)   FAR с детекцией (ср)  ↑95 худш.  люди (хоть раз/худш.) ↑95 люди\n");
            for (Source s : Source.values()) {
                for (Score score : s.sface ? Score.values() : new Score[] {Score.RATIO}) {
                    Counts c = evaluate(bySource[s.ordinal()].probes(), score, alpha);
                    String label = s.label + (s.sface ? (score == Score.BEST ? ", 1 − cos₁" : ", (1 − cos₁)/(1 − cos₂)") : ", ε₁/ε₂");
                    row(text, csv, id, alpha, label, c, s == Source.BASE ? null : base, boot);
                }
            }
        }
    }

    private static void row(StringBuilder text, List<String> csv, String id, double alpha, String label,
                            Counts c, Counts base, int[][] boot) {
        double[] ci = frrCi(c, boot);
        String diff = "—";
        if (base != null) {
            double[] d = diffCi(c, base, boot);
            String mark = d[1] < 0 ? " *" : d[0] > 0 ? " !" : "";
            diff = String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", 100 * (c.frr() - base.frr()), 100 * d[0], 100 * d[1], mark);
        }
        int total = c.sum(c.attempts);
        double farMean = 0;
        double farDetMean = 0;
        int worst = 0;
        int worstAttempts = 0;
        for (int g = 0; g < c.groups; g++) {
            farMean += c.falseAccept[g] / (double) c.impostorAttempts[g];
            farDetMean += c.impostorDetected[g] == 0 ? 0 : c.falseAccept[g] / (double) c.impostorDetected[g];
            if (c.falseAccept[g] >= worst) {
                worst = c.falseAccept[g];
                worstAttempts = c.impostorAttempts[g];
            }
        }
        farMean /= c.groups;
        farDetMean /= c.groups;
        int peopleWorst = Arrays.stream(c.peopleAccepted).max().orElse(0);
        int peoplePerGroup = c.impostorPeople; // в каждой группе (ротации) — те же чужие люди
        text.append(String.format(Locale.ROOT,
                "%-37s %5s [%5s; %5s]   %5s / %5s            %-22s %5s     %4s/%4s            %5s                 %5s      %d/%d / %d/%d          %5s%n",
                label, pct(c.frr()), pct(ci[0]), pct(ci[1]),
                pct(c.sum(c.detectorRejects) / (double) total), pct((c.sum(c.rejects) - c.sum(c.detectorRejects)) / (double) total),
                diff, pct(c.sum(c.correct) / (double) total),
                pct(farMean), pct(worstAttempts == 0 ? 0 : worst / (double) worstAttempts), pct(farDetMean),
                pct(FaceEvaluation.binomialUpperBound(worst, worstAttempts, FaceEvaluation.CONFIDENCE)),
                c.peopleAny, peoplePerGroup, peopleWorst, peoplePerGroup,
                pct(FaceEvaluation.binomialUpperBound(peopleWorst, peoplePerGroup, FaceEvaluation.CONFIDENCE))));
        csv.add(String.format(Locale.ROOT, "%s,%.2f,\"%s\",%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d",
                id, alpha, label, c.theta, c.frr(), ci[0], ci[1],
                c.sum(c.detectorRejects) / (double) total, c.sum(c.correct) / (double) total,
                farMean, farDetMean, c.valRejected / (double) c.valKnown, worst, worstAttempts, peopleWorst));
    }

    static double[] frrCi(Counts c, int[][] boot) {
        double[] values = new double[boot.length];
        for (int b = 0; b < boot.length; b++) {
            values[b] = TrainingSizeCurve.bootRatio(boot[b], c.rejects, c.attempts);
        }
        return TrainingSizeCurve.percentile95(values);
    }

    static double[] diffCi(Counts a, Counts b, int[][] boot) {
        double[] values = new double[boot.length];
        for (int i = 0; i < boot.length; i++) {
            values[i] = TrainingSizeCurve.bootRatio(boot[i], a.rejects, a.attempts)
                    - TrainingSizeCurve.bootRatio(boot[i], b.rejects, b.attempts);
        }
        return TrainingSizeCurve.percentile95(values);
    }

    private static void appendDetection(StringBuilder text, int[] detectedByPadding, int chosen,
                                        FaceAlignment.Result[][] results) {
        text.append("\n==================== Детекция и выравнивание (400 снимков ORL) ====================\n");
        for (int k = 0; k < PADDINGS.length; k++) {
            text.append(String.format(Locale.ROOT, "Поля %2.0f %%: найдено лиц %d/400 (неудачных %d)%s%n",
                    100 * PADDINGS[k], detectedByPadding[k], 400 - detectedByPadding[k], k == chosen ? "  ← выбрано" : ""));
        }
        List<String> failed = new ArrayList<>();
        double outside1 = 0;
        double outside2 = 0;
        double maxRoll = 0;
        int n = 0;
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                FaceAlignment.Result r = results[p][i];
                if (!r.detected()) {
                    failed.add("s" + (p + 1) + "/" + (i + 1));
                    continue;
                }
                n++;
                outside1 += r.alignCropOutside();
                outside2 += r.ownAffineOutside();
                maxRoll = Math.max(maxRoll, Math.abs(r.roll()));
            }
        }
        text.append("Неудачные детекции (при выбранных полях): ").append(String.join(", ", failed)).append('\n');
        text.append(String.format(Locale.ROOT, "Средняя доля пикселей кадра вне исходного снимка: (1) alignCrop %.1f %%, (2) своё аффинное %.1f %% (по %d снимкам).%n",
                100 * outside1 / n, 100 * outside2 / n, n));
        text.append(String.format(Locale.ROOT, "Наибольший наклон линии глаз: %.1f°.%n", maxRoll));
    }

    /** Все выровненные кадры: aligned/<способ>/sX/Y.png (неудачные детекции не сохраняются). */
    private static void saveAligned(Path dir, FaceAlignment.Result[][] results) throws IOException {
        for (FaceAlignment.Method m : FaceAlignment.Method.values()) {
            for (int p = 0; p < OrlDataset.PERSONS; p++) {
                Path personDir = dir.resolve(m.label).resolve("s" + (p + 1));
                Files.createDirectories(personDir);
                for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) {
                    FaceAlignment.Result r = results[p][i];
                    if (r.detected()) {
                        Imgcodecs.imwrite(personDir.resolve((i + 1) + ".png").toString(),
                                m == FaceAlignment.Method.ALIGN_CROP ? r.alignCropGray() : r.ownAffineGray());
                    }
                }
            }
        }
    }

    /** Отбор снимков для мозаики: неудачные, самые наклонённые, затем первые снимки людей по порядку. */
    static List<int[]> mosaicSelection(FaceAlignment.Result[][] results) {
        Set<Long> chosen = new LinkedHashSet<>();
        int capacity = MOSAIC_COLS * MOSAIC_ROWS;
        List<int[]> all = new ArrayList<>();
        for (int p = 0; p < OrlDataset.PERSONS; p++) {
            for (int i = 0; i < OrlDataset.IMAGES_PER_PERSON; i++) all.add(new int[] {p, i});
        }
        all.stream().filter(a -> !results[a[0]][a[1]].detected()).limit(6)
                .forEach(a -> chosen.add(key(a)));
        all.stream().filter(a -> results[a[0]][a[1]].detected())
                .sorted(Comparator.comparingDouble((int[] a) -> -Math.abs(results[a[0]][a[1]].roll())))
                .limit(6).forEach(a -> chosen.add(key(a)));
        for (int p = 0; p < OrlDataset.PERSONS && chosen.size() < capacity; p++) {
            chosen.add(key(new int[] {p, 0}));
        }
        List<int[]> selection = new ArrayList<>();
        for (long k : chosen) {
            if (selection.size() == capacity) break;
            selection.add(new int[] {(int) (k / 100), (int) (k % 100)});
        }
        return selection;
    }

    private static long key(int[] a) {
        return a[0] * 100L + a[1];
    }

    private static void saveMosaics(Path dir, FaceAlignment.Result[][] results) {
        List<int[]> selection = mosaicSelection(results);
        for (FaceAlignment.Method m : FaceAlignment.Method.values()) {
            Mat mosaic = new Mat(MOSAIC_ROWS * (TILE + LABEL), MOSAIC_COLS * TILE, CvType.CV_8UC3, new Scalar(255, 255, 255));
            for (int t = 0; t < selection.size(); t++) {
                int p = selection.get(t)[0];
                int i = selection.get(t)[1];
                FaceAlignment.Result r = results[p][i];
                int x0 = (t % MOSAIC_COLS) * TILE;
                int y0 = (t / MOSAIC_COLS) * (TILE + LABEL);
                Mat tile = r.detected()
                        ? (m == FaceAlignment.Method.ALIGN_CROP ? r.alignCropGray() : r.ownAffineGray())
                        : r.original();
                Mat bgr = new Mat();
                Imgproc.cvtColor(tile, bgr, Imgproc.COLOR_GRAY2BGR);
                int w = Math.min(TILE, bgr.cols());
                int h = Math.min(TILE, bgr.rows());
                bgr.submat(0, h, 0, w).copyTo(mosaic.submat(new Rect(x0 + (TILE - w) / 2, y0, w, h)));
                String caption = "s" + (p + 1) + "/" + (i + 1)
                        + (r.detected() ? String.format(Locale.ROOT, " %.0fdeg", r.roll()) : " FAIL");
                Imgproc.putText(mosaic, caption, new Point(x0 + 2, y0 + TILE + 12), Imgproc.FONT_HERSHEY_PLAIN, 0.9,
                        r.detected() ? new Scalar(0, 0, 0) : new Scalar(0, 0, 255), 1);
                if (!r.detected()) {
                    Imgproc.rectangle(mosaic, new Point(x0, y0), new Point(x0 + TILE - 1, y0 + TILE - 1), new Scalar(0, 0, 255), 2);
                }
            }
            Imgcodecs.imwrite(dir.resolve("mosaic_" + m.label + ".png").toString(), mosaic);
        }
    }

    static boolean hasModel(Role role, int person, List<Integer> persons) {
        return (role != Role.KNOWN_VAL && role != Role.KNOWN_TEST) || persons.contains(person);
    }

    /**
     * Подмножество задачи 1: только снимки с успешной детекцией; попытки своих,
     * у которых в этой галерее нет модели, исключаются (не считаются отказом).
     */
    static List<Probe> subset(List<Probe> probes) {
        return probes.stream().filter(p -> p.detected() && p.ownModel()).toList();
    }

    /** Число исключённых контрольных попыток своих без модели (с успешной детекцией). */
    static long excludedWithoutModel(List<Probe> probes) {
        return probes.stream().filter(p -> p.role() == Role.KNOWN_TEST && p.detected() && !p.ownModel()).count();
    }

    private static long count(List<Probe> probes, Role role) {
        return probes.stream().filter(p -> p.role() == role).count();
    }

    /** Строка FAR: сумма x/n, среднее и худшая группа, ↑95 худшей; люди — хоть раз / худшая группа, ↑95. */
    private static String farText(Counts c) {
        int x = c.sum(c.falseAccept);
        int n = c.sum(c.impostorAttempts);
        int worst = 0;
        int worstN = 0;
        double mean = 0;
        for (int g = 0; g < c.groups; g++) {
            mean += c.falseAccept[g] / (double) c.impostorAttempts[g];
            if (c.falseAccept[g] >= worst) {
                worst = c.falseAccept[g];
                worstN = c.impostorAttempts[g];
            }
        }
        mean /= c.groups;
        int peopleWorst = Arrays.stream(c.peopleAccepted).max().orElse(0);
        return String.format(Locale.ROOT, "FAR %d/%d (ср. %s, худш. %d/%d, ↑95 %s); люди %d/%d (худш. %d/%d, ↑95 %s)",
                x, n, pct(mean), worst, worstN,
                pct(FaceEvaluation.binomialUpperBound(worst, worstN, FaceEvaluation.CONFIDENCE)),
                c.peopleAny, c.impostorPeople, peopleWorst, c.impostorPeople,
                pct(FaceEvaluation.binomialUpperBound(peopleWorst, c.impostorPeople, FaceEvaluation.CONFIDENCE)));
    }

    /** Задача 1: база и (2) на одном подмножестве снимков с успешной детекцией. */
    private static void reportSubset(StringBuilder text, String name, Result base, Result own, int[][] boot) {
        List<Probe> baseProbes = subset(base.probes());
        List<Probe> ownProbes = subset(own.probes());
        text.append(String.format(Locale.ROOT, "%n==================== Протокол: %s ====================%n", name));
        text.append(String.format(Locale.ROOT,
                "Попыток: свои контроля %d, чужие контроля %d, чужие валидации %d (у базы и (2) одинаково: %b).%n",
                count(ownProbes, Role.KNOWN_TEST), count(ownProbes, Role.IMPOSTOR_TEST), count(ownProbes, Role.IMPOSTOR_VAL),
                count(baseProbes, Role.KNOWN_TEST) == count(ownProbes, Role.KNOWN_TEST)
                        && count(baseProbes, Role.IMPOSTOR_TEST) == count(ownProbes, Role.IMPOSTOR_TEST)
                        && count(baseProbes, Role.IMPOSTOR_VAL) == count(ownProbes, Role.IMPOSTOR_VAL)));
        text.append(String.format(Locale.ROOT,
                "Исключено контрольных попыток своих без модели (галерея × человек, меньше 2 обучающих снимков с детекцией): "
                        + "база %d, (2) %d; моделей не построено: база %d, (2) %d.%n",
                excludedWithoutModel(base.probes()), excludedWithoutModel(own.probes()),
                base.modelsMissing(), own.modelsMissing()));
        for (double alpha : ALPHAS) {
            Counts b = evaluate(baseProbes, Score.RATIO, alpha);
            Counts o = evaluate(ownProbes, Score.RATIO, alpha);
            double[] bCi = frrCi(b, boot);
            double[] oCi = frrCi(o, boot);
            double[] d = diffCi(o, b, boot);
            String mark = d[1] < 0 ? " * (значимо лучше)" : d[0] > 0 ? " ! (значимо хуже)" : " (незначимо)";
            text.append(String.format(Locale.ROOT, "%n--- α = %.2f ---%n", alpha));
            text.append(String.format(Locale.ROOT, "  база, ε₁/ε₂:              FRR %s %% [%s; %s] (%d/%d), argmin %s %%; %s%n",
                    pct(b.frr()), pct(bCi[0]), pct(bCi[1]), b.sum(b.rejects), b.sum(b.attempts),
                    pct(b.sum(b.correct) / (double) b.sum(b.attempts)), farText(b)));
            text.append(String.format(Locale.ROOT, "  (2) своё аффинное, ε₁/ε₂: FRR %s %% [%s; %s] (%d/%d), argmin %s %%; %s%n",
                    pct(o.frr()), pct(oCi[0]), pct(oCi[1]), o.sum(o.rejects), o.sum(o.attempts),
                    pct(o.sum(o.correct) / (double) o.sum(o.attempts)), farText(o)));
            text.append(String.format(Locale.ROOT, "  Δ FRR (2) − база: %+.1f п.п. [%+.1f; %+.1f]%s%n",
                    100 * (o.frr() - b.frr()), 100 * d[0], 100 * d[1], mark));
        }
    }

    /** Задача 2: SFace, 1 − cos₁, при α ∈ {0,05; 0,02; 0,01; 0,005; 0}; запас разделимости на валидации. */
    private static void reportSfaceAlpha(StringBuilder text, String name, Result sface, boolean curve) {
        List<Probe> probes = sface.probes();
        long n = probes.stream().filter(p -> p.role() == Role.IMPOSTOR_VAL && p.detected()).count();
        text.append(String.format(Locale.ROOT, "%n==================== Протокол: %s ====================%n", name));
        text.append(String.format(Locale.ROOT,
                "Попыток чужих с детекцией в объединённой валидации n = %d (разрешающая способность α = 1/n = %.4f %%).%n",
                n, 100.0 / n));
        appendMargin(text, probes, -1, "объединённая валидация");
        if (curve) {
            double worst = Double.POSITIVE_INFINITY;
            int worstGroup = -1;
            for (int g = 0; g < TrainingSizeCurve.ROTATIONS; g++) {
                double m = margin(probes, g)[2];
                appendMargin(text, probes, g, "ротация " + g);
                if (m < worst) {
                    worst = m;
                    worstGroup = g;
                }
            }
            text.append(String.format(Locale.ROOT, "  Худшая ротация по запасу: %d (разность %.4f).%n", worstGroup, worst));
        }
        text.append("  α       ⌊α·n⌋  θ          FRR общий (детектор + порог)   FRR по порогу (свои с детекцией)   FAR все попытки / с детекцией; люди\n");
        for (double alpha : new double[] {0.05, 0.02, 0.01, 0.005, 0.0}) {
            Counts c = evaluate(probes, Score.BEST, alpha);
            int allowed = (int) Math.floor(alpha * n + 1e-9);
            int detectedOwn = c.sum(c.attempts) - c.sum(c.detectorRejects);
            int thresholdRejects = c.sum(c.rejects) - c.sum(c.detectorRejects);
            int detectedImp = c.sum(c.impostorDetected);
            String note = alpha > 0 && allowed == 0 ? "  [⌊α·n⌋ = 0, совпадает с α = 0]" : "";
            text.append(String.format(Locale.ROOT,
                    "  %-6s  %-5d  %.6f   %s %% (%d/%d = %d + %d)          %s %% (%d/%d)                  %s; с детекцией %d/%d%s%n",
                    String.format(Locale.ROOT, "%.3f", alpha), allowed, c.theta,
                    pct(c.frr()), c.sum(c.rejects), c.sum(c.attempts), c.sum(c.detectorRejects), thresholdRejects,
                    pct(thresholdRejects / (double) detectedOwn), thresholdRejects, detectedOwn,
                    farText(c), c.sum(c.falseAccept), detectedImp, note));
        }
    }

    /** {наибольшая оценка своих, наименьшая оценка чужих, разность} на валидации; group = −1 — все группы. */
    static double[] margin(List<Probe> probes, int group) {
        double maxOwn = Double.NEGATIVE_INFINITY;
        double minImp = Double.POSITIVE_INFINITY;
        for (Probe p : probes) {
            if (!p.detected() || (group >= 0 && p.group() != group)) {
                continue;
            }
            if (p.role() == Role.KNOWN_VAL && p.ownModel()) {
                maxOwn = Math.max(maxOwn, p.value(Score.BEST));
            } else if (p.role() == Role.IMPOSTOR_VAL) {
                minImp = Math.min(minImp, p.value(Score.BEST));
            }
        }
        return new double[] {maxOwn, minImp, minImp - maxOwn};
    }

    private static void appendMargin(StringBuilder text, List<Probe> probes, int group, String label) {
        double[] m = margin(probes, group);
        text.append(String.format(Locale.ROOT,
                "  Запас разделимости (%s): наибольшая 1 − cos₁ своих %.4f, наименьшая чужих %.4f, разность %+.4f (%s).%n",
                label, m[0], m[1], m[2], m[2] > 0 ? "зазор есть" : "зазора нет"));
    }

    private static String subsetHeader(EvaluationProtocol protocol) {
        return "Задача 1: база и (2) своё аффинное на одном подмножестве (подготовка к шагу 5)\n"
                + "Только снимки с успешной детекцией YuNet (порог 0,9, без полей): неудачные исключены из обучения,\n"
                + "валидации и контроля у обоих вариантов. Попытки своих, у которых в галерее нет модели (меньше 2\n"
                + "обучающих снимков с детекцией), исключены из знаменателя FRR у обоих вариантов (не считаются отказом).\n"
                + "Скоринг ε₁/ε₂, k = n − 1, порог — Нейман – Пирсон на объединённой валидации; seed = " + protocol.getSeed() + ".\n"
                + "ДИ — бутстреп по людям; Δ — парная разность на тех же выборках.\n";
    }

    private static String alphaHeader(EvaluationProtocol protocol) {
        return "Задача 2: SFace, оценка 1 − cos₁ (шаблон — нормированное среднее), сетка α (подготовка к шагу 5)\n"
                + "Неудачная детекция — отказ (свой) / не принят (чужой). Порог — Нейман – Пирсон на объединённой валидации\n"
                + "по попыткам чужих с детекцией: допускается ⌊α·n⌋ принятых; α = 0 — порог чуть ниже наименьшей оценки\n"
                + "чужих на валидации. Запас разделимости — на валидации: наибольшая оценка своих с детекцией против\n"
                + "наименьшей оценки чужих. seed = " + protocol.getSeed() + ".\n"
                + "Происхождение обучающего набора SFace — открытый вопрос (см. alignment.txt).\n";
    }

    private static void writeExtra(Path file, String header, StringBuilder text) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8))) {
            out.print(header);
            out.print(text);
        }
    }

    private static String pct(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.ROOT, "%.1f", 100 * v);
    }

    private static void writeFiles(Path outDir, StringBuilder text, List<String> csv, EvaluationProtocol protocol,
                                   Path datasetDir, Path modelsDir, double padding, int frameWidth,
                                   int frameHeight) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("alignment.txt"), StandardCharsets.UTF_8))) {
            out.println("Выравнивание лиц на ORL и эталон SFace (подготовка к шагу 5)");
            out.println("База: " + datasetDir + " (The Database of Faces, AT&T Laboratories Cambridge)");
            out.println("Модели: " + modelsDir + " — " + FaceAlignment.YUNET_FILE + " (YuNet, MIT), "
                    + FaceAlignment.SFACE_FILE + " (SFace, Apache 2.0), opencv_zoo.");
            out.println("SFace: MobileFaceNet, обученная с функцией потерь SFace; ONNX — конвертация из github.com/zhongyy/SFace.");
            out.println("  Набор обучения выпущенной модели в карточке opencv_zoo не указан; в статье и коде SFace — CASIA-WebFace,");
            out.println("  VGGFace2 и MS1MV2 (производная MS-Celeb-1M, отозванной Microsoft в 2019 г.). Какой именно — не установлено.");
            out.printf(Locale.ROOT, "seed = %d; детекция YuNet: порог %.1f, лицо с наибольшей оценкой; поля — copyMakeBorder BORDER_REPLICATE, выбрано %.0f %%.%n",
                    protocol.getSeed(), FaceAlignment.SCORE_THRESHOLD, 100 * padding);
            out.println("(1) alignCrop: кадр 112×112, заполнение внутри OpenCV (warpAffine, BORDER_CONSTANT = 0);");
            out.println("(2) своё аффинное: estimateAffinePartial2D (LMEDS) на шаблон ArcFace, сдвинутый в кадр " + frameWidth + "×"
                    + frameHeight + ", warpAffine BORDER_REPLICATE.");
            out.println("SVD: подпространство на человека, k = n − 1 (n — обучающие снимки с успешной детекцией), скоринг ε₁/ε₂.");
            out.println("SFace: признак 128 (L2), шаблон — нормированное среднее; оценка 1 − cos, лучший и отношение.");
            out.println("Неудачная детекция: обучающий снимок исключается; запрос — отказ. Порог — Нейман – Пирсон на объединённой");
            out.println("валидации по попыткам чужих с успешной детекцией. FRR = отказ детектора + отказ по порогу.");
            out.println("FAR все — по всем попыткам чужих (неудачная детекция = не принят); FAR с детекцией — только по попыткам с детекцией.");
            out.println("Δ — парная разность FRR с базой на бутстреп-выборках людей: * — значимо лучше, ! — значимо хуже.");
            out.println("Мозаики: reports/faces/aligned/mosaic_align_crop.png, mosaic_own_affine.png (неудачные — исходный снимок в красной рамке).");
            out.print(text);
        }
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(outDir.resolve("alignment.csv"), StandardCharsets.UTF_8))) {
            out.println("protocol,alpha,source,theta,frr,frr_lo95,frr_hi95,frr_detector,argmin_accuracy,far_mean,"
                    + "far_detected_mean,val_frr,far_worst,far_worst_attempts,people_worst");
            for (String row : csv) {
                out.println(row);
            }
        }
    }
}
