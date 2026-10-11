package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.storage.SettingsStore;

/**
 * Качество кадров по сессиям GT (PLAN.md, п. 4.2, «Качество кадров по сессиям GT»), без распознавания. Меры по кадру «а»
 * (a/gt экспорта, 92×112, серый, 8 бит): дисперсия лапласиана (Imgproc.Laplacian, ksize = 1 — ядро 3×3 [0 1 0; 1 −4 1;
 * 0 1 0]); резкость с поправкой на контраст — дисперсия лапласиана / дисперсия яркости; σ шума по Immerkær (1996): σ =
 * √(π/2)·Σ|I ∗ N| / (6 (W − 2)(H − 2)), N = [1 −2 1; −2 4 −2; 1 −2 1], сумма по внутренним пикселям; средняя яркость;
 * СКО яркости. По исходному снимку — межглазье в пикселях по точкам YuNet (detections.tsv). Свои — разности «пробы −
 * галерея» по людям (медианы по снимкам), бутстреп по людям GT (те же веса, что XBaseEval); пробы — ошибка L1 (П2, FAR 1 %,
 * межсессионно, из xbase_gt_scores.tsv.gz) по квартилям разности «проба − медиана галереи её человека». Контроль внутри
 * сессии — те же разности для половин даты (деление gt-in). Отчёт xbase_gt_quality.txt, по снимкам — xbase_gt_quality.tsv.
 *
 * @author ssv
 */
final class XBaseQuality {

    static final String[] NAMES = {"дисперсия лапласиана", "лапласиан / дисперсия яркости", "σ шума (Immerkær)", "средняя яркость", "СКО яркости",
        "межглазье, px исходного снимка"};
    static final String[] IDS = {"lap_var", "lap_var_rel", "noise_sigma", "mean", "std", "eye_dist"};
    static final int MEASURES = NAMES.length;

    private XBaseQuality() {
    }

    /** Меры 1–5 по серому кадру 8 бит: {дисперсия лапласиана, она же / дисперсия яркости, σ шума, средняя, СКО}. */
    static double[] frameMeasures(Mat gray8) {
        Mat f = new Mat();
        gray8.convertTo(f, CvType.CV_64F);
        Mat lap = new Mat();
        Imgproc.Laplacian(f, lap, CvType.CV_64F, 1);
        MatOfDouble m = new MatOfDouble();
        MatOfDouble s = new MatOfDouble();
        Core.meanStdDev(lap, m, s);
        double lapVar = s.get(0, 0)[0] * s.get(0, 0)[0];
        Core.meanStdDev(f, m, s);
        double mean = m.get(0, 0)[0];
        double std = s.get(0, 0)[0];
        double[] out = {lapVar, std > 0 ? lapVar / (std * std) : Double.NaN, noiseSigma(f), mean, std};
        for (Mat x : new Mat[] {f, lap, m, s}) x.release();
        return out;
    }

    /** σ шума по Immerkær (1996) по кадру CV_64F: свёртка с N, сумма модулей по внутренним пикселям. */
    static double noiseSigma(Mat f64) {
        int w = f64.cols();
        int h = f64.rows();
        double[] v = new double[w * h];
        f64.get(0, 0, v);
        double sum = 0;
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double c = v[(y - 1) * w + x - 1] - 2 * v[(y - 1) * w + x] + v[(y - 1) * w + x + 1]
                        - 2 * v[y * w + x - 1] + 4 * v[y * w + x] - 2 * v[y * w + x + 1]
                        + v[(y + 1) * w + x - 1] - 2 * v[(y + 1) * w + x] + v[(y + 1) * w + x + 1];
                sum += Math.abs(c);
            }
        }
        return Math.sqrt(Math.PI / 2) * sum / (6.0 * (w - 2) * (h - 2));
    }

    /** Межглазье по строке YuNet (15 чисел: рамка, глаза (4, 5) и (6, 7), …). */
    static double eyeDistance(double[] row) {
        return Math.hypot(row[4] - row[6], row[5] - row[7]);
    }

    /** Строки YuNet по detections.tsv: «человек/снимок» GT → 15 чисел. */
    static Map<String, double[]> gtRows(Path detections) throws IOException {
        Map<String, double[]> out = new HashMap<>();
        for (String l : Files.readAllLines(detections, StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (!c[0].equals("gt") || !c[4].equals("1")) continue;
            double[] row = new double[15];
            for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[5 + i]);
            out.put(c[1] + "/" + Integer.parseInt(c[2].replaceAll("\\D", "")), row);
        }
        return out;
    }

    static double median(List<Double> v) {
        double[] a = v.stream().mapToDouble(Double::doubleValue).toArray();
        return FeiBootstrap.quantile(FeiBootstrap.sorted(a), 0.5);
    }

    /** Взвешенное среднее d[i] людей p[i] при весах w (нет веса — NaN). */
    static double wmean(double[] d, int[] p, int[] w) {
        double s = 0;
        double n = 0;
        for (int i = 0; i < d.length; i++) {
            s += w[p[i]] * d[i];
            n += w[p[i]];
        }
        return n > 0 ? s / n : Double.NaN;
    }

    /** Группа квартиля (0–3) значения x при границах q (25, 50, 75 %): ≤ q₁, ≤ q₂, ≤ q₃, больше. */
    static int quartile(double x, double[] q) {
        if (x <= q[0]) return 0;
        if (x <= q[1]) return 1;
        if (x <= q[2]) return 2;
        return 3;
    }

    /** Строка таблицы «пробы − галерея» по людям для меры k. */
    static String diffRow(int k, List<String> own, Map<String, List<String>> gal, Map<String, List<String>> pro, Map<String, double[]> meas,
                          Map<String, Integer> pIdx, int[][] w) {
        double[] d = new double[own.size()];
        int[] p = new int[own.size()];
        List<Double> gm = new ArrayList<>();
        List<Double> pm = new ArrayList<>();
        for (int i = 0; i < own.size(); i++) {
            List<Double> a = new ArrayList<>();
            List<Double> b = new ArrayList<>();
            for (String key : gal.get(own.get(i))) a.add(meas.get(key)[k]);
            for (String key : pro.get(own.get(i))) b.add(meas.get(key)[k]);
            double ma = median(a);
            double mb = median(b);
            gm.add(ma);
            pm.add(mb);
            d[i] = mb - ma;
            p[i] = pIdx.get(own.get(i));
        }
        double mean = wmean(d, p, FeiBootstrap.unit(pIdx.size()));
        String ci = "";
        if (w != null) {
            double[] reps = new double[w.length];
            for (int q = 0; q < w.length; q++) reps[q] = wmean(d, p, w[q]);
            double[] c = FeiNorm.ci(reps);
            ci = String.format(Locale.ROOT, " [%s; %s]%s", fmt(c[0]), fmt(c[1]), c[0] > 0 || c[1] < 0 ? " *" : "");
        }
        int same = 0;
        int zero = 0;
        for (double x : d) {
            if (x == 0) zero++;
            else if (Math.signum(x) == Math.signum(mean)) same++;
        }
        List<Double> dl = new ArrayList<>();
        for (double x : d) dl.add(x);
        return String.format(Locale.ROOT, "%s | %s | %s | %s%s | %s | %d/%d | %d%n", NAMES[k], fmt(median(gm)), fmt(median(pm)), fmt(mean), ci, fmt(median(dl)),
                same, own.size(), zero);
    }

    static String fmt(double x) {
        double a = Math.abs(x);
        return String.format(Locale.ROOT, a >= 100 ? "%.1f" : a >= 1 ? "%.3f" : "%.4f", x);
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                    Path out) throws IOException {
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        Map<String, double[]> det = gtRows(dir.resolve("detections.tsv"));
        // Меры по всем снимкам GT после отбора (ключ — Sample.file().toString()).
        Map<String, double[]> meas = new HashMap<>();
        for (Map.Entry<String, Sample> e : g.samples.entrySet()) {
            Mat f = export.frame(e.getValue(), Variant.A);
            if (f == null) throw new IllegalStateException("Нет кадра а: " + e.getKey());
            double[] fm = frameMeasures(f);
            f.release();
            double[] row = det.get(e.getValue().person() + "/" + XBaseEval.image(e.getValue()));
            if (row == null) throw new IllegalStateException("Нет детекции: " + e.getKey());
            double[] all = new double[MEASURES];
            System.arraycopy(fm, 0, all, 0, fm.length);
            all[MEASURES - 1] = eyeDistance(row);
            meas.put(e.getKey(), all);
        }
        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : null;
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryKeys, g.seed);

        // Ошибка L1 межсессионно: П2 при FAR 1 % по «чужим» галереи, выбор модели по позе (как XBaseSep).
        Path scores = out.resolve("xbase_gt_scores.tsv.gz");
        Map<String, Row> rows = XBaseSep.read(scores);
        Map<String, Double> rOf = new HashMap<>();
        Map<String, String> keyOf = new HashMap<>();
        for (Map.Entry<String, Sample> e : g.samples.entrySet()) {
            String pk = e.getValue().person() + "/" + XBaseEval.image(e.getValue());
            rOf.put(pk, g.rOf.get(e.getKey()));
            keyOf.put(pk, e.getKey());
        }
        XBaseSep.Data cvd = XBaseSep.data(rows, FeiMethods.LBP_MLDA, true, rOf, g.r0, g.pIdx);
        double theta = FeiBootstrap.threshold(cvd.cvR, cvd.cvP, FeiBootstrap.unit(g.persons.size()), 0.01);
        double[][] thetaRep = null;
        if (w != null) {
            thetaRep = new double[1][w.length];
            for (int q = 0; q < w.length; q++) thetaRep[0][q] = FeiBootstrap.threshold(cvd.cvR, cvd.cvP, w[q], 0.01);
        }
        // Пробы своих: человек, r оценки, верный кандидат, ключ.
        List<String> probeKey = new ArrayList<>();
        List<double[]> probe = new ArrayList<>();
        for (Row r : rows.values()) {
            if (!r.method().equals(FeiMethods.LBP_MLDA) || !r.role().equals(FeiScores.OWN)) continue;
            String pk = r.person() + "/" + r.image();
            boolean front = rOf.get(pk) <= g.r0;
            Row c = front ? r : rows.get(FeiMethods.LBP_MLDA + FeiMethods.MIRROR + "\t" + r.role() + "\t" + r.person() + "\t" + r.image());
            probeKey.add(keyOf.get(pk));
            probe.add(new double[] {g.pIdx.get(r.person()), FeiScores.ratio(c, null), c.bestId().equals(c.trueId()) ? 1 : 0});
        }
        long nErr = probe.stream().filter(a -> !(a[2] == 1 && a[1] <= theta)).count();

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 4.2: качество кадров по сессиям GT (XBaseEval gt-quality, без распознавания)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt — кадр а 92×112; detections.tsv)\n");
        t.append("Файл оценок L1: ").append(scores).append(" (").append(FeiThreshold.sha(scores)).append(")\n");
        t.append("Меры (по кадру «а», серый 8 бит, 92×112; параметры не подбирались):\n"
                + "  дисперсия лапласиана — Imgproc.Laplacian, ksize = 1 (ядро 3×3 [0 1 0; 1 −4 1; 0 1 0]), дисперсия по кадру;\n"
                + "  лапласиан / дисперсия яркости — резкость с поправкой на контраст;\n"
                + "  σ шума — Immerkær (1996): √(π/2)·Σ|I ∗ N| / (6 (W − 2)(H − 2)), N = [1 −2 1; −2 4 −2; 1 −2 1], внутренние пиксели;\n"
                + "  средняя яркость и СКО яркости по кадру (0–255);\n"
                + "  межглазье — расстояние между глазами по точкам YuNet в пикселях исходного снимка (detections.tsv).\n"
                + "Оговорка: меры 1–5 — после приведения к 92×112; разрешение исходника видно по межглазью.\n");
        t.append(String.format(Locale.ROOT, "Свои: %d человек; галерея и пробы — как в xbase_gt.txt (межсессионно) и xbase_gt_in.txt (половины даты, "
                + "seed %d + %d).%n", g.own.size(), g.seed, XBaseEval.WITHIN_SEED_OFFSET));
        if (w != null) {
            t.append(String.format(Locale.ROOT, "Бутстреп по людям Georgia Tech целиком: B = %d, seed = %d, людей %d (те же веса, что XBaseEval); "
                    + "[2,5; 97,5] — процентили, «*» — интервал не содержит 0.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        }
        t.append(String.format(Locale.ROOT, "Ошибка L1 (5-lbp-mlda-ratio+pose, П2, FAR 1 %%, межсессионно): порог θ = %.6f по «чужим» галереи; ошибок "
                + "%d из %d проб (как в xbase_gt.txt).%n", theta, nErr, probe.size()));

        String head = "мера | медиана галереи (медиана по людям) | медиана проб (медиана по людям) | разность «пробы − галерея»: среднее по людям "
                + "| медиана по людям | людей с тем же знаком | нулевых\n";
        t.append("\n=== Межсессионно: разность «пробы − галерея» по людям (медианы по снимкам человека) ===\n").append(head);
        for (int k = 0; k < MEASURES; k++) t.append(diffRow(k, g.own, g.galleryKeys, g.probeKeys, meas, g.pIdx, w));
        t.append("\n=== Внутри сессии (половины даты галереи): разность «пробы − галерея» по людям ===\n").append(head);
        for (int k = 0; k < MEASURES; k++) t.append(diffRow(k, g.own, split.get(0), split.get(1), meas, g.pIdx, w));

        t.append("\n=== Ошибка L1 по квартилям разности «проба − медиана галереи её человека» (межсессионно, П2, FAR 1 %) ===\n");
        t.append("ошибка — отказ или неверный кандидат; (ii) — верный кандидат, отказ по порогу; интервалы — бутстреп по людям, порог заново в "
                + "реплике\n");
        t.append("мера | квартиль | границы разности | проб | ошибка | (ii)\n");
        Map<String, Double> galMed = new HashMap<>();
        for (int k = 0; k < MEASURES; k++) {
            for (String p : g.own) {
                List<Double> a = new ArrayList<>();
                for (String key : g.galleryKeys.get(p)) a.add(meas.get(key)[k]);
                galMed.put(p + "\t" + k, median(a));
            }
            double[] diff = new double[probe.size()];
            for (int i = 0; i < probe.size(); i++) {
                String key = probeKey.get(i);
                diff[i] = meas.get(key)[k] - galMed.get(g.samples.get(key).person() + "\t" + k);
            }
            double[] s = FeiBootstrap.sorted(diff);
            double[] q = {FeiBootstrap.quantile(s, 0.25), FeiBootstrap.quantile(s, 0.5), FeiBootstrap.quantile(s, 0.75)};
            String[] bounds = {"≤ " + fmt(q[0]), "(" + fmt(q[0]) + "; " + fmt(q[1]) + "]", "(" + fmt(q[1]) + "; " + fmt(q[2]) + "]", "> " + fmt(q[2])};
            for (int qq = 0; qq < 4; qq++) {
                int n = 0;
                for (double x : diff) if (quartile(x, q) == qq) n++;
                double[] pt = shares(probe, diff, q, qq, FeiBootstrap.unit(g.persons.size()), theta);
                String e = String.format(Locale.ROOT, "%.1f %%", pt[0]);
                String r2 = String.format(Locale.ROOT, "%.1f %%", pt[1]);
                if (w != null) {
                    double[] re = new double[w.length];
                    double[] rr = new double[w.length];
                    for (int b = 0; b < w.length; b++) {
                        double[] x = shares(probe, diff, q, qq, w[b], thetaRep[0][b]);
                        re[b] = x[0];
                        rr[b] = x[1];
                    }
                    double[] ce = FeiNorm.ci(dropNaN(re));
                    double[] cr = FeiNorm.ci(dropNaN(rr));
                    e += String.format(Locale.ROOT, " [%.1f; %.1f]", ce[0], ce[1]);
                    r2 += String.format(Locale.ROOT, " [%.1f; %.1f]", cr[0], cr[1]);
                }
                t.append(String.join(" | ", NAMES[k], "Q" + (qq + 1), bounds[qq], Integer.toString(n), e, r2)).append('\n');
            }
        }
        FarMethods.write(out.resolve("xbase_gt_quality.txt"), t);
        System.out.print(t);

        // Таблица по снимкам.
        Map<String, String> crossRole = new HashMap<>();
        Map<String, String> inRole = new HashMap<>();
        for (String p : g.own) {
            for (String key : g.galleryKeys.get(p)) crossRole.put(key, "gallery");
            for (String key : g.probeKeys.get(p)) crossRole.put(key, "probe");
            for (String key : split.get(0).get(p)) inRole.put(key, "gallery");
            for (String key : split.get(1).get(p)) inRole.put(key, "probe");
        }
        Map<String, Integer> dateOf = new HashMap<>();
        g.byPerson.forEach((p, m) -> m.forEach((d, ks) -> ks.forEach(k -> dateOf.put(k, d))));
        StringBuilder tsv = new StringBuilder("person\timage\tdate\tcross_role\twithin_role");
        for (String id : IDS) tsv.append('\t').append(id);
        tsv.append('\n');
        g.samples.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach(e -> {
            String key = e.getKey();
            tsv.append(e.getValue().person()).append('\t').append(XBaseEval.image(e.getValue())).append('\t').append(dateOf.get(key)).append('\t')
                    .append(crossRole.getOrDefault(key, g.own.contains(e.getValue().person()) ? "-" : "ctrl")).append('\t').append(inRole.getOrDefault(key, "-"));
            for (double v : meas.get(key)) tsv.append('\t').append(String.format(Locale.ROOT, "%.6f", v));
            tsv.append('\n');
        });
        Files.writeString(out.resolve("xbase_gt_quality.tsv"), tsv, StandardCharsets.UTF_8);
    }

    /** Доли ошибки и (ii) в квартиле qq при весах w и пороге θ, %; пустая группа — NaN. */
    static double[] shares(List<double[]> probe, double[] diff, double[] q, int qq, int[] w, double theta) {
        double n = 0;
        double err = 0;
        double rej = 0;
        for (int i = 0; i < probe.size(); i++) {
            if (quartile(diff[i], q) != qq) continue;
            double[] a = probe.get(i);
            int wt = w[(int) a[0]];
            n += wt;
            boolean ok = a[2] == 1;
            boolean acc = a[1] <= theta;
            if (!(ok && acc)) err += wt;
            if (ok && !acc) rej += wt;
        }
        return n > 0 ? new double[] {100 * err / n, 100 * rej / n} : new double[] {Double.NaN, Double.NaN};
    }

    static double[] dropNaN(double[] v) {
        return java.util.Arrays.stream(v).filter(x -> !Double.isNaN(x)).toArray();
    }
}
