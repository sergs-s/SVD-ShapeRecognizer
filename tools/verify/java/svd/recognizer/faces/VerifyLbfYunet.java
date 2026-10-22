package svd.recognizer.faces;

import java.io.IOException;
import java.io.Writer;
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
import java.util.TreeSet;
import java.util.function.Predicate;
import org.opencv.core.Mat;
import org.opencv.face.Facemark;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.HaarFrames.Det;
import svd.recognizer.faces.LbfFrames.Lm;

/**
 * (4б) «рамка YuNet + точки LBF» без распознавания (tools/verify, ветка cloud-verify; Жорик №14 от 20.10.2026): точки Facemark LBF
 * по рамке YuNet (derived/detections.tsv, поля 0–3 как есть – LbfFrames.fit округляет до целых, как рамку Хаара) на fa/fb; центры
 * глаз – LbfFrames.Lm.five() (средние 36–41 и 42–47); расхождение с ручной разметкой – FeretData.eyeError, снимки – как в
 * LbfFrames.manual (размечены, YuNet нашёл, |r| ≤ R); рядом Ю5 и LBF по рамке Хаара (derived/lbf.tsv). Код проекта не меняется.
 *
 * Аргументы: каталог моделей, корень FERET, fei_selection.tsv (строка «# R»), кэш LBF по YuNet (читается, если есть, иначе
 * считается и пишется), отчёт.
 */
public final class VerifyLbfYunet {

    static final long SEED = 42 + FeiBootstrap.SEED_OFFSET;
    static final int B = 2000;

    private VerifyLbfYunet() {
    }

    static Map<String, double[]> yunet(Path feret) throws IOException {
        Map<String, double[]> out = new HashMap<>();
        for (String l : Files.readAllLines(feret.resolve(FeretData.DERIVED).resolve(FeretData.DETECTIONS), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (c.length != 17) continue;
            double[] row = null;
            if (c[1].equals("1")) {
                row = new double[15];
                for (int i = 0; i < 15; i++) row[i] = Double.parseDouble(c[2 + i]);
            }
            out.put(c[0], row);
        }
        return out;
    }

    static Map<String, Lm> lbf(Path file) throws IOException {
        Map<String, Lm> out = new HashMap<>();
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#")) continue;
            String[] c = l.split("\t", -1);
            if (!c[1].equals("1")) {
                out.put(c[0], null);
                continue;
            }
            double[][] p = new double[LbfFrames.POINTS][];
            for (int i = 0; i < LbfFrames.POINTS; i++) p[i] = new double[] {Double.parseDouble(c[2 + 2 * i]), Double.parseDouble(c[3 + 2 * i])};
            out.put(c[0], new Lm(p));
        }
        return out;
    }

    static void compute(Path models, List<Img> imgs, Map<String, double[]> yu, Path cache) throws IOException {
        Facemark fm = LbfFrames.load(models);
        int n = 0;
        try (Writer w = Files.newBufferedWriter(cache, StandardCharsets.UTF_8)) {
            w.write("# lbf по рамке YuNet (detections.tsv, поля 0–3)|" + LbfFrames.signature(models) + "\n");
            for (Img im : imgs) {
                double[] row = yu.get(im.key());
                if (row == null) continue;
                Mat color = FeretData.decode(im.file());
                Mat in = HaarFrames.input(color, 1.0);
                color.release();
                Lm lm = LbfFrames.fit(fm, in, 1.0, new Det(1, new double[] {row[0], row[1], row[2], row[3]}, null, null));
                in.release();
                StringBuilder m = new StringBuilder(im.key()).append('\t').append(lm == null ? "0" : "1");
                for (int i = 0; i < LbfFrames.POINTS; i++)
                    m.append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][0])).append('\t').append(lm == null ? "" : Double.toString(lm.p()[i][1]));
                w.write(m.append('\n').toString());
                if (++n % 250 == 0) {
                    w.flush();
                    System.out.printf(Locale.ROOT, "VerifyLbfYunet: %d%n", n);
                }
            }
        }
    }

    static double[] err(Img im, Lm lm) {
        if (lm == null) return null;
        double[] f = lm.five();
        double[] e = FeretData.eyeError(im, new double[] {f[4], f[5]}, new double[] {f[6], f[7]});
        return new double[] {e[1], e[0], e[2]};
    }

    /** Меры по постановке №14 (среднее двух глаз на снимок): медиана, 90 %, 95 %, доли > 0,1 и > 0,25. */
    static String measures(String group, String name, double[][] e, List<Integer> idx) {
        double[] m = FeiBootstrap.sorted(idx.stream().mapToDouble(i -> e[i][2]).toArray());
        double[] g = FeiBootstrap.sorted(idx.stream().flatMapToDouble(i -> Arrays.stream(new double[] {e[i][0], e[i][1]})).toArray());
        return String.format(Locale.ROOT, "%s | %s | %d | %.4f | %.4f | %.4f | %.1f %% | %.1f %% | %.4f | %.4f | %.4f | %.1f %% | %.1f %%%n", group, name,
                idx.size(), FeiBootstrap.quantile(m, 0.5), FeiBootstrap.quantile(m, 0.9), FeiBootstrap.quantile(m, 0.95), 100 * XBaseEyePts.share(m, 0.10),
                100 * XBaseEyePts.share(m, 0.25), FeiBootstrap.quantile(g, 0.5), FeiBootstrap.quantile(g, 0.9), FeiBootstrap.quantile(g, 0.95),
                100 * XBaseEyePts.share(g, 0.10), 100 * XBaseEyePts.share(g, 0.25));
    }

    /** Парная разность «c − y» по мерам №14 (медиана, 90 %, 95 %, доли > 0,1 и > 0,25), бутстреп по людям. */
    static String pair(String name, double[] c, double[] y, int[] owner, int[][] wp) {
        int n = c.length;
        double[] qs = {0.5, 0.9, 0.95};
        double[] ts = {0.10, 0.25};
        int k = qs.length + ts.length;
        int[] unit = new int[n];
        Arrays.fill(unit, 1);
        double[] point = new double[k];
        double[][] rep = new double[k][wp.length];
        for (int q = -1; q < wp.length; q++) {
            int[] wi = unit;
            if (q >= 0) {
                wi = new int[n];
                for (int i = 0; i < n; i++) wi[i] = wp[q][owner[i]];
            }
            double[] v = new double[k];
            for (int j = 0; j < qs.length; j++) v[j] = XBaseEyePts.weightedQuantile(c, wi, qs[j]) - XBaseEyePts.weightedQuantile(y, wi, qs[j]);
            for (int j = 0; j < ts.length; j++)
                v[qs.length + j] = XBaseEyePts.weightedShare(c, wi, ts[j]) - XBaseEyePts.weightedShare(y, wi, ts[j]);
            for (int j = 0; j < k; j++) {
                if (q < 0) point[j] = v[j];
                else rep[j][q] = v[j];
            }
        }
        List<String> cells = new ArrayList<>(List.of(name, Integer.toString(n)));
        for (int j = 0; j < k; j++) cells.add(j < qs.length ? XBaseEyePts.diff(point[j], rep[j], 1, "%+.4f") : XBaseEyePts.diff(point[j], rep[j], 100, "%+.1f"));
        return String.join(" | ", cells) + "\n";
    }

    /** Среднее и медиана смещения (dx, dy) точки глаза от ручной, доли ручного межглазья; x – по снимку (вправо +). */
    static String offset(String name, List<Img> imgs, List<double[][]> pts) {
        List<Double> dx = new ArrayList<>();
        List<Double> dy = new ArrayList<>();
        for (int i = 0; i < imgs.size(); i++) {
            double[][] p = pts.get(i);
            if (p == null) continue;
            Img im = imgs.get(i);
            double iod = Math.hypot(im.leftEye()[0] - im.rightEye()[0], im.leftEye()[1] - im.rightEye()[1]);
            dx.add((p[0][0] - im.rightEye()[0]) / iod);
            dy.add((p[0][1] - im.rightEye()[1]) / iod);
            dx.add((p[1][0] - im.leftEye()[0]) / iod);
            dy.add((p[1][1] - im.leftEye()[1]) / iod);
        }
        double[] x = FeiBootstrap.sorted(dx.stream().mapToDouble(Double::doubleValue).toArray());
        double[] y = FeiBootstrap.sorted(dy.stream().mapToDouble(Double::doubleValue).toArray());
        return String.format(Locale.ROOT, "%s | %d | %+.4f | %+.4f | %+.4f | %+.4f%n", name, x.length, Arrays.stream(x).average().orElse(Double.NaN),
                FeiBootstrap.quantile(x, 0.5), Arrays.stream(y).average().orElse(Double.NaN), FeiBootstrap.quantile(y, 0.5));
    }

    public static void main(String[] a) throws IOException {
        svd.recognizer.OpenCvLoader.load();
        Path models = Paths.get(a[0]);
        Path feret = Paths.get(a[1]);
        double rLimit = XBaseEval.poseLimit(Files.readAllLines(Paths.get(a[2]), StandardCharsets.UTF_8));
        Path cache = Paths.get(a[3]);
        Path out = Paths.get(a[4]);
        List<Img> fr = FeretData.frontal(feret);
        Map<String, double[]> yu = yunet(feret);
        if (!Files.exists(cache)) compute(models, fr, yu, cache);
        Map<String, Lm> lY = lbf(cache);
        Map<String, Lm> lH = lbf(feret.resolve(FeretData.DERIVED).resolve(LbfFrames.CACHE));
        Map<String, Det> haar = new HashMap<>();
        for (String l : Files.readAllLines(feret.resolve(FeretData.DERIVED).resolve(HaarFrames.CACHE), StandardCharsets.UTF_8)) {
            String[] c = l.split("\t", -1);
            if (c.length != 12) continue;
            double[] f = c[2].isEmpty() ? null : new double[] {Double.parseDouble(c[2]), Double.parseDouble(c[3]), Double.parseDouble(c[4]), Double.parseDouble(c[5])};
            haar.put(c[0], new Det(Integer.parseInt(c[1]), f, null, null));
        }

        List<Img> imgs = new ArrayList<>();
        for (Img im : fr) {
            double[] row = yu.get(im.key());
            if (im.leftEye() == null || im.rightEye() == null || row == null || Math.abs(FeiDataset.pose(row)) > rLimit) continue;
            imgs.add(im);
        }
        int n = imgs.size();
        double[][] eY = new double[n][];
        double[][] eH = new double[n][];
        double[][] eLY = new double[n][];
        List<double[][]> pY = new ArrayList<>();
        List<double[][]> pH = new ArrayList<>();
        List<double[][]> pLY = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Img im = imgs.get(i);
            double[] row = yu.get(im.key());
            double[] e = FeretData.eyeError(im, new double[] {row[4], row[5]}, new double[] {row[6], row[7]});
            eY[i] = new double[] {e[1], e[0], e[2]};
            eH[i] = err(im, lH.get(im.key()));
            eLY[i] = err(im, lY.get(im.key()));
            pY.add(new double[][] {{row[4], row[5]}, {row[6], row[7]}});
            for (Object[] o : new Object[][] {{lH, pH}, {lY, pLY}}) {
                @SuppressWarnings("unchecked") Lm lm = ((Map<String, Lm>) o[0]).get(im.key());
                @SuppressWarnings("unchecked") List<double[][]> dst = (List<double[][]>) o[1];
                dst.add(lm == null ? null : new double[][] {{lm.five()[4], lm.five()[5]}, {lm.five()[6], lm.five()[7]}});
            }
        }
        List<String> persons = new ArrayList<>(new TreeSet<>(imgs.stream().map(Img::person).toList()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        int[][] wp = FeiBootstrap.weights(persons.size(), B, SEED);

        StringBuilder r = new StringBuilder();
        long fa = fr.stream().filter(im -> yu.get(im.key()) != null).count();
        long fitY = lY.values().stream().filter(v -> v != null).count();
        r.append("(4б) «рамка YuNet + точки LBF», без распознавания – FERET fa/fb (Жорик №14 от 20.10.2026). Код проекта не менялся.\n");
        r.append(String.format(Locale.ROOT, "fa/fb: %d; YuNet нашёл лицо: %d; LBF по рамке YuNet дал точки: %d (не дал: %d).%n", fr.size(), fa, fitY, lY.size() - fitY));
        r.append("Рамка YuNet – поля 0–3 строки detections.tsv как есть (x, y, w, h), LbfFrames.fit округляет их до целых (LbfFrames.java:111–115).\n");
        r.append("Центр глаза LBF – LbfFrames.Lm.five(): средние точек 36–41 (глаз слева на снимке) и 42–47 (LbfFrames.java:64–72, 74–83).\n");
        r.append("Расхождение – FeretData.eyeError (FeretData.java:335–354), доли ручного межглазья; снимки – как LbfFrames.manual (LbfFrames.java:323–330):\n");
        r.append(String.format(Locale.ROOT, "размечены оба глаза, YuNet нашёл, |r| ≤ R = %.4f: %d снимков, людей %d.%n", rLimit, n, persons.size()));
        r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d (FeiBootstrap.weights); «*» – интервал разности не содержит 0.%n", B, SEED));

        Map<String, Predicate<Img>> groups = new LinkedHashMap<>();
        groups.put("все", im -> true);
        groups.put("без очков", im -> !(im.flag().equals("a") || im.flag().equals("c")));
        groups.put("в очках (a, c)", im -> im.flag().equals("a") || im.flag().equals("c"));
        StringBuilder t1 = new StringBuilder("\n=== Формат feret-eyepts / abl-lbf (XBaseMpPts.errLine) ===\n"
                + "группа | кандидат | снимков | среднее: медиана | 95 % | макс | > 5 % | > 10 % | глаза: медиана | > 5 % | > 10 %\n");
        StringBuilder t2 = new StringBuilder("\n=== Меры №14 (среднее двух глаз на снимок; рядом – по отдельным глазам) ===\n"
                + "группа | кандидат | снимков | медиана | 90 % | 95 % | > 0,1 | > 0,25 | глаза: медиана | 90 % | 95 % | > 0,1 | > 0,25\n");
        for (Map.Entry<String, Predicate<Img>> g : groups.entrySet()) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) if (g.getValue().test(imgs.get(i))) idx.add(i);
            List<Integer> iY = idx.stream().filter(i -> eLY[i] != null).toList();
            List<Integer> iH = idx.stream().filter(i -> eH[i] != null).toList();
            List<Integer> iB = iY.stream().filter(i -> eH[i] != null).toList();
            Object[][] rows = {{"Ю5", eY, idx}, {"LBF по рамке Хаара", eH, iH}, {"LBF по рамке YuNet", eLY, iY}, {"Ю5 (где LBF YuNet)", eY, iY},
                {"LBF YuNet (где есть оба LBF)", eLY, iB}, {"LBF Хаар (где есть оба LBF)", eH, iB}};
            for (Object[] o : rows) {
                @SuppressWarnings("unchecked") List<Integer> ii = (List<Integer>) o[2];
                t1.append(XBaseMpPts.errLine(g.getKey(), (String) o[0], (double[][]) o[1], ii));
                t2.append(measures(g.getKey(), (String) o[0], (double[][]) o[1], ii));
            }
        }
        r.append(t1).append(t2);

        r.append("\n=== Парные разности (среднее двух глаз на снимок), бутстреп по людям; «−» – первый точнее ===\n");
        r.append("разность | снимков | медиана | 90 % | 95 % | доля > 0,1, п. п. | доля > 0,25, п. п.\n");
        List<Integer> li = new ArrayList<>();
        for (int i = 0; i < n; i++) if (eLY[i] != null) li.add(i);
        int[] ow = li.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray();
        r.append(pair("LBF по YuNet − Ю5", li.stream().mapToDouble(i -> eLY[i][2]).toArray(), li.stream().mapToDouble(i -> eY[i][2]).toArray(), ow, wp));
        List<Integer> lb = li.stream().filter(i -> eH[i] != null).toList();
        int[] ob = lb.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray();
        r.append(pair("LBF по YuNet − LBF по Хаару", lb.stream().mapToDouble(i -> eLY[i][2]).toArray(), lb.stream().mapToDouble(i -> eH[i][2]).toArray(), ob, wp));
        r.append("Формат abl-lbf (XBaseMpPts.pairDiff: медиана, доли > 5 % и > 10 %):\n");
        r.append((String) XBaseMpPts.pairDiff("глаза: LBF по YuNet − Ю5", li.stream().mapToDouble(i -> eLY[i][2]).toArray(),
                li.stream().mapToDouble(i -> eY[i][2]).toArray(), ow, wp)[0]);
        List<Integer> lh = new ArrayList<>();
        for (int i = 0; i < n; i++) if (eH[i] != null) lh.add(i);
        r.append((String) XBaseMpPts.pairDiff("глаза: LBF (Хаар) − Ю5 [сверка с abl-lbf]", lh.stream().mapToDouble(i -> eH[i][2]).toArray(),
                lh.stream().mapToDouble(i -> eY[i][2]).toArray(), lh.stream().mapToInt(i -> pIdx.get(imgs.get(i).person())).toArray(), wp)[0]);

        r.append("\n=== Смещение центров глаз от ручных (доли ручного межглазья; x вправо по снимку, y вниз; оба глаза) ===\n");
        r.append("кандидат | точек | dx среднее | dx медиана | dy среднее | dy медиана\n");
        r.append(offset("Ю5", imgs, pY)).append(offset("LBF по рамке Хаара", imgs, pH)).append(offset("LBF по рамке YuNet", imgs, pLY));

        r.append("\n=== Рамки YuNet и Хаара на этих снимках (где есть обе; доли ширины рамки YuNet) ===\n");
        List<double[]> bx = new ArrayList<>();
        for (Img im : imgs) {
            double[] y = yu.get(im.key());
            Det h = haar.get(im.key());
            if (h == null || h.face() == null) continue;
            double[] f = h.face();
            bx.add(new double[] {f[2] / y[2], f[3] / y[3], (f[0] + f[2] / 2 - y[0] - y[2] / 2) / y[2], (f[1] + f[3] / 2 - y[1] - y[3] / 2) / y[2],
                y[3] / y[2]});
        }
        String[] names = {"ширина Хаара / ширина YuNet", "высота Хаара / высота YuNet", "центр Хаара − центр YuNet по x", "центр Хаара − центр YuNet по y",
            "высота / ширина YuNet"};
        r.append("величина | снимков | 5 % | 25 % | 50 % | 75 % | 95 %\n");
        for (int j = 0; j < names.length; j++) {
            final int jj = j;
            double[] s = FeiBootstrap.sorted(bx.stream().mapToDouble(v -> v[jj]).toArray());
            r.append(String.format(Locale.ROOT, "%s | %d | %.3f | %.3f | %.3f | %.3f | %.3f%n", names[j], s.length, FeiBootstrap.quantile(s, 0.05),
                    FeiBootstrap.quantile(s, 0.25), FeiBootstrap.quantile(s, 0.5), FeiBootstrap.quantile(s, 0.75), FeiBootstrap.quantile(s, 0.95)));
        }
        Files.writeString(out, r.toString(), StandardCharsets.UTF_8);
        System.out.print(r);
    }
}
