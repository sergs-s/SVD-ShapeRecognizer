package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.storage.SettingsStore;

/**
 * Пачка 4 – подстройка параметров конфигурации (32) (задание Жорика №1 от 01.11.2026, дополнение №4). Режим p4-retina-sheet –
 * лист кадров «а» 8 снимков GT: оригинал, Тан – Триггс (параметры конфигурации), ретинальный фильтр Vu (RetinaFilter, параметры
 * автора) – для проверки глазами до прогона; рядом – доля усечённых значений (у автора 2–4 %).
 *
 * @author ssv
 */
final class XBasePart4 {

    private XBasePart4() {
    }

    /** Вещественный кадр → 8 бит по min..max кадра. */
    static Mat toGray(double[] v, int w, int h) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) b[i] = (byte) (hi > lo ? Math.round(255 * (v[i] - lo) / (hi - lo)) : 128);
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        m.put(0, 0, b);
        return m;
    }

    /** Режим p4-retina-sheet: кадры GT из экспорта (a/gt/sNN/MM.png), человек 1 + 7i, снимок i + 1, i = 0…7. */
    static void retinaSheet(SettingsStore settings, Path out) throws IOException {
        String ed = System.getenv("FACES_EXPORT_DIR");
        if (ed == null) throw new IllegalStateException("p4-retina-sheet: задайте FACES_EXPORT_DIR");
        IlluminationNorm.NormParams p = IlluminationNorm.NormParams.of(settings);
        Path o = out.resolve("part4");
        Files.createDirectories(o);
        StringBuilder rep = new StringBuilder("Пачка 4, Б1′: лист кадров – оригинал | Тан – Триггс | ретинальный фильтр Vu (XBaseEval p4-retina-sheet)\nКод: коммит "
                + FarMethods.commit() + "\n");
        rep.append(String.format(Locale.ROOT, "Тан – Триггс: γ = %s, σ₀ = %s, σ₁ = %s, α = %s, τ = %s; ретинальный: σ₁ = σ₂ = %s, σ_Ph = %s, σ_H = %s, Th = %s "
                + "(дисс. Vu, 2010, разд. 3.4.1.3; кадры автора – Yale B 192×168, у нас 92×112, σ не масштабируются)%n", p.gamma(), p.sigma0(), p.sigma1(),
                p.alpha(), p.tau(), RetinaFilter.AUTHOR.sigma1(), RetinaFilter.AUTHOR.sigmaPh(), RetinaFilter.AUTHOR.sigmaH(), RetinaFilter.AUTHOR.th()));
        rep.append("снимок | доля усечённых (|I_nor| ≥ Th)\n");
        List<Mat> rows = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            String key = String.format(Locale.ROOT, "gt/s%02d/%02d.png", 1 + 7 * i, i + 1);
            Path f = Paths.get(ed, "a").resolve(key);
            Mat a = Imgcodecs.imdecode(new MatOfByte(Files.readAllBytes(f)), Imgcodecs.IMREAD_GRAYSCALE);
            double[] tt = IlluminationNorm.TAN_TRIGGS.vector(a, p);
            double[] rf = RetinaFilter.filter(a, RetinaFilter.AUTHOR);
            rep.append(String.format(Locale.ROOT, "%s | %.2f %%%n", key, 100 * RetinaFilter.clippedShare(rf, RetinaFilter.AUTHOR)));
            List<Mat> tiles = List.of(a, toGray(tt, a.cols(), a.rows()), toGray(rf, a.cols(), a.rows()));
            Mat row = new Mat();
            Core.hconcat(tiles, row);
            rows.add(row);
        }
        Mat sheet = new Mat();
        Core.vconcat(rows, sheet);
        Mat big = new Mat();
        Imgproc.resize(sheet, big, new Size(sheet.cols() * 3, sheet.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2BGR);
        String[] cap = {"orig", "Tan-Triggs", "retina Vu"};
        for (int c = 0; c < 3; c++) Imgproc.putText(big, cap[c], new Point(c * 92 * 3 + 6, 22), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 255), 2);
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".png", big, buf);
        Files.write(o.resolve("p4_retina_sheet.png"), buf.toArray());
        FarMethods.write(o.resolve("p4_retina_sheet.txt"), rep);
        System.out.print(rep);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- А: β по кандидату, r0 (без обучения, векторы v8 пачки 2)

    /** Вариант 8 пачки 2 – (б) (П2) + MS-LBP, конфигурация (32). */
    static final int V8 = XBaseFeat2.MS_JOIN;
    static final double[] BETAS = {0.5, 1, 2};
    static final String[] BETA_S = {"0,5", "1", "2"};
    /** r0 у L1-tt-xs-F: база 0,12 первой, затем 0,08 и 0,18. */
    static final double[] R0S = {0.12, 0.08, 0.18};

    /** Сочетания (β_tt, β_xs) номерами BETAS: база (1; 1) первой, далее по порядку {0,5; 1; 2}² без базы. */
    static int[][] betaPairs() {
        List<int[]> l = new ArrayList<>();
        l.add(new int[] {1, 1});
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (i != 1 || j != 1) l.add(new int[] {i, j});
        return l.toArray(new int[0][]);
    }

    static String[] names(int kind) {
        if (kind == 1) return new String[] {"r0 = 0,12 (база)", "r0 = 0,08", "r0 = 0,18"};
        int[][] bp = betaPairs();
        String[] n = new String[bp.length];
        for (int i = 0; i < bp.length; i++) n[i] = "β_tt = " + BETA_S[bp[i][0]] + ", β_xs = " + BETA_S[bp[i][1]] + (i == 0 ? " (база)" : "");
        return n;
    }

    /**
     * Слияние с β по кандидату: ℓ(g) = w ẑ_tt(g) + (1 − w) ẑ_xs(g); лучший b и второй c – по ℓ; s = w[(1+β_tt) ẑ_tt(b) − β_tt ẑ_tt(c)] +
     * (1 − w)[(1+β_xs) ẑ_xs(b) − β_xs ẑ_xs(c)]; при β_tt = β_xs = β – s = (1+β) ℓ(b) − β ℓ(c) (как XBaseVec.decide; при β = 1 строки
     * побайтно XBaseVec.fuseLines). Модель L1-tt-xs-F – по позе (|r| ≤ r0 – X), L1-xs-F – X+mirror.
     */
    static String fuseBeta(List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, double w, double[] st, double[] sx, Map<String, Double> rOf, double r0, double btt,
                           double bxs, int[] clamped) {
        StringBuilder out = new StringBuilder();
        Map<String, XBaseVec.Model> byTask = new LinkedHashMap<>();
        for (XBaseVec.Model m : tt) byTask.putIfAbsent(m.task(), m);
        for (String task : byTask.keySet()) {
            XBaseVec.Model tX = XBaseVec.find(tt, task, false);
            XBaseVec.Model tM = XBaseVec.find(tt, task, true);
            XBaseVec.Model xM = XBaseVec.find(xs, task, true);
            if (!tX.gallery().equals(xM.gallery()) || tX.rows().size() != xM.rows().size()) throw new IllegalStateException("p4a: слияние – разные задачи");
            for (int i = 0; i < tX.rows().size(); i++) {
                XBaseVec.Row rx = xM.rows().get(i);
                XBaseVec.Row r0r = tX.rows().get(i);
                if (!rx.key().equals(r0r.key()) || !rx.role().equals(r0r.role())) throw new IllegalStateException("p4a: слияние – разные пробы");
                XBaseVec.Row rt = rOf.get(r0r.key()) <= r0 ? r0r : tM.rows().get(i);
                int ng = rt.dg().length;
                double[] zt = new double[ng];
                double[] zx = new double[ng];
                double[] ln = new double[ng];
                for (int g = 0; g < ng; g++) {
                    if (Double.isInfinite(rt.dg()[g]) || Double.isInfinite(rx.dg()[g])) {
                        ln[g] = Double.POSITIVE_INFINITY;
                        continue;
                    }
                    zt[g] = (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1];
                    zx[g] = (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                    ln[g] = w * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - w) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                }
                int b = -1;
                int c = -1;
                for (int g = 0; g < ng; g++) {
                    if (b < 0 || ln[g] < ln[b]) {
                        c = b;
                        b = g;
                    } else if (c < 0 || ln[g] < ln[c]) {
                        c = g;
                    }
                }
                for (XBaseVec.Model m : List.of(tX, tM)) {
                    if (c < 0 || Double.isInfinite(ln[c]) || Double.isInfinite(ln[b])) {
                        out.append(FeiScores.line(m.method(), m.s(), 0, rt.role(), rt.person(), rt.image(), rt.trueIdx(), m.gallery(), rt.dg().clone(), false));
                        continue;
                    }
                    double s = btt == bxs ? (1 + btt) * ln[b] - btt * ln[c] : w * ((1 + btt) * zt[b] - btt * zt[c]) + (1 - w) * ((1 + bxs) * zx[b] - bxs * zx[c]);
                    double[] v = new double[ng];
                    for (int g = 0; g < ng; g++) v[g] = XBaseAbl.ABS_C * 10;
                    v[b] = XBaseAbl.ABS_C * XBaseVec.squash(s, clamped);
                    v[c] = XBaseAbl.ABS_C;
                    out.append(FeiScores.line(m.method(), m.s(), 0, rt.role(), rt.person(), rt.image(), rt.trueIdx(), m.gallery(), v, false));
                }
            }
        }
        return out.toString();
    }

    /** Строки варианта слияния блока: kind 0 – β (номер сочетания), 1 – r0 (номер); база сверяется с XBaseVec.fuseLines побайтно. */
    static String fused(XBaseVec.Blk b, List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, boolean pool, int kind, int k, int[] clamped) {
        double[] st = XBaseVec.fuseStats(tt, pool);
        double[] sx = XBaseVec.fuseStats(xs, pool);
        if (Math.abs(b.g().r0 - R0S[0]) > 1e-12) throw new IllegalStateException("p4a: r0 блока " + b.g().r0 + " ≠ 0,12");
        int[] bp = kind == 0 ? betaPairs()[k] : new int[] {1, 1};
        double r0 = kind == 1 && k > 0 ? R0S[k] : b.g().r0;
        String l = fuseBeta(tt, xs, XBaseFeat2.W_FUSE, st, sx, b.g().rOf, r0, BETAS[bp[0]], BETAS[bp[1]], clamped);
        if (k == 0 && !l.equals(XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, st, sx, b.g().rOf, b.g().r0, null))) {
            throw new IllegalStateException("p4a: база ≠ XBaseVec.fuseLines (" + b.name() + ")");
        }
        return l;
    }

    static final String NOTE_A = "Пачка 4, А (без обучения, векторы v8 пачки 2 – (б) (П2) + MS-LBP, конфигурация (32)): 1) β по кандидату – лучший и второй по "
            + "слитому ℓ,\ns = w[(1+β_tt) ẑ_tt(1) − β_tt ẑ_tt(2)] + (1 − w)[(1+β_xs) ẑ_xs(1) − β_xs ẑ_xs(2)], w = 0,75; 2) r0 выбора модели L1-tt-xs-F (β = 1). "
            + "Решающее – слияние;\nкандидаты отдельно при β ∈ {0,5; 1; 2} – справочно. Правило – уточнённое (в пределах 1 п. п. во всех трёх наборах, самый "
            + "простой – база первой).\n";

    /** pool-p4a. */
    static void poolA(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0);
        for (Mat m : u.fr.values()) m.release();
        Path vd = XBaseFeat2.dir(u, 0);
        int[] clamped = new int[2];
        List<String> sets = u.groups;
        double[][][] crit = new double[2][sets.size()][];
        double[][][] cand = new double[2][sets.size()][BETAS.length];
        for (int s = 0; s < sets.size(); s++) {
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(sets.get(s))) bi.add(i);
            List<List<XBaseVec.Model>> tt = new ArrayList<>();
            List<List<XBaseVec.Model>> xs = new ArrayList<>();
            for (int k : bi) {
                tt.add(XBaseFeat2.read(vd, u.blocks.get(k), 1, V8));
                xs.add(XBaseFeat2.read(vd, u.blocks.get(k), 0, V8));
            }
            for (int kind = 0; kind < 2; kind++) {
                int n = names(kind).length;
                crit[kind][s] = new double[n];
                for (int v = 0; v < n; v++) {
                    List<String> ls = new ArrayList<>();
                    for (int k = 0; k < bi.size(); k++) ls.add(fused(u.blocks.get(bi.get(k)), tt.get(k), xs.get(k), true, kind, v, clamped));
                    crit[kind][s][v] = XBaseFeat2.poolSep(u, bi, ls, 1, true).crit();
                }
            }
            for (int f = 0; f < 2; f++) for (int j = 0; j < BETAS.length; j++) {
                List<String> ls = new ArrayList<>();
                for (int k = 0; k < bi.size(); k++) {
                    StringBuilder sb = new StringBuilder();
                    for (XBaseVec.Model m : f == 1 ? tt.get(k) : xs.get(k)) for (XBaseVec.Row r : m.rows()) sb.append(XBaseVec.line(m.method(), m.s(), m.gallery(), r));
                    ls.add(XBaseAbl3.beta(sb.toString(), BETAS[j]));
                }
                cand[f][s][j] = XBaseFeat2.poolSep(u, bi, ls, f, false).crit();
            }
        }
        StringBuilder t = new StringBuilder("Пачка 4, А: β по кандидату и r0 – pool (XBaseEval pool-p4a; векторы " + vd + ")\nКод: коммит " + FarMethods.commit() + "\n")
                .append(NOTE_A);
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBasePart4 pool-p4a): часть, номер, вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        String[] title = {"А1. β по кандидату – слияние w = 0,75: вариант × набор (критерий, в скобках – отставание от лучшего, п. п.)",
            "А2. r0 у L1-tt-xs-F – слияние w = 0,75: вариант × набор"};
        StringBuilder gt = new StringBuilder("\n(Набор GT; наборы FERET – только в D:\\data\\Feret\\reports\\eq\\)\n");
        for (int kind = 0; kind < 2; kind++) {
            String[] nm = names(kind);
            int ch = XBaseFeat2.chooseAll(crit[kind]);
            t.append(XBaseFeat2.lagTable(title[kind], nm, crit[kind], sets, ch));
            if (ch >= 0) {
                t.append("Выбор: ").append(nm[ch]).append('\n');
            } else {
                int b = 0;
                for (int v = 1; v < nm.length; v++) if (XBaseFeat2.maxLag(crit[kind], v) < XBaseFeat2.maxLag(crit[kind], b)) b = v;
                t.append(String.format(Locale.ROOT, "Выбор: во всех трёх в пределах 1 п. п. – нет; наименьшее наибольшее отставание – %s (%.1f), условие Δ – на своих%n",
                        nm[b], XBaseFeat2.maxLag(crit[kind], b)));
            }
            sel.append(kind == 0 ? "β" : "r0").append('\t').append(ch).append('\t').append(ch >= 0 ? nm[ch] : "нет").append('\n');
            for (int v = 0; v < nm.length; v++) gt.append(String.format(Locale.ROOT, "%s | %.1f %%%n", nm[v], crit[kind][0][v]));
        }
        for (int f = 1; f >= 0; f--) {
            String[] nm = new String[BETAS.length];
            double[][] c = new double[sets.size()][];
            for (int j = 0; j < BETAS.length; j++) nm[j] = XBaseRule.FEATS[f] + ", β = " + BETA_S[j] + (j == 1 ? " (база)" : "");
            for (int s = 0; s < sets.size(); s++) c[s] = cand[f][s];
            t.append(XBaseFeat2.lagTable("Справочно: " + XBaseRule.FEATS[f] + " отдельно, β × набор", nm, c, sets, -1));
        }
        Files.writeString(u.out.resolve("xbase_pool_p4a_select.tsv"), sel.toString());
        FarMethods.write(u.out.resolve("xbase_pool_p4a.txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_p4a_gt.txt"), new StringBuilder(t.substring(0, t.indexOf("\n===") + 1)).append(gt));
        System.out.print(t);
        System.out.println("DONE");
    }

    /** feret-p4a, gt-p4a: все варианты А на своих; пары «вариант − база»; П3; условие Δ (GT внутри). */
    static void ownA(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        XBaseVec.Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, 0);
        for (Mat m : u.fr.values()) m.release();
        Path vd = XBaseFeat2.dir(u, 0);
        Path sf = (base.equals("gt") ? u.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : u.out).resolve("xbase_pool_p4a_select.tsv");
        StringBuilder t = new StringBuilder("Пачка 4, А: β по кандидату и r0 – " + base + " (XBaseEval " + base + "-p4a; векторы " + vd + ")\nКод: коммит "
                + FarMethods.commit() + "\n").append(NOTE_A);
        t.append("\n=== Выбор по пулу (").append(sf.getFileName()).append(") ===\n");
        if (Files.exists(sf)) {
            for (String l : Files.readAllLines(sf, StandardCharsets.UTF_8)) if (!l.startsWith("#")) t.append(l).append('\n');
        } else {
            t.append("(файла выбора нет – сначала pool-p4a)\n");
        }
        int[] clamped = new int[2];
        StringBuilder body = new StringBuilder();
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            List<List<XBaseVec.Model>> tt = new ArrayList<>();
            List<List<XBaseVec.Model>> xs = new ArrayList<>();
            for (int k : bi) {
                tt.add(XBaseFeat2.read(vd, u.blocks.get(k), 1, V8));
                xs.add(XBaseFeat2.read(vd, u.blocks.get(k), 0, V8));
            }
            for (int kind = 0; kind < 2; kind++) {
                String[] nm = names(kind);
                Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
                List<String[]> pf = new ArrayList<>();
                String bl = null;
                for (int v = 0; v < nm.length; v++) {
                    List<XBaseThr.Unit> us = new ArrayList<>();
                    for (int k = 0; k < bi.size(); k++) {
                        XBaseVec.Blk b = u.blocks.get(bi.get(k));
                        us.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), fused(b, tt.get(k), xs.get(k), false, kind, v, clamped), null, t2));
                    }
                    String lb = grp + " слияние w = 0,75, " + nm[v];
                    XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                    of.put(lb, o);
                    if (v == 0) bl = lb;
                    else pf.add(new String[] {lb, bl});
                    if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
                }
                XBaseLbp.tables(body, prefix + " – " + (kind == 0 ? "А1, β по кандидату" : "А2, r0 у L1-tt-xs-F"), of, u.persons, u.tpl, pf, t2);
                if (!t2) for (String[] p : pf) {
                    double[] d = XBaseVt.delta(of.get(p[0]), of.get(p[1]), XBaseThr.WAYS[0], u.persons, u.tpl);
                    dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                            : "НЕ выполнено"));
                }
            }
        }
        t.append(hold).append(base.equals("gt") ? dl : "").append(body);
        FarMethods.write(u.out.resolve("xbase_" + base + "_p4a.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }
}
