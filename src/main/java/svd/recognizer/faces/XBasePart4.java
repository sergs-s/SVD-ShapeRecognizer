package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
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

    // ---------------------------------------------------------------- В: чувствительность к R отбора по позе (только пробы)

    /** Пробный режим FACES_P4_SMOKE: 0 – нет; 2 – только первый блок «межс.» (замер). */
    static int smoke() {
        String v = System.getenv("FACES_P4_SMOKE");
        return v == null || v.isBlank() ? 0 : Integer.parseInt(v.trim());
    }

    /** Значения R: база (из fei_selection.tsv) первой, затем 0,30 и 0,45; справочно – без отбора (+∞). */
    static double[] rValues(double r0) {
        return new double[] {r0, 0.30, 0.45, Double.POSITIVE_INFINITY};
    }

    static String rName(double r, double r0) {
        return Double.isInfinite(r) ? "R = +∞ (без отбора, справочно)" : r == r0 ? String.format(Locale.ROOT, "R = %s (база)", fmtR(r)) : "R = " + fmtR(r);
    }

    static String fmtR(double r) {
        return String.format(Locale.ROOT, "%.3f", r).replaceAll("0+$", "").replace('.', ',');
    }

    /** Копия Gt с другими rOf, samples и ctrl (byPerson); прочие поля – как у g (XBaseAbl.copy). */
    static XBaseEval.Gt withProbes(XBaseEval.Gt g, Map<String, Double> rOf, Map<String, Sample> samples, Map<String, Map<Integer, List<String>>> ctrlBy) {
        XBaseEval.Gt c = XBaseAbl.copy(g, g.cohort, rOf, g.r0);
        c.poseR0 = g.poseR0;
        c.samples = samples;
        c.byPerson = new LinkedHashMap<>(g.byPerson);
        c.byPerson.putAll(ctrlBy);
        return c;
    }

    /** Модели блока с основными моделями, у которых строки проб отобраны предикатом keep (части – как есть). */
    static List<XBaseVec.Model> filtered(List<XBaseVec.Model> ms, java.util.function.Predicate<XBaseVec.Row> keep) {
        List<XBaseVec.Model> out = new ArrayList<>();
        for (XBaseVec.Model m : ms) {
            if (!m.task().equals("main")) {
                out.add(m);
                continue;
            }
            out.add(new XBaseVec.Model(m.task(), m.method(), m.s(), m.gallery(), m.rows().stream().filter(keep).toList(), m.z()));
        }
        return out;
    }

    /** Как XBaseThr.evaluate (без сокращённого набора), но «чужие» (Т1) – по своим ключам t1Keys (порог П3 – как в базе). */
    static XBaseThr.Out evalFixed(String label, List<XBaseThr.Unit> units, List<Map<String, List<String>>> t1Keys, int persons, XBaseEval.Gt tpl) {
        XBaseThr.Out o = new XBaseThr.Out();
        List<FeiDecompose.Cfg> cfgs = new ArrayList<>();
        List<List<FeiThreshold.Imp>> imps = List.of(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        long seed = tpl.seed + FeiBootstrap.SEED_OFFSET;
        for (int i = 0; i < units.size(); i++) {
            XBaseThr.Unit u = units.get(i);
            XBaseEval.Parsed p = XBaseEval.parse(u.g(), java.util.EnumSet.of(XBaseEval.V.L1), u.gal(), XBaseEval.probes(u.g(), u.pro()), u.parts(), u.lines());
            cfgs.add(p.cfg.get(XBaseEval.V.L1));
            imps.get(0).add(p.imp.get(XBaseEval.V.L1));
            FeiThreshold.Imp a = XBaseThr.t1(u, XBaseThr.rows(u.lines()), t1Keys.get(i));
            imps.get(1).add(a);
            imps.get(2).add(XBaseCalib.union(p.imp.get(XBaseEval.V.L1), a));
            o.vol.append(String.format(Locale.ROOT, "%s | проход %d | (Т0) %d снимков, %d людей | (Т1) %d снимков, %d людей%n", label, i + 1,
                    p.imp.get(XBaseEval.V.L1).r().length, p.imp.get(XBaseEval.V.L1).persons(), a.r().length, a.persons()));
        }
        for (int w = 0; w < 3; w++) o.res.put(XBaseThr.WAYS[w], FeiThreshold.evaluate(label + " " + XBaseThr.WAYS[w], XBaseEval.V.L1.ratioId(), cfgs, imps.get(w), persons,
                tpl.b, seed));
        for (double r : XBaseThr.rs(imps.get(0))) o.imp0.add(new double[] {r});
        for (double r : XBaseThr.rs(imps.get(1))) o.imp1.add(new double[] {r});
        return o;
    }

    /** Наименьший score YuNet по строкам детекций (последнее поле) и число строк без лица. */
    static double[] minScore(List<String> rows) {
        double min = Double.POSITIVE_INFINITY;
        int miss = 0;
        for (String l : rows) {
            String[] c = l.split("\t", -1);
            String s = c[c.length - 1];
            if (s.isEmpty()) miss++;
            else min = Math.min(min, Double.parseDouble(s));
        }
        return new double[] {min, miss, rows.size()};
    }

    /** feret-p4c, gt-p4c: роль «межс.»; R ∈ {база, 0,30, 0,45} и справочно без отбора – только к пробам своих и контрольных. */
    static void ownC(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        XBaseVec.Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, 0);
        boolean feret = base.equals("feret");
        double rBase = XBaseEval.poseLimit(selLines);
        double[] rs = rValues(rBase);
        // Все найденные снимки: ключ → |r|, дата, человек; GT – ещё образцы экспорта.
        Map<String, Double> rAll = new HashMap<>();
        Map<String, String> dateOf = new HashMap<>();
        Map<String, List<String>> byPerson = new java.util.TreeMap<>();
        Map<String, Sample> samples = new HashMap<>();
        StringBuilder yu = new StringBuilder("\n=== Порог YuNet 0,6 / 0,7 / 0,8 ===\n");
        if (feret) {
            for (FeretData.Img im : u.su.imgs) {
                Double r = u.su.rOf.get(im.key());
                if (r == null) continue;
                rAll.put(im.key(), r);
                dateOf.put(im.key(), im.date());
                byPerson.computeIfAbsent(im.person(), p -> new ArrayList<>()).add(im.key());
            }
            double[] ms = minScore(Files.readAllLines(u.su.dir.resolve(FeretData.DERIVED).resolve("detections.tsv"), StandardCharsets.UTF_8));
            yu.append(String.format(Locale.ROOT, "FERET fa/fb: строк детекций %d, без лица %d, наименьший score найденного лица %.4f.%n", (int) ms[2], (int) ms[1], ms[0]));
        } else {
            Map<String, Double> pose = XBaseEval.poses(dir.resolve("detections.tsv"));
            Map<String, Integer> sess = XBaseEval.sessions(dir.resolve(XBaseEval.SESSIONS));
            for (Map.Entry<String, List<Sample>> e : new java.util.TreeMap<>(export.data().gt()).entrySet()) {
                for (Sample s : e.getValue()) {
                    String name = s.file().getFileName().toString();
                    String k = s.file().toString();
                    rAll.put(k, Math.abs(pose.get("gt/" + s.person() + "/" + name)));
                    dateOf.put(k, String.valueOf(sess.get(s.person() + "/" + name)));
                    byPerson.computeIfAbsent(s.person(), p -> new ArrayList<>()).add(k);
                    samples.put(k, s);
                }
            }
            List<String> rows = Files.readAllLines(dir.resolve("detections.tsv"), StandardCharsets.UTF_8).stream().filter(l -> l.startsWith("gt\t")).toList();
            double[] ms = minScore(rows);
            yu.append(String.format(Locale.ROOT, "GT: строк детекций %d, без лица %d, наименьший score найденного лица %.4f.%n", (int) ms[2], (int) ms[1], ms[0]));
        }
        yu.append("Из найденных лиц берётся лицо с наибольшим score (YunetTwoPass.best), у всех снимков он не ниже 0,8 – при пороге 0,6, 0,7 и 0,8 "
                + "отказов детектора нет и выбранное лицо то же;\nпорог YuNet в этом диапазоне на FERET fa/fb и GT ни на что не влияет (прогон не нужен).\n");
        List<XBaseVec.Blk> blocks = new ArrayList<>();
        for (XBaseVec.Blk b : u.blocks) if (b.group().equals("межс.") && (sm == 0 || blocks.isEmpty())) blocks.add(b);
        // Кандидаты блока: свои – все снимки с датой не из дат галереи, контрольные – все снимки; база – с |r| ≤ R базы (сверка).
        Map<String, Double> rOfExt = new HashMap<>(blocks.get(0).g().rOf);
        Map<String, Sample> samplesExt = blocks.get(0).g().samples == null ? null : new HashMap<>(blocks.get(0).g().samples);
        List<Map<String, List<String>>> proAll = new ArrayList<>();
        List<Map<String, List<String>>> ctrlAll = new ArrayList<>();
        Set<String> newKeys = new java.util.TreeSet<>();
        StringBuilder check = new StringBuilder();
        for (XBaseVec.Blk b : blocks) {
            Map<String, List<String>> pa = new LinkedHashMap<>();
            for (String p : b.g().own) {
                Set<String> gd = new HashSet<>();
                for (String k : b.gal().get(p)) gd.add(dateOf.get(k));
                List<String> c = byPerson.get(p).stream().filter(k -> !gd.contains(dateOf.get(k))).sorted().toList();
                List<String> sel = c.stream().filter(k -> rAll.get(k) <= rBase).toList();
                if (!new HashSet<>(sel).equals(new HashSet<>(b.pro().get(p)))) throw new IllegalStateException("p4c: пробы своего " + p + " (" + b.name()
                        + ") ≠ кандидаты с |r| ≤ R базы");
                pa.put(p, c);
            }
            Map<String, List<String>> ca = new LinkedHashMap<>();
            for (String p : b.g().ctrl) {
                List<String> c = byPerson.get(p).stream().sorted().toList();
                List<String> baseKeys = new ArrayList<>();
                b.g().byPerson.get(p).values().forEach(baseKeys::addAll);
                if (!new HashSet<>(c.stream().filter(k -> rAll.get(k) <= rBase).toList()).equals(new HashSet<>(baseKeys))) {
                    throw new IllegalStateException("p4c: снимки контрольного " + p + " (" + b.name() + ") ≠ кандидаты с |r| ≤ R базы");
                }
                ca.put(p, c);
            }
            proAll.add(pa);
            ctrlAll.add(ca);
            for (List<String> l : pa.values()) for (String k : l) if (rAll.get(k) > rBase) newKeys.add(k);
            for (List<String> l : ca.values()) for (String k : l) if (rAll.get(k) > rBase) newKeys.add(k);
            check.append(String.format(Locale.ROOT, "%s: пробы своих и контрольных при |r| ≤ R базы = пробам базы – да%n", b.name()));
        }
        for (String k : newKeys) {
            rOfExt.put(k, rAll.get(k));
            if (samplesExt != null) samplesExt.put(k, samples.get(k));
        }
        // Кадры новых проб; признаки v8 обоих кандидатов по всем кадрам.
        if (feret) {
            for (String k : newKeys) u.fr.put(k, FeretData.frame(u.su.dir, "a", k));
        } else {
            u.fr.putAll(XBaseEval.frames(export, newKeys.stream().map(samples::get).toList(), null));
        }
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        Path vd = XBaseFeat2.dir(u, 0);
        int[] clamped = new int[2];
        StringBuilder t = new StringBuilder("Пачка 4, В: чувствительность к R отбора по позе – " + base + " межс. (XBaseEval " + base + "-p4c)" + (sm == 0 ? ""
                : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\nКод: коммит " + FarMethods.commit() + "; векторы базы – " + vd + "\n");
        t.append("R – только к пробам (свои и контрольные; решение Жорика №2 от 01.11): галерея, пул, когорта, W, модели частей и порог П3 по (Т2) – как в базе "
                + "((Т1) – по пробам базы).\nНовые пробы (|r| > R базы) – строки основных моделей (X и X+mirror обоих кандидатов) заново, на тех же W; строки "
                + "проб базы сверяются с векторами v8 побайтно.\nКонфигурация (32): (П2) + MS-LBP, слияние w = 0,75, β = 1. Отказ – проба с |r| > R; «отказ или "
                + "ошибка» – от всех проб, найденных детектором.\n");
        t.append(yu);
        StringBuilder cnt = new StringBuilder("\n=== Пробы по R (все найденные детектором) ===\nR | свои: принято / всего (отказ по позе) | контрольные: принято / всего "
                + "снимков (отказ) | контрольных людей с ≥ 1 принятым снимком\n");
        StringBuilder rej = new StringBuilder("\n=== «Отказ или ошибка» своих при П3 (Т2) FAR 1 % (от всех проб) ===\nR | отказ по позе | ошибка среди принятых | "
                + "отказ или ошибка\n");
        double[][] xs = new double[2][];
        List<List<List<XBaseVec.Model>>> ext = new ArrayList<>();
        for (int bi = 0; bi < blocks.size(); bi++) ext.add(new ArrayList<>(List.of(List.of(), List.of())));
        for (int f = 1; f >= 0; f--) {
            long tf = System.nanoTime();
            XBaseEval x = XBaseFeat2.vectors(u.fr, fs, f, V8);
            System.out.printf(Locale.ROOT, "%s-p4c: f%d признаки (%.0f с)%n", base, f, (System.nanoTime() - tf) / 1e9);
            for (int bi = 0; bi < blocks.size(); bi++) {
                XBaseVec.Blk b = blocks.get(bi);
                Map<String, List<String>> pa = proAll.get(bi);
                Map<String, Map<Integer, List<String>>> cb = new LinkedHashMap<>();
                for (Map.Entry<String, List<String>> e : ctrlAll.get(bi).entrySet()) cb.put(e.getKey(), new java.util.TreeMap<>(Map.of(0, e.getValue())));
                XBaseEval.Gt gx = withProbes(b.g(), rOfExt, samplesExt, cb);
                List<Probe> probes = XBaseEval.probes(gx, pa);
                List<List<String>> fullGallery = b.g().own.stream().map(b.gal()::get).toList();
                long tm = System.nanoTime();
                List<Callable<XBaseVec.Model>> jobs = new ArrayList<>();
                for (boolean m : new boolean[] {false, true}) {
                    String method = XBaseEval.V.L1.id + (m ? FeiMethods.MIRROR : "");
                    jobs.add(() -> XBaseVec.model(x, m, method, b.s(), b.g().own, fullGallery, b.g().cohort, u.cin, u.cout, probes, "main"));
                }
                List<XBaseVec.Model> mains = XBaseVec.run(List.of(jobs), u.tpl.threads).get(0);
                System.out.printf(Locale.ROOT, "%s-p4c: %s f%d основные модели (%.0f с)%n", base, b.name(), f, (System.nanoTime() - tm) / 1e9);
                List<XBaseVec.Model> file = XBaseFeat2.read(vd, b, f, V8);
                List<XBaseVec.Model> merged = new ArrayList<>();
                for (XBaseVec.Model m : file) {
                    if (!m.task().equals("main")) {
                        merged.add(m);
                        continue;
                    }
                    XBaseVec.Model nm = mains.get(m.method().endsWith(FeiMethods.MIRROR) ? 1 : 0);
                    Map<String, XBaseVec.Row> byKey = new HashMap<>();
                    for (XBaseVec.Row r : nm.rows()) byKey.put(r.role() + "\t" + r.key(), r);
                    for (XBaseVec.Row r : m.rows()) {
                        XBaseVec.Row q = byKey.get(r.role() + "\t" + r.key());
                        if (q == null || !java.util.Arrays.equals(q.dg(), r.dg()) || !java.util.Arrays.equals(q.dci(), r.dci()) || !java.util.Arrays.equals(q.dco(), r.dco())
                                || r.trueIdx() != q.trueIdx() || r.image() != q.image()) {
                            throw new IllegalStateException("p4c: строка пробы базы " + r.key() + " (" + b.name() + ", f" + f + ", " + m.method() + ") ≠ векторам v8");
                        }
                    }
                    merged.add(nm);
                }
                ext.get(bi).set(f, merged);
                check.append(String.format(Locale.ROOT, "%s f%d: строки %d проб базы основных моделей = векторам v8 (%s) побайтно; новых проб %d%n", b.name(), f,
                        file.stream().filter(m -> m.task().equals("main")).findFirst().orElseThrow().rows().size(), vd.getFileName(), probes.size()
                        - file.stream().filter(m -> m.task().equals("main")).findFirst().orElseThrow().rows().size()));
            }
        }
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        // Варианты R: строки слияния по отобранным пробам; при R базы – побайтно XBaseVec.fuseLines по векторам v8.
        Map<String, XBaseThr.Out> outs = new LinkedHashMap<>();
        List<String[]> prs = new ArrayList<>();
        String bl = null;
        for (double r : rs) {
            List<XBaseThr.Unit> us = new ArrayList<>();
            List<Map<String, List<String>>> t1 = new ArrayList<>();
            int[] c = new int[5];
            Set<String> ctrlP = new HashSet<>();
            for (int bi = 0; bi < blocks.size(); bi++) {
                XBaseVec.Blk b = blocks.get(bi);
                java.util.function.Predicate<XBaseVec.Row> keep = row -> rOfExt.get(row.key()) <= r;
                List<XBaseVec.Model> tt = filtered(ext.get(bi).get(1), keep);
                List<XBaseVec.Model> xx = filtered(ext.get(bi).get(0), keep);
                String lines = XBaseVec.fuseLines(tt, xx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, false), XBaseVec.fuseStats(xx, false), rOfExt, b.g().r0, clamped);
                if (r == rBase) {
                    List<XBaseVec.Model> ft = XBaseFeat2.read(vd, b, 1, V8);
                    List<XBaseVec.Model> fx = XBaseFeat2.read(vd, b, 0, V8);
                    String ref = XBaseVec.fuseLines(ft, fx, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(ft, false), XBaseVec.fuseStats(fx, false), b.g().rOf, b.g().r0, null);
                    if (!ref.equals(lines)) throw new IllegalStateException("p4c: строки слияния при R базы ≠ векторам v8 (" + b.name() + ")");
                    check.append(b.name()).append(": строки слияния при R базы = слиянию по векторам v8 побайтно\n");
                }
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (Map.Entry<String, List<String>> e : proAll.get(bi).entrySet()) {
                    pro.put(e.getKey(), e.getValue().stream().filter(k -> rOfExt.get(k) <= r).toList());
                    c[0] += pro.get(e.getKey()).size();
                    c[1] += e.getValue().size();
                }
                Map<String, Map<Integer, List<String>>> cb = new LinkedHashMap<>();
                for (Map.Entry<String, List<String>> e : ctrlAll.get(bi).entrySet()) {
                    List<String> l = e.getValue().stream().filter(k -> rOfExt.get(k) <= r).toList();
                    cb.put(e.getKey(), new java.util.TreeMap<>(Map.of(0, l)));
                    c[2] += l.size();
                    c[3] += e.getValue().size();
                    if (!l.isEmpty()) ctrlP.add(e.getKey());
                }
                c[4] += ctrlAll.get(bi).size();
                us.add(new XBaseThr.Unit(withProbes(b.g(), rOfExt, samplesExt, cb), b.gal(), pro, b.parts(), lines, null, true));
                t1.add(b.pro());
            }
            String lb = "межс. слияние w = 0,75, " + rName(r, rBase);
            XBaseThr.Out o = evalFixed(lb, us, t1, u.persons, u.tpl);
            outs.put(lb, o);
            if (r == rBase) bl = lb;
            else prs.add(new String[] {lb, bl});
            cnt.append(String.format(Locale.ROOT, "%s | %d / %d (%.1f %%) | %d / %d (%.1f %%) | %d из %d%n", rName(r, rBase), c[0], c[1], 100.0 * (c[1] - c[0]) / c[1], c[2],
                    c[3], 100.0 * (c[3] - c[2]) / c[3], ctrlP.size(), c[4]));
            FeiThreshold.Res res = o.res.get(XBaseThr.WAYS[2]);
            double[] k3 = res.counts[3][0][0];
            double n = java.util.Arrays.stream(k3).sum();
            double err = n - k3[FeiDecompose.G_OK];
            rej.append(String.format(Locale.ROOT, "%s | %d / %d = %.1f %% | %.0f / %.0f = %.1f %% | %.0f / %d = %.1f %%%n", rName(r, rBase), c[1] - c[0], c[1], 100.0 * (c[1]
                    - c[0]) / c[1], err, n, 100 * err / n, c[1] - c[0] + err, c[1], 100.0 * (c[1] - c[0] + err) / c[1]));
        }
        t.append("\n=== Сверка ===\n").append(check).append(cnt).append(rej);
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        for (Map.Entry<String, XBaseThr.Out> e : outs.entrySet()) hold.append(XBaseAbl3.holds(e.getKey(), e.getValue(), XBaseThr.WAYS[2]));
        t.append(hold);
        XBaseLbp.tables(t, (feret ? "FERET" : "GT") + " межс. – R отбора по позе (только пробы)", outs, u.persons, u.tpl, prs, true);
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с (потоков %d).%n", (System.nanoTime() - t0) / 1e9, u.tpl.threads));
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        FarMethods.write(u.out.resolve("xbase_" + base + "_p4c" + sfx + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- Г: порог при малой галерее – таблица N → t (Н) против П3

    static final int[] NS = {10, 50, 100, 200};

    /** Узлы N проверки p4d: FACES_P4_NS (через запятую, малая галерея 07.11) или NS. */
    static int[] dNs() {
        String v = System.getenv("FACES_P4_NS");
        return v == null || v.isBlank() ? NS : java.util.Arrays.stream(v.split(",")).mapToInt(s -> Integer.parseInt(s.trim())).toArray();
    }

    /** Узел таблицы для справочных строк на тех же галереях (FACES_P4_TREF, например 10) или null. */
    static Integer dRef() {
        String v = System.getenv("FACES_P4_TREF");
        return v == null || v.isBlank() ? null : Integer.parseInt(v.trim());
    }

    static final int GALLERIES = 20;
    /** Сдвиг seed галерей Г: faces.seed + 10800 + 100 · строка + N (строка: FERET проход 1 – 0, проход 2 – 1, GT – 2). */
    static final long D_SEED_OFFSET = 10800;

    /** Строка таблицы (Н): t и постоянные приведения μ_tt, σ_tt, μ_xs, σ_xs. */
    record TRow(double t, String by, double[] st, double[] sx) {}

    static Map<Integer, TRow> readTable(Path f) throws IOException {
        Map<Integer, TRow> out = new java.util.TreeMap<>();
        for (String l : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("N\t") || l.isBlank()) continue;
            String[] c = l.split("\t");
            out.put(Integer.parseInt(c[0]), new TRow(Double.parseDouble(c[1]), c[2], new double[] {Double.parseDouble(c[3]), Double.parseDouble(c[4])},
                    new double[] {Double.parseDouble(c[5]), Double.parseDouble(c[6])}));
        }
        return out;
    }

    /** Модель как XBaseVec.model без когорт приведения, W – MldaFast (DuelThr.nModelFast). */
    static List<Callable<XBaseVec.Model>> jobsD(XBaseEval x, int f, XBaseEval.Gt g, Map<String, List<String>> gal, List<Probe> main, List<List<String>> parts,
                                                List<List<Probe>> extra) {
        List<Callable<XBaseVec.Model>> out = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            if (f == 0 && !m) continue;
            String method = XBaseEval.V.L1.id + (m ? FeiMethods.MIRROR : "");
            List<List<String>> fullGallery = g.own.stream().map(gal::get).toList();
            out.add(() -> DuelThr.nModelFast(x, m, method, g.own, fullGallery, g.cohort, main, "main"));
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> own = g.own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = own.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : g.own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, g.img(k), k, -1));
                pr.addAll(extra.get(q));
                String task = "part " + q;
                out.add(() -> DuelThr.nModelFast(x, m, method, own, gk, g.cohort, pr, task));
            }
        }
        return out;
    }

    /** Галерея Г: Gt, галерея, пробы своих (только люди галереи), части, пробы (Т1). */
    record DG(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<String>> pro, List<List<String>> parts, List<Probe> main, List<List<Probe>> extra) {}

    static DG gallery(XBaseVec.Blk b, List<String> own, long seed, int k, boolean feret) {
        XBaseEval.Gt g = feret ? XBaseFeret.withOwn(b.g(), own) : XBaseDates.withOwn(b.g(), own);
        g.poseR0 = b.g().poseR0;
        Map<String, List<String>> gal = new LinkedHashMap<>();
        Map<String, List<String>> pro = new LinkedHashMap<>();
        for (String p : own) {
            gal.put(p, b.gal().get(p));
            pro.put(p, b.pro().get(p));
        }
        List<List<String>> parts = XBaseOther.cvParts(own, seed, k);
        return new DG(g, gal, pro, parts, XBaseEval.probes(g, pro), XBaseThr.extra(g, pro, parts));
    }

    /** Оценки s = 2ℓ₁ − ℓ₂ основной модели при приведении st / sx и порог t: {свои ошибки, свои всего, принятые контрольные, контрольные всего}. */
    static long[] countT(List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, double[] st, double[] sx, Map<String, Double> rOf, double r0, double t) {
        XBaseVec.Model tX = XBaseVec.find(tt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(tt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(xs, "main", true);
        long[] c = new long[4];
        for (int i = 0; i < tX.rows().size(); i++) {
            XBaseVec.Row rx = xM.rows().get(i);
            XBaseVec.Row rt = rOf.get(tX.rows().get(i).key()) <= r0 ? tX.rows().get(i) : tM.rows().get(i);
            if (!rx.key().equals(rt.key())) throw new IllegalStateException("p4d: пробы кандидатов разошлись");
            int ng = rt.dg().length;
            int b = -1;
            int s2 = -1;
            double[] ln = new double[ng];
            for (int g = 0; g < ng; g++) {
                ln[g] = XBaseFeat2.W_FUSE * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                if (b < 0 || ln[g] < ln[b]) {
                    s2 = b;
                    b = g;
                } else if (s2 < 0 || ln[g] < ln[s2]) {
                    s2 = g;
                }
            }
            double s = 2 * ln[b] - ln[s2];
            if (rt.trueIdx() >= 0) {
                c[1]++;
                if (b != rt.trueIdx() || s > t) c[0]++;
            } else if (rt.role().equals(FeiScores.CTRL)) {
                c[3]++;
                if (s <= t) c[2]++;
            }
        }
        return c;
    }

    static double quantile(List<Double> v, double q) {
        double[] a = v.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return FeiBootstrap.quantile(a, q);
    }

    /** feret-p4d, gt-p4d. */
    static void ownD(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        boolean feret = base.equals("feret");
        XBaseVec.Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, 0);
        String tf = System.getenv("FACES_P4_TABLE");
        if (tf == null) throw new IllegalStateException("p4d: задайте FACES_P4_TABLE – таблица (Н) duel_thr_N.tsv");
        Map<Integer, TRow> table = readTable(Paths.get(tf));
        int[] ns = dNs();
        Integer ref = dRef();
        TRow trRef = ref == null ? null : table.get(ref);
        if (ref != null && trRef == null) throw new IllegalStateException("p4d: в таблице нет узла " + ref);
        List<XBaseVec.Blk> blocks = new ArrayList<>();
        for (XBaseVec.Blk b : u.blocks) if (b.group().equals("межс.") && (sm == 0 || blocks.isEmpty())) blocks.add(b);
        int nGal = sm == 0 ? GALLERIES : 2;
        // Галереи: [блок][N] → список галерей (люди по возрастанию).
        List<Map<Integer, List<DG>>> gals = new ArrayList<>();
        StringBuilder t = new StringBuilder("Пачка 4, Г: порог при малой галерее – таблица N → t (Н) против П3 – " + base + " межс. (XBaseEval " + base + "-p4d)"
                + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\nКод: коммит " + FarMethods.commit() + "; таблица (Н) – " + tf + " (SHA-256 " + DuelSet.sha256(Paths.get(tf))
                + ")\n");
        t.append("Галереи: " + nGal + " на N, люди – случайная выборка своих блока (Collections.shuffle, Random(faces.seed + 10800 + 100 · строка + N), строка: "
                + "FERET проход 1 – 0, проход 2 – 1, GT – 2);\nW – заново на галерее + когорта FEI 20 + пул FERET (MldaFast), 5 частей галереи (XBaseOther.cvParts("
                + "галерея, faces.seed + 10800 + 100 · строка + N, номер галереи)) с переобучением W – для П3 по (Т2) самой галереи.\nПробы – межсессионные "
                + "пробы людей галереи и контрольные блока (как в базе). Конфигурация (32): (П2) + MS-LBP, слияние w = 0,75, β = 1.\nВарианты: (а) П3 по (Т2) "
                + "галереи, FAR 1 %; (б) t(N) таблицы с постоянными приведениями таблицы; (в) t(N) с приведениями по (Т2) галереи. Приём – s ≤ t.\n");
        for (int bi = 0; bi < blocks.size(); bi++) {
            XBaseVec.Blk b = blocks.get(bi);
            int row = feret ? bi : 2;
            Map<Integer, List<DG>> m = new java.util.TreeMap<>();
            for (int n : ns) {
                if (n > b.g().own.size() || (sm != 0 && n != ns[0])) continue;
                long seed = u.tpl.seed + D_SEED_OFFSET + 100L * row + n;
                java.util.Random rnd = new java.util.Random(seed);
                List<DG> l = new ArrayList<>();
                for (int k = 0; k < nGal; k++) {
                    List<String> perm = new ArrayList<>(b.g().own);
                    java.util.Collections.shuffle(perm, rnd);
                    List<String> own = new ArrayList<>(perm.subList(0, n));
                    java.util.Collections.sort(own);
                    l.add(gallery(b, own, seed, k, feret));
                }
                m.put(n, l);
            }
            gals.add(m);
            t.append(String.format(Locale.ROOT, "%s: своих %d; N = %s%s%n", b.name(), b.g().own.size(), m.keySet(), m.size() < ns.length && sm == 0
                    ? "; N больше числа своих – строки пропущены" : ""));
        }
        // Модели: по признаку, все галереи; контрольная точка – файл на (блок, N, признак).
        String nsTag = ns == NS ? "" : "_n" + java.util.Arrays.stream(ns).mapToObj(Integer::toString).collect(java.util.stream.Collectors.joining("_"));
        Path cd = u.out.resolve("xbase_" + base + "_p4d" + nsTag + (sm == 0 ? "" : "_smoke" + sm));
        Files.createDirectories(cd);
        String commit = FarMethods.commit();
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        for (int f = 1; f >= 0; f--) {
            XBaseEval x = null;
            for (int bi = 0; bi < blocks.size(); bi++) {
                for (Map.Entry<Integer, List<DG>> e : gals.get(bi).entrySet()) {
                    Path p = cd.resolve(blocks.get(bi).name() + "_n" + e.getKey() + "_f" + f + ".tsv.gz");
                    if (Files.exists(p)) {
                        if (!XBaseVec.readHead(p).contains(commit) && XBaseFinal.acceptedCommit(XBaseVec.readHead(p)) == null) throw new IllegalStateException("p4d: " + p + " – другой коммит кода; удалите каталог " + cd);
                        continue;
                    }
                    if (x == null) {
                        long tv = System.nanoTime();
                        x = XBaseFeat2.vectors(u.fr, fs, f, V8);
                        System.out.printf(Locale.ROOT, "%s-p4d: f%d признаки (%.0f с)%n", base, f, (System.nanoTime() - tv) / 1e9);
                    }
                    long tm = System.nanoTime();
                    List<List<Callable<XBaseVec.Model>>> jobs = new ArrayList<>();
                    for (DG d : e.getValue()) jobs.add(jobsD(x, f, d.g(), d.gal(), d.main(), d.parts(), d.extra()));
                    List<List<XBaseVec.Model>> done = XBaseVec.run(jobs, u.tpl.threads);
                    java.io.Writer w = XBaseVec.writer(p, "# XBasePart4 " + base + "-p4d: " + blocks.get(bi).name() + ", N = " + e.getKey() + ", " + XBaseRule.FEATS[f]
                            + ", галерей " + done.size() + "\n# Код: коммит " + commit + "\n");
                    for (int k = 0; k < done.size(); k++) XBaseVec.write(w, "g" + k, done.get(k));
                    XBaseVec.finish(w, p);
                    System.out.printf(Locale.ROOT, "%s-p4d: %s N = %d f%d – %d галерей (%.0f с)%n", base, blocks.get(bi).name(), e.getKey(), f, done.size(),
                            (System.nanoTime() - tm) / 1e9);
                }
            }
        }
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        XBaseEval.Gt tpl0 = XBaseAbl.copy(u.tpl, u.tpl.cohort, u.tpl.rOf, u.tpl.r0);
        tpl0.b = 0;
        StringBuilder res = new StringBuilder("\n=== По N и варианту: FAR контроля по сумме попыток (КП ↑95), 97,5 % по галереям, ошибка своих по сумме; держит FAR 1 % – КП ↑95 "
                + "по сумме ≤ 1 % ===\nN | вариант | галерей | FAR контроля | 97,5 % по галереям | ошибка своих | держит FAR 1 % | t / приведения\n");
        StringBuilder st2 = new StringBuilder("\n=== Приведения по (Т2) галереи (медиана [2,5 %; 97,5 %] по галереям): μ_tt, σ_tt, μ_xs, σ_xs ===\n");
        for (int n : ns) {
            List<double[]> va = new ArrayList<>();
            List<double[]> vb = new ArrayList<>();
            List<double[]> vc = new ArrayList<>();
            List<double[]> vd = new ArrayList<>();
            List<double[]> ve = new ArrayList<>();
            List<double[]> stats = new ArrayList<>();
            TRow tr = table.get(n);
            for (int bi = 0; bi < blocks.size(); bi++) {
                List<DG> l = gals.get(bi).get(n);
                if (l == null) continue;
                Map<String, List<XBaseVec.Model>> mt = XBaseVec.read(cd.resolve(blocks.get(bi).name() + "_n" + n + "_f1.tsv.gz"));
                Map<String, List<XBaseVec.Model>> mx = XBaseVec.read(cd.resolve(blocks.get(bi).name() + "_n" + n + "_f0.tsv.gz"));
                for (int k = 0; k < l.size(); k++) {
                    DG d = l.get(k);
                    List<XBaseVec.Model> tt = mt.get("g" + k);
                    List<XBaseVec.Model> xs = mx.get("g" + k);
                    double[] stt = XBaseVec.fuseStats(tt, false);
                    double[] sxs = XBaseVec.fuseStats(xs, false);
                    stats.add(new double[] {stt[0], stt[1], sxs[0], sxs[1]});
                    String lines = XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, stt, sxs, d.g().rOf, d.g().r0, null);
                    XBaseThr.Out o = XBaseThr.evaluate("N = " + n + " g" + k, List.of(new XBaseThr.Unit(d.g(), d.gal(), d.pro(), d.parts(), lines, null, true)), u.persons,
                            tpl0);
                    FeiThreshold.Res r = o.res.get(XBaseThr.WAYS[2]);
                    double[] cnt = r.counts[3][0][0];
                    double own = java.util.Arrays.stream(cnt).sum();
                    va.add(new double[] {own - cnt[FeiDecompose.G_OK], own, r.far[3][0][0], r.far[3][0][1]});
                    if (tr != null) {
                        long[] cb = countT(tt, xs, tr.st(), tr.sx(), d.g().rOf, d.g().r0, tr.t());
                        long[] cc = countT(tt, xs, stt, sxs, d.g().rOf, d.g().r0, tr.t());
                        vb.add(new double[] {cb[0], cb[1], cb[2], cb[3]});
                        vc.add(new double[] {cc[0], cc[1], cc[2], cc[3]});
                    }
                    if (trRef != null) {
                        long[] cd2 = countT(tt, xs, trRef.st(), trRef.sx(), d.g().rOf, d.g().r0, trRef.t());
                        long[] ce = countT(tt, xs, stt, sxs, d.g().rOf, d.g().r0, trRef.t());
                        vd.add(new double[] {cd2[0], cd2[1], cd2[2], cd2[3]});
                        ve.add(new double[] {ce[0], ce[1], ce[2], ce[3]});
                    }
                }
            }
            if (va.isEmpty()) {
                res.append(String.format(Locale.ROOT, "%d | – | 0 | N больше числа своих – строка пропущена%n", n));
                continue;
            }
            String[] vn = {"(а) П3 по (Т2) галереи", "(б) t(N) таблицы, приведения таблицы", "(в) t(N) таблицы, приведения по (Т2) галереи",
                    "(г) t(" + ref + ") таблицы, приведения узла " + ref, "(д) t(" + ref + ") таблицы, приведения по (Т2) галереи"};
            List<List<double[]>> all = List.of(va, vb, vc, vd, ve);
            for (int v = 0; v < 5; v++) {
                List<double[]> l = all.get(v);
                if (l.isEmpty()) continue;
                TRow trv = v >= 3 ? trRef : tr;
                long e0 = 0, o0 = 0, a0 = 0, c0 = 0;
                List<Double> fr = new ArrayList<>();
                for (double[] a : l) {
                    e0 += (long) a[0];
                    o0 += (long) a[1];
                    a0 += (long) a[2];
                    c0 += (long) a[3];
                    fr.add(100 * a[2] / a[3]);
                }
                double cp = 100 * FaceEvaluation.binomialUpperBound((int) a0, (int) c0, 0.95);
                String extra = v == 0 ? "П3 FAR 1 %" : String.format(Locale.ROOT, "t = %.2f (%s)%s", trv.t(), trv.by(), v == 1 || v == 3 ? String.format(Locale.ROOT, "; μ_tt %.4f σ_tt %.4f "
                        + "μ_xs %.4f σ_xs %.4f", trv.st()[0], trv.st()[1], trv.sx()[0], trv.sx()[1]) : "");
                res.append(String.format(Locale.ROOT, "%d | %s | %d | %d/%d = %.2f %%, КП ↑%.2f %% | %.2f %% | %d/%d = %.1f %% | %s | %s%n", n, vn[v], l.size(), a0, c0,
                        100.0 * a0 / c0, cp, quantile(fr, 0.975), e0, o0, 100.0 * e0 / o0, cp <= 1 ? "да" : "нет", extra));
            }
            double[][] q = new double[4][];
            for (int j = 0; j < 4; j++) {
                List<Double> s = new ArrayList<>();
                for (double[] a : stats) s.add(a[j]);
                q[j] = new double[] {quantile(s, 0.5), quantile(s, 0.025), quantile(s, 0.975)};
            }
            st2.append(String.format(Locale.ROOT, "N = %d: %.4f [%.4f; %.4f]; %.4f [%.4f; %.4f]; %.4f [%.4f; %.4f]; %.4f [%.4f; %.4f]%n", n, q[0][0], q[0][1], q[0][2], q[1][0],
                    q[1][1], q[1][2], q[2][0], q[2][1], q[2][2], q[3][0], q[3][1], q[3][2]));
        }
        t.append(res).append(st2);
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с (потоков %d).%n", (System.nanoTime() - t0) / 1e9, u.tpl.threads));
        FarMethods.write(u.out.resolve("xbase_" + base + "_p4d" + nsTag + (sm == 0 ? "" : "_smoke" + sm) + XBaseFinalReport.tag() + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- Б: параметры признака (пересчёт векторов, FeatP4)

    /** Вариант Б: номер, группа (параметр), название, источник, параметры, признаки, которые меняются (1 – L1-tt-xs-F, 0 – L1-xs-F). */
    record BV(int no, String group, String name, String src, FeatP4.P p, int[] feats) {}

    static final String SRC_TT = "Tan, Triggs, IEEE TIP 19(6), 2010, разд. II, табл. I (рукопись HAL hal-00565029): γ ∈ [0; 0,5], σ₁ до 4; σ₀ = 0,5 – дисс. Vu, 2010, "
            + "табл. 3.1 (DoG σ 0,5 и 3 на Yale B)";
    static final String SRC_RET = "дисс. Vu, 2010, гл. 3, разд. 3.3, ур. 3.3.1–3.3.10, параметры разд. 3.4.1.3 (σ₁ = σ₂ = 1, σ_Ph = 0,5, σ_H = 3, Th = 3,5); кадры "
            + "автора Yale B 192×168, у нас 92×112 и k = 2 184×224 (σ × 2), σ не масштабируются";
    static final String SRC_POEM = "Vu, Caplier, ECCV 2010, разд. 5.1, рис. 4 (ориентации без знака 2–7) и рис. 5 (ячейка 3–8, блок 5–11); дисс. Vu, 2010, разд. "
            + "4.4.1, рис. 4.4–4.6 (выбор: 3 без знака, w = 7, L = 10)";
    static final String SRC_MS = "Chen, Cao, Wen, Sun, CVPR 2013, разд. 5.1 (окно 40×40, 4×4 ячейки, 5 масштабов), рис. 5 (1–5 масштабов, дальше выигрыш мал); "
            + "размер окна и вес авторы не варьировали – значения постановки (Жорик №1 от 01.11)";

    static List<BV> variantsB(IlluminationNorm.NormParams tt) {
        FeatP4.P b = FeatP4.base(tt);
        List<BV> l = new ArrayList<>();
        int[] t1 = {1};
        int[] x0 = {0};
        int[] both = {1, 0};
        java.util.function.BiFunction<Double, double[], IlluminationNorm.NormParams> ttp = (g, s) -> new IlluminationNorm.NormParams(tt.claheClip(), tt.claheTile(), g,
                s[0], s[1], tt.alpha(), tt.tau());
        l.add(new BV(1, "Б1", "Тан – Триггс γ = 0,1", SRC_TT, withTt(b, ttp.apply(0.1, new double[] {tt.sigma0(), tt.sigma1()})), t1));
        l.add(new BV(2, "Б1", "Тан – Триггс γ = 0,3", SRC_TT, withTt(b, ttp.apply(0.3, new double[] {tt.sigma0(), tt.sigma1()})), t1));
        l.add(new BV(3, "Б1", "Тан – Триггс σ₁ = 3", SRC_TT, withTt(b, ttp.apply(tt.gamma(), new double[] {tt.sigma0(), 3})), t1));
        l.add(new BV(4, "Б1", "Тан – Триггс σ₁ = 4", SRC_TT, withTt(b, ttp.apply(tt.gamma(), new double[] {tt.sigma0(), 4})), t1));
        l.add(new BV(5, "Б1", "Тан – Триггс σ₀ = 0,5 (σ₁ = 2)", SRC_TT, withTt(b, ttp.apply(tt.gamma(), new double[] {0.5, tt.sigma1()})), t1));
        l.add(new BV(6, "Б1′-tt", "L1-tt-xs-F: ретинальный фильтр вместо Тан – Триггса", SRC_RET, new FeatP4.P(FeatP4.In.RETINA, b.inXs(), tt, b.orient(), b.cell(),
                b.radius(), b.scales(), b.patch(), b.msWeight()), t1));
        l.add(new BV(7, "Б1′-xs", "L1-xs-F: ретинальный фильтр (вместо кадра)", SRC_RET, new FeatP4.P(b.inTt(), FeatP4.In.RETINA, tt, b.orient(), b.cell(),
                b.radius(), b.scales(), b.patch(), b.msWeight()), x0));
        l.add(new BV(8, "Б2", "POEM m = 2", SRC_POEM, poemP(b, 2, b.cell(), b.radius()), both));
        l.add(new BV(9, "Б2", "POEM ячейка w = 5", SRC_POEM, poemP(b, b.orient(), 5, b.radius()), both));
        l.add(new BV(10, "Б2", "POEM блок L = 8 (радиус 4)", SRC_POEM, poemP(b, b.orient(), b.cell(), 4), both));
        l.add(new BV(11, "Б2", "POEM блок L = 11 (радиус 5,5)", SRC_POEM, poemP(b, b.orient(), b.cell(), 5.5), both));
        l.add(new BV(12, "Б2", "POEM m = 4", SRC_POEM, poemP(b, 4, b.cell(), b.radius()), both));
        l.add(new BV(13, "Б3", "MS-LBP 4 масштаба", SRC_MS, msP(b, 4, b.patch(), 1), both));
        l.add(new BV(14, "Б3", "MS-LBP окно 32", SRC_MS, msP(b, b.scales(), 32, 1), both));
        l.add(new BV(15, "Б3", "MS-LBP вес × 0,5", SRC_MS, msP(b, b.scales(), b.patch(), 0.5), both));
        l.add(new BV(16, "Б3", "MS-LBP вес × 2", SRC_MS, msP(b, b.scales(), b.patch(), 2), both));
        l.add(new BV(17, "Б3", "MS-LBP окно 48", SRC_MS, msP(b, b.scales(), 48, 1), both));
        l.add(new BV(18, "Б3", "MS-LBP 6 масштабов", SRC_MS, msP(b, 6, b.patch(), 1), both));
        return l;
    }

    static FeatP4.P withTt(FeatP4.P b, IlluminationNorm.NormParams tt) {
        return new FeatP4.P(b.inTt(), b.inXs(), tt, b.orient(), b.cell(), b.radius(), b.scales(), b.patch(), b.msWeight());
    }

    static FeatP4.P poemP(FeatP4.P b, int m, int w, double r) {
        return new FeatP4.P(b.inTt(), b.inXs(), b.tt(), m, w, r, b.scales(), b.patch(), b.msWeight());
    }

    static FeatP4.P msP(FeatP4.P b, int s, int patch, double wt) {
        return new FeatP4.P(b.inTt(), b.inXs(), b.tt(), b.orient(), b.cell(), b.radius(), s, patch, wt);
    }

    static final List<String> GROUPS_B = List.of("Б1", "Б1′-tt", "Б1′-xs", "Б2", "Б3");

    /** Признаки варианта (как XBaseFeat2.vectors, отражения – FaceMirror для кадра «а» и k = 2). */
    static XBaseEval vectorsB(Map<String, Mat> fr, Map<String, FrameK2.F> k2, FeatP4.P p, int f, int threads) {
        Map<String, double[]> vec = new java.util.concurrent.ConcurrentHashMap<>();
        Map<String, double[]> mir = new java.util.concurrent.ConcurrentHashMap<>();
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(threads);
        try {
            pool.submit(() -> fr.entrySet().parallelStream().forEach(e -> {
                Mat a = e.getValue();
                FrameK2.F k = k2.get(e.getKey());
                if (k == null) throw new IllegalStateException("p4b: нет кадра k = 2 " + e.getKey());
                Mat am = FaceMirror.mirror(a);
                Mat cm = FaceMirror.mirror(k.frame());
                vec.put(e.getKey(), FeatP4.feature(p, f, a, k.frame(), k.pts()));
                mir.put(e.getKey(), FeatP4.feature(p, f, am, cm, FrameK2.mirrorPts(k.pts())));
                am.release();
                cm.release();
            })).get();
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("p4b: признаки " + e, e);
        } finally {
            pool.shutdown();
        }
        XBaseEval x = new XBaseEval();
        x.vec.put(XBaseEval.V.L1, new HashMap<>(vec));
        x.mir.put(XBaseEval.V.L1, new HashMap<>(mir));
        return x;
    }

    static Path dirB(XBaseVec.Setup u, int sm) {
        return u.out.resolve("xbase_" + u.base + "_p4b" + (sm == 0 ? "" : "_smoke" + sm));
    }

    static Path fileB(Path d, XBaseVec.Blk b, int f, int v) {
        return d.resolve(b.name() + "_f" + f + "_b" + v + ".tsv.gz");
    }

    /** Модели блока: v = 0 – векторы v8 пачки 2; иначе – файл варианта (признак f не меняется вариантом – v8). */
    static List<XBaseVec.Model> modelsB(XBaseVec.Setup u, int sm, XBaseVec.Blk b, int f, BV v) throws IOException {
        if (v == null || java.util.Arrays.stream(v.feats()).noneMatch(x -> x == f)) return XBaseFeat2.read(XBaseFeat2.dir(u, 0), b, f, V8);
        return XBaseVec.read(fileB(dirB(u, sm), b, f, v.no())).values().iterator().next();
    }

    /** Векторы вариантов vs по всем блокам; готовые файлы с тем же коммитом пропускаются. Возвращает время по вариантам. */
    static Map<Integer, Double> computeB(XBaseVec.Setup u, XBaseFeat2.Feats fs, List<BV> vs, int sm) throws IOException {
        Path d = dirB(u, sm);
        Files.createDirectories(d);
        String commit = FarMethods.commit();
        Map<Integer, Double> sec = new LinkedHashMap<>();
        for (BV v : vs) {
            for (int f : v.feats()) {
                List<XBaseVec.Blk> todo = new ArrayList<>();
                for (XBaseVec.Blk b : u.blocks) {
                    Path p = fileB(d, b, f, v.no());
                    if (Files.exists(p)) {
                        if (!XBaseVec.readHead(p).contains(commit)) throw new IllegalStateException("p4b: " + p + " – другой коммит кода; удалите каталог " + d);
                        continue;
                    }
                    todo.add(b);
                }
                if (todo.isEmpty()) continue;
                long t0 = System.nanoTime();
                XBaseEval x = vectorsB(u.fr, fs.k2, v.p(), f, fs.threads);
                System.out.printf(Locale.ROOT, "%s-p4b: b%d f%d признаки (%.0f с)%n", u.base, v.no(), f, (System.nanoTime() - t0) / 1e9);
                int step = Math.max(1, Math.min(4, todo.size()));
                for (int i = 0; i < todo.size(); i += step) {
                    List<List<Callable<XBaseVec.Model>>> jobs = new ArrayList<>();
                    for (XBaseVec.Blk b : todo.subList(i, Math.min(todo.size(), i + step))) {
                        jobs.add(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout));
                    }
                    List<List<XBaseVec.Model>> done = XBaseVec.run(jobs, u.tpl.threads);
                    for (int k = 0; k < done.size(); k++) {
                        XBaseVec.Blk b = todo.get(i + k);
                        Path p = fileB(d, b, f, v.no());
                        java.io.Writer w = XBaseVec.writer(p, "# Векторы 1 − cos (XBasePart4 " + u.base + "-p4b): вариант " + v.no() + " – " + v.name() + ", " + XBaseRule.FEATS[f]
                                + "\n# Код: коммит " + commit + "\n");
                        XBaseVec.write(w, String.format(b.abl2(), XBaseRule.FEATS[f]) + v.name(), done.get(k));
                        XBaseVec.finish(w, p);
                    }
                    System.out.printf(Locale.ROOT, "%s-p4b: b%d f%d блоки %d–%d из %d%n", u.base, v.no(), f, i + 1, i + jobs.size(), todo.size());
                }
                sec.merge(v.no(), (System.nanoTime() - t0) / 1e9, Double::sum);
            }
        }
        return sec;
    }

    /** Список вариантов из FACES_P4_VARIANTS (номера через запятую; пусто – все). */
    static List<BV> pickB(List<BV> all) {
        String s = System.getenv("FACES_P4_VARIANTS");
        if (s == null || s.isBlank()) return all;
        Set<Integer> want = new HashSet<>();
        for (String x : s.split(",")) want.add(Integer.parseInt(x.trim()));
        return all.stream().filter(v -> want.contains(v.no())).toList();
    }

    static final String NOTE_B = "Пачка 4, Б: параметры признака конфигурации (32) – (П2) + MS-LBP, слияние w = 0,75, β = 1, MLDA с пулом FERET, c = 1, 1 − cos. "
            + "Б1 и Б1′-tt меняют только L1-tt-xs-F,\nБ1′-xs – только L1-xs-F, Б2 и Б3 – оба кандидата (код признака один); решающее – слияние, кандидаты отдельно – "
            + "справочно. Выбор – уточнённое правило (в пределах 1 п. п. во\nвсех трёх наборах, самый простой; база первой; дальше – порядок номеров: Б2 – "
            + "меньшая размерность раньше, Б3 – 4 масштаба, окно 32, веса, окно 48, 6 масштабов).\nВекторы базы – v8 пачки 2 (XBaseFeat2), вариант – FeatP4 "
            + "(при умолчаниях побайтно v8, тест FeatP4Test).\n";

    /** pool-p4b. */
    static void poolB(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, sm);
        List<BV> all = variantsB(u.tpl.params);
        List<BV> vs = pickB(all);
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        Map<Integer, Double> sec = computeB(u, fs, vs, sm);
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        double total = (System.nanoTime() - t0) / 1e9;
        StringBuilder t = new StringBuilder("Пачка 4, Б: параметры признака – pool (XBaseEval pool-p4b)" + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\nКод: коммит "
                + FarMethods.commit() + "; векторы вариантов – " + dirB(u, sm) + "\n").append(NOTE_B);
        t.append("\n=== Варианты и источники ===\n");
        for (BV v : all) t.append(String.format(Locale.ROOT, "%d | %s | %s | %s%n", v.no(), v.group(), v.name(), v.src()));
        t.append("\n=== Время (признаки и модели; потоков ").append(u.tpl.threads).append(") ===\n");
        for (Map.Entry<Integer, Double> e : sec.entrySet()) t.append(String.format(Locale.ROOT, "вариант %d: %.0f с%n", e.getKey(), e.getValue()));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с%n", total));
        if (sm == 2) {
            double frac = 1.0 / (10.0 / XBaseNap.SMOKE2_BLOCKS * (1 + XBaseNap.FERET_TO_GT));
            t.append(String.format(Locale.ROOT, "Замер (доля полного прогона %.3f по блокам; признаки – полностью): %.0f с%n", frac, total));
        }
        int[] clamped = new int[2];
        List<String> sets = u.groups;
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBasePart4 pool-p4b): группа, номер (0 – база), вариант\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder gt = new StringBuilder("\n(Набор GT; наборы FERET – только в D:\\data\\Feret\\reports\\eq\\)\n");
        for (String grp : GROUPS_B) {
            List<BV> gv = vs.stream().filter(v -> v.group().equals(grp)).toList();
            if (gv.isEmpty()) continue;
            String[] nm = new String[gv.size() + 1];
            nm[0] = "база (32)";
            for (int i = 0; i < gv.size(); i++) nm[i + 1] = gv.get(i).no() + ". " + gv.get(i).name();
            double[][] crit = new double[sets.size()][nm.length];
            int[] featsShown = gv.get(0).feats();
            double[][][] cand = new double[2][sets.size()][nm.length];
            for (int s = 0; s < sets.size(); s++) {
                List<Integer> bi = new ArrayList<>();
                for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(sets.get(s))) bi.add(i);
                for (int j = 0; j < nm.length; j++) {
                    BV v = j == 0 ? null : gv.get(j - 1);
                    List<String> ls = new ArrayList<>();
                    List<List<String>> cl = List.of(new ArrayList<>(), new ArrayList<>());
                    for (int k : bi) {
                        XBaseVec.Blk b = u.blocks.get(k);
                        List<XBaseVec.Model> tt = modelsB(u, sm, b, 1, v);
                        List<XBaseVec.Model> xs = modelsB(u, sm, b, 0, v);
                        ls.add(XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, true), XBaseVec.fuseStats(xs, true), b.g().rOf, b.g().r0, clamped));
                        cl.get(1).add(XBaseVec.normLines(tt, null, clamped));
                        cl.get(0).add(XBaseVec.normLines(xs, null, clamped));
                    }
                    crit[s][j] = XBaseFeat2.poolSep(u, bi, ls, 1, true).crit();
                    for (int f : featsShown) cand[f][s][j] = XBaseFeat2.poolSep(u, bi, cl.get(f), f, false).crit();
                }
            }
            int ch = XBaseFeat2.chooseAll(crit);
            t.append(XBaseFeat2.lagTable(grp + " – слияние w = 0,75: вариант × набор (критерий, в скобках – отставание от лучшего, п. п.)", nm, crit, sets, ch));
            if (ch >= 0) {
                t.append("Выбор ").append(grp).append(": ").append(nm[ch]).append('\n');
            } else {
                int b = 0;
                for (int v = 1; v < nm.length; v++) if (XBaseFeat2.maxLag(crit, v) < XBaseFeat2.maxLag(crit, b)) b = v;
                t.append(String.format(Locale.ROOT, "Выбор %s: во всех трёх в пределах 1 п. п. – нет; наименьшее наибольшее отставание – %s (%.1f), условие Δ – на своих%n",
                        grp, nm[b], XBaseFeat2.maxLag(crit, b)));
            }
            sel.append(grp).append('\t').append(ch < 0 ? -1 : ch == 0 ? 0 : gv.get(ch - 1).no()).append('\t').append(ch >= 0 ? nm[ch] : "нет").append('\n');
            for (int f : featsShown) {
                double[][] c = new double[sets.size()][];
                for (int s = 0; s < sets.size(); s++) c[s] = cand[f][s];
                t.append(XBaseFeat2.lagTable("Справочно: " + grp + ", " + XBaseRule.FEATS[f] + " отдельно", nm, c, sets, -1));
            }
            for (int j = 0; j < nm.length; j++) gt.append(String.format(Locale.ROOT, "%s | %s | %.1f %%%n", grp, nm[j], crit[0][j]));
        }
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        Files.writeString(u.out.resolve("xbase_pool_p4b_select" + sfx + ".tsv"), sel.toString());
        FarMethods.write(u.out.resolve("xbase_pool_p4b" + sfx + ".txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_p4b" + sfx + "_gt.txt"), new StringBuilder(t.substring(0, t.indexOf("\n=== Время"))).append(gt));
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Совместный вариант: параметры выбранных вариантов разных групп (номер 99). */
    static BV joint(List<BV> chosen, IlluminationNorm.NormParams tt) {
        FeatP4.P p = FeatP4.base(tt);
        Set<Integer> fs = new java.util.TreeSet<>();
        StringBuilder nm = new StringBuilder("совместно:");
        for (BV v : chosen) {
            FeatP4.P q = v.p();
            FeatP4.P b = FeatP4.base(tt);
            p = new FeatP4.P(q.inTt() != b.inTt() ? q.inTt() : p.inTt(), q.inXs() != b.inXs() ? q.inXs() : p.inXs(), q.tt().equals(b.tt()) ? p.tt() : q.tt(),
                    q.orient() != b.orient() ? q.orient() : p.orient(), q.cell() != b.cell() ? q.cell() : p.cell(), q.radius() != b.radius() ? q.radius() : p.radius(),
                    q.scales() != b.scales() ? q.scales() : p.scales(), q.patch() != b.patch() ? q.patch() : p.patch(), q.msWeight() != b.msWeight() ? q.msWeight()
                    : p.msWeight());
            for (int f : v.feats()) fs.add(f);
            nm.append(' ').append(v.no()).append(". ").append(v.name()).append(';');
        }
        return new BV(99, "совместно", nm.toString(), "по выбранным вариантам", p, fs.stream().sorted(java.util.Comparator.reverseOrder()).mapToInt(Integer::intValue)
                .toArray());
    }

    /** feret-p4b, gt-p4b: база, выбранные по пулу варианты (не база) и, если меняются две и больше групп, совместный. */
    static void ownB(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        XBaseVec.Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, 0);
        Path sf = (base.equals("gt") ? u.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : u.out).resolve("xbase_pool_p4b_select" + (sm == 0 ? "" : "_smoke"
                + sm) + ".tsv");
        if (!Files.exists(sf)) throw new IllegalStateException(base + "-p4b: нет файла выбора " + sf + " (сначала pool-p4b)");
        List<BV> all = variantsB(u.tpl.params);
        List<BV> chosen = new ArrayList<>();
        List<String> undecided = new ArrayList<>();
        for (String l : Files.readAllLines(sf, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] c = l.split("\t");
            int no = Integer.parseInt(c[1]);
            if (no > 0) chosen.add(all.stream().filter(v -> v.no() == no).findFirst().orElseThrow());
            if (no < 0) undecided.add(c[0]);
        }
        StringBuilder t = new StringBuilder("Пачка 4, Б: параметры признака – " + base + " (XBaseEval " + base + "-p4b)" + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ " + sm)
                + "\nКод: коммит " + FarMethods.commit() + "; выбор – " + sf.getFileName() + "\n").append(NOTE_B);
        t.append("Выбор по пулу: ").append(chosen.isEmpty() ? "во всех группах – база" : chosen.stream().map(v -> v.no() + ". " + v.name()).toList().toString())
                .append(undecided.isEmpty() ? "" : "; без решения по пулу (наименьшее отставание – на своих): " + undecided).append('\n');
        if (chosen.isEmpty() && undecided.isEmpty()) {
            for (Mat m : u.fr.values()) m.release();
            t.append("Свои не считаются: по пулу выбрана база во всех группах.\n");
            FarMethods.write(u.out.resolve("xbase_" + base + "_p4b.txt"), t);
            System.out.print(t);
            System.out.println("DONE");
            return;
        }
        List<BV> vs = new ArrayList<>(chosen);
        if (chosen.size() >= 2) vs.add(joint(chosen, u.tpl.params));
        ownEval(u, base, vs, sm, t, t0, export, dir, settings, "p4b");
    }

    /** Свои: векторы вариантов vs, слияние «база − вариант» по группам, П3, условие Δ; отчёт xbase_&lt;база&gt;_&lt;tag&gt;.txt. */
    static void ownEval(XBaseVec.Setup u, String base, List<BV> vs, int sm, StringBuilder t, long t0, FarExport.Loaded export, Path dir, SettingsStore settings,
                        String tag) throws IOException {
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        Map<Integer, Double> sec = computeB(u, fs, vs, sm);
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        int[] clamped = new int[2];
        StringBuilder body = new StringBuilder();
        StringBuilder hold = new StringBuilder("\n=== Держит ли П3 FAR при (Т2) (контроль по снимкам; КП – верхняя 95 % граница Клоппера – Пирсона) ===\n");
        StringBuilder dl = new StringBuilder("\n=== Условие Δ (GT внутри сессии, разделимость при FAR 1 %, «вариант − база») ===\n");
        for (String grp : u.groups) {
            boolean t2 = !grp.equals("внутри");
            String prefix = (base.equals("feret") ? "FERET " : "GT ") + grp;
            List<Integer> bi = new ArrayList<>();
            for (int i = 0; i < u.blocks.size(); i++) if (u.blocks.get(i).group().equals(grp)) bi.add(i);
            Map<String, XBaseThr.Out> of = new LinkedHashMap<>();
            List<String[]> pf = new ArrayList<>();
            String bl = grp + " слияние w = 0,75, база (32)";
            List<BV> rows = new ArrayList<>();
            rows.add(null);
            rows.addAll(vs);
            for (BV v : rows) {
                List<XBaseThr.Unit> us = new ArrayList<>();
                for (int k : bi) {
                    XBaseVec.Blk b = u.blocks.get(k);
                    List<XBaseVec.Model> tt = modelsB(u, sm, b, 1, v);
                    List<XBaseVec.Model> xs = modelsB(u, sm, b, 0, v);
                    us.add(new XBaseThr.Unit(b.g(), b.gal(), b.pro(), b.parts(), XBaseVec.fuseLines(tt, xs, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, false), XBaseVec
                            .fuseStats(xs, false), b.g().rOf, b.g().r0, clamped), null, t2));
                }
                String lb = v == null ? bl : grp + " слияние w = 0,75, " + v.no() + ". " + v.name();
                XBaseThr.Out o = XBaseThr.evaluate(lb, us, u.persons, u.tpl);
                of.put(lb, o);
                if (v != null) pf.add(new String[] {lb, bl});
                if (t2) hold.append(XBaseAbl3.holds(lb, o, XBaseThr.WAYS[2]));
            }
            XBaseLbp.tables(body, prefix + " – слияние кандидатов", of, u.persons, u.tpl, pf, t2);
            if (!t2) for (String[] p : pf) {
                double[] d = XBaseVt.delta(of.get(p[0]), of.get(p[1]), XBaseThr.WAYS[0], u.persons, u.tpl);
                dl.append(String.format(Locale.ROOT, "%s: разность %+.1f п. п., верхняя граница %+.1f п. п. – условие %s%n", p[0], d[0], d[1], d[1] <= 3 ? "выполнено"
                        : "НЕ выполнено"));
            }
        }
        t.append("\n=== Время ===\n");
        for (Map.Entry<Integer, Double> e : sec.entrySet()) t.append(String.format(Locale.ROOT, "вариант %d: %.0f с%n", e.getKey(), e.getValue()));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с (потоков %d)%n", (System.nanoTime() - t0) / 1e9, u.tpl.threads));
        t.append(hold).append(base.equals("gt") ? dl : "").append(body);
        FarMethods.write(u.out.resolve("xbase_" + base + "_" + tag + (sm == 0 ? "" : "_smoke" + sm) + ".txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    // ---------------------------------------------------------------- Тан – Триггс у L1-xs-F (решение (36), Жорик №4 от 02.11)

    /** Вариант 19: L1-xs-F – Тан – Триггс с параметрами L1-tt-xs-F (у MS-LBP – σ × 2); L1-tt-xs-F не меняется. */
    static BV variantE(IlluminationNorm.NormParams tt) {
        FeatP4.P b = FeatP4.base(tt);
        return new BV(19, "Т-xs", "L1-xs-F: Тан – Триггс (как у L1-tt-xs-F)", "решение Хозяина (36), постановка Жорика №4 от 02.11; параметры – (32): γ = 0,2, "
                + "σ₀ = 1, σ₁ = 2, α = 0,1, τ = 10", new FeatP4.P(b.inTt(), FeatP4.In.TT, tt, b.orient(), b.cell(), b.radius(), b.scales(), b.patch(), b.msWeight()),
                new int[] {0});
    }

    /** pool-p4e: векторы варианта 19, критерий и парная разность с бутстрепом по людям пула (как pool-noise); слияние решает, L1-xs-F отдельно – справочно. */
    static void poolE(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, sm);
        BV v = variantE(u.tpl.params);
        XBaseFeat2.Feats fs = XBaseFeat2.feats(u, export, dir, settings, true);
        Map<Integer, Double> sec = computeB(u, fs, List.of(v), sm);
        for (Mat m : u.fr.values()) m.release();
        for (FrameK2.F k : fs.k2.values()) k.frame().release();
        int[] clamped = new int[2];
        List<String> sets = u.groups;
        String[] rows = {"база (32): слияние w = 0,75", "вариант: слияние w = 0,75, L1-xs-F с Тан – Триггсом", "L1-xs-F отдельно, база (справочно)",
            "L1-xs-F отдельно, Тан – Триггс (справочно)"};
        FeiDecompose.Cfg[][] cfg = new FeiDecompose.Cfg[rows.length][sets.size()];
        for (int s = 0; s < sets.size(); s++) {
            List<List<FeiDecompose.Cfg>> acc = new ArrayList<>();
            for (int r = 0; r < rows.length; r++) acc.add(new ArrayList<>());
            for (XBaseVec.Blk b : u.blocks) {
                if (!b.group().equals(sets.get(s))) continue;
                List<XBaseVec.Model> tt = modelsB(u, sm, b, 1, null);
                List<XBaseVec.Model> x8 = modelsB(u, sm, b, 0, null);
                List<XBaseVec.Model> xv = modelsB(u, sm, b, 0, v);
                String[] l = {XBaseVec.fuseLines(tt, x8, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, true), XBaseVec.fuseStats(x8, true), b.g().rOf, b.g().r0, clamped),
                    XBaseVec.fuseLines(tt, xv, XBaseFeat2.W_FUSE, XBaseVec.fuseStats(tt, true), XBaseVec.fuseStats(xv, true), b.g().rOf, b.g().r0, clamped),
                    XBaseVec.normLines(x8, null, clamped), XBaseVec.normLines(xv, null, clamped)};
                for (int r = 0; r < rows.length; r++) {
                    XBaseEval.Gt g = r >= 2 ? XBaseAbl3.mirrorGt(b.g(), true) : b.g();
                    acc.get(r).add(XBaseEval.parse(g, java.util.EnumSet.of(XBaseEval.V.L1), b.gal(), b.probes(), XBaseFeat2.noParts(), l[r]).cfg.get(XBaseEval.V.L1));
                }
            }
            for (int r = 0; r < rows.length; r++) cfg[r][s] = XBaseRule.merge(acc.get(r));
        }
        long seed = u.tpl.seed + NOISE_SEED_OFFSET;
        int n = u.persons;
        int[][] w = FeiBootstrap.weights(n, NOISE_B, seed);
        int[] unit = FeiBootstrap.unit(n);
        double[][] pt = new double[rows.length][sets.size()];
        double[][][] rep = new double[rows.length][sets.size()][NOISE_B];
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(u.tpl.threads);
        try {
            for (int r = 0; r < rows.length; r++) for (int s = 0; s < sets.size(); s++) {
                FeiDecompose.Cfg c = cfg[r][s];
                pt[r][s] = critW(c, unit);
                double[] o1 = rep[r][s];
                pool.submit(() -> java.util.stream.IntStream.range(0, NOISE_B).parallel().forEach(q -> o1[q] = critW(c, w[q]))).get();
            }
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("pool-p4e: " + e, e);
        } finally {
            pool.shutdown();
        }
        StringBuilder t = new StringBuilder("Тан – Триггс у L1-xs-F (решение (36), Жорик №4 от 02.11) – pool (XBaseEval pool-p4e)" + (sm == 0 ? "" : " – ПРОБНЫЙ РЕЖИМ "
                + sm) + "\nКод: коммит " + FarMethods.commit() + "; векторы варианта – " + dirB(u, sm) + "; база – векторы v8 pool-feat2 (" + XBaseFeat2.dir(u, 0) + ")\n");
        t.append("База (32): (П2) + MS-LBP, σ₀ = 1, слияние w = 0,75, β = 1, L1-xs-F всегда X+mirror; вариант – ").append(v.name()).append(" (").append(v.src())
                .append(");\nL1-tt-xs-F не меняется. Строки базы – из векторов v8 pool-feat2 (те же файлы, что pool-feat2b, pool-p4a, pool-noise). Критерий – среднее "
                        + "ошибок своих при FAR 1, 2, 5 %.\n");
        t.append(String.format(Locale.ROOT, "Бутстреп по людям пула: B = %d, seed = faces.seed + %d = %d, людей %d (как pool-noise).%n", NOISE_B, NOISE_SEED_OFFSET, seed, n));
        t.append("\n=== Строки: критерий, точка [95 %] ===\nстрока");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        for (int r = 0; r < rows.length; r++) {
            t.append(rows[r]);
            for (int s = 0; s < sets.size(); s++) {
                double[] a = FeiBootstrap.sorted(rep[r][s].clone());
                t.append(String.format(Locale.ROOT, " | %.2f %% [%.2f; %.2f]", pt[r][s], FeiBootstrap.quantile(a, 0.025), FeiBootstrap.quantile(a, 0.975)));
            }
            t.append('\n');
        }
        t.append("\n=== Парная разность «вариант − база», п. п., точка [95 %]; * – значимо ===\nпара");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        boolean better = false;
        for (int[] p : new int[][] {{1, 0}, {3, 2}}) {
            t.append(p[0] == 1 ? "слияние (решающее)" : "L1-xs-F отдельно (справочно)");
            for (int s = 0; s < sets.size(); s++) {
                double[] d = new double[NOISE_B];
                for (int q = 0; q < NOISE_B; q++) d[q] = rep[p[0]][s][q] - rep[p[1]][s][q];
                double[] ds = FeiBootstrap.sorted(d);
                double lo = FeiBootstrap.quantile(ds, 0.025);
                double hi = FeiBootstrap.quantile(ds, 0.975);
                if (p[0] == 1 && hi < 0) better = true;
                t.append(String.format(Locale.ROOT, " | %+.2f [%+.2f; %+.2f]%s", pt[p[0]][s] - pt[p[1]][s], lo, hi, lo > 0 || hi < 0 ? " *" : ""));
            }
            t.append('\n');
        }
        t.append("По пулу (правило (36)): ").append(better ? "слияние с вариантом значимо лучше базы хотя бы на одном наборе" : "значимо лучше ни на одном наборе – "
                + "решение по своим").append('\n');
        t.append("\n=== Время ===\n");
        for (Map.Entry<Integer, Double> e : sec.entrySet()) t.append(String.format(Locale.ROOT, "вариант %d: %.0f с%n", e.getKey(), e.getValue()));
        t.append(String.format(Locale.ROOT, "Всего: %.0f с (потоков %d)%n", (System.nanoTime() - t0) / 1e9, u.tpl.threads));
        String sfx = sm == 0 ? "" : "_smoke" + sm;
        FarMethods.write(u.out.resolve("xbase_pool_p4e" + sfx + ".txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_p4e" + sfx + "_gt.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** feret-p4e, gt-p4e: база (32) и вариант 19 на своих (как feret-p4b, gt-p4b). */
    static void ownE(String base, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path out) throws IOException {
        long t0 = System.nanoTime();
        int sm = smoke();
        XBaseVec.Setup u = XBaseVec.setup(base, export, dir, feiKeys, selLines, settings, out, 0);
        BV v = variantE(u.tpl.params);
        StringBuilder t = new StringBuilder("Тан – Триггс у L1-xs-F (решение (36), Жорик №4 от 02.11) – " + base + " (XBaseEval " + base + "-p4e)" + (sm == 0 ? ""
                : " – ПРОБНЫЙ РЕЖИМ " + sm) + "\nКод: коммит " + FarMethods.commit() + "\n").append(NOTE_B);
        t.append("Вариант: ").append(v.name()).append(" (").append(v.src()).append("); правило (36) на своих: значимо лучше межсессионно при FAR 1 % либо EER "
                + "при выполненном Δ.\n");
        ownEval(u, base, List.of(v), sm, t, t0, export, dir, settings, "p4e");
    }

    // ---------------------------------------------------------------- шум правила выбора (Зануда 02.11, п. 2.2; Жорик №3 от 02.11)

    static final int NOISE_B = 10000;
    /** Сдвиг seed бутстрепа шума: faces.seed + 11000. */
    static final long NOISE_SEED_OFFSET = 11000;

    /** Критерий пула (среднее ошибок своих при FAR 1, 2, 5 %) по объединённой конфигурации при весах людей w. */
    static double critW(FeiDecompose.Cfg c, int[] w) {
        double s = 0;
        for (int k = 0; k < XBaseRule.CRIT; k++) s += XBaseOther.errAtFar(List.of(c), w, XBaseRule.FARS[k])[0];
        return s / XBaseRule.CRIT;
    }

    /** pool-noise: бутстреп по людям пула критерия базы и варианта и парной разности для решений (35), (32), (31), (30) и отвергнутого (Т+М). */
    static void noise(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        XBaseVec.Setup u = XBaseVec.setup("pool", export, dir, feiKeys, selLines, settings, out, 0);
        for (Mat m : u.fr.values()) m.release();
        Path vf = XBaseFeat2.dir(u, 0);
        Path vb = dirB(u, 0);
        Path abl2 = u.out.resolve("xbase_pool_abl2_scores.tsv.gz");
        for (Path p : List.of(vf, vb, abl2)) if (!Files.exists(p)) throw new IllegalStateException("pool-noise: нет " + p);
        Map<String, String> ab = XBaseSubsp.blocks(abl2);
        int[] clamped = new int[2];
        // Строки: имя → [набор] объединённая конфигурация.
        String[] rows = {"(32) слияние (П2) + MS-LBP, w = 0,75", "(35) то же, Тан – Триггс σ₀ = 0,5", "слияние (П2), w = 0,75", "слияние (П2), w = 0,5",
            "L1-tt-xs-F (П2) отдельно", "слияние (П2) + MS-LBP, w = 0,5", "L1-tt-xs-F abl2, β = 1", "L1-tt-xs-F abl2, β = 0", "L1-xs-F abl2, β = 1",
            "L1-xs-F abl2, β = 0", "L1-xs-F (М) всегда X+mirror", "L1-xs-F (Т+М) Тан – Триггс, всегда X+mirror"};
        List<String> sets = u.groups;
        FeiDecompose.Cfg[][] cfg = new FeiDecompose.Cfg[rows.length][sets.size()];
        BV v5 = variantsB(u.tpl.params).stream().filter(v -> v.no() == 5).findFirst().orElseThrow();
        for (int s = 0; s < sets.size(); s++) {
            List<List<FeiDecompose.Cfg>> acc = new ArrayList<>();
            for (int r = 0; r < rows.length; r++) acc.add(new ArrayList<>());
            for (XBaseVec.Blk b : u.blocks) {
                if (!b.group().equals(sets.get(s))) continue;
                List<XBaseVec.Model> t8 = XBaseFeat2.read(vf, b, 1, V8);
                List<XBaseVec.Model> x8 = XBaseFeat2.read(vf, b, 0, V8);
                List<XBaseVec.Model> t0m = XBaseFeat2.read(vf, b, 1, 0);
                List<XBaseVec.Model> x0m = XBaseFeat2.read(vf, b, 0, 0);
                List<XBaseVec.Model> t5 = modelsB(u, 0, b, 1, v5);
                java.util.function.BiFunction<XBaseEval.Gt, String, FeiDecompose.Cfg> parse = (g, l) -> XBaseEval.parse(g, java.util.EnumSet.of(XBaseEval.V.L1), b.gal(),
                        b.probes(), XBaseFeat2.noParts(), l).cfg.get(XBaseEval.V.L1);
                java.util.function.BiFunction<List<List<XBaseVec.Model>>, Double, String> fuse = (m, w) -> XBaseVec.fuseLines(m.get(0), m.get(1), w, XBaseVec.fuseStats(m
                        .get(0), true), XBaseVec.fuseStats(m.get(1), true), b.g().rOf, b.g().r0, clamped);
                XBaseEval.Gt g = b.g();
                XBaseEval.Gt gm = XBaseAbl3.mirrorGt(g, true);
                String[] l = new String[rows.length];
                l[0] = fuse.apply(List.of(t8, x8), 0.75);
                l[1] = fuse.apply(List.of(t5, x8), 0.75);
                l[2] = fuse.apply(List.of(t0m, x0m), 0.75);
                l[3] = fuse.apply(List.of(t0m, x0m), 0.5);
                l[4] = XBaseVec.normLines(t0m, null, clamped);
                l[5] = fuse.apply(List.of(t8, x8), 0.5);
                String bt = String.format(b.abl2(), XBaseRule.FEATS[1]);
                String bx = String.format(b.abl2(), XBaseRule.FEATS[0]);
                String tb = XBaseAbl3.need(ab, bt + XBaseAbl2.rname(1, 0), abl2);
                String xb = XBaseAbl3.need(ab, bx + XBaseAbl2.rname(0, 0), abl2);
                l[6] = XBaseAbl3.beta(tb, 1);
                l[7] = XBaseAbl3.beta(tb, 0);
                l[8] = XBaseAbl3.beta(xb, 1);
                l[9] = XBaseAbl3.beta(xb, 0);
                l[10] = xb;
                l[11] = XBaseAbl3.need(ab, bx + XBaseAbl2.rname(0, 3), abl2);
                for (int r = 0; r < rows.length; r++) acc.get(r).add(parse.apply(r >= 10 ? gm : g, l[r]));
            }
            for (int r = 0; r < rows.length; r++) cfg[r][s] = XBaseRule.merge(acc.get(r));
        }
        // Пары: вариант, база (номера строк), подпись.
        int[][] pairs = {{1, 0}, {0, 2}, {2, 4}, {3, 2}, {5, 0}, {6, 7}, {8, 9}, {11, 10}};
        String[] pn = {"1) (35): σ₀ = 0,5 − σ₀ = 1 (слияние (32))", "2) (32): слияние (П2) + MS-LBP − слияние (П2) (w = 0,75)",
            "3а) (31): слияние (П2) w = 0,75 − L1-tt-xs-F (П2) отдельно", "3б) (31): слияние (П2) w = 0,5 − w = 0,75 (пачка 1)",
            "3в) слияние (П2) + MS-LBP w = 0,5 − w = 0,75 (pool-feat2b)", "4а) (30): L1-tt-xs-F β = 1 − β = 0", "4б) (30): L1-xs-F β = 1 − β = 0",
            "5) отвергнутый: L1-xs-F (Т+М) − (М)"};
        long seed = u.tpl.seed + NOISE_SEED_OFFSET;
        int n = u.persons;
        int[][] w = FeiBootstrap.weights(n, NOISE_B, seed);
        int[] unit = FeiBootstrap.unit(n);
        double[][] pt = new double[rows.length][sets.size()];
        double[][][] rep = new double[rows.length][sets.size()][NOISE_B];
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(u.tpl.threads);
        try {
            for (int r = 0; r < rows.length; r++) {
                for (int s = 0; s < sets.size(); s++) {
                    FeiDecompose.Cfg c = cfg[r][s];
                    pt[r][s] = critW(c, unit);
                    double[] out1 = rep[r][s];
                    pool.submit(() -> java.util.stream.IntStream.range(0, NOISE_B).parallel().forEach(q -> out1[q] = critW(c, w[q]))).get();
                }
                System.out.printf(Locale.ROOT, "pool-noise: строка %d из %d (%.0f с)%n", r + 1, rows.length, (System.nanoTime() - t0) / 1e9);
            }
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("pool-noise: " + e, e);
        } finally {
            pool.shutdown();
        }
        StringBuilder t = new StringBuilder("Шум правила выбора по пулу – pool (XBaseEval pool-noise; рецензия Зануды 02.11, п. 2.2; Жорик №3 от 02.11)\nКод: коммит "
                + FarMethods.commit() + "\n");
        t.append(String.format(Locale.ROOT, "Бутстреп по людям пула: B = %d, seed = faces.seed + %d = %d, людей %d (человек вытягивается целиком, одни веса на все строки "
                + "и наборы).%nКритерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; интервалы – процентили 2,5–97,5 %%; «значимо» – интервал "
                + "разности не содержит 0.%nИсточники: векторы pool-feat2 (v0 – (П2), v8 – (П2) + MS-LBP) – %s; pool-p4b (вариант 5) – %s; файл оценок pool-abl2 – %s "
                + "(β и (Т+М), как pool-comb).%n", NOISE_B, NOISE_SEED_OFFSET, seed, n, vf, vb, abl2.getFileName()));
        t.append("\n=== Строки: критерий, точка [95 %] по наборам ===\nстрока");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        for (int r = 0; r < rows.length; r++) {
            t.append(rows[r]);
            for (int s = 0; s < sets.size(); s++) {
                double[] a = FeiBootstrap.sorted(rep[r][s].clone());
                t.append(String.format(Locale.ROOT, " | %.2f %% [%.2f; %.2f]", pt[r][s], FeiBootstrap.quantile(a, 0.025), FeiBootstrap.quantile(a, 0.975)));
            }
            t.append('\n');
        }
        t.append("\n=== Пары «вариант − база»: разность критерия, п. п., точка [95 %]; * – значимо ===\nпара");
        for (String s : sets) t.append(" | ").append(s);
        t.append('\n');
        for (int p = 0; p < pairs.length; p++) {
            t.append(pn[p]);
            for (int s = 0; s < sets.size(); s++) {
                double[] d = new double[NOISE_B];
                for (int q = 0; q < NOISE_B; q++) d[q] = rep[pairs[p][0]][s][q] - rep[pairs[p][1]][s][q];
                double[] ds = FeiBootstrap.sorted(d);
                double lo = FeiBootstrap.quantile(ds, 0.025);
                double hi = FeiBootstrap.quantile(ds, 0.975);
                t.append(String.format(Locale.ROOT, " | %+.2f [%+.2f; %+.2f]%s", pt[pairs[p][0]][s] - pt[pairs[p][1]][s], lo, hi, lo > 0 || hi < 0 ? " *" : ""));
            }
            t.append('\n');
        }
        t.append("\n=== Шум критерия: полуширина 95 % интервала базы (32), п. п. ===\n");
        for (int s = 0; s < sets.size(); s++) {
            double[] a = FeiBootstrap.sorted(rep[0][s].clone());
            t.append(String.format(Locale.ROOT, "%s: %.2f%n", sets.get(s), (FeiBootstrap.quantile(a, 0.975) - FeiBootstrap.quantile(a, 0.025)) / 2));
        }
        t.append(String.format(Locale.ROOT, "%nВремя: %.0f с (потоков %d).%n", (System.nanoTime() - t0) / 1e9, u.tpl.threads));
        FarMethods.write(u.out.resolve("xbase_pool_noise.txt"), t);
        Files.createDirectories(out);
        FarMethods.write(out.resolve("xbase_pool_noise_gt.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }
}
