package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Правило оценки на признаке L1 (ревизия 23.10.2026, пп. 1.2, 2.1, 2.7; задание №1 Жорика от 24.10.2026, план
 * подтверждён Хозяином). Проекция MLDA с пулом FERET (L1-xs-F, L1-tt-xs-F) обучается как сейчас (XBaseEval.fit), по одной
 * модели считаются все строки: (0) база – евклид по всем осям; (а) усечение до k первых осей (оси упорядочены по
 * собственным значениям S_b), евклид; (б) 1 − cos до центра класса, все оси и k; (в) оси с весом √λ_j (λ_j – собственное
 * значение S_b оси), евклид и косинус; (г) WPCA + косинус на векторах L1: PCA и отбеливание – на пуле FERET, m компонент,
 * центр – среднее отбелённых векторов шаблона, MLDA не используется. Решение – d₁/d₂ (FeiScores.ratio), модель X / X+mirror по
 * |r| пробы. Режимы: pool-rule – выбор по пулу FERET перекрёстно по людям (только D:\data\Feret\reports\eq\), gt-rule – GT
 * межс. N = 6 и внутри N = 3 (как gt-subsp), feret-rule – роль «межс.» feret-xs2; строка (0) сверяется с прежними файлами
 * оценок побайтно до разбора.
 *
 * @author ssv
 */
final class XBaseRule {

    /** Семейство строки: евклид, косинус, евклид и косинус с весом √λ, WPCA + косинус. */
    enum Fam {
        EUC, COS, WEUC, WCOS, WPCA
    }

    /** Строка: подпись, семейство, k (осей) или m (компонент WPCA); ALL – все. */
    record Rv(String label, Fam fam, int k) {}

    static final int ALL = Integer.MAX_VALUE;
    static final String BASE = "(0) база";
    static final String CHOICE = "выбор по пулу";
    /** Строки в порядке «от простого к сложному» (правило выбора по пулу, №2 Жорика от 24.10.2026, п. 2). */
    static final List<Rv> RVS = List.of(new Rv(BASE, Fam.EUC, ALL), new Rv("(б) косинус, все оси", Fam.COS, ALL), new Rv("(а) k = 20", Fam.EUC, 20),
            new Rv("(б) косинус, k = 20", Fam.COS, 20), new Rv("(а) k = 50", Fam.EUC, 50), new Rv("(б) косинус, k = 50", Fam.COS, 50),
            new Rv("(а) k = 100", Fam.EUC, 100), new Rv("(б) косинус, k = 100", Fam.COS, 100), new Rv("(в) √λ, евклид", Fam.WEUC, ALL),
            new Rv("(в) √λ, косинус", Fam.WCOS, ALL), new Rv("(г) WPCA m = 100", Fam.WPCA, 100), new Rv("(г) WPCA m = 200", Fam.WPCA, 200),
            new Rv("(г) WPCA m = 500", Fam.WPCA, 500), new Rv("(г) WPCA m = 900", Fam.WPCA, 900));
    static final int MAX_M = 900;
    static final String[] FEATS = {"L1-xs-F", "L1-tt-xs-F"};
    static final double[] FARS = {0.01, 0.02, 0.05, 0.10};
    /** Критерий выбора – среднее ошибок своих при фактическом FAR контроля 1, 2 и 5 % (первые CRIT уровней FARS). */
    static final int CRIT = 3;
    static final double NEAR = 1.0;
    static final long PART_SEED_OFFSET = 10200;
    static final long HALF_SEED_OFFSET = 10201;
    static final String SELECT = "xbase_pool_rule_select.tsv";

    private XBaseRule() {
    }

    // ---------------------------------------------------------------- модели

    /** PCA пула: среднее, направления (строки) по убыванию σ, σ. */
    record Pca(double[] mean, double[][] basis, double[] sigma) {
        static Pca of(List<double[]> xs) {
            LdaMath.Data d = LdaMath.Data.of(List.of(xs));
            LdaMath.Span s = LdaMath.span(d, 1e-10, MAX_M);
            return new Pca(d.mean(), s.basis(), s.sigma());
        }

        /** Отбелённые координаты: z_j = basisⱼ·(x − mean) / σ_j (общий множитель √(N − 1) на косинус не влияет). */
        double[] whiten(double[] x) {
            double[] z = new double[basis.length];
            for (int j = 0; j < z.length; j++) {
                double s = 0;
                double[] b = basis[j];
                for (int i = 0; i < x.length; i++) s += b[i] * (x[i] - mean[i]);
                z[j] = s / sigma[j];
            }
            return z;
        }
    }

    /** Модель одного обучения: LDA, собственные значения S_b по осям, центры WPCA галереи. */
    static final class Fit {
        final LdaModel lm;
        final double[] lamB;
        final Pca pca;
        final double[][] wCenters;

        Fit(LdaModel lm, double[] lamB, Pca pca, double[][] wCenters) {
            this.lm = lm;
            this.lamB = lamB;
            this.pca = pca;
            this.wCenters = wCenters;
        }

        int axes() {
            return lm.projection().length;
        }

        /** Оценки пробы x по всем строкам RVS (в их порядке). */
        double[][] scores(double[] x) {
            double[] y = lm.embed(x);
            double[] z = pca.whiten(x);
            double[][] c = lm.centers();
            double[][] out = new double[RVS.size()][c.length];
            for (int v = 0; v < RVS.size(); v++) {
                Rv rv = RVS.get(v);
                for (int g = 0; g < c.length; g++) {
                    if (c[g] == null) {
                        out[v][g] = Double.POSITIVE_INFINITY;
                        continue;
                    }
                    out[v][g] = switch (rv.fam()) {
                        case EUC -> rv.k() == ALL ? Math.sqrt(NearestVectorScorer.distance2(y, c[g])) : euc(y, c[g], Math.min(rv.k(), y.length), null);
                        case COS -> cos(y, c[g], Math.min(rv.k(), y.length), null);
                        case WEUC -> euc(y, c[g], y.length, lamB);
                        case WCOS -> cos(y, c[g], y.length, lamB);
                        case WPCA -> cos(z, wCenters[g], Math.min(rv.k(), z.length), null);
                    };
                }
            }
            return out;
        }
    }

    /** Евклид по первым k координатам; w – веса √w_j (null – без весов). */
    static double euc(double[] a, double[] b, int k, double[] w) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            double d = a[j] - b[j];
            s += w == null ? d * d : w[j] * d * d;
        }
        return Math.sqrt(s);
    }

    /** 1 − cos по первым k координатам; w – веса координат √w_j (null – без весов). */
    static double cos(double[] a, double[] b, int k, double[] w) {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int j = 0; j < k; j++) {
            double q = w == null ? 1 : w[j];
            ab += q * a[j] * b[j];
            aa += q * a[j] * a[j];
            bb += q * b[j] * b[j];
        }
        if (aa == 0 || bb == 0) return 1;
        return 1 - ab / Math.sqrt(aa * bb);
    }

    /**
     * Собственные значения S_b по осям: в отбелённом пространстве MLDA ось j – собственный вектор с λ_j, равным
     * взвешенному межклассовому разбросу проекций средних классов обучения Σ n_k ȳ_kj² (общее среднее проекций – 0).
     */
    static double[] lambdaB(LdaModel lm, List<List<double[]>> cls) {
        double[] lam = new double[lm.projection().length];
        for (List<double[]> list : cls) {
            if (list.isEmpty()) continue;
            double[] m = new double[list.get(0).length];
            for (double[] v : list) for (int i = 0; i < m.length; i++) m[i] += v[i];
            for (int i = 0; i < m.length; i++) m[i] /= list.size();
            double[] y = lm.embed(m);
            for (int j = 0; j < lam.length; j++) lam[j] += list.size() * y[j] * y[j];
        }
        return lam;
    }

    /** Обучение MLDA (как XBaseEval.fit для V.L1) и центры WPCA галереи по PCA пула pca. */
    static Fit fit(XBaseEval x, boolean mirror, List<List<String>> gallery, List<List<String>> cohort, Pca pca) {
        return fit(x, mirror, gallery, cohort, pca, 1.0);
    }

    /** То же с полом MLDA λ̄·c (ревизия 1.3, XBaseReg); pca == null – без центров WPCA (строки (г) не считаются). */
    static Fit fit(XBaseEval x, boolean mirror, List<List<String>> gallery, List<List<String>> cohort, Pca pca, double floorC) {
        List<List<double[]>> cls = new ArrayList<>();
        for (List<List<String>> group : List.of(gallery, cohort)) {
            for (List<String> keys : group) {
                List<double[]> list = new ArrayList<>();
                for (String k : keys) list.add(x.vec.get(V.L1).get(k));
                if (mirror) for (String k : keys) list.add(x.mir.get(V.L1).get(k));
                cls.add(list);
            }
        }
        LdaModel lm = (LdaModel) new MldaScorer(floorC).fit(cls, gallery.size());
        double[][] wc = new double[gallery.size()][];
        for (int c = 0; c < gallery.size() && pca != null; c++) {
            List<double[]> list = cls.get(c);
            if (list.isEmpty()) continue;
            double[] m = new double[pca.basis().length];
            for (double[] v : list) {
                double[] z = pca.whiten(v);
                for (int j = 0; j < m.length; j++) m[j] += z[j];
            }
            for (int j = 0; j < m.length; j++) m[j] /= list.size();
            wc[c] = m;
        }
        return new Fit(lm, lambdaB(lm, cls), pca, wc);
    }

    /** PCA пула по ключам (с отражениями у модели +mirror). */
    static Pca pca(XBaseEval x, List<String> keys, boolean mirror) {
        List<double[]> xs = new ArrayList<>();
        for (String k : keys) xs.add(x.vec.get(V.L1).get(k));
        if (mirror) for (String k : keys) xs.add(x.mir.get(V.L1).get(k));
        return Pca.of(xs);
    }

    /** Строки оценок всех вариантов одной модели: подпись строки → строки FeiScores. */
    static List<StringBuilder> lines(XBaseEval x, String method, int s, Fit f, List<String> gallery, List<Probe> probes) {
        List<StringBuilder> out = new ArrayList<>();
        for (int v = 0; v < RVS.size(); v++) out.add(new StringBuilder());
        for (Probe p : probes) {
            double[][] sc = f.scores(x.vec.get(V.L1).get(p.key()));
            for (int v = 0; v < RVS.size(); v++) {
                out.get(v).append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc[v], false));
            }
        }
        return out;
    }

    /** Сведения о модели: оси, ранг PCA, строка MLDA (заменено собственных значений S_w). */
    record Info(int axes, int pcaRank, String mlda) {}

    /** Результат fitLines: строки по вариантам (в порядке RVS) и сведения об основных моделях X и X+mirror. */
    record Lines(List<String> byRv, Info[] info) {}

    /**
     * Строки оценок вариантов (как XBaseEval.fitLines для V.L1, тот же порядок задач): основная модель и модели без части q,
     * оба зеркала; parts == null – только основная модель (выбор по пулу).
     */
    static Lines fitLines(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, int s,
                          Pca[] pcas) {
        return fitLines(gt, x, galleryKeys, mainProbes, parts, s, pcas, null);
    }

    /** То же; extra – дополнительные пробы моделей без части q (ревизия 1.14, XBaseThr), null – нет. */
    static Lines fitLines(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> galleryKeys, List<Probe> mainProbes, List<List<String>> parts, int s,
                          Pca[] pcas, List<List<Probe>> extra) {
        List<String> own = gt.own;
        List<List<String>> cohort = gt.cohort;
        List<List<String>> fullGallery = own.stream().map(galleryKeys::get).toList();
        Info[] info = new Info[2];
        List<Callable<List<StringBuilder>>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            Pca pc = pcas[m ? 1 : 0];
            tasks.add(() -> {
                Fit f = fit(x, m, fullGallery, cohort, pc);
                info[m ? 1 : 0] = new Info(f.axes(), pc.basis().length, f.lm.info());
                return lines(x, method, s, f, own, mainProbes);
            });
            if (parts == null) continue;
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(galleryKeys::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : galleryKeys.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                tasks.add(() -> lines(x, method, s, fit(x, m, gk, cohort, pc), g, pr));
            }
        }
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(gt.threads);
        List<StringBuilder> all = new ArrayList<>();
        for (int v = 0; v < RVS.size(); v++) all.add(new StringBuilder());
        try {
            for (java.util.concurrent.Future<List<StringBuilder>> f : ex.invokeAll(tasks)) {
                List<StringBuilder> r = f.get();
                for (int v = 0; v < RVS.size(); v++) all.get(v).append(r.get(v));
            }
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseRule: " + e, e);
        } finally {
            ex.shutdown();
        }
        return new Lines(all.stream().map(StringBuilder::toString).toList(), info);
    }

    /** Строки и сведения из контрольной точки (null – нет всех частей). */
    static Lines read(XBaseOther.Checkpoint cp, String tag) throws IOException {
        List<String> ls = new ArrayList<>();
        for (int v = 0; v < RVS.size(); v++) {
            String r = cp.read(tag + "_" + v);
            if (r == null) return null;
            ls.add(r);
        }
        String inf = cp.read(tag + "_info");
        if (inf == null) return null;
        String[] p = inf.split("\n");
        Info[] info = new Info[2];
        for (int m = 0; m < 2; m++) {
            String[] c = p[m].split("\t", 3);
            info[m] = new Info(Integer.parseInt(c[0]), Integer.parseInt(c[1]), c[2]);
        }
        return new Lines(ls, info);
    }

    static void save(XBaseOther.Checkpoint cp, String tag, Lines l) throws IOException {
        for (int v = 0; v < RVS.size(); v++) cp.save(tag + "_" + v, l.byRv().get(v));
        StringBuilder t = new StringBuilder();
        for (Info i : l.info()) t.append(i.axes()).append('\t').append(i.pcaRank()).append('\t').append(i.mlda()).append('\n');
        cp.save(tag + "_info", t.toString());
    }

    // ---------------------------------------------------------------- меры

    /** Разделимость строки: ошибка своих при фактическом FAR контроля (FARS), EER, медианы, реплики бутстрепа. */
    static final class Sep {
        double[] err = new double[FARS.length];
        double[] far = new double[FARS.length];
        double[][] errRep;
        double[] eer;
        double[] eerRep;
        double medOwn;
        double medCtrl;

        double crit() {
            double s = 0;
            for (int k = 0; k < CRIT; k++) s += err[k];
            return s / CRIT;
        }
    }

    static Sep sep(List<FeiDecompose.Cfg> cs, int[] unit, int[][] w) {
        Sep s = new Sep();
        s.errRep = new double[FARS.length][w.length];
        for (int k = 0; k < FARS.length; k++) {
            double[] p = XBaseOther.errAtFar(cs, unit, FARS[k]);
            s.err[k] = p[0];
            s.far[k] = p[1];
            for (int q = 0; q < w.length; q++) s.errRep[k][q] = XBaseOther.errAtFar(cs, w[q], FARS[k])[0];
        }
        XBaseSep.Data d = XBaseOther.pooled(cs);
        s.eer = XBaseSep.eer(d, unit);
        s.eerRep = new double[w.length];
        for (int q = 0; q < w.length; q++) s.eerRep[q] = XBaseSep.eer(d, w[q])[3];
        List<Double> ok = new ArrayList<>();
        for (int i = 0; i < d.ownR.length; i++) if (d.ownOk[i]) ok.add(d.ownR[i]);
        s.medOwn = ok.isEmpty() ? Double.NaN : FeiBootstrap.quantile(FeiBootstrap.sorted(ok.stream().mapToDouble(Double::doubleValue).toArray()), 0.5);
        s.medCtrl = FeiBootstrap.quantile(FeiBootstrap.sorted(d.ctrlR), 0.5);
        return s;
    }

    /** Все конфигурации одной (для выбора по пулу: порог по объединённым контрольным). */
    static FeiDecompose.Cfg merge(List<FeiDecompose.Cfg> cs) {
        XBaseSep.Data d = XBaseOther.pooled(cs);
        FeiDecompose.Cfg c = new FeiDecompose.Cfg();
        c.ownP = d.ownP;
        c.ownR = d.ownR;
        c.ownOk = d.ownOk;
        c.ctrlP = d.ctrlP;
        c.ctrlR = d.ctrlR;
        return c;
    }

    /**
     * Выбор по пулу (№2 Жорика от 24.10.2026, п. 2): критерий – среднее ошибок своих при FAR 1, 2, 5 %; база в пределах NEAR
     * п. п. от лучшего – база; иначе первый в порядке RVS среди строк в пределах NEAR от лучшего.
     */
    static int choose(double[] crit) {
        double best = Double.POSITIVE_INFINITY;
        for (double c : crit) best = Math.min(best, c);
        if (crit[0] <= best + NEAR) return 0;
        for (int v = 0; v < crit.length; v++) if (crit[v] <= best + NEAR) return v;
        throw new IllegalStateException("XBaseRule: выбор не определён");
    }

    static String fmt(double v) {
        return Double.isNaN(v) ? "–" : String.format(Locale.ROOT, "%.1f %%", v);
    }

    // ---------------------------------------------------------------- таблицы

    /** Строки итога: подпись → Res и Sep. */
    static void tables(StringBuilder t, String title, Map<String, FeiThreshold.Res> res, int persons, XBaseEval.Gt tpl, List<String[]> pairs) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), sep(e.getValue().cfgs, FeiBootstrap.unit(persons), w));
        t.append("\n\n######## ").append(title).append(" ########\n");
        t.append("\n=== Разделимость (ошибка своих при фактическом FAR контроля; порог по контрольным чужим своей конфигурации), EER, П3 FAR 1 %, "
                + "медианы d₁/d₂ ===\n");
        t.append("строка | разд. 1 % | 2 % | 5 % | 10 % | EER | П3 FAR 1 %: ошибка своих | FAR контроля | медиана d₁/d₂: свои с верным лучшим | "
                + "контрольные чужие\n");
        for (Map.Entry<String, Sep> e : sep.entrySet()) {
            Sep s = e.getValue();
            FeiThreshold.Res r = res.get(e.getKey());
            List<String> row = new ArrayList<>(List.of(e.getKey()));
            for (int k = 0; k < FARS.length; k++) row.add(XBaseSepLtp.cell(s.err[k], s.errRep[k]) + String.format(Locale.ROOT, " (%.2f %%)", s.far[k]));
            row.add(XBaseSepLtp.cell(s.eer[3], s.eerRep));
            row.add(FeiThreshold.errCell(r, 3, 0, 0, tpl.b));
            row.add(FeiThreshold.farCell(r, 3, 0, tpl.b));
            row.add(String.format(Locale.ROOT, "%.4f", s.medOwn));
            row.add(String.format(Locale.ROOT, "%.4f", s.medCtrl));
            t.append(String.join(" | ", row)).append('\n');
        }
        if (tpl.b > 0) {
            t.append("\n=== Парные разности «строка − база» разделимости, п. п. (одни веса бутстрепа; «*» – интервал [2,5; 97,5] не содержит 0) ===\n");
            t.append("пара | 1 % | 2 % | 5 % | 10 % | EER\n");
            for (String[] p : pairs) {
                Sep a = sep.get(p[0]);
                Sep z = sep.get(p[1]);
                List<String> row = new ArrayList<>(List.of(p[0] + " − " + p[1]));
                for (int k = 0; k < FARS.length; k++) row.add(XBaseSepLtp.diff(a.err[k] - z.err[k], a.errRep[k], z.errRep[k]));
                row.add(XBaseSepLtp.diff(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                t.append(String.join(" | ", row)).append('\n');
            }
            t.append("\n=== Парные разности «строка − база» при П3, п. п. ===\n");
            t.append("пара | способ | ").append(LbpNormReport.DIFF_HEAD).append('\n');
            for (String[] p : pairs) t.append(LbpNormReport.diffRow(List.of(p[0] + " − " + p[1], FeiThreshold.METHODS[3]), res.get(p[0]), res.get(p[1]), 3));
        }
    }

    static final String MODEL_NOTE = "Строки (одна модель MLDA на обучение, без переобучения): (0) база – евклид по всем осям MLDA; (а) первые k осей (по "
            + "убыванию собственных значений S_b), евклид; (б) 1 − cos до центра класса в пространстве MLDA (начало координат – общее среднее\n"
            + "обучения); (в) оси с весом √λ_j (λ_j – собственное значение S_b оси, Σ n_k ȳ_kj² по средним классов обучения), евклид и косинус; (г) WPCA:\n"
            + "PCA векторов L1 (после корня гистограмм) только пула FERET (у модели +mirror – с отражениями), отбеливание делением на σ_j, первые m\n"
            + "компонент, центр – среднее отбелённых векторов шаблона, 1 − cos; MLDA в (г) не используется. k и m ограничены числом осей и рангом\n"
            + "(фактические – в таблице «Модели»). Решение – d₁/d₂ (FeiScores.ratio, п. 5 – округление 1 − cos ниже нуля), модель X / X+mirror по |r|\n"
            + "пробы.\n";

    static String infoRow(String head, Info[] in) {
        return String.format(Locale.ROOT, "%s | %d / %d | %d / %d | %s%n", head, in[0].axes(), in[1].axes(), in[0].pcaRank(), in[1].pcaRank(), replaced(in[0].mlda()));
    }

    /** «заменено собственных значений в оболочке A из B» из строки MLDA. */
    static String replaced(String info) {
        int i = info.indexOf("заменено");
        return i < 0 ? info : info.substring(i);
    }

    static final String INFO_HEAD = "\n=== Модели (основная модель X / X+mirror): осей MLDA | ранг PCA пула (наибольшее фактическое m) | S_w: пол λ̄ ===\n"
            + "строка | осей | ранг PCA | заменено собственных значений S_w (X)\n";

    // ---------------------------------------------------------------- выбор по пулу

    /** Выбранные строки: набор (GT, FERET-1, FERET-2) → признак → индекс в RVS. */
    static Map<String, Map<String, Integer>> readSelect(Path f) throws IOException {
        if (!Files.exists(f)) throw new IllegalStateException("XBaseRule: нет файла выбора " + f + " – сначала pool-rule");
        Map<String, Map<String, Integer>> out = new TreeMap<>();
        for (String l : Files.readAllLines(f)) {
            if (l.startsWith("#") || l.isBlank()) continue;
            String[] c = l.split("\t");
            int v = -1;
            for (int i = 0; i < RVS.size(); i++) if (RVS.get(i).label().equals(c[2])) v = i;
            if (v < 0) throw new IllegalStateException("XBaseRule: неизвестная строка в " + f + ": " + l);
            out.computeIfAbsent(c[0], k -> new TreeMap<>()).put(c[1], v);
        }
        return out;
    }

    /** Части по людям: по возрастанию, перестановка Random(seed), i-й – в часть i mod 5; каждая по возрастанию. */
    static List<List<String>> parts5(Set<String> persons, long seed) {
        List<String> p = new ArrayList<>(new TreeSet<>(persons));
        Collections.shuffle(p, new Random(seed));
        List<List<String>> out = new ArrayList<>();
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) out.add(new ArrayList<>());
        for (int i = 0; i < p.size(); i++) out.get(i % FeiMethods.CV_PARTS).add(p.get(i));
        for (List<String> l : out) Collections.sort(l);
        return out;
    }

    /** Общая подготовка FERET: пул, половины №2, когорта FEI, векторы L1 и L1-tt. */
    static final class FSetup {
        XBaseA0.FeretSetup su;
        XBaseEval.Gt tpl;
        Path out;
        Map<String, List<List<String>>> pool;
        List<Set<String>> halves;
        List<List<String>> cohort;
        XBaseEval[] xs;
    }

    static FSetup fsetup(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                         boolean roles) throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("XBaseRule: только равные галереи (faces.feret.gallery = eq)");
        FSetup s = new FSetup();
        s.su = XBaseA0.feretSetup(exportDir, selLines, settings);
        s.tpl = s.su.tpl;
        s.out = s.su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Files.createDirectories(s.out);
        s.pool = XBaseXsF.pool(s.su.ro, s.su.byKey);
        List<String> perm = new ArrayList<>(s.pool.keySet());
        Collections.shuffle(perm, new Random(s.tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        s.halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, s.tpl.seed);
        s.cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            s.cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        Set<String> used = new TreeSet<>();
        if (roles) {
            XBaseFeret.Roles r = s.su.ro;
            for (Map<String, List<String>> m : List.of(r.gal1, r.pro1, r.g2, r.g2p, r.pro2, r.ctrlKeys)) m.values().forEach(used::addAll);
        }
        for (List<List<String>> ds : s.pool.values()) for (List<String> d : ds) used.addAll(d);
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(s.su.dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        s.xs = new XBaseEval[] {new XBaseEval(), new XBaseEval()};
        s.xs[0].vectors(fr, EnumSet.of(V.L1), s.tpl.params);
        s.xs[1].vectors(fr, EnumSet.of(V.L1), s.tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        return s;
    }

    static List<String> keys(Map<String, List<List<String>>> pool, Set<String> persons) {
        List<String> out = new ArrayList<>();
        for (String p : new TreeSet<>(persons)) for (List<String> d : pool.get(p)) out.addAll(d);
        return out;
    }

    /**
     * pool-rule: выбор строки по пулу FERET перекрёстно по людям. Наборы: GT – весь пул (выбор для gt-rule); FERET-1 – половина B
     * (для прохода 1 feret-rule, свои – A), FERET-2 – половина A. Набор – 5 частей по людям (Random(faces.seed + 10200));
     * часть q отложена, внутри неё люди пополам (Random(faces.seed + 10201 + q)): одна половина – свои (галерея – первая дата
     * пула, fa и fb; пробы – остальные даты), другая – контрольные чужие (все снимки), затем половины меняются. Обучение MLDA –
     * галерея своих + когорта FEI 20 + люди остальных 4 частей (человек – один класс по всем датам); PCA (г) – только эти 4
     * части. Оценки 5 частей × 2 смены объединяются; критерий – среднее ошибок своих при FAR 1, 2, 5 % (порог по объединённым
     * контрольным).
     */
    static void pool(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                     Path gtOut) throws IOException {
        FSetup s = fsetup(export, exportDir, feiKeys, selLines, settings, false);
        XBaseEval.Gt tpl = s.tpl;
        List<String> all = new ArrayList<>(s.pool.keySet());
        XBaseEval.Gt base = XBaseFeret.gt(tpl, all, List.of(), Map.of(), s.cohort, s.su.rOf, s.su.imageNo);
        Map<String, Set<String>> sets = new LinkedHashMap<>();
        sets.put("GT", new TreeSet<>(all));
        sets.put("FERET-1", s.halves.get(1));
        sets.put("FERET-2", s.halves.get(0));
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "pool-rule %s; seed %d; r0 %s; пул %d; половины %d/%d", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_pool_rule_eq"), fp);
        StringBuilder head = new StringBuilder();
        head.append("Ревизия 23.10, пп. 1.2, 2.1: выбор правила оценки по пулу FERET (XBaseEval pool-rule)\n");
        head.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        head.append("Данные: пул Color FERET (люди с ≥ 2 датами, ").append(s.pool.size()).append(", снимки по правилу eq); когорта FEI – экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        head.append(String.format(Locale.ROOT, "Наборы: GT – весь пул (выбор для gt-rule); FERET-1 – половина B (%d; выбор для прохода 1 feret-rule, свои – A); "
                + "FERET-2 – половина A (%d; для прохода 2). Половины №2 – Random(faces.seed + %d).%n", s.halves.get(1).size(), s.halves.get(0).size(),
                XBaseXsF.SPLIT_SEED_OFFSET));
        head.append(String.format(Locale.ROOT, "Схема: 5 частей по людям набора (Random(faces.seed + %d)); часть q отложена, её люди пополам (Random(faces.seed + "
                + "%d + q)): одна половина – свои (галерея – первая дата пула, fa и fb; пробы – снимки остальных дат), другая – контрольные чужие (все\n"
                + "снимки); затем половины меняются. Обучение MLDA – галерея своих + когорта FEI 20 + люди остальных 4 частей (человек – один класс по\n"
                + "всем датам); PCA (г) – только люди 4 частей. Оценки 5 частей × 2 смены объединены, порог разделимости – по объединённым\n"
                + "контрольным. Критерий – среднее ошибок своих при фактическом FAR контроля 1, 2, 5 %%; база в пределах %.0f п. п. от лучшего – база,\n"
                + "иначе первая в пределах %.0f п. п. в порядке: (б) косинус по всем осям; (а) или (б) с меньшим k (при равном k – (а)); (в) евклид,\n"
                + "косинус; (г) с меньшим m. Совпадающие фактически строки дают равные оценки – из них берётся первая по порядку.%n", PART_SEED_OFFSET,
                HALF_SEED_OFFSET, NEAR, NEAR));
        head.append(MODEL_NOTE);
        StringBuilder vol = new StringBuilder("\n=== Объём ===\nнабор | часть | смена | своих | проб своих | контрольных людей | контрольных проб | людей в обучении "
                + "из пула\n");
        StringBuilder info = new StringBuilder(INFO_HEAD);
        StringBuilder body = new StringBuilder();
        StringBuilder sel = new StringBuilder("# Выбор по пулу (XBaseRule pool-rule): набор, признак, строка\n# Код: коммит " + FarMethods.commit() + "\n");
        StringBuilder gtBody = new StringBuilder();
        StringBuilder scores = new StringBuilder();
        double fitS = 0;
        for (Map.Entry<String, Set<String>> se : sets.entrySet()) {
            String set = se.getKey();
            List<List<String>> parts = parts5(se.getValue(), tpl.seed + PART_SEED_OFFSET);
            StringBuilder tb = new StringBuilder();
            tb.append("\n\n######## Набор ").append(set).append(" (").append(se.getValue().size()).append(" человек) ########\n");
            tb.append("строка | признак | ошибка своих при FAR 1 % | 2 % | 5 % | среднее (критерий) | EER | медиана d₁/d₂: свои с верным лучшим | контрольные\n");
            StringBuilder choice = new StringBuilder();
            for (int f = 0; f < FEATS.length; f++) {
                List<List<FeiDecompose.Cfg>> cfgs = new ArrayList<>();
                for (int v = 0; v < RVS.size(); v++) cfgs.add(new ArrayList<>());
                for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                    List<String> held = new ArrayList<>(parts.get(q));
                    Set<String> train = new TreeSet<>(se.getValue());
                    held.forEach(train::remove);
                    Collections.shuffle(held, new Random(tpl.seed + HALF_SEED_OFFSET + q));
                    int h0 = (held.size() + 1) / 2;
                    List<List<String>> hv = List.of(new ArrayList<>(new TreeSet<>(held.subList(0, h0))), new ArrayList<>(new TreeSet<>(held.subList(h0, held.size()))));
                    Map<String, List<List<String>>> trainPool = new TreeMap<>();
                    for (String p : train) trainPool.put(p, s.pool.get(p));
                    List<List<String>> extra = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), trainPool, Set.of(), Map.of()).extra();
                    Pca[] pcas = null;
                    for (int h = 0; h < 2; h++) {
                        List<String> own = hv.get(h);
                        List<String> ctrl = hv.get(1 - h);
                        XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                        List<List<String>> aug = new ArrayList<>(s.cohort);
                        aug.addAll(extra);
                        g.cohort = aug;
                        Map<String, List<String>> gal = new LinkedHashMap<>();
                        List<Probe> probes = new ArrayList<>();
                        for (int i = 0; i < own.size(); i++) {
                            List<List<String>> ds = s.pool.get(own.get(i));
                            gal.put(own.get(i), ds.get(0));
                            for (int d = 1; d < ds.size(); d++) for (String k : ds.get(d)) probes.add(new Probe(FeiScores.OWN, own.get(i), g.img(k), k, i));
                        }
                        int nOwn = probes.size();
                        for (String p : ctrl) for (List<String> d : s.pool.get(p)) for (String k : d) probes.add(new Probe(FeiScores.CTRL, p, g.img(k), k, -1));
                        String tag = String.format(Locale.ROOT, "%s_q%d_h%d_f%d", set, q, h, f);
                        Lines ls = read(cp, tag);
                        if (ls == null) {
                            long t0 = System.nanoTime();
                            if (pcas == null) {
                                List<String> pk = keys(s.pool, train);
                                pcas = new Pca[] {pca(s.xs[f], pk, false), pca(s.xs[f], pk, true)};
                            }
                            ls = fitLines(g, s.xs[f], gal, probes, null, 2 * q + h, pcas);
                            double sec = (System.nanoTime() - t0) / 1e9;
                            fitS += sec;
                            save(cp, tag, ls);
                            System.out.printf(Locale.ROOT, "pool-rule: %s ready (%.0f s)%n", tag, sec);
                        }
                        if (f == 0) vol.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d | %d | %d | %d%n", set, q, h, own.size(), nOwn, ctrl.size(),
                                probes.size() - nOwn, train.size()));
                        info.append(infoRow(String.format(Locale.ROOT, "%s %s часть %d смена %d", set, FEATS[f], q, h), ls.info()));
                        List<List<String>> noParts = new ArrayList<>();
                        for (int p = 0; p < FeiMethods.CV_PARTS; p++) noParts.add(List.of());
                        for (int v = 0; v < RVS.size(); v++) {
                            scores.append("# ").append(set).append(", ").append(FEATS[f]).append(", часть ").append(q).append(", смена ").append(h).append(", ")
                                    .append(RVS.get(v).label()).append('\n').append(ls.byRv().get(v));
                            XBaseEval.Parsed ps = XBaseEval.parse(g, EnumSet.of(V.L1), gal, probes, noParts, ls.byRv().get(v));
                            cfgs.get(v).add(ps.cfg.get(V.L1));
                        }
                    }
                }
                double[] crit = new double[RVS.size()];
                for (int v = 0; v < RVS.size(); v++) {
                    FeiDecompose.Cfg m = merge(cfgs.get(v));
                    Sep sp = sep(List.of(m), FeiBootstrap.unit(all.size()), new int[0][]);
                    crit[v] = sp.crit();
                    tb.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %s | %s | %.4f | %.4f%n", RVS.get(v).label(), FEATS[f], fmt(sp.err[0]),
                            fmt(sp.err[1]), fmt(sp.err[2]), fmt(crit[v]), fmt(sp.eer[3]), sp.medOwn, sp.medCtrl));
                }
                int c = choose(crit);
                choice.append(String.format(Locale.ROOT, "Выбор %s %s: %s (критерий %.1f %%; база %.1f %%; лучший %.1f %%)%n", set, FEATS[f], RVS.get(c).label(),
                        crit[c], crit[0], java.util.Arrays.stream(crit).min().orElseThrow()));
                sel.append(set).append('\t').append(FEATS[f]).append('\t').append(RVS.get(c).label()).append('\n');
            }
            tb.append('\n').append(choice);
            body.append(tb);
            if (set.equals("GT")) gtBody.append(tb);
        }
        StringBuilder t = new StringBuilder(head).append(vol).append(info)
                .append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads)).append(body);
        Files.writeString(s.out.resolve(SELECT), sel.toString());
        XBaseOther.writeScores(s.out.resolve("xbase_pool_rule_scores.tsv.gz"), "# Выбор правила оценки по пулу FERET (XBaseRule pool-rule), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", scores.toString());
        FarMethods.write(s.out.resolve("xbase_pool_rule.txt"), t);
        // Для GT – тот же выбор без имён снимков и без описи частей (копия в SVD-faces-data допустима).
        StringBuilder g = new StringBuilder(head).append(gtBody);
        Files.createDirectories(gtOut);
        FarMethods.write(gtOut.resolve("xbase_pool_rule_gt.txt"), g);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- feret-rule

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        FSetup s = fsetup(export, exportDir, feiKeys, selLines, settings, true);
        XBaseEval.Gt tpl = s.tpl;
        Map<String, Map<String, Integer>> select = readSelect(s.out.resolve(SELECT));
        XBaseFeret.Roles ro = XBaseFeret.equalize(s.su.ro, s.su.byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        Path xs2 = s.out.resolve("xbase_feret_xs2_scores.tsv.gz");
        Map<String, String> xs2b = XBaseSubsp.blocks(xs2);
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-rule %s; seed %d; r0 %s; пул %d; половины %d/%d", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_rule_eq"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 23.10, пп. 1.2, 2.1: правило оценки, FERET межсессионно (XBaseEval feret-rule): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("Роль «межс.» feret-xs2: половины №2, проход 1 – свои A, пул проекции – B, проход 2 – наоборот; контрольные 200; порог П3 – по «чужим»\n"
                + "галереи своих прохода (5 частей, seed faces.seed + 8000 + 100), свой на проход; итог – объединение проходов.\n");
        t.append(MODEL_NOTE);
        t.append("Сетка строк на своих – диагностика, по ней ничего не выбирается; «выбор по пулу» – строка, выбранная pool-rule по пулу прохода\n"
                + "(проход 1 – набор FERET-1, половина B; проход 2 – FERET-2, половина A), прогон на своих – один.\n");
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для обоих проходов и всех строк: B = %d, seed = %d.%n",
                persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        StringBuilder check = new StringBuilder("\n=== Проверка (0) = строка feret-xs2 (строки L1 блока «межс.», побайтно) ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD);
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (файл " + SELECT + ") ===\n");
        StringBuilder lines = new StringBuilder();
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        double fitS = 0;
        for (int f = 0; f < FEATS.length; f++) {
            Map<Integer, List<XBaseSubsp.Pass>> passes = new TreeMap<>();
            for (int v = 0; v <= RVS.size(); v++) passes.put(v, new ArrayList<>());
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own = ro.own1.stream().filter(mine::contains).toList();
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    gal.put(p, ro.gal1.get(p));
                    pro.put(p, ro.pro1.get(p));
                }
                Map<String, List<List<String>>> projPool = new TreeMap<>();
                for (String p : s.halves.get(1 - ps)) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S);
                List<Probe> probes = XBaseEval.probes(g, pro);
                String tag = String.format(Locale.ROOT, "p%d_f%d", ps, f);
                Lines ls = read(cp, tag);
                if (ls == null) {
                    long t0 = System.nanoTime();
                    List<String> pk = keys(s.pool, s.halves.get(1 - ps));
                    Pca[] pcas = {pca(s.xs[f], pk, false), pca(s.xs[f], pk, true)};
                    ls = fitLines(g, s.xs[f], gal, probes, parts, 0, pcas);
                    double sec = (System.nanoTime() - t0) / 1e9;
                    fitS += sec;
                    save(cp, tag, ls);
                    System.out.printf(Locale.ROOT, "feret-rule: pass %d, %s ready (%.0f s)%n", ps + 1, FEATS[f], sec);
                }
                String bn = "проход " + (ps + 1) + ", конфигурация межс., строка " + FEATS[f];
                String old = xs2b.get(bn);
                if (old == null) throw new IllegalStateException("feret-rule: нет блока «" + bn + "» в " + xs2);
                boolean same = XBaseSubsp.l1Rows(old).equals(XBaseSubsp.l1Rows(ls.byRv().get(0)));
                check.append(String.format(Locale.ROOT, "%s проход %d: %s (%d строк)%n", FEATS[f], ps + 1, same ? "совпало" : "НЕ СОВПАЛО", XBaseSubsp.l1Rows(old).size()));
                if (!same) throw new IllegalStateException("feret-rule: (0) не совпала с feret-xs2: " + bn);
                info.append(infoRow(String.format(Locale.ROOT, "межс. %s проход %d", FEATS[f], ps + 1), ls.info()));
                int c = select.get("FERET-" + (ps + 1)).get(FEATS[f]);
                sel.append(String.format(Locale.ROOT, "%s проход %d: %s%n", FEATS[f], ps + 1, RVS.get(c).label()));
                for (int v = 0; v < RVS.size(); v++) {
                    lines.append("# проход ").append(ps + 1).append(", ").append(FEATS[f]).append(", ").append(RVS.get(v).label()).append('\n')
                            .append(ls.byRv().get(v));
                    passes.get(v).add(new XBaseSubsp.Pass(g, gal, pro, parts, ls.byRv().get(v)));
                }
                passes.get(RVS.size()).add(new XBaseSubsp.Pass(g, gal, pro, parts, ls.byRv().get(c)));
            }
            for (int v = 0; v <= RVS.size(); v++) {
                String l = "межс. " + FEATS[f] + " " + (v < RVS.size() ? RVS.get(v).label() : CHOICE);
                res.put(l, XBaseSubsp.result(l, passes.get(v), persons.size(), tpl, new int[2]));
                if (v > 0) pairs.add(new String[] {l, "межс. " + FEATS[f] + " " + BASE});
            }
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));
        tables(t, "FERET межсессионно (роль «межс.» feret-xs2, оба прохода)", res, persons.size(), tpl, pairs);
        XBaseOther.writeScores(s.out.resolve("xbase_feret_rule_scores.tsv.gz"), "# FERET, правило оценки (XBaseRule feret-rule), только локально\n# Код: коммит "
                + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_rule.txt"), t);
        System.out.print(t);
        cp.delete();
    }

    // ---------------------------------------------------------------- gt-rule

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-rule: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, Map<String, Integer>> select = readSelect(su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)).resolve(SELECT));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        XBaseEval[] xs = {g.x, new XBaseEval()};
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        xs[1].vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        Map<String, Mat> pf = new LinkedHashMap<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) pf.put(k, FeretData.frame(su.dir, "a", k));
        XBaseEval p0 = new XBaseEval();
        p0.vectors(pf, EnumSet.of(V.L1), g.params);
        XBaseEval p1 = new XBaseEval();
        p1.vectors(pf, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : pf.values()) m.release();
        xs[0].vec.get(V.L1).putAll(p0.vec.get(V.L1));
        xs[0].mir.get(V.L1).putAll(p0.mir.get(V.L1));
        xs[1].vec.get(V.L1).putAll(p1.vec.get(V.L1));
        xs[1].mir.get(V.L1).putAll(p1.mir.get(V.L1));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        List<String> poolKeys = keys(pool, pool.keySet());

        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g.own, g.galleryAll, g.seed);
        Map<String, List<String>> galI = XBaseEval.equalize(g.own, split.get(0), eqSeed);
        Object[][] modes = {{"межс.", galX, g.probeKeys}, {"внутри", galI, split.get(1)}};
        List<List<String>> parts = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        Path subsp = out.resolve("xbase_gt_subsp_scores.tsv.gz");
        if (!Files.exists(subsp)) throw new IllegalStateException("gt-rule: для сверки (0) нужен " + subsp + " (gt-subsp)");
        Map<String, String> old = XBaseSubsp.blocks(subsp);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-rule; seed %d; r0 %s; пул FERET %d", g.seed, g.r0, pool.size()));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_rule"), fp);
        StringBuilder t = new StringBuilder();
        t.append("Ревизия 23.10, пп. 1.2, 2.1: правило оценки, GT (XBaseEval gt-rule)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt; когорта – a/fei); пул L1-xs-F и "
                + "PCA (г) – Color FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет)\n");
        t.append("Протокол gt; галерея по правилу №8 (seed faces.seed + 9800): межсессионно N = 6, внутри сессии N = 3 (контроль «внутри сессии не "
                + "хуже», Δ = 3 п. п.); проекция – как gt-subsp k = 0.\n");
        t.append(MODEL_NOTE);
        t.append("Сетка строк на своих – диагностика, по ней ничего не выбирается; «выбор по пулу» – строка, выбранная pool-rule (набор GT – весь пул\n"
                + "FERET), прогон на своих – один.\n");
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET,
                g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверка (0) = gt-subsp k = 0 (строки L1 блока, побайтно) ===\n");
        StringBuilder info = new StringBuilder(INFO_HEAD);
        StringBuilder sel = new StringBuilder("\n=== Выбор по пулу (набор GT) ===\n");
        StringBuilder lines = new StringBuilder();
        StringBuilder body = new StringBuilder();
        double fitS = 0;
        List<List<String>> cohort0 = g.cohort;
        Pca[][] pcas = new Pca[FEATS.length][];
        try {
            for (Object[] md : modes) {
                String mode = (String) md[0];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
                @SuppressWarnings("unchecked")
                Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
                List<Probe> probes = XBaseEval.probes(g, pro);
                Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
                List<String[]> pairs = new ArrayList<>();
                for (int f = 0; f < FEATS.length; f++) {
                    List<List<String>> aug = new ArrayList<>(cohort0);
                    aug.addAll(poolCls);
                    g.cohort = aug;
                    String tag = (mode.equals("межс.") ? "x" : "i") + "_f" + f;
                    Lines ls = read(cp, tag);
                    if (ls == null) {
                        if (pcas[f] == null) pcas[f] = new Pca[] {pca(xs[f], poolKeys, false), pca(xs[f], poolKeys, true)};
                        long t0 = System.nanoTime();
                        ls = fitLines(g, xs[f], gal, probes, parts, 0, pcas[f]);
                        double sec = (System.nanoTime() - t0) / 1e9;
                        fitS += sec;
                        save(cp, tag, ls);
                        System.out.printf(Locale.ROOT, "gt-rule: %s ready (%.0f s)%n", tag, sec);
                    }
                    String bn = mode + ", проекция " + FEATS[f] + ", k = 0";
                    String ob = old.get(bn);
                    if (ob == null) throw new IllegalStateException("gt-rule: нет блока «" + bn + "» в " + subsp);
                    boolean same = XBaseSubsp.l1Rows(ob).equals(XBaseSubsp.l1Rows(ls.byRv().get(0)));
                    check.append(String.format(Locale.ROOT, "%s %s: %s (%d строк)%n", mode, FEATS[f], same ? "совпало" : "НЕ СОВПАЛО", XBaseSubsp.l1Rows(ob).size()));
                    if (!same) throw new IllegalStateException("gt-rule: (0) не совпала с gt-subsp: " + bn);
                    info.append(infoRow(mode + " " + FEATS[f], ls.info()));
                    int c = select.get("GT").get(FEATS[f]);
                    if (mode.equals("межс.")) sel.append(String.format(Locale.ROOT, "%s: %s%n", FEATS[f], RVS.get(c).label()));
                    for (int v = 0; v <= RVS.size(); v++) {
                        String body0 = ls.byRv().get(v < RVS.size() ? v : c);
                        String l = mode + " " + FEATS[f] + " " + (v < RVS.size() ? RVS.get(v).label() : CHOICE);
                        if (v < RVS.size()) lines.append("# ").append(mode).append(", ").append(FEATS[f]).append(", ").append(RVS.get(v).label()).append('\n')
                                .append(body0);
                        res.put(l, XBaseSubsp.result(l, List.of(new XBaseSubsp.Pass(g, gal, pro, parts, body0)), g.persons.size(), g, new int[2]));
                        if (v > 0) pairs.add(new String[] {l, mode + " " + FEATS[f] + " " + BASE});
                    }
                }
                tables(body, "GT " + mode, res, g.persons.size(), g, pairs);
            }
        } finally {
            g.cohort = cohort0;
        }
        t.append(check).append(sel).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        t.append(body);
        XBaseOther.writeScores(out.resolve("xbase_gt_rule_scores.tsv.gz"), "# GT, правило оценки (XBaseRule gt-rule)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_rule.txt"), t);
        System.out.print(t);
        cp.delete();
    }
}
