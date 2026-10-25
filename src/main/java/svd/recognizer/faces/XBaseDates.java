package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
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
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.GalleryEvaluation.Variant;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Кривая «1 → 2 → 3 даты регистрации» (п. 2; задание №10 Жорика от 25.10.2026, план утверждён в №11, подтверждён Хозяином).
 * Симметричные галереи: у всех своих первые s дат, на каждой fa и fb по правилу eq, пробы – все отобранные снимки дат после
 * s-й; (i) по s – люди с ≥ s + 1 датами; (ii) парно – люди с ≥ 4 датами, пробы – даты с 4-й, s = 1, 2, 3 ((i) s = 3 = (ii)
 * s = 3). Шаблон (м) – центр класса, при s ≥ 2 ещё (с) – центр каждой сессии, оценка человека – наименьшая по его сессиям; модель
 * MLDA одна на постановку (обучение по людям), (м) и (с) отличаются только оценкой. Строки L1-xs-F (7×7) и L1-tt-xs-F (9×9,
 * решение (27)): MLDA с пулом FERET, пол λ̄ (c = 1), 1 − cos до центра по всем осям, зеркала и выбор модели по |r|; SFace –
 * справочно (минимум снимков шаблона 1, только разделимость). Порог П2/П3 – (Т2), рядом (Т0). FERET – половины №2 (как
 * feret-xs2); GT – (а) s = 1 (протокол gt-lbp «межс.»), (б) люди с ≥ 3 датами, s = 1, 2. Режимы XBaseEval feret-dates и
 * gt-dates.
 *
 * @author ssv
 */
final class XBaseDates {

    /** Признаки строк: L1-xs-F – база 7×7 (вариант 0 XBaseLbp), L1-tt-xs-F – 9×9 (вариант (2)). */
    static final LbpFeature.Params[] PARAMS = {XBaseLbp.VARS[0], XBaseLbp.VARS[2]};
    static final int[] VNO = {0, 2};
    static final IlluminationNorm[] NORMS = {IlluminationNorm.NONE, IlluminationNorm.TAN_TRIGGS};
    static final String[] TPL = {"(м)", "(с)"};
    static final String[] BINS = {"≤ 30 дн.", "31–365 дн.", "> 365 дн."};
    /** Номера конфигураций GT (б): части П2/П3 – seed faces.seed + 8000 + 100 + номер. */
    static final int GT_B1 = 74;
    static final int GT_B2 = 75;
    /** Выбор снимков даты j в GT (б): seed faces.seed + 9800 + GT_EQ + j. */
    static final int GT_EQ = 10;

    /** Пробный режим FACES_DATES_SMOKE=1: не больше SMOKE_N своих на проход, без сверок, файлы *_smoke. */
    static final int SMOKE_N = 12;
    static final int SMOKE_POOL = 15;

    private XBaseDates() {
    }

    static boolean smoke() {
        return "1".equals(System.getenv("FACES_DATES_SMOKE"));
    }

    static List<String> trim(List<String> own) {
        return smoke() && own.size() > SMOKE_N ? own.subList(0, SMOKE_N) : own;
    }

    static int bin(long days) {
        return days <= 30 ? 0 : days <= 365 ? 1 : 2;
    }

    /** Постановка: подпись, s, наименьшее число дат, первая дата проб (с 0), номер конфигурации (части – CV_S + conf). */
    record Post(String name, int s, int minDates, int from, int conf, String hubs) {}

    /** FERET: (i) s1, s2, s3 = (ii) s3, (ii) s1, s2; номера частей – №11 Жорика (s = 1 (i) – как feret-xs2). */
    static final List<Post> POSTS = List.of(new Post("(i) s1", 1, 2, 1, 0, "(i) s1"), new Post("(i) s2", 2, 3, 2, 70, "(i) s2"),
            new Post("(i)=(ii) s3", 3, 4, 3, 71, "(i) s3"), new Post("(ii) s1", 1, 4, 3, 72, "(ii) s1"), new Post("(ii) s2", 2, 4, 3, 73, "(ii) s2"));

    // ---------------------------------------------------------------- оценки

    /** Строки (с): центр каждой сессии в пространстве модели lm, оценка человека – наименьшее 1 − cos по его сессиям. */
    static String sessLines(XBaseEval x, String method, int s, LdaModel lm, boolean mirror, List<String> gallery, Map<String, List<List<String>>> sess,
                            List<Probe> probes) {
        List<List<double[]>> cls = new ArrayList<>();
        List<Integer> owner = new ArrayList<>();
        for (int i = 0; i < gallery.size(); i++) {
            for (List<String> d : sess.get(gallery.get(i))) {
                List<double[]> list = new ArrayList<>();
                for (String k : d) list.add(x.vec.get(V.L1).get(k));
                if (mirror) for (String k : d) list.add(x.mir.get(V.L1).get(k));
                cls.add(list);
                owner.add(i);
            }
        }
        double[][] c = LdaModel.of(cls, cls.size(), lm.mean(), lm.projection(), lm.info()).centers();
        StringBuilder out = new StringBuilder();
        for (Probe p : probes) {
            double[] y = lm.embed(x.vec.get(V.L1).get(p.key()));
            double[] sc = new double[gallery.size()];
            Arrays.fill(sc, Double.POSITIVE_INFINITY);
            for (int j = 0; j < c.length; j++) if (c[j] != null) sc[owner.get(j)] = Math.min(sc[owner.get(j)], XBaseRule.cos(y, c[j], y.length, null));
            out.append(FeiScores.line(method, s, 0, p.role(), p.person(), p.image(), p.trueIdx(), gallery, sc, false));
        }
        return out.toString();
    }

    /**
     * Задачи блока: как XBaseLbp.job (тот же порядок задач и модели), каждая задача – {строки (м) косинуса, строки (с) или ""}; sess == null –
     * без (с).
     */
    static XBaseLbp.Job job(XBaseEval.Gt gt, XBaseEval x, Map<String, List<String>> gal, Map<String, List<List<String>>> sess, List<Probe> main,
                            List<List<String>> parts, int s, List<List<Probe>> extra) {
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> fullGallery = own.stream().map(gal::get).toList();
        String[] info = new String[2];
        int[] axes = new int[2];
        List<Callable<String[]>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> {
                XBaseRule.Fit f = XBaseRule.fit(x, m, fullGallery, cohort, null);
                info[m ? 1 : 0] = f.lm.info();
                axes[m ? 1 : 0] = f.axes();
                return new String[] {XBaseReg.lines(x, method, s, f.lm, own, main)[1], sess == null ? "" : sessLines(x, method, s, f.lm, m, own, sess, main)};
            });
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(parts.get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(gal::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (extra != null) pr.addAll(extra.get(q));
                tasks.add(() -> {
                    LdaModel lm = XBaseRule.fit(x, m, gk, cohort, null).lm;
                    return new String[] {XBaseReg.lines(x, method, s, lm, g, pr)[1], sess == null ? "" : sessLines(x, method, s, lm, m, g, sess, pr)};
                });
            }
        }
        return new XBaseLbp.Job(tasks, info, axes);
    }

    /** Проход постановки: Gt, галерея, сессии галереи, пробы своих, части, пробы модели, пробы (Т1). */
    record Pass(XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<List<String>>> sess, Map<String, List<String>> pro, List<List<String>> parts,
                List<Probe> probes, List<List<Probe>> extra) {}

    /** Строка таблицы: подпись → проходы (строки оценок по проходам). */
    record Line(String label, List<Pass> passes, List<String> lines) {}

    static Map<String, XBaseThr.Out> outs(List<Line> rows, int persons, XBaseEval.Gt tpl) {
        Map<String, XBaseThr.Out> o = new LinkedHashMap<>();
        for (Line l : rows) {
            List<XBaseThr.Unit> us = new ArrayList<>();
            for (int i = 0; i < l.passes().size(); i++) {
                Pass p = l.passes().get(i);
                us.add(new XBaseThr.Unit(p.g(), p.gal(), p.pro(), p.parts(), l.lines().get(i), null, true));
            }
            o.put(l.label(), XBaseThr.evaluate(l.label(), us, persons, tpl));
        }
        return o;
    }

    /** SFace (справочно): разделимость при фактическом FAR контроля 1, 2, 5, 10 % и EER по проходам вместе. */
    static void sface(StringBuilder t, String title, List<Line> rows, int persons, XBaseEval.Gt tpl, List<String[]> pairs) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseRule.Sep> sep = new LinkedHashMap<>();
        for (Line l : rows) {
            List<FeiDecompose.Cfg> cs = new ArrayList<>();
            for (int i = 0; i < l.passes().size(); i++) {
                Pass p = l.passes().get(i);
                cs.add(XBaseEval.parse(p.g(), EnumSet.of(V.SFACE), p.gal(), p.probes(), p.parts(), l.lines().get(i)).cfg.get(V.SFACE));
            }
            sep.put(l.label(), XBaseRule.sep(cs, FeiBootstrap.unit(persons), w));
        }
        t.append("\n=== ").append(title).append(": SFace справочно (минимум снимков шаблона 1; d₁/d₂ по 1 − cos) – разделимость (ошибка своих при фактическом FAR "
                + "контроля; порог по контрольным чужим своей конфигурации) и EER ===\n");
        t.append("строка | разд. 1 % | 2 % | 5 % | 10 % | EER\n");
        for (Map.Entry<String, XBaseRule.Sep> e : sep.entrySet()) {
            XBaseRule.Sep s = e.getValue();
            List<String> row = new ArrayList<>(List.of(e.getKey()));
            for (int k = 0; k < XBaseRule.FARS.length; k++) row.add(XBaseSepLtp.cell(s.err[k], s.errRep[k]) + String.format(Locale.ROOT, " (%.2f %%)", s.far[k]));
            row.add(XBaseSepLtp.cell(s.eer[3], s.eerRep));
            t.append(String.join(" | ", row)).append('\n');
        }
        if (tpl.b > 0 && !pairs.isEmpty()) {
            t.append("пара | 1 % | 2 % | 5 % | 10 % | EER (п. п.; «*» – интервал [2,5; 97,5] не содержит 0)\n");
            for (String[] p : pairs) {
                XBaseRule.Sep a = sep.get(p[0]);
                XBaseRule.Sep z = sep.get(p[1]);
                List<String> row = new ArrayList<>(List.of(p[0] + " − " + p[1]));
                for (int k = 0; k < XBaseRule.FARS.length; k++) row.add(XBaseSepLtp.diff(a.err[k] - z.err[k], a.errRep[k], z.errRep[k]));
                row.add(XBaseSepLtp.diff(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                t.append(String.join(" | ", row)).append('\n');
            }
        }
    }

    /**
     * Ревизия 1.8: ошибка своих по интервалу «дата пробы − последняя дата шаблона» и по интервалу между первой и последней датой шаблона, при
     * П3 (Т2) FAR 1 % (порог прохода) и при разделимости 1 % (порог по контрольным чужим прохода); вес – одна проба.
     */
    static void bins(StringBuilder t, Line l, XBaseThr.Out o, XBaseEval.Gt tpl, Map<String, LocalDate> dateOf, Map<String, List<LocalDate>> dates, int s) {
        FeiThreshold.Res r = o.res.get(XBaseThr.WAYS[2]);
        int[][][] c = new int[2][BINS.length][3];
        List<Set<String>> people = new ArrayList<>();
        for (int i = 0; i < 2 * BINS.length; i++) people.add(new TreeSet<>());
        for (int i = 0; i < l.passes().size(); i++) {
            Pass p = l.passes().get(i);
            Map<String, Row> rows = XBaseThr.rows(l.lines().get(i));
            double th3 = r.theta[3][0][i];
            double thS = XBaseSepLtp.ctrlThreshold(XBaseOther.data(r.cfgs.get(i)), FeiBootstrap.unit(tpl.persons.size()), 0.01);
            for (Probe q : p.probes()) {
                if (!q.role().equals(FeiScores.OWN)) continue;
                Row rw = XBaseEval.choose(rows, V.L1, FeiScores.OWN, q.person(), q.image(), tpl.rOf.get(q.key()) <= tpl.r0);
                double sc = FeiScores.ratio(rw, null);
                boolean ok = rw.bestId().equals(rw.trueId());
                List<LocalDate> ds = dates.get(q.person());
                int[] b = {bin(ChronoUnit.DAYS.between(ds.get(s - 1), dateOf.get(q.key()))), bin(ChronoUnit.DAYS.between(ds.get(0), ds.get(s - 1)))};
                for (int v = 0; v < 2; v++) {
                    int[] x = c[v][b[v]];
                    x[0]++;
                    if (!(ok && sc <= th3)) x[1]++;
                    if (!(ok && sc <= thS)) x[2]++;
                    people.get(v * BINS.length + b[v]).add(q.person());
                }
            }
        }
        for (int v = 0; v < 2; v++) {
            for (int b = 0; b < BINS.length; b++) {
                int[] x = c[v][b];
                if (x[0] == 0 && v == 1) continue;
                t.append(String.format(Locale.ROOT, "%s | %s %s | %d | %d | %s | %s%n", l.label(), v == 0 ? "проба − последняя дата шаблона" : "первая − последняя дата "
                        + "шаблона", BINS[b], x[0], people.get(v * BINS.length + b).size(), pct(x[1], x[0]), pct(x[2], x[0])));
            }
        }
    }

    static String pct(int e, int n) {
        return n == 0 ? "–" : String.format(Locale.ROOT, "%.1f %% (%d)", 100.0 * e / n, e);
    }

    static final String BIN_HEAD = "\n=== Ревизия 1.8: ошибка своих по интервалам (вес – проба; у s = 1 интервал между датами шаблона – 0) ===\n"
            + "строка | интервал | проб | людей | ошибка своих П3 (Т2) FAR 1 % (ошибок) | при разделимости 1 % (ошибок)\n";

    static final String NOTE = "Галереи симметричные: у всех своих первые s дат (на каждой – fa и fb по правилу eq), пробы – все отобранные снимки дат после "
            + "s-й. Шаблон (м) – центр класса\nчеловека (среднее снимков всех s дат); (с) – центр каждой даты (сессии), оценка человека – наименьшее 1 − cos по "
            + "его сессиям, второй кандидат – ближайший\nдругой человек; модель MLDA одна на постановку (обучение по людям), (м) и (с) различаются только "
            + "оценкой. С ростом s растёт и число снимков\nшаблона (2, 4, 6 при s = 1, 2, 3): кривая отражает и число дат, и число снимков, разделить их здесь "
            + "нельзя.\nСтроки: L1-xs-F – LBP 7×7 (8, 2); L1-tt-xs-F – Tan – Triggs, LBP 9×9 (8, 2) (решение (27)); корень гистограмм, MLDA с пулом FERET, пол "
            + "λ̄ (c = 1),\n1 − cos до центра по всем осям (решение (26)), решение – d₁/d₂, модель X / X+mirror по |r| пробы. SFace – справочно (минимум "
            + "снимков шаблона 1).\n";

    // ---------------------------------------------------------------- feret-dates

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        XBaseRule.FSetup s = XBaseRule.fsetup(export, exportDir, feiKeys, selLines, settings, true);
        s.xs = null;
        XBaseEval.Gt tpl = s.tpl;
        Map<String, Img> byKey = s.su.byKey;
        XBaseFeret.Roles ro0 = s.su.ro;
        XBaseFeret.Roles ro = XBaseFeret.equalize(ro0, byKey);
        TreeSet<String> ps0 = new TreeSet<>(s.pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, s.cohort, s.su.rOf, s.su.imageNo);
        tpl.persons = persons;
        tpl.rOf = s.su.rOf;
        // Снимки своих по датам: все отобранные (пробы) и fa, fb по правилу eq (галерея – сессии пула).
        Map<String, List<List<String>>> kept = new TreeMap<>();
        Map<String, List<LocalDate>> dates = new TreeMap<>();
        Map<String, LocalDate> dateOf = new HashMap<>();
        for (String p : ro0.own1) {
            Map<String, List<String>> byDate = new TreeMap<>();
            List<String> all = new ArrayList<>(ro0.gal1.get(p));
            all.addAll(ro0.pro1.get(p));
            for (String k : all) byDate.computeIfAbsent(byKey.get(k).date(), d -> new ArrayList<>()).add(k);
            List<List<String>> ks = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (Map.Entry<String, List<String>> e : byDate.entrySet()) {
                ks.add(XBaseFeret.sorted(e.getValue()));
                LocalDate w = FeretData.date(e.getKey());
                ws.add(w);
                for (String k : e.getValue()) dateOf.put(k, w);
            }
            kept.put(p, ks);
            dates.put(p, ws);
            if (ks.size() != s.pool.get(p).size()) throw new IllegalStateException("feret-dates: у " + p + " число дат не совпало с пулом");
        }
        Path lbpScores = s.out.resolve("xbase_feret_lbp_scores.tsv.gz");
        Path lbpTxt = s.out.resolve("xbase_feret_lbp.txt");
        Path hubsScores = s.out.resolve("xbase_feret_hubs_scores.tsv.gz");
        for (Path f : List.of(lbpScores, lbpTxt, hubsScores)) if (!Files.exists(f)) throw new IllegalStateException("feret-dates: нужен " + f + " (feret-lbp, feret-hubs)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        Map<String, String> hubsBlocks = XBaseSubsp.blocks(hubsScores);
        StringBuilder posts = new StringBuilder();
        for (Post p : POSTS) posts.append(String.format(Locale.ROOT, "%s s%d ≥%d с%d №%d; ", p.name(), p.s(), p.minDates(), p.from(), p.conf()));
        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-dates %s; seed %d; r0 %s; пул %d; половины %d/%d; %s", s.su.dir, tpl.seed,
                tpl.r0, s.pool.size(), s.halves.get(0).size(), s.halves.get(1).size(), posts));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(s.su.dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_dates_eq" + (smoke() ? "_smoke" : "")), fp);

        // Векторы: L1 (7×7) и L1-tt (9×9) по кадрам «а»; SFace – по кадрам alignCrop.
        Map<String, Mat> fr = XBaseLbp.frames(s, export, true);
        XBaseEval[] xs = new XBaseEval[2];
        for (int f = 0; f < 2; f++) {
            xs[f] = new XBaseEval();
            xs[f].vectors(fr, tpl.params, NORMS[f], PARAMS[f]);
        }
        Set<String> sfKeys = new TreeSet<>();
        kept.values().forEach(ds -> ds.forEach(sfKeys::addAll));
        ro.ctrlKeys.values().forEach(sfKeys::addAll);
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String k : sfKeys) sf.put(k, FeretData.frame(s.su.dir, "sface", k));
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        Path model = java.nio.file.Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
        xsf.sfaceVectors(sf, FaceRecognizerSF.create(model.toString(), ""));
        for (Mat m : fr.values()) m.release();
        for (Mat m : sf.values()) m.release();

        StringBuilder t = new StringBuilder();
        t.append("П. 2: кривая «1 → 2 → 3 даты регистрации», FERET (XBaseEval feret-dates): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(s.su.dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append("; SFace – ").append(model.getFileName()).append(" (").append(FeiThreshold.sha(model)).append(")\n");
        t.append(XBaseFeret.EQ_NOTE);
        t.append(NOTE);
        t.append("Половины №2 (как feret-xs2): проход 1 – свои из половины A, пул проекции – половина B целиком, проход 2 – наоборот; порог – свой на проход, "
                + "итог – объединение;\nконтрольные – те же 200. Постановки: (i) s = 1 – люди с ≥ 2 датами (= роль «межс.» feret-xs2); (i) s = 2 – с ≥ 3 "
                + "датами; s = 3 – с ≥ 4\nдатами, пробы – даты с 4-й ((i) s = 3 = (ii) s = 3); (ii) s = 1, 2 – те же люди с ≥ 4 датами и те же пробы, что s = 3. "
                + "В (i) люди и пробы при\nразных s разные – сравнение между s только в (ii) (парные разности на общих пробах).\n");
        t.append("Части «чужих» П2/П3 – seed faces.seed + 8000 + 100 + номер: ");
        for (Post p : POSTS) t.append(p.name()).append(" – ").append(p.conf()).append("; ");
        t.append("номера 70–73 не заняты другими режимами (проверено по коду: XBaseHubs 0–24, прочие ниже 70, метод Прокаева 76–90).\n");
        t.append(XBaseThr.NOTE);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям пула и контроля (%d), одни веса для всех постановок, проходов, строк и способов: B = "
                + "%d, seed = %d.%n", persons.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));

        StringBuilder check = new StringBuilder("\n=== Проверки ===\n");
        StringBuilder vol = new StringBuilder("\n=== Объём по проходам ===\nпостановка | проход | своих | снимков шаблона (на человека) | проб своих | людей с пробами\n");
        StringBuilder info = new StringBuilder(XBaseLbp.INFO_HEAD).append("постановка, проход, признак | вариант | размерность | ранг оболочки | заменено | осей\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Line> rowsL1 = new LinkedHashMap<>();
        Map<String, Line> rowsSf = new LinkedHashMap<>();
        Map<String, Post> postOf = new HashMap<>();
        double fitS = 0;
        for (Post pt : POSTS) {
            List<Pass> passes = new ArrayList<>();
            for (int ps = 0; ps < 2; ps++) {
                Set<String> mine = s.halves.get(ps);
                List<String> own = trim(ro0.own1.stream().filter(p -> mine.contains(p) && kept.get(p).size() >= pt.minDates()).toList());
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<List<String>>> sess = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    List<List<String>> ss = new ArrayList<>(s.pool.get(p).subList(0, pt.s()));
                    sess.put(p, ss);
                    List<String> g = new ArrayList<>();
                    ss.forEach(g::addAll);
                    gal.put(p, XBaseFeret.sorted(g));
                    List<String> pr = new ArrayList<>();
                    for (int j = pt.from(); j < kept.get(p).size(); j++) pr.addAll(kept.get(p).get(j));
                    pro.put(p, XBaseFeret.sorted(pr));
                }
                Map<String, List<List<String>>> projPool = new TreeMap<>();
                for (String p : s.halves.get(1 - ps)) if (!smoke() || projPool.size() < SMOKE_POOL) projPool.put(p, s.pool.get(p));
                XBaseEval.Gt g = XBaseFeret.withOwn(base, own);
                List<List<String>> aug = new ArrayList<>(s.cohort);
                aug.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), projPool, Set.of(), Map.of()).extra());
                g.cohort = aug;
                List<List<String>> parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + pt.conf());
                List<Probe> probes = XBaseEval.probes(g, pro);
                passes.add(new Pass(g, gal, sess, pro, parts, probes, XBaseThr.extra(g, pro, parts)));
                int nPro = (int) probes.stream().filter(q -> q.role().equals(FeiScores.OWN)).count();
                vol.append(String.format(Locale.ROOT, "%s | %d | %d | %d (%s) | %d | %d%n", pt.name(), ps + 1, own.size(), XBaseFeret.sum(gal),
                        gal.values().stream().map(List::size).distinct().sorted().map(String::valueOf).reduce((a, b) -> a + ", " + b).orElse("–"), nPro,
                        pro.values().stream().filter(l -> !l.isEmpty()).count()));
                // Состав (i) s = 2 – с ролью G2 (equalize: g2, pro2), s = 1 – с ролью «межс.» (gal1, pro1): только сведения.
                if (pt.conf() == 70) check.append(composition("(i) s2 – роль G2 (feret-subsp, feret)", ps, own, gal, pro, ro0.own2, ro.g2, ro.pro2, mine));
                if (pt.conf() == 0) check.append(composition("(i) s1 – роль «межс.» (feret-xs2)", ps, own, gal, pro, ro0.own1, ro.gal1, ro0.pro1, mine));
            }
            // Состав проб своих и контроля – с feret-hubs, часть 1 (обе половины вместе): остановка при несовпадении.
            if (!smoke()) check.append(hubsCheck(pt, passes, hubsBlocks, hubsScores));

            // Блоки постановки: недостающие в контрольной точке – одной пачкой задач.
            XBaseLbp.One[][] got = new XBaseLbp.One[2][2];
            List<XBaseLbp.Job> jobs = new ArrayList<>();
            List<int[]> todo = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                for (int ps = 0; ps < 2; ps++) {
                    String tag = String.format(Locale.ROOT, "c%d_p%d_f%d", pt.conf(), ps, f);
                    got[f][ps] = XBaseLbp.read(cp, tag);
                    if (got[f][ps] != null) continue;
                    Pass p = passes.get(ps);
                    jobs.add(job(p.g(), xs[f], p.gal(), pt.s() >= 2 ? p.sess() : null, p.probes(), p.parts(), pt.conf(), p.extra()));
                    todo.add(new int[] {f, ps});
                }
            }
            String[][] sfl = new String[2][2];
            for (int ps = 0; ps < 2; ps++) for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) sfl[ps][m] = cp.read(String.format(Locale.ROOT, "c%d_p%d_sf%d", pt.conf(), ps, m));
            long t0 = System.nanoTime();
            if (!jobs.isEmpty()) {
                List<XBaseLbp.One> done = XBaseLbp.run(jobs, tpl.threads);
                for (int i = 0; i < todo.size(); i++) {
                    int[] fp2 = todo.get(i);
                    got[fp2[0]][fp2[1]] = done.get(i);
                    XBaseLbp.save(cp, String.format(Locale.ROOT, "c%d_p%d_f%d", pt.conf(), fp2[1], fp2[0]), done.get(i));
                }
            }
            for (int ps = 0; ps < 2; ps++) {
                Pass p = passes.get(ps);
                XBaseEval.Gt gs = XBaseFeret.withOwn(p.g(), p.g().own);
                gs.cohort = List.of();
                for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                    if (sfl[ps][m] != null) continue;
                    sfl[ps][m] = m == 0 ? XBaseEval.fitLines(gs, xsf, EnumSet.of(V.SFACE), p.gal(), p.probes(), p.parts(), pt.conf())
                            : XBaseFeret.fitLinesSessions(gs, xsf, EnumSet.of(V.SFACE), p.sess(), p.gal(), p.probes(), p.parts(), pt.conf());
                    cp.save(String.format(Locale.ROOT, "c%d_p%d_sf%d", pt.conf(), ps, m), sfl[ps][m]);
                }
            }
            double sec = (System.nanoTime() - t0) / 1e9;
            fitS += sec;
            System.out.printf(Locale.ROOT, "feret-dates: %s ready, %d blocks (%.0f s)%n", pt.name(), jobs.size(), sec);

            for (int f = 0; f < 2; f++) {
                String feat = XBaseRule.FEATS[f];
                for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                    String label = pt.name() + (pt.s() >= 2 ? " " + TPL[m] : "") + " " + feat;
                    List<String> ls = new ArrayList<>();
                    for (int ps = 0; ps < 2; ps++) {
                        String l = got[f][ps].rows()[m];
                        ls.add(l);
                        lines.append("# ").append(label).append(", проход ").append(ps + 1).append('\n').append(l);
                        if (pt.conf() == 0 && !smoke()) check.append(XBaseReg.check("проход " + (ps + 1) + ", " + feat + ", " + XBaseLbp.VNAME[VNO[f]] + ", косинус", l, lbpBlocks,
                                lbpScores));
                    }
                    rowsL1.put(label, new Line(label, passes, ls));
                    postOf.put(label, pt);
                }
                for (int ps = 0; ps < 2; ps++) info.append(XBaseLbp.infoRow(pt.name() + ", проход " + (ps + 1) + ", " + feat, VNO[f], got[f][ps]));
            }
            for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                String label = pt.name() + (pt.s() >= 2 ? " " + TPL[m] : "") + " SFace";
                List<String> ls = List.of(sfl[0][m], sfl[1][m]);
                for (int ps = 0; ps < 2; ps++) lines.append("# ").append(label).append(", проход ").append(ps + 1).append('\n').append(ls.get(ps));
                rowsSf.put(label, new Line(label, passes, ls));
            }
        }
        // Проверка порогов (i) s = 1 – ячейки (Т0), (Т2) П3 FAR 1 % отчёта feret-lbp.
        for (int f = 0; f < 2 && !smoke(); f++) {
            String feat = XBaseRule.FEATS[f];
            XBaseThr.Out o = outs(List.of(rowsL1.get("(i) s1 " + feat)), persons.size(), tpl).get("(i) s1 " + feat);
            String rl = "межс. " + feat + " " + XBaseLbp.VNAME[VNO[f]];
            for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), lbpReport, tpl.b, lbpTxt));
        }
        t.append(check).append(vol).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, tpl.threads));

        // Таблицы: (i) и (ii).
        StringBuilder binsT = new StringBuilder(BIN_HEAD);
        for (int view = 0; view < 2; view++) {
            List<String> names = view == 0 ? List.of("(i) s1", "(i) s2", "(i)=(ii) s3") : List.of("(ii) s1", "(ii) s2", "(i)=(ii) s3");
            List<Line> l1 = new ArrayList<>();
            List<Line> sfr = new ArrayList<>();
            List<String[]> pairs = new ArrayList<>();
            List<String[]> sfPairs = new ArrayList<>();
            for (String feat : new String[] {XBaseRule.FEATS[0], XBaseRule.FEATS[1], "SFace"}) {
                for (String n : names) {
                    for (String m : n.endsWith("s1") ? List.of("") : List.of(" (м)", " (с)")) {
                        String label = n + m + " " + feat;
                        (feat.equals("SFace") ? sfr : l1).add((feat.equals("SFace") ? rowsSf : rowsL1).get(label));
                    }
                }
                List<String[]> pl = feat.equals("SFace") ? sfPairs : pairs;
                String f = " " + feat;
                if (view == 0) {
                    pl.add(new String[] {"(i) s2 (с)" + f, "(i) s2 (м)" + f});
                    pl.add(new String[] {"(i)=(ii) s3 (с)" + f, "(i)=(ii) s3 (м)" + f});
                } else {
                    pl.add(new String[] {"(ii) s2 (м)" + f, "(ii) s1" + f});
                    pl.add(new String[] {"(i)=(ii) s3 (м)" + f, "(ii) s2 (м)" + f});
                    pl.add(new String[] {"(i)=(ii) s3 (м)" + f, "(ii) s1" + f});
                    pl.add(new String[] {"(ii) s2 (с)" + f, "(ii) s1" + f});
                    pl.add(new String[] {"(i)=(ii) s3 (с)" + f, "(ii) s2 (с)" + f});
                    pl.add(new String[] {"(ii) s2 (с)" + f, "(ii) s2 (м)" + f});
                    pl.add(new String[] {"(i)=(ii) s3 (с)" + f, "(i)=(ii) s3 (м)" + f});
                }
            }
            Map<String, XBaseThr.Out> o = outs(l1, persons.size(), tpl);
            if (view == 0) for (Line l : l1) bins(binsT, l, o.get(l.label()), tpl, dateOf, dates, postOf.get(l.label()).s());
            String title = view == 0 ? "FERET (i): по s – люди с ≥ s + 1 датами (разные люди и пробы при разных s)"
                    : "FERET (ii): s = 1, 2, 3 – одни и те же люди с ≥ 4 датами, пробы – даты с 4-й (общие пробы; выборка мала)";
            XBaseLbp.tables(t, title, o, persons.size(), tpl, pairs, true);
            sface(t, title, sfr, persons.size(), tpl, sfPairs);
        }
        t.append(binsT);
        String sfx = smoke() ? "_smoke" : "";
        XBaseOther.writeScores(s.out.resolve("xbase_feret_dates" + sfx + "_scores.tsv.gz"), "# FERET, кривая 1 → 2 → 3 даты (XBaseDates feret-dates), только локально\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(s.out.resolve("xbase_feret_dates" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    /** Сведения о составе прохода против роли: люди, ключи галереи, ключи проб (без остановки). */
    static String composition(String what, int ps, List<String> own, Map<String, List<String>> gal, Map<String, List<String>> pro, List<String> roleOwn,
                              Map<String, List<String>> roleGal, Map<String, List<String>> rolePro, Set<String> half) {
        Set<String> a = new TreeSet<>(own);
        Set<String> z = new TreeSet<>(roleOwn.stream().filter(half::contains).toList());
        int galSame = 0;
        int proSame = 0;
        for (String p : own) {
            if (gal.get(p).equals(roleGal.get(p))) galSame++;
            if (pro.get(p).equals(rolePro.get(p))) proSame++;
        }
        Set<String> onlyA = new TreeSet<>(a);
        onlyA.removeAll(z);
        Set<String> onlyZ = new TreeSet<>(z);
        onlyZ.removeAll(a);
        return String.format(Locale.ROOT, "Состав %s, проход %d: своих %d, у роли %d; только здесь %s, только у роли %s; галерея совпала у %d, пробы – у %d%n", what,
                ps + 1, a.size(), z.size(), onlyA, onlyZ, galSame, proSame);
    }

    /** Пробы своих и контрольные (человек, номер снимка) – как у feret-hubs, часть 1 (L1, (м), обе половины вместе). */
    static String hubsCheck(Post pt, List<Pass> passes, Map<String, String> hubs, Path src) {
        String bn = "часть 1 " + pt.hubs() + " (м), набор none";
        String old = hubs.get(bn);
        if (old == null) throw new IllegalStateException("feret-dates: нет блока «" + bn + "» в " + src);
        Set<String> z = new TreeSet<>();
        for (String l : old.lines().toList()) {
            Row r = FeiScores.parse(l);
            if (r.method().equals(V.L1.id) && (r.role().equals(FeiScores.OWN) || r.role().equals(FeiScores.CTRL))) z.add(r.role() + "\t" + r.person() + "\t" + r.image());
        }
        Set<String> a = new TreeSet<>();
        Set<String> ctrl = new TreeSet<>();
        for (Pass p : passes) {
            for (Probe q : p.probes()) {
                String k = q.role() + "\t" + q.person() + "\t" + q.image();
                if (q.role().equals(FeiScores.OWN)) a.add(k);
                else ctrl.add(k);
            }
        }
        if (passes.size() == 2) {
            Set<String> c2 = new TreeSet<>();
            for (Probe q : passes.get(1).probes()) if (q.role().equals(FeiScores.CTRL)) c2.add(q.role() + "\t" + q.person() + "\t" + q.image());
            Set<String> c1 = new TreeSet<>();
            for (Probe q : passes.get(0).probes()) if (q.role().equals(FeiScores.CTRL)) c1.add(q.role() + "\t" + q.person() + "\t" + q.image());
            if (!c1.equals(c2)) throw new IllegalStateException("feret-dates: контроль проходов различается");
        }
        a.addAll(ctrl);
        if (!a.equals(z)) throw new IllegalStateException("feret-dates: состав проб " + pt.name() + " не совпал с feret-hubs «" + bn + "»: здесь " + a.size()
                + ", там " + z.size());
        return String.format(Locale.ROOT, "%s: пробы своих и контрольные (человек, снимок) = feret-hubs «%s» – совпало (%d)%n", pt.name(), bn, z.size());
    }

    // ---------------------------------------------------------------- gt-dates

    /** Копия Gt с другим списком своих (люди бутстрепа, контроль, снимки – прежние). */
    static XBaseEval.Gt withOwn(XBaseEval.Gt b, List<String> own) {
        XBaseEval.Gt g = XBaseFeret.withOwn(b, own);
        g.samples = b.samples;
        return g;
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("gt-dates: только равные галереи FERET (faces.feret.gallery = eq)");
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        List<Sample> gtSamples = g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString())).toList();
        Map<String, Mat> fr = XBaseEval.frames(export, gtSamples, null);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) for (String k : d) fr.put(k, FeretData.frame(su.dir, "a", k));
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        if (smoke()) poolCls = poolCls.subList(0, Math.min(SMOKE_POOL, poolCls.size()));
        XBaseEval[] xs = new XBaseEval[2];
        for (int f = 0; f < 2; f++) {
            xs[f] = new XBaseEval();
            xs[f].vectors(fr, g.params, NORMS[f], PARAMS[f]);
        }
        for (Mat m : fr.values()) m.release();
        Path model = java.nio.file.Paths.get(settings.loadFacesModelsDir()).resolve(FaceAlignment.SFACE_FILE);
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (Sample s : gtSamples) {
            Mat c = export.frame(s, Variant.SFACE);
            if (c == null) throw new IllegalStateException("gt-dates: нет кадра SFace снимка " + s.file());
            sf.put(s.file().toString(), c);
        }
        XBaseEval xsf = new XBaseEval();
        xsf.sfaceMin = 1;
        xsf.sfaceVectors(sf, FaceRecognizerSF.create(model.toString(), ""));
        for (Mat m : sf.values()) m.release();

        // (а) s = 1: протокол gt-lbp «межс.».
        long eqSeed = g.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g.own, g.galleryAll, eqSeed);
        List<List<String>> partsA = FeiMethods.cvParts(g.own, g.seed, XBaseEval.CV_S);
        // (б): люди с ≥ 3 датами (после отбора), первые две даты – поровну снимков на дату, пробы – даты с 3-й.
        List<String> own3 = g.own.stream().filter(p -> dated(g, p).size() >= 3).toList();
        int n = Integer.MAX_VALUE;
        for (String p : own3) for (int j = 0; j < 2; j++) n = Math.min(n, g.byPerson.get(p).get(dated(g, p).get(j)).size());
        List<Map<String, List<String>>> pick = new ArrayList<>();
        for (int j = 0; j < 2; j++) {
            Random rnd = new Random(eqSeed + GT_EQ + j);
            Map<String, List<String>> m = new LinkedHashMap<>();
            for (String p : own3) {
                List<String> ks = new ArrayList<>(g.byPerson.get(p).get(dated(g, p).get(j)));
                ks.sort(null);
                java.util.Collections.shuffle(ks, rnd);
                List<String> sub = new ArrayList<>(ks.subList(0, n));
                sub.sort(null);
                m.put(p, sub);
            }
            pick.add(m);
        }
        Map<String, List<String>> pro3 = new LinkedHashMap<>();
        for (String p : own3) {
            List<String> pr = new ArrayList<>();
            List<Integer> ds = dated(g, p);
            for (int j = 2; j < ds.size(); j++) pr.addAll(g.byPerson.get(p).get(ds.get(j)));
            pr.sort(null);
            pro3.put(p, pr);
        }

        Path lbpScores = out.resolve("xbase_gt_lbp_scores.tsv.gz");
        Path lbpTxt = out.resolve("xbase_gt_lbp.txt");
        if (!Files.exists(lbpScores) || !Files.exists(lbpTxt)) throw new IllegalStateException("gt-dates: нужны " + lbpScores + " и " + lbpTxt + " (gt-lbp)");
        Map<String, String> lbpBlocks = XBaseSubsp.blocks(lbpScores);
        List<String> lbpReport = Files.readAllLines(lbpTxt, StandardCharsets.UTF_8);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-dates; seed %d; r0 %s; пул FERET %d; (б) %s n %d; части %d, %d", g.seed, g.r0,
                pool.size(), own3, n, GT_B1, GT_B2));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_dates" + (smoke() ? "_smoke" : "")), fp);

        List<List<String>> aug = new ArrayList<>(g.cohort);
        aug.addAll(poolCls);
        record GPost(String name, int s, int conf, XBaseEval.Gt g, Map<String, List<String>> gal, Map<String, List<List<String>>> sess, Map<String, List<String>> pro,
                     List<List<String>> parts) {}
        List<GPost> posts = new ArrayList<>();
        XBaseEval.Gt ga = withOwn(g, g.own);
        ga.cohort = aug;
        if (smoke()) {
            ga.own = trim(g.own);
            partsA = XBaseOther.cvParts(ga.own, g.seed, XBaseEval.CV_S);
        }
        posts.add(new GPost("(а) s1", 1, 0, ga, galX, null, g.probeKeys, partsA));
        for (int s = 1; s <= 2; s++) {
            XBaseEval.Gt gb = withOwn(g, own3);
            gb.cohort = aug;
            Map<String, List<String>> gal = new LinkedHashMap<>();
            Map<String, List<List<String>>> sess = new LinkedHashMap<>();
            for (String p : own3) {
                List<List<String>> ss = new ArrayList<>();
                for (int j = 0; j < s; j++) ss.add(pick.get(j).get(p));
                sess.put(p, ss);
                List<String> all = new ArrayList<>();
                ss.forEach(all::addAll);
                all.sort(null);
                gal.put(p, all);
            }
            int conf = s == 1 ? GT_B1 : GT_B2;
            posts.add(new GPost("(б) s" + s, s, conf, gb, gal, sess, pro3, XBaseOther.cvParts(own3, g.seed, XBaseEval.CV_S + conf)));
        }

        StringBuilder t = new StringBuilder();
        t.append("П. 2: кривая «1 → 2 → 3 даты регистрации», GT (XBaseEval gt-dates)\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, sface/gt; когорта – a/fei); пул L1-xs-F – "
                + "Color FERET, ").append(pool.size()).append(" человек (только обучение; имён снимков FERET в отчёте и оценках нет); SFace – ")
                .append(model.getFileName()).append(" (").append(FeiThreshold.sha(model)).append(")\n");
        t.append(NOTE);
        t.append(String.format(Locale.ROOT, "(а) s = 1 – протокол gt-lbp «межс.»: свои с ≥ 2 датами (%d), галерея – дата с наибольшим числом снимков, N = 6 по правилу "
                + "№8 (seed faces.seed + 9800), пробы – другие даты; части – seed faces.seed + 8000 + 100.%n", g.own.size()));
        t.append(String.format(Locale.ROOT, "(б) люди с ≥ 3 датами после отбора (%d: %s): s = 1 – первая дата, s = 2 – первые две, на каждой дате %d снимков "
                + "(наименьшее общее число; выбор – seed faces.seed + 9800 + %d + номер даты, шаблон s = 1 – часть шаблона s = 2), пробы – даты с 3-й; "
                + "части – seed faces.seed + 8000 + 100 + %d и %d.%nВЫБОРКА МАЛА: %d человек – П2/П3 по «чужим» галереи из %d человек не имеют смысла, "
                + "разделимость и пары – справочно, выводов не делать. Интервалы ревизии 1.8 – только FERET.%n", own3.size(), own3, n, GT_EQ, GT_B1, GT_B2, own3.size(),
                own3.size()));
        t.append("Контроль – односессионные люди GT; когорта – FEI 20 и пул FERET.\n");
        t.append(XBaseThr.NOTE);
        if (g.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям GT: B = %d, seed = %d, людей %d.%n", g.b, g.seed + FeiBootstrap.SEED_OFFSET, g.persons.size()));
        StringBuilder check = new StringBuilder("\n=== Проверки ===\n");
        StringBuilder vol = new StringBuilder("\n=== Объём ===\nпостановка | своих | снимков шаблона (на человека) | проб своих\n");
        StringBuilder info = new StringBuilder(XBaseLbp.INFO_HEAD).append("постановка, признак | вариант | размерность | ранг оболочки | заменено | осей\n");
        StringBuilder lines = new StringBuilder();
        Map<String, Line> rowsL1 = new LinkedHashMap<>();
        Map<String, Line> rowsSf = new LinkedHashMap<>();
        double fitS = 0;
        for (GPost pt : posts) {
            List<Probe> probes = XBaseEval.probes(pt.g(), pt.pro());
            Pass pass = new Pass(pt.g(), pt.gal(), pt.sess(), pt.pro(), pt.parts(), probes, XBaseThr.extra(pt.g(), pt.pro(), pt.parts()));
            vol.append(String.format(Locale.ROOT, "%s | %d | %d (%s) | %d%n", pt.name(), pt.g().own.size(), XBaseFeret.sum(pt.gal()), pt.gal().values().stream()
                    .map(List::size).distinct().sorted().map(String::valueOf).reduce((a, b) -> a + ", " + b).orElse("–"), probes.stream()
                    .filter(q -> q.role().equals(FeiScores.OWN)).count()));
            XBaseLbp.One[] got = new XBaseLbp.One[2];
            List<XBaseLbp.Job> jobs = new ArrayList<>();
            List<Integer> todo = new ArrayList<>();
            for (int f = 0; f < 2; f++) {
                got[f] = XBaseLbp.read(cp, String.format(Locale.ROOT, "c%d_f%d", pt.conf(), f));
                if (got[f] != null) continue;
                jobs.add(job(pt.g(), xs[f], pt.gal(), pt.s() >= 2 ? pt.sess() : null, probes, pt.parts(), pt.conf(), pass.extra()));
                todo.add(f);
            }
            long t0 = System.nanoTime();
            if (!jobs.isEmpty()) {
                List<XBaseLbp.One> done = XBaseLbp.run(jobs, g.threads);
                for (int i = 0; i < todo.size(); i++) {
                    got[todo.get(i)] = done.get(i);
                    XBaseLbp.save(cp, String.format(Locale.ROOT, "c%d_f%d", pt.conf(), todo.get(i)), done.get(i));
                }
            }
            String[] sfl = new String[2];
            XBaseEval.Gt gs = withOwn(pt.g(), pt.g().own);
            gs.cohort = List.of();
            for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                sfl[m] = cp.read(String.format(Locale.ROOT, "c%d_sf%d", pt.conf(), m));
                if (sfl[m] != null) continue;
                sfl[m] = m == 0 ? XBaseEval.fitLines(gs, xsf, EnumSet.of(V.SFACE), pt.gal(), probes, pt.parts(), pt.conf())
                        : XBaseFeret.fitLinesSessions(gs, xsf, EnumSet.of(V.SFACE), pt.sess(), pt.gal(), probes, pt.parts(), pt.conf());
                cp.save(String.format(Locale.ROOT, "c%d_sf%d", pt.conf(), m), sfl[m]);
            }
            double sec = (System.nanoTime() - t0) / 1e9;
            fitS += sec;
            System.out.printf(Locale.ROOT, "gt-dates: %s ready, %d blocks (%.0f s)%n", pt.name(), jobs.size(), sec);
            for (int f = 0; f < 2; f++) {
                String feat = XBaseRule.FEATS[f];
                for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                    String label = pt.name() + (pt.s() >= 2 ? " " + TPL[m] : "") + " " + feat;
                    String l = got[f].rows()[m];
                    lines.append("# ").append(label).append('\n').append(l);
                    if (pt.conf() == 0 && !smoke()) check.append(XBaseReg.check("межс., " + feat + ", " + XBaseLbp.VNAME[VNO[f]] + ", косинус", l, lbpBlocks, lbpScores));
                    rowsL1.put(label, new Line(label, List.of(pass), List.of(l)));
                }
                info.append(XBaseLbp.infoRow(pt.name() + ", " + feat, VNO[f], got[f]));
            }
            for (int m = 0; m < (pt.s() >= 2 ? 2 : 1); m++) {
                String label = pt.name() + (pt.s() >= 2 ? " " + TPL[m] : "") + " SFace";
                lines.append("# ").append(label).append('\n').append(sfl[m]);
                rowsSf.put(label, new Line(label, List.of(pass), List.of(sfl[m])));
            }
        }
        for (int f = 0; f < 2 && !smoke(); f++) {
            String feat = XBaseRule.FEATS[f];
            XBaseThr.Out o = outs(List.of(rowsL1.get("(а) s1 " + feat)), g.persons.size(), g).get("(а) s1 " + feat);
            String rl = "межс. " + feat + " " + XBaseLbp.VNAME[VNO[f]];
            for (int w : new int[] {0, 2}) check.append(XBaseReg.checkThr(rl, XBaseThr.WAYS[w], o.res.get(XBaseThr.WAYS[w]), lbpReport, g.b, lbpTxt));
        }
        t.append(check).append(vol).append(info);
        t.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с (без блоков из контрольной точки); потоков %d.%n", fitS, g.threads));
        for (int view = 0; view < 2; view++) {
            List<String> names = view == 0 ? List.of("(а) s1") : List.of("(б) s1", "(б) s2 (м)", "(б) s2 (с)");
            List<Line> l1 = new ArrayList<>();
            List<Line> sfr = new ArrayList<>();
            List<String[]> pairs = new ArrayList<>();
            List<String[]> sfPairs = new ArrayList<>();
            for (String feat : new String[] {XBaseRule.FEATS[0], XBaseRule.FEATS[1], "SFace"}) {
                boolean isSf = feat.equals("SFace");
                for (String nm : names) (isSf ? sfr : l1).add((isSf ? rowsSf : rowsL1).get(nm + " " + feat));
                if (view == 1) {
                    List<String[]> pl = isSf ? sfPairs : pairs;
                    pl.add(new String[] {"(б) s2 (м) " + feat, "(б) s1 " + feat});
                    pl.add(new String[] {"(б) s2 (с) " + feat, "(б) s1 " + feat});
                    pl.add(new String[] {"(б) s2 (с) " + feat, "(б) s2 (м) " + feat});
                }
            }
            String title = view == 0 ? "GT (а) s = 1 (= gt-lbp «межс.»)" : "GT (б) s = 1, 2 – " + own3.size() + " человека с ≥ 3 датами (СПРАВОЧНО, выборка мала)";
            XBaseLbp.tables(t, title, outs(l1, g.persons.size(), g), g.persons.size(), g, pairs, true);
            sface(t, title, sfr, g.persons.size(), g, sfPairs);
        }
        String sfx = smoke() ? "_smoke" : "";
        XBaseOther.writeScores(out.resolve("xbase_gt_dates" + sfx + "_scores.tsv.gz"), "# GT, кривая 1 → 2 → 3 даты (XBaseDates gt-dates)\n# Код: коммит " + FarMethods.commit()
                + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_dates" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    /** Номера дат человека GT по возрастанию (без снимков без даты). */
    static List<Integer> dated(XBaseEval.Gt g, String p) {
        return g.byPerson.get(p).keySet().stream().filter(d -> d > 0).sorted().toList();
    }
}
