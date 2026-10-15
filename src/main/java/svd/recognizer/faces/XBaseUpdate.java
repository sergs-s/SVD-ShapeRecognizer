package svd.recognizer.faces;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPOutputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Моделирование обновления шаблона, часть 1 (PLAN.md п. 5.5, очередь (4); задание Жорика №4 от 15.10.2026, план с поправками
 * №5): FERET, равные галереи, половины №2 (свои – половина пула людей с ≥ 2 датами, пул проекции – другая половина). Галерея –
 * все свои половины, старт – первые 2 даты; шаг j – (2 + j)-я дата каждого человека (пара fa/fb – один проход, решение по
 * лучшему d₁/d₂ из двух). Варианты: (а) статично; (б) окно M = 2; (в) M = 3; (г) без окна; (д) оракул M = 3 (обновление по
 * истинной метке); (в) + худший случай (принятый контрольный чужой – сессией в шаблон того, за кого принят). Обновление –
 * после всех проб шага: принятая дата (прошла порог приёма) – сессией в шаблон лучшего кандидата (верного или нет). MLDA
 * обучается один раз на старте (полная модель и 5 моделей частей «чужих» П2/П3) и дальше не меняется; на шагах
 * пересчитываются только центры классов по текущим шаблонам. Порог приёма – П3 (основной, решение (12)) или П2, заново на
 * каждом шаге. Меры – только по людям с ≥ 4 датами. Режим XBaseEval feret-update; отчёт – только D:\data\Feret\reports\eq\.
 *
 * @author ssv
 */
final class XBaseUpdate {

    private XBaseUpdate() {
    }

    /** Дат в стартовом шаблоне. */
    static final int START = 2;
    /** Наименьшее число дат у человека, по которому считаются меры. */
    static final int MIN_DATES = 4;
    /** Номер s частей «чужих» П2/П3 (seed faces.seed + 8000 + 100 + s; s 0–6 заняты прежними режимами FERET). */
    static final int PARTS_S = 7;
    /** Шаги, которые даются по отдельности; дальше – только в «шаги ≥ 2». */
    static final int TABLE_STEPS = 4;
    static final double FAR = 0.01;

    /** Вариант обновления: подпись, M, оракул, худший случай (контрольные чужие тоже обновляют). */
    enum Var {
        A("(а) статично", 0, false, false), B("(б) M = 2", 2, false, false), C("(в) M = 3", 3, false, false),
        D("(г) без окна", TemplateWindow.UNBOUNDED, false, false), E("(д) оракул M = 3", 3, true, false),
        /** Сверх задания: чистое старение (обновляются все свои, ширина у всех 2 сессии) – (е) − (а). */
        F("(е) оракул M = 2", 2, true, false), W("(в) + худший случай", 3, false, true);

        final String label;
        final int m;
        final boolean oracle;
        final boolean worst;

        Var(String label, int m, boolean oracle, boolean worst) {
            this.label = label;
            this.m = m;
            this.oracle = oracle;
            this.worst = worst;
        }
    }

    /** Порог приёма: П3 (индекс FeiThreshold 3) или П2 (2). */
    enum Gate {
        P3("П3", 3), P2("П2", 2);

        final String label;
        final int m;

        Gate(String label, int m) {
            this.label = label;
            this.m = m;
        }
    }

    /** Решение по проходу: лучший кандидат (индекс в галерее), второй, d₁/d₂, оценка истинного, анфас решающего снимка. */
    record Att(int best, double r, double d2, double dTrue, boolean front, String key) {}

    // ---------------------------------------------------------------- модель прохода

    /** Проход и проекция: векторы, галерея, даты, модели старта (полная и 5 частей, оба зеркала), кэши. */
    static final class Ctx {
        XBaseEval x;
        double r0;
        Map<String, Double> rOf;
        List<String> gal;
        Map<String, Integer> gi = new HashMap<>();
        /** Человек галереи → даты (пары ключей) по времени; дата → строка yymmdd. */
        Map<String, List<List<String>>> dates;
        Map<String, String> dateOf;
        LdaModel[] full = new LdaModel[2];
        LdaModel[][] part = new LdaModel[2][FeiMethods.CV_PARTS];
        List<List<String>> parts;
        int[] partOf;
        final Map<String, double[]> centerCache = new ConcurrentHashMap<>();
        final Map<String, double[]> embedCache = new ConcurrentHashMap<>();

        /** Центр класса модели lm (зеркало m) по ключам шаблона – как LdaModel.of (среднее векторов, затем проекция). */
        double[] center(String id, LdaModel lm, int m, List<String> keys) {
            String ck = id + "|" + String.join(",", keys);
            double[] c = centerCache.get(ck);
            if (c != null) return c;
            List<double[]> list = new ArrayList<>();
            for (String k : keys) list.add(x.vec.get(V.L1).get(k));
            if (m == 1) for (String k : keys) list.add(x.mir.get(V.L1).get(k));
            double[] mean = new double[lm.mean().length];
            for (double[] v : list) for (int i = 0; i < mean.length; i++) mean[i] += v[i];
            for (int i = 0; i < mean.length; i++) mean[i] /= list.size();
            c = LdaMath.project(lm.projection(), mean, lm.mean());
            centerCache.put(ck, c);
            return c;
        }

        double[] embed(String id, LdaModel lm, String key) {
            return embedCache.computeIfAbsent(id + "|" + key, z -> lm.embed(x.vec.get(V.L1).get(key)));
        }

        /** Решение по одному снимку: модель по позе; centers[m] – центры людей list (null – нет кандидата). */
        Att one(String[] ids, LdaModel[] lms, double[][][] centers, String key, int trueIdx) {
            boolean front = rOf.get(key) <= r0;
            int m = front ? 0 : 1;
            double[] y = embed(ids[m], lms[m], key);
            double[][] cs = centers[m];
            double[] d = new double[cs.length];
            for (int c = 0; c < d.length; c++) d[c] = cs[c] == null ? Double.POSITIVE_INFINITY : Math.sqrt(NearestVectorScorer.distance2(y, cs[c]));
            int b = 0;
            for (int p = 1; p < d.length; p++) if (d[p] < d[b]) b = p;
            int s = -1;
            for (int p = 0; p < d.length; p++) if (p != b && (s < 0 || d[p] < d[s])) s = p;
            double r = FeiScores.ratio(d[b], d[s], false, () -> key, null);
            return new Att(b, r, d[s], trueIdx < 0 ? Double.NaN : d[trueIdx], front, key);
        }

        /** Решение по проходу (пара снимков даты): лучший по d₁/d₂ (при равенстве – первый по ключу). */
        Att pair(String[] ids, LdaModel[] lms, double[][][] centers, List<String> keys, int trueIdx) {
            Att best = null;
            for (String k : keys) {
                Att a = one(ids, lms, centers, k, trueIdx);
                if (best == null || a.r() < best.r()) best = a;
            }
            return best;
        }
    }

    /** Центры людей persons по шаблонам (null – нет в шаблоне снимков) для модели lm, зеркало m. */
    static double[][] centers(Ctx c, String id, LdaModel lm, int m, List<String> persons, Map<String, TemplateWindow> tpl) {
        double[][] out = new double[persons.size()][];
        for (int i = 0; i < out.length; i++) {
            List<String> keys = tpl.get(persons.get(i)).keys();
            if (!keys.isEmpty()) out[i] = c.center(id, lm, m, keys);
        }
        return out;
    }

    // ---------------------------------------------------------------- шаг

    /** Итог шага одного прохода: конфигурация (свои – только люди с ≥ 4 датами), чужие П2/П3, сведения для таблиц. */
    static final class Step {
        FeiDecompose.Cfg cfg;
        FeiThreshold.Imp imp;
        /** Только шаг 1: чужие П2/П3 по общей W (полная модель, центры части исключены). */
        FeiThreshold.Imp impCommon;
        double th2;
        double th3;
        /** Контрольные: лучший кандидат каждого прохода. */
        List<String> ctrlBest = new ArrayList<>();
        int ctrlAccepted;
        /** Людей галереи; людей с шаблоном ≥ 3 сессий; лучших у контроля на них. */
        int galN;
        int wideN;
        int ctrlOnWide;
        /** Обновления шага: своих верных, своих неверных (дрейф), чужих (худший случай); вытеснено сессий. */
        int updOk;
        int updWrong;
        int updCtrl;
        /** Возраст шаблона у проб людей с ≥ 4 датами: дата пробы − самая старая дата шаблона, дней (сумма и число). */
        double ageSum;
        int ageN;
        /** Люди с ≥ 4 датами, у которых в шаблоне есть чужие сессии, и их ошибки при пороге приёма. */
        int foreignHolders;
        int foreignErr;
        /** Строки файла оценок шага. */
        StringBuilder lines = new StringBuilder();
    }

    static LocalDate day(String yymmdd) {
        return LocalDate.of(1900 + Integer.parseInt(yymmdd.substring(0, 2)), Integer.parseInt(yymmdd.substring(2, 4)), Integer.parseInt(yymmdd.substring(4, 6)));
    }

    /** Чужие П2/П3 по моделям частей (или по общей W при common) при текущих шаблонах: сессии людей части q против остальных. */
    static FeiThreshold.Imp impostors(Ctx c, Map<String, TemplateWindow> tpl, Map<String, Integer> pIdx, boolean common) {
        List<double[]> list = new ArrayList<>();
        int frontal = 0;
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
            Set<String> pq = new HashSet<>(c.parts.get(q));
            List<String> rest = c.gal.stream().filter(p -> !pq.contains(p)).toList();
            LdaModel[] lms = common ? c.full : new LdaModel[] {c.part[0][q], c.part[1][q]};
            String[] ids = common ? new String[] {"full0", "full1"} : new String[] {"part0_" + q, "part1_" + q};
            double[][][] cs = {centers(c, ids[0], lms[0], 0, rest, tpl), centers(c, ids[1], lms[1], 1, rest, tpl)};
            for (String p : c.parts.get(q)) {
                for (TemplateWindow.Session s : tpl.get(p).sessions()) {
                    Att a = c.pair(ids, lms, cs, s.keys(), -1);
                    if (a.front()) frontal++;
                    list.add(new double[] {pIdx.get(p), a.r()});
                }
            }
        }
        return FeiThreshold.Imp.of(list, frontal);
    }

    /** Контрольный проход: человек, дата, пара ключей. */
    record Ctrl(String person, String date, List<String> keys) {}

    /**
     * Моделирование одного варианта на одном проходе: шаги 1…maxStep; persons – индекс бутстрепа; measured – люди с ≥ 4
     * датами.
     */
    static List<Step> simulate(Ctx c, Var v, Gate gate, List<Ctrl> ctrl, Map<String, Integer> pIdx, Set<String> measured, int maxStep, String tag) {
        Map<String, TemplateWindow> tpl = new HashMap<>();
        for (String p : c.gal) {
            TemplateWindow w = new TemplateWindow(v == Var.A ? TemplateWindow.UNBOUNDED : v.m);
            for (int d = 0; d < START; d++) w.add(new TemplateWindow.Session(c.dateOf.get(c.dates.get(p).get(d).get(0)), p, c.dates.get(p).get(d)));
            tpl.put(p, w);
        }
        String[] ids = {"full0", "full1"};
        int[] unit = FeiBootstrap.unit(pIdx.size());
        List<Step> out = new ArrayList<>();
        for (int j = 1; j <= maxStep; j++) {
            Step st = new Step();
            st.imp = impostors(c, tpl, pIdx, false);
            if (j == 1) st.impCommon = impostors(c, tpl, pIdx, true);
            double[][][] cs = {centers(c, ids[0], c.full[0], 0, c.gal, tpl), centers(c, ids[1], c.full[1], 1, c.gal, tpl)};
            // Свои: проход – дата START + j − 1 (по порядку дат человека).
            List<String> probers = new ArrayList<>();
            List<Att> ownAtt = new ArrayList<>();
            for (String p : c.gal) {
                if (c.dates.get(p).size() < START + j) continue;
                probers.add(p);
                ownAtt.add(c.pair(ids, c.full, cs, c.dates.get(p).get(START + j - 1), c.gi.get(p)));
            }
            List<Att> ctrlAtt = new ArrayList<>();
            for (Ctrl k : ctrl) ctrlAtt.add(c.pair(ids, c.full, cs, k.keys(), -1));
            // Конфигурация (свои – только люди с ≥ 4 датами).
            FeiDecompose.Cfg cfg = new FeiDecompose.Cfg();
            List<Integer> mi = new ArrayList<>();
            for (int i = 0; i < probers.size(); i++) if (measured.contains(probers.get(i))) mi.add(i);
            int n = mi.size();
            cfg.ownP = new int[n];
            cfg.ownR = new double[n];
            cfg.ownOk = new boolean[n];
            cfg.noSecond = new boolean[n];
            cfg.trueInf = new boolean[n];
            cfg.poseFrontal = new boolean[n];
            for (int t = 0; t < n; t++) {
                Att a = ownAtt.get(mi.get(t));
                String p = probers.get(mi.get(t));
                cfg.ownP[t] = pIdx.get(p);
                cfg.ownR[t] = a.r();
                cfg.ownOk[t] = a.best() == c.gi.get(p);
                cfg.noSecond[t] = Double.isInfinite(a.d2());
                cfg.trueInf[t] = Double.isInfinite(a.dTrue());
                cfg.poseFrontal[t] = a.front();
            }
            cfg.thrP = new int[0];
            cfg.thrR = new double[0];
            cfg.ctrlP = ctrl.stream().mapToInt(k -> pIdx.get(k.person())).toArray();
            cfg.ctrlR = ctrlAtt.stream().mapToDouble(Att::r).toArray();
            cfg.j = j;
            st.cfg = cfg;
            st.th2 = FeiThreshold.threshold(2, cfg, st.imp, unit, FAR).theta();
            st.th3 = FeiThreshold.threshold(3, cfg, st.imp, unit, FAR).theta();
            double gateTh = gate == Gate.P3 ? st.th3 : st.th2;

            // Хабы и широкие шаблоны.
            st.galN = c.gal.size();
            Set<Integer> wide = new HashSet<>();
            for (String p : c.gal) if (tpl.get(p).size() >= 3) wide.add(c.gi.get(p));
            st.wideN = wide.size();
            for (Att a : ctrlAtt) {
                st.ctrlBest.add(c.gal.get(a.best()));
                if (wide.contains(a.best())) st.ctrlOnWide++;
                if (a.r() <= gateTh) st.ctrlAccepted++;
            }
            // Возраст шаблона и чужие сессии у измеряемых.
            for (int t = 0; t < n; t++) {
                String p = probers.get(mi.get(t));
                String pd = c.dateOf.get(c.dates.get(p).get(START + j - 1).get(0));
                st.ageSum += ChronoUnit.DAYS.between(day(tpl.get(p).oldestDate()), day(pd));
                st.ageN++;
                if (tpl.get(p).foreign(p) > 0) {
                    st.foreignHolders++;
                    Att a = ownAtt.get(mi.get(t));
                    if (!(a.best() == c.gi.get(p) && a.r() <= gateTh)) st.foreignErr++;
                }
            }
            // Строки файла оценок.
            for (int i = 0; i < probers.size(); i++) {
                Att a = ownAtt.get(i);
                String p = probers.get(i);
                st.lines.append(String.join("\t", tag, Integer.toString(j), FeiScores.OWN, p, measured.contains(p) ? "1" : "0", a.key(), c.gal.get(a.best()),
                        Double.toString(a.r()), a.front() ? "1" : "0")).append('\n');
            }
            for (int i = 0; i < ctrl.size(); i++) {
                Att a = ctrlAtt.get(i);
                st.lines.append(String.join("\t", tag, Integer.toString(j), FeiScores.CTRL, ctrl.get(i).person(), "-", a.key(), c.gal.get(a.best()),
                        Double.toString(a.r()), a.front() ? "1" : "0")).append('\n');
            }
            st.lines.append(String.format(Locale.ROOT, "%s\t%d\tthreshold\t-\t-\t-\t-\tП2 %s; П3 %s\t-%n", tag, j, Double.toString(st.th2), Double.toString(st.th3)));

            // Обновления – после всех проб шага.
            if (v != Var.A) {
                for (int i = 0; i < probers.size(); i++) {
                    String p = probers.get(i);
                    Att a = ownAtt.get(i);
                    List<String> keys = c.dates.get(p).get(START + j - 1);
                    TemplateWindow.Session s = new TemplateWindow.Session(c.dateOf.get(keys.get(0)), p, keys);
                    if (v.oracle) {
                        tpl.get(p).add(s);
                        st.updOk++;
                    } else if (a.r() <= gateTh) {
                        String to = c.gal.get(a.best());
                        tpl.get(to).add(s);
                        if (to.equals(p)) st.updOk++;
                        else st.updWrong++;
                    }
                }
                if (v.worst) {
                    for (int i = 0; i < ctrl.size(); i++) {
                        Att a = ctrlAtt.get(i);
                        if (a.r() > gateTh) continue;
                        tpl.get(c.gal.get(a.best())).add(new TemplateWindow.Session(ctrl.get(i).date(), ctrl.get(i).person(), ctrl.get(i).keys()));
                        st.updCtrl++;
                    }
                }
            }
            out.add(st);
        }
        return out;
    }

    // ---------------------------------------------------------------- проверка шага 1

    /**
     * Оценки шага 1 по центрам старта (путь моделирования: центры из шаблонов) совпадают с прямой моделью XBaseEval.fit (её
     * собственные центры по тем же классам), побайтно.
     */
    static int checkStart(Ctx c, List<Ctrl> ctrl) {
        Map<String, TemplateWindow> tpl = new HashMap<>();
        for (int i = 0; i < c.gal.size(); i++) {
            TemplateWindow w = new TemplateWindow(START);
            for (int d = 0; d < START; d++) w.add(new TemplateWindow.Session("", c.gal.get(i), c.dates.get(c.gal.get(i)).get(d)));
            tpl.put(c.gal.get(i), w);
        }
        int n = 0;
        for (int m = 0; m < 2; m++) {
            LdaModel direct = c.full[m];
            double[][] cs = centers(c, "check" + m, c.full[m], m, c.gal, tpl);
            List<String> keys = new ArrayList<>();
            for (String p : c.gal) if (c.dates.get(p).size() > START) keys.addAll(c.dates.get(p).get(START));
            for (Ctrl k : ctrl) keys.addAll(k.keys());
            for (String k : keys) {
                double[] a = direct.scores(c.x.vec.get(V.L1).get(k));
                double[] y = c.full[m].embed(c.x.vec.get(V.L1).get(k));
                for (int i = 0; i < a.length; i++) {
                    double b = Math.sqrt(NearestVectorScorer.distance2(y, cs[i]));
                    if (Double.doubleToLongBits(a[i]) != Double.doubleToLongBits(b)) {
                        throw new IllegalStateException("feret-update: оценка старта не совпала с прямой моделью: " + k + ", человек " + c.gal.get(i));
                    }
                }
                n++;
            }
        }
        return n;
    }

    // ---------------------------------------------------------------- запуск

    /** Проекция: подпись, признак (0 – L1, 1 – L1-tt), пул в обучении MLDA. */
    record Proj(String label, int f, boolean pool) {}

    static final List<Proj> PROJS = List.of(new Proj("L1-xs-F", 0, true), new Proj("L1-tt-xs-F", 1, true), new Proj("L1", 0, false));

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-update: только равные галереи (faces.feret.gallery = eq)");
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        XBaseFeret.Roles roAll = su.ro;
        Map<String, List<List<String>>> pool = XBaseXsF.pool(roAll, su.byKey);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }
        // Контроль: 200 людей с одной датой; проход – пара fa/fb даты по правилу eq (XBaseFeret.pairOf).
        List<Ctrl> ctrl = new ArrayList<>();
        int ctrlSingle = 0;
        for (String p : roAll.ctrl) {
            List<String> ks = XBaseFeret.sorted(XBaseFeret.pairOf(roAll.ctrlKeys.get(p), su.byKey, false));
            if (ks.size() == 1) ctrlSingle++;
            ctrl.add(new Ctrl(p, su.byKey.get(ks.get(0)).date(), ks));
        }
        Set<String> used = new TreeSet<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) used.addAll(d);
        for (Ctrl k : ctrl) used.addAll(k.keys());
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        System.out.printf(Locale.ROOT, "feret-update: векторы готовы (%.0f с)%n", (System.nanoTime() - t0) / 1e9);

        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        for (Ctrl k : ctrl) ps0.add(k.person());
        List<String> persons = new ArrayList<>(ps0);
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        Set<String> measured = new TreeSet<>();
        for (Map.Entry<String, List<List<String>>> e : pool.entrySet()) if (e.getValue().size() >= MIN_DATES) measured.add(e.getKey());
        Map<String, String> dateOf = new HashMap<>();
        for (String k : used) dateOf.put(k, su.byKey.get(k).date());

        StringBuilder t = new StringBuilder();
        t.append("PLAN.md п. 5.5, очередь (4): моделирование обновления шаблона, часть 1 (XBaseEval feret-update): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append("""
                Схема (задание №4 от 15.10.2026, поправки №5):
                - половины №2 (seed faces.seed + 9910): свои – половина пула (люди FERET с ≥ 2 датами, на дату – пара fa/fb по правилу eq), пул
                  проекции L1-xs-F / L1-tt-xs-F – другая половина целиком (пул не пересекается со своими); два прохода, итог – объединение;
                - галерея – все свои половины, старт – первые 2 даты (по времени); шаг j – (2 + j)-я дата каждого человека (по номеру даты
                  человека, не по календарю); обновления шага – после всех его проб; меры – только по людям с ≥ 4 датами, остальные – статичные
                  участники галереи и возможные цели дрейфа (их даты тоже подаются пробами и обновляют шаблоны по правилу варианта);
                - проход – пара снимков даты (fa, fb), решение – по лучшему (наименьшему) d₁/d₂ из двух, модель по позе снимка; принятая дата –
                  одна сессия из её снимков; контрольные чужие (200 людей с одной датой) – так же парой fa/fb своей даты по правилу eq, каждый шаг
                  все 200 – пробы на текущую галерею (один проход на человека на шаг: FAR по проходам = FAR по людям);
                - MLDA обучается ОДИН РАЗ на старте (свои по 2 датам + когорта FEI + пул) и дальше не меняется; при обновлении пересчитываются
                  только центры классов в пространстве MLDA (среднее векторов шаблона, затем проекция; у модели +mirror – с отражениями);
                - порог П2 / П3 (FAR 1 %) – заново на каждом шаге по текущей галерее: 5 моделей частей «чужих» (MLDA без людей части q,
                  seed faces.seed + 8000 + 100 + 7) обучены на старте и не меняются, в них пересчитываются только центры по текущим шаблонам;
                  «чужие» – сессии шаблонов людей части q (парой, решение по лучшему из двух) против остальных; справочно на шаге 1 – «общая W»
                  (полная модель, центры части исключены);
                - порог приёма (обновление шаблона) – П3 (основной, решение (12)) или П2 – два отдельных хода моделирования;
                - обновление: дата принята (d₁/d₂ ≤ порога приёма) – сессией в шаблон лучшего кандидата (верного или нет – дрейф как в ПО);
                  (а) статично; (б) окно M = 2; (в) M = 3; (г) без окна; (д) оракул M = 3 – по истинной метке, независимо от приёма;
                  (в) + худший случай – ещё и принятый контрольный чужой сессией в шаблон того, за кого принят; сверх задания – (е) оракул
                  M = 2: обновляются все свои по истинной метке, ширина у всех 2 сессии – (е) − (а) – мера старения без примеси отбора
                  (у (б) − (а) обновлены только принятые, состав обновлённых зависит от порога);
                - шаг 1 у всех вариантов одинаков (обновлений ещё не было) и дан один раз; разности вариантов – с шага 2;
                - бутстреп по людям – поверх точечной траектории обновлений (траектория в репликах не пересчитывается).
                """);
        if (tpl.b > 0) t.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d.%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        t.append(String.format(Locale.ROOT, "Контроль: %d людей, проходов из одного снимка (нет пары fa/fb) – %d.%n", ctrl.size(), ctrlSingle));

        // Опись.
        StringBuilder vol = new StringBuilder("\n=== Опись ===\nпроход | галерея (людей) | с ≥ 4 датами | число дат у них | проходов своих по шагам 1, 2, 3, …\n");
        int maxStep = 0;
        for (String p : measured) maxStep = Math.max(maxStep, pool.get(p).size() - START);
        for (int pp = 0; pp < 2; pp++) {
            List<String> gal = new ArrayList<>(halves.get(pp));
            Map<Integer, Integer> nd = new TreeMap<>();
            int[] perStep = new int[maxStep + 1];
            int mn = 0;
            for (String p : gal) {
                if (measured.contains(p)) {
                    mn++;
                    nd.merge(pool.get(p).size(), 1, Integer::sum);
                    for (int j = 1; j <= pool.get(p).size() - START; j++) perStep[j]++;
                }
            }
            StringBuilder ps = new StringBuilder();
            for (int j = 1; j <= maxStep; j++) ps.append(j > 1 ? ", " : "").append(perStep[j]);
            vol.append(String.format(Locale.ROOT, "%d | %d | %d | %s | %s%n", pp + 1, gal.size(), mn, nd, ps));
        }
        t.append(vol);

        StringBuilder scores = new StringBuilder();
        StringBuilder check = new StringBuilder("\n=== Проверка: оценки шага 1 по центрам старта = прямая модель XBaseEval.fit (побайтно) ===\n");
        StringBuilder body = new StringBuilder();
        double simS = 0;
        Map<String, Map<Gate, Map<Var, List<List<Step>>>>> keep = new LinkedHashMap<>();
        for (Proj pj : PROJS) {
            // [gate][var][pass] → шаги.
            Map<Gate, Map<Var, List<List<Step>>>> all = new LinkedHashMap<>();
            for (Gate g : Gate.values()) {
                Map<Var, List<List<Step>>> m = new LinkedHashMap<>();
                for (Var v : Var.values()) m.put(v, new ArrayList<>());
                all.put(g, m);
            }
            for (int pp = 0; pp < 2; pp++) {
                long ts = System.nanoTime();
                Ctx c = ctx(xs[pj.f()], tpl, su.rOf, halves.get(pp), halves.get(1 - pp), pool, pj.pool(), cohort, dateOf);
                check.append(String.format(Locale.ROOT, "%s проход %d: совпало %d оценок (оба зеркала)%n", pj.label(), pp + 1, checkStart(c, ctrl)));
                List<java.util.concurrent.Callable<List<Step>>> tasks = new ArrayList<>();
                for (Gate g : Gate.values()) {
                    for (Var v : Var.values()) {
                        String tag = pj.label() + "\t" + (pp + 1) + "\t" + g.label + "\t" + v.label;
                        int ms = maxStep;
                        tasks.add(() -> simulate(c, v, g, ctrl, pIdx, measured, ms, tag));
                    }
                }
                List<List<Step>> done = runAll(tasks, tpl.threads);
                int ti = 0;
                for (Gate g : Gate.values()) {
                    for (Var v : Var.values()) {
                        List<Step> steps = done.get(ti++);
                        all.get(g).get(v).add(steps);
                        for (Step s : steps) scores.append(s.lines);
                    }
                }
                double sec = (System.nanoTime() - ts) / 1e9;
                simS += sec;
                System.out.printf(Locale.ROOT, "feret-update: %s проход %d готов (%.0f с)%n", pj.label(), pp + 1, sec);
            }
            for (Gate g : Gate.values()) body.append(group(pj, g, all.get(g), persons.size(), tpl, maxStep));
            keep.put(pj.label(), all);
        }

        // К4 (каскад 0б для глаз): (а) и (в) на кадрах derived/a_k4_0b – если они есть для всех нужных снимков FERET.
        Path k4 = dir.resolve(FeretData.DERIVED).resolve(FeretData.K4_0B);
        List<String> missing = used.stream().filter(k -> !Files.exists(k4.resolve(k + ".png"))).toList();
        if (!missing.isEmpty()) {
            t.append(String.format(Locale.ROOT, "%nК4: нет кадров для %d из %d нужных снимков FERET (снимки без ручной разметки; добыть – режим feret-k4-add, "
                    + "несколько минут) – строки К4 не строились.%n", missing.size(), used.size()));
        } else {
            long tk = System.nanoTime();
            Map<String, Mat> fk = new LinkedHashMap<>();
            for (String k : used) fk.put(k, FeretData.frame(dir, FeretData.K4_0B, k));
            fk.putAll(XBaseEval.frames(export, fei, cohortKeys));
            XBaseEval[] xk = {new XBaseEval(), new XBaseEval()};
            xk[0].vectors(fk, EnumSet.of(V.L1), tpl.params);
            xk[1].vectors(fk, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
            for (Mat m : fk.values()) m.release();
            body.append("\n\n################ К4: кадры по точкам глаз 0б (каскад eye_tree_eyeglasses), нос и рот YuNet; (а) и (в); пары «К4 − кадр а» "
                    + "по (в) ################\n");
            for (Proj pj : PROJS) {
                Map<Gate, Map<Var, List<List<Step>>>> allK = new LinkedHashMap<>();
                for (Gate g : Gate.values()) {
                    Map<Var, List<List<Step>>> m = new LinkedHashMap<>();
                    m.put(Var.A, new ArrayList<>());
                    m.put(Var.C, new ArrayList<>());
                    allK.put(g, m);
                }
                Proj pk = new Proj("К4 " + pj.label(), pj.f(), pj.pool());
                for (int pp = 0; pp < 2; pp++) {
                    Ctx c = ctx(xk[pj.f()], tpl, su.rOf, halves.get(pp), halves.get(1 - pp), pool, pj.pool(), cohort, dateOf);
                    for (Gate g : Gate.values()) {
                        for (Var v : List.of(Var.A, Var.C)) {
                            List<Step> steps = simulate(c, v, g, ctrl, pIdx, measured, maxStep, pk.label() + "\t" + (pp + 1) + "\t" + g.label + "\t" + v.label);
                            allK.get(g).get(v).add(steps);
                            for (Step s : steps) scores.append(s.lines);
                        }
                    }
                }
                for (Gate g : Gate.values()) {
                    body.append(group(pk, g, allK.get(g), persons.size(), tpl, maxStep, keep.get(pj.label()).get(g).get(Var.C), "(кадр а)"));
                }
            }
            simS += (System.nanoTime() - tk) / 1e9;
            System.out.printf(Locale.ROOT, "feret-update: К4 готово (%.0f с)%n", (System.nanoTime() - tk) / 1e9);
        }
        t.append(check);
        t.append(String.format(Locale.ROOT, "Обучение и моделирование: %.0f с; всего %.0f с; потоков %d.%n", simS, (System.nanoTime() - t0) / 1e9, tpl.threads));
        t.append(body);
        Path sf = out.resolve("xbase_feret_update_scores.tsv.gz");
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(sf), 1 << 16), StandardCharsets.UTF_8)) {
            w.write("# FERET, моделирование обновления шаблона (XBaseUpdate feret-update), только локально\n# Код: коммит " + FarMethods.commit() + "\n");
            w.write("# Проход своего (роль own) или контрольного (ctrl): решающий снимок пары, лучший кандидат, d₁/d₂, анфас; строка threshold – пороги шага\n");
            w.write("proj\tpass\tgate\tvariant\tstep\trole\tperson\tmeasured\tkey\tbest\tratio\tfrontal\n");
            w.write(scores.toString());
        }
        FarMethods.write(out.resolve("xbase_feret_update.txt"), t);
        System.out.print(t);
    }

    /** Задачи в threads потоках (только чистая Java), результаты – в порядке задач. */
    static <T> List<T> runAll(List<java.util.concurrent.Callable<T>> tasks, int threads) {
        java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<T> out = new ArrayList<>();
        try {
            for (java.util.concurrent.Future<T> f : ex.invokeAll(tasks)) out.add(f.get());
        } catch (InterruptedException | java.util.concurrent.ExecutionException e) {
            throw new IllegalStateException("XBaseUpdate: " + e, e);
        } finally {
            ex.shutdown();
        }
        return out;
    }

    static List<List<String>> aug(List<List<String>> cohort, Set<String> poolHalf, Map<String, List<List<String>>> pool, boolean usePool) {
        List<List<String>> a = new ArrayList<>(cohort);
        if (usePool) {
            Map<String, List<List<String>>> pp = new TreeMap<>();
            for (String p : poolHalf) pp.put(p, pool.get(p));
            a.addAll(XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pp, Set.of(), Map.of()).extra());
        }
        return a;
    }

    /** Контекст прохода: галерея own, старт – 2 даты, модели старта (полная и части) обоих зеркал. */
    static Ctx ctx(XBaseEval x, XBaseEval.Gt tpl, Map<String, Double> rOf, Set<String> own, Set<String> poolHalf, Map<String, List<List<String>>> pool,
                   boolean usePool, List<List<String>> cohort, Map<String, String> dateOf) {
        Ctx c = new Ctx();
        c.x = x;
        c.r0 = tpl.r0;
        c.rOf = rOf;
        c.gal = new ArrayList<>(own);
        for (int i = 0; i < c.gal.size(); i++) c.gi.put(c.gal.get(i), i);
        c.dates = new HashMap<>();
        for (String p : c.gal) c.dates.put(p, pool.get(p));
        c.dateOf = dateOf;
        List<List<String>> a = aug(cohort, poolHalf, pool, usePool);
        Map<String, List<String>> start = new LinkedHashMap<>();
        for (String p : c.gal) {
            List<String> k = new ArrayList<>();
            for (int d = 0; d < START; d++) k.addAll(pool.get(p).get(d));
            start.put(p, k);
        }
        c.parts = XBaseOther.cvParts(c.gal, tpl.seed, XBaseEval.CV_S + PARTS_S);
        List<java.util.concurrent.Callable<Void>> tasks = new ArrayList<>();
        for (int m = 0; m < 2; m++) {
            final int mm = m;
            boolean mir = m == 1;
            tasks.add(() -> {
                c.full[mm] = (LdaModel) x.fit(V.L1, mir, c.gal.stream().map(start::get).toList(), a);
                return null;
            });
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                final int qq = q;
                Set<String> pq = new HashSet<>(c.parts.get(q));
                tasks.add(() -> {
                    c.part[mm][qq] = (LdaModel) x.fit(V.L1, mir, c.gal.stream().filter(p -> !pq.contains(p)).map(start::get).toList(), a);
                    return null;
                });
            }
        }
        runAll(tasks, tpl.threads);
        return c;
    }

    // ---------------------------------------------------------------- таблицы

    static List<FeiDecompose.Cfg> cfgs(List<List<Step>> passes, int from, int to) {
        List<FeiDecompose.Cfg> out = new ArrayList<>();
        for (List<Step> ps : passes) for (Step s : ps) if (s.cfg.j >= from && s.cfg.j <= to && s.cfg.ownR.length > 0) out.add(s.cfg);
        return out;
    }

    static List<FeiThreshold.Imp> imps(List<List<Step>> passes, int from, int to, boolean common) {
        List<FeiThreshold.Imp> out = new ArrayList<>();
        for (List<Step> ps : passes) for (Step s : ps) if (s.cfg.j >= from && s.cfg.j <= to && s.cfg.ownR.length > 0) out.add(common ? s.impCommon : s.imp);
        return out;
    }

    static List<Step> steps(List<List<Step>> passes, int from, int to) {
        List<Step> out = new ArrayList<>();
        for (List<Step> ps : passes) for (Step s : ps) if (s.cfg.j >= from && s.cfg.j <= to) out.add(s);
        return out;
    }

    /** Таблицы проекции и порога приёма: ход по шагам, стандартные таблицы, парные разности. */
    static String group(Proj pj, Gate g, Map<Var, List<List<Step>>> all, int persons, XBaseEval.Gt tpl, int maxStep) {
        return group(pj, g, all, persons, tpl, maxStep, null, null);
    }

    /**
     * То же; ref (не null) – (в) того же порога на кадрах «а» для пар «К4 − кадр а» (строки refTag; те же реплики бутстрепа).
     */
    static String group(Proj pj, Gate g, Map<Var, List<List<Step>>> all, int persons, XBaseEval.Gt tpl, int maxStep, List<List<Step>> ref, String refTag) {
        StringBuilder t = new StringBuilder();
        t.append("\n\n######## ").append(pj.label()).append(", порог приёма ").append(g.label).append(" ########\n");
        // Шаг 1 одинаков у всех вариантов.
        List<List<Step>> base = all.get(Var.A);
        for (Var v : all.keySet()) {
            List<FeiDecompose.Cfg> a = cfgs(base, 1, 1);
            List<FeiDecompose.Cfg> b = cfgs(all.get(v), 1, 1);
            for (int i = 0; i < a.size(); i++) {
                if (!Arrays.equals(a.get(i).ownR, b.get(i).ownR) || !Arrays.equals(a.get(i).ctrlR, b.get(i).ctrlR)) {
                    throw new IllegalStateException("feret-update: шаг 1 у " + v.label + " не совпал с (а)");
                }
            }
        }
        t.append("Проверка: шаг 1 у всех вариантов совпал (d₁/d₂ своих и контрольных).\n");

        // Ход по шагам (точка).
        t.append("\n=== Ход по шагам (точка; пороги – П2 / П3 своего шага; «разд.» – порог по контрольным, FAR 1 %) ===\n");
        t.append("вариант | шаг | людей / проходов своих | ошибка своих П3 | П2 | разд. | FAR контроля П3 | П2 | доля первого хаба (лучший у контроля) | "
                + "два самых частых | анти-хабов (без единого лучшего) | лучших у контроля на шаблонах ≥ 3 сессий / их доля в галерее | обновлений "
                + "после шага: верных / ошибочных своих / чужих | возраст шаблона, дней (среднее: дата пробы − самая старая дата шаблона) | людей с "
                + "чужой сессией в шаблоне (из них ошибок при пороге приёма)\n"
                + "(хабы, обновления – по всей галерее обоих проходов; ошибки, FAR, возраст – по людям с ≥ 4 датами и контролю)\n");
        for (Var v : all.keySet()) {
            for (int j = 1; j <= maxStep; j++) {
                List<Step> ss = steps(all.get(v), j, j);
                List<FeiDecompose.Cfg> cs = cfgs(all.get(v), j, j);
                if (cs.isEmpty()) continue;
                t.append(stepRow(v.label, Integer.toString(j), ss, cs, persons));
            }
        }

        // Стандартные таблицы.
        Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
        List<String[]> pairs = new ArrayList<>();
        String p0 = pj.label() + " " + g.label + " ";
        res.put(p0 + "шаг 1", FeiThreshold.evaluate(p0 + "шаг 1", V.L1.ratioId(), cfgs(base, 1, 1), imps(base, 1, 1, false), persons, tpl.b, tpl.seed
                + FeiBootstrap.SEED_OFFSET));
        res.put(p0 + "шаг 1, общая W", FeiThreshold.evaluate(p0 + "шаг 1, общая W", V.L1.ratioId(), cfgs(base, 1, 1), imps(base, 1, 1, true), persons, tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        pairs.add(new String[] {p0 + "шаг 1, общая W", p0 + "шаг 1"});
        Map<String, List<List<Step>>> rows = new LinkedHashMap<>();
        for (Var v : all.keySet()) rows.put(v.label, all.get(v));
        if (ref != null) rows.put(Var.C.label + " " + refTag, ref);
        for (Map.Entry<String, List<List<Step>>> e : rows.entrySet()) {
            for (int j = 2; j <= TABLE_STEPS + 1; j++) {
                int from = j <= TABLE_STEPS ? j : 2;
                int to = j <= TABLE_STEPS ? j : maxStep;
                List<FeiDecompose.Cfg> cs = cfgs(e.getValue(), from, to);
                if (cs.isEmpty()) continue;
                String l = p0 + e.getKey() + (j <= TABLE_STEPS ? " шаг " + j : " шаги ≥ 2");
                res.put(l, FeiThreshold.evaluate(l, V.L1.ratioId(), cs, imps(e.getValue(), from, to, false), persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
            }
        }
        for (String s : new String[] {" шаг 2", " шаги ≥ 2"}) {
            pairs.add(new String[] {p0 + Var.C.label + s, p0 + Var.A.label + s});
            pairs.add(new String[] {p0 + Var.B.label + s, p0 + Var.C.label + s});
            pairs.add(new String[] {p0 + Var.B.label + s, p0 + Var.A.label + s});
            pairs.add(new String[] {p0 + Var.F.label + s, p0 + Var.A.label + s});
            pairs.add(new String[] {p0 + Var.E.label + s, p0 + Var.C.label + s});
            pairs.add(new String[] {p0 + Var.W.label + s, p0 + Var.C.label + s});
            pairs.add(new String[] {p0 + Var.D.label + s, p0 + Var.C.label + s});
            if (ref != null) pairs.add(new String[] {p0 + Var.C.label + s, p0 + Var.C.label + " " + refTag + s});
        }
        pairs.removeIf(p -> !res.containsKey(p[0]) || !res.containsKey(p[1]));
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : res.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, FeiBootstrap.unit(persons), w));
        XBaseOther.tables(t, res, sep, tpl.b, pairs);
        return t.toString();
    }

    /** Строка хода по шагам: объединение проходов шага. */
    static String stepRow(String var, String step, List<Step> ss, List<FeiDecompose.Cfg> cs, int persons) {
        int[] unit = FeiBootstrap.unit(persons);
        int n = 0;
        Set<Integer> people = new HashSet<>();
        double e2 = 0;
        double e3 = 0;
        double f2 = 0;
        double f3 = 0;
        double fn = 0;
        for (Step s : ss) {
            FeiDecompose.Cfg c = s.cfg;
            for (int i = 0; i < c.ownR.length; i++) {
                n++;
                people.add(c.ownP[i]);
                if (!(c.ownOk[i] && c.ownR[i] <= s.th2)) e2++;
                if (!(c.ownOk[i] && c.ownR[i] <= s.th3)) e3++;
            }
            for (double r : c.ctrlR) {
                fn++;
                if (r <= s.th2) f2++;
                if (r <= s.th3) f3++;
            }
        }
        double sepErr = XBaseOther.sep(cs, unit, new int[0][]).err[0];
        // Хабы по объединению проходов шага (галереи проходов не пересекаются).
        Map<String, Integer> cnt = new TreeMap<>();
        int ctrlN = 0;
        int galN = 0;
        int onWide = 0;
        int wideN = 0;
        int[] upd = new int[3];
        double age = 0;
        int ageN = 0;
        int fh = 0;
        int fe = 0;
        for (Step s : ss) {
            for (String b : s.ctrlBest) cnt.merge(b, 1, Integer::sum);
            ctrlN += s.ctrlBest.size();
            galN += s.galN;
            onWide += s.ctrlOnWide;
            wideN += s.wideN;
            upd[0] += s.updOk;
            upd[1] += s.updWrong;
            upd[2] += s.updCtrl;
            age += s.ageSum;
            ageN += s.ageN;
            fh += s.foreignHolders;
            fe += s.foreignErr;
        }
        List<Integer> v = new ArrayList<>(cnt.values());
        v.sort(Collections.reverseOrder());
        int top1 = v.isEmpty() ? 0 : v.get(0);
        int top2 = top1 + (v.size() > 1 ? v.get(1) : 0);
        int anti = galN - cnt.size();
        return String.format(Locale.ROOT, "%s | %s | %d / %d | %.1f %% | %.1f %% | %.1f %% | %.2f %% (%d/%d) | %.2f %% (%d/%d) | %.1f %% | %.1f %% | %d из %d | "
                + "%.1f %% / %.1f %% | %d / %d / %d | %.0f | %d (%d)%n", var, step, people.size(), n, 100.0 * e3 / n, 100.0 * e2 / n, sepErr, 100.0 * f3 / fn,
                (int) f3, (int) fn, 100.0 * f2 / fn, (int) f2, (int) fn, 100.0 * top1 / ctrlN, 100.0 * top2 / ctrlN, anti, galN, 100.0 * onWide / ctrlN,
                galN == 0 ? 0 : 100.0 * wideN / galN, upd[0], upd[1], upd[2], ageN == 0 ? 0 : age / ageN, fh, fe);
    }
}
