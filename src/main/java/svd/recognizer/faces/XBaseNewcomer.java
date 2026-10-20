package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import java.util.concurrent.Callable;
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.faces.XBaseUpdate.Att;
import svd.recognizer.faces.XBaseUpdate.Ctx;
import svd.recognizer.faces.XBaseUpdate.Proj;
import svd.recognizer.storage.SettingsStore;

/**
 * Моделирование обновления шаблона, часть 2 – новичок среди старожилов (PLAN.md п. 5.5, очередь (4); задание Жорика №6 от
 * 15.10.2026, план с поправками №7). FERET eq, половины №2 (свои – половина пула, пул проекции – другая половина), контроль –
 * 200 человек парой fa/fb. Галерея – вся половина. Новички – первые 10 % и 30 % перестановки половины (seed faces.seed + 9970 +
 * r, r = 0, 1, 2; выбор вложенный), шаблон – 1-я дата; остальные – старожилы, шаблон – 2 даты. MLDA и 5 моделей частей «чужих»
 * обучаются на старте на старожилах (2 даты) + когорта + пул, без новичков; при обновлении пересчитываются только центры.
 * Обновление – по подтверждённой личности (решение (15), вариант (д) части 1): после проб шага каждому, кто подал пробу,
 * добавляется сессия даты пробы, окно M = 3. Шаг 1: новички – 2-я дата, старожилы – 3-я; шаг 2: новички – 3-я, старожилы – 4-я.
 * Варианты: (н0) новичков нет (те же W и части; будущие новички – старожилы с 2 датами); (н1) как есть; (н2) + синтетическая
 * сессия (а3, диапазоны (в2) по людям вне своей половины и контроля), выбывает первой; (н3) ширина оценки и порога – наименьшее
 * число сессий в галерее на шаге; (н4) свой П3 (П2) новичка – по «чужим», у которых лучший кандидат – новичок; (н4') справочно –
 * по «чужим» против целей, урезанных до одной сессии. Справочно (н1-W) – W с 1-й датой новичков. Режим XBaseEval
 * feret-newcomer; отчёт – только D:\data\Feret\reports\eq\.
 *
 * @author ssv
 */
final class XBaseNewcomer {

    private XBaseNewcomer() {
    }

    static final long PERM_SEED_OFFSET = 9970;
    static final long SYNTH_SEED_OFFSET = 9980;
    static final double[] SHARES = {0.10, 0.30};
    static final int R = 3;
    static final int STEPS = 2;
    static final int M = 3;
    static final double FAR = 0.01;

    /** Траектории шаблонов (обновление по подтверждённой личности от порога не зависит). */
    enum Traj {
        N0("(н0) без новичков"), N1("(н1) как есть"), N2("(н2) + синтез"), N3("(н3) выравнивание вниз"), N1W("(н1-W) W с новичками");

        final String label;

        Traj(String label) {
            this.label = label;
        }
    }

    /** Строки таблиц: траектория и правило порога. */
    enum Row {
        N0(Traj.N0, 0, "(н0) без новичков"), N1(Traj.N1, 0, "(н1) как есть"), N2(Traj.N2, 0, "(н2) + синтез"), N3(Traj.N3, 0, "(н3) выравнивание вниз"),
        N4(Traj.N1, 1, "(н4) свой порог новичка"), N4T(Traj.N1, 2, "(н4') свой порог, цели по 1 сессии (справочно)"), N1W(Traj.N1W, 0, "(н1-W) W с новичками");

        final Traj traj;
        /** 0 – один порог; 1 – свой порог по «чужим» с лучшим-новичком; 2 – свой порог по «чужим» против целей из 1 сессии. */
        final int rule;
        final String label;

        Row(Traj traj, int rule, String label) {
            this.traj = traj;
            this.rule = rule;
            this.label = label;
        }
    }

    // ---------------------------------------------------------------- данные шага

    /** Проба своего: человек, новичок, d₁/d₂, верный, анфас, лучший (индекс бутстрепа), лучший – новичок. */
    record OwnA(int p, boolean nw, double r, boolean ok, boolean front, int best, boolean bestNew, String key) {}

    /** Проба контрольного: человек, d₁/d₂, лучший, лучший – новичок, дата в шаблоне лучшего, ожидаемое у новичков / старожилов. */
    record CtrlA(int p, double r, int best, boolean bestNew, boolean same, double expNew, double expOld, String key) {}

    /** «Чужой» калибровки: человек части, d₁/d₂, человек части – новичок, лучший – новичок. */
    record ImpA(int p, double r, boolean ownerNew, boolean bestNew) {}

    /** Шаг одной конфигурации (выбор r, доля, проход, проекция, траектория). */
    static final class SD {
        int step;
        List<OwnA> own = new ArrayList<>();
        List<CtrlA> ctrl = new ArrayList<>();
        List<ImpA> imp = new ArrayList<>();
        /** (н4'): «чужие» против целей, урезанных до одной сессии (только у (н1)). */
        List<ImpA> impT = new ArrayList<>();
        /** Галерея: индексы бутстрепа и признак «новичок». */
        int[] galP;
        boolean[] galNew;
        /** Ширина оценки (н3), сессий; иначе 0. */
        int width;

        String encode() {
            StringBuilder t = new StringBuilder();
            t.append("S\t").append(step).append('\t').append(width).append('\n');
            for (int i = 0; i < galP.length; i++) t.append("G\t").append(galP[i]).append('\t').append(galNew[i] ? 1 : 0).append('\n');
            for (OwnA a : own) t.append(String.join("\t", "O", "" + a.p(), b(a.nw()), Double.toString(a.r()), b(a.ok()), b(a.front()), "" + a.best(), b(a.bestNew()),
                    a.key())).append('\n');
            for (CtrlA a : ctrl) t.append(String.join("\t", "C", "" + a.p(), Double.toString(a.r()), "" + a.best(), b(a.bestNew()), b(a.same()),
                    Double.toString(a.expNew()), Double.toString(a.expOld()), a.key())).append('\n');
            for (ImpA a : imp) t.append(String.join("\t", "I", "" + a.p(), Double.toString(a.r()), b(a.ownerNew()), b(a.bestNew()))).append('\n');
            for (ImpA a : impT) t.append(String.join("\t", "T", "" + a.p(), Double.toString(a.r()), b(a.ownerNew()), b(a.bestNew()))).append('\n');
            return t.toString();
        }

        static List<SD> decode(String s) {
            List<SD> out = new ArrayList<>();
            SD cur = null;
            List<int[]> g = new ArrayList<>();
            for (String l : s.split("\n")) {
                if (l.isEmpty()) continue;
                String[] c = l.split("\t", -1);
                switch (c[0]) {
                    case "S" -> {
                        if (cur != null) finish(cur, g);
                        cur = new SD();
                        cur.step = Integer.parseInt(c[1]);
                        cur.width = Integer.parseInt(c[2]);
                        out.add(cur);
                        g = new ArrayList<>();
                    }
                    case "G" -> g.add(new int[] {Integer.parseInt(c[1]), Integer.parseInt(c[2])});
                    case "O" -> cur.own.add(new OwnA(Integer.parseInt(c[1]), c[2].equals("1"), Double.parseDouble(c[3]), c[4].equals("1"), c[5].equals("1"),
                            Integer.parseInt(c[6]), c[7].equals("1"), c[8]));
                    case "C" -> cur.ctrl.add(new CtrlA(Integer.parseInt(c[1]), Double.parseDouble(c[2]), Integer.parseInt(c[3]), c[4].equals("1"), c[5].equals("1"),
                            Double.parseDouble(c[6]), Double.parseDouble(c[7]), c[8]));
                    case "I" -> cur.imp.add(new ImpA(Integer.parseInt(c[1]), Double.parseDouble(c[2]), c[3].equals("1"), c[4].equals("1")));
                    case "T" -> cur.impT.add(new ImpA(Integer.parseInt(c[1]), Double.parseDouble(c[2]), c[3].equals("1"), c[4].equals("1")));
                    default -> throw new IllegalStateException("feret-newcomer: строка контрольной точки " + l);
                }
            }
            if (cur != null) finish(cur, g);
            return out;
        }

        private static void finish(SD sd, List<int[]> g) {
            sd.galP = g.stream().mapToInt(x -> x[0]).toArray();
            sd.galNew = new boolean[g.size()];
            for (int i = 0; i < g.size(); i++) sd.galNew[i] = g.get(i)[1] == 1;
        }

        private static String b(boolean v) {
            return v ? "1" : "0";
        }
    }

    // ---------------------------------------------------------------- моделирование

    /** Схема галереи: человек → стартовые сессии; номер даты пробы на шаге 1; новичок. */
    record Plan(Map<String, List<TemplateWindow.Session>> start, Map<String, Integer> first, Set<String> newcomers) {}

    /** Ключи шаблона для оценки: при width > 0 – последние width сессий. */
    static List<String> keys(TemplateWindow w, int width) {
        List<TemplateWindow.Session> s = w.sessions();
        if (width > 0 && s.size() > width) s = s.subList(s.size() - width, s.size());
        List<String> out = new ArrayList<>();
        for (TemplateWindow.Session x : s) out.addAll(x.keys());
        return out;
    }

    static List<TemplateWindow.Session> sessions(TemplateWindow w, int width) {
        List<TemplateWindow.Session> s = w.sessions();
        return width > 0 && s.size() > width ? s.subList(s.size() - width, s.size()) : s;
    }

    /** (н3): ширина оценки шага – наименьшее число сессий у людей галереи, не больше M. */
    static int width(Map<String, TemplateWindow> tpl, List<String> gal) {
        int w = M;
        for (String p : gal) w = Math.min(w, tpl.get(p).size());
        return w;
    }

    static double[][] centers(Ctx c, String id, LdaModel lm, int m, List<String> persons, Map<String, TemplateWindow> tpl, int width) {
        double[][] out = new double[persons.size()][];
        for (int i = 0; i < out.length; i++) {
            List<String> k = keys(tpl.get(persons.get(i)), width);
            if (!k.isEmpty()) out[i] = c.center(id, lm, m, k);
        }
        return out;
    }

    /** «Чужие» калибровки: сессии (в пределах ширины) людей части q против остальных; targetWidth – ширина целей. */
    static List<ImpA> impostors(Ctx c, Map<String, TemplateWindow> tpl, Map<String, Integer> pIdx, Set<String> nw, int width, int targetWidth) {
        List<ImpA> out = new ArrayList<>();
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
            Set<String> pq = new HashSet<>(c.parts.get(q));
            List<String> rest = c.gal.stream().filter(p -> !pq.contains(p)).toList();
            LdaModel[] lms = {c.part[0][q], c.part[1][q]};
            String[] ids = {"part0_" + q, "part1_" + q};
            double[][][] cs = {centers(c, ids[0], lms[0], 0, rest, tpl, targetWidth), centers(c, ids[1], lms[1], 1, rest, tpl, targetWidth)};
            for (String p : c.parts.get(q)) {
                for (TemplateWindow.Session s : sessions(tpl.get(p), width)) {
                    Att a = c.pair(ids, lms, cs, s.keys(), -1);
                    out.add(new ImpA(pIdx.get(p), a.r(), nw.contains(p), nw.contains(rest.get(a.best()))));
                }
            }
        }
        return out;
    }

    /** Моделирование одной траектории: шаги 1…steps; н3 – ширина оценки по правилу width(); withT – «чужие» (н4'). */
    static List<SD> simulate(Ctx c, Plan plan, boolean n3, boolean withT, List<XBaseUpdate.Ctrl> ctrl, Map<String, Integer> pIdx, int steps) {
        Map<String, TemplateWindow> tpl = new HashMap<>();
        for (String p : c.gal) {
            TemplateWindow w = new TemplateWindow(M);
            for (TemplateWindow.Session s : plan.start().get(p)) w.add(s);
            tpl.put(p, w);
        }
        Set<String> nw = plan.newcomers();
        String[] ids = {"full0", "full1"};
        List<SD> out = new ArrayList<>();
        for (int j = 1; j <= steps; j++) {
            SD sd = new SD();
            sd.step = j;
            int width = n3 ? width(tpl, c.gal) : 0;
            sd.width = width;
            sd.galP = c.gal.stream().mapToInt(pIdx::get).toArray();
            sd.galNew = new boolean[c.gal.size()];
            for (int i = 0; i < c.gal.size(); i++) sd.galNew[i] = nw.contains(c.gal.get(i));
            sd.imp = impostors(c, tpl, pIdx, nw, width, width);
            if (withT) sd.impT = impostors(c, tpl, pIdx, nw, width, 1);
            double[][][] cs = {centers(c, ids[0], c.full[0], 0, c.gal, tpl, width), centers(c, ids[1], c.full[1], 1, c.gal, tpl, width)};
            // Даты шаблонов (для меры общей сессии).
            Map<String, Set<String>> dates = new HashMap<>();
            for (String p : c.gal) {
                Set<String> ds = new HashSet<>();
                for (TemplateWindow.Session s : sessions(tpl.get(p), width)) ds.add(s.date());
                dates.put(p, ds);
            }
            int nNew = nw.size();
            int nOld = c.gal.size() - nNew;
            List<String> probers = new ArrayList<>();
            List<List<String>> probeKeys = new ArrayList<>();
            for (String p : c.gal) {
                int d = plan.first().get(p) + j - 1;
                if (c.dates.get(p).size() <= d) continue;
                probers.add(p);
                List<String> ks = c.dates.get(p).get(d);
                probeKeys.add(ks);
                Att a = c.pair(ids, c.full, cs, ks, c.gi.get(p));
                String b = c.gal.get(a.best());
                sd.own.add(new OwnA(pIdx.get(p), nw.contains(p), a.r(), a.best() == c.gi.get(p), a.front(), pIdx.get(b), nw.contains(b), a.key()));
            }
            for (XBaseUpdate.Ctrl k : ctrl) {
                Att a = c.pair(ids, c.full, cs, k.keys(), -1);
                String b = c.gal.get(a.best());
                int en = 0;
                int eo = 0;
                for (String p : c.gal) {
                    if (!dates.get(p).contains(k.date())) continue;
                    if (nw.contains(p)) en++;
                    else eo++;
                }
                sd.ctrl.add(new CtrlA(pIdx.get(k.person()), a.r(), pIdx.get(b), nw.contains(b), dates.get(b).contains(k.date()), nNew == 0 ? 0 : (double) en / nNew,
                        nOld == 0 ? 0 : (double) eo / nOld, a.key()));
            }
            // Обновление по подтверждённой личности – всем, кто подал пробу.
            for (int i = 0; i < probers.size(); i++) {
                List<String> ks = probeKeys.get(i);
                tpl.get(probers.get(i)).add(new TemplateWindow.Session(c.dateOf.get(ks.get(0)), probers.get(i), ks));
            }
            out.add(sd);
        }
        return out;
    }

    /** Контекст: галерея gal (вся половина), W и модели частей – по train (человек → ключи; люди вне train – не в обучении). */
    static Ctx ctx(XBaseEval x, XBaseEval.Gt tpl, Map<String, Double> rOf, List<String> gal, Map<String, List<String>> train, List<List<String>> aug,
                   Map<String, List<List<String>>> pool, Map<String, String> dateOf) {
        Ctx c = new Ctx();
        c.x = x;
        c.r0 = tpl.r0;
        c.rOf = rOf;
        c.gal = new ArrayList<>(gal);
        for (int i = 0; i < c.gal.size(); i++) c.gi.put(c.gal.get(i), i);
        c.dates = new HashMap<>();
        for (String p : c.gal) c.dates.put(p, pool.get(p));
        c.dateOf = dateOf;
        c.parts = XBaseOther.cvParts(c.gal, tpl.seed, XBaseEval.CV_S + XBaseUpdate.PARTS_S);
        List<Callable<Void>> tasks = new ArrayList<>();
        for (int m = 0; m < 2; m++) {
            final int mm = m;
            boolean mir = m == 1;
            List<List<String>> all = c.gal.stream().filter(train::containsKey).map(train::get).toList();
            tasks.add(() -> {
                c.full[mm] = (LdaModel) x.fit(V.L1, mir, all, aug);
                return null;
            });
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                final int qq = q;
                Set<String> pq = new HashSet<>(c.parts.get(q));
                List<List<String>> g = c.gal.stream().filter(p -> !pq.contains(p) && train.containsKey(p)).map(train::get).toList();
                tasks.add(() -> {
                    c.part[mm][qq] = (LdaModel) x.fit(V.L1, mir, g, aug);
                    return null;
                });
            }
        }
        XBaseUpdate.runAll(tasks, tpl.threads);
        return c;
    }

    // ---------------------------------------------------------------- оценка

    /** Сортированные «чужие» для порога: r по возрастанию и человек. */
    record Sorted(double[] r, int[] p) {
        static Sorted of(List<ImpA> l, java.util.function.Predicate<ImpA> f) {
            List<ImpA> s = new ArrayList<>(l.stream().filter(f).toList());
            s.sort((a, b) -> Double.compare(a.r(), b.r()));
            return new Sorted(s.stream().mapToDouble(ImpA::r).toArray(), s.stream().mapToInt(ImpA::p).toArray());
        }
    }

    /** Конфигурация оценки: шаг и заранее отсортированные «чужие» (все, с лучшим-новичком, против целей из 1 сессии). */
    static final class Cfg {
        final SD sd;
        final Sorted all;
        final Sorted toNew;
        final Sorted toNewT;
        final double[] ctrlR;
        final int[] ctrlP;

        Cfg(SD sd) {
            this.sd = sd;
            all = Sorted.of(sd.imp, a -> true);
            toNew = Sorted.of(sd.imp, ImpA::bestNew);
            toNewT = Sorted.of(sd.impT, ImpA::bestNew);
            List<CtrlA> c = new ArrayList<>(sd.ctrl);
            c.sort((a, b) -> Double.compare(a.r(), b.r()));
            ctrlR = c.stream().mapToDouble(CtrlA::r).toArray();
            ctrlP = c.stream().mapToInt(CtrlA::p).toArray();
        }
    }

    static double thr(int m, Sorted s, int[] w) {
        if (s.r().length == 0) return Double.NaN;
        return m == 2 ? FeiBootstrap.threshold(s.r(), s.p(), w, FAR) : FeiThreshold.margin(s.r(), s.p(), w, FAR).theta();
    }

    /**
     * Меры строки при весах w: [0] ошибка новичков П3, [1] старожилов П3, [2] новичков П2, [3] старожилов П2, [4] FAR контроля П3,
     * [5] FAR П2, [6] ошибка новичков при пороге по контролю (FAR 1 %), [7] старожилов при нём. NaN – нет проб.
     */
    static double[] metrics(List<Cfg> cs, int rule, int[] w) {
        double[] x = new double[8];
        double[] n = new double[8];
        boolean[] nan = new boolean[8]; // порог не определён у пробы с весом – мера NaN (решение Хозяина 20.10.2026, №9 Жорика)
        for (Cfg c : cs) {
            double[] th = new double[4]; // П3 общий, П3 новичка, П2 общий, П2 новичка
            th[0] = thr(3, c.all, w);
            th[2] = thr(2, c.all, w);
            Sorted sub = rule == 1 ? c.toNew : rule == 2 ? c.toNewT : null;
            th[1] = sub == null || sub.r().length == 0 ? th[0] : thr(3, sub, w);
            th[3] = sub == null || sub.r().length == 0 ? th[2] : thr(2, sub, w);
            double sep = FeiBootstrap.threshold(c.ctrlR, c.ctrlP, w, FAR);
            for (OwnA a : c.sd.own) {
                int q = w[a.p()];
                if (q == 0) continue;
                int g = a.nw() ? 0 : 1;
                for (int k = 0; k < 2; k++) {
                    double t = a.bestNew() ? th[2 * k + 1] : th[2 * k];
                    n[2 * k + g] += q;
                    nan[2 * k + g] |= Double.isNaN(t);
                    if (!(a.ok() && a.r() <= t)) x[2 * k + g] += q;
                }
                n[6 + g] += q;
                nan[6 + g] |= Double.isNaN(sep);
                if (!(a.ok() && a.r() <= sep)) x[6 + g] += q;
            }
            for (CtrlA a : c.sd.ctrl) {
                int q = w[a.p()];
                if (q == 0) continue;
                for (int k = 0; k < 2; k++) {
                    double t = a.bestNew() ? th[2 * k + 1] : th[2 * k];
                    n[4 + k] += q;
                    nan[4 + k] |= Double.isNaN(t);
                    if (a.r() <= t) x[4 + k] += q;
                }
            }
        }
        double[] out = new double[8];
        for (int i = 0; i < 8; i++) out[i] = n[i] == 0 || nan[i] ? Double.NaN : 100.0 * x[i] / n[i];
        return out;
    }

    /** Реплика «доля той же сессии − ожидаемая», п. п., по суммам z = {проб, той же сессии, ожидаемо}; пустая реплика – NaN (№9 Жорика). */
    static double sameExcess(double[] z) {
        return z[0] == 0 ? Double.NaN : 100 * (z[1] - z[2]) / z[0];
    }

    /** Строка таблицы: точка и реплики мер. */
    static final class Res {
        double[] point;
        double[][] reps;
        int nNew;
        int nOld;
        int pNew;
        int pOld;
    }

    static Res evaluate(List<SD> sds, int rule, int persons, int[][] w) {
        List<Cfg> cs = sds.stream().map(Cfg::new).toList();
        Res r = new Res();
        r.point = metrics(cs, rule, FeiBootstrap.unit(persons));
        r.reps = new double[8][w.length];
        java.util.stream.IntStream.range(0, w.length).parallel().forEach(q -> {
            double[] m = metrics(cs, rule, w[q]);
            for (int i = 0; i < 8; i++) r.reps[i][q] = m[i];
        });
        Set<Integer> pn = new HashSet<>();
        Set<Integer> po = new HashSet<>();
        for (SD s : sds) {
            for (OwnA a : s.own) {
                if (a.nw()) {
                    r.nNew++;
                    pn.add(a.p());
                } else {
                    r.nOld++;
                    po.add(a.p());
                }
            }
        }
        r.pNew = pn.size();
        r.pOld = po.size();
        return r;
    }

    static String cell(double point, double[] reps) {
        if (Double.isNaN(point)) return "–";
        double[] ok = Arrays.stream(reps).filter(v -> !Double.isNaN(v)).toArray();
        if (ok.length == 0) return String.format(Locale.ROOT, "%.1f %%", point);
        double[] s = FeiBootstrap.sorted(ok);
        return String.format(Locale.ROOT, "%.1f %% [%.1f; %.1f]", point, FeiBootstrap.quantile(s, 0.025), FeiBootstrap.quantile(s, 0.975));
    }

    static String diffCell(Res a, Res b, int i) {
        double p = a.point[i] - b.point[i];
        if (Double.isNaN(p)) return "–";
        double[] d = new double[a.reps[i].length];
        int k = 0;
        for (int q = 0; q < d.length; q++) {
            double v = a.reps[i][q] - b.reps[i][q];
            if (!Double.isNaN(v)) d[k++] = v;
        }
        double[] s = FeiBootstrap.sorted(Arrays.copyOf(d, k));
        double lo = FeiBootstrap.quantile(s, 0.025);
        double hi = FeiBootstrap.quantile(s, 0.975);
        return String.format(Locale.ROOT, "%+.1f [%+.1f; %+.1f]%s", p, lo, hi, lo > 0 || hi < 0 ? " *" : "");
    }

    // ---------------------------------------------------------------- запуск

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        if (!settings.loadFacesFeretGalleryEqual()) throw new IllegalStateException("feret-newcomer: только равные галереи (faces.feret.gallery = eq)");
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
        List<XBaseUpdate.Ctrl> ctrl = new ArrayList<>();
        for (String p : roAll.ctrl) {
            List<String> ks = XBaseFeret.sorted(XBaseFeret.pairOf(roAll.ctrlKeys.get(p), su.byKey, false));
            ctrl.add(new XBaseUpdate.Ctrl(p, su.byKey.get(ks.get(0)).date(), ks));
        }
        Set<String> used = new TreeSet<>();
        for (List<List<String>> ds : pool.values()) for (List<String> d : ds) used.addAll(d);
        for (XBaseUpdate.Ctrl k : ctrl) used.addAll(k.keys());
        Map<String, String> dateOf = new HashMap<>();
        for (String k : used) dateOf.put(k, su.byKey.get(k).date());

        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        for (XBaseUpdate.Ctrl k : ctrl) ps0.add(k.person());
        List<String> persons = new ArrayList<>(ps0);
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);

        // Выбор новичков: на повтор r и проход – перестановка половины, новички – первые 10 % и 30 %.
        Map<String, List<String>> permOf = new HashMap<>();
        Map<String, Set<String>> newOf = new LinkedHashMap<>();
        for (int r = 0; r < R; r++) {
            for (int pp = 0; pp < 2; pp++) {
                List<String> hp = new ArrayList<>(halves.get(pp));
                Collections.shuffle(hp, new Random(tpl.seed + PERM_SEED_OFFSET + r));
                permOf.put(r + "_" + pp, hp);
                for (int si = 0; si < SHARES.length; si++) newOf.put(r + "_" + si + "_" + pp, new TreeSet<>(hp.subList(0, (int) Math.round(SHARES[si] * hp.size()))));
            }
        }

        // Векторы: кадры «а» пула и контроля, когорта, синтез (н2) для новичков 30 % (надмножество 10 %).
        Map<String, Mat> fr = new LinkedHashMap<>();
        for (String k : used) fr.put(k, FeretData.frame(dir, "a", k));
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        StringBuilder synthNote = new StringBuilder();
        Map<String, List<String>> synthKeys = synth(su, tpl, pool, halves, newOf, roAll.ctrl, fr, synthNote);
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        for (Mat m : fr.values()) m.release();
        for (Map.Entry<String, List<String>> e : synthKeys.entrySet()) {
            String src = e.getKey().substring(e.getKey().indexOf('|') + 1);
            for (int i = 0; i < e.getValue().size(); i++) {
                String sk = e.getValue().get(i);
                String base = pool.get(src).get(0).get(i % pool.get(src).get(0).size());
                su.rOf.put(sk, su.rOf.get(base));
                dateOf.put(sk, dateOf.get(base));
            }
        }
        System.out.printf(Locale.ROOT, "feret-newcomer: векторы готовы (%.0f с)%n", (System.nanoTime() - t0) / 1e9);

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-newcomer %s; снимков %d; seed %d; r0 %s; пул %d; R %d; доли %s", dir,
                su.imgs.size(), tpl.seed, tpl.r0, pool.size(), R, Arrays.toString(SHARES)));
        // Пробный режим (отладка отчёта): FACES_NEWCOMER_SMOKE=1 – только L1, r = 0, без проверки п. 2 и (н1-W), отчёт *_smoke.
        boolean smoke = "1".equals(System.getenv("FACES_NEWCOMER_SMOKE"));
        String sfx = smoke ? "_smoke" : "";
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve("checkpoint_feret_newcomer_eq" + sfx), fp);
        StringBuilder t = new StringBuilder();
        header(t, dir, exportDir, tpl);
        if (smoke) t.append("ПРОБНЫЙ РЕЖИМ (FACES_NEWCOMER_SMOKE=1): только L1, r = 0, без проверки п. 2 и (н1-W) – не для выводов.\n");
        t.append(synthNote);

        // П. 2: проверка – новичков нет, W на всех: строки (д) №4 (L1-xs-F, П3, шаги ≥ 2, люди с ≥ 4 датами) побайтно.
        if (!smoke) t.append(check(xs[0], tpl, su, halves, pool, cohort, ctrl, pIdx, dateOf, out, cp));
        // П. 1: механизм (б) по оценкам №4.
        t.append(mechanismB(out, halves, pool, ctrl, tpl));

        // Конфигурации: проекция × r × доля × проход; траектории (н0)–(н3).
        Map<String, List<SD>> res = new HashMap<>();
        double simS = 0;
        List<Proj> projs = smoke ? List.of(XBaseUpdate.PROJS.get(2)) : XBaseUpdate.PROJS;
        int rN = smoke ? 1 : R;
        for (Proj pj : projs) {
            for (int r = 0; r < rN; r++) {
                for (int si = 0; si < SHARES.length; si++) {
                    for (int pp = 0; pp < 2; pp++) {
                        String name = String.format(Locale.ROOT, "%s_r%d_s%d_p%d", pj.label(), r, si, pp);
                        String cached = cp.read(name);
                        if (cached == null) {
                            long ts = System.nanoTime();
                            StringBuilder enc = new StringBuilder();
                            Set<String> nw = newOf.get(r + "_" + si + "_" + pp);
                            Map<Traj, List<SD>> m = runConfig(xs[pj.f()], tpl, su.rOf, halves.get(pp), halves.get(1 - pp), pool, pj.pool(), cohort, dateOf, nw,
                                    synthKeys, r, ctrl, pIdx, false);
                            for (Map.Entry<Traj, List<SD>> e : m.entrySet()) {
                                enc.append("# ").append(e.getKey().name()).append('\n');
                                for (SD sd : e.getValue()) enc.append(sd.encode());
                            }
                            cp.save(name, enc.toString());
                            cached = enc.toString();
                            double sec = (System.nanoTime() - ts) / 1e9;
                            simS += sec;
                            System.out.printf(Locale.ROOT, "feret-newcomer: %s готово (%.0f с)%n", name, sec);
                        }
                        for (Map.Entry<String, String> e : blocks(cached).entrySet()) {
                            res.computeIfAbsent(pj.label() + "|" + si + "|" + e.getKey(), z -> new ArrayList<>()).addAll(SD.decode(e.getValue()));
                            res.computeIfAbsent(pj.label() + "|" + si + "|" + e.getKey() + "|r" + r, z -> new ArrayList<>()).addAll(SD.decode(e.getValue()));
                        }
                    }
                }
            }
        }
        // (н1-W): W с 1-й датой новичков; L1-xs-F и L1-tt-xs-F, доля 30 %, r = 0.
        for (Proj pj : projs) {
            if (!pj.pool() || smoke) continue;
            for (int pp = 0; pp < 2; pp++) {
                String name = String.format(Locale.ROOT, "%s_W_p%d", pj.label(), pp);
                String cached = cp.read(name);
                if (cached == null) {
                    long ts = System.nanoTime();
                    Map<Traj, List<SD>> m = runConfig(xs[pj.f()], tpl, su.rOf, halves.get(pp), halves.get(1 - pp), pool, pj.pool(), cohort, dateOf,
                            newOf.get("0_1_" + pp), synthKeys, 0, ctrl, pIdx, true);
                    StringBuilder enc = new StringBuilder("# " + Traj.N1W.name() + "\n");
                    for (SD sd : m.get(Traj.N1W)) enc.append(sd.encode());
                    cp.save(name, enc.toString());
                    cached = enc.toString();
                    double sec = (System.nanoTime() - ts) / 1e9;
                    simS += sec;
                    System.out.printf(Locale.ROOT, "feret-newcomer: %s готово (%.0f с)%n", name, sec);
                }
                for (Map.Entry<String, String> e : blocks(cached).entrySet()) {
                    res.computeIfAbsent(pj.label() + "|1|" + e.getKey() + "|r0", z -> new ArrayList<>()).addAll(SD.decode(e.getValue()));
                }
            }
        }
        t.append(volume(res, newOf));
        t.append(String.format(Locale.ROOT, "Обучение и моделирование: %.0f с (без конфигураций из контрольной точки); всего %.0f с; потоков %d.%n", simS,
                (System.nanoTime() - t0) / 1e9, tpl.threads));
        t.append(tables(res, persons.size(), tpl, projs));
        FarMethods.write(out.resolve("xbase_feret_newcomer" + sfx + ".txt"), t);
        System.out.print(t);
        cp.delete();
    }

    static Map<String, String> blocks(String enc) {
        Map<String, StringBuilder> b = new LinkedHashMap<>();
        String cur = null;
        for (String l : enc.split("\n")) {
            if (l.startsWith("# ")) {
                cur = l.substring(2);
                b.put(cur, new StringBuilder());
            } else if (cur != null) {
                b.get(cur).append(l).append('\n');
            }
        }
        Map<String, String> out = new LinkedHashMap<>();
        b.forEach((k, v) -> out.put(k, v.toString()));
        return out;
    }

    /** Одна конфигурация: W и части – по старожилам (или, при withNewW, ещё по 1-й дате новичков); траектории. */
    static Map<Traj, List<SD>> runConfig(XBaseEval x, XBaseEval.Gt tpl, Map<String, Double> rOf, Set<String> half, Set<String> poolHalf,
                                         Map<String, List<List<String>>> pool, boolean usePool, List<List<String>> cohort, Map<String, String> dateOf,
                                         Set<String> nw, Map<String, List<String>> synthKeys, int r, List<XBaseUpdate.Ctrl> ctrl, Map<String, Integer> pIdx,
                                         boolean withNewW) {
        List<String> gal = new ArrayList<>(half);
        Map<String, List<String>> train = new LinkedHashMap<>();
        for (String p : gal) {
            List<String> k = new ArrayList<>();
            if (nw.contains(p)) {
                if (!withNewW) continue;
                k.addAll(pool.get(p).get(0));
            } else {
                for (int d = 0; d < XBaseUpdate.START; d++) k.addAll(pool.get(p).get(d));
            }
            train.put(p, k);
        }
        Ctx c = ctx(x, tpl, rOf, gal, train, XBaseUpdate.aug(cohort, poolHalf, pool, usePool), pool, dateOf);
        Map<Traj, List<SD>> out = new LinkedHashMap<>();
        List<Callable<List<SD>>> tasks = new ArrayList<>();
        List<Traj> order = withNewW ? List.of(Traj.N1W) : List.of(Traj.N0, Traj.N1, Traj.N2, Traj.N3);
        for (Traj tr : order) {
            Set<String> nwT = tr == Traj.N0 ? Set.of() : nw;
            Plan plan = plan(gal, pool, nwT, tr == Traj.N2 ? synthKeys : null, r, dateOf);
            tasks.add(() -> simulate(c, plan, tr == Traj.N3, tr == Traj.N1, ctrl, pIdx, STEPS));
        }
        List<List<SD>> done = XBaseUpdate.runAll(tasks, tpl.threads);
        for (int i = 0; i < order.size(); i++) out.put(order.get(i), done.get(i));
        // (н0): признак «новичок этого выбора» (будущие новички – старожилы с 2 датами вне обучения W) – для групп и хабов; пороги
        // (н0) от признака не зависят (свой порог новичка у (н0) не считается).
        if (!withNewW) {
            Set<Integer> ni = nwIdx(nw, pIdx);
            for (SD sd : out.get(Traj.N0)) {
                List<OwnA> own = new ArrayList<>();
                for (OwnA a : sd.own) own.add(new OwnA(a.p(), ni.contains(a.p()), a.r(), a.ok(), a.front(), a.best(), ni.contains(a.best()), a.key()));
                sd.own = own;
                List<CtrlA> ct = new ArrayList<>();
                for (CtrlA a : sd.ctrl) ct.add(new CtrlA(a.p(), a.r(), a.best(), ni.contains(a.best()), a.same(), a.expNew(), a.expOld(), a.key()));
                sd.ctrl = ct;
                for (int i = 0; i < sd.galP.length; i++) sd.galNew[i] = ni.contains(sd.galP[i]);
            }
        }
        return out;
    }

    static Set<Integer> nwIdx(Set<String> nw, Map<String, Integer> pIdx) {
        Set<Integer> s = new HashSet<>();
        for (String p : nw) s.add(pIdx.get(p));
        return s;
    }

    /** Стартовые сессии: новичок – (н2: синтетическая сессия, затем) 1-я дата, проба шага 1 – 2-я; старожил – 2 даты, проба – 3-я. */
    static Plan plan(List<String> gal, Map<String, List<List<String>>> pool, Set<String> nw, Map<String, List<String>> synthKeys, int r,
                     Map<String, String> dateOf) {
        Map<String, List<TemplateWindow.Session>> start = new HashMap<>();
        Map<String, Integer> first = new HashMap<>();
        for (String p : gal) {
            List<TemplateWindow.Session> s = new ArrayList<>();
            List<List<String>> ds = pool.get(p);
            if (nw.contains(p)) {
                if (synthKeys != null) {
                    List<String> sk = synthKeys.get(r + "|" + p);
                    if (sk == null) throw new IllegalStateException("feret-newcomer: нет синтеза для " + p + ", r = " + r);
                    s.add(new TemplateWindow.Session(dateOf.get(ds.get(0).get(0)), p, sk));
                }
                s.add(new TemplateWindow.Session(dateOf.get(ds.get(0).get(0)), p, ds.get(0)));
                first.put(p, 1);
            } else {
                for (int d = 0; d < XBaseUpdate.START; d++) s.add(new TemplateWindow.Session(dateOf.get(ds.get(d).get(0)), p, ds.get(d)));
                first.put(p, XBaseUpdate.START);
            }
            start.put(p, s);
        }
        return new Plan(start, first, nw);
    }

    // ---------------------------------------------------------------- синтез (н2)

    /**
     * Синтетическая сессия новичка (а3, схема (в2)): по одному синтезу на каждый снимок 1-й даты (fa, fb); диапазоны – люди вне
     * своей половины и вне контроля (для половины pp – люди другой половины и не из пула), даты – любые (их люди в моделировании
     * пробами не участвуют); Random на человека – seed faces.seed + 9980 + 1000·r + номер человека. Кадры – в fr.
     */
    static Map<String, List<String>> synth(XBaseA0.FeretSetup su, XBaseEval.Gt tpl, Map<String, List<List<String>>> pool, List<Set<String>> halves,
                                           Map<String, Set<String>> newOf, List<String> ctrlP, Map<String, Mat> fr, StringBuilder note) throws IOException {
        Map<String, List<String>> out = new LinkedHashMap<>();
        Map<String, double[]> cache = new HashMap<>();
        Set<String> ctrlSet = new HashSet<>(ctrlP);
        for (int pp = 0; pp < 2; pp++) {
            List<Double> noise = new ArrayList<>();
            List<Double> asym = new ArrayList<>();
            List<double[]> align = new ArrayList<>();
            for (Img im : su.imgs) {
                double[] row = su.det.get(im.key());
                if (row == null || su.rOf.get(im.key()) > tpl.rLimit) continue;
                if (halves.get(pp).contains(im.person()) || ctrlSet.contains(im.person())) continue;
                double[] m = XBaseA0Check.measures(su.src, im.key(), cache);
                asym.add(m[0]);
                noise.add(m[1]);
                double[][] man = FeretData.manual4(im);
                if (man != null) align.add(FaceSynth.alignDelta(FeretData.yunet4(row), man));
            }
            if (noise.size() < XBaseSynth.RANGE_MIN || align.size() < XBaseSynth.RANGE_MIN) {
                throw new IllegalStateException(String.format(Locale.ROOT, "feret-newcomer: мало снимков для диапазонов синтеза (σ и A %d, выравнивание %d)",
                        noise.size(), align.size()));
            }
            FaceSynth.Ranges ranges = new FaceSynth.Ranges(XBaseSynth.sortedArr(noise), XBaseSynth.sortedArr(asym), align);
            note.append(String.format(Locale.ROOT, "Синтез (н2), проход %d: диапазоны по %d снимкам (σ и A), выравнивание – %d размеченных.%n", pp + 1, noise.size(),
                    align.size()));
            for (int r = 0; r < R; r++) {
                for (String p : newOf.get(r + "_1_" + pp)) {
                    List<String> real = pool.get(p).get(0);
                    Random rnd = new Random(tpl.seed + SYNTH_SEED_OFFSET + 1000L * r + Integer.parseInt(p));
                    List<String> ks = new ArrayList<>();
                    for (int i = 0; i < real.size(); i++) {
                        String b = real.get(i);
                        Mat color = FeretData.decode(su.byKey.get(b).file());
                        FaceSynth.Params prm = FaceSynth.sample(ranges, rnd);
                        Mat a = FaceSynth.synth(color, su.det.get(b), prm, rnd);
                        color.release();
                        String key = b + "#n" + r + "_" + i;
                        fr.put(key, a);
                        ks.add(key);
                    }
                    out.put(r + "|" + p, ks);
                }
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- проверка п. 2 и механизм (б)

    static String check(XBaseEval x, XBaseEval.Gt tpl, XBaseA0.FeretSetup su, List<Set<String>> halves, Map<String, List<List<String>>> pool,
                        List<List<String>> cohort, List<XBaseUpdate.Ctrl> ctrl, Map<String, Integer> pIdx, Map<String, String> dateOf, Path out,
                        XBaseOther.Checkpoint cp) throws IOException {
        StringBuilder t = new StringBuilder("\n=== П. 2, проверка кода: новичков нет, W на всех (как №4) – строки (д) feret-update, L1-xs-F, П3, шаги ≥ 2, люди с ≥ 4 "
                + "датами, побайтно (ключ решающего снимка, лучший, d₁/d₂) ===\n");
        String cached = cp.read("check");
        if (cached != null) return t.append(cached).toString();
        Map<String, String> old = new HashMap<>();
        Path sf = out.resolve(XBaseSession.SCORES);
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(sf), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                String[] c = l.split("\t", -1);
                if (c.length < 12 || !c[0].equals("L1-xs-F") || !c[2].equals("П3") || !c[3].equals(XBaseUpdate.Var.E.label) || !c[5].equals(FeiScores.OWN)) continue;
                if (Integer.parseInt(c[4]) < 2 || !c[7].equals("1")) continue;
                old.put(c[1] + "\t" + c[4] + "\t" + c[6], c[8] + "\t" + c[9] + "\t" + c[10]);
            }
        }
        int maxStep = 0;
        for (List<List<String>> ds : pool.values()) if (ds.size() >= XBaseUpdate.MIN_DATES) maxStep = Math.max(maxStep, ds.size() - XBaseUpdate.START);
        int same = 0;
        int all = 0;
        for (int pp = 0; pp < 2; pp++) {
            Ctx c = XBaseUpdate.ctx(x, tpl, su.rOf, halves.get(pp), halves.get(1 - pp), pool, true, cohort, dateOf);
            Plan plan = plan(new ArrayList<>(halves.get(pp)), pool, Set.of(), null, 0, dateOf);
            List<SD> sds = simulate(c, plan, false, false, ctrl, pIdx, maxStep);
            Map<Integer, String> byIdx = new HashMap<>();
            for (Map.Entry<String, Integer> e : pIdx.entrySet()) byIdx.put(e.getValue(), e.getKey());
            for (SD sd : sds) {
                if (sd.step < 2) continue;
                for (OwnA a : sd.own) {
                    String p = byIdx.get(a.p());
                    if (pool.get(p).size() < XBaseUpdate.MIN_DATES) continue;
                    String k = (pp + 1) + "\t" + sd.step + "\t" + p;
                    String now = a.key() + "\t" + byIdx.get(a.best()) + "\t" + Double.toString(a.r());
                    all++;
                    if (now.equals(old.get(k))) same++;
                    else throw new IllegalStateException("feret-newcomer: проверка п. 2 не совпала: " + k + " – было " + old.get(k) + ", сейчас " + now);
                }
            }
        }
        if (all != old.size()) throw new IllegalStateException("feret-newcomer: проверка п. 2 – строк " + all + ", в файле №4 " + old.size());
        String r = String.format(Locale.ROOT, "совпало %d из %d строк (файл %s, %s)%n", same, old.size(), sf.getFileName(), FeiThreshold.sha(sf));
        cp.save("check", r);
        System.out.print("feret-newcomer: проверка п. 2 – " + r);
        return t.append(r).toString();
    }

    /** П. 1: доля обновлённых после шага 1 среди лучших кандидатов контрольных чужих, (б), шаг 2, порог приёма П3, против их доли в галерее. */
    static String mechanismB(Path out, List<Set<String>> halves, Map<String, List<List<String>>> pool, List<XBaseUpdate.Ctrl> ctrl, XBaseEval.Gt tpl)
            throws IOException {
        Path sf = out.resolve(XBaseSession.SCORES);
        Map<String, List<String[]>> rows = new HashMap<>();
        Map<String, Double> th = new HashMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(sf), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") || l.startsWith("proj\t")) continue;
                String[] c = l.split("\t", -1);
                if (!c[2].equals("П3") || !c[3].equals(XBaseUpdate.Var.B.label)) continue;
                String k = c[0] + "\t" + c[1] + "\t" + c[4];
                if (c[5].equals("threshold")) th.put(k, Double.parseDouble(c[10].substring(c[10].indexOf("П3 ") + 3)));
                else rows.computeIfAbsent(k, z -> new ArrayList<>()).add(c);
            }
        }
        StringBuilder t = new StringBuilder("\n=== П. 1: механизм (б) – доля обновлённых после шага 1 среди лучших кандидатов контрольных чужих, (б) M = 2, шаг 2, "
                + "порог приёма П3 (по файлу оценок feret-update " + FeiThreshold.sha(sf) + ") ===\n"
                + "проекция | обновлённых в галерее (доля) | лучших у контроля на обновлённых [95 %, бутстреп по контрольным людям] | разность\n");
        List<String> ctrlP = ctrl.stream().map(XBaseUpdate.Ctrl::person).sorted().toList();
        int[][] w = FeiBootstrap.weights(ctrlP.size(), tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        for (Proj pj : XBaseUpdate.PROJS) {
            int upd = 0;
            int galN = 0;
            Map<String, double[]> perCtrl = new TreeMap<>(); // контрольный → {на обновлённых, проб, сумма долей обновлённых}
            for (int pp = 0; pp < 2; pp++) {
                Set<String> updated = new HashSet<>();
                double t1 = th.get(pj.label() + "\t" + (pp + 1) + "\t1");
                for (String[] c : rows.get(pj.label() + "\t" + (pp + 1) + "\t1")) if (c[5].equals(FeiScores.OWN) && Double.parseDouble(c[10]) <= t1) updated.add(c[9]);
                upd += updated.size();
                galN += halves.get(pp).size();
                double share = (double) updated.size() / halves.get(pp).size();
                for (String[] c : rows.get(pj.label() + "\t" + (pp + 1) + "\t2")) {
                    if (!c[5].equals(FeiScores.CTRL)) continue;
                    double[] a = perCtrl.computeIfAbsent(c[6], z -> new double[3]);
                    if (updated.contains(c[9])) a[0]++;
                    a[1]++;
                    a[2] += share;
                }
            }
            double[] p = new double[3];
            for (double[] a : perCtrl.values()) for (int i = 0; i < 3; i++) p[i] += a[i];
            double[] reps = new double[w.length];
            double[] dreps = new double[w.length];
            for (int q = 0; q < w.length; q++) {
                double[] s = new double[3];
                for (int i = 0; i < ctrlP.size(); i++) {
                    double[] a = perCtrl.get(ctrlP.get(i));
                    if (a == null) continue;
                    for (int z = 0; z < 3; z++) s[z] += w[q][i] * a[z];
                }
                reps[q] = 100 * s[0] / s[1];
                dreps[q] = 100 * (s[0] - s[2]) / s[1];
            }
            double[] ci = FeiNorm.ci(reps);
            double[] cd = FeiNorm.ci(dreps);
            t.append(String.format(Locale.ROOT, "%s | %d из %d (%.1f %%) | %.1f %% (%d/%d) [%.1f; %.1f] | %+.1f [%+.1f; %+.1f]%s%n", pj.label(), upd, galN, 100.0 * upd / galN,
                    100 * p[0] / p[1], (int) p[0], (int) p[1], ci[0], ci[1], 100 * (p[0] - p[2]) / p[1], cd[0], cd[1], cd[0] > 0 || cd[1] < 0 ? " *" : ""));
        }
        return t.toString();
    }

    // ---------------------------------------------------------------- отчёт

    static void header(StringBuilder t, Path dir, Path exportDir, XBaseEval.Gt tpl) {
        t.append("PLAN.md п. 5.5, очередь (4): моделирование обновления шаблона, часть 2 – новички (XBaseEval feret-newcomer): Color FERET — только локально\n");
        t.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        t.append("Данные: ").append(dir).append("; когорта FEI — экспорт ").append(exportDir).append(", коммит данных ").append(FarMethods.dataCommit(exportDir))
                .append('\n');
        t.append(XBaseFeret.EQ_NOTE);
        t.append(String.format(Locale.ROOT, """
                Схема (задание №6 от 15.10.2026, план с поправками №7):
                - половины №2 (seed faces.seed + 9910); галерея – вся половина; контроль – 200 человек парой fa/fb, каждый шаг;
                - новички: на повтор r (r = 0, 1, 2) и проход – перестановка половины Random(faces.seed + %d + r), новички – первые 10 %% и
                  первые 30 %% (выбор вложенный; округление числа до целого); шаблон новичка – 1-я дата, старожила – 2 даты;
                - MLDA и 5 моделей частей «чужих» (seed faces.seed + 8000 + 100 + 7) обучаются на старте на старожилах (2 даты) + когорта FEI +
                  пул (L1-xs-F, L1-tt-xs-F), БЕЗ новичков (как в ПО: W обучена до прихода новичка); при обновлении – только центры;
                - обновление – по подтверждённой личности (решение (15)): после проб шага каждому, кто подал пробу, – сессия её даты, окно M = 3;
                  траектория от порога не зависит; шаг 1 – новички 2-я дата, старожилы 3-я; шаг 2 – новички 3-я, старожилы 4-я; у кого даты
                  нет – без пробы (статичный участник);
                - калибровка П2 / П3 (FAR 1 %%) – на каждом шаге по моделям частей, новички участвуют как «чужие» своей части и как цели;
                  проход – пара, решение по лучшему d₁/d₂;
                - (н0) новичков нет: те же W и модели частей, что у выбора; будущие новички – старожилы с 2 датами; группа «новички» в (н0) –
                  те же люди (их проба – 3-я дата), сравнивать (н1) − (н0) только по старожилам и FAR;
                - (н2) синтетическая сессия (а3, схема (в2): по синтезу на каждый снимок 1-й даты; диапазоны – люди вне своей половины и
                  контроля; Random на человека – seed faces.seed + %d + 1000·r + номер) – одна сессия окна, выбывает первой;
                - (н3) ширина оценки и порога на шаге – наименьшее число сессий у людей галереи (≤ M): берутся последние сессии;
                - (н4) свой порог новичка: если лучший кандидат – новичок, П3 (П2) – по «чужим» калибровки, у которых лучший – новичок;
                  иначе общий; (н4') справочно – по «чужим» против целей, урезанных до одной (последней) сессии;
                - (н1-W) справочно: W и модели частей обучены ещё и с 1-й датой новичков (L1-xs-F, L1-tt-xs-F, доля 30 %%, r = 0);
                - «разд.» – один порог по контрольным чужим (FAR 1 %%) на всех, ошибки по группам при нём;
                - бутстреп по людям поверх повторов и проходов: B = %d, seed = %d; «*» – интервал разности [2,5; 97,5] не содержит 0.
                """, PERM_SEED_OFFSET, SYNTH_SEED_OFFSET, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
    }

    static String volume(Map<String, List<SD>> res, Map<String, Set<String>> newOf) {
        StringBuilder t = new StringBuilder("\n=== Опись: новичков по выбору и проходу ===\nдоля | r | проход 1 | проход 2\n");
        for (int si = 0; si < SHARES.length; si++) {
            for (int r = 0; r < R; r++) {
                t.append(String.format(Locale.ROOT, "%.0f %% | %d | %d | %d%n", 100 * SHARES[si], r, newOf.get(r + "_" + si + "_0").size(), newOf.get(r + "_" + si + "_1")
                        .size()));
            }
        }
        return t.toString();
    }

    static String tables(Map<String, List<SD>> res, int persons, XBaseEval.Gt tpl, List<Proj> projs) {
        StringBuilder t = new StringBuilder();
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        for (Proj pj : projs) {
            t.append("\n\n######## ").append(pj.label()).append(" ########\n");
            Map<String, Res> rs = new LinkedHashMap<>();
            for (int si = 0; si < SHARES.length; si++) {
                for (Row row : Row.values()) {
                    if (row == Row.N1W) continue;
                    for (int j = 1; j <= STEPS; j++) {
                        List<SD> sds = steps(res.get(pj.label() + "|" + si + "|" + row.traj.name()), j);
                        rs.put(key(si, row, j, ""), evaluate(sds, row.rule, persons, w));
                    }
                }
            }
            // (н1-W) и (н1) на тех же конфигурациях (r = 0, 30 %).
            if (pj.pool()) {
                for (int j = 1; j <= STEPS; j++) {
                    rs.put(key(1, Row.N1W, j, " r0"), evaluate(steps(res.get(pj.label() + "|1|" + Traj.N1W.name() + "|r0"), j), 0, persons, w));
                    rs.put(key(1, Row.N1, j, " r0"), evaluate(steps(res.get(pj.label() + "|1|" + Traj.N1.name() + "|r0"), j), 0, persons, w));
                }
            }
            t.append("\n=== Ошибка своих по группам и FAR контроля (точка [95 %]) ===\n"
                    + "доля | строка | шаг | новичков: проб / людей | старожилов: проб / людей | ошибка новичков П3 | старожилов П3 | новичков П2 | "
                    + "старожилов П2 | FAR контроля П3 | П2 | разд.: ошибка новичков | старожилов\n");
            for (Map.Entry<String, Res> e : rs.entrySet()) {
                Res r = e.getValue();
                String[] k = e.getKey().split("\\|");
                t.append(String.join(" | ", k[0], k[1], k[2], r.nNew + " / " + r.pNew, r.nOld + " / " + r.pOld, cell(r.point[0], r.reps[0]), cell(r.point[1], r.reps[1]),
                        cell(r.point[2], r.reps[2]), cell(r.point[3], r.reps[3]), cell(r.point[4], r.reps[4]), cell(r.point[5], r.reps[5]), cell(r.point[6], r.reps[6]),
                        cell(r.point[7], r.reps[7]))).append('\n');
            }
            t.append("\n=== Парные разности, п. п. (те же реплики) ===\n"
                    + "доля | пара | шаг | ошибка новичков П3 | П2 | разд. | ошибка старожилов П3 | П2 | разд. | FAR контроля П3 | П2\n");
            for (int si = 0; si < SHARES.length; si++) {
                for (Row[] pr : new Row[][] {{Row.N2, Row.N1}, {Row.N3, Row.N1}, {Row.N4, Row.N1}, {Row.N4T, Row.N1}, {Row.N1, Row.N0}}) {
                    for (int j = 1; j <= STEPS; j++) {
                        Res a = rs.get(key(si, pr[0], j, ""));
                        Res b = rs.get(key(si, pr[1], j, ""));
                        t.append(diffRow(String.format(Locale.ROOT, "%.0f %%", 100 * SHARES[si]), pr[0].label + " − " + pr[1].label, j, a, b, pr[1] == Row.N0));
                    }
                }
            }
            if (pj.pool()) {
                for (int j = 1; j <= STEPS; j++) {
                    t.append(diffRow("30 %, r = 0", Row.N1W.label + " − " + Row.N1.label, j, rs.get(key(1, Row.N1W, j, " r0")), rs.get(key(1, Row.N1, j, " r0")), false));
                }
            }
            t.append(hubs(res, pj));
            t.append(calib(res, pj));
            t.append(session(res, pj, w));
        }
        return t.toString();
    }

    static String key(int si, Row row, int j, String tag) {
        return String.format(Locale.ROOT, "%.0f %%%s|%s|%d", 100 * SHARES[si], tag, row.label, j);
    }

    static List<SD> steps(List<SD> sds, int j) {
        return sds.stream().filter(s -> s.step == j).toList();
    }

    static String diffRow(String share, String pair, int j, Res a, Res b, boolean onlyOld) {
        String n = onlyOld ? "–" : null;
        return String.join(" | ", share, pair, Integer.toString(j), n != null ? n : diffCell(a, b, 0), n != null ? n : diffCell(a, b, 2), n != null ? n : diffCell(a, b, 6),
                diffCell(a, b, 1), diffCell(a, b, 3), diffCell(a, b, 7), diffCell(a, b, 4), diffCell(a, b, 5)) + "\n";
    }

    /** Хабы по группам: лучшие у контроля на новичках против доли новичков в галерее; анти-хабы; два самых частых. */
    static String hubs(Map<String, List<SD>> res, Proj pj) {
        StringBuilder t = new StringBuilder("\n=== Хабы у контрольных чужих по группам (все конфигурации доли вместе) ===\n"
                + "доля | траектория | шаг | лучших на новичках / доля новичков в галерее | анти-хабов: новичков (из) | старожилов (из) | "
                + "два самых частых лучших (доля проб; из них новичков)\n");
        for (int si = 0; si < SHARES.length; si++) {
            for (Traj tr : List.of(Traj.N0, Traj.N1, Traj.N2, Traj.N3)) {
                for (int j = 1; j <= STEPS; j++) {
                    List<SD> sds = steps(res.get(pj.label() + "|" + si + "|" + tr.name()), j);
                    int onNew = 0;
                    int n = 0;
                    int galNew = 0;
                    int gal = 0;
                    int antiNew = 0;
                    int antiOld = 0;
                    int newN = 0;
                    int oldN = 0;
                    int top2 = 0;
                    int top2New = 0;
                    for (SD sd : sds) {
                        Map<Integer, Integer> cnt = new HashMap<>();
                        for (CtrlA a : sd.ctrl) {
                            cnt.merge(a.best(), 1, Integer::sum);
                            if (a.bestNew()) onNew++;
                            n++;
                        }
                        for (int i = 0; i < sd.galP.length; i++) {
                            gal++;
                            boolean isNew = sd.galNew[i];
                            if (isNew) {
                                galNew++;
                                newN++;
                            } else {
                                oldN++;
                            }
                            if (!cnt.containsKey(sd.galP[i])) {
                                if (isNew) antiNew++;
                                else antiOld++;
                            }
                        }
                        Set<Integer> newSet = new HashSet<>();
                        for (int i = 0; i < sd.galP.length; i++) if (sd.galNew[i]) newSet.add(sd.galP[i]);
                        List<Map.Entry<Integer, Integer>> s = new ArrayList<>(cnt.entrySet());
                        s.sort((a, b) -> b.getValue() - a.getValue() != 0 ? b.getValue() - a.getValue() : a.getKey() - b.getKey());
                        for (int i = 0; i < Math.min(2, s.size()); i++) {
                            top2 += s.get(i).getValue();
                            if (newSet.contains(s.get(i).getKey())) top2New++;
                        }
                    }
                    t.append(String.format(Locale.ROOT, "%.0f %% | %s | %d | %.1f %% / %.1f %% | %d (%d) | %d (%d) | %.1f %%; %d из %d%n", 100 * SHARES[si], tr.label, j,
                            100.0 * onNew / n, 100.0 * galNew / gal, antiNew, newN, antiOld, oldN, 100.0 * top2 / n, top2New, 2 * sds.size()));
                }
            }
        }
        return t.toString();
    }

    /** Калибровка: доля новичков среди «чужих» (люди части) и среди целей (лучший кандидат); «чужих» с лучшим-новичком (н4, н4'). */
    static String calib(Map<String, List<SD>> res, Proj pj) {
        StringBuilder t = new StringBuilder("\n=== Калибровка П2 / П3: новички среди «чужих» и целей; объём подмножества (н4) ===\n"
                + "доля | траектория | шаг | «чужих» всего | из них от новичков | целей-новичков (лучший – новичок) | на конфигурацию (н4): мин. – макс. | "
                + "(н4') целей-новичков всего\n");
        for (int si = 0; si < SHARES.length; si++) {
            for (Traj tr : List.of(Traj.N1, Traj.N2, Traj.N3)) {
                for (int j = 1; j <= STEPS; j++) {
                    List<SD> sds = steps(res.get(pj.label() + "|" + si + "|" + tr.name()), j);
                    int all = 0;
                    int fromNew = 0;
                    int toNew = 0;
                    int mn = Integer.MAX_VALUE;
                    int mx = 0;
                    int toNewT = 0;
                    for (SD sd : sds) {
                        int k = 0;
                        for (ImpA a : sd.imp) {
                            all++;
                            if (a.ownerNew()) fromNew++;
                            if (a.bestNew()) {
                                toNew++;
                                k++;
                            }
                        }
                        mn = Math.min(mn, k);
                        mx = Math.max(mx, k);
                        for (ImpA a : sd.impT) if (a.bestNew()) toNewT++;
                    }
                    t.append(String.format(Locale.ROOT, "%.0f %% | %s | %d | %d | %d (%.1f %%) | %d (%.1f %%) | %d – %d | %s%n", 100 * SHARES[si], tr.label, j, all, fromNew,
                            100.0 * fromNew / all, toNew, 100.0 * toNew / all, mn, mx, tr == Traj.N1 ? Integer.toString(toNewT) : "–"));
                }
            }
        }
        return t.toString();
    }

    /** Мера общей сессии у контрольных чужих: лучший – новичок / старожил; ожидаемое – доля людей той же группы с датой в шаблоне. */
    static String session(Map<String, List<SD>> res, Proj pj, int[][] w) {
        StringBuilder t = new StringBuilder("\n=== Общая сессия у контрольных чужих: дата чужого в шаблоне лучшего кандидата (наблюдаемое / ожидаемое при случайном "
                + "кандидате той же группы) ===\n"
                + "доля | траектория | шаг | лучший – новичок: проб, наблюдаемое / ожидаемое | лучший – старожил: проб, наблюдаемое / ожидаемое, разность [95 %]\n");
        for (int si = 0; si < SHARES.length; si++) {
            for (Traj tr : List.of(Traj.N0, Traj.N1, Traj.N2, Traj.N3)) {
                for (int j = 1; j <= STEPS; j++) {
                    List<SD> sds = steps(res.get(pj.label() + "|" + si + "|" + tr.name()), j);
                    double[][] s = new double[2][3];
                    Map<Integer, double[]> byP = new HashMap<>();
                    for (SD sd : sds) {
                        for (CtrlA a : sd.ctrl) {
                            // (н0): ожидаемое посчитано по всей галерее – без деления на группы.
                            int g = tr != Traj.N0 && a.bestNew() ? 0 : 1;
                            s[g][0]++;
                            if (a.same()) s[g][1]++;
                            s[g][2] += g == 0 ? a.expNew() : a.expOld();
                            if (g == 1) {
                                double[] b = byP.computeIfAbsent(a.p(), z -> new double[3]);
                                b[0]++;
                                if (a.same()) b[1]++;
                                b[2] += a.expOld();
                            }
                        }
                    }
                    double[] reps = new double[w.length];
                    for (int q = 0; q < w.length; q++) {
                        double[] z = new double[3];
                        for (Map.Entry<Integer, double[]> e : byP.entrySet()) for (int i = 0; i < 3; i++) z[i] += w[q][e.getKey()] * e.getValue()[i];
                        reps[q] = sameExcess(z);
                    }
                    double[] ci = w.length > 0 ? FeiNorm.ci(reps) : new double[] {Double.NaN, Double.NaN};
                    String oldCell = s[1][0] == 0 ? "–" : String.format(Locale.ROOT, "%d, %.1f %% / %.1f %%, %+.1f%s", (int) s[1][0], 100 * s[1][1] / s[1][0],
                            100 * s[1][2] / s[1][0], 100 * (s[1][1] - s[1][2]) / s[1][0], s[1][0] < XBaseSession.MIN_N ? " (мало проб, без интервала)"
                                    : String.format(Locale.ROOT, " [%+.1f; %+.1f]", ci[0], ci[1]) + FeiBootstrap.note(reps));
                    t.append(String.format(Locale.ROOT, "%.0f %% | %s | %d | %s | %s%n", 100 * SHARES[si], tr.label + (tr == Traj.N0 ? " (вся галерея)" : ""), j,
                            s[0][0] == 0 ? "–" : String.format(Locale.ROOT, "%d, %.1f %% / %.1f %%", (int) s[0][0], 100 * s[0][1] / s[0][0], 100 * s[0][2] / s[0][0]),
                            oldCell));
                }
            }
        }
        return t.toString();
    }
}
