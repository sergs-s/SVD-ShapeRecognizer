package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
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
import java.util.stream.IntStream;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.ProkaevMethod.Variant;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№1 (ред. 4; план Васи от 19.10.2026, принят Жориком с поправками, PLAN.md п. 5.5, (9) (а2г), Д2–Д8): метод Прокаева
 * ({@link ProkaevMethod}) на протоколах (4а) – справочная строка, как SFace, в правило отключений не входит.
 * <p>
 * Роли, галереи, пробы, контроль и части «чужих» П3 – те же, что у базы (4а) (XBaseAbl; здесь те же вызовы, XBaseAbl не меняется);
 * до счёта – сверка наборов проб и частей с блоками базы в файле оценок (4а), расхождение – остановка. Все снимки галереи – точками.
 * Кадр – {@link ProkaevFrames} 400 × 600; справочно – кадр «а» 92×112. Гости – по одному снимку на человека, вне своих, контроля и
 * «чужих» порога (проверка в коде): FERET – люди другой половины №2 (пула прохода), GT – люди FERET (условные номера FG01…; имена
 * снимков FERET – только в каталоге FERET); 10 / 20 / 30 (вложенные), seed faces.seed + 9976 + номер протокола; справочно – 10
 * человек FEI (seed + 9975) и 10 гостей по правилу автора (пул 20 человек по 3 снимка вне гостей и проб; оценка – доля снимков с
 * долей своего ≥ 0,95 без гостей; верхние 10 человек, по их самому «принимаемому» снимку). Оценка для наших мер – 1 − доля лучшего
 * своего, ничьи – по d₁ (в файле оценок – обе через составную оценку; второе расстояние – постоянная, как (4) XBaseAbl); точка по
 * правилу автора (0,95) – отдельно.
 * <p>
 * Одна матрица Грама на задание: строки обучения всех моделей (галерея, гости всех наборов, пул отбора), скалярные произведения проб
 * с ними; каждая модель – собственные векторы подматрицы.
 *
 * @author ssv
 */
final class ProkaevEval {

    static final int K = 6;
    static final int L = 3;
    static final int KMAX = 20;
    static final long GUEST_SEED = 9976;
    static final long FEI_SEED = 9975;
    static final long AUTH_SEED = 9977;
    static final long HALF_SEED = 9978;
    static final int AUTH_PEOPLE = 20;
    static final int AUTH_IMAGES = 3;
    static final int AUTH_TOP = 10;
    static final String BASE_ROW = "база (4а) L1-xs-F";
    static final String BASE_ROW_TT = "база (4а) L1-tt-xs-F";
    static final String MAIN = "метод Прокаева (Б2-К), гости 10";
    /** FACES_PROKAEV_ONLY=main – считается только основная строка (Д-№3); иначе все строки cfgs(). */
    static final Set<Integer> ONLY = "main".equals(System.getenv("FACES_PROKAEV_ONLY")) ? Set.of(0) : null;

    enum Gs {
        NONE, G10, G20, G30, FEI, AUTH, HALF
    }

    /** Строка: подпись, набор гостей, базис по галерее, кадр «а», вариант, k, l, оценка d₁/d₂ (справочно). */
    record Cfg(String row, Gs gs, boolean galBasis, boolean frameA, Variant v, int k, int l, boolean ratio) {}

    static List<Cfg> cfgs() {
        List<Cfg> c = new ArrayList<>();
        c.add(new Cfg(MAIN, Gs.G10, false, false, Variant.K, K, L, false));
        c.add(new Cfg("метод Прокаева, гости 20", Gs.G20, false, false, Variant.K, K, L, false));
        c.add(new Cfg("метод Прокаева, гости 30", Gs.G30, false, false, Variant.K, K, L, false));
        c.add(new Cfg("Б2-н (справочно)", Gs.G10, false, false, Variant.N, K, L, false));
        c.add(new Cfg("Б2-с (справочно)", Gs.G10, false, false, Variant.S, K, L, false));
        c.add(new Cfg("без гостей (справочно)", Gs.NONE, false, false, Variant.K, K, L, false));
        c.add(new Cfg("базис только по галерее (справочно)", Gs.G10, true, false, Variant.K, K, L, false));
        c.add(new Cfg("гости FEI 10 – другая съёмка (справочно)", Gs.FEI, false, false, Variant.K, K, L, false));
        c.add(new Cfg("гости по правилу автора 10 (справочно)", Gs.AUTH, false, false, Variant.K, K, L, false));
        for (int k : new int[] {3, 4, 5, 10, 20}) c.add(new Cfg("k = " + k + " (НЕ ДЛЯ ВЫБОРА)", Gs.G10, false, false, Variant.K, k, L, false));
        for (int l : new int[] {1, 5}) c.add(new Cfg("l = " + l + " (НЕ ДЛЯ ВЫБОРА)", Gs.G10, false, false, Variant.K, K, l, false));
        c.add(new Cfg("кадр «а» 92×112 (справочно)", Gs.G10, false, true, Variant.K, K, L, false));
        c.add(new Cfg("d₁/d₂ в пространстве метода – наше решающее правило на проекции метода, не метод Прокаева (справочно)", Gs.G10, false, false,
                Variant.K, K, L, true));
        c.add(new Cfg("гости GT, контроль – половины (справочно)", Gs.HALF, false, false, Variant.K, K, L, false));
        return c;
    }

    /** Задание: постановка, проход, роли базы (4а), гости, строки базы из файла (4а). */
    static final class Job {
        String conf;
        int pass;
        int proto;
        XBaseEval.Gt g;
        Map<String, List<String>> gal;
        List<Probe> probes;
        List<List<String>> parts;
        int s;
        List<String> g30 = new ArrayList<>();
        List<String> fei10 = new ArrayList<>();
        List<List<String>> authPool = new ArrayList<>();
        List<String> auth = new ArrayList<>();
        List<String> half = new ArrayList<>();
        /** Номер j строк оценок (FEI – номер контрольного снимка; прочие – 0). */
        int cfgJ;
        String baseL1;
        /** Строка базы для пары (подпись) и прочие строки рядом: {подпись, строки оценок, вариант}. */
        String pairBase = BASE_ROW;
        List<Object[]> extraBases = new ArrayList<>();
        /** Только строка «контроль – половины» (GT справочно). */
        boolean halfOnly;
        /** Считать только строки cfgs() с этими номерами (п. 5 дуэли); null – все. */
        Set<Integer> only;

        List<String> guests(Gs gs) {
            return switch (gs) {
                case NONE -> List.of();
                case G10 -> g30.subList(0, Math.min(10, g30.size()));
                case G20 -> g30.subList(0, Math.min(20, g30.size()));
                case G30 -> g30;
                case FEI -> fei10;
                case AUTH -> auth;
                case HALF -> half;
            };
        }
    }

    /** Счёт: накопители строк (вариант L1 разбора), тальи правила автора, строки оценок. */
    static final class Run {
        final Map<String, XBaseAbl.Acc> acc = new LinkedHashMap<>();
        final Map<String, long[]> rule = new LinkedHashMap<>();
        final Map<String, Set<String>> ruleCtrlPersons = new HashMap<>();
        final Map<String, Set<String>> ruleCtrlAll = new HashMap<>();
        final StringBuilder lines = new StringBuilder();
        final StringBuilder check = new StringBuilder();
        final StringBuilder guests = new StringBuilder();
        final List<int[]> galSizes = new ArrayList<>();
        long peak;
        final Set<String> bases = new HashSet<>();
        /** Свой разбор строк (FEI): {задание, подпись, строки}; null – разбор протоколов (4а). */
        java.util.function.Consumer<Object[]> parse;
    }

    // ---------------------------------------------------------------- векторы

    /** Строки пикселей кадров 400 × 600 (и кадра «а») по ключам. */
    static final class Vecs {
        final Map<String, float[]> big = new HashMap<>();
        final Map<String, float[]> small = new HashMap<>();
        int clipped;
    }

    static void addBig(Vecs v, String key, Mat color, double[] row) {
        int[] c = new int[1];
        Mat f = ProkaevFrames.crop(color, row, c);
        v.clipped += c[0];
        v.big.put(key, ProkaevFrames.row(f));
        f.release();
    }

    static void addSmall(Vecs v, String key, Mat frameA) {
        v.small.put(key, ProkaevFrames.row(frameA));
        frameA.release();
    }

    // ---------------------------------------------------------------- счёт задания

    static double[][] gram(List<float[]> t) {
        int m = t.size();
        double[][] g = new double[m][m];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = i; j < m; j++) g[i][j] = ProkaevMethod.dot(t.get(i), t.get(j));
        });
        for (int i = 0; i < m; i++) for (int j = 0; j < i; j++) g[i][j] = g[j][i];
        return g;
    }

    static double[][] dots(List<float[]> z, List<float[]> t) {
        double[][] d = new double[z.size()][t.size()];
        IntStream.range(0, z.size()).parallel().forEach(i -> {
            for (int j = 0; j < t.size(); j++) d[i][j] = ProkaevMethod.dot(z.get(i), t.get(j));
        });
        return d;
    }

    /** Данные задания по набору кадров: ключи обучения T (индексы), Грам, произведения проб основного протокола. */
    record Mats(Map<String, Integer> idx, double[][] g, Map<String, double[]> probeDots) {}

    static Mats mats(Job j, Map<String, float[]> vec) {
        List<String> t = new ArrayList<>();
        for (String p : j.g.own) t.addAll(j.gal.get(p));
        for (Gs gs : Gs.values()) for (String k : j.guests(gs)) if (!t.contains(k)) t.add(k);
        for (List<String> pp : j.authPool) for (String k : pp) if (!t.contains(k)) t.add(k);
        Map<String, Integer> idx = new LinkedHashMap<>();
        List<float[]> rows = new ArrayList<>();
        for (String k : t) {
            float[] v = vec.get(k);
            if (v == null) throw new IllegalStateException("prokaev: нет кадра для " + k);
            idx.put(k, rows.size());
            rows.add(v);
        }
        double[][] g = gram(rows);
        List<String> pk = new ArrayList<>();
        for (Probe q : j.probes) if (!idx.containsKey(q.key()) && !pk.contains(q.key())) pk.add(q.key());
        List<float[]> pr = new ArrayList<>();
        for (String k : pk) {
            float[] v = vec.get(k);
            if (v == null) throw new IllegalStateException("prokaev: нет кадра пробы " + k);
            pr.add(v);
        }
        double[][] d = dots(pr, rows);
        Map<String, double[]> pd = new HashMap<>();
        for (int i = 0; i < pk.size(); i++) pd.put(pk.get(i), d[i]);
        for (Map.Entry<String, Integer> e : idx.entrySet()) pd.put(e.getKey(), g[e.getValue()]);
        return new Mats(idx, g, pd);
    }

    /** Модель: галерея своих без части q (q = −1 – все), гости набора gs (в базисе или точками). */
    static final class Model {
        ProkaevMethod pm;
        int[] basis;
    }

    static Model model(Job j, Mats mt, Gs gs, boolean galBasis, int q) {
        Set<String> part = q < 0 ? Set.of() : new HashSet<>(j.parts.get(q));
        List<Integer> b = new ArrayList<>();
        List<String> bl = new ArrayList<>();
        for (String p : j.g.own) {
            if (part.contains(p)) continue;
            for (String k : j.gal.get(p)) {
                b.add(mt.idx().get(k));
                bl.add(p);
            }
        }
        List<Integer> ex = new ArrayList<>();
        for (String k : j.guests(gs)) {
            if (galBasis) ex.add(mt.idx().get(k));
            else {
                b.add(mt.idx().get(k));
                bl.add(ProkaevMethod.GUEST);
            }
        }
        int m = b.size();
        double[][] g = new double[m][m];
        for (int x = 0; x < m; x++) for (int y = 0; y < m; y++) g[x][y] = mt.g()[b.get(x)][b.get(y)];
        double[][] ed = new double[ex.size()][m];
        for (int x = 0; x < ex.size(); x++) for (int y = 0; y < m; y++) ed[x][y] = mt.g()[ex.get(x)][b.get(y)];
        String[] el = new String[ex.size()];
        java.util.Arrays.fill(el, ProkaevMethod.GUEST);
        Model md = new Model();
        md.pm = ProkaevMethod.fitGram(g, bl.toArray(String[]::new), ed, el, KMAX);
        md.basis = b.stream().mapToInt(Integer::intValue).toArray();
        return md;
    }

    static ProkaevMethod.Result decide(Model md, double[] dotsAll, Cfg c) {
        double[] d = new double[md.basis.length];
        for (int i = 0; i < d.length; i++) d[i] = dotsAll[md.basis[i]];
        double[] w = md.pm.projectDots(d, c.v());
        return md.pm.decide(w, c.v(), Math.min(c.k(), md.pm.axes()), c.l());
    }

    /** Оценки по людям галереи (меньше – ближе): основная – 1 + (1 − доля) + 1e-6·d/(1 + d); d₁/d₂ – ближайшие точки. */
    static double[] scores(ProkaevMethod.Result r, List<String> gallery, boolean ratio) {
        double[] v = new double[gallery.size()];
        for (int i = 0; i < v.length; i++) {
            String p = gallery.get(i);
            double near = r.near().getOrDefault(p, Double.POSITIVE_INFINITY);
            v[i] = ratio ? near : 1 + (1 - r.share().getOrDefault(p, 0.0)) + 1e-6 * near / (1 + near);
        }
        if (ratio && r.nn1Class() != null) {
            // Второе расстояние – ближайшая точка любого другого класса, включая гостей.
            int c1 = gallery.indexOf(r.nn1Class());
            int o = -1;
            for (int i = 0; i < v.length; i++) if (i != c1 && (o < 0 || v[i] < v[o])) o = i;
            if (o >= 0 && r.nn2() < v[o]) v[o] = r.nn2();
        }
        return v;
    }

    /** Счёт задания и разбор: compute – текст (строки оценок и решения по правилу автора; он же – контрольная точка), absorb – разбор. */
    static void job(Job j, Vecs vecs, Run run) {
        absorb(j, compute(j, vecs), run);
    }

    /**
     * Текст задания: «## i» – строка cfgs()[i], затем строки оценок, затем «#R роль человек гость(0/1) не_тот(0/1)» по основным пробам;
     * «#G …» – запись отбора гостей по правилу автора.
     */
    static String compute(Job j, Vecs vecs) {
        StringBuilder out = new StringBuilder();
        Mats big = mats(j, vecs.big);
        Mats small = null;
        List<Cfg> cl = cfgs();
        for (int ci = 0; ci < cl.size(); ci++) if (cl.get(ci).frameA() && !j.halfOnly && (j.only == null || j.only.contains(ci))) small = mats(j, vecs.small);
        // Отбор гостей по правилу автора: модель без гостей, оценка – доля снимков пула с долей своего ≥ 0,95.
        if (!j.authPool.isEmpty() && j.auth.isEmpty()) {
            Model m0 = model(j, big, Gs.NONE, false, -1);
            Cfg c0 = new Cfg("", Gs.NONE, false, false, Variant.K, K, L, false);
            List<Object[]> rank = new ArrayList<>();
            for (List<String> pp : j.authPool) {
                int acc = 0;
                double sum = 0;
                String bestK = null;
                double bestS = -1;
                for (String k : pp) {
                    ProkaevMethod.Result r = decide(m0, big.probeDots().get(k), c0);
                    if (r.bestShare() >= ProkaevMethod.RULE) acc++;
                    sum += r.bestShare();
                    if (r.bestShare() > bestS) {
                        bestS = r.bestShare();
                        bestK = k;
                    }
                }
                rank.add(new Object[] {(double) acc / pp.size(), sum / pp.size(), bestK});
            }
            rank.sort((a, b) -> {
                int x = Double.compare((double) b[0], (double) a[0]);
                return x != 0 ? x : Double.compare((double) b[1], (double) a[1]);
            });
            for (int i = 0; i < Math.min(AUTH_TOP, rank.size()); i++) j.auth.add((String) rank.get(i)[2]);
            out.append(String.format(Locale.ROOT, "#G\t%s, проход %d: отбор по правилу автора – доля принятых у верхних %d: %s%n", j.conf, j.pass + 1,
                    Math.min(AUTH_TOP, rank.size()), rank.subList(0, Math.min(AUTH_TOP, rank.size())).stream().map(a -> String.format(Locale.ROOT, "%.2f", (double) a[0]))
                            .toList()));
        }
        Map<String, Model> cache = new HashMap<>();
        List<Cfg> all = cfgs();
        for (int ci = 0; ci < all.size(); ci++) {
            Cfg c = all.get(ci);
            if (j.halfOnly != (c.gs() == Gs.HALF)) continue;
            if (c.gs() == Gs.AUTH && j.auth.isEmpty()) continue;
            if (j.only != null && !j.only.contains(ci)) continue;
            Mats mt = c.frameA() ? small : big;
            String method = "prokaev-" + ci;
            out.append("## ").append(ci).append('\n');
            StringBuilder rule = new StringBuilder();
            for (int q = -1; q < FeiMethods.CV_PARTS; q++) {
                if (q >= 0 && j.parts == null) break;
                String key = c.gs() + "|" + c.galBasis() + "|" + c.frameA() + "|" + q;
                final int qq = q;
                Model md = cache.computeIfAbsent(key, x -> model(j, mt, c.gs(), c.galBasis(), qq));
                Set<String> part = q < 0 ? Set.of() : new HashSet<>(j.parts.get(q));
                List<String> gallery = j.g.own.stream().filter(p -> !part.contains(p)).toList();
                List<Probe> pr = new ArrayList<>();
                if (q < 0) pr.addAll(j.probes);
                else for (String p : j.g.own) if (part.contains(p)) for (String k : j.gal.get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, j.g.img(k), k, -1));
                for (Probe p : pr) {
                    ProkaevMethod.Result r = decide(md, mt.probeDots().get(p.key()), c);
                    out.append(FeiScores.line(method, j.s, j.cfgJ, p.role(), p.person(), p.image(), p.trueIdx(), gallery, scores(r, gallery, c.ratio()), false));
                    if (q < 0 && !c.ratio()) rule.append("#R\t").append(p.role()).append('\t').append(p.person()).append('\t').append(r.guest() ? 1 : 0).append('\t')
                            .append(!r.guest() && !r.best().equals(p.person()) ? 1 : 0).append('\n');
                }
            }
            out.append(rule);
        }
        return out.toString();
    }

    /** Разбор текста задания: накопители строк, правило автора, строки базы, размеры галереи. */
    static void absorb(Job j, String text, Run run) {
        List<Cfg> all = cfgs();
        Map<Integer, StringBuilder> blocks = new LinkedHashMap<>();
        Map<Integer, List<String[]>> rules = new LinkedHashMap<>();
        Integer cur = null;
        for (String l : text.split("\n")) {
            if (l.isEmpty()) continue;
            if (l.startsWith("#G\t")) {
                run.guests.append(l.substring(3)).append('\n');
            } else if (l.startsWith("## ")) {
                cur = Integer.parseInt(l.substring(3).trim());
                blocks.put(cur, new StringBuilder());
                rules.put(cur, new ArrayList<>());
            } else if (l.startsWith("#R\t")) {
                rules.get(cur).add(l.split("\t", -1));
            } else {
                blocks.get(cur).append(l).append('\n');
            }
        }
        for (Map.Entry<Integer, StringBuilder> e : blocks.entrySet()) {
            Cfg c = all.get(e.getKey());
            String method = "prokaev-" + e.getKey();
            String ls = e.getValue().toString();
            run.lines.append("# ").append(j.conf).append(", проход ").append(j.pass + 1).append(", ").append(c.row()).append('\n').append(ls);
            String forParse = ls.replace(method + "\t", V.L1.id + "\t");
            if (!c.ratio()) forParse = XBaseAbl.abs(forParse);
            String label = j.conf + " " + (c.gs() == Gs.HALF ? "гости GT, контроль – половины, со сводкой (4а) не сопоставимо напрямую (справочно)" : c.row());
            if (run.parse != null) {
                run.parse.accept(new Object[] {j, label, forParse});
            } else {
                XBaseEval.Gt gp = XBaseAbl.copy(j.g, j.g.cohort, j.g.rOf, Double.POSITIVE_INFINITY);
                run.acc.computeIfAbsent(label, x -> new XBaseAbl.Acc(V.L1)).add(XBaseEval.parse(gp, EnumSet.of(V.L1), j.gal, j.probes, j.parts, forParse));
            }
            String lab = j.conf + " " + c.row();
            for (String[] rr : rules.get(e.getKey())) {
                long[] t = run.rule.computeIfAbsent(lab, x -> new long[6]);
                boolean guest = rr[3].equals("1");
                if (rr[1].equals(FeiScores.OWN)) {
                    t[0]++;
                    if (guest) t[1]++;
                    else if (rr[4].equals("1")) t[2]++;
                } else if (rr[1].equals(FeiScores.CTRL)) {
                    t[3]++;
                    run.ruleCtrlAll.computeIfAbsent(lab, x -> new HashSet<>()).add(j.pass + "|" + rr[2]);
                    if (!guest) {
                        t[4]++;
                        run.ruleCtrlPersons.computeIfAbsent(lab, x -> new HashSet<>()).add(j.pass + "|" + rr[2]);
                    }
                }
            }
        }
        run.peak = Math.max(run.peak, Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        if (!j.halfOnly && j.baseL1 != null) {
            XBaseEval.Gt gb = XBaseAbl.copy(j.g, j.g.cohort, j.g.rOf, j.g.r0);
            run.bases.add(j.conf + " " + j.pairBase);
            run.acc.computeIfAbsent(j.conf + " " + j.pairBase, x -> new XBaseAbl.Acc(V.L1)).add(XBaseEval.parse(gb, EnumSet.of(V.L1), j.gal, j.probes, j.parts, j.baseL1));
            for (Object[] eb : j.extraBases) {
                V v = (V) eb[2];
                run.bases.add(j.conf + " " + eb[0]);
                run.acc.computeIfAbsent(j.conf + " " + eb[0], x -> new XBaseAbl.Acc(v)).add(XBaseEval.parse(gb, EnumSet.of(v), j.gal, j.probes, j.parts, (String) eb[1]));
            }
        }
        if (!j.halfOnly) {
            int[] sz = j.g.own.stream().mapToInt(p -> j.gal.get(p).size()).sorted().toArray();
            run.galSizes.add(new int[] {sz[0], sz[sz.length / 2], sz[sz.length - 1]});
        }
    }

    /** Сверка ролей с блоком базы (4а): пробы и части «чужих» (роль, человек, снимок) совпадают; GT – пробы ⊆ блока (там ещё пробы U). */
    static void checkRoles(Run run, Job j, boolean subset) {
        Set<String> base = new TreeSet<>();
        for (String l : j.baseL1.split("\n")) {
            if (l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            if (c[0].equals(V.L1.id)) base.add(c[3] + "\t" + c[4] + "\t" + c[5]);
        }
        Set<String> mine = new TreeSet<>();
        for (Probe q : j.probes) mine.add(q.role() + "\t" + q.person() + "\t" + q.image());
        for (int q = 0; q < FeiMethods.CV_PARTS; q++) for (String p : j.parts.get(q)) for (String k : j.gal.get(p)) mine.add(FeiMethods.CV_ROLE + q + "\t" + p + "\t" + j.g.img(k));
        boolean ok = subset ? base.containsAll(mine) : base.equals(mine);
        run.check.append(String.format(Locale.ROOT, "%s, проход %d: пробы и части «чужих» – %d, в блоке базы (4а) – %d: %s%n", j.conf, j.pass + 1, mine.size(), base.size(),
                ok ? (subset ? "все в блоке базы" : "совпали") : "НЕ СОВПАЛИ"));
        if (!ok) throw new IllegalStateException("prokaev: роли не совпали с (4а): " + j.conf + ", проход " + (j.pass + 1));
    }

    /** Пересечения гостей и пула отбора со своими, контролем, частями «чужих» и пробами – исключение. */
    static void checkGuests(Job j, Map<String, String> personOf) {
        Set<String> own = new HashSet<>(j.g.own);
        Set<String> ctrl = new HashSet<>(j.g.ctrl);
        Set<String> probeKeys = new HashSet<>();
        for (Probe q : j.probes) probeKeys.add(q.key());
        Set<String> guestPersons = new HashSet<>();
        for (Gs gs : new Gs[] {Gs.G30, Gs.FEI, Gs.HALF}) {
            for (String k : j.guests(gs)) {
                String p = personOf.get(k);
                if (p == null) throw new IllegalStateException("prokaev: неизвестен человек гостя " + k);
                if (own.contains(p) || ctrl.contains(p) || probeKeys.contains(k)) throw new IllegalStateException("prokaev: гость пересекается со своими, контролем или пробами: " + p);
                guestPersons.add(p);
            }
        }
        for (List<String> pp : j.authPool) {
            for (String k : pp) {
                String p = personOf.get(k);
                if (own.contains(p) || ctrl.contains(p) || probeKeys.contains(k) || guestPersons.contains(p))
                    throw new IllegalStateException("prokaev: пул отбора пересекается с гостями, своими, контролем или пробами: " + p);
            }
        }
    }

    /** Гости из людей persons (порядок seed): 30 гостей по одному снимку, затем пул отбора 20 человек по 3 снимка. */
    static void pickGuests(Job j, List<String> persons, Map<String, List<String>> keysOf, long seed) {
        List<String> ps = new ArrayList<>(persons);
        Collections.sort(ps);
        Random rnd = new Random(seed + GUEST_SEED + j.proto);
        Collections.shuffle(ps, rnd);
        for (int i = 0; i < ps.size() && j.g30.size() < 30; i++) {
            List<String> ks = new ArrayList<>(keysOf.get(ps.get(i)));
            Collections.sort(ks);
            j.g30.add(ks.get(rnd.nextInt(ks.size())));
        }
        Random ra = new Random(seed + AUTH_SEED + j.proto);
        for (int i = 30; i < ps.size() && j.authPool.size() < AUTH_PEOPLE; i++) {
            List<String> ks = new ArrayList<>(keysOf.get(ps.get(i)));
            Collections.sort(ks);
            Collections.shuffle(ks, ra);
            j.authPool.add(new ArrayList<>(ks.subList(0, Math.min(AUTH_IMAGES, ks.size()))));
        }
    }

    /** Гости FEI: 10 человек вне когорты (seed + 9975), по одному снимку. */
    static List<String> feiGuests(Map<String, Map<Integer, String>> feiKeys, List<String> cohortP, long seed, Map<String, String> personOf) {
        List<String> ps = new ArrayList<>(new TreeSet<>(feiKeys.keySet()));
        ps.removeAll(cohortP);
        Random rnd = new Random(seed + FEI_SEED);
        Collections.shuffle(ps, rnd);
        List<String> out = new ArrayList<>();
        for (String p : ps.subList(0, 10)) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            Collections.sort(ks);
            String k = ks.get(rnd.nextInt(ks.size()));
            out.add(k);
            personOf.put(k, "FEI " + p);
        }
        return out;
    }

    // ---------------------------------------------------------------- таблицы

    static void tables(StringBuilder r, StringBuilder sum, Run run, int persons, XBaseEval.Gt tpl, List<String> confs, String base, String pairBase) {
        int[][] w = tpl.b > 0 ? FeiBootstrap.weights(persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons);
        for (String conf : confs) {
            Map<String, FeiThreshold.Res> res = new LinkedHashMap<>();
            Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
            List<String[]> pairs = new ArrayList<>();
            for (Map.Entry<String, XBaseAbl.Acc> e : run.acc.entrySet()) {
                if (!e.getKey().startsWith(conf + " ")) continue;
                XBaseAbl.Acc a = e.getValue();
                res.put(e.getKey(), FeiThreshold.evaluate(e.getKey(), V.L1.ratioId(), a.cfgs, a.imps, persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
                sep.put(e.getKey(), XBaseOther.sep(a.cfgs, unit, w));
            }
            String b = conf + " " + pairBase;
            for (String l : res.keySet()) if (!run.bases.contains(l) && !l.contains("половины") && res.containsKey(b)) pairs.add(new String[] {l, b});
            r.append("\n\n######## ").append(base).append(": ").append(conf).append(" ########\n");
            XBaseOther.tables(r, res, sep, tpl.b, pairs);
            // Точка по правилу автора.
            r.append("\n=== Точка по правилу автора (0,95): гость – отказ ===\n");
            r.append("строка | проб своих | отвергнуто своих (гость) | принят не тот | верно принято | контрольных снимков | FAR контроля по снимкам [↑95 К–П] | "
                    + "FAR по людям (принят хотя бы один снимок)\n");
            for (Map.Entry<String, long[]> e : run.rule.entrySet()) {
                if (!e.getKey().startsWith(conf + " ")) continue;
                long[] t = e.getValue();
                int pa = run.ruleCtrlPersons.getOrDefault(e.getKey(), Set.of()).size();
                int pn = run.ruleCtrlAll.getOrDefault(e.getKey(), Set.of()).size();
                r.append(String.format(Locale.ROOT, "%s | %d | %.1f %% | %.1f %% | %.1f %% | %d | %.2f %% [%.2f %%] | %d/%d = %.1f %%%n", e.getKey(), t[0], 100.0 * t[1] / t[0],
                        100.0 * t[2] / t[0], 100.0 * (t[0] - t[1] - t[2]) / t[0], t[3], 100.0 * t[4] / t[3],
                        100 * FaceEvaluation.binomialUpperBound((int) t[4], (int) t[3], 0.95), pa, pn, 100.0 * pa / Math.max(1, pn)));
            }
            // Строка для сводки (4а).
            String m = conf + " " + MAIN;
            if (res.containsKey(m)) {
                XBaseOther.Sep a = sep.get(m);
                FeiThreshold.Res rm = res.get(m);
                long[] t = run.rule.get(m);
                List<String> cells = new ArrayList<>(List.of(base, conf, "метод Прокаева (Д7), в правило отключений не входит",
                        String.format(Locale.ROOT, "%.1f", a.err[0]), FeiThreshold.errCell(rm, 3, 0, 0, tpl.b),
                        String.format(Locale.ROOT, "%.2f", FeiThreshold.farPct(rm, 3, 0)), String.format(Locale.ROOT, "%.1f", a.eer[3])));
                if (sep.containsKey(b)) {
                    XBaseOther.Sep z = sep.get(b);
                    cells.addAll(XBaseAbl.diffCells(a.err[0] - z.err[0], a.errRep[0], z.errRep[0]));
                    cells.addAll(XBaseAbl.diffCells(a.eer[3] - z.eer[3], a.eerRep, z.eerRep));
                } else {
                    cells.addAll(List.of("база (4а) ожидается", "", "", "", "", ""));
                }
                cells.add(String.format(Locale.ROOT, "%.1f", 100.0 * t[1] / t[0]));
                cells.add(String.format(Locale.ROOT, "%.2f", 100.0 * t[4] / t[3]));
                sum.append(String.join("\t", cells)).append('\n');
            }
        }
    }

    static double num(String s) {
        return Double.parseDouble(s);
    }

    static String sumHead() {
        return "# база\tпостановка\tстрока\tразд. 1 %\tП3 FAR 1 %\tFAR контроля П3 %\tEER\tразд. 1 %: метод − база (4а)\tниз\tверх\tEER: метод − база\tниз\tверх\t"
                + "правило 0,95: отвергнуто своих %\tFAR контроля по снимкам %\n";
    }

    static void summaryText(StringBuilder r, String sum) {
        r.append("\n=== Строка для сводки (4а): «метод Прокаева (Д7), в правило отключений не входит» ===\n");
        r.append("постановка | разд. при FAR контроля 1 % | П3 FAR 1 % | FAR контроля при П3 | EER | разд.: метод − база (4а) L1-xs-F [2,5; 97,5] | EER: метод − база | "
                + "правило 0,95: отвергнуто своих | FAR контроля\n");
        for (String l : sum.split("\n")) {
            if (l.startsWith("#") || l.isEmpty()) continue;
            String[] c = l.split("\t", -1);
            // В файле сводки – полная точность (одно округление – при выводе, №11 Жорика); в тексте – два знака.
            String d1 = c[7].startsWith("база") ? c[7] : String.format(Locale.ROOT, "%.2f [%.2f; %.2f]", num(c[7]), num(c[8]), num(c[9]));
            String d2 = c[7].startsWith("база") ? "–" : String.format(Locale.ROOT, "%.2f [%.2f; %.2f]", num(c[10]), num(c[11]), num(c[12]));
            r.append(String.join(" | ", c[1], c[3] + " %", c[4], c[5] + " %", c[6] + " %", d1, d2, c[13] + " %", c[14] + " %")).append('\n');
        }
    }

    // ---------------------------------------------------------------- FERET

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(exportDir, selLines, settings);
        XBaseEval.Gt tpl = su.tpl;
        Path dir = su.dir;
        Path out = dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Map<String, String> ablBlocks = XBaseSubsp.blocks(out.resolve("xbase_feret_abl_scores.tsv.gz"));
        XBaseFeret.Roles ro = XBaseFeret.equalize(su.ro, su.byKey);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<String> perm = new ArrayList<>(pool.keySet());
        Collections.shuffle(perm, new Random(tpl.seed + XBaseXsF.SPLIT_SEED_OFFSET));
        int half = (perm.size() + 1) / 2;
        List<Set<String>> halves = List.of(new TreeSet<>(perm.subList(0, half)), new TreeSet<>(perm.subList(half, perm.size())));
        TreeSet<String> ps0 = new TreeSet<>(pool.keySet());
        ps0.addAll(ro.ctrl);
        List<String> persons = new ArrayList<>(ps0);
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        for (String p : cohortP) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
        XBaseEval.Gt base = XBaseFeret.gt(tpl, persons, ro.ctrl, ro.ctrlKeys, cohort, su.rOf, su.imageNo);
        Map<String, String> personOf = new HashMap<>();
        for (Img im : su.imgs) personOf.put(im.key(), im.person());
        Map<String, List<String>> keysOf = new HashMap<>();
        pool.forEach((p, ds) -> {
            List<String> ks = new ArrayList<>();
            ds.forEach(ks::addAll);
            keysOf.put(p, ks);
        });
        List<String> fei10 = feiGuests(feiKeys, cohortP, tpl.seed, personOf);
        Map<String, Integer> protoOf = Map.of("межс.", 1, "G2", 2, "внутри", 3);
        List<Job> jobs = new ArrayList<>();
        Set<String> used = new TreeSet<>();
        for (XBaseXsF.Conf c : XBaseMp2.confs(ro)) {
            for (int ps = 0; ps < 2; ps++) {
                Job j = new Job();
                j.conf = c.name();
                j.pass = ps;
                j.proto = protoOf.get(c.name());
                List<String> own = c.own().stream().filter(halves.get(ps)::contains).toList();
                j.gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    j.gal.put(p, c.gal().get(p));
                    pro.put(p, c.pro().get(p));
                }
                j.g = XBaseFeret.withOwn(base, own);
                j.probes = XBaseEval.probes(j.g, pro);
                j.parts = XBaseOther.cvParts(own, tpl.seed, XBaseEval.CV_S + c.s());
                j.s = c.s();
                pickGuests(j, new ArrayList<>(halves.get(1 - ps)), keysOf, tpl.seed);
                j.fei10 = fei10;
                j.baseL1 = ablBlocks.get("схема база, " + c.name() + ", кандидат L1-xs-F, проход " + (ps + 1));
                {
                    String tt = ablBlocks.get("схема база, " + c.name() + ", кандидат L1-tt-xs-F, проход " + (ps + 1));
                    if (tt != null) j.extraBases.add(new Object[] {BASE_ROW_TT, tt, V.L1});
                }
                if (j.baseL1 == null) throw new IllegalStateException("prokaev: нет блока базы (4а) " + c.name() + ", проход " + (ps + 1) + " – сначала feret-abl");
                checkGuests(j, personOf);
                j.only = ONLY;
                jobs.add(j);
                for (String p : own) used.addAll(c.gal().get(p));
                for (Probe q : j.probes) used.add(q.key());
                used.addAll(j.g30);
                j.authPool.forEach(used::addAll);
            }
        }
        // Кадры: FERET – исходные снимки (в памяти), кадр «а» – derived/a; гости FEI – исходные снимки и кадр «а» экспорта.
        Vecs v = new Vecs();
        int n = 0;
        for (String k : used) {
            Mat c = FeretData.decode(su.byKey.get(k).file());
            addBig(v, k, c, su.det.get(k));
            c.release();
            addSmall(v, k, FeretData.frame(dir, "a", k));
            if (++n % 500 == 0) System.out.printf(Locale.ROOT, "feret-prokaev: кадры – %d / %d%n", n, used.size());
        }
        feiFrames(export, exportDir, new HashSet<>(fei10), settings, v);
        Run run = new Run();
        for (Job j : jobs) {
            checkRoles(run, j, false);
            job(j, v, run);
            System.out.printf(Locale.ROOT, "feret-prokaev: %s, проход %d ready%n", j.conf, j.pass + 1);
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 (ред. 4, Д8): метод Прокаева на протоколах (4а), FERET eq (XBaseEval feret-prokaev): Color FERET – только "
                + "локально; справочная строка, в правило отключений (4а) не входит\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb, YuNet – derived/detections.tsv; кадр «а» – derived/a); база (4а) – ").append(out.resolve("xbase_feret_abl_scores.tsv.gz"))
                .append(" (").append(FeiThreshold.sha(out.resolve("xbase_feret_abl_scores.tsv.gz"))).append("); гости FEI – экспорт ").append(exportDir).append(", коммит данных ")
                .append(FarMethods.dataCommit(exportDir)).append('\n');
        header(r, tpl, v, run, jobs, "FERET: гости – люди другой половины №2 (пула прохода; в методе Прокаева пула нет), номер протокола: межс. 1, G2 2, внутри 3");
        StringBuilder sum = new StringBuilder(sumHead());
        tables(r, sum, run, persons.size(), tpl, List.of("межс.", "G2", "внутри"), "FERET", BASE_ROW);
        summaryText(r, sum.toString());
        XBaseOther.writeScores(out.resolve("xbase_feret_prokaev" + ProkaevFrames.tag() + "_scores.tsv.gz"), "# FERET, Д-№1 метод Прокаева (ProkaevEval feret-prokaev), только локально; блоки – "
                + "постановка, проход, строка; оценка – 1 + (1 − доля лучшего своего) + 1e-6·d/(1 + d) (меньше – ближе), у строки d₁/d₂ – расстояния до ближайших точек\n"
                + "# Код: коммит " + FarMethods.commit() + "\n", run.lines.toString());
        FarMethods.write(out.resolve("xbase_feret_prokaev" + ProkaevFrames.tag() + "_summary.tsv"), sum);
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; наибольшая занятая куча (выборки после каждой строки) – %d МБ из %d МБ.%n", (System.nanoTime() - t0) / 1e9,
                run.peak >> 20, Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_feret_prokaev" + ProkaevFrames.tag() + ".txt"), r);
        System.out.print(r);
    }

    static void header(StringBuilder r, XBaseEval.Gt tpl, Vecs v, Run run, List<Job> jobs, String guests) {
        r.append(String.format(Locale.ROOT, "Метод (ProkaevMethod): SVD без центрирования по общей матрице «галерея + гости», k = %d, l = %d, точка – строка L_k, проба – "
                + "w_j = z·R_j / s_j, доли по формуле 7, правило автора – гость (отказ), если доля гостей > 0 и доля каждого своего < 0,95.%n", K, L));
        r.append(String.format(Locale.ROOT, "Кадр: %s. Справочно – кадр «а» 92×112.%n", ProkaevFrames.describe(v.clipped)));
        r.append("Оценка для наших мер (разделимость, П3, EER): 1 − доля лучшего своего, ничьи – по d₁ (расстояние до ближайшей точки этого человека); выбор "
                + "модели по позе не применяется (зеркал у метода нет). Роли, пробы, контроль и части «чужих» – как у базы (4а).\n");
        r.append(guests).append("; 10 / 20 / 30 – вложенные, по одному снимку, seed faces.seed + ").append(GUEST_SEED).append(" + номер протокола; гости FEI – seed + ")
                .append(FEI_SEED).append("; пул отбора по правилу автора – seed + ").append(AUTH_SEED).append(" + номер протокола (20 человек по 3 снимка; принят – доля "
                + "своего ≥ 0,95 в модели без гостей; верхние 10 человек, их самый «принимаемый» снимок).\n");
        r.append(run.guests);
        List<String> sz = new ArrayList<>();
        for (int i = 0; i < jobs.size(); i++) if (!jobs.get(i).halfOnly) sz.add(jobs.get(i).conf + " " + (jobs.get(i).pass + 1) + ": " + run.galSizes.get(sz.size())[0] + "/"
                + run.galSizes.get(sz.size())[1] + "/" + run.galSizes.get(sz.size())[2]);
        boolean two = run.galSizes.stream().anyMatch(x -> x[0] <= 2);
        r.append("Снимков галереи на человека (мин/медиана/макс): ").append(String.join("; ", sz)).append(two ? ". При 2 точках на человека и l = 3 третья "
                + "ближайшая точка почти всегда чужая – свойство метода при таком протоколе.\n" : ".\n");
        if (tpl.b > 0) r.append(String.format(Locale.ROOT, "Бутстреп по людям: B = %d, seed = %d (как у базы (4а)); «*» – интервал разности не содержит 0.%n", tpl.b,
                tpl.seed + FeiBootstrap.SEED_OFFSET));
        r.append("\n=== Сверка ролей с базой (4а) ===\n").append(run.check);
    }

    /** Кадры гостей FEI: исходный снимок и YuNet экспорта (400 × 600), кадр «а» экспорта. */
    static void feiFrames(FarExport.Loaded export, Path exportDir, Set<String> keys, SettingsStore settings, Vecs v) throws IOException {
        Path feiDir = Paths.get(settings.loadFacesFeiDir());
        Map<String, Path> raw = new HashMap<>();
        for (List<Sample> l : FeiDataset.originals(feiDir).values()) for (Sample x : l) raw.put(FarExport.stem(x.file()), x.file());
        Map<String, double[]> rows = XBaseMpPts.exportRows(exportDir, "fei");
        List<Sample> sel = new ArrayList<>();
        for (List<Sample> l : export.data().fei().values()) for (Sample x : l) if (keys.contains(x.file().toString())) sel.add(x);
        for (Sample x : sel) {
            Mat c = Imgcodecs.imread(raw.get(FarExport.stem(x.file())).toString(), Imgcodecs.IMREAD_COLOR);
            double[] row = rows.get(x.person() + "/" + x.file().getFileName().toString());
            if (row == null) throw new IllegalStateException("prokaev: нет рамки YuNet гостя FEI " + x.file());
            addBig(v, x.file().toString(), c, row);
            c.release();
        }
        Map<String, Mat> fa = XBaseEval.frames(export, sel, keys);
        for (Map.Entry<String, Mat> e : fa.entrySet()) addSmall(v, e.getKey(), e.getValue());
        if (sel.size() != keys.size()) throw new IllegalStateException("prokaev: не найдены снимки гостей FEI");
    }

    // ---------------------------------------------------------------- GT

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("gt-prokaev: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> det = XBaseMpPts.exportRows(dir, "gt");
        Map<String, Double> rOld = new HashMap<>();
        det.forEach((pk, row) -> {
            if (row != null) rOld.put(pk, Math.abs(FeiDataset.pose(row)));
        });
        XBaseEval.Gt g0 = XBaseMp2.gtRoles(export, dir, feiKeys, settings, pk -> rOld.containsKey(pk) && rOld.get(pk) <= rLimit, rOld, r0);
        g0.rLimit = rLimit;
        long eqSeed = g0.seed + XBaseEval.EQ_SEED_OFFSET;
        Map<String, List<String>> galX = XBaseEval.equalize(g0.own, g0.galleryAll, eqSeed);
        List<Map<String, List<String>>> split = XBaseEval.withinSplit(g0.own, g0.galleryAll, g0.seed);
        Map<String, List<String>> galIn = XBaseEval.equalize(g0.own, split.get(0), eqSeed);
        List<List<String>> parts = FeiMethods.cvParts(g0.own, g0.seed, XBaseEval.CV_S);
        Map<String, String> ablBlocks = XBaseSubsp.blocks(out.resolve("xbase_gt_abl_scores.tsv.gz"));
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        Map<String, List<String>> keysOf = new HashMap<>();
        pool.forEach((p, ds) -> {
            List<String> ks = new ArrayList<>();
            ds.forEach(ks::addAll);
            keysOf.put(p, ks);
        });
        Map<String, String> personOf = new HashMap<>();
        for (Img im : su.imgs) personOf.put(im.key(), "FERET " + im.person());
        Map<String, Sample> sampleOf = new HashMap<>();
        for (List<Sample> l : export.data().gt().values()) for (Sample s : l) {
            sampleOf.put(s.file().toString(), s);
            personOf.put(s.file().toString(), s.person());
        }
        List<String> fei10 = feiGuests(feiKeys, g0.cohortP, g0.seed, personOf);
        Object[][] modes = {{"межс.", galX, g0.probeKeys, 4}, {"внутри", galIn, split.get(1), 5}};
        List<Job> jobs = new ArrayList<>();
        List<String> ctrl = new ArrayList<>(g0.ctrl);
        Collections.sort(ctrl);
        Collections.shuffle(ctrl, new Random(g0.seed + HALF_SEED));
        List<List<String>> ctrlHalves = List.of(ctrl.subList(0, ctrl.size() / 2), ctrl.subList(ctrl.size() / 2, ctrl.size()));
        for (Object[] md : modes) {
            @SuppressWarnings("unchecked")
            Map<String, List<String>> gal = (Map<String, List<String>>) md[1];
            @SuppressWarnings("unchecked")
            Map<String, List<String>> pro = (Map<String, List<String>>) md[2];
            Job j = new Job();
            j.conf = (String) md[0];
            j.pass = 0;
            j.proto = (Integer) md[3];
            j.g = g0;
            j.gal = gal;
            j.probes = XBaseEval.probes(g0, pro);
            j.parts = parts;
            j.s = 0;
            pickGuests(j, new ArrayList<>(pool.keySet()), keysOf, g0.seed);
            j.fei10 = fei10;
            j.baseL1 = ablBlocks.get("схема база, " + j.conf + ", кандидат L1-xs-F, проход 1");
            {
                    String tt = ablBlocks.get("схема база, " + j.conf + ", кандидат L1-tt-xs-F, проход 1");
                    if (tt != null) j.extraBases.add(new Object[] {BASE_ROW_TT, tt, V.L1});
                }
            if (j.baseL1 == null) throw new IllegalStateException("prokaev: нет блока базы (4а) GT " + j.conf + " – сначала gt-abl");
            checkGuests(j, personOf);
            j.only = ONLY;
            jobs.add(j);
            // Справочно: контроль пополам, гости – люди GT другой половины (по одному снимку), два прохода.
            for (int h = 0; h < 2; h++) {
                Job jh = new Job();
                jh.conf = j.conf;
                jh.pass = h;
                jh.proto = j.proto;
                jh.halfOnly = true;
                XBaseEval.Gt gh = XBaseAbl.copy(g0, g0.cohort, g0.rOf, g0.r0);
                gh.ctrl = new ArrayList<>(ctrlHalves.get(1 - h));
                Collections.sort(gh.ctrl);
                jh.g = gh;
                jh.gal = gal;
                jh.probes = XBaseEval.probes(gh, pro);
                jh.parts = parts;
                Random rh = new Random(g0.seed + HALF_SEED + 1 + h);
                for (String p : new TreeSet<>(ctrlHalves.get(h))) {
                    List<String> ks = new ArrayList<>();
                    g0.byPerson.get(p).values().forEach(ks::addAll);
                    Collections.sort(ks);
                    jh.half.add(ks.get(rh.nextInt(ks.size())));
                }
                checkGuests(jh, personOf);
                if (ONLY == null) jobs.add(jh);
            }
        }
        // Кадры: GT – исходные снимки gt_db и YuNet экспорта, кадр «а» – экспорт; гости FERET – исходные снимки, derived/a; FEI – экспорт.
        Vecs v = new Vecs();
        Set<String> gtKeys = new TreeSet<>(g0.samples.keySet());
        for (String k : gtKeys) {
            Sample s = sampleOf.get(k);
            String pk = s.person() + "/" + s.file().getFileName().toString();
            Mat c = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            addBig(v, k, c, det.get(pk));
            c.release();
        }
        for (Map.Entry<String, Mat> e : XBaseEval.frames(export, gtKeys.stream().map(sampleOf::get).toList(), null).entrySet()) addSmall(v, e.getKey(), e.getValue());
        Set<String> fk = new TreeSet<>();
        for (Job j : jobs) {
            fk.addAll(j.g30);
            j.authPool.forEach(fk::addAll);
        }
        for (String k : fk) {
            Mat c = FeretData.decode(su.byKey.get(k).file());
            addBig(v, k, c, su.det.get(k));
            c.release();
            addSmall(v, k, FeretData.frame(su.dir, "a", k));
        }
        feiFrames(export, dir, new HashSet<>(fei10), settings, v);
        Run run = new Run();
        for (Job j : jobs) {
            if (!j.halfOnly) checkRoles(run, j, true);
            job(j, v, run);
            System.out.printf(Locale.ROOT, "gt-prokaev: %s%s ready%n", j.conf, j.halfOnly ? ", половины контроля, проход " + (j.pass + 1) : "");
        }
        // Условные номера гостей FERET (имена снимков – только в каталоге FERET).
        StringBuilder map = new StringBuilder("# Гости FERET в gt-prokaev: условный номер → снимок FERET (только локально)\n");
        List<String> all = new ArrayList<>();
        for (Job j : jobs) for (String k : j.g30) if (!all.contains(k)) all.add(k);
        for (Job j : jobs) for (String k : j.auth) if (!all.contains(k)) all.add(k);
        for (int i = 0; i < all.size(); i++) map.append(String.format(Locale.ROOT, "FG%02d\t%s%n", i + 1, all.get(i)));
        Path fdir = su.dir.resolve("reports").resolve(XBaseEval.eqDir(0, true));
        FarMethods.write(fdir.resolve("prokaev_gt_guests.tsv"), map);
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 (ред. 4, Д8): метод Прокаева на протоколах (4а), GT (XBaseEval gt-prokaev); справочная строка, в правило "
                + "отключений (4а) не входит\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (a/gt, YuNet – detections.tsv); снимки GT – ").append(gtDir)
                .append("; база (4а) – ").append(out.resolve("xbase_gt_abl_scores.tsv.gz")).append(" (").append(FeiThreshold.sha(out.resolve("xbase_gt_abl_scores.tsv.gz")))
                .append("); гости – Color FERET (условные номера FG01–FG").append(String.format(Locale.ROOT, "%02d", all.size())).append(", имён снимков нет)\n");
        header(r, g0, v, run, jobs, "GT: гости – люди FERET (пул, 173 человека; номер протокола: межс. 4, внутри 5); справочно – контроль пополам (seed faces.seed + "
                + HALF_SEED + "), гости – люди GT другой половины по одному снимку, два прохода, итог – объединение");
        StringBuilder sum = new StringBuilder(sumHead());
        tables(r, sum, run, g0.persons.size(), g0, List.of("межс.", "внутри"), "GT", BASE_ROW);
        summaryText(r, sum.toString());
        XBaseOther.writeScores(out.resolve("xbase_gt_prokaev" + ProkaevFrames.tag() + "_scores.tsv.gz"), "# GT, Д-№1 метод Прокаева (ProkaevEval gt-prokaev); блоки – постановка, проход, строка\n"
                + "# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных " + FarMethods.dataCommit(dir) + "\n", run.lines.toString());
        FarMethods.write(out.resolve("xbase_gt_prokaev" + ProkaevFrames.tag() + "_summary.tsv"), sum);
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; наибольшая занятая куча (выборки после каждой строки) – %d МБ из %d МБ.%n", (System.nanoTime() - t0) / 1e9,
                run.peak >> 20, Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_gt_prokaev" + ProkaevFrames.tag() + ".txt"), r);
        System.out.print(r);
    }
}
