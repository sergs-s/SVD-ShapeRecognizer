package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.opencv.core.Mat;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.FeretData.Img;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Исключение даты пробы (PLAN.md п. 5.5, (а2); план подтверждён Хозяином 12.10.2026): для пробы человека с даты d в галерее и
 * в обучении MLDA нет ни одного снимка этого человека с даты d (исключается вся дата). Варианты галереи человека для пробы:
 * <ul>
 *   <li>Г-все — все другие его даты (основной);</li>
 *   <li>Г-прошлые — только даты раньше даты пробы (справочно; первая дата пробой не бывает);</li>
 *   <li>Г-первая — только первая дата, пробы — как у Г-прошлые (= роль (1) feret, своя калибровка порога).</li>
 * </ul>
 * Схема моделей: свои делятся на 5 частей (XBaseOther.cvParts, s = 0); модель (q, k) — у людей части q с ≥ k датами
 * убирается k-я дата (Г-все) или даты с k-й (Г-прошлые), у прочих — все даты; пробы модели — снимки k-й даты людей части q.
 * Порог Г-все и Г-прошлые — одна калибровка П2/П3 на метод по полной галерее (все даты всех своих), контрольные чужие — против
 * полной модели. Режимы XBaseEval feret-excl (отчёт только в каталоге базы FERET) и gt-excl.
 *
 * @author ssv
 */
final class XBaseExcl {

    /** Вариант галереи: подпись, номер s в строках оценок. */
    enum G {
        ALL("Г-все", 2), PAST("Г-прошлые", 3), FIRST("Г-первая", 0);

        final String label;
        final int s;

        G(String label, int s) {
            this.label = label;
            this.s = s;
        }
    }

    /** Номер s строк полной модели (контроль и «чужие» П2/П3 вариантов Г-все и Г-прошлые). */
    static final int FULL_S = 1;

    static final String[] SESS_NAMES = {"1", "2", "3", "≥ 4"};
    static final String[] IMG_NAMES = {"1–2", "3–4", "5–6", "≥ 7"};
    /** Ячейка разбивки помечается «мало» при числе людей меньше этого. */
    static final int FEW = 10;

    private XBaseExcl() {
    }

    // ---------------------------------------------------------------- схема

    /**
     * Свои: человек → даты с отобранными снимками по возрастанию → ключи (пробы); даты съёмки (для интервалов); gsess — те же
     * даты → ключи галереи и обучения MLDA (№9: на FERET по умолчанию правило №8 — один fa и один fb на дату; иначе = sess).
     */
    record Data(List<String> own, Map<String, List<List<String>>> sess, Map<String, List<LocalDate>> when, Map<String, List<List<String>>> gsess) {
        Data(List<String> own, Map<String, List<List<String>>> sess, Map<String, List<LocalDate>> when) {
            this(own, sess, when, sess);
        }
    }

    /** №9: сессии галереи по правилу №8 (XBaseFeret.pairOf на каждой дате). */
    static Map<String, List<List<String>>> pairSessions(Map<String, List<List<String>>> sess, Map<String, Img> byKey) {
        Map<String, List<List<String>>> out = new LinkedHashMap<>();
        sess.forEach((p, ds) -> {
            List<List<String>> gs = new ArrayList<>();
            for (List<String> keys : ds) gs.add(sorted(XBaseFeret.pairOf(keys, byKey, false)));
            out.put(p, gs);
        });
        return out;
    }

    /** №9: строка шапки отчётов feret-excl и feret-hubs при равных галереях. */
    static final String EQ_NOTE = "№9 (faces.feret.gallery = eq, по умолчанию; правило №8, (А)): галерея и обучение MLDA – на каждой дате один fa и "
            + "один fb, варианты (_a и др.) исключены; нет чистого fa (fb) – первый вариант по имени; пробы – все отобранные снимки даты; прежние "
            + "галереи – faces.feret.gallery = old.\n";

    /** Модель схемы: часть q, раунд k (номер даты с 1), галерея (человек → ключи), пробы своих (человек → ключи). */
    record Model(int q, int k, Map<String, List<String>> gal, Map<String, List<String>> pro) {}

    /** Номера дат галереи человека для пробы с даты i (с 0) из n дат. */
    static List<Integer> galleryDates(G v, int i, int n) {
        List<Integer> out = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            boolean in = switch (v) {
                case ALL -> j != i;
                case PAST -> j < i;
                case FIRST -> j == 0;
            };
            if (in) out.add(j);
        }
        return out;
    }

    /** Модели варианта ALL или PAST (FIRST — одна модель, XBaseEval.fitLines); пустые раунды пропускаются. */
    static List<Model> models(Data d, List<List<String>> parts, G v) {
        if (v == G.FIRST) throw new IllegalArgumentException("Г-первая — одна модель");
        int maxK = d.own().stream().mapToInt(p -> d.sess().get(p).size()).max().orElse(0);
        List<Model> out = new ArrayList<>();
        for (int q = 0; q < parts.size(); q++) {
            Set<String> part = new HashSet<>(parts.get(q));
            for (int k = v == G.PAST ? 2 : 1; k <= maxK; k++) {
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : d.own()) if (part.contains(p) && d.sess().get(p).size() >= k) pro.put(p, sorted(d.sess().get(p).get(k - 1)));
                if (pro.isEmpty()) continue;
                Map<String, List<String>> gal = new LinkedHashMap<>();
                for (String p : d.own()) {
                    List<List<String>> ds = d.gsess().get(p);
                    List<String> keys = new ArrayList<>();
                    if (pro.containsKey(p)) {
                        for (int j : galleryDates(v, k - 1, ds.size())) keys.addAll(ds.get(j));
                    } else {
                        ds.forEach(keys::addAll);
                    }
                    gal.put(p, sorted(keys));
                }
                out.add(new Model(q, k, gal, pro));
            }
        }
        return out;
    }

    /** Галерея: человек → снимки дат с номерами из dates (null — все). */
    static Map<String, List<String>> gallery(Data d, Set<Integer> dates) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (String p : d.own()) {
            List<String> keys = new ArrayList<>();
            List<List<String>> ds = d.gsess().get(p);
            for (int j = 0; j < ds.size(); j++) if (dates == null || dates.contains(j)) keys.addAll(ds.get(j));
            out.put(p, sorted(keys));
        }
        return out;
    }

    /** Пробы своих варианта: Г-все — все снимки, прочие — снимки дат со второй. */
    static Map<String, List<String>> probeKeys(Data d, G v) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (String p : d.own()) {
            List<String> keys = new ArrayList<>();
            List<List<String>> ds = d.sess().get(p);
            for (int j = v == G.ALL ? 0 : 1; j < ds.size(); j++) keys.addAll(ds.get(j));
            out.put(p, sorted(keys));
        }
        return out;
    }

    /** Разбивка пробы: {число сессий галереи, число снимков галереи, интервал до ближайшей даты галереи} — номера интервалов. */
    static int[] bins(Data d, String p, int i, G v) {
        List<List<String>> ds = d.gsess().get(p);
        List<Integer> gd = galleryDates(v, i, ds.size());
        int imgs = 0;
        long near = Long.MAX_VALUE;
        for (int j : gd) {
            imgs += ds.get(j).size();
            near = Math.min(near, Math.abs(ChronoUnit.DAYS.between(d.when().get(p).get(j), d.when().get(p).get(i))));
        }
        int s = gd.size();
        return new int[] {Math.min(s, 4) - 1, imgs <= 2 ? 0 : imgs <= 4 ? 1 : imgs <= 6 ? 2 : 3, XBaseFeret.bin(near)};
    }

    static List<String> sorted(List<String> l) {
        List<String> s = new ArrayList<>(l);
        Collections.sort(s);
        return s;
    }

    // ---------------------------------------------------------------- обучение и оценки

    /** Строки оценок проб своих моделей схемы (s — вариант, j — раунд k). */
    static String fitModels(XBaseEval.Gt g, XBaseEval x, Set<V> vs, List<Model> models, int s) {
        List<String> own = g.own;
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < own.size(); i++) idx.put(own.get(i), i);
        List<Callable<String>> tasks = new ArrayList<>();
        for (V v : vs) {
            for (boolean m : v == V.SFACE ? new boolean[] {false} : new boolean[] {false, true}) {
                String method = v.id + (m ? FeiMethods.MIRROR : "");
                for (Model md : models) {
                    List<List<String>> gk = own.stream().map(md.gal()::get).toList();
                    List<Probe> pr = new ArrayList<>();
                    md.pro().forEach((p, ks) -> {
                        for (String k : ks) pr.add(new Probe(FeiScores.OWN, p, g.img(k), k, idx.get(p)));
                    });
                    tasks.add(() -> x.lines(method, s, md.k(), x.fit(v, m, gk, g.cohort), v, own, pr));
                }
            }
        }
        StringBuilder all = new StringBuilder();
        XBaseEval.run(tasks, g.threads).forEach(all::append);
        return all.toString();
    }

    /**
     * Прогон трёх вариантов: g — свои d.own(), контроль, когорта, люди бутстрепа; xs и sets — векторы и наборы вариантов V
     * (как XBaseFeret.SETS). Пишет в r таблицы, в lines — строки оценок блоками.
     */
    static void evaluate(XBaseEval.Gt g, Data d, XBaseEval[] xs, List<Set<V>> sets, XBaseOther.Checkpoint cp, StringBuilder r, StringBuilder lines,
                         String code) throws IOException {
        List<List<String>> parts = XBaseOther.cvParts(d.own(), g.seed, XBaseEval.CV_S);
        Map<String, List<String>> fullGal = gallery(d, null);
        Map<String, List<String>> firstGal = gallery(d, Set.of(0));
        Map<G, List<Probe>> main = new LinkedHashMap<>();
        for (G v : G.values()) main.put(v, XBaseEval.probes(g, probeKeys(d, v)));
        List<Probe> ctrlProbes = main.get(G.ALL).stream().filter(p -> p.role().equals(FeiScores.CTRL)).toList();
        Map<G, List<Model>> models = new LinkedHashMap<>();
        models.put(G.ALL, models(d, parts, G.ALL));
        models.put(G.PAST, models(d, parts, G.PAST));

        Map<G, Map<String, FeiThreshold.Res>> res = new LinkedHashMap<>();
        for (G v : G.values()) res.put(v, new LinkedHashMap<>());
        Map<String, FeiThreshold.Imp> imps = new HashMap<>();
        double fitS = 0;
        int resumed = 0;
        for (int set = 0; set < sets.size(); set++) {
            Set<V> vs = sets.get(set);
            String setName = set == 0 ? "none" : set == 1 ? "tt" : "SFace";
            Map<String, String> ls = new LinkedHashMap<>();
            for (String block : List.of("first", "full", "all", "past")) {
                String name = block + "_" + set;
                String l = cp.read(name);
                if (l == null) {
                    long t1 = System.nanoTime();
                    l = switch (block) {
                        case "first" -> XBaseEval.fitLines(g, xs[set], vs, firstGal, main.get(G.FIRST), parts, G.FIRST.s);
                        case "full" -> XBaseEval.fitLines(g, xs[set], vs, fullGal, ctrlProbes, parts, FULL_S);
                        case "all" -> fitModels(g, xs[set], vs, models.get(G.ALL), G.ALL.s);
                        default -> fitModels(g, xs[set], vs, models.get(G.PAST), G.PAST.s);
                    };
                    double s = (System.nanoTime() - t1) / 1e9;
                    fitS += s;
                    cp.save(name, l);
                    System.out.printf(Locale.ROOT, "%s: блок %s, набор %s готов (%.0f с)%n", code, block, setName, s);
                } else {
                    resumed++;
                    System.out.printf(Locale.ROOT, "%s: блок %s, набор %s — из контрольной точки%n", code, block, setName);
                }
                ls.put(block, l);
                lines.append("# блок ").append(block).append(", набор ").append(setName).append('\n').append(l);
            }
            Map<G, XBaseEval.Parsed> parsed = new LinkedHashMap<>();
            parsed.put(G.ALL, XBaseEval.parse(g, vs, fullGal, main.get(G.ALL), parts, ls.get("full") + ls.get("all")));
            parsed.put(G.PAST, XBaseEval.parse(g, vs, fullGal, main.get(G.PAST), parts, ls.get("full") + ls.get("past")));
            parsed.put(G.FIRST, XBaseEval.parse(g, vs, firstGal, main.get(G.FIRST), parts, ls.get("first")));
            for (G v : G.values()) {
                for (XBaseFeret.Method m : XBaseFeret.METHODS) {
                    if (m.set() != set) continue;
                    String label = v.label + " " + m.label();
                    XBaseEval.Parsed p = parsed.get(v);
                    imps.put(label, p.imp.get(m.v()));
                    res.get(v).put(label, FeiThreshold.evaluate(label, m.v().ratioId(), List.of(p.cfg.get(m.v())), List.of(p.imp.get(m.v())),
                            g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET));
                }
            }
        }
        Map<String, FeiThreshold.Res> all = new LinkedHashMap<>();
        for (G v : G.values()) for (XBaseFeret.Method m : XBaseFeret.METHODS) all.put(v.label + " " + m.label(), res.get(v).get(v.label + " " + m.label()));

        // Сведения о схеме и объёмах.
        Map<String, Integer> dateOf = new HashMap<>();
        for (String p : d.own()) {
            List<List<String>> ds = d.sess().get(p);
            for (int j = 0; j < ds.size(); j++) for (String k : ds.get(j)) dateOf.put(k, j);
        }
        r.append(String.format(Locale.ROOT, "Части своих (5, seed + 8000 + 100 + 0): размеры %s; моделей схемы на метод и зеркало: Г-все %d, Г-прошлые %d "
                + "(+ полная и 5 для порога; Г-первая — 1 + 5).%n", parts.stream().map(q -> Integer.toString(q.size())).toList(), models.get(G.ALL).size(),
                models.get(G.PAST).size()));
        r.append("вариант | своих с пробами | проб своих | снимков контроля | «чужих» П2/П3\n");
        for (G v : G.values()) {
            List<Probe> own = main.get(v).stream().filter(p -> p.role().equals(FeiScores.OWN)).toList();
            Map<String, List<String>> calib = v == G.FIRST ? firstGal : fullGal;
            r.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d%n", v.label, own.stream().map(Probe::person).distinct().count(), own.size(),
                    ctrlProbes.size(), calib.values().stream().mapToInt(List::size).sum()));
        }
        r.append(String.format(Locale.ROOT, "Обучение и оценки: %.0f с; из контрольной точки — блоков %d; потоков %d.%n", fitS, resumed, g.threads));

        int[][] w = g.b > 0 ? FeiBootstrap.weights(g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(g.persons.size());
        Map<String, XBaseOther.Sep> sep = new LinkedHashMap<>();
        for (Map.Entry<String, FeiThreshold.Res> e : all.entrySet()) sep.put(e.getKey(), XBaseOther.sep(e.getValue().cfgs, unit, w));
        List<String[]> pairs = new ArrayList<>();
        for (G v : G.values()) {
            pairs.add(new String[] {v.label + " L1-tt", v.label + " L1"});
            pairs.add(new String[] {v.label + " LTP-in-tt", v.label + " L1-tt"});
            pairs.add(new String[] {v.label + " LTP-in-tt", v.label + " L1"});
        }
        for (XBaseFeret.Method m : XBaseFeret.METHODS) pairs.add(new String[] {G.PAST.label + " " + m.label(), G.FIRST.label + " " + m.label()});
        XBaseOther.tables(r, all, sep, g.b, pairs);

        // Разбивки ошибки своих.
        Map<G, List<int[]>> bins = new LinkedHashMap<>();
        for (G v : G.values()) {
            List<int[]> b = new ArrayList<>();
            for (Probe q : main.get(v)) if (q.role().equals(FeiScores.OWN)) b.add(bins(d, q.person(), dateOf.get(q.key()), v));
            bins.put(v, b);
        }
        String[][] names = {SESS_NAMES, IMG_NAMES, XBaseFeret.BIN_NAMES};
        String[] titles = {"по числу сессий (дат) в галерее человека для пробы", "по числу снимков в галерее человека для пробы",
                "по интервалу от ближайшей даты галереи до даты пробы"};
        for (int t = 0; t < names.length; t++) {
            r.append("\n=== Ошибка своих ").append(titles[t]).append(" (порог своей строки; «мало» — меньше ").append(FEW).append(" людей) ===\n");
            r.append("строка | группа | проб | людей | П2 FAR 1 % | П3 FAR 1 % | П2 FAR 5 % | П3 FAR 5 %\n");
            for (G v : G.values()) {
                List<int[]> b = bins.get(v);
                for (Map.Entry<String, FeiThreshold.Res> e : res.get(v).entrySet()) {
                    FeiDecompose.Cfg c = e.getValue().cfgs.get(0);
                    for (int k = 0; k < names[t].length; k++) {
                        List<Integer> idx = new ArrayList<>();
                        for (int i = 0; i < b.size(); i++) if (b.get(i)[t] == k) idx.add(i);
                        if (idx.isEmpty()) continue;
                        FeiDecompose.Cfg s = XBaseFeret.subset(c, idx);
                        long people = idx.stream().map(i -> c.ownP[i]).distinct().count();
                        FeiThreshold.Res sr = FeiThreshold.evaluate(e.getKey(), e.getValue().id, List.of(s), List.of(imps.get(e.getKey())), g.persons.size(), g.b,
                                g.seed + FeiBootstrap.SEED_OFFSET);
                        r.append(String.join(" | ", e.getKey(), names[t][k], Integer.toString(idx.size()), people + (people < FEW ? " (мало)" : ""),
                                FeiThreshold.errCell(sr, 2, 0, 0, g.b), FeiThreshold.errCell(sr, 3, 0, 0, g.b), FeiThreshold.errCell(sr, 2, 1, 0, g.b),
                                FeiThreshold.errCell(sr, 3, 1, 0, g.b))).append('\n');
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- шапка

    static void protocol(StringBuilder r, XBaseEval.Gt g, String base) {
        r.append("Протокол (а2): для пробы человека с даты d в галерее и в обучении MLDA нет ни одного его снимка с даты d (исключается вся дата);\n"
                + "у прочих людей — все даты. Г-все — галерея человека для пробы — все его другие даты (основной); Г-прошлые — только даты\n"
                + "раньше даты пробы (справочно, как проходная; первая дата пробой не бывает); Г-первая — только первая дата, пробы те же, что\n"
                + "у Г-прошлые (роль (1) feret; справочно, своя калибровка).\n");
        r.append("Схема моделей: свои — 5 частей по людям (XBaseOther.cvParts, seed faces.seed + 8000 + 100 + 0); модель (q, k): у людей части q с ≥ k\n"
                + "датами из галереи и обучения убирается k-я дата (Г-все) или даты с k-й (Г-прошлые), у прочих — все даты; пробы модели — снимки\n"
                + "k-й даты людей части q. Оговорка: у соседей пробы по части q в той же модели тоже нет k-й даты — это влияет только на\n"
                + "второго кандидата (d₂) и путь «верный кандидат», не на требование к пробе.\n");
        r.append("Порог: Г-все и Г-прошлые — одна калибровка П2/П3 на метод по полной галерее (все даты всех своих; 5 частей, модель без части,\n"
                + "«чужие» — все снимки людей части); контрольные чужие — против полной модели. Г-первая — как роль (1) feret: П2/П3 по «чужим»\n"
                + "галереи первой даты, контроль — против её модели. Рядом — разделимость при равном фактическом FAR контроля (от порога не\n"
                + "зависит). Зеркала и выбор модели по позе r0 = ").append(g.r0).append(" (кроме SFace); оценка d₁/d₂.\n");
        r.append("Методы: L1 = LBP u2 (8, 2) 7×7 + MLDA (none); L1-tt — то же на выходе Tan – Triggs; LTP-in-tt — LTP по выходу tt, внутренние\n"
                + "блоки; A = CLAHE + MLDA (справочно); SFace — alignCrop 112×112, шаблон — среднее признаков (минимум снимков шаблона 1);\n"
                + "когорта MLDA — FEI 20 (seed faces.seed + 9000). Даты пробы и галереи — по ").append(base).append(".\n");
        if (g.b > 0) {
            r.append(String.format(Locale.ROOT, "Бутстреп по людям (свои и контроль, %d): B = %d, seed = %d; одни веса для всех строк; порог — заново в каждой "
                    + "реплике; «*» — интервал разности не содержит 0.%n", g.persons.size(), g.b, g.seed + FeiBootstrap.SEED_OFFSET));
        }
    }

    // ---------------------------------------------------------------- FERET

    static void feret(FarExport.Loaded export, Path exportDir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings)
            throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("XBaseEval feret-excl: задайте faces.feret.dir / FACES_FERET_DIR");
        Path dir = Paths.get(rawDir);
        double t = XBaseLtp.threshold(settings.loadFacesLtpT());
        XBaseEval.Gt tpl = new XBaseEval.Gt();
        tpl.dir = exportDir;
        tpl.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval feret-excl: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        tpl.r0 = FeiPose.r0(rawR0, "+pose");
        tpl.b = settings.loadFacesFeiBootstrap();
        tpl.threads = settings.loadFacesFeiCvThreads();
        tpl.rLimit = XBaseEval.poseLimit(selLines);
        tpl.params = IlluminationNorm.NormParams.of(settings);
        boolean eq = settings.loadFacesFeretGalleryEqual(); // №9: равные галереи — по умолчанию, old — прежние
        Path out = eq ? dir.resolve("reports").resolve(XBaseEval.eqDir(0, true)) : dir.resolve("reports");
        Files.createDirectories(out);

        // Снимки, детекции, кадры — как XBaseFeret.run.
        List<Img> imgs = FeretData.frontal(dir);
        Path models = Paths.get(settings.loadFacesModelsDir());
        Path sfaceModel = models.resolve(FaceAlignment.SFACE_FILE);
        FaceRecognizerSF sface = FaceRecognizerSF.create(sfaceModel.toString(), "");
        Map<String, double[]> det = FeretData.detect(imgs, dir, models, sface);
        Map<String, Double> rOf = new HashMap<>();
        Map<String, Integer> imageNo = new HashMap<>();
        Map<String, Map<String, List<Img>>> kept = new TreeMap<>();
        Map<String, Integer> seen = new HashMap<>();
        int notFound = 0;
        int poseOut = 0;
        for (Img im : imgs) {
            imageNo.put(im.key(), seen.merge(im.person(), 1, Integer::sum));
            double[] row = det.get(im.key());
            if (row == null) {
                notFound++;
                continue;
            }
            double r = Math.abs(FeiDataset.pose(row));
            rOf.put(im.key(), r);
            if (r > tpl.rLimit) {
                poseOut++;
                continue;
            }
            kept.computeIfAbsent(im.person(), p -> new TreeMap<>()).computeIfAbsent(im.date(), x -> new ArrayList<>()).add(im);
        }
        XBaseFeret.Roles ro = XBaseFeret.roles(XBaseFeret.dates(imgs), kept, tpl.seed);

        // Свои — роль (1) feret (первая дата и хотя бы одна более поздняя с отобранными снимками); даты — с отобранными снимками.
        Map<String, List<List<String>>> sess = new LinkedHashMap<>();
        Map<String, List<LocalDate>> when = new LinkedHashMap<>();
        for (String p : ro.own1) {
            List<List<String>> ds = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (Map.Entry<String, List<Img>> e : kept.get(p).entrySet()) {
                ds.add(XBaseFeret.keys(e.getValue()));
                ws.add(FeretData.date(e.getKey()));
            }
            sess.put(p, ds);
            when.put(p, ws);
        }
        Map<String, Img> byKey = new HashMap<>();
        for (Img im : imgs) byKey.put(im.key(), im);
        Data d = new Data(ro.own1, sess, when, eq ? pairSessions(sess, byKey) : sess);

        // Когорта FEI 20.
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, tpl.seed);
        List<List<String>> cohort = new ArrayList<>();
        Set<String> cohortKeys = new HashSet<>();
        for (String p : cohortP) {
            List<String> ks = new ArrayList<>(feiKeys.get(p).values());
            cohort.add(ks);
            cohortKeys.addAll(ks);
        }

        // Векторы: снимки своих (все даты), контроль, когорта.
        Set<String> used = new TreeSet<>();
        sess.values().forEach(ds -> ds.forEach(used::addAll));
        ro.ctrlKeys.values().forEach(used::addAll);
        Map<String, Mat> fr = new LinkedHashMap<>();
        Map<String, Mat> sf = new LinkedHashMap<>();
        for (String k : used) {
            fr.put(k, FeretData.frame(dir, "a", k));
            sf.put(k, FeretData.frame(dir, "sface", k));
        }
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval[] xs = {new XBaseEval(), new XBaseEval(), new XBaseEval()};
        xs[0].vectors(fr, EnumSet.of(V.L1, V.A), tpl.params);
        xs[1].vectors(fr, EnumSet.of(V.L1), tpl.params, IlluminationNorm.TAN_TRIGGS);
        xs[1].ltpVectors(fr, tpl.params, t);
        xs[2].sfaceMin = 1;
        xs[2].sfaceVectors(sf, sface);
        fr.values().forEach(Mat::release);
        sf.values().forEach(Mat::release);

        TreeSet<String> ps = new TreeSet<>(d.own());
        ps.addAll(ro.ctrl);
        XBaseEval.Gt g = XBaseFeret.withOwn(XBaseFeret.gt(tpl, new ArrayList<>(ps), ro.ctrl, ro.ctrlKeys, cohort, rOf, imageNo), d.own());

        String fp = FeiCheckpoint.fingerprint(exportDir, String.format(Locale.ROOT, "feret-excl %s; снимков %d; детекций найдено %d; seed %d; r0 %s; t %s", dir,
                imgs.size(), imgs.size() - notFound, tpl.seed, tpl.r0, t) + (eq ? "; №8 галерея fa/fb" : ""));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(dir.resolve(FeretData.DERIVED).resolve(eq ? "checkpoint_feret_excl_eq" : "checkpoint_feret_excl"), fp);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а2) исключение даты пробы на FERET (XBaseEval feret-excl): Color FERET — только локально (Release Agreement v1)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: ").append(dir).append(" (fa и fb; YuNet и кадры — как feret, кэш derived/); когорта FEI — экспорт ").append(exportDir)
                .append(", коммит данных ").append(FarMethods.dataCommit(exportDir)).append('\n');
        r.append(String.format(Locale.ROOT, "Снимков fa/fb: %d; YuNet не нашёл лица: %d; отбор по позе |r| ≤ R = %.4f исключил: %d.%n", imgs.size(), notFound,
                tpl.rLimit, poseOut));
        r.append(String.format(Locale.ROOT, "Свои — люди с анфасом на ≥ 2 датах (роль (1) feret): %d (выбыли после отбора %d); дат с отобранными снимками у "
                + "своих: %s.%n", d.own().size(), ro.dropped[0], dateHist(d)));
        r.append(String.format(Locale.ROOT, "Контрольные чужие — те же, что в feret: %d человек с одной датой (Random(seed + %d)), %d снимков.%n", ro.ctrl.size(),
                XBaseFeret.CTRL_SEED_OFFSET, XBaseFeret.sum(ro.ctrlKeys)));
        r.append("Модель SFace: ").append(sfaceModel.getFileName()).append(" (").append(FeiThreshold.sha(sfaceModel)).append(").\n");
        if (eq) r.append(EQ_NOTE);
        protocol(r, g, "дате в имени файла FERET");
        StringBuilder lines = new StringBuilder();
        evaluate(g, d, xs, XBaseFeret.SETS, cp, r, lines, "feret-excl");
        XBaseOther.writeScores(out.resolve("xbase_feret_excl_scores.tsv.gz"), "# FERET (а2), XBaseExcl, только локально; s: 0 — Г-первая, 1 — полная модель "
                + "(контроль, cv), 2 — Г-все, 3 — Г-прошлые; j — раунд k\n# Код: коммит " + FarMethods.commit() + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_feret_excl.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /** «2 даты — 112, 3 — 36, …». */
    static String dateHist(Data d) {
        Map<Integer, Integer> h = new TreeMap<>();
        for (String p : d.own()) h.merge(d.sess().get(p).size(), 1, Integer::sum);
        List<String> out = new ArrayList<>();
        h.forEach((n, c) -> out.add(n + " — " + c));
        return String.join(", ", out);
    }

    // ---------------------------------------------------------------- GT

    /** Дата каждой сессии GT: «человек/номер» → дата (первое поле exif_datetime). */
    static Map<String, LocalDate> gtDates(Path file) throws IOException {
        Map<String, LocalDate> out = new HashMap<>();
        for (String l : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (l.startsWith("#") || l.startsWith("person\t") || l.isBlank()) continue;
            String[] c = l.split("\t", -1);
            if (c[3].equals("-")) continue;
            String[] ymd = c[2].substring(0, 10).split(":");
            LocalDate dt = LocalDate.of(Integer.parseInt(ymd[0]), Integer.parseInt(ymd[1]), Integer.parseInt(ymd[2]));
            LocalDate old = out.putIfAbsent(c[0] + "/" + c[3], dt);
            if (old != null && !old.equals(dt)) throw new IllegalStateException("GT: у сессии " + c[0] + "/" + c[3] + " две даты");
        }
        return out;
    }

    static void gt(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        double t = XBaseLtp.threshold(settings.loadFacesLtpT());
        XBaseEval.Gt g = XBaseEval.prepare(export, dir, feiKeys, selLines, settings, true, false);
        g.x.sfaceMin = 1;
        Map<String, LocalDate> dates = gtDates(dir.resolve(XBaseEval.SESSIONS));

        // Свои — ≥ 2 датами с отобранными снимками (снимки без даты не берутся).
        List<String> own = new ArrayList<>();
        List<String> out1 = new ArrayList<>();
        int noDate = 0;
        Map<String, List<List<String>>> sess = new LinkedHashMap<>();
        Map<String, List<LocalDate>> when = new LinkedHashMap<>();
        for (String p : g.own) {
            List<List<String>> ds = new ArrayList<>();
            List<LocalDate> ws = new ArrayList<>();
            for (Map.Entry<Integer, List<String>> e : g.byPerson.get(p).entrySet()) {
                if (e.getKey() == 0) {
                    noDate += e.getValue().size();
                    continue;
                }
                ds.add(sorted(e.getValue()));
                ws.add(dates.get(p + "/" + e.getKey()));
            }
            if (ds.size() < 2) {
                out1.add(p);
                continue;
            }
            own.add(p);
            sess.put(p, ds);
            when.put(p, ws);
        }
        g.own = own;
        Data d = new Data(own, sess, when);

        // Векторы tt (L1-tt, LTP-in-tt) — как gt-ltp.
        Map<String, Mat> fr = XBaseEval.frames(export, g.samples.values().stream().sorted((u, w) -> u.file().toString().compareTo(w.file().toString()))
                .toList(), null);
        Set<String> cohortKeys = new HashSet<>();
        g.cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
        XBaseEval xt = new XBaseEval();
        xt.vectors(fr, EnumSet.of(V.L1), g.params, IlluminationNorm.TAN_TRIGGS);
        xt.ltpVectors(fr, g.params, t);
        fr.values().forEach(Mat::release);
        XBaseEval[] xs = {g.x, xt, g.x};

        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "gt-excl; данные %s; seed %d; r0 %s; t %s", FarMethods.dataCommit(dir), g.seed, g.r0,
                t));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_gt_excl"), fp);
        StringBuilder r = new StringBuilder();
        r.append("PLAN.md п. 5.5, (а2) исключение даты пробы на Georgia Tech (XBaseEval gt-excl)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir))
                .append(" (a/gt — кадр а 92×112; sface/gt — alignCrop 112×112; когорта — a/fei; даты — reports/gt_sessions.tsv)\n");
        r.append("Оговорка: у большинства своих 2 даты — галерея почти везде односессионная (почти повтор gt с обменом ролей дат); снимки внутри\n"
                + "даты — почти дубли; 30 своих — различимы только крупные эффекты.\n");
        r.append(String.format(Locale.ROOT, "Свои — %d (люди GT с ≥ 2 датами и ≥ 2 датами после отбора; выбыли %s; снимков своих без даты, не взяты: %d); "
                + "дат у своих: %s; контрольные чужие — %d человек.%n", own.size(), out1.isEmpty() ? "нет" : String.join(", ", out1), noDate, dateHist(d),
                g.ctrl.size()));
        r.append("Модель SFace: ").append(g.sfaceModel).append(".\n");
        protocol(r, g, "дате EXIF (gt_sessions.tsv)");
        StringBuilder lines = new StringBuilder();
        evaluate(g, d, xs, XBaseFeret.SETS, cp, r, lines, "gt-excl");
        XBaseOther.writeScores(out.resolve("xbase_gt_excl_scores.tsv.gz"), "# GT (а2), XBaseExcl; s: 0 — Г-первая, 1 — полная модель (контроль, cv), "
                + "2 — Г-все, 3 — Г-прошлые; j — раунд k\n# Код: коммит " + FarMethods.commit() + "\n# Данные: экспорт " + dir + ", коммит данных "
                + FarMethods.dataCommit(dir) + "\n", lines.toString());
        FarMethods.write(out.resolve("xbase_gt_excl.txt"), r);
        System.out.print(r);
        cp.delete();
    }
}
