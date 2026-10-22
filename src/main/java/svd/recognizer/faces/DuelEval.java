package svd.recognizer.faces;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import svd.recognizer.faces.DuelSet.Item;
import svd.recognizer.faces.FeiScores.Row;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.ProkaevMethod.Variant;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№2 (дуэль; PLAN.md п. 5.5, (1а), Д12, Д13): наш расчёт на наборе для коллеги – режим duel-eval, по файлам набора (FACES_DUEL_DIR) и
 * ключу reports/faces/duel/key.tsv (duel-set); до счёта – сверка файлов набора с описью set_sha256.txt (расхождение – остановка).
 * <p>
 * Строки (по порядку): L1-tt-xs-F, L1-xs-F – YuNet 0,7 в два прохода на PNG 400 × 600 → отбор |r| ≤ R → кадр «а» → LBP, схема (Wа): MLDA на
 * 10 людях галереи, когорте FEI 20 и пуле FERET (люди с ≥ 2 датами, только чтение), модель X / X+mirror по |r| ≤ r0; порог – П3 по «чужим»
 * галереи (5 частей по 2 человека, seed faces.seed + 8100, как протокол GT), рядом – разделимость; SFace – тот же конвейер до alignCrop,
 * шаблон – нормированное среднее, 1 − cos ≤ 0,53 (решение Хозяина (24), N = 10); метод Прокаева (Б2-К, k = 6, l = 3) – серый кадр набора
 * целиком, гости G01…G10 в обучающей матрице, правило автора 0,95. Справочно – Б2-н, Б2-с и все три без гостей. Отказ детектора или отбора у
 * наших методов – отвергнутая проба (снимок галереи с отказом – вне шаблона).
 *
 * @author ssv
 */
final class DuelEval {

    static final double SFACE_T = 0.53;
    static final int K = ProkaevEval.K;
    static final int L = ProkaevEval.L;

    private DuelEval() {
    }

    /** Строка наших методов: подпись, вектор (L1 без / с Tan – Triggs или SFace), пул в обучении. */
    record Meth(String label, int feat, V v, boolean pool) {}

    /** Итог пробы одной строки: прошла ли детектор и отбор (иначе причина), лучший кандидат, верен ли он, оценка, принята ли при рабочем пороге. */
    record Out(String code, String role, String person, String fail, String best, boolean ok, double score, boolean accepted, String second, double secondScore) {}

    /** Строка метода Прокаева: подпись, вариант, гости в обучении. */
    record PCfg(String label, Variant v, boolean guests) {}

    static final List<PCfg> PCFGS = List.of(new PCfg("метод Прокаева (Б2-К), гости G01…G10", Variant.K, true),
            new PCfg("Б2-н, гости G01…G10 (справочно)", Variant.N, true), new PCfg("Б2-с, гости G01…G10 (справочно)", Variant.S, true),
            new PCfg("Б2-К без гостей (справочно)", Variant.K, false), new PCfg("Б2-н без гостей (справочно)", Variant.N, false),
            new PCfg("Б2-с без гостей (справочно)", Variant.S, false));

    static boolean probe(Item it) {
        return it.code().startsWith("Q");
    }

    static boolean own(Item it) {
        return it.role().equals(DuelSet.SAME_ROLE) || it.role().equals(DuelSet.OTHER_ROLE);
    }

    /** Классы метода Прокаева по убыванию доли, при равных – по расстоянию до ближайшей точки; гости – «гость». */
    static List<Map.Entry<String, Double>> ranked(ProkaevMethod.Result r) {
        List<Map.Entry<String, Double>> e = new ArrayList<>(r.share().entrySet());
        e.sort(Comparator.<Map.Entry<String, Double>>comparingDouble(x -> -x.getValue()).thenComparingDouble(x -> r.near().get(x.getKey())));
        return e;
    }

    static String cls(String c) {
        return c.equals(ProkaevMethod.GUEST) ? DuelSet.GUEST_ROLE : c;
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        String rawDuel = System.getenv("FACES_DUEL_DIR");
        if (rawDuel == null) throw new IllegalStateException("duel-eval: задайте FACES_DUEL_DIR (каталог набора)");
        Path duel = Paths.get(rawDuel);
        Path rep = out.getParent().resolve("duel");
        // Сверка набора с описью duel-set.
        List<String> inv = Files.readAllLines(rep.resolve("set_sha256.txt"), StandardCharsets.UTF_8);
        List<Path> files;
        try (Stream<Path> st = Files.walk(duel)) {
            files = st.filter(Files::isRegularFile).sorted().toList();
        }
        if (files.size() != inv.size()) throw new IllegalStateException("duel-eval: файлов набора " + files.size() + ", в описи " + inv.size());
        for (int i = 0; i < files.size(); i++) {
            String l = DuelSet.sha256(files.get(i)) + "  " + duel.relativize(files.get(i)).toString().replace('\\', '/');
            if (!l.equals(inv.get(i))) throw new IllegalStateException("duel-eval: набор не совпал с описью: " + l);
        }
        List<Item> key = DuelSet.readKey(rep.resolve(DuelSet.KEY));
        long seed = settings.loadFacesSeed();
        double rLimit = XBaseEval.poseLimit(selLines);
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("duel-eval: задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        double r0 = FeiPose.r0(rawR0, "+pose");
        Path models = Paths.get(settings.loadFacesModelsDir());
        FaceDetectorYN detector = OwnDetectorThreshold.create(models, FeretData.SCORE);
        FaceRecognizerSF sface = FaceRecognizerSF.create(models.resolve(FaceAlignment.SFACE_FILE).toString(), "");
        IlluminationNorm.NormParams params = IlluminationNorm.NormParams.of(settings);
        int threads = settings.loadFacesFeiCvThreads();

        // Когорта FEI 20 и пул FERET (только чтение кадров «а»).
        List<String> cohortP = XBaseEval.cohortPersons(feiKeys, seed);
        List<List<String>> cohort = new ArrayList<>();
        for (String p : cohortP) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
        Set<String> cohortKeys = new HashSet<>();
        cohort.forEach(cohortKeys::addAll);
        List<Sample> fei = new ArrayList<>();
        export.data().fei().values().forEach(fei::addAll);
        XBaseA0.FeretSetup su = XBaseA0.feretSetup(dir, selLines, settings);
        Map<String, List<List<String>>> pool = XBaseXsF.pool(su.ro, su.byKey);
        List<List<String>> poolCls = XBaseXsF.train(XBaseXsF.Kind.XS, List.of(), Map.of(), pool, Set.of(), Map.of()).extra();
        Set<String> poolKeys = new TreeSet<>();
        pool.values().forEach(ds -> ds.forEach(poolKeys::addAll));

        List<String> ownP = key.stream().filter(it -> it.code().startsWith("P")).map(it -> it.code().substring(0, 3)).distinct().sorted().toList();
        Map<String, String> codeOf = new HashMap<>();
        key.stream().filter(it -> it.code().startsWith("P")).forEach(it -> codeOf.put(it.person(), it.code().substring(0, 3)));
        List<String> strangers = key.stream().filter(it -> it.role().equals(DuelSet.STRANGER_ROLE)).map(Item::person).distinct().sorted().toList();
        List<Meth> meths = List.of(new Meth("L1-tt-xs-F", 1, V.L1, true), new Meth("L1-xs-F", 0, V.L1, true), new Meth("SFace", 0, V.SFACE, false));

        StringBuilder r = new StringBuilder("Д-№2 (дуэль; PLAN.md п. 5.5, (1а), Д12, Д13): наш расчёт на наборе для коллеги (XBaseEval duel-eval)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (detections.tsv, gt_sessions.tsv, a/fei); "
                + "снимки – gt_db; пул – Color FERET (только чтение, имён снимков нет)\n");
        r.append(String.format(Locale.ROOT, "Набор: %s, файлов %d (опись – set_sha256.txt, сверена); seed: свои %d, чужие %d, перемешивание %d, гости %d; части П3 – "
                + "seed %d.%n", duel, files.size(), seed + DuelSet.OWN_SEED, seed + DuelSet.STRANGER_SEED, seed + DuelSet.PERM_SEED, seed + DuelSet.GUEST_SEED,
                seed + 8000 + XBaseEval.CV_S));
        Map<String, Integer> roles = new LinkedHashMap<>();
        key.forEach(it -> roles.merge(it.role(), 1, Integer::sum));
        r.append("Состав (GT, отбор |r| ≤ R = ").append(rLimit).append("): ").append(roles).append("; свои 10 человек, чужие ").append(strangers.size())
                .append(" человек, гости 10.\n");
        r.append("Кадры набора вырезаны по рамке YuNet на исходных снимках (одинаково для всех методов). На файлах набора метод Прокаева детектор не "
                + "запускает; наши методы запускают YuNet и отбор по позе заново; их отказы засчитаны как отвергнутые, число отказов – отдельно.\n");
        r.append("10 человек – П3 ненадёжен (verify_p3_e6c768d), выводов о FAR не делать. FAR – числом принятых чужих проб из 20 и людей из 10.\n");
        StringBuilder probesTsv = new StringBuilder("set\tmethod\tQ\trole\tperson\tfail\tbest\tbest_score\tsecond\tsecond_score\tok\taccepted\n");

        for (int s = 0; s < DuelSet.SETS.length; s++) {
            String set = DuelSet.SETS[s];
            String tag = DuelSet.SET_TAGS[s];
            Path base = duel.resolve(set);
            // --- наш конвейер на файлах набора
            Map<String, Mat> fa = new LinkedHashMap<>();
            Map<String, Mat> fs = new LinkedHashMap<>();
            Map<String, Double> rOf = new HashMap<>();
            Map<String, String> fail = new LinkedHashMap<>();
            Map<String, float[]> gray = new HashMap<>();
            for (Item it : key) {
                Mat color = DuelSet.readColor(base.resolve(it.rel()));
                Mat g = new Mat();
                Imgproc.cvtColor(color, g, Imgproc.COLOR_BGR2GRAY);
                gray.put(it.code(), ProkaevFrames.row(g));
                g.release();
                if (!it.role().equals(DuelSet.GUEST_ROLE)) {
                    double[] first = YunetTwoPass.firstPass(detector, color, 1.0);
                    double[] row = null;
                    if (first != null) {
                        double[] second = YunetTwoPass.secondPass(detector, color, new double[] {first[0], first[1], first[2], first[3]});
                        row = second != null ? second : first;
                    }
                    String k = tag + "/" + it.code();
                    if (row == null) fail.put(it.code(), "детектор");
                    else if (Math.abs(FeiDataset.pose(row)) > rLimit) fail.put(it.code(), "отбор");
                    else {
                        rOf.put(k, Math.abs(FeiDataset.pose(row)));
                        fa.put(k, FeretData.frameA(color, row));
                        fs.put(k, FeretData.frameSface(color, row, sface));
                    }
                }
                color.release();
            }
            Map<String, Integer> imageNo = new HashMap<>();
            Map<String, List<String>> gal = new LinkedHashMap<>();
            for (String p : ownP) gal.put(p, new ArrayList<>());
            for (Item it : key) {
                String k = tag + "/" + it.code();
                if (it.code().startsWith("P")) {
                    imageNo.put(k, Integer.parseInt(it.code().substring(4)));
                    if (!fail.containsKey(it.code())) gal.get(it.code().substring(0, 3)).add(k);
                } else if (probe(it)) {
                    imageNo.put(k, Integer.parseInt(it.code().substring(1)));
                }
            }
            for (Map.Entry<String, List<String>> e : gal.entrySet()) if (e.getValue().size() < 2) throw new IllegalStateException("duel-eval: у " + e.getKey() + " в галерее "
                    + e.getValue().size() + " снимков после отказов");
            List<Probe> probes = new ArrayList<>();
            List<String> persons = new ArrayList<>(ownP);
            persons.addAll(strangers);
            Map<String, Integer> pIdx = new HashMap<>();
            for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
            for (Item it : key) {
                if (!probe(it) || fail.containsKey(it.code())) continue;
                String k = tag + "/" + it.code();
                if (own(it)) probes.add(new Probe(FeiScores.OWN, codeOf.get(it.person()), imageNo.get(k), k, ownP.indexOf(codeOf.get(it.person()))));
                else probes.add(new Probe(FeiScores.CTRL, it.person(), imageNo.get(k), k, -1));
            }
            Map<String, Mat> fr = new LinkedHashMap<>(fa);
            fr.putAll(XBaseEval.frames(export, fei, cohortKeys));
            for (String k : poolKeys) fr.put(k, FeretData.frame(su.dir, "a", k));
            XBaseEval[] xa = {XBaseAbl.vectors(XBaseAbl.Sch.BASE, 0, fr, params), XBaseAbl.vectors(XBaseAbl.Sch.BASE, 1, fr, params)};
            fr.values().forEach(Mat::release);
            XBaseEval xs = new XBaseEval();
            xs.sfaceVectors(fs, sface);
            fs.values().forEach(Mat::release);
            List<List<String>> parts = FeiMethods.cvParts(ownP, seed, XBaseEval.CV_S);

            Map<String, List<Out>> res = new LinkedHashMap<>();
            Map<String, Double> thr = new LinkedHashMap<>();
            for (Meth m : meths) {
                XBaseEval.Gt g = new XBaseEval.Gt();
                g.seed = seed;
                g.r0 = r0;
                g.threads = threads;
                g.params = params;
                g.own = ownP;
                g.ctrl = strangers;
                g.persons = persons;
                g.pIdx = pIdx;
                g.rOf = rOf;
                g.imageNo = imageNo;
                g.cohort = new ArrayList<>(cohort);
                if (m.pool()) g.cohort.addAll(poolCls);
                XBaseEval x = m.v() == V.SFACE ? xs : xa[m.feat()];
                String lines = XBaseEval.fitLines(g, x, EnumSet.of(m.v()), gal, probes, parts, 0);
                Map<String, Row> rows = new HashMap<>();
                for (String l : lines.lines().toList()) {
                    Row rw = FeiScores.parse(l);
                    rows.put(rw.method() + "\t" + rw.role() + "\t" + rw.person() + "\t" + rw.image(), rw);
                }
                double theta;
                if (m.v() == V.SFACE) {
                    theta = SFACE_T;
                } else {
                    XBaseEval.Parsed pa = XBaseEval.parse(g, EnumSet.of(m.v()), gal, probes, parts, lines);
                    theta = FeiThreshold.threshold(3, pa.cfg.get(m.v()), pa.imp.get(m.v()), FeiBootstrap.unit(persons.size()), 0.01).theta();
                }
                thr.put(m.label(), theta);
                List<Out> outs = new ArrayList<>();
                for (Item it : key) {
                    if (!probe(it)) continue;
                    if (fail.containsKey(it.code())) {
                        outs.add(new Out(it.code(), it.role(), it.person(), fail.get(it.code()), "-", false, Double.POSITIVE_INFINITY, false, "-", Double.POSITIVE_INFINITY));
                        continue;
                    }
                    String k = tag + "/" + it.code();
                    boolean o = own(it);
                    Row rw = XBaseEval.choose(rows, m.v(), o ? FeiScores.OWN : FeiScores.CTRL, o ? codeOf.get(it.person()) : it.person(), imageNo.get(k),
                            rOf.get(k) <= r0);
                    double sc = m.v() == V.SFACE ? rw.best() : FeiScores.ratio(rw, null);
                    boolean ok = o && rw.bestId().equals(rw.trueId());
                    outs.add(new Out(it.code(), it.role(), it.person(), "", rw.bestId(), ok, sc, sc <= theta, rw.secondId(), rw.second()));
                }
                res.put(m.label(), outs);
            }
            // --- метод Прокаева: серые кадры набора целиком
            Map<String, List<Out>> pres = new LinkedHashMap<>();
            StringBuilder csv = new StringBuilder(DuelSet.BOM + "Q;кандидат 1;% 1;кандидат 2;% 2;решение (свой / гость);примечание\n");
            for (PCfg pc : PCFGS) {
                List<float[]> a = new ArrayList<>();
                List<String> lab = new ArrayList<>();
                for (Item it : key) {
                    if (it.code().startsWith("P")) {
                        a.add(gray.get(it.code()));
                        lab.add(it.code().substring(0, 3));
                    } else if (it.code().startsWith("G") && pc.guests()) {
                        a.add(gray.get(it.code()));
                        lab.add(ProkaevMethod.GUEST);
                    }
                }
                ProkaevMethod pm = ProkaevMethod.fit(a.toArray(float[][]::new), lab.toArray(String[]::new), null, ProkaevEval.KMAX);
                List<Out> outs = new ArrayList<>();
                for (Item it : key) {
                    if (!probe(it)) continue;
                    ProkaevMethod.Result pr = pm.decide(pm.project(gray.get(it.code()), pc.v()), pc.v(), K, L);
                    List<Map.Entry<String, Double>> rk = ranked(pr);
                    String c1 = cls(rk.get(0).getKey());
                    String c2 = rk.size() > 1 ? cls(rk.get(1).getKey()) : "-";
                    double p1 = rk.get(0).getValue();
                    double p2 = rk.size() > 1 ? rk.get(1).getValue() : 0;
                    boolean o = own(it);
                    boolean ok = o && pr.best().equals(codeOf.get(it.person()));
                    outs.add(new Out(it.code(), it.role(), it.person(), "", pr.best(), ok, pr.score(), !pr.guest(), c1 + " " + fmt(p1) + " / " + c2 + " " + fmt(p2), Double.NaN));
                    if (pc == PCFGS.get(0)) csv.append(String.format(Locale.ROOT, "%s;%s;%s;%s;%s;%s;%n", it.code(), c1, fmt(p1), c2, rk.size() > 1 ? fmt(p2) : "",
                            pr.guest() ? DuelSet.GUEST_ROLE : "свой " + pr.best()));
                }
                pres.put(pc.label(), outs);
            }
            Files.writeString(rep.resolve("prokaev_" + tag + ".csv"), csv.toString(), StandardCharsets.UTF_8);

            // --- отчёт комплекта
            r.append("\n================ Комплект ").append(set).append(" ================\n");
            Map<String, int[]> fc = new LinkedHashMap<>();
            for (Item it : key) {
                if (it.role().equals(DuelSet.GUEST_ROLE)) continue;
                int[] c = fc.computeIfAbsent(it.role(), x -> new int[3]);
                c[0]++;
                String f = fail.get(it.code());
                if ("детектор".equals(f)) c[1]++;
                if ("отбор".equals(f)) c[2]++;
            }
            r.append("Отказы наших методов (YuNet 0,7 в два прохода на файлах набора; отбор |r| ≤ R): роль – снимков / детектор / отбор: ");
            fc.forEach((k, c) -> r.append(String.format(Locale.ROOT, "%s %d / %d / %d; ", k, c[0], c[1], c[2])));
            r.append("\nПороги: ");
            thr.forEach((k, v) -> r.append(String.format(Locale.ROOT, "%s %s; ", k, k.equals("SFace") ? "1 − cos ≤ " + SFACE_T + " (решение (24), N = 10)"
                    : "П3 θ = " + v)));
            r.append("метод Прокаева – правило автора (доля гостей > 0 и доля каждого своего < 0,95 – гость)\n");
            Map<String, List<Out>> all = new LinkedHashMap<>(res);
            all.put(PCFGS.get(0).label(), pres.get(PCFGS.get(0).label()));
            table(r, all, "Основные строки");
            Map<String, List<Out>> ref = new LinkedHashMap<>(pres);
            ref.remove(PCFGS.get(0).label());
            table(r, ref, "Справочно – варианты метода Прокаева (ниже черты)");
            sep(r, res);
            for (Map.Entry<String, List<Out>> e : res.entrySet()) for (Out o : e.getValue()) probesTsv.append(String.join("\t", tag, e.getKey(), o.code(), o.role(),
                    o.person(), o.fail(), o.best(), Double.toString(o.score()), o.second(),
                    Double.toString(o.secondScore()), Boolean.toString(o.ok()), Boolean.toString(o.accepted()))).append('\n');
            for (Map.Entry<String, List<Out>> e : pres.entrySet()) for (Out o : e.getValue()) probesTsv.append(String.join("\t", tag, e.getKey(), o.code(), o.role(),
                    o.person(), "", o.best(), Double.toString(o.score()), o.second(), "",
                    Boolean.toString(o.ok()), Boolean.toString(o.accepted()))).append('\n');
        }
        r.append("\nПо каждой пробе – duel_70_probes.tsv; таблицы метода Прокаева (Б2-К, гости G01…G10) в виде таблицы коллеги – prokaev_face.csv, prokaev_plain.csv.\n");
        r.append("\n=== SHA-256 файлов набора ===\n");
        inv.forEach(l -> r.append(l).append('\n'));
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с.%n", (System.nanoTime() - t0) / 1e9));
        FarMethods.write(rep.resolve("duel_70_probes.tsv"), probesTsv);
        FarMethods.write(rep.resolve("duel_70.txt"), r);
        System.out.print(r);
    }

    static String fmt(double share) {
        return String.format(Locale.ROOT, "%.1f", 100 * share);
    }

    /** Главные меры: верный первый кандидат, свои – приняты верно / за другого / отвергнуты, отдельно «тот же день» и «другой день»; чужие – принятые. */
    static void table(StringBuilder r, Map<String, List<Out>> rows, String title) {
        r.append("\n=== ").append(title).append(" ===\n");
        r.append("строка | тот же день (20): верный первый; приняты верно / за другого / отвергнуты | другой день (30): то же | чужие приняты: проб из 20, "
                + "людей из 10\n");
        for (Map.Entry<String, List<Out>> e : rows.entrySet()) {
            StringBuilder line = new StringBuilder(e.getKey());
            for (String role : new String[] {DuelSet.SAME_ROLE, DuelSet.OTHER_ROLE}) {
                int n = 0;
                int first = 0;
                int good = 0;
                int other = 0;
                int rej = 0;
                for (Out o : e.getValue()) {
                    if (!o.role().equals(role)) continue;
                    n++;
                    if (o.ok()) first++;
                    if (!o.accepted()) rej++;
                    else if (o.ok()) good++;
                    else other++;
                }
                line.append(String.format(Locale.ROOT, " | %d из %d; %d / %d / %d", first, n, good, other, rej));
            }
            int acc = 0;
            int n = 0;
            Set<String> accP = new TreeSet<>();
            for (Out o : e.getValue()) {
                if (!o.role().equals(DuelSet.STRANGER_ROLE)) continue;
                n++;
                if (o.accepted()) {
                    acc++;
                    accP.add(o.person());
                }
            }
            line.append(String.format(Locale.ROOT, " | %d из %d, людей %d%n", acc, n, accP.size()));
            r.append(line);
        }
    }

    /** Разделимость (L1-tt-xs-F, L1-xs-F, SFace): порог по 20 чужим – принято 0 и 1; ошибка своих «тот же день» / «другой день». */
    static void sep(StringBuilder r, Map<String, List<Out>> res) {
        r.append("\n=== Разделимость (порог подобран на этих же 20 чужих; мера разнесения оценок, не рабочий порог) ===\n");
        r.append("строка | принято 0 из 20: ошибка своих тот же день / другой день | принято 1 из 20: то же\n");
        for (Map.Entry<String, List<Out>> e : res.entrySet()) {
            double[] st = e.getValue().stream().filter(o -> o.role().equals(DuelSet.STRANGER_ROLE)).mapToDouble(Out::score).sorted().toArray();
            StringBuilder line = new StringBuilder(e.getKey());
            for (int lev = 0; lev < 2; lev++) {
                double th = lev < st.length ? Math.nextDown(st[lev]) : Double.POSITIVE_INFINITY;
                int accepted = 0;
                for (double x : st) if (x <= th) accepted++;
                line.append(" |");
                for (String role : new String[] {DuelSet.SAME_ROLE, DuelSet.OTHER_ROLE}) {
                    int n = 0;
                    int err = 0;
                    for (Out o : e.getValue()) {
                        if (!o.role().equals(role)) continue;
                        n++;
                        if (!(o.ok() && o.score() <= th)) err++;
                    }
                    line.append(String.format(Locale.ROOT, " %d из %d", err, n)).append(role.equals(DuelSet.SAME_ROLE) ? " /" : "");
                }
                if (accepted != lev) line.append(String.format(Locale.ROOT, " (ничьи: принято %d)", accepted));
            }
            r.append(line).append('\n');
        }
    }
}
