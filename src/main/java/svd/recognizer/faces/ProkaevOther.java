package svd.recognizer.faces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.zip.GZIPInputStream;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import svd.recognizer.faces.GalleryEvaluation.Sample;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№1, п. 5: метод Прокаева на других базах по их протоколам (справочно; ProkaevEval – протоколы (4а)).
 * <ul>
 *   <li>muct-prokaev, orl-prokaev – протоколы режимов muct и orl (XBaseOther: 5 ротаций по людям, снимки пополам; ORL – 5 × 2), роли
 *       сверяются с файлами оценок xbase_72c3195 (SVD-faces-data), рядом – L1, L1-in (и A у ORL) из них же, без пересчёта. Гости
 *       MUCT – люди своей когорты MUCT (20; в методе Прокаева когорта не нужна), 30 – не хватает людей; ORL – людей вне ролей нет,
 *       гости – 10 человек FEI (другая база). Контрольная точка – после каждой конфигурации.</li>
 *   <li>fei-prokaev – протокол п. 3а (FeiProtocol, 5 разбиений × 8 конфигураций): гости – люди когорты разбиения (20), справочно –
 *       10 человек GT (другая съёмка); порог – Нейман – Пирсон по снимкам пороговых чужих (FAR 1 % и 5 %), ошибка своих и FAR
 *       контроля – по снимкам, бутстреп по людям FEI. L1 – справочно из готового отчёта fei_630f21a (fei_scores.txt).</li>
 * </ul>
 *
 * @author ssv
 */
final class ProkaevOther {

    static final String REF_DIR = "reports/xbase_72c3195";

    private ProkaevOther() {
    }

    /** Строки файла оценок без шапки, по конфигурации s. */
    static Map<Integer, StringBuilder> bySConf(Path file) throws IOException {
        Map<Integer, StringBuilder> out = new TreeMap<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(Files.newInputStream(file), 1 << 16), StandardCharsets.UTF_8))) {
            String l;
            while ((l = in.readLine()) != null) {
                if (l.startsWith("#") || l.startsWith("method\t") || l.isEmpty()) continue;
                int s = Integer.parseInt(l.substring(l.indexOf('\t') + 1, l.indexOf('\t', l.indexOf('\t') + 1)));
                out.computeIfAbsent(s, x -> new StringBuilder()).append(l).append('\n');
            }
        }
        return out;
    }

    /** Строки метода m (и m+mirror) из строк конфигурации. */
    static String method(String lines, String m) {
        StringBuilder b = new StringBuilder();
        for (String l : lines.split("\n")) {
            String id = l.substring(0, l.indexOf('\t'));
            if (id.equals(m) || id.equals(m + FeiMethods.MIRROR)) b.append(l).append('\n');
        }
        return b.toString();
    }

    // ---------------------------------------------------------------- MUCT, ORL

    static void other(String mode, FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings,
                      Path out) throws IOException {
        long t0 = System.nanoTime();
        boolean muct = mode.equals("muct-prokaev");
        String code = muct ? "muct" : "orl";
        String title = muct ? "MUCT" : "ORL";
        XBaseEval.Gt t = new XBaseEval.Gt();
        t.dir = dir;
        t.seed = settings.loadFacesSeed();
        String rawR0 = settings.loadFacesFeiPoseR0();
        if (rawR0 == null) throw new IllegalStateException("XBaseEval " + mode + ": задайте faces.fei.pose.r0 / FACES_FEI_POSE_R0");
        t.r0 = FeiPose.r0(rawR0, "+pose");
        t.b = settings.loadFacesFeiBootstrap();
        t.threads = settings.loadFacesFeiCvThreads();
        t.rLimit = XBaseEval.poseLimit(selLines);
        Map<String, Double> pose = XBaseEval.poses(dir.resolve("detections.tsv"));
        XBaseOther.Base b = XBaseOther.base(code, title, muct ? export.data().muct() : export.data().orl(), pose, t.rLimit);
        List<String> feiCohortP = XBaseEval.cohortPersons(feiKeys, t.seed);
        List<String> all = new ArrayList<>(b.keys.keySet());
        List<String> cohortP = muct ? XBaseOther.muctCohort(all, t.seed) : feiCohortP;
        List<List<String>> cohort = new ArrayList<>();
        if (muct) for (String p : cohortP) cohort.add(b.keys.get(p));
        else for (String p : feiCohortP) cohort.add(new ArrayList<>(feiKeys.get(p).values()));
        List<String> rest = all.stream().filter(p -> !muct || !cohortP.contains(p)).toList();
        t.persons = rest;
        for (int i = 0; i < rest.size(); i++) t.pIdx.put(rest.get(i), i);
        List<List<String>> parts = XBaseOther.parts(rest, t.seed, muct ? XBaseOther.MUCT_PART_OFFSET : XBaseOther.ORL_PART_OFFSET);
        Map<String, List<String>> restKeys = new TreeMap<>();
        for (String p : rest) restKeys.put(p, b.keys.get(p));
        List<Map<String, List<String>>> half = XBaseOther.halves(restKeys, t.seed, muct ? XBaseOther.MUCT_HALF_OFFSET : XBaseOther.ORL_HALF_OFFSET);
        Path ref = dir.resolve(REF_DIR).resolve("xbase_" + code + "_scores.tsv.gz");
        Map<Integer, StringBuilder> refLines = bySConf(ref);
        Map<String, String> personOf = new HashMap<>();
        for (Map.Entry<String, Sample> e : b.samples.entrySet()) personOf.put(e.getKey(), e.getValue().person());
        List<String> fei10 = ProkaevEval.feiGuests(feiKeys, muct ? List.of() : feiCohortP, t.seed, personOf);
        if (muct) {
            // Когорта MUCT – в samples базы (ключи), её люди – гости.
            for (String p : cohortP) for (String k : b.keys.get(p)) personOf.put(k, p);
        }
        Map<String, List<String>> keysOf = new HashMap<>();
        for (String p : cohortP) if (muct) keysOf.put(p, b.keys.get(p));
        List<ProkaevEval.Job> jobs = new ArrayList<>();
        for (int jj = 0; jj < XBaseOther.PARTS; jj++) {
            Set<String> ctrlSet = new HashSet<>(parts.get(jj));
            List<String> own = rest.stream().filter(p -> !ctrlSet.contains(p)).toList();
            for (int k = 0; k < (muct ? 1 : 2); k++) {
                Map<String, List<String>> gal = new LinkedHashMap<>();
                Map<String, List<String>> pro = new LinkedHashMap<>();
                for (String p : own) {
                    gal.put(p, half.get(k).get(p));
                    pro.put(p, half.get(1 - k).get(p));
                }
                XBaseOther.Conf c = new XBaseOther.Conf(muct ? jj : 2 * jj + k, own, parts.get(jj), gal, pro);
                ProkaevEval.Job j = new ProkaevEval.Job();
                j.conf = title;
                j.pass = c.s();
                j.proto = muct ? 7 : 8;
                j.g = XBaseOther.gt(t, b, c, cohort);
                j.gal = gal;
                j.probes = XBaseEval.probes(j.g, pro);
                j.parts = XBaseOther.cvParts(j.g.own, t.seed, XBaseEval.CV_S + c.s());
                j.s = c.s();
                if (muct) ProkaevEval.pickGuests(j, cohortP, keysOf, t.seed);
                else j.g30 = new ArrayList<>(fei10);
                j.fei10 = fei10;
                String rl = refLines.get(c.s()).toString();
                j.baseL1 = method(rl, V.L1.id);
                j.pairBase = "L1 (xbase_72c3195)";
                j.extraBases.add(new Object[] {"L1-in (xbase_72c3195)", method(rl, V.L1IN.id), V.L1IN});
                if (!muct) j.extraBases.add(new Object[] {"A (xbase_72c3195)", method(rl, V.A.id), V.A});
                ProkaevEval.checkGuests(j, personOf);
                jobs.add(j);
            }
        }
        // Кадры: исходные снимки базы и YuNet экспорта; кадр «а» – экспорт; гости FEI – как в ProkaevEval.
        ProkaevEval.Vecs v = new ProkaevEval.Vecs();
        Map<String, double[]> rows = XBaseMpPts.exportRows(dir, code);
        Path raw = Paths.get(muct ? settings.loadFacesMuctDir() : settings.loadFacesDatasetDir());
        Set<String> keys = new TreeSet<>();
        for (String p : rest) keys.addAll(b.keys.get(p));
        for (ProkaevEval.Job j : jobs) keys.addAll(j.g30);
        keys.removeAll(fei10);
        int n = 0;
        List<Sample> list = new ArrayList<>();
        for (String k : keys) {
            Sample s = b.samples.get(k);
            String name = s.file().getFileName().toString();
            Path f = muct ? raw.resolve(name + ".jpg") : raw.resolve("s" + Integer.parseInt(s.person().substring(1))).resolve(name + ".pgm");
            Mat c = Imgcodecs.imread(f.toString(), Imgcodecs.IMREAD_COLOR);
            if (c.empty()) throw new IOException("prokaev: не прочитан " + f);
            double[] row = rows.get(s.person() + "/" + name);
            if (row == null) throw new IllegalStateException("prokaev: нет рамки YuNet " + s.file());
            ProkaevEval.addBig(v, k, c, row);
            c.release();
            list.add(s);
            if (++n % 500 == 0) System.out.printf(Locale.ROOT, "%s: кадры – %d / %d%n", mode, n, keys.size());
        }
        for (Map.Entry<String, Mat> e : XBaseEval.frames(export, list, null).entrySet()) ProkaevEval.addSmall(v, e.getKey(), e.getValue());
        ProkaevEval.feiFrames(export, dir, new HashSet<>(fei10), settings, v);
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "%s; seed %d; r0 %s; R %s; конфигураций %d; H %s", mode, t.seed, t.r0, t.rLimit, jobs.size(),
                ProkaevFrames.BOX_SHARE));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_" + mode), fp);
        ProkaevEval.Run run = new ProkaevEval.Run();
        for (ProkaevEval.Job j : jobs) {
            ProkaevEval.checkRoles(run, j, false);
            String name = String.format(Locale.ROOT, "conf_%02d", j.s);
            String text = cp.read(name);
            if (text == null) {
                text = ProkaevEval.compute(j, v);
                cp.save(name, text);
                System.out.printf(Locale.ROOT, "%s: конфигурация %d ready%n", mode, j.s);
            }
            ProkaevEval.absorb(j, text, run);
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 (ред. 4, п. 5): метод Прокаева на " + title + " по протоколу режима " + code + " (XBaseEval " + mode
                + "); справочно\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append("; снимки – ").append(raw).append("; L1, L1-in")
                .append(muct ? "" : ", A").append(" – ").append(ref).append(" (").append(FeiThreshold.sha(ref)).append(")\n");
        r.append(muct ? "Оговорка: MUCT – одна сессия съёмки (результаты внутрисессионные).\n" : "Оговорка: ORL – часть людей снята в разное время без разметки сессий.\n");
        ProkaevEval.header(r, t, v, run, jobs, muct ? "MUCT: гости – люди когорты MUCT (20 человек; 30 – не хватает людей, строка «гости 30» = 20), пул отбора по правилу автора – "
                + "людей нет (строка не считается)" : "ORL: людей вне ролей нет – гости 10 человек FEI (другая база; строки «гости 10/20/30» одинаковы)");
        StringBuilder sum = new StringBuilder(ProkaevEval.sumHead());
        ProkaevEval.tables(r, sum, run, rest.size(), t, List.of(title), title, "L1 (xbase_72c3195)");
        ProkaevEval.summaryText(r, sum.toString());
        XBaseOther.writeScores(out.resolve("xbase_" + code + "_prokaev_scores.tsv.gz"), "# " + title + ", Д-№1 метод Прокаева (ProkaevOther " + mode + "); блоки – база, "
                + "конфигурация, строка\n# Код: коммит " + FarMethods.commit() + "\n", run.lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; наибольшая занятая куча (выборки после каждой конфигурации) – %d МБ из %d МБ.%n", (System.nanoTime() - t0) / 1e9,
                run.peak >> 20, Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_" + code + "_prokaev.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    // ---------------------------------------------------------------- FEI

    /** Конфигурация FEI строки: свои (человек, оценка, верный лучший), пороговые (по возрастанию), контроль. */
    static final class FCfg {
        int[] ownP;
        double[] ownR;
        boolean[] ownOk;
        int[] thrP;
        double[] thrR;
        int[] ctrlP;
        double[] ctrlR;
    }

    static void fei(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out)
            throws IOException {
        long t0 = System.nanoTime();
        long seed = settings.loadFacesSeed();
        int bs = settings.loadFacesFeiBootstrap();
        Map<String, Set<Integer>> included = new TreeMap<>();
        feiKeys.forEach((p, m) -> included.put(p, new TreeSet<>(m.keySet())));
        List<FeiProtocol.Split> splits = FeiProtocol.splits(included, seed);
        List<String> persons = new ArrayList<>(new TreeSet<>(feiKeys.keySet()));
        Map<String, Integer> pIdx = new HashMap<>();
        for (int i = 0; i < persons.size(); i++) pIdx.put(persons.get(i), i);
        Map<String, String> personOf = new HashMap<>();
        Map<String, Integer> numOf = new HashMap<>();
        feiKeys.forEach((p, m) -> m.forEach((num, k) -> {
            personOf.put(k, p);
            numOf.put(k, num);
        }));
        // Гости другой съёмки – 10 человек GT (seed + 9975), по одному снимку.
        Map<String, Sample> gtS = new TreeMap<>();
        for (List<Sample> l : export.data().gt().values()) for (Sample s : l) gtS.put(s.file().toString(), s);
        List<String> gtPersons = new ArrayList<>(new TreeSet<>(export.data().gt().keySet()));
        java.util.Random rg = new java.util.Random(seed + ProkaevEval.FEI_SEED);
        java.util.Collections.shuffle(gtPersons, rg);
        List<String> gt10 = new ArrayList<>();
        for (String p : gtPersons.subList(0, 10)) {
            List<String> ks = export.data().gt().get(p).stream().map(s -> s.file().toString()).sorted().toList();
            String k = ks.get(rg.nextInt(ks.size()));
            gt10.add(k);
            personOf.put(k, "GT " + p);
        }
        List<ProkaevEval.Job> jobs = new ArrayList<>();
        Map<String, List<String>> cohortKeysOf = new HashMap<>();
        for (FeiProtocol.Split sp : splits) {
            for (String p : sp.cohort()) cohortKeysOf.put(p, new ArrayList<>(feiKeys.get(p).values()));
            for (int jn = 0; jn < FeiProtocol.NUMBERS.length; jn++) {
                ProkaevEval.Job j = new ProkaevEval.Job();
                j.conf = "FEI";
                j.pass = sp.index() * FeiProtocol.NUMBERS.length + jn;
                j.proto = 6;
                j.s = sp.index();
                j.cfgJ = jn;
                XBaseEval.Gt g = new XBaseEval.Gt();
                g.own = sp.gallery();
                g.ctrl = sp.control();
                g.imageNo = numOf;
                g.persons = persons;
                g.pIdx = pIdx;
                j.g = g;
                j.gal = new LinkedHashMap<>();
                List<Probe> pr = new ArrayList<>();
                int nj = FeiProtocol.NUMBERS[jn];
                for (int i = 0; i < sp.gallery().size(); i++) {
                    String p = sp.gallery().get(i);
                    List<String> gk = new ArrayList<>();
                    for (int num : FeiProtocol.trainNumbers(jn)) gk.add(feiKeys.get(p).get(num));
                    j.gal.put(p, gk);
                    pr.add(new Probe(FeiScores.OWN, p, nj, feiKeys.get(p).get(nj), i));
                }
                for (String p : sp.threshold()) for (Map.Entry<Integer, String> e : new TreeMap<>(feiKeys.get(p)).entrySet()) pr.add(new Probe(FeiScores.THR, p, e.getKey(), e.getValue(), -1));
                for (String p : sp.control()) for (Map.Entry<Integer, String> e : new TreeMap<>(feiKeys.get(p)).entrySet()) pr.add(new Probe(FeiScores.CTRL, p, e.getKey(), e.getValue(), -1));
                j.probes = pr;
                j.parts = null;
                ProkaevEval.pickGuests(j, sp.cohort(), cohortKeysOf, seed + sp.index());
                j.fei10 = gt10;
                ProkaevEval.checkGuests(j, personOf);
                // Пороговые – тоже вне гостей.
                for (String k : j.guests(ProkaevEval.Gs.G30)) if (sp.threshold().contains(personOf.get(k))) throw new IllegalStateException("prokaev: гость FEI среди пороговых");
                jobs.add(j);
            }
        }
        // Кадры FEI – исходные снимки и YuNet экспорта; гости GT – исходные gt_db.
        ProkaevEval.Vecs v = new ProkaevEval.Vecs();
        Set<String> keys = new TreeSet<>(personOf.keySet());
        keys.removeAll(gt10);
        ProkaevEval.feiFrames(export, dir, keys, settings, v);
        Path gtDir = Paths.get(settings.loadFacesGtDir());
        Map<String, double[]> gtRows = XBaseMpPts.exportRows(dir, "gt");
        List<Sample> gl = new ArrayList<>();
        for (String k : gt10) {
            Sample s = gtS.get(k);
            String pk = s.person() + "/" + s.file().getFileName().toString();
            Mat c = Imgcodecs.imread(gtDir.resolve(pk + ".jpg").toString(), Imgcodecs.IMREAD_COLOR);
            ProkaevEval.addBig(v, k, c, gtRows.get(pk));
            c.release();
            gl.add(s);
        }
        for (Map.Entry<String, Mat> e : XBaseEval.frames(export, gl, null).entrySet()) ProkaevEval.addSmall(v, e.getKey(), e.getValue());
        ProkaevEval.Run run = new ProkaevEval.Run();
        Map<String, List<FCfg>> fc = new LinkedHashMap<>();
        run.parse = o -> {
            String label = (String) o[1];
            FCfg c = new FCfg();
            List<double[]> own = new ArrayList<>();
            List<double[]> thr = new ArrayList<>();
            List<double[]> ctrl = new ArrayList<>();
            for (String l : ((String) o[2]).split("\n")) {
                if (l.isEmpty()) continue;
                FeiScores.Row r = FeiScores.parse(l);
                double x = FeiScores.ratio(r, null);
                int p = pIdx.get(r.person());
                if (r.own()) own.add(new double[] {p, x, r.bestId().equals(r.trueId()) ? 1 : 0});
                else (r.role().equals(FeiScores.THR) ? thr : ctrl).add(new double[] {p, x});
            }
            thr.sort((a, z) -> Double.compare(a[1], z[1]));
            c.ownP = own.stream().mapToInt(a -> (int) a[0]).toArray();
            c.ownR = own.stream().mapToDouble(a -> a[1]).toArray();
            c.ownOk = new boolean[own.size()];
            for (int i = 0; i < own.size(); i++) c.ownOk[i] = own.get(i)[2] == 1;
            c.thrP = thr.stream().mapToInt(a -> (int) a[0]).toArray();
            c.thrR = thr.stream().mapToDouble(a -> a[1]).toArray();
            c.ctrlP = ctrl.stream().mapToInt(a -> (int) a[0]).toArray();
            c.ctrlR = ctrl.stream().mapToDouble(a -> a[1]).toArray();
            fc.computeIfAbsent(label, x -> new ArrayList<>()).add(c);
        };
        String fp = FeiCheckpoint.fingerprint(dir, String.format(Locale.ROOT, "fei-prokaev; seed %d; конфигураций %d; H %s", seed, jobs.size(), ProkaevFrames.BOX_SHARE));
        XBaseOther.Checkpoint cp = new XBaseOther.Checkpoint(out.resolve("checkpoint_fei_prokaev"), fp);
        for (ProkaevEval.Job j : jobs) {
            String name = String.format(Locale.ROOT, "conf_%02d", j.pass);
            String text = cp.read(name);
            if (text == null) {
                text = ProkaevEval.compute(j, v);
                cp.save(name, text);
                System.out.printf(Locale.ROOT, "fei-prokaev: разбиение %d, конфигурация %d ready%n", j.s, j.cfgJ);
            }
            ProkaevEval.absorb(j, text, run);
        }
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№1 (ред. 4, п. 5): метод Прокаева на FEI по протоколу п. 3а (XBaseEval fei-prokaev); справочно, контроль "
                + "«внутри сессии не хуже» (Δ = 3 п. п.)\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Данные: экспорт ").append(dir).append(", коммит данных ").append(FarMethods.dataCommit(dir)).append(" (fei_selection.tsv, YuNet – detections.tsv); снимки FEI – ")
                .append(settings.loadFacesFeiDir()).append("; гости другой съёмки – GT\n");
        ProkaevEval.header(r, new XBaseEval.Gt(), v, run, jobs, "FEI: протокол п. 3а (5 разбиений × 8 конфигураций; свои 100, пороговые 40, контроль 40); гости – "
                + "люди когорты разбиения (20; методу Прокаева когорта не нужна; 30 – не хватает людей, строка «гости 30» = 20), справочно – 10 человек GT; П3 и "
                + "разделимость по частям «чужих» на FEI не считаются – порог по снимкам пороговых (как fei_scores)");
        int[][] w = bs > 0 ? FeiBootstrap.weights(persons.size(), bs, seed + FeiBootstrap.SEED_OFFSET) : new int[0][];
        int[] unit = FeiBootstrap.unit(persons.size());
        r.append("\n=== Ошибка своих при пороге по снимкам пороговых чужих (Нейман – Пирсон), FAR контроля по снимкам; [2,5; 97,5] – бутстреп по людям FEI ===\n");
        r.append("строка | FAR 1 %: ошибка своих | FAR контроля | FAR 5 %: ошибка своих | FAR контроля\n");
        for (Map.Entry<String, List<FCfg>> e : fc.entrySet()) {
            List<String> row = new ArrayList<>(List.of(e.getKey()));
            for (double far : new double[] {0.01, 0.05}) {
                double[] p = feiErr(e.getValue(), unit, far);
                double[] rep = new double[w.length];
                for (int q = 0; q < w.length; q++) rep[q] = feiErr(e.getValue(), w[q], far)[0];
                row.add(XBaseSepLtp.cell(p[0], rep));
                row.add(String.format(Locale.ROOT, "%.2f %%", p[1]));
            }
            r.append(String.join(" | ", row)).append('\n');
        }
        r.append("Справочно, L1 (5-lbp-mlda-ratio+pose, fei_630f21a/fei_scores.txt, порог по снимкам пороговых): FAR 1 % – ошибка своих 73/4000 = 1,8 %; "
                + "мера «не хуже» – разность с L1 ≤ +3 п. п. (без парного бутстрепа: строки L1 – из готового отчёта).\n");
        r.append("\n=== Точка по правилу автора (0,95) ===\n");
        r.append("строка | проб своих | отвергнуто своих (гость) | принят не тот | контрольных снимков | FAR контроля по снимкам | FAR по людям\n");
        for (Map.Entry<String, long[]> e : run.rule.entrySet()) {
            long[] t = e.getValue();
            int pa = run.ruleCtrlPersons.getOrDefault(e.getKey(), Set.of()).size();
            int pn = run.ruleCtrlAll.getOrDefault(e.getKey(), Set.of()).size();
            r.append(String.format(Locale.ROOT, "%s | %d | %.1f %% | %.1f %% | %d | %.2f %% | %d/%d%n", e.getKey(), t[0], 100.0 * t[1] / t[0], 100.0 * t[2] / t[0], t[3],
                    100.0 * t[4] / t[3], pa, pn));
        }
        XBaseOther.writeScores(out.resolve("xbase_fei_prokaev_scores.tsv.gz"), "# FEI, Д-№1 метод Прокаева (ProkaevOther fei-prokaev); s – разбиение, j – конфигурация; "
                + "роли own / thr / ctrl\n# Код: коммит " + FarMethods.commit() + "\n", run.lines.toString());
        r.append(String.format(Locale.ROOT, "%nВремя: %.0f с; наибольшая занятая куча – %d МБ из %d МБ.%n", (System.nanoTime() - t0) / 1e9, run.peak >> 20,
                Runtime.getRuntime().maxMemory() >> 20));
        FarMethods.write(out.resolve("xbase_fei_prokaev.txt"), r);
        System.out.print(r);
        cp.delete();
    }

    /** {ошибка своих %, FAR контроля %} при пороге по пороговым (веса людей w), сумма по конфигурациям. */
    static double[] feiErr(List<FCfg> cs, int[] w, double far) {
        double e = 0;
        double n = 0;
        double fx = 0;
        double fn = 0;
        for (FCfg c : cs) {
            double th = FeiBootstrap.threshold(c.thrR, c.thrP, w, far);
            for (int i = 0; i < c.ownR.length; i++) {
                n += w[c.ownP[i]];
                if (!(c.ownOk[i] && c.ownR[i] <= th)) e += w[c.ownP[i]];
            }
            for (int i = 0; i < c.ctrlR.length; i++) {
                fn += w[c.ctrlP[i]];
                if (c.ctrlR[i] <= th) fx += w[c.ctrlP[i]];
            }
        }
        return new double[] {100 * e / n, 100 * fx / fn};
    }
}
