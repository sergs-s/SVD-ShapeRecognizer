package svd.recognizer.faces;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.storage.SettingsStore;

/**
 * О-8, управляемая галерея (Жорик №2 от 06.11), GT межс.: подтверждается ли эффект смены очков и лечит ли его регистрация в обоих состояниях. Участники – свои, носившие очки, у кого на дате галереи
 * n_осн ≥ N и n_доп ≥ 1; N – наибольшее из 6, 5, 4, 3, 2 при не меньше 10 участниках. Варианты галереи (по N снимков у всех своих): (а) у участников только преобладающее состояние; (б) ⌊N/2⌋ другого
 * (сколько есть), остальные – преобладающего. Прочие люди – N снимков даты, одинаковых в обоих. Выбор – Random(faces.seed + 11000 + номер человека). W и модели частей – заново на вариант. Преобладающее при
 * равенстве – без очков (объявлено до прогона, TASKS.md).
 *
 * @author ssv
 */
final class XBaseGlassGal {

    private XBaseGlassGal() {
    }

    static final long SEED_OFFSET = 11000;
    static final int MIN_PARTICIPANTS = 10;
    static final int[] NS = {6, 5, 4, 3, 2};

    /** Наибольшее N, при котором участвуют не меньше minP; иначе −1. nMain, nOther – по людям, носившим очки. */
    static int chooseN(int[] nMain, int[] nOther, int minP) {
        for (int n : NS) {
            int c = 0;
            for (int i = 0; i < nMain.length; i++) if (nMain[i] >= n && nOther[i] >= 1) c++;
            if (c >= minP) return n;
        }
        return -1;
    }

    /** Выбор снимков участника: main и other уже перемешаны; both – вариант (б). Вариант (б) – начало main и начало other, поэтому снимки совпадают с (а), где возможно. */
    static List<String> pick(List<String> main, List<String> other, int n, boolean both) {
        int k = both ? Math.min(n / 2, other.size()) : 0;
        List<String> out = new ArrayList<>(main.subList(0, n - k));
        out.addAll(other.subList(0, k));
        Collections.sort(out);
        return out;
    }

    /** Перемешивание в постоянном порядке: сначала сортировка, затем общий Random. */
    static List<String> shuffled(List<String> keys, Random rnd) {
        List<String> l = new ArrayList<>(keys);
        Collections.sort(l);
        Collections.shuffle(l, rnd);
        return l;
    }

    static int personNo(String person) {
        return Integer.parseInt(person.replaceAll("\\D", ""));
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("glassgal: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup("gt", false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        XBaseEval.Gt tpl = u.tpl;
        Path gtFile = Paths.get(settings.loadFacesGtDir()).getParent().resolve(XBaseGlasses.GT_FILE);
        Map<String, String> labels = XBaseGlasses.readGt(gtFile);
        XBaseVec.Blk b0 = u.blocks.stream().filter(b -> b.group().equals("межс.")).findFirst().orElseThrow();
        Sel sel = select(tpl, labels);
        Set<String> part = sel.part();
        Map<String, Boolean> mainIsGlasses = sel.mainIsGlasses();
        int n = sel.n();
        int minDate = sel.minDate();
        StringBuilder t = new StringBuilder("О-8: управляемая галерея – GT межс. (XBaseEval gt-glassgal; Жорик №2 от 06.11)\nКод: коммит " + FarMethods.commit() + "; разметка – " + gtFile + ", SHA-256 " + FrameB.sha256(gtFile)
                + String.format(Locale.ROOT, "; бутстреп по людям B = %d, seed %d; выбор снимков – Random(faces.seed + %d + номер человека)%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET, SEED_OFFSET));
        t.append(sel.tab());
        Path root = r.dir().resolveSibling("glassgal");
        Files.createDirectories(root);
        Path rf = root.resolve("xbase_gt_glassgal.txt");
        if (n < 0 || n > minDate) {
            t.append(String.format(Locale.ROOT, "%nОСТАНОВКА до прогона: N = %d, наименьшее число снимков на дате галереи у своих %d; участников меньше %d при N = 2 или N больше доступного. Нужно решение.%n", n, minDate, MIN_PARTICIPANTS));
            FarMethods.write(rf, t);
            System.out.print(t);
            System.out.println("DONE");
            return;
        }
        t.append(String.format(Locale.ROOT, "%nВыбрано N = %d; участников %d из %d; не участвуют: %s%n", n, part.size(), part.size() + sel.notPart().size(), sel.notPart()));
        Map<String, List<String>> galA = sel.galA();
        Map<String, List<String>> galB = sel.galB();
        t.append(String.format(Locale.ROOT, "Снимков другого состояния в галереях (б): %d (на участника ⌊N/2⌋ = %d, сколько есть)%n", sel.otherInB(), n / 2));
        List<Probe> probes = b0.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList();
        XBaseVec.Blk bA = new XBaseVec.Blk(b0.name(), b0.abl2(), b0.g(), galA, b0.pro(), b0.parts(), probes, b0.extra(), b0.s(), b0.group());
        XBaseVec.Blk bB = new XBaseVec.Blk(b0.name(), b0.abl2(), b0.g(), galB, b0.pro(), b0.parts(), probes, b0.extra(), b0.s(), b0.group());
        Set<String> need = new HashSet<>();
        for (XBaseVec.Blk b : List.of(bA, bB)) {
            b.gal().values().forEach(need::addAll);
            for (Probe q : b.probes()) need.add(q.key());
            if (b.extra() != null) for (List<Probe> l : b.extra()) for (Probe q : l) need.add(q.key());
        }
        b0.g().cohort.forEach(need::addAll);
        u.cin.forEach(need::addAll);
        u.cout.forEach(need::addAll);
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        if (!fr.keySet().containsAll(need)) throw new IllegalStateException("glassgal: не все кадры загружены");
        String code = FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBaseGlassGal.java");
        XBaseFinal.Run rA = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("a"), false, 0);
        XBaseFinal.Run rB = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("b"), false, 0);
        Files.createDirectories(root.resolve("a").resolve("vec"));
        Files.createDirectories(root.resolve("b").resolve("vec"));
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            XBaseEval x = null;
            for (Object[] pair : new Object[][] {{rA, bA, "а"}, {rB, bB, "б"}}) {
                XBaseFinal.Run rr = (XBaseFinal.Run) pair[0];
                XBaseVec.Blk b = (XBaseVec.Blk) pair[1];
                Path vf = XBaseFinal.vfile(rr, b, c);
                if (Files.exists(vf)) {
                    if (!XBaseVec.readHead(vf).contains(code)) throw new IllegalStateException("glassgal: другой код: " + vf);
                    continue;
                }
                if (x == null) x = XBasePart4.vectorsB(fr, r.k2(), XBaseFinal.param(c, tpl.params), c.f(), tpl.threads);
                List<XBaseVec.Model> ms = XBaseVec.run(List.of(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout)), Math.min(8, tpl.threads)).get(0);
                java.io.Writer w = XBaseVec.writer(vf, "# Управляемая галерея (XBaseGlassGal), вариант " + pair[2] + ", " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; " + code + "\n");
                XBaseVec.write(w, b.name() + " " + pair[2] + " " + c.name(), ms);
                XBaseVec.finish(w, vf);
                System.out.printf(Locale.ROOT, "glassgal: %s вариант %s готово (%.0f с)%n", c.id(), pair[2], (System.nanoTime() - t0) / 1e9);
            }
            if (x != null) {
                x.vec.clear();
                x.mir.clear();
            }
        }
        // Группы проб участников: 0 – преобладающее состояние, 1 – другое, −1 – не участник
        List<Probe> own = XBaseEval.probes(b0.g(), b0.pro()).stream().filter(q -> q.role().equals(FeiScores.OWN)).toList();
        int[] groups = groups(own, part, labels, mainIsGlasses);
        int[] counts = new int[2];
        Set<String> pMain = new HashSet<>();
        Set<String> pOther = new HashSet<>();
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] < 0) continue;
            counts[groups[i]]++;
            (groups[i] == XBaseGlasses.MATCH ? pMain : pOther).add(own.get(i).person());
        }
        t.append(String.format(Locale.ROOT, "%nПробы участников: преобладающее – %d проб, %d людей; другое – %d проб, %d людей%n", counts[0], pMain.size(), counts[1], pOther.size()));
        int[] zeroDays = new int[groups.length];
        XBaseGlasses.Line la = new XBaseGlasses.Line("а", List.of(XBaseGoalPrep.cfg(bA, bA.g(), XBaseGoalPrep.fused(rA, bA, XBaseFinal.TT1, bA.g().rOf, bA.g().r0))), List.of(groups), List.of(zeroDays));
        XBaseGlasses.Line lb = new XBaseGlasses.Line("б", List.of(XBaseGoalPrep.cfg(bB, bB.g(), XBaseGoalPrep.fused(rB, bB, XBaseFinal.TT1, bB.g().rOf, bB.g().r0))), List.of(groups), List.of(zeroDays));
        int[][] w = FeiBootstrap.weights(u.persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        int[] unit = FeiBootstrap.unit(u.persons);
        String[] gName = {"преобладающее", "другое"};
        t.append("\n=== Участники, слияние: порог – по контрольным чужим своего варианта; запись (а) | (б) | (а) − (б) [95 %] ===\nгруппа | FAR (факт. а / б, %) | ошибка своих (а) | (б) | разность\n");
        for (int g : new int[] {XBaseGlasses.CHANGE, XBaseGlasses.MATCH}) {
            for (double far : new double[] {0.01, 0.05}) {
                double a = XBaseGlasses.ownErr(la, unit, far, g)[0];
                double c = XBaseGlasses.ownErr(lb, unit, far, g)[0];
                double[] rep = new double[w.length];
                for (int q = 0; q < w.length; q++) rep[q] = XBaseGlasses.ownErr(la, w[q], far, g)[0] - XBaseGlasses.ownErr(lb, w[q], far, g)[0];
                t.append(String.format(Locale.ROOT, "%s | %.2f / %.2f | %.1f %% | %.1f %% | %s%n", gName[g], XBaseOther.errAtFar(la.cfgs(), unit, far)[1], XBaseOther.errAtFar(lb.cfgs(), unit, far)[1], a, c, XBaseGlasses.interval(a - c, rep)));
            }
        }
        t.append("\nВерный лучший кандидат: группа | (а) | (б) | (а) − (б) [95 %]\n");
        for (int g : new int[] {XBaseGlasses.CHANGE, XBaseGlasses.MATCH}) {
            double a = XBaseGlasses.ownErr(la, unit, 0.01, g)[1];
            double c = XBaseGlasses.ownErr(lb, unit, 0.01, g)[1];
            double[] rep = new double[w.length];
            for (int q = 0; q < w.length; q++) rep[q] = XBaseGlasses.ownErr(la, w[q], 0.01, g)[1] - XBaseGlasses.ownErr(lb, w[q], 0.01, g)[1];
            t.append(String.format(Locale.ROOT, "%s | %.1f %% | %.1f %% | %s%n", gName[g], a, c, XBaseGlasses.interval(a - c, rep)));
        }
        XBaseFinalReport.M mA = XBaseFinalReport.measure(la.cfgs(), u.persons, w);
        XBaseFinalReport.M mB = XBaseFinalReport.measure(lb.cfgs(), u.persons, w);
        XBaseFinalReport.M m32 = XBaseFinalReport.measure(List.of(XBaseGoalPrep.cfg(b0, b0.g(), XBaseGoalPrep.fused(r, b0, XBaseFinal.TT1, b0.g().rOf, b0.g().r0))), u.persons, w);
        t.append("\n=== Вся строка GT межс., справочно ===\nстрока | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.level("(32), случайные 6 снимков", m32)).append(XBaseEsrc.level("(а) одно состояние", mA)).append(XBaseEsrc.level("(б) оба состояния", mB));
        t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.row("(а) − (б)", mA, mB));
        t.append(String.format(Locale.ROOT, "%nВремя: всего %.0f с%n", (System.nanoTime() - t0) / 1e9));
        for (Mat q : u.fr.values()) q.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        FarMethods.write(rf, t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Группы проб участников: 0 – преобладающее состояние, 1 – другое, −1 – не участник. */
    static int[] groups(List<Probe> own, Set<String> part, Map<String, String> labels, Map<String, Boolean> mainIsGlasses) {
        int[] groups = new int[own.size()];
        for (int i = 0; i < groups.length; i++) {
            Probe q = own.get(i);
            if (!part.contains(q.person())) {
                groups[i] = -1;
                continue;
            }
            String s = labels.get(q.key());
            if (s == null) throw new IllegalStateException("glassgal: нет пробы в разметке " + q.key());
            boolean glasses = s.equals("1");
            groups[i] = glasses == mainIsGlasses.get(q.person()) ? XBaseGlasses.MATCH : XBaseGlasses.CHANGE;
        }
        return groups;
    }

    /** Выбор опыта: таблица, N, участники, галереи (а) и (б); при N вне допустимого галереи пусты. */
    record Sel(String tab, int n, int minDate, Set<String> part, List<String> notPart, Map<String, Boolean> mainIsGlasses, Map<String, List<String>> galA, Map<String, List<String>> galB, int otherInB) {}

    static Sel select(XBaseEval.Gt tpl, Map<String, String> labels) {
        Map<String, List<String>> all = tpl.galleryAll;
        // Люди, носившие очки: любой снимок человека в разметке = «1»
        Set<String> wearers = new HashSet<>();
        for (Map.Entry<String, String> e : labels.entrySet()) {
            if (e.getValue().equals("1")) wearers.add(Paths.get(e.getKey()).getName(1).toString());
        }
        Map<String, List<String>> ones = new HashMap<>();
        Map<String, List<String>> zeros = new HashMap<>();
        List<String> own14 = new ArrayList<>();
        int minDate = Integer.MAX_VALUE;
        for (String p : tpl.own) {
            List<String> y = new ArrayList<>();
            List<String> n = new ArrayList<>();
            for (String k : all.get(p)) {
                String s = labels.get(k);
                if (s == null) throw new IllegalStateException("glassgal: нет снимка в разметке " + k);
                if (s.equals("1")) y.add(k);
                else if (s.equals("0")) n.add(k);
                else throw new IllegalStateException("glassgal: «?» на дате галереи " + k);
            }
            ones.put(p, y);
            zeros.put(p, n);
            minDate = Math.min(minDate, all.get(p).size());
            if (wearers.contains(p)) own14.add(p);
        }
        int[] nm = new int[own14.size()];
        int[] no = new int[own14.size()];
        StringBuilder tab = new StringBuilder("\n=== Свои, носившие очки: на дате галереи ===\nчеловек | в очках | без очков | преобладающее | n_осн | n_доп\n");
        Map<String, Boolean> mainIsGlasses = new HashMap<>();
        for (int i = 0; i < nm.length; i++) {
            String p = own14.get(i);
            int a = ones.get(p).size();
            int c = zeros.get(p).size();
            boolean mg = a > c; // при равенстве – без очков
            mainIsGlasses.put(p, mg);
            nm[i] = mg ? a : c;
            no[i] = mg ? c : a;
            tab.append(String.format(Locale.ROOT, "%s | %d | %d | %s | %d | %d%n", p, a, c, mg ? "в очках" : "без очков", nm[i], no[i]));
        }
        int n = chooseN(nm, no, MIN_PARTICIPANTS);
        if (n < 0 || n > minDate) return new Sel(tab.toString(), n, minDate, Set.of(), List.of(), mainIsGlasses, Map.of(), Map.of(), 0);
        Set<String> part = new HashSet<>();
        List<String> notPart = new ArrayList<>();
        for (int i = 0; i < nm.length; i++) {
            if (nm[i] >= n && no[i] >= 1) part.add(own14.get(i));
            else notPart.add(own14.get(i));
        }
        Map<String, List<String>> galA = new LinkedHashMap<>();
        Map<String, List<String>> galB = new LinkedHashMap<>();
        int otherInB = 0;
        for (String p : tpl.own) {
            Random rnd = new Random(tpl.seed + SEED_OFFSET + personNo(p));
            if (part.contains(p)) {
                boolean mg = mainIsGlasses.get(p);
                List<String> main = shuffled(mg ? ones.get(p) : zeros.get(p), rnd);
                List<String> other = shuffled(mg ? zeros.get(p) : ones.get(p), rnd);
                galA.put(p, pick(main, other, n, false));
                galB.put(p, pick(main, other, n, true));
                otherInB += Math.min(n / 2, other.size());
            } else {
                List<String> s = shuffled(all.get(p), rnd);
                List<String> g = new ArrayList<>(s.subList(0, n));
                Collections.sort(g);
                galA.put(p, g);
                galB.put(p, g);
            }
        }
        return new Sel(tab.toString(), n, minDate, part, notPart, mainIsGlasses, galA, galB, otherInB);
    }
}
