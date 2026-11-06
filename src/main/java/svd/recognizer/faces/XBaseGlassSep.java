package svd.recognizer.faces;

import java.io.IOException;
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
import org.opencv.core.Mat;
import svd.recognizer.faces.XBaseEval.Probe;
import svd.recognizer.faces.XBaseEval.V;
import svd.recognizer.storage.SettingsStore;

/**
 * О-8, раздельные шаблоны (Жорик №3 от 06.11), GT межс.: вариант (в) рядом с (а) и (б) управляемой галереи. Снимки галереи (в) – те же, что (б), поэтому W и модели частей те же: W учится по снимкам классов и от
 * числа центров не зависит. У участника два центра – по снимкам преобладающего состояния и по снимкам другого; расстояние до человека – наименьшее 1 − cos по его центрам, отдельно у каждого кандидата; у прочих –
 * один центр. Приведения (Т2) – тем же правилом по новым расстояниям. Моделей (б) на диске нет, поэтому те же модели обучаются заново; расстояния (б) сверяются с векторами gt-glassgal побайтно, иначе остановка.
 * Риск хаба: доля контрольных проб с лучшим кандидатом – участником, FAR при рабочем пороге (34).
 *
 * @author ssv
 */
final class XBaseGlassSep {

    private XBaseGlassSep() {
    }

    /** Наименьшее 1 − cos пробы по центрам каждого человека галереи. */
    static double[] minDistances(double[] y, double[][][] centers) {
        double[] d = new double[centers.length];
        for (int g = 0; g < d.length; g++) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] c : centers[g]) best = Math.min(best, XBaseVec.cos(y, c));
            d[g] = best;
        }
        return d;
    }

    /** Снимки галереи участника по состояниям: [преобладающее, другое]; пустая часть – ошибка. */
    static List<List<String>> splitState(String person, List<String> keys, Map<String, String> labels, boolean mainIsGlasses) {
        List<String> main = new ArrayList<>();
        List<String> other = new ArrayList<>();
        for (String k : keys) {
            String s = labels.get(k);
            if (!"0".equals(s) && !"1".equals(s)) throw new IllegalStateException("glasssep: нет разметки снимка галереи " + k);
            (s.equals("1") == mainIsGlasses ? main : other).add(k);
        }
        if (main.isEmpty() || other.isEmpty()) throw new IllegalStateException("glasssep: у участника " + person + " одно состояние в галерее");
        return List.of(main, other);
    }

    /** Модель (б) и (в) на одной W: те же строки проб и порядок, что XBaseVec.model; когорты приведения не нужны (слияние их не использует). */
    private static XBaseVec.Model[] model(XBaseEval x, boolean mirror, String method, int s, List<String> own, List<List<String>> gallery, List<List<String>> cohort, List<Probe> probes, String task,
                                          Map<String, List<List<String>>> sep) {
        XBaseRule.Fit f = XBaseRule.fit(x, mirror, gallery, cohort, null);
        double[][] c = f.lm.centers();
        double[][][] cc = new double[c.length][][];
        for (int g = 0; g < c.length; g++) {
            List<List<String>> two = sep.get(own.get(g));
            if (two == null) {
                cc[g] = new double[][] {c[g]};
                continue;
            }
            // Центр по всем снимкам тем же способом обязан совпасть с центром модели: иначе центры (в) посчитаны не в той W.
            double[] all = XBaseVec.center(f.lm, x, gallery.get(g), mirror);
            for (int i = 0; i < all.length; i++) if (Math.abs(all[i] - c[g][i]) > 1e-8) throw new IllegalStateException("glasssep: центр ≠ центру модели, " + own.get(g));
            cc[g] = new double[][] {XBaseVec.center(f.lm, x, two.get(0), mirror), XBaseVec.center(f.lm, x, two.get(1), mirror)};
        }
        List<XBaseVec.Row> rb = new ArrayList<>();
        List<XBaseVec.Row> rc = new ArrayList<>();
        double[] none = new double[0];
        for (Probe p : probes) {
            double[] y = f.lm.embed(x.vec.get(V.L1).get(p.key()));
            double[] db = new double[c.length];
            for (int g = 0; g < c.length; g++) db[g] = XBaseVec.cos(y, c[g]);
            rb.add(new XBaseVec.Row(p.role(), p.person(), p.image(), p.key(), p.trueIdx(), db, none, none));
            rc.add(new XBaseVec.Row(p.role(), p.person(), p.image(), p.key(), p.trueIdx(), minDistances(y, cc), none, none));
        }
        double[][] z = new double[c.length][];
        return new XBaseVec.Model[] {new XBaseVec.Model(task, method, s, own, rb, z), new XBaseVec.Model(task, method, s, own, rc, z)};
    }

    /** Задачи блока – как XBaseVec.job (тот же порядок моделей и проб). */
    private static List<Callable<XBaseVec.Model[]>> job(XBaseVec.Blk b, XBaseEval x, Map<String, List<List<String>>> sep) {
        XBaseEval.Gt gt = b.g();
        List<String> own = gt.own;
        List<List<String>> cohort = new ArrayList<>(gt.cohort);
        List<List<String>> full = own.stream().map(b.gal()::get).toList();
        List<Callable<XBaseVec.Model[]>> tasks = new ArrayList<>();
        for (boolean m : new boolean[] {false, true}) {
            String method = V.L1.id + (m ? FeiMethods.MIRROR : "");
            tasks.add(() -> model(x, m, method, b.s(), own, full, cohort, b.probes(), "main", sep));
            for (int q = 0; q < FeiMethods.CV_PARTS; q++) {
                Set<String> part = new HashSet<>(b.parts().get(q));
                List<String> g = own.stream().filter(p -> !part.contains(p)).toList();
                List<List<String>> gk = g.stream().map(b.gal()::get).toList();
                List<Probe> pr = new ArrayList<>();
                for (String p : own) if (part.contains(p)) for (String k : b.gal().get(p)) pr.add(new Probe(FeiMethods.CV_ROLE + q, p, gt.img(k), k, -1));
                if (b.extra() != null) pr.addAll(b.extra().get(q));
                int qq = q;
                tasks.add(() -> model(x, m, method, b.s(), g, gk, cohort, pr, "part " + qq, sep));
            }
        }
        return tasks;
    }

    /** Контроль: расстояния (б), посчитанные здесь, побайтно равны векторам gt-glassgal (те же галереи, W, пробы). */
    static void checkSame(List<XBaseVec.Model> mine, List<XBaseVec.Model> file, String what) {
        if (mine.size() != file.size()) throw new IllegalStateException("glasssep: разное число моделей, " + what);
        for (int i = 0; i < mine.size(); i++) {
            XBaseVec.Model a = mine.get(i);
            XBaseVec.Model c = file.get(i);
            if (!a.task().equals(c.task()) || !a.method().equals(c.method()) || !a.gallery().equals(c.gallery()) || a.rows().size() != c.rows().size())
                throw new IllegalStateException("glasssep: модели разошлись, " + what + ", " + a.task() + " " + a.method());
            for (int k = 0; k < a.rows().size(); k++) {
                XBaseVec.Row p = a.rows().get(k);
                XBaseVec.Row q = c.rows().get(k);
                if (!p.key().equals(q.key()) || !p.role().equals(q.role()) || !java.util.Arrays.equals(p.dg(), q.dg()))
                    throw new IllegalStateException("glasssep: (б) не совпала с векторами gt-glassgal, " + what + ", " + a.task() + " " + a.method() + ", проба " + p.key());
            }
        }
    }

    /** Исход контрольной пробы: лучший кандидат и s слияния. */
    record Ctrl(String person, String key, String best, double s) {}

    /** Контрольные пробы основной модели: те же ℓ, выбор модели и s, что XBaseFinalReport.atT. */
    static List<Ctrl> ctrl(List<XBaseVec.Model> tt, List<XBaseVec.Model> xs, Map<String, Double> rOf, double r0) {
        double[] st = XBaseVec.fuseStats(tt, false);
        double[] sx = XBaseVec.fuseStats(xs, false);
        XBaseVec.Model tX = XBaseVec.find(tt, "main", false);
        XBaseVec.Model tM = XBaseVec.find(tt, "main", true);
        XBaseVec.Model xM = XBaseVec.find(xs, "main", true);
        List<Ctrl> out = new ArrayList<>();
        for (int i = 0; i < tX.rows().size(); i++) {
            XBaseVec.Row rx = xM.rows().get(i);
            XBaseVec.Row rt = rOf.get(tX.rows().get(i).key()) <= r0 ? tX.rows().get(i) : tM.rows().get(i);
            if (!rx.key().equals(rt.key())) throw new IllegalStateException("glasssep: пробы кандидатов разошлись");
            if (!rt.role().equals(FeiScores.CTRL)) continue;
            int b = -1;
            int s2 = -1;
            double[] ln = new double[rt.dg().length];
            for (int g = 0; g < ln.length; g++) {
                ln[g] = XBaseFeat2.W_FUSE * (XBaseVec.ell(rt.dg()[g]) - st[0]) / st[1] + (1 - XBaseFeat2.W_FUSE) * (XBaseVec.ell(rx.dg()[g]) - sx[0]) / sx[1];
                if (b < 0 || ln[g] < ln[b]) {
                    s2 = b;
                    b = g;
                } else if (s2 < 0 || ln[g] < ln[s2]) {
                    s2 = g;
                }
            }
            out.add(new Ctrl(rt.person(), rt.key(), tX.gallery().get(b), 2 * ln[b] - ln[s2]));
        }
        return out;
    }

    private static XBaseGlasses.Line line(String name, XBaseVec.Blk b, XBaseFinal.Run r, int[] groups) throws IOException {
        return new XBaseGlasses.Line(name, List.of(XBaseGoalPrep.cfg(b, b.g(), XBaseGoalPrep.fused(r, b, XBaseFinal.TT1, b.g().rOf, b.g().r0))), List.of(groups), List.of(new int[groups.length]));
    }

    /** Строка парной разности ошибки своих: (в) − сравниваемый. */
    private static String diff(XBaseGlasses.Line c, XBaseGlasses.Line o, int[] unit, int[][] w, double far, int grp, int idx) {
        double a = XBaseGlasses.ownErr(c, unit, far, grp)[idx];
        double z = XBaseGlasses.ownErr(o, unit, far, grp)[idx];
        double[] rep = new double[w.length];
        for (int q = 0; q < w.length; q++) rep[q] = XBaseGlasses.ownErr(c, w[q], far, grp)[idx] - XBaseGlasses.ownErr(o, w[q], far, grp)[idx];
        return XBaseGlasses.interval(a - z, rep);
    }

    private static XBaseGlasses.Line regroup(XBaseGlasses.Line l, int[] g) {
        return new XBaseGlasses.Line(l.name(), l.cfgs(), List.of(g), l.days());
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("glasssep: FACES_FRAMES=b");
        String tf = System.getenv("FACES_P4_TABLE");
        if (tf == null) throw new IllegalStateException("glasssep: задайте FACES_P4_TABLE – таблица (Н) итогового прогона");
        Map<Integer, XBasePart4.TRow> table = XBasePart4.readTable(Paths.get(tf));
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup("gt", false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        XBaseEval.Gt tpl = u.tpl;
        Path gtFile = Paths.get(settings.loadFacesGtDir()).getParent().resolve(XBaseGlasses.GT_FILE);
        Map<String, String> labels = XBaseGlasses.readGt(gtFile);
        XBaseVec.Blk b0 = u.blocks.stream().filter(b -> b.group().equals("межс.")).findFirst().orElseThrow();
        XBaseGlassGal.Sel sel = XBaseGlassGal.select(tpl, labels);
        if (sel.part().size() != 11 || sel.n() != 5) throw new IllegalStateException("glasssep: ожидалось 11 участников при N = 5, а не " + sel.part().size() + " при " + sel.n());
        Path root = r.dir().resolveSibling("glassgal");
        Path rf = root.resolve("xbase_gt_glasssep.txt");
        List<Probe> probes = b0.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList();
        XBaseVec.Blk bA = new XBaseVec.Blk(b0.name(), b0.abl2(), b0.g(), sel.galA(), b0.pro(), b0.parts(), probes, b0.extra(), b0.s(), b0.group());
        XBaseVec.Blk bB = new XBaseVec.Blk(b0.name(), b0.abl2(), b0.g(), sel.galB(), b0.pro(), b0.parts(), probes, b0.extra(), b0.s(), b0.group());
        StringBuilder t = new StringBuilder("О-8: раздельные шаблоны – GT межс. (XBaseEval gt-glasssep; Жорик №3 от 06.11)\nКод: коммит " + FarMethods.commit() + "; разметка – " + gtFile + ", SHA-256 " + FrameB.sha256(gtFile)
                + "\nТаблица (Н) – " + tf + ", SHA-256 " + FrameB.sha256(Paths.get(tf)) + String.format(Locale.ROOT, "; бутстреп по людям B = %d, seed %d%n", tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        Map<String, List<List<String>>> sep = new LinkedHashMap<>();
        t.append("\n=== Центры участников в (в): снимков преобладающего | другого ===\n");
        for (String p : tpl.own) {
            if (!sel.part().contains(p)) continue;
            List<List<String>> two = splitState(p, sel.galB().get(p), labels, sel.mainIsGlasses().get(p));
            sep.put(p, two);
            t.append(String.format(Locale.ROOT, "%s | %d | %d%n", p, two.get(0).size(), two.get(1).size()));
        }
        Set<String> need = new HashSet<>();
        bB.gal().values().forEach(need::addAll);
        for (Probe q : bB.probes()) need.add(q.key());
        if (bB.extra() != null) for (List<Probe> l : bB.extra()) for (Probe q : l) need.add(q.key());
        b0.g().cohort.forEach(need::addAll);
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        if (!fr.keySet().containsAll(need)) throw new IllegalStateException("glasssep: не все кадры загружены");
        XBaseFinal.Run rA = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("a"), false, 0);
        XBaseFinal.Run rB = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("b"), false, 0);
        XBaseFinal.Run rC = new XBaseFinal.Run(u, r.k2(), Map.of(), root.resolve("c"), false, 0);
        Files.createDirectories(root.resolve("c").resolve("vec"));
        String code = FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBaseGlassSep.java");
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            Path vf = XBaseFinal.vfile(rC, bB, c);
            if (Files.exists(vf)) {
                if (!XBaseVec.readHead(vf).contains(code)) throw new IllegalStateException("glasssep: другой код: " + vf);
                continue;
            }
            XBaseEval x = XBasePart4.vectorsB(fr, r.k2(), XBaseFinal.param(c, tpl.params), c.f(), tpl.threads);
            List<Callable<XBaseVec.Model[]>> tasks = job(bB, x, sep);
            java.util.concurrent.ExecutorService ex = java.util.concurrent.Executors.newFixedThreadPool(Math.min(8, tpl.threads));
            List<XBaseVec.Model> mb = new ArrayList<>();
            List<XBaseVec.Model> mc = new ArrayList<>();
            try {
                for (java.util.concurrent.Future<XBaseVec.Model[]> ff : ex.invokeAll(tasks)) {
                    XBaseVec.Model[] two = ff.get();
                    mb.add(two[0]);
                    mc.add(two[1]);
                }
            } finally {
                ex.shutdown();
            }
            checkSame(mb, XBaseFinal.models(rB, bB, c, false), c.name());
            java.io.Writer w = XBaseVec.writer(vf, "# Раздельные шаблоны (XBaseGlassSep), вариант в, " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; " + code + "\n");
            XBaseVec.write(w, bB.name() + " в " + c.name(), mc);
            XBaseVec.finish(w, vf);
            x.vec.clear();
            x.mir.clear();
            System.out.printf(Locale.ROOT, "glasssep: %s готово, (б) совпала с векторами gt-glassgal (%.0f с)%n", c.id(), (System.nanoTime() - t0) / 1e9);
        }
        t.append("\nКонтроль: расстояния (б), посчитанные на тех же W, побайтно равны векторам gt-glassgal (glassgal/b/vec; при продолжении после обрыва – для кандидатов, посчитанных в этом запуске и в прерванном).\n");

        List<Probe> own = XBaseEval.probes(b0.g(), b0.pro()).stream().filter(q -> q.role().equals(FeiScores.OWN)).toList();
        int[] groups = XBaseGlassGal.groups(own, sel.part(), labels, sel.mainIsGlasses());
        int[] partAll = new int[groups.length];
        int[] nonPart = new int[groups.length];
        for (int i = 0; i < groups.length; i++) {
            partAll[i] = groups[i] < 0 ? -1 : 0;
            nonPart[i] = groups[i] < 0 ? 0 : -1;
        }
        XBaseGlasses.Line la = line("а", bA, rA, groups);
        XBaseGlasses.Line lb = line("б", bB, rB, groups);
        XBaseGlasses.Line lc = line("в", bB, rC, groups);
        int[][] w = FeiBootstrap.weights(u.persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        int[] unit = FeiBootstrap.unit(u.persons);
        Object[][] sets = {{"другое", groups, XBaseGlasses.CHANGE}, {"преобладающее", groups, XBaseGlasses.MATCH}, {"обе группы", partAll, 0}, {"неучастники, справочно", nonPart, 0}};
        t.append("\n=== Слияние: порог – по контрольным чужим своего варианта; ошибка своих, разности [95 %] ===\nгруппа | проб | FAR (факт. а / б / в, %) | (а) | (б) | (в) | (в) − (б) | (в) − (а)\n");
        for (Object[] s : sets) {
            int[] g = (int[]) s[1];
            int gi = (Integer) s[2];
            XBaseGlasses.Line a = regroup(la, g);
            XBaseGlasses.Line b = regroup(lb, g);
            XBaseGlasses.Line c = regroup(lc, g);
            long cnt = java.util.Arrays.stream(g).filter(v -> v == gi).count();
            for (double far : new double[] {0.01, 0.05}) {
                t.append(String.format(Locale.ROOT, "%s | %d | %.2f / %.2f / %.2f | %.1f %% | %.1f %% | %.1f %% | %s | %s%n", s[0], cnt, XBaseOther.errAtFar(la.cfgs(), unit, far)[1], XBaseOther.errAtFar(lb.cfgs(), unit, far)[1],
                        XBaseOther.errAtFar(lc.cfgs(), unit, far)[1], XBaseGlasses.ownErr(a, unit, far, gi)[0], XBaseGlasses.ownErr(b, unit, far, gi)[0], XBaseGlasses.ownErr(c, unit, far, gi)[0],
                        diff(c, b, unit, w, far, gi, 0), diff(c, a, unit, w, far, gi, 0)));
            }
        }
        t.append("\nВерный лучший кандидат: группа | (а) | (б) | (в) | (в) − (б) | (в) − (а)\n");
        for (Object[] s : sets) {
            int[] g = (int[]) s[1];
            int gi = (Integer) s[2];
            XBaseGlasses.Line a = regroup(la, g);
            XBaseGlasses.Line b = regroup(lb, g);
            XBaseGlasses.Line c = regroup(lc, g);
            t.append(String.format(Locale.ROOT, "%s | %.1f %% | %.1f %% | %.1f %% | %s | %s%n", s[0], XBaseGlasses.ownErr(a, unit, 0.01, gi)[1], XBaseGlasses.ownErr(b, unit, 0.01, gi)[1], XBaseGlasses.ownErr(c, unit, 0.01, gi)[1],
                    diff(c, b, unit, w, 0.01, gi, 1), diff(c, a, unit, w, 0.01, gi, 1)));
        }

        // Риск хаба и рабочий порог (34)
        int nGal = b0.g().own.size();
        double[] tu = XBaseFinalReport.tOf(table, nGal);
        double share = 100.0 * sel.part().size() / nGal;
        t.append(String.format(Locale.ROOT, "%n=== Риск хаба: лучший кандидат контрольной пробы – участник; доля участников в галерее %d из %d = %.1f %% ===%nвариант | контрольных проб | лучший – участник | доля, %%%n", sel.part().size(), nGal, share));
        StringBuilder acc = new StringBuilder(String.format(Locale.ROOT, "%n=== Рабочий порог (34): N = %d, t = %.2f (узел %d), приведения по (Т2) своей галереи; s ≤ t – приём ===%nвариант | принято чужих из контрольных | доля, %% | КП ↑95, %% | ошибка своих и прочее%n", nGal, tu[0], (int) tu[1]));
        StringBuilder list = new StringBuilder("\nПринятые чужие: вариант | проба | человек пробы | принят к | участник? | s\n");
        Object[][] vars = {{"(а)", rA, bA}, {"(б)", rB, bB}, {"(в)", rC, bB}};
        for (Object[] v : vars) {
            XBaseFinal.Run rr = (XBaseFinal.Run) v[1];
            XBaseVec.Blk b = (XBaseVec.Blk) v[2];
            List<XBaseVec.Model> mt = XBaseFinal.models(rr, b, XBaseFinal.TT1, false);
            List<XBaseVec.Model> mx = XBaseFinal.models(rr, b, XBaseFinal.XS, false);
            List<Ctrl> cs = ctrl(mt, mx, b.g().rOf, b.g().r0);
            long hub = cs.stream().filter(c -> sel.part().contains(c.best())).count();
            t.append(String.format(Locale.ROOT, "%s | %d | %d | %.1f%n", v[0], cs.size(), hub, 100.0 * hub / cs.size()));
            List<Ctrl> ok = cs.stream().filter(c -> c.s() <= tu[0]).toList();
            double[] st = XBaseVec.fuseStats(mt, false);
            double[] sx = XBaseVec.fuseStats(mx, false);
            List<XBaseFinalReport.Out1> o1 = XBaseFinalReport.atT(mt, mx, st, sx, b.g().rOf, b.g().r0, tu[0], ro -> ro.equals(FeiScores.OWN) || ro.equals(FeiScores.CTRL));
            long acc1 = o1.stream().filter(o -> !o.own() && o.accepted()).count();
            if (acc1 != ok.size()) throw new IllegalStateException("glasssep: принятые чужие разошлись с atT: " + acc1 + " и " + ok.size());
            acc.append(String.format(Locale.ROOT, "%s | %d из %d | %.2f | %.2f | %s%n", v[0], ok.size(), cs.size(), 100.0 * ok.size() / cs.size(), 100 * FaceEvaluation.binomialUpperBound(ok.size(), cs.size(), 0.95), XBaseFinalReport.summary(o1)));
            for (Ctrl c : ok) list.append(String.format(Locale.ROOT, "%s | %s | %s | %s | %s | %.3f%n", v[0], c.key(), c.person(), c.best(), sel.part().contains(c.best()) ? "да" : "нет", c.s()));
        }
        t.append(acc).append(list);

        XBaseFinalReport.M mA = XBaseFinalReport.measure(la.cfgs(), u.persons, w);
        XBaseFinalReport.M mB = XBaseFinalReport.measure(lb.cfgs(), u.persons, w);
        XBaseFinalReport.M mC = XBaseFinalReport.measure(lc.cfgs(), u.persons, w);
        t.append("\n=== Вся строка GT межс., справочно ===\nстрока | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.level("(а) одно состояние", mA)).append(XBaseEsrc.level("(б) оба состояния", mB)).append(XBaseEsrc.level("(в) раздельные шаблоны", mC));
        t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.row("(в) − (б)", mC, mB)).append(XBaseEsrc.row("(в) − (а)", mC, mA));
        t.append(String.format(Locale.ROOT, "%nВремя: всего %.0f с%n", (System.nanoTime() - t0) / 1e9));
        for (Mat q : u.fr.values()) q.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        FarMethods.write(rf, t);
        System.out.print(t);
        System.out.println("DONE");
    }
}
