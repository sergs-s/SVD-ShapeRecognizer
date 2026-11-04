package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import svd.recognizer.storage.SettingsStore;

/**
 * Этап «к цели», проба 3 (Жорик №1 от 05.11): вариация внутри сессии при регистрации, GT. Галерея – N снимков даты галереи (межс. 6, внутри 3 из своей половины), вместо случайного выбора –
 * правило вслепую «наиболее разные»: признаки снимка – поза r (YuNet, со знаком) и средняя яркость кадра Б, каждая стандартизована по всем снимкам дат галереи GT; жадно – первый ближайший к среднему,
 * далее наибольшее наименьшее расстояние до отобранных, при равенстве – меньший номер снимка. W и модели частей обучаются заново, остальное – как итоговый прогон (кадры Б).
 *
 * @author ssv
 */
final class XBaseVarSel {

    private XBaseVarSel() {
    }

    /** Точка отбора. */
    record Pt(String key, int no, double a, double b) {}

    /** Жадный отбор n самых далёких точек (координаты уже стандартизованы); результат – в порядке отбора. */
    static List<String> greedy(List<Pt> pts, int n) {
        List<Pt> rest = new ArrayList<>(pts);
        rest.sort(Comparator.comparingInt(Pt::no));
        List<Pt> sel = new ArrayList<>();
        if (rest.size() <= n) return rest.stream().map(Pt::key).toList();
        Pt first = null;
        double bd = Double.POSITIVE_INFINITY;
        for (Pt p : rest) {
            double d = p.a() * p.a() + p.b() * p.b();
            if (d < bd) {
                bd = d;
                first = p;
            }
        }
        sel.add(first);
        rest.remove(first);
        while (sel.size() < n) {
            Pt best = null;
            double bv = -1;
            for (Pt p : rest) {
                double m = Double.POSITIVE_INFINITY;
                for (Pt q : sel) m = Math.min(m, (p.a() - q.a()) * (p.a() - q.a()) + (p.b() - q.b()) * (p.b() - q.b()));
                if (m > bv) {
                    bv = m;
                    best = p;
                }
            }
            sel.add(best);
            rest.remove(best);
        }
        return sel.stream().map(Pt::key).toList();
    }

    private static Map<String, List<String>> choose(List<String> own, Map<String, List<String>> src, Map<String, double[]> z, XBaseEval.Gt g, int n) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (String p : own) {
            List<Pt> pts = new ArrayList<>();
            for (String k : src.get(p)) pts.add(new Pt(k, g.img(k), z.get(k)[0], z.get(k)[1]));
            List<String> s = new ArrayList<>(greedy(pts, n));
            s.sort(null);
            out.put(p, s);
        }
        return out;
    }

    static void run(FarExport.Loaded export, Path dir, Map<String, Map<Integer, String>> feiKeys, List<String> selLines, SettingsStore settings, Path out) throws Exception {
        if (!FrameB.on()) throw new IllegalStateException("varsel: FACES_FRAMES=b");
        long t0 = System.nanoTime();
        XBaseFinal.Run r = XBaseFinal.setup("gt", false, export, dir, feiKeys, selLines, settings, out);
        XBaseVec.Setup u = r.u();
        XBaseEval.Gt tpl = u.tpl;
        Map<String, double[]> rows = FrameK2.exportRows(dir);
        // Стандартизация по всем снимкам дат галереи GT
        List<String> allKeys = new ArrayList<>();
        for (String p : tpl.own) allKeys.addAll(tpl.galleryAll.get(p));
        double[][] raw = new double[allKeys.size()][2];
        for (int i = 0; i < raw.length; i++) {
            String k = allKeys.get(i);
            raw[i][0] = FeiDataset.pose(rows.get(k));
            raw[i][1] = Core.mean(u.fr.get(k)).val[0];
        }
        double[] m = new double[2];
        double[] sd = new double[2];
        for (int j = 0; j < 2; j++) {
            for (double[] v : raw) m[j] += v[j] / raw.length;
            for (double[] v : raw) sd[j] += (v[j] - m[j]) * (v[j] - m[j]);
            sd[j] = Math.sqrt(sd[j] / (raw.length - 1));
        }
        Map<String, double[]> z = new HashMap<>();
        for (int i = 0; i < raw.length; i++) z.put(allKeys.get(i), new double[] {(raw[i][0] - m[0]) / sd[0], (raw[i][1] - m[1]) / sd[1]});
        StringBuilder t = new StringBuilder("Этап «к цели», проба 3 – наиболее разные снимки при регистрации – GT (XBaseEval gt-varsel; Жорик №1 от 05.11)\nКод: коммит " + FarMethods.commit() + "\n");
        t.append(String.format(Locale.ROOT, "Стандартизация по %d снимкам дат галереи: r среднее %.4f, разброс %.4f; яркость среднее %.2f, разброс %.2f; бутстреп по людям B = %d, seed %d%n", raw.length, m[0], sd[0], m[1], sd[1], tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET));
        List<XBaseVec.Blk> news = new ArrayList<>();
        List<Map<String, List<String>>> withinHalf = XBaseEval.withinSplit(tpl.own, tpl.galleryAll, tpl.seed);
        StringBuilder diff = new StringBuilder("\n=== Выбор против случайного (правило №8) ===\nгруппа | снимков на человека | людей | людей с иным выбором | снимков заменено | из | разброс внутри шаблона r / яркость (среднее по людям, в единицах разброса): случайный → наиболее разные\n");
        for (XBaseVec.Blk b : u.blocks) {
            int n = b.group().equals("межс.") ? b.gal().get(tpl.own.get(0)).size() : b.gal().get(tpl.own.get(0)).size();
            Map<String, List<String>> src = b.group().equals("межс.") ? tpl.galleryAll : withinHalf.get(0);
            Map<String, List<String>> gal = choose(tpl.own, src, z, b.g(), n);
            int chP = 0;
            int chK = 0;
            int tot = 0;
            for (String p : tpl.own) {
                Set<String> a = new HashSet<>(b.gal().get(p));
                tot += a.size();
                int same = 0;
                for (String k : gal.get(p)) if (a.contains(k)) same++;
                if (same != a.size() || same != gal.get(p).size()) chP++;
                chK += a.size() - same;
            }
            diff.append(String.format(Locale.ROOT, "%s | %d | %d | %d | %d | %d | %.3f / %.3f → %.3f / %.3f%n", b.group(), n, tpl.own.size(), chP, chK, tot, spread(b.gal(), z, 0), spread(b.gal(), z, 1), spread(gal, z, 0), spread(gal, z, 1)));
            news.add(new XBaseVec.Blk(b.name(), b.abl2(), b.g(), gal, b.pro(), b.parts(), b.probes().stream().filter(q -> !q.role().equals(XBaseFinal.CTRL_X)).toList(), b.extra(), b.s(), b.group()));
        }
        t.append(diff);
        Path root = r.dir().resolveSibling("varsel");
        Files.createDirectories(root.resolve("vec"));
        XBaseFinal.Run rs = new XBaseFinal.Run(u, r.k2(), Map.of(), root, false, 0);
        Set<String> need = new HashSet<>();
        for (XBaseVec.Blk b : news) {
            b.gal().values().forEach(need::addAll);
            b.g().cohort.forEach(need::addAll);
            for (XBaseEval.Probe q : b.probes()) need.add(q.key());
            if (b.extra() != null) for (List<XBaseEval.Probe> l : b.extra()) for (XBaseEval.Probe q : l) need.add(q.key());
        }
        Map<String, Mat> fr = new HashMap<>();
        for (Map.Entry<String, Mat> e : u.fr.entrySet()) if (need.contains(e.getKey())) fr.put(e.getKey(), e.getValue());
        String commit = FarMethods.git("rev-parse", "HEAD:src/main/java/svd/recognizer/faces/XBaseVarSel.java");
        double fitS = 0;
        for (XBaseFinal.Cfg c : List.of(XBaseFinal.TT1, XBaseFinal.XS)) {
            XBaseEval x = null;
            for (XBaseVec.Blk b : news) {
                java.nio.file.Path vf = XBaseFinal.vfile(rs, b, c);
                if (Files.exists(vf)) {
                    if (!XBaseVec.readHead(vf).contains(commit)) throw new IllegalStateException("varsel: другой код: " + vf);
                    continue;
                }
                if (x == null) x = XBasePart4.vectorsB(fr, r.k2(), XBaseFinal.param(c, tpl.params), c.f(), tpl.threads);
                long ts = System.nanoTime();
                List<XBaseVec.Model> ms = XBaseVec.run(List.of(XBaseVec.job(b.g(), x, b.gal(), b.probes(), b.parts(), b.s(), b.extra(), u.cin, u.cout)), Math.min(8, tpl.threads)).get(0);
                java.io.Writer w = XBaseVec.writer(vf, "# Наиболее разные (XBaseVarSel), " + c.name() + "\n# Код: коммит " + FarMethods.commit() + "; " + commit + "\n");
                XBaseVec.write(w, b.name() + " " + c.name(), ms);
                XBaseVec.finish(w, vf);
                fitS += (System.nanoTime() - ts) / 1e9;
                System.out.printf(Locale.ROOT, "varsel: %s %s готово (%.0f с)%n", c.id(), b.name(), (System.nanoTime() - t0) / 1e9);
            }
            if (x != null) {
                x.vec.clear();
                x.mir.clear();
            }
        }
        int[][] w = FeiBootstrap.weights(u.persons, tpl.b, tpl.seed + FeiBootstrap.SEED_OFFSET);
        t.append("\n=== Наиболее разные − (32): запись: разность [95 %]; у pAUC больше – лучше, у прочих меньше – лучше; условие Δ по №1 не считается, внутри – справочно ===\n");
        for (int i = 0; i < news.size(); i++) {
            XBaseVec.Blk bo = u.blocks.get(i);
            XBaseVec.Blk bn = news.get(i);
            XBaseFinalReport.M mB = XBaseFinalReport.measure(List.of(XBaseGoalPrep.cfg(bo, bo.g(), XBaseGoalPrep.fused(r, bo, XBaseFinal.TT1, bo.g().rOf, bo.g().r0))), u.persons, w);
            XBaseFinalReport.M mN = XBaseFinalReport.measure(List.of(XBaseGoalPrep.cfg(bn, bn.g(), XBaseGoalPrep.fused(rs, bn, XBaseFinal.TT1, bn.g().rOf, bn.g().r0))), u.persons, w);
            t.append("\n").append(bo.group()).append(": строка | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.level("(32) слияние, случайный выбор", mB)).append(XBaseEsrc.level("наиболее разные, слияние", mN));
            t.append("пара | pAUC ≤ 10 % | EER | FAR 5 % | FAR 1 %\n").append(XBaseEsrc.row("наиболее разные − (32)", mN, mB));
        }
        t.append("\nFERET не считался: по правилу eq на дате галереи один fa и один fb – выбора нет (№1, п. 1).\n");
        t.append(String.format(Locale.ROOT, "%nВремя: обучение и оценка %.0f с, всего %.0f с%n", fitS, (System.nanoTime() - t0) / 1e9));
        for (Mat q : u.fr.values()) q.release();
        for (FrameK2.F k : r.k2().values()) k.frame().release();
        FarMethods.write(root.resolve("xbase_gt_varsel.txt"), t);
        System.out.print(t);
        System.out.println("DONE");
    }

    /** Средний по людям разброс признака внутри шаблона. */
    private static double spread(Map<String, List<String>> gal, Map<String, double[]> z, int j) {
        double s = 0;
        for (List<String> keys : gal.values()) {
            double m = 0;
            for (String k : keys) m += z.get(k)[j] / keys.size();
            double v = 0;
            for (String k : keys) v += (z.get(k)[j] - m) * (z.get(k)[j] - m);
            s += Math.sqrt(v / Math.max(1, keys.size() - 1));
        }
        return s / gal.size();
    }
}
