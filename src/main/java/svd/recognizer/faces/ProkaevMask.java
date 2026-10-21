package svd.recognizer.faces;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import svd.recognizer.storage.SettingsStore;

/**
 * Д-№3 (Жорик-дуэль, 21.10.2026): фон против лица в методе Прокаева – сводка по готовым файлам оценок основной строки (Б2-К, k = 6, l = 3,
 * гости 10) четырёх комплектов: К-лицо и К-обычный (полные прогоны Д10), Т1 «только лицо» и Т2 «лицо закрашено» (FACES_PROKAEV_FRAME =
 * plain-t1 / plain-t2, FACES_PROKAEV_ONLY = main). Без пересчёта. Меры: верный первый кандидат (без порога), ошибка своих при фактическом FAR
 * контроля 1 % (порог по контрольным чужим каждой конфигурации, XBaseOther.errAtFar); FEI – ещё по порогу пороговых чужих (FAR 1 %, по
 * снимкам, как fei-prokaev). Точки, без интервалов.
 *
 * @author ssv
 */
final class ProkaevMask {

    static final String[][] SETS = {{"_face0.700", "К-лицо"}, {"_plain", "К-обычный"}, {"_plain_t1_main", "Т1 «только лицо»"},
        {"_plain_t2_main", "Т2 «лицо закрашено»"}};

    private ProkaevMask() {
    }

    /** Конфигурации разбора по блокам основной строки постановки conf (FEI – «FEI»); null – нет файла. */
    static List<FeiDecompose.Cfg> cfgs(Path file, String conf, Map<String, Integer> pIdx) throws IOException {
        if (!Files.exists(file)) return null;
        List<FeiDecompose.Cfg> out = new ArrayList<>();
        for (Map.Entry<String, String> e : XBaseSubsp.blocks(file).entrySet()) {
            if (!e.getKey().startsWith(conf + ", проход ") || !e.getKey().endsWith(", " + ProkaevEval.MAIN)) continue;
            String ls = XBaseAbl.abs(e.getValue());
            for (String l : ls.split("\n")) if (!l.isEmpty()) pIdx.computeIfAbsent(FeiScores.parse(l).person(), x -> pIdx.size());
            FeiDecompose.Cfg c = ProkaevN.cfg(ls, pIdx);
            List<double[]> thr = new ArrayList<>();
            for (String l : ls.split("\n")) {
                if (l.isEmpty()) continue;
                FeiScores.Row r = FeiScores.parse(l);
                if (r.role().equals(FeiScores.THR)) thr.add(new double[] {pIdx.get(r.person()), FeiScores.ratio(r, null)});
            }
            thr.sort((a, z) -> Double.compare(a[1], z[1]));
            c.thrP = thr.stream().mapToInt(a -> (int) a[0]).toArray();
            c.thrR = thr.stream().mapToDouble(a -> a[1]).toArray();
            out.add(c);
        }
        if (out.isEmpty()) throw new IllegalStateException("prokaev-mask: нет блоков основной строки «" + conf + "» в " + file);
        return out;
    }

    /** {верный первый %, ошибка своих при FAR контроля 1 %, фактический FAR %, ошибка своих по порогу пороговых (FEI) %, FAR контроля при нём %}. */
    static double[] measures(List<FeiDecompose.Cfg> cs, int persons, boolean fei) {
        int[] unit = FeiBootstrap.unit(persons);
        long good = 0;
        long all = 0;
        for (FeiDecompose.Cfg c : cs) {
            all += c.ownOk.length;
            for (boolean b : c.ownOk) if (b) good++;
        }
        double[] sep = XBaseOther.errAtFar(cs, unit, 0.01);
        double[] m = {100.0 * good / all, sep[0], sep[1], Double.NaN, Double.NaN};
        if (fei) {
            double e = 0;
            double n = 0;
            double fx = 0;
            double fn = 0;
            for (FeiDecompose.Cfg c : cs) {
                double th = FeiBootstrap.threshold(c.thrR, c.thrP, unit, 0.01);
                for (int i = 0; i < c.ownR.length; i++) {
                    n++;
                    if (!(c.ownOk[i] && c.ownR[i] <= th)) e++;
                }
                for (double x : c.ctrlR) {
                    fn++;
                    if (x <= th) fx++;
                }
            }
            m[3] = 100 * e / n;
            m[4] = 100 * fx / fn;
        }
        return m;
    }

    static void run(SettingsStore settings, Path out) throws IOException {
        String rawDir = settings.loadFacesFeretDir();
        if (rawDir == null) throw new IllegalStateException("prokaev-mask: задайте faces.feret.dir / FACES_FERET_DIR");
        Path fe = Paths.get(rawDir).resolve("reports").resolve(XBaseEval.eqDir(0, true));
        Object[][] bases = {{"FERET", "межс.", fe, "feret"}, {"FERET", "внутри", fe, "feret"}, {"GT", "межс.", out, "gt"}, {"GT", "внутри", out, "gt"},
            {"FEI", "FEI", out, "fei"}};
        StringBuilder r = new StringBuilder("PLAN.md п. 5.5, Д-№3: фон против лица в методе Прокаева (XBaseEval prokaev-mask) – по готовым файлам оценок, без "
                + "пересчёта; справочно, в правило отключений не входит\n");
        r.append("Код: коммит ").append(FarMethods.commit()).append('\n');
        r.append("Строка – основная (Б2-К, k = 6, l = 3, гости 10). Комплекты: К-лицо (H = h_рамки / 0,70) и К-обычный – полные прогоны Д10; Т1 «только лицо» "
                + "(вне эллипса – ровный серый) и Т2 «лицо закрашено» (внутри эллипса – ровный серый) – К-обычный с маской, эллипс вписан в рамку YuNet, "
                + "расширенную в 1,10 раза по каждой оси, серый – среднее кадра; прогон только основной строки (FACES_PROKAEV_ONLY = main).\n");
        r.append("Меры: верный первый кандидат – доля проб своих с верным лучшим (без порога); «разд.» – ошибка своих при фактическом FAR контроля 1 % "
                + "(порог по контрольным чужим каждой конфигурации), в скобках – фактический FAR контроля; FEI – ещё ошибка своих по порогу пороговых "
                + "чужих (FAR 1 %, по снимкам) и FAR контроля при нём. Точки, без интервалов.\n");
        r.append("\nФайлы оценок:\n");
        StringBuilder t = new StringBuilder("\n=== Фон против лица: база, постановка × комплект ===\n");
        t.append("база | постановка | комплект | верный первый, % | разд. FAR 1 %, % (фактический FAR, %) | FEI: ошибка своих по порогу пороговых, % "
                + "(FAR контроля, %)\n");
        for (Object[] b : bases) {
            for (String[] s : SETS) {
                Path f = ((Path) b[2]).resolve("xbase_" + b[3] + "_prokaev" + s[0] + "_scores.tsv.gz");
                if (b[1].equals("межс.") || b[0].equals("FEI")) r.append(f).append(Files.exists(f) ? " (" + FeiThreshold.sha(f) + ")" : " – нет файла").append('\n');
                Map<String, Integer> pIdx = new HashMap<>();
                List<FeiDecompose.Cfg> cs = cfgs(f, (String) b[1], pIdx);
                if (cs == null) {
                    t.append(String.format(Locale.ROOT, "%s | %s | %s | нет файла оценок | – | –%n", b[0], b[1], s[1]));
                    continue;
                }
                double[] m = measures(cs, pIdx.size(), b[0].equals("FEI"));
                t.append(String.format(Locale.ROOT, "%s | %s | %s | %.1f | %.1f (%.2f) | %s%n", b[0], b[1], s[1], m[0], m[1], m[2],
                        Double.isNaN(m[3]) ? "–" : String.format(Locale.ROOT, "%.1f (%.2f)", m[3], m[4])));
            }
        }
        r.append(t);
        FarMethods.write(out.resolve("xbase_prokaev_mask.txt"), r);
        System.out.print(r);
    }
}
