package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Проверка реализации Fisherfaces на ORL (шаг 5, этап 2, конфигурация 2-orl-check): 40 человек, 5 снимков обучение,
 * 5 контроль, опознание argmin (закрытое множество, без порога). Разбиения: фиксированное (1–5 / 6–10) и 10 случайных
 * (seed + 3000 + номер). Решение — ближайший центр класса (как в FarMethods) и ближайший обучающий снимок в пространстве
 * LDA (как в литературе). Fisherfaces — с PCA до N − C и с меньшим числом компонент PCA (40, 80, 120); рядом —
 * подпространство на человека (ε) и MLDA. Кадр а 92×112 из экспорта (без нормализации и Tan – Triggs) и 32×32 — как в
 * ориентире.
 * Ориентир: Cai, He, Han, Zhang. Orthogonal Laplacianfaces for Face Recognition. IEEE TIP 15(11), 2006, 3608–3614 —
 * Fisherfaces на ORL, 5 обучающих: ошибка 7,75 % (кадры 32×32, случайные разбиения, ближайший сосед).
 *
 * @author ssv
 */
final class OrlLdaCheck {

    static final int TRAIN = 5;
    static final int RANDOM_SPLITS = 10;

    private OrlLdaCheck() {
    }

    static String run(FarMethods fm) {
        StringBuilder t = new StringBuilder();
        t.append("Fisherfaces на ORL (закрытое множество 40 человек, 5 обучение / 5 контроль, опознание argmin, 200 проб на разбиение).\n");
        t.append("Ориентир (по выдаче поиска; сайт статьи закрыт прокси, с текстом не сверено): Cai, He, Han, Zhang, IEEE TIP 15(11),\n"
                + "2006, 3608–3614 — Fisherfaces на ORL, 5 обучающих: ошибка 7,75 % (32×32, случайные разбиения, ближайший сосед).\n");
        List<String> persons = new ArrayList<>(fm.data.orl().keySet());
        Collections.sort(persons);
        long seed = fm.settings.loadFacesSeed();
        // Условия ориентира: кадры 32×32 (INTER_AREA), без нормализации.
        Map<String, double[]> v32 = new java.util.HashMap<>();
        for (Map.Entry<String, org.opencv.core.Mat> e : fm.frames.entrySet()) {
            org.opencv.core.Mat small = new org.opencv.core.Mat();
            org.opencv.imgproc.Imgproc.resize(e.getValue(), small, new org.opencv.core.Size(32, 32), 0, 0,
                    org.opencv.imgproc.Imgproc.INTER_AREA);
            v32.put(e.getKey(), IlluminationNorm.NONE.vector(small, fm.params));
            small.release();
        }
        String[] labels = {"без нормализации, 92×112", "Tan – Triggs, 92×112", "без нормализации, 32×32 (как в ориентире)"};
        List<Map<String, double[]>> sets = List.of(fm.vectors(IlluminationNorm.NONE), fm.vectors(IlluminationNorm.TAN_TRIGGS), v32);
        for (int set = 0; set < sets.size(); set++) {
            Map<String, double[]> v = sets.get(set);
            String[] names = {"Fisherfaces (PCA N − C = 160), центр класса", "Fisherfaces (PCA N − C = 160), ближайший снимок",
                "Fisherfaces PCA 40, центр класса", "Fisherfaces PCA 80, центр класса", "Fisherfaces PCA 120, центр класса",
                "MLDA, центр класса", "подпространство на человека, ε"};
            int[] fixed = new int[names.length];
            int[] rnd = new int[names.length];
            int fixedN = 0;
            int rndN = 0;
            String fisherInfo = "";
            for (int split = -1; split < RANDOM_SPLITS; split++) {
                List<List<double[]>> train = new ArrayList<>();
                List<List<double[]>> test = new ArrayList<>();
                for (String p : persons) {
                    List<GalleryEvaluation.Sample> imgs = new ArrayList<>(fm.data.orl().get(p));
                    imgs.sort((a, b) -> Integer.compare(number(a), number(b)));
                    if (split >= 0) Collections.shuffle(imgs, new Random(seed + 3000 + split));
                    List<double[]> tr = new ArrayList<>();
                    List<double[]> te = new ArrayList<>();
                    for (int i = 0; i < imgs.size(); i++) {
                        double[] x = v.get(imgs.get(i).file().toString());
                        if (x == null) continue;
                        (i < TRAIN ? tr : te).add(x);
                    }
                    train.add(tr);
                    test.add(te);
                }
                int n = persons.size();
                LdaModel fisher = (LdaModel) new FisherScorer().fit(train, n);
                if (split < 0) fisherInfo = fisher.info();
                GalleryScorer.ScoreModel[] fisherK = {new FisherScorer(40).fit(train, n), new FisherScorer(80).fit(train, n),
                    new FisherScorer(120).fit(train, n)};
                GalleryScorer.ScoreModel mlda = new MldaScorer().fit(train, n);
                GalleryScorer.ScoreModel sub = new SubspaceScorer(false).fit(train, n);
                List<double[]> emb = new ArrayList<>();
                List<Integer> lab = new ArrayList<>();
                for (int c = 0; c < n; c++) {
                    for (double[] x : train.get(c)) {
                        emb.add(fisher.embed(x));
                        lab.add(c);
                    }
                }
                int[] ok = new int[names.length];
                int probes = 0;
                for (int c = 0; c < n; c++) {
                    for (double[] x : test.get(c)) {
                        probes++;
                        if ((int) FarMethods.best(fisher.scores(x))[1] == c) ok[0]++;
                        double[] y = fisher.embed(x);
                        int nn = 0;
                        double bd = Double.POSITIVE_INFINITY;
                        for (int k = 0; k < emb.size(); k++) {
                            double d = NearestVectorScorer.distance2(y, emb.get(k));
                            if (d < bd) {
                                bd = d;
                                nn = lab.get(k);
                            }
                        }
                        if (nn == c) ok[1]++;
                        for (int k = 0; k < fisherK.length; k++) if ((int) FarMethods.best(fisherK[k].scores(x))[1] == c) ok[2 + k]++;
                        if ((int) FarMethods.best(mlda.scores(x))[1] == c) ok[5]++;
                        if ((int) FarMethods.best(sub.scores(x))[1] == c) ok[6]++;
                    }
                }
                for (int k = 0; k < names.length; k++) {
                    if (split < 0) fixed[k] += ok[k];
                    else rnd[k] += ok[k];
                }
                if (split < 0) fixedN = probes;
                else rndN += probes;
            }
            t.append(String.format(Locale.ROOT, "  %s; Fisherfaces (фиксированное разбиение): %s%n", labels[set], fisherInfo));
            for (int k = 0; k < names.length; k++) {
                t.append(String.format(Locale.ROOT, "    %s: фиксированное %d/%d = %.1f %%; %d случайных — %d/%d = %.1f %%%n", names[k],
                        fixed[k], fixedN, 100.0 * fixed[k] / fixedN, RANDOM_SPLITS, rnd[k], rndN, 100.0 * rnd[k] / rndN));
            }
        }
        return t.toString();
    }

    /** Номер снимка ORL из имени файла (1…10). */
    static int number(GalleryEvaluation.Sample s) {
        return Integer.parseInt(FarExport.stem(s.file()).replaceAll("\\D", ""));
    }
}
