package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Блочный метод (шаг 5, п. 4.2 PLAN.md): базовый оценщик на прямоугольной области выровненного кадра 92×112.
 * Вектор кадра — построчно (FaceAlignment.toVector); из него берутся пиксели области, тоже построчно.
 *
 * @author ssv
 */
final class RegionScorer implements GalleryScorer {

    static final int FRAME_W = 92;
    static final int FRAME_H = 112;

    /** Область кадра 92×112: левый верхний угол (x, y), ширина и высота в пикселях. */
    record Region(String id, String name, int x, int y, int w, int h) {
        Region {
            if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > FRAME_W || y + h > FRAME_H) {
                throw new IllegalArgumentException("Область вне кадра 92×112: " + id);
            }
        }

        String label() {
            return String.format(Locale.ROOT, "%s: x %d–%d, y %d–%d (%d×%d)", name, x, x + w - 1, y, y + h - 1, w, h);
        }

        /** Пиксели области из вектора кадра. */
        double[] crop(double[] v) {
            if (v.length != FRAME_W * FRAME_H) throw new IllegalArgumentException("Вектор не кадра 92×112: " + v.length);
            double[] out = new double[w * h];
            for (int r = 0; r < h; r++) System.arraycopy(v, (y + r) * FRAME_W + x, out, r * w, w);
            return out;
        }
    }

    /**
     * Области (шаблон 5 точек в кадре 92×112: глаза (28,3; 51,7) и (63,5; 51,5), нос (46,0; 71,7), углы рта
     * (31,5; 92,4) и (60,7; 92,2)); проверены глазами на выровненных кадрах своей базы, GT, MUCT, ORL.
     */
    static final Region EYES = new Region("eyes", "глаза (полоса с бровями)", 8, 38, 76, 26);
    static final Region NOSE = new Region("nose", "нос", 30, 52, 32, 30);
    static final Region MOUTH = new Region("mouth", "рот", 20, 80, 52, 24);
    static final Region FACE = new Region("face", "всё лицо", 0, 0, FRAME_W, FRAME_H);

    private final GalleryScorer base;
    private final Region region;

    RegionScorer(GalleryScorer base, Region region) {
        this.base = base;
        this.region = region;
    }

    @Override
    public String label() {
        return "область " + region.label() + "; " + base.label();
    }

    @Override
    public ScoreModel fit(List<List<double[]>> classes, int gallery) {
        List<List<double[]>> cropped = new ArrayList<>();
        for (List<double[]> c : classes) {
            List<double[]> list = new ArrayList<>();
            for (double[] x : c) list.add(region.crop(x));
            cropped.add(list);
        }
        ScoreModel m = base.fit(cropped, gallery);
        return new ScoreModel() {
            @Override
            public double[] scores(double[] x) {
                return m.scores(region.crop(x));
            }

            @Override
            public String info() {
                return m.info();
            }
        };
    }
}
