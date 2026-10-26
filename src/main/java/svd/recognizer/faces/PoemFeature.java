package svd.recognizer.faces;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Гистограммы POEM (Vu, Caplier, «Face Recognition with Patterns of Oriented Edge Magnitudes», ECCV 2010; п. 5.5, «Постановка
 * 2.4, POEM», Жорик 25.10.2026). Градиент – Собель 3×3 по внутренним пикселям (без досочинения края): gx – правый столбец
 * минус левый, gy – нижняя строка минус верхняя (веса 1, 2, 1); величина √(gx² + gy²), ориентация без знака θ = atan2(gy, gx)
 * mod π, M = 3 интервала [0, π/3), [π/3, 2π/3), [2π/3, π), жёсткое отнесение (величина целиком в свой интервал). Ячейка:
 * сумма величин своей ориентации по квадрату CELL×CELL (7×7; только пиксели с полной ячейкой). Код: 8 соседей на окружности
 * радиуса RADIUS (5) – билинейная интерполяция (LbpFeature.sample), бит p – сосед под углом 2πp/8, против часовой стрелки от
 * направления вправо (как LBP), бит 1 при «сосед − центр > τ», τ = TAU_REL · наибольшее значение карты ориентации (своё у
 * каждой ориентации); равномерные коды (метки LbpFeature.LABEL, 59). Коды – у пикселей с полной окрестностью: кадр 92×112 →
 * 74×94 (край 1 + 3 + 5 = 9); сетка блоков по кадру кодов – как у LBP (границы ⌊i·74/G⌋, ⌊i·94/G⌋); гистограмма каждого
 * (блока, ориентации) нормирована на сумму 1; порядок – по строкам блоков, в блоке ориентации 0, 1, 2; длина G²·3·59.
 *
 * @author ssv
 */
final class PoemFeature {

    static final int ORIENT = 3;
    static final int CELL = 7;
    static final double RADIUS = 5;
    static final double TAU_REL = 1e-6;
    /** Край кадра без кодов: Собель 1 + полуячейка 3 + радиус 5. */
    static final int BORDER = 1 + CELL / 2 + (int) RADIUS;
    /** Множитель части POEM в сцеплении (П2) после корня: квадрат нормы на блок – 3 гистограммы · 1/3 = 1, как у LBP (поправка Жорика, №6 от 26.10). */
    static final double JOIN_SCALE = 1 / Math.sqrt(ORIENT);

    private static final double[][] OFFSETS = offsets();

    private PoemFeature() {
    }

    private static double[][] offsets() {
        double[][] o = new double[LbpFeature.NEIGHBORS][2];
        for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
            double a = 2 * Math.PI * p / LbpFeature.NEIGHBORS;
            o[p][0] = Math.round(RADIUS * Math.cos(a) * 1e9) / 1e9;
            o[p][1] = Math.round(-RADIUS * Math.sin(a) * 1e9) / 1e9;
        }
        return o;
    }

    static int length(int grid) {
        return grid * grid * ORIENT * LbpFeature.BINS;
    }

    /** Номер интервала ориентации (gx, gy): θ = atan2(gy, gx) mod π. */
    static int bin(double gx, double gy) {
        double t = Math.atan2(gy, gx);
        if (t < 0) t += Math.PI;
        if (t >= Math.PI) t -= Math.PI;
        return Math.min(ORIENT - 1, (int) (t * ORIENT / Math.PI));
    }

    /** Карты величин по ориентациям [o][(h − 2)·(w − 2)] (Собель по внутренним пикселям). */
    static double[][] magnitudes(double[] img, int w, int h) {
        int gw = w - 2;
        int gh = h - 2;
        double[][] m = new double[ORIENT][gw * gh];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int u = (y - 1) * w;
                int c = y * w;
                int d = (y + 1) * w;
                double gx = img[u + x + 1] + 2 * img[c + x + 1] + img[d + x + 1] - img[u + x - 1] - 2 * img[c + x - 1] - img[d + x - 1];
                double gy = img[d + x - 1] + 2 * img[d + x] + img[d + x + 1] - img[u + x - 1] - 2 * img[u + x] - img[u + x + 1];
                m[bin(gx, gy)][(y - 1) * gw + x - 1] = Math.sqrt(gx * gx + gy * gy);
            }
        }
        return m;
    }

    /** Суммы по квадрату CELL×CELL у пикселей с полной ячейкой: [(gh − CELL + 1)·(gw − CELL + 1)] (по строкам, затем по столбцам). */
    static double[] cells(double[] g, int gw, int gh) {
        int aw = gw - CELL + 1;
        int ah = gh - CELL + 1;
        double[] row = new double[aw * gh];
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < aw; x++) {
                double s = 0;
                for (int k = 0; k < CELL; k++) s += g[y * gw + x + k];
                row[y * aw + x] = s;
            }
        }
        double[] out = new double[aw * ah];
        for (int y = 0; y < ah; y++) {
            for (int x = 0; x < aw; x++) {
                double s = 0;
                for (int k = 0; k < CELL; k++) s += row[(y + k) * aw + x];
                out[y * aw + x] = s;
            }
        }
        return out;
    }

    /** Коды карты a (ширина aw, высота ah) с порогом tau у пикселей с полной окрестностью радиуса RADIUS. */
    static int[] codes(double[] a, int aw, int ah, double tau) {
        int r = (int) RADIUS;
        int cw = aw - 2 * r;
        int ch = ah - 2 * r;
        int[] out = new int[cw * ch];
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double c = a[(y + r) * aw + x + r];
                int code = 0;
                for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
                    if (LbpFeature.sample(a, aw, x + r + OFFSETS[p][0], y + r + OFFSETS[p][1]) - c > tau) code |= 1 << p;
                }
                out[y * cw + x] = code;
            }
        }
        return out;
    }

    /** Коды по ориентациям [o][(h − 2·BORDER)·(w − 2·BORDER)]. */
    static int[][] codes(double[] img, int w, int h) {
        if (w <= 2 * BORDER || h <= 2 * BORDER) throw new IllegalArgumentException("POEM: кадр " + w + "×" + h + " меньше окрестности");
        double[][] m = magnitudes(img, w, h);
        int gw = w - 2;
        int gh = h - 2;
        int aw = gw - CELL + 1;
        int ah = gh - CELL + 1;
        int[][] out = new int[ORIENT][];
        for (int o = 0; o < ORIENT; o++) {
            double[] a = cells(m[o], gw, gh);
            double max = 0;
            for (double v : a) max = Math.max(max, v);
            out[o] = codes(a, aw, ah, TAU_REL * max);
        }
        return out;
    }

    /** Вектор гистограмм (length(grid)) по кадру img (по строкам, ширина w). */
    static double[] histogram(double[] img, int w, int h, int grid) {
        int[][] codes = codes(img, w, h);
        int cw = w - 2 * BORDER;
        int ch = h - 2 * BORDER;
        int bins = LbpFeature.BINS;
        double[] v = new double[length(grid)];
        for (int by = 0; by < grid; by++) {
            int y0 = by * ch / grid;
            int y1 = (by + 1) * ch / grid;
            for (int bx = 0; bx < grid; bx++) {
                int x0 = bx * cw / grid;
                int x1 = (bx + 1) * cw / grid;
                double n = (double) (y1 - y0) * (x1 - x0);
                for (int o = 0; o < ORIENT; o++) {
                    int off = ((by * grid + bx) * ORIENT + o) * bins;
                    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) v[off + LbpFeature.LABEL[codes[o][y * cw + x]]]++;
                    for (int k = 0; k < bins; k++) v[off + k] /= n;
                }
            }
        }
        return v;
    }

    /** Вход признака: яркости кадра (NONE) или выход Tan – Triggs (TAN_TRIGGS), как у LbpFeature. */
    static double[] input(Mat gray8, IlluminationNorm norm, IlluminationNorm.NormParams p) {
        if (gray8.type() != CvType.CV_8UC1) throw new IllegalArgumentException("POEM: кадр не 8 бит одноканальный");
        return switch (norm) {
            case NONE -> {
                byte[] b = new byte[gray8.cols() * gray8.rows()];
                gray8.get(0, 0, b);
                double[] img = new double[b.length];
                for (int i = 0; i < b.length; i++) img[i] = b[i] & 0xFF;
                yield img;
            }
            case TAN_TRIGGS -> IlluminationNorm.TAN_TRIGGS.vector(gray8, p);
            default -> throw new IllegalArgumentException("POEM: нормировка " + norm + " не предусмотрена");
        };
    }

    /** Сцепление (П2): корень гистограмм LBP, затем корень гистограмм POEM, умноженный на JOIN_SCALE. */
    static double[] join(double[] lbpSqrt, double[] poemSqrt) {
        double[] v = new double[lbpSqrt.length + poemSqrt.length];
        System.arraycopy(lbpSqrt, 0, v, 0, lbpSqrt.length);
        for (int i = 0; i < poemSqrt.length; i++) v[lbpSqrt.length + i] = poemSqrt[i] * JOIN_SCALE;
        return v;
    }
}
