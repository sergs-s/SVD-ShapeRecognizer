package svd.recognizer.faces;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Гистограммы LBP (Ahonen, Hadid, Pietikäinen, 2004; п. 4.2, «LBP-гистограммы»): параметры фиксированы заранее и на FEI
 * не подбираются. Равномерные коды LBP⁽ᵘ²⁾ с 8 соседями на окружности радиуса 2 (значения соседей — билинейной
 * интерполяцией, бит 1 — сосед ≥ центра, бит p — сосед под углом 2πp/8, против часовой стрелки от направления вправо);
 * 58 равномерных кодов (не больше двух переходов 0/1 по кругу) — метки 0…57 по возрастанию кода, остальные — метка 58.
 * Коды — у пикселей с полной окрестностью (кадр 92×112 → 88×108); сетка 7×7 блоков по кадру кодов (границы ⌊i·88/7⌋,
 * ⌊i·108/7⌋); гистограмма блока нормирована на сумму 1 (блоки чуть разного размера); гистограммы блоков сцеплены по
 * строкам блоков — вектор 7·7·59 = 2891; весов блоков нет. Вход — кадр «а» без нормализации освещения (LBP не меняется
 * при монотонном изменении яркости).
 *
 * @author ssv
 */
final class LbpFeature {

    static final int NEIGHBORS = 8;
    static final double RADIUS = 2;
    static final int GRID = 7;
    static final int BINS = 59;
    static final int LENGTH = GRID * GRID * BINS;
    /** Метка кода 0…255: равномерные — 0…57, прочие — 58. */
    static final int[] LABEL = labels();
    /** Смещения соседей (x, y), округлены до 1e-9 (у p = 0, 2, 4, 6 — ровно целые). */
    private static final double[][] OFFSETS = offsets();

    private LbpFeature() {
    }

    static boolean uniform(int code) {
        int rotated = ((code >> 1) | ((code & 1) << (NEIGHBORS - 1))) & 0xFF;
        return Integer.bitCount((code ^ rotated) & 0xFF) <= 2;
    }

    private static int[] labels() {
        int[] l = new int[1 << NEIGHBORS];
        int next = 0;
        for (int c = 0; c < l.length; c++) l[c] = uniform(c) ? next++ : -1;
        if (next != BINS - 1) throw new IllegalStateException("Равномерных кодов " + next + " вместо 58");
        for (int c = 0; c < l.length; c++) if (l[c] < 0) l[c] = BINS - 1;
        return l;
    }

    private static double[][] offsets() {
        double[][] o = new double[NEIGHBORS][2];
        for (int p = 0; p < NEIGHBORS; p++) {
            double a = 2 * Math.PI * p / NEIGHBORS;
            o[p][0] = Math.round(RADIUS * Math.cos(a) * 1e9) / 1e9;
            o[p][1] = Math.round(-RADIUS * Math.sin(a) * 1e9) / 1e9;
        }
        return o;
    }

    /** Значение в точке (x, y) билинейной интерполяцией (img — по строкам, ширина w). */
    static double sample(double[] img, int w, double x, double y) {
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        double fx = x - x0;
        double fy = y - y0;
        double v = (1 - fx) * (1 - fy) * img[y0 * w + x0];
        if (fx > 0) v += fx * (1 - fy) * img[y0 * w + x0 + 1];
        if (fy > 0) v += (1 - fx) * fy * img[(y0 + 1) * w + x0];
        if (fx > 0 && fy > 0) v += fx * fy * img[(y0 + 1) * w + x0 + 1];
        return v;
    }

    /** Коды LBP пикселей с полной окрестностью: [(h − 4)·(w − 4)] по строкам. */
    static int[] codes(double[] img, int w, int h) {
        int r = (int) RADIUS;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        int[] out = new int[cw * ch];
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double c = img[(y + r) * w + x + r];
                int code = 0;
                for (int p = 0; p < NEIGHBORS; p++) {
                    if (sample(img, w, x + r + OFFSETS[p][0], y + r + OFFSETS[p][1]) >= c) code |= 1 << p;
                }
                out[y * cw + x] = code;
            }
        }
        return out;
    }

    /** Вектор гистограмм (2891) по яркостям кадра. */
    static double[] histogram(double[] img, int w, int h) {
        int r = (int) RADIUS;
        int cw = w - 2 * r;
        int ch = h - 2 * r;
        int[] codes = codes(img, w, h);
        double[] v = new double[LENGTH];
        for (int by = 0; by < GRID; by++) {
            int y0 = by * ch / GRID;
            int y1 = (by + 1) * ch / GRID;
            for (int bx = 0; bx < GRID; bx++) {
                int x0 = bx * cw / GRID;
                int x1 = (bx + 1) * cw / GRID;
                int off = (by * GRID + bx) * BINS;
                for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) v[off + LABEL[codes[y * cw + x]]]++;
                double n = (double) (y1 - y0) * (x1 - x0);
                for (int k = 0; k < BINS; k++) v[off + k] /= n;
            }
        }
        return v;
    }

    /** Вектор гистограмм кадра (8 бит, один канал). */
    static double[] histogram(Mat gray8) {
        if (gray8.type() != CvType.CV_8UC1) throw new IllegalArgumentException("LBP: кадр не 8 бит одноканальный");
        int w = gray8.cols();
        int h = gray8.rows();
        byte[] b = new byte[w * h];
        gray8.get(0, 0, b);
        double[] img = new double[b.length];
        for (int i = 0; i < b.length; i++) img[i] = b[i] & 0xFF;
        return histogram(img, w, h);
    }

    /**
     * Маска блоков сетки 7×7 (п. 4.2, «LBP: внутренние и краевые блоки»; фиксирована заранее, на FEI не подбирается):
     * INNER — строки и столбцы блоков 1–5 (25 блоков), RING — краевое кольцо (строка или столбец 0 или 6, 24 блока).
     */
    enum Mask {
        INNER, RING;

        boolean has(int by, int bx) {
            boolean in = by >= 1 && by <= GRID - 2 && bx >= 1 && bx <= GRID - 2;
            return this == INNER ? in : !in;
        }

        int blocks() {
            int n = 0;
            for (int by = 0; by < GRID; by++) for (int bx = 0; bx < GRID; bx++) if (has(by, bx)) n++;
            return n;
        }
    }

    /** Гистограммы блоков маски (вектор длины LENGTH, по строкам блоков) — в том же порядке, сцеплены: blocks()·59. */
    static double[] select(double[] v, Mask mask) {
        if (v.length != LENGTH) throw new IllegalArgumentException("LBP: длина вектора " + v.length + " вместо " + LENGTH);
        double[] out = new double[mask.blocks() * BINS];
        int k = 0;
        for (int by = 0; by < GRID; by++) {
            for (int bx = 0; bx < GRID; bx++) {
                if (!mask.has(by, bx)) continue;
                System.arraycopy(v, (by * GRID + bx) * BINS, out, k, BINS);
                k += BINS;
            }
        }
        return out;
    }

    /** Поэлементный квадратный корень (L1). */
    static double[] sqrt(double[] v) {
        double[] s = new double[v.length];
        for (int i = 0; i < v.length; i++) s[i] = Math.sqrt(v[i]);
        return s;
    }
}
