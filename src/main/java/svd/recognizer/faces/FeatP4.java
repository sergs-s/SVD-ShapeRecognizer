package svd.recognizer.faces;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Признак конфигурации (32) – (б) (П2) + MS-LBP (вариант 8 пачки 2) – с настраиваемыми параметрами для пачки 4, Б (задание Жорика №1–№5
 * от 01.11.2026): вход (кадр, Тан – Триггс с параметрами варианта, ретинальный фильтр Vu), POEM – число ориентаций m, ячейка w, радиус
 * блока L / 2 (PoemFeature – те же шаги с постоянными 3, 7, 5), MS-LBP – число масштабов S, окно (4×4 ячейки), множитель веса части. При
 * параметрах по умолчанию признак побайтно равен XBaseFeat2.feature(…, 8, …) (тест FeatP4Test). Классы PoemFeature, MsLbpFeature,
 * XBaseFeat2 не меняются.
 *
 * @author ssv
 */
final class FeatP4 {

    /** Вход признака. */
    enum In {
        NONE, TT, RETINA
    }

    /**
     * Параметры варианта: вход L1-tt-xs-F и L1-xs-F (по умолчанию TT и NONE), параметры Тан – Триггса кадра «а» (k = 2 – σ × 2), POEM (m, w,
     * радиус), MS-LBP (масштабы, окно, множитель веса).
     */
    record P(In inTt, In inXs, IlluminationNorm.NormParams tt, int orient, int cell, double radius, int scales, int patch, double msWeight) {
        In in(int f) {
            return f == 1 ? inTt : inXs;
        }
    }

    static P base(IlluminationNorm.NormParams tt) {
        return new P(In.TT, In.NONE, tt, PoemFeature.ORIENT, PoemFeature.CELL, PoemFeature.RADIUS, MsLbpFeature.SCALES, MsLbpFeature.PATCH, 1);
    }

    private FeatP4() {
    }

    static double[] gray(Mat gray8) {
        if (gray8.type() != CvType.CV_8UC1) throw new IllegalArgumentException("FeatP4: кадр не 8 бит одноканальный");
        byte[] b = new byte[gray8.cols() * gray8.rows()];
        gray8.get(0, 0, b);
        double[] img = new double[b.length];
        for (int i = 0; i < b.length; i++) img[i] = b[i] & 0xFF;
        return img;
    }

    /** Вход кадра: k – масштаб σ (1 – кадр «а», 2 – кадр k = 2). */
    static double[] input(Mat gray8, In in, IlluminationNorm.NormParams tt, int k) {
        return switch (in) {
            case NONE -> gray(gray8);
            case TT -> IlluminationNorm.TAN_TRIGGS.vector(gray8, k == 1 ? tt : XBaseFeat2.scaledTt(tt, k));
            case RETINA -> RetinaFilter.filter(gray8, k == 1 ? RetinaFilter.AUTHOR : RetinaFilter.AUTHOR.scaled(k));
        };
    }

    // ---------------------------------------------------------------- POEM с параметрами (шаги PoemFeature)

    static int bin(double gx, double gy, int orient) {
        double t = Math.atan2(gy, gx);
        if (t < 0) t += Math.PI;
        if (t >= Math.PI) t -= Math.PI;
        return Math.min(orient - 1, (int) (t * orient / Math.PI));
    }

    static double[][] magnitudes(double[] img, int w, int h, int orient) {
        int gw = w - 2;
        int gh = h - 2;
        double[][] m = new double[orient][gw * gh];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int u = (y - 1) * w;
                int c = y * w;
                int d = (y + 1) * w;
                double gx = img[u + x + 1] + 2 * img[c + x + 1] + img[d + x + 1] - img[u + x - 1] - 2 * img[c + x - 1] - img[d + x - 1];
                double gy = img[d + x - 1] + 2 * img[d + x] + img[d + x + 1] - img[u + x - 1] - 2 * img[u + x] - img[u + x + 1];
                m[bin(gx, gy, orient)][(y - 1) * gw + x - 1] = Math.sqrt(gx * gx + gy * gy);
            }
        }
        return m;
    }

    static double[] cells(double[] g, int gw, int gh, int cell) {
        int aw = gw - cell + 1;
        int ah = gh - cell + 1;
        double[] row = new double[aw * gh];
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < aw; x++) {
                double s = 0;
                for (int k = 0; k < cell; k++) s += g[y * gw + x + k];
                row[y * aw + x] = s;
            }
        }
        double[] out = new double[aw * ah];
        for (int y = 0; y < ah; y++) {
            for (int x = 0; x < aw; x++) {
                double s = 0;
                for (int k = 0; k < cell; k++) s += row[(y + k) * aw + x];
                out[y * aw + x] = s;
            }
        }
        return out;
    }

    /** Край окрестности кода: ⌈радиус⌉ (при целом радиусе – как PoemFeature). */
    static int reach(double radius) {
        return (int) Math.ceil(radius - 1e-9);
    }

    static double[][] offsets(double radius) {
        double[][] o = new double[LbpFeature.NEIGHBORS][2];
        for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
            double a = 2 * Math.PI * p / LbpFeature.NEIGHBORS;
            o[p][0] = Math.round(radius * Math.cos(a) * 1e9) / 1e9;
            o[p][1] = Math.round(-radius * Math.sin(a) * 1e9) / 1e9;
        }
        return o;
    }

    static int[] codes(double[] a, int aw, int ah, double tau, double radius, double[][] off) {
        int r = reach(radius);
        int cw = aw - 2 * r;
        int ch = ah - 2 * r;
        int[] out = new int[cw * ch];
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                double c = a[(y + r) * aw + x + r];
                int code = 0;
                for (int p = 0; p < LbpFeature.NEIGHBORS; p++) {
                    if (LbpFeature.sample(a, aw, x + r + off[p][0], y + r + off[p][1]) - c > tau) code |= 1 << p;
                }
                out[y * cw + x] = code;
            }
        }
        return out;
    }

    /** Гистограммы POEM (корня нет) по кадру img: блоки grid × grid, в блоке – ориентации подряд; длина grid²·m·59. */
    static double[] poem(double[] img, int w, int h, int grid, int orient, int cell, double radius) {
        return poem(img, w, h, grid, orient, cell, radius, null);
    }

    /** Маска эллипса кадра w × h ((Ч+О)): XBaseChin.mask, true – внутри; кэш по размеру. */
    private static final java.util.Map<Long, boolean[]> ELLIPSE = new java.util.concurrent.ConcurrentHashMap<>();

    static boolean[] ellipse(int w, int h) {
        return ELLIPSE.computeIfAbsent(((long) w << 32) | h, k -> {
            Mat m = XBaseChin.mask(w, h);
            byte[] b = new byte[w * h];
            m.get(0, 0, b);
            m.release();
            boolean[] out = new boolean[w * h];
            for (int i = 0; i < out.length; i++) out[i] = b[i] != 0;
            return out;
        });
    }

    /**
     * LBP с маской: как LbpFeature.histogram(img, w, h, lp), но код учитывается, только если его центр (пиксель кадра (x + r, y + r)) внутри
     * маски; доля – от числа учтённых кодов ячейки (ячейка без учтённых – нули).
     */
    static double[] lbpMasked(double[] img, int w, int h, LbpFeature.Params lp, boolean[] mask) {
        return lbpMasked(img, w, h, lp, mask, 0);
    }

    /**
     * Граница ряда блоков by (0 … rows) в области кодов высоты ch: chb ≤ 0 – прежнее деление ch на grid рядов; chb > 0 ((Чр), Жорик №2 от
     * 10.11) – ряды 0 … grid как у кадра базы с областью кодов высоты chb (by · chb / grid), плюс нижний ряд chb … ch.
     */
    static int rowEdge(int by, int ch, int grid, int chb) {
        if (chb <= 0) return by * ch / grid;
        return by <= grid ? by * chb / grid : ch;
    }

    /**
     * LBP с маской (mask == null – без маски) и сеткой (Чр): hb > 0 – высота кадра базы (112): ряды grid как у кадра базы, плюс нижний ряд до
     * низа кадра (длина grid · (grid + 1) · 59 на радиус); hb = 0 – сетка grid × grid (как lbpMasked).
     */
    static double[] lbpMasked(double[] img, int w, int h, LbpFeature.Params lp, boolean[] mask, int hb) {
        int grid = lp.grid();
        int rows = hb > 0 ? grid + 1 : grid;
        double[] v = new double[lp.radii().length * grid * rows * LbpFeature.BINS];
        for (int ri = 0; ri < lp.radii().length; ri++) {
            int r = (int) lp.radii()[ri];
            int cw = w - 2 * r;
            int ch = h - 2 * r;
            int[] codes = LbpFeature.codes(img, w, h, lp.radii()[ri]);
            int chb = hb > 0 ? hb - 2 * r : 0;
            int base = ri * grid * rows * LbpFeature.BINS;
            for (int by = 0; by < rows; by++) {
                int y0 = rowEdge(by, ch, grid, chb);
                int y1 = rowEdge(by + 1, ch, grid, chb);
                for (int bx = 0; bx < grid; bx++) {
                    int x0 = bx * cw / grid;
                    int x1 = (bx + 1) * cw / grid;
                    int off = base + (by * grid + bx) * LbpFeature.BINS;
                    int n = 0;
                    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
                        if (mask != null && !mask[(y + r) * w + x + r]) continue;
                        v[off + LbpFeature.LABEL[codes[y * cw + x]]]++;
                        n++;
                    }
                    if (n > 0) for (int k = 0; k < LbpFeature.BINS; k++) v[off + k] /= n;
                }
            }
        }
        return v;
    }

    /**
     * POEM; mask != null – код учитывается, только если центр его окна ячейки (пиксель кадра (x + r + 1 + (cell − 1)/2, …)) внутри маски;
     * доля – от числа учтённых кодов ячейки.
     */
    static double[] poem(double[] img, int w, int h, int grid, int orient, int cell, double radius, boolean[] mask) {
        return poem(img, w, h, grid, orient, cell, radius, mask, 0);
    }

    /** POEM с сеткой (Чр): hb > 0 – ряды grid как у кадра базы высоты hb, плюс нижний ряд до низа кадра (длина grid · (grid + 1) · m · 59). */
    static double[] poem(double[] img, int w, int h, int grid, int orient, int cell, double radius, boolean[] mask, int hb) {
        int border = 1 + cell / 2 + reach(radius);
        if (w <= 2 * border || h <= 2 * border) throw new IllegalArgumentException("FeatP4: кадр меньше окрестности POEM");
        double[][] m = magnitudes(img, w, h, orient);
        int gw = w - 2;
        int gh = h - 2;
        int aw = gw - cell + 1;
        int ah = gh - cell + 1;
        double[][] off = offsets(radius);
        int[][] codes = new int[orient][];
        for (int o = 0; o < orient; o++) {
            double[] a = cells(m[o], gw, gh, cell);
            double max = 0;
            for (double v : a) max = Math.max(max, v);
            codes[o] = codes(a, aw, ah, PoemFeature.TAU_REL * max, radius, off);
        }
        int cw = aw - 2 * reach(radius);
        int ch = ah - 2 * reach(radius);
        int bins = LbpFeature.BINS;
        int rows = hb > 0 ? grid + 1 : grid;
        int chb = hb > 0 ? hb - 2 - cell + 1 - 2 * reach(radius) : 0;
        double[] v = new double[grid * rows * orient * bins];
        for (int by = 0; by < rows; by++) {
            int y0 = rowEdge(by, ch, grid, chb);
            int y1 = rowEdge(by + 1, ch, grid, chb);
            for (int bx = 0; bx < grid; bx++) {
                int x0 = bx * cw / grid;
                int x1 = (bx + 1) * cw / grid;
                double n = (double) (y1 - y0) * (x1 - x0);
                if (mask != null) {
                    int sh = reach(radius) + 1 + (cell - 1) / 2;
                    n = 0;
                    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) if (mask[(y + sh) * w + x + sh]) n++;
                }
                for (int o = 0; o < orient; o++) {
                    int offv = ((by * grid + bx) * orient + o) * bins;
                    if (mask == null) {
                        for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) v[offv + LbpFeature.LABEL[codes[o][y * cw + x]]]++;
                    } else {
                        int sh = reach(radius) + 1 + (cell - 1) / 2;
                        for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) if (mask[(y + sh) * w + x + sh]) v[offv + LbpFeature.LABEL[codes[o][y * cw + x]]]++;
                    }
                    if (n > 0) for (int k = 0; k < bins; k++) v[offv + k] /= n;
                }
            }
        }
        return v;
    }

    // ---------------------------------------------------------------- MS-LBP с параметрами (шаги MsLbpFeature)

    static double[] patch(double[] img, int w, int h, int cx, int cy, int size) {
        int margin = MsLbpFeature.MARGIN;
        int n = size + 2 * margin;
        int x0 = cx - size / 2 - margin;
        int y0 = cy - size / 2 - margin;
        double[] p = new double[n * n];
        for (int y = 0; y < n; y++) {
            int yy = Math.max(0, Math.min(h - 1, y0 + y));
            for (int x = 0; x < n; x++) {
                int xx = Math.max(0, Math.min(w - 1, x0 + x));
                p[y * n + x] = img[yy * w + xx];
            }
        }
        return p;
    }

    static double[] msCells(double[] patch, int size) {
        int n = size + 2 * MsLbpFeature.MARGIN;
        int cells = MsLbpFeature.CELLS;
        int cell = size / cells;
        int[] codes = LbpFeature.codes(patch, n, n, LbpFeature.RADIUS);
        double[] v = new double[cells * cells * LbpFeature.BINS];
        for (int cy = 0; cy < cells; cy++) {
            for (int cx = 0; cx < cells; cx++) {
                int off = (cy * cells + cx) * LbpFeature.BINS;
                for (int y = cy * cell; y < (cy + 1) * cell; y++) for (int x = cx * cell; x < (cx + 1) * cell; x++) v[off + LbpFeature.LABEL[codes[y * size + x]]]++;
            }
        }
        for (int i = 0; i < v.length; i++) v[i] = Math.sqrt(v[i] / (cell * cell));
        return v;
    }

    /** MS-LBP (корни) по кадру k = 2: точки × масштабы × ячейки; длина 5·S·16·59. */
    static double[] msLbp(double[] img, int w, int h, double[][] pts, int scales, int size) {
        if (pts.length != MsLbpFeature.POINTS) throw new IllegalArgumentException("FeatP4: точек " + pts.length);
        int len = MsLbpFeature.CELLS * MsLbpFeature.CELLS * LbpFeature.BINS;
        double[] v = new double[MsLbpFeature.POINTS * scales * len];
        int k = 0;
        double[][] sc = new double[scales][];
        for (int s = 0; s < scales; s++) sc[s] = MsLbpFeature.scaled(img, w, h, s);
        for (double[] p : pts) {
            for (int s = 0; s < scales; s++) {
                int ws = s == 0 ? w : MsLbpFeature.size(w, s);
                int hs = s == 0 ? h : MsLbpFeature.size(h, s);
                int cx = (int) Math.round((p[0] + 0.5) * ws / w - 0.5);
                int cy = (int) Math.round((p[1] + 0.5) * hs / h - 0.5);
                double[] c = msCells(patch(sc[s], ws, hs, cx, cy, size), size);
                System.arraycopy(c, 0, v, k, c.length);
                k += c.length;
            }
        }
        return v;
    }

    /** Множитель части MS-LBP: правило равных норм √(2G² / (5·S·16)) × множитель варианта (решение Жорика №2 от 01.11). */
    static double msScale(int grid, int scales, double weight) {
        return Math.sqrt(2.0 * grid * grid / (MsLbpFeature.POINTS * scales * MsLbpFeature.CELLS * MsLbpFeature.CELLS)) * weight;
    }

    // ---------------------------------------------------------------- признак

    /** Признак кандидата f: (П2) – LBP (параметры кандидата) и POEM · 1/√m по входу кадра «а»; затем MS-LBP по входу кадра k = 2 · вес. */
    static double[] feature(P p, int f, Mat a, Mat c, double[][] pts) {
        In in = p.in(f);
        double[] img = input(a, in, p.tt(), 1);
        int w = a.cols();
        int h = a.rows();
        int grid = XBasePoem.grid(f);
        // (Ч+О), FACES_CHIN = co: пиксели вне эллипса не входят в гистограммы LBP и POEM; Тан – Триггс – по всему кадру (выше).
        boolean[] mk = ("co".equals(FrameB.chin()) || "cro".equals(FrameB.chin())) && w == FrameB.W && h == XBaseChin.height() ? ellipse(w, h) : null;
        // (Чр), FACES_CHIN = cr: ряды блоков LBP и POEM – как у кадра базы 92×112 (границы побайтно те же), плюс нижний ряд под подбородок.
        int hb = ("cr".equals(FrameB.chin()) || "cro".equals(FrameB.chin())) && w == FrameB.W && h == XBaseChin.height() ? FrameB.H : 0;
        double[] lbp = LbpFeature.sqrt(mk == null && hb == 0 ? LbpFeature.histogram(img, w, h, XBaseDates.PARAMS[f]) : lbpMasked(img, w, h, XBaseDates.PARAMS[f], mk, hb));
        double[] poem = LbpFeature.sqrt(poem(img, w, h, grid, p.orient(), p.cell(), p.radius(), mk, hb));
        double js = p.orient() == PoemFeature.ORIENT ? PoemFeature.JOIN_SCALE : 1 / Math.sqrt(p.orient());
        double[] ck = input(c, in, p.tt(), FrameK2.K);
        double[] ms = msLbp(ck, c.cols(), c.rows(), pts, p.scales(), p.patch());
        double sc = p.scales() == MsLbpFeature.SCALES && p.msWeight() == 1 ? MsLbpFeature.joinScale(grid) : msScale(grid, p.scales(), p.msWeight());
        double[] out = new double[lbp.length + poem.length + ms.length];
        System.arraycopy(lbp, 0, out, 0, lbp.length);
        for (int i = 0; i < poem.length; i++) out[lbp.length + i] = poem[i] * js;
        for (int i = 0; i < ms.length; i++) out[lbp.length + poem.length + i] = ms[i] * sc;
        return out;
    }
}
