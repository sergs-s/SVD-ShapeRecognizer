package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * (а3) Синтез снимков галереи (PLAN.md п. 5.5; задание №3 от 14.10.2026, план подтверждён Хозяином). Синтезируется то, что
 * меняет коды LBP и различается между сессиями:
 * <ul>
 *   <li>ошибка выравнивания — к 5 точкам YuNet применяется подобие из эмпирического набора «YuNet → ручная разметка» (сдвиг
 *       в долях межглазья, поворот, масштаб), кадр «а» вырезается заново из исходного снимка;</li>
 *   <li>направленный свет — плавный горизонтальный перепад с асимметрией A = |ln(m_L / m_R)| из распределения базы и
 *       случайным знаком, либо (в доле SHADOW) односторонняя тень со сглаженной границей около середины лица, глубина — по
 *       верхней четверти распределения A;</li>
 *   <li>шум — гауссов, до σ (Immerkær) из распределения базы.</li>
 * </ul>
 * Яркость, контраст и гамма не синтезируются (LBP к ним инвариантен). Диапазоны этапа 1 — по всем сессиям базы
 * (оптимистично: распределения видели и даты проб).
 *
 * @author ssv
 */
final class FaceSynth {

    /** Доля синтезов с односторонней тенью вместо плавного перепада. */
    static final double SHADOW = 1.0 / 3;
    /** Ширина сглаживания границы тени, пиксели кадра «а». */
    static final double SHADOW_S = 3;
    /** Разброс положения границы тени около середины кадра, пиксели. */
    static final double SHADOW_X = 6;

    /** Диапазоны: σ шума и A по кадрам «а» (по возрастанию), подобия выравнивания {u_x, u_y, θ, ln s} в долях межглазья. */
    record Ranges(double[] noise, double[] asym, List<double[]> align) {}

    /** Параметры одного синтеза. */
    record Params(double[] align, double noise, double asym, int sign, boolean shadow, double shadowX) {}

    private FaceSynth() {
    }

    /** Подобие from → to в долях межглазья точек from: {u_x, u_y, θ, ln s} (см. apply). */
    static double[] alignDelta(double[][] from, double[][] to) {
        Mat s = FaceAlignment.similarity(from, to);
        double a = s.get(0, 0)[0];
        double b = s.get(1, 0)[0];
        double tx = s.get(0, 2)[0];
        double ty = s.get(1, 2)[0];
        s.release();
        double sc = Math.hypot(a, b);
        double th = Math.atan2(b, a);
        double[] c = FeretData.mid(from[0], from[1]);
        double d = Math.hypot(from[1][0] - from[0][0], from[1][1] - from[0][1]);
        double ux = (a * c[0] - b * c[1] + tx - c[0]) / d;
        double uy = (b * c[0] + a * c[1] + ty - c[1]) / d;
        return new double[] {ux, uy, th, Math.log(sc)};
    }

    /** Строка YuNet с 5 точками, сдвинутыми подобием delta (центр — середина глаз, единица — межглазье). */
    static double[] apply(double[] row, double[] delta) {
        double[] out = row.clone();
        double cx = (row[4] + row[6]) / 2;
        double cy = (row[5] + row[7]) / 2;
        double d = Math.hypot(row[6] - row[4], row[7] - row[5]);
        double s = Math.exp(delta[3]);
        double cos = Math.cos(delta[2]) * s;
        double sin = Math.sin(delta[2]) * s;
        for (int j = 0; j < 5; j++) {
            double x = row[4 + 2 * j] - cx;
            double y = row[5 + 2 * j] - cy;
            out[4 + 2 * j] = cx + cos * x - sin * y + d * delta[0];
            out[5 + 2 * j] = cy + sin * x + cos * y + d * delta[1];
        }
        return out;
    }

    static Params sample(Ranges r, Random rnd) {
        double[] al = r.align().get(rnd.nextInt(r.align().size()));
        double noise = r.noise()[rnd.nextInt(r.noise().length)];
        boolean shadow = rnd.nextDouble() < SHADOW;
        double asym;
        if (shadow) {
            int from = (int) Math.floor(0.75 * r.asym().length);
            asym = r.asym()[from + rnd.nextInt(Math.max(1, r.asym().length - from))];
        } else {
            asym = r.asym()[rnd.nextInt(r.asym().length)];
        }
        int sign = rnd.nextBoolean() ? 1 : -1;
        double shadowX = 46 + (2 * rnd.nextDouble() - 1) * SHADOW_X;
        return new Params(al, noise, asym, sign, shadow, shadowX);
    }

    /** Синтез кадра «а»: исходный снимок color, строка YuNet row, параметры p, шум — из rnd. */
    static Mat synth(Mat color, double[] row, Params p, Random rnd) {
        Mat a = FeretData.frameA(color, apply(row, p.align()));
        int w = a.cols();
        int h = a.rows();
        byte[] px = new byte[w * h];
        a.get(0, 0, px);
        double[] v = new double[w * h];
        for (int i = 0; i < v.length; i++) v[i] = px[i] & 0xff;
        // Свет: множитель по столбцу.
        for (int x = 0; x < w; x++) {
            double f;
            if (p.shadow()) {
                double depth = 1 - Math.exp(-p.asym());
                double sig = 1 / (1 + Math.exp(-(x - p.shadowX()) / SHADOW_S));
                f = 1 - depth * (p.sign() > 0 ? sig : 1 - sig);
            } else {
                f = Math.exp(p.sign() * p.asym() * (2.0 * x / (w - 1) - 1) / 2);
            }
            for (int y = 0; y < h; y++) v[y * w + x] *= f;
        }
        // Шум: до σ цели (Immerkær) по текущему кадру.
        Mat f64 = new Mat(h, w, CvType.CV_64FC1);
        f64.put(0, 0, v);
        double cur = XBaseQuality.noiseSigma(f64);
        f64.release();
        double add = Math.sqrt(Math.max(0, p.noise() * p.noise() - cur * cur));
        for (int i = 0; i < v.length; i++) {
            double z = v[i] + add * rnd.nextGaussian();
            px[i] = (byte) (int) Math.round(Math.max(0, Math.min(255, z)));
        }
        a.put(0, 0, px);
        return a;
    }

    /** Медиана по массиву (копия сортируется). */
    static double median(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        return s.length == 0 ? Double.NaN : s.length % 2 == 1 ? s[s.length / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
    }

    /** Смещение глаз (среднее двух) в долях межглазья для подобия delta — для сверки с измеренной медианой 5,5 %. */
    static double eyeShift(double[] delta) {
        double[] row = new double[15];
        row[4] = -0.5;
        row[6] = 0.5;
        double[] o = apply(row, delta);
        return (Math.hypot(o[4] + 0.5, o[5]) + Math.hypot(o[6] - 0.5, o[7])) / 2;
    }

    static List<Double> list(double[] a) {
        List<Double> l = new ArrayList<>();
        for (double x : a) l.add(x);
        return l;
    }
}
