package svd.recognizer.faces;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * FeatP4: при параметрах по умолчанию – побайтно XBaseFeat2.feature(…, 8, …) у обоих кандидатов; варианты пачки 4, Б – длины и конечность.
 *
 * @author ssv
 */
public class FeatP4Test {

    private static final IlluminationNorm.NormParams TT = new IlluminationNorm.NormParams(2.0, 8, 0.2, 1.0, 2.0, 0.1, 10.0);
    private static final double[][] PTS = {{60, 90}, {124, 90}, {92, 130}, {68, 165}, {116, 165}};

    @BeforeClass
    public static void load() {
        svd.recognizer.OpenCvLoader.load();
    }

    private static Mat frame(int w, int h, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC1);
        Core.setRNGSeed((int) seed);
        Core.randu(m, 0, 256);
        Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(5, 5), 1.5);
        return m;
    }

    @Test
    public void baseEqualsFeat2() {
        Mat a = frame(92, 112, 1);
        Mat c = frame(184, 224, 2);
        XBaseFeat2.Feats fs = new XBaseFeat2.Feats();
        fs.params = TT;
        fs.paramsK2 = XBaseFeat2.scaledTt(TT, FrameK2.K);
        for (int f = 0; f < 2; f++) {
            double[] ref = XBaseFeat2.feature(fs, f, XBaseFeat2.MS_JOIN, a, c, PTS);
            double[] mine = FeatP4.feature(FeatP4.base(TT), f, a, c, PTS);
            assertArrayEquals("кандидат " + f, ref, mine, 0.0);
        }
    }

    @Test
    public void variants() {
        Mat a = frame(92, 112, 3);
        Mat c = frame(184, 224, 4);
        FeatP4.P b = FeatP4.base(TT);
        FeatP4.P[] vs = {new FeatP4.P(b.inTt(), b.inXs(), TT, 2, 7, 5, 5, 40, 1), new FeatP4.P(b.inTt(), b.inXs(), TT, 4, 7, 5, 5, 40, 1),
            new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 5, 5, 5, 40, 1), new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 7, 4, 5, 40, 1),
            new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 7, 5.5, 5, 40, 1), new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 7, 5, 4, 40, 1),
            new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 7, 5, 6, 48, 2), new FeatP4.P(b.inTt(), b.inXs(), TT, 3, 7, 5, 5, 32, 0.5),
            new FeatP4.P(FeatP4.In.RETINA, FeatP4.In.RETINA, TT, 3, 7, 5, 5, 40, 1)};
        for (FeatP4.P p : vs) {
            for (int f = 0; f < 2; f++) {
                int g = XBasePoem.grid(f);
                double[] v = FeatP4.feature(p, f, a, c, PTS);
                int len = XBaseDates.PARAMS[f].length() + g * g * p.orient() * LbpFeature.BINS + MsLbpFeature.POINTS * p.scales() * 16 * LbpFeature.BINS;
                assertEquals(len, v.length);
                for (double x : v) assertTrue(Double.isFinite(x) && x >= 0);
            }
        }
    }
}
