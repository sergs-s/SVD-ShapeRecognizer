package svd.recognizer.faces;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

/**
 * Двухпроходный YuNet (2′): первый проход — на уменьшенном входе, второй — по области
 * лица из полного разрешения (рамка первого прохода с полями {@value #MARGIN} её размера,
 * масштаб — лицо около {@value #FACE} px). Строки — 15 чисел YuNet (рамка, 5 точек,
 * оценка) в координатах полного снимка. Общий код для OwnEvaluation и GalleryEvaluation.
 *
 * @author ssv
 */
public final class YunetTwoPass {

    /** Второй проход: поле вокруг рамки первого прохода — доля её размера с каждой стороны. */
    static final double MARGIN = 0.5;
    /** Второй проход: размер лица (ширина рамки первого прохода) во входе второго прохода, px. */
    static final double FACE = 300;

    private YunetTwoPass() {
    }

    /** Первый проход на входе с масштабом scale: лучшая по оценке строка в координатах полного снимка или null. */
    static double[] firstPass(FaceDetectorYN detector, Mat input, double scale) {
        detector.setInputSize(input.size());
        Mat faces = new Mat();
        detector.detect(input, faces);
        if (faces.rows() == 0) return null;
        int best = best(faces);
        double[] first = new double[15];
        for (int j = 0; j < 15; j++) first[j] = j == 14 ? faces.get(best, j)[0] : faces.get(best, j)[0] / scale;
        return first;
    }

    /**
     * Второй проход по рамке box = {x, y, w, h} (полный снимок): лучшая строка в координатах
     * полного снимка или null, если лица не нашлось.
     */
    static double[] secondPass(FaceDetectorYN detector, Mat fullColor, double[] box) {
        double[] b = box;
        int x0 = (int) Math.max(0, Math.floor(b[0] - MARGIN * b[2]));
        int y0 = (int) Math.max(0, Math.floor(b[1] - MARGIN * b[3]));
        int x1 = (int) Math.min(fullColor.cols(), Math.ceil(b[0] + b[2] + MARGIN * b[2]));
        int y1 = (int) Math.min(fullColor.rows(), Math.ceil(b[1] + b[3] + MARGIN * b[3]));
        double f = FACE / b[2];
        Mat crop = new Mat();
        Imgproc.resize(new Mat(fullColor, new Rect(x0, y0, x1 - x0, y1 - y0)), crop,
                new Size(Math.round((x1 - x0) * f), Math.round((y1 - y0) * f)), 0, 0, f < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        detector.setInputSize(crop.size());
        Mat faces = new Mat();
        detector.detect(crop, faces);
        if (faces.rows() == 0) return null;
        int best = best(faces);
        double[] row = new double[15];
        for (int j = 0; j < 15; j++) row[j] = faces.get(best, j)[0];
        // В полный кадр: x = x0 + x_crop / f; ширина и высота — / f.
        double[] full = new double[15];
        for (int j = 0; j < 14; j++) {
            boolean isX = j % 2 == 0;
            full[j] = j == 2 || j == 3 ? row[j] / f : (isX ? x0 : y0) + row[j] / f;
        }
        full[14] = row[14];
        return full;
    }

    /** Оба прохода: строка второго прохода, если он нашёл лицо, иначе первого; null — лица нет. */
    static double[] detect(FaceDetectorYN detector, Mat fullColor, Mat input, double scale) {
        double[] first = firstPass(detector, input, scale);
        if (first == null) return null;
        double[] second = secondPass(detector, fullColor, new double[] {first[0], first[1], first[2], first[3]});
        return second == null ? first : second;
    }

    private static int best(Mat faces) {
        int best = 0;
        for (int i = 1; i < faces.rows(); i++) {
            if (faces.get(i, 14)[0] > faces.get(best, 14)[0]) best = i;
        }
        return best;
    }
}
