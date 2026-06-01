package svd.recognizer.core;

/**
 *
 * @author ssv
 */
public class ImagePreprocessor {

    /**
     * Preprocesses an image for SVD feature extraction.
     * Converts to grayscale, applies Gaussian blur, thresholding, morphology.
     *
     * @param imagePath path to the input image
     * @return preprocessed binary image as Mat
     */
    public org.opencv.core.Mat preprocess(String imagePath) {
        org.opencv.core.Mat src = org.opencv.imgcodecs.Imgcodecs.imread(imagePath);
        if (src.empty()) {
            throw new IllegalArgumentException("Cannot load image: " + imagePath);
        }
        return preprocessMat(src);
    }

    /**
     * Preprocesses an already loaded Mat.
     *
     * @param src input Mat
     * @return preprocessed binary Mat
     */
    public org.opencv.core.Mat preprocessMat(org.opencv.core.Mat src) {
        org.opencv.core.Mat gray = new org.opencv.core.Mat();
        org.opencv.core.Mat blurred = new org.opencv.core.Mat();
        org.opencv.core.Mat binary = new org.opencv.core.Mat();
        org.opencv.core.Mat morphed = new org.opencv.core.Mat();

        if (src.channels() > 1) {
            org.opencv.imgproc.Imgproc.cvtColor(src, gray, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = src.clone();
        }

        org.opencv.imgproc.Imgproc.GaussianBlur(gray, blurred, new org.opencv.core.Size(5, 5), 0);

        org.opencv.imgproc.Imgproc.threshold(blurred, binary, 0, 255,
                org.opencv.imgproc.Imgproc.THRESH_BINARY_INV + org.opencv.imgproc.Imgproc.THRESH_OTSU);

        org.opencv.core.Mat kernel = org.opencv.imgproc.Imgproc.getStructuringElement(
                org.opencv.imgproc.Imgproc.MORPH_RECT, new org.opencv.core.Size(3, 3));
        org.opencv.imgproc.Imgproc.morphologyEx(binary, morphed,
                org.opencv.imgproc.Imgproc.MORPH_CLOSE, kernel);

        return morphed;
    }
}
