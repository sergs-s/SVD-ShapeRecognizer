package svd.recognizer.core;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import svd.recognizer.model.ShapeTemplate;
import java.util.List;

/**
 *
 * @author ssv
 */
public class SVDTrainer {

    /**
     * Trains a ShapeTemplate from a list of feature vectors using SVD.
     *
     * @param label       shape label (e.g. "circle", "square", "triangle")
     * @param featureVecs list of feature vectors from training samples
     * @return trained ShapeTemplate
     */
    public ShapeTemplate train(String label, List<double[]> featureVecs) {
        if (featureVecs == null || featureVecs.isEmpty()) {
            throw new IllegalArgumentException("No feature vectors for label: " + label);
        }
        int rows = featureVecs.size();
        int cols = featureVecs.get(0).length;

        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            data[i] = featureVecs.get(i);
        }

        RealMatrix matrix = new Array2DRowRealMatrix(data);
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);

        // Keep top singular vectors as the template basis
        RealMatrix U = svd.getU();
        double[] singularValues = svd.getSingularValues();

        return new ShapeTemplate(label, U.getData(), singularValues);
    }
}
