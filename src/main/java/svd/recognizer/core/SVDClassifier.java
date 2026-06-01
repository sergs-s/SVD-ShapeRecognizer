package svd.recognizer.core;

import svd.recognizer.model.ShapeTemplate;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import java.util.List;

/**
 *
 * @author ssv
 */
public class SVDClassifier {

    private final double threshold;

    /**
     * @param threshold maximum distance to accept recognition result
     */
    public SVDClassifier(double threshold) {
        this.threshold = threshold;
    }

    /**
     * Classifies a feature vector against a list of templates.
     *
     * @param featureVec   query feature vector
     * @param templates    list of trained ShapeTemplate
     * @return label of the best match, or "Unknown" if below threshold
     */
    public String classify(double[] featureVec, List<ShapeTemplate> templates) {
        String bestLabel = "Unknown";
        double bestDistance = Double.MAX_VALUE;

        for (ShapeTemplate t : templates) {
            double dist = computeSubspaceDistance(featureVec, t);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestLabel = t.getLabel();
            }
        }

        return bestDistance <= threshold ? bestLabel : "Unknown";
    }

    /**
     * Returns all distances for debugging/UI display.
     */
    public double[] getDistances(double[] featureVec, List<ShapeTemplate> templates) {
        double[] dists = new double[templates.size()];
        for (int i = 0; i < templates.size(); i++) {
            dists[i] = computeSubspaceDistance(featureVec, templates.get(i));
        }
        return dists;
    }

    private double computeSubspaceDistance(double[] vec, ShapeTemplate t) {
        RealMatrix U = new Array2DRowRealMatrix(t.getBasis());
        RealMatrix v = new Array2DRowRealMatrix(new double[][]{vec}).transpose();
        // Projection: P = U * U^T * v
        RealMatrix proj = U.multiply(U.transpose()).multiply(v);
        // Residual distance
        RealMatrix diff = v.subtract(proj);
        double norm = 0;
        for (double d : diff.getColumn(0)) norm += d * d;
        return Math.sqrt(norm);
    }
}
