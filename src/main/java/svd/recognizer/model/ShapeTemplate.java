package svd.recognizer.model;

import java.io.Serializable;

/**
 *
 * @author ssv
 */
public class ShapeTemplate implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String label;
    private final double[][] basis;
    private final double[] singularValues;

    /**
     * @param label         shape class label
     * @param basis         SVD U matrix (basis vectors)
     * @param singularValues singular values from SVD
     */
    public ShapeTemplate(String label, double[][] basis, double[] singularValues) {
        this.label = label;
        this.basis = basis;
        this.singularValues = singularValues;
    }

    public String getLabel() { return label; }
    public double[][] getBasis() { return basis; }
    public double[] getSingularValues() { return singularValues; }
}
