package svd.recognizer.model;

import java.io.Serializable;

/**
 * Неизменяемый DTO — результат обучения подпространства одного класса.
 * Содержит средний вектор и ортонормированный базис подпространства.
 */
public final class SubspaceModel implements Serializable {
    private static final long serialVersionUID = 1L;

    private final double[] meanVector;     // длина 4096
    private final double[][] basisMatrix;  // 4096 x k
    private final int k;

    public SubspaceModel(double[] meanVector, double[][] basisMatrix, int k) {
        this.meanVector = meanVector;
        this.basisMatrix = basisMatrix;
        this.k = k;
    }

    public double[] getMeanVector() {return meanVector;}

    public double[][] getBasisMatrix() {return basisMatrix;}

    public int getK() {return k;}
}