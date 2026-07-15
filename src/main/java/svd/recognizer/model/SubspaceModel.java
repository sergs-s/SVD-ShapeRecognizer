package svd.recognizer.model;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Неизменяемый DTO — результат обучения подпространства одного класса.
 *
 * Содержит:
 * - meanVector: средний вектор класса (длина 4096)
 * - basisMatrix: матрица базиса подпространства (4096 x k)
 * - k: размерность подпространства
 *
 * @author ssv
 */
public final class SubspaceModel implements Serializable {
    private static final long serialVersionUID = 1L;

    private final double[] meanVector;
    private final double[][] basisMatrix;
    private final int k;

    public SubspaceModel(double[] meanVector, double[][] basisMatrix, int k) {
        this.meanVector = meanVector.clone();
        this.basisMatrix = deepCopy(basisMatrix);
        this.k = k;
    }

    public double[] getMeanVector() {return meanVector.clone();}

    public double[][] getBasisMatrix() {return deepCopy(basisMatrix);}

    public int getK() {return k;}

    /**
     * Глубокое копирование матрицы для обеспечения неизменяемости.
     */
    private static double[][] deepCopy(double[][] matrix) {
        if (matrix == null) {
            return null;
        }
        double[][] copy = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            copy[i] = matrix[i].clone();
        }
        return copy;
    }

    @Override
    public String toString() {
        return "SubspaceModel{" +
                "k=" + k +
                ", meanVectorLength=" + (meanVector != null ? meanVector.length : 0) +
                ", basisRows=" + (basisMatrix != null ? basisMatrix.length : 0) +
                ", basisCols=" + (basisMatrix != null && basisMatrix.length > 0 ? basisMatrix[0].length : 0) +
                '}';
    }
}