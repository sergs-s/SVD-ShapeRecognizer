package svd.recognizer.math;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * Реализация интерфейса SvdEngine на базе Apache Commons Math.
 *
 * Причины выбора этой реализации:
 * 1. SingularValueDecomposition предоставляет getSingularValues(), getU() и getV()
 * 2. Сингулярные значения возвращаются в невозрастающем порядке
 * 3. Текущей задаче распознавания фигур достаточно первых σ-значений
 * 4. Позже эту реализацию можно заменить на EjmlSvdEngine без изменения верхнего уровня
 *
 * @author ssv
 */
public class CommonsMathSvdEngine implements SvdEngine {
    @Override
    public SvdResult decompose(double[][] matrix) {
        RealMatrix a = new Array2DRowRealMatrix(matrix);
        SingularValueDecomposition svd = new SingularValueDecomposition(a);
        return new SvdResult(
                svd.getSingularValues(),
                svd.getU().getData(),
                svd.getV().getData()
        );
    }
}