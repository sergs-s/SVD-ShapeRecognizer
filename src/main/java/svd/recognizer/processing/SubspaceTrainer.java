package svd.recognizer.processing;

import svd.recognizer.math.SvdEngine;
import svd.recognizer.math.SvdResult;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

import java.util.List;

/**
 * Сервис построения подпространства класса по эталонам.
 *
 * Алгоритм (согласно ТЗ "переход на подпространства", раздел 2.3):
 * 1. Для каждого эталона класса взять Template.getNormalizedMatrix()
 *    и развернуть в вектор длины 4096 через ImageVectorizer.toVector.
 * 2. Вычислить средний вектор класса — покомпонентное среднее всех векторов.
 * 3. Центрировать образцы (вычесть средний из каждого) и собрать матрицу X
 *    размера 4096 x n, где столбцы — центрированные образцы.
 * 4. Выполнить SVD матрицы X через существующий SvdEngine.
 * 5. Взять первые k столбцов матрицы U как базис подпространства.
 *
 * @author ssv
 */
public class SubspaceTrainer {

    public static final int DEFAULT_K = 4;
    private static final int VECTOR_LENGTH = 64 * 64; // 4096
    private static final int IMAGE_SIZE = 64;

    private final SvdEngine svdEngine;

    public SubspaceTrainer(SvdEngine svdEngine) {
        this.svdEngine = svdEngine;
    }

    /**
     * Строит подпространство класса с размерностью по умолчанию (DEFAULT_K = 4).
     */
    public SubspaceModel train(TemplateStore store) {
        return train(store, DEFAULT_K);
    }

    /**
     * Строит подпространство класса по хранилищу эталонов.
     *
     * @param store хранилище эталонов класса (непустое, у всех эталонов
     *              должно быть заполнено normalizedMatrix)
     * @param k желаемая размерность подпространства
     * @return обученная модель подпространства класса
     * @throws IllegalArgumentException если хранилище пусто или у эталона
     *         нет normalizedMatrix
     */
    public SubspaceModel train(TemplateStore store, int k) {
        validateStore(store);

        List<Template> templates = store.getTemplates();
        int n = templates.size();

        System.out.println("SubspaceTrainer: обучение класса " + store.getShapeClass() +
                " с " + n + " эталонами, k=" + k);

        // Шаг 1: Векторизация всех эталонов
        double[][] vectors = vectorizeTemplates(templates, store.getShapeClass());

        // Шаг 2: Вычисление среднего вектора
        double[] mean = computeMeanVector(vectors, n);

        // Шаг 3: Центрирование и формирование матрицы X (4096 x n)
        double[][] x = buildCenteredMatrix(vectors, mean, n);

        // Шаг 4: Сингулярное разложение матрицы X
        System.out.println("SubspaceTrainer: выполнение SVD для " + store.getShapeClass());
        SvdResult svd = svdEngine.decompose(x);
        double[][] u = svd.getU();

        // Шаг 5: Отбор первых k столбцов U как базис
        double[][] basis = extractBasis(u, k);

        // Создание и возврат модели
        SubspaceModel model = new SubspaceModel(mean, basis, basis[0].length);
        System.out.println("SubspaceTrainer: обучение завершено: " + model);

        return model;
    }

    /**
     * Проверяет, что хранилище не пустое.
     *
     * @param store хранилище эталонов
     * @throws IllegalArgumentException если хранилище пусто
     */
    private void validateStore(TemplateStore store) {
        if (store == null) {
            throw new IllegalArgumentException("Хранилище не может быть null");
        }
        if (store.getTemplates().isEmpty()) {
            throw new IllegalArgumentException(
                    "Класс " + store.getShapeClass() + ": нет эталонов для обучения"
            );
        }
    }

    /**
     * Векторизует все эталоны класса.
     *
     * @param templates  список эталонов
     * @param shapeClass класс фигуры (для сообщений об ошибках)
     * @return массив векторов (каждый длины 4096)
     * @throws IllegalArgumentException если у эталона нет normalizedMatrix
     */
    private double[][] vectorizeTemplates(List<Template> templates, ShapeClass shapeClass) {
        int n = templates.size();
        double[][] vectors = new double[n][];

        for (int i = 0; i < n; i++) {
            double[][] matrix = templates.get(i).getNormalizedMatrix();
            validateTemplateMatrix(matrix, i, shapeClass);
            vectors[i] = ImageVectorizer.toVector(matrix);
        }

        return vectors;
    }

    /**
     * Проверяет, что у эталона есть нормализованная матрица корректного размера.
     *
     * @param matrix     матрица эталона
     * @param index      индекс эталона в списке
     * @param shapeClass класс фигуры (для сообщений об ошибках)
     * @throws IllegalArgumentException если матрица отсутствует или имеет некорректный размер
     */
    private void validateTemplateMatrix(double[][] matrix, int index, ShapeClass shapeClass) {
        if (matrix == null) {
            throw new IllegalArgumentException(
                    "Эталон #" + index + " класса " + shapeClass +
                            " не содержит normalizedMatrix. " +
                            "Удалите и перезагрузите эталоны этого класса."
            );
        }
        if (matrix.length != IMAGE_SIZE || matrix[0].length != IMAGE_SIZE) {
            throw new IllegalArgumentException(
                    "Эталон #" + index + " класса " + shapeClass +
                            " имеет некорректный размер матрицы: " + matrix.length + "x" + matrix[0].length
            );
        }
    }

    /**
     * Вычисляет средний вектор класса (покомпонентное среднее всех векторов).
     *
     * @param vectors массив векторов эталонов
     * @param n       количество векторов
     * @return средний вектор длины 4096
     */
    private double[] computeMeanVector(double[][] vectors, int n) {
        double[] mean = new double[VECTOR_LENGTH];
        for (double[] vector : vectors) {
            for (int j = 0; j < VECTOR_LENGTH; j++) {
                mean[j] += vector[j];
            }
        }
        for (int j = 0; j < VECTOR_LENGTH; j++) {
            mean[j] /= n;
        }
        return mean;
    }

    /**
     * Строит центрированную матрицу X размера 4096 x n.
     * Каждый столбец — центрированный вектор эталона.
     *
     * @param vectors массив векторов эталонов
     * @param mean    средний вектор класса
     * @param n       количество векторов
     * @return матрица X размера 4096 x n
     */
    private double[][] buildCenteredMatrix(double[][] vectors, double[] mean, int n) {
        double[][] x = new double[VECTOR_LENGTH][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < VECTOR_LENGTH; j++) {
                x[j][i] = vectors[i][j] - mean[j];
            }
        }
        return x;
    }

    /**
     * Извлекает первые k столбцов матрицы U как базис подпространства.
     *
     * @param u матрица левых сингулярных векторов (размер 4096 x n)
     * @param k желаемая размерность подпространства
     * @return матрица базиса размера 4096 x actualK
     */
    private double[][] extractBasis(double[][] u, int k) {
        int availableVectors = u[0].length;
        int actualK = Math.min(k, availableVectors);

        System.out.println("SubspaceTrainer: доступно " + availableVectors +
                " сингулярных векторов, используем " + actualK);

        double[][] basis = new double[VECTOR_LENGTH][actualK];
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            System.arraycopy(u[i], 0, basis[i], 0, actualK);
        }
        return basis;
    }
}