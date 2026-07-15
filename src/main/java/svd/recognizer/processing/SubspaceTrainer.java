package svd.recognizer.processing;

import svd.recognizer.math.SvdEngine;
import svd.recognizer.math.SvdResult;
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

    private final SvdEngine svdEngine;

    public SubspaceTrainer(SvdEngine svdEngine) {
        this.svdEngine = svdEngine;
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
        // Проверка: хранилище не должно быть пустым
        List<Template> templates = store.getTemplates();
        if (templates.isEmpty()) {
            throw new IllegalArgumentException(
                    "Класс " + store.getShapeClass() + ": нет эталонов для обучения"
            );
        }

        int n = templates.size();
        System.out.println("SubspaceTrainer: обучение класса " + store.getShapeClass() +
                " с " + n + " эталонами, k=" + k);

        // Шаг 1: Векторизация всех эталонов
        double[][] vectors = new double[n][];
        for (int i = 0; i < n; i++) {
            double[][] matrix = templates.get(i).getNormalizedMatrix();
            if (matrix == null) {
                throw new IllegalArgumentException(
                        "Эталон #" + i + " класса " + store.getShapeClass() +
                                " не содержит normalizedMatrix. " +
                                "Удалите и перезагрузите эталоны этого класса."
                );
            }

            // Проверка размера матрицы
            if (matrix.length != 64 || matrix[0].length != 64) {
                throw new IllegalArgumentException(
                        "Эталон #" + i + " класса " + store.getShapeClass() +
                                " имеет некорректный размер матрицы: " + matrix.length + "x" + matrix[0].length
                );
            }

            vectors[i] = ImageVectorizer.toVector(matrix);
        }

        // Шаг 2: Вычисление среднего вектора
        double[] mean = new double[VECTOR_LENGTH];
        for (double[] vector : vectors) {
            for (int j = 0; j < VECTOR_LENGTH; j++) {
                mean[j] += vector[j];
            }
        }
        for (int j = 0; j < VECTOR_LENGTH; j++) {
            mean[j] /= n;
        }

        // Шаг 3: Центрирование и формирование матрицы X (4096 x n)
        // В SvdEngine.decompose(matrix) матрица имеет размер [rows][cols],
        // где rows = количество строк, cols = количество столбцов.
        // Для X мы хотим 4096 строк и n столбцов.
        double[][] x = new double[VECTOR_LENGTH][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < VECTOR_LENGTH; j++) {
                x[j][i] = vectors[i][j] - mean[j];
            }
        }

        // Шаг 4: Сингулярное разложение матрицы X
        System.out.println("SubspaceTrainer: выполнение SVD для " + store.getShapeClass());
        SvdResult svd = svdEngine.decompose(x);
        double[][] u = svd.getU();

        // Шаг 5: Отбор первых k столбцов U как базис
        // U имеет размер [VECTOR_LENGTH][n] (в экономичной форме)
        int actualK = Math.min(k, u[0].length);
        System.out.println("SubspaceTrainer: доступно " + u[0].length +
                " сингулярных векторов, используем " + actualK);

        double[][] basis = new double[VECTOR_LENGTH][actualK];
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            System.arraycopy(u[i], 0, basis[i], 0, actualK);
        }

        // Создание и возврат модели
        SubspaceModel model = new SubspaceModel(mean, basis, actualK);
        System.out.println("SubspaceTrainer: обучение завершено: " + model);

        return model;
    }

    /**
     * Строит подпространство класса с размерностью по умолчанию (DEFAULT_K = 4).
     *
     * @param store хранилище эталонов класса
     * @return обученная модель подпространства класса
     * @throws IllegalArgumentException если хранилище пусто или у эталона
     *         нет normalizedMatrix
     */
    public SubspaceModel train(TemplateStore store) {
        return train(store, DEFAULT_K);
    }

    /**
     * Проверяет, можно ли обучить подпространство для данного хранилища.
     *
     * @param store хранилище эталонов
     * @return true, если хранилище содержит как минимум один эталон с
     *         заполненным normalizedMatrix
     */
    public boolean isTrainable(TemplateStore store) {
        if (store == null || store.getTemplates().isEmpty()) {
            return false;
        }

        for (Template template : store.getTemplates()) {
            if (template.getNormalizedMatrix() == null) {
                return false;
            }
        }
        return true;
    }

    /**
     * Проверяет, можно ли обучить подпространство для всех классов.
     *
     * @param stores карта хранилищ всех классов
     * @return true, если все хранилища содержат как минимум один эталон с
     *         заполненным normalizedMatrix
     */
    public boolean isTrainableAll(java.util.Map<svd.recognizer.model.ShapeClass, TemplateStore> stores) {
        if (stores == null) {
            return false;
        }

        for (svd.recognizer.model.ShapeClass sc : svd.recognizer.model.ShapeClass.values()) {
            TemplateStore store = stores.get(sc);
            if (!isTrainable(store)) {
                return false;
            }
        }
        return true;
    }
}