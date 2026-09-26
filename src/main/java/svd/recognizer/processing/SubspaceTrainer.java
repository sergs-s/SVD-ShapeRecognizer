package svd.recognizer.processing;

import svd.recognizer.math.SvdEngine;
import svd.recognizer.math.SvdResult;
import svd.recognizer.model.ShapeClass;
import svd.recognizer.model.SubspaceModel;
import svd.recognizer.model.Template;
import svd.recognizer.model.TemplateStore;

import java.util.List;

/**
 * Сервис построения подпространства класса по обучающим векторам.
 *
 * Алгоритм (согласно ТЗ "переход на подпространства", раздел 2.3):
 * 1. Получить векторы класса одинаковой длины d (для фигур — эталоны
 *    Template.getNormalizedMatrix() 64×64, развёрнутые в d = 4096 через
 *    ImageVectorizer.toVector; для лиц ORL — снимки 92×112, d = 10 304).
 * 2. Вычислить средний вектор класса — покомпонентное среднее всех векторов.
 * 3. Центрировать образцы (вычесть средний из каждого) и собрать матрицу X
 *    размера d x n, где столбцы — центрированные образцы.
 * 4. Выполнить SVD матрицы X через существующий SvdEngine.
 * 5. Взять первые k столбцов матрицы U как базис подпространства
 *    (k ≤ n − 1; k задаётся явно или по доле энергии η).
 *
 * @author ssv
 */
public class SubspaceTrainer {

    /** Размер эталона фигуры (64×64); для лиц не используется. */
    private static final int IMAGE_SIZE = 64;

    private final SvdEngine svdEngine;

    public SubspaceTrainer(SvdEngine svdEngine) {
        this.svdEngine = svdEngine;
    }

    /**
     * Строит подпространство класса по хранилищу эталонов.
     *
     * @param store хранилище эталонов класса (непустое, у всех эталонов
     *              должно быть заполнено normalizedMatrix)
     * @param k желаемая размерность подпространства (значение по умолчанию —
     *          {@link svd.recognizer.storage.SettingsStore#loadSubspaceK()})
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

        // Векторизация всех эталонов
        double[][] vectors = vectorizeTemplates(templates, store.getShapeClass());

        System.out.println("SubspaceTrainer: выполнение SVD для " + store.getShapeClass());
        SubspaceModel model = train(vectors, k);
        System.out.println("SubspaceTrainer: доступно " + n +
                " сингулярных векторов, используем " + model.getK());
        System.out.println("SubspaceTrainer: обучение завершено: " + model);

        return model;
    }

    /**
     * Строит подпространство по набору векторов произвольной (но одинаковой)
     * длины d: средний вектор, центрированная матрица X (d x n), SVD, базис из
     * первых min(k, n − 1) столбцов U.
     *
     * @param vectors обучающие векторы класса (n штук, n ≥ 2)
     * @param k       желаемая размерность подпространства
     * @return обученная модель подпространства
     * @throws IllegalArgumentException если векторов меньше двух или их длины различаются
     */
    public SubspaceModel train(double[][] vectors, int k) {
        double[] mean = computeMeanVector(vectors);
        SvdResult svd = svdEngine.decompose(buildCenteredMatrix(vectors, mean));
        double[][] basis = extractBasis(svd.getU(), k);
        return new SubspaceModel(mean, basis, basis[0].length);
    }

    /**
     * Строит подпространство, выбирая размерность по доле энергии:
     * k_c = min{ k ≤ n − 1 : Σ_{i≤k} σᵢ² / Σ_{i≤n−1} σᵢ² ≥ η }.
     *
     * @param vectors обучающие векторы класса (n штук, n ≥ 2)
     * @param eta     требуемая доля энергии, 0 &lt; η ≤ 1
     * @return обученная модель подпространства (k_c = getK())
     * @throws IllegalArgumentException если векторов меньше двух, их длины
     *         различаются или η вне (0, 1]
     */
    public SubspaceModel trainByEnergy(double[][] vectors, double eta) {
        if (!(eta > 0.0 && eta <= 1.0)) {
            throw new IllegalArgumentException("Доля энергии η должна быть в (0, 1], получено " + eta);
        }
        double[] mean = computeMeanVector(vectors);
        SvdResult svd = svdEngine.decompose(buildCenteredMatrix(vectors, mean));
        int k = energyK(svd.getSingularValues(), vectors.length - 1, eta);
        double[][] basis = extractBasis(svd.getU(), k);
        return new SubspaceModel(mean, basis, basis[0].length);
    }

    /**
     * Наименьшее k ≤ maxK, при котором доля энергии первых k сингулярных
     * значений не меньше η. При нулевой энергии возвращает 1.
     *
     * @param sigma сингулярные значения в невозрастающем порядке
     * @param maxK  верхняя граница k (ранг центрированной матрицы, n − 1)
     * @param eta   требуемая доля энергии
     * @return выбранная размерность k (1 ≤ k ≤ maxK)
     */
    static int energyK(double[] sigma, int maxK, double eta) {
        int limit = Math.min(maxK, sigma.length);
        double total = 0.0;
        for (int i = 0; i < limit; i++) {
            total += sigma[i] * sigma[i];
        }
        if (total <= 0.0) {
            return 1;
        }
        double acc = 0.0;
        for (int i = 0; i < limit; i++) {
            acc += sigma[i] * sigma[i];
            if (acc / total >= eta) {
                return i + 1;
            }
        }
        return limit;
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
     * Вычисляет средний вектор (покомпонентное среднее всех векторов).
     *
     * @param vectors обучающие векторы одинаковой длины d, не меньше двух
     * @return средний вектор длины d
     * @throws IllegalArgumentException если векторов меньше двух или их длины различаются
     */
    private double[] computeMeanVector(double[][] vectors) {
        if (vectors == null || vectors.length < 2) {
            throw new IllegalArgumentException(
                    "Для построения подпространства нужно минимум 2 вектора, есть " +
                            (vectors == null ? 0 : vectors.length));
        }
        int dim = vectors[0].length;
        int n = vectors.length;
        double[] mean = new double[dim];
        for (double[] vector : vectors) {
            if (vector.length != dim) {
                throw new IllegalArgumentException(
                        "Векторы разной длины: " + dim + " и " + vector.length);
            }
            for (int j = 0; j < dim; j++) {
                mean[j] += vector[j];
            }
        }
        for (int j = 0; j < dim; j++) {
            mean[j] /= n;
        }
        return mean;
    }

    /**
     * Строит центрированную матрицу X размера d x n.
     * Каждый столбец — центрированный обучающий вектор.
     *
     * @param vectors обучающие векторы
     * @param mean    средний вектор
     * @return матрица X размера d x n
     */
    private double[][] buildCenteredMatrix(double[][] vectors, double[] mean) {
        int dim = mean.length;
        int n = vectors.length;
        double[][] x = new double[dim][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                x[j][i] = vectors[i][j] - mean[j];
            }
        }
        return x;
    }

    /**
     * Извлекает первые k столбцов матрицы U как базис подпространства.
     *
     * Ранг центрированной матрицы из n векторов не больше n − 1 (столбцы в
     * сумме дают ноль), поэтому осмысленных сингулярных векторов не больше
     * n − 1: actualK = min(k, n − 1).
     *
     * @param u матрица левых сингулярных векторов (размер d x n)
     * @param k желаемая размерность подпространства
     * @return матрица базиса размера d x actualK
     * @throws IllegalArgumentException если вектор один (ранг равен нулю)
     */
    private double[][] extractBasis(double[][] u, int k) {
        int availableVectors = u[0].length;
        if (availableVectors < 2) {
            throw new IllegalArgumentException(
                    "Для построения подпространства нужно минимум 2 эталона, есть " + availableVectors);
        }
        int actualK = Math.min(k, availableVectors - 1);

        int dim = u.length;
        double[][] basis = new double[dim][actualK];
        for (int i = 0; i < dim; i++) {
            System.arraycopy(u[i], 0, basis[i], 0, actualK);
        }
        return basis;
    }
}
