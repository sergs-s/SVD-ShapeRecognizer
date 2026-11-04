package svd.recognizer.faces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Решатель ℓ1 для ESRC (Deng, Hu, Guo, TPAMI 2012): min ‖x‖₁ при ‖y − Φx‖₂ ≤ ε – гомотопия по λ (LARS с удалением переменных) от x = 0 до первой
 * точки пути с невязкой ε; если весь путь (до λ = 0) не достигает ε – решение наименьших квадратов на конце пути. Вход – предвычисленные
 * G = ΦᵀΦ, c = Φᵀy и ‖y‖²; активная матрица – разложение Холецкого с обновлениями. Столбец, линейно зависимый от активных (остаток &lt; RANK_TOL),
 * блокируется до следующего удаления.
 *
 * @author ssv
 */
final class EsrcSolver {

    private EsrcSolver() {
    }

    static final double RANK_TOL = 1e-10;
    private static final double TINY = 1e-13;

    /** x – решение; steps – шагов пути; rr – квадрат невязки; capped – оборвано по пределу шагов. */
    record Result(double[] x, int steps, double rr, boolean capped) {}

    static Result solve(double[][] g, double[] c0, double yy, double eps, int maxSteps) {
        int p = c0.length;
        double[] x = new double[p];
        double rr = yy;
        double target = eps * eps;
        if (rr <= target || p == 0) return new Result(x, 0, rr, false);
        double[] c = c0.clone();
        int first = -1;
        double lambda = 0;
        for (int j = 0; j < p; j++) {
            if (Math.abs(c[j]) > lambda) {
                lambda = Math.abs(c[j]);
                first = j;
            }
        }
        if (first < 0) return new Result(x, 0, rr, false);
        List<Integer> act = new ArrayList<>();
        boolean[] in = new boolean[p];
        boolean[] blocked = new boolean[p];
        double[] sign = new double[p];
        List<double[]> chol = new ArrayList<>();
        add(g, first, act, chol);
        in[first] = true;
        sign[first] = Math.signum(c[first]);
        int dropped = -1;
        double[] a = new double[p];
        for (int step = 1; step <= maxSteps; step++) {
            int k = act.size();
            double[] s = new double[k];
            int[] ai = new int[k];
            for (int i = 0; i < k; i++) {
                ai[i] = act.get(i);
                s[i] = sign[ai[i]];
            }
            double[] w = cholSolve(chol, s);
            double q = 0;
            for (int i = 0; i < k; i++) q += s[i] * w[i];
            for (int j = 0; j < p; j++) {
                double[] gj = g[j];
                double v = 0;
                for (int i = 0; i < k; i++) v += gj[ai[i]] * w[i];
                a[j] = v;
            }
            double gamma = lambda;
            int enter = -1;
            int leave = -1;
            for (int j = 0; j < p; j++) {
                if (in[j] || blocked[j]) continue;
                double d1 = 1 - a[j];
                double d2 = 1 + a[j];
                // только что вышедший столбец с прежним знаком сразу не возвращается; с обратным знаком – может в этом же отрезке
                if (d1 > TINY && !(j == dropped && sign[j] > 0)) {
                    double t = (lambda - c[j]) / d1;
                    if (t > TINY && t < gamma) {
                        gamma = t;
                        enter = j;
                    }
                }
                if (d2 > TINY && !(j == dropped && sign[j] < 0)) {
                    double t = (lambda + c[j]) / d2;
                    if (t > TINY && t < gamma) {
                        gamma = t;
                        enter = j;
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                if (w[i] == 0) continue;
                double t = -x[ai[i]] / w[i];
                if (t > TINY && t < gamma) {
                    gamma = t;
                    leave = i;
                    enter = -1;
                }
            }
            // rr(γ) = rr − 2γλq + γ²q; первая точка с rr(γ) = ε² на отрезке – конец пути
            double after = rr - 2 * gamma * lambda * q + gamma * gamma * q;
            boolean finish = after <= target;
            if (finish) {
                double disc = lambda * lambda - (rr - target) / q;
                gamma = Math.min(gamma, lambda - Math.sqrt(Math.max(disc, 0)));
                after = target;
            }
            for (int i = 0; i < k; i++) x[ai[i]] += gamma * w[i];
            for (int j = 0; j < p; j++) c[j] -= gamma * a[j];
            lambda -= gamma;
            rr = after;
            if (finish || lambda <= TINY) return new Result(x, step, rr, false);
            dropped = -1;
            if (leave >= 0) {
                int j = ai[leave];
                x[j] = 0;
                in[j] = false;
                dropped = j;
                remove(leave, act, chol);
                Arrays.fill(blocked, false);
                if (act.isEmpty()) {
                    // все вышли: следующий – наибольшая корреляция, кроме только что вышедшего
                    int b = -1;
                    double m = 0;
                    for (int t = 0; t < p; t++) {
                        if (t != j && Math.abs(c[t]) > m) {
                            m = Math.abs(c[t]);
                            b = t;
                        }
                    }
                    if (b < 0) return new Result(x, step, rr, false);
                    add(g, b, act, chol);
                    in[b] = true;
                    sign[b] = Math.signum(c[b]);
                    lambda = m;
                }
            } else if (enter >= 0) {
                if (add(g, enter, act, chol)) {
                    in[enter] = true;
                    sign[enter] = Math.signum(c[enter]);
                } else {
                    blocked[enter] = true;
                }
            }
        }
        return new Result(x, maxSteps, rr, true);
    }

    /** Добавить столбец j в разложение; false – столбец линейно зависит от активных. */
    private static boolean add(double[][] g, int j, List<Integer> act, List<double[]> chol) {
        int k = act.size();
        double[] row = new double[k + 1];
        double sum = 0;
        for (int i = 0; i < k; i++) {
            double[] li = chol.get(i);
            double v = g[act.get(i)][j];
            for (int t = 0; t < i; t++) v -= li[t] * row[t];
            v /= li[i];
            row[i] = v;
            sum += v * v;
        }
        double d = g[j][j] - sum;
        if (d <= RANK_TOL) return false;
        row[k] = Math.sqrt(d);
        chol.add(row);
        act.add(j);
        return true;
    }

    /** Убрать столбец с позиции pos (вращения Гивенса). */
    private static void remove(int pos, List<Integer> act, List<double[]> chol) {
        act.remove(pos);
        chol.remove(pos);
        int k = act.size();
        for (int i = pos; i < k; i++) {
            double[] ri = chol.get(i);
            // строка i (была i + 1) длиной i + 2: обнулить элемент [i][i + 1]
            double p = ri[i];
            double q = ri[i + 1];
            double h = Math.hypot(p, q);
            double cs = p / h;
            double sn = q / h;
            for (int t = i + 1; t < k; t++) {
                double[] rt = chol.get(t);
                double u = rt[i];
                double v = rt[i + 1];
                rt[i] = cs * u + sn * v;
                rt[i + 1] = -sn * u + cs * v;
            }
            double[] nr = Arrays.copyOf(ri, i + 1);
            nr[i] = h;
            chol.set(i, nr);
        }
    }

    private static double[] cholSolve(List<double[]> chol, double[] s) {
        int k = s.length;
        double[] z = new double[k];
        for (int i = 0; i < k; i++) {
            double[] li = chol.get(i);
            double v = s[i];
            for (int t = 0; t < i; t++) v -= li[t] * z[t];
            z[i] = v / li[i];
        }
        for (int i = k - 1; i >= 0; i--) {
            double v = z[i];
            for (int t = i + 1; t < k; t++) v -= chol.get(t)[i] * z[t];
            z[i] = v / chol.get(i)[i];
        }
        return z;
    }
}
