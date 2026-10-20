package svd.recognizer.faces;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.HashMap;
import java.util.List;
import org.junit.Test;

/**
 * Неопределённый порог и пустая реплика вне основного конвейера (решение Хозяина 20.10.2026, №9 Жорика): меры при пороге NaN –
 * NaN (не 100 % и не 0), пустая реплика – NaN, пометка «мало людей» – только у определённой точки.
 *
 * @author ssv
 */
public class UndefinedThresholdTest {

    /** 1. FeiBootstrap.metrics: пороговые люди без веса – ошибки и FAR NaN; все веса 1 – числа. */
    @Test
    public void feiBootstrapMetrics() {
        FeiBootstrap.Cfg c = new FeiBootstrap.Cfg(true);
        c.ownP = new int[] {0, 1};
        c.ownS = new double[] {0.2, 0.9};
        c.ownOk = new boolean[] {true, false};
        c.thrP = new int[] {2, 3};
        c.thrS = new double[] {0.3, 0.6};
        c.ctrlP = new int[] {4, 5};
        c.ctrlS = new double[] {0.25, 0.8};
        c.ctrlPeople = new int[] {4, 5};
        double[] m = FeiBootstrap.metrics(List.of(c), new int[] {1, 1, 0, 0, 1, 1});
        for (double x : m) assertTrue(Double.isNaN(x));
        double[] u = FeiBootstrap.metrics(List.of(c), FeiBootstrap.unit(6));
        // Все конфигурации – анфас: ошибка по повёрнутым – 0/0 = NaN по построению.
        for (int k : new int[] {FeiBootstrap.ERR1, FeiBootstrap.ERR5, FeiBootstrap.ERR5_FRONTAL, FeiBootstrap.FAR1, FeiBootstrap.FAR5, FeiBootstrap.FARP1})
            assertFalse(Double.isNaN(u[k]));
    }

    /** 2. XBaseEqCompare.ownErrW: «чужие» П2 без веса – NaN, не 100 %. */
    @Test
    public void eqCompareOwnErr() {
        XBaseA0.Pair p = new XBaseA0.Pair();
        p.cvR = new double[] {0.3, 0.6};
        p.cvP = new int[] {0, 1};
        assertTrue(Double.isNaN(XBaseEqCompare.ownErrW(p, new HashMap<>(), new int[] {0, 0})));
    }

    /** 3. ProkaevOther.feiErr: пороговые без веса – {NaN, NaN}, не {100, 0}. */
    @Test
    public void prokaevFeiErr() {
        ProkaevOther.FCfg c = new ProkaevOther.FCfg();
        c.ownP = new int[] {0};
        c.ownR = new double[] {0.2};
        c.ownOk = new boolean[] {true};
        c.thrP = new int[] {1, 2};
        c.thrR = new double[] {0.3, 0.6};
        c.ctrlP = new int[] {3};
        c.ctrlR = new double[] {0.1};
        double[] e = ProkaevOther.feiErr(List.of(c), new int[] {1, 0, 0, 1}, 0.01);
        assertTrue(Double.isNaN(e[0]) && Double.isNaN(e[1]));
        double[] u = ProkaevOther.feiErr(List.of(c), FeiBootstrap.unit(4), 0.01);
        assertEquals(0.0, u[0], 0.0);
        assertEquals(100.0, u[1], 0.0);
    }

    /** 4. XBaseNewcomer: пустая реплика доли той же сессии – NaN, не 0. */
    @Test
    public void newcomerEmptyReplicate() {
        assertTrue(Double.isNaN(XBaseNewcomer.sameExcess(new double[] {0, 0, 0})));
        assertEquals(25.0, XBaseNewcomer.sameExcess(new double[] {4, 2, 1}), 1e-12);
    }

    /** 5. XBaseNewcomer.metrics: «чужие» без веса – меры П2 и П3 NaN; контроль без веса – и мера при пороге по контролю. */
    @Test
    public void newcomerMetrics() {
        XBaseNewcomer.SD sd = new XBaseNewcomer.SD();
        sd.own.add(new XBaseNewcomer.OwnA(0, true, 0.2, true, true, 0, true, "a"));
        sd.own.add(new XBaseNewcomer.OwnA(1, false, 0.4, true, true, 1, false, "b"));
        sd.imp.add(new XBaseNewcomer.ImpA(2, 0.3, false, true));
        sd.imp.add(new XBaseNewcomer.ImpA(3, 0.6, false, false));
        sd.ctrl.add(new XBaseNewcomer.CtrlA(4, 0.5, 0, true, false, 0, 0, "c"));
        XBaseNewcomer.Cfg c = new XBaseNewcomer.Cfg(sd);
        double[] m = XBaseNewcomer.metrics(List.of(c), 0, new int[] {1, 1, 0, 0, 0});
        for (double x : m) assertTrue(Double.isNaN(x));
        double[] u = XBaseNewcomer.metrics(List.of(c), 0, FeiBootstrap.unit(5));
        for (double x : u) assertFalse(Double.isNaN(x));
    }

    /** Оформление: при точке NaN пометки нет, при определённой точке – есть. */
    @Test
    public void noteOnlyForDefinedPoint() {
        double[] reps = new double[100];
        for (int i = 0; i < 100; i++) reps[i] = i < 50 ? i : Double.NaN;
        assertFalse(FeiBootstrap.ci(Double.NaN, reps, 1).contains("мало людей"));
        assertTrue(FeiBootstrap.ci(1.0, reps, 1).contains("мало людей"));
        assertFalse(FeiBootstrap.far(Double.NaN, reps, new int[] {0, 10}).contains("мало людей"));
        assertTrue(FeiBootstrap.far(1.0, reps, new int[] {0, 10}).contains("мало людей"));
    }
}
