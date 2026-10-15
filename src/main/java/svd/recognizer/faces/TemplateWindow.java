package svd.recognizer.faces;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Шаблон человека как окно последних M сессий (PLAN.md п. 5.5, очередь (4), моделирование обновления шаблона; задание №4 от
 * 15.10.2026): сессия – дата и ключи её снимков; при добавлении сверх M вытесняется самая старая по порядку добавления.
 * M = {@link #UNBOUNDED} – без окна (копятся все сессии). Порядок добавления – порядок шагов моделирования; чужая сессия
 * (ошибочное обновление) вытесняет так же, как своя.
 *
 * @author ssv
 */
final class TemplateWindow {

    /** Без окна. */
    static final int UNBOUNDED = Integer.MAX_VALUE;

    /** Сессия: дата (для интервала «дата пробы – самая старая дата шаблона»), чья (человек, чьи это снимки), ключи снимков. */
    record Session(String date, String owner, List<String> keys) {
        Session {
            keys = List.copyOf(keys);
        }
    }

    private final int m;
    private final Deque<Session> sessions = new ArrayDeque<>();

    TemplateWindow(int m) {
        if (m < 1) throw new IllegalArgumentException("TemplateWindow: M ≥ 1, задано " + m);
        this.m = m;
    }

    /** Копия (то же M, те же сессии). */
    TemplateWindow copy() {
        TemplateWindow w = new TemplateWindow(m);
        w.sessions.addAll(sessions);
        return w;
    }

    /** Добавить сессию; вытесненная (самая старая сверх M) – результат, иначе null. */
    Session add(Session s) {
        sessions.addLast(s);
        return sessions.size() > m ? sessions.removeFirst() : null;
    }

    int size() {
        return sessions.size();
    }

    /** Сессии от старой к новой. */
    List<Session> sessions() {
        return new ArrayList<>(sessions);
    }

    /** Ключи снимков текущего шаблона (по сессиям, от старой к новой). */
    List<String> keys() {
        List<String> out = new ArrayList<>();
        for (Session s : sessions) out.addAll(s.keys());
        return out;
    }

    /** Самая старая дата среди сессий (строки дат FERET yymmdd сравниваются как строки); пусто – null. */
    String oldestDate() {
        String d = null;
        for (Session s : sessions) if (d == null || s.date().compareTo(d) < 0) d = s.date();
        return d;
    }

    /** Есть ли в шаблоне сессии чужих (owner ≠ person). */
    int foreign(String person) {
        int n = 0;
        for (Session s : sessions) if (!s.owner().equals(person)) n++;
        return n;
    }
}
