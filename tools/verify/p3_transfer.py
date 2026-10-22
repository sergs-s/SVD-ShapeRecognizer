"""Переносимость порога П3 (Жорик №9 от 22.10.2026): по готовым файлам оценок, без кода проекта.

Для строк таблицы С1: П3 FAR 1 % – по «чужим» галереи (cv, measures.p3_threshold, как в коде), фактический FAR контроля при нём;
порог по контрольным чужим при FAR 1 % и 0,1 % (Нейман – Пирсон, measures.np_threshold) и ошибка своих при нём; распределения
оценок d₁/d₂ своих, «чужих» галереи и контрольных чужих (квантили), худший свой (наибольшая оценка) и самый похожий чужой
(наименьшая); сколько людей и попыток у «чужих» галереи и у контроля. Оценка «меньше – ближе», порог – «принят при r ≤ θ».

python3 -I p3_transfer.py <вывод>
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_check  # noqa: E402
import compare  # noqa: E402
from compare import FERET, read_scores  # noqa: E402
from measures import Row, evaluate, np_threshold, p3_threshold, ratio  # noqa: E402

DATA_MAIN = os.environ.get("VERIFY_DATA_MAIN", "")
FAR_TOL = 1.1  # допуск отчётов: точка FAR контроля при FAR 1 % не выше 1,1 %


def plain_rows(blocks, method, score):
    """Попытки одной модели (без X / X+mirror): {s: {own|ctrl|cv: [(r, wrong, person, 0)]}}."""
    out = {}
    for tag, b in enumerate(blocks):
        for d in b:
            if d["method"] != method:
                continue
            role = "cv" if d["role"].startswith("cv") else d["role"]
            r = score(d)
            out.setdefault(tag * 1000 + int(d["s"]), {"own": [], "ctrl": [], "cv": []})[role].append(
                (r, d["best_id"] != d["true_id"], d["person"], 0.0))
    return out


def mk_row(conf):
    persons = sorted({a[2] for c in conf.values() for L in c.values() for a in L})
    return Row(conf, {p: i for i, p in enumerate(persons)})


def q(v, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    return " / ".join("%.4f" % float(np.quantile(v, p, method="inverted_cdf")) for p in ps) if len(v) else "–"


def analyse(label, row, ref=None):
    ones = np.ones(row.np_, dtype=np.int64)
    res = evaluate(row, ones, fars=(0.01,))
    p3 = res[("P3", 0.01)]
    sep = res[("sep", 0.01)]
    far_p3 = 100 * p3["cacc"] / p3["cn"]
    err_p3 = 100 * p3["err"]["all"] / p3["nown"]["all"]
    own_r = np.concatenate([row.c[s]["own"][0] for s in row.s])
    own_wr = np.concatenate([row.c[s]["own"][1] for s in row.s]) != 0
    ct_r = np.concatenate([row.c[s]["ctrl"][0] for s in row.s])
    cv_r = np.concatenate([row.c[s]["cv"][0] for s in row.s])
    cv_p = {(s, int(p)) for s in row.s for p in row.c[s]["cv"][2]}
    ct_p = {(s, int(p)) for s in row.s for p in row.c[s]["ctrl"][2]}
    L = ["### " + label, ""]
    L.append("| мера | значение |")
    L.append("|---|---|")
    L.append("| П3 FAR 1 %%: ошибка своих / факт. FAR контроля | %.1f %% / %.2f %% %s |" % (
        err_p3, far_p3, ("(допуск ≤ %.1f %% – %s)" % (FAR_TOL, "да" if far_p3 <= FAR_TOL else "нет"))))
    L.append("| разд. FAR 1 %% (порог по контролю): ошибка своих / факт. FAR | %.1f %% / %.2f %% |" % (
        100 * sep["err"]["all"] / sep["nown"]["all"], 100 * sep["cacc"] / sep["cn"]))
    # порог по контролю 0,1 % – по каждой конфигурации, как разд.
    e01 = n01 = a01 = c01 = 0.0
    th_p3 = []
    th_c1 = []
    for s in row.s:
        d = row.c[s]
        kr = d["ctrl"][0]
        kw = np.ones(len(kr))
        t = np_threshold(kr, kw, 0.001)
        orr, owk = d["own"][0], d["own"][1]
        e01 += ((orr > t) | (owk != 0)).sum()
        n01 += len(orr)
        a01 += (kr <= t).sum()
        c01 += len(kr)
        cr = d["cv"][0]
        th3, _ = p3_threshold(cr, np.ones(len(cr)), d["cv"][2], (np.ones(row.np_) * (d["cv_n"] > 0)), d["cv_n"], 0.01)
        th_p3.append(th3)
        th_c1.append(np_threshold(kr, kw, 0.01))
    L.append("| порог по контролю FAR 0,1 %%: ошибка своих / факт. FAR | %.1f %% / %.2f %% |" % (100 * e01 / n01, 100 * a01 / c01))
    L.append("| θ П3 FAR 1 %% (по конфигурациям) | %s |" % " / ".join("%.4f" % t for t in th_p3))
    L.append("| θ по контролю FAR 1 %% (по конфигурациям) | %s |" % " / ".join("%.4f" % t for t in th_c1))
    L.append("| «чужие» галереи (П3): попыток / людей (сумма по конфигурациям) | %d / %d |" % (len(cv_r), len(cv_p)))
    L.append("| контрольные чужие: попыток / людей | %d / %d |" % (len(ct_r), len(ct_p)))
    L.append("| квантили 1 / 5 / 25 / 50 / 75 / 95 / 99 %%: свои | %s |" % q(own_r))
    L.append("| то же: «чужие» галереи | %s |" % q(cv_r))
    L.append("| то же: контрольные чужие | %s |" % q(ct_r))
    L.append("| худший свой (наибольшая r) / самый похожий контрольный (наименьшая r) / самый похожий «чужой» галереи | %.4f / %.4f / %.4f |" % (
        own_r.max(), ct_r.min(), cv_r.min()))
    gap = ct_r.min() - own_r[~own_wr].max() if (~own_wr).any() else float("nan")
    L.append("| запас: наименьшая r контроля − наибольшая r верных своих | %+.4f%s |" % (gap, " (полное разделение)" if gap > 0 else ""))
    acc_lo = (ct_r < np.quantile(cv_r, 0.01, method="inverted_cdf")).mean() * 100 if len(cv_r) else float("nan")
    L.append("| доля контроля ниже 1-го процентиля «чужих» галереи | %.2f %% |" % acc_lo)
    if ref is not None:
        L.append("| сверка с отчётом (П3: ошибка своих, FAR контроля) | отчёт %s; здесь %.1f %% / %.2f %% |" % (ref, err_p3, far_p3))
    L.append("")
    return L, far_p3


def main(out):
    lines = ["# Переносимость порога П3 – по файлам оценок (Жорик №9 от 22.10.2026) – Ахмед", ""]
    summary = []
    # (4а): L1-xs-F, L1-tt-xs-F, (9б) – строки abl (FERET, GT)
    for kind in ("feret", "gt"):
        rows, rep, persons = abl_check.abl_rows(kind)
        for lab, row in rows.items():
            if not (lab.endswith(" база") or lab.endswith("(9б) Хаар + LBF")):
                continue
            L, f = analyse("%s %s" % (kind.upper(), lab), row)
            lines += L
            summary.append(("%s %s" % (kind.upper(), lab), f))
    # SFace FERET – протокол feret
    bl = {t: b for t, b in read_scores(FERET + "/xbase_feret_scores.tsv.gz")}
    for conf in ("межс.", "внутри", "G2-ii"):
        b = bl["конфигурация %s, набор SFace" % conf]
        row = mk_row(plain_rows([b], "0-sface", lambda d: ratio(float(d["best_score"]), float(d["second_score"]))))
        L, f = analyse("FERET SFace %s (xbase_feret_scores, протокол feret)" % conf, row)
        lines += L
        summary.append(("FERET SFace " + conf, f))
    # SFace GT – gt-diag
    for name, f_ in (("межс.", "xbase_gt_diag_scores.tsv.gz"), ("внутри", "xbase_gt_in_diag_scores.tsv.gz")):
        b = read_scores(DATA_MAIN + "/reports/xbase_2ea947f/" + f_)[0][1]
        row = mk_row(plain_rows([b], "0-sface", lambda d: ratio(float(d["best_score"]), float(d["second_score"]))))
        L, f = analyse("GT SFace %s (%s, протокол gt)" % (name, f_), row)
        lines += L
        summary.append(("GT SFace " + name, f))
    # Прокаев GT – основная строка «Б2-К, гости 10»; оценка – best_score как есть (соглашение проверяется сверкой с отчётом)
    for kit in ("face0.700", "plain"):
        for title, b in read_scores(DATA_MAIN + "/reports/xbase_d1af442/xbase_gt_prokaev_%s_scores.tsv.gz" % kit):
            if "метод Прокаева (Б2-К), гости 10" not in title:
                continue
            ms = collections.Counter(d["method"] for d in b)
            m = sorted(ms)[0]
            row = mk_row(plain_rows([b], m, lambda d: float(d["best_score"])))
            L, f = analyse("GT Прокаев %s %s (%s)" % (kit, title.split(",")[0], m), row)
            lines += L
            summary.append(("GT Прокаев %s %s" % (kit, title.split(",")[0]), f))
    head = ["## Сводка: факт. FAR контроля при П3 FAR 1 % (допуск ≤ 1,1 %)", "", "| строка | FAR контроля при П3 | допуск |", "|---|---|---|"]
    head += ["| %s | %.2f %% | %s |" % (l, f, "да" if f <= FAR_TOL else "нет") for l, f in summary]
    text = "\n".join(lines[:2] + head + [""] + lines[2:]) + "\n"
    open(out, "w", encoding="utf-8").write(text)
    print("\n".join(head))


if __name__ == "__main__":
    main(sys.argv[1])
