"""Размер набора внешних чужих для порога по внешним чужим (Жорик №2, №3 от 22.10.2026), FERET, по готовым файлам оценок.

Вариант (а), решение Хозяина 22.10: внешние чужие – часть контрольных людей FERET (одна дата); проверка – остальные контрольные.
Это отличие от соглашения кода: в XBaseCalib калибровочные – люди пула (≥ 2 дат). Довод – в ПО набор К2 – посторонние, снятые
по разу.

Для каждой выборки (seed SEED + номер): калибровочные люди – случайные n из 200 контрольных, снимков на человека – 1 (случайный)
или 2 (до двух, у кого есть); проверочные – остальные люди (все их снимки). Пороги – по каждой конфигурации (проход / роль), как в
отчётах:
- К1 – Нейман – Пирсон по калибровочным попыткам (measures.np_threshold: принято не больше ⌊FAR·n⌋);
- К2 – П3 по ним же (measures.p3_threshold, Клоппер – Пирсон по n_eff); при n < 299 попыток (FAR 1 %) и < 2995 (0,1 %) К2
  вырожден – порог ниже всех (XBaseCalib.K2_MIN = 299).
Меры по выборке: фактический FAR на проверочных людях (сумма по конфигурациям), ошибка своих (все пробы своих).

python3 -I k2_size.py <вывод>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_check  # noqa: E402
from compare import FERET, read_scores  # noqa: E402
from measures import np_threshold, p3_threshold, ratio  # noqa: E402
from p3_transfer import mk_row, plain_rows  # noqa: E402

SEED = 7042 + 9970  # выборки: numpy default_rng(SEED + 1000·размер + 10·снимков + номер)
SAMPLES = 100
SIZES = (20, 50, 100)
FARS = ((0.01, 1.1), (0.001, 0.11))


def rows():
    out = {}
    for kind in ("feret",):
        r, _, _ = abl_check.abl_rows(kind)
        for lab, row in r.items():
            if lab.endswith(" база") and ("L1-xs-F" in lab or "L1-tt-xs-F" in lab):
                out[lab.replace(" база", "")] = row
    bl = {t: b for t, b in read_scores(FERET + "/xbase_feret_scores.tsv.gz")}
    for conf, lab in (("межс.", "межс."), ("G2-ii", "G2"), ("внутри", "внутри")):
        b = bl["конфигурация %s, набор SFace" % conf]
        out["%s SFace" % lab] = mk_row(plain_rows([b], "0-sface", lambda d: ratio(float(d["best_score"]), float(d["second_score"]))))
    return out


def one(row, cal_set, per_person, rng_choice, far, k2):
    """FAR на проверке и ошибка своих при пороге по калибровочным (К1 или К2), сумма по конфигурациям; θ по конфигурациям."""
    acc = n = e = no = 0
    ths = []
    natt = []
    for s in row.s:
        d = row.c[s]
        kr, _, kpid, _ = d["ctrl"]
        cal = np.isin(kpid, cal_set)
        idx = np.nonzero(cal)[0]
        # снимки калибровочного человека: выбор фиксирован выборкой (rng_choice – ранги снимков человека)
        keep = []
        for p in np.unique(kpid[idx]):
            ii = idx[kpid[idx] == p]
            order = ii[np.argsort(rng_choice[ii])]
            keep.extend(order[:per_person])
        keep = np.sort(np.array(keep, dtype=int))
        cr = kr[keep]
        cw = np.ones(len(cr))
        if k2:
            cnt = np.bincount(kpid[keep], minlength=row.np_)
            th, _ = p3_threshold(cr, cw, kpid[keep], (np.ones(row.np_) * (cnt > 0)), cnt, far)
        else:
            th = np_threshold(cr, cw, far)
        ths.append(th)
        natt.append(len(cr))
        test = ~cal
        acc += (kr[test] <= th).sum()
        n += test.sum()
        orr, owk = d["own"][0], d["own"][1]
        e += ((orr > th) | (owk != 0)).sum()
        no += len(orr)
    return 100.0 * acc / n, 100.0 * e / no, ths, natt


def pct(v, p):
    return float(np.quantile(v, p, method="inverted_cdf"))


def main(out):
    R = rows()
    L = ["# Размер набора внешних чужих для порога К1 / К2 – FERET, деление контроля (Жорик №2, №3 от 22.10.2026) – Ахмед", ""]
    L.append("Выборок на размер: %d; seed выборок – numpy default_rng(%d + 1000·n + 10·снимков + номер)." % (SAMPLES, SEED))
    L.append("")
    best = {}
    for lab, row in R.items():
        ctrl_people = np.unique(np.concatenate([row.c[s]["ctrl"][2] for s in row.s]))
        L.append("## " + lab + " (контрольных людей %d, попыток контроля на конфигурацию %s)" % (
            len(ctrl_people), " / ".join(str(len(row.c[s]["ctrl"][0])) for s in row.s)))
        L.append("")
        L.append("| порог | цель FAR | людей × снимков | попыток калибровки (медиана) | θ: медиана [5; 95 %] | FAR проверки: медиана [5; 95 %], макс | доля выборок с FAR выше допуска | ошибка своих: медиана [5; 95 %] |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k2 in (False, True):
            for far, tol in FARS:
                for n_ in SIZES:
                    for pp in (1, 2):
                        fs, es, th0, na = [], [], [], []
                        for i in range(SAMPLES):
                            rng = np.random.default_rng(SEED + 1000 * n_ + 10 * pp + i)
                            cal = rng.choice(ctrl_people, size=n_, replace=False)
                            # ранги снимков – один набор на выборку (общий для конфигураций): по номеру попытки в первой конфигурации
                            choice = {s: rng.random(len(row.c[s]["ctrl"][0])) for s in row.s}
                            acc = n = e = no = 0
                            ths = []
                            nat = []
                            for s in row.s:
                                sub = type("R", (), {})()
                                sub.s, sub.c, sub.np_ = [s], {s: row.c[s]}, row.np_
                                f1, e1, t1, a1 = one(sub, cal, pp, choice[s], far, k2)
                                d = row.c[s]
                                test_n = (~np.isin(d["ctrl"][2], cal)).sum()
                                acc += f1 * test_n / 100
                                n += test_n
                                e += e1 * len(d["own"][0]) / 100
                                no += len(d["own"][0])
                                ths += t1
                                nat += a1
                            fs.append(100 * acc / n)
                            es.append(100 * e / no)
                            th0.append(ths[0])
                            na.append(np.median(nat))
                        fs, es, th0 = np.array(fs), np.array(es), np.array(th0)
                        bad = (fs > tol).mean() * 100
                        name = "К2 (П3)" if k2 else "К1 (Н–П)"
                        L.append("| %s | %.1f %% | %d × %d | %.0f | %.4f [%.4f; %.4f] | %.2f [%.2f; %.2f], %.2f | %.0f %% | %.1f [%.1f; %.1f] |" % (
                            name, 100 * far, n_, pp, np.median(na), np.median(th0), pct(th0, 0.05), pct(th0, 0.95), np.median(fs), pct(fs, 0.05),
                            pct(fs, 0.95), fs.max(), bad, np.median(es), pct(es, 0.05), pct(es, 0.95)))
                        key = (lab.split(" ", 1)[1], name, far)
                        if bad <= 5:
                            best.setdefault(key, {}).setdefault(lab.split(" ", 1)[0], []).append((n_ * pp, n_, pp, np.median(es)))
        L.append("")
    L.append("## Наименьший набор (людей × снимков), при котором FAR проверки в допуске в ≥ 95 % выборок")
    L.append("")
    L.append("| метод | порог | цель FAR | межс. | G2 | внутри |")
    L.append("|---|---|---|---|---|---|")
    meths = sorted({k[0] for k in best} | {lab.split(" ", 1)[1] for lab in R})
    for m in meths:
        for name in ("К1 (Н–П)", "К2 (П3)"):
            for far, _ in FARS:
                cells = []
                for prot in ("межс.", "G2", "внутри"):
                    v = best.get((m, name, far), {}).get(prot)
                    if not v:
                        cells.append("нет (до 100 × 2)")
                    else:
                        v.sort()
                        cells.append("%d × %d (ошибка своих %.1f %%)" % (v[0][1], v[0][2], v[0][3]))
                L.append("| %s | %s | %.1f %% | %s |" % (m, name, 100 * far, " | ".join(cells)))
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(meths) * 4 + 2):]))


if __name__ == "__main__":
    main(sys.argv[1])
