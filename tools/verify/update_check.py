"""Сверка точек сводки feret-update и feret-wstatus4 по их файлам оценок (только то, что из файла вычислимо).

В файлах – решение прохода (пара fa/fb: лучший кандидат и d₁/d₂ решающего снимка) и строки threshold с порогами П2 и П3 шага
(FAR 1 %); оценок «чужих» галереи и порогов FAR 5 % нет. Поэтому сверяются: ошибка своих при П2 и П3 FAR 1 % (по записанным
порогам) с FAR контроля, «разд.» при FAR 1 % и 5 % (порог по контрольным своей конфигурации – проход и шаг), EER. Интервалы
бутстрепа не сверяются (траектория обновлений в репликах не пересчитывается, порядок реплик в коде не описан).

python3 -I update_check.py update|wstatus4 [вывод]
"""
import gzip
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import Cmp, FERET, fmt, parse_report, split_cell  # noqa: E402
from measures import eer, np_threshold  # noqa: E402

# «шаги ≥ 2»: конфигурации (проход, шаг) без проб своих не входят (контроль и EER – только там, где есть свои); подобрано по
# расхождению FAR контроля в строках «шаги ≥ 2» (до чтения кода).
SKIP_EMPTY = True


def load(kind):
    path = FERET + "/xbase_feret_%s_scores.tsv.gz" % kind
    head = None
    conf = {}  # (метка без шага, шаг) → {(проход): {own, ctrl, th}}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("#") or not line:
                continue
            p = line.split("\t")
            if head is None:
                head = p
                continue
            d = dict(zip(head, p))
            if kind == "update":
                lab = "%s %s %s" % (d["proj"], d["gate"], d["variant"])
            else:
                lab = "%s %s %s" % (d["proj"], d["scheme"], d["variant"])
            c = conf.setdefault((lab, int(d["step"])), {}).setdefault(d["pass"], {"own": [], "ctrl": [], "th": None})
            if d["role"] == "threshold":
                t = dict(x.strip().split(" ") for x in d["ratio"].split(";"))
                c["th"] = (float(t["П2"]), float(t["П3"]))
            elif d["role"] == "own":
                if d["measured"] == "1":
                    c["own"].append((float(d["ratio"]), d["best"] != d["person"]))
            elif d["role"] == "ctrl":
                c["ctrl"].append(float(d["ratio"]))
            else:
                raise ValueError(d["role"])
    return conf


def measures_of(parts):
    """parts – список конфигураций (проход, шаг): own [(r, wrong)], ctrl [r], th (П2, П3)."""
    out = {}
    for name, idx in (("P2", 0), ("P3", 1)):
        e = n = ca = cn = 0
        for c in parts:
            th = c["th"][idx]
            for r, w in c["own"]:
                e += (r > th) or w
                n += 1
            ca += sum(r <= th for r in c["ctrl"])
            cn += len(c["ctrl"])
        out[name] = (e, n, ca, cn)
    for far in (0.01, 0.05):
        e = n = ca = cn = 0
        for c in parts:
            k = np.sort(np.array(c["ctrl"]))
            th = np_threshold(k, np.ones(len(k)), far)
            for r, w in c["own"]:
                e += (r > th) or w
                n += 1
            ca += int((k <= th).sum())
            cn += len(k)
        out[("sep", far)] = (e, n, ca, cn)
    own = [x for c in parts for x in c["own"]]
    ctrl = [x for c in parts for x in c["ctrl"]]
    orr = np.array([x[0] for x in own])
    owr = np.array([x[1] for x in own], dtype=bool)
    cr = np.array(ctrl)
    out["EER"] = eer(orr, owr, np.ones(len(orr)), cr, np.ones(len(cr)))
    return out


def main(kind, out_path=None):
    conf = load(kind)
    rep = parse_report(FERET + "/xbase_feret_%s.txt" % kind)
    C = Cmp()
    skipped = []
    for lab, cells in rep["sum"].items():
        parts = None
        if lab.endswith("шаги ≥ 2"):
            base = lab[:-len(" шаги ≥ 2")]
            parts = [p for (l, s), v in conf.items() if l == base and s >= 2 for p in v.values() if p["own"] or not SKIP_EMPTY]
        elif " шаг " in lab:
            base, step = lab.rsplit(" шаг ", 1)
            if not step.isdigit():
                skipped.append(lab)
                continue
            if kind == "update" and int(step) == 1 and base.count(" ") == 1:
                base += " (а) статично"  # шаг 1 у всех вариантов одинаков, дан один раз
            parts = list(conf.get((base, int(step)), {}).values())
        if not parts:
            skipped.append(lab)
            continue
        m = measures_of(parts)
        for i, key in ((0, "P2"), (1, "P3")):
            e, n, ca, cn = m[key]
            p, lo, hi, tail = split_cell(cells[i])
            C.pt("%s | %s FAR 1 %%" % (lab, key), p + " " + tail, "%d/%d = %s %% (%s %%)" % (e, n, fmt(100 * e / n, 1), fmt(100 * ca / cn, 2)))
        for i, far in ((2, 0.01), (5, 0.05)):
            e, n, ca, cn = m[("sep", far)]
            p, lo, hi, tail = split_cell(cells[i])
            C.pt("%s | разд. %g %%" % (lab, far * 100), p + " " + tail, "%s %% (%s %%)" % (fmt(100 * e / n, 1), fmt(100 * ca / cn, 2)))
        p, lo, hi, tail = split_cell(cells[6])
        C.pt("%s | EER" % lab, p, fmt(100 * m["EER"][3], 1) + " %")
    head = ["%s: точки %d из %d совпали; строк сводки без данных в файле (не сверялись): %d – %s"
            % (kind, C.ok_pt, C.n_pt, len(skipped), "; ".join(sorted(set(s.split(" шаг")[0] + " шаг …" for s in skipped))))]
    text = "\n".join(head + ["", "РАСХОЖДЕНИЯ:"] + C.bad + ["", "ВСЕ:"] + C.lines) + "\n"
    if out_path:
        open(out_path, "w", encoding="utf-8").write(text)
    print(head[0])
    for b in C.bad[:30]:
        print(b)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
