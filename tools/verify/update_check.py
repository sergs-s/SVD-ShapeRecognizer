"""Строки отчётов feret-update и feret-wstatus4 по их файлам оценок (для compare.py, наборы feret-update, feret-wstatus4).

В файлах – решение прохода (пара fa/fb: лучший кандидат и d₁/d₂ решающего снимка, признак анфаса) и строки threshold с
порогами П2 и П3 шага при FAR 1 %; оценок «чужих» галереи нет. Поэтому:
- П2 и П3 при FAR 1 % – по записанным порогам шага; в бутстрепе здесь порог тот же (предположение; для П2 сверкой
  опровергнуто – интервалы отчёта шире, код, видимо, пересчитывает порог по «чужим» в каждой реплике, а их в файле нет);
- П2 и П3 при FAR 5 % из файла не вычислимы (порогов нет, «чужих» нет);
- «разд.» и EER – по контрольным своей конфигурации (проход, шаг), в репликах – заново.
Строки «шаг 1, общая W» – попыток в файле нет. Строки «К4 … (в) M = 3 (кадр а)» – те же попытки, что строки «… (в) M = 3»
без К4 (пара «К4 − кадр а»; совпадение чисел видно в отчёте).
"""
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import FERET, parse_report  # noqa: E402
from measures import Row  # noqa: E402

# «шаги ≥ 2»: конфигурации (проход, шаг) без проб своих не входят (контроль и EER – только там, где есть свои); подобрано по
# расхождению FAR контроля в строках «шаги ≥ 2» (до чтения кода).
SKIP_EMPTY = True


def load(kind):
    """(метка без шага, шаг) → {проход: {own: [(r, wrong, person, поза)], ctrl: [...], th: (П2, П3)}}."""
    path = FERET + "/xbase_feret_%s_scores.tsv.gz" % kind
    head = None
    conf = {}
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
                continue
            # поза: 0 – анфас, 1 – поворот (по столбцу frontal; граница r0 = 0,12 в Row)
            a = (float(d["ratio"]), d["best"] != d["person"], d["person"], 0.0 if d["frontal"] == "1" else 1.0)
            if d["role"] == "own":
                if d["measured"] == "1":
                    c["own"].append(a)
            elif d["role"] == "ctrl":
                c["ctrl"].append(a)
            else:
                raise ValueError(d["role"])
    return conf


def parts_of(conf, lab):
    """Метка строки отчёта → [(проход, шаг, конфигурация)] или None (попыток в файле нет)."""
    if "общая W" in lab:
        return None
    if lab.startswith("К4 ") and " (кадр а)" in lab:
        lab = lab[3:].replace(" (кадр а)", "")
    if lab.endswith(" шаги ≥ 2"):
        base = lab[:-len(" шаги ≥ 2")]
        out = [(ps, s, c) for (l, s), v in conf.items() if l == base and s >= 2 for ps, c in v.items()
               if c["own"] or not SKIP_EMPTY]
        return out or None
    base, step = lab.rsplit(" шаг ", 1)
    step = int(step)
    if (base, step) not in conf:
        base += " (а) статично"  # шаг 1 у всех вариантов одинаков и дан один раз
    if (base, step) not in conf:
        return None
    return [(ps, step, c) for ps, c in conf[(base, step)].items()]


def update_rows(kind):
    conf = load(kind)
    rep_path = FERET + "/xbase_feret_%s.txt" % kind
    rep = parse_report(rep_path)
    labels = set(rep.get("sum", {})) | set(rep.get("eer", {})) | {k[0] for k in rep.get("dec", {})} | \
        {k[0] for k in rep.get("far", {})}
    for k in list(rep.get("pair", {})) + [(p,) for p in rep.get("psep", {})]:
        labels |= set(k[0].split(" − "))
    persons = sorted({a[2] for v in conf.values() for c in v.values() for a in c["own"] + c["ctrl"]})
    pidx = {p: i for i, p in enumerate(persons)}
    rows = {}
    for lab in labels:
        parts = parts_of(conf, lab)
        if parts is None:
            continue
        cf, fixed = {}, {}
        for ps, step, c in parts:
            s = int(ps) * 100 + step
            cf[s] = {"own": c["own"], "ctrl": c["ctrl"], "cv": []}
            fixed[s] = {("P2", 0.01): c["th"][0], ("P3", 0.01): c["th"][1]}
        r = Row(cf, pidx)
        r.fixed = fixed
        rows[lab] = r
    return rows, rep_path, persons
