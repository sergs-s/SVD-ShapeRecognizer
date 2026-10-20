"""Калибровка r (поза по точкам YuNet) → номинальный угол поворота головы, FERET (Жорик №10 от 20.10.2026).

r = (x носа − x середины глаз) / |x глаза слева − x глаза справа| по строке YuNet – как в коде отбора и выбора модели
(FeiDataset.java:96–97: (row[8] − (row[4] + row[6]) / 2) / |row[6] − row[4]|). Угол – номинальный yaw серии из ground_truths
FERET (name_value/*.txt, поле yaw; куда человека просили смотреть, а не измеренный поворот).

python3 -I calib_yunet.py [вывод]
"""
import collections
import glob
import math
import os
import sys

import numpy as np

FERET = os.environ.get("VERIFY_FERET", "/home/user/svd-feret-cloud")
MAIN = ["fa", "fb", "rb", "rc", "ql", "qr", "ra", "rd"]
REF = ["hl", "hr", "re", "pl", "pr"]
THR = [("r0 (выбор модели)", 0.12), ("R (отбор)", 0.3869555959643167)]
B = 2000
SEED = 7042


def yaw_table():
    """Ключ снимка → номинальный yaw из ground_truths."""
    out = {}
    for f in glob.glob(FERET + "/dvd*/data/ground_truths/name_value/*/*_*_*.txt"):
        key = os.path.basename(f)[:-4]
        for line in open(f, encoding="utf-8", errors="replace"):
            if line.startswith("yaw="):
                out[key] = float(line.strip()[4:])
    return out


def pose(v):
    return (v[8] - (v[4] + v[6]) / 2) / abs(v[6] - v[4])


def detections():
    out = {}
    for name in ("detections.tsv", "detections_allpose.tsv"):
        for line in open(FERET + "/derived/" + name, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if c[1] != "1":
                out.setdefault(c[0], None)
                continue
            out[c[0]] = pose([float(x) for x in c[2:17]])
    return out


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def interp_angle(angles, med, t):
    """Наименьший угол ≥ 0, при котором кусочно-линейная медиана |r| (по возрастанию угла) достигает t; None – не достигает."""
    for i in range(len(angles) - 1):
        a0, a1, m0, m1 = angles[i], angles[i + 1], med[i], med[i + 1]
        if m0 <= t <= m1 and m1 > m0:
            return a0 + (t - m0) / (m1 - m0) * (a1 - a0)
        if m0 >= t and i == 0:
            return a0
    return None


def main(out_path=None):
    yaw = yaw_table()
    det = detections()
    series = collections.defaultdict(list)  # серия → [(человек, r)]
    ang = {}
    miss = collections.Counter()
    for key, r in det.items():
        p = key.split("_")
        s = p[2]
        if key not in yaw:
            miss[s] += 1
            continue
        ang.setdefault(s, set()).add(yaw[key])
        if r is None:
            continue
        series[s].append((p[0], r))
    L = []
    L.append("Калибровка r YuNet → номинальный yaw FERET; r – FeiDataset.java:96–97; угол – ground_truths (yaw=).")
    L.append("Оговорка: угол номинальный (куда просили смотреть), разброс r включает неточность самой позы.")
    L.append("Снимков без yaw в ground_truths: %s" % (dict(miss) or "нет"))
    L.append("")
    L.append("серия | yaw | снимков с точками | людей | медиана r | 5 % | 95 % | медиана |r| | людей: медиана r по человеку – 5 % / 50 % / 95 %")
    per_person = {}
    for s in MAIN + REF:
        rs = np.array([r for _, r in series[s]])
        byp = collections.defaultdict(list)
        for person, r in series[s]:
            byp[person].append(r)
        pm = np.array([np.median(v) for v in byp.values()])
        per_person[s] = {k: float(np.median(v)) for k, v in byp.items()}
        L.append("%s%s | %s | %d | %d | %+.3f | %+.3f | %+.3f | %.3f | %+.3f / %+.3f / %+.3f" % (
            s, " (справочно)" if s in REF else "", "/".join("%g" % a for a in sorted(ang.get(s, []))), len(rs), len(byp),
            q(rs, 50), q(rs, 5), q(rs, 95), q(np.abs(rs), 50), q(pm, 5), q(pm, 50), q(pm, 95)))
    # знак и симметрия
    L.append("")
    L.append("Знак и симметрия (медианы r по снимкам): пары «+угол» / «−угол»:")
    for a, b in (("rb", "rc"), ("ql", "qr"), ("ra", "rd"), ("hl", "hr"), ("pl", "pr")):
        ra_ = np.median([r for _, r in series[a]])
        rb_ = np.median([r for _, r in series[b]])
        L.append("  %s (%+g°) %+.3f | %s (%+g°) %+.3f | сумма %+.3f (0 – симметрично)" % (
            a, list(ang[a])[0], ra_, b, list(ang[b])[0], rb_, ra_ + rb_))
    # общие люди
    common = set.intersection(*[set(per_person[s]) for s in MAIN])
    L.append("")
    L.append("Общие люди (есть во всех сериях %s): %d" % (", ".join(MAIN), len(common)))
    L.append("серия | yaw | медиана r по людям | 5 % | 95 %")
    for s in MAIN:
        v = np.array([per_person[s][p] for p in sorted(common)])
        L.append("%s | %s | %+.3f | %+.3f | %+.3f" % (s, "/".join("%g" % a for a in sorted(ang[s])), q(v, 50), q(v, 5), q(v, 95)))
    # перевод в градусы
    L.append("")
    L.append("Перевод порогов в градусы. Способ: |угол| 0 (fa, fb), 15 (rb, rc), 22,5 (ql, qr), 45 (ra, rd); по каждому |углу| – |r|")
    L.append("снимков обеих сторон; кусочно-линейная интерполяция медианы |r| по |углу|; интервал – бутстреп по людям (B = %d, seed %d,"
             % (B, SEED))
    L.append("люди вытягиваются целиком, все их снимки во всех сериях), процентили 2,5 / 97,5. Рядом – по сторонам отдельно и угол, при")
    L.append("котором порог пересекают 25 % / 75 % людей (по медиане r человека) – разброс между людьми в градусах.")
    groups = [(0.0, ["fa", "fb"]), (15.0, ["rb", "rc"]), (22.5, ["ql", "qr"]), (45.0, ["ra", "rd"])]
    sides = {"+": [(0.0, ["fa", "fb"]), (15.0, ["rb"]), (22.5, ["ql"]), (45.0, ["ra"])],
             "−": [(0.0, ["fa", "fb"]), (15.0, ["rc"]), (22.5, ["qr"]), (45.0, ["rd"])]}
    persons = sorted({p for s in MAIN for p, _ in series[s]})
    pidx = {p: i for i, p in enumerate(persons)}
    data = {s: (np.array([pidx[p] for p, _ in series[s]]), np.abs(np.array([r for _, r in series[s]]))) for s in MAIN}

    def medians(gr, w=None):
        out = []
        for a, ss in gr:
            ids = np.concatenate([data[s][0] for s in ss])
            v = np.concatenate([data[s][1] for s in ss])
            if w is None:
                out.append(float(np.median(v)))
            else:
                rep = np.repeat(v, w[ids])
                out.append(float(np.median(rep)) if len(rep) else float("nan"))
        return out

    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(persons), len(persons)), minlength=len(persons)) for _ in range(B)]
    for name, t in THR:
        L.append("")
        L.append("%s = %.4f:" % (name, t))
        for lab, gr in [("обе стороны", groups), ("сторона +", sides["+"]), ("сторона −", sides["−"])]:
            angs = [a for a, _ in gr]
            m = medians(gr)
            pt = interp_angle(angs, m, t)
            reps = []
            for w in W:
                v = interp_angle(angs, medians(gr, w), t)
                reps.append(np.nan if v is None else v)
            reps = np.array(reps)
            ok = reps[~np.isnan(reps)]
            ci = (q(ok, 2.5), q(ok, 97.5)) if len(ok) else (float("nan"), float("nan"))
            L.append("  %s: медианы |r| по |углу| %s → угол %s° [%.1f; %.1f] (реплик без пересечения: %d)" % (
                lab, " / ".join("%g°: %.3f" % (a, x) for a, x in zip(angs, m)), "—" if pt is None else "%.1f" % pt, ci[0], ci[1],
                int(np.isnan(reps).sum())))
        # разброс между людьми: углы, где порог пересекают 25 / 75 % людей
        pp = []
        for a, ss in groups:
            vals = []
            for s in ss:
                vals += [abs(v) for v in per_person[s].values()]
            pp.append((a, np.array(vals)))
        for qq in (25, 75):
            m = [q(v, qq) for _, v in pp]
            v = interp_angle([a for a, _ in pp], m, t)
            L.append("  %d-й процентиль |r| людей по |углу| %s → угол %s°" % (
                qq, " / ".join("%g°: %.3f" % (a, x) for (a, _), x in zip(pp, m)), "—" if v is None else "%.1f" % v))
    # однозначность: разброс между людьми против шага между сериями
    L.append("")
    L.append("Однозначность: межквартильный размах |r| людей в серии против шага медиан |r| между соседними сериями:")
    prev = None
    for (a, v) in pp:
        iqr = q(v, 75) - q(v, 25)
        med = q(v, 50)
        step = "" if prev is None else "; шаг медианы от %g° – %.3f" % (prev[0], med - prev[1])
        L.append("  %g°: медиана %.3f, межквартильный размах %.3f, 5–95 %% – %.3f%s" % (a, med, iqr, q(v, 95) - q(v, 5), step))
        prev = (a, med)
    text = "\n".join(L) + "\n"
    if out_path:
        open(out_path, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
