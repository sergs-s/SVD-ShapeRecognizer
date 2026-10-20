"""Калибровка r_LBF (поза по точкам Facemark LBF на рамке Хаара) → номинальный угол поворота головы, FERET (Жорик №12, №13 от
20.10.2026). Способ – как calib_yunet.py; точки – вывод VerifyHaarLbf (формат LbfFrames.cached).

r_LBF = FeiDataset.pose(Lm.five()) (LbfFrames.java:64–72, 84–85; FeiDataset.java:96–97): глаза – средние точек 36–41 и 42–47,
нос – 30. Угол – номинальный yaw серии из ground_truths FERET (name_value/*.txt, поле yaw).

python3 -I calib_lbf.py <haar fa/fb> <lbf fa/fb> <haar серий> <lbf серий> [вывод]
"""
import collections
import glob
import math
import os
import sys

import numpy as np

FERET = os.environ.get("VERIFY_FERET", "/home/user/svd-feret-cloud")
MAIN = ["fa", "fb", "rb", "rc", "ql", "qr", "ra", "rd"]
REF = []
THR = [("r0_LBF (выбор модели)", 0.0841), ("R_LBF (отбор)", 0.2486)]
B = 2000
SEED = 7042
FILES = []


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


def five(p):
    """Строка в формате YuNet по 68 точкам (LbfFrames.Lm.five): индексы 4–13 – глаз слева, глаз справа, нос, рот."""
    def mean(a, b):
        x = y = 0.0
        for i in range(a, b + 1):
            x += p[i][0]
            y += p[i][1]
        n = b - a + 1
        return (x / n, y / n)
    row = [0.0] * 15
    for j, f in enumerate([mean(36, 41), mean(42, 47), p[30], p[48], p[54]]):
        row[4 + 2 * j], row[5 + 2 * j] = f
    return row


HAAR = {}  # ключ → "нет лица" / "нет двух глаз" / "лицо и два глаза"


def detections():
    """Ключ → r_LBF (None – нет точек LBF); попутно HAAR – состояние Хаара по ключу."""
    out = {}
    for hp, lp in ((FILES[0], FILES[1]), (FILES[2], FILES[3])):
        for line in open(hp, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            HAAR[c[0]] = "нет лица" if c[2] == "" else ("лицо и два глаза" if c[8] != "" and c[10] != "" else "нет двух глаз")
        for line in open(lp, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[1] != "1":
                out[c[0]] = None
                continue
            v = [float(x) for x in c[2:]]
            out[c[0]] = pose(five([(v[2 * i], v[2 * i + 1]) for i in range(68)]))
    return out


def shares(det, yaw):
    """По сериям: снимков, нет лица Хаара, лицо без точек LBF, с точками; доли |r_LBF| ≤ порогов среди всех снимков серии."""
    L = ["", "Отказы по сериям (доли от всех снимков серии; лицо Хаара – наибольшая рамка, точки LBF – только при найденном лице;",
         "доли |r_LBF| ≤ порога – от всех снимков серии, снимок без точек не проходит):",
         "серия | yaw | снимков | нет лица Хаара | лицо без двух глаз Хаара | лицо есть, нет точек LBF | с точками | "
         "|r_LBF| ≤ r0_LBF | |r_LBF| ≤ R_LBF (из всех)"]
    by = collections.defaultdict(list)
    for k, r in det.items():
        by[k.split("_")[2]].append((k, r))
    for s in MAIN:
        v = by[s]
        n = len(v)
        nf = sum(1 for k, _ in v if HAAR[k] == "нет лица")
        ne = sum(1 for k, _ in v if HAAR[k] == "нет двух глаз")
        nl = sum(1 for k, r in v if HAAR[k] != "нет лица" and r is None)
        ok = [r for _, r in v if r is not None]
        a0 = sum(1 for r in ok if abs(r) <= THR[0][1])
        a1 = sum(1 for r in ok if abs(r) <= THR[1][1])
        L.append("%s | %s | %d | %d (%.1f %%) | %d (%.1f %%) | %d (%.1f %%) | %d | %.1f %% | %.1f %%" % (
            s, "/".join("%g" % a for a in sorted({yaw[k] for k, _ in v if k in yaw})), n, nf, 100 * nf / n, ne, 100 * ne / n,
            nl, 100 * nl / n, len(ok), 100 * a0 / n, 100 * a1 / n))
    return L


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


def main(files, out_path=None):
    FILES[:] = files
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
    L.append("Калибровка r_LBF → номинальный yaw FERET; r_LBF – FeiDataset.pose(LbfFrames.Lm.five()) (LbfFrames.java:64–72, 84–85;")
    L.append("FeiDataset.java:96–97); точки – VerifyHaarLbf (HaarFrames.detect, LbfFrames.fit, масштаб 1,0); угол – ground_truths (yaw=).")
    L.append("Оговорка: угол номинальный (куда просили смотреть), разброс r включает неточность самой позы.")
    L.append("Снимков без yaw в ground_truths: %s" % (dict(miss) or "нет"))
    L.append("")
    L += shares(det, yaw)
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
    for a, b in (("rb", "rc"), ("ql", "qr"), ("ra", "rd")):
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
    main(sys.argv[1:5], sys.argv[5] if len(sys.argv) > 5 else None)
