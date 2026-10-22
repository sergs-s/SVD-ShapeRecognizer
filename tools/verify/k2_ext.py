"""Порог по внешним чужим FERET (Жорик №4 от 22.10.2026): калибровка – внешние чужие (роль cal, VerifyK2Ext), проверка – контроль.

Калибровка – n людей из внешних (100 / 200 / 300 / 500 / все), по 1 или 2 снимка (случайные); проверка – контроль FERET (200
людей, все снимки). Пороги – по каждой конфигурации (проход), К1 – Нейман – Пирсон, К2 – П3 (measures.p3_threshold). Модель X /
X+mirror – по |r| пробы ≤ 0,12 (как в отчётах), SFace – одна модель.

python3 -I k2_ext.py <файл оценок VerifyK2Ext> <вывод>
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
from measures import R0, np_threshold, p3_threshold, ratio  # noqa: E402

SEED = 7042 + 9980  # numpy default_rng(SEED + 1000·размер + 10·снимков + номер)
SAMPLES = 100
SIZES = (100, 200, 300, 500, 0)  # 0 – все
FARS = ((0.01, 1.1), (0.001, 0.11))


def load(path):
    """Блоки → [(заголовок, строки-словари)]."""
    out = []
    head = None
    cols = None
    cur = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# "):
                cur = []
                out.append((line[2:], cur))
                continue
            if line.startswith("#"):
                continue
            if line.startswith("method\t"):
                cols = line.split("\t")
                continue
            if cur is None or cols is None:
                continue
            cur.append(dict(zip(cols, line.split("\t"))))
    return out


def conf_of(rows, pose, plain):
    """{роль: (r, wrong, person)} одной конфигурации; модель по позе (classic) или одна (SFace)."""
    by = {}
    for d in rows:
        key = (d["role"], d["person"], d["image"])
        by.setdefault(key, {})[d["method"]] = d
    out = {"own": [], "ctrl": [], "cv": [], "cal": []}
    for (role, person, image), m in by.items():
        if plain:
            d = next(iter(m.values()))
        else:
            base = [k for k in m if not k.endswith("+mirror")][0]
            d = m[base] if abs(pose(person, image)) <= R0 else m[base + "+mirror"]
        rr = "cv" if role.startswith("cv") else role
        out[rr].append((ratio(float(d["best_score"]), float(d["second_score"])), d["best_id"] != d["true_id"], person))
    return {k: (np.array([a[0] for a in v]), np.array([a[1] for a in v]), np.array([a[2] for a in v])) for k, v in out.items()}


def main(path, out):
    pose = compare.feret_pose()
    blocks = load(path)
    lines = {}
    for title, rows in blocks:
        t = dict(x.strip().split(" ", 1) for x in title.split(","))
        if "набор" in t:
            lab = "%s SFace" % t["конфигурация"]
            lines.setdefault(lab, []).append(conf_of(rows, pose, True))
        else:
            lab = "%s %s" % (t["конфигурация"], t["строка"])
            lines.setdefault(lab, []).append(conf_of(rows, pose, False))
    L = ["# Порог по внешним чужим FERET – калибровка 100…все, проверка – контроль (Жорик №4 от 22.10.2026) – Ахмед", "",
         "Выборок на размер: %d; seed – numpy default_rng(%d + 1000·n + 10·снимков + номер); «все» – n = 0 в seed." % (SAMPLES, SEED), ""]
    best = {}
    for lab, confs in lines.items():
        ext = sorted(set(confs[0]["cal"][2]))
        L.append("## %s (внешних людей %d, снимков %d; контроль %d попыток на конфигурацию; конфигураций %d)" % (
            lab, len(ext), len(confs[0]["cal"][0]), len(confs[0]["ctrl"][0]), len(confs)))
        L.append("")
        L.append("| порог | цель FAR | людей × снимков | попыток (медиана) | FAR контроля: медиана [5; 95 %], макс | доля выборок выше допуска | ошибка своих: медиана [5; 95 %] |")
        L.append("|---|---|---|---|---|---|---|")
        for k2 in (False, True):
            for far, tol in FARS:
                for n_ in SIZES:
                    for pp in (1, 2):
                        fs, es, na = [], [], []
                        for i in range(SAMPLES):
                            rng = np.random.default_rng(SEED + 1000 * n_ + 10 * pp + i)
                            people = ext if n_ == 0 else list(rng.choice(ext, size=n_, replace=False))
                            pset = set(people)
                            acc = n = e = no = 0
                            nat = []
                            for c in confs:
                                cr, _, cp = c["cal"]
                                order = rng.permutation(len(cr))
                                taken = {}
                                keep = []
                                for j in order:
                                    p = cp[j]
                                    if p in pset and taken.get(p, 0) < pp:
                                        taken[p] = taken.get(p, 0) + 1
                                        keep.append(j)
                                keep = np.array(sorted(keep, key=lambda j: cr[j]))
                                r = cr[keep]
                                w = np.ones(len(r))
                                if k2:
                                    names = cp[keep]
                                    uniq = {p: i for i, p in enumerate(sorted(set(names)))}
                                    pid = np.array([uniq[p] for p in names])
                                    cnt = np.bincount(pid, minlength=len(uniq))
                                    th, _ = p3_threshold(r, w, pid, np.ones(len(uniq)), cnt, far)
                                else:
                                    th = np_threshold(r, w, far)
                                nat.append(len(r))
                                kr = c["ctrl"][0]
                                acc += (kr <= th).sum()
                                n += len(kr)
                                orr, owr, _ = c["own"]
                                e += ((orr > th) | owr).sum()
                                no += len(orr)
                            fs.append(100 * acc / n)
                            es.append(100 * e / no)
                            na.append(np.median(nat))
                        fs, es = np.array(fs), np.array(es)
                        bad = (fs > tol).mean() * 100
                        name = "К2 (П3)" if k2 else "К1 (Н–П)"
                        size = "все %d" % len(ext) if n_ == 0 else str(n_)
                        q = lambda v, p: float(np.quantile(v, p, method="inverted_cdf"))  # noqa: E731
                        L.append("| %s | %.1f %% | %s × %d | %.0f | %.2f [%.2f; %.2f], %.2f | %.0f %% | %.1f [%.1f; %.1f] |" % (
                            name, 100 * far, size, pp, np.median(na), np.median(fs), q(fs, 0.05), q(fs, 0.95), fs.max(), bad, np.median(es),
                            q(es, 0.05), q(es, 0.95)))
                        if bad <= 5:
                            key = (lab, name, far)
                            nn = len(ext) if n_ == 0 else n_
                            if key not in best or (nn * pp, nn) < best[key][:2]:
                                best[key] = (nn * pp, nn, pp, np.median(es))
        L.append("")
    L.append("## Наименьший набор (людей × снимков), при котором FAR контроля в допуске в ≥ 95 % выборок")
    L.append("")
    L.append("| строка | порог | цель FAR | набор | ошибка своих (медиана) |")
    L.append("|---|---|---|---|---|")
    for lab in lines:
        for name in ("К1 (Н–П)", "К2 (П3)"):
            for far, _ in FARS:
                b = best.get((lab, name, far))
                L.append("| %s | %s | %.1f %% | %s | %s |" % (lab, name, 100 * far, "%d × %d" % (b[1], b[2]) if b else "нет (до всех × 2)",
                                                               "%.1f %%" % b[3] if b else "–"))
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
