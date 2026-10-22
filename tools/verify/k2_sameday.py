"""Слои S / N для порога по внешним чужим (Жорик №5 от 22.10.2026): по готовым файлам оценок VerifyK2Ext, без прогонов.

Слой S – дата снимка (по описи, Img.date – дата в ключе снимка) совпадает с одной из дат галереи прохода и конфигурации, N – нет;
как XBaseSameDay (XBaseSameDay.java:98–117: allDates – даты всех снимков галереи своих прохода). Снимки галереи прохода
восстанавливаются по попыткам «чужих» галереи (роль cv<q>: каждый снимок галереи – проба своей части).

1) доля S у контроля и у внешних чужих; 2) FAR контроля по слоям при К2 по всем внешним (цель 1 %; 300 × 1, все × 1, все × 2);
3) К2 по S-попыткам внешних с проверкой на S-контроле и то же для N (если в слое не меньше 30 попыток).

python3 -I k2_sameday.py <файл оценок VerifyK2Ext> <вывод>
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_check  # noqa: E402
import compare  # noqa: E402
from k2_ext import load  # noqa: E402
from measures import R0, p3_threshold, ratio  # noqa: E402

SEED = 7042 + 9990  # numpy default_rng(SEED + 1000·размер + 10·снимков + номер)
SAMPLES = 100
FAR, TOL = 0.01, 1.1
MIN_N = 30


def conf_full(rows, pose, keys, plain):
    """{роль: список (r, wrong, person, date)} одной конфигурации."""
    by = {}
    for d in rows:
        by.setdefault((d["role"], d["person"], d["image"]), {})[d["method"]] = d
    out = {"own": [], "ctrl": [], "cv": [], "cal": []}
    for (role, person, image), m in by.items():
        if plain:
            d = next(iter(m.values()))
        else:
            base = [k for k in m if not k.endswith("+mirror")][0]
            d = m[base] if abs(pose(person, image)) <= R0 else m[base + "+mirror"]
        rr = "cv" if role.startswith("cv") else role
        date = keys[(person, image)].split("_")[1]
        out[rr].append((ratio(float(d["best_score"]), float(d["second_score"])), d["best_id"] != d["true_id"], person, date))
    return out


def k2(r_sorted, people):
    names = sorted(set(people))
    idx = {p: i for i, p in enumerate(names)}
    pid = np.array([idx[p] for p in people])
    cnt = np.bincount(pid, minlength=len(names))
    th, _ = p3_threshold(r_sorted, np.ones(len(r_sorted)), pid, np.ones(len(names)), cnt, FAR)
    return th


def pick(cal, pp, rng, people=None):
    """Калибровочные попытки: люди (все или подмножество), не больше pp снимков на человека; по возрастанию r."""
    order = rng.permutation(len(cal))
    taken = {}
    keep = []
    for j in order:
        p = cal[j][2]
        if people is not None and p not in people:
            continue
        if taken.get(p, 0) < pp:
            taken[p] = taken.get(p, 0) + 1
            keep.append(j)
    keep.sort(key=lambda j: cal[j][0])
    return np.array([cal[j][0] for j in keep]), [cal[j][2] for j in keep]


def q(v, p):
    return float(np.quantile(v, p, method="inverted_cdf"))


def main(path, out):
    pose = compare.feret_pose()
    keys = abl_check.feret_keys()
    lines = {}
    for title, rows in load(path):
        if "конфигурация" not in title:
            continue
        t = dict(x.strip().split(" ", 1) for x in title.split(","))
        plain = "набор" in t
        lab = "%s SFace" % t["конфигурация"] if plain else "%s %s" % (t["конфигурация"], t["строка"])
        c = conf_full(rows, pose, keys, plain)
        gal_dates = {a[3] for a in c["cv"]} | set()
        # даты галереи: снимки галереи = пробы cv (даты своих по ключу снимка)
        for role in ("ctrl", "cal"):
            c[role] = [a + (a[3] in gal_dates,) for a in c[role]]
        lines.setdefault(lab, []).append(c)
    L = ["# Слои S / N для порога по внешним чужим FERET (Жорик №5 от 22.10.2026) – Ахмед", "",
         "S – дата снимка среди дат галереи прохода (как XBaseSameDay), N – прочие. Выборок %d, seed – numpy default_rng(%d + 1000·n + 10·снимков + номер); "
         "К2 – П3, цель 1 %%, допуск 1,1 %%." % (SAMPLES, SEED), ""]
    L.append("## 1. Доля S")
    L.append("")
    L.append("| строка | контроль: S / всего (доля) | внешние чужие: S / всего (доля) | дат галереи (по проходам) |")
    L.append("|---|---|---|---|")
    for lab, confs in lines.items():
        cs = sum(sum(a[4] for a in c["ctrl"]) for c in confs)
        cn = sum(len(c["ctrl"]) for c in confs)
        es = sum(sum(a[4] for a in c["cal"]) for c in confs)
        en = sum(len(c["cal"]) for c in confs)
        nd = " / ".join(str(len({a[3] for a in c["cv"]})) for c in confs)
        L.append("| %s | %d / %d (%.1f %%) | %d / %d (%.1f %%) | %s |" % (lab, cs, cn, 100 * cs / cn, es, en, 100 * es / en, nd))
    L.append("")
    L.append("## 2. FAR контроля по слоям при К2 по всем внешним чужим (цель 1 %)")
    L.append("")
    L.append("| строка | набор | FAR контроля S: медиана [5; 95 %], доля выше 1,1 % | FAR контроля N: то же | FAR всего: то же |")
    L.append("|---|---|---|---|---|")
    for lab, confs in lines.items():
        ext = sorted({a[2] for a in confs[0]["cal"]})
        for name, n_, pp in (("300 × 1", 300, 1), ("все × 1", 0, 1), ("все × 2", 0, 2)):
            res = {"S": [], "N": [], "all": []}
            for i in range(SAMPLES):
                rng = np.random.default_rng(SEED + 1000 * n_ + 10 * pp + i)
                people = None if n_ == 0 else set(rng.choice(ext, size=n_, replace=False))
                acc = {"S": 0, "N": 0}
                tot = {"S": 0, "N": 0}
                for c in confs:
                    r, ps = pick(c["cal"], pp, rng, people)
                    th = k2(r, ps)
                    for a in c["ctrl"]:
                        lay = "S" if a[4] else "N"
                        tot[lay] += 1
                        acc[lay] += a[0] <= th
                for lay in ("S", "N"):
                    res[lay].append(100 * acc[lay] / tot[lay] if tot[lay] else np.nan)
                res["all"].append(100 * (acc["S"] + acc["N"]) / (tot["S"] + tot["N"]))
            cells = []
            for lay in ("S", "N", "all"):
                v = np.array(res[lay])
                if np.isnan(v).all():
                    cells.append("слой пуст")
                    continue
                cells.append("%.2f [%.2f; %.2f], %.0f %%" % (np.median(v), q(v, 0.05), q(v, 0.95), 100 * (v > TOL).mean()))
            L.append("| %s | %s | %s |" % (lab, name, " | ".join(cells)))
    L.append("")
    L.append("## 3. К2 по слою внешних чужих, проверка на том же слое контроля")
    L.append("")
    L.append("Калибровка – все внешние люди слоя, по 1 или 2 случайных снимка; seed – default_rng(%d + 500 + 10·снимков + номер), "
             "у слоя N ещё + 50000. При числе попыток калибровки < 299 порог П3 – ниже наименьшей оценки (по одной попытке)." % SEED)
    L.append("")
    L.append("| строка | слой | внешних в слое: людей / снимков (по проходам) | контроль слоя: попыток (по проходам) | FAR × 1: медиана [5; 95 %], доля выше 1,1 % | FAR × 2: то же |")
    L.append("|---|---|---|---|---|---|")
    for lab, confs in lines.items():
        for lay, flag in (("S", True), ("N", False)):
            ncal = [sum(1 for a in c["cal"] if a[4] == flag) for c in confs]
            nctl = [sum(1 for a in c["ctrl"] if a[4] == flag) for c in confs]
            if min(ncal) < MIN_N or min(nctl) < MIN_N:
                L.append("| %s | %s | попыток калибровки %s, контроля %s – меньше %d, без расчёта | | | |" % (
                    lab, lay, " / ".join(map(str, ncal)), " / ".join(map(str, nctl)), MIN_N))
                continue
            cells = []
            npeople = []
            for pp in (1, 2):
                v = []
                for i in range(SAMPLES):
                    rng = np.random.default_rng(SEED + 500 + 10 * pp + i + (0 if flag else 50000))
                    acc = tot = 0
                    for c in confs:
                        sub = [a for a in c["cal"] if a[4] == flag]
                        r, ps = pick(sub, pp, rng)
                        if i == 0 and pp == 1:
                            npeople.append("%d / %d" % (len(set(a[2] for a in sub)), len(sub)))
                        th = k2(r, ps)
                        for a in c["ctrl"]:
                            if a[4] == flag:
                                tot += 1
                                acc += a[0] <= th
                    v.append(100 * acc / tot)
                v = np.array(v)
                cells.append("%.2f [%.2f; %.2f], %.0f %%" % (np.median(v), q(v, 0.05), q(v, 0.95), 100 * (v > TOL).mean()))
            L.append("| %s | %s | %s | %s | %s | %s |" % (lab, lay, "; ".join(npeople), " / ".join(map(str, nctl)), cells[0], cells[1]))
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
