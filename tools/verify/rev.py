"""Разбор VerifyRev (ревизия, пп. 1.11–1.13; Жорик №7, №8 от 23.10.2026): только файлы оценок и stats, без кода проекта.

Меры (measures.evaluate): разделимость – ошибка своих при фактическом FAR контроля 1, 2, 5, 10 % (порог по контрольным чужим),
EER, П3 FAR 1 % (ошибка своих и FAR контроля). Выбор модели X / X+mirror – по позе варианта (r или r′) и его r0. Парные разности
«вариант − база» – на общих пробах (своих и контрольных, прошедших отбор в обоих); «чужие» П2/П3 – у каждого свои. Бутстреп по
людям: B = 2000, numpy default_rng(7042), одни веса на все строки; «*» – интервал 2,5–97,5 % без 0.

python3 -I rev.py <каталог FERET (оценки, поза, stats)> <каталог GT (оценки, поза)> <stats прочих баз> <R′> <r0′> <вывод .md>
"""
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_check  # noqa: E402
from measures import R0, Row, evaluate, ratio, read_scores  # noqa: E402

FARS = (0.01, 0.02, 0.05, 0.10)
B = int(os.environ.get("VERIFY_B", "2000"))
SEED = 7042
VARS = ("base", "A", "B", "C", "ABC")
ROWS = ("L1-xs-F", "L1-tt-xs-F")


def load_pose(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        out[(c[0], c[1])] = (float(c[2]), float(c[3]))
    return out


def confs(path, pose, prime, r0, gt):
    """{(постановка, строка): {s: {own|ctrl|cv: [(r, wrong, person, pose, id)]}}}."""
    out = {}
    for title, rows in read_scores(path):
        if gt:
            mode, row = title.split(", проекция ")
            s = 0
        else:
            t = dict(x.strip().split(" ", 1) for x in title.split(","))
            mode, row, s = t["конфигурация"], t["строка"], int(t["проход"]) - 1
        by = {}
        for d in rows:
            m = d["method"]
            key = (d["role"], d["person"], d["image"])
            by.setdefault(key, {})[m.endswith("+mirror")] = d
        c = out.setdefault((mode, row), {}).setdefault(s, {"own": [], "ctrl": [], "cv": []})
        for (role, person, image), pair in by.items():
            pk = (person, "%02d" % int(image)) if gt else (person, image)
            pr = pose[pk][1 if prime else 0]
            d = pair[abs(pr) > r0]
            if d["no_agreement"] != "-":
                raise ValueError("no_agreement")
            r = ratio(float(d["best_score"]), float(d["second_score"]))
            rr = "cv" if role.startswith("cv") else role
            c[rr].append((r, d["best_id"] != d["true_id"], person, pr, (role, person, image)))
    return out


def restrict(conf, keep):
    """Только своих и контрольных из keep[s]; cv – как есть."""
    out = {}
    for s, d in conf.items():
        out[s] = {"cv": d["cv"], "own": [a for a in d["own"] if a[4] in keep[s]], "ctrl": [a for a in d["ctrl"] if a[4] in keep[s]]}
    return out


def ids(conf):
    return {s: {a[4] for r in ("own", "ctrl") for a in d[r]} for s, d in conf.items()}


def mk(conf, pidx, r0):
    return Row({s: {r: [a[:4] for a in L_] for r, L_ in d.items()} for s, d in conf.items()}, pidx, r0)


def metrics(res):
    v = {}
    for far in FARS:
        x = res[("sep", far)]
        v[("sep", far)] = 100.0 * x["err"]["all"] / x["nown"]["all"] if x["nown"]["all"] > 0 else np.nan
        v[("sepfar", far)] = 100.0 * x["cacc"] / x["cn"] if x["cn"] > 0 else np.nan
    x = res[("P3", 0.01)]
    v["P3err"] = 100.0 * x["err"]["all"] / x["nown"]["all"] if x["nown"]["all"] > 0 else np.nan
    v["P3far"] = 100.0 * x["cacc"] / x["cn"] if x["cn"] > 0 else np.nan
    v["EER"] = 100.0 * res["EER"][3]
    return v


KEYS = [("sep", f) for f in FARS] + ["EER", "P3err", "P3far"]
NAMES = {("sep", 0.01): "разд. 1 %", ("sep", 0.02): "разд. 2 %", ("sep", 0.05): "разд. 5 %", ("sep", 0.10): "разд. 10 %", "EER": "EER",
         "P3err": "П3 1 %: ошибка своих", "P3far": "П3 1 %: FAR контроля"}


def ev(row, w):
    return metrics(evaluate(row, w, fars=FARS))


def ci(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def fmt_d(pt, lo, hi):
    star = " *" if (lo > 0 or hi < 0) else ""
    return "%+.1f [%+.1f; %+.1f]%s" % (pt, lo, hi, star)


def main(fdir, gdir, other_stats, rp, r0p, out):
    rp, r0p = float(rp), float(r0p)
    L = ["# Ревизия, пп. 1.11–1.13: варианты А, Б, В конвейера (Жорик №7, №8 от 23.10.2026) – Ахмед", ""]
    # ---- данные
    data = {}
    for prot, d, gt in (("FERET", fdir, False), ("GT", gdir, True)):
        for v in VARS + ("base_nc", "B_nc"):
            path = os.path.join(d, "verify_rev_%s_%s_scores.tsv.gz" % ("gt" if gt else "feret", v))
            if not os.path.exists(path):
                continue
            pose = load_pose(os.path.join(d, "verify_rev_%s_%s_pose.tsv" % ("gt" if gt else "feret", v)))
            a = "A" in v
            data[(prot, v)] = confs(path, pose, a, r0p if a else R0, gt)
    persons = sorted({a[2] for c in data.values() for cc in c.values() for d in cc.values() for L_ in d.values() for a in L_})
    pidx = {p: i for i, p in enumerate(persons)}
    rows = {}
    sizes = []
    for (prot, v), c in data.items():
        for (mode, rl), conf in c.items():
            lab = (prot, mode, rl)
            r0 = r0p if "A" in v else R0
            rows[lab + (v, "full")] = mk(conf, pidx, r0)
            n = {r: sum(len(conf[s][r]) for s in conf) for r in ("own", "ctrl", "cv")}
            if v in VARS and v != "base" and (prot, "base") in data:
                bconf = data[(prot, "base")][(mode, rl)]
                ib, iv = ids(bconf), ids(conf)
                common = {s: ib[s] & iv[s] for s in conf}
                rows[lab + (v, "common")] = mk(restrict(conf, common), pidx, r0)
                rows[lab + ("base", "common:" + v)] = mk(restrict(bconf, common), pidx, R0)
                nb = {r: sum(len(bconf[s][r]) for s in bconf) for r in ("own", "ctrl", "cv")}
                nc = {r: sum(sum(1 for a in conf[s][r] if a[4] in common[s]) for s in conf) for r in ("own", "ctrl")}
                sizes.append((lab, v, n, nb, nc))
            elif v == "base":
                sizes.append((lab, v, n, None, None))
    w1 = np.ones(len(persons), dtype=np.int64)
    point = {k: ev(r, w1) for k, r in rows.items()}
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in rows}
    for i in range(B):
        w = np.bincount(rng.integers(0, len(persons), len(persons)), minlength=len(persons))
        for k, r in rows.items():
            boot[k].append(ev(r, w))
    L.append("Бутстреп по людям: B = %d, numpy default_rng(%d), людей %d (FERET и GT вместе, одни веса); «*» – интервал разности без 0." % (B, SEED, len(persons)))
    L.append("Вариант А: R′ = %.6f, r0′ = %.6f (95-е процентили |r′|: своя база 113 снимков, FEI №11–14 из fei_selection.tsv, 776 снимков); база: R = 0,386956, r0 = 0,12." % (rp, r0p))
    L.append("")
    L.append("## Наборы проб: своих / контрольных / «чужих» П2–П3 (сумма по проходам); у вариантов – общих с базой своих / контрольных")
    L.append("")
    L.append("| постановка | строка | вариант | своих | контрольных | «чужих» П2/П3 | общих с базой: своих / контрольных | выпало из базы: своих / контр. | вошло нового: своих / контр. |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab, v, n, nb, nc in sizes:
        if nb is None:
            L.append("| %s %s | %s | %s | %d | %d | %d | – | – | – |" % (lab[0], lab[1], lab[2], v, n["own"], n["ctrl"], n["cv"]))
        else:
            L.append("| %s %s | %s | %s | %d | %d | %d | %d / %d | %d / %d | %d / %d |" % (lab[0], lab[1], lab[2], v, n["own"], n["ctrl"], n["cv"], nc["own"], nc["ctrl"],
                     nb["own"] - nc["own"], nb["ctrl"] - nc["ctrl"], n["own"] - nc["own"], n["ctrl"] - nc["ctrl"]))
    L.append("")

    def cell(k, key):
        lo, hi = ci([m[key] for m in boot[k]])
        return "%.1f [%.1f; %.1f]" % (point[k][key], lo, hi)

    labs = sorted({k[:3] for k in rows}, key=lambda x: (x[0], x[1] != "межс.", x[2]))
    L.append("## Меры на полном наборе каждого варианта (%, точка [2,5; 97,5 %])")
    L.append("")
    for lab in labs:
        L.append("### %s %s, %s" % lab)
        L.append("")
        L.append("| вариант | " + " | ".join(NAMES[k] for k in KEYS) + " |")
        L.append("|---|" + "---|" * len(KEYS))
        for v in VARS + ("base_nc", "B_nc"):
            k = lab + (v, "full")
            if k in rows:
                L.append("| %s | %s |" % (v, " | ".join(cell(k, x) for x in KEYS)))
        L.append("")
    L.append("## Парные разности «вариант − база» на общих пробах (п. п., точка [2,5; 97,5 %])")
    L.append("")
    for lab in labs:
        L.append("### %s %s, %s" % lab)
        L.append("")
        L.append("| вариант | " + " | ".join(NAMES[k] for k in KEYS) + " |")
        L.append("|---|" + "---|" * len(KEYS))
        for v in VARS[1:]:
            a, b = lab + (v, "common"), lab + ("base", "common:" + v)
            if a not in rows:
                continue
            cells = []
            for key in KEYS:
                d = np.array([m[key] for m in boot[a]]) - np.array([m[key] for m in boot[b]])
                lo, hi = ci(d)
                cells.append(fmt_d(point[a][key] - point[b][key], lo, hi))
            L.append("| %s | %s |" % (v, " | ".join(cells)))
        L.append("")
    # когорта
    lab = ("GT", "межс.", "L1-xs-F")
    keys = [lab + (v, "full") for v in ("base", "B", "base_nc", "B_nc")]
    if all(k in rows for k in keys):
        L.append("## Когорта FEI (прежние кадры): GT межс., L1-xs-F – «Б − база» с когортой и без неё")
        L.append("")
        L.append("| мера | Б − база (с когортой) | Б − база (без когорты) | разность разностей |")
        L.append("|---|---|---|---|")
        for key in KEYS:
            x = [np.array([m[key] for m in boot[k]]) for k in keys]
            d1, d2 = x[1] - x[0], x[3] - x[2]
            p1 = point[keys[1]][key] - point[keys[0]][key]
            p2 = point[keys[3]][key] - point[keys[2]][key]
            L.append("| %s | %s | %s | %s |" % (NAMES[key], fmt_d(p1, *ci(d1)), fmt_d(p2, *ci(d2)), fmt_d(p1 - p2, *ci(d1 - d2))))
        L.append("")
    # ---- диагностика без распознавания
    st = []
    for f in (os.path.join(fdir, "stats_feret.tsv"), other_stats):
        lines = open(f, encoding="utf-8").read().split("\n")
        h = lines[0].split("\t")
        st += [dict(zip(h, x.split("\t"))) for x in lines[1:] if x]
    by = collections.defaultdict(list)
    for d in st:
        by[d["base"]].append(d)
    L.append("## 1.13: инлайеры LMEDS и масштаб подобия")
    L.append("")
    L.append("| база | снимков | инлайеров < 5 | из них 3 / 4 | 1 / масштаб LMEDS: медиана [мин; макс] | k (Б, LMEDS): распределение | кадр базы из сырого = кадр экспорта |")
    L.append("|---|---|---|---|---|---|---|")
    for b in ("feret", "gt", "fei", "muct", "orl", "own"):
        v = by[b]
        inl = collections.Counter(int(d["lmeds_inliers"]) for d in v)
        sl = np.array([float(d["scale_lmeds"]) for d in v])
        k = collections.Counter(int(d["k_lmeds"]) for d in v)
        fs = collections.Counter(d["frame_same"] for d in v)
        L.append("| %s | %d | %.1f %% | %d / %d | %.2f [%.2f; %.2f] | %s | %s |" % (
            b, len(v), 100 * (inl[3] + inl[4] + inl[2] + inl[1] + inl[0]) / len(v), inl[3], inl[4], np.median(1 / sl), (1 / sl).min(), (1 / sl).max(),
            ", ".join("%d: %d" % kv for kv in sorted(k.items())), "%d из %d" % (fs["1"], len(v)) if fs["-"] == 0 else "–"))
    L.append("")
    # fa / fb одного дня
    fv = {d["key"]: d for d in by["feret"]}
    grp = collections.defaultdict(dict)
    for k in sorted(fv):
        p, date, pose = k.split("_")[:3]
        grp[(p, date)].setdefault(pose, k)
    dl, dq, inl_any = [], [], []
    for g, m in grp.items():
        if "fa" in m and "fb" in m:
            a, b_ = fv[m["fa"]], fv[m["fb"]]
            dl.append(abs(math.log(float(a["scale_lmeds"]) / float(b_["scale_lmeds"]))))
            dq.append(abs(math.log(float(a["scale_lsq"]) / float(b_["scale_lsq"]))))
            inl_any.append(int(a["lmeds_inliers"]) < 5 or int(b_["lmeds_inliers"]) < 5)
    dl, dq, inl_any = np.array(dl), np.array(dq), np.array(inl_any)
    L.append("FERET, fa и fb одного дня (%d пар; первый снимок позы дня): |ln(s_fa / s_fb)|, %% – медиана / 95-й процентиль / максимум:" % len(dl))
    L.append("")
    L.append("| подобие | все пары | пары, где у fa или fb < 5 инлайеров (%d) | остальные (%d) |" % (inl_any.sum(), (~inl_any).sum()))
    L.append("|---|---|---|---|")
    for nm, x in (("LMEDS", dl), ("наименьшие квадраты", dq)):
        f = lambda y: "%.2f / %.2f / %.2f" % (100 * np.median(y), 100 * np.quantile(y, 0.95), 100 * y.max())  # noqa: E731
        L.append("| %s | %s | %s | %s |" % (nm, f(x), f(x[inl_any]), f(x[~inl_any])))
    L.append("")
    # шум
    L.append("## 1.12: σ шума кадра «а» (Immerkær), база и Б: медиана по людям медиан человека; разность «пробы − галерея» – среднее по людям")
    L.append("")
    L.append("| постановка | кадр | галерея | пробы | пробы − галерея [2,5; 97,5 % бутстреп по людям] | людей |")
    L.append("|---|---|---|---|---|---|")
    fkeys = abl_check.feret_keys()
    allst = {("feret", d["key"]): d for d in by["feret"]}
    allst.update({("gt", d["key"]): d for d in by["gt"]})
    for prot, base in (("FERET", "feret"), ("GT", "gt")):
        if (prot, "base") not in data:
            continue
        conf = data[(prot, "base")][("межс.", "L1-xs-F")]
        gal, pro = collections.defaultdict(set), collections.defaultdict(set)
        for s, d in conf.items():
            for a in d["cv"]:
                gal[a[2]].add(a[4][2])
            for a in d["own"]:
                pro[a[2]].add(a[4][2])
        ps = sorted(set(gal) & set(pro))
        key = (lambda p, i: fkeys[(p, i)]) if prot == "FERET" else (lambda p, i: "%s/%02d" % (p, int(i)))
        for col, nm in (("sigma_base", "база"), ("sigma_B", "Б")):
            g = np.array([np.median([float(allst[(base, key(p, i))][col]) for i in gal[p]]) for p in ps])
            q = np.array([np.median([float(allst[(base, key(p, i))][col]) for i in pro[p]]) for p in ps])
            d = q - g
            r2 = np.random.default_rng(SEED)
            bm = [d[r2.integers(0, len(d), len(d))].mean() for _ in range(B)]
            lo, hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
            L.append("| %s межс. | %s | %.3f | %.3f | %+.3f [%+.3f; %+.3f] | %d |" % (prot, nm, np.median(g), np.median(q), d.mean(), lo, hi, len(ps)))
    L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:7])
