"""Сверка диагностических разделов отчётов, вычислимых из файлов оценок (задача №2 от 20.10.2026).

python3 -I diag_check.py [вывод]

- ORL, MUCT: «Правило d₁/d₂ у «чужих» П2 (L1; случаи 1–6)» – число случаев по выбранной по позе модели;
- GT и FERET subsp: «Сосредоточенность контроля» – контрольные пробы, у которых лучший кандидат – один из двух самых
  частых лучших своего прохода; «Объём» (своих, снимков шаблона = попыток cv, проб);
- FERET xs2: «Объём по проходам» (своих, проб своих, новых классов L1-xs-F); «Лучший кандидат – хаб»;
- FERET update: «Опись» и столбцы «Ход по шагам» до «анти-хабов» включительно.
"""
import collections
import gzip
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import (DATA, FERET, R0, Cmp, feret_pose, fmt, gt_pose, muct_pose, orl_pose,  # noqa: E402
                     read_scores)
from measures import np_threshold, ratio  # noqa: E402

import numpy as np  # noqa: E402


def pick(rows, method, pose_of, r0=R0):
    """Попытки после выбора модели X / X+mirror по |r| пробы: [словарь строки файла]."""
    by = {}
    for d in rows:
        if d["method"] in (method, method + "+mirror"):
            by.setdefault((d["s"], d["j"], d["role"], d["person"], d["image"]), {})[d["method"]] = d
    out = []
    for (s, j, role, person, image), pair in by.items():
        out.append(pair[method] if abs(pose_of(person, image)) <= r0 else pair[method + "+mirror"])
    return out


def report_lines(path):
    with open(path, encoding="utf-8") as f:
        return [x.rstrip("\n").rstrip("\r") for x in f]


def section(lines, title):
    """Строки таблиц всех разделов, чьё название начинается с title: {метка: [ячейки]}."""
    out = {}
    on = False
    for x in lines:
        if x.startswith("=== "):
            on = x.strip("= ").startswith(title)
            continue
        if on and " | " in x and not x.startswith("строка") and not x.startswith("роль") \
                and not x.startswith("постановка") and not x.startswith("проход") and not x.startswith("вариант"):
            c = x.split(" | ")
            out.setdefault(c[0], []).append(c[1:])
        elif on and not x.strip():
            pass
    return out


def frac(x, n, dec=1):
    return "%d/%d = %s %%" % (x, n, fmt(100 * x / n, dec))


# ---------- ORL, MUCT: правило вырожденных случаев ----------

def degenerate(C):
    for name, path, rep, pose in (
            ("ORL", DATA + "/reports/xbase_72c3195/xbase_orl_scores.tsv.gz", DATA + "/reports/xbase_72c3195/xbase_orl.txt",
             orl_pose()),
            ("MUCT", DATA + "/reports/xbase_72c3195/xbase_muct_scores.tsv.gz", DATA + "/reports/xbase_72c3195/xbase_muct.txt",
             muct_pose())):
        rows = pick(read_scores(path)[0][1], "5-lbp-mlda", pose)
        cnt = collections.Counter()
        for d in rows:
            if not d["role"].startswith("cv"):
                continue
            d1, d2 = float(d["best_score"]), float(d["second_score"])
            if d2 == math.inf:
                cnt[4] += 1
            elif -1e-12 <= d1 < 0 or -1e-12 <= d2 < 0:
                cnt[5] += 1
            elif d1 == 0 and d2 == 0:
                cnt[2] += 1
            elif d1 == 0 or (d2 > 0 and d1 / d2 < 1e-12):
                cnt[1] += 1
            try:
                ratio(d1, d2)
            except ValueError:
                cnt[3] += 1
        ours = "все конфигурации: " + ", ".join("%d: %d" % (k, cnt[k]) for k in range(1, 7))
        ref = [x for x in report_lines(rep) if x.startswith("Правило d₁/d₂")][0]
        C.pt("%s | правило d₁/d₂ у «чужих» П2 (L1)" % name, ref.split("), ", 1)[1], ours)


# ---------- сосредоточенность, объём ----------

def top2(best):
    c = collections.Counter(best)
    return sum(v for _, v in c.most_common(2))


def blocks_feret(kind):
    """FERET subsp / xs2: метка строки → {проход: [строки после выбора модели]}."""
    pose = feret_pose()
    out = {}
    for title, b in read_scores(FERET + "/xbase_feret_%s_scores.tsv.gz" % kind):
        if kind == "subsp":
            role, proj, ps, k = [x.strip() for x in title.split(",")]
            lab = "%s %s %s" % (role, proj.replace("проекция ", ""), k)
            ps = ps.replace("проход ", "")
        else:
            t = dict(x.strip().split(" ", 1) for x in title.split(","))
            lab = "%s %s" % (t["конфигурация"], t["строка"])
            ps = t["проход"]
        out.setdefault(lab, {})[ps] = pick(b, "5-lbp-mlda", pose)
    return out


def concentration(C):
    # GT subsp – один проход
    pose = gt_pose()
    rep = report_lines(DATA + "/reports/xbase_f12513c/xbase_gt_subsp.txt")
    ref = section(rep, "Сосредоточенность контроля")
    for title, b in read_scores(DATA + "/reports/xbase_f12513c/xbase_gt_subsp_scores.tsv.gz"):
        mode, proj, k = [x.strip() for x in title.split(",")]
        lab = "%s %s %s" % (mode, proj.replace("проекция ", ""), k)
        rows = [d for d in pick(b, "5-lbp-mlda", pose) if d["role"] == "ctrl"]
        C.pt("GT %s | сосредоточенность" % lab, ref[lab][0][0], frac(top2(d["best_id"] for d in rows), len(rows)))
    # FERET subsp – по проходам
    rep = report_lines(FERET + "/xbase_feret_subsp.txt")
    ref = section(rep, "Сосредоточенность контроля")
    vol = section(rep, "Объём")
    bl = blocks_feret("subsp")
    seen = set()
    for lab, by in bl.items():
        x = n = 0
        for ps, rows in by.items():
            ctrl = [d for d in rows if d["role"] == "ctrl"]
            x += top2(d["best_id"] for d in ctrl)
            n += len(ctrl)
        C.pt("FERET subsp %s | сосредоточенность" % lab, ref[lab][0][0], frac(x, n))
        role = lab.split(" L1")[0]
        if role in seen or not lab.endswith("k = 0") or " L1-xs-F " not in lab:
            continue
        seen.add(role)
        for ps in sorted(by):
            rows = by[ps]
            own = [d for d in rows if d["role"] == "own"]
            cv = [d for d in rows if d["role"].startswith("cv")]
            ours = "%d | %d | %d" % (len({d["person"] for d in own}), len(cv), len(own))
            refc = [c for c in vol[role] if c[0] == ps][0]
            C.pt("FERET subsp объём %s проход %s | своих, снимков шаблона, проб" % (role, ps), " | ".join(refc[1:4]), ours)


# ---------- FERET xs2: объём, хабы ----------

def feret_dates():
    dates = collections.defaultdict(set)
    with open(FERET + "/derived/detections.tsv", encoding="utf-8") as f:
        for line in f:
            key = line.split("\t", 1)[0]
            p = key.split("_")
            dates[p[0]].add(p[1])
    return {k: len(v) for k, v in dates.items()}


def xs2(C):
    rep = report_lines(FERET + "/xbase_feret_xs2.txt")
    vol = section(rep, "Объём по проходам")
    hub = section(rep, "Лучший кандидат – хаб")
    bl = blocks_feret("xs2")
    nd = feret_dates()
    own_of = {}
    for lab, by in bl.items():
        for ps, rows in by.items():
            own_of.setdefault((lab.split(" ")[0], ps), {d["person"] for d in rows if d["role"] == "own"})
    for (role, ps), persons in sorted(own_of.items()):
        rows = bl["%s L1" % role][ps]
        own = [d for d in rows if d["role"] == "own"]
        other = own_of[("межс.", "2" if ps == "1" else "1")]  # пул прохода – вся другая половина
        ours = "%d | %d | %d" % (len(persons), len(own), len(other))
        refc = [c for c in vol[role] if c[0] == ps][0]
        C.pt("FERET xs2 объём %s проход %s | своих, проб своих, новых классов L1-xs-F" % (role, ps),
             "%s | %s | %s" % (refc[1], refc[2], refc[3].split(" / ")[0]), ours)
    # хаб – человек пула (все 173 человека с ≥ 2 датами, обе половины) с ≥ 5 датами
    for lab, by in bl.items():
        if lab not in hub:
            continue
        role = lab.split(" ")[0]
        xo = no = xn = xc = nc = 0
        for ps, rows in by.items():
            pool = own_of[("межс.", "1")] | own_of[("межс.", "2")]
            hubs = {p for p in pool if nd.get(p, 0) >= 5}
            for d in rows:
                if d["role"] == "own":
                    no += 1
                    if d["best_id"] in hubs:
                        xo += 1
                        xn += d["best_id"] != d["true_id"]
                elif d["role"] == "ctrl":
                    nc += 1
                    xc += d["best_id"] in hubs
        C.pt("FERET xs2 %s | лучший – хаб" % lab, " | ".join(hub[lab][0]), "%s | %d | %s" % (frac(xo, no), xn, frac(xc, nc)))


# ---------- FERET update: опись, ход по шагам ----------

def update(C):
    from update_check import load, parts_of
    conf = load("update")
    rep = report_lines(FERET + "/xbase_feret_update.txt")
    # опись – по строкам L1-xs-F П3 (а) (свои с ≥ 4 датами измеряются; число дат = последний шаг + 2)
    inv = section(rep, "Опись")
    steps = collections.defaultdict(lambda: collections.defaultdict(int))
    last = collections.defaultdict(dict)
    gal = collections.defaultdict(set)
    path = FERET + "/xbase_feret_update_scores.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if head is None:
                head = p
                continue
            d = dict(zip(head, p))
            if d["proj"] != "L1-xs-F" or d["gate"] != "П3" or d["variant"] != "(а) статично" or d["role"] != "own":
                continue
            gal[d["pass"]].add(d["person"])
            if d["measured"] == "1":
                steps[d["pass"]][int(d["step"])] += 1
                last[d["pass"]][d["person"]] = max(last[d["pass"]].get(d["person"], 0), int(d["step"]))
    # галерея прохода – вся половина (люди с 2–3 датами проб не имеют и в файле update их нет): берётся из файла xs2
    # (те же половины №2, seed faces.seed + 9910; проверяется вложение людей файла update в половину)
    halves = {}
    for ps, rows in blocks_feret("xs2")["межс. L1"].items():
        halves[ps] = {d["person"] for d in rows if d["role"] == "own"}
    for ps in gal:
        assert gal[ps] <= halves[ps], "люди update вне половины xs2"
    GALLERY.clear()
    GALLERY.update(halves["1"] | halves["2"])
    for ps in sorted(gal):
        nd = collections.Counter(v + 2 for v in last[ps].values())
        mx = max(max(s for v in steps.values() for s in v), 1)
        ours = "%d | %d | {%s} | %s" % (len(halves[ps]), len(last[ps]), ", ".join("%d=%d" % (k, nd[k]) for k in sorted(nd)),
                                        ", ".join(str(steps[ps].get(s, 0)) for s in range(1, mx + 1)))
        refc = [c for c in inv[ps]][0] if ps in inv else None
        C.pt("FERET update опись проход %s" % ps, " | ".join(refc[:4]) if refc else "—", ours)
    # ход по шагам: таблицы идут по порядку разделов «######## X, порог приёма G ########»
    cur = None
    gsize = sum(len(v) for v in gal.values())
    for x in rep:
        if x.startswith("######## ") and "порог приёма" in x:
            t = x.strip("# ").split(", порог приёма ")
            cur = (t[0], t[1])
            continue
        if cur is None or " | " not in x or x.startswith("вариант") or x.startswith("(") is False:
            continue
        c = x.split(" | ")
        if len(c) < 11 or not c[1].isdigit():
            continue
        var, step = c[0], int(c[1])
        lab = "%s %s %s шаг %d" % (cur[0], cur[1], var, step)
        parts = parts_of(conf, lab)
        if parts is None:
            C.na.append("ход по шагам: " + lab)
            continue
        own = [a for _, _, p in parts for a in p["own"]]
        res = []
        res.append("%d / %d" % (len({a[2] for a in own}), len(own)))
        for idx in (1, 0):  # П3, П2
            e = sum((a[0] > p["th"][idx]) or a[1] for _, _, p in parts for a in p["own"])
            res.append("%s %%" % fmt(100 * e / len(own), 1) if own else "— %")
        e = 0
        for _, _, p in parts:
            k = np.sort(np.array([a[0] for a in p["ctrl"]]))
            th = np_threshold(k, np.ones(len(k)), 0.01)
            e += sum((a[0] > th) or a[1] for a in p["own"])
        res.append("%s %%" % fmt(100 * e / len(own), 1) if own else "— %")
        for idx in (1, 0):
            ca = sum(a[0] <= p["th"][idx] for _, _, p in parts for a in p["ctrl"])
            cn = sum(len(p["ctrl"]) for _, _, p in parts)
            res.append("%s %% (%d/%d)" % (fmt(100 * ca / cn, 2), ca, cn))
        C.pt("FERET update ход %s | своих, ошибки П3/П2/разд., FAR П3/П2" % lab, " | ".join(c[2:8]), " | ".join(res))
        # хабы – по всей галерее обоих проходов: лучшие у контроля (кандидаты-люди галереи)
        conf_all = parts_of(conf, lab)
        best = []
        with_best = set()
        # лучшего у контроля в load нет – читаем отдельно
        best = CTRL_BEST.get((cur[0], cur[1], var if step > 1 else "(а) статично", step), [])
        gbest = [b for b in best if b in GALLERY]
        cn = len(best)
        cnt = collections.Counter(gbest)
        top = cnt.most_common(2)
        h1 = top[0][1] if top else 0
        h2 = sum(v for _, v in top)
        anti = sum(1 for p in GALLERY if cnt[p] == 0)
        C.pt("FERET update ход %s | доля первого хаба, двух, анти-хабов" % lab, " | ".join(c[8:11]),
             "%s %% | %s %% | %d из %d" % (fmt(100 * h1 / cn, 1), fmt(100 * h2 / cn, 1), anti, len(GALLERY)))
        del conf_all, with_best


CTRL_BEST = {}
GALLERY = set()


def load_ctrl_best():
    path = FERET + "/xbase_feret_update_scores.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if head is None:
                head = p
                continue
            d = dict(zip(head, p))
            if d["role"] == "ctrl":
                CTRL_BEST.setdefault((d["proj"], d["gate"], d["variant"], int(d["step"])), []).append(d["best"])


if __name__ == "__main__":
    C = Cmp()
    degenerate(C)
    concentration(C)
    xs2(C)
    load_ctrl_best()
    update(C)
    head = "диагностические разделы: точки %d из %d совпали; не вычислимо: %d" % (C.ok_pt, C.n_pt, len(C.na))
    text = "\n".join([head, "", "РАСХОЖДЕНИЯ:"] + C.bad + ["", "НЕ ВЫЧИСЛИМО:"] + C.na + ["", "ВСЕ:"] + C.lines) + "\n"
    if len(sys.argv) > 1:
        open(sys.argv[1], "w", encoding="utf-8").write(text)
    print(head)
    for b in C.bad[:40]:
        print(b)
