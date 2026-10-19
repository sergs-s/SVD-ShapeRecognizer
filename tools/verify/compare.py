"""Сверка мер отчётов XBaseEval с независимым расчётом (measures.py).

python3 -I compare.py <набор> [B] – наборы: orl, muct, gt-subsp, gt-xs-f; пути – переменные VERIFY_DATA (SVD-faces-data).
Вывод – таблица «отчёт / Ахмед» по ячейкам и итог.
"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measures import (R0, Row, cp_upper, deff_of, detections_pose, evaluate, read_scores,  # noqa: E402
                      select_by_pose)

DATA = os.environ.get("VERIFY_DATA", "/home/user/svd-faces-data")
FARS = (0.01, 0.05)
WAYS = {"П2": "P2", "П3": "P3"}


def fmt(x, dec):
    """Округление половины вверх (как String.format в Java)."""
    q = Decimal(1).scaleb(-dec)
    return str(Decimal(x).quantize(q, rounding=ROUND_HALF_UP))


def fsign(x, dec):
    s = fmt(x, dec)
    return s if s.startswith("-") else "+" + s


# ---------- наборы ----------

def gt_pose():
    pose, _ = detections_pose(DATA + "/detections.tsv", "gt")
    return lambda p, i: pose[(p, "%02d" % int(i))]


def orl_pose():
    pose, _ = detections_pose(DATA + "/detections.tsv", "orl")
    return lambda p, i: pose[(p, i)]


def muct_pose():
    pose, order = detections_pose(DATA + "/detections.tsv", "muct")
    return lambda p, i: pose[(p, order[p][int(i) - 1])]


def dataset(name):
    """Строки отчёта: метка → (блоки файла, метод, фильтр s); отчёт; функция позы."""
    rows = {}
    if name == "orl":
        rep = DATA + "/reports/xbase_72c3195/xbase_orl.txt"
        b = read_scores(DATA + "/reports/xbase_72c3195/xbase_orl_scores.tsv.gz")[0][1]
        for lab, m in (("L1", "5-lbp-mlda"), ("L1-in", "5-lbp-mlda-in"), ("A", "2c-mlda")):
            rows[lab] = (b, m, None)
        return rows, rep, orl_pose()
    if name == "muct":
        d = DATA + "/reports/xbase_72c3195/"
        rep = d + "xbase_muct.txt"
        b = read_scores(d + "xbase_muct_scores.tsv.gz")[0][1]
        bf = read_scores(d + "xbase_muct_fei_scores.tsv.gz")[0][1]
        ba = read_scores(d + "xbase_muct_a_scores.tsv.gz")[0][1]
        rows["L1"] = (b, "5-lbp-mlda", None)
        rows["L1-in"] = (b, "5-lbp-mlda-in", None)
        rows["L1 (когорта FEI)"] = (bf, "5-lbp-mlda", None)
        rows["A (ротация 0)"] = (ba, "2c-mlda", None)
        rows["L1 (ротация 0)"] = (b, "5-lbp-mlda", {0})
        return rows, rep, muct_pose()
    if name == "gt-subsp":
        d = DATA + "/reports/xbase_f12513c/"
        rep = d + "xbase_gt_subsp.txt"
        for title, b in read_scores(d + "xbase_gt_subsp_scores.tsv.gz"):
            mode, proj, k = [x.strip() for x in title.split(",")]
            lab = "%s %s %s" % (mode.rstrip("."), proj.replace("проекция ", ""), k)
            lab = lab.replace("межс ", "межс. ")
            rows[lab] = (b, "5-lbp-mlda", None)
        return rows, rep, gt_pose()
    if name == "gt-xs-f":
        d = DATA + "/reports/xbase_1d78ea7/"
        rep = d + "xbase_gt_xs_f.txt"
        for title, b in read_scores(d + "xbase_gt_xs_f_scores.tsv.gz"):
            mode, var = [x.strip() for x in title.split(",")]
            lab = "%s %s" % (mode.replace("режим ", ""), var.replace("вариант ", ""))
            rows[lab] = (b, "5-lbp-mlda", None)
        return rows, rep, gt_pose()
    if name.startswith("feret-"):
        return feret(name[6:])
    raise SystemExit("набор: " + name)


FERET = os.environ.get("VERIFY_FERET", "/home/user/svd-feret-cloud")


def feret_pose():
    """Номер снимка в файле оценок – порядковый номер снимка человека в описи fa/fb (соглашение из кода:
    XBaseA0.java:562, imageNo); порядок описи – порядок строк derived/detections.tsv (проверяется по числам анфас/поворот)."""
    from measures import pose_r
    pose = {}
    seen = {}
    with open(FERET + "/derived/detections.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            person = p[0].split("_")[0]
            seen[person] = seen.get(person, 0) + 1
            pose[(person, str(seen[person]))] = pose_r([float(x) for x in p[2:17]]) if p[1] == "1" else None
    return lambda p, i: pose[(p, i)]


R0_MP = 5.2589


def mp_yaw():
    """yaw МП (градусы) по номеру снимка человека – как feret_pose; кэш derived/mp_points_crop3.tsv."""
    yaw = {}
    with open(FERET + "/derived/mp_points_crop3.tsv", encoding="utf-8") as f:
        head = None
        for line in f:
            p = line.rstrip("\n").split("\t")
            if line.startswith("#"):
                continue
            if head is None:
                head = p
                iy = head.index("yaw")
                continue
            yaw[p[0]] = float(p[iy]) if p[1] == "1" else None
    for v in yaw.values():
        if v is not None and abs(abs(v) - R0_MP) < 5e-5:
            raise SystemExit("yaw МП в пределах округления r0_МП")
    no = {}
    seen = {}
    with open(FERET + "/derived/detections.tsv", encoding="utf-8") as f:
        for line in f:
            key = line.split("\t", 1)[0]
            person = key.split("_")[0]
            seen[person] = seen.get(person, 0) + 1
            no[(person, str(seen[person]))] = yaw[key]
    return lambda p, i: no[(p, i)]


def feret(kind):
    """FERET: блоки по проходам объединяются в одну строку, конфигурация – (проход, s)."""
    rows = {}
    if kind == "xs2":
        f = FERET + "/xbase_feret_xs2_scores.tsv.gz"
        for title, b in read_scores(f):
            t = dict(x.strip().split(" ", 1) for x in title.split(","))
            rows.setdefault("%s %s" % (t["конфигурация"], t["строка"]), []).append(b)
    elif kind == "subsp":
        f = FERET + "/xbase_feret_subsp_scores.tsv.gz"
        for title, b in read_scores(f):
            role, proj, ps, k = [x.strip() for x in title.split(",")]
            rows.setdefault("%s %s %s" % (role, proj.replace("проекция ", ""), k), []).append(b)
    elif kind == "mp2":
        # Схема «МП»: выбор модели по |yaw МП| ≤ r0_МП (значение из шапки отчёта, 4 знака; снимков FERET с |yaw| в пределах
        # округления от него нет – проверяется ниже); схема «база» – r YuNet, r0 = 0,12. U = I (отчёт), поэтому (а) и (б) –
        # одни и те же попытки.
        f = FERET + "/xbase_feret_mp2_scores.tsv.gz"
        yp = mp_yaw()
        for title, b in read_scores(f):
            t = dict(x.strip().split(" ", 1) for x in title.split(","))
            for tab in ("(а)", "(б)"):
                rows.setdefault("%s %s %s %s" % (t["конфигурация"], t["строка"], t["схема"], tab), (t["схема"], []))[1].append(b)
        spec = {}
        for lab, (sch, v) in rows.items():
            if sch == "МП":
                spec[lab] = (tuple(v), "5-lbp-mlda", None, yp, R0_MP)
            else:
                spec[lab] = (tuple(v), "5-lbp-mlda", None)
        return spec, FERET + "/xbase_feret_mp2.txt", feret_pose()
    else:
        raise SystemExit("FERET: " + kind)
    spec = {lab: (tuple(v), "5-lbp-mlda", None) for lab, v in rows.items()}
    return spec, FERET + "/xbase_feret_%s.txt" % kind, feret_pose()


# ---------- отчёт ----------

def parse_report(path):
    """Разделы отчёта: имя раздела → {ключ строки: [ячейки]}."""
    sec = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("=== "):
                t = line.strip("= ")
                if t.startswith("Сводка"):
                    cur = "sum"
                elif t.startswith("Ошибка своих, разложение"):
                    cur = "dec"
                elif t.startswith("FAR контроля при П2"):
                    cur = "far"
                elif t.startswith("Разделимость: точка равных"):
                    cur = "eer"
                elif t.startswith("Парные разности разделимости"):
                    cur = "psep"
                elif t.startswith("Парные разности"):
                    cur = "pair"
                else:
                    cur = None
                continue
            if cur is None or " | " not in line or line.startswith("строка") or line.startswith("пара"):
                continue
            c = line.split(" | ")
            if cur in ("sum", "eer", "psep"):
                sec.setdefault(cur, {})[c[0]] = c[1:]
            elif cur == "pair":
                sec.setdefault(cur, {})[(c[0], c[1])] = c[2:]
            else:
                sec.setdefault(cur, {})[(c[0], c[1], c[2])] = c[3:]
    return sec


# ---------- расчёт ----------

def metrics(res):
    """Плоский вектор мер для бутстрепа (доли)."""
    v = {}
    for far in FARS:
        for way in ("P2", "P3", "sep"):
            x = res[(way, far)]
            e, n = x["err"], x["nown"]
            v[(way, far, "all")] = e["all"] / n["all"]
            v[(way, far, "fr")] = e["fr"] / n["fr"] if n["fr"] > 0 else np.nan
            v[(way, far, "tu")] = e["tu"] / n["tu"] if n["tu"] > 0 else np.nan
            for q in ("i", "ii", "iii", "iv"):
                v[(way, far, q)] = e[q] / n["all"]
            v[(way, far, "cand")] = (e["i"] + e["ii"]) / n["all"]
            v[(way, far, "far")] = x["cacc"] / x["cn"]
    v["EER"] = res["EER"][3]
    return v


def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run(name, B, seed=7042):
    spec, rep_path, pose_of = dataset(name)
    persons = set()
    for lab, (b, m, sf, *_) in spec.items():
        for part in (b if isinstance(b, tuple) else (b,)):
            for d in part:
                persons.add(d["person"])
    persons = sorted(persons)
    pidx = {p: i for i, p in enumerate(persons)}
    rows = {}
    for lab, (b, m, sf, *rest) in spec.items():
        po, r0 = rest if rest else (pose_of, R0)
        conf = {}
        for tag, part in enumerate(b if isinstance(b, tuple) else (b,)):
            for s_, v in select_by_pose(part, m, po, r0).items():
                conf[tag * 1000 + s_] = v
        if sf is not None:
            conf = {s: v for s, v in conf.items() if s in sf}
        rows[lab] = Row(conf, pidx, r0)
    ones = np.ones(len(persons), dtype=np.int64)
    point = {lab: evaluate(r, ones) for lab, r in rows.items()}
    rng = np.random.default_rng(seed)
    boot = {lab: [] for lab in rows}
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(persons), len(persons)), minlength=len(persons))
        for lab, r in rows.items():
            boot[lab].append(metrics(evaluate(r, w)))
    return spec, rep_path, rows, point, boot, persons


def ci(boot, lab, key, scale=100.0):
    a = [m[key] for m in boot[lab]]
    return pct(a, 2.5) * scale, pct(a, 97.5) * scale


def diff_ci(boot, a, b, key, scale=100.0):
    x = np.array([m[key] for m in boot[a]]) - np.array([m[key] for m in boot[b]])
    return pct(x, 2.5) * scale, pct(x, 97.5) * scale


class Cmp:
    def __init__(self):
        self.lines = []
        self.n_pt = self.ok_pt = 0
        self.n_ci = self.ok_ci = 0
        self.bad = []

    def pt(self, where, ref, ours):
        self.n_pt += 1
        ok = ref == ours
        self.ok_pt += ok
        if not ok:
            self.bad.append("ТОЧКА %s: отчёт «%s», Ахмед «%s»" % (where, ref, ours))
        self.lines.append("%s\t%s\t%s\t%s" % (where, ref, ours, "=" if ok else "≠"))

    def iv(self, where, ref_lo, ref_hi, lo, hi, tol=2.0):
        self.n_ci += 1
        d = max(abs(ref_lo - lo), abs(ref_hi - hi))
        ok = d <= tol
        self.ok_ci += ok
        if not ok:
            self.bad.append("ИНТЕРВАЛ %s: отчёт [%.2f; %.2f], Ахмед [%.2f; %.2f], Δ %.2f" % (where, ref_lo, ref_hi, lo, hi, d))
        self.lines.append("%s\t[%.2f; %.2f]\t[%.2f; %.2f]\t%s Δ %.2f" % (where, ref_lo, ref_hi, lo, hi, "ок" if ok else "ВНЕ", d))
        return d


def split_cell(c):
    """«x/n = p % [lo; hi] …» → (точка до «[», lo, hi, хвост)."""
    i = c.index("[")
    j = c.index("]")
    lo, hi = c[i + 1:j].split(";")
    return c[:i].strip(), float(lo), float(hi), c[j + 1:].strip()


def compare(name, B):
    spec, rep_path, rows, point, boot, persons = run(name, B)
    rep = parse_report(rep_path)
    C = Cmp()
    for lab, cells in rep.get("sum", {}).items():
        if lab not in rows:
            C.bad.append("строка отчёта без данных: " + lab)
            continue
        res = point[lab]
        k = 0
        for far in FARS:
            for way in ("P2", "P3", "sep"):
                x = res[(way, far)]
                e, n = x["err"]["all"], x["nown"]["all"]
                farc = 100 * x["cacc"] / x["cn"]
                if way == "sep":
                    ours = "%s %%" % fmt(100 * e / n, 1)
                else:
                    ours = "%d/%d = %s %%" % (e, n, fmt(100 * e / n, 1))
                p, lo, hi, tail = split_cell(cells[k])
                C.pt("%s | сводка %s %g%%" % (lab, way, far * 100), p + " " + tail, ours + " (" + fmt(farc, 2) + " %)")
                blo, bhi = ci(boot, lab, (way, far, "all"))
                C.iv("%s | сводка %s %g%%" % (lab, way, far * 100), lo, hi, blo, bhi)
                k += 1
        p, lo, hi, tail = split_cell(cells[6])
        C.pt("%s | EER" % lab, p, fmt(100 * res["EER"][3], 1) + " %")
        blo, bhi = ci(boot, lab, "EER")
        C.iv("%s | EER" % lab, lo, hi, blo, bhi)
    names = ["all", "fr", "tu", "i", "ii", "iii", "iv", "cand"]
    for (lab, way, far_s), cells in rep.get("dec", {}).items():
        res = point[lab]
        far = float(far_s.split()[1]) / 100
        w = WAYS[way]
        x = res[(w, far)]
        e, nn = x["err"], x["nown"]
        vals = [(e["all"], nn["all"]), (e["fr"], nn["fr"]), (e["tu"], nn["tu"]), (e["i"], nn["all"]), (e["ii"], nn["all"]),
                (e["iii"], nn["all"]), (e["iv"], nn["all"]), (e["i"] + e["ii"], nn["all"])]
        for q, (num, den), cell in zip(names, vals, cells):
            p, lo, hi, _ = split_cell(cell)
            ours = "%d/%d = %s %%" % (num, den, fmt(100 * num / den if den else 0, 1))
            C.pt("%s | разлож. %s %s %s" % (lab, way, far_s, q), p, ours)
            blo, bhi = ci(boot, lab, (w, far, q))
            C.iv("%s | разлож. %s %s %s" % (lab, way, far_s, q), lo, hi, blo, bhi)
    for (lab, way, far_s), cells in rep.get("far", {}).items():
        res = point[lab]
        far = float(far_s.split()[1]) / 100
        w = WAYS[way]
        x = res[(w, far)]
        cacc, cn = x["cacc"], x["cn"]
        p, lo, hi, tail = split_cell(cells[0])
        a = np.array([m[(w, far, "far")] for m in boot[lab]])
        up = pct(a, 95) * 100
        C.pt("%s | FAR контроля %s %s по снимкам" % (lab, way, far_s), p, "%d/%d = %s %%" % (cacc, cn, fmt(100 * cacc / cn, 2)))
        blo, bhi = ci(boot, lab, (w, far, "far"))
        C.iv("%s | FAR контроля %s %s по снимкам" % (lab, way, far_s), lo, hi, blo, bhi)
        ref_up = float(tail.replace("↑", "").replace("%", "").strip())
        C.iv("%s | FAR контроля %s %s ↑95" % (lab, way, far_s), ref_up, ref_up, up, up)
        acc, pid, wts = x["ctrl"]
        deff, neff = deff_of(acc, pid, wts)
        xe = (cacc / cn) * neff
        kp = cp_upper(xe, neff) if np.isfinite(deff) else 1.0
        C.pt("%s | FAR контроля %s %s КП_eff" % (lab, way, far_s), cells[1],
             "↑%s %% (%s; %s)" % (fmt(100 * kp, 2), fmt(deff, 2), fmt(neff, 1)))
        pa, pn = x["pers_acc"], x["pers_n"]
        C.pt("%s | FAR контроля %s %s по людям" % (lab, way, far_s), cells[2],
             "%d/%d = %s %% (↑%s %%)" % (pa, pn, fmt(100 * pa / pn, 1), fmt(100 * cp_upper(pa, pn), 1)))
        lim = 1.1 if far == 0.01 else 5.5
        C.pt("%s | FAR контроля %s %s допуск" % (lab, way, far_s), cells[3],
             "да" if float(fmt(100 * cacc / cn, 2)) <= lim else "нет")
    for lab, cells in rep.get("eer", {}).items():
        th, eo, ef, ee = point[lab]["EER"]
        C.pt("%s | EER θ, ошибка, FAR" % lab, " | ".join(cells[:3]),
             "%s | %s %% | %s %%" % (fmt(th, 6), fmt(100 * eo, 1), fmt(100 * ef, 2)))
    for (pair, way), cells in rep.get("pair", {}).items():
        a, b = pair.split(" − ")
        w = WAYS[way]
        keys = [(w, 0.01, "all", 1), (w, 0.01, "fr", 1), (w, 0.01, "tu", 1), (w, 0.05, "all", 1), (w, 0.01, "cand", 1),
                (w, 0.01, "far", 2), (w, 0.05, "far", 2)]
        for kk, cell in zip(keys, cells):
            key, dec = kk[:3], kk[3]
            pa_, pb_ = metrics(point[a])[key], metrics(point[b])[key]
            p, lo, hi, _ = split_cell(cell)
            C.pt("%s | пара %s %s" % (pair, way, key[1:]), p, fsign(100 * pa_ - 100 * pb_, dec))
            blo, bhi = diff_ci(boot, a, b, key)
            C.iv("%s | пара %s %s" % (pair, way, key[1:]), lo, hi, blo, bhi)
    for pair, cells in rep.get("psep", {}).items():
        a, b = pair.split(" − ")
        for key, cell in zip([("sep", 0.01, "all"), ("sep", 0.05, "all"), "EER"], cells):
            pa_, pb_ = metrics(point[a])[key], metrics(point[b])[key]
            p, lo, hi, _ = split_cell(cell)
            C.pt("%s | пара разд. %s" % (pair, key), p, fsign(100 * pa_ - 100 * pb_, 1))
            blo, bhi = diff_ci(boot, a, b, key)
            C.iv("%s | пара разд. %s" % (pair, key), lo, hi, blo, bhi)
    p3m0 = sum(point[l][(w, f)]["p3_m0"] for l in point for (w, f) in [("P3", 0.01), ("P3", 0.05)])
    return C, len(persons), p3m0


if __name__ == "__main__":
    name = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    C, npers, p3m0 = compare(name, B)
    out = sys.argv[3] if len(sys.argv) > 3 else None
    head = ["набор %s; бутстреп B = %d, людей %d; П3 с недопустимым m = 0: %d" % (name, B, npers, p3m0),
            "точки: %d из %d совпали; интервалы: %d из %d в допуске ±2 п. п." % (C.ok_pt, C.n_pt, C.ok_ci, C.n_ci)]
    text = "\n".join(head + ["", "РАСХОЖДЕНИЯ:"] + C.bad + ["", "ВСЕ ЯЧЕЙКИ (где | отчёт | Ахмед | итог):"] + C.lines) + "\n"
    if out:
        open(out, "w", encoding="utf-8").write(text)
    print("\n".join(head))
    for b in C.bad[:60]:
        print(b)
