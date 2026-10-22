"""Фиксированный порог SFace по 1 − cos (Жорик №6 от 22.10.2026, решение Хозяина (24)): по признакам VerifySfaceEmb, без детекций.

Правило – как в ПО: проба принята, если 1 − cos с лучшим шаблоном ≤ t, имя – лучший шаблон. Шаблон – нормированное среднее
нормированных признаков (SFaceScorer, GalleryEvaluation.train), наименьшее число снимков шаблона – как в протоколе базы (FERET 1,
прочие 2).

Базы и роли:
- FERET межс. и внутри – протокол feret, роль (1) (xbase_feret_scores.tsv.gz, блоки «набор SFace»): шаблоны – по снимкам галереи
  (попытки cv), свои – own, чужие – контроль (ctrl); сверка оценок own и ctrl с файлом (допуск 1e-6);
- GT межс. и внутри – файлы gt-diag (метод 0-sface), основной – xbase_2ed0021/eq9800, справочно – xbase_871f450; так же;
- FEI (кадры sface/fei – отобранные по fei_selection.tsv), MUCT, ORL, своя база (справочно; деление по моментам own_moments.tsv):
  у человека шаблон – ⌈n/2⌉ снимков (случайно), пробы – остальные; чужой – проба против шаблонов прочих людей.

Галерея из N людей – случайные подвыборки (REPS, seed – numpy default_rng(SEED_G + 1000·номер базы-роли + N)); чужие при ней – у
FERET и GT контроль, у прочих – пробы людей вне подвыборки; «все» – вся галерея протокола (у баз с половинами – все прочие люди,
N = людей − 1).

python3 -I sface_thr.py <признаки FERET> <признаки прочих баз> <файл оценок FERET> <каталог отчётов данных> <вывод .md> <вывод .tsv>
"""
import gzip
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_check  # noqa: E402
from measures import cp_upper  # noqa: E402

SEED_HALF = 17100  # default_rng(SEED_HALF + номер базы) – деление на половины
SEED_G = 17200
REPS = 200
GRID = np.round(np.arange(0.300, 0.80001, 0.005), 3)
NS = (10, 50, 100, 200)
EXTRA = (500, 1000)
TARGETS = (0.01, 0.001)
DEMO_COS = 0.363  # opencv_zoo models/face_recognition_sface/sface.py:24 (_threshold_cosine), main 48b10fa
DEMO_L2 = 1.128  # там же, строка 25 (_threshold_norml2); для единичных векторов 1 − cos = l2² / 2
TWINS = ("247", "248")


def load_emb(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            out.setdefault(c[0], {})[c[1]] = np.array([float(x) for x in c[2:]])
    return out


def template(vs):
    t = np.sum(vs, axis=0)
    return t / np.linalg.norm(t)


def blocks(path):
    """{заголовок: строки-словари}; заголовок – последняя строка «# …» перед строками (у файлов gt-diag – шапка)."""
    out = {}
    cols = None
    title = ""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                title = line[2:]
                continue
            if line.startswith("method\t"):
                cols = line.split("\t")
                continue
            out.setdefault(title, []).append(dict(zip(cols, line.split("\t"))))
    return out


class Data:
    """Шаблоны T (людей P), пробы X с людьми px; kind – ctrl (чужие – контроль) или half (чужие – пробы прочих людей)."""

    def __init__(self, name, main, kind, tp, T, px, X, is_ctrl, note=""):
        self.name, self.main, self.kind, self.note = name, main, kind, note
        self.tp = list(tp)
        self.T = np.array(T)
        self.px = np.array(px)
        self.X = np.array(X)
        self.is_ctrl = np.array(is_ctrl, dtype=bool)
        self.D = 1.0 - self.X @ self.T.T
        idx = {p: i for i, p in enumerate(self.tp)}
        self.pi = np.array([idx.get(p, -1) for p in self.px])  # номер своего шаблона (−1 – чужой контроля)


def from_scores(name, main, rows, emb, keyf, mini):
    """Роль из файла оценок: шаблоны – по попыткам cv, свои – own, чужие – ctrl; сверка оценок."""
    gal = {}
    for d in rows:
        if d["role"].startswith("cv"):
            gal.setdefault(d["person"], []).append(emb[keyf(d["person"], d["image"])])
    tp = sorted(gal)
    T = []
    for p in tp:
        if len(gal[p]) < mini:
            raise SystemExit("%s: у %s снимков шаблона %d < %d" % (name, "человека", len(gal[p]), mini))
        T.append(template(gal[p]))
    px, X, ic, ref = [], [], [], []
    for d in rows:
        if d["role"] in ("own", "ctrl"):
            px.append(d["person"])
            X.append(emb[keyf(d["person"], d["image"])])
            ic.append(d["role"] == "ctrl")
            ref.append(d)
    ds = Data(name, main, "ctrl", tp, T, px, X, ic)
    dev = 0.0
    bad = 0
    for i, d in enumerate(ref):
        s = ds.D[i]
        o = np.argsort(s, kind="stable")
        if tp[o[0]] != d["best_id"]:
            bad += 1
        dev = max(dev, abs(s[o[0]] - float(d["best_score"])), abs(s[o[1]] - float(d["second_score"])))
        if d["role"] == "own":
            dev = max(dev, abs(s[tp.index(d["person"])] - float(d["true_score"])))
    ds.check = (len(ref), bad, dev)
    return ds


def duplicates(emb):
    """Пары «дубль личности» (правило вслепую): медиана 1 − cos между снимками двух людей ≤ наибольшей по людям базы медианы
    1 − cos пар снимков одного человека. → [(медиана, человек a, человек b)], порог правила."""
    ks = sorted(emb)
    X = np.array([emb[k] for k in ks])
    P = np.array([k.split("/")[0] for k in ks])
    D = 1.0 - X @ X.T
    ps = sorted(set(P))
    idx = {p: np.where(P == p)[0] for p in ps}
    own = max(float(np.median(D[np.ix_(idx[p], idx[p])][np.triu_indices(len(idx[p]), 1)])) for p in ps if len(idx[p]) > 1)
    out = []
    for a in range(len(ps)):
        for b in range(a + 1, len(ps)):
            m = float(np.median(D[np.ix_(idx[ps[a]], idx[ps[b]])]))
            if m <= own:
                out.append((m, ps[a], ps[b]))
    return out, own


def halves(name, main, emb, person_of, unit_of, seed, mini=2, note=""):
    """Шаблон – ⌈n/2⌉ единиц (снимков или моментов) человека, случайно; пробы – остальные."""
    rng = np.random.default_rng(seed)
    by = {}
    for k in sorted(emb):
        by.setdefault(person_of(k), {}).setdefault(unit_of(k), []).append(k)
    tp, T, px, X = [], [], [], []
    skipped = []
    for p in sorted(by):
        units = sorted(by[p])
        n = len(units)
        h = (n + 1) // 2
        timg = sum(len(by[p][u]) for u in units)
        if n - h < 1 or (h < mini and sum(len(by[p][u]) for u in units[:h]) < mini):
            skipped.append((p, timg))
            continue
        perm = rng.permutation(n)
        tu = [units[j] for j in perm[:h]]
        pu = [units[j] for j in perm[h:]]
        tk = [k for u in tu for k in by[p][u]]
        if len(tk) < mini:
            skipped.append((p, timg))
            continue
        tp.append(p)
        T.append(template([emb[k] for k in tk]))
        for u in pu:
            for k in by[p][u]:
                px.append(p)
                X.append(emb[k])
    ds = Data(name, main, "half", tp, T, px, X, [False] * len(px), note)
    ds.skipped = skipped
    ds.nimg = sum(len(v) for b in by.values() for v in b.values())
    return ds


def run_gallery(ds, G):
    """Галерея G (номера шаблонов; None – вся). → (оценки чужих, люди чужих, own: лучшая, чужое имя, своя)."""
    if G is None:
        if ds.kind == "ctrl":
            Ds = ds.D[ds.is_ctrl]
            sm = Ds.min(axis=1)
            sp = ds.px[ds.is_ctrl]
        else:
            Dm = ds.D.copy()
            Dm[np.arange(len(Dm)), ds.pi] = np.inf
            sm = Dm.min(axis=1)
            sp = ds.px
        own = ~ds.is_ctrl
        Do = ds.D[own]
        best = Do.min(axis=1)
        wrong = Do.argmin(axis=1) != ds.pi[own]
        true = Do[np.arange(len(Do)), ds.pi[own]]
        return sm, sp, best, wrong, true
    inG = np.zeros(len(ds.tp), dtype=bool)
    inG[G] = True
    if ds.kind == "ctrl":
        sel = ds.is_ctrl
    else:
        sel = ~inG[ds.pi]
    sm = ds.D[sel][:, G].min(axis=1)
    sp = ds.px[sel]
    own = (~ds.is_ctrl) & (ds.pi >= 0) & inG[np.maximum(ds.pi, 0)]
    Do = ds.D[own][:, G]
    gpos = {g: j for j, g in enumerate(G)}
    tj = np.array([gpos[i] for i in ds.pi[own]])
    best = Do.min(axis=1)
    wrong = Do.argmin(axis=1) != tj
    true = Do[np.arange(len(Do)), tj]
    return sm, sp, best, wrong, true


def curves(sm, sp, best, wrong, true):
    """По сетке: принятых чужих попыток, n, принятых чужих людей, людей, ошибка своих, чужое имя принято."""
    s = np.sort(sm)
    k = np.searchsorted(s, GRID, side="right")
    pm = {}
    for v, p in zip(sm, sp):
        pm[p] = min(pm.get(p, np.inf), v)
    ps = np.sort(np.array(list(pm.values())))
    kp = np.searchsorted(ps, GRID, side="right")
    err = np.array([np.mean(wrong | (true > t)) if len(true) else np.nan for t in GRID])
    wn = np.array([np.mean(wrong & (best <= t)) if len(true) else np.nan for t in GRID])
    return k, len(s), kp, len(ps), err, wn


def cp_vec(k, n):
    return np.array([cp_upper(int(x), n) for x in k])


def pair_far(ds):
    if ds.kind == "ctrl":
        v = ds.D[ds.is_ctrl].ravel()
    else:
        M = np.ones_like(ds.D, dtype=bool)
        M[np.arange(len(ds.D)), ds.pi] = False
        v = ds.D[M]
    s = np.sort(v)
    return np.searchsorted(s, GRID, side="right") / len(s), v


def fq(v):
    return " / ".join("%.3f" % float(np.quantile(v, p, method="inverted_cdf")) for p in (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1))


def main(emb_feret, emb_bases, feret_scores, rep_dir, out_md, out_tsv):
    ef = load_emb(emb_feret)["feret"]
    eb = load_emb(emb_bases)
    keys = abl_check.feret_keys()
    fb = blocks(feret_scores)
    dss = []
    for role, t in (("межс.", "конфигурация межс., набор SFace"), ("внутри", "конфигурация внутри, набор SFace")):
        dss.append(from_scores("FERET " + role, True, fb[t], ef, lambda p, i: keys[(p, i)], 1))
    gt = eb["gt"]
    gk = lambda p, i: "%s/%02d" % (p, int(i))  # noqa: E731
    for sub, main_ in (("xbase_2ed0021/eq9800", True), ("xbase_871f450", False)):
        for role, fn in (("межс.", "xbase_gt_diag_scores.tsv.gz"), ("внутри", "xbase_gt_in_diag_scores.tsv.gz")):
            rows = [d for v in blocks(os.path.join(rep_dir, sub, fn)).values() for d in v if d["method"] == "0-sface"]
            dss.append(from_scores("GT %s%s" % (role, "" if main_ else " (871f450)"), main_, rows, gt, gk, 2))
    first = lambda k: k.split("/")[0]  # noqa: E731
    ident = lambda k: k  # noqa: E731
    moments = {}
    with open(os.path.join(os.path.dirname(rep_dir.rstrip("/")), "own_moments.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            moments["%s/%s" % (c[0], c[2])] = c[1]
    dup = {}
    for b, (base, main_, unit) in enumerate((("fei", True, ident), ("muct", True, ident), ("orl", True, ident),
                                             ("own", False, lambda k: moments[k]))):
        nm = {"fei": "FEI", "muct": "MUCT", "orl": "ORL", "own": "своя база"}[base]
        dp, lim = duplicates(eb[base])
        dup[nm] = (dp, lim)
        drop = {x[2] for x in dp}
        e = {k: v for k, v in eb[base].items() if first(k) not in drop}
        dss.append(halves(nm, main_, e, first, unit, SEED_HALF + b, note="моменты" if base == "own" else ""))
        if drop:
            dss.append(halves(nm + " (как есть, с дублями)", False, eb[base], first, unit, SEED_HALF + b))
    muct = [d for d in dss if d.name == "MUCT"][0]
    # MUCT без близнецов – справочно
    keep = [i for i, p in enumerate(muct.tp) if p not in TWINS]
    pk = np.isin(muct.px, [p for p in muct.tp if p not in TWINS])
    m2 = Data("MUCT без 247, 248", False, "half", [muct.tp[i] for i in keep], muct.T[keep], muct.px[pk], muct.X[pk], [False] * int(pk.sum()))
    dss.append(m2)

    L = ["# Фиксированный порог SFace по 1 − cos (Жорик №6 от 22.10.2026) – Ахмед", ""]
    L.append("Сетка t – %.3f…%.3f, шаг 0,005; подвыборок галереи %d, seed – numpy default_rng(%d + 1000·номер строки + N); "
             "половины – default_rng(%d + номер базы: FEI 0, MUCT 1, ORL 2, своя 3)." % (GRID[0], GRID[-1], REPS, SEED_G, SEED_HALF))
    L.append("")
    L.append("## 0. Данные и сверка")
    L.append("")
    L.append("| строка | в выборе | людей галереи | своих попыток | чужих попыток (вся галерея) | чужих людей | сверка: попыток / иной лучший / макс. |Δ| |")
    L.append("|---|---|---|---|---|---|---|")
    full = {}
    for ds in dss:
        sm, sp, best, wrong, true = run_gallery(ds, None)
        full[ds.name] = (sm, sp, best, wrong, true)
        chk = "%d / %d / %.1e" % ds.check if hasattr(ds, "check") else "–"
        L.append("| %s | %s | %d | %d | %d | %d | %s |" % (ds.name, "да" if ds.main else "справочно", len(ds.tp), len(true), len(sm),
                                                         len(set(sp)), chk))
    L.append("")
    for nm, (dp, lim) in dup.items():
        L.append("- %s: дубли личности (медиана 1 − cos между людьми ≤ %.3f – наибольшей медианы своих пар): %s." % (
            nm, lim, ", ".join("%s и %s (%.3f; второй исключён из основной строки)" % (a, b, m) for m, a, b in dp) or "нет"))
    for ds in dss:
        if hasattr(ds, "skipped"):
            L.append("- %s: снимков %d; без шаблона или без проб (мало снимков) – людей %d%s." % (
                ds.name, ds.nimg, len(ds.skipped), " (" + ", ".join("%d сн." % n for _, n in ds.skipped) + ")" if ds.skipped else ""))
    L.append("")

    # 2. распределения
    L.append("## 2. Распределения 1 − cos (вся галерея): квантили 0 / 1 / 5 / 50 / 95 / 99 / 100 %")
    L.append("")
    L.append("| строка | свои (к своему шаблону) | чужие – к лучшему шаблону | чужие – пары «проба – чужой шаблон» | зазор | чужое имя |")
    L.append("|---|---|---|---|---|---|")
    pairs = {}
    for ds in dss:
        sm, sp, best, wrong, true = full[ds.name]
        pf, pv = pair_far(ds)
        pairs[ds.name] = pf
        L.append("| %s | %s | %s | %s | %.3f | %d из %d |" % (ds.name, fq(true), fq(sm), fq(pv), sm.min() - true.max(), wrong.sum(), len(wrong)))
    L.append("")
    tw = [(muct.px[i], muct.tp[j], muct.D[i, j]) for i in range(len(muct.px)) for j in range(len(muct.tp))
          if muct.px[i] in TWINS and muct.tp[j] in TWINS and muct.px[i] != muct.tp[j]]
    tws = np.array([x[2] for x in tw])
    L.append("Близнецы MUCT 247 и 248: проба одного против шаблона другого – %d попыток, 1 − cos %s (мин. / медиана / макс.)." % (
        len(tws), " / ".join("%.3f" % v for v in (tws.min(), np.median(tws), tws.max()))))
    L.append("")

    # 3. сетка по N
    res = {}  # (имя, N) → словарь массивов
    for di, ds in enumerate(dss):
        P = len(ds.tp)
        nmax = P if ds.kind == "ctrl" else P - 1
        for N in NS + ("все",):
            if N == "все":
                k, n, kp, npp, err, wn = curves(*full[ds.name])
                res[(ds.name, N)] = dict(N=nmax, n=n, npp=npp, far=k / n, cp=cp_vec(k, n), farp=kp / npp, cpp=cp_vec(kp, npp),
                                         err=err, wn=wn, kmed=k)
                continue
            if N >= nmax:
                continue
            rng = np.random.default_rng(SEED_G + 1000 * di + N)
            K, C, KP, CPP, E, W, NN = [], [], [], [], [], [], []
            for _ in range(REPS):
                G = np.sort(rng.choice(P, size=N, replace=False))
                k, n, kp, npp, err, wn = curves(*run_gallery(ds, G))
                K.append(k / n)
                C.append(cp_vec(k, n))
                KP.append(kp / npp)
                CPP.append(cp_vec(kp, npp))
                E.append(err)
                W.append(wn)
                NN.append(n)
            res[(ds.name, N)] = dict(N=N, n=int(np.median(NN)), npp=None, far=np.median(K, axis=0), cp=np.median(C, axis=0),
                                     farp=np.median(KP, axis=0), cpp=np.median(CPP, axis=0), err=np.median(E, axis=0),
                                     wn=np.median(W, axis=0), kmed=np.median(np.array(K) * np.array(NN)[:, None], axis=0))
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("# Сетка t: строка, N (число людей галереи), чужих попыток (медиана), t, FAR попыток (медиана), его ↑95 КП (медиана),"
                " FAR людей, ↑95, ошибка своих, чужое имя принято, FAR пары, оценка 1−(1−p)^N\n")
        f.write("row\tN\tn\tt\tfar\tfar_cp\tfar_person\tfar_person_cp\town_err\twrong_name\tpair_far\tfar_from_pair\n")
        for (name, Nk), r in res.items():
            for j, t in enumerate(GRID):
                p = pairs[name][j]
                f.write("%s\t%s\t%d\t%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (
                    name, Nk if Nk == "все" else Nk, r["n"], t, r["far"][j], r["cp"][j], r["farp"][j], r["cpp"][j], r["err"][j], r["wn"][j],
                    p, 1 - (1 - p) ** r["N"]))

    def tmax(name, Nk, target, zero_ok=False):
        """Наибольший t сетки с ↑95 КП FAR попыток ≤ цели (None – нет такого); zero_ok – при n < нужного: медиана принятых = 0."""
        r = res[(name, Nk)]
        ok = r["cp"] <= target + 1e-12
        if zero_ok and math.log(0.05) / math.log(1 - target) > r["n"]:
            ok = r["kmed"] == 0
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return None
        # монотонность: берём последний t, до которого все подряд удовлетворяют
        j = 0
        while j < len(ok) and ok[j]:
            j += 1
        return GRID[j - 1] if j > 0 else None

    def gidx(t):
        return int(np.argmin(np.abs(GRID - t)))

    L.append("## 3. Сетка t (полностью – `sface_thr_grid.tsv`)")
    L.append("")
    L.append("Справочно – порог демонстрационного кода opencv_zoo (models/face_recognition_sface/sface.py, строки 24–25, ветка main "
             "48b10fa): cos ≥ %.3f → 1 − cos ≤ %.3f; по L2 ≤ %.3f → 1 − cos ≤ %.4f (для нормированных признаков)." % (
                 DEMO_COS, 1 - DEMO_COS, DEMO_L2, DEMO_L2 ** 2 / 2))
    L.append("")
    L.append("Выдержка: t = 0,30…0,70 через 0,05 и порог демо. FAR попыток – медиана по подвыборкам, ↑95 – медиана верхних границ "
             "Клоппера – Пирсона; «из пары» – 1 − (1 − p)^N по FAR одной пары p.")
    L.append("")
    show = [round(x, 3) for x in np.arange(0.30, 0.701, 0.05)] + [0.635]
    for ds in dss:
        L.append("### %s" % ds.name)
        L.append("")
        L.append("| N | чужих попыток | t | FAR попыток, % [↑95] | из пары, % | FAR людей, % [↑95] | ошибка своих, % | чужое имя принято, % |")
        L.append("|---|---|---|---|---|---|---|---|")
        for Nk in NS + ("все",) + EXTRA:
            if Nk in EXTRA:
                for t in show:
                    p = pairs[ds.name][gidx(t)]
                    L.append("| %d (экстраполяция) | – | %.3f | – | %.2f | – | – | – |" % (Nk, t, 100 * (1 - (1 - p) ** Nk)))
                continue
            if (ds.name, Nk) not in res:
                continue
            r = res[(ds.name, Nk)]
            for t in show:
                j = gidx(t)
                p = pairs[ds.name][j]
                L.append("| %s | %d | %.3f | %.2f [%.2f] | %.2f | %.2f [%.2f] | %.1f | %.2f |" % (
                    "все (%d)" % r["N"] if Nk == "все" else Nk, r["n"], t, 100 * r["far"][j], 100 * r["cp"][j], 100 * (1 - (1 - p) ** r["N"]),
                    100 * r["farp"][j], 100 * r["cpp"][j], 100 * r["err"][j], 100 * r["wn"][j]))
        L.append("")

    # 4. выбор
    mains = [d.name for d in dss if d.main]
    L.append("## 4. Выбор t*(N): наибольший t, при котором ↑95 FAR попыток ≤ цели на всех основных строках")
    L.append("")
    L.append("Основные строки: %s. Справочно (не в выборе): %s." % (", ".join(mains), ", ".join(d.name for d in dss if not d.main)))
    L.append("Строка, у которой чужих попыток меньше, чем нужно для ↑95 ≤ цели при нуле принятых (299 для 1 %, 2995 для 0,1 %), "
             "в основной выбор не входит («недостижимо»); вариант t*₀ включает её с условием «ни одного принятого» (медиана).")
    L.append("")
    choice = {}
    for target in TARGETS:
        need = math.ceil(math.log(0.05) / math.log(1 - target))
        L.append("### Цель %.1f %% (нужно ≥ %d чужих попыток)" % (100 * target, need))
        L.append("")
        L.append("| N | t*(N) | определяет | t*₀(N) | определяет | недостижимо (попыток) | нет N в базе |")
        L.append("|---|---|---|---|---|---|---|")
        for Nk in NS:
            lim, lim0, una, absent = {}, {}, [], []
            for name in mains:
                if (name, Nk) not in res:
                    absent.append(name)
                    continue
                r = res[(name, Nk)]
                if r["n"] < need:
                    una.append("%s (%d)" % (name, r["n"]))
                else:
                    lim[name] = tmax(name, Nk, target)
                lim0[name] = tmax(name, Nk, target, zero_ok=True)

            def best_of(d):
                if not d:
                    return None, "–"
                if any(v is None for v in d.values()):
                    return None, ", ".join(k for k, v in d.items() if v is None) + " – ниже %.3f (край сетки)" % GRID[0]
                m = min(d.values())
                return m, ", ".join(k for k, v in d.items() if v == m)
            t1, w1 = best_of(lim)
            t0, w0 = best_of(lim0)
            choice[(target, Nk)] = (t1, t0, lim, lim0)
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (Nk, "%.3f" % t1 if t1 is not None else "нет", w1, "%.3f" % t0 if t0 is not None else "нет",
                                                          w0, ", ".join(una) or "–", ", ".join(absent) or "–"))
        L.append("")
    # пределы по строкам и ошибка своих при t*
    L.append("### Предел t по каждой строке и ошибка своих при t*(N), цель 1 %")
    L.append("")
    L.append("Предел строки – наибольший t с ↑95 ≤ 1 % (при n < 299 – с нулём принятых, помечено «₀»). Ошибка своих – медиана по "
             "подвыборкам галереи из N людей и при всей галерее строки.")
    L.append("")
    L.append("| N | t*₀(N) | строка | предел строки | наименьший 1 − cos чужих (вся галерея) | запас: мин. чужих − t* | ошибка своих при t*, N / вся галерея, % | чужое имя принято, % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for Nk in NS:
        t1, t0, lim, lim0 = choice[(0.01, Nk)]
        if t0 is None:
            continue
        for ds in dss:
            if (ds.name, Nk) not in res:
                continue
            r = res[(ds.name, Nk)]
            ra = res[(ds.name, "все")]
            j = gidx(t0)
            sm = full[ds.name][0]
            small = r["n"] < 299
            lm = tmax(ds.name, Nk, 0.01, zero_ok=small)
            L.append("| %s | %.3f | %s%s | %s | %.3f | %+.3f | %.1f / %.1f | %.2f |" % (
                Nk, t0, ds.name, "" if ds.main else " (справ.)", ("%.3f" % lm if lm is not None else "< %.3f" % GRID[0]) + ("₀" if small else ""),
                sm.min(), sm.min() - t0, 100 * r["err"][j], 100 * ra["err"][j], 100 * r["wn"][j]))
    L.append("")
    L.append("### Вся галерея строки (N – своё у каждой строки), цель 1 %")
    L.append("")
    L.append("| строка | N | чужих попыток | предел строки | ошибка своих при пределе, % | FAR попыток при t*₀(50), % [↑95] |")
    L.append("|---|---|---|---|---|---|")
    t50 = choice[(0.01, 50)][1]
    for ds in dss:
        r = res[(ds.name, "все")]
        small = r["n"] < 299
        lm = tmax(ds.name, "все", 0.01, zero_ok=small)
        j50 = gidx(t50)
        L.append("| %s%s | %d | %d | %s | %s | %.2f [%.2f] |" % (
            ds.name, "" if ds.main else " (справ.)", r["N"], r["n"], ("%.3f" % lm if lm is not None else "< %.3f" % GRID[0]) + ("₀" if small else ""),
            "%.1f" % (100 * r["err"][gidx(lm)]) if lm is not None else "–", 100 * r["far"][j50], 100 * r["cp"][j50]))
    L.append("")
    # переносимость
    L.append("### Переносимость: t*(N) по основным строкам без одной базы, проверка на исключённой (цель 1 %, вариант t*₀)")
    L.append("")
    L.append("| N | исключена | t* по остальным | FAR попыток исключённой, % [↑95] | в допуске (↑95 ≤ 1 % или 0 принятых при малом n) |")
    L.append("|---|---|---|---|---|")
    bases = sorted({n.split(" ")[0] for n in mains})
    for Nk in NS:
        _, _, _, lim0 = choice[(0.01, Nk)]
        for b in bases:
            rest = {k: v for k, v in lim0.items() if not k.startswith(b + " ") and k != b}
            excl = [k for k in lim0 if k.startswith(b + " ") or k == b]
            if not rest or not excl or any(v is None for v in rest.values()):
                continue
            t = min(rest.values())
            for name in excl:
                r = res[(name, Nk)]
                j = gidx(t)
                need = math.ceil(math.log(0.05) / math.log(0.99))
                ok = r["cp"][j] <= 0.01 + 1e-12 if r["n"] >= need else r["kmed"][j] == 0
                L.append("| %s | %s | %.3f | %.2f [%.2f] | %s |" % (Nk, name, t, 100 * r["far"][j], 100 * r["cp"][j], "да" if ok else "**нет**"))
    L.append("")
    # экстраполяция
    L.append("### Оценка по FAR одной пары: наибольший t, при котором 1 − (1 − p)^N ≤ 1 %")
    L.append("")
    L.append("| N | " + " | ".join(d.name for d in dss if d.main) + " | все основные | без MUCT | t*₀(N) по подвыборкам (сверка) |")
    L.append("|---|" + "---|" * (len(mains) + 3))
    for Nk in NS + EXTRA:
        lim = {}
        for name in mains:
            p = pairs[name]
            ok = 1 - (1 - p) ** Nk <= 0.01
            j = 0
            while j < len(ok) and ok[j]:
                j += 1
            lim[name] = GRID[j - 1] if j > 0 else None
        fmt = lambda v: "%.3f" % v if v is not None else "< %.3f" % GRID[0]  # noqa: E731
        allm = None if any(v is None for v in lim.values()) else min(lim.values())
        nom = [v for k, v in lim.items() if k != "MUCT"]
        nom = None if any(v is None for v in nom) else min(nom)
        ch = choice.get((0.01, Nk))
        L.append("| %d%s | %s | %s | %s | %s |" % (Nk, " (экстраполяция)" if Nk in EXTRA else "", " | ".join(fmt(lim[n]) for n in mains), fmt(allm),
                                              fmt(nom), "%.3f" % ch[1] if ch and ch[1] is not None else "–"))
    L.append("")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main(*sys.argv[1:7])
