"""Независимый расчёт мер по файлам оценок (задача №1 от 19.10.2026, (б)).

Определения – CLAUDE.md и PLAN.md; соглашения, взятые из кода, помечены «соглашение из кода» с файлом и строкой.
Только numpy и scipy.special; Java-код мер не используется.
"""
import gzip
import math

import numpy as np
from scipy.special import betaincinv

RHO_MIN = 1e-12
R0 = 0.12
# Вариант для разбора (найден после расхождения интервалов, сверен с FeiThreshold.java:117, 121, 131–133): поиск порога П3
# идёт по всем попыткам, включая людей с нулевым весом в реплике бутстрепа (их оценки задают места θ, в том числе
# θ = nextDown(r[0]) при недопустимом m = 0). По определению (люди вытягиваются целиком) – только попытки с весом > 0.
P3_ZERO_W = False
# Вариант для разбора (п. 2 Жорика №4, 20.10): EER в реплике без своих или без контрольных – как в коде, заготовка
# {−∞, 100 %, 0 %, 50 %} (XBaseSep.java:157, 171–175: 0/0 = NaN, «gap < bestGap» ложно); по умолчанию – NaN.
EER_EMPTY_50 = False
# Вариант для разбора (Жорик №5, 20.10): выбор точки EER при равных |ошибка − FAR| как в коде – по процентам в плавающей
# точке, первое строгое «<»; по умолчанию – доли и точное равенство (первый θ среди равных с допуском 1e-12).
EER_JAVA_FLOAT = False


def ratio(d1, d2):
    """r = d₁/d₂ по правилу вырожденных случаев (CLAUDE.md, «Правило вырожденных случаев»)."""
    if math.isnan(d1) or math.isnan(d2):
        raise ValueError("NaN")
    if d2 == math.inf:
        return math.inf
    if d1 < 0 and d1 >= -1e-12:
        d1 = 0.0
    if d2 < 0 and d2 >= -1e-12:
        d2 = 0.0
    if d1 < 0 or d2 < 0 or d1 > d2:
        raise ValueError("d1=%r d2=%r" % (d1, d2))
    if d1 == 0 and d2 == 0:
        return 1.0
    if d2 == 0:
        raise ValueError("d2=0")
    r = d1 / d2
    return max(r, RHO_MIN)


def read_scores(path):
    """Блоки файла оценок: [(заголовок блока или None, [строки-словари])]."""
    blocks = []
    header = None
    title = None
    cur = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("#"):
                if header is None:
                    continue  # шапка файла до строки столбцов
                if cur or title is not None:
                    blocks.append((title, cur))
                title = line[1:].strip()
                cur = []
                continue
            if header is None:
                header = line.split("\t")
                continue
            cur.append(dict(zip(header, line.split("\t"))))
    blocks.append((title, cur))
    return [b for b in blocks if b[1]]


def detections_pose(path, base):
    """(person, name) → r позы (смещение носа от середины глаз в долях межглазья по x, точки YuNet)."""
    out = {}
    order = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] != base:
                continue
            order.setdefault(p[1], []).append(p[2])
            if p[4] != "1":
                out[(p[1], p[2])] = None
                continue
            v = [float(x) for x in p[5:20]]
            out[(p[1], p[2])] = pose_r(v)
    return out, order


def pose_r(v):
    """v – 15 чисел YuNet: x, y, w, h, глаз справа на лице (re), глаз слева (le), нос, углы рта, score."""
    rex, lex, nx = v[4], v[6], v[8]
    return (nx - (rex + lex) / 2) / (lex - rex)


class Attempts:
    """Попытки одной строки отчёта после выбора модели по позе: по конфигурациям s."""

    def __init__(self):
        self.conf = {}  # s → dict роль → списки


def select_by_pose(rows, method, pose_of, r0=R0):
    """Выбор модели X / X+mirror по |r| пробы; возвращает {s: {own|ctrl|cv: [(r, wrong, person, pose)]}}."""
    by = {}
    for d in rows:
        m = d["method"]
        if m != method and m != method + "+mirror":
            continue
        key = (d["s"], d["j"], d["role"], d["person"], d["image"])
        by.setdefault(key, {})[m] = d
    out = {}
    for key, pair in by.items():
        s, j, role, person, image = key
        if len(pair) != 2:
            raise ValueError("нет пары X / X+mirror: %r" % (key,))
        pr = pose_of(person, image)
        d = pair[method] if abs(pr) <= r0 else pair[method + "+mirror"]
        if d["no_agreement"] != "-":
            raise ValueError("no_agreement: %r" % (key,))
        r = ratio(float(d["best_score"]), float(d["second_score"]))
        wrong = d["best_id"] != d["true_id"]
        rr = "cv" if role.startswith("cv") else role
        out.setdefault(int(s), {"own": [], "ctrl": [], "cv": []})[rr].append((r, wrong, person, pr))
    return out


# ---------- пороги ----------

def np_threshold(sorted_r, sorted_w, far):
    """Нейман – Пирсон: наибольший θ, при котором принято не больше ⌊FAR·n⌋ (вес) чужих.

    Соглашение из кода: θ = nextDown(r[m]) по возрастанию, m = n – +∞ (FeiScores.java:37, 601; FeiThreshold.java:102).
    """
    n = sorted_w.sum()
    allowed = math.floor(far * n)
    cum = np.cumsum(sorted_w)
    i = int(np.searchsorted(cum, allowed, side="right"))
    if i >= len(sorted_r):
        return math.inf
    return float(np.nextafter(sorted_r[i], -np.inf))


def cp_upper(x, n, conf=0.95):
    """Верхняя односторонняя граница Клоппера – Пирсона (x, n – могут быть дробными)."""
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(betaincinv(x + 1.0, n - x, conf))


def p3_threshold(sorted_r, sorted_w, sorted_pid, pid_weight, pid_n, far):
    """П3: КП по n_eff = n/deff; соглашения из кода – FeiThreshold.java:54, 75–78, 88, 100–102, 117.

    sorted_pid – номер человека у попытки; pid_weight – вес человека (бутстреп), pid_n – число его попыток (без веса).
    """
    pos = sorted_w > 0
    if P3_ZERO_W:
        pos = np.ones(len(sorted_w), dtype=bool)
    r = sorted_r[pos]
    pid = sorted_pid[pos]
    w = sorted_w[pos]
    n = float(w.sum())
    present = pid_weight > 0
    k = float(pid_weight[present].sum())
    C = float((pid_weight * pid_n.astype(float) ** 2).sum())
    # x_i до каждой попытки – сколько попыток того же человека раньше по возрастанию оценки
    m = len(r)
    o = np.argsort(pid, kind="stable")
    ps = pid[o]
    start = np.r_[0, np.nonzero(ps[1:] != ps[:-1])[0] + 1]
    gstart = np.repeat(start, np.diff(np.r_[start, m]))
    xi_before = np.empty(m)
    xi_before[o] = np.arange(m) - gstart
    wi = pid_weight[pid].astype(float)
    ni = pid_n[pid].astype(float)
    dA = wi * (2 * xi_before + 1)
    dB = wi * ni
    A = np.cumsum(dA)
    B = np.cumsum(dB)
    X = np.cumsum(w.astype(float))
    # граница после принятия первых t+1 попыток; допустимы только места между разными оценками
    p = X / n
    S2 = A - 2 * p * B + p * p * C
    v = k / (k - 1) * S2 / (n * n) if k > 1 else np.full(m, math.inf)
    base = p * (1 - p) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        deff = np.where(base > 0, np.maximum(1.0, v / base), math.inf)
    deff = np.where(p >= 1, math.inf, deff)
    deff = np.where(X == 0, 1.0, deff)
    if k <= 1:
        deff[:] = math.inf
    neff = n / deff
    xeff = p * neff
    # m = 0
    if cp_upper(0.0, n) > far:
        return float(np.nextafter(r[0], -np.inf)), True
    boundary = np.ones(m, dtype=bool)
    boundary[:-1] = r[:-1] < r[1:]
    tb = np.nonzero(boundary)[0]
    fin = np.isfinite(deff[tb]) & (xeff[tb] < neff[tb])
    ub = np.ones(len(tb))
    ub[fin] = betaincinv(xeff[tb][fin] + 1.0, neff[tb][fin] - xeff[tb][fin], 0.95)
    bad = np.nonzero(ub > far)[0]
    if len(bad):
        t = tb[bad[0]]
        # недопустимо принять до t включительно: θ – перед группой равных оценок, содержащей t
        g0 = int(np.searchsorted(r, r[t], side="left"))
        return float(np.nextafter(r[g0], -np.inf)), False
    return math.inf, False


def deff_of(acc, person_ids, weights):
    """deff и n_eff для доли принятых по кластерам (люди); соглашение – FeiThreshold.java:75–78, 88."""
    ids = np.unique(person_ids)
    n = float(weights.sum())
    X = float((weights * acc).sum())
    p = X / n
    if X == 0:
        return 1.0, n
    k = 0.0
    S = 0.0
    for i in ids:
        sel = person_ids == i
        wi = weights[sel][0]
        if wi == 0:
            continue
        k += wi
        xi = float(acc[sel].sum())
        ni = float(sel.sum())
        S += wi * (xi - p * ni) ** 2
    if k <= 1 or p >= 1:
        return math.inf, 0.0
    v = k / (k - 1) * S / (n * n)
    deff = max(1.0, v / (p * (1 - p) / n))
    return deff, n / deff


def eer(own_r, own_wrong, own_w, ctrl_r, ctrl_w):
    """EER по объединённым попыткам; соглашение из кода – XBaseSep.java:136–138 (кандидаты – все r своих и чужих,
    первый θ по возрастанию с наименьшим |ошибка своих − FAR|, EER – полусумма)."""
    cand = np.sort(np.concatenate([own_r[own_w > 0], ctrl_r[ctrl_w > 0]]))
    o = np.argsort(own_r, kind="stable")
    orr, ow, owr = own_r[o], own_w[o].astype(float), (np.asarray(own_wrong)[o] == 1)
    co = np.argsort(ctrl_r, kind="stable")
    crr, cw = ctrl_r[co], ctrl_w[co].astype(float)
    Wo, Wc = ow.sum(), cw.sum()
    if Wo == 0 or Wc == 0:
        if EER_EMPTY_50:
            return -math.inf, 1.0, 0.0, 0.5
        return math.nan, math.nan, math.nan, math.nan
    cum_ow = np.concatenate([[0.0], np.cumsum(ow)])
    cum_owr = np.concatenate([[0.0], np.cumsum(ow * owr)])
    cum_cw = np.concatenate([[0.0], np.cumsum(cw)])
    io = np.searchsorted(orr, cand, side="right")
    ic = np.searchsorted(crr, cand, side="right")
    acc_o = cum_ow[io]
    acc_wrong = cum_owr[io]
    if EER_JAVA_FLOAT:
        # как XBaseSep.java:171–176: проценты 100·(ownN − okN)/ownN и 100·acc/ctrlN в плавающей точке, выбор – первое
        # строгое «gap < bestGap» от bestGap = 100 (то есть первый точный минимум); при всех gap ≥ 100 – заготовка
        ok_n = acc_o - acc_wrong
        frr = 100 * (Wo - ok_n) / Wo
        farp = 100 * cum_cw[ic] / Wc
        g = np.abs(frr - farp)
        i = int(np.argmin(g))
        if not g[i] < 100:
            return -math.inf, 1.0, 0.0, 0.5
        return float(cand[i]), float(frr[i] / 100), float(farp[i] / 100), float((frr[i] + farp[i]) / 2 / 100)
    err = (Wo - acc_o + acc_wrong) / Wo
    far = cum_cw[ic] / Wc
    d = np.abs(err - far)
    # равенство |разностей| – точно, без шума округления долей (найдено на feret-update, К4 L1 П2 (в) M = 3, шаги ≥ 2)
    i = int(np.nonzero(d <= d.min() + 1e-12)[0][0])
    return float(cand[i]), float(err[i]), float(far[i]), float((err[i] + far[i]) / 2)


# ---------- строка отчёта ----------

class Row:
    """Попытки строки по конфигурациям, в массивах; людям – общие номера (для одних весов бутстрепа)."""

    def __init__(self, conf, pindex, r0=R0):
        self.r0 = r0
        self.s = sorted(conf)
        self.c = {}
        for s in self.s:
            d = {}
            for role in ("own", "ctrl", "cv"):
                L = conf[s][role]
                r = np.array([a[0] for a in L], dtype=float)
                # 0 – верный лучший, 1 – неверный, 2 – принудительный отказ (нет лица, отвергнут отбором): ошибка, но ни (ii), ни (iii)
                wrong = np.array([int(a[1]) for a in L], dtype=np.int8)
                pid = np.array([pindex[a[2]] for a in L], dtype=np.int64)
                pose = np.array([a[3] for a in L], dtype=float)
                o = np.argsort(r, kind="stable")
                d[role] = (r[o], wrong[o], pid[o], pose[o])
            cvpid = d["cv"][2]
            cnt = np.bincount(cvpid, minlength=len(pindex))
            d["cv_n"] = cnt
            self.c[s] = d
        self.np_ = len(pindex)
        # Пороги П2 / П3, записанные в файле оценок (feret-update, feret-wstatus4): {s: {(way, far): θ}}; None – по cv.
        self.fixed = None


def evaluate(row, pw, fars=(0.01, 0.05)):
    """Все меры строки при весах людей pw (единицы – точечная оценка)."""
    res = {}
    for far in fars:
        for way in ("P2", "P3", "sep"):
            if row.fixed is not None and way != "sep" and any((way, far) not in row.fixed[s] for s in row.s):
                continue
            err = {"all": 0.0, "fr": 0.0, "tu": 0.0, "i": 0.0, "ii": 0.0, "iii": 0.0, "iv": 0.0}
            nown = {"all": 0.0, "fr": 0.0, "tu": 0.0}
            cacc = 0.0
            cn = 0.0
            pers_acc = 0
            pers_n = 0
            ctrl_acc_all = []
            ctrl_pid_all = []
            ctrl_w_all = []
            p3_m0 = 0
            for s in row.s:
                d = row.c[s]
                cr, _, cpid, _ = d["cv"]
                cw = pw[cpid]
                kr, _, kpid, _ = d["ctrl"]
                kw = pw[kpid]
                if row.fixed is not None and way != "sep":
                    th = row.fixed[s][(way, far)]
                elif way == "P2":
                    th = np_threshold(cr, cw, far)
                elif way == "P3":
                    th, m0 = p3_threshold(cr, cw, cpid, (pw * (d["cv_n"] > 0)), d["cv_n"], far)
                    p3_m0 += m0
                else:
                    th = np_threshold(kr, kw, far)
                orr, owk, opid, opose = d["own"]
                owr = owk == 1
                forced = owk == 2
                ow = pw[opid].astype(float)
                acc = orr <= th
                front = np.abs(opose) <= row.r0
                e = (~acc) | owr
                err["all"] += (ow * e).sum()
                err["fr"] += (ow * e * front).sum()
                err["tu"] += (ow * e * ~front).sum()
                err["i"] += (ow * (acc & ~owr)).sum()
                err["ii"] += (ow * (~acc & ~owr & ~forced)).sum()
                err["iii"] += (ow * (~acc & owr)).sum()
                err["iv"] += (ow * (acc & owr)).sum()
                nown["all"] += ow.sum()
                nown["fr"] += (ow * front).sum()
                nown["tu"] += (ow * ~front).sum()
                ka = kr <= th
                cacc += (kw * ka).sum()
                cn += kw.sum()
                # по людям: человек в конфигурации
                present = np.bincount(kpid, minlength=row.np_) > 0
                pacc = np.bincount(kpid, weights=ka.astype(float), minlength=row.np_) > 0
                pers_n += (pw * present).sum()
                pers_acc += (pw * pacc).sum()
                ctrl_acc_all.append(ka.astype(float))
                ctrl_pid_all.append(kpid)
                ctrl_w_all.append(kw.astype(float))
            key = (way, far)
            res[key] = dict(err=err, nown=nown, cacc=cacc, cn=cn, pers_acc=pers_acc, pers_n=pers_n, p3_m0=p3_m0,
                            ctrl=(np.concatenate(ctrl_acc_all), np.concatenate(ctrl_pid_all), np.concatenate(ctrl_w_all)))
    # EER – по объединённым попыткам
    own_r = np.concatenate([row.c[s]["own"][0] for s in row.s])
    own_wr = np.concatenate([row.c[s]["own"][1] for s in row.s])
    own_w = np.concatenate([pw[row.c[s]["own"][2]] for s in row.s])
    ct_r = np.concatenate([row.c[s]["ctrl"][0] for s in row.s])
    ct_w = np.concatenate([pw[row.c[s]["ctrl"][2]] for s in row.s])
    res["EER"] = eer(own_r, own_wr, own_w, ct_r, ct_w)
    return res
