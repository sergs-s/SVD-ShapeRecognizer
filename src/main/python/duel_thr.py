"""Д-№1 (дуэль, 30.10.2026; PLAN.md Д21–Д25): фиксированный порог классики (Д) – по векторам 1 − cos (XBaseEval duel-thr-vec).

Команды:
  split <каталог экспорта> <вывод .tsv>
      Деление FEI, MUCT, ORL на шаблон и пробы – тем же способом и seed, что у порога SFace (tools/verify/sface_thr.py, cloud-verify
      cafb7fc): ключи – кадры sface/<база> экспорта (как у VerifySfaceEmb), у человека шаблон – ⌈n/2⌉ снимков (numpy
      default_rng(17100 + номер базы: FEI 0, MUCT 1, ORL 2)), пробы – остальные; дубли личности, найденные правилом Ахмеда по SFace
      (FEI 002 и 071, MUCT 088 и 429), – второй номер пары исключён до деления, как в основных строках SFace.
  table <каталог векторов (не FERET)> <каталог векторов FERET> <вывод .tsv> <вывод отчёта .txt>
      Подвыборки галереи по N, сетка t, FAR по попыткам и по людям, ошибка своих, выбор t(N); проверка (Б).

Запуск: D:/data/models/mediapipe/venv/Scripts/python.exe -I src/main/python/duel_thr.py …
"""
import gzip
import hashlib
import math
import os
import sys
import time

import numpy as np

T_START = time.time()
sys.stdout.reconfigure(encoding="utf-8")

SEED_HALF = 17100
DROP = {"fei": {"071"}, "muct": {"429"}, "orl": set()}
# Пересчёт по решениям (37), (38): DUEL_EXCL – люди вне всех ролей (строки не читаются); DUEL_EXCL_R – вне файлов переобучения (перешли в W: подвыборки
# прежние); DUEL_KEEP_OLD – новые люди W, не участвующие в розыгрыше подвыборок (прочие подвыборки – прежние); DUEL_RULE37 – выбор t только по межсессионным строкам.
N_EXCL = set(filter(None, os.environ.get("DUEL_EXCL", "").split(",")))
N_EXCL_R = set(filter(None, os.environ.get("DUEL_EXCL_R", "").split(",")))
N_KEEP_OLD = set(filter(None, os.environ.get("DUEL_KEEP_OLD", "").split(",")))
BASES = ("fei", "muct", "orl")


def split(export, out):
    lines = ["# Деление «шаблон / пробы» для duel-thr-vec (duel_thr.py split; способ и seed – tools/verify/sface_thr.py, cafb7fc)",
             "base\tperson\tkey\tpart"]
    for b, base in enumerate(BASES):
        root = os.path.join(export, "sface", base)
        keys = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".png"):
                    rel = os.path.relpath(os.path.join(dp, f), root).replace("\\", "/")
                    keys.append(rel[:-4])
        keys = [k for k in keys if k.split("/")[0] not in DROP[base]]
        rng = np.random.default_rng(SEED_HALF + b)
        by = {}
        for k in sorted(keys):
            by.setdefault(k.split("/")[0], []).append(k)
        np_, nt, npr, sk = 0, 0, 0, []
        for p in sorted(by):
            units = sorted(by[p])
            n = len(units)
            h = (n + 1) // 2
            if n - h < 1 or h < 2:  # как sface_thr.halves при mini = 2 (единица – снимок)
                sk.append(p)
                continue
            perm = rng.permutation(n)
            for j in perm[:h]:
                lines.append("%s\t%s\t%s\tt" % (base, p, units[j]))
            for j in perm[h:]:
                lines.append("%s\t%s\t%s\tp" % (base, p, units[j]))
            np_ += 1
            nt += h
            npr += n - h
        print("%s: снимков %d, людей %d, шаблон %d, проб %d, пропущены %s" % (base, len(keys), np_, nt, npr, sk or "нет"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("SHA-256", hashlib.sha256(open(out, "rb").read()).hexdigest())


# ---------------------------------------------------------------- разбор (table)

SEED_G = 10742  # numpy default_rng(SEED_G + 1000·номер строки + N + 500·проход); 10742 = faces.seed (42) + 10700
REPS = 200
NS = (10, 20, 50)
TARGET = 0.01
RHO_MIN = 1e-12
FEATS = ("L1-xs-F", "L1-tt-xs-F")
SCORES = ("d1/d2", "d1")
GRIDS = {"d1/d2": np.round(np.arange(0.300, 1.00001, 0.005), 3), "d1": np.round(np.arange(0.000, 1.50001, 0.005), 3)}
# Строки протокола: имя, файлы проходов, вид (ctrl – чужие – контроль, half – пробы людей вне подвыборки), в основном выборе (без GT).
ROWS = (("FERET межс.", ("feret_x_p0", "feret_x_p1"), "ctrl", True), ("FERET внутри", ("feret_i_p0", "feret_i_p1"), "ctrl", True),
        ("GT межс.", ("gt_x",), "ctrl", False), ("GT внутри", ("gt_i",), "ctrl", False),
        ("FEI", ("fei",), "half", True), ("MUCT", ("muct",), "half", True), ("ORL", ("orl",), "half", True))


_LF = np.zeros(1)  # ln i! для i = 0…len − 1 (math.lgamma), наращивается по нужде
_CP = {}  # кэш cp_upper на весь прогон


def _lf(n):
    global _LF
    if len(_LF) <= n:
        _LF = np.array([math.lgamma(i + 1) for i in range(max(n + 1, 2 * len(_LF)))])
    return _LF


def cp_upper(k, n, a=0.05):
    """Точная односторонняя верхняя граница Клоппера – Пирсона (1 − a) – бисекцией по биномиальной функции распределения
    (члены суммы – numpy по таблице ln i!; результат кэшируется по (k, n, a))."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    key = (k, n, a)
    if key in _CP:
        return _CP[key]
    lf = _lf(n)
    i = np.arange(k + 1)
    base = lf[n] - lf[i] - lf[n - i]

    def cdf(p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        lp, lq = math.log(p), math.log1p(-p)
        return float(np.exp(base + i * lp + (n - i) * lq).sum())
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) > a:
            lo = mid
        else:
            hi = mid
    _CP[key] = hi
    return hi


def binom_q(n, p, q):
    """Квантиль Binomial(n, p) уровня q (наименьший k с F(k) ≥ q)."""
    c = 0.0
    for k in range(n + 1):
        c += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))
        if c >= q - 1e-15:
            return k
    return n


class Pass:
    """Проход: галерея P, пробы (роль, человек, расстояния до центров P)."""

    def __init__(self, gallery, rows):
        self.P = list(gallery)
        idx = {p: i for i, p in enumerate(self.P)}
        self.role = np.array([r[1] for r in rows])
        self.px = np.array([r[2] for r in rows])
        self.keys = [r[3] for r in rows]
        self.D = np.array([[float(x) for x in r[5:]] for r in rows])
        self.D[(self.D < 0) & (self.D >= -1e-12)] = 0.0
        if (self.D < 0).any() or np.isnan(self.D).any():
            raise SystemExit("недопустимое расстояние")
        self.is_ctrl = self.role == "ctrl"
        self.pi = np.array([idx.get(p, -1) if not c else -1 for p, c in zip(self.px, self.is_ctrl)])


def read_vec(path):
    """Файл duel-thr-vec / duel-thr-b → [(подпись, Pass)]."""
    out = []
    title, gal, rows = None, None, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] == "#S":
                if title is not None:
                    out.append((title, Pass(gal, rows)))
                title, rows = c[1], []
            elif c[0] == "#G":
                gal = c[1:]
            elif c[0] == "R":
                rows.append(c)
    if title is not None:
        out.append((title, Pass(gal, rows)))
    return out


def score(d, kind):
    """d – расстояния (строки – пробы) до центров подвыборки, отсортированные по строке: (оценка, номер лучшего)."""
    o = np.argsort(d, axis=1, kind="stable")
    d1 = d[np.arange(len(d)), o[:, 0]]
    if kind == "d1":
        return d1, o[:, 0]
    d2 = d[np.arange(len(d)), o[:, 1]]
    r = np.where((d1 == 0) & (d2 == 0), 1.0, np.where(np.isinf(d2), np.inf, d1 / np.where(d2 == 0, 1.0, d2)))
    if ((d2 == 0) & (d1 > 0)).any():
        raise SystemExit("d2 = 0 при d1 > 0")
    return np.maximum(r, RHO_MIN), o[:, 0]


def run(ps, G, kind, sk):
    """Галерея G (номера в ps.P; None – вся): оценки чужих, их люди; у своих – оценка, чужое имя."""
    if G is None:
        G = np.arange(len(ps.P))
        if kind == "ctrl":
            sel = ps.is_ctrl
            s, _ = score(ps.D[sel], sk)
        else:
            Dm = ps.D.copy()
            Dm[np.arange(len(Dm)), ps.pi] = np.inf
            sel = np.ones(len(Dm), dtype=bool)
            s, _ = score(Dm, sk)
        own = ~ps.is_ctrl
        so, b = score(ps.D[own], sk)
        return s, ps.px[sel], so, b != ps.pi[own]
    inG = np.zeros(len(ps.P), dtype=bool)
    inG[G] = True
    sel = ps.is_ctrl if kind == "ctrl" else ~inG[np.maximum(ps.pi, 0)]
    s, _ = score(ps.D[sel][:, G], sk)
    own = (~ps.is_ctrl) & inG[np.maximum(ps.pi, 0)] & (ps.pi >= 0)
    gpos = {g: j for j, g in enumerate(G)}
    tj = np.array([gpos[i] for i in ps.pi[own]], dtype=int)
    so, b = score(ps.D[own][:, G], sk)
    return s, ps.px[sel], so, b != tj


def curves(s, sp, so, wrong, grid):
    k = np.searchsorted(np.sort(s), grid, side="right")
    pm = {}
    for v, p in zip(s, sp):
        pm[p] = min(pm.get(p, np.inf), v)
    kp = np.searchsorted(np.sort(np.array(list(pm.values()))), grid, side="right")
    err = np.array([np.mean(wrong | (so > t)) if len(so) else np.nan for t in grid])
    wn = np.array([np.mean(wrong & (so <= t)) if len(so) else np.nan for t in grid])
    return k, len(s), kp, len(pm), err, wn


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def table(vec_dir, feret_dir, out_tsv, out_txt):
    data = {}  # (строка, признак) → [Pass по проходам]
    files = {}
    for name, fs, kind, main in ROWS:
        for f in range(2):
            ps = []
            for fn in fs:
                d = feret_dir if fn.startswith("feret") else vec_dir
                p = os.path.join(d, "%s_f%d.tsv.gz" % (fn, f))
                files[p] = sha(p)
                ps.append(read_vec(p)[0][1])
            data[(name, f)] = ps
    L = ["Д-№1 (дуэль), фиксированный порог классики, конфигурация (Д) – duel_thr.py table", ""]
    L.append("Конфигурация (Д) – как duel-eval-24 / duel-eval-24-yunet400 (699f3b9, e0c3554): признак LBP (L1-xs-F 7×7 (8, 2); L1-tt-xs-F Тан – "
             "Триггс, 9×9 (8, 2)), 1 − cos в W до центров людей, модель X / X+mirror по |r| ≤ r0 = 0,12; решающая оценка d₁/d₂; справочно – d₁.")
    L.append("W – одна на строку (все люди галереи строки + когорта FEI 20 (кроме строки FEI) + пул FERET; у FERET – половины №2, два прохода); "
             "подвыборки галереи по N – по векторам, без переобучения (решение Жорика (дуэль), ответ на план, п. 1.4).")
    L.append("Подвыборок %d на N (у FERET – по %d на проход), seed – numpy default_rng(%d + 1000·номер строки + N + 500·проход); FAR при N – медиана "
             "по подвыборкам, ↑95 – медиана верхних границ Клоппера – Пирсона; t(N) – наибольший t сетки, при котором ↑95 FAR попыток ≤ 1 %% на "
             "всех строках выбора (если чужих попыток < 299 – ни одного принятого, как у SFace)." % (REPS, REPS // 2, SEED_G))
    L.append("Основной выбор – без строк GT (люди дуэли – люди GT; ответ Жорика (дуэль), п. 1.3); с GT – справочно. Порог SFace по решению (24) "
             "выбран с GT.")
    L.append("")
    L.append("Файлы векторов (SHA-256):")
    for p, h in files.items():
        L.append("  %s  %s" % (h, os.path.basename(p)))
    L.append("")
    T = ["row\tfeat\tscore\tN\tt\tfar\tfar_cp\tfar_person\tfar_person_cp\town_err\twrong_name\tn_imp"]
    res = {}  # (строка, признак, оценка, N) → (медиана far, cp, far_person, cp_person, err, wn, n)
    L.append("Строки: людей галереи (по проходам), своих попыток, чужих попыток (вся галерея)")
    for name, fs, kind, main in ROWS:
        ps = data[(name, 0)]
        L.append("  %-13s людей %s, своих %s, чужих %s, в выборе: %s" % (name, " + ".join(str(len(p.P)) for p in ps),
                 " + ".join(str(int((~p.is_ctrl).sum())) for p in ps), " + ".join(str(int(p.is_ctrl.sum()) if kind == "ctrl" else len(p.D)) for p in ps),
                 "да" if main else "справочно (GT)"))
    L.append("")
    for di, (name, fs, kind, main) in enumerate(ROWS):
        for f in range(2):
            ps = data[(name, f)]
            for sk in SCORES:
                grid = GRIDS[sk]
                for N in NS + ("все",):
                    acc = []
                    if N == "все":
                        parts = [curves(*run(p, None, kind, sk), grid) for p in ps]
                        k = sum(x[0] for x in parts)
                        n = sum(x[1] for x in parts)
                        kp = sum(x[2] for x in parts)
                        npn = sum(x[3] for x in parts)
                        no = [int((~p.is_ctrl).sum()) for p in ps]
                        err = sum(x[4] * w for x, w in zip(parts, no)) / sum(no)
                        wn = sum(x[5] * w for x, w in zip(parts, no)) / sum(no)
                        far, cp = k / n, np.array([cp_upper(int(x), n) for x in k])
                        fp, cpp = kp / npn, np.array([cp_upper(int(x), npn) for x in kp])
                        res[(name, f, sk, N)] = (far, cp, fp, cpp, err, wn, n)
                        continue
                    if any(N > len(p.P) for p in ps):
                        continue
                    for pi, p in enumerate(ps):
                        rng = np.random.default_rng(SEED_G + 1000 * di + N + 500 * pi)
                        for _ in range(REPS // len(ps)):
                            G = np.sort(rng.choice(len(p.P), size=N, replace=False))
                            acc.append(curves(*run(p, G, kind, sk), grid))
                    n = int(np.median([x[1] for x in acc]))
                    far = np.median([x[0] / x[1] for x in acc], axis=0)
                    cpc = {}
                    cp = np.median([[cpc.setdefault((int(v), x[1]), cp_upper(int(v), x[1])) for v in x[0]] for x in acc], axis=0)
                    fp = np.median([x[2] / x[3] for x in acc], axis=0)
                    cpp = np.median([[cpc.setdefault((int(v), x[3]), cp_upper(int(v), x[3])) for v in x[2]] for x in acc], axis=0)
                    err = np.median([x[4] for x in acc], axis=0)
                    wn = np.median([x[5] for x in acc], axis=0)
                    res[(name, f, sk, N)] = (far, cp, fp, cpp, err, wn, n)
                print("ход: %s, %s, %s – %.0f с" % (name, FEATS[f], sk, time.time() - T_START), flush=True)
                for N in NS + ("все",):
                    if (name, f, sk, N) not in res:
                        continue
                    far, cp, fp, cpp, err, wn, n = res[(name, f, sk, N)]
                    for j, t in enumerate(grid):
                        T.append("\t".join([name, FEATS[f], sk, str(N), "%.3f" % t, "%.6f" % far[j], "%.6f" % cp[j], "%.6f" % fp[j], "%.6f" % cpp[j],
                                            "%.6f" % err[j], "%.6f" % wn[j], str(n)]))

    def limit(name, f, sk, N):
        far, cp, fp, cpp, err, wn, n = res[(name, f, sk, N)]
        grid = GRIDS[sk]
        ok = (far * n < 0.5) if n < 299 else (cp <= TARGET + 1e-12)
        j = -1
        for i in range(len(grid)):
            if ok[i]:
                j = i
            else:
                break
        return j

    sel = {}
    for sk in SCORES:
        L.append("=== t(N) по оценке %s: наибольший t, при котором ↑95 FAR попыток ≤ 1 %% (по строке; «–» – N больше людей строки) ===" % sk)
        for f in range(2):
            L.append("")
            L.append("%s:" % FEATS[f])
            hdr = "  %-14s" % "строка" + "".join("%12s" % ("N = %s" % N) for N in NS + ("все",))
            L.append(hdr)
            lim = {}
            for name, fs, kind, main in ROWS:
                cells = []
                for N in NS + ("все",):
                    if (name, f, sk, N) not in res:
                        cells.append("%12s" % "–")
                        continue
                    j = limit(name, f, sk, N)
                    lim[(name, N)] = j
                    cells.append("%12s" % ("< сетки" if j < 0 else "%.3f" % GRIDS[sk][j]))
                L.append("  %-14s" % (name + ("" if main else " *")) + "".join(cells))
            for with_gt in (False, True):
                cells = []
                for N in NS:
                    cand = [(lim[(name, N)], name) for name, fs, kind, main in ROWS if (main or with_gt) and (name, N) in lim]
                    j, who = min(cand)
                    t = GRIDS[sk][j] if j >= 0 else float("nan")
                    sel[(sk, f, N, with_gt)] = (t, who)
                    cells.append("%12s" % ("%.3f (%s)" % (t, who.split()[0]) if j >= 0 else "< сетки"))
                L.append("  %-14s" % ("t(N) с GT *" if with_gt else "t(N) без GT") + "".join(cells))
        L.append("  (* – GT, справочно; в скобках – строка, задающая t)")
        L.append("")
    L.append("=== Порог по умолчанию (Д), d₁/d₂, округлено вниз до 0,005 сетки – основной выбор без GT ===")
    for f in range(2):
        for N in NS:
            t, who = sel[("d1/d2", f, N, False)]
            tg, whog = sel[("d1/d2", f, N, True)]
            L.append("  %-11s N = %-3d t = %.3f (задаёт %s); с GT – %.3f (%s)%s" % (FEATS[f], N, t, who, tg, whog,
                                                                                  "" if abs(t - tg) < 1e-9 else " – различаются"))
    L.append("")
    L.append("=== При t(10) (без GT), d₁/d₂: по строкам при N = 10 – FAR попыток % [↑95], FAR людей % [↑95], ошибка своих %, чужое имя % ===")
    for f in range(2):
        t, _ = sel[("d1/d2", f, 10, False)]
        j = int(np.argmin(np.abs(GRIDS["d1/d2"] - t)))
        L.append("%s, t = %.3f:" % (FEATS[f], t))
        for name, fs, kind, main in ROWS:
            far, cp, fp, cpp, err, wn, n = res[(name, f, "d1/d2", 10)]
            fa, cpa, fpa, cppa, erra, wna, na = res[(name, f, "d1/d2", "все")]
            L.append("  %-14s N = 10: %.2f [%.2f], %.2f [%.2f], %.1f, %.2f; вся галерея: FAR %.2f [%.2f], ошибка своих %.1f (чужих попыток %d)"
                     % (name, 100 * far[j], 100 * cp[j], 100 * fp[j], 100 * cpp[j], 100 * err[j], 100 * wn[j], 100 * fa[j], 100 * cpa[j],
                        100 * erra[j], na))
    L.append("")
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# duel_thr.py table: сетка t, медианы по подвыборкам; N = все – вся галерея строки\n" + "\n".join(T) + "\n")
    # Таблица порогов (Д) для фиксации до прогона на наборе.
    thr = os.path.join(os.path.dirname(out_txt), "duel_thr_D.tsv")
    TL = ["# Пороги (Д) – duel_thr.py table; оценка d1/d2 (основная) и d1 (справочно); выбор без GT (основной) и с GT (справочно); "
          "t – наибольший t сетки, при котором медиана ↑95 КП FAR попыток ≤ 1 % на всех строках выбора",
          "feat\tscore\tN\twith_gt\tt\tdetermined_by"]
    for sk in SCORES:
        for f in range(2):
            for N in NS:
                for wg in (False, True):
                    t, who = sel[(sk, f, N, wg)]
                    TL.append("%s\t%s\t%d\t%s\t%.3f\t%s" % (FEATS[f], sk, N, "yes" if wg else "no", t, who))
    with open(thr, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(TL) + "\n")
    L.append("Таблица порогов: %s, SHA-256 %s" % (os.path.basename(thr), sha(thr)))
    L.append("")
    # Проверка (Б).
    L.append("=== Проверка (Б): W заново на 10 людях подвыборки (duel-thr-b) против общей W на тех же подвыборках, d₁/d₂, порог t(10) без GT ===")
    for name, fn, d in (("GT межс.", "gt_x", vec_dir), ("FERET межс.", "feret_x_p0", feret_dir)):
        for f in range(2):
            pb = os.path.join(d, "b_%s_f%d.tsv.gz" % (fn, f))
            if not os.path.exists(pb):
                L.append("  %s %s: файла %s нет" % (name, FEATS[f], os.path.basename(pb)))
                continue
            files[pb] = sha(pb)
            common = data[(name, f)][0]
            t, _ = sel[("d1/d2", f, 10, False)]
            kb = nb = kc = nc = 0
            eb, ec, no = 0.0, 0.0, 0
            for title, p in read_vec(pb):
                s, sp, so, wrong = run(p, None, "ctrl", "d1/d2")
                kb += int((s <= t).sum())
                nb += len(s)
                eb += float((wrong | (so > t)).sum())
                G = np.array(sorted(common.P.index(x) for x in p.P))
                s2, sp2, so2, wr2 = run(common, G, "ctrl", "d1/d2")
                kc += int((s2 <= t).sum())
                nc += len(s2)
                ec += float((wr2 | (so2 > t)).sum())
                no += len(so)
                if len(so) != len(so2):
                    raise SystemExit("(Б): разное число своих попыток")
            lo, hi = binom_q(nb, TARGET, 0.025), binom_q(nb, TARGET, 0.975)
            de = 100 * (eb - ec) / no
            pos = "ниже интервала (FAR меньше цели)" if kb < lo else ("в интервале" if kb <= hi else "ВЫШЕ интервала")
            L.append("  %-12s %-11s t = %.3f: FAR (Б) %d/%d = %.2f %% [↑95 %.2f], общая W %d/%d = %.2f %%; 95 %% интервал цели 1 %% – %d…%d принятых → "
                     "%s; ошибка своих (Б) %.1f %%, общая %.1f %% (разность %+.1f п. п., %s)" % (
                         name, FEATS[f], t, kb, nb, 100 * kb / nb, 100 * cp_upper(kb, nb), kc, nc, 100 * kc / nc, lo, hi,
                         pos, 100 * eb / no, 100 * ec / no, de, "≤ 1" if abs(de) <= 1 else "БОЛЬШЕ 1"))
    with open(out_txt, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- п. 2.3: сверка с таблицами коллеги (check)

def read_xlsx(path):
    """Листы xlsx (стандартной библиотекой): {имя листа: [[11 чисел] × 70]}."""
    import re
    import xml.etree.ElementTree as ET
    import zipfile
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    z = zipfile.ZipFile(path)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    ss = []  # общие строки: часть чисел у коллеги записана текстом
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            ss.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    out = {}
    for sh in wb.find("m:sheets", ns):
        t = target[sh.get("{%s}id" % ns["r"])]
        t = t.lstrip("/")
        if not t.startswith("xl/"):
            t = "xl/" + t
        rows = []
        for r in ET.fromstring(z.read(t)).find("m:sheetData", ns).findall("m:row", ns):
            vals = [0.0] * 11
            for c in r.findall("m:c", ns):
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                j = ord(col) - ord("A")
                v = c.find("m:v", ns)
                if v is None:
                    continue
                vals[j] = float(ss[int(v.text)].replace(",", ".")) if c.get("t") == "s" else float(v.text)
            rows.append(vals)
        m = re.search(r"(\d)\s*([аб])", sh.get("name"))
        out[m.group(1) + m.group(2)] = np.array(rows)
    return out


def decision(row, b):
    """Решение: номер класса 1–10 или «гость» (только в «б»: столбец 11 = 1)."""
    if b and row[10] >= 0.5:
        return "гость"
    return str(int(np.argmax(row[:10])) + 1)


def check(tsv, xl_face, xl_plain, key, out):
    ours = {}
    with open(tsv, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            ours.setdefault((c[0], c[1], c[2]), {})[int(c[3])] = np.array([float(x) for x in c[4:15]])
    role = {}
    with open(key, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if line.startswith("#") or c[0] == "file" or not c[0].startswith("пробы"):
                continue
            role[int(c[0].split("/")[1][:3])] = c[2]
    L = ["=== Часть 3. Сверка нашей симуляции метода Прокаева (ProkaevMethod) с программой коллеги ===",
         "Его таблицы: %s (SHA-256 %s), %s (SHA-256 %s); наши доли – duel_24.10_prokaev_check.tsv (XBaseEval duel-prokaev-check), SHA-256 %s."
         % (os.path.basename(xl_face), sha(xl_face), os.path.basename(xl_plain), sha(xl_plain), sha(tsv)),
         "Настройки (Д20): обучение – 80 снимков (10 × 8), гостей в обучающей матрице нет; кадр – серый, уменьшен в 10 раз (800 × 1200 → 80 × 120); "
         "k = 7; вар. 1 – l = 3, формула 8 (ProkaevMethod.Variant.S); 2 – l = 5, формула 8; 3 – l = 5, формула 7 (Variant.K); «б» – гость, если "
         "наибольшая доля своего < 0,6 (у коллеги – столбец 11 = 1, прочие 0; проверено по листам).",
         "Способы уменьшения: area – OpenCV BGR2GRAY (uint8) и INTER_AREA (uint8); block – OpenCV BGR2GRAY, среднее блоков 10 × 10 без округления; "
         "matlab – rgb2gray MATLAB (uint8) и imresize(…, 0.1) по умолчанию (бикубическое со сглаживанием, по строкам, затем по столбцам, uint8 после "
         "каждого прохода); matlab-d – то же без округления. Первый кандидат – наибольшая доля среди 10 своих; решение – класс или «гость» (в «б»).",
         ""]
    best = {}
    for set_, xl in (("К-лицо", xl_face), ("К-обычный", xl_plain)):
        his = read_xlsx(xl)
        L.append("--- Комплект %s ---" % set_)
        L.append("  способ     лист  тот же первый  то же решение  макс. |Δ доли|")
        score_way = {}
        for way in ("area", "block", "matlab", "matlab-d"):
            tot = 0
            for sheet in ("1а", "1б", "2а", "2б", "3а", "3б"):
                H = his[sheet]
                O = ours[(set_, way, sheet)]
                if H.shape != (70, 11):
                    raise SystemExit("%s %s: размер %s" % (xl, sheet, H.shape))
                b = sheet.endswith("б")
                first = sum(np.argmax(H[i][:10]) == np.argmax(O[i + 1][:10]) for i in range(70) if not (b and (H[i][10] >= 0.5 or O[i + 1][10] >= 0.5)))
                nfirst = sum(1 for i in range(70) if not (b and (H[i][10] >= 0.5 or O[i + 1][10] >= 0.5)))
                dec = sum(decision(H[i], b) == decision(O[i + 1], b) for i in range(70))
                md = max(float(np.max(np.abs(H[i] - O[i + 1]))) for i in range(70))
                tot += dec
                L.append("  %-9s  %-4s  %3d из %-3d     %2d из 70       %.3g" % (way, sheet, first, nfirst, dec, md))
            score_way[way] = tot
        bw = max(score_way, key=lambda w: (score_way[w], -list(score_way).index(w)))
        best[set_] = bw
        L.append("  Совпало решений по всем листам: %s; лучший способ – %s." % (", ".join("%s %d из 420" % kv for kv in score_way.items()), bw))
        L.append("  Расхождения при способе %s (проба, роль по ключу, его решение / наше, его доли / наши, |Δ| > 1e-6):" % bw)
        for sheet in ("1а", "1б", "2а", "2б", "3а", "3б"):
            H = his[sheet]
            O = ours[(set_, bw, sheet)]
            b = sheet.endswith("б")
            for i in range(70):
                d = float(np.max(np.abs(H[i] - O[i + 1])))
                if d > 1e-6 or decision(H[i], b) != decision(O[i + 1], b):
                    fmt = lambda v: " ".join("%d:%.3f" % (j + 1, x) for j, x in enumerate(v) if x > 1e-9)  # noqa: E731
                    L.append("    %s %03d (%s): %s / %s; {%s} / {%s}" % (sheet, i + 1, role.get(i + 1, "?"), decision(H[i], b), decision(O[i + 1], b),
                                                                        fmt(H[i]), fmt(O[i + 1])))
        L.append("")
    with open(out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- (Н): (П2) + MS-LBP, слияние w = 0,75, β = 1 (ntable)

N_NS = (10, 50, 100, 200)
W_FUSE = 0.75
N_GRID = np.round(np.arange(-12.0, 4.00001, 0.01), 2)
N_GRID_D1 = np.round(np.arange(-8.0, 4.00001, 0.01), 2)
# Строки (Н): имя, файлы, в выборе t (без GT; ORL – справочно), номер строки.
N_ROWS = (("FERET межс.", ("feret_x_p0", "feret_x_p1"), True, 0), ("FERET внутри", ("feret_i_p0", "feret_i_p1"), not os.environ.get("DUEL_RULE37"), 1),
          ("GT межс.", ("gt_x",), False, 2), ("GT внутри", ("gt_i",), False, 3),
          ("FEI", ("fei",), True, 4), ("MUCT", ("muct",), True, 5), ("ORL", ("orl",), False, 6))


class NPass:
    """Проход (Н): галерея (W), строки own / P / Z по кандидату; (Т2), если есть."""

    def __init__(self, path, drop=frozenset()):
        self.t2 = None
        gal, rows = None, []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[0] == "#G":
                    gal = c[1:]
                elif c[0] == "#T2":
                    self.t2 = (float(c[1]), float(c[2]))
                elif c[0] == "R":
                    if c[2] not in N_EXCL and c[2] not in drop:
                        rows.append(c)
        self.P = gal
        idx = {p: i for i, p in enumerate(gal)}
        self.role = np.array([r[1] for r in rows])
        self.px = np.array([r[2] for r in rows])
        self.keys = [r[3] for r in rows]
        D = np.array([[float(x) for x in r[5:]] for r in rows])
        D[(D < 0) & (D >= -1e-12)] = 0.0
        if (D < 0).any() or np.isnan(D).any():
            raise SystemExit("недопустимое расстояние: " + path)
        self.L = np.log(np.maximum(D, RHO_MIN))
        self.pi = np.array([idx[p] if r == "own" else -1 for p, r in zip(self.px, self.role)])


def n_load(vec_dir, feret_dir, fn):
    d = feret_dir if fn.startswith("feret") else vec_dir
    ps = []
    for f in range(2):
        p = os.path.join(d, "%s_f%d_n.tsv.gz" % (fn, f))
        ps.append((NPass(p), sha(p), os.path.basename(p)))
    tt, xs = ps[1][0], ps[0][0]
    if tt.keys != xs.keys or tt.P != xs.P:
        raise SystemExit("(Н): пробы или галерея кандидатов разошлись: " + fn)
    return tt, xs, ps


def n_scores(tt, xs, G, w=W_FUSE):
    """Подвыборка G: приведение по (П), слитые ℓ, оценки s (β = 1) и s₀ = ℓ₁ (β = 0) у own и Z; μ, σ по кандидатам."""
    P = tt.role == "P"
    st = []
    for c in (tt, xs):
        v = c.L[P][:, G].ravel()
        st.append((float(v.mean()), float(v.std(ddof=1))))
    F = w * (tt.L[:, G] - st[0][0]) / st[0][1] + (1 - w) * (xs.L[:, G] - st[1][0]) / st[1][1]
    o = np.argsort(F, axis=1, kind="stable")
    l1 = F[np.arange(len(F)), o[:, 0]]
    l2 = F[np.arange(len(F)), o[:, 1]]
    s = 2 * l1 - l2
    return s, l1, o[:, 0], st


def n_run(tt, xs, G):
    s, s0, best, st = n_scores(tt, xs, G)
    Z = tt.role == "Z"
    gpos = {g: j for j, g in enumerate(G)}
    own = (tt.role == "own") & np.isin(tt.pi, G)
    tj = np.array([gpos[i] for i in tt.pi[own]], dtype=int)
    return (s[Z], s0[Z], tt.px[Z], s[own], s0[own], best[own] != tj, st)


# Переобучение W (решение основного Жорика №10, 30.10): N = 10 – FERET межс., FEI, MUCT; N = 50 – проверка на FERET межс. (проход 1) и FEI (в GT межс. 30 человек в W – 50 не выбрать).
R_ROWS = {10: ("FERET межс.", "FEI", "MUCT"), 50: ("FERET межс.", "FEI")}
R_FULL = ("FERET межс.", "FEI", "MUCT")  # строки переобучения в таблице (N = 10 и, если проверка N = 50 не прошла, N = 50 – файлы r50f)


def n_draw(rng, tt, N):
    """Подвыборка индексов галереи: новые люди W (N_KEEP_OLD) в розыгрыше не участвуют – прежние подвыборки не меняются."""
    idx = [i for i, p in enumerate(tt.P) if p not in N_KEEP_OLD]
    return np.array(idx)[np.sort(rng.choice(len(idx), size=N, replace=False))]


def n_rng(rn, N, pi):
    return np.random.default_rng(SEED_G + 1000 * rn + N + 500 * pi + 50000)


def nsubs(vec_dir, feret_dir, out, N, reps, mode="check"):
    """Файл подвыборок для переобучения: те же подвыборки, что у общей W в ntable (первые reps по строке; FERET – reps/2 на проход;
    при N = 50 – только проход 1 FERET)."""
    N, reps = int(N), int(reps)
    L = ["# duel_thr.py nsubs N = %d, подвыборок %d на строку; seed – как ntable: default_rng(%d + 1000·номер строки + N + 500·проход + 50000)" % (N, reps, SEED_G)]
    for name, fs, main, rn in N_ROWS:
        if name not in (R_FULL if mode == "full" else R_ROWS[N]):
            continue
        use = fs if (N == 10 or mode == "full") else fs[:1]
        for pi, fn in enumerate(use):
            tt, xs, _ = n_load(vec_dir, feret_dir, fn)
            rng = n_rng(rn, N, pi)
            for k in range(reps // len(use)):
                G = np.sort(rng.choice(len(tt.P), size=N, replace=False))
                L.append("%s\t%d\t%s" % (fn, k, ",".join(tt.P[g] for g in G)))
    with open(out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("подвыборок:", len(L) - 1, "SHA-256", sha(out))


def fastcheck(feret_dir, fast_dir, out):
    """Сверка MldaFast с MldaScorer на FERET внутри (общая W): относительная разница расстояний, лучший кандидат по каждой пробе."""
    L = ["Сверка MldaFast (LAPACK dsyevd, вариант (в)) с MldaScorer: FERET внутри, общая W, оба прохода, оба кандидата"]
    ok = True
    for fn in ("feret_i_p0", "feret_i_p1"):
        for f in (1, 0):
            a = NPass(os.path.join(feret_dir, "%s_f%d_n.tsv.gz" % (fn, f)))
            b = NPass(os.path.join(fast_dir, "%s_f%d_n.tsv.gz" % (fn, f)))
            if a.keys != b.keys or a.P != b.P or list(a.role) != list(b.role):
                raise SystemExit("fastcheck: пробы или галерея разошлись: " + fn)
            da, db = np.exp(a.L), np.exp(b.L)
            rel = float(np.max(np.abs(da - db) / np.maximum(np.abs(da), 1e-12)))
            same = int((np.argmin(da, axis=1) == np.argmin(db, axis=1)).sum())
            good = rel <= 1e-6 and same == len(da)
            ok &= good
            L.append("  %s f%d: проб %d, людей %d; наибольшая относительная разница расстояний %.3g; лучший кандидат совпал %d из %d – %s" % (
                fn, f, len(da), len(a.P), rel, same, len(da), "да" if good else "НЕТ"))
    L.append("Итог: %s" % ("сверка пройдена (≤ 1e-6, решения совпали)" if ok else "сверка НЕ пройдена"))
    with open(out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))
    if not ok:
        sys.exit(3)


def r_load(d, tag, fn):
    """Переобученные подвыборки: [(tt, xs)] по блокам файлов <tag>_<fn>_f1 / _f0."""
    pt = os.path.join(d, "%s_%s_f1_n.tsv.gz" % (tag, fn))
    px_ = os.path.join(d, "%s_%s_f0_n.tsv.gz" % (tag, fn))
    if not (os.path.exists(pt) and os.path.exists(px_)):
        return None, []
    bt, bx = n_split_blocks(pt), n_split_blocks(px_)
    out = []
    for (tl, a), (xl, b) in zip(bt, bx):
        if tl != xl or a.keys != b.keys or a.P != b.P:
            raise SystemExit("(Н): блоки переобучения кандидатов разошлись: " + fn)
        out.append((a, b))
    return out, [(sha(pt), os.path.basename(pt)), (sha(px_), os.path.basename(px_))]


def ntable(vec_dir, feret_dir, out_tsv, out_txt):
    L = ["Д-№1 (дуэль), таблица N → t классики, конфигурация (Н) – duel_thr.py ntable", ""]
    L.append("Конфигурация (Н) (решения (26)–(32)): признак (П2) + MS-LBP (XBaseFeat2 вариант 8; L1-tt-xs-F 9×9, L1-xs-F 7×7), MLDA с пулом FERET, c = 1, "
             "косинус; L1-tt-xs-F – модель по позе (r0 = 0,12), L1-xs-F – всегда X+mirror; слияние по человеку ℓ(g) = 0,75·ℓ̃_tt + 0,25·ℓ̃_xs, "
             "ℓ̃ = (ln(1 − cos) − μ) / σ; s = 2ℓ₁ − ℓ₂ (β = 1); справочно s₀ = ℓ₁ (β = 0) при N = 10.")
    L.append("W – одна на строку; люди (П) – приведение ℓ (μ, σ по всем расстояниям «проба (П) – человек подвыборки», у каждого кандидата) и (З) – "
             "замер FAR и выбор t – вне W (ответ Жорика (дуэль) на №3; деление – seed faces.seed + 10710 + номер строки). FERET межс. и GT – векторы "
             "Васи v8 (пачка 2, ec68b50), прочие – duel-thr-n.")
    L.append("Подвыборок %d на N (FERET – по %d на проход), seed – numpy default_rng(%d + 1000·номер строки + N + 500·проход + 50000); "
             "t(N) – наибольший t сетки (шаг 0,01), при котором медиана ↑95 КП FAR попыток (З) ≤ 1 %% на строках выбора (FERET межс. и внутри, FEI, "
             "MUCT); при < 299 попыток – ни одного принятого. GT и ORL – справочно (только замер)." % (REPS, REPS // 2, SEED_G))
    L.append("")
    data, files = {}, []
    for name, fs, main, rn in N_ROWS:
        data[name] = []
        for fn in fs:
            tt, xs, ps = n_load(vec_dir, feret_dir, fn)
            data[name].append((tt, xs))
            files += [(h, b) for _, h, b in ps]
    rfull = {}  # (строка, N) → [(tt, xs)]: подвыборки таблицы с переобучением W (N = 10 – r10, N = 50 – r50f)
    for N, tag in ((10, "r10"), (50, "r50f")):
        for name, fs, main, rn in N_ROWS:
            if name not in R_FULL:
                continue
            blocks, ok = [], True
            for fn in fs:
                d = feret_dir if fn.startswith("feret") else vec_dir
                b, h = r_load(d, tag, fn)
                if b is None:
                    ok = False
                    break
                blocks += b
                files += h
            if ok:
                rfull[(name, N)] = blocks
    rdata = {}  # (строка, N) → [(tt, xs)] переобученных подвыборок
    for N, names in ((50, R_ROWS[50]),):  # проверка N = 50 (5 подвыборок, проход 1 FERET)
        for name, fs, main, rn in N_ROWS:
            if name not in names:
                continue
            blocks, ok = [], True
            for fn in (fs if N == 10 else fs[:1]):
                d = feret_dir if fn.startswith("feret") else vec_dir
                b, h = r_load(d, "r%d" % N, fn)
                if b is None:
                    ok = False
                    break
                blocks += b
                files += h
            if ok:
                rdata[(name, N)] = blocks
    L.append("Файлы векторов (SHA-256):")
    L += ["  %s  %s" % x for x in files]
    L.append("")
    L.append("Строки: людей в W (по проходам); своих попыток; (П) людей / попыток; (З) людей / попыток")
    for name, fs, main, rn in N_ROWS:
        cells = []
        for tt, xs in data[name]:
            cells.append("W %d; своих %d; (П) %d / %d; (З) %d / %d" % (len(tt.P), int((tt.role == "own").sum()), len(set(tt.px[tt.role == "P"])),
                                                                     int((tt.role == "P").sum()), len(set(tt.px[tt.role == "Z"])), int((tt.role == "Z").sum())))
        L.append("  %-13s %s%s" % (name, " | ".join(cells), "" if main else " (справочно)"))
    L.append("")
    # Условие (а): (Т2) против (П) на полной галерее.
    L.append("=== Условие (а): постоянные приведения на полной галерее – (Т2) (векторы Васи, модели без части) против (П); равносильный w ===")
    stop = False
    for name in ("FERET межс.", "GT межс.", "GT внутри"):
        for pi, (tt, xs) in enumerate(data[name]):
            G = np.arange(len(tt.P))
            _, _, _, st = n_scores(tt, xs, G)
            if tt.t2 is None or xs.t2 is None:
                raise SystemExit("(Н): нет (Т2) у " + name)
            a = W_FUSE * tt.t2[1] / st[0][1]
            b = (1 - W_FUSE) * xs.t2[1] / st[1][1]
            weq = a / (a + b)
            ok = 0.7 <= weq <= 0.8
            stop |= not ok
            L.append("  %-12s проход %d: tt (Т2) μ %.4f σ %.4f, (П) μ %.4f σ %.4f; xs (Т2) μ %.4f σ %.4f, (П) μ %.4f σ %.4f → w_eq = %.3f %s" % (
                name, pi + 1, tt.t2[0], tt.t2[1], st[0][0], st[0][1], xs.t2[0], xs.t2[1], st[1][0], st[1][1], weq, "в [0,7; 0,8]" if ok else "ВНЕ [0,7; 0,8]"))
    L.append("")
    if stop:
        L.append("w_eq вне [0,7; 0,8] – таблица не считается (условие (а)), доклад Жорику.")
        with open(out_txt, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("\n".join(L) + "\n")
        print("\n".join(L))
        return
    T = ["row\tscore\tN\tt\tfar\tfar_cp\tfar_person\tfar_person_cp\town_err\twrong_name\tn_imp"]
    res, consts, accs, owns, retrained = {}, {}, {}, {}, set()
    for di, (name, fs, main, rn) in enumerate(N_ROWS):
        for sk, grid in (("s", N_GRID), ("s0", N_GRID_D1)):
            for N in N_NS:
                if any(N > len(tt.P) for tt, xs in data[name]):
                    continue
                if sk == "s0" and N != 10:
                    continue
                acc, cs, nown = [], [], []
                if (name, N) in rfull:
                    retrained.add((name, N))
                    for tt, xs in rfull[(name, N)]:
                        zs, zs0, zp, os_, os0, wr, st = n_run(tt, xs, np.arange(len(tt.P)))
                        a, b = (zs, os_) if sk == "s" else (zs0, os0)
                        acc.append(curves(a, zp, b, wr, grid))
                        nown.append(len(b))
                        cs.append([st[0][0], st[0][1], st[1][0], st[1][1]])
                else:
                    for pi, (tt, xs) in enumerate(data[name]):
                        rng = n_rng(rn, N, pi)
                        reps = 1 if N == len(tt.P) else REPS // len(data[name])
                        for _ in range(reps):
                            G = n_draw(rng, tt, N)
                            zs, zs0, zp, os_, os0, wr, st = n_run(tt, xs, G)
                            a, b = (zs, os_) if sk == "s" else (zs0, os0)
                            acc.append(curves(a, zp, b, wr, grid))
                            nown.append(len(b))
                            cs.append([st[0][0], st[0][1], st[1][0], st[1][1]])
                accs[(name, sk, N)] = acc
                owns[(name, sk, N)] = nown
                n = int(np.median([x[1] for x in acc]))
                cpc = {}
                far = np.median([x[0] / x[1] for x in acc], axis=0)
                cp = np.median([[cpc.setdefault((int(v), x[1]), cp_upper(int(v), x[1])) for v in x[0]] for x in acc], axis=0)
                fp = np.median([x[2] / x[3] for x in acc], axis=0)
                cpp = np.median([[cpc.setdefault((int(v), x[3]), cp_upper(int(v), x[3])) for v in x[2]] for x in acc], axis=0)
                err = np.median([x[4] for x in acc], axis=0)
                wn = np.median([x[5] for x in acc], axis=0)
                res[(name, sk, N)] = (far, cp, fp, cpp, err, wn, n)
                cs = np.array(cs)
                consts[(name, N)] = (np.median(cs, axis=0), np.quantile(cs, 0.05, axis=0), np.quantile(cs, 0.95, axis=0))
                for j, t in enumerate(grid):
                    T.append("\t".join([name, sk, str(N), "%.2f" % t, "%.6f" % far[j], "%.6f" % cp[j], "%.6f" % fp[j], "%.6f" % cpp[j], "%.6f" % err[j],
                                        "%.6f" % wn[j], str(n)]))

    def limit(key, grid):
        far, cp, fp, cpp, err, wn, n = res[key]
        ok = (far * n < 0.5) if n < 299 else (cp <= TARGET + 1e-12)
        j = -1
        for i in range(len(grid)):
            if ok[i]:
                j = i
            else:
                break
        return j

    L.append("=== t(N) по s (β = 1, слияние): наибольший t, при котором ↑95 FAR попыток (З) ≤ 1 % (по строке; «–» – N больше людей в W) ===")
    L.append("  %-14s" % "строка" + "".join("%11s" % ("N = %d" % N) for N in N_NS))
    sel = {}
    for name, fs, main, rn in N_ROWS:
        cells = []
        for N in N_NS:
            if (name, "s", N) not in res:
                cells.append("%11s" % "–")
                continue
            j = limit((name, "s", N), N_GRID)
            cells.append("%11s" % ("< сетки" if j < 0 else "%.2f" % N_GRID[j]))
        L.append("  %-14s" % (name + ("" if main else " *")) + "".join(cells))
    cells = []
    for N in N_NS:
        cand = [(limit((name, "s", N), N_GRID), name) for name, fs, main, rn in N_ROWS if main and (name, "s", N) in res]
        j, who = min(cand)
        sel[N] = (N_GRID[j] if j >= 0 else float("nan"), who)
        cells.append("%11s" % ("%.2f" % sel[N][0]))
    L.append("  %-14s" % "t(N)" + "".join(cells))
    L.append("  %-14s" % "задаёт" + "".join("%11s" % sel[N][1].split()[0] for N in N_NS))
    L.append("  (* – справочно, в выборе не участвует)")
    L.append("  W: переобучение на каждой подвыборке у строк %s (подвыборок: %s); прочие строки и N – общая W "
             "(«без переобучения W»; N = 50 – по проверке ниже)" % (", ".join("%s (N = %d)" % k for k in sorted(retrained)) or "–",
                                                                  ", ".join("%s N = %d – %d" % (k[0], k[1], len(v)) for k, v in rfull.items()) or "–"))
    L.append("")
    L.append("=== Разброс фактического FAR попыток (З) на выбранном t(N) по подвыборкам: медиана [2,5 %; 97,5 %], подвыборок; "
             "отметка – верхний край 97,5 % > 2 % ===")
    for N in N_NS:
        t = sel[N][0]
        j = int(np.argmin(np.abs(N_GRID - t)))
        for name, fs, main, rn in N_ROWS:
            if (name, "s", N) not in accs:
                continue
            v = np.array([x[0][j] / x[1] for x in accs[(name, "s", N)]])
            lo, md, hi = np.quantile(v, 0.025), np.median(v), np.quantile(v, 0.975)
            wide = hi > 0.02  # только верхний край (Жорик №1 от 31.10)
            L.append("  N = %-3d %-14s %.2f %% [%.2f; %.2f], подвыборок %d%s%s" % (N, name, 100 * md, 100 * lo, 100 * hi, len(v),
                     "; W заново" if (name, N) in retrained else "; без переобучения W", " – верхний край > 2 %" if wide else ""))
    L.append("")
    L.append("=== Постоянные приведения по N (медиана [5 %; 95 %] по подвыборкам): μ_tt, σ_tt, μ_xs, σ_xs ===")
    for name, fs, main, rn in N_ROWS:
        for N in N_NS:
            if (name, N) not in consts:
                continue
            m, lo, hi = consts[(name, N)]
            L.append("  %-13s N = %-3d " % (name, N) + "; ".join("%.4f [%.4f; %.4f]" % (m[i], lo[i], hi[i]) for i in range(4)))
    L.append("")
    L.append("=== При t(N): по строкам – FAR попыток (З) % [↑95], FAR людей % [↑95], ошибка своих %, чужое имя %, попыток (З) ===")
    for N in N_NS:
        t = sel[N][0]
        j = int(np.argmin(np.abs(N_GRID - t)))
        L.append("N = %d, t = %.2f:" % (N, t))
        for name, fs, main, rn in N_ROWS:
            if (name, "s", N) not in res:
                continue
            far, cp, fp, cpp, err, wn, n = res[(name, "s", N)]
            L.append("  %-14s %.2f [%.2f], %.2f [%.2f], %.1f, %.2f, %d" % (name, 100 * far[j], 100 * cp[j], 100 * fp[j], 100 * cpp[j], 100 * err[j], 100 * wn[j], n))
    L.append("")
    L.append("=== При t(N), по строкам: подвыборок; своих и (З) попыток на подвыборку (медиана); схема W; ошибка своих % и FAR попыток (З) % – "
             "по сумме попыток по всем подвыборкам (основная мера; k/n), рядом [медиана по подвыборкам, справочно] ===")
    for N in N_NS:
        t = sel[N][0]
        j = int(np.argmin(np.abs(N_GRID - t)))
        L.append("N = %d, t = %.2f:" % (N, t))
        for name, fs, main, rn in N_ROWS:
            key = (name, "s", N)
            if key not in accs:
                continue
            acc, no = accs[key], np.array(owns[key])
            errs = np.array([x[4][j] for x in acc])
            kk = np.array([x[0][j] for x in acc])
            nn = np.array([x[1] for x in acc])
            scheme = "W заново на подвыборке" if (name, N) in retrained else "общая W (без переобучения W)"
            L.append("  %-14s подвыборок %d; своих %d, (З) %d; %s; ошибка своих %.1f [%.1f]; FAR %.2f (%d/%d) [%.2f]%s" % (
                name, len(acc), int(np.median(no)), int(np.median(nn)), scheme, 100 * (errs * no).sum() / no.sum(), 100 * np.median(errs),
                100 * kk.sum() / nn.sum(), int(kk.sum()), int(nn.sum()), 100 * np.median(kk / nn), "" if main else " (справочно)"))
    L.append("")
    L.append("=== Запас: t(N) против t₁% – наибольшего t, при котором FAR попыток (З) по сумме попыток ≤ 1 % на строке, задающей t(N); ошибка своих – "
             "по сумме попыток; [t₁% по медиане FAR по подвыборкам и ошибка своих (медиана) при нём – справочно] ===")

    def first_run(ok):
        j1 = -1
        for i in range(len(ok)):
            if ok[i]:
                j1 = i
            else:
                break
        return j1

    for N in N_NS:
        t, who = sel[N]
        acc, no = accs[(who, "s", N)], np.array(owns[(who, "s", N)])
        kk = np.sum([x[0] for x in acc], axis=0)
        nn = float(np.sum([x[1] for x in acc]))
        perr = np.sum([x[4] * w for x, w in zip(acc, no)], axis=0) / no.sum()
        j1 = first_run(kk / nn <= TARGET + 1e-12)
        far, cp, fp, cpp, err, wn, n = res[(who, "s", N)]
        jm = first_run(far <= TARGET + 1e-12)
        j = int(np.argmin(np.abs(N_GRID - t)))
        L.append("  N = %-3d задаёт %-13s t(N) = %.2f, t₁%% = %.2f, запас %.2f; ошибка своих при t(N) %.1f %%, при t₁%% %.1f %% "
                 "(FAR при t₁%% %.2f %%) [по медиане: t₁%% = %.2f, запас %.2f, ошибка своих %.1f %% → %.1f %%]" % (
                     N, who, t, N_GRID[j1], N_GRID[j1] - t, 100 * perr[j], 100 * perr[j1], 100 * kk[j1] / nn, N_GRID[jm], N_GRID[jm] - t,
                     100 * err[j], 100 * err[jm]))
    L.append("")
    L.append("=== Справочно: s₀ = ℓ₁ (β = 0), N = 10 – t по строкам ===")
    cand = []
    for name, fs, main, rn in N_ROWS:
        if (name, "s0", 10) not in res:
            continue
        j = limit((name, "s0", 10), N_GRID_D1)
        if main:
            cand.append((j, name))
        L.append("  %-14s %s" % (name + ("" if main else " *"), "< сетки" if j < 0 else "%.2f" % N_GRID_D1[j]))
    j, who = min(cand)
    L.append("  t₀(10) = %.2f (задаёт %s)" % (N_GRID_D1[j], who))
    L.append("")
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# duel_thr.py ntable: сетка t по s (β = 1) и s0 (β = 0, N = 10), медианы по подвыборкам\n" + "\n".join(T) + "\n")
    thr = os.path.join(os.path.dirname(out_txt), "duel_thr_N.tsv")
    TL = ["# Пороги (Н) – duel_thr.py ntable; s = 2ℓ₁ − ℓ₂ после слияния (w = 0,75), приведение по (П); постоянные – медианы по подвыборкам",
          "N\tt\tdetermined_by\tmu_tt\tsigma_tt\tmu_xs\tsigma_xs"]
    for N in N_NS:
        # Постоянные – у строки, задающей t (у FERET – медиана по обоим проходам).
        m = consts[(sel[N][1], N)][0]
        TL.append("%d\t%.2f\t%s\t%.6f\t%.6f\t%.6f\t%.6f" % (N, sel[N][0], sel[N][1], m[0], m[1], m[2], m[3]))
    with open(thr, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(TL) + "\n")
    L.append("Таблица порогов: %s, SHA-256 %s" % (os.path.basename(thr), sha(thr)))
    # Проверка (Б) для (Н).
    L.append("")
    L.append("=== Проверка (Б) для (Н): W заново на 10 людях против общей W на тех же подвыборках, s, порог t(10) ===")
    t10 = sel[10][0]
    for name, fn, d, bi in (("GT межс.", "gt_x", vec_dir, 0), ("FERET межс.", "feret_x_p0", feret_dir, 0)):
        pt = os.path.join(d, "b_%s_f1_n.tsv.gz" % fn)
        px_ = os.path.join(d, "b_%s_f0_n.tsv.gz" % fn)
        if not (os.path.exists(pt) and os.path.exists(px_)):
            L.append("  %s: файлов проверки (Б) нет" % name)
            continue
        btt = n_split_blocks(pt)
        bxs = n_split_blocks(px_)
        common_tt, common_xs = data[name][bi]
        kb = nb = kc = nc = 0
        eb = ec = no = 0
        for (tl, a), (xl, b) in zip(btt, bxs):
            G0 = np.arange(len(a.P))
            zs, _, _, os_, _, wr, _ = n_run(a, b, G0)
            kb += int((zs <= t10).sum())
            nb += len(zs)
            eb += int((wr | (os_ > t10)).sum())
            G = np.array(sorted(common_tt.P.index(x) for x in a.P))
            zs2, _, _, os2, _, wr2, _ = n_run(common_tt, common_xs, G)
            kc += int((zs2 <= t10).sum())
            nc += len(zs2)
            ec += int((wr2 | (os2 > t10)).sum())
            no += len(os_)
        lo, hi = binom_q(nb, TARGET, 0.025), binom_q(nb, TARGET, 0.975)
        de = 100 * (eb - ec) / no
        pos = "ниже интервала (FAR меньше цели)" if kb < lo else ("в интервале" if kb <= hi else "ВЫШЕ интервала")
        L.append("  %-12s t = %.2f: FAR (Б) %d/%d = %.2f %% [↑95 %.2f], общая W %d/%d = %.2f %%; 95 %% интервал цели 1 %% – %d…%d принятых → %s; ошибка своих "
                 "(Б) %.1f %%, общая %.1f %% (%+.1f п. п., %s)" % (name, t10, kb, nb, 100 * kb / nb, 100 * cp_upper(kb, nb), kc, nc, 100 * kc / nc, lo, hi,
                                                                  pos, 100 * eb / no, 100 * ec / no, de, "≤ 1" if abs(de) <= 1 else "БОЛЬШЕ 1"))
    # Проверка N = 50 (основной Жорик №9): W заново на 5 подвыборках против общей W на тех же, порог t(50) общей W.
    L.append("")
    L.append("=== Проверка N = 50: W заново на подвыборке против общей W на тех же подвыборках, s, порог t(50) ===")
    t50 = sel[50][0]
    need50 = False
    for name, fs, main, rn in N_ROWS:
        if (name, 50) not in rdata:
            if name in R_ROWS[50]:
                L.append("  %s: файлов переобучения r50 нет" % name)
            continue
        common_tt, common_xs = data[name][0]
        kb = nb = kc = nc = eb = ec = no = 0
        for a, b in rdata[(name, 50)]:
            zs, _, _, os_, _, wr, _ = n_run(a, b, np.arange(len(a.P)))
            kb += int((zs <= t50).sum())
            nb += len(zs)
            eb += int((wr | (os_ > t50)).sum())
            G = np.array(sorted(common_tt.P.index(x) for x in a.P))
            zs2, _, _, os2, _, wr2, _ = n_run(common_tt, common_xs, G)
            kc += int((zs2 <= t50).sum())
            nc += len(zs2)
            ec += int((wr2 | (os2 > t50)).sum())
            no += len(os_)
        lo, hi = binom_q(nb, TARGET, 0.025), binom_q(nb, TARGET, 0.975)
        de = 100 * (eb - ec) / no
        out_i = kb < lo or kb > hi
        need50 |= out_i or abs(de) > 1
        L.append("  %-12s t = %.2f, подвыборок %d: FAR (W заново) %d/%d = %.2f %%, общая W %d/%d = %.2f %%; 95 %% интервал цели – %d…%d → %s; ошибка "
                 "своих %.1f / %.1f %% (%+.1f п. п.)" % (name, t50, len(rdata[(name, 50)]), kb, nb, 100 * kb / nb, kc, nc, 100 * kc / nc, lo, hi,
                                                     "вне интервала" if out_i else "в интервале", 100 * eb / no, 100 * ec / no, de))
    if any((n, 50) in rdata for n in R_ROWS[50]):
        L.append("  Итог по критерию (FAR вне интервала в любую сторону или ошибка своих > 1 п. п.): N = 50 – %s" % (
            "нужно переобучение W" if need50 else "общая W"))
    with open(out_txt, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))


def n_split_blocks(path):
    """Файл duel-thr-n-b → [(подпись, NPass)] по подвыборкам (через временные куски)."""
    out = []
    head, cur = [], None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#S\t"):
                if cur is not None:
                    out.append(cur)
                cur = [line]
            elif cur is not None:
                cur.append(line)
    if cur is not None:
        out.append(cur)
    res = []
    for blk in out:
        tmp = path + ".part.gz"
        with gzip.open(tmp, "wt", encoding="utf-8") as fo:
            fo.writelines(blk)
        res.append((blk[0].rstrip("\n"), NPass(tmp, N_EXCL_R)))
        os.remove(tmp)
    return res


# ---------------------------------------------------------------- п. 2.2: дуэльный набор с фиксированным порогом (duel)

ROLES = (("свой-тот-же-день", "тот же день"), ("свой-другой-день", "другой день"))


def duel(thr_tsv, plain_tsv, face_tsv, out):
    """Строки L1-tt-xs-F, L1-xs-F с порогом (Д) N = 10 по оценкам файлов проб duel-eval-24 (К-обычный) и duel-eval-24-yunet400 (К-лицо)."""
    thr = {}
    with open(thr_tsv, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if line.startswith("#") or c[0] == "feat":
                continue
            thr[(c[0], c[1], int(c[2]), c[3])] = float(c[4])
    L = ["=== Часть 2. Дуэльный набор GT_24.10 с фиксированным порогом (Д), N = 10 ===",
         "Таблица порогов – %s, SHA-256 %s (сообщена Жорику (дуэль) до этого шага)." % (os.path.basename(thr_tsv), sha(thr_tsv)),
         "Оценки – файлы проб прежних прогонов без пересчёта: К-обычный – duel_24.10_probes.tsv (duel-eval-24, 699f3b9; SHA-256 %s), К-лицо – "
         "duel_24.10_yunet400_probes.tsv (duel-eval-24-yunet400, e0c3554; вход YuNet 400 × 600; SHA-256 %s). У L1-tt-xs-F и L1-xs-F в них best_score = "
         "d₁/d₂ (FeiScores.ratio), second_score = d₂; меняется только порог – лучший кандидат и оценки те же. d₁ = d₁/d₂ · d₂ (справочно)."
         % (sha(plain_tsv), sha(face_tsv)),
         "Меры: верный первый кандидат; свои – приняты верно / за другого / отвергнуты (включая отказы детектора и отбора); чужие (20 проб, 10 "
         "человек) – принято проб и людей. 10 человек и 20 чужих – выводов о FAR нет (Д12).", ""]
    for set_, path, title in (("plain", plain_tsv, "К-обычный (duel-eval-24)"), ("face", face_tsv, "К-лицо (вход YuNet 400 × 600, duel-eval-24-yunet400)")):
        rows = {}
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[0] == set_:
                    rows.setdefault(c[1], []).append(c)
        lines = []
        for m in ("L1-tt-xs-F", "L1-xs-F"):
            for sk, nm in (("d1/d2", ""), ("d1", ", справочно d₁")):
                t = thr[(m, sk, 10, "no")]
                lines.append(("%s, фикс. порог %s ≤ %.3f%s" % (m, "d₁/d₂" if sk == "d1/d2" else "d₁", t, nm), m, sk, t))
            lines.append(("%s, П3 (прежняя строка)" % m, m, None, None))
        lines.append(("SFace, 1 − cos ≤ 0,53 (прежняя строка)", "SFace", None, None))
        L.append("--- Комплект %s ---" % title)
        L.append("%-48s | %-26s | %-26s | чужие: проб из 20 / людей из 10" % ("строка", "тот же день (20): первый; верно / другой / отв.",
                                                                                  "другой день (30): то же"))
        for lab, m, sk, t in lines:
            cells = []
            acc_st, acc_p = 0, set()
            for rk, rn in ROLES:
                sel = [c for c in rows[m] if c[4] == rk]
                first = sum(c[11] == "true" for c in sel)
                ok = oth = rej = 0
                for c in sel:
                    sc = float(c[8])
                    if sk is None:
                        a = c[12] == "true"
                    elif c[6] != "":
                        a = False
                    else:
                        a = (sc if sk == "d1/d2" else sc * float(c[10])) <= t
                    if a and c[11] == "true":
                        ok += 1
                    elif a:
                        oth += 1
                    else:
                        rej += 1
                cells.append("%2d; %2d / %2d / %2d (%d %%)" % (first, ok, oth, rej, round(100 * ok / len(sel))))
            st = [c for c in rows[m] if c[4] == "чужой"]
            for c in st:
                sc = float(c[8])
                a = (c[12] == "true") if sk is None else (c[6] == "" and (sc if sk == "d1/d2" else sc * float(c[10])) <= t)
                if a:
                    acc_st += 1
                    acc_p.add(c[5])
            L.append("%-48s | %-26s | %-26s | %2d (%d %%) / %2d (%d %%)" % (lab, cells[0], cells[1], acc_st, round(100 * acc_st / len(st)), len(acc_p),
                                                                           round(100 * len(acc_p) / len({c[5] for c in st}))))
        L.append("")
    with open(out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "ntable":
        ntable(*sys.argv[2:6])
    elif sys.argv[1] == "duel":
        duel(*sys.argv[2:6])
    elif sys.argv[1] == "check":
        check(*sys.argv[2:7])
    elif sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "nsubs":
        nsubs(*sys.argv[2:8])
    elif sys.argv[1] == "fastcheck":
        fastcheck(*sys.argv[2:5])
    elif sys.argv[1] == "table":
        table(*sys.argv[2:6])
