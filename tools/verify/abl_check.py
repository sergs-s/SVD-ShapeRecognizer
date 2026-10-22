"""(4а) поочерёдное отключение: строки отчётов feret-abl и gt-abl по их файлам оценок (Жорик №8 от 20.10.2026).

Строки собираются по постановке PLAN.md, (4а): блоки файла – схемы база, (1), (2), (3), (8), (9), (9б); строки (4)–(6) – из
блока базы:
- (4) – оценка d₁/1e6 вместо d₁/d₂ (второе расстояние – постоянная);
- (5) – всегда модель X;
- (6) – всегда X+mirror.

Строки Хаара (9):
- всегда X+mirror;
- (i) – проба без двух глаз Хаара – отказ своему и непринятый чужой;
- (ii) – как в файле;
- проба без лица (её нет в блоке (9)) – отказ и непринятый в обеих.

Справочная «(9)(i) + отбор и модель по r_H»: отбор |r_H| ≤ R_H, модель X при |r_H| ≤ r0_H.

(9б) – модель по r_LBF ≤ r0_LBF (вариант «yunet» – по r YuNet).

[Хаар: лицо и два глаза] – только пробы с лицом и двумя глазами Хаара.

GT, (7):
- «база» – пробы I (|r| ≤ R);
- «база (а)» – пробы U, отвергнутая отбором – отказ и непринятый;
- «отключено» – пробы U как есть.

Кэши Хаара и LBF: FERET – derived/haar.tsv, lbf.tsv; GT – VERIFY_GT_RAW/haar.tsv, lbf.tsv (SVD-faces-data main 7f40554e, raw/; Жорик
№17 от 20.10.2026). Без VERIFY_GT_RAW строки GT, которым нужны кэши, не строятся (как прежде).
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
from compare import DATA, FERET, read_scores  # noqa: E402
from measures import R0, Row, pose_r, ratio  # noqa: E402

GT_ABL = os.environ.get("VERIFY_GT_ABL", "")
GT_RAW = os.environ.get("VERIFY_GT_RAW", "")
R = 0.3869555959643167
# r0_H в отчёте – 4 знака (0.0549); точное значение ≥ 0,054945 (|r_H| снимка, модель X по отчёту) – берём 0,054948
R_H, R0_H = 0.0594, 0.054948
R_LBF, R0_LBF = 0.2486, 0.0841
LBF_MODEL = "lbf"  # «lbf» – выбор модели по r_LBF; «yunet» – по r YuNet (проверка соглашения)
INF = float("inf")


def feret_keys():
    """(человек, номер снимка) → ключ; номер – порядковый в описи fa/fb (как compare.feret_pose)."""
    out = {}
    seen = {}
    with open(FERET + "/derived/detections.tsv", encoding="utf-8") as f:
        for line in f:
            key = line.split("\t", 1)[0]
            p = key.split("_")[0]
            seen[p] = seen.get(p, 0) + 1
            out[(p, str(seen[p]))] = key
    return out


def haar(path=None):
    out = {}
    with open(path or FERET + "/derived/haar.tsv", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) != 12:
                continue
            face = None if c[2] == "" else [float(x) for x in c[2:6]]
            le = None if c[8] == "" else (float(c[8]), float(c[9]))
            re_ = None if c[10] == "" else (float(c[10]), float(c[11]))
            rh = float("nan")
            if face and le and re_:
                rh = ((le[0] + re_[0]) / 2 - (face[0] + face[2] / 2)) / abs(re_[0] - le[0])
            out[c[0]] = (face is not None, (le is not None) + (re_ is not None), rh)
    return out


def lbf(path=None):
    out = {}
    with open(path or FERET + "/derived/lbf.tsv", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 2 or c[1] != "1":
                out[c[0]] = float("nan")
                continue
            v = [float(x) for x in c[2:2 + 136]]
            pt = lambda i: (v[2 * i], v[2 * i + 1])  # noqa: E731
            lx = sum(pt(i)[0] for i in range(36, 42)) / 6
            rx = sum(pt(i)[0] for i in range(42, 48)) / 6
            out[c[0]] = (pt(30)[0] - (lx + rx) / 2) / (rx - lx)
    return out


def blocks(path, two_pass):
    """(схема, постановка, кандидат) → [(проход, строки)]."""
    out = collections.defaultdict(list)
    for title, b in read_scores(path):
        t = [x.strip() for x in title.split(",")]
        scheme = t[0].replace("схема ", "")
        out[(scheme, t[1], t[2].replace("кандидат ", ""))].append((t[3].replace("проход ", ""), b))
    return out


def attempts(rows, methods):
    """Попытки блока: ключ (s, j, роль, человек, снимок) → {метод: строка}."""
    by = collections.defaultdict(dict)
    for d in rows:
        if d["method"] in methods:
            by[(d["s"], d["j"], d["role"], d["person"], d["image"])][d["method"]] = d
    return by


def build(parts, base_parts, method, choose, score="ratio", reject=None, include=None, missing_fail=False, pose_of=None):
    """Строка отчёта по блокам проходов → conf {s: {own, ctrl, cv}}.

    choose(person, image) → True – модель X+mirror; reject(person, image) → проба отклонена (отказ / непринят);
    include(person, image) → проба в строке; missing_fail – попытки базы, которых нет в блоке, – отказ / непринят.
    """
    conf = {}
    for tag, (ps, rows) in enumerate(parts):
        by = attempts(rows, (method, method + "+mirror"))
        base_by = None
        if missing_fail:
            base_by = attempts(dict(base_parts)[ps], ("5-lbp-mlda", "5-lbp-mlda+mirror"))
        # попытки базы, которых нет в блоке (нет лица), – только пробы своих и контроля; снимок галереи без лица выпадает
        # из галереи и «чужих» П2/П3 (их попыток cv в блоке нет – так и остаётся)
        keys = set(by) | ({k for k in base_by if not k[2].startswith("cv")} if base_by else set())
        for key in keys:
            s, j, role, person, image = key
            rr = "cv" if role.startswith("cv") else role
            if include is not None and rr != "cv" and not include(person, image):
                continue
            # «анфас / поворот» в отчётах – по выбранной модели: X – анфас, X+mirror – поворот
            pose = 1.0 if choose(person, image) else 0.0
            slot = conf.setdefault(int(ps) * 1000 + int(s), {"own": [], "ctrl": [], "cv": []})[rr]
            if key not in by:  # нет лица – отказ своему, непринятый чужой
                slot.append((INF, 2, person, pose))
                continue
            pair = by[key]
            d = pair[method + "+mirror"] if choose(person, image) else pair[method]
            if d["no_agreement"] != "-":
                raise ValueError(key)
            d1, d2 = float(d["best_score"]), float(d["second_score"])
            r = ratio(d1, 1e6) if score == "d1" else ratio(d1, d2)
            wrong = d["best_id"] != d["true_id"]
            if rr != "cv" and reject is not None and reject(person, image):
                r, wrong = INF, 2  # принудительный отказ: в ошибке, но не в (ii) и не в (iii) (соглашение отчёта, подобрано сверкой)
            slot.append((r, wrong, person, pose))
    return conf


def rows_feret():
    path = FERET + "/xbase_feret_abl_scores.tsv.gz"
    bl = blocks(path, True)
    pose = compare.feret_pose()
    keys = feret_keys()
    hz = haar()
    lb = lbf()
    yun = lambda p, i: abs(pose(p, i)) > R0  # noqa: E731 – X+mirror при |r| > r0
    h_two = lambda p, i: hz.get(keys[(p, i)], (False, 0, 0))[1] == 2 and hz[keys[(p, i)]][0]  # noqa: E731
    out = {}
    for role in ("межс.", "G2", "внутри"):
        for cand in ("L1-xs-F", "L1-tt-xs-F"):
            b = bl[("база", role, cand)]
            p = "%s %s " % (role, cand)
            out[p + "база"] = build(b, b, "5-lbp-mlda", yun, pose_of=pose)
            out[p + "(1) без пула"] = build(bl[("(1) без пула", role, cand)], b, "5-lbp-mlda", yun, pose_of=pose)
            out[p + "(2) без корня"] = build(bl[("(2) без корня", role, cand)], b, "5-lbp-mlda", yun, pose_of=pose)
            out[p + "(3) χ² L0-mean"] = build(bl[("(3) χ²", role, cand)], b, "5-lbp-chi-mean", yun, pose_of=pose)
            out[p + "(3) χ² L0 (справочно)"] = build(bl[("(3) χ²", role, cand)], b, "5-lbp-chi", yun, pose_of=pose)
            out[p + "(4) d₁ вместо d₁/d₂"] = build(b, b, "5-lbp-mlda", yun, score="d1", pose_of=pose)
            out[p + "(5) без зеркал"] = build(b, b, "5-lbp-mlda", lambda p_, i: False, pose_of=pose)
            out[p + "(6) без выбора модели"] = build(b, b, "5-lbp-mlda", lambda p_, i: True, pose_of=pose)
            out[p + "(8) кадр по рамке YuNet"] = build(bl[("(8) кадр по рамке", role, cand)], b, "5-lbp-mlda", yun, pose_of=pose)
            h = bl[("(9) Хаар", role, cand)]
            out[p + "(9)(i) Хаар"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, reject=lambda p_, i: not h_two(p_, i),
                                           missing_fail=True, pose_of=pose)
            # проба без лица Хаара остаётся в файле (оценена по кадру-заглушке), отказ ей ставится при разборе (по коду:
            # XBaseAbl.java:442, 455, refuse(ls, noFace)) – здесь по кэшу haar.tsv
            no_face = lambda p_, i: not hz.get(keys[(p_, i)], (False, 0, 0))[0]  # noqa: E731
            out[p + "(9)(ii) Хаар, глаза по рамке"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, reject=no_face, missing_fail=True,
                                                             pose_of=pose)

            def rh(p_, i):
                # без двух глаз r_H = 0 (модель X) – соглашение кода (XBaseAbl.java:446–453), найдено после расхождения
                v = hz[keys[(p_, i)]][2]
                return 0.0 if v != v else v
            out[p + "(9)(i) Хаар + отбор и модель по r_H (справочно)"] = build(
                h, b, "5-lbp-mlda", lambda p_, i: not (abs(rh(p_, i)) <= R0_H),
                reject=lambda p_, i: not h_two(p_, i) or not (abs(rh(p_, i)) <= R_H), missing_fail=True, pose_of=pose)
            if LBF_MODEL == "lbf":
                # нет точек LBF – r_LBF = 0 (модель X; проба всё равно отклонена) – как в коде (XBaseAbl.java:471–477)
                lchoose = lambda p_, i: not (abs(lb.get(keys[(p_, i)], 0.0) if lb.get(keys[(p_, i)], 0.0) == lb.get(keys[(p_, i)], 0.0)
                                             else 0.0) <= R0_LBF)  # noqa: E731
            else:
                lchoose = yun
            # отбор |r_LBF| ≤ R_LBF, нет точек – отказ (постановка; по коду – XBaseAbl.java:471–478)
            lrej = lambda p_, i: not (abs(lb.get(keys[(p_, i)], float("nan"))) <= R_LBF)  # noqa: E731
            out[p + "(9б) Хаар + LBF"] = build(bl[("(9б) Хаар + LBF", role, cand)], b, "5-lbp-mlda", lchoose, reject=lrej,
                                                missing_fail=True, pose_of=pose)
            out[p + "база [Хаар: лицо и два глаза]"] = build(b, b, "5-lbp-mlda", yun, include=h_two, pose_of=pose)
            out[p + "(9)(ii) Хаар [Хаар: лицо и два глаза]"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, include=h_two,
                                                                       missing_fail=True, pose_of=pose)
    # люди бутстрепа – пул (173) и контроль (как код)
    persons = set()
    for (sch, role, cand), parts in bl.items():
        for ps, rows in parts:
            for d in rows:
                persons.add(d["person"])
    return out, FERET + "/xbase_feret_abl.txt", sorted(persons)


def rows_gt():
    path = GT_ABL + "/xbase_gt_abl_scores.tsv.gz"
    bl = blocks(path, False)
    pose = compare.gt_pose()
    yun = lambda p, i: abs(pose(p, i)) > R0  # noqa: E731
    inI = lambda p, i: abs(pose(p, i)) <= R  # noqa: E731
    out = {}
    for role in ("межс.", "внутри"):
        for cand in ("L1-xs-F", "L1-tt-xs-F"):
            b = bl[("база", role, cand)]
            p = "%s %s " % (role, cand)
            out[p + "база"] = build(b, b, "5-lbp-mlda", yun, include=inI, pose_of=pose)
            out[p + "(1) без пула"] = build(bl[("(1) без пула", role, cand)], b, "5-lbp-mlda", yun, include=inI, pose_of=pose)
            out[p + "(2) без корня"] = build(bl[("(2) без корня", role, cand)], b, "5-lbp-mlda", yun, include=inI, pose_of=pose)
            out[p + "(3) χ² L0-mean"] = build(bl[("(3) χ²", role, cand)], b, "5-lbp-chi-mean", yun, include=inI, pose_of=pose)
            out[p + "(3) χ² L0 (справочно)"] = build(bl[("(3) χ²", role, cand)], b, "5-lbp-chi", yun, include=inI, pose_of=pose)
            out[p + "(4) d₁ вместо d₁/d₂"] = build(b, b, "5-lbp-mlda", yun, score="d1", include=inI, pose_of=pose)
            out[p + "(5) без зеркал"] = build(b, b, "5-lbp-mlda", lambda p_, i: False, include=inI, pose_of=pose)
            out[p + "(6) без выбора модели"] = build(b, b, "5-lbp-mlda", lambda p_, i: True, include=inI, pose_of=pose)
            out[p + "база (а): отвергнутые отбором – отказ"] = build(b, b, "5-lbp-mlda", yun, reject=lambda p_, i: not inI(p_, i), pose_of=pose)
            out[p + "(7) без отбора по позе"] = build(b, b, "5-lbp-mlda", yun, pose_of=pose)
            out[p + "(8) кадр по рамке YuNet"] = build(bl[("(8) кадр по рамке", role, cand)], b, "5-lbp-mlda", yun, include=inI,
                                                       pose_of=pose)
            h = bl[("(9) Хаар", role, cand)]
            if not GT_RAW:
                out[p + "(9)(ii) Хаар, глаза по рамке"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, include=inI, missing_fail=True,
                                                                 pose_of=pose)
                continue
            # ключ кэша GT – «человек/номер снимка» (как detections.tsv экспорта, compare.gt_pose)
            hz = haar(GT_RAW + "/haar.tsv")
            lb = lbf(GT_RAW + "/lbf.tsv")
            key = lambda p_, i: "%s/%02d" % (p_, int(i))  # noqa: E731
            h_two = lambda p_, i: hz[key(p_, i)][0] and hz[key(p_, i)][1] == 2  # noqa: E731
            no_face = lambda p_, i: not hz[key(p_, i)][0]  # noqa: E731
            out[p + "(9)(i) Хаар"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, reject=lambda p_, i: not h_two(p_, i), include=inI,
                                           missing_fail=True, pose_of=pose)
            out[p + "(9)(ii) Хаар, глаза по рамке"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, reject=no_face, include=inI,
                                                             missing_fail=True, pose_of=pose)

            def rh(p_, i):
                v = hz[key(p_, i)][2]
                return 0.0 if v != v else v
            out[p + "(9)(i) Хаар + отбор и модель по r_H (справочно)"] = build(
                h, b, "5-lbp-mlda", lambda p_, i: not (abs(rh(p_, i)) <= R0_H),
                reject=lambda p_, i: not h_two(p_, i) or not (abs(rh(p_, i)) <= R_H), include=inI, missing_fail=True, pose_of=pose)

            def rl(p_, i):
                v = lb.get(key(p_, i), float("nan"))
                return v
            lchoose = lambda p_, i: not (abs(rl(p_, i) if rl(p_, i) == rl(p_, i) else 0.0) <= R0_LBF)  # noqa: E731
            lrej = lambda p_, i: not (abs(rl(p_, i)) <= R_LBF)  # noqa: E731
            out[p + "(9б) Хаар + LBF"] = build(bl[("(9б) Хаар + LBF", role, cand)], b, "5-lbp-mlda", lchoose, reject=lrej, include=inI,
                                                missing_fail=True, pose_of=pose)
            both = lambda p_, i: inI(p_, i) and h_two(p_, i)  # noqa: E731
            out[p + "база [Хаар: лицо и два глаза]"] = build(b, b, "5-lbp-mlda", yun, include=both, pose_of=pose)
            out[p + "(9)(ii) Хаар [Хаар: лицо и два глаза]"] = build(h, b, "5-lbp-mlda", lambda p_, i: True, include=both,
                                                                       missing_fail=True, pose_of=pose)
    persons = set()
    for parts in bl.values():
        for ps, rows in parts:
            for d in rows:
                persons.add(d["person"])
    return out, GT_ABL + "/xbase_gt_abl.txt", sorted(persons)


def abl_rows(kind):
    confs, rep, persons = rows_feret() if kind == "feret" else rows_gt()
    pidx = {p: i for i, p in enumerate(persons)}
    return {lab: Row(c, pidx) for lab, c in confs.items()}, rep, persons
