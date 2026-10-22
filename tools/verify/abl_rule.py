"""(4а): правило «убрать» (PLAN.md, (4а), «Правило, заранее») по своим числам сверки (файлы ячеек compare для abl-feret и
abl-gt) – разность разделимости при FAR контроля 1 % «отключено − база», точка и [2,5; 97,5].

Правило: приём – кандидат на удаление, если на всех основных протоколах (FERET 1 и 2 даты, GT межсессионно) отключение не хуже
(точка ≤ 0 или верхняя граница ≤ +3 п. п.) и внутри сессии (FERET, GT) не хуже на 3 п. п. (точка ≤ +3); иначе – остаётся.

python3 -I abl_rule.py <ячейки FERET> <ячейки GT>
"""
import sys

ROWS = ["(1) без пула", "(2) без корня", "(3) χ² L0-mean", "(4) d₁ вместо d₁/d₂", "(5) без зеркал", "(6) без выбора модели",
        "(7) без отбора по позе", "(8) кадр по рамке YuNet", "(9)(i) Хаар", "(9б) Хаар + LBF"]
MAIN = [("FERET", "межс."), ("FERET", "G2"), ("GT", "межс.")]
INSIDE = [("FERET", "внутри"), ("GT", "внутри")]


def load(path):
    """(постановка, кандидат, строка) → (точка, нижняя, верхняя) – мои числа пары разделимости FAR 1 %."""
    out = {}
    pts = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 4 or "пара разд. ('sep', 0.01, 'all')" not in p[0]:
            continue
        pair = p[0].split(" | ")[0]
        a, b = pair.split(" − ")
        role, cand = a.split(" ")[0], a.split(" ")[1]
        row = a[len(role) + len(cand) + 2:]
        key = (role, cand, row)
        if p[1].startswith("["):
            lo, hi = [float(x) for x in p[2].strip("[]").split(";")]
            out[key] = (pts.get(key), lo, hi)
        else:
            pts[key] = float(p[2])
    for k, v in out.items():
        out[k] = (pts.get(k), v[1], v[2])
    return out


def main(fp, gp):
    d = {"FERET": load(fp), "GT": load(gp)}
    lines = []
    for cand in ("L1-xs-F", "L1-tt-xs-F"):
        lines.append("=== %s ===" % cand)
        for row in ROWS:
            cells, ok_all, any_data = [], True, False
            for base, role in MAIN + INSIDE:
                v = d[base].get((role, cand, row))
                if v is None or v[0] is None:
                    cells.append("%s %s: –" % (base, role))
                    continue
                any_data = True
                pt, lo, hi = v
                if (base, role) in MAIN:
                    ok = pt <= 0 or hi <= 3
                else:
                    ok = pt <= 3
                ok_all &= ok
                cells.append("%s %s: %+.1f [%+.1f; %+.1f] %s" % (base, role, pt, lo, hi, "не хуже" if ok else "хуже"))
            verdict = "кандидат на удаление" if any_data and ok_all else ("остаётся" if any_data else "нет данных")
            if row.startswith("(7)"):
                verdict += " (только GT; FERET – отбор ничего не отвергает)"
            lines.append("%s | %s | %s" % (row, " | ".join(cells), verdict))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
