"""Точки и поза MediaPipe Face Landmarker для этапа 1 пункта (8) (PLAN.md п. 5.5) – кэш для XBaseEval feret-mppts / gt-mppts.

Запуск – venv с mediapipe (CLAUDE.md, «Тестовые данные»):
  python mp_points.py feret <каталог FERET> <face_landmarker.task> <выходной TSV>
  python mp_points.py gt <каталог gt_db> <face_landmarker.task> <выходной TSV>
  python mp_points.py dump <каталог FERET или gt_db> <ключ>[,<ключ>…] <каталог PNG>   – раскодированные снимки для сверки с Java

Сеть в процессе запрещена (подключения дают ошибку): модель читается только с диска. FERET: все *.ppm.bz2 обоих дисков,
распаковка в памяти (на диск не пишется). GT: все sNN/*.jpg. Снимок раскодируется OpenCV этого venv (BGR → RGB для
MediaPipe). Лиц ищется до 2, берётся наибольшее (площадь прямоугольника точек) – как правило ПО «наибольшая рамка».
Выход – TSV через временную дозапись: уже записанные ключи пропускаются (продолжение после обрыва), оборванная
последняя строка отбрасывается. Координаты – пиксели полного снимка (x·w, y·h); z – в тех же единицах (z·w).
"""
import bz2
import hashlib
import math
import os
import socket
import sys


def _deny(*args, **kwargs):
    raise RuntimeError("mp_points: сеть запрещена")


socket.socket.connect = _deny
socket.create_connection = _deny
socket.getaddrinfo = _deny

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

# Имя – индекс из 478. Радужки: 468 – центр радужки глаза слева на снимке (правый глаз человека), 473 – справа.
POINTS = [("iris_re", 468), ("iris_le", 473), ("eye_re_out", 33), ("eye_re_in", 133), ("eye_le_in", 362), ("eye_le_out", 263),
          ("nose", 1), ("mouth_r", 61), ("mouth_l", 291)]
BLINK = ["eyeBlinkRight", "eyeBlinkLeft"]


def header():
    h = ["key", "found", "faces", "w", "h"]
    for name, _ in POINTS:
        h += [name + "_x", name + "_y", name + "_z"]
    h += ["yaw", "pitch", "roll"] + ["m%d%d" % (i, j) for i in range(4) for j in range(4)] + BLINK
    return h


def images(kind, root):
    out = []
    if kind == "feret":
        for dvd in ("dvd1", "dvd2"):
            base = os.path.join(root, dvd, "data", "images")
            for d, _, files in os.walk(base):
                for f in files:
                    if f.endswith(".ppm.bz2"):
                        out.append((f[:-len(".ppm.bz2")], os.path.join(d, f)))
    else:
        for p in sorted(os.listdir(root)):
            pd = os.path.join(root, p)
            if os.path.isdir(pd):
                for f in sorted(os.listdir(pd)):
                    if f.lower().endswith(".jpg"):
                        out.append((p + "/" + f[:-4], os.path.join(pd, f)))
    out.sort()
    return out


def decode(path):
    data = open(path, "rb").read()
    if path.endswith(".bz2"):
        data = bz2.decompress(data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("не декодируется: " + path)
    return img


def angles(m):
    r = m[:3, :3]
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def row(key, bgr, lm):
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    n = len(res.face_landmarks)
    if n == 0:
        return [key, "0", "0", str(w), str(h)] + [""] * (len(header()) - 5)
    areas = []
    for f in res.face_landmarks:
        xs = [p.x for p in f]
        ys = [p.y for p in f]
        areas.append((max(xs) - min(xs)) * w * (max(ys) - min(ys)) * h)
    k = int(np.argmax(areas))
    f = res.face_landmarks[k]
    out = [key, "1", str(n), str(w), str(h)]
    for _, i in POINTS:
        out += ["%.3f" % (f[i].x * w), "%.3f" % (f[i].y * h), "%.3f" % (f[i].z * w)]
    m = np.array(res.facial_transformation_matrixes[k])
    out += ["%.4f" % a for a in angles(m)]
    out += ["%.6g" % v for v in m.flatten()]
    bs = {c.category_name: c.score for c in res.face_blendshapes[k]}
    out += ["%.5f" % bs[b] for b in BLINK]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(kind, root, model, out):
    done = set()
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        good = [l for l in lines if l.startswith("#") or len(l.split("\t")) == len(header())]
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(l + "\n" for l in good))
        done = {l.split("\t")[0] for l in good if not l.startswith("#") and not l.startswith("key\t")}
    todo = [(k, p) for k, p in images(kind, root) if k not in done]
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.IMAGE,
                                        num_faces=2, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    with vision.FaceLandmarker.create_from_options(opts) as lm, open(out, "a", encoding="utf-8", newline="\n") as f:
        if not done:
            f.write("# mediapipe %s; opencv %s; numpy %s; python %s\n" % (mp.__version__, cv2.__version__, np.__version__, sys.version.split()[0]))
            f.write("# model %s SHA-256 %s\n" % (os.path.basename(model), sha256(model)))
            f.write("# %s %s; координаты – пиксели снимка; 468 – iris_re (на снимке слева), 473 – iris_le\n" % (kind, root))
            f.write("\t".join(header()) + "\n")
        for i, (k, p) in enumerate(todo):
            f.write("\t".join(row(k, decode(p), lm)) + "\n")
            f.flush()
            if (i + 1) % 500 == 0:
                print("mp_points %s: %d / %d" % (kind, i + 1, len(todo)), flush=True)
    print("mp_points %s: готово, новых %d, всего %d" % (kind, len(todo), len(done) + len(todo)), flush=True)


def dump(root, keys, outdir):
    os.makedirs(outdir, exist_ok=True)
    kind = "gt" if os.path.isdir(os.path.join(root, "s01")) else "feret"
    paths = dict(images(kind, root))
    for k in keys:
        cv2.imwrite(os.path.join(outdir, k.replace("/", "_") + ".png"), decode(paths[k]))
    print("mp_points dump: %d снимков -> %s" % (len(keys), outdir))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] in ("feret", "gt"):
        run(a[0], a[1], a[2], a[3])
    elif len(a) == 4 and a[0] == "dump":
        dump(a[1], a[2].split(","), a[3])
    else:
        sys.exit(__doc__)
