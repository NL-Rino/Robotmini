"""Can nha: nhieu phong, ban ghe, nguoi di lai.

Ban truoc mo phong MOT PHONG 6,4 x 4,8 m - to gap 20 lan con xe, va tu
giua phong thi LiDAR nhin thay gan het. Trong cai phong do khong co gi de
kham pha, va "di toi cho moi" khong co nghia gi.

Day la mot CAN NHA 12 x 9 m chia lam bon phong thong nhau qua o cua. Tu mot
cho bat ky, xe chi nhin thay phong dang dung; muon biet phong ben canh co
gi thi phai di qua cua. Do la dieu kien de viec KHAM PHA co nghia.

Do dac duoi mat LiDAR (cao khoang 20 cm):

  - ban va ghe: chi con CHAN, tuc la nhung cham tron 2,5 cm. Mot cai ghe la
    bon cham cach nhau 35 cm. Do la thu kho nhat trong can nha: no gan
    giong nhieu cam bien, va bo do hoc rat de nham bon cham do voi mot cai
    hoc.
  - ghe sofa, tu: khoi dac, hien ra la mot canh thang.
  - nguoi: hai cai chan, mot cai nhap nhay theo nhip buoc (xem `Mover`).

Hoc sac dat o CAC PHONG KHAC NHAU, nen xe khong the dung mot cho ma thay
het - phai nho duong.
"""

import math
import random

from . import params as P
from .geometry import SegmentSet, point_segment_distance
from .world import Beacon, Dock, Mover, World


# --------------------------------------------------------------- tien ich
def _seg(pts, tag):
    return SegmentSet.from_polyline(pts, closed=False, tag=tag)


def _wall_with_door(x0, y0, x1, y1, door_at, door_w, tag=("wall", None)):
    """Mot buc tuong thang, khoet mot o cua o `door_at` (do tu dau tuong)."""
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return []
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    a = max(0.0, door_at - 0.5 * door_w)
    b = min(L, door_at + 0.5 * door_w)
    out = []
    if a > 0.05:
        out.append(_seg([(x0, y0), (x0 + ux * a, y0 + uy * a)], tag))
    if b < L - 0.05:
        out.append(_seg([(x0 + ux * b, y0 + uy * b), (x1, y1)], tag))
    return out


def _legs_rect(cx, cy, w, h, theta):
    """Bon chan o bon goc mot hinh chu nhat (ban, ghe)."""
    out = []
    c, s = math.cos(theta), math.sin(theta)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            lx, ly = sx * w, sy * h
            out.append((cx + lx * c - ly * s, cy + lx * s + ly * c,
                        P.LEG_RADIUS))
    return out


class _Plan:
    """Cho trong trong nha, de dat do khong chong len nhau."""

    def __init__(self):
        self.taken = []          # (x, y, ban kinh giu cho)

    def free(self, x, y, r):
        for px, py, pr in self.taken:
            if math.hypot(x - px, y - py) < r + pr:
                return False
        return True

    def put(self, x, y, r):
        self.taken.append((x, y, r))


# --------------------------------------------------------------- can nha
def make_house(seed=0, n_docks=3, n_decoys=1, width=P.HOUSE_W,
               height=P.HOUSE_H, n_movers=3, n_beacons=None,
               with_void=True):
    """Sinh mot can nha. Cung hat giong thi ra dung cung can nha."""
    rng = random.Random(seed)
    n_beacons = P.TASK_BEACONS if n_beacons is None else n_beacons

    floor = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    parts = [SegmentSet.from_polyline(
        [(0.02, 0.02), (width - 0.02, 0.02), (width - 0.02, height - 0.02),
         (0.02, height - 0.02)], closed=True, tag=("wall", None))]

    # Hai buc tuong trong, moi buc mot o cua -> bon phong thong nhau.
    ym = rng.uniform(0.42, 0.58) * height          # tuong ngang
    xt = rng.uniform(0.33, 0.48) * width           # tuong doc phia tren
    xb = rng.uniform(0.55, 0.70) * width           # tuong doc phia duoi
    dw = P.DOOR_W
    parts += _wall_with_door(0.02, ym, width - 0.02, ym,
                             rng.uniform(0.18, 0.40) * width, dw)
    parts += _wall_with_door(xt, ym, xt, height - 0.02,
                             rng.uniform(0.3, 0.7) * (height - ym), dw)
    parts += _wall_with_door(xb, 0.02, xb, ym,
                             rng.uniform(0.3, 0.7) * (ym - 0.02), dw)

    rooms = [(0.02, 0.02, xb, ym), (xb, 0.02, width - 0.02, ym),
             (0.02, ym, xt, height - 0.02), (xt, ym, width - 0.02, height - 0.02)]

    plan = _Plan()
    legs = []
    obs_parts = []

    # --- hoc sac: moi cai mot phong khac nhau, lung ap tuong ngoai -------
    back_off = 0.02 + P.DOCK_OUTER_D
    cho = []
    for i, (rx0, ry0, rx1, ry1) in enumerate(rooms):
        # canh nao cua phong nay la tuong NGOAI thi dat duoc; hai cho moi
        # canh de con du cho khi can nhieu hoc hon so phong
        for _ in range(2):
            if ry0 < 0.05:
                cho.append((rng.uniform(rx0 + 0.7, rx1 - 0.7), back_off,
                            math.pi / 2, i))
            if ry1 > height - 0.05:
                cho.append((rng.uniform(rx0 + 0.7, rx1 - 0.7),
                            height - back_off, -math.pi / 2, i))
            if rx0 < 0.05:
                cho.append((back_off, rng.uniform(ry0 + 0.7, ry1 - 0.7),
                            0.0, i))
            if rx1 > width - 0.05:
                cho.append((width - back_off,
                            rng.uniform(ry0 + 0.7, ry1 - 0.7), math.pi, i))
    rng.shuffle(cho)

    docks, dung_phong = [], set()

    def them_hoc(x, y, th):
        docks.append(Dock(x, y, th, code=101 + len(docks),
                          name=f"D{len(docks) + 1}"))
        plan.put(x, y, 1.2)

    # Luot dau: moi phong mot hoc. Luot sau (chi khi can nhieu hoc hon so
    # phong): cho nao con trong thi dat.
    for x, y, th, ph in cho:
        if len(docks) >= n_docks:
            break
        if ph in dung_phong:
            continue
        dung_phong.add(ph)
        them_hoc(x, y, th)
    for x, y, th, ph in cho:
        if len(docks) >= n_docks:
            break
        if plan.free(x, y, 1.2):
            them_hoc(x, y, th)
    for x, y, th, ph in cho:
        if len(docks) >= n_docks + n_decoys:
            break
        if not plan.free(x, y, 1.2):
            continue
        docks.append(Dock(x, y, th, code=None, ir_on=False, powered=False,
                          name=f"moi{len(docks) - n_docks + 1}"))
        plan.put(x, y, 1.2)

    # --- ho (chieu nghi cau thang) ---------------------------------------
    # Dat TRUOC do dac: de sau thi nha day ban ghe roi, khong con cho nao
    # vua, va muoi can nha chi mot hai cai co ho - xe khong hoc duoc cach
    # tranh vuc.
    tuong_da_co = SegmentSet.concat(parts)
    voids = []
    if with_void:
        # Chieu nghi cau thang. Phai CACH XA TUONG: dat de len mot buc tuong
        # thi cai ho an mat o cua, va mot phong thanh khong vao duoc.
        for _ in range(400):
            vx = rng.uniform(1.4, width - 1.4)
            vy = rng.uniform(1.4, height - 1.4)
            d = point_segment_distance(vx, vy, tuong_da_co)
            if d.size and float(d.min()) < 1.35:
                continue
            if plan.free(vx, vy, 1.05):
                voids.append([(vx - 0.45, vy - 0.65), (vx + 0.45, vy - 0.65),
                              (vx + 0.45, vy + 0.65), (vx - 0.45, vy + 0.65)])
                plan.put(vx, vy, 1.3)
                break


    # --- do dac -----------------------------------------------------------
    def cho_trong(r, lan=120):
        for _ in range(lan):
            rx0, ry0, rx1, ry1 = rooms[rng.randrange(len(rooms))]
            if rx1 - rx0 < 2 * r + 0.6 or ry1 - ry0 < 2 * r + 0.6:
                continue
            x = rng.uniform(rx0 + r + 0.5, rx1 - r - 0.5)
            y = rng.uniform(ry0 + r + 0.5, ry1 - r - 0.5)
            if plan.free(x, y, r + 0.45):
                return x, y
        return None

    for _ in range(rng.randint(2, 3)):            # ban: 4 chan
        g = cho_trong(0.7)
        if g:
            th = rng.uniform(0, math.pi)
            legs += _legs_rect(g[0], g[1], 1.25, 0.72, th)
            plan.put(g[0], g[1], 1.0)
    for _ in range(rng.randint(4, 7)):            # ghe: 4 chan nho
        g = cho_trong(0.3)
        if g:
            th = rng.uniform(0, math.pi)
            legs += _legs_rect(g[0], g[1], 0.38, 0.38, th)
            plan.put(g[0], g[1], 0.55)
    for _ in range(rng.randint(1, 3)):            # sofa / tu: khoi dac
        g = cho_trong(0.9)
        if g:
            w = rng.uniform(1.2, 1.9)
            h = rng.uniform(0.5, 0.8)
            obs_parts.append(SegmentSet.from_rect(
                g[0], g[1], w, h, rng.uniform(0, math.pi),
                tag=("obstacle", None)))
            plan.put(g[0], g[1], 0.5 * max(w, h) + 0.4)

    # `World._rebuild` tu gop `obstacles` vao roi, khong cho vao `walls` nua
    # keo moi doan dem hai lan.
    obstacles = SegmentSet.concat(obs_parts) if obs_parts else SegmentSet()
    walls = SegmentSet.concat(parts)

    # --- cham goi: rai deu ra cac phong ----------------------------------
    beacons = []
    for k in range(n_beacons):
        rx0, ry0, rx1, ry1 = rooms[k % len(rooms)]
        for _ in range(60):
            x = rng.uniform(rx0 + 0.6, rx1 - 0.6)
            y = rng.uniform(ry0 + 0.6, ry1 - 0.6)
            if plan.free(x, y, 0.5):
                beacons.append(Beacon(x, y, on=True))
                plan.put(x, y, 0.7)
                break

    # --- nguoi di lai: duong di xuyen qua cua ----------------------------
    movers = []
    for _ in range(n_movers):
        wps = []
        for _ in range(3):
            rx0, ry0, rx1, ry1 = rooms[rng.randrange(len(rooms))]
            wps.append((rng.uniform(rx0 + 0.7, rx1 - 0.7),
                        rng.uniform(ry0 + 0.7, ry1 - 0.7)))
        movers.append(Mover(wps[0][0], wps[0][1], waypoints=wps,
                            speed=rng.uniform(0.3, 0.55),
                            phase=rng.random()))

    return World(floor, walls, obstacles, docks, beacons, movers, voids,
                 legs=legs, name=f"nha-{seed}")
