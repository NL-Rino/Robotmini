# -*- coding: utf-8 -*-
"""Sinh the gioi 3D theo lo.

Moi moi truong co mat bang rieng. Dung MOT LAN luc khoi tao (vong lap Python
o day khong sao vi no chi chay mot lan), roi xep thanh tensor; tu do tro di
moi thu deu la phep tinh tensor.

Ba thu chi co o ban 3D nay, khong the co o ban 2D:

  - CAI BAN co gam. LiDAR quet o do cao 10 cm nen chi thay bon cai chan;
    camera thi thay ca mat ban. Xe chui duoc xuong gam.
  - VAT THAP hon tia LiDAR (cai tham, bac them): LiDAR khong thay gi ca,
    chi co camera thay.
  - BANG MA gan CAO TREN thanh sau cua hoc, cao hon vach hoc. Muon doc phai
    NGUA CAMERA LEN. Day la ly do cai servo ton tai.
"""

import math
import random

import torch

from . import params as P

# Bang mau cua cac o tren bang ma hoc sac.
PALETTE = [
    (0.92, 0.18, 0.16),   # do
    (0.16, 0.62, 0.92),   # xanh duong
    (0.98, 0.80, 0.12),   # vang
    (0.20, 0.78, 0.36),   # xanh la
]

COL_WALL = (0.62, 0.60, 0.57)
COL_OBST = (0.45, 0.38, 0.32)
COL_TABLE = (0.55, 0.40, 0.26)
COL_DOCK = (0.28, 0.30, 0.34)
COL_PLATE = (0.90, 0.90, 0.88)
COL_PERSON = (0.35, 0.30, 0.45)
COL_ROBOT = (0.20, 0.35, 0.50)
COL_FLOOR_A = (0.52, 0.51, 0.49)
COL_FLOOR_B = (0.43, 0.42, 0.41)
COL_CEIL = (0.50, 0.51, 0.53)


class _Build:
    """Gom hinh khoi cua mot the gioi truoc khi xep thanh tensor."""

    def __init__(self):
        self.box = []      # (cx,cy,cz, hx,hy,hz, yaw, r,g,b, mat, c0,c1,c2)
        self.cyl = []      # (cx,cy, r, z0,z1, r,g,b)

    def add_box(self, cx, cy, cz, hx, hy, hz, yaw=0.0, col=COL_WALL,
                mat=0, code=(0, 0, 0)):
        self.box.append((cx, cy, cz, hx, hy, hz, yaw) + tuple(col)
                        + (mat,) + tuple(code))

    def add_cyl(self, cx, cy, r, z0, z1, col=COL_PERSON):
        self.cyl.append((cx, cy, r, z0, z1) + tuple(col))


def _add_dock(b, x, y, theta, code, powered=True):
    """Hoc chu U dung 3D: thanh sau, hai vach ben, hai canh vat, bang ma."""
    w2 = 0.5 * P.DOCK_OUTER_W
    cav2 = 0.5 * P.DOCK_CAVITY_W
    depth = P.DOCK_CAVITY_D
    outer = P.DOCK_OUTER_D
    ch = 0.08
    hz = 0.5 * P.DOCK_HEIGHT
    c, s = math.cos(theta), math.sin(theta)

    def put(lx, ly, hx, hy, extra_yaw=0.0, **kw):
        b.add_box(x + lx * c - ly * s, y + lx * s + ly * c, hz,
                  hx, hy, hz, theta + extra_yaw, **kw)

    # Thanh sau: mat trong o x = -depth, mat ngoai o x = -outer.
    # Viet sai cho nay mot lan roi: hop bi day vao trong long hoc 11 cm, long
    # hoc chi con sau 20 cm thay vi 31 cm, va CAM SAC THANH BAT KHA THI. Bai
    # kiem thu cu dat xe thang vao toa do nen khong chay qua vat ly, va khong
    # bat duoc. Gio co bai lai xe vao that.
    put(-(outer + depth) * 0.5, 0.0, 0.5 * (outer - depth), w2, col=COL_DOCK)
    # hai vach ben
    for sgn in (1, -1):
        put(-0.5 * outer, sgn * 0.5 * (w2 + cav2), 0.5 * outer,
            0.5 * (w2 - cav2), col=COL_DOCK)
    # hai canh vat o mieng: khong vat thi khong dan dong vi sai nao cam noi
    seg = math.hypot(ch, w2 - cav2)
    ang = math.atan2(-(w2 - cav2), -ch)
    for sgn in (1, -1):
        put(-0.5 * ch, sgn * 0.5 * (w2 + cav2), 0.5 * seg, 0.012,
            extra_yaw=sgn * ang, col=COL_DOCK)
    # BANG MA: gan tren thanh sau, cao hon vach hoc -> phai ngua camera len
    if code is not None:
        mz = 0.5 * (P.MARK_Z0 + P.MARK_Z1)
        b.add_box(x + (-depth + 0.01) * c, y + (-depth + 0.01) * s, mz,
                  0.010, 0.125, 0.5 * (P.MARK_Z1 - P.MARK_Z0), theta,
                  col=COL_PLATE, mat=1, code=code)
    else:
        # hoc moi nhu: co bang trang tron, khong ma, khong dien
        mz = 0.5 * (P.MARK_Z0 + P.MARK_Z1)
        b.add_box(x + (-depth + 0.01) * c, y + (-depth + 0.01) * s, mz,
                  0.010, 0.125, 0.5 * (P.MARK_Z1 - P.MARK_Z0), theta,
                  col=COL_PLATE, mat=0)


def build_one(seed, n_docks=3, n_decoys=1, width=6.4, height=4.8,
              n_people=2, n_parked=1):
    """Dung mot the gioi. Tra ve (_Build, thong tin hoc, lo thung, kich thuoc)."""
    rng = random.Random(seed)
    b = _Build()
    t = 0.08                      # do day tuong
    wall_h = 1.20
    for cx, cy, hx, hy in ((width / 2, -t, width / 2 + t, t),
                           (width / 2, height + t, width / 2 + t, t),
                           (-t, height / 2, t, height / 2 + t),
                           (width + t, height / 2, t, height / 2 + t)):
        b.add_box(cx, cy, wall_h / 2, hx, hy, wall_h / 2, col=COL_WALL)

    back = 0.02 + P.DOCK_OUTER_D
    slots = []
    n_bottom = max(2, (n_docks + n_decoys) // 2 + 1)
    n_top = n_docks + n_decoys - n_bottom
    margin = 0.55
    for i in range(n_bottom):
        fx = margin + (width - 2 * margin) * (i + 0.5) / max(1, n_bottom)
        slots.append((fx, back, math.pi / 2))
    for i in range(max(0, n_top)):
        fx = margin + (width - 2 * margin) * (i + 0.5) / max(1, n_top)
        slots.append((fx, height - back, -math.pi / 2))
    rng.shuffle(slots)

    codes = set()
    docks = []
    for i in range(n_docks):
        while True:
            code = tuple(rng.randrange(P.MARK_COLORS) for _ in range(P.MARK_CELLS))
            if code not in codes:
                codes.add(code)
                break
        x, y, th = slots[i]
        _add_dock(b, x, y, th, code)
        docks.append(dict(x=x, y=y, theta=th, code=code, powered=True))
    for j in range(n_decoys):
        if n_docks + j >= len(slots):
            break
        x, y, th = slots[n_docks + j]
        _add_dock(b, x, y, th, None)
        docks.append(dict(x=x, y=y, theta=th, code=None, powered=False))

    def clear_of_docks(cx, cy, r=1.0):
        for d in docks:
            ax = d["x"] + 0.95 * math.cos(d["theta"])
            ay = d["y"] + 0.95 * math.sin(d["theta"])
            if math.hypot(cx - ax, cy - ay) < r:
                return False
            if math.hypot(cx - d["x"], cy - d["y"]) < 1.05:
                return False
        return True

    # CAI BAN: bon chan mong + mat ban o tren. Xe chui duoc xuong gam; LiDAR
    # chi thay bon cai chan, camera thay ca mat ban.
    for _ in range(40):
        tx = rng.uniform(1.4, width - 1.4)
        ty = rng.uniform(1.5, height - 1.5)
        if clear_of_docks(tx, ty, 1.2):
            tw, td, th_ = 0.55, 0.38, 0.42
            for sx in (-1, 1):
                for sy in (-1, 1):
                    b.add_box(tx + sx * (tw - 0.05), ty + sy * (td - 0.05),
                              th_ / 2, 0.025, 0.025, th_ / 2, col=COL_TABLE)
            b.add_box(tx, ty, th_ + 0.02, tw, td, 0.02, col=COL_TABLE)
            break

    placed = []
    for _ in range(3):
        for _try in range(30):
            cx = rng.uniform(1.0, width - 1.0)
            cy = rng.uniform(1.2, height - 1.2)
            hw = rng.uniform(0.18, 0.40)
            hh = rng.uniform(0.18, 0.40)
            # Mot phan ba so vat can THAP hon tia LiDAR: chi camera thay.
            tall = rng.random() > 0.33
            hz = rng.uniform(0.25, 0.45) if tall else rng.uniform(0.03, 0.055)
            if not clear_of_docks(cx, cy, 1.0):
                continue
            if any(math.hypot(cx - px, cy - py) < 1.0 for px, py in placed):
                continue
            placed.append((cx, cy))
            b.add_box(cx, cy, hz, hw, hh, hz, rng.uniform(0, math.pi),
                      col=COL_OBST)
            break

    people = []
    for _ in range(n_people):
        px = rng.uniform(1.0, width - 1.0)
        py = rng.uniform(1.2, height - 1.2)
        qx = rng.uniform(1.0, width - 1.0)
        qy = rng.uniform(1.2, height - 1.2)
        b.add_cyl(px, py, 0.18, 0.0, 1.65, COL_PERSON)
        people.append((px, py, qx, qy, rng.uniform(0.28, 0.5)))
    for _ in range(n_parked):
        # xe khac dang do trong mot cai hoc -> hoc do ban
        d = docks[rng.randrange(len(docks))]
        dep = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.01
        b.add_cyl(d["x"] - dep * math.cos(d["theta"]),
                  d["y"] - dep * math.sin(d["theta"]),
                  P.BODY_RADIUS, 0.0, P.BODY_HEIGHT, COL_ROBOT)
        d["blocked"] = True

    voids = []
    vx, vy = width - 0.75, 0.5 * height
    voids.append((vx - 0.35, vy - 0.55, vx + 0.35, vy + 0.55))

    # Cac cho dung trong, tinh san mot lan. Luc huan luyen phai dat lai xe
    # hang chuc nghin lan moi phut; boc san o day thi luc do chi con la mot
    # phep tra bang.
    spots = []
    solids = [(bx[0], bx[1], bx[3], bx[4], bx[6]) for bx in b.box
              if bx[2] - bx[5] < P.BODY_HEIGHT and bx[2] + bx[5] > 0.0
              and abs(bx[3]) + abs(bx[4]) > 1e-6]
    guard = P.BODY_RADIUS + 0.08
    tries = 0
    while len(spots) < 64 and tries < 4000:
        tries += 1
        px = rng.uniform(0.45, width - 0.45)
        py = rng.uniform(0.45, height - 0.45)
        if any(vx0 - guard < px < vx1 + guard and vy0 - guard < py < vy1 + guard
               for vx0, vy0, vx1, vy1 in voids):
            continue
        ok = True
        for cx, cy, hx, hy, yaw in solids:
            ca, sa = math.cos(-yaw), math.sin(-yaw)
            dx, dy = px - cx, py - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            qx = max(-hx, min(hx, lx))
            qy = max(-hy, min(hy, ly))
            if math.hypot(lx - qx, ly - qy) < guard:
                ok = False
                break
        if ok:
            spots.append((px, py))
    while len(spots) < 64:
        spots.append((width / 2, height / 2))

    return b, docks, voids, (width, height), people, spots


def make_batch(n_worlds, device, seed0=0, n_docks=3, n_decoys=1, **kw):
    """Dung n_worlds the gioi, xep thanh tensor de doi tia theo lo."""
    builds, metas = [], []
    for i in range(n_worlds):
        b, docks, voids, size, people, spots = build_one(
            seed0 + i, n_docks=n_docks, n_decoys=n_decoys, **kw)
        builds.append(b)
        metas.append(dict(docks=docks, voids=voids, size=size, people=people,
                          spots=spots, n_people=len(people)))

    nb = max(len(b.box) for b in builds)
    nc = max(len(b.cyl) for b in builds)
    nv = max(len(m["voids"]) for m in metas)
    f = torch.float32

    box = torch.zeros(n_worlds, nb, 14, dtype=f)
    for i, b in enumerate(builds):
        if b.box:
            box[i, :len(b.box)] = torch.tensor(b.box, dtype=f)
    cyl = torch.zeros(n_worlds, max(1, nc), 8, dtype=f)
    for i, b in enumerate(builds):
        if b.cyl:
            cyl[i, :len(b.cyl)] = torch.tensor(b.cyl, dtype=f)

    void = torch.zeros(n_worlds, nv, 4, dtype=f)
    floor = torch.zeros(n_worlds, 4, dtype=f)
    for i, m in enumerate(metas):
        for j, v in enumerate(m["voids"]):
            void[i, j] = torch.tensor(v, dtype=f)
        floor[i] = torch.tensor([0.0, 0.0, m["size"][0], m["size"][1]], dtype=f)

    # Hop nao chan duong xe: chi nhung hop co phan nam trong khoang cao cua
    # than xe. Mat ban o 0,42 m hay bang ma o 0,34 m thi xe chui qua duoc.
    z0 = box[..., 2] - box[..., 5]
    z1 = box[..., 2] + box[..., 5]
    solid = (z0 < P.BODY_HEIGHT) & (z1 > 0.0) & (box[..., 3:6].abs().sum(-1) > 1e-6)

    nd = max(len(m["docks"]) for m in metas)
    dk = torch.zeros(n_worlds, nd, 3, dtype=f)
    dcode = torch.zeros(n_worlds, nd, P.MARK_CELLS, dtype=torch.long)
    dpow = torch.zeros(n_worlds, nd, dtype=torch.bool)
    dblk = torch.zeros(n_worlds, nd, dtype=torch.bool)
    for i, m in enumerate(metas):
        for j, d in enumerate(m["docks"]):
            dk[i, j] = torch.tensor([d["x"], d["y"], d["theta"]], dtype=f)
            if d["code"] is not None:
                dcode[i, j] = torch.tensor(d["code"], dtype=torch.long)
                dpow[i, j] = True
            dblk[i, j] = bool(d.get("blocked", False))

    sc = dict(
        box_solid=solid.to(device),
        dock_pose=dk.to(device), dock_code=dcode.to(device),
        dock_powered=dpow.to(device), dock_blocked=dblk.to(device),
        box_c=box[..., 0:3].to(device), box_h=box[..., 3:6].to(device),
        box_yaw=box[..., 6].to(device), box_col=box[..., 7:10].to(device),
        box_mat=box[..., 10].long().to(device),
        box_code=box[..., 11:14].long().to(device),
        cyl_c=cyl[..., 0:2].to(device), cyl_r=cyl[..., 2].to(device),
        cyl_z=cyl[..., 3:5].to(device), cyl_col=cyl[..., 5:8].to(device),
        void=void.to(device), floor=floor.to(device),
        floor_col=torch.tensor([[COL_FLOOR_A, COL_FLOOR_B]], dtype=f
                               ).repeat(n_worlds, 1, 1).to(device),
        spots=torch.tensor([m["spots"] for m in metas], dtype=f).to(device),
        mover_n=torch.tensor([m["n_people"] for m in metas],
                             dtype=torch.long).to(device),
        mover_path=torch.zeros(n_worlds, max(1, nc), 5, dtype=f).to(device),
        ceil_z=torch.full((n_worlds,), 2.45, dtype=f).to(device),
        ceil_col=torch.tensor([COL_CEIL], dtype=f).repeat(n_worlds, 1).to(device),
        palette=torch.tensor(PALETTE, dtype=f).to(device),
    )
    path = torch.zeros(n_worlds, max(1, nc), 5, dtype=f)
    for i, m in enumerate(metas):
        for j, (px, py, qx, qy, sp) in enumerate(m["people"]):
            path[i, j] = torch.tensor([px, py, qx, qy, sp], dtype=f)
    sc["mover_path"] = path.to(device)
    return sc, metas
