# -*- coding: utf-8 -*-
"""Doi mat bang cua ban v1 sang tensor de chay theo lo.

KHONG viet lai bo sinh mat bang. Goi thang `sim.world.make_fleet_map` cua
ban v1 roi doi sang tensor. Nho vay hai ban chay tren DUNG cung mot can
phong, va moi so do cua ban cu van con nghia.

Mot xe = mot hang trong moi tensor. Nhieu xe co the dung chung mat bang;
luc dung tensor thi trai phang ra het, vi 144 hang x 64 doan thang chi het
147 KB - khong dang de tiet kiem.
"""


import torch

from sim import params as P
from sim.world import make_fleet_map


def _pad(rows, n, width, dtype=torch.float32):
    out = torch.zeros(len(rows), n, width, dtype=dtype)
    for i, r in enumerate(rows):
        if len(r):
            out[i, :len(r)] = torch.tensor(r, dtype=dtype)
    return out


class BatchWorld:
    """Mot lo the gioi. Moi chi so la mot XE (nhieu xe co the chung mat bang)."""

    def __init__(self, worlds, world_of_robot, home_of_robot, device):
        self.device = device
        self.worlds = worlds
        self.world_of = list(world_of_robot)
        self.R = len(self.world_of)

        segs, docks, voids, floors, movers, beacons = [], [], [], [], [], []
        for w in worlds:
            s = w.static_segments
            segs.append([(float(s.ax[i]), float(s.ay[i]), float(s.bx[i]),
                          float(s.by[i])) for i in range(len(s))])
            docks.append([(d.x, d.y, d.theta,
                           -1.0 if d.code is None else float(d.code),
                           1.0 if d.powered else 0.0) for d in w.docks])
            voids.append([(v[0][0], v[0][1], v[2][0], v[2][1]) for v in w.voids])
            x0, y0, x1, y1 = w.bounds
            floors.append([x0, y0, x1, y1])
            movers.append([(m.x, m.y, m.radius) for m in w.movers])
            beacons.append([(b.x, b.y, 1.0 if b.on else 0.0) for b in w.beacons])

        ns = max(len(s) for s in segs)
        nd = max(len(d) for d in docks)
        nv = max(1, max(len(v) for v in voids))
        nm = max(1, max(len(m) for m in movers))
        nb = max(1, max(len(b) for b in beacons))

        idx = torch.tensor(self.world_of, dtype=torch.long)
        self._widx = idx.to(device)
        self.seg = _pad(segs, ns, 4)[idx].to(device)
        self.dock = _pad(docks, nd, 5)[idx].to(device)
        self.void = _pad(voids, nv, 4)[idx].to(device)
        self.floor = torch.tensor(floors, dtype=torch.float32)[idx].to(device)
        self.mover = _pad(movers, nm, 3)[idx].to(device)
        self.beacon = _pad(beacons, nb, 3)[idx].to(device)
        self.n_dock = nd

        # Doan rong (dai 0) bi phep giao tia loai ngay vi mau so bang 0, nen
        # khong can mat na rieng.
        self.seg_e = self.seg[..., 2:4] - self.seg[..., 0:2]

        self.home = torch.tensor(list(home_of_robot), dtype=torch.long,
                                 device=device)
        self.dock_code = self.dock[..., 3]
        self.dock_powered = self.dock[..., 4] > 0.5
        self.my_code = self.dock_code.gather(1, self.home[:, None]).squeeze(1)

        # Quy dao nguoi di lai: giu nguyen doi tuong CPU de buoc di, roi moi
        # buoc chep vi tri sang tensor. Nguoi chi co vai nguoi moi the gioi
        # nen phan nay khong phai nut that.
        self._mover_src = [w.movers for w in worlds]
        self._beacon_src = [w.beacons for w in worlds]

    def freeze_movers(self):
        """Bo het nguoi di lai.

        Chi zero cai tensor thoi thi khong an thua: cuoi moi buoc
        `step_dynamics` chep lai vi tri tu cac doi tuong CPU, nen nguoi song
        lai ngay o buoc sau. Muon do vat ly hay do bo do hoc canh ban v1 thi
        phai cat ca nguon.
        """
        self._mover_src = [[] for _ in self.worlds]
        self.mover.zero_()
        self._frozen = True

    def step_dynamics(self, dt, rng):
        """Buoc nguoi di lai va den goi.

        Dung mot vong lap tren SO MAT BANG (2-6 cai) roi trai ra cho tung
        xe bang mot phep chi so. Truoc day o day co mot vong lap tren SO XE:
        voi 1.536 xe thi moi buoc mo phong phai chay 3.000 vong Python va
        tao 1.536 tensor ti hon - ton hon ca phep ban tia, va ton cang nhieu
        khi quan the cang lon, dung cai ma ca ban nay sinh ra de tranh.
        """
        for w in self.worlds:
            w.step_dynamics(dt, rng)
        pos, bea = [], []
        for ms, bs in zip(self._mover_src, self._beacon_src):
            pos.append([(m.x, m.y, m.radius) for m in ms])
            bea.append([(b.x, b.y, 1.0 if b.on else 0.0) for b in bs])
        self.mover = _pad(pos, self.mover.shape[1], 3).to(self.device)[self._widx]
        self.beacon = _pad(bea, self.beacon.shape[1], 3).to(self.device)[self._widx]

    def home_pose(self):
        g = self.home[:, None, None].expand(-1, 1, 5)
        return self.dock.gather(1, g).squeeze(1)[:, :3]


def build(map_seeds, robots_per_map, device, n_docks=3, n_decoys=1, copies=1):
    """Dung mot lo the gioi tu danh sach hat giong mat bang.

    Moi hat giong sinh ra DUNG cai mat bang ma ban v1 sinh ra voi hat giong
    do - de hai ban so sanh duoc voi nhau.

    `copies` nhan cho ES: mot the he cham diem P bo trong so, va CHUNG SO
    NGAU NHIEN doi hoi ca P bo phai chay tren DUNG cung mat bang, cung cho
    dat xe, cung nguoi di lai. Nen ta lap DANH SACH XE len P lan ma van dung
    CHUNG cac doi tuong mat bang - nguoi di lai buoc mot lan, ca P ban sao
    deu thay y het. Hang thu (p*U + u) la ban sao p cua don vi u.
    """
    worlds, world_of, home_of = [], [], []
    for wi, seed in enumerate(map_seeds):
        w = make_fleet_map(seed, n_docks=n_docks, n_decoys=n_decoys)
        worlds.append(w)
        coded = [i for i, d in enumerate(w.docks) if d.code is not None]
        for k in range(robots_per_map):
            world_of.append(wi)
            home_of.append(coded[k % len(coded)])
    return BatchWorld(worlds, world_of * copies, home_of * copies, device)


def dock_local(px, py, pose):
    """Doi diem sang he quy chieu hoc: +x la truc ra, goc o giua mieng."""
    dx = px[:, None] - pose[..., 0]
    dy = py[:, None] - pose[..., 1]
    ca = torch.cos(-pose[..., 2])
    sa = torch.sin(-pose[..., 2])
    return dx * ca - dy * sa, dx * sa + dy * ca


def approach_point(pose, dist=0.55):
    return (pose[:, 0] + dist * torch.cos(pose[:, 2]),
            pose[:, 1] + dist * torch.sin(pose[:, 2]))


CAVITY_HALF = 0.5 * P.DOCK_CAVITY_W
DEPTH = P.DOCK_CAVITY_D
SPAWN_DEPTH = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005


def spawn_in_dock(pose):
    """Tu the xe khi nam san trong hoc: duoi o phia trong, mui huong ra."""
    return (pose[:, 0] - SPAWN_DEPTH * torch.cos(pose[:, 2]),
            pose[:, 1] - SPAWN_DEPTH * torch.sin(pose[:, 2]),
            pose[:, 2].clone())


def free_spots(worlds, n=48):
    """Cac cho dung trong, tinh mot lan tren CPU."""
    import random
    from sim.geometry import point_segment_distance
    out = []
    for w in worlds:
        rng = random.Random(hash(w.name) & 0xFFFF)
        x0, y0, x1, y1 = w.bounds
        spots = []
        tries = 0
        while len(spots) < n and tries < 3000:
            tries += 1
            px = rng.uniform(x0 + 0.45, x1 - 0.45)
            py = rng.uniform(y0 + 0.45, y1 - 0.45)
            if not w.on_floor(px, py):
                continue
            d = point_segment_distance(px, py, w.static_segments)
            if d.size and float(d.min()) < P.BODY_RADIUS + 0.10:
                continue
            spots.append((px, py))
        while len(spots) < n:
            spots.append((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        out.append(spots)
    return out
