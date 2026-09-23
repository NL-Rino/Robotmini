"""Mot lan danh gia mot bo nao.

Diem mau chot cua ca cach huan luyen nay nam o day: GIAO TRINH NGUOC.

Mot chu ky sac day du dai hang phut mo phong. Neu moi lan danh gia deu bat
dau tu luc xe day pin trong hoc thi phai chay het vai nghin buoc moi toi
duoc cho co phan thuong, va trong hang tram the he dau tien thi khong mot ca
the nao cham duoc vao no - ES khong co gi de so sanh, chi xao trong so ngau
nhien. Ban cu chay 5.900 the he ma van thua bo luat viet tay chinh vi vay.

Cach chua la dat xe vao mot diem BAT KY tren chu ky, khong phai luon luon o
dau. Nam pha, tu de nhat den kho nhat:

  sap-cam     lui dang do vao mot cai hoc, pin can    -> con 20 cm nua
  truoc-mieng dung truoc mieng mot cai hoc, pin can   -> con quay va lui
  pin-yeu     dung bat ky dau, pin duoi nguong        -> con tim duong ve
  long-nhong  dung bat ky dau, pin con nhieu          -> chi can khong dam
  trong-hoc   nam trong hoc cua minh, pin day         -> hoc cach ra di

Nhung the he dau danh phan lon suat cho hai pha cuoi bang, noi phan thuong
chi cach vai chuc buoc. Kha len toi dau thi day dan suat ve phia dau bang.

Va: cai hoc dung o pha "truoc-mieng"/"sap-cam" duoc chon NGAU NHIEN, nhieu
khi khong phai hoc cua xe. Neu lan nao cung dat dung hoc cua no thi bo nao
se hoc duoc rang cu cam la co dien, va se khong bao gio nhin toi tin hieu
bat tay.

DE BAI DAI, LAN DANH GIA NGAN. Lam xong de bai (5 cham goi + 3 lan sac hop
le) mat vai phut mo phong, mot lan danh gia chi mot phut. Nen giao trinh con
cap san TIEN DO: xe bat dau voi "da an k cham, da sac m lan" ngau nhien, va
kha nang cao la chi con thieu mot hai viec. Nho vay phan thuong HOAN THANH
cham duoc ngay tu nhung the he dau, va bo nao hoc duoc cac dau vao tien do
(task_beacons, task_charges) co nghia gi.

Pha moi "dang-sac": nam trong hoc cua minh, DANG trong mot lan sac hop le,
pin moi len duoc nua chung. Day la cho hoc "dung co chay ra giua chung".
"""

import math
import random

import numpy as np

from sim import params as P
from sim.fleet import FleetSim
from sim.world import make_fleet_map

from .policy import PolicyBrain
from .reward import RewardTracker

PHASES = ("sap-cam", "truoc-mieng", "pin-yeu", "long-nhong", "trong-hoc",
          "dang-sac")

# Suat cua tung pha o dau va o cuoi qua trinh huan luyen.
MIX_EARLY = (0.26, 0.26, 0.14, 0.14, 0.06, 0.14)
MIX_LATE = (0.06, 0.14, 0.26, 0.32, 0.12, 0.10)


def _task_progress(rng, progress):
    """Tien do de bai cap san cho mot xe: (so cham da an, so lan da sac).

    Dau huan luyen: phan lon chi con thieu mot viec. Cuoi huan luyen: bat
    dau tu so 0 nhieu hon, de bo nao hoc ca chang duong dai.
    """
    p = min(1.0, max(0.0, float(progress)))
    if rng.random() < 0.55 - 0.30 * p:
        # gan xong: thieu mot cham HOAC mot lan sac
        if rng.random() < 0.5:
            return P.TASK_BEACONS - 1, P.TASK_CHARGES
        return P.TASK_BEACONS, P.TASK_CHARGES - 1
    return (rng.randint(0, P.TASK_BEACONS), rng.randint(0, P.TASK_CHARGES))


def phase_mix(progress):
    """Tron hai bo suat theo tien do 0..1."""
    p = min(1.0, max(0.0, float(progress)))
    return tuple(a + (b - a) * p for a, b in zip(MIX_EARLY, MIX_LATE))


def _pick_phase(rng, mix):
    x = rng.random() * sum(mix)
    acc = 0.0
    for name, w in zip(PHASES, mix):
        acc += w
        if x <= acc:
            return name
    return PHASES[-1]


class RolloutResult:
    __slots__ = ("score", "charged", "wrong", "falls", "flats", "beacons",
                 "charges_ok", "cells", "complete", "clawed",
                 "obs_sum", "obs_sqsum", "obs_n", "steps")

    def __init__(self):
        self.score = 0.0
        self.charged = 0.0
        self.wrong = 0
        self.falls = 0
        self.flats = 0
        self.beacons = 0
        self.charges_ok = 0       # lan sac HOP LE (duoi 20% -> day 100%)
        self.cells = 0            # o san nha moi nhin thay
        self.complete = 0         # so xe lam xong de bai
        self.clawed = 0.0         # tien tam ung bi thu lai (rut ra giua chung)
        self.obs_sum = None
        self.obs_sqsum = None
        self.obs_n = 0
        self.steps = 0


def _place(sim, rng, mix, station_drift, progress=0.0):
    """Dat tung xe vao mot pha, moi xe mot cai hoc khac nhau."""
    docks = [d for d in sim.world.docks]
    rng.shuffle(docks)
    # Mot cai hoc chi cho MOT xe. Truoc day xe nay duoc dat vao hoc cua no
    # con xe kia duoc dat ngau nhien vao dung cai hoc do: hai than xe chong
    # len nhau, bo giai va cham day nhau ra, va ca hai bat dau lan danh gia
    # bang mot cu va vao vach.
    taken = set()
    for rid, r in enumerate(sim.robots):
        phase = _pick_phase(rng, mix)
        drift = rng.uniform(0.0, station_drift)
        r.task_beacons, r.task_charges = _task_progress(rng, progress)

        if phase in ("trong-hoc", "dang-sac"):
            d = sim.home[rid]
            if id(d) in taken:
                phase = "long-nhong"
            else:
                taken.add(id(d))
        if phase in ("trong-hoc", "dang-sac"):
            d = sim.home[rid]
            depth = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005
            if phase == "dang-sac":
                # Mot lan sac hop le dang do: pin moi len duoc mot phan.
                # Con thieu it nhat mot lan sac thi moi co y nghia.
                r.task_charges = min(r.task_charges, P.TASK_CHARGES - 1)
                batt = rng.uniform(0.10, 0.85)
            else:
                batt = rng.uniform(0.75, 1.0)
            sim.place_robot(rid, d.x - depth * math.cos(d.theta),
                            d.y - depth * math.sin(d.theta), d.theta,
                            battery=batt, station_drift=0.0)
            if phase == "dang-sac" and r.charging:
                r.charge_valid = True
            continue

        if phase in ("truoc-mieng", "sap-cam"):
            # Hoc duoc chon ngau nhien: khoang mot nua so lan la hoc CUA XE,
            # so con lai la hoc cua xe khac hoac hoc moi nhu. Neu lan nao
            # cung dat dung hoc cua no thi bo nao se hoc duoc rang "cu cam
            # la co dien" va se khong bao gio nhin toi tin hieu bat tay.
            free = [x for x in docks if id(x) not in taken]
            if not free:
                x, y, th = sim.free_pose(rng)
                sim.place_robot(rid, x, y, th, battery=rng.uniform(0.05, 0.19),
                                station_drift=drift, rng=rng)
                continue
            own = sim.home[rid]
            if rng.random() < 0.55 and id(own) not in taken:
                d = own
            else:
                d = free[0]
            taken.add(id(d))
            # Trong moi pha van co dai kho de: co lan gan nhu xong roi, co
            # lan lech nhieu. Mot pha chi co mot muc kho thi hoac de qua
            # (khong hoc them duoc gi) hoac kho qua (khong ai cham toi).
            hard = rng.random()
            if phase == "truoc-mieng":
                dist = 0.35 + 0.60 * hard
                x = d.x + dist * math.cos(d.theta)
                y = d.y + dist * math.sin(d.theta)
                # Huong mui: de thi da quay san duoi vao hoc, kho thi quay
                # lung tung.
                th = d.theta + rng.gauss(0.0, 0.15 + 1.6 * hard)
                batt = rng.uniform(0.05, 0.19)
            else:
                depth = 0.16 - 0.14 * hard
                x = d.x - depth * math.cos(d.theta)
                y = d.y - depth * math.sin(d.theta)
                th = d.theta + rng.gauss(0.0, 0.03 + 0.25 * hard)
                batt = rng.uniform(0.04, 0.19)
            sim.place_robot(rid, x, y, th, battery=batt,
                            station_drift=drift, rng=rng)
            continue

        x, y, th = sim.free_pose(rng)
        # long-nhong: pin tu 22% tro len. Phan lon se tut duoi 20% ngay
        # trong lan danh gia - do la luc phai tu biet quay ve.
        batt = (rng.uniform(0.05, P.BATT_LOW - 0.005) if phase == "pin-yeu"
                else rng.uniform(0.22, 0.95))
        sim.place_robot(rid, x, y, th, battery=batt, station_drift=drift,
                        rng=rng)


def rollout(policy, seed, steps=1200, n_robots=3, progress=0.0,
            station_drift=2.5, collect_obs=True, n_decoys=1):
    """Chay mot lan va cham diem. Cung `seed` thi cung mat bang, cung cho
    dat xe, cung nhieu cam bien - do la dieu kien de so sanh hai bo nao."""
    world = make_fleet_map(seed, n_docks=n_robots, n_decoys=n_decoys)
    sim = FleetSim(world, n_robots=n_robots, seed=seed)
    rng = random.Random(seed * 7919 + 13)
    _place(sim, rng, phase_mix(progress), station_drift, progress)

    brains = [PolicyBrain(policy, r.id) for r in sim.robots]
    track = [RewardTracker(r.id, sim.home[r.id], r) for r in sim.robots]

    res = RolloutResult()
    if collect_obs:
        res.obs_sum = np.zeros(policy.n_in, dtype=np.float64)
        res.obs_sqsum = np.zeros(policy.n_in, dtype=np.float64)

    for _ in range(steps):
        obs = sim.observe()
        cmds = {}
        was = {}
        for r in sim.robots:
            o = obs[r.id]
            if collect_obs:
                res.obs_sum += o
                res.obs_sqsum += np.square(o, dtype=np.float64)
                res.obs_n += 1
            was[r.id] = r.charging
            cmds[r.id] = brains[r.id](o, sim.t)
        sim.step(cmds)
        for i, r in enumerate(sim.robots):
            track[i].latch_charge(r, was[r.id])
            track[i].step(world, r, sim.dt)
    res.steps = steps

    n = float(len(track))
    res.score = sum(t.total for t in track) / n
    res.charged = sum(t.charged for t in track)
    res.wrong = sum(t.wrong for t in track)
    res.falls = sum(t.fell for t in track)
    res.flats = sum(t.flat for t in track)
    res.beacons = sum(t.beacons for t in track)
    res.charges_ok = sum(t.charges_ok for t in track)
    res.cells = sum(t.cells for t in track)
    res.complete = sum(1 for t, r in zip(track, sim.robots)
                       if t.complete and r.task_beacons >= P.TASK_BEACONS
                       and r.task_charges >= P.TASK_CHARGES)
    res.clawed = sum(t.clawed for t in track)
    return res


def evaluate(policy, seeds, steps=1200, n_robots=3, progress=1.0, **kw):
    """Chay nhieu lan, tra ve diem trung binh va thong ke gop."""
    out = RolloutResult()
    tot = 0.0
    for s in seeds:
        r = rollout(policy, s, steps=steps, n_robots=n_robots,
                    progress=progress, collect_obs=False, **kw)
        tot += r.score
        out.charged += r.charged
        out.wrong += r.wrong
        out.falls += r.falls
        out.flats += r.flats
        out.beacons += r.beacons
        out.charges_ok += r.charges_ok
        out.cells += r.cells
        out.complete += r.complete
        out.clawed += r.clawed
    out.score = tot / max(1, len(seeds))
    out.steps = steps * len(seeds)
    return out
