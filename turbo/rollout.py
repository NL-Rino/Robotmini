# -*- coding: utf-8 -*-
"""Cham diem CA QUAN THE trong mot lan chay.

Cung giao trinh nguoc cua ban v1 (`train/rollout.py`): nam pha, suat cua
tung pha day dan ve phia kho khi bo nao kha len. Cho dat xe tinh tren CPU
bang DUNG doan ma cua ban v1 - mot the he chi dat vai chuc con, khong dang
de viet lai theo lo va lam lech giao trinh.

Cho khac: mot lan chay o day cham diem P bo trong so cung mot luc, va ca P
bo chay tren DUNG cung mat bang, cung cho dat xe, cung nhieu cam bien. Do
la chung so ngau nhien - thu re nhat va an thua nhat trong ca danh sach.

Bo cuc hang: hang (p*U + u) la ban sao p cua don vi u. Nen doi ve (P, U) la
mot phep `view`, khong phai phep chep.
"""

import math
import random

import torch

from sim import params as P
from sim.geometry import point_segment_distance
from train.rollout import (MIX_EARLY, MIX_LATE, PHASES, _task_progress,
                           phase_mix)

from . import dock as D, perception as PC, world as TW
from .reward import BatchReward
from .sim import BatchSim

__all__ = ["MIX_EARLY", "MIX_LATE", "PHASES", "phase_mix", "Rollout"]


def _pick_phase(rng, mix):
    x = rng.random() * sum(mix)
    acc = 0.0
    for name, w in zip(PHASES, mix):
        acc += w
        if x <= acc:
            return name
    return PHASES[-1]


def _free_pose(w, rng, taken_xy, tries=60):
    x0, y0, x1, y1 = w.bounds
    for _ in range(tries):
        x = rng.uniform(x0 + 0.4, x1 - 0.4)
        y = rng.uniform(y0 + 0.4, y1 - 0.4)
        if not w.on_floor(x, y):
            continue
        d = point_segment_distance(x, y, w.static_segments)
        if d.size and float(d.min()) < P.BODY_RADIUS + 0.10:
            continue
        if any(math.hypot(x - cx, y - cy) < P.BODY_RADIUS + cr + 0.08
               for cx, cy, cr in w.solid_circles()):
            continue
        if any(math.hypot(x - ax, y - ay) < 2 * P.BODY_RADIUS + 0.1
               for ax, ay in taken_xy):
            continue
        return x, y, rng.uniform(-math.pi, math.pi)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), rng.uniform(-math.pi, math.pi)


def start_states(bw, unit, rng, mix, station_drift, progress=0.0):
    """Cho dat cho `unit` con xe dau (mot ban sao). Tra ve cac danh sach.

    Kem theo tien do de bai cap san cho tung xe: (so cham, so lan sac, dang
    o giua mot lan sac hop le hay khong) - xem `train/rollout.py`.
    """
    xs, ys, ths, batts, sts = [], [], [], [], []
    tasks = []
    by_world = {}
    for u in range(unit):
        wi = bw.world_of[u]
        by_world.setdefault(wi, []).append(u)

    out = [None] * unit
    for wi, rows in by_world.items():
        w = bw.worlds[wi]
        docks = list(w.docks)
        rng.shuffle(docks)
        taken = set()
        placed = []
        for u in rows:
            home = w.docks[int(bw.home[u])]
            phase = _pick_phase(rng, mix)
            drift = rng.uniform(0.0, station_drift)
            tb, tc = _task_progress(rng, progress)
            valid = False

            if phase in ("trong-hoc", "dang-sac") and id(home) in taken:
                phase = "long-nhong"
            if phase in ("trong-hoc", "dang-sac"):
                taken.add(id(home))
                depth = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005
                x = home.x - depth * math.cos(home.theta)
                y = home.y - depth * math.sin(home.theta)
                th = home.theta
                if phase == "dang-sac":
                    tc = min(tc, P.TASK_CHARGES - 1)
                    batt = rng.uniform(0.10, 0.85)
                    valid = True
                else:
                    batt = rng.uniform(0.75, 1.0)
                drift = 0.0
            elif phase in ("truoc-mieng", "sap-cam"):
                free = [d for d in docks if id(d) not in taken]
                if not free:
                    x, y, th = _free_pose(w, rng, placed)
                    batt = rng.uniform(0.05, 0.19)
                else:
                    # Hoc chon NGAU NHIEN: khoang mot nua so lan khong phai
                    # hoc cua xe. Lan nao cung dat dung hoc cua no thi bo nao
                    # se hoc rang cu cam la co dien.
                    d = home if (rng.random() < 0.55 and id(home) not in taken) \
                        else free[0]
                    taken.add(id(d))
                    hard = rng.random()
                    if phase == "truoc-mieng":
                        dist = 0.35 + 0.60 * hard
                        x = d.x + dist * math.cos(d.theta)
                        y = d.y + dist * math.sin(d.theta)
                        th = d.theta + rng.gauss(0.0, 0.15 + 1.6 * hard)
                        batt = rng.uniform(0.05, 0.19)
                    else:
                        depth = 0.16 - 0.14 * hard
                        x = d.x - depth * math.cos(d.theta)
                        y = d.y - depth * math.sin(d.theta)
                        th = d.theta + rng.gauss(0.0, 0.03 + 0.25 * hard)
                        batt = rng.uniform(0.04, 0.19)
            else:
                x, y, th = _free_pose(w, rng, placed)
                batt = (rng.uniform(0.05, P.BATT_LOW - 0.005)
                        if phase == "pin-yeu" else rng.uniform(0.22, 0.95))

            placed.append((x, y))
            sx, sy, sth = home.x, home.y, home.theta
            if drift > 0.0:
                ang = rng.uniform(-math.pi, math.pi)
                sx += drift * math.cos(ang)
                sy += drift * math.sin(ang)
                sth += rng.gauss(0.0, 0.12 * drift)
            out[u] = (x, y, th, batt, sx, sy, sth, (tb, tc, valid))

    for rec in out:
        x, y, th, batt, sx, sy, sth, task = rec
        xs.append(x)
        ys.append(y)
        ths.append(th)
        batts.append(batt)
        sts.append((sx, sy, sth))
        tasks.append(task)
    return xs, ys, ths, batts, sts, tasks


class Rollout:
    """Mot lo the gioi dung san, dung lai cho nhieu the he.

    Dung lai duoc la co y: tao mat bang moi moi the he ton vai tram mili
    giay CPU va khong dinh gi toi GPU. O day mat bang giu nguyen, chi dat
    lai xe - `reset` het chua toi mot phan muoi thoi gian do.
    """

    def __init__(self, map_seeds, robots_per_map, n_pop, device,
                 n_decoys=1, seed=0):
        self.device = device
        self.n_pop = int(n_pop)
        self.bw = TW.build(list(map_seeds), robots_per_map, device,
                           n_docks=robots_per_map, n_decoys=n_decoys,
                           copies=self.n_pop)
        self.unit = len(map_seeds) * robots_per_map
        self.sim = BatchSim(self.bw, device, seed=seed, copies=self.n_pop)
        self.rw = None

    # ------------------------------------------------------------------
    def reset(self, seed, progress, station_drift=2.5):
        """Dat lai ca lo. Cung `seed` thi cung cho dat xe - do la CRN."""
        rng = random.Random(seed * 7919 + 13)
        xs, ys, ths, batts, sts, tasks = start_states(
            self.bw, self.unit, rng, phase_mix(progress), station_drift,
            progress)
        rep = self.n_pop
        # Tao tren CPU roi moi chuyen sang may: tao THANG tren card lien
        # (DirectML) tu mot danh sach Python la thu no hay khong lam duoc.
        t = lambda v: torch.tensor(v * rep, dtype=torch.float32).to(self.device)
        st = torch.tensor(sts, dtype=torch.float32).repeat(rep, 1).to(self.device)
        self.sim.t = 0.0
        # Khong dung `zero_()`: phep tai cho tren kieu bool la thu card
        # lien hay tu choi, va no tu choi bang mot cau 'unknown error'.
        self.sim.reset_beacons()
        self.sim.seen = torch.zeros_like(self.sim.seen)
        self.sim._cursor = 0.0
        self.sim._rev = 0
        self.sim.place(torch.arange(self.sim.R).to(self.device),
                       t(xs), t(ys), t(ths), battery=t(batts), station=st)
        # Trang thai tiep dien phai dung NGAY tu buoc dau: xe dat san trong
        # hoc cua no la "dang sac", va ham phan thuong doc cai do de biet co
        # bat nguoi ta di lam mot vong khong.
        ins, idok, _ = self.sim.contact()
        self.sim.in_slot, self.sim.id_ok, self.sim.charging = ins, idok, idok
        self.sim.task_beacons = t([float(a[0]) for a in tasks])
        self.sim.task_charges = t([float(a[1]) for a in tasks])
        # "dang-sac": dang o giua mot lan sac hop le - chi khi that su co dien.
        self.sim.charge_valid = (t([1.0 if a[2] else 0.0 for a in tasks]) > 0.5) \
            & idok
        self.rw = BatchReward(self.sim)
        self._pre = self.rw.complete > 0.5
        self.docks = torch.zeros(self.sim.R, P.N_DOCK_CANDIDATES, 4,
                                 device=self.device)

    # ------------------------------------------------------------------
    def run(self, policy, theta, steps, collect_obs=True):
        """theta: (P, n_params). Tra ve diem trung binh moi ban sao (P,)."""
        p = policy.unpack(theta)
        h = policy.new_state(self.n_pop, self.unit)
        s = self.sim
        obs_sum = obs_sq = None
        for _ in range(steps):
            if s.lidar_step():
                self.docks = D.detect(s.scan_r, s.scan_b, s.scan_ok)
            s.cover()
            obs = PC.build(s, self.docks)
            if collect_obs:
                # Cong don bang so thuc 32 bit roi moi doi sang 64 bit tren
                # CPU o cuoi: card lien khong co so thuc 64 bit, ma 64 dau
                # vao deu nam trong [-1;1] nen 32 bit thua do chinh xac.
                s1 = obs.sum(0)
                s2 = (obs * obs).sum(0)
                if obs_sum is None:
                    obs_sum, obs_sq = s1, s2
                else:
                    obs_sum = obs_sum + s1
                    obs_sq = obs_sq + s2
            y, h = policy.step(p, obs.view(self.n_pop, self.unit, -1), h)
            ev = s.step(y.reshape(s.R, -1))
            self.rw.step(ev)
        n_obs = steps * s.R
        tot = self.rw.total.view(self.n_pop, self.unit).mean(dim=1)
        return tot, (obs_sum, obs_sq, n_obs)

    # ------------------------------------------------------------------
    def stats(self):
        """Thong ke gop, tinh tren ban sao dau tien (de xem bo nao dang lam gi)."""
        v = lambda a: a.view(self.n_pop, self.unit)
        return dict(charged=float(v(self.rw.charged).sum(1).mean()),
                    wrong=float(v(self.rw.wrong).sum(1).mean()),
                    falls=float(v(self.rw.fell).sum(1).mean()),
                    flats=float(v(self.rw.flat).sum(1).mean()),
                    beacons=float(v(self.rw.beacons).sum(1).mean()),
                    charges_ok=float(v(self.rw.charges_ok).sum(1).mean()),
                    cells=float(v(self.rw.cells).sum(1).mean()),
                    complete=float(v(self._complete()).sum(1).mean()),
                    clawed=float(v(self.rw.clawed).sum(1).mean()))

    def _complete(self):
        """Xe lam xong de bai TRONG lan chay nay (khong tinh xe da xong san)."""
        s = self.sim
        return ((self.rw.complete > 0.5) & (s.task_beacons >= P.TASK_BEACONS)
                & (s.task_charges >= P.TASK_CHARGES) & ~self._pre).float()
