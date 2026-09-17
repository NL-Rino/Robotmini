# -*- coding: utf-8 -*-
"""Ham phan thuong cua ban v1, tinh cho ca lo mot luc.

KHONG dat lai mot con so nao. Tat ca hang so deu lay tu `train/reward.py`,
vi hai ban phai cho cung mot cau tra loi cho cung mot hanh vi - neu khong
thi moi so do cua ban v1 (ngoi trong hoc quay tit: -22 thay vi 994) khong
con noi gi ve ban nay.

Cho khac duy nhat: den goi tat theo TUNG XE chu khong tat tren mat bang, vi
mot mat bang o day duoc ca quan the dung chung.
"""

import torch

from sim import params as P
from train.reward import (ALIVE, AWAY_DIST, BEACON, BEACON_RADIUS, BUMP,
                          CHARGE_ENERGY, CHARGE_LATCH, CLIFF_CAP, CLIFF_WARN,
                          FALL, FLAT, FULL_ENOUGH, HOMING, LOITER, LOITER_CAP,
                          SPIN, SPIN_FREE, SPIN_IN_DOCK, WRONG_DOCK)
from .ops import hypot
from .world import approach_point


class BatchReward:
    """Mot bo dem cho moi xe trong lo."""

    def __init__(self, sim):
        self.sim = sim
        dev = sim.device
        R = sim.R
        hp = sim.w.home_pose()
        self.ax, self.ay = approach_point(hp)
        self.hx, self.hy = hp[:, 0], hp[:, 1]

        z = lambda: torch.zeros(R, device=dev)
        self.total = z()
        self.charged = z()
        self.wrong = z()
        self.fell = z()
        self.flat = z()
        self.beacons = z()
        self.homing_on = torch.zeros(R, dtype=torch.bool, device=dev)
        self.loiter_paid = z()
        self.spin_in_dock = z()
        self.cliff_paid = z()
        self.reset_start()

    def reset_start(self):
        """Goi NGAY SAU khi dat xe: chot moc khoang cach, pin, han muc sac."""
        s = self.sim
        self.prev_dist = hypot(s.x - self.ax, s.y - self.ay)
        self.prev_batt = s.batt.clone()
        # Xe xuat phat NGOAI hoc thi coi nhu da di xa roi. Chi xe xuat phat
        # trong hoc va dang co dien moi phai di mot vong roi ve.
        docked = s.charging
        self.max_away = torch.where(docked, torch.zeros_like(s.batt),
                                    torch.full_like(s.batt, AWAY_DIST))
        self.budget = torch.where(docked, torch.zeros_like(s.batt),
                                  (1.0 - s.batt).clamp(min=0.0))

    # ------------------------------------------------------------------
    def step(self, ev):
        """`ev` la tu dien su kien tu `BatchSim.step`. Tra ve diem buoc nay."""
        s = self.sim
        zero = torch.zeros_like(s.batt)
        r = torch.zeros_like(s.batt)

        # chet: tra mot lan roi thoi, va tu do khong tinh gi nua
        new_fall = ev["fell"] & (self.fell < 0.5)
        new_flat = ev["flat"] & (self.flat < 0.5)
        r = r + torch.where(new_fall, torch.full_like(r, FALL), zero)
        r = r + torch.where(new_flat, torch.full_like(r, FLAT), zero)
        self.fell = torch.where(ev["fell"], torch.ones_like(self.fell), self.fell)
        self.flat = torch.where(ev["flat"], torch.ones_like(self.flat), self.flat)
        dead = s.fallen | s.stranded

        live = ~dead
        r = r + torch.where(live, torch.full_like(r, ALIVE), zero)

        inside = s.in_cavity()

        # va cham: dem theo LAN, mien khi that su dang trong long hoc
        r = r + torch.where(ev["bump"] & ~inside & live,
                            torch.full_like(r, BUMP), zero)

        cl, cr = s.cliff()
        warn = (cl | cr) & live
        # `max(CLIFF_WARN, con lai)`: tra -0,8 moi buoc cho toi khi cham tran
        pen = torch.clamp(CLIFF_CAP - self.cliff_paid, min=CLIFF_WARN)
        pen = torch.where(warn, pen, zero)
        r = r + pen
        self.cliff_paid = self.cliff_paid + pen

        spin = (s.wv.abs() > 0.8 * P.W_MAX) & live
        r = r + torch.where(spin, torch.full_like(r, SPIN), zero)

        # TRONG LONG HOC THI KHONG DUOC QUAY NGUOI.
        excess = (s.wv.abs() / P.W_MAX - SPIN_FREE).clamp(min=0.0)
        pen = torch.where(inside & live, SPIN_IN_DOCK * excess, zero)
        r = r + pen
        self.spin_in_dock = self.spin_in_dock - pen

        # thuong mot lan khi vua cam dung hoc CUA MINH - neu da di lam ve
        latch = ev["latch"] & (self.max_away >= AWAY_DIST) & live
        r = r + torch.where(latch, torch.full_like(r, CHARGE_LATCH), zero)
        self.max_away = torch.where(latch, zero, self.max_away)
        self.budget = torch.where(latch, (1.0 - s.batt).clamp(min=0.0),
                                  self.budget)

        # sac: tra theo NANG LUONG nap duoc, moi lan vao hoc mot han muc
        d_batt = s.batt - self.prev_batt
        pay = torch.minimum(d_batt.clamp(min=0.0), self.budget.clamp(min=0.0))
        pay = torch.where(s.charging & live, pay, zero)
        r = r + CHARGE_ENERGY * pay
        self.budget = self.budget - pay
        self.charged = self.charged + torch.where(s.charging, d_batt.clamp(min=0.0), zero)
        self.prev_batt = s.batt.clone()

        # pin day ma van nam trong hoc thi bat dau lo von
        full = s.charging & (s.batt > FULL_ENOUGH) & live
        pen = torch.clamp(LOITER_CAP - self.loiter_paid, min=LOITER)
        pen = torch.where(full, pen, zero)
        r = r + pen
        self.loiter_paid = self.loiter_paid + pen

        away = hypot(s.x - self.hx, s.y - self.hy)
        self.max_away = torch.maximum(self.max_away, away)

        r = r + torch.where(ev["wrong"] & live, torch.full_like(r, WRONG_DOCK), zero)
        self.wrong = self.wrong + (ev["wrong"] & live).float()

        got = s.take_beacon(BEACON_RADIUS) & live
        r = r + torch.where(got, torch.full_like(r, BEACON), zero)
        self.beacons = self.beacons + got.float()

        # dan duong: chi khi den bao sac dang sang, va dan toi DIEM DUNG
        # TRUOC MIENG chu khong phai toi cai hoc
        dist = hypot(s.x - self.ax, s.y - self.ay)
        gain = HOMING * (self.prev_dist - dist)
        r = r + torch.where(s.low_lamp & self.homing_on & live, gain, zero)
        self.homing_on = s.low_lamp
        self.prev_dist = dist

        # da chet thi khong con cong gi nua ngoai khoan phat mot lan o tren
        r = torch.where(dead & ~(new_fall | new_flat), zero, r)
        self.total = self.total + r
        return r
