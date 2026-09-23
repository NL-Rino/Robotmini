# -*- coding: utf-8 -*-
"""Ham phan thuong cua ban v1, tinh cho ca lo mot luc.

KHONG dat lai mot con so nao. Tat ca hang so deu lay tu `train/reward.py`,
vi hai ban phai cho cung mot cau tra loi cho cung mot hanh vi - neu khong
thi moi so do cua ban v1 (ngoi trong hoc quay tit: -22 thay vi 994) khong
con noi gi ve ban nay.

Cho khac duy nhat: cham goi tat theo TUNG XE chu khong tat tren mat bang,
vi mot mat bang o day duoc ca quan the dung chung.

De bai (xem `train/reward.py`): 5 cham goi + 3 lan sac HOP LE. Tien tra dan
trong luc sac chi la tam ung, rut ra truoc khi day thi bi thu lai.
"""

import torch

from sim import params as P
from train.reward import (ALIVE, BEACON, BEACON_EXTRA, BUMP, CHARGE_DONE,
                          CHARGE_ENERGY, CHARGE_LATCH, CLIFF_CAP, CLIFF_WARN,
                          COMPLETE, EXPLORE, FALL, FLAT, FULL_ENOUGH, HOMING,
                          LOITER, LOITER_CAP, SPIN, SPIN_FREE, SPIN_IN_DOCK,
                          WRONG_DOCK)
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
        self.charges_ok = z()
        self.cells = z()
        self.clawed = z()
        self.homing_on = torch.zeros(R, dtype=torch.bool, device=dev)
        self.loiter_paid = z()
        self.spin_in_dock = z()
        self.cliff_paid = z()
        self.reset_start()

    def reset_start(self):
        """Goi NGAY SAU khi dat xe va dat tien do de bai: chot cac moc."""
        s = self.sim
        self.prev_dist = hypot(s.x - self.ax, s.y - self.ay)
        self.prev_batt = s.batt.clone()
        # Tien do cap san boi giao trinh khong duoc tra lai.
        self.prev_beacons = s.task_beacons.clone()
        self.prev_charges = s.task_charges.clone()
        self.advance = torch.zeros_like(s.batt)
        self.complete = self._done().float()

    def _done(self):
        s = self.sim
        return ((s.task_beacons >= P.TASK_BEACONS)
                & (s.task_charges >= P.TASK_CHARGES))

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

        # Sac. Tam ung khi vua cam HOP LE (pin < 20%) va theo nang luong nap
        # trong lan hop le; day 100% -> tra CHARGE_DONE; rut ra giua chung
        # -> thu lai toan bo tam ung. Cung thu tu voi train/reward.py:
        # latch_charge chay truoc step.
        latch = ev["latch"] & s.charge_valid & live
        # (lan sac day ngay trong buoc vua cam thi charge_valid da tat -
        # khong xay ra duoc vi nap mot buoc chi 0,25%)
        r = r + torch.where(latch, torch.full_like(r, CHARGE_LATCH), zero)
        self.advance = self.advance + torch.where(
            latch, torch.full_like(r, CHARGE_LATCH), zero)

        d_batt = s.batt - self.prev_batt
        n_done = s.task_charges - self.prev_charges
        done = n_done > 0.5
        r = r + torch.where(done, CHARGE_DONE * n_done, zero)
        self.charges_ok = self.charges_ok + torch.where(done, n_done, zero)
        self.prev_charges = s.task_charges.clone()
        pay = torch.where(~done & s.charging & s.charge_valid & live,
                          CHARGE_ENERGY * d_batt.clamp(min=0.0), zero)
        r = r + pay
        claw = ~done & ~s.charge_valid & (self.advance > 0.0) & live
        r = r - torch.where(claw, self.advance, zero)
        self.clawed = self.clawed + torch.where(claw, self.advance, zero)
        self.advance = torch.where(done | claw, zero, self.advance + pay)
        self.charged = self.charged + torch.where(s.charging, d_batt.clamp(min=0.0), zero)
        self.prev_batt = s.batt.clone()

        # Cham goi: sim da an va dem; o day chi doc so dem.
        n_b = s.task_beacons - self.prev_beacons
        within = (self.prev_beacons < P.TASK_BEACONS).float()
        r = r + torch.where(n_b > 0.5, n_b * (BEACON * within
                                              + BEACON_EXTRA * (1.0 - within)),
                            zero)
        self.beacons = self.beacons + n_b
        self.prev_beacons = s.task_beacons.clone()

        # Kham pha: o moi trong vong quet vua roi.
        cells = torch.round(s.new_area * 12.0)
        r = r + torch.where(live, EXPLORE * cells, zero)
        self.cells = self.cells + torch.where(live, cells, zero)

        # Lam xong de bai: thuong mot lan.
        fin = self._done() & (self.complete < 0.5) & live
        r = r + torch.where(fin, torch.full_like(r, COMPLETE), zero)
        self.complete = torch.where(fin, torch.ones_like(self.complete),
                                    self.complete)

        # pin day ma van nam trong hoc thi bat dau lo von
        full = s.charging & (s.batt > FULL_ENOUGH) & live
        pen = torch.clamp(LOITER_CAP - self.loiter_paid, min=LOITER)
        pen = torch.where(full, pen, zero)
        r = r + pen
        self.loiter_paid = self.loiter_paid + pen

        r = r + torch.where(ev["wrong"] & live, torch.full_like(r, WRONG_DOCK), zero)
        self.wrong = self.wrong + (ev["wrong"] & live).float()

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
