# -*- coding: utf-8 -*-
"""Kiem thu de bai moi o ban chay theo lo - phai giong `tests/test_task.py`.

  - lan sac hop le: cam luc pin < 20%, nam toi khi day -> dem mot lan
  - cam luc pin con nhieu -> khong dem; chay ra giua chung -> thu lai tam ung
  - cham goi: an thi dem, tat, roi sang lai o cho khac
  - luoi kham pha dem o moi sau moi vong quet
  - LiDAR thay chan ban ghe va HAI CHAN nguoi, va cham thi tinh than nguoi
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch

from sim import params as P
from turbo import sim as TS, world as TW
from turbo.reward import BatchReward

DEV = torch.device("cpu")


def docked_sim(battery, n=3, seed=3):
    bw = TW.build([seed], n, DEV, n_docks=n, n_decoys=1)
    bw.freeze_movers()
    s = TS.BatchSim(bw, DEV, seed=1)
    x, y, th = TW.spawn_in_dock(bw.home_pose())
    s.place(torch.arange(s.R), x, y, th,
            battery=torch.full((s.R,), float(battery)))
    return s


def run(s, rw, steps, action=None):
    a = torch.zeros(s.R, 2) if action is None else action
    for _ in range(steps):
        s.lidar_step()
        s.cover()
        ev = s.step(a)
        if rw is not None:
            rw.step(ev)


class TestSacHopLe(unittest.TestCase):
    def test_duoi_20_phan_tram_thi_dem_mot_lan(self):
        s = docked_sim(0.12)
        rw = BatchReward(s)
        run(s, rw, int(20.0 / P.DT))
        self.assertTrue(bool(s.charging.all()))
        self.assertTrue(bool((s.batt >= P.BATT_FULL).all()))
        self.assertEqual(s.task_charges.tolist(), [1.0] * s.R)
        self.assertEqual(rw.charges_ok.tolist(), [1.0] * s.R)

    def test_cam_luc_pin_con_nhieu_thi_khong_dem(self):
        s = docked_sim(0.5)
        rw = BatchReward(s)
        run(s, rw, int(12.0 / P.DT))
        self.assertTrue(bool((s.batt >= P.BATT_FULL).all()))
        self.assertEqual(s.task_charges.tolist(), [0.0] * s.R)

    def test_chay_ra_giua_chung_thi_thu_lai(self):
        # Pin 12% nhung dat xe ngoai hoc, lui vao roi chay ra.
        bw = TW.build([3], 1, DEV, n_docks=1, n_decoys=0)
        bw.freeze_movers()
        s = TS.BatchSim(bw, DEV, seed=1)
        hp = bw.home_pose()
        x = hp[:, 0] + 0.30 * torch.cos(hp[:, 2])
        y = hp[:, 1] + 0.30 * torch.sin(hp[:, 2])
        s.place(torch.arange(1), x, y, hp[:, 2].clone(),
                battery=torch.tensor([0.12]))
        rw = BatchReward(s)
        back = torch.tensor([[-0.3, -0.3]])
        for _ in range(200):
            run(s, rw, 1, back)
            if bool(s.charging[0]):
                break
        self.assertTrue(bool(s.charging[0]))
        self.assertTrue(bool(s.charge_valid[0]))
        run(s, rw, int(6.0 / P.DT))
        run(s, rw, 60, torch.tensor([[0.4, 0.4]]))
        self.assertFalse(bool(s.charging[0]))
        self.assertFalse(bool(s.charge_valid[0]))
        self.assertEqual(float(s.task_charges[0]), 0.0)
        self.assertGreater(float(rw.clawed[0]), 20.0)


class TestChamGoi(unittest.TestCase):
    def test_an_roi_sang_lai_cho_khac(self):
        bw = TW.build([3], 1, DEV, n_docks=1, n_decoys=0)
        bw.freeze_movers()
        s = TS.BatchSim(bw, DEV, seed=1)
        bx, by = float(s.bpos[0, 0, 0]), float(s.bpos[0, 0, 1])
        s.place(torch.arange(1), torch.tensor([bx - 0.6]), torch.tensor([by]),
                torch.tensor([0.0]), battery=torch.tensor([0.9]))
        rw = BatchReward(s)
        run(s, rw, 30, torch.tensor([[0.5, 0.5]]))
        self.assertEqual(float(s.task_beacons[0]), 1.0)
        self.assertFalse(bool(s.bon[0, 0]))
        self.assertGreater(float(rw.total[0]), 20.0)
        run(s, rw, int(P.BEACON_RESPAWN[1] / P.DT) + 40)
        self.assertTrue(bool(s.bon[0, 0]))
        moved = math.hypot(float(s.bpos[0, 0, 0]) - bx,
                           float(s.bpos[0, 0, 1]) - by)
        self.assertGreater(moved, 0.3)


class TestKhamPha(unittest.TestCase):
    def test_vong_quet_dau_tien_thay_nhieu_o_roi_het_dan(self):
        s = docked_sim(0.9)
        rw = BatchReward(s)
        run(s, rw, 40)
        first = rw.cells.clone()
        self.assertTrue(bool((first > 5).all()))
        run(s, rw, 200)
        self.assertTrue(bool((rw.cells - first <= 0.2 * first + 3).all()),
                        "dung yen thi het o moi")


class TestChanNguoi(unittest.TestCase):
    def test_lidar_thay_chan_va_cham_tinh_than(self):
        bw = TW.build([3], 1, DEV, n_docks=1, n_decoys=0)
        s = TS.BatchSim(bw, DEV, seed=1)
        lid = s._lidar_circles()
        col = s._circles()
        n_leg = bw.legs.shape[1]
        self.assertEqual(lid.shape[1], n_leg + bw.mover_legs.shape[1])
        self.assertEqual(col.shape[1], n_leg + bw.mover.shape[1])
        # than nguoi to hon chan nguoi
        body = bw.mover[0, :, 2]
        legs = bw.mover_legs[0, :, 2]
        self.assertTrue(bool((body[body > 0] > 0.15).all()))
        self.assertTrue(bool((legs[legs > 0] < 0.08).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
