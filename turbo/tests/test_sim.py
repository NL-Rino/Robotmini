# -*- coding: utf-8 -*-
"""Kiem thu ban mo phong theo lo.

Ban turbo phai la CUNG MOT THE GIOI voi ban v1, chi khac cach tinh. Neu vat
ly lech nhau thi moi con so do duoc cua ban v1 deu mat nghia, va bo nao nuoi
o day mang ve may that se cu xu khac. Nen kiem truc tiep: chay song song v1
va turbo tu cung mot diem, cung mot chuoi lenh, roi so tung buoc.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from sim import params as P
from sim.robot import Robot
from sim.world import make_fleet_map
from turbo import sim as TS, world as TW

DEV = torch.device("cpu")


class TestSameWorld(unittest.TestCase):
    """Mat bang cua turbo la mat bang cua v1, khong phai ban sao gan giong."""

    def test_mat_bang_khop(self):
        bw = TW.build([3, 7], 2, DEV, n_docks=3, n_decoys=1)
        for wi, w in enumerate(bw.worlds):
            self.assertIs(w, bw.worlds[wi])
            r = wi * 2
            self.assertEqual(int(bw.seg[r].shape[0]),
                             int(bw.seg[r + 1].shape[0]))
            n = len(w.static_segments)
            for i in range(0, n, max(1, n // 11)):
                self.assertAlmostEqual(float(bw.seg[r, i, 0]),
                                       float(w.static_segments.ax[i]), places=5)
                self.assertAlmostEqual(float(bw.seg[r, i, 3]),
                                       float(w.static_segments.by[i]), places=5)

    def test_moi_xe_mot_hoc_rieng(self):
        bw = TW.build([3], 3, DEV, n_docks=3, n_decoys=1)
        codes = [float(bw.my_code[i]) for i in range(3)]
        self.assertEqual(len(set(codes)), 3, "hai xe an chung mot ma hoc")
        self.assertTrue(all(c >= 0 for c in codes), "xe nhan hoc moi lam")


class TestPhysicsMatchesV1(unittest.TestCase):
    """Vat ly trung voi ban v1 trong sai so may."""

    def test_di_400_buoc(self):
        seed = 7
        w = make_fleet_map(seed, n_docks=3, n_decoys=1)
        bw = TW.build([seed], 1, DEV, n_docks=3, n_decoys=1)
        bw.freeze_movers()

        pose = bw.home_pose()
        x0 = float(pose[0, 0] + 1.2 * math.cos(pose[0, 2]))
        y0 = float(pose[0, 1] + 1.2 * math.sin(pose[0, 2]))
        th0 = float(pose[0, 2]) + math.pi

        s = TS.BatchSim(bw, DEV, seed=1)
        s.place(torch.arange(1), torch.tensor([x0]), torch.tensor([y0]),
                torch.tensor([th0]))
        r1 = Robot(0, x0, y0, th0, dock_code=0, seed=1)
        # dung DUNG sai so odometry cua turbo, vi hai ben boc so ngau nhien
        # khac nhau - cho nay kiem vat ly, khong kiem bo sinh so ngau nhien.
        r1._odom_scale_l = float(s.od_l[0])
        r1._odom_scale_r = float(s.od_r[0])
        r1._odom_drift = float(s.od_drift[0])

        w.movers = []
        free = all_ = odo = 0.0
        touched = False
        for i in range(400):
            a = float(np.sin(i * 0.031) * 0.8)
            b = float(np.cos(i * 0.017) * 0.8)
            s.step(torch.tensor([[a, b]]))
            hit, _ = r1.step_motion(a, b, P.DT, w, [])
            d = math.hypot(float(s.x[0]) - r1.x, float(s.y[0]) - r1.y)
            all_ = max(all_, d)
            odo = max(odo, math.hypot(float(s.ox[0]) - r1.ox,
                                      float(s.oy[0]) - r1.oy))
            self.assertEqual(hit, bool(s.bump[0] > 0.99),
                             f"buoc {i}: mot ben bao cham, ben kia khong")
            touched = touched or hit
            if not touched:
                free = max(free, d)
        self.assertTrue(touched, "quy dao nay le ra phai co luc cham tuong")
        # Chay tu do: phai trung trong sai so cua so float32.
        self.assertLess(free, 5e-5, f"chay tu do lech {free * 1e6:.1f} um")
        # Luc day ra khoi tuong thi hai ban KHAC NHAU co chu y: ban v1 day
        # lan luot khoi tung doan mot, ban turbo day khoi doan an sau nhat
        # roi tinh lai. O goc tuong (cham hai doan cung luc) ket qua lech vai
        # milimet. Khong gop duoc thanh mot phep tinh neu con day lan luot.
        self.assertLess(all_, 0.02, f"sau va cham lech {all_ * 1000:.1f} mm")
        self.assertLess(odo, 0.02, f"odometry lech {odo * 1000:.1f} mm")


class TestBatchIndependence(unittest.TestCase):
    """Xe trong lo khong duoc anh huong nhau qua phep tinh."""

    def test_them_xe_khong_doi_ket_qua(self):
        bw1 = TW.build([3], 1, DEV, n_docks=3, n_decoys=1)
        bw4 = TW.build([3], 4, DEV, n_docks=3, n_decoys=1)
        bw1.freeze_movers()
        bw4.freeze_movers()
        p1, p4 = bw1.home_pose(), bw4.home_pose()
        x = float(p1[0, 0] + 1.0 * math.cos(p1[0, 2]))
        y = float(p1[0, 1] + 1.0 * math.sin(p1[0, 2]))
        th = float(p1[0, 2]) + math.pi

        a = TS.BatchSim(bw1, DEV, seed=2)
        a.place(torch.arange(1), torch.tensor([x]), torch.tensor([y]),
                torch.tensor([th]))
        b = TS.BatchSim(bw4, DEV, seed=2)
        b.place(torch.arange(4), torch.full((4,), x), torch.full((4,), y),
                torch.full((4,), th))
        for i in range(120):
            u = float(np.sin(i * 0.05))
            a.step(torch.tensor([[u, 0.6]]))
            b.step(torch.tensor([[u, 0.6]] * 4))
        for k in range(4):
            self.assertAlmostEqual(float(a.x[0]), float(b.x[k]), places=5)
            self.assertAlmostEqual(float(a.y[0]), float(b.y[k]), places=5)


class TestDockRules(unittest.TestCase):
    """Bon dieu cua ban v1 phai con nguyen o ban turbo."""

    def test_cam_dau_vao_khong_sac(self):
        bw = TW.build([3], 1, DEV, n_docks=3, n_decoys=1)
        pose = bw.home_pose()
        s = TS.BatchSim(bw, DEV, seed=0)
        # mui huong VAO hoc (nguoc voi kieu lui duoi vao)
        nose = (pose[:, 0] - 0.10 * torch.cos(pose[:, 2]),
                pose[:, 1] - 0.10 * torch.sin(pose[:, 2]),
                pose[:, 2] + math.pi)
        s.place(torch.arange(1), *nose)
        for _ in range(60):
            s.step(torch.tensor([[-0.6, -0.6]]))
        self.assertFalse(bool(s.charging[0]),
                         "cam dau vao hoc ma van sac duoc")

    def test_lui_duoi_vao_thi_sac(self):
        bw = TW.build([3], 1, DEV, n_docks=3, n_decoys=1)
        pose = bw.home_pose()
        s = TS.BatchSim(bw, DEV, seed=0)
        x, y, th = TW.spawn_in_dock(pose)
        s.place(torch.arange(1), x, y, th, battery=torch.tensor([0.4]))
        for _ in range(40):
            s.step(torch.tensor([[-0.4, -0.4]]))
        self.assertTrue(bool(s.charging[0]), "lui dung hoc cua minh ma khong sac")
        self.assertGreater(float(s.batt[0]), 0.4, "khong nap them duoc pin")

    def test_hoc_nguoi_khac_khong_sac(self):
        bw = TW.build([3], 2, DEV, n_docks=3, n_decoys=1)
        s = TS.BatchSim(bw, DEV, seed=0)
        # dat xe 0 vao hoc cua xe 1
        pose = bw.dock[1, int(bw.home[1]), :3][None, :]
        x, y, th = TW.spawn_in_dock(pose)
        s.place(torch.tensor([0]), x, y, th, battery=torch.tensor([0.4]))
        for _ in range(40):
            s.step(torch.zeros(2, 2))
        self.assertTrue(bool(s.in_slot[0]), "khong cham duoc tiep diem")
        self.assertFalse(bool(s.charging[0]), "cam nham hoc ma van co dien")
        self.assertLess(float(s.batt[0]), 0.4, "pin phai tiep tuc voi")

    def test_den_bao_pin_yeu(self):
        bw = TW.build([3], 1, DEV, n_docks=3, n_decoys=1)
        s = TS.BatchSim(bw, DEV, seed=0)
        p = bw.home_pose()
        s.place(torch.arange(1), p[:, 0] + 1.0 * torch.cos(p[:, 2]),
                p[:, 1] + 1.0 * torch.sin(p[:, 2]), p[:, 2],
                battery=torch.tensor([P.BATT_LOW + 0.004]))
        self.assertFalse(bool(s.low_lamp[0]))
        for _ in range(1500):
            s.step(torch.zeros(1, 2))
            if bool(s.low_lamp[0]):
                break
        self.assertTrue(bool(s.low_lamp[0]), "xuong duoi 15% ma den khong bao")


if __name__ == "__main__":
    unittest.main(verbosity=2)
