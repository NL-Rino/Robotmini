"""Kiem thu DE BAI moi va can nha.

  - hoan thanh = an du 5 cham goi VA sac du 3 lan HOP LE
  - mot lan sac hop le: cam vao luc pin < 20%, nam yen cho toi khi day 100%;
    rut ra giua chung thi khong duoc gi, ke ca tien tam ung cung bi thu lai
  - can nha to nhieu phong, ban ghe duoi mat LiDAR chi con la chan
  - toi cho LiDAR thay khu vuc MOI thi duoc cong diem
  - nguoi di lai la hai cai chan, mot cai hien ra roi bien mat theo nhip buoc
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim import params as P, perception as PC
from sim.fleet import FleetSim
from sim.world import Mover, make_fleet_map
from train.reward import (BEACON, CHARGE_DONE, COMPLETE, EXPLORE,
                          RewardTracker)


def play(sim, tr, policy, steps):
    r = sim.robots[0]
    for i in range(steps):
        sim.observe()
        was = r.charging
        sim.step({0: policy(i, r)})
        tr.latch_charge(r, was)
        tr.step(sim.world, r, sim.dt)


def lui_vao(sim, battery):
    """Dat xe truoc mieng hoc cua no, mui quay ra - chi can lui thang."""
    d = sim.home[0]
    sim.place_robot(0, d.x + 0.30 * math.cos(d.theta),
                    d.y + 0.30 * math.sin(d.theta), d.theta, battery=battery)
    return sim.robots[0]


# ---------------------------------------------------------------- can nha
class TestCanNha(unittest.TestCase):
    def test_nha_to_hon_xe_rat_nhieu_va_co_ban_ghe(self):
        w = make_fleet_map(3)
        x0, y0, x1, y1 = w.bounds
        self.assertGreaterEqual((x1 - x0) * (y1 - y0), 100.0,
                                "can nha phai to (>= 100 m2)")
        self.assertGreater(len(w.legs), 12, "phai co ban ghe (chan)")
        for _x, _y, r in w.legs:
            self.assertLess(r, 0.05, "chan ghe chi la mot cham nho")

    def test_moi_hoc_o_mot_phong_khac(self):
        for seed in range(8):
            w = make_fleet_map(seed, n_docks=3)
            coded = [d for d in w.docks if d.code is not None]
            self.assertEqual(len(coded), 3)
            for a in coded:
                for b in coded:
                    if a is not b:
                        self.assertGreater(math.hypot(a.x - b.x, a.y - b.y),
                                           1.5)

    def test_chan_ghe_hien_tren_lidar(self):
        sim = FleetSim(seed=3, n_robots=1)
        lx, ly, _r = sim.world.legs[0]
        # dat xe cach chan ghe 1 m, mui huong vao no
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            x = lx + math.cos(ang)
            y = ly + math.sin(ang)
            if not sim.world.on_floor(x, y):
                continue
            sim.place_robot(0, x, y, ang + math.pi, battery=0.9)
            break
        sim.world.movers = []
        for _ in range(8):
            sim.observe()
        scan = sim.robots[0].lidar.scan
        near = scan.valid & (np.abs(scan.bearings) < 0.2)
        self.assertTrue(near.any())
        self.assertLess(float(scan.ranges[near].min()), 1.05)


class TestNguoiLaHaiChan(unittest.TestCase):
    def test_dung_yen_thay_ca_hai_chan(self):
        m = Mover(2.0, 2.0, waypoints=[])
        m.step(0.05, random.Random(0))
        legs = m.leg_circles()
        self.assertEqual(len(legs), 2)
        self.assertTrue(all(r > 0 for _x, _y, r in legs))

    def test_di_thi_chan_hien_ra_roi_bien_mat(self):
        m = Mover(1.0, 1.0, speed=0.5, waypoints=[(9.0, 1.0), (1.0, 1.0)])
        rng = random.Random(0)
        seen = []
        for _ in range(80):
            m.step(0.05, rng)
            seen.append(tuple(r > 0 for _x, _y, r in m.leg_circles()))
        self.assertIn((True, True), seen, "co luc thay ca hai chan")
        self.assertTrue(any(not a or not b for a, b in seen),
                        "co luc mot chan nhac len, bien khoi mat quet")
        self.assertNotIn((False, False), seen, "khong bao gio mat ca hai")

    def test_va_cham_van_la_than_nguoi(self):
        w = make_fleet_map(3)
        solid = w.solid_circles()
        for m in w.movers:
            self.assertIn(m.as_circle(), solid)
            for leg in m.leg_circles():
                self.assertNotIn(leg, solid)


# ---------------------------------------------------------------- cham goi
class TestChamGoi(unittest.TestCase):
    def test_cham_goi_khong_tu_tat(self):
        w = make_fleet_map(3)
        rng = random.Random(1)
        for _ in range(4000):
            w.step_dynamics(0.05, rng)
        self.assertTrue(all(b.on for b in w.beacons))

    def test_an_cham_thi_dem_va_duoc_thuong_roi_sang_cho_khac(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        b = sim.world.beacons[0]
        bx, by = b.x, b.y
        sim.place_robot(0, bx - 0.6, by, 0.0, battery=0.9)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr: (0.5, 0.5), 30)
        self.assertEqual(r.task_beacons, 1)
        self.assertEqual(tr.beacons, 1)
        self.assertFalse(b.on)
        # sau vai giay no sang lai, o CHO KHAC
        play(sim, tr, lambda i, rr: (0.0, 0.0),
             int(P.BEACON_RESPAWN[1] / P.DT) + 40)
        self.assertTrue(b.on)
        self.assertGreater(math.hypot(b.x - bx, b.y - by), 0.3)
        self.assertGreater(tr.total, BEACON - 10.0)

    def test_dau_vao_tien_do_cham(self):
        sim = FleetSim(seed=3, n_robots=1)
        sim.robots[0].task_beacons = 2
        v = sim.observe()[0]
        self.assertAlmostEqual(v[PC.I_TASK + 0], 2.0 / P.TASK_BEACONS)


# ---------------------------------------------------------------- sac hop le
class TestLanSacHopLe(unittest.TestCase):
    def test_duoi_20_phan_tram_sac_day_thi_tinh_mot_lan(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = lui_vao(sim, battery=0.15)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr: (0.0, 0.0) if rr.charging else (-0.3, -0.3),
             int(22.0 / P.DT))
        self.assertGreaterEqual(r.battery, P.BATT_FULL)
        self.assertEqual(r.task_charges, 1)
        self.assertEqual(tr.charges_ok, 1)
        self.assertGreater(tr.total, CHARGE_DONE)

    def test_cam_luc_pin_con_nhieu_thi_khong_tinh(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = lui_vao(sim, battery=0.45)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr: (0.0, 0.0) if rr.charging else (-0.3, -0.3),
             int(16.0 / P.DT))
        self.assertGreaterEqual(r.battery, P.BATT_FULL)
        self.assertEqual(r.task_charges, 0, "cam luc 45% thi khong phai lan hop le")
        self.assertEqual(tr.charges_ok, 0)
        # khong co tien sac nao: chi con song + kham pha
        self.assertLess(tr.total, 0.02 * 320 + EXPLORE * tr.cells + 1.0)

    def test_rut_ra_giua_chung_thi_mat_het(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = lui_vao(sim, battery=0.12)
        tr = RewardTracker(0, sim.home[0], r)
        state = {"t_in": None}

        def pol(i, rr):
            if rr.charging and state["t_in"] is None:
                state["t_in"] = i
            if state["t_in"] is None:
                return (-0.3, -0.3)
            if i - state["t_in"] < int(8.0 / P.DT):
                return (0.0, 0.0)          # sac duoc mot nua...
            return (0.4, 0.4)              # ...roi chay ra

        play(sim, tr, pol, int(14.0 / P.DT))
        self.assertLess(r.battery, P.BATT_FULL)
        self.assertEqual(r.task_charges, 0)
        self.assertFalse(r.charge_valid)
        self.assertGreater(tr.clawed, 20.0, "tien tam ung phai bi thu lai")
        # khong duoc loi gi tu lan sac do (chi con song + kham pha)
        self.assertLess(tr.total, 0.02 * 280 + EXPLORE * tr.cells + 1.0)

    def test_dau_vao_lan_sac_dang_hop_le(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = lui_vao(sim, battery=0.10)
        for _ in range(200):
            sim.observe()
            sim.step({0: (0.0, 0.0) if r.charging else (-0.3, -0.3)})
            if r.charging:
                break
        self.assertTrue(r.charging)
        v = sim.observe()[0]
        self.assertEqual(v[PC.I_TASK + 2], 1.0)


# ---------------------------------------------------------------- kham pha
class TestKhamPha(unittest.TestCase):
    def test_di_sang_phong_khac_duoc_them_o_moi(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        rng = random.Random(4)
        x, y, th = sim.free_pose(rng)
        sim.place_robot(0, x, y, th, battery=0.9)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr: (0.0, 0.0), 60)
        dung_yen = tr.cells
        self.assertGreater(dung_yen, 0)
        play(sim, tr, lambda i, rr: (0.0, 0.0), 200)
        # Chi con le te vai o (nguoi di qua che/ho ra), khong phai mot nguon.
        self.assertLess(tr.cells - dung_yen, 0.2 * dung_yen + 3,
                        "dung yen mot cho thi het o moi rat nhanh")
        # Sang mot phong khac (o goc doi dien can nha) thi lai co o moi.
        truoc = tr.cells
        xa = max((sim.free_pose(rng) for _ in range(30)),
                 key=lambda p: math.hypot(p[0] - x, p[1] - y))
        sim.place_robot(0, xa[0], xa[1], xa[2], battery=0.9)
        play(sim, tr, lambda i, rr: (0.0, 0.0), 60)
        self.assertGreater(tr.cells - truoc, 10)

    def test_moi_xe_mot_ban_do_rieng(self):
        sim = FleetSim(seed=3, n_robots=2)
        sim.observe()
        for _ in range(10):
            sim.observe()
            sim.step({0: (0.0, 0.0), 1: (0.0, 0.0)})
        a = sim.robots[0].seen_cells
        b = sim.robots[1].seen_cells
        self.assertFalse(np.array_equal(a, b))


# ---------------------------------------------------------------- hoan thanh
class TestHoanThanh(unittest.TestCase):
    def test_lam_xong_de_bai_duoc_thuong_lon_mot_lan(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        b = sim.world.beacons[0]
        sim.place_robot(0, b.x - 0.6, b.y, 0.0, battery=0.9)
        r.task_beacons = P.TASK_BEACONS - 1
        r.task_charges = P.TASK_CHARGES
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr: (0.5, 0.5), 30)
        self.assertEqual(tr.complete, 1)
        self.assertGreater(tr.total, COMPLETE)
        before = tr.total
        play(sim, tr, lambda i, rr: (0.0, 0.0), 20)
        self.assertLess(tr.total - before, 5.0, "thuong hoan thanh chi mot lan")

    def test_tien_do_cap_san_khong_duoc_tra_lai(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        r.task_beacons = P.TASK_BEACONS
        r.task_charges = P.TASK_CHARGES
        tr = RewardTracker(0, sim.home[0], r)
        self.assertEqual(tr.complete, 1)
        play(sim, tr, lambda i, rr: (0.0, 0.0), 10)
        self.assertLess(tr.total, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
