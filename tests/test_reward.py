"""Kiem thu ham phan thuong - cac lo hong da bit.

Moi bai o day deu ung voi mot hanh vi ma bo nao DA tim ra hoac SE tim ra
neu khong chan. Phan thuong khong phai chuyen tham my: bo nao giai dung cai
bai toan ta viet ra, ke ca khi ta viet nham.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.rule_brain import RuleBrain
from sim import params as P, sensors
from sim.fleet import FleetSim
from train.reward import (
    CLIFF_CAP, FALL, FLAT, LOITER_CAP, RewardTracker,
)


def docked_sim(seed=3, battery=0.35):
    """Mot xe, dat san trong hoc cua no, tiep diem da an."""
    sim = FleetSim(seed=seed, n_robots=1)
    r = sim.robots[0]
    d = sim.home[0]
    dep = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005
    sim.place_robot(0, d.x - dep * math.cos(d.theta),
                    d.y - dep * math.sin(d.theta), d.theta, battery=battery)
    c = sensors.dock_contact(sim.world, r.x, r.y, r.th, r.code)
    r.step_power(0.0, c)
    return sim, r, d


def play(sim, tracker, policy, steps):
    r = sim.robots[0]
    for i in range(steps):
        obs = sim.observe()
        was = r.charging
        sim.step({0: policy(i, r, obs[0])})
        tracker.latch_charge(r, was)
        tracker.step(sim.world, r, sim.dt)
    return tracker.total


class TestCampingTrongSac(unittest.TestCase):
    """Bo nao cu ngoi trong sac quay qua quay lai de an thuong."""

    def test_quay_trong_hoc_bi_phat_nang(self):
        sim, _r, _d = docked_sim(battery=0.9)
        tr = RewardTracker(0, sim.home[0], sim.robots[0])
        play(sim, tr, lambda i, r, o: (0.8, -0.8), 600)
        self.assertLess(tr.total, -100.0,
                        "quay nguoi trong long hoc phai lo nang")
        self.assertGreater(tr.spin_in_dock, 50.0)

    def test_quay_NGOAI_hoc_thi_khong_bi_phat_cai_do(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        x, y, th = sim.free_pose(__import__("random").Random(1))
        sim.place_robot(0, x, y, th, battery=0.9)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr, o: (0.8, -0.8), 400)
        self.assertEqual(tr.spin_in_dock, 0.0,
                         "ra ngoai roi thi muon quay bao nhieu cung duoc")

    def test_lac_ra_lac_vao_khong_an_duoc_thuong_cam(self):
        sim, r, _d = docked_sim(battery=0.5)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr, o: ((0.55, 0.55) if (i // 14) % 2 == 0
                                        else (-0.55, -0.55)), 900)
        self.assertGreater(r.n_charges, 5, "phai that su cam vao nhieu lan")
        self.assertLess(tr.total, 80.0,
                        "cam di cam lai tai cho khong duoc tra tien")

    def test_ngoi_yen_khi_pin_day_thi_lo_dan(self):
        sim, r, _d = docked_sim(battery=0.99)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr, o: (0.0, 0.0), 500)
        self.assertLess(tr.total, 0.0, "pin day ma van nam trong hoc thi lo")
        self.assertLess(tr.loiter_paid, -10.0)

    def test_xa_roi_nap_lai_trong_hoc_khong_duoc_tra_hai_lan(self):
        """Han muc nang luong moi lan vao hoc."""
        sim, r, _d = docked_sim(battery=0.30)
        tr = RewardTracker(0, sim.home[0], r)
        # xuat phat trong hoc -> han muc bang 0, sac day cung khong duoc gi
        play(sim, tr, lambda i, rr, o: (0.0, 0.0), 1400)
        self.assertGreater(r.battery, 0.97, "van phai sac day that")
        self.assertLess(tr.total, 30.0,
                        "sac ma chua di lam thi khong duoc tra tien")

    def test_di_lam_ve_thi_DUOC_tra_tien(self):
        """Phia nguoc lai: bit lo hong ma bit luon duong song thi hong."""
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        d = sim.home[0]
        ax, ay = d.x + 0.55 * math.cos(d.theta), d.y + 0.55 * math.sin(d.theta)
        # dat xe cach hoc 2 m, pin can -> no da "di lam" that
        sim.place_robot(0, ax + 2.0 * math.cos(d.theta),
                        ay + 2.0 * math.sin(d.theta), d.theta, battery=0.20)
        tr = RewardTracker(0, d, r)
        before = r.battery
        # lui thang ve hoc
        play(sim, tr, lambda i, rr, o: (-0.35, -0.35), 400)
        self.assertGreaterEqual(r.n_charges, 1, "phai cam duoc vao that")
        self.assertGreater(tr.total, 25.0,
                           "di lam ve cam dung hoc thi phai duoc tra tien")
        self.assertGreater(r.battery, before)


class TestTranTrenCacKhoanPhat(unittest.TestCase):
    """Moi khoan phat cong don phai NHE HON cai chet.

    Neu khong thi dung canh ho dat hon lao xuong ho, va bo nao se chon lao
    xuong - dung cai bay (h) cua ban cu, chi khac dau.
    """

    def test_phat_vuc_va_phat_nam_li_deu_nhe_hon_chet(self):
        self.assertGreater(CLIFF_CAP, FALL)
        self.assertGreater(LOITER_CAP, FLAT)
        self.assertGreater(CLIFF_CAP + LOITER_CAP, FALL,
                           "hai khoan cong lai van phai nhe hon mot cai chet")

    def test_dung_canh_ho_mai_thi_phat_khong_tang_vo_han(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        v = sim.world.voids[0]
        vx = 0.5 * (v[0][0] + v[2][0])
        # dung ngay sat mep ho, mui huong vao ho
        sim.place_robot(0, vx, v[0][1] - P.BODY_RADIUS - 0.02, math.pi / 2,
                        battery=0.9)
        tr = RewardTracker(0, sim.home[0], r)
        play(sim, tr, lambda i, rr, o: (0.0, 0.0), 900)
        self.assertGreaterEqual(tr.cliff_paid, CLIFF_CAP - 1e-6)
        self.assertGreater(tr.total, FALL, "dung canh ho phai re hon roi xuong")


class TestLuiVaoKhongBiPhatOan(unittest.TestCase):
    def test_nan_huong_nhe_luc_lui_vao_thi_khong_bi_phat(self):
        """Luc lui vao van phai nan huong. Phat ca cai do thi khong xe nao
        cam duoc vao hoc nua."""
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        d = sim.home[0]
        sim.place_robot(0, d.x + 0.45 * math.cos(d.theta),
                        d.y + 0.45 * math.sin(d.theta), d.theta, battery=0.5)
        tr = RewardTracker(0, d, r)
        # lui vao voi mot chut nan huong, dung nhu bo luat mau lam
        play(sim, tr, lambda i, rr, o: (-0.13 - 0.03, -0.13 + 0.03), 200)
        self.assertLess(tr.spin_in_dock, 1.0,
                        "nan huong nhe luc lui vao khong duoc tinh la quay")

    def test_bo_luat_mau_khong_bi_phat_quay_trong_hoc(self):
        sim, r, d = docked_sim(battery=0.30)
        tr = RewardTracker(0, d, r)
        b = RuleBrain(0, 1)
        play(sim, tr, lambda i, rr, o: b(o, sim.t), 3000)
        self.assertLess(tr.spin_in_dock, 5.0,
                        "bo luat mau phai di thang ra roi moi quay")


class TestLamViecVanLaChienLuocTotNhat(unittest.TestCase):
    def test_lam_viec_an_dut_moi_kieu_an_gian(self):
        def score(policy, steps=2500, battery=0.35):
            sim, r, d = docked_sim(battery=battery)
            tr = RewardTracker(0, d, r)
            play(sim, tr, policy, steps)
            return tr.total

        b = RuleBrain(0, 1)
        lam_viec = score(lambda i, r, o: b(o, i * P.DT))
        ngoi_yen = score(lambda i, r, o: (0.0, 0.0))
        quay_tit = score(lambda i, r, o: (0.8, -0.8))
        lac = score(lambda i, r, o: ((0.55, 0.55) if (i // 14) % 2 == 0
                                     else (-0.55, -0.55)))
        for name, val in (("ngoi yen", ngoi_yen), ("quay tit", quay_tit),
                          ("lac ra vao", lac)):
            self.assertGreater(lam_viec, val + 100.0,
                               f"lam viec ({lam_viec:.0f}) phai an dut "
                               f"{name} ({val:.0f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
