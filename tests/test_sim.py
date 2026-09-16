"""Kiem thu phan mo phong.

Trong tam nam o bon dieu moi:
  1. Chay MAI - chi dut ket noi bo nao moi dung, xe chet khong dung the gioi
  2. Duoi 15% pin chi co MOT DAU VAO NHAP NHAY, khong co lop cuong ep nao
  3. Muon sac thi phai LUI DUOI VAO, cam dau vao thi khong an thua
  4. Moi xe mot MA HOC SAC; cam nham hoc thi cham duoc nhung khong co dien
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from brain.rule_brain import RuleBrain
from sim import dock_detector, params as P, perception as PC, sensors
from sim.fleet import FleetSim, LocalBrains
from sim.geometry import wrap_pi_scalar
from sim.lidar import Lidar
from sim.robot import Robot
from sim.geometry import SegmentSet
from sim.world import Dock, World, make_fleet_map


def scan_from(world, x, y, th, steps=8, seed=7, circles=()):
    lid = Lidar(seed=seed)
    out = None
    for i in range(steps):
        s = lid.update(P.DT, (x, y, th), (x, y, th), world.static_segments,
                       list(circles), i * P.DT)
        if s is not None:
            out = s
    return out


def docked_pose(dock, lateral=0.0, back_off=0.0, spin=0.0):
    """Tu the xe khi da lui duoi vao het co."""
    depth = P.DOCK_CAVITY_D - P.BODY_RADIUS - back_off
    x = dock.x - depth * math.cos(dock.theta) - lateral * math.sin(dock.theta)
    y = dock.y - depth * math.sin(dock.theta) + lateral * math.cos(dock.theta)
    return x, y, wrap_pi_scalar(dock.theta + spin)


# ---------------------------------------------------------------- hinh hoc
class TestWorld(unittest.TestCase):
    def test_kich_thuoc_hoc(self):
        d = Dock(2.0, 2.0, 0.0, code=7)
        xs = [p[0] for p in d.polygon]
        ys = [p[1] for p in d.polygon]
        self.assertAlmostEqual(max(ys) - min(ys), P.DOCK_OUTER_W, places=6)
        self.assertAlmostEqual(max(xs) - min(xs), P.DOCK_OUTER_D, places=6)
        self.assertAlmostEqual(math.hypot(d.contact_x - d.x, d.contact_y - d.y),
                               P.DOCK_CAVITY_D, places=6)

    def test_than_xe_lot_duoc_vao_long_hoc(self):
        self.assertGreater(P.DOCK_CAVITY_W, 2 * P.BODY_RADIUS)
        self.assertLess(P.DOCK_CAVITY_W - 2 * P.BODY_RADIUS, 0.02,
                        "khe ho moi ben phai duoi 1 cm, neu khong thi bai toan de qua")

    def test_vuc_la_ngoai_san(self):
        w = make_fleet_map(3)
        self.assertTrue(w.on_floor(3.0, 2.0))
        vx = sum(p[0] for p in w.voids[0]) / 4.0
        vy = sum(p[1] for p in w.voids[0]) / 4.0
        self.assertFalse(w.on_floor(vx, vy))

    def test_moi_hoc_mot_ma_khac_nhau(self):
        w = make_fleet_map(3, n_docks=5)
        codes = [d.code for d in w.docks if d.code is not None]
        self.assertEqual(len(codes), 5)
        self.assertEqual(len(set(codes)), 5, "hai hoc trung ma thi het y nghia")


# ---------------------------------------------------------------- LiDAR
class TestLidar(unittest.TestCase):
    def test_gom_du_mot_vong_moi_tra_ket_qua(self):
        w = make_fleet_map(3)
        lid = Lidar(seed=1)
        got = [lid.update(P.DT, (3.0, 2.4, 0.0), (3.0, 2.4, 0.0),
                          w.static_segments, [], i * P.DT) is not None
               for i in range(7)]
        self.assertEqual(got.count(True), 2, "6 Hz, buoc 50 ms -> ~1 vong moi 3,3 buoc")
        self.assertEqual(len(lid.scan), lid.n)

    def test_do_dung_khoang_cach_toi_tuong(self):
        # Phong tron, khong vat can, khong hoc: chi co bon buc tuong.
        floor = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
        walls = SegmentSet.from_polyline(
            [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)], closed=True)
        w = World(floor, walls)
        sc = scan_from(w, 2.0, 2.0, 0.0)
        for want_bearing, want_range in ((0.0, 4.0), (math.pi, 2.0),
                                         (math.pi / 2, 2.0), (-math.pi / 2, 2.0)):
            # Camsense rot ~2% diem, nen phai lay diem HOP LE gan phuong vi
            # can do nhat, chu khong phai diem gan nhat.
            err = np.abs(wrap_pi_scalar(sc.bearings - want_bearing))
            err = np.where(sc.valid, err, np.inf)
            i = int(np.argmin(err))
            self.assertLess(float(err[i]), 0.05)
            self.assertAlmostEqual(float(sc.ranges[i]), want_range, delta=0.08)

    def test_khu_nhoe_chi_dung_odometry(self):
        """Neu bo nhoe bang goc THAT thi robot that se khong lam duoc nhu vay.

        Cho odometry lech han mot goc so voi su that: vong quet phai nam
        trong he quy chieu ODOM, tuc la phuong vi phai lech theo dung goc do.
        """
        w = make_fleet_map(3)
        lid_a = Lidar(seed=5)
        lid_b = Lidar(seed=5)
        off = 0.4
        sa = sb = None
        for i in range(8):
            r = lid_a.update(P.DT, (3.0, 2.4, 0.0), (3.0, 2.4, 0.0),
                             w.static_segments, [], i * P.DT)
            if r is not None:
                sa = r
            r = lid_b.update(P.DT, (3.0, 2.4, 0.0), (3.0, 2.4, off),
                             w.static_segments, [], i * P.DT)
            if r is not None:
                sb = r
        np.testing.assert_allclose(sa.ranges, sb.ranges, atol=1e-9)
        d = wrap_pi_scalar(sb.bearings - sa.bearings)
        self.assertAlmostEqual(float(np.median(d)), 0.0, delta=1e-6,
                               msg="vong quet phai nam trong he odom, khong phai he that")


# ---------------------------------------------------------------- bo do hoc
class TestDockDetector(unittest.TestCase):
    def test_nhin_thang_vao_hoc_thi_do_dung(self):
        w = make_fleet_map(3)
        d = w.docks[0]
        for dist in (0.6, 0.9, 1.3):
            px = d.x + dist * math.cos(d.theta)
            py = d.y + dist * math.sin(d.theta)
            th = d.theta + math.pi
            cands = dock_detector.detect(scan_from(w, px, py, th))
            self.assertTrue(cands, f"khong thay hoc o cu ly {dist} m")
            c = cands[0]
            cx = px + c.range * math.cos(th + c.bearing)
            cy = py + c.range * math.sin(th + c.bearing)
            self.assertLess(math.hypot(cx - d.x, cy - d.y), 0.12)
            self.assertLess(abs(math.degrees(wrap_pi_scalar(th + c.axis - d.theta))), 12.0)

    def test_dung_giua_phong_trong_thi_khong_bao_bua(self):
        w = make_fleet_map(3, n_docks=5, with_void=False)
        far = [c for c in dock_detector.detect(scan_from(w, 3.2, 2.4, 0.0))]
        for c in far:
            cx = 3.2 + c.range * math.cos(c.bearing)
            cy = 2.4 + c.range * math.sin(c.bearing)
            near = min(math.hypot(cx - d.x, cy - d.y) for d in w.docks)
            self.assertLess(near, 1.0, "bao mot cai hoc o cho khong co hoc nao")


# ------------------------------------------------- LUI DUOI VAO + MA HOC SAC
class TestDockContact(unittest.TestCase):
    def setUp(self):
        self.w = make_fleet_map(3)
        self.d = self.w.docks[0]
        self.code = self.d.code

    def test_lui_duoi_vao_dung_ma_thi_co_dien(self):
        x, y, th = docked_pose(self.d)
        c = sensors.dock_contact(self.w, x, y, th, self.code)
        self.assertTrue(c.in_slot)
        self.assertTrue(c.id_signal)
        self.assertTrue(c.charging)

    def test_cam_dau_vao_thi_khong_cham_duoc(self):
        """Quay 180 do tai cung cho: than xe tron nen van lot, nhung chan
        tiep dien o duoi xe luc nay chi ra ngoai cua."""
        x, y, th = docked_pose(self.d, spin=math.pi)
        c = sensors.dock_contact(self.w, x, y, th, self.code)
        self.assertFalse(c.in_slot)
        self.assertFalse(c.charging)

    def test_cam_nham_hoc_thi_cham_duoc_nhung_khong_ra_dien(self):
        other = [d for d in self.w.docks
                 if d.code is not None and d.code != self.code][0]
        x, y, th = docked_pose(other)
        c = sensors.dock_contact(self.w, x, y, th, self.code)
        self.assertTrue(c.in_slot, "chan tiep dien phai cham that")
        self.assertFalse(c.id_signal, "hoc nguoi khac thi khong phat tin hieu")
        self.assertFalse(c.charging)
        self.assertTrue(c.wrong_dock)

    def test_hoc_moi_nhu_khong_phat_gi_ca(self):
        decoy = [d for d in self.w.docks if d.code is None][0]
        x, y, th = docked_pose(decoy)
        c = sensors.dock_contact(self.w, x, y, th, self.code)
        self.assertTrue(c.in_slot)
        self.assertFalse(c.charging)
        self.assertFalse(decoy.ir_on, "hoc moi nhu cung khong duoc phat hong ngoai")

    def test_lech_ngang_hay_chua_vao_het_thi_truot_chan(self):
        for kw in ({"lateral": 0.04}, {"back_off": 0.06}, {"spin": math.radians(20)}):
            x, y, th = docked_pose(self.d, **kw)
            self.assertFalse(sensors.dock_contact(self.w, x, y, th, self.code).in_slot,
                             f"khong duoc cham voi {kw}")

    def test_co_cam_manh_vao_hoc_nguoi_khac_cung_khong_sac_duoc(self):
        other = [d for d in self.w.docks
                 if d.code is not None and d.code != self.code][0]
        x, y, th = docked_pose(other, back_off=0.30)
        r = Robot(0, x, y, th, dock_code=self.code, seed=1)
        r.battery = 0.5
        for _ in range(120):                       # lui het ga vao trong
            r.step_motion(-1.0, -1.0, P.DT, self.w, [])
            r.step_power(P.DT, sensors.dock_contact(self.w, r.x, r.y, r.th, r.code))
        self.assertTrue(r.in_slot, "phai cam duoc vao that")
        self.assertFalse(r.charging)
        self.assertLess(r.battery, 0.5, "cam nham hoc ma pin van tut, khong he nap")
        self.assertGreaterEqual(r.n_wrong_dock, 1)

    def test_dem_cam_nham_theo_LAN_chu_khong_theo_buoc(self):
        other = [d for d in self.w.docks
                 if d.code is not None and d.code != self.code][0]
        x, y, th = docked_pose(other)
        r = Robot(0, x, y, th, dock_code=self.code, seed=1)
        for _ in range(60):
            r.step_power(P.DT, sensors.dock_contact(self.w, r.x, r.y, r.th, r.code))
        self.assertEqual(r.n_wrong_dock, 1)


# ---------------------------------------------------------------- den bao sac
class TestLowBatteryLamp(unittest.TestCase):
    def test_tren_nguong_thi_khong_nhap_nhay(self):
        r = Robot(0, 0, 0, 0, dock_code=1, seed=1)
        r.battery = 0.5
        r.step_power(P.DT, sensors.ContactState())
        self.assertFalse(any(r.low_battery_blink(t * 0.05) for t in range(200)))

    def test_duoi_15_phan_tram_thi_nhap_nhay_MAI(self):
        r = Robot(0, 0, 0, 0, dock_code=1, seed=1)
        r.battery = 0.10
        r.step_power(P.DT, sensors.ContactState())
        vals = [r.low_battery_blink(t * 0.05) for t in range(400)]
        self.assertIn(0.0, vals)
        self.assertIn(1.0, vals)
        # va no khong tat di sau mot luc
        self.assertIn(1.0, vals[-60:], "den phai nhap nhay lien tuc, khong tu tat")

    def test_khong_con_lop_cuong_ep_ve_sac_nao(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertFalse(os.path.exists(os.path.join(here, "sim", "failsafe.py")),
                         "lop cuong ep ve sac da bi bo han")

    def test_pin_thap_khong_lam_doi_lenh_ga(self):
        """Den bao sac CHI la mot dau vao. Khong ai cuop quyen lai ca.

        Cho mot bo nao lai het ga thang toi, roi vat pin xuong 2%. Neu con
        lop cuong ep nao thi ga se bi de xuong khac di; o day phai y nguyen.
        """
        sim = FleetSim(seed=3, n_robots=2)
        for r in sim.robots:
            r.battery = 0.02
        seen = []

        class Ga:
            def __init__(self, rid):
                self.rid = rid

            def __call__(self, obs, t):
                seen.append(float(obs[PC.I_BATTERY + 1]))
                return 0.7, 0.7

        lb = LocalBrains(lambda rid: Ga(rid), sim.robot_ids)
        sim.run(lb, max_seconds=1.0)
        for r in sim.robots:
            self.assertAlmostEqual(r.cmd_l, 0.7, places=6)
            self.assertAlmostEqual(r.cmd_r, 0.7, places=6)
        self.assertIn(1.0, seen, "den bao phai co luc sang")
        self.assertIn(0.0, seen, "va co luc tat - tuc la dang nhap nhay")


# ---------------------------------------------------------------- dau vao
class TestPerception(unittest.TestCase):
    def test_dung_48_dau_vao_va_deu_huu_han(self):
        sim = FleetSim(seed=3, n_robots=3)
        obs = sim.observe()
        self.assertEqual(len(PC.INPUT_NAMES), P.N_INPUTS)
        for v in obs.values():
            self.assertEqual(v.shape, (P.N_INPUTS,))
            self.assertTrue(np.all(np.isfinite(v)))
            self.assertTrue(np.all(np.abs(v) <= 1.0 + 1e-6))

    def test_luc_xuat_phat_xe_dang_cam_dung_hoc_cua_no(self):
        sim = FleetSim(seed=3, n_robots=5)
        obs = sim.observe()
        for rid, v in obs.items():
            self.assertEqual(v[PC.I_CONTACT + 0], 1.0, "phai dang nam trong hoc")
            self.assertEqual(v[PC.I_CONTACT + 1], 1.0, "va hoc phai phat tin hieu")
            self.assertEqual(v[PC.I_STATION + 0], 1.0, "biet cho tram ngay tu buoc 0")

    def test_ga_bao_lai_la_ga_da_chay_chu_khong_phai_ga_duoc_yeu_cau(self):
        sim = FleetSim(seed=3, n_robots=1)
        r = sim.robots[0]
        r.stranded = True                     # het pin: banh bi khoa
        sim.step({0: (1.0, -1.0)})
        obs = sim.observe()
        self.assertEqual(obs[0][PC.I_MOTION + 2], 0.0)
        self.assertEqual(obs[0][PC.I_MOTION + 3], 0.0)


# ------------------------------------------------- chay mai toi khi mat nao
class TestRunForever(unittest.TestCase):
    def test_chi_dung_khi_dut_ket_noi_bo_nao(self):
        sim = FleetSim(seed=3, n_robots=3)
        lb = LocalBrains(lambda rid: (lambda o, t: (0.5, 0.4)), sim.robot_ids)

        def hook(s):
            if s.steps == 250:
                lb.disconnect()

        rep = sim.run(lb, max_seconds=None, on_step=hook)
        self.assertEqual(rep.steps, 250)
        self.assertIn("ngat ket noi", rep.stop_reason)

    def test_mat_nao_tung_xe_mot_thi_van_chay_tiep(self):
        sim = FleetSim(seed=3, n_robots=3)
        lb = LocalBrains(lambda rid: (lambda o, t: (0.5, 0.4)), sim.robot_ids)

        def hook(s):
            if s.steps == 60:
                lb.disconnect(0)
            elif s.steps == 120:
                lb.disconnect(1)
            elif s.steps == 200:
                lb.disconnect(2)

        rep = sim.run(lb, max_seconds=None, on_step=hook)
        self.assertEqual(rep.steps, 200, "chi xe cuoi cung mat nao moi dung")
        self.assertTrue(sim.robots[0].brain_lost)

    def test_xe_het_pin_khong_lam_dung_the_gioi(self):
        sim = FleetSim(seed=3, n_robots=3)
        for r in sim.robots:
            r.battery = 0.001
        lb = LocalBrains(lambda rid: (lambda o, t: (1.0, 1.0)), sim.robot_ids)
        rep = sim.run(lb, max_seconds=30.0)
        self.assertTrue(all(r.stranded for r in sim.robots))
        self.assertGreater(rep.sim_seconds, 29.0, "the gioi phai chay het 30 giay")

    def test_xe_roi_xuong_vuc_khong_lam_dung_the_gioi(self):
        sim = FleetSim(seed=3, n_robots=2)
        v = sim.world.voids[0]
        cx = sum(p[0] for p in v) / 4.0
        cy = sum(p[1] for p in v) / 4.0
        sim.robots[0].x, sim.robots[0].y = cx, cy
        lb = LocalBrains(lambda rid: (lambda o, t: (0.3, 0.3)), sim.robot_ids)
        rep = sim.run(lb, max_seconds=5.0)
        self.assertTrue(sim.robots[0].fallen)
        self.assertGreater(rep.sim_seconds, 4.0)


# ---------------------------------------------------------------- doi xe
class TestFleet(unittest.TestCase):
    def test_nam_xe_nam_ma_khac_nhau_moi_xe_mot_hoc(self):
        sim = FleetSim(seed=3, n_robots=5)
        codes = [r.code for r in sim.robots]
        self.assertEqual(len(set(codes)), 5)
        for rid, dock in sim.home.items():
            self.assertEqual(dock.code, sim.robots[rid].code)

    def test_khong_du_hoc_co_ma_thi_bao_loi_ngay(self):
        with self.assertRaises(ValueError):
            FleetSim(seed=3, n_robots=7)

    def test_cac_xe_nhin_thay_nhau_tren_lidar(self):
        sim = FleetSim(seed=3, n_robots=2)
        a, b = sim.robots
        a.x, a.y, a.th = 3.0, 2.4, 0.0
        b.x, b.y = 3.7, 2.4
        for _ in range(4):
            sim.observe()
        obs = sim.observe()
        front = (1.0 - obs[a.id][PC.I_FANS]) * P.LIDAR_MAX
        self.assertLess(front, 0.75, "phai thay cai xe dung truoc mat")


# ---------------------------------------------------------------- bo luat mau
class TestRuleBrain(unittest.TestCase):
    def test_chay_duoc_va_khong_no(self):
        sim = FleetSim(seed=3, n_robots=5)
        lb = LocalBrains(lambda rid: RuleBrain(rid, 1), sim.robot_ids)
        rep = sim.run(lb, max_seconds=120.0)
        self.assertGreater(rep.steps, 2000)
        self.assertTrue(all(r.distance > 1.0 for r in sim.robots),
                        "xe nao cung phai ra khoi hoc va chay duoc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
