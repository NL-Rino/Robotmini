# -*- coding: utf-8 -*-
"""Kiem thu ban v2.

Trong tam: nhung thu chi ban 3D moi co, va nhung thu ma neu hong thi khong
ai bao gi ca - chi la hang chuc gio thue GPU troi qua vo ich.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch

from v2 import params as P
from v2 import raytrace as RT
from v2.camera import render
from v2.env import FleetEnv3D, scalar_names
from v2.policy import ActorCritic
from v2.ppo import PPO, Buffer, collect
from v2.world3d import PALETTE, make_batch

CPU = torch.device("cpu")


def tiny_scene(box=(3.0, 0.0, 0.5, 0.2, 1.0, 0.5)):
    cx, cy, cz, hx, hy, hz = box
    return dict(
        box_c=torch.tensor([[[cx, cy, cz]]]), box_h=torch.tensor([[[hx, hy, hz]]]),
        box_yaw=torch.zeros(1, 1), box_col=torch.tensor([[[1.0, 0.0, 0.0]]]),
        box_mat=torch.zeros(1, 1, dtype=torch.long),
        box_code=torch.zeros(1, 1, 3, dtype=torch.long),
        cyl_c=torch.zeros(1, 0, 2), cyl_r=torch.zeros(1, 0),
        cyl_z=torch.zeros(1, 0, 2), cyl_col=torch.zeros(1, 0, 3),
        floor=torch.tensor([[-10.0, -10.0, 10.0, 10.0]]),
        void=torch.zeros(1, 0, 4),
        floor_col=torch.tensor([[[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]]),
        ceil_z=torch.tensor([2.45]), ceil_col=torch.tensor([[0.5, 0.5, 0.5]]),
        palette=torch.eye(3))


class TestRaytrace(unittest.TestCase):
    def test_cham_hop_dung_khoang_cach_va_phap_tuyen(self):
        sc = tiny_scene()
        o = torch.tensor([[[0.0, 0.0, 0.5]]])
        d = torch.tensor([[[1.0, 0.0, 0.0]]])
        t, _a, n, k = RT.trace(o, d, sc)
        self.assertAlmostEqual(float(t[0, 0]), 2.8, places=5)
        self.assertEqual(int(k[0, 0]), RT.KIND_BOX)
        self.assertAlmostEqual(float(n[0, 0, 0]), -1.0, places=5)

    def test_hop_xoay(self):
        sc = tiny_scene(box=(3.0, 0.0, 0.5, 0.5, 0.05, 0.5))
        sc["box_yaw"] = torch.tensor([[math.pi / 2]])
        o = torch.tensor([[[0.0, 0.0, 0.5]]])
        d = torch.tensor([[[1.0, 0.0, 0.0]]])
        t, _a, _n, _k = RT.trace(o, d, sc)
        # xoay 90 do thi be day 0,05 quay ra truoc -> cham o 2,95
        self.assertAlmostEqual(float(t[0, 0]), 2.95, places=4)

    def test_san_tran_va_lo_thung(self):
        sc = tiny_scene()
        o = torch.tensor([[[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]]])
        d = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
        t, _a, _n, k = RT.trace(o, d, sc)
        self.assertAlmostEqual(float(t[0, 0]), 0.3, places=5)
        self.assertEqual(int(k[0, 0]), RT.KIND_FLOOR)
        self.assertAlmostEqual(float(t[0, 1]), 2.15, places=5)
        self.assertEqual(int(k[0, 1]), RT.KIND_CEIL)

        sc["void"] = torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]])
        t2, _a, _n, k2 = RT.trace(o, d, sc)
        self.assertNotEqual(int(k2[0, 0]), RT.KIND_FLOOR,
                            "tia chieu vao lo thung thi khong duoc cham san")

    def test_tru_dung(self):
        sc = tiny_scene()
        sc["box_h"] = torch.zeros(1, 1, 3)
        sc["cyl_c"] = torch.tensor([[[2.0, 0.0]]])
        sc["cyl_r"] = torch.tensor([[0.25]])
        sc["cyl_z"] = torch.tensor([[[0.0, 1.6]]])
        sc["cyl_col"] = torch.tensor([[[0.0, 1.0, 0.0]]])
        o = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 1.9]]])
        d = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        t, _a, _n, k = RT.trace(o, d, sc)
        self.assertAlmostEqual(float(t[0, 0]), 1.75, places=4)
        self.assertEqual(int(k[0, 0]), RT.KIND_CYL)
        self.assertNotEqual(int(k[0, 1]), RT.KIND_CYL,
                            "tia bay cao hon dinh tru thi khong duoc cham")


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.sc, self.metas = make_batch(1, CPU, seed0=3)
        self.dock = self.metas[0]["docks"][1]

    def _look(self, dist, tilt_deg, w=P.CAM_W, h=P.CAM_H):
        d = self.dock
        x = torch.tensor([d["x"] + dist * math.cos(d["theta"])])
        y = torch.tensor([d["y"] + dist * math.sin(d["theta"])])
        th = torch.tensor([d["theta"] + math.pi])
        tl = torch.tensor([math.radians(tilt_deg)])
        return render(self.sc, x, y, th, tl, w=w, h=h)

    def test_anh_dung_kich_thuoc_va_trong_khoang(self):
        img = self._look(1.2, 0.0)
        self.assertEqual(tuple(img.shape), (1, 3, P.CAM_H, P.CAM_W))
        self.assertGreaterEqual(float(img.min()), 0.0)
        self.assertLessEqual(float(img.max()), 1.0)

    def test_DOC_DUOC_BANG_MA_TREN_HOC(self):
        """Day la ly do ca du an v2 ton tai.

        Ngua camera len nhin bang ma o cu ly gan, roi doi chieu mau tung o
        voi ma that cua hoc. Neu bai nay hong thi camera chi la mot manh
        trang tri dat tien.
        """
        code = self.dock["code"]
        img = self._look(1.0, 20.0, w=256, h=192)[0]
        # tim hang anh co nhieu mau ruc nhat -> do la dai bang ma
        sat = img.max(0).values - img.min(0).values
        row = int(sat.mean(dim=1).argmax())
        band = img[:, row, :]
        vivid = sat[row] > 0.18
        cols = torch.nonzero(vivid).flatten()
        self.assertGreater(cols.numel(), 12, "khong thay dai mau nao")
        x0, x1 = int(cols.min()), int(cols.max())
        pal = torch.tensor(PALETTE)
        read = []
        for c in range(P.MARK_CELLS):
            a = x0 + (x1 - x0 + 1) * c // P.MARK_CELLS
            b = x0 + (x1 - x0 + 1) * (c + 1) // P.MARK_CELLS
            px = band[:, (a + b) // 2]
            read.append(int((pal - px[None] / px.max()).abs().sum(1).argmin()))
        self.assertEqual(tuple(read), tuple(code),
                         f"doc duoc {read} nhung ma that la {list(code)}")

    def test_servo_doi_thuc_su_goc_nhin(self):
        up = self._look(1.5, 35.0)[0]
        dn = self._look(1.5, -35.0)[0]
        # cui xuong thi nua duoi anh phai la san (sang hon tran/tuong xa)
        self.assertGreater(float(dn[:, P.CAM_H // 2:, :].mean()),
                           float(up[:, P.CAM_H // 2:, :].mean()) - 1.0)
        self.assertGreater(float((up - dn).abs().mean()), 0.05,
                           "hai goc servo phai cho ra hai anh khac han nhau")

    def test_cang_xa_cang_kho_doc(self):
        near = self._look(1.0, 20.0, w=256, h=192)[0]
        far = self._look(4.0, 20.0, w=256, h=192)[0]
        s_near = float((near.max(0).values - near.min(0).values).max())
        s_far = float((far.max(0).values - far.min(0).values).max())
        self.assertGreater(s_near, s_far,
                           "o gan phai thay mau ruc hon o xa")


class TestEnv3D(unittest.TestCase):
    def setUp(self):
        self.e = FleetEnv3D(n_envs=6, device=CPU, seed=2)

    def test_quan_sat_dung_dang(self):
        img, s = self.e.observe()
        self.assertEqual(tuple(img.shape), (6, 3, P.CAM_H, P.CAM_W))
        self.assertEqual(tuple(s.shape), (6, P.N_SCALARS))
        self.assertEqual(len(scalar_names()), P.N_SCALARS)
        self.assertTrue(bool(torch.isfinite(s).all()))
        self.assertLessEqual(float(s.abs().max()), 1.0 + 1e-5)

    def _dock_pose(self):
        pose = self.e.sc["dock_pose"]
        return pose.gather(1, self.e.home[:, None, None].expand(-1, 1, 3)
                           ).squeeze(1)

    def _put(self, xoff=0.0, yoff=0.0, spin=0.0):
        hp = self._dock_pose()
        dep = P.DOCK_CAVITY_D - P.BODY_RADIUS + xoff
        self.e.x = hp[:, 0] - dep * torch.cos(hp[:, 2]) - yoff * torch.sin(hp[:, 2])
        self.e.y = hp[:, 1] - dep * torch.sin(hp[:, 2]) + yoff * torch.cos(hp[:, 2])
        self.e.th = hp[:, 2] + spin
        self.e.in_slot[:] = False
        return self.e._contact()

    def test_phai_lui_duoi_vao_moi_sac_duoc(self):
        ins, ok, _w = self._put()
        self.assertTrue(bool(ins.all()) and bool(ok.all()))
        ins, ok, _w = self._put(spin=math.pi)
        self.assertFalse(bool(ins.any()), "cam dau vao thi khong duoc cham")

    def test_cam_nham_hoc_thi_khong_ra_dien(self):
        pose = self.e.sc["dock_pose"]
        other = (self.e.home + 1) % self.e.n_docks
        op = pose.gather(1, other[:, None, None].expand(-1, 1, 3)).squeeze(1)
        dep = P.DOCK_CAVITY_D - P.BODY_RADIUS
        self.e.x = op[:, 0] - dep * torch.cos(op[:, 2])
        self.e.y = op[:, 1] - dep * torch.sin(op[:, 2])
        self.e.th = op[:, 2]
        self.e.in_slot[:] = False
        ins, ok, _w = self.e._contact()
        self.assertTrue(bool(ins.all()), "chan tiep dien phai cham that")
        self.assertFalse(bool(ok.any()), "nhung hoc nguoi khac thi khong ra dien")

    def test_lech_ngang_hay_lech_goc_thi_truot_chan(self):
        for kw in ({"yoff": 0.04}, {"spin": math.radians(20)}, {"xoff": -0.07}):
            ins, _o, _w = self._put(**kw)
            self.assertFalse(bool(ins.any()), f"khong duoc cham voi {kw}")

    def test_XE_CHUI_DUOC_XUONG_GAM_BAN(self):
        """Thu chi ban 3D moi co.

        Mat ban o do cao 0,42 m khong chan duong xe cao 0,12 m. O ban 2D thi
        moi vat can deu la mot buc tuong cao vo han.
        """
        solid = self.e.sc["box_solid"][0]
        zc = self.e.sc["box_c"][0, :, 2]
        zh = self.e.sc["box_h"][0, :, 2]
        high = (zc - zh > P.BODY_HEIGHT) & (zh > 1e-6)
        self.assertGreater(int(high.sum()), 0, "mat bang nao cung phai co vat cao")
        self.assertFalse(bool(solid[high].any()),
                         "vat cao hon than xe thi khong duoc chan duong")

    def test_LIDAR_KHONG_THAY_VAT_THAP_MA_CAMERA_THAY(self):
        """LiDAR quet o 10 cm. Cai tham day 3 cm thi no khong thay."""
        sc, _m = make_batch(1, CPU, seed0=11)
        o = torch.tensor([[[1.0, 1.0, P.LIDAR_HEIGHT]]])
        d = torch.tensor([[[1.0, 0.0, 0.0]]])
        # dat mot tam thap ngay truoc mat
        sc["box_c"] = torch.cat([sc["box_c"],
                                 torch.tensor([[[2.0, 1.0, 0.015]]])], dim=1)
        sc["box_h"] = torch.cat([sc["box_h"],
                                 torch.tensor([[[0.3, 0.3, 0.015]]])], dim=1)
        for k, v in (("box_yaw", torch.zeros(1, 1)),
                     ("box_col", torch.tensor([[[0.9, 0.2, 0.2]]])),
                     ("box_mat", torch.zeros(1, 1, dtype=torch.long)),
                     ("box_code", torch.zeros(1, 1, 3, dtype=torch.long))):
            sc[k] = torch.cat([sc[k], v], dim=1)
        t, _a, _n, _k = RT.trace(o, d, sc, want_floor=False)
        self.assertGreater(float(t[0, 0]), 2.5, "LiDAR khong duoc thay tam thap")
        img = render(sc, torch.tensor([1.0]), torch.tensor([1.0]),
                     torch.zeros(1), torch.tensor([-0.35]))
        self.assertGreater(float(img[0, 0].max() - img[0, 1].max()), 0.03,
                           "camera cui xuong thi phai thay tam mau do")

    def test_roi_xuong_lo_thung(self):
        v = self.e.sc["void"][0, 0]
        self.e.x[0] = 0.5 * (v[0] + v[2])
        self.e.y[0] = 0.5 * (v[1] + v[3])
        self.e.step(torch.zeros(6, 3))
        self.assertTrue(bool(self.e.fallen[0]))

    def test_servo_co_cu_chan_co_hoc(self):
        a = torch.zeros(6, 3)
        a[:, 2] = 1.0
        for _ in range(60):
            self.e.step(a)
        self.assertLessEqual(float(self.e.tilt.max()), P.TILT_MAX + 1e-5)
        a[:, 2] = -1.0
        for _ in range(120):
            self.e.step(a)
        self.assertGreaterEqual(float(self.e.tilt.min()), P.TILT_MIN - 1e-5)

    def test_khong_chui_duoc_vao_tuong(self):
        a = torch.zeros(6, 3)
        a[:, 0] = 1.0
        a[:, 1] = 1.0
        for _ in range(200):
            self.e.step(a)
        bc = self.e.sc["box_c"]
        bh = self.e.sc["box_h"]
        solid = self.e.sc["box_solid"]
        dx = self.e.x[:, None] - bc[..., 0]
        dy = self.e.y[:, None] - bc[..., 1]
        ca, sa = torch.cos(-self.e.sc["box_yaw"]), torch.sin(-self.e.sc["box_yaw"])
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        qx = torch.clamp(lx, -bh[..., 0], bh[..., 0])
        qy = torch.clamp(ly, -bh[..., 1], bh[..., 1])
        d = torch.hypot(lx - qx, ly - qy)
        d = torch.where(solid, d, torch.full_like(d, 9.9))
        self.assertGreater(float(d.min()), P.BODY_RADIUS - 0.02)

    def test_cung_hat_giong_thi_cung_ket_qua(self):
        a = torch.zeros(6, 3)
        a[:, 0] = 0.4
        a[:, 1] = 0.3
        outs = []
        for _ in range(2):
            e = FleetEnv3D(n_envs=6, device=CPU, seed=5)
            for _ in range(10):
                e.step(a)
            outs.append(e.x.clone())
        torch.testing.assert_close(outs[0], outs[1])


class TestLearning(unittest.TestCase):
    def test_mot_vong_ppo_chay_duoc_va_doi_trong_so(self):
        env = FleetEnv3D(n_envs=4, device=CPU, seed=1, max_steps=20)
        m = ActorCritic().to(CPU)
        before = [p.clone() for p in m.parameters()]
        algo = PPO(m, CPU, minibatches=2, epochs=2)
        buf = Buffer(8, 4, (3, P.CAM_H, P.CAM_W), P.N_SCALARS, 3, CPU)
        st = dict(h=m.initial_state(4, CPU), done=torch.zeros(4))
        last_v = collect(env, m, buf, st, CPU)
        stats = algo.update(buf, last_v)
        for k, v in stats.items():
            self.assertTrue(math.isfinite(v), f"{k} khong huu han")
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(before, m.parameters())))

    def test_bo_dem_giu_anh_dang_uint8(self):
        buf = Buffer(4, 4, (3, P.CAM_H, P.CAM_W), P.N_SCALARS, 3, CPU)
        self.assertEqual(buf.img.dtype, torch.uint8)

    def test_GRU_quen_dung_cho_khi_het_tap(self):
        m = ActorCritic().to(CPU)
        img = torch.rand(3, 2, 3, P.CAM_H, P.CAM_W)
        sca = torch.rand(3, 2, P.N_SCALARS)
        h0 = torch.randn(1, 2, m.hidden)
        d = torch.zeros(3, 2)
        d[1, 0] = 1.0
        a, _v, _h = m.forward_seq(img, sca, h0, d)
        b, _v, _h = m.forward_seq(img, sca, h0, torch.zeros(3, 2))
        self.assertFalse(torch.allclose(a[2, 0], b[2, 0]),
                         "ket thuc tap ma khong xoa tri nho GRU")
        torch.testing.assert_close(a[2, 1], b[2, 1])


@unittest.skipUnless(torch.cuda.is_available(), "may nay khong co GPU")
class TestCuda(unittest.TestCase):
    def test_GPU_va_CPU_cho_cung_ket_qua(self):
        dev = torch.device("cuda:0")
        a = torch.zeros(4, 3)
        a[:, 0] = 0.5
        a[:, 1] = 0.35
        e1 = FleetEnv3D(n_envs=4, device=CPU, seed=7)
        e2 = FleetEnv3D(n_envs=4, device=dev, seed=7)
        for _ in range(12):
            e1.step(a)
            e2.step(a.to(dev))
        torch.testing.assert_close(e1.x, e2.x.cpu(), rtol=2e-4, atol=2e-4)
        i1, _s1 = e1.observe()
        i2, _s2 = e2.observe()
        self.assertLess(float((i1 - i2.cpu()).abs().mean()), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
