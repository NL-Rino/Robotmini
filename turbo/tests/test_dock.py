# -*- coding: utf-8 -*-
"""Kiem thu bo do hoc theo lo.

Bo nay la mot THUAT TOAN KHAC ban v1, nen khong the kiem bang cach so tung
so voi ban cu. Kiem bang bon dieu:
  1. Dat xe truoc hoc cua no, duong nhin trong -> phai thay, dung cho, dung truc
  2. Dat xe truoc buc tuong phang -> phai KHONG bao gi
  3. Chay mot xe rieng va chay no trong lo 8 xe -> ket qua y het
  4. Do do chinh xac tren mat bang ngau nhien: khong te hon ban v1 mot cach
     co he thong (sai so vi tri, sai so truc, ti le bao gia)
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
from turbo import dock as TD, sim as TS, world as TW

DEV = torch.device("cpu")


def _clear_shot(w, dock, dist):
    """Diem cach mieng hoc `dist` tren truc, va co thay hoc that khong."""
    from sim.geometry import raycast_segments
    px = dock.x + dist * math.cos(dock.theta)
    py = dock.y + dist * math.sin(dock.theta)
    if not w.on_floor(px, py):
        return None
    # tia tu do ve thanh trong phai khong bi gi chan
    d = float(raycast_segments(px, py, np.array([dock.theta + math.pi]),
                               w.static_segments, P.LIDAR_MAX)[0])
    if d < dist + P.DOCK_CAVITY_D - 0.02:
        return None
    for m in w.movers:
        if math.hypot(m.x - px, m.y - py) < m.radius + 0.6:
            return None
    # Can nha co ban ghe: chan ghe dung giua duong nhin (hay sat mieng hoc)
    # thi khong con la "duong nhin trong" nua.
    ux, uy = -math.cos(dock.theta), -math.sin(dock.theta)
    for cx, cy, _r in w.legs:
        t = (cx - px) * ux + (cy - py) * uy
        if -0.3 < t < dist + 0.3:
            lat = abs(-(cx - px) * uy + (cy - py) * ux)
            if lat < 0.6:
                return None
    return px, py


def _scan(bw, xs, ys, ths, seed=0):
    s = TS.BatchSim(bw, DEV, seed=seed)
    s.place(torch.arange(len(xs)), torch.tensor(xs), torch.tensor(ys),
            torch.tensor(ths))
    for _ in range(4000):
        s.step(torch.zeros(len(xs), 2))
        if s.lidar_step():
            break
    return s


class TestFrontOfDock(unittest.TestCase):
    """Dieu 1: nhin thang vao hoc cua minh thi phai thay."""

    def test_thay_hoc_cua_minh(self):
        found = 0
        tried = 0
        for seed in (3, 5, 8, 11):
            bw = TW.build([seed], 4, DEV, n_docks=3, n_decoys=1)
            bw.freeze_movers()
            w = bw.worlds[0]
            cases = []
            for i in range(4):
                dk = w.docks[int(bw.home[i])]
                got = _clear_shot(w, dk, 0.55 + 0.45 * i)
                cases.append((got, dk))
            xs, ys, ths = [], [], []
            for got, dk in cases:
                if got is None:
                    xs.append(dk.x + 1.0)
                    ys.append(dk.y)
                    ths.append(0.0)
                else:
                    xs.append(got[0])
                    ys.append(got[1])
                    ths.append(dk.theta + math.pi)
            s = _scan(bw, xs, ys, ths)
            out = TD.detect(s.scan_r, s.scan_b, s.scan_ok)
            for i, (got, dk) in enumerate(cases):
                if got is None:
                    continue
                tried += 1
                best = None
                for k in range(out.shape[1]):
                    if out[i, k, 0] <= 0.0:
                        continue
                    r, b = float(out[i, k, 1]), float(out[i, k, 2])
                    gx = xs[i] + r * math.cos(ths[i] + b)
                    gy = ys[i] + r * math.sin(ths[i] + b)
                    e = math.hypot(gx - dk.x, gy - dk.y)
                    if best is None or e < best[0]:
                        best = (e, float(out[i, k, 3]))
                if best is None or best[0] > 0.12:
                    continue
                ax = abs(math.degrees(math.atan2(
                    math.sin(ths[i] + best[1] - dk.theta),
                    math.cos(ths[i] + best[1] - dk.theta))))
                self.assertLess(ax, 15.0, "truc hoc lech qua nhieu")
                found += 1
        self.assertGreater(tried, 8, "khong du truong hop de kiem")
        self.assertGreaterEqual(found, int(0.8 * tried),
                                f"chi thay {found}/{tried} hoc khi nhin thang")


class TestPlainWall(unittest.TestCase):
    """Dieu 2: buc tuong phang khong phai cai hoc."""

    def test_tuong_phang_khong_bao(self):
        n = 400
        b = torch.linspace(-math.pi, math.pi, n + 1)[:n][None, :]
        # tuong thang x = 2 m, quet 360 do; huong ra sau la khoang trong
        r = torch.where(torch.cos(b) > 0.05, 2.0 / torch.cos(b).clamp(min=0.05),
                        torch.full_like(b, P.LIDAR_MAX))
        ok = (r < 6.0)
        r = r + torch.randn(r.shape, generator=torch.Generator().manual_seed(1)) * 0.01
        out = TD.detect(r, b, ok)
        self.assertEqual(float(out[..., 0].max()), 0.0,
                         "bao co hoc giua mot buc tuong phang")


class TestBatchInvariance(unittest.TestCase):
    """Dieu 3: chay rieng hay chay chung mot lo phai ra y het."""

    def test_rieng_bang_chung(self):
        bw = TW.build([3, 5], 4, DEV, n_docks=3, n_decoys=1)
        pose = bw.home_pose()
        xs = (pose[:, 0] + 0.8 * torch.cos(pose[:, 2])).tolist()
        ys = (pose[:, 1] + 0.8 * torch.sin(pose[:, 2])).tolist()
        ths = TS.wrap(pose[:, 2] + math.pi).tolist()
        s = _scan(bw, xs, ys, ths)
        full = TD.detect(s.scan_r, s.scan_b, s.scan_ok)
        for i in (0, 3, 7):
            one = TD.detect(s.scan_r[i:i + 1], s.scan_b[i:i + 1],
                            s.scan_ok[i:i + 1])
            self.assertTrue(torch.allclose(one[0], full[i], atol=1e-5),
                            f"xe {i} ra khac khi chay rieng")


class TestAccuracy(unittest.TestCase):
    """Dieu 4: khong te hon ban v1 mot cach co he thong."""

    def test_do_chinh_xac(self):
        from turbo.tools.measure import _grade, _scan_batch
        bw, s, poses = _scan_batch([3, 5, 8], 40, DEV)
        out = TD.detect(s.scan_r, s.scan_b, s.scan_ok)
        cands = [[tuple(float(x) for x in out[r, k])
                  for k in range(out.shape[1])] for r in range(len(poses))]
        tp, fp, pos, ax = _grade(bw, poses, cands)
        self.assertGreater(tp, 25, "thay duoc qua it hoc")
        self.assertLess(fp / max(1, tp + fp), 0.12, "bao gia qua nhieu")
        self.assertLess(float(np.median(pos)), 0.045,
                        "lech vi tri trung vi qua 4,5 cm")
        self.assertLess(float(np.median(ax)), 8.0,
                        "lech truc trung vi qua 8 do")


if __name__ == "__main__":
    unittest.main(verbosity=2)
