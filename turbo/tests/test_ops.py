# -*- coding: utf-8 -*-
"""Kiem thu lop cho may yeu.

Card lien (Intel HD 620 qua DirectML) thieu mot so phep ma card roi co.
`turbo/ops.py` viet lai nhung phep do. Bai kiem thu o day chung minh hai
dieu:

  1. Cach viet moi cho ra DUNG ket qua cua cach cu.
  2. Duong vong (dung khi may thieu phep) cho ra dung ket qua cua duong
     nhanh - ke ca khi chay ca mot lan danh gia day du.

Khong co may nao thieu phep o cho dang chay bai kiem thu nay, nen duong
vong duoc BAT EP bang cach bao voi `ops` la may khong co phep do.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from turbo import device as DEV, ops
from turbo.policy import BatchPolicy
from turbo.rollout import Rollout

DEV_CPU = torch.device("cpu")


class _NoFastPath:
    """Gia vo la may nay khong co `scatter_reduce`."""

    def __init__(self, *names):
        self.names = names

    def __enter__(self):
        self.old = dict(ops._CAPS)
        c = dict(ops.caps(DEV_CPU))
        for n in self.names:
            c[n] = False
        ops._CAPS["cpu"] = c
        return self

    def __exit__(self, *a):
        ops._CAPS.clear()
        ops._CAPS.update(self.old)


class TestEquivalent(unittest.TestCase):
    """Cach viet moi = cach viet cu, tung so mot."""

    def test_hypot(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randn(64, 32, generator=g) * 5
        b = torch.randn(64, 32, generator=g) * 5
        self.assertTrue(torch.allclose(ops.hypot(a, b), torch.hypot(a, b),
                                       atol=1e-6))

    def test_cum_and(self):
        g = torch.Generator().manual_seed(2)
        x = torch.rand(16, 9, 40, generator=g) > 0.35
        want = torch.cummin(x.to(torch.uint8), dim=1).values
        got = ops.cum_and(x, dim=1)
        self.assertTrue(torch.equal(got.to(torch.uint8), want))
        # tong cua no phai la SO BE RONG lien tiep con dung
        self.assertTrue(torch.equal(got.sum(1).long(), want.sum(1).long()))

    def test_min_window(self):
        g = torch.Generator().manual_seed(3)
        x = torch.rand(8, 120, generator=g) * 8
        for half in (1, 5, 40):
            xp = torch.cat((x[:, -half:], x, x[:, :half]), 1)[:, None, :]
            want = -F.max_pool1d(-xp, 2 * half + 1, stride=1)[:, 0, :]
            self.assertTrue(torch.allclose(ops.min_window_circ(x, half), want,
                                           atol=1e-6), f"half={half}")

    def test_one_hot_at(self):
        m = torch.tensor([[True, True, False], [False, False, False]])
        col = torch.tensor([1, 0])
        got = ops.one_hot_at(m, col)
        self.assertTrue(torch.equal(
            got, torch.tensor([[False, True, False], [False, False, False]])))

    def test_host_rng_khong_phu_thuoc_may(self):
        a = ops.HostRng(7, DEV_CPU).randn(3, 5)
        b = ops.HostRng(7, DEV_CPU).randn(3, 5)
        self.assertTrue(torch.equal(a, b), "cung hat giong ra khac nhau")


class TestFallback(unittest.TestCase):
    """Duong vong cho ket qua y het duong nhanh."""

    def _sim(self):
        ro = Rollout([3, 5], 3, 2, DEV_CPU, seed=1)
        ro.reset(11, 0.3)
        for _ in range(12):
            ro.sim.step(torch.zeros(ro.sim.R, 2))
            ro.sim.lidar_step()
        return ro

    def test_gom_tia_vao_quat(self):
        ro = self._sim()
        fast = ro.sim.fans()
        with _NoFastPath("scatter_reduce"):
            slow = ro.sim.fans()
        self.assertTrue(torch.allclose(fast, slow, atol=1e-6))

    def test_ca_mot_lan_danh_gia(self):
        pol = BatchPolicy(16, DEV_CPU)
        th = (torch.randn(4, pol.n_params, generator=
                          torch.Generator().manual_seed(5)) * 0.2)

        def run():
            ro = Rollout([3], 3, 4, DEV_CPU, seed=2)
            ro.reset(33, 0.5)
            sc, _ = ro.run(pol, th, 80, collect_obs=False)
            return sc

        fast = run()
        with _NoFastPath("scatter_reduce"):
            slow = run()
        self.assertTrue(torch.allclose(fast, slow, atol=1e-4),
                        f"duong vong ra diem khac: {fast} vs {slow}")


class TestDevicePicker(unittest.TestCase):

    def test_luon_co_CPU(self):
        keys = [d["key"] for d in DEV.list_devices()]
        self.assertIn("cpu", keys)
        self.assertEqual(keys[-1], "cpu", "CPU phai o cuoi danh sach")

    def test_resolve(self):
        self.assertEqual(DEV.resolve("cpu").type, "cpu")

    def test_engines(self):
        eng = DEV.engines()
        self.assertGreaterEqual(len(eng), 2)
        for label, mod, key, tip in eng:
            self.assertIn(mod, ("train.train", "turbo.train"))
            self.assertTrue(label and tip)
            if mod == "turbo.train":
                self.assertTrue(key)
        # May khong co card thi "tung xe mot" phai dung dau - do la cai
        # nhanh hon o quan the nho, va app.py lay muc dau lam mac dinh.
        if not any(k and k != "cpu" for _l, _m, k, _t in eng):
            self.assertEqual(eng[0][1], "train.train")


if __name__ == "__main__":
    unittest.main(verbosity=2)
