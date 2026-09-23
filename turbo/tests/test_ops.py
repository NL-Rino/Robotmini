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


class TestNoScalarAssign(unittest.TestCase):
    """Khong duoc gan mot SO PYTHON THUAN vao lat cat cua tensor.

    Tren CPU thi `v[:, 5] = 1.0` chay binh thuong. Tren card lien no bao
    `scatter(): Expected self.dtype to be equal to src.dtype`, vi so Python
    thanh mot so 64 bit con tensor la 32 bit. Da vap dung loi nay o buoc
    "dung 64 dau vao". Bai kiem thu nay bat ca LOP loi do, khong phai mot
    dong cu the.
    """

    def test_ca_vong_chay_khong_gan_so_thuan(self):
        real = torch.Tensor.__setitem__
        bad = []

        def spy(self, key, value):
            if isinstance(value, (int, float, bool)):
                import traceback
                where = [l for l in traceback.format_stack()[:-1]
                         if "/turbo/" in l.replace("\\", "/")]
                bad.append(where[-1].strip() if where else "?")
            return real(self, key, value)

        torch.Tensor.__setitem__ = spy
        try:
            from turbo import dock as D, perception as PC
            ro = Rollout([3], 3, 2, DEV_CPU, seed=1)
            ro.reset(11, 0.3)
            pol = BatchPolicy(12, DEV_CPU)
            th = torch.randn(2, pol.n_params) * 0.2
            p = pol.unpack(th)
            h = pol.new_state(2, ro.unit)
            dk = torch.zeros(ro.sim.R, 2, 4)
            for _ in range(30):
                if ro.sim.lidar_step():
                    dk = D.detect(ro.sim.scan_r, ro.sim.scan_b, ro.sim.scan_ok)
                v = PC.build(ro.sim, dk)
                y, h = pol.step(p, v.view(2, ro.unit, -1), h)
                ro.rw.step(ro.sim.step(y.reshape(ro.sim.R, -1)))
        finally:
            torch.Tensor.__setitem__ = real
        self.assertEqual(bad, [], "gan so Python thuan vao tensor:\n  "
                                 + "\n  ".join(dict.fromkeys(bad)))


class TestNoFloat64(unittest.TestCase):
    """Khong duoc de lot mot mang SO THUC 64 BIT nao.

    Card lien khong lam duoc nhieu phep tren 64 bit va no bao loi bang mot
    cau "unknown error" khong chi cho nao ca. Da mat nhieu vong vi dung mot
    dong: `torch.full(..., 8.0, device=...)` khong ghi `dtype` thi tren
    card lien ra 64 bit. Te hon nua, loi do chi hien ra o xe VUA DAT LAI:
    quet xong mot vong la mang do bi thay bang ban 32 bit va loi bien mat.
    """

    def test_trang_thai_deu_32_bit(self):
        ro = Rollout([3, 5], 3, 2, DEV_CPU, seed=1)
        ro.reset(11, 0.3)
        self.assertEqual(ops.dtype_audit(ro.sim), [])
        self.assertEqual(ops.dtype_audit(ro.sim.w), [])
        # NGAY SAU khi dat lai, chua quet vong nao - day la luc de lot nhat
        self.assertEqual(ro.sim.fans().dtype, torch.float32)

    def test_ca_vong_chay_khong_sinh_64_bit(self):
        from turbo import perception as PC

        class Bat(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                xs = out if isinstance(out, (tuple, list)) else (out,)
                for x in xs:
                    if torch.is_tensor(x) and x.dtype == torch.float64:
                        raise AssertionError(
                            f"{getattr(func, '__name__', func)} sinh ra so "
                            f"thuc 64 bit")
                return out

        ro = Rollout([3], 3, 2, DEV_CPU, seed=2)
        pol = BatchPolicy(12, DEV_CPU)
        th = torch.randn(2, pol.n_params, generator=
                         torch.Generator().manual_seed(4)) * 0.2
        with Bat():
            ro.reset(21, 0.4)
            self.assertEqual(PC.build(ro.sim, ro.docks).dtype, torch.float32)
            ro.run(pol, th, 30, collect_obs=False)


class TestDevicePicker(unittest.TestCase):

    def test_luon_co_CPU(self):
        keys = [d["key"] for d in DEV.list_devices()]
        self.assertIn("cpu", keys)
        self.assertEqual(keys[-1], "cpu", "CPU phai o cuoi danh sach")

    def test_resolve(self):
        self.assertEqual(DEV.resolve("cpu").type, "cpu")

    def test_directml_noi_ro_ly_do(self):
        ok, why = DEV.directml_status()
        self.assertIsInstance(ok, bool)
        self.assertTrue(why, "khong co DirectML thi phai noi duoc ly do")
        if not ok:
            self.assertNotEqual(why.strip(), "")

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
