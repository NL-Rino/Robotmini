# -*- coding: utf-8 -*-
"""Kiem thu phan huan luyen theo lo.

Ba dieu phai dung, va deu la thu neu sai thi khong co gi bao:

  1. Bo nao chay theo lo phai cho DUNG hanh dong nhu ban v1 voi cung trong
     so va cung dau vao. Sai o day thi bo nao nuoi bang GPU mang ve may se
     cu xu khac han, va khong cho nao keu.
  2. Chung so ngau nhien phai that: hai ca the trong cung mot the he phai
     gap DUNG mot the gioi.
  3. Ba cai hang rao phan thuong cua ban v1 (ngoi li trong hoc, quay nguoi
     trong hoc, cam ra cam vao an thuong) phai con nguyen o ban nay.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from train.policy import GRUPolicy
from train.reward import CHARGE_LATCH, LOITER_CAP
from turbo import sim as TS, world as TW
from turbo.policy import BatchPolicy
from turbo.reward import BatchReward
from turbo.rollout import Rollout

DEV = torch.device("cpu")


class TestPolicyMatchesV1(unittest.TestCase):

    def test_cung_trong_so_cung_hanh_dong(self):
        H = 24
        v1 = GRUPolicy(n_hidden=H, seed=11)
        rng = np.random.default_rng(2)
        # bo chuan hoa KHONG tam thuong - day la cho de sai nhat
        v1.norm.load(rng.normal(0, 0.5, 48), rng.uniform(0.2, 3.0, 48), 1e5)

        bp = BatchPolicy(H, DEV)
        bp.set_norm(*v1.norm.state()[:2])
        th = torch.tensor(v1.theta, dtype=torch.float32)[None, :]
        p = bp.unpack(th)
        h = bp.new_state(1, 1)
        hv = v1.new_state()
        worst = 0.0
        for _ in range(40):
            o = rng.normal(0.0, 1.0, 48)
            y2, h = bp.step(p, torch.tensor(o, dtype=torch.float32)[None, None],
                            h)
            y1, hv = v1.step(o, hv)
            worst = max(worst, float(np.max(np.abs(y2[0, 0].numpy() - y1))))
        self.assertLess(worst, 2e-5, f"lech hanh dong {worst:.2e}")

    def test_khuon_tham_so_khop(self):
        for H in (12, 16, 24, 32):
            self.assertEqual(BatchPolicy(H, DEV).n_params,
                             GRUPolicy(n_hidden=H).n_params)


class TestCommonRandomNumbers(unittest.TestCase):

    def test_moi_ban_sao_gap_cung_the_gioi(self):
        ro = Rollout([5], 3, 4, DEV, seed=3)
        ro.reset(seed=21, progress=0.4)
        s = ro.sim
        U = ro.unit
        for _ in range(60):
            s.step(torch.zeros(s.R, 2))
            s.lidar_step()
        for name in ("x", "y", "th", "batt", "scan_r"):
            a = getattr(s, name)
            base = a[:U]
            for c in range(1, 4):
                self.assertTrue(
                    torch.allclose(a[c * U:(c + 1) * U], base, atol=1e-6),
                    f"'{name}': ban sao {c} khac ban sao 0 - chung so ngau "
                    f"nhien khong con dung")

    def test_cung_trong_so_cung_diem(self):
        ro = Rollout([5, 8], 3, 4, DEV, seed=3)
        ro.reset(seed=21, progress=0.2)
        bp = BatchPolicy(16, DEV)
        one = torch.randn(1, bp.n_params) * 0.2
        th = one.repeat(4, 1)
        sc, _ = ro.run(bp, th, 90)
        self.assertLess(float(sc.max() - sc.min()), 1e-3,
                        "cung trong so ma diem khac nhau")


class TestRewardBarriers(unittest.TestCase):
    """Ba hang rao khien bo nao khong an gian duoc."""

    def _in_dock(self, batt=1.0, n=1):
        bw = TW.build([3], 1, DEV, n_docks=3, n_decoys=1)
        bw.freeze_movers()
        s = TS.BatchSim(bw, DEV, seed=0)
        x, y, th = TW.spawn_in_dock(bw.home_pose())
        s.place(torch.arange(1), x, y, th, battery=torch.tensor([batt]))
        ins, idok, _ = s.contact()
        s.in_slot, s.id_ok, s.charging = ins, idok, idok
        return s, BatchReward(s)

    def test_ngoi_li_trong_hoc_thi_lo_von(self):
        s, rw = self._in_dock(batt=1.0)
        self.assertTrue(bool(s.charging[0]), "chua cam duoc vao hoc")
        for _ in range(1200):
            rw.step(s.step(torch.zeros(1, 2)))
        self.assertLess(float(rw.total[0]), -50.0,
                        "nam trong hoc pin day ma van co lai")
        # va khoan lo phai CO TRAN: lo hon chet thi xe se hoc cach lao xuong vuc
        self.assertGreaterEqual(float(rw.loiter_paid[0]), LOITER_CAP - 1e-3)
        self.assertGreater(float(rw.total[0]), -150.0,
                           "nam li trong hoc dat hon chet - dung cai bay cu")

    def test_quay_nguoi_trong_hoc_bi_phat(self):
        s, rw = self._in_dock(batt=0.5)
        spin = torch.tensor([[-1.0, 1.0]])
        for _ in range(60):
            rw.step(s.step(spin))
        self.assertGreater(float(rw.spin_in_dock[0]), 1.0,
                           "quay tit trong hoc ma khong bi phat gi")

    def test_cam_ra_cam_vao_khong_duoc_tra_hai_lan(self):
        s, rw = self._in_dock(batt=0.5)
        got = 0.0
        for i in range(600):
            # lui ra mot doan roi cam lai, van quanh quan trong hoc
            a = 0.8 if (i // 40) % 2 == 0 else -0.8
            r = rw.step(s.step(torch.tensor([[a, a]])))
            got += float(r)
        self.assertLess(float(rw.total[0]), CHARGE_LATCH,
                        "lac ra lac vao van an duoc thuong cam hoc")


class TestTrainLoop(unittest.TestCase):

    def test_chay_luu_va_chay_tiep(self):
        from turbo.train import train
        out = tempfile.mkdtemp()
        try:
            train(out=out, hidden=12, pop=8, maps=1, robots=3, steps=60,
                  gens=2, eval_every=2, device=DEV, quiet=True)
            st = os.path.join(out, "state.npz")
            self.assertTrue(os.path.exists(st))
            d = np.load(st, allow_pickle=True)
            self.assertEqual(int(d["gen"]), 2)
            # file phai mo duoc bang ban v1 - do la diem cua ca viec nay
            pol, meta = GRUPolicy.load(os.path.join(out, "best.npz"))
            self.assertEqual(pol.n_h, 12)
            y, _h = pol.step(np.zeros(48), pol.new_state())
            self.assertEqual(y.shape, (2,))

            train(out=out, hidden=12, pop=8, maps=1, robots=3, steps=60,
                  gens=2, eval_every=2, resume=out, device=DEV, quiet=True)
            d = np.load(st, allow_pickle=True)
            self.assertEqual(int(d["gen"]), 4, "chay tiep khong noi duoc so")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_file_STOP_dung_an_toan(self):
        """Nut Dung cua app.py chinh la file nay. Phai dung SAU khi the he
        dang chay ghi xong, chu khong phai bo do giua chung."""
        from turbo import train as T
        out = tempfile.mkdtemp()
        real = T._status
        try:
            def spy(path, rec):
                real(path, rec)
                if rec["gen"] == 2:
                    open(os.path.join(out, "STOP"), "w").close()
            T._status = spy
            T.train(out=out, hidden=12, pop=8, maps=1, robots=3, steps=40,
                    gens=9, eval_every=99, device=DEV, quiet=True)
            d = np.load(os.path.join(out, "state.npz"), allow_pickle=True)
            self.assertEqual(int(d["gen"]), 2, "thay STOP ma van chay tiep")
        finally:
            T._status = real
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
