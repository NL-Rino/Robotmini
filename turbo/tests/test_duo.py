# -*- coding: utf-8 -*-
"""Kiem thu viec chia quan the cho NHIEU MAY cung chay.

Dieu phai dung, va la dieu duy nhat lam cho viec nay co nghia: chia quan
the ra hai may roi ghep diem lai phai ra DUNG cai ma chay nguyen mot lo ra.

Neu sai thi ES van chay, van "hoc" duoc, nhung no dang xep hang cac ca the
theo mot thang do lech nhau - va khong cho nao bao gi ca.
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

from sim import params as P
from turbo.duo import Duo
from turbo.policy import BatchPolicy
from turbo.rollout import Rollout

DEV = torch.device("cpu")
NORM = (np.zeros(P.N_INPUTS), np.ones(P.N_INPUTS))


class TestSplitMatchesWhole(unittest.TestCase):

    def _theta(self, pop, hidden):
        bp = BatchPolicy(hidden, DEV)
        g = torch.Generator().manual_seed(3)
        return torch.randn(pop, bp.n_params, generator=g) * 0.2

    def test_chia_doi_ra_dung_diem_cua_ca_lo(self):
        pop, hidden, steps = 8, 12, 90
        th = self._theta(pop, hidden)

        ro = Rollout([3, 5], 3, pop, DEV, seed=77)
        ro.reset(77, 0.3)
        bp = BatchPolicy(hidden, DEV)
        bp.set_norm(*NORM)
        mot, _ = ro.run(bp, th.clone(), steps)

        duo = Duo([DEV, DEV], hidden)
        duo.rate = [1.0, 1.0]
        hai, _, _st, share = duo.run([3, 5], 3, pop, 77, 0.3, th.clone(),
                                     steps, NORM)
        self.assertEqual(sum(share), pop)
        self.assertEqual(len(share), 2)
        # Khong con doi hoi trung toi 1e-4 nhu truoc. Lo 8 va lo 4 tinh sin/
        # cos bang duong vector khac nhau o phan duoi mang, nen vi tri xe
        # lech nhau ~1e-5 m - xua nay van vay. Nhung gio co luoi kham pha: mot
        # tia lech 1e-5 m roi sang o ben canh la thanh MOT O (0,3 diem) va
        # mot dau vao "vua thay cho moi" khac nhau. Mot o tren ca lan chay la
        # sai so lam tron, khong phai chia sai; chia sai thi lech ca chuc diem.
        self.assertLess(float((mot.cpu() - hai).abs().max()), 0.5,
                        "chia ra roi ghep lai khong ra dung diem cu")

    def test_chia_lech_van_dung(self):
        """Ti le 7-1 cung phai dung - khong duoc phu thuoc vao chia deu."""
        pop, hidden, steps = 8, 12, 60
        th = self._theta(pop, hidden)
        ro = Rollout([3], 3, pop, DEV, seed=5)
        ro.reset(5, 0.5)
        bp = BatchPolicy(hidden, DEV)
        bp.set_norm(*NORM)
        mot, _ = ro.run(bp, th.clone(), steps)

        duo = Duo([DEV, DEV], hidden)
        duo.rate = [7.0, 1.0]
        hai, _, _st, share = duo.run([3], 3, pop, 5, 0.5, th.clone(),
                                     steps, NORM)
        self.assertEqual(share, [7, 1])
        self.assertLess(float((mot.cpu() - hai).abs().max()), 1e-4)

    def test_quan_sat_gop_lai_dung(self):
        pop, hidden, steps = 8, 12, 40
        th = self._theta(pop, hidden)
        ro = Rollout([3], 3, pop, DEV, seed=9)
        ro.reset(9, 0.2)
        bp = BatchPolicy(hidden, DEV)
        bp.set_norm(*NORM)
        _sc, (s1, s2, n1) = ro.run(bp, th.clone(), steps)

        duo = Duo([DEV, DEV], hidden)
        duo.rate = [1.0, 1.0]
        _h, (t1, t2, n2), _st, _sh = duo.run([3], 3, pop, 9, 0.2, th.clone(),
                                             steps, NORM)
        self.assertEqual(n1, n2)
        self.assertLess(float((s1.double() - t1).abs().max()), 1e-2)
        self.assertLess(float((s2.double() - t2).abs().max()), 1e-2)


class TestSplitRule(unittest.TestCase):

    def test_chia_du_va_khong_ai_bi_bo_doi(self):
        duo = Duo([DEV, DEV, DEV], 12)
        for rate in ([1, 1, 1], [10, 1, 1], [1, 0.001, 5], [None, None, None]):
            duo.rate = list(rate)
            for pop in (4, 8, 16, 64):
                sh = duo.split(pop)
                self.assertEqual(sum(sh), pop, f"{rate} {pop}")
                self.assertTrue(all(x >= 1 for x in sh), f"{rate} {pop}")

    def test_mot_may_thi_lay_het(self):
        duo = Duo([DEV], 12)
        self.assertEqual(duo.split(30), [30])


class TestTrainTwoDevices(unittest.TestCase):

    def test_chay_duoc_va_luu_duoc(self):
        from turbo.train import train
        from train.policy import GRUPolicy
        out = tempfile.mkdtemp()
        try:
            train(out=out, hidden=12, pop=8, maps=1, robots=3, steps=40,
                  gens=2, eval_every=2, device="cpu,cpu", quiet=True)
            d = np.load(os.path.join(out, "state.npz"), allow_pickle=True)
            self.assertEqual(int(d["gen"]), 2)
            pol, _m = GRUPolicy.load(os.path.join(out, "best.npz"))
            self.assertEqual(pol.n_h, 12)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
