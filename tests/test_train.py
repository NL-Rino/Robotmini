"""Kiem thu phan huan luyen.

Trong tam: nhung thu ma neu hong thi khong ai bao gi ca, chi la hang nghin
the he troi qua vo ich.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from train import train as T
from sim import params as P
from train.es import AdamW, centered_ranks, gradient, noise
from train.policy import GRUPolicy, ObsNorm, PolicyBrain
from train.rollout import PHASES, phase_mix, rollout


class TestPolicy(unittest.TestCase):
    def test_kich_thuoc_va_tinh_xac_dinh(self):
        p = GRUPolicy(n_hidden=16, seed=3)
        # 3 cong x (16x64 + 16x16 + 16) + dau ra 2x16 + 2
        self.assertEqual(p.n_params, 3922)
        self.assertEqual(p.theta.shape, (p.n_params,))
        o = np.linspace(0, 1, P.N_INPUTS)
        h = p.new_state()
        a1, h1 = p.step(o, h)
        a2, h2 = p.step(o, p.new_state())
        np.testing.assert_allclose(a1, a2)
        self.assertFalse(np.allclose(h1, h))

    def test_dau_ra_luon_trong_khoang_ga_hop_le(self):
        p = GRUPolicy(n_hidden=16, seed=1)
        p.set_theta(p.theta * 50.0)          # ep tanh bao hoa
        h = p.new_state()
        for _ in range(20):
            y, h = p.step(np.random.rand(P.N_INPUTS), h)
            self.assertTrue(np.all(np.abs(y) <= 1.0))

    def test_ghi_va_doc_lai_khong_sai_lech(self):
        p = GRUPolicy(n_hidden=12, seed=5)
        p.norm.update(np.full(P.N_INPUTS, 0.3), np.full(P.N_INPUTS, 0.04), 1000)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npz")
            p.save(path, meta=dict(gen=42))
            q, meta = GRUPolicy.load(path)
        np.testing.assert_allclose(p.theta, q.theta)
        np.testing.assert_allclose(p.norm.mean, q.norm.mean)
        self.assertEqual(int(meta["gen"]), 42)
        o = np.random.rand(P.N_INPUTS)
        np.testing.assert_allclose(p.step(o, p.new_state())[0],
                                   q.step(o, q.new_state())[0])

    def test_chuan_hoa_dau_vao_dung_trung_binh_va_phuong_sai(self):
        n = ObsNorm(3)
        data = np.random.default_rng(0).normal(5.0, 2.0, (4000, 3))
        for chunk in np.array_split(data, 8):
            n.update(chunk.mean(axis=0), chunk.var(axis=0), chunk.shape[0])
        np.testing.assert_allclose(n.mean, data.mean(axis=0), atol=1e-8)
        np.testing.assert_allclose(n.var, data.var(axis=0), atol=1e-6)
        z = n.apply(data)
        self.assertLess(abs(float(z.mean())), 0.01)
        self.assertLess(abs(float(z.std()) - 1.0), 0.02)


class TestES(unittest.TestCase):
    def test_xep_hang_giua(self):
        r = centered_ranks([10.0, 20.0, 30.0])
        np.testing.assert_allclose(r, [-0.5, 0.0, 0.5])
        self.assertAlmostEqual(float(np.mean(r)), 0.0)

    def test_ca_quan_the_bang_nhau_thi_KHONG_sinh_gradient(self):
        """Loi (l) cua ban cu: cho nay tung sinh gradient tu hu khong."""
        np.testing.assert_allclose(centered_ranks([7.0] * 8), np.zeros(8))
        g = gradient(50, 1, [3.0] * 6, [3.0] * 6, 0.05)
        np.testing.assert_allclose(g, np.zeros(50))

    def test_adamw_tach_rieng_phan_suy_giam(self):
        """Gradient bang 0 thi trong so phai co lai, khong duoc dung yen.

        Adam + weight_decay cong vao gradient thi phan suy giam bi chia cho
        sqrt(v) va mat tac dung - dung cho lam trong so phinh den muc tanh
        bao hoa va bo nao tra ve (+1,+1) voi moi dau vao.
        """
        th = np.ones(10)
        opt = AdamW(10, lr=0.1, weight_decay=0.5)
        for _ in range(5):
            th = opt.step(th, np.zeros(10))
        self.assertLess(float(np.max(np.abs(th))), 1.0)

    def test_nhieu_tai_tao_lai_duoc_tu_hat_giong(self):
        a = noise(100, 7, 3)
        b = noise(100, 7, 3)
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, noise(100, 7, 4)))

    def test_es_giai_duoc_bai_toan_do_choi(self):
        n = 30
        target = np.linspace(-1, 1, n)
        th = np.zeros(n)
        opt = AdamW(n, lr=0.05, weight_decay=0.0)
        sigma = 0.1
        for gen in range(200):
            sp, sm = [], []
            for i in range(16):
                e = noise(n, gen, i)
                sp.append(-float(np.sum((th + sigma * e - target) ** 2)))
                sm.append(-float(np.sum((th - sigma * e - target) ** 2)))
            th = opt.step(th, gradient(n, gen, sp, sm, sigma))
        self.assertLess(float(np.abs(th - target).mean()), 0.05)


class TestRollout(unittest.TestCase):
    def test_cung_hat_giong_thi_cung_diem(self):
        """Day la dieu kien de CHUNG SO NGAU NHIEN co y nghia: neu cung mot
        bo nao tren cung hat giong ma ra hai diem khac nhau thi khong the so
        sanh hai ca the voi nhau duoc."""
        p = GRUPolicy(n_hidden=12, seed=2)
        a = rollout(p, 123, steps=120, n_robots=2, collect_obs=False)
        b = rollout(p, 123, steps=120, n_robots=2, collect_obs=False)
        self.assertAlmostEqual(a.score, b.score, places=9)

    def test_doi_hat_giong_thi_doi_diem(self):
        p = GRUPolicy(n_hidden=12, seed=2)
        a = rollout(p, 1, steps=120, n_robots=2, collect_obs=False)
        b = rollout(p, 2, steps=120, n_robots=2, collect_obs=False)
        self.assertNotAlmostEqual(a.score, b.score, places=3)

    def test_doi_trong_so_thi_doi_diem(self):
        p = GRUPolicy(n_hidden=12, seed=2)
        a = rollout(p, 5, steps=120, n_robots=2, collect_obs=False)
        p.set_theta(p.theta + 0.3)
        b = rollout(p, 5, steps=120, n_robots=2, collect_obs=False)
        self.assertNotAlmostEqual(a.score, b.score, places=3)

    def test_giao_trinh_chay_tu_kho_ve_de(self):
        early = dict(zip(PHASES, phase_mix(0.0)))
        late = dict(zip(PHASES, phase_mix(1.0)))
        self.assertGreater(early["sap-cam"], late["sap-cam"])
        self.assertLess(early["trong-hoc"], late["trong-hoc"])
        for mix in (phase_mix(0.0), phase_mix(0.5), phase_mix(1.0)):
            self.assertAlmostEqual(sum(mix), 1.0, places=6)

    def test_khong_dat_hai_xe_chong_len_nhau(self):
        """Mot cai hoc chi cho mot xe.

        Truoc day xe nay duoc dat vao hoc cua no con xe kia duoc dat ngau
        nhien vao dung cai hoc do: hai than xe chong len nhau, bo giai va
        cham day nhau ra, va ca hai bat dau lan danh gia bang mot cu va
        vao vach - diem thap vi mot loi dat xe, khong phai vi bo nao do.
        """
        import math
        import random as _r
        from sim import params as _P
        from sim.fleet import FleetSim as _F
        from sim.world import make_fleet_map as _mk
        from train.rollout import _place
        for seed in range(40):
            sim = _F(_mk(seed, n_docks=3, n_decoys=1), n_robots=3, seed=seed)
            _place(sim, _r.Random(seed * 7919 + 13), phase_mix(0.0), 2.5)
            for i, a in enumerate(sim.robots):
                for b in sim.robots[i + 1:]:
                    self.assertGreaterEqual(
                        math.hypot(a.x - b.x, a.y - b.y),
                        2 * _P.BODY_RADIUS - 0.01,
                        f"hai xe chong len nhau o mat bang {seed}")

    def test_thu_thap_duoc_thong_ke_dau_vao(self):
        p = GRUPolicy(n_hidden=12, seed=2)
        r = rollout(p, 9, steps=60, n_robots=2, collect_obs=True)
        self.assertEqual(r.obs_n, 60 * 2)
        mean = r.obs_sum / r.obs_n
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.abs(mean) <= 1.0 + 1e-6))


class TestTrainLoop(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _tiny(self, **kw):
        args = dict(out=self.d, hidden=12, pop=4, steps=60, robots=2,
                    episodes=1, gens=3, jobs=0, eval_every=2, quiet=True,
                    curriculum_gens=10)
        args.update(kw)
        return T.train(**args)

    @staticmethod
    def _recs(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_chay_va_ghi_du_file(self):
        self._tiny()
        self.assertTrue(os.path.exists(os.path.join(self.d, "state.npz")))
        recs = self._recs(os.path.join(self.d, "status.jsonl"))
        self.assertEqual([r["gen"] for r in recs], [1, 2, 3])
        for r in recs:
            for k in ("score", "secs", "sat", "progress", "charged"):
                self.assertIn(k, r)

    def test_chay_tiep_dung_cho_da_dung(self):
        self._tiny()
        self._tiny(resume=self.d, gens=2)
        recs = self._recs(os.path.join(self.d, "status.jsonl"))
        self.assertEqual([r["gen"] for r in recs], [1, 2, 3, 4, 5])
        d = np.load(os.path.join(self.d, "state.npz"), allow_pickle=True)
        self.assertEqual(int(d["gen"]), 5)

    def test_file_STOP_dung_vong_lap_va_van_luu(self):
        """Dat file STOP TRONG LUC dang chay thi vong lap phai dung lai.

        File STOP co san tu truoc luc bat dau thi bi xoa di - neu khong thi
        mot lan dung an toan hom qua se chan mat lan chay hom nay.
        """
        import threading
        status = os.path.join(self.d, "status.jsonl")
        stop = os.path.join(self.d, "STOP")
        th = threading.Thread(target=self._tiny, kwargs=dict(gens=500),
                              daemon=True)
        th.start()
        t0 = time.time()
        while time.time() - t0 < 60:
            if os.path.exists(status) and len(self._recs(status)) >= 2:
                break
            time.sleep(0.05)
        n_before = len(self._recs(status))
        open(stop, "w").close()
        th.join(timeout=60)
        self.assertFalse(th.is_alive(), "STOP phai dung duoc vong lap")
        recs = self._recs(status)
        self.assertLess(len(recs), 60, "phai dung som chu khong chay het 500")
        self.assertGreaterEqual(len(recs), n_before)
        d = np.load(os.path.join(self.d, "state.npz"), allow_pickle=True)
        self.assertEqual(int(d["gen"]), recs[-1]["gen"],
                         "phai luu dung the he DA CHAY XONG")

    def test_bat_dau_tu_mot_bo_nao_co_san(self):
        src = os.path.join(self.d, "goc.npz")
        p = GRUPolicy(n_hidden=12, seed=9)
        p.save(src, meta=dict(gen=0))
        out = os.path.join(self.d, "run")
        T.train(out=out, hidden=12, pop=4, steps=40, robots=2, episodes=1,
                gens=1, jobs=0, eval_every=99, quiet=True, init=src)
        self.assertTrue(os.path.exists(os.path.join(out, "state.npz")))

    def test_khong_khop_so_no_thi_bao_loi_ngay(self):
        src = os.path.join(self.d, "goc.npz")
        GRUPolicy(n_hidden=12, seed=9).save(src, meta=dict(gen=0))
        with self.assertRaises(SystemExit):
            T.train(out=os.path.join(self.d, "r2"), hidden=24, pop=4, steps=20,
                    robots=2, episodes=1, gens=1, jobs=0, quiet=True, init=src)


class TestPolicyBrain(unittest.TestCase):
    def test_vua_khuon_bo_nao_cua_mo_phong(self):
        from sim.fleet import FleetSim, LocalBrains
        p = GRUPolicy(n_hidden=12, seed=4)
        sim = FleetSim(seed=3, n_robots=2)
        lb = LocalBrains(lambda rid: PolicyBrain(p, rid), sim.robot_ids)
        rep = sim.run(lb, max_seconds=5.0)
        self.assertGreater(rep.steps, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
