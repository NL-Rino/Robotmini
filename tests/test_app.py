# -*- coding: utf-8 -*-
"""Kiem thu phan KHONG phai giao dien cua app.py.

Cai de sai nhat o app.py khong phai cac nut bam ma la DONG LENH no dung
de chay huan luyen: sai mot chu la tien trinh con chet ngay, ma no chay o
cua so an nen khong ai thay thong bao loi.

Tkinter duoc gia lap o day, nen bai kiem thu nay chay duoc ca tren may chu
khong co giao dien.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _m in ("tkinter", "tkinter.messagebox", "tkinter.ttk"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
_tk = sys.modules["tkinter"]
for _n in ("Canvas", "Tk", "Frame", "Toplevel", "Label", "Entry", "Text",
           "StringVar", "Button"):
    if not hasattr(_tk, _n):
        setattr(_tk, _n, object)
_tk.messagebox = sys.modules["tkinter.messagebox"]
_tk.ttk = sys.modules["tkinter.ttk"]

import app                                              # noqa: E402

CFG = dict(pop=24, steps=400, robots=3, episodes=2, jobs=4, gens=5000,
           curriculum_gens=600)


def _sau_m(args):
    """Cac tham so, bo phan `python -u -m <module>` o dau."""
    return args[4:]


class TestTrainCommand(unittest.TestCase):

    def test_ban_v1(self):
        a = app.train_command("train.train", None, "runs/x", 16, CFG)
        self.assertEqual(a[:4], [sys.executable, "-u", "-m", "train.train"])
        self.assertIn("--episodes", a)
        self.assertIn("--jobs", a)
        self.assertNotIn("--device", a, "ban v1 khong biet --device")
        self.assertNotIn("--maps", a, "ban v1 khong biet --maps")

    def test_ban_theo_lo(self):
        a = app.train_command("turbo.train", "cpu", "runs/x", 16, CFG)
        self.assertEqual(a[3], "turbo.train")
        self.assertIn("--maps", a)
        self.assertEqual(a[a.index("--device") + 1], "cpu")
        self.assertNotIn("--jobs", a, "ban theo lo khong co tien trinh nao")
        self.assertNotIn("--episodes", a)

    def test_hai_may(self):
        a = app.train_command("turbo.train", "dml:0,cpu", "runs/x", 16, CFG)
        self.assertEqual(a[a.index("--device") + 1], "dml:0,cpu")

    def test_thieu_may_thi_ve_CPU(self):
        a = app.train_command("turbo.train", None, "runs/x", 16, CFG)
        self.assertEqual(a[a.index("--device") + 1], "cpu")

    def test_chay_tiep_va_bat_dau_tu_bo_nao_co_san(self):
        a = app.train_command("train.train", None, "runs/x", 16, CFG,
                              resume=True)
        self.assertEqual(a[a.index("--resume") + 1], "runs/x")
        self.assertNotIn("--init", a, "chay tiep thi khong kem --init")
        b = app.train_command("train.train", None, "runs/x", 16, CFG,
                              init="brains/a.npz")
        self.assertEqual(b[b.index("--init") + 1], "brains/a.npz")
        self.assertNotIn("--resume", b)

    def test_moi_tham_so_deu_co_that(self):
        """Dua dong lenh cho CHINH bo doc tham so cua tung ban.

        Sai mot chu thi tien trinh con chet ngay khi vua chay, ma no chay o
        cua so an nen khong ai thay. Cho `--gens 0` de no khong huan luyen
        gi, chi doc tham so roi thoat.
        """
        import shutil
        import tempfile
        for mod, dev in (("train.train", None), ("turbo.train", "cpu")):
            d = tempfile.mkdtemp()
            try:
                a = app.train_command(mod, dev, d, 12, dict(CFG, gens=0))
                m = __import__(mod, fromlist=["main"])
                try:
                    m.main(a[4:] + ["--quiet"])
                except SystemExit as e:
                    if e.code:
                        self.fail(f"{mod} tu choi dong lenh: {e}")
            finally:
                shutil.rmtree(d, ignore_errors=True)


class TestEngines(unittest.TestCase):

    def test_co_it_nhat_mot_bo_may_chay(self):
        eng = app._engines()
        self.assertGreaterEqual(len(eng), 1)
        for label, mod, key, tip in eng:
            self.assertTrue(label and tip)
            self.assertIn(mod, ("train.train", "turbo.train"))

    def test_muc_dau_dung_duoc_ngay(self):
        """app.py lay muc DAU lam mac dinh, nen no phai chay duoc."""
        label, mod, key, _tip = app._engines()[0]
        a = app.train_command(mod, key, "runs/x", 16, CFG)
        self.assertTrue(os.path.basename(a[0]).startswith("python"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
