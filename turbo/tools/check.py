# -*- coding: utf-8 -*-
"""Thu xem may cua ban chay duoc ban theo lo den dau.

    python -m turbo.tools.check
    python -m turbo.tools.check --device dml      (card lien Intel/AMD)
    python -m turbo.tools.check --device cuda

Card roi NVIDIA chay duoc gan het moi phep cua PyTorch. Card LIEN (Intel HD
620 chang han) chay qua DirectML va THIEU mot so phep - thieu cai nao thi
thong bao loi cua no thuong khong noi ro. Chay file nay truoc, no se noi
thang cho nao chay duoc, cho nao khong, roi mo phong thu vai buoc that.

Chay xong ma co dong nao mau do thi chup man hinh gui lai - do dung la thu
can biet.
"""

import argparse
import re
import sys
import time
import warnings

import torch

from turbo import device as DEV, ops

def _ok(name, note=""):
    print(f"  [ duoc  ] {name}" + (f"   {note}" if note else ""))


def _fail(name, err):
    """In ca vet goi ham. "unknown error" cua DirectML khong noi gi ca -
    phai biet no chet o DONG NAO thi moi viet duong vong duoc."""
    import traceback
    print(f"  [ KHONG ] {name}")
    print(f"            {type(err).__name__}: {err}")
    print("            --- chet o day ---")
    for ln in traceback.format_exc().strip().splitlines()[-9:]:
        print("            " + ln.rstrip())


def check(dev, robots=48, quiet=False):
    robots = max(3, robots - robots % 3)
    pop = robots // 3
    print(f"\nMay tinh: {dev}")
    bad = 0

    print("\n1. Cac phep de thieu")
    for k, v in ops.caps(dev).items():
        mark = "duoc " if v else "KHONG"
        note = {
            "scatter_reduce": "gom tia vao quat (co duong vong, cham hon 19 lan)",
            "float64": "so thuc 64 bit (khong can, chi de biet)",
            "generator": "bo sinh so rieng (khong can, da sinh tren CPU)",
            "index_copy": "dat lai xe - CAN CO",
            "index_copy_bool": "dat lai xe, kieu dung/sai (co duong vong)",
            "index_fill_bool": "dat lai xe, kieu dung/sai - CAN CO",
            "cumprod": "do hoc sac - CAN CO",
            "bmm": "chay bo nao - CAN CO",
            "atan2": "moi phep goc - CAN CO",
            "gather": "lay diem quanh hoc - CAN CO",
            "sort": "khong dung, chi de biet",
        }.get(k, "")
        print(f"  [ {mark} ] {k:16s} {note}")
        if not v and "CAN CO" in note:
            bad += 1

    print("\n2. Chay thu")
    from turbo import dock as D, perception as PC, sim as TS, world as TW
    from turbo.policy import BatchPolicy
    try:
        bw = TW.build([3], 3, dev, n_docks=3, n_decoys=1, copies=pop)
        s = TS.BatchSim(bw, dev, seed=0, copies=pop)
        p = bw.home_pose()
        # Dat kem MUC PIN: duong dat pin di qua mot phep khac (chep theo
        # hang tren kieu dung/sai) ma duong khong pin khong cham toi.
        s.place(torch.arange(s.R).to(dev),
                p[:, 0] + 0.9 * torch.cos(p[:, 2]),
                p[:, 1] + 0.9 * torch.sin(p[:, 2]), p[:, 2] + 3.14159,
                battery=torch.full((s.R,), 0.5).to(dev))
        _ok("dung the gioi va dat xe", f"{s.R} xe")
    except Exception as e:
        _fail("dung the gioi va dat xe", e)
        return 1

    try:
        for _ in range(40):
            s.step(torch.zeros(s.R, 2, device=dev))
            s.lidar_step()
        _ok("mo phong + LiDAR", "40 buoc")
    except Exception as e:
        _fail("mo phong + LiDAR", e)
        return 1

    try:
        dk = D.detect(s.scan_r, s.scan_b, s.scan_ok)
        n = int((dk[..., 0] > 0).sum())
        _ok("do hoc sac", f"{n}/{s.R} xe thay hoc")
    except Exception as e:
        _fail("do hoc sac", e)
        return 1

    try:
        v = PC.build(s, dk)
        assert v.shape == (s.R, 48)
        _ok("dung 48 dau vao")
    except Exception as e:
        _fail("dung 48 dau vao", e)
        return 1

    try:
        bp = BatchPolicy(16, dev)
        th = (torch.randn(pop, bp.n_params) * 0.2).to(dev)
        pr = bp.unpack(th)
        h = bp.new_state(pop, 3)
        y, h = bp.step(pr, v.view(pop, 3, -1), h)
        assert y.shape == (pop, 3, 2)
        _ok("chay bo nao")
    except Exception as e:
        _fail("chay bo nao", e)
        return 1

    print("\n3. Phep nao dang phai nho CPU tinh ho")
    lag = _fallbacks(s, dev, bp, pr, h, v)
    if lag:
        print("  Card nay thieu cac phep sau, PyTorch dang lang le chep sang")
        print("  CPU tinh roi chep ve - moi buoc mo phong mat mot vong di ve:")
        for name in lag:
            print(f"    - {name}")
        print("  Gui danh sach nay lai, viet duong vong cho chung duoc.")
    else:
        print("  Khong co phep nao phai nho CPU. Tot.")

    print("\n4. Chay mot lan danh gia day du")
    from turbo.policy import BatchPolicy as BP
    from turbo.rollout import Rollout
    ro = bp2 = th2 = None
    try:
        ro = Rollout([3], 3, pop, dev, seed=1)
        _ok("dung lo the gioi", f"{ro.sim.R} xe, {ro.unit} xe moi bo trong so")
    except Exception as e:
        _fail("dung lo the gioi", e)
        return 1
    try:
        ro.reset(11, 0.3)
        _ok("dat lai xe theo giao trinh")
    except Exception as e:
        _fail("dat lai xe theo giao trinh", e)
        return 1
    try:
        bp2 = BP(16, dev)
        th2 = (torch.randn(pop, bp2.n_params) * 0.2).to(dev)
        ro.run(bp2, th2, 1)
        _ok("mot buoc: bo nao + vat ly + phan thuong")
    except Exception as e:
        _fail("mot buoc: bo nao + vat ly + phan thuong", e)
        return 1
    try:
        ro.run(bp2, th2, 20)
        _sync(dev)
        _ok("hai muoi buoc lien")
    except Exception as e:
        _fail("hai muoi buoc lien", e)
        return 1

    print("\n5. Nhanh cham the nao")
    for n in (pop, 4 * pop):
        try:
            r2 = Rollout([3], 3, n, dev, seed=1)
            b2 = BP(16, dev)
            t2 = (torch.randn(n, b2.n_params) * 0.2).to(dev)
            r2.reset(11, 0.3)
            r2.run(b2, t2, 5)
            _sync(dev)
            r2.reset(11, 0.3)
            t = time.perf_counter()
            r2.run(b2, t2, 100)
            _sync(dev)
            dt = time.perf_counter() - t
            print(f"  quan the {n:4d} ({r2.sim.R:4d} xe): "
                  f"{100 * r2.sim.R / dt:9,.0f} buoc-xe/giay")
        except Exception as e:
            _fail(f"quan the {n}", e)
            break

    if bad:
        print(f"\nCon {bad} phep bat buoc khong chay duoc tren may nay.")
        return 1
    print("\nChay duoc het. So o muc 5 cang lon cang tot. So voi ban "
          "'tung xe mot'\nbang lenh:  chay_dml.bat do")
    return 0


def _fallbacks(s, dev, bp, pr, h, v):
    """Chay mot vong day du va nhat het cac loi canh bao "chay nho CPU"."""
    from turbo import dock as D, perception as PC
    names = []
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        try:
            for _ in range(12):
                s.step(torch.zeros(s.R, 2, device=dev))
                if s.lidar_step():
                    dk = D.detect(s.scan_r, s.scan_b, s.scan_ok)
                    PC.build(s, dk)
            y, _h2 = bp.step(pr, v.view(v.shape[0] // 3, 3, -1), h)
            float(y.sum())
        except Exception as e:
            print(f"  (chay thu de nhat canh bao thi loi: "
                  f"{type(e).__name__}: {e})")
    for w in got:
        m = re.search(r"operator '([^']+)'", str(w.message))
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


def _sync(dev):
    """Doi may lam xong. Card chay bat dong bo, khong doi thi do ra so ao."""
    try:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "xpu":
            torch.xpu.synchronize()
        elif dev.type != "cpu":
            # Card lien khong co lenh doi rieng; doc mot so ve CPU thi buoc
            # no phai lam xong het viec dang xep hang.
            float(torch.ones(1, device=dev).add_(1.0).cpu())
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thu may co chay duoc ban theo lo")
    ap.add_argument("--robots", type=int, default=48)
    DEV.add_argument(ap)
    a = ap.parse_args(argv)
    print(f"Python {sys.version.split()[0]}  |  torch {torch.__version__}")
    print("\nCac may tinh tim thay:")
    for d in DEV.list_devices():
        print(f"  {d['key']:10s} {DEV.describe(d)}")
    ok, why = DEV.directml_status()
    if not ok:
        print(f"  (khong co DirectML: {why})")
    dev = DEV.from_args(a)
    return check(dev, robots=a.robots)


if __name__ == "__main__":
    sys.exit(main())
