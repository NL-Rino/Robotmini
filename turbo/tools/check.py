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
import sys
import time

import torch

from turbo import device as DEV, ops

def _ok(name, note=""):
    print(f"  [ duoc  ] {name}" + (f"   {note}" if note else ""))


def _fail(name, err):
    print(f"  [ KHONG ] {name}")
    print(f"            {type(err).__name__}: {err}")


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
        s.place(torch.arange(s.R).to(dev),
                p[:, 0] + 0.9 * torch.cos(p[:, 2]),
                p[:, 1] + 0.9 * torch.sin(p[:, 2]), p[:, 2] + 3.14159)
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

    print("\n3. Nhanh cham the nao")
    try:
        from turbo.policy import BatchPolicy as BP
        from turbo.rollout import Rollout
        for pop in (16, 64):   # noqa: F402
            ro = Rollout([3], 3, pop, dev, seed=1)
            bp = BP(16, dev)
            th = (torch.randn(pop, bp.n_params) * 0.2).to(dev)
            ro.reset(11, 0.3)
            ro.run(bp, th, 5)
            _sync(dev)
            ro.reset(11, 0.3)
            t = time.perf_counter()
            ro.run(bp, th, 100)
            _sync(dev)
            dt = time.perf_counter() - t
            print(f"  quan the {pop:4d} ({ro.sim.R:4d} xe): "
                  f"{100 * ro.sim.R / dt:9,.0f} buoc-xe/giay")
    except Exception as e:
        _fail("do toc do", e)
        return 1

    if bad:
        print(f"\nCon {bad} phep bat buoc khong chay duoc tren may nay.")
        return 1
    print("\nChay duoc het. So o muc 3 cang lon cang tot; so sanh voi muc "
          "'tung xe mot'\nbang lenh:  python -m turbo.tools.measure speed")
    return 0


def _sync(dev):
    try:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "xpu":
            torch.xpu.synchronize()
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
