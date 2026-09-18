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
import gc
import re
import sys
import time
import warnings

import torch

from sim import params as P
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


def check(dev, robots=48, quiet=False, only=None, dev_key=None,
          retry=True):
    robots = max(3, robots - robots % 3)
    pop = robots // 3
    run = set(only) if only else {1, 2, 3, 4, 5, 6}
    print(f"\nMay tinh: {dev}   (gom tia: {ops.FAN_MODE})")
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

    from turbo import dock as D, perception as PC, sim as TS, world as TW
    from turbo.policy import BatchPolicy
    bw = s = dk = v = bp = pr = h = y = None
    if 2 not in run:
        print("\n2. Chay thu - bo qua")
        return _four(dev, pop, PC, bad, run, dev_key, retry)

    print("\n2. Chay thu")
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
        _touch(s.x)
        _sync(dev)
        _ok("dung the gioi va dat xe", f"{s.R} xe")
    except Exception as e:
        _fail("dung the gioi va dat xe", e)
        return 1

    # Gom tia vao quat khi CHUA CO VONG QUET NAO: day la trang thai xe vua
    # bat len, va no khac han trang thai sau khi da quet - dung du lieu
    # khac thi card co the hong o cho khac.
    _, good = _stage("gom tia vao quat (chua quet lan nao)",
                     lambda: s.fans(), dev)
    if not good:
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

    if 3 not in run:
        print("\n3. Phep nao dang phai nho CPU - bo qua")
        return _four(dev, pop, PC, bad, run, dev_key, retry)

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

    # Tra lai bo nho cua muc 2 va 3 truoc khi sang muc 4. Card lien co
    # kho tai nguyen nho hon card roi nhieu; giu lai het ca lo the gioi cu
    # roi dung them mot lo nua la mot cach chac chan de no het cho.
    bw = s = dk = v = bp = pr = h = y = None
    gc.collect()
    return _four(dev, pop, PC, bad, run, dev_key, retry)


def _four(dev, pop, PC, bad, run, dev_key=None, retry=True):
    print("\n4. Chay mot lan danh gia day du")
    print("   (moi buoc deu DOI cho card lam xong roi moi bao, de loi no")
    print("    dung cho gay ra chu khong no o cho sau)")
    from turbo.policy import BatchPolicy as BP
    from turbo.rollout import Rollout

    ro, good = _stage("dung lo the gioi",
                      lambda: Rollout([3], 3, pop, dev, seed=1), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("dat lai xe theo giao trinh",
                     lambda: ro.reset(11, 0.3), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("ban tia mot buoc", lambda: ro.sim.lidar_step(), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("gom tia vao quat", lambda: ro.sim.fans(), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("do vuc", lambda: ro.sim.cliff(), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("tiep dien", lambda: ro.sim.contact(), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("hong ngoai", lambda: ro.sim.ir_dock(), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    obs, good = _stage("dung 48 dau vao",
                       lambda: PC.build(ro.sim, ro.docks), dev)
    if not good:
        _deep_build(ro.sim, ro.docks, dev)
        return _thu_cach_khac(dev_key, retry, pop)
    bp2 = BP(16, dev)
    th2 = (torch.randn(pop, bp2.n_params) * 0.2).to(dev)
    st = bp2.new_state(pop, ro.unit)
    y, good = _stage("chay bo nao",
                     lambda: bp2.step(bp2.unpack(th2),
                                      obs.view(pop, ro.unit, -1), st), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    ev, good = _stage("mot buoc vat ly",
                      lambda: ro.sim.step(y[0].reshape(ro.sim.R, -1)), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("cham diem", lambda: ro.rw.step(ev), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)
    _, good = _stage("hai muoi buoc lien",
                     lambda: ro.run(bp2, th2, 20), dev)
    if not good:
        return _thu_cach_khac(dev_key, retry, pop)

    if 5 in run:
        print("\n5. Nhanh cham the nao")
    for n in ((pop, 4 * pop) if 5 in run else ()):
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

    if 6 in run:
        print("\n6. Giu duoc bao lau")
        print("   (huan luyen that tao mot lo the gioi MOI moi the he; neu"
              "\n    may het cho sau vai lan thi phai biet truoc)")
        from turbo.policy import BatchPolicy as BP3
        from turbo.rollout import Rollout as RO3
        n_ok = 0
        try:
            for g in range(12):
                r3 = RO3([3 + g], 3, pop, dev, seed=g)
                b3 = BP3(16, dev)
                t3 = (torch.randn(pop, b3.n_params) * 0.2).to(dev)
                r3.reset(g * 7 + 1, 0.3)
                r3.run(b3, t3, 12)
                _sync(dev)
                del r3, b3, t3
                gc.collect()
                n_ok = g + 1
            print("  [ duoc  ] 12 the he lien tiep, khong het cho")
        except Exception as e:
            _fail(f"the he thu {n_ok + 1} (da qua {n_ok} the he)", e)
            print("  -> may het cho sau vai the he. Giam quan the xuong,"
                  "\n     hoac bao lai so the he chay duoc.")
            return 1

    if bad:
        print(f"\nCon {bad} phep bat buoc khong chay duoc tren may nay.")
        return 1
    print("\nChay duoc het. So o muc 5 cang lon cang tot. So voi ban "
          "'tung xe mot'\nbang lenh:  chay_dml.bat do")
    return 0


def _deep_build(sim, docks, dev):
    """Tim cho card chet trong "dung 48 dau vao", theo hai buoc.

    Den day thi da biet: khong phai het tai nguyen (chay rieng cung hong),
    khong phai cach gom tia (ca ba cach deu hong). Con lai la mot phep cu
    the, va viec con lai la chi dung ten no.

    Buoc mot: chay chinh ham that, nhung chen mot cai moc sau moi khoi -
    moc do doi cho card lam xong roi in ten khoi. Khoi cuoi cung IN RA la
    khoi cuoi cung chay duoc; cho hong nam ngay sau no.

    Buoc hai: neu hong o khoi dau (quat LiDAR), thu vai cach viet khac cua
    chinh cho do, de biet nen doi sang cach nao.
    """
    from turbo import perception as PC2

    print("\n   Di tung khoi mot trong 'dung 48 dau vao':")
    xong = []

    def hook(ten, x):
        _touch(x)
        _sync(dev)
        lt = "" if (not torch.is_tensor(x) or x.is_contiguous()) \
            else "  KHONG LIEN TUC"
        print(f"     [ duoc  ] {ten}{lt}")
        xong.append(ten)

    PC2.STEP_HOOK = hook
    try:
        PC2.build(sim, docks)
        print("     (lan nay ca ham lai chay duoc - loi khong on dinh)")
        return
    except Exception as e:
        sau = xong[-1] if xong else "truoc khoi dau tien"
        print(f"     [ KHONG ] khoi NGAY SAU '{sau}'")
        print(f"               {type(e).__name__}: {e}")
    finally:
        PC2.STEP_HOOK = None

    if xong:
        return

    # Hong ngay o khoi dau: cat nho ra nua.
    print("\n   Khoi dau la 'quat LiDAR'. Cat nho ra:")

    def step(name, fn):
        try:
            r = fn()
            _touch(r)
            _sync(dev)
        except Exception as e:
            print(f"     [ KHONG ] {name}")
            print(f"               {type(e).__name__}: {e}")
            return None, False
        if torch.is_tensor(r):
            lt = "" if r.is_contiguous() else "  KHONG LIEN TUC"
            print(f"     [ duoc  ] {name:32s} {str(tuple(r.shape)):12s}"
                  f" {str(r.dtype).replace('torch.', '')}{lt}")
        else:
            print(f"     [ duoc  ] {name}")
        return r, True

    v, ok = step("tao mang 48 dau vao",
                 lambda: torch.zeros(sim.R, 48, device=dev))
    if not ok:
        return
    f, ok = step("sim.fans()", lambda: sim.fans())
    if not ok:
        return

    c, ok = step("clamp(0, 8)", lambda: f.clamp(0.0, P.LIDAR_MAX))
    if not ok:
        print("       -> thu cac cach viet khac cua chinh phep nay:")
        fc, ok2 = step("  chep ra ban lien tuc",
                       lambda: f.contiguous().clone())
        if ok2:
            c, ok = step("  clamp tren ban chep",
                         lambda: fc.clamp(0.0, P.LIDAR_MAX))
        if not ok:
            c, ok = step("  clamp_min roi clamp_max",
                         lambda: f.clamp_min(0.0).clamp_max(P.LIDAR_MAX))
        if not ok:
            c, ok = step("  minimum/maximum thay clamp",
                         lambda: torch.minimum(
                             torch.maximum(f, torch.zeros_like(f)),
                             torch.full_like(f, P.LIDAR_MAX)))
        if not ok:
            c, ok = step("  cong 0 roi clamp",
                         lambda: (f + 0.0).clamp(0.0, P.LIDAR_MAX))
        if not ok:
            return

    d, ok = step("chia cho 8", lambda: c / P.LIDAR_MAX)
    if not ok:
        return
    e, ok = step("1 - x", lambda: 1.0 - d)
    if not ok:
        return

    def gan():
        v[:, 0:P.N_LIDAR_FANS] = e
        return v

    _, ok = step("gan vao lat cat cua mang", gan)
    if not ok:
        def gan2():
            v.narrow(1, 0, P.N_LIDAR_FANS).copy_(e)
            return v
        step("  copy_ vao narrow thay cho gan", gan2)


def _thu_cach_khac(dev_key, retry, pop):
    """Muc 4 hong -> tu chay lai muc 4 trong TIEN TRINH MOI theo vai cach.

    Muc dich la tra loi mot cau duy nhat: hong vi PHEP TINH, hay vi da
    dung qua nhieu tai nguyen truoc do? Tien trinh moi khong mang theo gi
    cua lan truoc, nen neu chay rieng thi qua -> la tai nguyen.
    """
    if not retry or not dev_key:
        return 1
    import subprocess
    print("\n   Thu lai muc 4 trong tien trinh MOI, theo vai cach:")
    thu = [("chay rieng, khong co muc 2-3", []),
           ("gom tia tren CPU", ["--fan", "cpu"]),
           ("gom tia bang vong lap", ["--fan", "loop"]),
           ("gom tia bang scatter", ["--fan", "scatter"]),
           (f"lo nho hon ({max(3, pop // 4 * 3)} xe)",
            ["--robots", str(max(3, pop // 4 * 3))])]
    duoc = []
    for ten, extra in thu:
        cmd = [sys.executable, "-m", "turbo.tools.check", "--device", dev_key,
               "--only", "4", "--no-retry"] + extra
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            ok = (r.returncode == 0)
        except Exception:
            ok = False
        print(f"     [ {'duoc ' if ok else 'KHONG'} ] {ten}")
        if ok:
            duoc.append(ten)
    print()
    if not duoc:
        print("   Cach nao cung hong -> khong phai tai nguyen, ma la mot")
        print("   phep tinh card nay khong lam duoc. Gui ca trang nay lai.")
    elif duoc == ["chay rieng, khong co muc 2-3"] or "chay rieng, khong co muc 2-3" in duoc:
        print("   Chay RIENG thi qua -> card het cho khi phai giu nhieu thu")
        print("   cung luc. Se sua bang cach tra bo nho som hon va dung lo")
        print("   nho hon. Gui ca trang nay lai.")
    else:
        print("   Co cach chay duoc (xem dong 'duoc' o tren). Gui lai trang")
        print("   nay, se dat cach do lam mac dinh cho card lien.")
    return 1


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


def _touch(x):
    """Doc mot so ve CPU. Card chay bat dong bo: khong doc thi loi cua phep
    vua roi se no o phep sau do, va ta di sua nham cho."""
    try:
        if torch.is_tensor(x) and x.numel():
            float(x.reshape(-1)[:1].cpu().sum())
        elif isinstance(x, (tuple, list)):
            for y in x:
                _touch(y)
    except Exception:
        raise


def _stage(name, fn, dev, note=""):
    """Chay mot buoc, DOI cho may lam xong, roi moi bao duoc hay khong."""
    try:
        out = fn()
        _touch(out)
        _sync(dev)
    except Exception as e:
        _fail(name, e)
        return None, False
    _ok(name, note)
    return out, True


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
    ap.add_argument("--fan", default="auto",
                    choices=("auto", "scatter", "loop", "cpu"),
                    help="cach gom tia vao quat - chi de thu khi may la")
    ap.add_argument("--only", default=None,
                    help="chi chay vai muc, vi du --only 4  hoac  --only 4,6")
    ap.add_argument("--no-retry", action="store_true",
                    help="hong thi thoi, dung tu thu lai cach khac")
    DEV.add_argument(ap)
    a = ap.parse_args(argv)
    ops.set_fan_mode(a.fan)
    only = [int(x) for x in a.only.split(",")] if a.only else None
    print(f"Python {sys.version.split()[0]}  |  torch {torch.__version__}")
    print("\nCac may tinh tim thay:")
    for d in DEV.list_devices():
        print(f"  {d['key']:10s} {DEV.describe(d)}")
    ok, why = DEV.directml_status()
    if not ok:
        print(f"  (khong co DirectML: {why})")
    dev = DEV.from_args(a)
    return check(dev, robots=a.robots, only=only,
                 dev_key=getattr(a, "device", None),
                 retry=not a.no_retry)


if __name__ == "__main__":
    sys.exit(main())
