# -*- coding: utf-8 -*-
"""Do ban turbo, va do BEN CANH ban v1 tren CUNG MOT VONG QUET.

    python -m turbo.tools.measure detector

Bo do hoc cua turbo la mot thuat toan khac han ban v1 (xem `turbo/dock.py`),
nen phai chung minh no khong te hon. Cach cong bang duy nhat la cho hai bo
doc CUNG du lieu: mo phong bang turbo, lay vong quet ra, roi dua cho ca hai.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from sim import params as P
from sim import dock_detector as D1
from sim.geometry import point_segment_distance, wrap_pi_scalar
from sim.lidar import Scan
from turbo import dock as D2, sim as TS, world as TW

MATCH = 0.30              # gan hoc hon the nay -> bao dung
FAR = 0.80                # xa moi hoc hon the nay -> bao gia that su


def _poses(worlds, per_map, seed=4):
    """Vi tri ngau nhien tren san, khong nam sat hoc, giong ban v1."""
    rng = random.Random(seed)
    out = []
    for wi, w in enumerate(worlds):
        got = 0
        while got < per_map:
            px = rng.uniform(0.5, 5.9)
            py = rng.uniform(0.5, 4.3)
            th = rng.uniform(-math.pi, math.pi)
            if not w.on_floor(px, py):
                continue
            if min(math.hypot(px - d.x, py - d.y) for d in w.docks) < 0.6:
                continue
            d = point_segment_distance(px, py, w.static_segments)
            if d.size and float(d.min()) < P.BODY_RADIUS + 0.02:
                continue
            out.append((wi, px, py, th))
            got += 1
    return out


def _scan_batch(seeds, per_map, device, lidar_seed=0):
    """Dat xe, quay du mot vong LiDAR, tra ve (the_gioi_lo, sim, tu_the)."""
    bw = TW.build(seeds, per_map, device, n_docks=3, n_decoys=1)
    bw.freeze_movers()                    # ban v1 do khi khong co nguoi di lai
    poses = _poses(bw.worlds, per_map)
    s = TS.BatchSim(bw, device, seed=lidar_seed)
    idx = torch.arange(len(poses))
    s.place(idx,
            torch.tensor([p[1] for p in poses]),
            torch.tensor([p[2] for p in poses]),
            torch.tensor([p[3] for p in poses]))
    for _ in range(4000):
        s.step(torch.zeros(len(poses), 2))
        if s.lidar_step():
            break
    return bw, s, poses


def _grade(bw, poses, cands):
    """cands[r] = danh sach (diem, cu_ly, phuong_vi, truc). Tra ve thong ke."""
    pos_err, ax_err = [], []
    tp = fp = 0
    for r, (wi, px, py, th) in enumerate(poses):
        docks = bw.worlds[wi].docks
        for sc, rng_, bea, axis in cands[r]:
            if sc <= 0.0:
                continue
            cx = px + rng_ * math.cos(th + bea)
            cy = py + rng_ * math.sin(th + bea)
            err, dk = min(((math.hypot(cx - d.x, cy - d.y), d) for d in docks),
                          key=lambda z: z[0])
            if err < MATCH:
                tp += 1
                pos_err.append(err)
                ax_err.append(abs(math.degrees(
                    wrap_pi_scalar(th + axis - dk.theta))))
            elif err > FAR:
                fp += 1
    return tp, fp, np.array(pos_err), np.array(ax_err)


def _report(name, tp, fp, pos, ax):
    rate = 100.0 * fp / max(1, tp + fp)
    print(f"  {name:6s} thay dung {tp:4d}  bao gia {fp:3d} ({rate:2.0f}%)  "
          f"lech vi tri trung vi {np.median(pos) * 100:4.1f} cm  "
          f"lech truc trung vi {np.median(ax):4.1f} do")


SETS = (((3, 5, 8), 0), ((11, 12, 13), 1))


def _merge(parts):
    tp = sum(p[0] for p in parts)
    fp = sum(p[1] for p in parts)
    return tp, fp, np.concatenate([p[2] for p in parts]), \
        np.concatenate([p[3] for p in parts])


def measure_detector(sets=SETS, per_map=60, device="cpu"):
    """Hai bo mat bang doc lap, vi voi n~60 thi chenh 3 lan la nhieu chu khong
    phai ket qua - do mot bo roi chinh tham so theo no la tu lua minh."""
    dev = torch.device(device)
    a, b = [], []
    for seeds, lseed in sets:
        bw, sim, poses = _scan_batch(list(seeds), per_map, dev, lidar_seed=lseed)
        print(f"mat bang {list(seeds)}: {len(poses)} vong quet, "
              f"{int(sim.scan_ok.sum())} diem hop le")
        old = []
        for r in range(len(poses)):
            sc = Scan(sim.scan_r[r].double().numpy(),
                      sim.scan_b[r].double().numpy(),
                      sim.scan_ok[r].numpy(), 0.0, 0)
            old.append([(c.score, c.range, c.bearing, c.axis)
                        for c in D1.detect(sc)])
        out = D2.detect(sim.scan_r, sim.scan_b, sim.scan_ok)
        new = [[tuple(float(x) for x in out[r, k]) for k in range(out.shape[1])]
               for r in range(len(poses))]
        a.append(_grade(bw, poses, old))
        b.append(_grade(bw, poses, new))
    _report("v1", *_merge(a))
    _report("turbo", *_merge(b))


def measure_speed(pop=32, steps=600, robots=3, maps=2, hidden=16,
                  device="cpu", also_v1=True):
    """Mot THE HE cua ban turbo canh mot the he cua ban v1.

    Dem bang BUOC-XE chu khong bang giay: hai ban lam cung khoi luong viec
    (pop x maps x robots x steps con xe buoc mot buoc) nen chia ra la so
    sanh duoc, va con so do giu nguyen nghia khi doi may.
    """
    import torch
    from turbo.policy import BatchPolicy
    from turbo.rollout import Rollout
    dev = torch.device(device)
    work = pop * maps * robots * steps
    print(f"mot the he = {work:,} buoc-xe "
          f"(quan the {pop} x {maps} mat bang x {robots} xe x {steps} buoc)")

    ro = Rollout([3, 5][:maps] or [3], robots, pop, dev, seed=1)
    bp = BatchPolicy(hidden, dev)
    th = torch.randn(pop, bp.n_params, device=dev) * 0.2
    ro.reset(11, 0.3)
    ro.run(bp, th, 5)
    ro.reset(11, 0.3)
    t = time.perf_counter()
    ro.run(bp, th, steps)
    dt = time.perf_counter() - t
    print(f"  turbo  {dt:7.1f} s  {work / dt:10,.0f} buoc-xe/giay  "
          f"({ro.sim.R} xe buoc cung luc)")

    if also_v1:
        import multiprocessing as mp
        from train import train as T1
        from train.policy import GRUPolicy
        pol = GRUPolicy(n_hidden=hidden, seed=0)
        nm, nv, nc = pol.norm.state()
        seeds = [101, 202][:maps]
        tasks = [(pol.theta, 1, i, sg, 0.05, seeds, steps, robots, 0.3,
                  nm, nv, nc)
                 for i in range(pop // 2) for sg in (1.0, -1.0)]
        jobs = min(os.cpu_count() or 1, 4)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(jobs, initializer=T1._init_worker,
                        initargs=(hidden, pol.n_in, pol.n_out))
        try:
            t = time.perf_counter()
            pool.map(T1._eval_one, tasks, chunksize=1)
            dt1 = time.perf_counter() - t
        finally:
            pool.close()
            pool.join()
        print(f"  v1     {dt1:7.1f} s  {work / dt1:10,.0f} buoc-xe/giay  "
              f"({jobs} tien trinh)")
        print(f"  -> turbo nhanh gap {dt1 / dt:.2f} lan tren may nay")


def measure_scale(pops=(16, 32, 64, 128, 256), steps=150, robots=3, maps=2,
                  hidden=16, device="cpu"):
    """Quan the cang lon thi ban nao duoi kip?

    Day moi la cho quan trong. Ban v1 chia quan the cho cac tien trinh: het
    loi may thi them ca the la them thoi gian, dung ti le. Ban turbo gop ca
    quan the vao mot phep tinh: them ca the la mang to hon, va mang cang to
    thi may cang chay dung suc - cho toi khi dung het duong truyen bo nho.
    """
    import torch
    from turbo.policy import BatchPolicy
    from turbo.rollout import Rollout
    dev = torch.device(device)
    bp = BatchPolicy(hidden, dev)
    print(f"{'quan the':>9} {'xe cung luc':>12} {'buoc-xe/giay':>14}")
    for pop in pops:
        ro = Rollout([3, 5][:maps] or [3], robots, pop, dev, seed=1)
        th = torch.randn(pop, bp.n_params, device=dev) * 0.2
        ro.reset(11, 0.3)
        ro.run(bp, th, 5)
        ro.reset(11, 0.3)
        t = time.perf_counter()
        ro.run(bp, th, steps)
        dt = time.perf_counter() - t
        print(f"{pop:9d} {ro.sim.R:12d} {steps * ro.sim.R / dt:14,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("detector", "speed", "scale"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--no-v1", action="store_true")
    a = ap.parse_args()
    if a.what == "detector":
        measure_detector(device=a.device)
    elif a.what == "speed":
        measure_speed(pop=a.pop, steps=a.steps, device=a.device,
                      also_v1=not a.no_v1)
    elif a.what == "scale":
        measure_scale(device=a.device)


if __name__ == "__main__":
    main()
