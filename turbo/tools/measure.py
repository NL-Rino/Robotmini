# -*- coding: utf-8 -*-
"""Do ban turbo, va do BEN CANH ban v1 tren CUNG MOT VONG QUET.

    python -m turbo.tools.measure detector

Bo do hoc cua turbo la mot thuat toan khac han ban v1 (xem `turbo/dock.py`),
nen phai chung minh no khong te hon. Cach cong bang duy nhat la cho hai bo
doc CUNG du lieu: mo phong bang turbo, lay vong quet ra, roi dua cho ca hai.
"""

import argparse
import math
import random

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("detector",))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.what == "detector":
        measure_detector(device=a.device)


if __name__ == "__main__":
    main()
