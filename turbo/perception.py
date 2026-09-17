# -*- coding: utf-8 -*-
"""Dung vector 48 dau vao cho ca lo cung mot luc.

Phai ra DUNG cai ma `sim/perception.py` ra, chi khac la 144 hang thay vi
mot. Bo cuc, thu tu, cach chuan hoa - tat ca deu chep tu ban v1, vi bo nao
nuoi o day phai cam vao duoc con robot that qua dung mot cai vector do.

Kiem cheo o `turbo/tests/test_perception.py`: dung cung mot trang thai roi
so tung o mot voi ban v1.
"""

import torch

from sim import params as P
from sim.perception import (I_BATTERY, I_CLIFF, I_CONTACT, I_DOCKS, I_FANS,
                            I_IR, I_MOTION, I_STATION, DOCK_RANGE_NORM,
                            STATION_RANGE_NORM)

from .ops import hypot

N_INPUTS = P.N_INPUTS


def _range_feat(d, norm):
    return (1.0 - d.clamp(max=norm) / norm).clamp(min=0.0)


def build(sim, docks):
    """sim: BatchSim. docks: (R, N_DOCK_CANDIDATES, 4) tu `turbo.dock.detect`."""
    R = sim.R
    v = torch.zeros(R, N_INPUTS, device=sim.device)

    fans = sim.fans()
    v[:, I_FANS:I_FANS + P.N_LIDAR_FANS] = \
        1.0 - fans.clamp(0.0, P.LIDAR_MAX) / P.LIDAR_MAX

    cl, cr = sim.cliff()
    v[:, I_CLIFF] = cl.float()
    v[:, I_CLIFF + 1] = cr.float()

    for k in range(P.N_DOCK_CANDIDATES):
        b = I_DOCKS + 6 * k
        sc, rng_, bea, axis = docks[:, k, 0], docks[:, k, 1], \
            docks[:, k, 2], docks[:, k, 3]
        on = sc > 0.0
        v[:, b + 0] = torch.where(on, sc.clamp(0.0, 1.0), torch.zeros_like(sc))
        v[:, b + 1] = torch.where(on, _range_feat(rng_, DOCK_RANGE_NORM),
                                  torch.zeros_like(sc))
        v[:, b + 2] = torch.where(on, torch.sin(bea), torch.zeros_like(sc))
        v[:, b + 3] = torch.where(on, torch.cos(bea), torch.zeros_like(sc))
        v[:, b + 4] = torch.where(on, torch.sin(axis), torch.zeros_like(sc))
        v[:, b + 5] = torch.where(on, torch.cos(axis), torch.zeros_like(sc))

    for off, (seen, bear) in ((0, sim.ir_beacon()), (3, sim.ir_dock())):
        f = seen.float()
        v[:, I_IR + off + 0] = f
        v[:, I_IR + off + 1] = torch.sin(bear) * f
        v[:, I_IR + off + 2] = torch.cos(bear) * f

    # Bo nho tram: hieu hai so CUNG HE ODOM nen phan troi triet tieu.
    dx = sim.station[:, 0] - sim.ox
    dy = sim.station[:, 1] - sim.oy
    dist = hypot(dx, dy)
    bear = _wrap(torch.atan2(dy, dx) - sim.oth)
    axis_rel = _wrap(sim.station[:, 2] - sim.oth)
    # `= 1.0` (so Python thuan) thi tren card lien no thanh mot so 64 bit va
    # phep gan bao "self.dtype khac src.dtype". Phai la tensor cung kieu.
    v[:, I_STATION + 0] = torch.ones_like(dist)
    v[:, I_STATION + 1] = _range_feat(dist, STATION_RANGE_NORM)
    v[:, I_STATION + 2] = torch.sin(bear)
    v[:, I_STATION + 3] = torch.cos(bear)
    v[:, I_STATION + 4] = torch.sin(axis_rel)
    v[:, I_STATION + 5] = torch.cos(axis_rel)

    v[:, I_BATTERY + 0] = sim.batt
    v[:, I_BATTERY + 1] = blink(sim.t, sim.low_lamp)
    v[:, I_BATTERY + 2] = sim.charging.float()

    v[:, I_CONTACT + 0] = sim.in_slot.float()
    v[:, I_CONTACT + 1] = sim.id_ok.float()

    v[:, I_MOTION + 0] = (sim.v / P.V_MAX).clamp(-1.0, 1.0)
    v[:, I_MOTION + 1] = (sim.wv / P.W_MAX).clamp(-1.0, 1.0)
    v[:, I_MOTION + 2] = sim.cmd[:, 0]
    v[:, I_MOTION + 3] = sim.cmd[:, 1]
    v[:, I_MOTION + 4] = sim.bump
    return v


def blink(t, low_lamp):
    """Den bao sac: nhap nhay 0/1 lien tuc, CHI the thoi."""
    on = 1.0 if int(t * P.BATT_BLINK_HZ * 2.0) % 2 == 0 else 0.0
    return low_lamp.float() * on


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))
