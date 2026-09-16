# -*- coding: utf-8 -*-
"""Camera tren cot, mot servo ngua len cui xuong.

Khong co servo quay trai phai: muon nhin sang ben thi xoay ca xe. Do la
rang buoc that cua phan cung, va no lam bai toan kho hon mot bac - xe phai
tu quyet dinh luc nao thi quay nguoi de nhin, danh doi voi viec dang chay.

Ham dung anh nhan ca lo the gioi cung mot luc; do la ly do ban nay chay
duoc tren GPU con ban truoc thi khong.
"""

import math

import torch

from . import params as P
from .raytrace import KIND_SKY, trace

SKY = (0.58, 0.66, 0.76)
LIGHT = (0.32, 0.22, 0.92)


def camera_frame(x, y, th, tilt):
    """Ba truc cua camera: nhin toi, sang phai, len tren."""
    cp, sp = torch.cos(tilt), torch.sin(tilt)
    ct, st = torch.cos(th), torch.sin(th)
    f = torch.stack((ct * cp, st * cp, sp), dim=-1)
    r = torch.stack((st, -ct, torch.zeros_like(st)), dim=-1)
    u = torch.stack((-ct * sp, -st * sp, cp), dim=-1)
    o = torch.stack((x + P.CAM_FORWARD * ct, y + P.CAM_FORWARD * st,
                     torch.full_like(x, P.CAM_HEIGHT)), dim=-1)
    return o, f, r, u


def pixel_dirs(f, r, u, w, h, fov_h, device, dtype):
    """Huong tia cua tung diem anh, theo mo hinh lo kim."""
    tx = math.tan(0.5 * fov_h)
    ty = tx * h / w
    cols = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2.0 - 1.0
    rows = 1.0 - (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2.0
    ax = (cols * tx)[None, :].expand(h, w).reshape(-1)     # (h*w,)
    ay = (rows * ty)[:, None].expand(h, w).reshape(-1)
    d = (f[:, None, :] + ax[None, :, None] * r[:, None, :]
         + ay[None, :, None] * u[:, None, :])
    return d / d.norm(dim=-1, keepdim=True)


def render(sc, x, y, th, tilt, w=P.CAM_W, h=P.CAM_H, fov_h=P.CAM_FOV_H,
           chunk=8, noise=0.0, fog=14.0, generator=None):
    """Dung anh RGB. Tra ve (B, 3, h, w) trong khoang [0,1]."""
    dev, dt = x.device, x.dtype
    o, f, r, u = camera_frame(x, y, th, tilt)
    d = pixel_dirs(f, r, u, w, h, fov_h, dev, dt)
    o = o[:, None, :].expand(-1, h * w, -1)

    t, alb, nrm, kind = trace(o, d, sc, chunk=chunk)

    lig = torch.tensor(LIGHT, device=dev, dtype=dt)
    lig = lig / lig.norm()
    lam = torch.clamp((nrm * lig).sum(dim=-1), min=0.0)
    shade = 0.42 + 0.58 * lam                      # nen sang + den tren cao
    col = alb * shade[..., None]

    sky = torch.tensor(SKY, device=dev, dtype=dt)
    is_sky = kind == KIND_SKY
    # Suong theo cu ly: vua la thuc te cua may anh re, vua cho bo nao mot
    # manh moi de doan xa gan.
    fade = 1.0 - torch.exp(-torch.clamp(t, max=60.0) / fog)
    col = col * (1.0 - fade[..., None]) + sky * fade[..., None]
    col = torch.where(is_sky[..., None], sky.expand_as(col), col)

    if noise > 0.0:
        col = col + torch.randn(col.shape, device=dev, dtype=dt,
                                generator=generator) * noise
    col = torch.clamp(col, 0.0, 1.0)
    return col.reshape(-1, h, w, 3).permute(0, 3, 1, 2).contiguous()


def depth(sc, x, y, th, tilt, w=P.CAM_W, h=P.CAM_H, fov_h=P.CAM_FOV_H,
          chunk=8):
    """Ban do cu ly - khong dua cho bo nao, chi de kiem thu va xem."""
    o, f, r, u = camera_frame(x, y, th, tilt)
    d = pixel_dirs(f, r, u, w, h, fov_h, x.device, x.dtype)
    o = o[:, None, :].expand(-1, h * w, -1)
    t, _a, _n, _k = trace(o, d, sc, chunk=chunk)
    return t.reshape(-1, h, w)
