# -*- coding: utf-8 -*-
"""Do hoc chu U theo lo - viet lai thanh KHOI LUONG CO DINH.

Ban `sim/dock_detector.py` cat vong quet thanh cac doan dai ngan tuy du lieu,
roi noi lai, roi phan tich tung doan. Dung dan va chinh xac, nhung moi vong
quet lam mot kieu khac nhau nen khong the gop 144 xe vao mot phep tinh - tuc
la khong dung duoc GPU.

O day doi cach hoi, va chia thanh hai chang:

CHANG 1 - TIM THANH TRONG (re, chay tren moi diem).
  Cai duy nhat cua hoc luon nhin thay duoc la THANH TRONG: mot mat phang
  rong 31 cm. Voi MOI diem quet, ta hoi "quanh diem nay co mot doan thang
  phang dai bao nhieu?" - bang cach thu cac be rong 7,11,15,21,29,41,57,79
  diem va lay be rong LON NHAT con phang. Tuong phong dai vai met thi be
  rong lon nhat la 79 -> doan qua dai -> loai. Thanh trong thi be rong nao
  tran ra hai canh hoc se het phang -> do duoc dung ~31 cm.
  Tat ca lam bang TONG TICH LUY (cumsum) nen mot be rong het O(n), khong
  phai O(n*w). Ca 8 be rong tren 500 diem cua 144 xe het vai phep tinh.

CHANG 2 - THU LAI 6 UNG VIEN DAU BANG HINH HOC DAY DU.
  Doi ca vong quet sang he quy chieu cua hoc DE NGHI roi kiem ba dieu ma
  chi cai hoc that thoa:
    - long hoc PHAI TRONG (khong co diem nao trong khoang sau 4,5..26,5 cm),
    - PHAI CO HAI CANH o hai ben (goc tuong lom chi co mot ben -> loai),
    - PHIA SAU thanh trong PHAI KHONG CO GI (than hoc day 40 cm che het).
  Chang nay chi chay cho 6 ung vien nen (R,6,500) - khong dang ke.

Doi lai: day la MOT THUAT TOAN KHAC ban cu. No khong ra ket qua y het, va
bo nao nuoi bang ban nay khong dung lai duoc cho ban kia. Do chinh xac o
`turbo/tests/test_dock.py`.
"""

import math

import torch

from sim import params as P

from . import ops
from .ops import hypot

# --------------------------------------------------------------- chang 1
HALVES = (2, 3, 5, 7, 10, 14, 20, 28, 39)  # be rong 5,7,11,15,21,29,41,57,79
# Nguong "phang" phai TI LE VOI CU LY: nhieu cua Camsense la ti le (1,2% cua
# cu ly), nen mot buc tuong o 0,9 m gon song 1,1 cm con o 2,3 m la 2,8 cm.
# Dat nguong co dinh thi o xa cai gi cung gap khuc, o gan cai gi cung phang.
FLAT_K = 1.8 * P.LIDAR_NOISE
FLAT_FLOOR = 0.008
RUN_MIN = 0.15            # doan phang ngan hon the nay khong phai thanh trong
RUN_MAX = 0.50            # dai hon the nay la tuong phong
RUN_BEST = 0.33           # dai do duoc cua thanh trong that (31 cm + tran ra)
MIN_VALID = 0.85          # 2% diem bi mat, doi het diem hop le la vo ich
# Gop bao nhieu be rong vao mot phep tinh. Gop het 9 thi it loi goi nhat
# (tot cho GPU) nhung mang trung gian 3,5 trieu o khong lot vao bo nho dem
# CPU; gop 1 thi nguoc lai. Do duoc tren CPU: gop 1 la 143 ms, gop 3 la 147,
# gop 9 la 165 - chenh 15%, nen lay khuc giua.
WIDTH_GROUP = 3
NB = 40                   # ban kinh lan can (~29 do) de do "co gi gan hon"
NEAR_MIN = 0.12           # phai co vat gan hon thanh trong it nhat the nay
# Goc toi cua tia len thanh trong. Lech hon the nay thi canh hoc BEN KIA da
# che gan het thanh trong, cho nen co do cung sai - ban v1 cung chi chinh xac
# trong khoang +-15 do. Gioi han nay con loai luon CANH HOC (mot tam phang
# 31 cm y het thanh trong, nhung phap tuyen quay ngang) ngay o chang 1.
MAX_INCID = math.radians(40.0)
TOPK = 12                 # so ung vien mang sang chang 2
# Chang 2 chi can cac diem QUANH ung vien, khong can ca vong quet: cai hoc
# rong 40 cm nen o 0,35 m no chiem 60 do, o xa hon thi it hon. Lay +-69 do
# la du roi ma re hon ca vong 2,6 lan.
VIEW = 96                 # nua so diem lay quanh ung vien
DEDUP = 0.12              # hai ung vien gan nhau hon the nay la mot (chang 1)
DEDUP_OUT = 0.25          # ... va sau khi da khop lai (chang 2)

# --------------------------------------------------------------- chang 2
DEPTH = P.DOCK_CAVITY_D           # 0.31
HALF_W = 0.5 * P.DOCK_CAVITY_W    # 0.155
# Be day lop diem duoc coi la "tren thanh trong". Cung phai theo nhieu: dat
# 3 cm co dinh thi o 2,3 m (nhieu 2,8 cm) ta cat mat nua may diem va khop
# duong thang bi lech - do chinh la cho ban dau lech truc 4,9 do.
TOL_K = 2.0 * P.LIDAR_NOISE
TOL_FLOOR = 0.030
BACK_SPAN = 0.155         # nua be rong vung lay diem thanh trong
N_REFIT = 3               # so lan khop lai
# Vung "long hoc phai trong" va "phia sau phai trong" phai LUI RA THEO NHIEU:
# nhieu la ti le cu ly, nen o 2,3 m mot diem tren thanh trong co the nhay ra
# truoc 8 cm. Dat bien co dinh 4,5 cm thi o xa hoc that nao cung tu loai minh.
SKIN_K = 3.2 * P.LIDAR_NOISE
SKIN_FLOOR = 0.045
INNER_U_HI = 0.265        # long hoc phai trong
INNER_V = 0.105
BEHIND_LO = -0.60
BEHIND_V = 0.130
# Hai canh hoc. Bien duoi theo u phai LON HON KHONG han: mot buc tuong
# phang dai co san diem o u~0 voi |v| lon, va neu tinh ca chung thi tuong
# nao cung co "hai canh" -> bo do bao gia 85%. Canh hoc that nho ra TRUOC
# mat thanh trong, nen chi dem diem o u >= 6 cm.
WING_U = (0.060, 0.380)
WING_V = (0.115, 0.270)
BACK_MIN = 0.22
BACK_MAX = 0.42

N_OUT = P.N_DOCK_CANDIDATES


def _csum(a, pad):
    """Tong tich luy cua mang da dem vong tron. Tra ve dai n+2*pad+1."""
    ap = torch.cat((a[:, -pad:], a, a[:, :pad]), dim=1)
    z = torch.zeros(a.shape[0], 1, dtype=ap.dtype, device=ap.device)
    return torch.cat((z, ap.cumsum(1)), dim=1)


def _near_min(r, device):
    """Cu ly nho nhat trong lan can +-NB diem, tinh cho moi diem."""
    return ops.min_window_circ(r, NB)


def _flat_runs(xs, ys, ok):
    """Voi moi diem: doan thang phang ROI NHAT quanh no.

    Tra ve (co, dai, tam_x, tam_y, goc_duong) - `goc_duong` la phuong cua
    doan thang (chua phai truc hoc).

    Ca 9 be rong tinh CUNG MOT LUC tren truc thu hai, khong phai mot vong
    lap Python 9 vong. Cung tung ay phep tinh, nhung la 20 loi goi thay vi
    180 - tren CPU thi do la bot 160 lan dung day chuyen, con tren GPU thi
    moi loi goi deu mat mot khoan cho co dinh nen cho nay an dam.

    "Be rong LON NHAT con phang" ra bang mot phep cong: `alive` la AND luy
    tien theo be rong (cummin tren 0/1), nen tong cua no chinh la so be rong
    lien tiep con phang - va do la chi so can lay.
    """
    device = xs.device
    R, n = xs.shape
    pad = HALVES[-1] + 1
    z = torch.zeros_like(xs)
    xv = torch.where(ok, xs, z)
    yv = torch.where(ok, ys, z)

    Cn = _csum(ok.to(xs.dtype), pad)
    Cx, Cy = _csum(xv, pad), _csum(yv, pad)
    Cxx, Cyy, Cxy = _csum(xv * xv, pad), _csum(yv * yv, pad), _csum(xv * yv, pad)

    base = torch.arange(n, device=device)[None, :]
    parts = []
    for i in range(0, len(HALVES), WIDTH_GROUP):
        hh = torch.tensor(HALVES[i:i + WIDTH_GROUP], device=device)[:, None]
        lo = base + pad - hh                       # (W,n)
        hi = base + pad + hh + 1
        w = (2 * hh + 1).to(xs.dtype)[None, :, :]  # (1,W,1)

        sel = lambda C: (C.index_select(1, hi.reshape(-1))
                         - C.index_select(1, lo.reshape(-1))
                         ).view(C.shape[0], hh.shape[0], -1)
        nv = sel(Cn)                               # (R,W,n)
        inv = 1.0 / nv.clamp(min=1.0)
        mx = sel(Cx) * inv
        my = sel(Cy) * inv
        sxx = sel(Cxx) * inv - mx * mx
        syy = sel(Cyy) * inv - my * my
        sxy = sel(Cxy) * inv - mx * my
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        disc = (0.25 * tr * tr - det).clamp(min=0.0).sqrt()
        rms = (0.5 * tr - disc).clamp(min=0.0).sqrt()
        thr = (FLAT_K * hypot(mx, my)).clamp(min=FLAT_FLOOR)
        flat = (nv >= MIN_VALID * w) & (rms < thr)
        # Dai doan: lay tu PHUONG SAI chu khong tu hai dau. Hai dau co the la
        # diem bi mat (2% bi mat), va diem mat thi toa do la rac.
        length = (12.0 * (0.5 * tr + disc).clamp(min=0.0)).sqrt()
        line = 0.5 * torch.atan2(2.0 * sxy, sxx - syy)
        parts.append((flat, length, mx, my, line))

    cat = lambda j: torch.cat([p[j] for p in parts], dim=1)
    alive = ops.cum_and(cat(0), dim=1)
    have = alive[:, 0] > 0
    k = (alive.sum(dim=1).long() - 1).clamp(min=0)[:, None, :]   # (R,1,n)
    g = lambda j: cat(j).gather(1, k).squeeze(1)
    return have, g(1), g(2), g(3), g(4)



def _to_sensor(ang, cx, cy):
    """Phap tuyen cua duong `ang`, chon chieu CHI VE cam bien (goc toa do)."""
    nx, ny = -torch.sin(ang), torch.cos(ang)
    flip = (nx * -cx + ny * -cy) < 0.0
    nx = torch.where(flip, -nx, nx)
    ny = torch.where(flip, -ny, ny)
    return torch.atan2(ny, nx)


def _stage_a(xs, ys, ok, scan_r):
    device = xs.device
    have, run, cx, cy, ang = _flat_runs(xs, ys, ok)
    rc = hypot(cx, cy)
    near = _near_min(scan_r, device)
    deep = rc - near                      # co gi gan hon thanh trong khong

    axis = _to_sensor(ang, cx, cy)
    incid = axis - torch.atan2(-cy, -cx)
    incid = torch.atan2(torch.sin(incid), torch.cos(incid)).abs()

    good = (have & (run >= RUN_MIN) & (run <= RUN_MAX)
            & (deep >= NEAR_MIN) & (incid <= MAX_INCID) & ok)
    score = (1.0 - (run - RUN_BEST).abs() / 0.26).clamp(min=0.0)
    score = score * (1.0 - 0.5 * incid / MAX_INCID)
    score = torch.where(good, score.clamp(max=1.0), torch.zeros_like(score))
    return score, cx, cy, axis


def _pick(score, cx, cy, axis, k):
    """Lay `k` ung vien roi nhat, loai trung trong ban kinh DEDUP."""
    ox, oy, oa, os_, oi = [], [], [], [], []
    for _ in range(k):
        best, wi = score.max(dim=1)
        g = wi[:, None]
        bx = cx.gather(1, g).squeeze(1)
        by = cy.gather(1, g).squeeze(1)
        os_.append(best)
        ox.append(bx)
        oy.append(by)
        oa.append(axis.gather(1, g).squeeze(1))
        oi.append(wi)
        near = hypot(cx - bx[:, None], cy - by[:, None]) < DEDUP
        score = torch.where(near, torch.zeros_like(score), score)
    return (torch.stack(os_, 1), torch.stack(ox, 1), torch.stack(oy, 1),
            torch.stack(oa, 1), torch.stack(oi, 1))


def _slice(xs, ys, ok, at):
    """Lay +-VIEW diem quanh moi ung vien. at: (R,K) -> (R,K,M)."""
    R, n = xs.shape
    K = at.shape[1]
    off = torch.arange(-VIEW, VIEW + 1, device=xs.device)
    g = ((at[:, :, None] + off) % n).reshape(R, -1)
    m = off.numel()
    return (xs.gather(1, g).view(R, K, m), ys.gather(1, g).view(R, K, m),
            ok.gather(1, g).view(R, K, m))


def _local(px, py, cx, cy, axis):
    """Doi diem sang he hoc de nghi: u = do sau tu thanh trong ra mieng."""
    ca, sa = torch.cos(axis)[..., None], torch.sin(axis)[..., None]
    dx = px - cx[..., None]
    dy = py - cy[..., None]
    return dx * ca + dy * sa, -dx * sa + dy * ca


def _refit(px, py, m):
    """Khop lai duong thang qua cac diem duoc chon. m: (R,K,M)."""
    wgt = m.to(px.dtype)
    c = wgt.sum(-1)
    inv = 1.0 / c.clamp(min=1.0)
    mx = (px * wgt).sum(-1) * inv
    my = (py * wgt).sum(-1) * inv
    dx = (px - mx[..., None]) * wgt
    dy = (py - my[..., None]) * wgt
    sxx = (dx * dx).sum(-1) * inv
    syy = (dy * dy).sum(-1) * inv
    sxy = (dx * dy).sum(-1) * inv
    return mx, my, 0.5 * torch.atan2(2.0 * sxy, sxx - syy), c


def _verify(px, py, m0, cx, cy, axis):
    """Cham diem cac ung vien bang hinh hoc day du. Tra ve (diem, mieng, truc)."""
    mx, my = cx, cy
    tol = (TOL_K * hypot(cx, cy)).clamp(min=TOL_FLOOR)[..., None]
    for _ in range(N_REFIT):
        u, v = _local(px, py, mx, my, axis)
        sel = m0 & (u.abs() < tol) & (v.abs() < BACK_SPAN)
        fx, fy, line, nf = _refit(px, py, sel)
        keep = nf >= 4
        axis = torch.where(keep, _to_sensor(line, fx, fy), axis)
        mx = torch.where(keep, fx, mx)
        my = torch.where(keep, fy, my)

    u, v = _local(px, py, mx, my, axis)
    big = torch.finfo(u.dtype).max

    back = m0 & (u.abs() < tol) & (v.abs() < BACK_SPAN)
    nb = back.sum(-1)
    hi = torch.where(back, v, torch.full_like(v, -big)).max(-1).values
    lo = torch.where(back, v, torch.full_like(v, big)).min(-1).values
    span = hi - lo
    rms = ((u.abs() * back).sum(-1) / nb.clamp(min=1)).clamp(min=0.0)

    skin = (SKIN_K * hypot(mx, my)).clamp(min=SKIN_FLOOR)[..., None]
    inner = (m0 & (u > skin) & (u < INNER_U_HI)
             & (v.abs() < INNER_V)).sum(-1)
    behind = (m0 & (u > BEHIND_LO) & (u < -skin)
              & (v.abs() < BEHIND_V)).sum(-1)
    wing = m0 & (u > skin + WING_U[0]) & (u < WING_U[1])
    wl = (wing & (v > WING_V[0]) & (v < WING_V[1])).sum(-1)
    wr = (wing & (-v > WING_V[0]) & (-v < WING_V[1])).sum(-1)

    good = ((nb >= 4) & (inner == 0) & (behind == 0)
            & (span >= BACK_MIN) & (span <= BACK_MAX)
            & (wl >= 1) & (wr >= 1) & (wl + wr >= 4))

    s_span = (1.0 - (span - P.DOCK_CAVITY_W).abs() / 0.20).clamp(0.0, 1.0)
    # Do phang phai tinh THEO CU LY, dung nhu `tol` o tren: nhieu LiDAR ti
    # le voi cu ly, nen o 2 m thanh trong phang tuyet doi van co do lech
    # trung binh ~2 cm. Nguong co dinh 2 cm thi o do diem ve 0 va xe dung
    # thang truoc hoc cua minh cach 1,9 m ma khong thay no.
    flat_tol = 0.020 + 1.6 * P.LIDAR_NOISE * hypot(mx, my)
    s_flat = (1.0 - rms / flat_tol).clamp(0.0, 1.0)
    s_wing = 0.55 + 0.45 * ((wl + wr).to(u.dtype) / 10.0).clamp(max=1.0)
    score = s_span * s_flat * s_wing
    score = torch.where(good, score, torch.zeros_like(score))

    ca, sa = torch.cos(axis), torch.sin(axis)
    return score, mx + DEPTH * ca, my + DEPTH * sa, axis


def detect(scan_r, scan_b, scan_ok):
    """Tra ve (R, N_OUT, 4): diem, cu ly mieng hoc, phuong vi, goc truc ra."""
    xs = scan_r * torch.cos(scan_b)
    ys = scan_r * torch.sin(scan_b)
    sa, cx, cy, ax = _stage_a(xs, ys, scan_ok, scan_r)
    sk, kx, ky, ka, ki = _pick(sa, cx, cy, ax, TOPK)
    kx = torch.where(sk > 0.0, kx, torch.full_like(kx, 1e4))
    lx, ly, lok = _slice(xs, ys, scan_ok, ki)
    score, mx, my, axis = _verify(lx, ly, lok, kx, ky, ka)
    score = torch.where(sk > 0.0, score, torch.zeros_like(score))

    # Loai trung LAN HAI. Hai cua so cach nhau 15 cm tren cung thanh trong
    # deu khop ve DUNG mot tu the, va neu de ca hai thi chung chiem het hai
    # cho ra va day cai hoc thu hai ra ngoai.
    out = torch.zeros(xs.shape[0], N_OUT, 4, device=xs.device)
    for k in range(N_OUT):
        best, wi = score.max(dim=1)
        g = wi[:, None]
        bx = mx.gather(1, g).squeeze(1)
        by = my.gather(1, g).squeeze(1)
        out[:, k, 0] = best
        out[:, k, 1] = hypot(bx, by)
        out[:, k, 2] = torch.atan2(by, bx)
        out[:, k, 3] = axis.gather(1, g).squeeze(1)
        if k + 1 < N_OUT:
            near = hypot(mx - bx[:, None], my - by[:, None]) < DEDUP_OUT
            score = torch.where(near, torch.zeros_like(score), score)
    return out
