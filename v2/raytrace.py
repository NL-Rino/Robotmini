# -*- coding: utf-8 -*-
"""Bo doi tia 3D chay theo lo, viet bang PyTorch thuan.

Vi sao tu viet chu khong dung thu vien do hoa: may thue chi co card va
dong lenh, khong co man hinh. OpenGL/Vulkan can EGL va driver do hoa, cai
dat tren may thue la mot mo hon. Con phep doi tia thi chi la phep tinh
tensor - chay y het nhau tren CPU va tren CUDA, khong them mot goi nao.

Va no hop voi viec huan luyen: 512 the gioi cung ve mot luc la 512 lan
nhan ma tran chu khong phai 512 lan goi driver.

Hinh khoi:
  hop xoay quanh truc dung  - tuong, vat can, vach hoc, bang ma
  tru dung                  - nguoi di lai, xe khac
  san phang z=0 co lo thung - mep ban, o cau thang

Tra ve cho moi tia: khoang cach, mau be mat, phap tuyen, loai vat.
"""

import torch

KIND_SKY = 0
KIND_FLOOR = 1
KIND_BOX = 2
KIND_CYL = 3
KIND_CEIL = 4

MAT_PLAIN = 0
MAT_MARK = 1          # bang ma hoc sac: mau tung o lay tu `box_code`

FAR = 1.0e9
EPS = 1e-6


def _rot_z(x, y, c, s):
    return x * c + y * s, -x * s + y * c


def _slab(o, d, c, h, yaw):
    """Phep thu theo lop cho hop xoay quanh truc dung.

    Lam theo tung truc mot, khong xep thanh vector 3 chieu. Xep vector thi
    phai cap phat them hai mang (B,R,K,3) va o kich thuoc nay tien chuyen
    bo nho dat hon tien tinh toan nhieu lan.
    """
    cs, sn = torch.cos(yaw), torch.sin(yaw)
    rx = o[..., 0] - c[..., 0]
    ry = o[..., 1] - c[..., 1]
    lx = rx * cs + ry * sn
    ly = -rx * sn + ry * cs
    lz = o[..., 2] - c[..., 2]
    dx0 = d[..., 0]
    dy0 = d[..., 1]
    dx = dx0 * cs + dy0 * sn
    dy = -dx0 * sn + dy0 * cs
    dz = d[..., 2]

    tlo = None
    thi = None
    axis = None
    for k, (l, dd, hh) in enumerate(((lx, dx, h[..., 0]),
                                     (ly, dy, h[..., 1]),
                                     (lz, dz, h[..., 2]))):
        inv = 1.0 / torch.where(dd.abs() < EPS, torch.full_like(dd, EPS), dd)
        t1 = (-hh - l) * inv
        t2 = (hh - l) * inv
        a1 = torch.minimum(t1, t2)
        b1 = torch.maximum(t1, t2)
        if tlo is None:
            tlo, thi = a1, b1
            axis = torch.zeros_like(a1, dtype=torch.uint8)
        else:
            newer = a1 > tlo
            axis = torch.where(newer, torch.full_like(axis, k), axis)
            tlo = torch.maximum(tlo, a1)
            thi = torch.minimum(thi, b1)
    near = torch.where(tlo > EPS, tlo, thi)
    hit = (thi >= torch.clamp(tlo, min=0.0)) & (thi > EPS) & (near > EPS)
    hit &= h.abs().sum(dim=-1) > EPS
    return torch.where(hit, near, torch.full_like(near, FAR)), axis, \
        (lx, ly, lz), (dx, dy, dz)


def _boxes(o, d, sc, t, alb, nrm, kind, chunk, palette):
    """Doi tia voi cac hop.

    Hai luot. Luot mot chi di tim xem hop nao gan nhat - phan nay chay tren
    (B,R,K) nen phai gon het muc. Luot hai moi tinh phap tuyen, mau va bang
    ma, va no chi chay cho MOT hop da thang, tuc la tren (B,R). Gop lam mot
    luot thi moi hop deu phai tra gia cho viec tinh mau du cuoi cung bi che
    khuat het.
    """
    nb = sc["box_c"].shape[1]
    best = torch.full_like(t, FAR)
    bidx = torch.zeros(t.shape, dtype=torch.long, device=t.device)
    for i0 in range(0, nb, chunk):
        i1 = min(nb, i0 + chunk)
        cand, _ax, _l, _dl = _slab(o[:, :, None, :], d[:, :, None, :],
                                   sc["box_c"][:, None, i0:i1, :],
                                   sc["box_h"][:, None, i0:i1, :],
                                   sc["box_yaw"][:, None, i0:i1])
        cb, cw = cand.min(dim=-1)
        upd = cb < best
        bidx = torch.where(upd, cw + i0, bidx)
        best = torch.where(upd, cb, best)

    take = best < t
    if not bool(take.any()):
        return t, alb, nrm, kind

    g3 = bidx[..., None].expand(-1, -1, 3)
    c1 = torch.gather(sc["box_c"], 1, g3)
    h1 = torch.gather(sc["box_h"], 1, g3)
    y1 = torch.gather(sc["box_yaw"], 1, bidx)
    _c, axis, loc, dl = _slab(o, d, c1, h1, y1)

    sign = torch.zeros_like(best)
    nl = [torch.zeros_like(best) for _ in range(3)]
    for k in range(3):
        on = axis == k
        sign = torch.where(on, -torch.sign(dl[k]), sign)
        nl[k] = torch.where(on, -torch.sign(dl[k]), torch.zeros_like(best))
    cy, sy = torch.cos(y1), torch.sin(y1)
    n_new = torch.stack((nl[0] * cy - nl[1] * sy,
                         nl[0] * sy + nl[1] * cy, nl[2]), dim=-1)
    a_new = torch.gather(sc["box_col"], 1, g3)

    mat = torch.gather(sc["box_mat"], 1, bidx)
    is_mark = (mat == MAT_MARK) & (axis == 0)
    if bool(is_mark.any()):
        hy = loc[1] + best * dl[1]
        v = (hy / torch.clamp(h1[..., 1], min=EPS) + 1.0) * 0.5
        cells = sc["box_code"].shape[-1]
        cell = torch.clamp((v * cells).long(), 0, cells - 1)
        code = torch.gather(sc["box_code"], 1,
                            bidx[..., None].expand(-1, -1, cells))
        ci = torch.gather(code, -1, cell[..., None]).squeeze(-1)
        a_new = torch.where(is_mark[..., None], palette[ci], a_new)

    t = torch.where(take, best, t)
    alb = torch.where(take[..., None], a_new, alb)
    nrm = torch.where(take[..., None], n_new, nrm)
    kind = torch.where(take, torch.full_like(kind, KIND_BOX), kind)
    return t, alb, nrm, kind


def _cylinders(o, d, sc, t, alb, nrm, kind, chunk):
    """Doi tia voi cac tru dung co chan tren chan duoi."""
    nc = sc["cyl_c"].shape[1]
    for i0 in range(0, nc, chunk):
        i1 = min(nc, i0 + chunk)
        c = sc["cyl_c"][:, None, i0:i1, :]           # (B,1,K,2)
        r = sc["cyl_r"][:, None, i0:i1]
        z0 = sc["cyl_z"][:, None, i0:i1, 0]
        z1 = sc["cyl_z"][:, None, i0:i1, 1]

        ox = o[:, :, None, 0] - c[..., 0]
        oy = o[:, :, None, 1] - c[..., 1]
        oz = o[:, :, None, 2].expand_as(ox)
        dx = d[:, :, None, 0].expand_as(ox)
        dy = d[:, :, None, 1].expand_as(ox)
        dz = d[:, :, None, 2].expand_as(ox)

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        cc = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * cc
        ok = (disc > 0.0) & (a > EPS) & (r > EPS)
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        a_safe = torch.where(a > EPS, a, torch.full_like(a, EPS))
        tn = (-b - sq) / (2.0 * a_safe)
        tf = (-b + sq) / (2.0 * a_safe)

        def side(tt):
            z = oz + tt * dz
            return ok & (tt > EPS) & (z >= z0) & (z <= z1)

        use_n = side(tn)
        cand = torch.where(use_n, tn, torch.where(side(tf), tf,
                                                  torch.full_like(tn, FAR)))
        # Chan tren (nhin tu tren xuong thay mat tron)
        tcap = (z1 - oz) / torch.where(dz.abs() < EPS,
                                       torch.full_like(dz, EPS), dz)
        px = ox + tcap * dx
        py = oy + tcap * dy
        cap_ok = (tcap > EPS) & (px * px + py * py <= r * r) & (r > EPS)
        cand = torch.where(cap_ok & (tcap < cand), tcap, cand)
        is_cap = cap_ok & (tcap <= cand + EPS)

        best, which = cand.min(dim=-1)
        take = best < t
        if not bool(take.any()):
            continue
        g = which[..., None]

        hx = (ox + best[..., None] * dx).gather(-1, g).squeeze(-1)
        hy = (oy + best[..., None] * dy).gather(-1, g).squeeze(-1)
        rr = r.expand(-1, o.shape[1], -1).gather(-1, g).squeeze(-1)
        cap = is_cap.gather(-1, g).squeeze(-1)
        nx = torch.where(cap, torch.zeros_like(hx), hx / torch.clamp(rr, min=EPS))
        ny = torch.where(cap, torch.zeros_like(hy), hy / torch.clamp(rr, min=EPS))
        nz = torch.where(cap, torch.ones_like(hx), torch.zeros_like(hx))
        n_new = torch.stack((nx, ny, nz), dim=-1)

        col = sc["cyl_col"][:, None, i0:i1, :].expand(-1, o.shape[1], -1, -1)
        a_new = col.gather(-2, g[..., None].expand(-1, -1, -1, 3)).squeeze(-2)

        t = torch.where(take, best, t)
        alb = torch.where(take[..., None], a_new, alb)
        nrm = torch.where(take[..., None], n_new, nrm)
        kind = torch.where(take, torch.full_like(kind, KIND_CYL), kind)
    return t, alb, nrm, kind


def _floor(o, d, sc, t, alb, nrm, kind, tile=0.30):
    """San phang z=0, co o ca ro, va co lo thung (mep ban / cau thang)."""
    dz = d[..., 2]
    down = dz < -EPS
    tf = torch.where(down, -o[..., 2] / torch.where(down, dz,
                                                    torch.full_like(dz, -1.0)),
                     torch.full_like(dz, FAR))
    px = o[..., 0] + tf * d[..., 0]
    py = o[..., 1] + tf * d[..., 1]

    inside = torch.ones_like(down)
    fl = sc["floor"]                                  # (B,4)
    inside &= (px >= fl[:, None, 0]) & (px <= fl[:, None, 2])
    inside &= (py >= fl[:, None, 1]) & (py <= fl[:, None, 3])
    vd = sc["void"]                                   # (B,NV,4)
    if vd.shape[1] > 0:
        hole = ((px[..., None] >= vd[:, None, :, 0]) &
                (px[..., None] <= vd[:, None, :, 2]) &
                (py[..., None] >= vd[:, None, :, 1]) &
                (py[..., None] <= vd[:, None, :, 3])).any(dim=-1)
        inside &= ~hole

    take = down & inside & (tf > EPS) & (tf < t)
    chk = ((torch.floor(px / tile) + torch.floor(py / tile)) % 2.0).abs()
    c0 = sc["floor_col"][:, None, 0, :]
    c1 = sc["floor_col"][:, None, 1, :]
    a_new = torch.where(chk[..., None] < 0.5, c0, c1)
    n_new = torch.zeros_like(d)
    n_new[..., 2] = 1.0

    t = torch.where(take, tf, t)
    alb = torch.where(take[..., None], a_new, alb)
    nrm = torch.where(take[..., None], n_new, nrm)
    kind = torch.where(take, torch.full_like(kind, KIND_FLOOR), kind)
    return t, alb, nrm, kind


def _ceiling(o, d, sc, t, alb, nrm, kind):
    """Tran nha. Khong co tran thi ngua camera len la thay mot mang troi to
    tuong - phong trong nha khong the nhu vay, va bo nao se hoc mot manh
    moi khong ton tai ngoai doi."""
    z = sc["ceil_z"][:, None]
    dz = d[..., 2]
    up = dz > EPS
    tc = torch.where(up, (z - o[..., 2]) / torch.where(up, dz,
                                                       torch.full_like(dz, 1.0)),
                     torch.full_like(dz, FAR))
    take = up & (tc > EPS) & (tc < t)
    px = o[..., 0] + tc * d[..., 0]
    py = o[..., 1] + tc * d[..., 1]
    # o den tran: mot manh moi on dinh cho bo nao doan huong va do cao
    lamp = (((px / 1.6) % 1.0 < 0.34) & ((py / 1.6) % 1.0 < 0.34))
    base = sc["ceil_col"][:, None, :]
    a_new = torch.where(lamp[..., None], torch.clamp(base * 1.7, max=1.0), base)
    n_new = torch.zeros_like(d)
    n_new[..., 2] = -1.0
    t = torch.where(take, tc, t)
    alb = torch.where(take[..., None], a_new.expand_as(alb), alb)
    nrm = torch.where(take[..., None], n_new, nrm)
    kind = torch.where(take, torch.full_like(kind, KIND_CEIL), kind)
    return t, alb, nrm, kind


def trace(o, d, sc, chunk=8, palette=None, want_floor=True):
    """Doi tia. o,d: (B,R,3). Tra ve (t, albedo, normal, kind)."""
    B, R = o.shape[0], o.shape[1]
    dev = o.device
    t = torch.full((B, R), FAR, device=dev, dtype=o.dtype)
    alb = torch.zeros((B, R, 3), device=dev, dtype=o.dtype)
    nrm = torch.zeros((B, R, 3), device=dev, dtype=o.dtype)
    kind = torch.zeros((B, R), device=dev, dtype=torch.long)

    if palette is None:
        palette = sc["palette"]
    t, alb, nrm, kind = _boxes(o, d, sc, t, alb, nrm, kind, chunk, palette)
    if sc["cyl_c"].shape[1] > 0:
        t, alb, nrm, kind = _cylinders(o, d, sc, t, alb, nrm, kind, chunk)
    if want_floor:
        t, alb, nrm, kind = _floor(o, d, sc, t, alb, nrm, kind)
        if "ceil_z" in sc:
            t, alb, nrm, kind = _ceiling(o, d, sc, t, alb, nrm, kind)
    return t, alb, nrm, kind
