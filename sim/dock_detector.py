"""Do hoc chu U bang HINH HOC, khong phai bang mang.

Quy trinh:
  1. Cat vong quet thanh cac doan lien tuc (dut khi khoang cach nhay).
  2. Noi hai dau doan thanh mot day cung.
  3. Do do LOM so voi day cung. Hoc thi lom ~31 cm; vat dac thi loi ra.
  4. Co day cung lai cho het phan loi (cai ban, cai chan tu dung canh hoc).
  5. KHOP DUONG THANG qua cac diem sau cung (thanh trong) de lay TRUC.
     Lay truc tu day cung thi lech trung vi ~8 do; lay tu thanh trong con ~2 do.

Ket qua la danh sach ung vien (cu ly, phuong vi, goc truc) trong he quy chieu
xe. Bo do KHONG biet hoc nao la hoc cua minh - viec do phai cam vao moi biet.
"""

import math

import numpy as np

from . import params as P
from .geometry import fit_line_angle, wrap_pi_scalar

# Nguong hinh hoc suy ra tu kich thuoc hoc that.
MIN_WIDTH = 0.22
MAX_WIDTH = 0.60
MIN_DEPTH = 0.20
MAX_DEPTH = 0.42
BREAK_GAP = 0.13          # khoang cach nhay -> cat doan
MERGE_GAP = 0.28          # noi lai hai doan ke nhau neu ho hep hon the nay
MIN_POINTS = 7
CONVEX_TOL = 0.035        # phan loi ra cho phep truoc khi co day cung


class DockCandidate:
    __slots__ = ("range", "bearing", "axis", "width", "depth", "score", "n_points")

    def __init__(self, rng_, bearing, axis, width, depth, score, n_points):
        self.range = rng_        # cu ly toi giua mieng hoc
        self.bearing = bearing   # phuong vi mieng hoc, he xe
        self.axis = axis         # goc TRUC RA cua hoc, he xe
        self.width = width
        self.depth = depth
        self.score = score
        self.n_points = n_points

    def __repr__(self):
        return (f"<Dock r={self.range:.2f} b={math.degrees(self.bearing):.0f} "
                f"axis={math.degrees(self.axis):.0f} w={self.width:.2f} "
                f"d={self.depth:.2f} s={self.score:.2f}>")


def _split_runs(xs, ys, rs, valid):
    """Cat vong quet thanh doan lien tuc.

    Giu ca doan RAT NGAN (2 diem). Khi nhin thang vao hoc, hai canh vat o
    mieng chi con 2-3 diem; vut chung di la mat luon hai dau day cung va do
    sau do duoc bi hut mot nua. MIN_POINTS chi ap o buoc phan tich nhom.

    Viet bang numpy chu khong phai vong lap Python: ham nay chay mot lan moi
    vong quet tren 500 diem va la mot trong nhung cho ton nhat cua mo phong.
    """
    n = xs.shape[0]
    if n < 2:
        return []
    v = np.asarray(valid, dtype=bool)
    # Mot diem mo dau doan moi khi: no la diem dau tien, hoac chinh no hong,
    # hoac diem truoc no hong, hoac khoang cach nhay vot.
    cut = np.empty(n, dtype=bool)
    cut[0] = True
    cut[1:] = (~v[1:]) | (~v[:-1]) | (np.abs(rs[1:] - rs[:-1]) > BREAK_GAP)

    bounds = np.nonzero(cut)[0]
    pieces = np.split(np.arange(n), bounds[1:])
    runs = [p for p in pieces if p.size >= 2 and v[p[0]]]

    # Vong quet la vong tron: noi doan dau voi doan cuoi neu lien tuc.
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        if abs(rs[0] - rs[n - 1]) <= BREAK_GAP:
            runs[0] = np.concatenate((runs[-1], runs[0]))
            runs.pop()
    return runs


def _analyse_run(idx, xs, ys):
    """Do lom cua mot doan, co day cung cho het loi, roi lay truc."""
    px = xs[idx]
    py = ys[idx]

    lo, hi = 0, px.shape[0] - 1
    for _ in range(6):
        if hi - lo + 1 < MIN_POINTS:
            return None
        ax, ay = px[lo], py[lo]
        bx, by = px[hi], py[hi]
        cx, cy = bx - ax, by - ay
        clen = math.hypot(cx, cy)
        if clen < 1e-6:
            return None
        # Phap tuyen cua day cung, chon chieu HUONG RA XA cam bien.
        nx, ny = -cy / clen, cx / clen
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        if nx * mx + ny * my < 0.0:
            nx, ny = -nx, -ny

        depth = (px[lo:hi + 1] - ax) * nx + (py[lo:hi + 1] - ay) * ny

        # Phan loi ra (am) o hai dau -> co day cung vao.
        trimmed = False
        if depth[0] < -CONVEX_TOL or depth[1] < -CONVEX_TOL:
            lo += max(1, int(0.08 * (hi - lo)))
            trimmed = True
        if depth[-1] < -CONVEX_TOL or depth[-2] < -CONVEX_TOL:
            hi -= max(1, int(0.08 * (hi - lo)))
            trimmed = True
        if not trimmed:
            break

    ax, ay = px[lo], py[lo]
    bx, by = px[hi], py[hi]
    cx, cy = bx - ax, by - ay
    width = math.hypot(cx, cy)
    if width < 1e-6:
        return None
    nx, ny = -cy / width, cx / width
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    if nx * mx + ny * my < 0.0:
        nx, ny = -nx, -ny

    sel = slice(lo, hi + 1)
    qx, qy = px[sel], py[sel]
    depth = (qx - ax) * nx + (qy - ay) * ny
    dmax = float(depth.max())
    dmin = float(depth.min())

    if not (MIN_WIDTH <= width <= MAX_WIDTH):
        return None
    if not (MIN_DEPTH <= dmax <= MAX_DEPTH):
        return None
    if dmin < -CONVEX_TOL * 2.0:
        return None

    # Thanh trong = cac diem sau cung.
    deep = depth >= 0.72 * dmax
    if int(deep.sum()) < 4:
        return None
    line_ang, straight = fit_line_angle(qx[deep], qy[deep])
    if straight < 0.35:
        return None

    # Kiem tra cau truc quan trong nhat: hoc that co THANH TRONG PHANG rong
    # ~31 cm. Mot goc tuong, hay khe giua cai tu voi buc tuong, cung lom va
    # cung du rong du sau, nhung cho sau nhat cua no la mot DIEM chu khong
    # phai mot mat phang - do trai cua dam diem sau cung gan bang 0.
    # Thieu phep thu nay thi bo do bao gia diem 0,5-0,6 o khap goc phong va
    # xe di cam vao tuong.
    bl = math.cos(line_ang) * qx[deep] + math.sin(line_ang) * qy[deep]
    back_extent = float(bl.max() - bl.min())
    if back_extent < 0.5 * P.DOCK_CAVITY_W or back_extent > 1.5 * P.DOCK_CAVITY_W:
        return None

    # Truc ra = phap tuyen cua thanh trong, huong ve phia cam bien.
    axis = wrap_pi_scalar(line_ang + math.pi / 2.0)
    bxw, byw = float(qx[deep].mean()), float(qy[deep].mean())
    if math.cos(axis) * (-bxw) + math.sin(axis) * (-byw) < 0.0:
        axis = wrap_pi_scalar(axis + math.pi)

    # Tam mieng hoc = tam thanh trong day ra mot doan bang do sau DANH DINH.
    # Do sau do duoc (dmax) bi hut khi nhin cheo, chi dung de cham diem;
    # do sau that cua hoc thi biet truoc vi hoc la do vat co san.
    mouth_x = bxw + P.DOCK_CAVITY_D * math.cos(axis)
    mouth_y = byw + P.DOCK_CAVITY_D * math.sin(axis)

    rng_ = math.hypot(mouth_x, mouth_y)
    bearing = math.atan2(mouth_y, mouth_x)

    # Cham diem: cang giong kich thuoc that cang cao.
    w_err = abs(width - (P.DOCK_CAVITY_W + 2 * P.DOCK_CHAMFER * 0.5)) / 0.25
    d_err = abs(dmax - P.DOCK_CAVITY_D) / 0.25
    b_err = abs(back_extent - P.DOCK_CAVITY_W) / 0.20
    score = (max(0.0, 1.0 - 0.35 * w_err - 0.35 * d_err - 0.30 * b_err)
             * (0.35 + 0.65 * straight))

    return DockCandidate(rng_, bearing, axis, width, dmax, score,
                         int(hi - lo + 1))


def _gap(i, j, xs, ys):
    return math.hypot(float(xs[i] - xs[j]), float(ys[i] - ys[j]))


def _groups(runs, xs, ys):
    """Sinh cac nhom diem de phan tich.

    Ngoai tung doan rieng, con noi cac doan KE NHAU ma khe ho giua chung hep.
    Can cai nay vi hai thanh trong cua hoc nhin gan nhu song song voi tia:
    ca thanh chi duoc 2-3 diem, du de lam khoang cach nhay vot va cat doan
    ngay giua long hoc. Khe ho toi thanh sau hoc thi rong hon han (>= do sau
    hoc) nen khong bi noi nham.
    """
    out = list(runs)
    n = len(runs)
    if n < 2:
        return out
    for start in range(n):
        chain = runs[start]
        prev = start
        for k in range(1, min(n, 4)):
            j = (start + k) % n
            if j == start:
                break
            if _gap(runs[prev][-1], runs[j][0], xs, ys) > MERGE_GAP:
                break
            chain = np.concatenate((chain, runs[j]))
            prev = j
            if chain.size > 400:
                break
            out.append(chain)
    return out


def detect(scan, max_candidates=P.N_DOCK_CANDIDATES):
    """Tra ve toi da `max_candidates` ung vien, sap theo diem giam dan."""
    if scan is None:
        return []
    valid = scan.valid
    if int(valid.sum()) < MIN_POINTS:
        return []
    r = scan.ranges
    b = scan.bearings
    xs = r * np.cos(b)
    ys = r * np.sin(b)

    runs = _split_runs(xs, ys, r, valid)
    out = []
    for gi in _groups(runs, xs, ys):
        if gi.size < MIN_POINTS:
            continue
        # Loc tho truoc khi phan tich: hoc chi rong 40 cm sau 31 cm, nen ca
        # cum diem cua no khong the trai qua mot o vuong 0,9 m. Phep thu nay
        # O(1) va vut di khoang chin phan muoi so nhom, con phan tich day du
        # thi ton hon nhieu lan.
        gx, gy = xs[gi], ys[gi]
        if (gx.max() - gx.min()) > 0.9 or (gy.max() - gy.min()) > 0.9:
            continue
        chord = math.hypot(gx[-1] - gx[0], gy[-1] - gy[0])
        if chord < 0.8 * MIN_WIDTH or chord > 2.0 * MAX_WIDTH:
            continue
        c = _analyse_run(gi, xs, ys)
        if c is not None:
            out.append(c)
    out.sort(key=lambda c: -c.score)

    # Gop cac ung vien trung nhau (cung mot hoc, nhieu cach cat doan).
    merged = []
    for c in out:
        dup = False
        for m in merged:
            dx = c.range * math.cos(c.bearing) - m.range * math.cos(m.bearing)
            dy = c.range * math.sin(c.bearing) - m.range * math.sin(m.bearing)
            if math.hypot(dx, dy) < 0.30:
                dup = True
                break
        if not dup:
            merged.append(c)
        if len(merged) >= max_candidates:
            break
    return merged
