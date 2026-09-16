"""Hinh hoc co ban: ray-cast vector hoa tren doan thang va duong tron.

Moi thu LiDAR "nhin thay" trong the gioi deu quy ve hai loai nguyen thuy:
  - doan thang (tuong, vach hoc sac, canh vat can)
  - duong tron  (nguoi di lai, va cac robot khac)

Ray-cast duoc viet bang numpy theo lo (M tia x N doan) vi mot vong quet
Camsense la ~460 tia va trong san co toi 5 robot cung quet.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_pi(a):
    """Dua goc ve khoang (-pi, pi]. Nhan scalar hoac ndarray."""
    return (np.asarray(a, dtype=float) + math.pi) % TWO_PI - math.pi


def wrap_pi_scalar(a):
    return (a + math.pi) % TWO_PI - math.pi


def angle_diff(a, b):
    """Goc a tru goc b, dua ve (-pi, pi]."""
    return wrap_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def rot(vx, vy, theta):
    """Quay vector (vx, vy) di goc theta."""
    c, s = math.cos(theta), math.sin(theta)
    return vx * c - vy * s, vx * s + vy * c


def to_local(px, py, ox, oy, otheta):
    """Doi diem toan cuc (px,py) sang he quy chieu gan tai (ox,oy,otheta)."""
    dx, dy = px - ox, py - oy
    c, s = math.cos(-otheta), math.sin(-otheta)
    return dx * c - dy * s, dx * s + dy * c


def to_world(px, py, ox, oy, otheta):
    """Nguoc lai cua to_local."""
    c, s = math.cos(otheta), math.sin(otheta)
    return ox + px * c - py * s, oy + px * s + py * c


class SegmentSet:
    """Tap doan thang tinh, luu duoi dang mang numpy de ray-cast mot lan."""

    __slots__ = ("ax", "ay", "bx", "by", "ex", "ey", "tag")

    def __init__(self, segments=(), tags=None):
        segs = list(segments)
        n = len(segs)
        self.ax = np.empty(n, dtype=float)
        self.ay = np.empty(n, dtype=float)
        self.bx = np.empty(n, dtype=float)
        self.by = np.empty(n, dtype=float)
        for i, (ax, ay, bx, by) in enumerate(segs):
            self.ax[i], self.ay[i], self.bx[i], self.by[i] = ax, ay, bx, by
        self.ex = self.bx - self.ax
        self.ey = self.by - self.ay
        self.tag = list(tags) if tags is not None else [None] * n

    def __len__(self):
        return self.ax.shape[0]

    @staticmethod
    def concat(parts):
        segs, tags = [], []
        for p in parts:
            for i in range(len(p)):
                segs.append((p.ax[i], p.ay[i], p.bx[i], p.by[i]))
                tags.append(p.tag[i])
        return SegmentSet(segs, tags)

    @staticmethod
    def from_polyline(points, closed=False, tag=None):
        segs = []
        pts = list(points)
        n = len(pts)
        last = n if closed else n - 1
        for i in range(last):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            segs.append((ax, ay, bx, by))
        return SegmentSet(segs, [tag] * len(segs))

    @staticmethod
    def from_rect(cx, cy, w, h, theta=0.0, tag=None):
        hw, hh = 0.5 * w, 0.5 * h
        corners = []
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            corners.append(to_world(lx, ly, cx, cy, theta))
        return SegmentSet.from_polyline(corners, closed=True, tag=tag)


def raycast_segments(ox, oy, angles, seg, max_range):
    """Ban M tia tu (ox,oy) theo `angles`, tra ve khoang cach toi doan gan nhat.

    Tia khong trung gi tra ve `max_range`. Do phuc tap M*N nhung chay
    hoan toan trong numpy nen mot vong quet 460 tia x ~80 doan van re.
    """
    angles = np.asarray(angles, dtype=float)
    if len(seg) == 0:
        return np.full(angles.shape, float(max_range))

    dx = np.cos(angles)[:, None]
    dy = np.sin(angles)[:, None]

    ex = seg.ex[None, :]
    ey = seg.ey[None, :]
    aox = (seg.ax - ox)[None, :]
    aoy = (seg.ay - oy)[None, :]

    denom = dx * ey - dy * ex
    safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)

    t = (aox * ey - aoy * ex) / safe          # doc theo tia
    u = (aox * dy - aoy * dx) / safe          # doc theo doan

    ok = (np.abs(denom) >= 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    t = np.where(ok, t, np.inf)
    best = t.min(axis=1)
    return np.minimum(best, float(max_range))


def raycast_circles(ox, oy, angles, circles, max_range):
    """Nhu tren nhung voi danh sach duong tron [(cx, cy, r), ...]."""
    angles = np.asarray(angles, dtype=float)
    if not circles:
        return np.full(angles.shape, float(max_range))

    arr = np.asarray(circles, dtype=float)
    cx = arr[:, 0][None, :] - ox
    cy = arr[:, 1][None, :] - oy
    r = arr[:, 2][None, :]

    dx = np.cos(angles)[:, None]
    dy = np.sin(angles)[:, None]

    b = dx * cx + dy * cy                       # chieu tam len tia
    c = cx * cx + cy * cy - r * r
    disc = b * b - c
    valid = disc >= 0.0
    sq = np.sqrt(np.where(valid, disc, 0.0))

    t_near = b - sq
    t_far = b + sq
    t = np.where(t_near > 1e-9, t_near, t_far)  # goc tia nam trong hinh tron
    ok = valid & (t > 1e-9)
    t = np.where(ok, t, np.inf)
    best = t.min(axis=1)
    return np.minimum(best, float(max_range))


def raycast(ox, oy, angles, seg, circles, max_range):
    """Gop ca hai loai nguyen thuy."""
    d1 = raycast_segments(ox, oy, angles, seg, max_range)
    d2 = raycast_circles(ox, oy, angles, circles, max_range)
    return np.minimum(d1, d2)


def line_of_sight(x0, y0, x1, y1, seg, circles=(), ignore_radius=0.0):
    """Co duong nhin thang tu (x0,y0) toi (x1,y1) khong?

    Dung cho hong ngoai: chum tia bi chinh hai vach hoc bop lai la he qua
    cua phep kiem tra nay, khong phai luat viet tay.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return True
    ang = math.atan2(dy, dx)
    hit_seg = float(raycast_segments(x0, y0, [ang], seg, dist + 1.0)[0])
    if hit_seg < dist - 1e-4:
        return False
    if circles:
        keep = [c for c in circles if math.hypot(c[0] - x1, c[1] - y1) > ignore_radius]
        if keep:
            hit_c = float(raycast_circles(x0, y0, [ang], keep, dist + 1.0)[0])
            if hit_c < dist - 1e-4:
                return False
    return True


def point_segment_distance(px, py, seg):
    """Khoang cach tu mot diem toi tung doan (tra ve mang do dai N)."""
    if len(seg) == 0:
        return np.empty(0, dtype=float)
    apx = px - seg.ax
    apy = py - seg.ay
    denom = seg.ex * seg.ex + seg.ey * seg.ey
    denom = np.where(denom < 1e-12, 1.0, denom)
    u = np.clip((apx * seg.ex + apy * seg.ey) / denom, 0.0, 1.0)
    qx = seg.ax + u * seg.ex
    qy = seg.ay + u * seg.ey
    return np.hypot(px - qx, py - qy)


def closest_point_on_segments(px, py, seg):
    """Tra ve (khoang cach, chi so doan, diem chieu) gan nhat."""
    d = point_segment_distance(px, py, seg)
    if d.size == 0:
        return float("inf"), -1, (0.0, 0.0)
    i = int(np.argmin(d))
    apx = px - seg.ax[i]
    apy = py - seg.ay[i]
    denom = seg.ex[i] ** 2 + seg.ey[i] ** 2
    u = 0.0 if denom < 1e-12 else min(1.0, max(0.0, (apx * seg.ex[i] + apy * seg.ey[i]) / denom))
    return float(d[i]), i, (seg.ax[i] + u * seg.ex[i], seg.ay[i] + u * seg.ey[i])


def point_in_polygon(px, py, poly):
    """Ray casting chan/le cho da giac kin."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            xint = (xj - xi) * (py - yi) / (yj - yi + 1e-18) + xi
            if px < xint:
                inside = not inside
        j = i
    return inside


def fit_line_angle(xs, ys):
    """Khop duong thang qua dam diem, tra ve goc phap tuyen huong ra ngoai.

    Dung tong binh phuong toan phan (PCA 2 chieu) chu khong phai binh phuong
    toi thieu theo y, vi thanh trong hoc sac co the gan nhu thang dung.
    Tra ve (goc_huong_duong_thang, do_thang 0..1).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return 0.0, 0.0
    mx, my = xs.mean(), ys.mean()
    dx, dy = xs - mx, ys - my
    sxx = float(np.dot(dx, dx))
    syy = float(np.dot(dy, dy))
    sxy = float(np.dot(dx, dy))
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    # tri rieng: phuong sai doc / phuong sai ngang
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, 0.25 * tr * tr - det)
    lam_big = 0.5 * tr + math.sqrt(disc)
    lam_small = 0.5 * tr - math.sqrt(disc)
    straightness = 0.0 if lam_big < 1e-12 else 1.0 - max(0.0, lam_small) / lam_big
    return theta, straightness
