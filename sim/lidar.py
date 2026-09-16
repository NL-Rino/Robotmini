"""Mo hinh Camsense X1/X2.

Ba dac diem cua con LiDAR nay quyet dinh phan con lai:
  1. No khong chup ca vong cung mot luc. 3000 diem/giay, quay ~6 Hz, nghia la
     mot vong mat ~167 ms va trong khoang do xe da di chuyen. Phai gom du
     mot vong moi xu ly.
  2. Bo nhoe (deskew) chi duoc phep dung ODOMETRY. Robot that khong co goc
     that. Lay hieu hai so cung mot he thi phan troi moi triet tieu.
  3. Tra ve milimet, co nhieu, va rot khoang 2% diem.
"""

import math

import numpy as np

from . import params as P
from .geometry import raycast


class Scan:
    """Mot vong quet da gom du, dua ve he quy chieu xe tai thoi diem KET THUC vong."""

    __slots__ = ("ranges", "bearings", "valid", "stamp", "index")

    def __init__(self, ranges, bearings, valid, stamp, index):
        self.ranges = ranges        # met
        self.bearings = bearings    # rad, he quy chieu xe, 0 = mui xe
        self.valid = valid          # bool
        self.stamp = stamp
        self.index = index

    def __len__(self):
        return self.ranges.shape[0]

    def points_xy(self):
        r = self.ranges[self.valid]
        b = self.bearings[self.valid]
        return r * np.cos(b), r * np.sin(b)


class Lidar:
    def __init__(self, seed=0, hz=P.LIDAR_HZ, point_rate=P.LIDAR_POINT_RATE,
                 rmin=P.LIDAR_MIN, rmax=P.LIDAR_MAX,
                 noise=P.LIDAR_NOISE, drop=P.LIDAR_DROP):
        self.rng = np.random.default_rng(seed)
        self.hz = float(hz)
        self.n = max(8, int(round(point_rate / hz)))
        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.drop = drop

        self._step_angle = 2.0 * math.pi / self.n
        self._cursor = 0.0            # vi tri kim quet, don vi "diem"
        self._buf_r = np.zeros(self.n)
        self._buf_ok = np.zeros(self.n, dtype=bool)
        self._buf_ox = np.zeros(self.n)
        self._buf_oy = np.zeros(self.n)
        self._buf_oth = np.zeros(self.n)
        self._rev = 0
        self.scan = None              # vong quet day du gan nhat
        self.new_scan = False         # True dung mot buoc khi vua xong vong

    def reset(self):
        self._cursor = 0.0
        self._buf_ok[:] = False
        self._rev = 0
        self.scan = None
        self.new_scan = False

    def update(self, dt, true_pose, odom_pose, segments, circles, stamp=0.0):
        """Ban them cac diem cua buoc nay. Tra ve Scan moi hoac None."""
        self.new_scan = False
        tx, ty, tth = true_pose
        ox, oy, oth = odom_pose

        advance = self.hz * self.n * dt
        start = self._cursor
        end = start + advance
        i0 = int(math.floor(start))
        i1 = int(math.floor(end))
        self._cursor = end

        count = i1 - i0
        if count <= 0:
            return None

        idx = np.arange(i0, i1)
        local_ang = (idx % self.n) * self._step_angle
        world_ang = tth + local_ang

        d = raycast(tx, ty, world_ang, segments, circles, self.rmax)

        # Nhieu ti le + rot diem + nguong gan.
        d = d * (1.0 + self.rng.normal(0.0, self.noise, d.shape))
        ok = (d >= self.rmin) & (d < self.rmax - 1e-6)
        ok &= self.rng.random(d.shape) >= self.drop
        # Camsense tra ve milimet nguyen.
        d = np.round(d * 1000.0) / 1000.0

        slot = idx % self.n
        self._buf_r[slot] = d
        self._buf_ok[slot] = ok
        self._buf_ox[slot] = ox
        self._buf_oy[slot] = oy
        self._buf_oth[slot] = oth

        finished = (i1 // self.n) > (i0 // self.n)
        if finished:
            self._rev += 1
            self.scan = self._finish(oth, ox, oy, stamp)
            self.new_scan = True
            return self.scan
        return None

    def _finish(self, oth_end, ox_end, oy_end, stamp):
        """Bo nhoe: dua moi diem ve he quy chieu xe tai cuoi vong.

        Chi dung so odometry. Neu o day lo dung goc that thi mo phong se de
        hon thuc te dung mot bac va bo nao se hoc nham.
        """
        r = self._buf_r
        ok = self._buf_ok.copy()
        ang = np.arange(self.n) * self._step_angle

        # Diem trong he odom (he "the gioi theo xe tu nghi").
        px = self._buf_ox + r * np.cos(self._buf_oth + ang)
        py = self._buf_oy + r * np.sin(self._buf_oth + ang)

        dx = px - ox_end
        dy = py - oy_end
        c, s = math.cos(-oth_end), math.sin(-oth_end)
        lx = dx * c - dy * s
        ly = dx * s + dy * c

        rr = np.hypot(lx, ly)
        bb = np.arctan2(ly, lx)
        rr = np.where(ok, rr, self.rmax)
        return Scan(rr, bb, ok, stamp, self._rev)


def fan_ranges(scan, n_fans=P.N_LIDAR_FANS, rmax=P.LIDAR_MAX):
    """Gom vong quet thanh n_fans quat, lay khoang cach GAN NHAT moi quat.

    Quat 0 nam giua mui xe; goc tang theo chieu duong (trai).
    """
    out = np.full(n_fans, rmax, dtype=float)
    if scan is None:
        return out
    width = 2.0 * math.pi / n_fans
    b = (scan.bearings + 0.5 * width) % (2.0 * math.pi)
    idx = np.floor(b / width).astype(int) % n_fans
    r = np.where(scan.valid, scan.ranges, rmax)
    np.minimum.at(out, idx, r)
    return out
