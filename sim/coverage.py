"""Da di toi nhung cho nao roi: luoi o vuong phu len san nha.

De bai co mot khoan: "khi robot toi 1 noi ma lidar bao phu duoc khu vuc moi
thi se duoc cong diem". Cach do dieu do:

Chia san nha thanh o vuong 50 cm. Sau MOI VONG QUET, lay cac diem quet va
danh dau nhung o ma tia di qua - ca o cham vao vat, lan cac o TRONG tren
duong tia (vi nhin thay khoang trong cung la biet cho do co gi). O nao lan
dau tien duoc danh dau thi cong diem mot lan, tu do ve sau khong cong nua.

Chi lay MOT TIA TREN BON va ba diem tren moi tia. Luoi 50 cm thi lay day
500 tia cung khong danh dau them duoc o nao, chi ton them thoi gian.

Cai nay do dung thu "xe da BIET duoc bao nhieu can nha", khong phai "xe da
di duoc bao xa". Dung yen mot cho quay tron thi cung duoc mot it diem -
dung, vi quay mot vong o giua phong that su co nhin thay ca phong - nhung
het rat nhanh, va muon them thi phai buoc qua cua sang phong khac.
"""

import math

import numpy as np

from . import params as P

RAY_STEP = 4                      # lay mot tia tren bon
ALONG = (0.35, 0.7, 1.0)          # ba diem tren moi tia


class Grid:
    """Luoi o vuong phu len mot mat bang."""

    __slots__ = ("x0", "y0", "cell", "nx", "ny", "n")

    def __init__(self, bounds, cell=P.COVER_CELL):
        x0, y0, x1, y1 = bounds
        self.x0, self.y0 = float(x0), float(y0)
        self.cell = float(cell)
        self.nx = max(1, int(math.ceil((x1 - x0) / self.cell)))
        self.ny = max(1, int(math.ceil((y1 - y0) / self.cell)))
        self.n = self.nx * self.ny

    def index(self, xs, ys):
        """Toa do -> chi so o. Ra ngoai luoi thi ghim vao mep."""
        cx = np.clip(((np.asarray(xs) - self.x0) / self.cell).astype(np.int64),
                     0, self.nx - 1)
        cy = np.clip(((np.asarray(ys) - self.y0) / self.cell).astype(np.int64),
                     0, self.ny - 1)
        return cy * self.nx + cx

    def new_cells(self, seen, x, y, th, scan):
        """Danh dau nhung o vong quet nay nhin thay. Tra ve SO O MOI.

        `seen` la mang bool dai `self.n`, bi sua tai cho.
        """
        if scan is None:
            return 0
        v = scan.valid[::RAY_STEP]
        if not v.any():
            return 0
        r = scan.ranges[::RAY_STEP][v]
        b = scan.bearings[::RAY_STEP][v] + th
        cs, sn = np.cos(b), np.sin(b)
        xs = np.concatenate([x + cs * r * f for f in ALONG])
        ys = np.concatenate([y + sn * r * f for f in ALONG])
        idx = self.index(xs, ys)
        truoc = int(seen.sum())
        seen[idx] = True
        return int(seen.sum()) - truoc

    def fraction(self, seen):
        return float(seen.sum()) / float(self.n)
