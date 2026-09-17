# -*- coding: utf-8 -*-
"""Mo phong ban v1 chay THEO LO.

Cung can phong, cung vat ly, cung 48 dau vao nhu `sim/` - chi khac mot dieu:
144 con xe buoc cung mot luc thay vi lan luot tung con.

Do la toan bo bi mat cua viec dung duoc GPU. Ban `sim/` khong cham vi numpy
cham; no cham vi moi buoc la vai chuc phep tinh ti hon tren mang 64 phan tu,
va o kich thuoc do thi tien goi ham dat hon tien tinh toan. Gop 144 xe lai
thi cung tung ay phep tinh nhung moi phep chay tren mang 1,4 trieu phan tu -
CPU nhanh len vai lan, con GPU thi dung cho no sinh ra.
"""

import math

import torch

from sim import params as P

from . import ops
from .ops import hypot

N_LIDAR = 500          # diem moi vong, giong `sim/lidar.py`


# Mot lan ban tia dung (R, K, S) so trung gian: 1.152 xe x 60 tia x 52 doan
# la 3,6 trieu o cho MOI bien tam - va co gan chuc bien tam. Tren CPU thi
# cai do khong lot noi vao bo nho dem, va phep tinh bi nghen o duong truyen
# bo nho chu khong phai o so hoc. Cat theo doan thanh tung manh nho roi giu
# mot cai nho dan la re hon 1,6 lan tren CPU; tren GPU thi khong khac may.
RAY_TILE = 2 << 20


def raycast(ox, oy, ang, seg, seg_e, rmax):
    """Ban tia. ox,oy: (R,)  ang: (R,K)  seg: (R,S,4) -> (R,K)."""
    dx = torch.cos(ang)[:, :, None]
    dy = torch.sin(ang)[:, :, None]
    n_seg = seg.shape[1]
    step = max(1, min(n_seg, RAY_TILE // max(1, ang.numel())))
    best = torch.full(ang.shape, float("inf"), device=ang.device,
                      dtype=ang.dtype)
    for i in range(0, n_seg, step):
        sl = slice(i, i + step)
        aox = (seg[:, sl, 0] - ox[:, None])[:, None, :]
        aoy = (seg[:, sl, 1] - oy[:, None])[:, None, :]
        ex = seg_e[:, sl, 0][:, None, :]
        ey = seg_e[:, sl, 1][:, None, :]

        den = dx * ey - dy * ex
        nu = aox * dy - aoy * dx
        nt = aox * ey - aoy * ex
        # `u` chi dung de kiem 0 <= u <= 1, khong can gia tri. Mot phep chia
        # tren mang 3,6 trieu o dat hon ca chuc phep so sanh, nen kiem dau
        # va do lon thay vi chia: u trong [0,1] <=> nu cung dau den va
        # |nu| <= |den|. Bo duoc mot trong hai phep chia cua moi tia.
        ad = den.abs()
        ok = (ad >= 1e-9) & (nu * den >= 0.0) & (nu.abs() <= ad) \
            & (nt * den > 0.0)
        safe = torch.where(ok, den, torch.ones_like(den))
        t = torch.where(ok, nt / safe, torch.full_like(nt, float("inf")))
        best = torch.minimum(best, t.min(dim=2).values)
    return torch.clamp(best, max=rmax)


def raycast_circles(ox, oy, ang, circ, rmax):
    """circ: (R,M,3) x,y,r. Ban kinh 0 = khong co."""
    cx = (circ[..., 0] - ox[:, None])[:, None, :]
    cy = (circ[..., 1] - oy[:, None])[:, None, :]
    rr = circ[..., 2][:, None, :]
    dx = torch.cos(ang)[:, :, None]
    dy = torch.sin(ang)[:, :, None]
    b = dx * cx + dy * cy
    c = cx * cx + cy * cy - rr * rr
    disc = b * b - c
    okd = (disc >= 0.0) & (rr > 1e-6)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    tn = b - sq
    tf = b + sq
    t = torch.where(tn > 1e-6, tn, tf)
    t = torch.where(okd & (t > 1e-6), t, torch.full_like(t, float("inf")))
    return torch.clamp(t.min(dim=2).values, max=rmax)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class BatchSim:
    """Vong lap the gioi theo lo. Mot chi so = mot xe."""

    def __init__(self, world, device, seed=0, dt=P.DT, copies=1):
        self.w = world
        self.device = device
        self.dt = dt
        self.R = world.R
        self.copies = int(copies)
        assert self.R % self.copies == 0, "so xe phai chia het cho so ban sao"
        self.unit = self.R // self.copies
        self.rng = ops.HostRng(seed + 991, device)
        import random
        self._pyrng = random.Random(seed + 4242)

        z = lambda *s: torch.zeros(*s, device=device)
        self.x, self.y, self.th = z(self.R), z(self.R), z(self.R)
        self.ox, self.oy, self.oth = z(self.R), z(self.R), z(self.R)
        self.vl, self.vr = z(self.R), z(self.R)
        self.cmd = z(self.R, 2)
        self.v, self.wv = z(self.R), z(self.R)
        self.batt = torch.ones(self.R, device=device)
        self.bump = z(self.R)
        self.t = 0.0
        self.station = z(self.R, 3)
        b = lambda: torch.zeros(self.R, dtype=torch.bool, device=device)
        self.in_slot, self.id_ok, self.charging = b(), b(), b()
        self.stranded, self.fallen, self.low_lamp = b(), b(), b()

        g = self._randn(3)
        common = 1.0 + g[:, 0] * P.ODOM_SCALE_ERR
        diff = g[:, 1] * P.ODOM_DIFF_ERR
        self.od_l = common * (1.0 - diff)
        self.od_r = common * (1.0 + diff)
        self.od_drift = g[:, 2] * P.ODOM_DRIFT

        # LiDAR: moi xe mot bo dem vong quet. Tat ca cac xe quay CUNG PHA
        # (giong ban v1: kim quet deu bat dau tu 0), nen chi so diem moi buoc
        # la chung cho ca lo - do la ly do gom duoc thanh mot phep tinh.
        self._cursor = 0.0
        self._rev = 0
        self.lr = torch.zeros(self.R, N_LIDAR, device=device)
        self.lok = torch.zeros(self.R, N_LIDAR, dtype=torch.bool, device=device)
        self.lox = torch.zeros(self.R, N_LIDAR, device=device)
        self.loy = torch.zeros(self.R, N_LIDAR, device=device)
        self.loth = torch.zeros(self.R, N_LIDAR, device=device)
        self.scan_r = torch.full((self.R, N_LIDAR), P.LIDAR_MAX, device=device)
        self.scan_b = torch.zeros(self.R, N_LIDAR, device=device)
        self.scan_ok = torch.zeros(self.R, N_LIDAR, dtype=torch.bool,
                                   device=device)
        self.new_scan = False
        self._lidar_step = 2.0 * math.pi / N_LIDAR
        # Den goi da bi nhat, theo TUNG XE. Ban v1 tat cai den tren mat bang,
        # nhung o day mot mat bang duoc P ban sao dung chung: ban sao nao toi
        # truoc se tat den cua ca P ban, va chung so ngau nhien mat nghia.
        self.beacon_off = torch.zeros(self.R, self.w.beacon.shape[1],
                                      dtype=torch.bool, device=device)

    # --------------------------------------------------- chung so ngau nhien
    def _randn(self, *tail):
        """Nhieu chuan (R, *tail) nhung GIONG HET NHAU giua cac ban sao.

        Day la chung so ngau nhien. Neu moi ban sao tu boc nhieu rieng thi
        chenh lech diem giua hai bo trong so phan lon la chenh lech VAN MAY,
        va ES se xep hang theo van may.
        """
        u = self.rng.randn(self.unit, *tail)
        return u.repeat(self.copies, *([1] * len(tail)))

    def _rand(self, *tail):
        u = self.rng.rand(self.unit, *tail)
        return u.repeat(self.copies, *([1] * len(tail)))

    # ------------------------------------------------------------------ dat xe
    def place(self, idx, x, y, th, battery=None, station=None, drift=0.0):
        # `a[idx] = b` khong phai may nao cung lam duoc; `index_copy_` thi co.
        th = wrap(th)
        for a, v in ((self.x, x), (self.y, y), (self.th, th),
                     (self.ox, x), (self.oy, y), (self.oth, th)):
            a.index_copy_(0, idx, v)
        for a in (self.vl, self.vr, self.cmd, self.bump):
            a.index_fill_(0, idx, 0.0)
        for a in (self.stranded, self.fallen, self.in_slot, self.id_ok,
                  self.charging):
            a.index_fill_(0, idx, False)
        if battery is not None:
            self.batt.index_copy_(0, idx, battery)
            self.low_lamp = ops.put_rows(self.low_lamp, idx,
                                         battery < P.BATT_LOW)
        hp = self.w.home_pose().index_select(0, idx)
        if station is None:
            station = hp
        if drift is not None and torch.is_tensor(drift):
            # Nguoi goi phai dua ca `drift` lan goc san neu muon chung so
            # ngau nhien; o day chi boc them khi khong co.
            ang = (self.rng.rand(len(idx)) * 2 - 1) * math.pi
            station = torch.stack((station[:, 0] + drift * torch.cos(ang),
                                   station[:, 1] + drift * torch.sin(ang),
                                   station[:, 2] + self.rng.randn(len(idx))
                                   * 0.12 * drift), dim=-1)
        self.station.index_copy_(0, idx, station)
        # Dat lai xe la dat lai ca LiDAR: xe vua bat len thi chua thay gi.
        # Khong xoa thi vai buoc dau tien xe con nhin bang vong quet cua lan
        # danh gia TRUOC - mot can phong khac han.
        self.lok.index_fill_(0, idx, False)
        self.scan_ok.index_fill_(0, idx, False)
        self.scan_r.index_fill_(0, idx, P.LIDAR_MAX)

    # ------------------------------------------------------------------ vat ly
    def _circles(self):
        """Nguoi di lai + cac xe KHAC trong cung the gioi."""
        return self.w.mover

    def _collide(self):
        seg = self.w.seg
        e = self.w.seg_e
        hit = torch.zeros(self.R, dtype=torch.bool, device=self.device)
        for _ in range(4):
            apx = self.x[:, None] - seg[..., 0]
            apy = self.y[:, None] - seg[..., 1]
            den = (e * e).sum(-1).clamp(min=1e-12)
            u = ((apx * e[..., 0] + apy * e[..., 1]) / den).clamp(0.0, 1.0)
            qx = seg[..., 0] + u * e[..., 0]
            qy = seg[..., 1] + u * e[..., 1]
            dx = self.x[:, None] - qx
            dy = self.y[:, None] - qy
            d = hypot(dx, dy)
            pen = torch.where(d < P.BODY_RADIUS, P.BODY_RADIUS - d,
                              torch.zeros_like(d))
            best, wi = pen.max(dim=1)
            m = best > 1e-6
            if not bool(m.any()):
                break
            g = wi[:, None]
            dd = d.gather(1, g).squeeze(1).clamp(min=1e-6)
            nx = dx.gather(1, g).squeeze(1) / dd
            ny = dy.gather(1, g).squeeze(1) / dd
            self.x = torch.where(m, self.x + nx * best, self.x)
            self.y = torch.where(m, self.y + ny * best, self.y)
            hit |= m

        c = self._circles()
        dx = self.x[:, None] - c[..., 0]
        dy = self.y[:, None] - c[..., 1]
        d = hypot(dx, dy).clamp(min=1e-6)
        over = (P.BODY_RADIUS + c[..., 2]) - d
        over = torch.where(c[..., 2] > 1e-6, over, torch.full_like(over, -1.0))
        best, wi = over.max(dim=1)
        m = best > 1e-6
        if bool(m.any()):
            g = wi[:, None]
            nx = dx.gather(1, g).squeeze(1) / d.gather(1, g).squeeze(1)
            ny = dy.gather(1, g).squeeze(1) / d.gather(1, g).squeeze(1)
            self.x = torch.where(m, self.x + nx * best, self.x)
            self.y = torch.where(m, self.y + ny * best, self.y)
            hit |= m
        return hit

    def on_floor(self, px, py):
        f = self.w.floor
        ok = (px >= f[:, 0]) & (px <= f[:, 2]) & (py >= f[:, 1]) & (py <= f[:, 3])
        v = self.w.void
        hole = ((px[:, None] >= v[..., 0]) & (px[:, None] <= v[..., 2]) &
                (py[:, None] >= v[..., 1]) & (py[:, None] <= v[..., 3])
                ).any(dim=1)
        return ok & ~hole

    def cliff(self):
        out = []
        for s in (1.0, -1.0):
            a = self.th + s * P.CLIFF_ANGLE
            out.append(~self.on_floor(self.x + P.CLIFF_RADIUS * torch.cos(a),
                                      self.y + P.CLIFF_RADIUS * torch.sin(a)))
        return out[0], out[1]

    def contact(self):
        rx = self.x - P.BODY_RADIUS * torch.cos(self.th)
        ry = self.y - P.BODY_RADIUS * torch.sin(self.th)
        from .world import dock_local
        lx, ly = dock_local(rx, ry, self.w.dock)
        h = (self.in_slot.to(self.x.dtype) * P.CONTACT_HYST)[:, None]
        touch = ((lx <= -P.DOCK_CAVITY_D + P.CONTACT_LONG_TOL + h)
                 & (ly.abs() <= P.CONTACT_LAT_TOL + h)
                 & (lx > -P.DOCK_CAVITY_D - 0.12))
        ins, which = ops.any_and_first(touch)
        code = self.w.dock_code.gather(1, which[:, None]).squeeze(1)
        pw = self.w.dock_powered.gather(1, which[:, None]).squeeze(1)
        return ins, ins & pw & (code == self.w.my_code), which

    def in_cavity(self):
        from .world import dock_local
        lx, ly = dock_local(self.x, self.y, self.w.dock)
        return ((lx <= 0.02) & (lx >= -P.DOCK_CAVITY_D - 0.02)
                & (ly.abs() <= 0.5 * P.DOCK_CAVITY_W + 0.02)).any(dim=1)

    # ------------------------------------------------------------------ mot buoc
    def step(self, action):
        a = torch.clamp(action, -1.0, 1.0)
        alive = ~(self.stranded | self.fallen)
        cl = torch.where(alive, a[:, 0], torch.zeros_like(a[:, 0]))
        cr = torch.where(alive, a[:, 1], torch.zeros_like(a[:, 1]))
        self.cmd = torch.stack((cl, cr), dim=-1)

        k = 1.0 - math.exp(-self.dt / P.MOTOR_TAU)
        self.vl = self.vl + (cl * P.V_MAX - self.vl) * k
        self.vr = self.vr + (cr * P.V_MAX - self.vr) * k
        v = 0.5 * (self.vl + self.vr)
        wv = (self.vr - self.vl) / P.WHEEL_BASE
        self.v, self.wv = v, wv

        th1 = self.th + wv * self.dt
        small = wv.abs() < 1e-6
        r = v / torch.where(small, torch.ones_like(wv), wv)
        self.x = torch.where(small, self.x + v * torch.cos(self.th) * self.dt,
                             self.x + r * (torch.sin(th1) - torch.sin(self.th)))
        self.y = torch.where(small, self.y + v * torch.sin(self.th) * self.dt,
                             self.y - r * (torch.cos(th1) - torch.cos(self.th)))
        self.th = wrap(th1)

        dl = self.vl * self.dt * self.od_l
        dr = self.vr * self.dt * self.od_r
        dv = 0.5 * (dl + dr)
        dw = (dr - dl) / P.WHEEL_BASE + self.od_drift * self.dt
        self.oth = wrap(self.oth + dw)
        self.ox = self.ox + dv * torch.cos(self.oth)
        self.oy = self.oy + dv * torch.sin(self.oth)

        hit = self._collide()
        was_bump = self.bump > 0.5
        self.bump = torch.where(hit, torch.ones_like(self.bump),
                                (self.bump - self.dt * 4.0).clamp(min=0.0))
        new_bump = hit & ~was_bump

        was_fallen = self.fallen
        self.fallen = self.fallen | ~self.on_floor(self.x, self.y)
        just_fell = self.fallen & ~was_fallen
        self.vl = torch.where(self.fallen, torch.zeros_like(self.vl), self.vl)
        self.vr = torch.where(self.fallen, torch.zeros_like(self.vr), self.vr)

        ins, idok, _wd = self.contact()
        was_slot, was_charge = self.in_slot, self.charging
        self.in_slot, self.id_ok, self.charging = ins, idok, idok

        prev_b = self.batt
        speed = (v.abs() / P.V_MAX + 0.35 * wv.abs() / P.W_MAX).clamp(max=1.0)
        drain = P.BATT_IDLE_DRAIN + P.BATT_MOVE_DRAIN * speed
        drain = torch.where(self.stranded | self.fallen,
                            torch.zeros_like(drain), drain)
        self.batt = torch.where(
            self.charging, (self.batt + P.BATT_CHARGE_RATE * self.dt).clamp(max=1.0),
            (self.batt - drain * self.dt).clamp(min=0.0))
        d_batt = (self.batt - prev_b).clamp(min=0.0)

        was_flat = self.stranded
        self.stranded = torch.where(self.batt <= 0.0,
                                    torch.ones_like(self.stranded),
                                    torch.where(self.batt > 0.05,
                                                torch.zeros_like(self.stranded),
                                                self.stranded))
        just_flat = self.stranded & ~was_flat
        self.low_lamp = torch.where(
            self.batt < P.BATT_LOW, torch.ones_like(self.low_lamp),
            torch.where(self.batt > P.BATT_LOW + 0.03,
                        torch.zeros_like(self.low_lamp), self.low_lamp))

        just_charge = self.charging & ~was_charge
        if bool(just_charge.any()):
            # Nho tram trong HE ODOM, dung y ban v1: ghi lai chinh cho dang
            # cam. Ghi toa do that thi phan odom troi khong triet tieu nua va
            # bo nho tram tro nen chinh xac hon doi thuc.
            hp = self.w.home_pose()
            st = torch.stack((self.ox, self.oy,
                              wrap(self.oth + (hp[:, 2] - self.th))), dim=-1)
            self.station = torch.where(just_charge[:, None], st, self.station)

        self.t += self.dt
        self.w.step_dynamics(self.dt, self._pyrng)
        return dict(bump=new_bump, fell=just_fell, flat=just_flat,
                    d_batt=d_batt, wrong=ins & ~idok & ~was_slot,
                    latch=just_charge)

    # ------------------------------------------------------------------ LiDAR
    def lidar_step(self):
        """Ban them cac diem cua buoc nay; tra ve True khi vua xong mot vong.

        Giu nguyen mo hinh cua ban v1: khong chup ca vong cung luc, va bo
        nhoe CHI bang odometry - robot that khong co goc that.
        """
        self.new_scan = False
        start = self._cursor
        end = start + P.LIDAR_HZ * N_LIDAR * self.dt
        i0, i1 = int(math.floor(start)), int(math.floor(end))
        self._cursor = end
        if i1 <= i0:
            return False
        idx = torch.arange(i0, i1, device=self.device)
        local = (idx % N_LIDAR).float() * self._lidar_step
        ang = self.th[:, None] + local[None, :]

        d = raycast(self.x, self.y, ang, self.w.seg, self.w.seg_e, P.LIDAR_MAX)
        dc = raycast_circles(self.x, self.y, ang, self._circles(), P.LIDAR_MAX)
        d = torch.minimum(d, dc)
        d = d * (1.0 + self._randn(d.shape[1]) * P.LIDAR_NOISE)
        ok = (d >= P.LIDAR_MIN) & (d < P.LIDAR_MAX - 1e-6)
        ok &= self._rand(d.shape[1]) >= P.LIDAR_DROP
        d = torch.round(d * 1000.0) / 1000.0

        slot = (idx % N_LIDAR)
        one = torch.ones(1, slot.numel(), device=self.device)
        self.lr.index_copy_(1, slot, d)
        self.lok.index_copy_(1, slot, ok)
        self.lox.index_copy_(1, slot, self.ox[:, None] * one)
        self.loy.index_copy_(1, slot, self.oy[:, None] * one)
        self.loth.index_copy_(1, slot, self.oth[:, None] * one)

        if (i1 // N_LIDAR) > (i0 // N_LIDAR):
            self._rev += 1
            self._finish_scan()
            self.new_scan = True
            return True
        return False

    def _finish_scan(self):
        ang = (torch.arange(N_LIDAR, device=self.device).float()
               * self._lidar_step)[None, :]
        px = self.lox + self.lr * torch.cos(self.loth + ang)
        py = self.loy + self.lr * torch.sin(self.loth + ang)
        dx = px - self.ox[:, None]
        dy = py - self.oy[:, None]
        c = torch.cos(-self.oth)[:, None]
        s = torch.sin(-self.oth)[:, None]
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        rr = hypot(lx, ly)
        self.scan_r = torch.where(self.lok, rr,
                                  torch.full_like(rr, P.LIDAR_MAX))
        self.scan_b = torch.atan2(ly, lx)
        self.scan_ok = self.lok.clone()

    def fans(self):
        wdt = 2.0 * math.pi / P.N_LIDAR_FANS
        b = (self.scan_b + 0.5 * wdt) % (2.0 * math.pi)
        i = (b / wdt).floor().long().clamp(0, P.N_LIDAR_FANS - 1)
        r = torch.where(self.scan_ok, self.scan_r,
                        torch.full_like(self.scan_r, P.LIDAR_MAX))
        return ops.fan_min(i, r, P.N_LIDAR_FANS, P.LIDAR_MAX, self.device)

    # ------------------------------------------------------------------ hong ngoai
    def ir_dock(self):
        pose = self.w.dock
        ex = pose[..., 0] - (P.DOCK_CAVITY_D - 0.002) * torch.cos(pose[..., 2])
        ey = pose[..., 1] - (P.DOCK_CAVITY_D - 0.002) * torch.sin(pose[..., 2])
        return self._ir(ex, ey, self.w.dock_powered)

    def ir_beacon(self):
        b = self.w.beacon
        return self._ir(b[..., 0], b[..., 1], (b[..., 2] > 0.5) & ~self.beacon_off)

    def take_beacon(self, radius):
        """Xe nao vua cham vao mot den goi dang sang; tat den do cho RIENG xe do."""
        b = self.w.beacon
        near = ((b[..., 2] > 0.5) & ~self.beacon_off
                & (hypot(b[..., 0] - self.x[:, None],
                               b[..., 1] - self.y[:, None]) < radius))
        got, wi = ops.any_and_first(near)
        # moi buoc chi nhat MOT den, giong ban v1
        self.beacon_off = self.beacon_off | ops.one_hot_at(near, wi)
        return got

    def _ir(self, ex, ey, on):
        dx = ex - self.x[:, None]
        dy = ey - self.y[:, None]
        dist = hypot(dx, dy)
        bear = wrap(torch.atan2(dy, dx) - self.th[:, None])
        ok = on & (dist < P.IR_RANGE) & (bear.abs() < P.IR_FOV)
        big = torch.where(ok, dist, torch.full_like(dist, 1e9))
        best, wi = big.min(dim=1)
        seen = best < 1e8
        g = wi[:, None]
        ang = torch.atan2(dy.gather(1, g).squeeze(1), dx.gather(1, g).squeeze(1))
        hit = raycast(self.x, self.y, ang[:, None], self.w.seg, self.w.seg_e,
                      1e4)[:, 0]
        hitc = raycast_circles(self.x, self.y, ang[:, None], self._circles(),
                               1e4)[:, 0]
        seen = seen & (torch.minimum(hit, hitc) >= best - 0.02)
        return seen, wrap(ang - self.th)
