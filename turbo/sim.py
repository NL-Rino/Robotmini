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

N_LIDAR = 500          # diem moi vong, giong `sim/lidar.py`


def raycast(ox, oy, ang, seg, seg_e, rmax):
    """Ban tia. ox,oy: (R,)  ang: (R,K)  seg: (R,S,4) -> (R,K)."""
    dx = torch.cos(ang)[:, :, None]
    dy = torch.sin(ang)[:, :, None]
    aox = (seg[..., 0] - ox[:, None])[:, None, :]
    aoy = (seg[..., 1] - oy[:, None])[:, None, :]
    ex = seg_e[..., 0][:, None, :]
    ey = seg_e[..., 1][:, None, :]

    den = dx * ey - dy * ex
    bad = den.abs() < 1e-9
    safe = torch.where(bad, torch.full_like(den, 1e-9), den)
    u = (aox * dy - aoy * dx) / safe
    t = (aox * ey - aoy * ex) / safe
    ok = ~bad & (u >= 0.0) & (u <= 1.0) & (t > 1e-6)
    t = torch.where(ok, t, torch.full_like(t, float("inf")))
    return torch.clamp(t.min(dim=2).values, max=rmax)


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

    def __init__(self, world, device, seed=0, dt=P.DT):
        self.w = world
        self.device = device
        self.dt = dt
        self.R = world.R
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 991)
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

        g = torch.randn(self.R, 3, device=device, generator=self.gen)
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

    # ------------------------------------------------------------------ dat xe
    def place(self, idx, x, y, th, battery=None, station=None, drift=0.0):
        self.x[idx], self.y[idx], self.th[idx] = x, y, wrap(th)
        self.ox[idx], self.oy[idx], self.oth[idx] = x, y, wrap(th)
        self.vl[idx] = 0.0
        self.vr[idx] = 0.0
        self.cmd[idx] = 0.0
        self.bump[idx] = 0.0
        self.stranded[idx] = False
        self.fallen[idx] = False
        self.in_slot[idx] = False
        self.id_ok[idx] = False
        self.charging[idx] = False
        if battery is not None:
            self.batt[idx] = battery
            self.low_lamp[idx] = self.batt[idx] < P.BATT_LOW
        hp = self.w.home_pose()[idx]
        if station is None:
            station = hp
        if drift is not None and torch.is_tensor(drift):
            ang = (torch.rand(len(idx), device=self.device,
                              generator=self.gen) * 2 - 1) * math.pi
            station = torch.stack((station[:, 0] + drift * torch.cos(ang),
                                   station[:, 1] + drift * torch.sin(ang),
                                   station[:, 2] + torch.randn(
                                       len(idx), device=self.device,
                                       generator=self.gen) * 0.12 * drift),
                                  dim=-1)
        self.station[idx] = station
        self.lok[idx] = False

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
            d = torch.hypot(dx, dy)
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
        d = torch.hypot(dx, dy).clamp(min=1e-6)
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
        h = torch.where(self.in_slot, P.CONTACT_HYST, 0.0)[:, None]
        touch = ((lx <= -P.DOCK_CAVITY_D + P.CONTACT_LONG_TOL + h)
                 & (ly.abs() <= P.CONTACT_LAT_TOL + h)
                 & (lx > -P.DOCK_CAVITY_D - 0.12))
        ins, which = touch.max(dim=1)
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
            hp = self.w.home_pose()
            self.station = torch.where(just_charge[:, None], hp, self.station)

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
        d = d * (1.0 + torch.randn(d.shape, device=self.device,
                                   generator=self.gen) * P.LIDAR_NOISE)
        ok = (d >= P.LIDAR_MIN) & (d < P.LIDAR_MAX - 1e-6)
        ok &= torch.rand(d.shape, device=self.device,
                         generator=self.gen) >= P.LIDAR_DROP
        d = torch.round(d * 1000.0) / 1000.0

        slot = (idx % N_LIDAR)
        self.lr[:, slot] = d
        self.lok[:, slot] = ok
        self.lox[:, slot] = self.ox[:, None]
        self.loy[:, slot] = self.oy[:, None]
        self.loth[:, slot] = self.oth[:, None]

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
        rr = torch.hypot(lx, ly)
        self.scan_r = torch.where(self.lok, rr,
                                  torch.full_like(rr, P.LIDAR_MAX))
        self.scan_b = torch.atan2(ly, lx)
        self.scan_ok = self.lok.clone()

    def fans(self):
        wdt = 2.0 * math.pi / P.N_LIDAR_FANS
        b = (self.scan_b + 0.5 * wdt) % (2.0 * math.pi)
        i = (b / wdt).floor().long().clamp(0, P.N_LIDAR_FANS - 1)
        out = torch.full((self.R, P.N_LIDAR_FANS), P.LIDAR_MAX,
                         device=self.device)
        r = torch.where(self.scan_ok, self.scan_r,
                        torch.full_like(self.scan_r, P.LIDAR_MAX))
        return out.scatter_reduce(1, i, r, reduce="amin", include_self=True)

    # ------------------------------------------------------------------ hong ngoai
    def ir_dock(self):
        pose = self.w.dock
        ex = pose[..., 0] - (P.DOCK_CAVITY_D - 0.002) * torch.cos(pose[..., 2])
        ey = pose[..., 1] - (P.DOCK_CAVITY_D - 0.002) * torch.sin(pose[..., 2])
        return self._ir(ex, ey, self.w.dock_powered)

    def ir_beacon(self):
        b = self.w.beacon
        return self._ir(b[..., 0], b[..., 1], b[..., 2] > 0.5)

    def _ir(self, ex, ey, on):
        dx = ex - self.x[:, None]
        dy = ey - self.y[:, None]
        dist = torch.hypot(dx, dy)
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
