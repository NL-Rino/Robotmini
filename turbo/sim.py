# -*- coding: utf-8 -*-
"""Mo phong ban v1 chay THEO LO.

Cung can nha, cung vat ly, cung 64 dau vao nhu `sim/` - chi khac mot dieu:
144 con xe buoc cung mot luc thay vi lan luot tung con.

Nhung thu cua ban "can nha" cung o day: chan ban ghe (hinh tron dung yen),
nguoi la hai cai chan nhap nhay tren LiDAR, luoi kham pha theo tung xe, cham
goi an roi sang lai cho khac, va dem lan sac HOP LE (cam luc < 20%, nam den
khi day).

Do la toan bo bi mat cua viec dung duoc GPU. Ban `sim/` khong cham vi numpy
cham; no cham vi moi buoc la vai chuc phep tinh ti hon tren mang 64 phan tu,
va o kich thuoc do thi tien goi ham dat hon tien tinh toan. Gop 144 xe lai
thi cung tung ay phep tinh nhung moi phep chay tren mang 1,4 trieu phan tu -
CPU nhanh len vai lan, con GPU thi dung cho no sinh ra.
"""

import math

import torch

from sim import params as P
from sim.coverage import ALONG, RAY_STEP

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
        # `dtype` phai ghi RO. `torch.full` voi mot so Python tren card lien
        # ra so thuc 64 BIT, va card do khong lam duoc `clamp` tren 64 bit.
        # Cai buoc mo phong dau tien chet la vi day: chua quet xong vong nao
        # thi `scan_r` van la mang goc nay, quet xong mot vong thi
        # `_finish_scan` thay no bang mang 32 bit - nen loi chi hien ra o xe
        # VUA DAT LAI, va bien mat sau mot vong quet.
        self.scan_r = torch.full((self.R, N_LIDAR), P.LIDAR_MAX,
                                 dtype=torch.float32, device=device)
        self.scan_b = torch.zeros(self.R, N_LIDAR, device=device)
        self.scan_ok = torch.zeros(self.R, N_LIDAR, dtype=torch.bool,
                                   device=device)
        self.new_scan = False
        self._lidar_step = 2.0 * math.pi / N_LIDAR
        # Cham goi theo TUNG XE. Ban v1 tat cai den tren mat bang, nhung o
        # day mot mat bang duoc P ban sao dung chung: ban sao nao toi truoc
        # se tat den cua ca P ban, va chung so ngau nhien mat nghia.
        self.reset_beacons()

        # Tien do de bai, theo tung xe (xem sim/params.py muc "de bai").
        self.task_beacons = z(self.R)
        self.task_charges = z(self.R)
        self.charge_valid = b()

        # Luoi kham pha: moi xe mot ban. Them MOT cot rac o cuoi cho nhung
        # tia khong hop le, de khong phai cat mang theo mat na.
        f = self.w.floor
        self.cov_x0, self.cov_y0 = f[:, 0].clone(), f[:, 1].clone()
        self.cov_nx = max(1, int(math.ceil(float((f[:, 2] - f[:, 0]).max())
                                           / P.COVER_CELL)))
        self.cov_ny = max(1, int(math.ceil(float((f[:, 3] - f[:, 1]).max())
                                           / P.COVER_CELL)))
        self.cov_n = self.cov_nx * self.cov_ny
        self.seen = z(self.R, self.cov_n + 1)
        self.new_area = z(self.R)

    def reset_beacons(self):
        b = self.w.beacon
        self.bpos = b[..., 0:2].clone()
        self.bon = b[..., 2] > 0.5
        self.bwait = torch.zeros_like(b[..., 2])

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
        # DAT HET CA LO la truong hop thuong gap nhat (moi the he mot lan),
        # va luc do khong can chep theo chi so gi ca - chep thang la xong.
        # Dieu do dang gia tren card lien: `index_copy_` o do khong co that,
        # PyTorch am tham chep tensor sang CPU roi chep nguoc lai. `copy_`
        # va `fill_` thi may nao cung co.
        full = int(idx.numel()) == self.R

        def dat(a, v):
            if full:
                a.copy_(v)
            else:
                a.index_copy_(0, idx, v)

        def xoa(a, val):
            if full:
                a.fill_(val)
            else:
                a.index_fill_(0, idx, val)

        th = wrap(th)
        for a, v in ((self.x, x), (self.y, y), (self.th, th),
                     (self.ox, x), (self.oy, y), (self.oth, th)):
            dat(a, v)
        for a in (self.vl, self.vr, self.cmd, self.bump):
            xoa(a, 0.0)
        for a in (self.stranded, self.fallen, self.in_slot, self.id_ok,
                  self.charging, self.charge_valid):
            xoa(a, False)
        xoa(self.new_area, 0.0)
        if battery is not None:
            dat(self.batt, battery)
            if full:
                self.low_lamp = battery < P.BATT_LOW
            else:
                self.low_lamp = ops.put_rows(self.low_lamp, idx,
                                             battery < P.BATT_LOW)
        hp = self.w.home_pose()
        hp = hp if full else hp.index_select(0, idx)
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
        dat(self.station, station)
        # Dat lai xe la dat lai ca LiDAR: xe vua bat len thi chua thay gi.
        # Khong xoa thi vai buoc dau tien xe con nhin bang vong quet cua lan
        # danh gia TRUOC - mot can phong khac han.
        xoa(self.lok, False)
        xoa(self.scan_ok, False)
        xoa(self.scan_r, P.LIDAR_MAX)

    # ------------------------------------------------------------------ vat ly
    def _circles(self):
        """Thu CHAN DUONG xe: chan ban ghe + than nguoi."""
        return torch.cat((self.w.legs, self.w.mover), dim=1)

    def _lidar_circles(self):
        """Thu LiDAR NHIN THAY: chan ban ghe + HAI CHAN moi nguoi (mot cai
        nhap nhay theo nhip buoc), khong phai mot khoi tron o than."""
        return torch.cat((self.w.legs, self.w.mover_legs), dim=1)

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

        # Cham goi: an truoc buoc pin, dung thu tu cua sim/fleet.py.
        # So ngau nhien boc MOI BUOC, du co can hay khong: neu chi boc khi
        # "co xe nao vua an", thi chia quan the ra hai may (duo.py) se lam
        # hai may boc so khac nhau va chung so ngau nhien vo.
        nb = self.bon.shape[1]
        u = self._rand(2 * nb)
        got = self.take_beacon(P.BEACON_RADIUS, ~(self.stranded | self.fallen),
                               u[:, :nb])
        self._respawn_beacons(u[:, nb:])

        ins, idok, _wd = self.contact()
        was_slot, was_charge = self.in_slot, self.charging
        self.in_slot, self.id_ok, self.charging = ins, idok, idok

        prev_b = self.batt
        speed = (v.abs() / P.V_MAX + 0.35 * wv.abs() / P.W_MAX).clamp(max=1.0)
        drain = P.BATT_IDLE_DRAIN + P.BATT_MOVE_DRAIN * speed
        drain = torch.where(self.stranded | self.fallen,
                            torch.zeros_like(drain), drain)
        # Lan sac HOP LE: luc vua cam, pin (truoc buoc nap) phai < 20%.
        # Rut ra khi chua day -> mat. Day 100% -> dem mot lan.
        just_on = self.charging & ~was_charge
        self.charge_valid = torch.where(just_on, prev_b < P.BATT_LOW,
                                        self.charge_valid) & self.charging
        self.batt = torch.where(
            self.charging, (self.batt + P.BATT_CHARGE_RATE * self.dt).clamp(max=1.0),
            (self.batt - drain * self.dt).clamp(min=0.0))
        d_batt = (self.batt - prev_b).clamp(min=0.0)
        done = self.charge_valid & (self.batt >= P.BATT_FULL)
        self.task_charges = self.task_charges + done.float()
        self.charge_valid = self.charge_valid & ~done

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
                    latch=just_charge, beacon=got, charge_done=done)

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
        dc = raycast_circles(self.x, self.y, ang, self._lidar_circles(),
                             P.LIDAR_MAX)
        d = torch.minimum(d, dc)
        d = d * (1.0 + self._randn(d.shape[1]) * P.LIDAR_NOISE)
        ok = (d >= P.LIDAR_MIN) & (d < P.LIDAR_MAX - 1e-6)
        ok &= self._rand(d.shape[1]) >= P.LIDAR_DROP
        d = torch.round(d * 1000.0) / 1000.0

        # Cac o can ghi luon LIEN TIEP (chi vong qua 0 mot lan moi vong
        # quet), nen cat lam mot hoac hai doan roi chep thang. Truoc day cho
        # nay dung `index_copy_` - phep ma card lien khong co that, va
        # PyTorch am tham chep ca mang sang CPU roi chep nguoc lai, NAM LAN
        # MOI BUOC.
        self._ghi_cot(i0 % N_LIDAR, i1 - i0,
                      ((self.lr, d), (self.lok, ok),
                       (self.lox, self.ox[:, None]),
                       (self.loy, self.oy[:, None]),
                       (self.loth, self.oth[:, None])))

        if (i1 // N_LIDAR) > (i0 // N_LIDAR):
            self._rev += 1
            self._finish_scan()
            self.new_scan = True
            return True
        return False

    def _ghi_cot(self, lo, n, cap):
        """Ghi vao cac cot [lo, lo+n) vong tron, bang narrow + copy_."""
        n1 = min(n, N_LIDAR - lo)
        for a, v in cap:
            if v.shape[1] == n:
                a.narrow(1, lo, n1).copy_(v[:, :n1])
                if n1 < n:
                    a.narrow(1, 0, n - n1).copy_(v[:, n1:])
            else:                       # (R,1): phat ra ca doan
                a.narrow(1, lo, n1).copy_(v)
                if n1 < n:
                    a.narrow(1, 0, n - n1).copy_(v)

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
        two_pi = 2.0 * math.pi
        # Khong dung `%` tren so thuc, va khong `clamp` tren so nguyen: hai
        # phep do de thieu tren card lien. `scan_b` luon nam trong (-pi, pi]
        # vi no ra tu atan2, nen cong/tru mot vong la du, khong can chia du.
        b = self.scan_b + 0.5 * wdt
        b = torch.where(b < 0.0, b + two_pi, b)
        b = torch.where(b >= two_pi, b - two_pi, b)
        i = (b / wdt).floor().clamp(0.0, P.N_LIDAR_FANS - 1.0).long()
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
        return self._ir(self.bpos[..., 0], self.bpos[..., 1], self.bon)

    def take_beacon(self, radius, live=None, u=None):
        """Xe nao vua cham vao mot cham goi dang sang: tat cham do cho RIENG
        xe do, hen gio sang lai o cho khac, va cong vao tien do de bai.

        `u`: so ngau nhien [0,1) (R, so cham) de hen gio; khong dua thi tu boc.
        """
        near = self.bon & (hypot(self.bpos[..., 0] - self.x[:, None],
                                 self.bpos[..., 1] - self.y[:, None]) < radius)
        if live is not None:
            near = near & live[:, None]
        got, wi = ops.any_and_first(near)
        # moi buoc chi an MOT cham, giong ban v1
        off = ops.one_hot_at(near, wi)
        if u is None:
            u = self._rand(self.bon.shape[1])
        lo, hi = P.BEACON_RESPAWN
        self.bon = self.bon & ~off
        self.bwait = torch.where(off, lo + (hi - lo) * u, self.bwait)
        self.task_beacons = self.task_beacons + got.float()
        return got

    def _respawn_beacons(self, u=None):
        """Cham da an het gio cho thi sang lai o mot cho moi, xa xe >= 1,5 m."""
        off = ~self.bon
        self.bwait = torch.where(off, (self.bwait - self.dt).clamp(min=0.0),
                                 self.bwait)
        due = off & (self.bwait <= 0.0)
        if u is None:
            u = self._rand(self.bon.shape[1])
        ns = self.w.spots.shape[1]
        k = (u * ns).floor().clamp(0.0, ns - 1.0).long()
        px = self.w.spots[..., 0].gather(1, k)
        py = self.w.spots[..., 1].gather(1, k)
        far = hypot(px - self.x[:, None], py - self.y[:, None]) >= 1.5
        put = due & far
        self.bpos = torch.where(put[..., None], torch.stack((px, py), -1),
                                self.bpos)
        self.bon = self.bon | put

    # ------------------------------------------------------------------ kham pha
    def cover(self):
        """Sau MOI VONG QUET: danh dau o nhin thay, dem o LAN DAU thay.

        Dung y `sim/coverage.py`: mot tia tren bon, ba diem tren moi tia,
        vi tri THAT cua xe. Tra ve (va giu o `new_area`) so o moi / 12.
        """
        if not self.new_scan:
            self.new_area = torch.zeros_like(self.new_area)
            return self.new_area
        r = self.scan_r[:, ::RAY_STEP]
        ok = self.scan_ok[:, ::RAY_STEP]
        ang = self.th[:, None] + self.scan_b[:, ::RAY_STEP]
        c, sn = torch.cos(ang), torch.sin(ang)
        cols = []
        for f in ALONG:
            px = self.x[:, None] + c * r * f
            py = self.y[:, None] + sn * r * f
            cx = ((px - self.cov_x0[:, None]) / P.COVER_CELL).floor() \
                .clamp(0.0, self.cov_nx - 1.0)
            cy = ((py - self.cov_y0[:, None]) / P.COVER_CELL).floor() \
                .clamp(0.0, self.cov_ny - 1.0)
            i = cy * self.cov_nx + cx
            cols.append(torch.where(ok, i, torch.full_like(i, float(self.cov_n))))
        idx = torch.cat(cols, dim=1).long()
        before = self.seen[:, :self.cov_n].sum(1)
        self.seen = self.seen.scatter(1, idx, torch.ones_like(idx,
                                                             dtype=self.seen.dtype))
        moi = self.seen[:, :self.cov_n].sum(1) - before
        self.new_area = moi / 12.0
        return self.new_area

    def cover_fraction(self):
        return self.seen[:, :self.cov_n].sum(1) / float(self.cov_n)

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
        hitc = raycast_circles(self.x, self.y, ang[:, None], self._lidar_circles(),
                               1e4)[:, 0]
        seen = seen & (torch.minimum(hit, hitc) >= best - 0.02)
        return seen, wrap(ang - self.th)
