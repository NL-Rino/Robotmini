# -*- coding: utf-8 -*-
"""Moi truong 3D chay theo lo tren GPU.

B the gioi chay song song, moi the gioi mot xe. Khong co vong lap Python nao
theo tung the gioi - moi buoc la mot chuoi phep tinh tensor. Do la ly do ban
nay dung duoc GPU con ban v1 thi khong.

Khac v1 mot cho co chu y: o day moi the gioi chi co MOT xe hoc, cac xe khac
la vat can hinh tru (co cai dang do trong hoc, chiem cho). Lam vay de mot
buoc la mot phep tinh tensor thay vi mot bai toan nhieu tac tu; bai toan cua
tung xe thi van y nguyen.

Hanh dong: 3 so trong [-1,1] = ga trai, ga phai, toc do servo camera.
Quan sat : anh (B,3,48,64) + 48 so vo huong.
"""

import math

import torch

from . import params as P
from .camera import render
from .raytrace import trace
from .world3d import make_batch

PHASES = ("sap-cam", "truoc-mieng", "pin-yeu", "long-nhong", "trong-hoc")
MIX_EARLY = (0.34, 0.34, 0.16, 0.10, 0.06)
MIX_LATE = (0.08, 0.17, 0.30, 0.25, 0.20)

# --- phan thuong (xem docs/DESIGN.md muc 4) ---
R_FALL = -150.0
R_FLAT = -150.0
R_ALIVE = 0.02
R_BUMP = -1.5
R_CLIFF = -0.8
R_CLIFF_CAP = -60.0       # tran. Do duoc o ban v1: xe ket canh cai ho, cam
                          # bien vuc keu suot 300 giay -> -1.713 diem, tuc la
                          # dung canh ho dat hon lao xuong ho (-150).
R_CHARGE = 150.0
R_LATCH = 30.0
R_AWAY_DIST = 1.5         # phai roi hoc xa chung nay thi lan cam sau moi
                          # duoc tra tien. Khong co no thi xe lac ra lac vao
                          # an +30 moi lan - do duoc 42 lan mot tap o ban v1.
R_FULL_ENOUGH = 0.97
R_LOITER = -0.10          # moi buoc con nam trong hoc khi pin da day
R_LOITER_CAP = -80.0
R_WRONG = -4.0
R_HOME = 3.0
# Trong long hoc CHI duoc lui vao va di thang ra. Long hoc 31 cm ma than xe
# 30 cm: quay nguoi trong do la co xat hai vach va giat chan tiep dien.
R_SPIN_DOCK = -2.5
R_SPIN_FREE = 0.35        # phan toc do quay toi da duoc quay tu do de con
                          # nan huong luc dang lui vao


def phase_mix(progress):
    p = min(1.0, max(0.0, float(progress)))
    return tuple(a + (b - a) * p for a, b in zip(MIX_EARLY, MIX_LATE))


def scalar_names():
    return (
        [f"fan{i}" for i in range(P.N_LIDAR_FANS)]
        + ["cliff_L", "cliff_R"]
        + ["ir_dock", "ir_sin", "ir_cos"]
        + ["st_known", "st_range", "st_bsin", "st_bcos", "st_asin", "st_acos"]
        + ["batt_soc", "batt_low_blink", "batt_charging"]
        + ["contact_in_slot", "contact_id_ok"]
        + ["mv_v", "mv_w", "mv_cmd_l", "mv_cmd_r", "mv_bump"]
        + ["tilt_sin", "tilt_cos"]
        + [f"mycode_{c}_{k}" for c in range(P.MARK_CELLS)
           for k in range(P.MARK_COLORS)]
        + ["tilt_at_limit"]
    )


class FleetEnv3D:
    def __init__(self, n_envs=64, device=None, seed=0, n_docks=3, n_decoys=1,
                 max_steps=600, chunk=8, img_noise=0.012, progress=0.0):
        self.n = int(n_envs)
        self.device = device or torch.device("cpu")
        self.max_steps = int(max_steps)
        self.chunk = chunk
        self.img_noise = img_noise
        self.progress = float(progress)
        self.dt = P.DT

        self.sc, self.metas = make_batch(self.n, self.device, seed0=seed * 100003,
                                         n_docks=n_docks, n_decoys=n_decoys)
        self.n_docks = self.sc["dock_pose"].shape[1]
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed + 12345)

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.x, self.y, self.th = z(self.n), z(self.n), z(self.n)
        self.vl, self.vr = z(self.n), z(self.n)
        self.ox, self.oy, self.oth = z(self.n), z(self.n), z(self.n)
        self.batt = torch.ones(self.n, device=self.device)
        self.tilt, self.tilt_v = z(self.n), z(self.n)
        self.cmd = z(self.n, 2)
        self.bump = z(self.n)
        self.t = z(self.n)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.station = z(self.n, 3)
        self.st_known = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.low_lamp = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.in_slot = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.id_ok = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.charging = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.stranded = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.home = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.prev_home_d = z(self.n)
        self.mover_u = torch.rand(self.sc["cyl_c"].shape[:2], device=self.device,
                                  generator=self.gen)
        self.mover_dir = torch.ones_like(self.mover_u)

        # Sai so odometry rieng cua tung xe.
        g = torch.randn(self.n, 3, device=self.device, generator=self.gen)
        common = 1.0 + g[:, 0] * P.ODOM_SCALE_ERR
        diff = g[:, 1] * P.ODOM_DIFF_ERR
        self.odom_l = common * (1.0 - diff)
        self.odom_r = common * (1.0 + diff)
        self.odom_drift = g[:, 2] * P.ODOM_DRIFT

        self.ep_charged = z(self.n)
        self.ep_wrong = z(self.n)
        self.ep_ret = z(self.n)
        self.ep_budget = z(self.n)      # han muc nang luong cua lan vao hoc nay
        self.ep_away = z(self.n)        # da roi hoc xa nhat bao nhieu
        self.ep_loiter = z(self.n)
        self.ep_cliff = z(self.n)
        self.reset_all()

    # ------------------------------------------------------------------ dat xe
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def reset_all(self):
        self.reset_idx(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def reset_idx(self, m):
        """Dat lai nhung moi truong co m=True, theo giao trinh nguoc."""
        k = int(m.sum())
        if k == 0:
            return
        idx = torch.nonzero(m, as_tuple=True)[0]
        mix = torch.tensor(phase_mix(self.progress), device=self.device)
        ph = torch.multinomial(mix.expand(k, -1), 1,
                               generator=self.gen).squeeze(-1)

        pose = self.sc["dock_pose"][idx]                   # (k, D, 3)
        powered = self.sc["dock_powered"][idx]
        blocked = self.sc["dock_blocked"][idx]
        good = powered & ~blocked
        # Hoc cua xe: mot trong nhung hoc co dien va chua bi chiem.
        w = good.float() + 1e-6
        home = torch.multinomial(w, 1, generator=self.gen).squeeze(-1)
        self.home[idx] = home
        hp = pose.gather(1, home[:, None, None].expand(-1, 1, 3)).squeeze(1)

        # Hoc de dat xe o pha "truoc mieng"/"sap cam": mot NUA so lan la hoc
        # cua no, con lai la hoc nguoi khac hoac hoc moi nhu. Neu lan nao
        # cung dat dung hoc cua no thi bo nao se hoc duoc rang "cu cam la co
        # dien" va se khong bao gio nhin len bang ma.
        other = torch.multinomial(torch.ones_like(w), 1,
                                  generator=self.gen).squeeze(-1)
        use_own = self._rand(k) < 0.55
        pick = torch.where(use_own, home, other)
        pp = pose.gather(1, pick[:, None, None].expand(-1, 1, 3)).squeeze(1)
        dx, dy, dth = pp[:, 0], pp[:, 1], pp[:, 2]
        hard = self._rand(k)

        # pha 0: sap cam - da lui dang do vao hoc
        dep = 0.16 - 0.14 * hard
        x0 = dx - dep * torch.cos(dth)
        y0 = dy - dep * torch.sin(dth)
        t0 = dth + torch.randn(k, device=self.device, generator=self.gen) * (
            0.03 + 0.25 * hard)
        b0 = 0.04 + 0.08 * self._rand(k)

        # pha 1: truoc mieng
        dist = 0.35 + 0.60 * hard
        x1 = dx + dist * torch.cos(dth)
        y1 = dy + dist * torch.sin(dth)
        t1 = dth + torch.randn(k, device=self.device, generator=self.gen) * (
            0.15 + 1.6 * hard)
        b1 = 0.05 + 0.09 * self._rand(k)

        # pha 2,3: dung bat ky dau
        spots = self.sc["spots"][idx]
        si = torch.randint(0, spots.shape[1], (k,), device=self.device,
                           generator=self.gen)
        sp = spots.gather(1, si[:, None, None].expand(-1, 1, 2)).squeeze(1)
        x2, y2 = sp[:, 0], sp[:, 1]
        t2 = (self._rand(k) * 2.0 - 1.0) * math.pi
        b2 = 0.05 + (P.BATT_LOW - 0.055) * self._rand(k)
        b3 = 0.25 + 0.70 * self._rand(k)

        # pha 4: trong hoc cua minh, pin day
        dep4 = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005
        x4 = hp[:, 0] - dep4 * torch.cos(hp[:, 2])
        y4 = hp[:, 1] - dep4 * torch.sin(hp[:, 2])
        t4 = hp[:, 2]
        b4 = 0.75 + 0.25 * self._rand(k)

        sel = lambda a0, a1, a2, a3, a4: torch.where(
            ph == 0, a0, torch.where(ph == 1, a1, torch.where(
                ph == 2, a2, torch.where(ph == 3, a3, a4))))
        self.x[idx] = sel(x0, x1, x2, x2, x4)
        self.y[idx] = sel(y0, y1, y2, y2, y4)
        self.th[idx] = sel(t0, t1, t2, t2, t4)
        self.batt[idx] = sel(b0, b1, b2, b3, b4)

        self.vl[idx] = 0.0
        self.vr[idx] = 0.0
        self.cmd[idx] = 0.0
        self.bump[idx] = 0.0
        self.tilt[idx] = (self._rand(k) * 2.0 - 1.0) * 0.2
        self.tilt_v[idx] = 0.0
        self.stranded[idx] = False
        self.fallen[idx] = False
        self.in_slot[idx] = False
        self.id_ok[idx] = False
        self.charging[idx] = False
        self.low_lamp[idx] = self.batt[idx] < P.BATT_LOW
        self.t[idx] = 0.0
        self.steps[idx] = 0
        self.ox[idx] = self.x[idx]
        self.oy[idx] = self.y[idx]
        self.oth[idx] = self.th[idx]

        # Bo nho tram: ghi vi tri hoc CUA NO nhung lech di vai met, dung nhu
        # odometry that sau mot chuyen di dai. Lan nao cung ghi dung thi bo
        # nao se hoc cach tin vao no, roi ra doi la chiu.
        drift = self._rand(k) * 2.5
        ang = (self._rand(k) * 2.0 - 1.0) * math.pi
        self.station[idx, 0] = hp[:, 0] + drift * torch.cos(ang)
        self.station[idx, 1] = hp[:, 1] + drift * torch.sin(ang)
        self.station[idx, 2] = hp[:, 2] + torch.randn(
            k, device=self.device, generator=self.gen) * 0.12 * drift
        self.st_known[idx] = True

        self.ep_charged[idx] = 0.0
        self.ep_wrong[idx] = 0.0
        self.ep_ret[idx] = 0.0
        self.ep_loiter[idx] = 0.0
        self.ep_cliff[idx] = 0.0
        # Xe dat NGOAI hoc thi coi nhu da di xa roi - no dang o ngoai that.
        # Chi xe dat san trong hoc va dang co dien moi phai di lam mot vong
        # roi ve thi lan sac moi duoc tra tien.
        rear_x = self.x[idx] - P.BODY_RADIUS * torch.cos(self.th[idx])
        rear_y = self.y[idx] - P.BODY_RADIUS * torch.sin(self.th[idx])
        lx, ly = self._to_dock_local(rear_x, rear_y, hp)
        docked = (lx <= -P.DOCK_CAVITY_D + P.CONTACT_LONG_TOL) & \
                 (ly.abs() <= P.CONTACT_LAT_TOL)
        self.ep_away[idx] = torch.where(docked, torch.zeros_like(rear_x),
                                        torch.full_like(rear_x, R_AWAY_DIST))
        self.ep_budget[idx] = torch.where(
            docked, torch.zeros_like(rear_x),
            torch.clamp(1.0 - self.batt[idx], min=0.0))
        self.prev_home_d[idx] = self._home_dist(idx)

    @staticmethod
    def _to_dock_local(px, py, pose):
        """Doi mot diem sang he quy chieu hoc: +x la truc ra, goc o mieng."""
        dx = px - pose[:, 0]
        dy = py - pose[:, 1]
        ca, sa = torch.cos(-pose[:, 2]), torch.sin(-pose[:, 2])
        return dx * ca - dy * sa, dx * sa + dy * ca

    def _in_cavity(self):
        """Tam xe co dang nam trong long mot cai hoc nao khong."""
        pose = self.sc["dock_pose"]
        dx = self.x[:, None] - pose[..., 0]
        dy = self.y[:, None] - pose[..., 1]
        ca, sa = torch.cos(-pose[..., 2]), torch.sin(-pose[..., 2])
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        return ((lx <= 0.02) & (lx >= -P.DOCK_CAVITY_D - 0.02)
                & (ly.abs() <= 0.5 * P.DOCK_CAVITY_W + 0.02)).any(dim=1)

    def _home_dist(self, idx=None):
        """Khoang cach toi DIEM DUNG TRUOC MIENG hoc cua no (khong phai toi hoc).

        Dan thang toi cai hoc thi xe bi hut vao suon hoc - cho gan nhat nhung
        la ngo cut, khong nhin thau long hoc ma cung khong bat duoc hong ngoai.
        """
        if idx is None:
            idx = slice(None)
        pose = self.sc["dock_pose"][idx]
        h = self.home[idx]
        hp = pose.gather(1, h[:, None, None].expand(-1, 1, 3)).squeeze(1)
        ax = hp[:, 0] + 0.55 * torch.cos(hp[:, 2])
        ay = hp[:, 1] + 0.55 * torch.sin(hp[:, 2])
        return torch.hypot(self.x[idx] - ax, self.y[idx] - ay)

    # ------------------------------------------------------------------ va cham
    def _resolve_collision(self):
        """Day xe ra khoi vat the. Chi tinh nhung hop chan duong - mat ban o
        0,42 m thi xe chui qua duoc, do la diem chi ban 3D moi co."""
        hit = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        R = P.BODY_RADIUS
        bc, bh, byaw = self.sc["box_c"], self.sc["box_h"], self.sc["box_yaw"]
        solid = self.sc["box_solid"]
        cc, cr = self.sc["cyl_c"], self.sc["cyl_r"]
        for _ in range(3):
            dx = self.x[:, None] - bc[..., 0]
            dy = self.y[:, None] - bc[..., 1]
            ca, sa = torch.cos(-byaw), torch.sin(-byaw)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            qx = torch.clamp(lx, -bh[..., 0], bh[..., 0])
            qy = torch.clamp(ly, -bh[..., 1], bh[..., 1])
            ex, ey = lx - qx, ly - qy
            d = torch.hypot(ex, ey)
            pen = torch.where(solid & (d < R), R - d,
                              torch.zeros_like(d))
            best, wi = pen.max(dim=1)
            any_hit = best > 1e-6
            if bool(any_hit.any()):
                g = wi[:, None]
                dd = d.gather(1, g).squeeze(1).clamp(min=1e-6)
                nx = ex.gather(1, g).squeeze(1) / dd
                ny = ey.gather(1, g).squeeze(1) / dd
                yb = byaw.gather(1, g).squeeze(1)
                wx = nx * torch.cos(yb) - ny * torch.sin(yb)
                wy = nx * torch.sin(yb) + ny * torch.cos(yb)
                self.x = torch.where(any_hit, self.x + wx * best, self.x)
                self.y = torch.where(any_hit, self.y + wy * best, self.y)
                hit |= any_hit
            else:
                break

        if cc.shape[1] > 0:
            dx = self.x[:, None] - cc[..., 0]
            dy = self.y[:, None] - cc[..., 1]
            d = torch.hypot(dx, dy).clamp(min=1e-6)
            over = (R + cr) - d
            best, wi = over.max(dim=1)
            m = (best > 1e-6) & (cr.gather(1, wi[:, None]).squeeze(1) > 1e-6)
            if bool(m.any()):
                g = wi[:, None]
                nx = dx.gather(1, g).squeeze(1) / d.gather(1, g).squeeze(1)
                ny = dy.gather(1, g).squeeze(1) / d.gather(1, g).squeeze(1)
                self.x = torch.where(m, self.x + nx * best, self.x)
                self.y = torch.where(m, self.y + ny * best, self.y)
                hit |= m
        return hit

    def _on_floor(self, px, py):
        fl = self.sc["floor"]
        ok = ((px >= fl[:, 0]) & (px <= fl[:, 2]) &
              (py >= fl[:, 1]) & (py <= fl[:, 3]))
        vd = self.sc["void"]
        if vd.shape[1] > 0:
            hole = ((px[:, None] >= vd[..., 0]) & (px[:, None] <= vd[..., 2]) &
                    (py[:, None] >= vd[..., 1]) & (py[:, None] <= vd[..., 3])
                    ).any(dim=1)
            ok &= ~hole
        return ok

    def _cliff(self):
        out = []
        for sgn in (1.0, -1.0):
            a = self.th + sgn * P.CLIFF_ANGLE
            out.append(~self._on_floor(self.x + P.CLIFF_RADIUS * torch.cos(a),
                                       self.y + P.CLIFF_RADIUS * torch.sin(a)))
        return out[0], out[1]

    def _contact(self):
        """Chan tiep dien o DUOI xe. Phai lui duoi vao hoc moi cham duoc, va
        cham roi thi hoc con phai BAT TAY DUNG MA moi ra dien."""
        rx = self.x - P.BODY_RADIUS * torch.cos(self.th)
        ry = self.y - P.BODY_RADIUS * torch.sin(self.th)
        pose = self.sc["dock_pose"]
        dx = rx[:, None] - pose[..., 0]
        dy = ry[:, None] - pose[..., 1]
        ca, sa = torch.cos(-pose[..., 2]), torch.sin(-pose[..., 2])
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        hyst = torch.where(self.in_slot, P.CONTACT_HYST, 0.0)[:, None]
        deep = lx <= -P.DOCK_CAVITY_D + P.CONTACT_LONG_TOL + hyst
        cent = ly.abs() <= P.CONTACT_LAT_TOL + hyst
        touch = deep & cent & (lx > -P.DOCK_CAVITY_D - 0.12)
        in_slot, which = touch.max(dim=1)

        g = which[:, None, None].expand(-1, 1, P.MARK_CELLS)
        dcode = self.sc["dock_code"].gather(1, g).squeeze(1)
        mycode = self.sc["dock_code"].gather(
            1, self.home[:, None, None].expand(-1, 1, P.MARK_CELLS)).squeeze(1)
        pw = self.sc["dock_powered"].gather(1, which[:, None]).squeeze(1)
        same = (dcode == mycode).all(dim=1)
        id_ok = in_slot & pw & same
        return in_slot, id_ok, which

    def _move_people(self):
        path = self.sc["mover_path"]
        n_mv = self.sc["mover_n"][:, None]
        idx = torch.arange(path.shape[1], device=self.device)[None, :]
        is_mv = idx < n_mv
        p0 = path[..., 0:2]
        p1 = path[..., 2:4]
        speed = path[..., 4]
        seg = (p1 - p0).norm(dim=-1).clamp(min=0.05)
        self.mover_u = self.mover_u + self.mover_dir * speed / seg * self.dt
        over = self.mover_u > 1.0
        under = self.mover_u < 0.0
        self.mover_dir = torch.where(over | under, -self.mover_dir, self.mover_dir)
        self.mover_u = torch.clamp(self.mover_u, 0.0, 1.0)
        pos = p0 + (p1 - p0) * self.mover_u[..., None]
        self.sc["cyl_c"] = torch.where(is_mv[..., None], pos, self.sc["cyl_c"])

    # ------------------------------------------------------------------ mot buoc
    def step(self, action):
        a = torch.clamp(action, -1.0, 1.0)
        alive = ~(self.stranded | self.fallen)
        cl = torch.where(alive, a[:, 0], torch.zeros_like(a[:, 0]))
        cr = torch.where(alive, a[:, 1], torch.zeros_like(a[:, 1]))
        ct = torch.where(alive, a[:, 2], torch.zeros_like(a[:, 2]))
        self.cmd = torch.stack((cl, cr), dim=-1)

        k = 1.0 - math.exp(-self.dt / P.MOTOR_TAU)
        self.vl = self.vl + (cl * P.V_MAX - self.vl) * k
        self.vr = self.vr + (cr * P.V_MAX - self.vr) * k
        v = 0.5 * (self.vl + self.vr)
        w = (self.vr - self.vl) / P.WHEEL_BASE

        th1 = self.th + w * self.dt
        small = w.abs() < 1e-6
        r = v / torch.where(small, torch.ones_like(w), w)
        self.x = torch.where(small, self.x + v * torch.cos(self.th) * self.dt,
                             self.x + r * (torch.sin(th1) - torch.sin(self.th)))
        self.y = torch.where(small, self.y + v * torch.sin(self.th) * self.dt,
                             self.y - r * (torch.cos(th1) - torch.cos(self.th)))
        self.th = torch.atan2(torch.sin(th1), torch.cos(th1))

        dl = self.vl * self.dt * self.odom_l
        dr = self.vr * self.dt * self.odom_r
        dv = 0.5 * (dl + dr)
        dw = (dr - dl) / P.WHEEL_BASE + self.odom_drift * self.dt
        self.oth = torch.atan2(torch.sin(self.oth + dw), torch.cos(self.oth + dw))
        self.ox = self.ox + dv * torch.cos(self.oth)
        self.oy = self.oy + dv * torch.sin(self.oth)

        # servo camera: chi len xuong, va co cu chan co hoc
        kt = 1.0 - math.exp(-self.dt / P.TILT_TAU)
        self.tilt_v = self.tilt_v + (ct * P.TILT_RATE - self.tilt_v) * kt
        raw = self.tilt + self.tilt_v * self.dt
        self.tilt = torch.clamp(raw, P.TILT_MIN, P.TILT_MAX)
        at_limit = (raw < P.TILT_MIN) | (raw > P.TILT_MAX)
        self.tilt_v = torch.where(at_limit, torch.zeros_like(self.tilt_v),
                                  self.tilt_v)

        self._move_people()
        hit = self._resolve_collision()
        was_bump = self.bump > 0.5
        self.bump = torch.where(hit, torch.ones_like(self.bump),
                                torch.clamp(self.bump - self.dt * 4.0, min=0.0))
        new_bump = hit & ~was_bump

        was_fallen = self.fallen
        self.fallen |= ~self._on_floor(self.x, self.y)
        just_fell = self.fallen & ~was_fallen
        self.vl = torch.where(self.fallen, torch.zeros_like(self.vl), self.vl)
        self.vr = torch.where(self.fallen, torch.zeros_like(self.vr), self.vr)

        in_slot, id_ok, _wd = self._contact()
        was_slot = self.in_slot
        was_charge = self.charging
        self.in_slot, self.id_ok = in_slot, id_ok
        self.charging = id_ok

        prev_b = self.batt
        speed = torch.clamp(v.abs() / P.V_MAX + 0.35 * w.abs() / P.W_MAX,
                            max=1.0)
        drain = P.BATT_IDLE_DRAIN + P.BATT_MOVE_DRAIN * speed
        drain = torch.where(self.stranded | self.fallen,
                            torch.zeros_like(drain), drain)
        charged = torch.clamp(self.batt + P.BATT_CHARGE_RATE * self.dt, max=1.0)
        drained = torch.clamp(self.batt - drain * self.dt, min=0.0)
        self.batt = torch.where(self.charging, charged, drained)
        d_batt = torch.clamp(self.batt - prev_b, min=0.0)

        was_flat = self.stranded
        self.stranded = torch.where(self.batt <= 0.0, torch.ones_like(
            self.stranded), torch.where(self.batt > 0.05, torch.zeros_like(
                self.stranded), self.stranded))
        just_flat = self.stranded & ~was_flat
        self.low_lamp = torch.where(
            self.batt < P.BATT_LOW, torch.ones_like(self.low_lamp),
            torch.where(self.batt > P.BATT_LOW + 0.03,
                        torch.zeros_like(self.low_lamp), self.low_lamp))

        # bo nho tram duoc ghi lai moi lan cam dung hoc
        just_charge = self.charging & ~was_charge
        if bool(just_charge.any()):
            pose = self.sc["dock_pose"]
            hp = pose.gather(1, self.home[:, None, None].expand(-1, 1, 3)
                             ).squeeze(1)
            self.station = torch.where(just_charge[:, None],
                                       torch.stack((hp[:, 0], hp[:, 1],
                                                    hp[:, 2]), dim=-1),
                                       self.station)

        wrong = in_slot & ~id_ok & ~was_slot
        cliff_l, cliff_r = self._cliff()

        # Da roi hoc xa nhat bao nhieu ke tu lan duoc tra tien truoc.
        pose = self.sc["dock_pose"]
        hp_now = pose.gather(1, self.home[:, None, None].expand(-1, 1, 3)
                             ).squeeze(1)
        away = torch.hypot(self.x - hp_now[:, 0], self.y - hp_now[:, 1])
        self.ep_away = torch.maximum(self.ep_away, away)

        # Vua cam dung hoc VA da thuc su di lam mot vong -> tra tien, va mo
        # han muc nang luong cho lan nay.
        paid_latch = just_charge & (self.ep_away >= R_AWAY_DIST)
        self.ep_away = torch.where(paid_latch, torch.zeros_like(self.ep_away),
                                   self.ep_away)
        self.ep_budget = torch.where(
            paid_latch, torch.clamp(1.0 - self.batt, min=0.0), self.ep_budget)

        # Tra tien nang luong trong han muc. Het han muc thi xa roi nap lai
        # ngay trong hoc cung khong duoc gi.
        pay = torch.minimum(d_batt, torch.clamp(self.ep_budget, min=0.0))
        self.ep_budget = torch.clamp(self.ep_budget - pay, min=0.0)

        # Pin day ma van nam trong hoc thi bat dau lo von.
        loiter = torch.where(self.charging & (self.batt > R_FULL_ENOUGH),
                             torch.full_like(self.batt, R_LOITER),
                             torch.zeros_like(self.batt))
        loiter = torch.maximum(loiter, R_LOITER_CAP - self.ep_loiter)
        loiter = torch.minimum(loiter, torch.zeros_like(loiter))
        self.ep_loiter = self.ep_loiter + loiter

        cliff_pen = torch.where(cliff_l | cliff_r,
                                torch.full_like(self.batt, R_CLIFF),
                                torch.zeros_like(self.batt))
        cliff_pen = torch.maximum(cliff_pen, R_CLIFF_CAP - self.ep_cliff)
        cliff_pen = torch.minimum(cliff_pen, torch.zeros_like(cliff_pen))
        self.ep_cliff = self.ep_cliff + cliff_pen

        # Quay nguoi trong long hoc.
        excess = torch.clamp(w.abs() / P.W_MAX - R_SPIN_FREE, min=0.0)
        spin_pen = torch.where(self._in_cavity(), R_SPIN_DOCK * excess,
                               torch.zeros_like(excess))

        rew = torch.full_like(self.x, R_ALIVE)
        rew = rew + R_BUMP * new_bump.float()
        rew = rew + cliff_pen
        rew = rew + R_CHARGE * pay
        rew = rew + loiter
        rew = rew + spin_pen
        rew = rew + R_LATCH * paid_latch.float()
        rew = rew + R_WRONG * wrong.float()
        rew = rew + R_FALL * just_fell.float()
        rew = rew + R_FLAT * just_flat.float()
        hd = self._home_dist()
        rew = rew + torch.where(self.low_lamp, R_HOME * (self.prev_home_d - hd),
                                torch.zeros_like(hd))
        self.prev_home_d = hd
        rew = torch.where(was_fallen | was_flat, torch.zeros_like(rew), rew)

        self.ep_charged += d_batt
        self.ep_wrong += wrong.float()
        self.ep_ret += rew
        self.t += self.dt
        self.steps += 1
        done = self.fallen | self.stranded | (self.steps >= self.max_steps)

        info = dict(charged=self.ep_charged.clone(), wrong=self.ep_wrong.clone(),
                    fell=just_fell, flat=just_flat, ret=self.ep_ret.clone())
        return rew, done, info

    # ------------------------------------------------------------------ cam nhan
    def lidar_fans(self):
        n = P.N_LIDAR_RAYS
        ang = torch.arange(n, device=self.device) * (2 * math.pi / n)
        a = self.th[:, None] + ang[None, :]
        o = torch.stack((self.x[:, None].expand(-1, n),
                         self.y[:, None].expand(-1, n),
                         torch.full((self.n, n), P.LIDAR_HEIGHT,
                                    device=self.device)), dim=-1)
        d = torch.stack((torch.cos(a), torch.sin(a),
                         torch.zeros_like(a)), dim=-1)
        t, _al, _nr, _kd = trace(o, d, self.sc, chunk=self.chunk,
                                 want_floor=False)
        t = torch.clamp(t, max=P.LIDAR_MAX)
        t = t * (1.0 + torch.randn(t.shape, device=self.device,
                                   generator=self.gen) * P.LIDAR_NOISE)
        drop = torch.rand(t.shape, device=self.device,
                          generator=self.gen) < P.LIDAR_DROP
        t = torch.where(drop | (t < P.LIDAR_MIN), torch.full_like(t, P.LIDAR_MAX), t)
        per = n // P.N_LIDAR_FANS
        # quat 0 nam giua mui xe: xoay di nua quat truoc khi gom
        t = torch.roll(t, shifts=per // 2, dims=1)
        return t.reshape(self.n, P.N_LIDAR_FANS, per).min(dim=2).values

    def observe(self):
        img = render(self.sc, self.x, self.y, self.th, self.tilt,
                     chunk=self.chunk, noise=self.img_noise, generator=self.gen)
        s = torch.zeros(self.n, P.N_SCALARS, device=self.device)
        i = 0
        s[:, i:i + P.N_LIDAR_FANS] = 1.0 - self.lidar_fans() / P.LIDAR_MAX
        i += P.N_LIDAR_FANS
        cl, cr = self._cliff()
        s[:, i] = cl.float()
        s[:, i + 1] = cr.float()
        i += 2

        seen, bear = self._ir_dock()
        s[:, i] = seen.float()
        s[:, i + 1] = torch.sin(bear) * seen
        s[:, i + 2] = torch.cos(bear) * seen
        i += 3

        dx = self.station[:, 0] - self.ox
        dy = self.station[:, 1] - self.oy
        dist = torch.hypot(dx, dy)
        bear = torch.atan2(dy, dx) - self.oth
        axis = self.station[:, 2] - self.oth
        s[:, i] = self.st_known.float()
        s[:, i + 1] = torch.clamp(1.0 - dist / 5.0, min=0.0)
        s[:, i + 2] = torch.sin(bear)
        s[:, i + 3] = torch.cos(bear)
        s[:, i + 4] = torch.sin(axis)
        s[:, i + 5] = torch.cos(axis)
        i += 6

        blink = ((self.t * P.BATT_BLINK_HZ * 2.0).long() % 2 == 0).float()
        s[:, i] = self.batt
        s[:, i + 1] = self.low_lamp.float() * blink
        s[:, i + 2] = self.charging.float()
        i += 3
        s[:, i] = self.in_slot.float()
        s[:, i + 1] = self.id_ok.float()
        i += 2
        v = 0.5 * (self.vl + self.vr)
        w = (self.vr - self.vl) / P.WHEEL_BASE
        s[:, i] = torch.clamp(v / P.V_MAX, -1, 1)
        s[:, i + 1] = torch.clamp(w / P.W_MAX, -1, 1)
        s[:, i + 2] = self.cmd[:, 0]
        s[:, i + 3] = self.cmd[:, 1]
        s[:, i + 4] = self.bump
        i += 5
        s[:, i] = torch.sin(self.tilt)
        s[:, i + 1] = torch.cos(self.tilt)
        i += 2

        # MA CUA CHINH XE: bo nao phai doi chieu cai nay voi cai no NHIN THAY
        # tren bang ma. Khong co no thi camera vo dung.
        mycode = self.sc["dock_code"].gather(
            1, self.home[:, None, None].expand(-1, 1, P.MARK_CELLS)).squeeze(1)
        oh = torch.zeros(self.n, P.MARK_CELLS, P.MARK_COLORS, device=self.device)
        oh.scatter_(2, mycode[..., None], 1.0)
        s[:, i:i + P.MARK_CELLS * P.MARK_COLORS] = oh.reshape(self.n, -1)
        i += P.MARK_CELLS * P.MARK_COLORS
        s[:, i] = ((self.tilt <= P.TILT_MIN + 1e-3) |
                   (self.tilt >= P.TILT_MAX - 1e-3)).float()
        return img, s

    def _ir_dock(self):
        """Den hong ngoai trong hoc. Bi hai vach ben bop lai con ~+-33 do -
        do la hinh hoc tu lo, khong phai mot tham so nao ca."""
        pose = self.sc["dock_pose"]
        ex = pose[..., 0] - (P.DOCK_CAVITY_D - 0.002) * torch.cos(pose[..., 2])
        ey = pose[..., 1] - (P.DOCK_CAVITY_D - 0.002) * torch.sin(pose[..., 2])
        dx = ex - self.x[:, None]
        dy = ey - self.y[:, None]
        dist = torch.hypot(dx, dy)
        bear = torch.atan2(dy, dx) - self.th[:, None]
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        ok = self.sc["dock_powered"] & (dist < P.IR_RANGE) & (bear.abs() < P.IR_FOV)
        big = torch.where(ok, dist, torch.full_like(dist, 1e9))
        best, wi = big.min(dim=1)
        seen = best < 1e8
        if bool(seen.any()):
            a = torch.atan2(dy.gather(1, wi[:, None]).squeeze(1),
                            dx.gather(1, wi[:, None]).squeeze(1))
            o = torch.stack((self.x, self.y,
                             torch.full_like(self.x, P.LIDAR_HEIGHT)), dim=-1)
            d = torch.stack((torch.cos(a), torch.sin(a),
                             torch.zeros_like(a)), dim=-1)
            t, _a, _n, _k = trace(o[:, None], d[:, None], self.sc,
                                  chunk=self.chunk, want_floor=False)
            seen &= t[:, 0] >= best - 0.02
        bearing = torch.atan2(dy.gather(1, wi[:, None]).squeeze(1),
                              dx.gather(1, wi[:, None]).squeeze(1)) - self.th
        return seen, torch.atan2(torch.sin(bearing), torch.cos(bearing))
