"""Dong luc hoc xe, odometry, va cham, pin, va bo nho tram.

Mot xe = mot than tron Ø30 cm, hai banh vi sai, mot LiDAR, hai mat vuc,
mot mat hong ngoai o dau, va MOT CHAN TIEP DIEN O DUOI XE.

Xe khong "chet" theo nghia ket thuc tap. Het pin thi nam im (`stranded`),
roi xuong vuc thi nam im (`fallen`); the gioi van chay tiep. Vong lap chi
dung khi dut ket noi bo nao - xem sim/fleet.py.
"""

import math

import numpy as np

from . import params as P
from . import sensors
from .geometry import wrap_pi_scalar
from .lidar import Lidar


class Robot:
    def __init__(self, robot_id, x, y, th, dock_code, seed=0,
                 radius=P.BODY_RADIUS):
        self.id = robot_id
        self.code = dock_code           # ma hoc sac cua RIENG xe nay
        self.radius = radius

        self.x, self.y, self.th = float(x), float(y), wrap_pi_scalar(float(th))
        self.ox, self.oy, self.oth = self.x, self.y, self.th   # odometry

        self.vl = 0.0                   # toc do banh thuc te
        self.vr = 0.0
        self.cmd_l = 0.0                # lenh ga buoc truoc (da bao lai)
        self.cmd_r = 0.0
        self.v = 0.0
        self.w = 0.0

        self.battery = 1.0
        self.low_lamp = False           # chot nguong bao sac, co tre

        # ---- tien do de bai (xem sim/params.py muc "de bai")
        self.task_beacons = 0           # da an may cham goi
        self.task_charges = 0           # da sac day may lan HOP LE
        self.charge_valid = False       # lan cam hien tai con duoc tinh khong
        self.new_area = 0.0             # vua thay them bao nhieu cho moi
        self.seen_cells = None          # luoi da di qua, FleetSim cap
        self.charging = False
        self.in_slot = False
        self.id_signal = False
        self.contact = sensors.ContactState()

        self.bump = 0.0
        self.stranded = False           # het pin, nam im
        self.fallen = False             # roi khoi san
        self.brain_lost = False         # mat ket noi bo nao

        # Bo nho tram: toa do trong HE ODOM cua chinh xe, nen troi theo odom.
        self.station = None             # (x, y, axis) hoac None
        self.dist_since_charge = 0.0

        self.lidar = Lidar(seed=seed + 1000 * (robot_id + 1))
        self._rng = np.random.default_rng(seed + 77 * (robot_id + 1))
        common = 1.0 + self._rng.normal(0.0, P.ODOM_SCALE_ERR)
        diff = self._rng.normal(0.0, P.ODOM_DIFF_ERR)
        self._odom_scale_l = common * (1.0 - diff)
        self._odom_scale_r = common * (1.0 + diff)
        self._odom_drift = self._rng.normal(0.0, P.ODOM_DRIFT)

        # thong ke
        self.n_charges = 0
        self.n_wrong_dock = 0
        self.n_flat = 0
        self.n_falls = 0
        self.n_bumps = 0
        self.charge_seconds = 0.0
        self.distance = 0.0

    # ------------------------------------------------------------------ tien ich
    def pose(self):
        return (self.x, self.y, self.th)

    def odom_pose(self):
        return (self.ox, self.oy, self.oth)

    def as_circle(self):
        return (self.x, self.y, self.radius)

    def rear_point(self):
        return sensors.rear_contact_point(self.x, self.y, self.th, self.radius)

    def disabled(self):
        return self.stranded or self.fallen

    def remember_station(self, dock_axis_world):
        """Ghi lai tram trong he odom, ngay tai cho dang cam."""
        axis_odom = wrap_pi_scalar(self.oth + (dock_axis_world - self.th))
        self.station = (self.ox, self.oy, axis_odom)
        self.dist_since_charge = 0.0

    # ------------------------------------------------------------------ dong luc hoc
    def _apply_motor(self, cmd_l, cmd_r, dt):
        cmd_l = float(np.clip(cmd_l, -1.0, 1.0))
        cmd_r = float(np.clip(cmd_r, -1.0, 1.0))
        if self.disabled() or self.brain_lost:
            cmd_l = cmd_r = 0.0
        tgt_l = cmd_l * P.V_MAX
        tgt_r = cmd_r * P.V_MAX
        k = 1.0 - math.exp(-dt / P.MOTOR_TAU)
        self.vl += (tgt_l - self.vl) * k
        self.vr += (tgt_r - self.vr) * k
        return cmd_l, cmd_r

    def step_motion(self, cmd_l, cmd_r, dt, world, circles):
        """Mot buoc dong luc hoc + va cham. Tra ve (da_cham, da_roi)."""
        cmd_l, cmd_r = self._apply_motor(cmd_l, cmd_r, dt)
        # Ghi lai ga THUC SU da chay (da qua moi lop de lenh), vi day la mot
        # trong cac dau vao cua buoc sau. Bao lai ga duoc yeu cau thay vi ga
        # da chay se lam trang thai GRU tren laptop troi khoi thuc te.
        self.cmd_l, self.cmd_r = cmd_l, cmd_r

        v = 0.5 * (self.vl + self.vr)
        w = (self.vr - self.vl) / P.WHEEL_BASE
        self.v, self.w = v, w

        x0, y0 = self.x, self.y
        if abs(w) < 1e-6:
            self.x += v * math.cos(self.th) * dt
            self.y += v * math.sin(self.th) * dt
        else:
            th1 = self.th + w * dt
            r = v / w
            self.x += r * (math.sin(th1) - math.sin(self.th))
            self.y += -r * (math.cos(th1) - math.cos(self.th))
            self.th = wrap_pi_scalar(th1)

        # ---- odometry (xe tu nghi minh o dau): sai so ti le + troi goc
        dl = self.vl * dt * self._odom_scale_l
        dr = self.vr * dt * self._odom_scale_r
        dv = 0.5 * (dl + dr)
        dw = (dr - dl) / P.WHEEL_BASE + self._odom_drift * dt
        self.oth = wrap_pi_scalar(self.oth + dw)
        self.ox += dv * math.cos(self.oth)
        self.oy += dv * math.sin(self.oth)
        self.dist_since_charge += abs(dv)

        hit = self._resolve_collision(world, circles)
        was_bump = self.bump > 0.5
        self.bump = 1.0 if hit else max(0.0, self.bump - dt * 4.0)
        if hit and not was_bump:
            # Dem LAN va cham, khong dem buoc. Mot xe nam ke vao tuong thi
            # moi buoc deu "dang cham", dem tung buoc ra con so vo nghia.
            self.n_bumps += 1

        self.distance += math.hypot(self.x - x0, self.y - y0)

        fell = False
        if not world.on_floor(self.x, self.y):
            if not self.fallen:
                self.n_falls += 1
            self.fallen = True
            self.vl = self.vr = self.v = self.w = 0.0
            fell = True
        return hit, fell

    def _resolve_collision(self, world, circles):
        """Day xe ra khoi vat the. Truot doc be mat chinh la thu giup xe
        chui duoc vao hoc: goc vat o mieng hoc bien lech thanh truot vao."""
        hit = False
        seg = world.static_segments
        for _ in range(3):
            moved = False
            # Chieu tam xe len tung doan mot lan bang numpy. Truoc day o day
            # dung mot SegmentSet moi cho MOI doan bi cham de goi lai ham
            # chung - moi buoc cap phat vai chuc mang numpy ti hon va an het
            # mot phan ba thoi gian chay.
            apx = self.x - seg.ax
            apy = self.y - seg.ay
            denom = seg.ex * seg.ex + seg.ey * seg.ey
            u = np.clip((apx * seg.ex + apy * seg.ey) / np.where(denom < 1e-12, 1.0, denom),
                        0.0, 1.0)
            qx = seg.ax + u * seg.ex
            qy = seg.ay + u * seg.ey
            dx = self.x - qx
            dy = self.y - qy
            d = np.hypot(dx, dy)
            idx = np.nonzero(d < self.radius)[0]
            for i in idx:
                dist = float(d[i])
                if dist < 1e-9:
                    nx, ny = math.cos(self.th + math.pi), math.sin(self.th + math.pi)
                else:
                    nx, ny = float(dx[i]) / dist, float(dy[i]) / dist
                push = self.radius - dist
                self.x += nx * push
                self.y += ny * push
                hit = moved = True
            if not moved:
                break

        for (cx, cy, cr) in circles:
            dx, dy = self.x - cx, self.y - cy
            dist = math.hypot(dx, dy)
            overlap = (self.radius + cr) - dist
            if overlap > 0.0:
                if dist < 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
                self.x += dx / dist * overlap
                self.y += dy / dist * overlap
                hit = True
        return hit

    # ------------------------------------------------------------------ pin
    def step_power(self, dt, contact):
        self.contact = contact
        was_in_slot = self.in_slot
        self.in_slot = contact.in_slot
        self.id_signal = contact.id_signal
        was_charging = self.charging
        self.charging = contact.charging

        if self.charging:
            if not was_charging:
                self.n_charges += 1
                self.remember_station(contact.dock.axis_out())
                # MOT LAN SAC HOP LE bat dau tu day: luc cam vao pin phai
                # dang duoi nguong bao dong. Cam luc pin con day thi lan do
                # khong bao gio duoc tinh, du co nam den khi day.
                self.charge_valid = self.battery < P.BATT_LOW
            self.charge_seconds += dt
            self.battery = min(1.0, self.battery + P.BATT_CHARGE_RATE * dt)
            if self.charge_valid and self.battery >= P.BATT_FULL:
                # Day roi: tinh mot lan, va tat co de khong dem hai lan.
                self.task_charges += 1
                self.charge_valid = False
        else:
            # Roi hoc khi chua day -> lan do mat, phai lam lai tu dau.
            self.charge_valid = False
            if contact.wrong_dock and contact.in_slot and not was_in_slot:
                # Vua cam vao mot hoc khong phai cua minh: chan tiep dien
                # cham that, nhung hoc khong phat tin hieu va khong co dien.
                # Dem mot lan moi lan cam, khong dem tung buoc.
                self.n_wrong_dock += 1
            speed = min(1.0, abs(self.v) / P.V_MAX + 0.35 * abs(self.w) / P.W_MAX)
            drain = P.BATT_IDLE_DRAIN + P.BATT_MOVE_DRAIN * speed
            if self.disabled():
                drain = 0.0
            self.battery = max(0.0, self.battery - drain * dt)

        if self.battery <= 0.0 and not self.charging:
            if not self.stranded:
                self.n_flat += 1
            self.stranded = True
        elif self.battery > 0.05:
            self.stranded = False

        # Den bao sac: bat duoi 15%, tat lai o 18% cho khoi chap chon.
        if self.battery < P.BATT_LOW:
            self.low_lamp = True
        elif self.battery > P.BATT_LOW + 0.03:
            self.low_lamp = False

    def low_battery_blink(self, t):
        """Dau vao nhap nhay: 0/1 lien tuc khi pin duoi 15%, ngoai ra 0.

        Day la TAT CA nhung gi xay ra khi pin thap. Khong co lop nao cuop
        quyen lai. Xe muon phot lo den nay thi cu viec - va se nam duong.
        """
        if not self.low_lamp:
            return 0.0
        return 1.0 if int(t * P.BATT_BLINK_HZ * 2.0) % 2 == 0 else 0.0

    # ------------------------------------------------------------------ cuu ho
    def rescue_to(self, dock):
        """Nguoi nhat xe bo lai vao hoc cua no. Chi dung khi bat --rescue."""
        depth = P.DOCK_CAVITY_D - self.radius
        self.x = dock.x - depth * math.cos(dock.theta)
        self.y = dock.y - depth * math.sin(dock.theta)
        self.th = dock.theta
        self.vl = self.vr = self.v = self.w = 0.0
        self.battery = max(self.battery, 0.25)
        self.stranded = False
        self.fallen = False
        self.lidar.reset()
        self.ox, self.oy, self.oth = self.x, self.y, self.th
        self.station = (self.ox, self.oy, self.oth)
        self.dist_since_charge = 0.0
