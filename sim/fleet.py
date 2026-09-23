"""Vong lap the gioi: NHIEU XE tren mot mat bang, chay lien tuc.

Hai diem khac han ban cu:

  1. KHONG CO TAP (episode). Khong co gioi han so buoc, khong reset. Vong lap
     chi dung khi DUT KET NOI BO NAO - tat ca cac xe deu mat nao. Xe het pin
     hay roi xuong vuc thi nam im tai cho va tro thanh vat can cua xe khac;
     the gioi van chay tiep.

  2. Mot mat bang tha 5 xe, MOI XE MOT MA HOC SAC RIENG. Nam cai hoc nhin tu
     ngoai giong het nhau. Xe chi biet do co phai hoc cua minh khong khi da
     lui duoi vao va chan tiep dien cham - luc do hoc moi phat tin hieu.
"""

import math
import random
import time

import numpy as np

from . import params as P
from . import sensors
from . import dock_detector
from . import perception
from .coverage import Grid
from .geometry import point_segment_distance, wrap_pi_scalar
from .robot import Robot
from .world import make_fleet_map


class BrainLink:
    """Duong day toi bo nao. Vong lap song chet theo cai nay."""

    def connected(self, robot_id):
        raise NotImplementedError

    def any_connected(self):
        raise NotImplementedError

    def act(self, robot_id, obs, t):
        raise NotImplementedError

    def close(self):
        pass


class LocalBrains(BrainLink):
    """Nao chay thang trong tien trinh nay (xem chay, kiem thu).

    "Ngat ket noi" o day la goi disconnect() - tu ban phim, tu giao dien,
    hay tu Ctrl-C.
    """

    def __init__(self, factory, robot_ids):
        self.brains = {rid: factory(rid) for rid in robot_ids}
        self.live = {rid: True for rid in robot_ids}

    def connected(self, robot_id):
        return self.live.get(robot_id, False)

    def any_connected(self):
        return any(self.live.values())

    def act(self, robot_id, obs, t):
        return self.brains[robot_id](obs, t)

    def disconnect(self, robot_id=None):
        if robot_id is None:
            for k in self.live:
                self.live[k] = False
        else:
            self.live[robot_id] = False

    def close(self):
        self.disconnect()


class RunReport:
    def __init__(self):
        self.steps = 0
        self.sim_seconds = 0.0
        self.wall_seconds = 0.0
        self.stop_reason = ""
        self.per_robot = {}

    def __repr__(self):
        return (f"<Run {self.steps} buoc / {self.sim_seconds:.0f} s mo phong, "
                f"dung vi: {self.stop_reason}>")


class FleetSim:
    def __init__(self, world=None, n_robots=5, seed=0, dt=P.DT,
                 rescue_seconds=None, log_limit=4000):
        self.world = (world if world is not None
                      else make_fleet_map(seed, n_docks=max(3, n_robots)))
        self.dt = float(dt)
        self.seed = seed
        self.rescue_seconds = rescue_seconds
        self.t = 0.0
        self.steps = 0
        # RNG rieng cua tung mo phong: hai FleetSim cung seed phai cho ra
        # dung mot ket qua, khong an ke nhau qua mot bien toan cuc.
        self._rng = random.Random(90210 + seed)
        self.log = []
        self.log_limit = log_limit

        coded = [d for d in self.world.docks if d.code is not None]
        if n_robots > len(coded):
            raise ValueError(
                f"mat bang chi co {len(coded)} hoc co ma, khong du cho "
                f"{n_robots} xe (moi xe phai co mot ma rieng)")
        self.home = {}
        self.robots = []
        for i in range(n_robots):
            d = coded[i]
            depth = P.DOCK_CAVITY_D - P.BODY_RADIUS - 0.005
            x = d.x - depth * math.cos(d.theta)
            y = d.y - depth * math.sin(d.theta)
            # Xuat phat TU TRONG HOC, DUOI o phia trong, mui huong ra cua.
            r = Robot(i, x, y, d.theta, dock_code=d.code, seed=seed + i)
            r.station = (r.ox, r.oy, r.oth)
            self.robots.append(r)
            self.home[i] = d
        self.robot_ids = [r.id for r in self.robots]
        self._cands = {r.id: [] for r in self.robots}
        self._disabled_since = {r.id: None for r in self.robots}

        # Luoi "da nhin thay cho nao": moi xe mot ban rieng. Xe A di het
        # tang tren khong lam xe B duoc diem kham pha.
        self.grid = Grid(self.world.bounds)
        for r in self.robots:
            r.seen_cells = np.zeros(self.grid.n, dtype=bool)

        # Doc tiep dien ngay tu buoc 0. Xe dang nam trong hoc cua no that,
        # nen dau vao "dang cam" va "hoc phat tin hieu" phai dung ngay tu
        # lan quan sat dau tien, truoc khi co buoc dong luc hoc nao chay.
        for r in self.robots:
            c = sensors.dock_contact(self.world, r.x, r.y, r.th, r.code)
            r.contact = c
            r.in_slot = c.in_slot
            r.id_signal = c.id_signal
            r.charging = c.charging
            if c.dock is not None:
                c.dock.occupied_by = r.id

    def place_robot(self, rid, x, y, th, battery=None, station=None,
                    station_drift=0.0, rng=None):
        """Dat mot xe vao trang thai bat ky. Dung cho giao trinh nguoc.

        `station_drift` la sai so cua BO NHO TRAM tinh bang met. Xe that chay
        mot lat la cho nho lech vai met; neu luc huan luyen lan nao cho nho
        cung dung chinh xac thi bo nao se hoc cach tin vao no, roi ra doi
        gap cho nho lech 4 m la chiu.
        """
        r = self.robots[rid]
        r.x, r.y, r.th = float(x), float(y), wrap_pi_scalar(float(th))
        r.vl = r.vr = r.v = r.w = 0.0
        r.cmd_l = r.cmd_r = 0.0
        r.bump = 0.0
        r.stranded = False
        r.fallen = False
        r.brain_lost = False
        r.lidar.reset()
        r.ox, r.oy, r.oth = r.x, r.y, r.th
        if battery is not None:
            r.battery = float(battery)
            r.low_lamp = r.battery < P.BATT_LOW
        r.dist_since_charge = 0.0
        r.charge_valid = False
        r.new_area = 0.0

        d = self.home[rid] if station is None else station
        sx, sy, sth = d.x, d.y, d.theta
        if station_drift > 0.0:
            g = rng if rng is not None else self._rng
            ang = g.uniform(-math.pi, math.pi)
            sx += station_drift * math.cos(ang)
            sy += station_drift * math.sin(ang)
            sth = wrap_pi_scalar(sth + g.gauss(0.0, 0.12 * station_drift))
        # Bo nho tram nam trong HE ODOM cua xe; ngay sau khi dat lai thi he
        # odom trung he that, nen ghi thang toa do vao duoc.
        r.station = (sx, sy, sth)

        c = sensors.dock_contact(self.world, r.x, r.y, r.th, r.code)
        r.contact = c
        r.in_slot = c.in_slot
        r.id_signal = c.id_signal
        r.charging = c.charging
        self._disabled_since[rid] = None
        self._cands[rid] = []

    def free_pose(self, rng, tries=60):
        """Mot cho dung duoc trong phong: tren san, khong dam vao gi."""
        x0, y0, x1, y1 = self.world.bounds
        for _ in range(tries):
            x = rng.uniform(x0 + 0.4, x1 - 0.4)
            y = rng.uniform(y0 + 0.4, y1 - 0.4)
            if not self.world.on_floor(x, y):
                continue
            d = point_segment_distance(x, y, self.world.static_segments)
            if d.size and float(d.min()) < P.BODY_RADIUS + 0.10:
                continue
            # Chan ban ghe va nguoi: dat xe de len chan ghe thi buoc dau
            # tien da la mot cu va.
            if any(math.hypot(x - cx, y - cy) < P.BODY_RADIUS + cr + 0.08
                   for cx, cy, cr in self.world.solid_circles()):
                continue
            if any(math.hypot(x - r.x, y - r.y) < 2 * P.BODY_RADIUS + 0.1
                   for r in self.robots):
                continue
            return x, y, rng.uniform(-math.pi, math.pi)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return cx, cy, rng.uniform(-math.pi, math.pi)

    # ------------------------------------------------------------------ cam nhan
    def _circles_for(self, robot):
        """Thu CHAN DUONG xe: chan ban ghe, than nguoi, va cac xe khac."""
        out = self.world.solid_circles()
        for o in self.robots:
            if o.id != robot.id:
                out.append(o.as_circle())
        return out

    def _seen_by(self, robot):
        """Thu LiDAR NHIN THAY. Khac cai tren o cho nguoi: quet thay HAI CAI
        CHAN, mot cai nhap nhay theo nhip buoc, chu khong phai mot khoi
        tron; con va cham thi van tinh bang than nguoi."""
        out = self.world.lidar_circles()
        for o in self.robots:
            if o.id != robot.id:
                out.append(o.as_circle())
        return out

    def observe(self):
        """Doc cam bien cua tat ca cac xe, tra ve {id: vector 64 so}."""
        obs = {}
        for r in self.robots:
            circles = self._seen_by(r)
            scan = r.lidar.update(self.dt, r.pose(), r.odom_pose(),
                                  self.world.static_segments, circles, self.t)
            r.new_area = 0.0
            if scan is not None:
                self._cands[r.id] = dock_detector.detect(scan)
                # Vong quet vua xong: danh dau nhung o no nhin thay, va dem
                # so o LAN DAU duoc nhin thay.
                moi = self.grid.new_cells(r.seen_cells, r.x, r.y, r.th, scan)
                r.new_area = moi / 12.0
            cliff = sensors.cliff_sensors(self.world, r.x, r.y, r.th)
            irb = sensors.ir_beacon(self.world, r.x, r.y, r.th, circles)
            ird = sensors.ir_dock(self.world, r.x, r.y, r.th, circles)
            obs[r.id] = perception.build(r, r.lidar.scan, self._cands[r.id],
                                         cliff, irb, ird, self.t)
        return obs

    # ------------------------------------------------------------------ mot buoc
    def step(self, commands):
        dt = self.dt
        self.world.step_dynamics(dt, self._rng)

        for r in self.robots:
            cl, cr = commands.get(r.id, (0.0, 0.0))
            circles = self._circles_for(r)
            was_fallen = r.fallen
            r.step_motion(cl, cr, dt, self.world, circles)
            if r.fallen and not was_fallen:
                self._event(r, "roi khoi san")

        self._collect_beacons()
        self._respawn_beacons()

        for r in self.robots:
            was_charge = r.charging
            n_wrong = r.n_wrong_dock
            was_flat = r.stranded
            c = sensors.dock_contact(self.world, r.x, r.y, r.th, r.code,
                                     engaged=r.in_slot)
            r.step_power(dt, c)
            if r.charging and not was_charge:
                self._event(r, f"cam dung hoc {c.dock.name}, bat dau sac")
            if r.n_wrong_dock > n_wrong:
                self._event(r, f"cam nham hoc {c.dock.name} (ma {c.dock.code}), "
                               f"khong co dien")
            if r.stranded and not was_flat:
                self._event(r, "het pin, nam duong")

        for d in self.world.docks:
            d.occupied_by = None
        for r in self.robots:
            if r.contact.dock is not None:
                r.contact.dock.occupied_by = r.id

        self._maybe_rescue()
        self.t += dt
        self.steps += 1

    def _collect_beacons(self):
        """Xe cham vao cham goi dang sang thi an no.

        De o DAY chu khong o ham phan thuong: so cham da an la mot thu robot
        that biet ve chinh no, va no la mot trong 64 dau vao.
        """
        for r in self.robots:
            if r.disabled():
                continue
            for b in self.world.beacons:
                if b.on and math.hypot(r.x - b.x, r.y - b.y) < P.BEACON_RADIUS:
                    b.on = False
                    b.wait = self._rng.uniform(*P.BEACON_RESPAWN)
                    r.task_beacons += 1
                    self._event(r, f"an cham goi ({r.task_beacons}"
                                   f"/{P.TASK_BEACONS})")
                    break

    def _respawn_beacons(self):
        """Cham da an het gio cho thi sang lai o mot cho moi, xa cac xe.

        Can vi ca dan xe dung chung mot bo cham: khong sang lai thi ba xe
        tranh nhau nam cham, khong xe nao an du nam.
        """
        for b in self.world.beacons:
            if b.on or b.wait > 0.0:
                continue
            spot = self.beacon_spot(self._rng)
            if spot is None:
                b.wait = 1.0
                continue
            b.x, b.y = spot
            b.on = True

    def beacon_spot(self, rng, tries=40):
        """Mot cho dat cham goi: tren san, khong sat tuong, xa moi xe."""
        x0, y0, x1, y1 = self.world.bounds
        legs = self.world.solid_circles()
        for _ in range(tries):
            x = rng.uniform(x0 + 0.5, x1 - 0.5)
            y = rng.uniform(y0 + 0.5, y1 - 0.5)
            if not self.world.on_floor(x, y):
                continue
            d = point_segment_distance(x, y, self.world.static_segments)
            if d.size and float(d.min()) < 0.35:
                continue
            if any(math.hypot(x - cx, y - cy) < cr + 0.25
                   for cx, cy, cr in legs):
                continue
            if any(math.hypot(x - r.x, y - r.y) < 1.5 for r in self.robots):
                continue
            if any(o.on and math.hypot(x - o.x, y - o.y) < 1.0
                   for o in self.world.beacons):
                continue
            return x, y
        return None

    def _event(self, robot, text):
        if len(self.log) < self.log_limit:
            self.log.append((self.t, robot.id, text))

    def _maybe_rescue(self):
        if self.rescue_seconds is None:
            return
        for r in self.robots:
            if r.disabled():
                if self._disabled_since[r.id] is None:
                    self._disabled_since[r.id] = self.t
                elif self.t - self._disabled_since[r.id] >= self.rescue_seconds:
                    r.rescue_to(self.home[r.id])
                    self._disabled_since[r.id] = None
                    self._event(r, "duoc nhat bo lai vao hoc")
            else:
                self._disabled_since[r.id] = None

    # ------------------------------------------------------------------ vong lap
    def run(self, link, max_seconds=None, on_step=None, realtime=False):
        """Chay cho toi khi DUT KET NOI BO NAO.

        `max_seconds` chi de kiem thu tu dong dung lai; mac dinh la None,
        tuc la khong co gioi han - dung dung nghia "chay mai".
        """
        rep = RunReport()
        wall0 = time.time()
        stop = "bo nao ngat ket noi"
        while True:
            if not link.any_connected():
                break
            if max_seconds is not None and self.t >= max_seconds:
                stop = f"cham tran {max_seconds:g}s (chi dung khi kiem thu)"
                break

            obs = self.observe()
            cmds = {}
            for r in self.robots:
                if link.connected(r.id):
                    r.brain_lost = False
                    try:
                        cmds[r.id] = link.act(r.id, obs[r.id], self.t)
                    except Exception:
                        cmds[r.id] = (0.0, 0.0)
                        raise
                else:
                    if not r.brain_lost:
                        self._event(r, "mat ket noi bo nao, dung banh")
                    r.brain_lost = True
                    cmds[r.id] = (0.0, 0.0)
            self.step(cmds)

            if on_step is not None:
                on_step(self)
            if realtime:
                lag = wall0 + self.t - time.time()
                if lag > 0:
                    time.sleep(lag)

        # Vong lap thoat vi khong con xe nao co nao. Nhung xe vua mat nao o
        # dung vong cuoi cung chua kip duoc danh dau trong than vong lap
        # (dieu kien thoat kiem tra truoc), nen danh dau o day.
        for r in self.robots:
            if not link.connected(r.id):
                if not r.brain_lost:
                    self._event(r, "mat ket noi bo nao, dung banh")
                r.brain_lost = True
                r.cmd_l = r.cmd_r = 0.0

        rep.steps = self.steps
        rep.sim_seconds = self.t
        rep.wall_seconds = time.time() - wall0
        rep.stop_reason = stop
        rep.per_robot = {r.id: self.stats(r) for r in self.robots}
        return rep

    def stats(self, r):
        return {
            "ma": r.code,
            "pin": round(r.battery, 3),
            "lan_sac": r.n_charges,
            "cam_nham_hoc": r.n_wrong_dock,
            "het_pin": r.n_flat,
            "roi": r.n_falls,
            "va_cham": r.n_bumps,
            "quang_duong_m": round(r.distance, 1),
            "giay_dang_sac": round(r.charge_seconds, 1),
            "trang_thai": ("roi" if r.fallen else
                           "het pin" if r.stranded else
                           "dang sac" if r.charging else
                           "cam nham hoc" if r.in_slot else "dang chay"),
        }
