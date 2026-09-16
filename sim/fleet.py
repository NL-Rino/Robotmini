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

from . import params as P
from . import sensors
from . import dock_detector
from . import perception
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
        self.world = world if world is not None else make_fleet_map(seed)
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

    # ------------------------------------------------------------------ cam nhan
    def _circles_for(self, robot):
        out = self.world.mover_circles()
        for o in self.robots:
            if o.id != robot.id:
                out.append(o.as_circle())
        return out

    def observe(self):
        """Doc cam bien cua tat ca cac xe, tra ve {id: vector 48 so}."""
        obs = {}
        for r in self.robots:
            circles = self._circles_for(r)
            scan = r.lidar.update(self.dt, r.pose(), r.odom_pose(),
                                  self.world.static_segments, circles, self.t)
            if scan is not None:
                self._cands[r.id] = dock_detector.detect(scan)
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
