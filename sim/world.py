"""Mat bang: phong, vat can, nguoi di lai, den goi, va hoc sac chu U.

Diem moi so voi ban cu: MOI HOC SAC CO MOT MA RIENG (`code`). Hoc chi phat
tin hieu bat tay va chi cap dien khi ma cua no trung ma cua xe dang cam vao.
Nhin tu ngoai thi nam cai hoc giong het nhau.
"""

import math
import random

from . import params as P
from .geometry import (
    SegmentSet,
    point_in_polygon,
    to_world,
    wrap_pi_scalar,
)


class Dock:
    """Hoc chu U. `theta` la huong TRUC RA (tu trong hoc nhin ra cua).

    He toa do rieng: goc o giua mieng hoc, +x chi ra ngoai.
    """

    def __init__(self, x, y, theta, code, ir_on=True, powered=True, name=""):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_pi_scalar(float(theta))
        self.code = code                  # ma rieng; None = hoc moi nhu
        self.ir_on = bool(ir_on)          # hoc moi nhu khong phat hong ngoai
        self.powered = bool(powered)
        self.name = name or (f"dock{code}" if code is not None else "moi-nhu")
        self.occupied_by = None           # id xe dang nam trong hoc

        w2 = 0.5 * P.DOCK_OUTER_W
        cav2 = 0.5 * P.DOCK_CAVITY_W
        depth = P.DOCK_CAVITY_D
        outer_d = P.DOCK_OUTER_D
        ch = P.DOCK_CHAMFER

        # Bien da giac vat lieu, di theo thu tu khong tu cat.
        self._local_poly = [
            (0.0, w2),
            (-outer_d, w2),
            (-outer_d, -w2),
            (0.0, -w2),
            (-ch, -cav2),
            (-depth, -cav2),
            (-depth, cav2),
            (-ch, cav2),
        ]
        self.polygon = [to_world(lx, ly, self.x, self.y, self.theta)
                        for lx, ly in self._local_poly]
        self.segments = SegmentSet.from_polyline(self.polygon, closed=True,
                                                 tag=("dock", self.name))

        # Long trong: dung de biet xe "da vao hoc" hay chua.
        self._cavity_local = [
            (0.0, cav2), (-depth, cav2), (-depth, -cav2), (0.0, -cav2),
        ]
        self.cavity = [to_world(lx, ly, self.x, self.y, self.theta)
                       for lx, ly in self._cavity_local]

        # Den hong ngoai va tiep diem: deu o giua thanh trong.
        self.ir_x, self.ir_y = to_world(-depth + 0.002, 0.0, self.x, self.y, self.theta)
        self.contact_x, self.contact_y = to_world(-depth, 0.0, self.x, self.y, self.theta)
        self.mouth_x, self.mouth_y = self.x, self.y

    def axis_out(self):
        return self.theta

    def contains(self, px, py):
        return point_in_polygon(px, py, self.cavity)

    def approach_point(self, dist=0.55):
        """Diem dung truoc mieng hoc, tren truc. Dung cho ban luat mau."""
        return (self.x + dist * math.cos(self.theta),
                self.y + dist * math.sin(self.theta))

    def __repr__(self):
        return f"<Dock {self.name} code={self.code} at ({self.x:.2f},{self.y:.2f}) th={math.degrees(self.theta):.0f}>"


class Beacon:
    """Den hong ngoai goi xe toi. Kenh khac voi kenh hoc sac."""

    def __init__(self, x, y, on=True):
        self.x = float(x)
        self.y = float(y)
        self.on = bool(on)

    def step(self, dt, rng):
        # Thinh thoang tat/bat de xe khong coi no la coc tieu co dinh.
        if rng.random() < 0.002:
            self.on = not self.on


class Mover:
    """Nguoi di lai: hinh tron, di theo cac diem moc, doi huong ngau nhien."""

    def __init__(self, x, y, radius=0.18, speed=0.45, waypoints=()):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.speed = float(speed)
        self.waypoints = list(waypoints)
        self.idx = 0
        self.pause = 0.0

    def step(self, dt, rng):
        if not self.waypoints:
            return
        if self.pause > 0.0:
            self.pause -= dt
            return
        tx, ty = self.waypoints[self.idx]
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        if d < 0.08:
            self.idx = (self.idx + 1) % len(self.waypoints)
            if rng.random() < 0.35:
                self.pause = rng.uniform(0.5, 2.5)
            return
        step = min(self.speed * dt, d)
        self.x += dx / d * step
        self.y += dy / d * step

    def as_circle(self):
        return (self.x, self.y, self.radius)


class World:
    """Mat bang tinh + cac thu dong day.

    `floor` la da giac san nha; ra ngoai no la VUC (mep ban / cau thang).
    `voids` la cac lo thung khoet trong san.
    """

    def __init__(self, floor, walls=None, obstacles=None, docks=(), beacons=(),
                 movers=(), voids=(), name="map"):
        self.name = name
        self.floor = list(floor)
        self.voids = [list(v) for v in voids]
        self.walls = walls if walls is not None else SegmentSet()
        self.obstacles = obstacles if obstacles is not None else SegmentSet()
        self.docks = list(docks)
        self.beacons = list(beacons)
        self.movers = list(movers)
        self._rebuild()

    def _rebuild(self):
        parts = [self.walls, self.obstacles]
        parts.extend(d.segments for d in self.docks)
        self.static_segments = SegmentSet.concat(parts)
        xs = [p[0] for p in self.floor]
        ys = [p[1] for p in self.floor]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))

    def add_dock(self, dock):
        self.docks.append(dock)
        self._rebuild()

    def on_floor(self, x, y):
        """False = duoi chan la khoang khong (vuc)."""
        if not point_in_polygon(x, y, self.floor):
            return False
        for v in self.voids:
            if point_in_polygon(x, y, v):
                return False
        return True

    def dock_by_code(self, code):
        for d in self.docks:
            if d.code == code:
                return d
        return None

    def step_dynamics(self, dt, rng):
        for m in self.movers:
            m.step(dt, rng)
        for b in self.beacons:
            b.step(dt, rng)

    def mover_circles(self):
        return [m.as_circle() for m in self.movers]


# --------------------------------------------------------------------------
# Sinh mat bang
# --------------------------------------------------------------------------

def _wall_rect(x0, y0, x1, y1):
    return SegmentSet.from_polyline(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, tag=("wall", None))


def make_fleet_map(seed=0, n_docks=5, width=6.4, height=4.8, n_movers=3,
                   n_decoys=1, with_void=True):
    """Mat bang mac dinh: mot phong, N hoc sac ma khac nhau dat sat tuong.

    Moi hoc mot ma. Ngoai ra co the tha them vai hoc MOI NHU: giong het ve
    hinh dang nhung khong phat hong ngoai va khong co dien.
    """
    rng = random.Random(seed)

    floor = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    walls = _wall_rect(0.02, 0.02, width - 0.02, height - 0.02)

    # Cho dat hoc: lung ap sat tuong, mieng huong vao trong phong.
    # Mieng hoc phai lui vao du DOCK_OUTER_D thi than hoc moi nam trong phong.
    back_off = 0.02 + P.DOCK_OUTER_D
    margin = 0.55
    slots = []
    n_bottom = max(2, n_docks // 2 + 1)
    n_top = n_docks + n_decoys - n_bottom
    for i in range(n_bottom):
        fx = margin + (width - 2 * margin) * (i + 0.5) / max(1, n_bottom)
        slots.append((fx, back_off, math.pi / 2))            # tuong duoi, mieng huong len
    for i in range(max(0, n_top)):
        fx = margin + (width - 2 * margin) * (i + 0.5) / max(1, n_top)
        slots.append((fx, height - back_off, -math.pi / 2))  # tuong tren, mieng huong xuong
    rng.shuffle(slots)

    docks = []
    for i in range(n_docks):
        x, y, th = slots[i]
        docks.append(Dock(x, y, th, code=101 + i, name=f"D{i + 1}"))
    for j in range(n_decoys):
        if n_docks + j >= len(slots):
            break
        x, y, th = slots[n_docks + j]
        docks.append(Dock(x, y, th, code=None, ir_on=False, powered=False,
                          name=f"moi{j + 1}"))

    # Vat can giua phong, tranh cho truoc mieng hoc.
    obs_parts = []
    tries = 0
    placed = []
    while len(placed) < 4 and tries < 200:
        tries += 1
        w = rng.uniform(0.35, 0.8)
        h = rng.uniform(0.35, 0.8)
        cx = rng.uniform(1.0, width - 1.0)
        cy = rng.uniform(1.2, height - 1.2)
        ok = True
        for d in docks:
            ax, ay = d.approach_point(0.95)
            if math.hypot(cx - ax, cy - ay) < 0.95:
                ok = False
                break
            if math.hypot(cx - d.x, cy - d.y) < 1.1:
                ok = False
                break
        for px, py, pw, ph in placed:
            if abs(cx - px) < 0.5 * (w + pw) + 0.8 and abs(cy - py) < 0.5 * (h + ph) + 0.8:
                ok = False
                break
        if ok:
            placed.append((cx, cy, w, h))
            obs_parts.append(SegmentSet.from_rect(cx, cy, w, h, rng.uniform(0, math.pi),
                                                  tag=("obstacle", None)))
    obstacles = SegmentSet.concat(obs_parts) if obs_parts else SegmentSet()

    movers = []
    for _ in range(n_movers):
        wps = [(rng.uniform(0.8, width - 0.8), rng.uniform(1.0, height - 1.0))
               for _ in range(3)]
        movers.append(Mover(wps[0][0], wps[0][1], waypoints=wps,
                            speed=rng.uniform(0.3, 0.55)))

    beacons = [Beacon(rng.uniform(1.0, width - 1.0), rng.uniform(1.2, height - 1.2))
               for _ in range(2)]

    voids = []
    if with_void:
        # Mot o thung (cau thang) sat tuong phai, xa mieng hoc.
        vx = width - 0.75
        vy = 0.5 * height
        voids.append([(vx - 0.35, vy - 0.55), (vx + 0.35, vy - 0.55),
                      (vx + 0.35, vy + 0.55), (vx - 0.35, vy + 0.55)])

    return World(floor, walls, obstacles, docks, beacons, movers, voids,
                 name=f"fleet-{seed}")
