"""Ham phan thuong.

Nguyen tac, rut ra tu nhat ky loi cua ban cu:

  (g) Roi khoi ban phat -40 trong khi di 2 m ve tram duoc +28 -> xe hoc cach
      lao ve tram roi roi. Phat roi phai LON HON moi thu kiem duoc tren
      duong. O day -150.
  (h) Sua xong (g) thi no hoc cach chet re hon: dung quay tai cho cho het
      pin. HAI CAI CHET PHAI CUNG GIA. Het pin cung -150.
  (i) Mien phat va cham voi vach hoc qua rong -> xe nam li vao vach 41% so
      buoc. Chi mien khi that su dang trong long hoc.

Va mot nguyen tac cua rieng ban nay:

  Phan thuong noi xe NEN O DAU, khong bao gio noi xe PHAI LAM GI. Khong co
  so hang nao thuong cho viec lui, cho viec quay dau, hay cho viec hoi ma
  hoc. Ba viec do de bo nao tu tim ra - dung nhu yeu cau "cai nay ko day".
  So hang dan duong duy nhat la khoang cach toi DIEM DUNG TRUOC MIENG HOC,
  va no chi bat khi den bao sac dang sang.
"""

import math

from sim import params as P

# chet
FALL = -150.0
FLAT = -150.0

# song va di lai
ALIVE = 0.02
BUMP = -1.5
CLIFF_WARN = -0.8
SPIN = -0.01          # phat nhe viec quay tit tai cho

# sac
CHARGE_ENERGY = 150.0     # nhan voi phan pin nap duoc trong buoc do
CHARGE_LATCH = 30.0       # thuong mot lan khi vua cam dung hoc
WRONG_DOCK = -4.0         # cam nham: chi phat NHE, vi do la cach hop le
                          # de biet hoc nao la cua minh

# den goi
BEACON = 25.0
BEACON_RADIUS = 0.40

# dan duong ve tram, chi khi den bao sac dang sang
HOMING = 3.0


def approach_point(dock, dist=0.55):
    return (dock.x + dist * math.cos(dock.theta),
            dock.y + dist * math.sin(dock.theta))


class RewardTracker:
    """Theo doi mot xe qua mot lan danh gia."""

    __slots__ = ("rid", "home", "prev_dist", "prev_batt", "prev_bumps",
                 "prev_wrong", "total", "charged", "wrong", "fell", "flat",
                 "beacons", "homing_on")

    def __init__(self, rid, home_dock, robot):
        self.rid = rid
        self.home = home_dock
        self.prev_dist = self._dist(robot)
        self.prev_batt = robot.battery
        self.prev_bumps = robot.n_bumps
        self.prev_wrong = robot.n_wrong_dock
        self.total = 0.0
        self.charged = 0.0
        self.wrong = 0
        self.fell = 0
        self.flat = 0
        self.beacons = 0
        self.homing_on = False

    def _dist(self, robot):
        ax, ay = approach_point(self.home)
        return math.hypot(robot.x - ax, robot.y - ay)

    def step(self, world, robot, dt):
        r = 0.0
        if robot.fallen:
            if not self.fell:
                self.fell = 1
                r += FALL
            self.total += r
            return r
        if robot.stranded:
            if not self.flat:
                self.flat = 1
                r += FLAT
            self.total += r
            return r

        r += ALIVE

        # va cham: dem theo LAN, va mien khi that su dang trong long hoc
        if robot.n_bumps > self.prev_bumps:
            inside = any(d.contains(robot.x, robot.y) for d in world.docks)
            if not inside:
                r += BUMP * (robot.n_bumps - self.prev_bumps)
            self.prev_bumps = robot.n_bumps

        # canh vuc keu: phat ngay ca khi chua roi
        cl = not world.on_floor(
            robot.x + P.CLIFF_RADIUS * math.cos(robot.th + P.CLIFF_ANGLE),
            robot.y + P.CLIFF_RADIUS * math.sin(robot.th + P.CLIFF_ANGLE))
        cr = not world.on_floor(
            robot.x + P.CLIFF_RADIUS * math.cos(robot.th - P.CLIFF_ANGLE),
            robot.y + P.CLIFF_RADIUS * math.sin(robot.th - P.CLIFF_ANGLE))
        if cl or cr:
            r += CLIFF_WARN

        if abs(robot.w) > 0.8 * P.W_MAX:
            r += SPIN

        # sac: thuong theo NANG LUONG thuc su nap duoc
        d_batt = robot.battery - self.prev_batt
        if robot.charging and d_batt > 0.0:
            r += CHARGE_ENERGY * d_batt
        self.prev_batt = robot.battery

        # cam nham hoc: phat nhe
        if robot.n_wrong_dock > self.prev_wrong:
            r += WRONG_DOCK * (robot.n_wrong_dock - self.prev_wrong)
            self.wrong += robot.n_wrong_dock - self.prev_wrong
            self.prev_wrong = robot.n_wrong_dock

        # den goi
        for b in world.beacons:
            if b.on and math.hypot(robot.x - b.x, robot.y - b.y) < BEACON_RADIUS:
                r += BEACON
                self.beacons += 1
                b.on = False
                break

        # dan duong: chi khi den bao sac dang sang, va dan toi DIEM DUNG
        # TRUOC MIENG chu khong phai toi cai hoc. Dan thang toi hoc thi xe
        # bi hut vao suon hoc - cho gan nhat nhung la ngo cut, khong nhin
        # thau long hoc ma cung khong bat duoc den hong ngoai.
        dist = self._dist(robot)
        if robot.low_lamp:
            if self.homing_on:
                r += HOMING * (self.prev_dist - dist)
            self.homing_on = True
        else:
            self.homing_on = False
        self.prev_dist = dist

        self.charged += max(0.0, d_batt) if robot.charging else 0.0
        self.total += r
        return r

    def latch_charge(self, robot, was_charging):
        """Thuong mot lan khi vua cam dung hoc cua minh."""
        if robot.charging and not was_charging:
            self.total += CHARGE_LATCH
            return CHARGE_LATCH
        return 0.0
