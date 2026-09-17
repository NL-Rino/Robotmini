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
CLIFF_CAP = -60.0     # tran cua khoan tren, va no BAT BUOC phai co.
                      # Do duoc: mot xe ket o goc giua buc tuong va cai ho,
                      # cam bien vuc keu suot 300 giay -> -1.713 diem, tuc la
                      # dung canh ho DAT HON LAO XUONG HO (-150). Khong chan
                      # lai thi bo nao se hoc cach nhay xuong cho xong - dung
                      # cai bay (h) cua ban cu, chi khac dau.
SPIN = -0.01          # phat nhe viec quay tit tai cho

# sac
#
# Ba con so duoi day deu la HANG RAO, khong phai tham so tinh chinh. Ban
# truoc khong co chung va bo nao tim ra ngay: ngoi trong hoc quay tit tai
# cho duoc 994 diem, lac ra lac vao duoc 1.300 diem, trong khi di lam viec
# that chi duoc ~300. No khong hong - no dang giai dung cai bai toan ta ra.
CHARGE_ENERGY = 150.0     # nhan voi phan pin nap duoc trong buoc do
CHARGE_LATCH = 30.0       # thuong mot lan khi vua cam dung hoc
AWAY_DIST = 1.5           # phai roi hoc xa chung nay thi lan cam sau moi duoc
                          # tra tien. Khong co no thi xe lac ra lac vao an
                          # +30 moi lan, 42 lan trong mot tap.
FULL_ENOUGH = 0.97
LOITER = -0.10            # moi buoc con nam trong hoc khi pin da day
LOITER_CAP = -80.0        # tran cua khoan tren. Phai co tran: neu de no vuot
                          # qua -150 thi nam li trong hoc dat hon chet, va xe
                          # se hoc cach lao xuong vuc cho xong. Dung cai bay
                          # (h) cua ban cu, chi khac dau.
WRONG_DOCK = -4.0         # cam nham: chi phat NHE, vi do la cach hop le
                          # de biet hoc nao la cua minh

# Trong long hoc thi CHI duoc lui vao va di thang ra. Long hoc rong 31 cm ma
# than xe 30 cm: quay nguoi trong do la co xat hai vach va giat chan tiep
# dien. Chua duoc quay tu do mot phan de con nan huong luc dang lui vao.
SPIN_IN_DOCK = -2.5
SPIN_FREE = 0.35          # phan cua toc do quay toi da duoc quay tu do

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
                 "beacons", "homing_on", "budget", "max_away", "loiter_paid",
                 "spin_in_dock", "cliff_paid")

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
        # Xe xuat phat NGOAI hoc thi coi nhu da di xa roi - no dang o ngoai
        # that. Chi xe xuat phat trong hoc va dang co dien moi phai di lam
        # mot vong roi ve thi lan sac moi duoc tra tien.
        docked = bool(robot.charging)
        self.max_away = 0.0 if docked else AWAY_DIST
        self.budget = 0.0 if docked else max(0.0, 1.0 - robot.battery)
        self.loiter_paid = 0.0
        self.spin_in_dock = 0.0
        self.cliff_paid = 0.0

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

        inside = any(d.contains(robot.x, robot.y) for d in world.docks)

        # va cham: dem theo LAN, va mien khi that su dang trong long hoc
        if robot.n_bumps > self.prev_bumps:
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
            pen = max(CLIFF_WARN, CLIFF_CAP - self.cliff_paid)
            r += pen
            self.cliff_paid += pen

        if abs(robot.w) > 0.8 * P.W_MAX:
            r += SPIN

        # TRONG LONG HOC THI KHONG DUOC QUAY NGUOI. Chi duoc lui vao roi di
        # thang ra; muon xoay thi ra khoi hoc da.
        if inside:
            excess = abs(robot.w) / P.W_MAX - SPIN_FREE
            if excess > 0.0:
                pen = SPIN_IN_DOCK * excess
                r += pen
                self.spin_in_dock -= pen

        # Sac: tra tien theo NANG LUONG nap duoc, nhung moi lan vao hoc chi
        # co mot han muc, dat luc vua cam. Khong co han muc nay thi xe co the
        # xa pin roi nap lai ngay trong hoc, moi vong an them mot khoan.
        d_batt = robot.battery - self.prev_batt
        if robot.charging and d_batt > 0.0 and self.budget > 0.0:
            pay = min(d_batt, self.budget)
            r += CHARGE_ENERGY * pay
            self.budget -= pay
        self.prev_batt = robot.battery

        # Pin da day ma van nam trong hoc thi bat dau lo von. Day la cau tra
        # loi cho "no cu ngoi yen trong sac": ngoi yen khong con mien phi.
        if robot.charging and robot.battery > FULL_ENOUGH:
            pen = max(LOITER, LOITER_CAP - self.loiter_paid)
            r += pen
            self.loiter_paid += pen

        away = math.hypot(robot.x - self.home.x, robot.y - self.home.y)
        if away > self.max_away:
            self.max_away = away

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
        """Thuong mot lan khi vua cam dung hoc cua minh - NEU da di lam ve.

        Cam vao roi rut ra roi cam lai ngay thi khong duoc gi ca. Muon duoc
        tra tien lan nua thi phai thuc su roi hoc di xa, tuc la phai lam mot
        vong viec da.
        """
        if not (robot.charging and not was_charging):
            return 0.0
        if self.max_away < AWAY_DIST:
            return 0.0
        self.max_away = 0.0
        self.budget = max(0.0, 1.0 - robot.battery)
        self.total += CHARGE_LATCH
        return CHARGE_LATCH
