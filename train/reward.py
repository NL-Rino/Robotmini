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

DE BAI (xem sim/params.py muc "de bai"): mot lan chay HOAN THANH khi xe da
an du 5 cham goi VA sac du 3 lan hop le. Mot lan sac hop le: cam vao luc
pin DUOI 20%, va nam yen cho toi khi DAY 100%. Rut ra giua chung thi khong
duoc gi - ke ca phan da tra dan trong luc sac cung bi THU LAI.

Va: moi o san nha lan dau lot vao tam quet LiDAR duoc cong mot it. Do la
cai day xe di qua cua sang phong khac thay vi quanh quan mot cho.
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
#
# Gio chi tra cho lan sac HOP LE (cam luc pin < 20%). Tien tra dan trong luc
# sac chi la "tien tam ung": rut ra truoc khi day thi bi thu lai het, dung
# nhu de bai "trong luc sac chay ra thi khong duoc cong diem".
CHARGE_ENERGY = 60.0      # nhan voi phan pin nap duoc - tam ung
CHARGE_LATCH = 20.0       # tam ung mot lan khi vua cam dung hoc, pin < 20%
CHARGE_DONE = 60.0        # tra THAT khi day 100%: mot lan sac duoc tinh
AWAY_DIST = 1.5           # (giu ten cu cho kiem thu; gio hang rao chinh la
                          # dieu kien pin < 20% luc cam)
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

# den goi: tra theo SO CHAM DA AN ma robot tu dem (sim/fleet.py an cham)
BEACON = 25.0
BEACON_EXTRA = 3.0        # an qua 5 cham thi van co chut it, nhung khong dang
BEACON_RADIUS = P.BEACON_RADIUS

# kham pha: moi o 50 cm lan dau lot vao tam quet. Can nha 432 o, nen di het
# ca nha cung chi ~130 - khong bang lam xong de bai.
EXPLORE = 0.3

# lam xong de bai: 5 cham + 3 lan sac hop le
COMPLETE = 300.0

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
                 "spin_in_dock", "cliff_paid", "prev_beacons",
                 "prev_charges", "advance", "charges_ok", "cells",
                 "complete", "clawed")

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
        # Tien do de bai tinh tu LUC BAT DAU danh gia: giao trinh co the cap
        # san "da an 3 cham, da sac 2 lan" - phan do khong duoc tra lai.
        self.prev_beacons = robot.task_beacons
        self.prev_charges = robot.task_charges
        self.advance = 0.0        # tien tam ung cua lan sac dang do
        self.charges_ok = 0
        self.cells = 0
        self.complete = int(robot.task_beacons >= P.TASK_BEACONS
                            and robot.task_charges >= P.TASK_CHARGES)
        self.clawed = 0.0

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

        # Sac: CHI lan sac hop le (cam luc pin < 20%) moi duoc tam ung theo
        # nang luong nap. Day 100% -> lan sac duoc tinh, tra CHARGE_DONE.
        # Rut ra truoc khi day -> thu lai toan bo tien tam ung.
        d_batt = robot.battery - self.prev_batt
        if robot.task_charges > self.prev_charges:
            n = robot.task_charges - self.prev_charges
            r += CHARGE_DONE * n
            self.charges_ok += n
            self.prev_charges = robot.task_charges
            self.advance = 0.0
        elif robot.charging and robot.charge_valid and d_batt > 0.0:
            pay = CHARGE_ENERGY * d_batt
            r += pay
            self.advance += pay
        elif self.advance > 0.0 and not robot.charge_valid:
            r -= self.advance
            self.clawed += self.advance
            self.advance = 0.0
        self.prev_batt = robot.battery

        # Cham goi: robot tu dem trong sim; o day chi doc so dem.
        if robot.task_beacons > self.prev_beacons:
            for k in range(self.prev_beacons, robot.task_beacons):
                r += BEACON if k < P.TASK_BEACONS else BEACON_EXTRA
            self.beacons += robot.task_beacons - self.prev_beacons
            self.prev_beacons = robot.task_beacons

        # Kham pha: o moi trong vong quet vua roi.
        if robot.new_area > 0.0:
            n = int(round(robot.new_area * 12.0))
            r += EXPLORE * n
            self.cells += n

        if (not self.complete and robot.task_beacons >= P.TASK_BEACONS
                and robot.task_charges >= P.TASK_CHARGES):
            self.complete = 1
            r += COMPLETE

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
        """Tam ung mot lan khi vua cam dung hoc cua minh - NEU pin < 20%.

        Cam luc pin con nhieu thi khong duoc gi (lan do khong bao gio tinh).
        Tien nay la tam ung: rut ra truoc khi day thi `step` thu lai.
        """
        if not (robot.charging and not was_charging):
            return 0.0
        if not robot.charge_valid:
            return 0.0
        self.max_away = 0.0
        self.advance += CHARGE_LATCH
        self.total += CHARGE_LATCH
        return CHARGE_LATCH
