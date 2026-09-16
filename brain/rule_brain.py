"""Bo luat viet tay - BAN MAU DE XEM MO PHONG CHAY, khong phai giao an.

Bo luat nay chi doc 48 dau vao giong het bo nao hoc duoc. No khong duoc
nhin vao toa do that, khong biet hoc nao mang ma nao, khong biet no dang o
dau tren ban do. Nho vay no chung minh duoc mot dieu: 48 dau vao la DU de
lam tron quy trinh - long nhong, thay den bao sac thi ve, lui duoi vao hoc,
hoi tin hieu, cam nham thi rut ra tim hoc khac.

LUU Y: dung dung file nay lam giao an bat chuoc cho phan CAM HOC. Viec
"phai lui duoi vao" va "phai hoi ma hoc" la de bo nao tu kham pha qua
phan thuong, khong day truoc. Bo luat nay chi de nguoi xem kiem chung rang
mo phong chay dung.
"""

import math
import random

from sim import params as P
from sim import perception as PC


def _drive(v, w):
    """Doi (toc do thang, toc do quay) ra hai ga banh trong [-1, 1]."""
    half = 0.5 * P.WHEEL_BASE * w
    l = (v - half) / P.V_MAX
    r = (v + half) / P.V_MAX
    m = max(1.0, abs(l), abs(r))
    return l / m, r / m


def _fan(obs, k):
    """Khoang cach met o quat k (0 = truoc mui, tang theo chieu trai)."""
    return (1.0 - float(obs[PC.I_FANS + (k % P.N_LIDAR_FANS)])) * P.LIDAR_MAX


def _cand(obs, k):
    """Ung vien hoc thu k: (diem, cu ly, phuong vi, goc truc ra) he quy chieu xe."""
    b = PC.I_DOCKS + 6 * k
    score = float(obs[b])
    if score <= 0.02:
        return None
    rng_ = (1.0 - float(obs[b + 1])) * PC.DOCK_RANGE_NORM
    bearing = math.atan2(float(obs[b + 2]), float(obs[b + 3]))
    axis = math.atan2(float(obs[b + 4]), float(obs[b + 5]))
    return score, rng_, bearing, axis


class _DockTrack:
    """Bam mot cai hoc qua nhieu vong quet.

    Camsense chi xong mot vong moi ~167 ms, va o cu ly xa thi co vong thay
    co vong khong. Neu cu thay mat ung vien la bo cuoc thi may trang thai
    giat nhu dong kinh va khong bao gio cam duoc. Giua hai vong thi doi vi
    tri da nho theo chinh chuyen dong cua xe (v va w deu co trong dau vao).
    """

    HOLD = 3.0      # giu bao lau sau lan thay cuoi
    GATE = 0.45     # ung vien moi lech qua the nay thi coi la hoc khac

    def __init__(self):
        self.x = self.y = self.axis = 0.0
        self.score = 0.0
        self.t_seen = -1e9
        self.alive = False

    def predict(self, v, w, dt):
        if not self.alive:
            return
        self.x -= v * dt
        a = -w * dt
        c, s = math.cos(a), math.sin(a)
        self.x, self.y = self.x * c - self.y * s, self.x * s + self.y * c
        self.axis = math.atan2(math.sin(self.axis + a), math.cos(self.axis + a))

    def update(self, cands, t):
        best = None
        for c in cands:
            if c is None:
                continue
            cx = c[1] * math.cos(c[2])
            cy = c[1] * math.sin(c[2])
            if self.alive and math.hypot(cx - self.x, cy - self.y) > self.GATE:
                continue
            if best is None or c[0] > best[0][0]:
                best = (c, cx, cy)
        if best is None:
            if self.alive and t - self.t_seen > self.HOLD:
                self.alive = False
            return
        c, cx, cy = best
        self.x, self.y, self.axis, self.score = cx, cy, c[3], c[0]
        self.t_seen = t
        self.alive = True

    def latch(self, cands, t):
        """Chua bam cai nao thi lay cai diem cao nhat lam moc."""
        if self.alive:
            return
        best = None
        for c in cands:
            if c is not None and (best is None or c[0] > best[0]):
                best = c
        if best is not None:
            self.x = best[1] * math.cos(best[2])
            self.y = best[1] * math.sin(best[2])
            self.axis = best[3]
            self.score = best[0]
            self.t_seen = t
            self.alive = True

    def drop(self):
        self.alive = False

    # --- toa do trong HE HOC: lech ngang va lech goc ---
    def cross_track(self):
        """Xe lech sang ben bao nhieu so voi truc hoc (met)."""
        return self.x * math.sin(self.axis) - self.y * math.cos(self.axis)

    def along_track(self):
        """Xe cach mieng hoc bao nhieu, do DOC TRUC. Duong = dang o ngoai."""
        return -(self.x * math.cos(self.axis) + self.y * math.sin(self.axis))

    def heading_err(self):
        """Goc giua mui xe va truc RA cua hoc. 0 = duoi dang chi dung vao hoc."""
        return -self.axis

    def range(self):
        return math.hypot(self.x, self.y)

    def approach_point(self, dist):
        return (self.x + dist * math.cos(self.axis),
                self.y + dist * math.sin(self.axis))


class RuleBrain:
    """Mot bo luat cho MOT xe. Tu giu trang thai rieng."""

    STATES = ("long-nhong", "ve-tram", "tim-hoc", "ap-mieng", "chinh-truc",
              "lui-vao", "hoi-ma", "dang-sac", "rut-ra")

    COMMIT_SCORE = 0.55     # duoi nguong nay thi phan lon la goc tuong

    def __init__(self, robot_id=0, seed=0):
        self.id = robot_id
        self.rng = random.Random(1000 + seed + robot_id)
        self.state = "long-nhong"
        self.t_state = 0.0
        self._last_t = 0.0
        self.turn_dir = 1
        self.wander_turn = 0.0
        self.wander_until = 0.0
        self.tries = 0
        self.orbit_dir = 1
        self.track = _DockTrack()
        # Tu do duong bang chinh hai so v va w trong dau vao. Dung de nho
        # xem vua cam nham cai hoc nao, khoi thu lai dung no.
        self.px = self.py = self.pth = 0.0
        self.bad = []          # [(x, y, t)] cac hoc da cam nhung khong ra dien

    # ------------------------------------------------------------------
    def _go(self, state, t):
        if state != self.state:
            self.state = state
            self.t_state = t

    def _in_state(self, t):
        return t - self.t_state

    def __call__(self, obs, t):
        dt = max(1e-3, t - self._last_t)
        self._last_t = t

        v_now = float(obs[PC.I_MOTION]) * P.V_MAX
        w_now = float(obs[PC.I_MOTION + 1]) * P.W_MAX
        self.pth += w_now * dt
        self.px += v_now * math.cos(self.pth) * dt
        self.py += v_now * math.sin(self.pth) * dt
        self.bad = [b for b in self.bad if t - b[2] < 400.0]

        self.track.predict(v_now, w_now, dt)
        self.track.update([_cand(obs, 0), _cand(obs, 1)], t)

        blink_low = float(obs[PC.I_BATTERY + 1])
        soc = float(obs[PC.I_BATTERY + 0])
        charging = float(obs[PC.I_BATTERY + 2]) > 0.5
        in_slot = float(obs[PC.I_CONTACT + 0]) > 0.5
        id_ok = float(obs[PC.I_CONTACT + 1]) > 0.5
        cliff_l = float(obs[PC.I_CLIFF]) > 0.5
        cliff_r = float(obs[PC.I_CLIFF + 1]) > 0.5

        # Vuc thi lui, khong ban gi khac. Phan xa, khong phai ke hoach.
        if cliff_l or cliff_r:
            self._go("long-nhong", t)
            return _drive(-0.35, (-1.0 if cliff_l else 1.0) * 2.0)

        if charging:
            self._go("dang-sac", t)
            self.tries = 0
            self.bad = []
            if soc <= 0.97:
                return 0.0, 0.0

        # Den bao sac nhap nhay: dau vao chi nhap nhay, KHONG ai cuop lai.
        # Bo luat nay chon nghe theo no; bo nao hoc duoc thi tuy no.
        if blink_low > 0.0 and self.state == "long-nhong":
            self._go("ve-tram", t)

        fn = getattr(self, "_st_" + self.state.replace("-", "_"))
        return fn(obs, t, dt, in_slot, id_ok, soc)

    # ------------------------------------------------------------------ doc dau vao
    @staticmethod
    def _station(obs):
        """Tram trong he quy chieu xe: (biet khong, vi tri, goc truc ra).

        Day la thu xe TU ghi lai bang odometry cua chinh no luc sac lan
        truoc, nen cang chay lau cang lech. Nhung no co ca HUONG TRUC, va
        do la thu quyet dinh: biet truc thi biet phai vong ra dung truoc
        mieng hoc, chu bo tới sat suon hoc thi khong nhin thau long hoc
        duoc ma cung khong bat duoc den hong ngoai.
        """
        if float(obs[PC.I_STATION + 0]) <= 0.5:
            return False, 0.0, 0.0, 0.0
        rng_ = (1.0 - float(obs[PC.I_STATION + 1])) * PC.STATION_RANGE_NORM
        bear = math.atan2(float(obs[PC.I_STATION + 2]), float(obs[PC.I_STATION + 3]))
        axis = math.atan2(float(obs[PC.I_STATION + 4]), float(obs[PC.I_STATION + 5]))
        return True, rng_ * math.cos(bear), rng_ * math.sin(bear), axis

    @staticmethod
    def _ir_dock(obs):
        if float(obs[PC.I_IR + 3]) <= 0.5:
            return False, 0.0
        return True, math.atan2(float(obs[PC.I_IR + 4]), float(obs[PC.I_IR + 5]))

    # ------------------------------------------------------------------ lai xe
    def _avoid(self, obs, hard=0.42):
        """Tra ve lenh tranh vat neu can, khong thi None."""
        front = min(_fan(obs, 0), _fan(obs, 1) * 0.9, _fan(obs, 11) * 0.9)
        if front >= hard:
            return None
        left = _fan(obs, 2) + _fan(obs, 3)
        right = _fan(obs, 10) + _fan(obs, 9)
        d = 1 if left > right else -1
        return _drive(0.05 if front > 0.25 else -0.18, d * 2.2)

    def _seek(self, tx, ty, v_max=0.42):
        """Chay toi mot diem trong he quy chieu xe."""
        d = math.hypot(tx, ty)
        head = math.atan2(ty, tx)
        if abs(head) > 1.0:
            return _drive(0.0, 2.0 if head > 0 else -2.0)
        return _drive(min(v_max, 1.0 * d), max(-2.0, min(2.0, 2.2 * head)))

    # ------------------------------------------------------------------ chon hoc
    def _to_self_frame(self, lx, ly):
        c, s_ = math.cos(self.pth), math.sin(self.pth)
        return self.px + lx * c - ly * s_, self.py + lx * s_ + ly * c

    def _is_bad(self, lx, ly):
        gx, gy = self._to_self_frame(lx, ly)
        return any(math.hypot(gx - bx, gy - by) < 0.60 for bx, by, _ in self.bad)

    def _mark_bad(self):
        if self.track.alive:
            gx, gy = self._to_self_frame(self.track.x, self.track.y)
            self.bad.append((gx, gy, self._last_t))

    def _pick(self, obs, t, max_range=2.0):
        """Chon ung vien hoc dang tin: bo hoc da cam hut, uu tien gan cho nho."""
        known, sx, sy, _sa = self._station(obs)
        if self.tries >= 2:
            known = False
        floor = self.COMMIT_SCORE if self.tries < 3 else 0.38
        best = None
        for c in (_cand(obs, 0), _cand(obs, 1)):
            if c is None or c[0] < floor or c[1] > max_range:
                continue
            lx, ly = c[1] * math.cos(c[2]), c[1] * math.sin(c[2])
            if self._is_bad(lx, ly):
                continue
            cost = -c[0] + (math.hypot(lx - sx, ly - sy) if known else 0.0)
            if best is None or cost < best[0]:
                best = (cost, c, lx, ly)
        return best

    def _commit(self, pick, t):
        _cost, c, lx, ly = pick
        self.track.x, self.track.y = lx, ly
        self.track.axis, self.track.score = c[3], c[0]
        self.track.t_seen = t
        self.track.alive = True
        self._go("ap-mieng", t)

    # ------------------------------------------------------------------ cac trang thai
    def _st_long_nhong(self, obs, t, dt, in_slot, id_ok, soc):
        a = self._avoid(obs, 0.45)
        if a is not None:
            return a
        seen = float(obs[PC.I_IR + 0]) > 0.5
        if seen:
            bear = math.atan2(float(obs[PC.I_IR + 1]), float(obs[PC.I_IR + 2]))
            return _drive(0.42, max(-1.8, min(1.8, 2.0 * bear)))
        if t > self.wander_until:
            self.wander_turn = self.rng.uniform(-1.0, 1.0)
            self.wander_until = t + self.rng.uniform(0.8, 2.5)
        return _drive(0.45, self.wander_turn)

    def _st_ve_tram(self, obs, t, dt, in_slot, id_ok, soc):
        """Vong ra DUNG TRUOC MIENG hoc da nho, chu khong bo thang toi hoc.

        Day la cho de sai nhat cua ca quy trinh. Bo thang toi toa do cai
        hoc thi xe dung sat suon no: bo do khong nhin thau long hoc (hai
        vach ben che), den hong ngoai cung bi hai vach do bop lai con
        +-33 do nen cung khong thay. Phai di toi diem nam TREN TRUC, cach
        mieng nua met, roi moi quay mat vao.
        """
        a = self._avoid(obs)
        if a is not None:
            return a

        known, sx, sy, sa = self._station(obs)
        # Da hoi hut vai lan roi thi cho nho chac chan la sai - bo di, quay
        # sang lan theo den hong ngoai va do tung cai hoc mot. Odometry chay
        # 130 m lech toi 4 m; bam mai vao mot cho sai thi nam duong o do.
        if not known or self.tries >= 2:
            ir_on, ir_bear = self._ir_dock(obs)
            if ir_on:
                if abs(ir_bear) > 0.15:
                    return _drive(0.12, max(-1.8, min(1.8, 2.0 * ir_bear)))
                if _fan(obs, 0) > 0.85:
                    return _drive(0.40, 0.0)
                self._go("tim-hoc", t)
                return 0.0, 0.0
            if self._in_state(t) > 6.0:
                self._go("tim-hoc", t)
                return 0.0, 0.0
            return self._st_long_nhong(obs, t, dt, in_slot, id_ok, soc)

        # Diem cho: tren truc, truoc mieng 0,65 m.
        tx = sx + 0.65 * math.cos(sa)
        ty = sy + 0.65 * math.sin(sa)
        if math.hypot(tx, ty) < 0.28 or self._in_state(t) > 60.0:
            self._go("tim-hoc", t)
            return 0.0, 0.0
        return self._seek(tx, ty)

    def _st_tim_hoc(self, obs, t, dt, in_slot, id_ok, soc):
        """Da dung truoc mieng: quay mat vao, do hoc, hoi den hong ngoai.

        Hinh dang khong thoi thi khong du. Goc tu, khe giua cai ban voi
        buc tuong cung lom cung rong bang cai hoc; bo do van cham 0,4-0,5.
        Va hoc MOI NHU thi giong het hoc that ma khong phat gi ca. Den hoc
        sac loai duoc ca hai. Nhung no KHONG cho biet do co phai hoc cua
        MINH khong - chuyen do phai lui duoi vao cham chan tiep dien moi ro.
        """
        known, sx, sy, _sa = self._station(obs)
        if self.tries >= 2:
            known = False
        ir_on, ir_bear = self._ir_dock(obs)

        pick = self._pick(obs, t, max_range=2.2)
        if pick is not None and (ir_on or self.tries >= 3):
            self._commit(pick, t)
            return _drive(0.2, 0.0)

        span = self._in_state(t)
        # Quay mat ve phia cho nho (hoac phia den) cho bo do nhin thau.
        aim = math.atan2(sy, sx) if known else (ir_bear if ir_on else None)
        if aim is not None and abs(aim) > 0.15 and span < 4.0:
            return _drive(0.0, max(-1.5, min(1.5, 2.0 * aim)))
        if span < 7.0:
            return _drive(0.0, 1.1 * self.orbit_dir)     # quet mot vong
        if span < 12.0:
            # Chua thay: di vong cung quanh cho nho de doi goc nhin.
            a = self._avoid(obs, 0.35)
            if a is not None:
                return a
            return _drive(0.28, 0.9 * self.orbit_dir)
        self.tries += 1
        self.orbit_dir = -self.orbit_dir
        self._go("ve-tram", t)
        return _drive(0.2, 0.0)

    def _st_ap_mieng(self, obs, t, dt, in_slot, id_ok, soc):
        """Dua xe len DUNG TRUC hoc, cach mieng nua met.

        Chi mot viec: triet tieu LECH NGANG. Lech goc de buoc sau; lech doc
        thi khong quan trong. Truoc day o day do bang khoang cach toi diem
        dich - mot so tron ca lech ngang lan lech doc - con buoc sau lai do
        rieng lech ngang, hai thuoc do khac nhau cho cung mot viec nen hai
        trang thai da qua da lai quanh nguong 8 cm cho toi khi het pin.
        """
        if not self.track.alive or self._in_state(t) > 22.0:
            self.track.drop()
            self.tries += 1
            self._go("tim-hoc", t)
            return _drive(0.2, 0.0)

        cross = self.track.cross_track()
        along = self.track.along_track()
        if abs(cross) < 0.06 and abs(along - 0.55) < 0.18:
            self._go("chinh-truc", t)
            return 0.0, 0.0
        if along < 0.25:
            return _drive(-0.22, 0.0)      # da chui vao mieng ma van lech: lui ra
        a = self._avoid(obs, 0.30)
        if a is not None and self.track.range() > 0.9:
            return a
        px, py = self.track.approach_point(0.55)
        return self._seek(px, py, v_max=0.32)

    def _st_chinh_truc(self, obs, t, dt, in_slot, id_ok, soc):
        """Quay cho MUI huong ra theo truc hoc, tuc la DUOI quay vao hoc.

        Chi quay, khong di. Lech ngang con lai de buoc lui tu chinh - vua
        lui vua nan con de hon la chay ra chay vao chinh truoc.
        """
        if not self.track.alive or self._in_state(t) > 10.0:
            self.track.drop()
            self.tries += 1
            self._go("tim-hoc", t)
            return _drive(0.15, 0.0)
        psi = self.track.heading_err()
        if abs(psi) < math.radians(5.0):
            self._go("lui-vao", t)
            return 0.0, 0.0
        # DAU: quay xe mot goc +w lam moi huong co dinh ngoai the gioi lui
        # di -w trong he quy chieu xe, nen psi = -axis TANG theo w. Muon
        # psi ve 0 thi phai lai w = -k*psi. Lai w = +k*psi thi 180 do bien
        # tu diem day thanh diem hut: xe dung im truoc mieng hoc, mui chui
        # vao trong, lac qua lac lai cho toi khi het pin.
        return _drive(0.0, max(-1.6, min(1.6, -3.0 * psi)))

    def _st_lui_vao(self, obs, t, dt, in_slot, id_ok, soc):
        """Lui duoi vao hoc, vua lui vua giu truc.

        Chi chinh goc thoi thi khong du: lech ngang 8 cm la ke hoc chan
        ngay, vi long hoc 31 cm ma than xe 30 cm. Dung luat lui chuong:
        goc mong muon ti le voi lech ngang, roi banh lai bam theo goc do.
        """
        if in_slot:
            self._go("hoi-ma", t)
            return 0.0, 0.0
        if self._in_state(t) > 8.0 or not self.track.alive:
            self.tries += 1
            self.track.drop()
            self._go("rut-ra", t)
            return _drive(0.3, 0.0)
        e = self.track.cross_track()          # lech ngang, met
        psi = self.track.heading_err()        # lech goc, rad
        psi_want = max(-0.5, min(0.5, 2.6 * e))
        w = max(-1.2, min(1.2, 2.2 * (psi_want - psi)))
        return _drive(-0.13, w)

    def _st_hoi_ma(self, obs, t, dt, in_slot, id_ok, soc):
        """Da cham chan tiep dien. Hoc co phat tin hieu khong?"""
        if id_ok:
            self._go("dang-sac", t)
            self.tries = 0
            return 0.0, 0.0
        if self._in_state(t) > 0.6:
            # Khong phat gi ca -> khong phai hoc cua minh. Co cam manh cung
            # khong ra dien. Ghi so cai hoc nay lai roi rut ra tim cai khac.
            self._mark_bad()
            self.tries += 1
            self.track.drop()
            self._go("rut-ra", t)
        return 0.0, 0.0

    def _st_dang_sac(self, obs, t, dt, in_slot, id_ok, soc):
        if soc > 0.97 or not in_slot:
            self._go("rut-ra", t)
            return _drive(0.3, 0.0)
        return 0.0, 0.0

    def _st_rut_ra(self, obs, t, dt, in_slot, id_ok, soc):
        self.track.drop()
        if self._in_state(t) < 1.8:
            return _drive(0.38, 0.0)
        if self._in_state(t) < 2.9:
            return _drive(0.1, 1.8 * (1 if self.tries % 2 else -1))
        low = (float(obs[PC.I_BATTERY + 1]) > 0.0
               or float(obs[PC.I_BATTERY]) < P.BATT_LOW)
        self._go("tim-hoc" if low else "long-nhong", t)
        return _drive(0.3, 0.0)


def factory(seed=0):
    return lambda rid: RuleBrain(rid, seed)
