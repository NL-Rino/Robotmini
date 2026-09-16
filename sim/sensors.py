"""Cam bien vuc, hong ngoai hai kenh, va TIEP DIEM SAC o duoi duoi xe.

Phan moi quan trong nhat nam o `dock_contact`:

  - Chan tiep dien nam o DUOI XE. Xe phai LUI DUOI vao hoc moi cham duoc.
    Than xe tron nen ke hoc khong can goc quay - cai chan xe la vi tri chan
    tiep dien, khong phai hinh dang than xe.
  - Cham duoc chan chua chac co dien. Hoc chi bat tay va cap dien khi MA
    cua hoc trung ma cua xe. Cam nham hoc thi `in_slot` len 1 nhung
    `id_signal` va `charging` van bang 0. Co cam manh cung khong sac duoc.
"""

import math

from . import params as P
from .geometry import (
    raycast_circles,
    raycast_segments,
    to_local,
    wrap_pi_scalar,
)


class ContactState:
    """Ket qua mot lan kiem tra tiep diem."""

    __slots__ = ("in_slot", "dock", "id_signal", "charging", "wrong_dock")

    def __init__(self, in_slot=False, dock=None, id_signal=False,
                 charging=False, wrong_dock=False):
        self.in_slot = in_slot        # chan tiep dien dang cham mot hoc nao do
        self.dock = dock
        self.id_signal = id_signal    # hoc co phat tin hieu bat tay khong
        self.charging = charging      # thuc su co dien chay vao
        self.wrong_dock = wrong_dock  # cham roi nhung khong phai hoc cua minh

    def __repr__(self):
        name = self.dock.name if self.dock else "-"
        return (f"<Contact slot={int(self.in_slot)} dock={name} "
                f"id={int(self.id_signal)} charge={int(self.charging)}>")


def rear_contact_point(x, y, th, radius=P.BODY_RADIUS):
    """Vi tri chan tiep dien: o DUOI xe, nguoc huong mui."""
    return x - radius * math.cos(th), y - radius * math.sin(th)


def dock_contact(world, x, y, th, robot_code, engaged=False):
    """Xe dang cam vao hoc nao, va hoc do co chiu cap dien khong.

    `engaged` = buoc truoc da cam roi. Khi da cam thi dung nguong rong hon
    mot chut: chan tiep dien that la mieng dong co lo xo, khong phai diem.
    """
    long_tol = P.CONTACT_LONG_TOL + (P.CONTACT_HYST if engaged else 0.0)
    lat_tol = P.CONTACT_LAT_TOL + (P.CONTACT_HYST if engaged else 0.0)
    rx, ry = rear_contact_point(x, y, th)
    for d in world.docks:
        # Doi ve he quy chieu hoc: +x la truc ra, goc o giua mieng hoc.
        lx, ly = to_local(rx, ry, d.x, d.y, d.theta)
        if lx > 0.02 or lx < -P.DOCK_CAVITY_D - 0.12:
            continue
        deep_enough = lx <= -P.DOCK_CAVITY_D + long_tol
        centred = abs(ly) <= lat_tol
        if not (deep_enough and centred):
            continue
        in_slot = True
        # Hoc chi phat tin hieu bat tay khi ma trung. Hoc moi nhu khong co ma.
        match = (d.code is not None and robot_code is not None
                 and d.code == robot_code)
        id_signal = bool(match and d.powered)
        return ContactState(in_slot, d, id_signal, id_signal, not id_signal)
    return ContactState()


def cliff_sensors(world, x, y, th):
    """Hai mat chieu xuong, cach tam 14,5 cm, lech +-35 do so voi mui.

    Tra ve (trai, phai): True = duoi do la khoang khong.
    """
    out = []
    for sign in (+1, -1):
        a = th + sign * P.CLIFF_ANGLE
        sx = x + P.CLIFF_RADIUS * math.cos(a)
        sy = y + P.CLIFF_RADIUS * math.sin(a)
        out.append(not world.on_floor(sx, sy))
    return out[0], out[1]


def _nearest_visible(x, y, th, sources, segments, circles, fov=P.IR_FOV,
                     rng_max=P.IR_RANGE):
    """Nguon hong ngoai gan nhat ma mat thu o dau xe nhin thay.

    Loc bang cu ly va goc mo truoc (re), roi moi ban tia kiem tra che khuat
    cho nhung nguon con lai - va ban MOT LAN cho ca nhom.
    """
    cand = []
    for ex, ey in sources:
        dx, dy = ex - x, ey - y
        dist = math.hypot(dx, dy)
        if dist > rng_max:
            continue
        bearing = wrap_pi_scalar(math.atan2(dy, dx) - th)
        if abs(bearing) > fov:
            continue
        cand.append((dist, bearing, ex, ey))
    if not cand:
        return False, 0.0
    cand.sort()
    angles = [math.atan2(ey - y, ex - x) for _d, _b, ex, ey in cand]
    far = max(c[0] for c in cand) + 1.0
    hit = raycast_segments(x, y, angles, segments, far)
    for i, (dist, bearing, _ex, _ey) in enumerate(cand):
        if hit[i] < dist - 1e-4:
            continue
        if circles:
            hc = raycast_circles(x, y, [angles[i]], circles, dist + 1.0)
            if float(hc[0]) < dist - 1e-4:
                continue
        return True, bearing
    return False, 0.0


def ir_beacon(world, x, y, th, circles=()):
    """Kenh 1: den goi xe toi."""
    src = [(b.x, b.y) for b in world.beacons if b.on]
    return _nearest_visible(x, y, th, src, world.static_segments, circles)


def ir_dock(world, x, y, th, circles=()):
    """Kenh 2: den hong ngoai trong hoc sac.

    Den gan giua thanh trong, chieu thang ra cua. Hai vach ben bop chum tia
    lai - o day khong cai dat goc chum nao ca, chi hoi "co nhin thay nhau
    khong", va hinh hoc tu lo phan con lai.

    Khong phan biet duoc hoc nao voi hoc nao: moi hoc deu phat giong nhau.
    Chi khi CAM VAO moi biet co phai hoc cua minh khong.
    """
    src = [(d.ir_x, d.ir_y) for d in world.docks if d.ir_on]
    return _nearest_visible(x, y, th, src, world.static_segments, circles)
