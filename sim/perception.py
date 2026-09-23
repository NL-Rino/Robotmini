"""CHO DUY NHAT dung vector dau vao cua bo nao.

Mo phong va robot that deu goi vao day. Neu hai ben dung vector khac nhau
mot chut thi bo nao nuoi trong mo phong se vo dung ngoai doi, va loi do rat
kho tim vi khong cho nao bao gi ca.

64 dau vao, bo cuc:

  0..23   (24) quat LiDAR      1 = vat sat mat, 0 = trong khong
  24..25   (2) vuc trai / phai 1 = duoi do la khoang khong
  26..37  (12) 2 ung vien hoc  moi cai [diem, cu ly, sin/cos phuong vi,
                               sin/cos TRUC RA]
  38..43   (6) hong ngoai      den goi [thay, sin, cos] + hoc sac [thay, sin, cos]
  44..49   (6) bo nho tram     [co, cu ly, sin/cos phuong vi, sin/cos truc]
  50..52   (3) pin             [muc pin, DEN BAO SAC NHAP NHAY, dang co dien]
  53..54   (2) tiep dien       [dang cam vao mot hoc, TIN HIEU DUNG HOC]
  55..59   (5) chuyen dong     [toc do thang, toc do quay, ga trai, ga phai, cham]
  60..63   (4) tien do de bai  [da an may cham, da sac may lan, lan sac nay
                               con hop le, vua thay cho moi]

Khoi 60..63 la thu ma robot THAT cung biet ve chinh no: no dem duoc no da
an may cai cham goi, da sac day may lan, lan cam hien tai co bat dau tu
duoi 20% hay khong. Khong cho no biet thi bai toan doi hoi mot bo nho dem
nam gon trong trang thai GRU - lam duoc, nhung la bat no hoc cong tru thay
vi hoc lai xe.

Hai dau vao dang chu y:

  [39] DEN BAO SAC. Duoi 15% pin thi dau vao nay nhap nhay 0/1 lien tuc va
       khong tat cho toi khi sac lai. No CHI nhap nhay. Khong co lop nao
       cuop quyen lai xe nua - xe phot lo thi xe nam duong.

  [42] TIN HIEU DUNG HOC. Chi co nghia khi [41] dang bang 1, tuc la chan
       tiep dien o duoi xe da cham vao mot hoc nao do. Hoc phat tin hieu
       thi do la hoc cua xe nay va dien dang vao; khong phat thi cam nham
       hoc cua xe khac, co cam manh cung khong sac duoc.
"""

import math

import numpy as np

from . import params as P
from .geometry import wrap_pi_scalar
from .lidar import fan_ranges

N_INPUTS = P.N_INPUTS

I_FANS = 0
I_CLIFF = I_FANS + P.N_LIDAR_FANS
I_DOCKS = I_CLIFF + 2
I_IR = I_DOCKS + 6 * P.N_DOCK_CANDIDATES
I_STATION = I_IR + 6
I_BATTERY = I_STATION + 6
I_CONTACT = I_BATTERY + 3
I_MOTION = I_CONTACT + 2
I_TASK = I_MOTION + 5

DOCK_RANGE_NORM = 4.0
STATION_RANGE_NORM = 8.0


def _range_feat(d, norm):
    return float(max(0.0, 1.0 - min(d, norm) / norm))


def build(robot, scan, dock_candidates, cliff, ir_beacon, ir_dock, t):
    """Dung vector dau vao tu trang thai cam bien tho.

    Moi doi so deu la thu robot that co the tu do duoc. Khong co tham so nao
    o day den tu "goc that" hay "vi tri that" cua mo phong.
    """
    v = np.zeros(N_INPUTS, dtype=np.float32)

    # ---- quat LiDAR
    fans = fan_ranges(scan, P.N_LIDAR_FANS, P.LIDAR_MAX)
    v[I_FANS:I_FANS + P.N_LIDAR_FANS] = 1.0 - np.clip(fans, 0.0, P.LIDAR_MAX) / P.LIDAR_MAX

    # ---- cam bien vuc
    v[I_CLIFF] = 1.0 if cliff[0] else 0.0
    v[I_CLIFF + 1] = 1.0 if cliff[1] else 0.0

    # ---- ung vien hoc
    for k in range(P.N_DOCK_CANDIDATES):
        b = I_DOCKS + 6 * k
        if k < len(dock_candidates):
            c = dock_candidates[k]
            v[b + 0] = float(min(1.0, max(0.0, c.score)))
            v[b + 1] = _range_feat(c.range, DOCK_RANGE_NORM)
            v[b + 2] = math.sin(c.bearing)
            v[b + 3] = math.cos(c.bearing)
            v[b + 4] = math.sin(c.axis)
            v[b + 5] = math.cos(c.axis)

    # ---- hong ngoai, hai kenh
    seen_b, bear_b = ir_beacon
    if seen_b:
        v[I_IR + 0] = 1.0
        v[I_IR + 1] = math.sin(bear_b)
        v[I_IR + 2] = math.cos(bear_b)
    seen_d, bear_d = ir_dock
    if seen_d:
        v[I_IR + 3] = 1.0
        v[I_IR + 4] = math.sin(bear_d)
        v[I_IR + 5] = math.cos(bear_d)

    # ---- bo nho tram (hieu hai so CUNG HE ODOM nen phan troi triet tieu)
    if robot.station is not None:
        sx, sy, sax = robot.station
        dx, dy = sx - robot.ox, sy - robot.oy
        dist = math.hypot(dx, dy)
        bear = wrap_pi_scalar(math.atan2(dy, dx) - robot.oth)
        axis_rel = wrap_pi_scalar(sax - robot.oth)
        v[I_STATION + 0] = 1.0
        v[I_STATION + 1] = _range_feat(dist, STATION_RANGE_NORM)
        v[I_STATION + 2] = math.sin(bear)
        v[I_STATION + 3] = math.cos(bear)
        v[I_STATION + 4] = math.sin(axis_rel)
        v[I_STATION + 5] = math.cos(axis_rel)

    # ---- pin
    v[I_BATTERY + 0] = float(robot.battery)
    v[I_BATTERY + 1] = robot.low_battery_blink(t)
    v[I_BATTERY + 2] = 1.0 if robot.charging else 0.0

    # ---- tiep dien / danh tinh hoc
    v[I_CONTACT + 0] = 1.0 if robot.in_slot else 0.0
    v[I_CONTACT + 1] = 1.0 if robot.id_signal else 0.0

    # ---- chuyen dong. cmd_l/cmd_r la ga THUC SU da chay, khong phai ga
    # duoc yeu cau: khi phan xa an toan de lenh ma khong bao lai thi trang
    # thai GRU tren laptop se troi khoi thuc te.
    v[I_MOTION + 0] = float(np.clip(robot.v / P.V_MAX, -1.0, 1.0))
    v[I_MOTION + 1] = float(np.clip(robot.w / P.W_MAX, -1.0, 1.0))
    v[I_MOTION + 2] = float(robot.cmd_l)
    v[I_MOTION + 3] = float(robot.cmd_r)
    v[I_MOTION + 4] = float(robot.bump)

    # ---- tien do de bai
    v[I_TASK + 0] = min(1.0, robot.task_beacons / float(P.TASK_BEACONS))
    v[I_TASK + 1] = min(1.0, robot.task_charges / float(P.TASK_CHARGES))
    v[I_TASK + 2] = 1.0 if robot.charge_valid else 0.0
    v[I_TASK + 3] = float(min(1.0, robot.new_area))
    return v


INPUT_NAMES = (
    [f"fan{i}" for i in range(P.N_LIDAR_FANS)]
    + ["cliff_L", "cliff_R"]
    + [n for k in range(P.N_DOCK_CANDIDATES)
       for n in (f"dock{k}_score", f"dock{k}_range", f"dock{k}_bsin",
                 f"dock{k}_bcos", f"dock{k}_asin", f"dock{k}_acos")]
    + ["ir_call", "ir_call_sin", "ir_call_cos",
       "ir_dock", "ir_dock_sin", "ir_dock_cos"]
    + ["st_known", "st_range", "st_bsin", "st_bcos", "st_asin", "st_acos"]
    + ["batt_soc", "batt_low_blink", "batt_charging"]
    + ["contact_in_slot", "contact_id_ok"]
    + ["mv_v", "mv_w", "mv_cmd_l", "mv_cmd_r", "mv_bump"]
    + ["task_beacons", "task_charges", "task_charge_ok", "task_new_area"]
)
assert len(INPUT_NAMES) == N_INPUTS, (len(INPUT_NAMES), N_INPUTS)
