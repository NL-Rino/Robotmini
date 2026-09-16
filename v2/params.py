# -*- coding: utf-8 -*-
"""Moi con so cua ban v2.

Phan xe va hoc sac sao chep nguyen tu ban v1 (xem ../docs/SIM.md) de hai ban
noi ve cung mot con robot. Phan camera va phan 3D la moi.

v2 la mot du an RIENG. No khong doc file nao cua v1 va khong sua gi cua v1;
chep ca thu muc v2/ sang may thue la chay duoc.
"""

import math

# ---------------------------------------------------------------- than xe
BODY_RADIUS = 0.15
BODY_HEIGHT = 0.12          # than xe cao 12 cm
WHEEL_BASE = 0.235
V_MAX = 0.50
MOTOR_TAU = 0.12
W_MAX = 2.0 * V_MAX / WHEEL_BASE

CLIFF_RADIUS = 0.145
CLIFF_ANGLE = math.radians(35.0)

IR_FOV = math.radians(46.0)
IR_RANGE = 6.0

CONTACT_LONG_TOL = 0.030
CONTACT_LAT_TOL = 0.025
CONTACT_HYST = 0.012

# ---------------------------------------------------------------- CAMERA (moi)
# Mot mat may tinh nho gan tren cot, CHI co mot servo 3 chan de ngua len cui
# xuong. Khong co servo quay trai phai: muon nhin sang ben thi xoay ca xe.
CAM_HEIGHT = 0.22           # tam ong kinh cao hon san 22 cm
CAM_FORWARD = 0.06          # nho ve phia truoc tam xe mot chut
CAM_W = 64                  # anh dua vao bo nao
CAM_H = 48
CAM_FOV_H = math.radians(70.0)
CAM_FOV_V = CAM_FOV_H * CAM_H / CAM_W

# Gioi han co hoc cua servo. Ngua len de doc bang ma tren hoc, cui xuong de
# nhin mep vuc - ca hai deu KHONG the thay bang LiDAR.
TILT_MIN = math.radians(-38.0)
TILT_MAX = math.radians(45.0)
TILT_RATE = math.radians(110.0)    # do/giay khi day het ga servo
TILT_TAU = 0.08

# ---------------------------------------------------------------- LiDAR
LIDAR_HZ = 6.0
LIDAR_POINT_RATE = 3000.0
LIDAR_HEIGHT = 0.10         # quet o do cao nay -> khong thay vat thap hon
LIDAR_MIN = 0.12
LIDAR_MAX = 8.0
LIDAR_NOISE = 0.012
LIDAR_DROP = 0.02
N_LIDAR_FANS = 12

# ---------------------------------------------------------------- odometry
ODOM_SCALE_ERR = 0.025
ODOM_DIFF_ERR = 0.003
ODOM_DRIFT = 0.001

# ---------------------------------------------------------------- hoc sac
DOCK_OUTER_W = 0.40
DOCK_OUTER_D = 0.40
DOCK_CAVITY_W = 0.31
DOCK_CAVITY_D = 0.31
DOCK_WALL = 0.045
DOCK_HEIGHT = 0.30          # vach hoc cao 30 cm
# Bang ma gan TREN thanh sau cua hoc, cao hon dinh vach. Muon doc phai NGUA
# CAMERA LEN - day la ly do cai servo ton tai.
MARK_Z0 = 0.34
MARK_Z1 = 0.50
MARK_CELLS = 3              # 3 o mau -> 4^3 = 64 ma khac nhau
MARK_COLORS = 4

# ---------------------------------------------------------------- pin
BATT_IDLE_DRAIN = 1.0 / 2400.0
BATT_MOVE_DRAIN = 1.0 / 870.0
BATT_CHARGE_RATE = 1.0 / 60.0
BATT_LOW = 0.15
BATT_BLINK_HZ = 2.0

# ---------------------------------------------------------------- vong lap
DT = 0.05

# ---------------------------------------------------------------- dau vao
N_SCALARS = 48              # xem v2/env.py, ham `scalar_names`
N_LIDAR_RAYS = 96           # moi buoc, trai deu 360 do
