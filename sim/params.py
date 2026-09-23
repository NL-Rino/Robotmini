"""Moi con so cua mo phong nam o day, de doc lai ma khong phai doi file.

Doi vi: met, giay, radian. Pin tinh theo ti le 0..1.
"""

import math

# ---------------------------------------------------------------- than xe
BODY_RADIUS = 0.15          # tron, duong kinh 30 cm
WHEEL_BASE = 0.235          # khoang cach hai banh
V_MAX = 0.50                # m/s
MOTOR_TAU = 0.12            # tre dong co, bac nhat
W_MAX = 2.0 * V_MAX / WHEEL_BASE   # suy ra: hai banh nguoc chieu het ga

# Cam bien vuc: 2 cai chieu xuong, cach tam 14,5 cm, lech +-35 do so voi mui
CLIFF_RADIUS = 0.145
CLIFF_ANGLE = math.radians(35.0)

# Mat thu hong ngoai o DAU xe, +-46 do
IR_FOV = math.radians(46.0)
IR_RANGE = 6.0

# Tiep diem sac nam o DUOI DUOI XE (mat sau), tai diem tam - R*huong.
# Muon sac thi phai LUI DUOI VAO HOC. Than xe tron nen goc quay khong bi
# ke hoc chan; cai chan la vi tri cua chan tiep dien o duoi xe.
CONTACT_LONG_TOL = 0.030    # con cach thanh trong bao nhieu thi cham
CONTACT_LAT_TOL = 0.025     # lech ngang bao nhieu thi truot chan tiep dien
# Chan tiep dien that la mieng dong co lo xo, nen lu bu vai mili met van con
# dinh. Khong co do nay thi mot cu huc nhe cua xe khac cung lam roi dien.
CONTACT_HYST = 0.012

# ---------------------------------------------------------------- LiDAR
LIDAR_HZ = 6.0              # vong/giay
LIDAR_POINT_RATE = 3000.0   # diem/giay -> ~500 diem/vong
LIDAR_MIN = 0.12
LIDAR_MAX = 8.0
LIDAR_NOISE = 0.012         # sai so ti le
LIDAR_DROP = 0.02           # 2% diem mat

# ---------------------------------------------------------------- odometry
# Phai tach lam hai. Neu cho MOI BANH mot sai so 2,5% doc lap thi hai banh
# lech nhau 5%, chia cho vet banh 0,235 m ra 0,21 rad moi met - tuc 12 do/m,
# di 20 m la quay du mot vong. Thuc te sai so 2,5% la sai so DUONG KINH CHUNG
# (do ca hai banh cung mon, cung bom), no lam sai QUANG DUONG chu khong lam
# sai HUONG. Cai lam sai huong la phan LECH GIUA HAI BANH, va no nho hon han.
ODOM_SCALE_ERR = 0.025      # sai so chung ca hai banh -> sai quang duong
ODOM_DIFF_ERR = 0.003       # lech giua hai banh -> sai huong, ~0,7 do/m
ODOM_DRIFT = 0.001          # rad/s, phan troi khong theo quang duong

# ---------------------------------------------------------------- hoc sac
DOCK_OUTER_W = 0.40
DOCK_OUTER_D = 0.40
DOCK_CAVITY_W = 0.31
DOCK_CAVITY_D = 0.31
DOCK_WALL = 0.045
DOCK_CHAMFER = 0.08         # vat goc trong o mieng; khong vat thi khong cam noi

# ---------------------------------------------------------------- pin
# Thang pin duoc chon de nguong 15% THUC SU la mot canh bao dung duoc.
# Ban cu co lop cuong ep ve sac nen 15% chi con 7,9 giay cung khong sao -
# lop do cuop lai va lai ve. Bay gio khong con lop nao ca: duoi 15% chi co
# mot den nhap nhay, xe nghe hay khong la viec cua xe.
#
# Nhung 15% phai du cho bao nhieu? Do do PHAI dem ca cai gia cua viec cam
# nham hoc. Xe chi co odometry cua chinh no: chay 130 m thi cho nho tram
# lech toi 4 m, tuc la no khong con biet trong nam cai hoc giong het nhau
# cai nao la cua minh. Cach duy nhat de biet la LUI DUOI VAO VA HOI - va
# moi lan hoi hut mat 25-30 giay. Vay 15% phai du cho bon nam lan hoi:
#   15% x 870 giay = 130 giay = ~65 m duong.
# Neu cat 15% xuong con vai chuc giay thi bai toan thanh khong giai duoc:
# xe biet phai ve nhung khong con du pin de tim ra hoc nao la hoc cua no.
# THANG PIN DA NEN LAI. Ban truoc: 870 giay mot lan xa, 60 giay mot lan sac
# -> mot vong xa-sac la 14.880 buoc, ma mot lan danh gia chi 600-1.200 buoc.
# Ba vong nhu de bai doi la 44.640 buoc: khong the huan luyen duoc.
#
# Gio 300 giay mot lan xa, 20 giay mot lan sac. Chon 300 vi phai giu MOT
# tinh chat: tu nguong bao dong, xe con du pin de TIM ra hoc. 20% x 300 s
# = 60 giay = ~30 m duong, ma can nha nhieu phong, di vong qua cua thi
# mot lan ve co khi 15-20 m. Thu 180 s thi bo luat viet tay nam duong ca
# loat. Cat ngan hon nua thi bai toan thanh khong giai duoc.
#
# Con "ba lan sac" thi khong bat mot lan danh gia phai chua ca ba: GIAO
# TRINH NGUOC cap san tien do (da an may cham, da sac may lan), nen mot lan
# danh gia ngan van cham toi duoc khuc cuoi. Xem `train/rollout.py`.
BATT_IDLE_DRAIN = 1.0 / 900.0    # moi giay khi dung yen
BATT_MOVE_DRAIN = 1.0 / 300.0    # moi giay khi chay het ga -> 5 phut lien
BATT_CHARGE_RATE = 1.0 / 20.0    # moi giay khi dang co dien -> day sau 20 s
BATT_LOW = 0.20                  # nguong den bao sac, VA nguong tinh mot lan
                                 # sac hop le (de bai: phai duoi 20% moi tinh)
BATT_BLINK_HZ = 2.0              # nhap nhay 2 lan/giay, lien tuc, khong tat
BATT_FULL = 0.999                # coi la day; phai day moi tinh mot lan sac

# ---------------------------------------------------------------- can nha
# Khong con mot phong 6,4 x 4,8 nua. Mot can nha nhieu phong, co ban ghe;
# chan ghe duoi mat LiDAR chi la mot cham tron 2,5 cm.
HOUSE_W = 12.0
HOUSE_H = 9.0
LEG_RADIUS = 0.025          # chan ban / chan ghe
DOOR_W = 0.95               # be rong o cua giua cac phong

# Nguoi di lai: duoi mat LiDAR (cao ~20 cm) KHONG phai mot khoi tron ma la
# HAI CAI CHAN, va khi buoc thi chan sau nhac len - cham do bien mat mot
# nhip roi hien lai. Do la thu de nhan ra nguoi that.
LEG_R_HUMAN = 0.055         # ban kinh mot chan nguoi
LEG_SPACING = 0.24          # hai chan cach nhau
STRIDE_HZ = 0.9             # so buoc chan moi giay
BODY_RADIUS_HUMAN = 0.20    # de tinh va cham (than nguoi), khong phai de quet

# ---------------------------------------------------------------- de bai
# Mot lan chay duoc coi la HOAN THANH khi lam xong ca hai:
TASK_BEACONS = 5            # an du 5 cham goi
TASK_CHARGES = 3            # va sac du 3 lan HOP LE
# Mot lan sac hop le: luc cam vao pin phai duoi BATT_LOW, va phai nam yen
# cho toi khi day. Rut ra giua chung thi lan do khong tinh, phai lam lai.

# Thuong cho viec di toi cho MOI: chia san nha thanh o vuong, o nao lan dau
# tien nam trong tam quet LiDAR thi duoc cong diem mot lan.
COVER_CELL = 0.5            # canh o, met
BEACON_RADIUS = 0.40        # toi gan hon the nay thi coi la da an cham goi
BEACON_RESPAWN = (4.0, 10.0)  # an xong, sau chung nay giay thi sang lai cho khac

# ---------------------------------------------------------------- vong lap
DT = 0.05                   # 20 Hz

N_LIDAR_FANS = 24           # can nha to hon thi can nhin min hon
N_DOCK_CANDIDATES = 2
N_INPUTS = 64               # xem sim/perception.py va docs/SIM.md
