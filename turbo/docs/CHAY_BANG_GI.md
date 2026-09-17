# Chay bang gi: tung xe mot, hay ca lo mot luc

## Cau hoi that su la gi

Ban hoi "sao de CPU lam het, khai thac con GPU di". Cau tra loi that:
**ban v1 khong the dung GPU duoc, va do la loi kien truc chu khong phai
loi cua GPU.**

Do duoc o ban v1: mot buoc mo phong mat khoang 0,45 ms, trong do mang than
kinh (GRU 48 -> 16 -> 2) chiem **2,4%**. Con 97,6% la mot con xe duy nhat
ban 60 tia LiDAR va do cai hoc bang numpy - toan mang 60 phan tu. Dua phan
2,4% do sang GPU thi du no nhanh vo han, ca buoc cung chi nhanh len 2,4%.
Do la dinh luat Amdahl, va no khong thuong luong duoc.

GPU nhanh khong phai vi moi phep tinh nhanh hon, ma vi no lam duoc **hang
nghin phep giong nhau cung luc**. Muon dung no thi phai co hang nghin phep
giong nhau de dua cho no. Mot con xe khong co. **Mot nghin nam tram con xe
thi co.**

## `turbo/` la gi

Chinh la ban v1 - cung can phong, cung vat ly, cung 48 dau vao, cung ham
phan thuong, cung giao trinh nguoc - nhung 1.536 con xe buoc **cung mot
phep tinh** thay vi lan luot tung con.

Mot the he cua ES cham diem P bo trong so. Truoc day P bo do chia cho 4
tien trinh CPU, moi tien trinh chay mot con xe mot luc. Bay gio ca P bo
nam trong mot phep tinh:

    theta      (P, 5306)          ca quan the
    quan sat   (P, G, 48)         P bo x G con xe moi bo
    GRU        11 phep bmm        thay vi P*G*11 phep nhan ma tran ti hon
    ban tia    (P*G, 60, 52)      3,6 trieu phep giao tia mot luc

## Do duoc tren may nay (4 loi CPU, khong co card)

Mot the he voi quan the 32 = 115.200 buoc-xe; quan the 128 = 460.800.

| chay bang | quan the 32 | quan the 128 |
|---|---:|---:|
| tung xe mot (4 tien trinh) | 23,5 s | 87,8 s |
| ca lo mot luc (CPU) | 24,4 s | **67,1 s** |
| | hoa (0,96 lan) | **nhanh gap 1,31 lan** |

Day la cho quan trong nhat trong ca trang nay: **khong co con so "nhanh
gap may lan" nao ca, co mot duong cong**.

| quan the | xe cung luc | buoc-xe/giay |
|---:|---:|---:|
| 16 | 96 | 3.652 |
| 32 | 192 | 4.762 |
| 64 | 384 | 5.802 |
| 128 | 768 | 6.874 |
| 256 | 1.536 | **7.584** |

Ban "tung xe mot" nam ngang o khoang 5.000 va se nam ngang mai: no chia
viec cho 4 loi, het 4 loi la het. Ban theo lo **di len theo kich thuoc
quan the** - quan the 256 chay nhanh gap doi quan the 16.

Tren CPU nay duong cong do bat dau chung lai o khoang 7.500 vi het **duong
truyen bo nho**, khong phai het phep tinh. Cho "het duong truyen bo nho"
chinh la cho card do hoa roi co ly: RTX A4000 co bo nho RIENG, rong gap
khoang 45 lan CPU nay. (Card LIEN thi khong: no dung chung ong voi CPU -
xem `INTEL_620.md`.)

## Toi CHUA do duoc tren GPU

Cho toi dang lam viec khong co card nao, nen bang tren la do tren CPU.
Toi khong dua cho ban mot con so GPU phong doan roi goi no la ket qua do.

Ban thue may xong thi chay dung mot lenh nay de tu do:

    python -m turbo.tools.check   --device cuda    # may chay duoc den dau
    python -m turbo.tools.measure scale --device cuda
    python -m turbo.tools.measure speed --device cuda

Tren may nha (card lien Intel) thi doi `cuda` thanh `dml`.

Cai dang xem la cot "buoc-xe/giay" co con di len khi quan the tang khong,
va no dung o dau.

## Card LIEN (Intel HD 620, UHD, Iris, AMD tich hop)

Khong co CUDA, va `torch.xpu` cung khong nhan (no can Gen12 tro len). Duong
duy nhat la **DirectML**, va no chay duoc. Cach cai va cach do: xem
`turbo/docs/INTEL_620.md`.

Mot dieu nen biet truoc: card lien **dung chung thanh RAM voi CPU**, ma cho
nghen cua chuong trinh nay la duong truyen bo nho chu khong phai so phep
tinh. Nen dung ky vong nhu card roi. Do di roi hay chon.

## Vay chon cai nao

- **Quan the duoi ~48, may khong co card**: chon "tung xe mot". Don gian
  hon va nhanh hon.
- **Quan the tu ~64 tro len**: chon "ca lo mot luc", ke ca tren CPU.
- **Co card do hoa**: chon "ca lo mot luc", va de quan the that to (128,
  256, 512). Quan the to khong chi nhanh hon moi dong ho - no con lam
  gradient cua ES bot nhieu, tuc la moi the he di dung huong hon.
- Muon doi giua hai cai: chon o o "Chay bang" trong man huan luyen. Hai ban
  ghi ra **cung mot khuon file**, nen dang chay dang do doi sang ban kia
  cung duoc - bam Dung, doi o chon, bam chay tiep.

## Mot bo nao, hai bo may chay

Day la dieu quan trong nhat va la thu duoc kiem thu ky nhat:

- `state.npz` / `best.npz` cua hai ban **doc duoc lan nhau**, bang chinh
  `GRUPolicy.load` ma con robot that dung.
- Cung trong so va cung dau vao thi hai ban cho ra hanh dong lech nhau
  duoi 2e-5 (`turbo/tests/test_train.py`).
- Vat ly: chay tu do hai ban lech 2 um sau 150 buoc.

Mot cho **khong** giong: bo do hoc. Ban v1 cat vong quet thanh doan dai
ngan tuy du lieu - moi xe mot kieu, khong gop duoc. Ban turbo hoi khac
han (xem `turbo/dock.py`). Do tren 360 vong quet, hai bo doc cung du lieu:

| | thay dung | bao gia | lech vi tri | lech truc |
|---|---:|---:|---:|---:|
| v1 | 130 | 13 (9%) | 2,3 cm | 2,8 do |
| turbo | 119 | 2 (2%) | 1,8 cm | 3,9 do |

Ban turbo thay it hon 8%, bao gia it hon 5 lan, dinh vi chinh hon mot chut,
do goc truc kem hon mot chut. Khong te hon, nhung **khac**: bo nao nuoi
bang ban nay thay mot the gioi hoi khac bo nao nuoi bang ban kia, nen dung
nuoi mot bo nao nua chung o ban nay roi chay tiep o ban kia va mong doi no
cu xu y het.

## Cho van con o CPU

Ba viec nho van chay tren CPU va co ly do:

1. **Sinh mat bang** (`make_fleet_map`): vai chuc mili giay mot the he,
   va giu chung ma voi ban v1 la dang gia hon la tiet kiem cho nay.
2. **Cho dat xe dau moi lan** (giao trinh nguoc): vai chuc con xe, mot vong
   lap Python. Viet lai theo lo thi de lech giao trinh ma khong duoc gi.
3. **Nguoi di lai**: buoc mot lan cho MOI MAT BANG (2-6 cai), roi trai ra
   cho tung xe bang mot phep chi so. Truoc day cho nay lap theo so XE -
   voi 1.536 xe la 3.000 vong Python moi buoc.
