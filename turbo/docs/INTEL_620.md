# Chay thu tren Intel HD 620 (may cua ban)

## Truoc het: card cua ban khong co CUDA

`i5-7200U` di kem `Intel HD Graphics 620` - card LIEN, khong phai card roi.
Nhung thu sau **khong** chay duoc no:

- CUDA: cua rieng NVIDIA.
- `torch.xpu` (duong Intel chinh thuc): can Gen12 tro len (Arc, Core Ultra).
  HD 620 la Gen9.5, doi 2017 - khong nam trong danh sach.

Con **mot** duong: **DirectML**. No bien DirectX 12 thanh mot may tinh cua
PyTorch, va HD 620 co DirectX 12.

## Cai dat (Windows)

`torch-directml` keo theo mot ban `torch` rieng cua no, nen **dung mot moi
truong ao rieng** de khong lam hong ban torch dang co:

    cd duong\dan\den\Robotmini
    python -m venv .venv-dml
    .venv-dml\Scripts\activate
    pip install torch-directml
    pip install numpy

Xong thi thu ngay:

    python -m turbo.tools.check --list-devices

Thay mot dong co chu **DirectML** la duoc. Khong thay thi xem muc "Khong
thay card" o duoi.

## Thu xem may chay duoc den dau

    python -m turbo.tools.check --device dml

No in ra ba muc:

1. **Cac phep de thieu** - card lien thieu phep la chuyen thuong. Chuong
   trinh da viet san duong vong cho hau het; dong nao ghi `KHONG` ma kem
   chu `CAN CO` moi la van de.
2. **Chay thu** - dung the gioi, dat xe, mo phong 40 buoc, do hoc sac,
   dung 48 dau vao, chay bo nao.
3. **Nhanh cham the nao** - so `buoc-xe/giay`.

**Co dong nao ghi `KHONG` thi chup man hinh gui lai.** Do dung la thu can
biet, va viet duong vong cho no thuong chi mat mot doan ngan.

## Roi so voi CPU

    python -m turbo.tools.check --device cpu
    python -m turbo.tools.measure speed --device cpu

So hai con so `buoc-xe/giay` voi nhau. Cai nao lon hon thi chon cai do o o
**"Chay bang"** trong `app.py`.

## Toi doan no nhanh hon bao nhieu

**Toi khong biet, va toi khong doan.** Cho dang lam viec khong co card nao
de do. Nhung co mot dieu ve phan cung ban nen biet truoc khi ky vong:

Cho nghen cua chuong trinh nay khong phai la **so phep tinh**, ma la
**duong truyen bo nho** - moi buoc phai doc ghi vai tram MB. Card roi
(RTX A4000) co bo nho RIENG, rong gap ~45 lan CPU, nen no thang dam. Card
LIEN thi **dung chung dung mot thanh RAM voi CPU** - cung mot cai ong.

Nen: HD 620 co suc tinh toan tho gap khoang 7 lan CPU 2 nhan cua ban
(~380 GFLOPS so voi ~50), nhung khong co them ti nao duong truyen. Ket qua
that nam o dau giua hai con so do, va con tru di phan DirectML phai dich
qua lai. Co the nhanh hon, co the bang, co the cham hon.

Cai chac chan la: **chay duoc**, va ban tu do duoc bang mot lenh.

## May 2 nhan thi chon gi

Neu DirectML khong hon, van nen thu `ca lo mot luc - CPU`. Ly do: ban
"tung xe mot" chia viec cho cac tien trinh, va may ban chi co **2 nhan**
that (4 luong) - no chi chia duoc lam 2. Ban theo lo khong quan tam may
co bao nhieu nhan. Tren may 4 nhan o cho toi do, hai ban hoa nhau o quan
the 32-64; tren may 2 nhan, diem hoa do se den som hon.

Nho dat quan the cho to: **64 tro len**, ban theo lo moi phat huy.

## Khong thay card

- Cap nhat trinh dieu khien do hoa Intel (trang Intel, khong phai Windows
  Update).
- Kiem tra card co DirectX 12: bam Win+R, go `dxdiag`, xem muc "Feature
  Levels" co `12_1` hoac `12_0` khong.
- May co ca card roi va card lien thi Windows co the dang giau bot: vao
  Settings > Display > Graphics.
- Van khong duoc thi cu chay bang CPU. Khong mat gi ca: cung mot chuong
  trinh, cung mot bo nao, cung mot khuon file.

## Vi sao no chay duoc tren card lien

Card lien thieu phep hon card roi. Bon cho tung phai sua:

| phep | card lien | thay bang |
|---|---|---|
| `torch.hypot` | hay thieu | `can(a^2+b^2)` |
| `torch.cummin` | hay thieu | `cumprod` tren 0/1 |
| `max_pool1d` | hay thieu | bang thua (log2 buoc) |
| bo sinh so ngau nhien rieng | khong co | sinh tren CPU roi chuyen sang |
| so thuc 64 bit | khong co | cong don 32 bit, doi ve 64 o CPU |

Ba cach viet dau lai **nhanh hon** cach cu ngay tren CPU (`cumprod` nhanh
gap 9 lan `cummin`, bang thua nhanh gap 3,5 lan `max_pool1d`), nen chung
duoc dung cho moi may chu khong rieng card lien.

Cach sinh so ngau nhien moi con them mot cai loi: cung hat giong thi
**may nao cung ra cung mot chuoi**, nen mot phien chay tren card so thang
duoc voi mot phien chay tren CPU.

Con MOT phep phai do chung: `scatter_reduce` (gom tia vao quat). Duong
vong cho no cham hon 19 lan, nen chuong trinh THU xem may co phep do
khong roi moi chon duong. Muc 1 cua `check` noi cho ban biet may minh roi
vao duong nao.
