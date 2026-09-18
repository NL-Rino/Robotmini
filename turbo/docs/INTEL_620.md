# Chay thu tren Intel HD 620 (may cua ban)

## Truoc het: card cua ban khong co CUDA

`i5-7200U` di kem `Intel HD Graphics 620` - card LIEN, khong phai card roi.
Nhung thu sau **khong** chay duoc no:

- CUDA: cua rieng NVIDIA.
- `torch.xpu` (duong Intel chinh thuc): can Gen12 tro len (Arc, Core Ultra).
  HD 620 la Gen9.5, doi 2017 - khong nam trong danh sach.

Con **mot** duong: **DirectML**. No bien DirectX 12 thanh mot may tinh cua
PyTorch, va HD 620 co DirectX 12.

## Cach nhanh nhat: chay san mot file

O thu muc goc co `cai_dml_windows.bat`. Bam doi vao no. No se:

  1. tim mot ban Python tu 3.12 tro xuong tren may ban;
  2. neu khong co, in ra dung lenh can go de cai them (va KHONG dong toi
     ban Python dang co);
  3. tao moi truong ao, cai `torch-directml`, roi chay `check` luon.

Cai xong roi thi lan sau bam doi `chay_dml.bat` (thu may) hoac go
`chay_dml.bat do` (do toc do card roi do lai tren CPU de so).

### Moi truong ao dat o dau, va tai sao khong dat canh du an

Dat o `%LOCALAPPDATA%\robotmini-dml`, tuc la
`C:\Users\<ten ban>\AppData\Local\robotmini-dml`.

Ly do: **Windows chi cho duong dan dai 260 ky tu**, ma goi `torch` co cay
thu muc sau toi khoang 150 ky tu (`...\torch\include\torch\csrc\api\
include\torch\nn\modules\...`). Du an nam trong OneDrive thi rieng phan
dau da hon 90 ky tu:

    C:\Users\Cuong\OneDrive\Tai lieu\#robot\GPU\3\Robotmini-claude-...\

Cong vao la tran, va pip do giua chung voi:

    ERROR: Could not install packages due to an OSError:
    [Errno 22] Invalid argument

Dat moi truong ao ra cho ngan thi het. **Ma nguon du an cu de nguyen trong
OneDrive cung duoc** - chi rieng moi truong ao phai ra ngoai.

Van do thi chep ca thu muc du an ra mot cho ngan, vi du `C:\robot\`.

## Cai dat (Windows)

### Truoc tien: Python phai la 3.12 tro XUONG

Day la cho vap dau tien va no khong hien ra ro rang. Ban moi nhat cua
`torch-directml` (0.2.5.dev240914) chi co goi cho **Python 3.8 den 3.12**,
va no ghim `torch==2.4.1` - ma torch 2.4.1 cung chi co toi 3.12.

Chay Python 3.13 tro len thi pip bao dung cau nay:

    ERROR: Could not find a version that satisfies the requirement
    torch-directml (from versions: none)
    ERROR: No matching distribution found for torch-directml

**"from versions: none" nghia la khong co goi nao hop voi Python nay** -
khong phai loi mang, khong phai loi pip cu.

Kiem tra ban dang chay ban nao:

    python -V
    py -0                 (liet ke moi ban Python dang co tren may)

### Co san Python 3.12 hoac cu hon

    cd duong\dan\den\Robotmini
    py -3.12 -m venv .venv-dml
    .venv-dml\Scripts\activate
    python -V                      (phai thay 3.12.x)
    pip install torch-directml numpy

Doi `-3.12` thanh `-3.11` hay `-3.10` neu may ban co ban do.

### Chua co ban nao tu 3.12 tro xuong

`py -0` chi hien `3.13` thi phai cai them. **Cai them KHONG lam hong ban
3.13 dang co** - trinh `py` cho nhieu ban Python song song tren cung mot
may, va `app.py` van chay bang ban cu nhu thuong.

**Cach 1 - winget** (co san tren Windows 10 doi moi):

    winget install Python.Python.3.12

Bao khong nhan ten do thi xem ten dung bang:

    winget search Python.Python

**Cach 2 - tai ve** tu `python.org/downloads`: keo xuong muc "Looking for a
specific release?", chon mot ban **3.12.x**, tai file *Windows installer
(64-bit)*, cai binh thuong. Khong bat buoc tich "Add to PATH" - ta goi qua
`py -3.12`.

Cai xong thi **mo mot cua so cmd MOI** (cua so cu chua thay ban vua cai),
roi:

    py -0                          (phai thay ca 3.12 va 3.13)
    py -3.12 -m venv .venv-dml

### Kiem tra

    python -m turbo.tools.check --list-devices

Dong dau in ra ban Python va ban torch. Thay mot dong co chu **DirectML**
la duoc. Khong thay thi chuong trinh se in ro LY DO ngay duoi - doc dong
do truoc khi lam gi tiep.

### Luu y ve moi truong ao nay

`torch-directml` keo theo `torch==2.4.1` cua rieng no. Moi truong `.venv-dml`
CHI dung de chay thu bang card lien. Ban torch dang co o may (de chay
`app.py` binh thuong) khong bi dung toi.

## Thu xem may chay duoc den dau

Lan sau muon chay lai thi bam doi `chay_dml.bat` (no tu bat moi truong ao
len ho). Muon go tay thi phai **bat moi truong ao truoc**, khong thi bao
`No module named 'torch'`:

    .venv-dml\Scripts\activate
    python -m turbo.tools.check --device dml

No in ra ba muc:

1. **Cac phep de thieu** - card lien thieu phep la chuyen thuong. Chuong
   trinh da viet san duong vong cho hau het; dong nao ghi `KHONG` ma kem
   chu `CAN CO` moi la van de.
2. **Chay thu** - dung the gioi, dat xe, mo phong 40 buoc, do hoc sac,
   dung 48 dau vao, chay bo nao.
3. **Phep nao dang phai nho CPU** - cho nay quan trong. DirectML khong bao
   loi khi thieu mot phep: no **lang le chep tensor sang CPU, tinh o do,
   roi chep nguoc lai**. Chay van chay, nhung moi buoc mo phong mat mot
   vong di ve, va do la cach chac chan nhat de card cham hon CPU. Muc nay
   liet ke dung ten cac phep do - **gui danh sach do lai**, viet duong
   vong cho chung thuong chi mat mot doan ngan.
4. **Nhanh cham the nao** - so `buoc-xe/giay`.

**Co dong nao ghi `KHONG` thi chup man hinh gui lai.** Do dung la thu can
biet, va viet duong vong cho no thuong chi mat mot doan ngan.

## Roi so voi CPU

    chay_dml.bat do

Lenh do chay ca hai roi in ra canh nhau. Go tay thi:

    .venv-dml\Scripts\activate
    python -m turbo.tools.measure speed --device dml --pop 64 --no-v1
    python -m turbo.tools.measure speed --device cpu --pop 64

So hai con so `buoc-xe/giay` voi nhau. Cai nao lon hon thi chon cai do o o
**"Chay bang"** trong `app.py`.

## Do duoc that tren HD 620 (18/09/2026)

Chay duoc het muc 1 den muc 6. Toc do, kem trang thai card luc do:

| quan the | xe cung luc | buoc-xe/giay | tang | GPU 3D | bo nho chung |
|---:|---:|---:|---|---:|---:|
| 16 | 48 | 350 | | 2-8% | 0,2 GB |
| 64 | 192 | 862 | 2,5 lan khi xe gap 4 | | |
| 256 | 768 | 1.315 | 1,5 lan khi xe gap 4 | 84% | 1,2/1,9 GB |

Doc bang nay tu trai sang phai:

**O lo NHO, card nam khong (3D chi 2-8%).** Do khong phai card yeu ma la
tien goi phep: moi lan PyTorch goi mot phep len DirectML, no phai dung mot
doi tuong DirectML roi day xuong hang doi - mat khoang nua mili giay BAT
KE mang to hay nho. Mot buoc mo phong goi vai tram phep, nen o lo nho thi
gan het thoi gian la tien goi.

**O lo TO, card da lam viec that (3D 84%)** va duong cong bat dau chung
lai: gap 4 lan xe chi con duoc 1,5 lan toc do. Bo nho chung cung da len
1,2 trong 1,9 GB, tuc la quan the 512 gan nhu chac chan khong vua.

Nen **tran cua HD 620 trong bai nay la khoang 1.300 buoc-xe/giay**, dat o
quan the 256. Lo to hon nua khong con cho.

Con no co hon CPU cua chinh may do khong thi phai do canh nhau:
`chay_dml.bat do`.

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

## Cap nhat ma nguon khi chua co git

Bam doi `capnhat.bat`. Co `git` thi no dung git; khong co thi no tai file
nen tu GitHub roi chep de len - chi de len ma nguon, khong dung toi `runs\`
hay `brains\` cua ban.

Cach chac chan hon la cai git mot lan cho xong:

    winget install Git.Git

roi mo CMD MOI va tu do chi can `git pull`.

**Tai lai ma nguon KHONG phai cai lai torch.** Moi truong ao nam o
`%LOCALAPPDATA%\robotmini-dml`, ngoai thu muc du an, nen thu muc du an moi
van dung lai duoc no - chi can chay `chay_dml.bat`.

## Loi hay gap

**`[Errno 22] Invalid argument` luc cai torch** -> duong dan qua dai. Xem
muc "Moi truong ao dat o dau" o tren. `cai_dml_windows.bat` ban moi da tu
dat ra cho ngan roi; van loi thi chep du an ra `C:\robot\`.

**`No module named 'torch'`** -> chua bat moi truong ao. Dung `chay_dml.bat`
thay vi go tay, no tu bat ho.

**OneDrive khoa file** -> bam chuot phai bieu tuong OneDrive o khay he
thong, chon *Pause syncing*, roi cai lai.

## Khong thay card

- `pip` bao `from versions: none` -> Python qua moi, xem muc cai dat o tren.
- Cap nhat trinh dieu khien do hoa Intel (trang Intel, khong phai Windows
  Update).
- Kiem tra card co DirectX 12: bam Win+R, go `dxdiag`, xem muc "Feature
  Levels" co `12_1` hoac `12_0` khong.
- May co ca card roi va card lien thi Windows co the dang giau bot: vao
  Settings > Display > Graphics.
- Van khong duoc thi cu chay bang CPU. Khong mat gi ca: cung mot chuong
  trinh, cung mot bo nao, cung mot khuon file.

## Cai bay lon nhat: so thuc 64 bit

Card lien khong lam duoc nhieu phep tren so thuc 64 bit, va no bao loi bang
dung mot cau:

    RuntimeError: unknown error

Khong ten phep, khong dong, khong gi ca. Cho sinh ra 64 bit cung khong ngo:

    torch.full((R, N), 8.0, device=may)      -> 64 BIT tren card lien
    torch.full((R, N), 8.0, dtype=torch.float32, device=may)   -> dung

Tren CPU thi `torch.full` lay kieu mac dinh (32 bit) nen khong ai thay gi;
tren card lien no lay kieu cua chinh con so Python, ma so Python la 64 bit.

Loi nay da lam mat nhieu vong do mot ly do nua: no CHI hien ra o xe vua dat
lai. Quet xong mot vong LiDAR la mang do bi thay bang ban 32 bit va loi bien
mat - nen chay thu 40 buoc thi qua, ma bat dau mot lan danh gia moi thi chet.

Gio `turbo/tools/check.py` co mot muc soi kieu so cua MOI mang trong mo
phong va bao ngay neu co cai nao khong phai 32 bit. Va co hai bai kiem thu
chan: mot bai soi trang thai, mot bai chay ca vong va bao loi neu co BAT KY
phep nao sinh ra 64 bit.

## Vi sao no chay duoc tren card lien

Card lien thieu phep hon card roi. Bon cho tung phai sua:

| phep | card lien | thay bang |
|---|---|---|
| `torch.hypot` | hay thieu | `can(a^2+b^2)` |
| `torch.cummin` | hay thieu | `cumprod` tren 0/1 |
| `max_pool1d` | hay thieu | bang thua (log2 buoc) |
| bo sinh so ngau nhien rieng | khong co | sinh tren CPU roi chuyen sang |
| so thuc 64 bit | khong co | cong don 32 bit, doi ve 64 o CPU |
| `clamp` tren 64 bit | khong co | ghi ro `dtype=torch.float32` luc tao mang |
| `%` tren so thuc | hay thieu | cong/tru mot vong |
| `clamp` tren so nguyen | hay thieu | clamp truoc khi ep kieu |

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
