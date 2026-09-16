# Thuê GPU chạy thử — hướng dẫn từng bước

## 0. Một điều phải nói trước

**Tôi chưa chạy được bản này trên card thật.** Máy tôi làm việc không có
GPU, nên mọi thứ ở đây đã được kiểm thử trên CPU (22/22 bài xanh) và viết để
chạy trên CUDA, nhưng con số tốc độ dưới đây là **ước lượng**, không phải đo.

Vì thế việc đầu tiên khi thuê máy là chạy `bench`. Nó cho số thật trong hai
phút, và bạn quyết sau.

Trong bộ kiểm thử có một bài tên `test_GPU_va_CPU_cho_cung_ket_qua` — trên
máy không có card thì nó bị bỏ qua, trên máy có card thì nó chạy. **Chạy bộ
kiểm thử ngay khi vừa thuê máy**, bài đó là thứ bắt lỗi lệch CPU/GPU.

## 1. Chọn card

| | RTX A4000 | RTX 3090 |
|---|---|---|
| giá | 14.300 đ/giờ | 18.500 đ/giờ |
| VRAM | 16 GB | 24 GB |
| băng thông bộ nhớ | ~448 GB/s | ~936 GB/s |
| tính toán FP32 | ~19 TFLOPS | ~36 TFLOPS |

**Chọn 3090.** Việc nặng nhất ở đây là đối tia — một chuỗi phép tính đơn giản
trên mảng rất lớn, tức là **bị chặn bởi băng thông bộ nhớ**, không phải bởi
sức tính. Về mặt đó 3090 hơn khoảng **2,1 lần** mà chỉ đắt hơn 1,29 lần. Tính
theo đồng trên mỗi triệu bước thì 3090 rẻ hơn.

24 GB so với 16 GB không quan trọng lắm ở đây — bộ đệm ảnh uint8 với 512 thế
giới × 64 bước chỉ hết khoảng 0,3 GB. Nhưng nó cho bạn thoải mái tăng số thế
giới song song, và điều đó thì **rất** quan trọng (mục 4).

## 2. Cài đặt, từ lúc ssh vào

```bash
# 1. lấy code
git clone -b claude/inspiring-noether-9n5p0d https://github.com/NL-Rino/Robotmini
cd Robotmini

# 2. torch bản CUDA (thường máy thuê đã có sẵn, kiểm tra trước)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# nếu False:  pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. kiểm thử — phải xanh hết, kể cả bài so CPU với GPU
python v2/tests/test_v2.py

# 4. đo máy này chạy được bao nhiêu
python -m v2.tools.bench --yes
```

`bench` in ra bảng theo từng cỡ lô, rồi tự quy ra tiền:

```
  512 the gioi | ve+cam bien  ... ms | ... | NNNNN buoc/s | VRAM  N.NN GB
nhanh nhat: 512 the gioi song song, NNNNN buoc/giay
chay 50 trieu buoc het N.N gio
gia 18.500 d/gio -> NNN.NNN d cho ca lan chay
```

## 3. Chạy huấn luyện

```bash
tmux new -s train           # để rớt mạng không mất phiên
python -m v2.train --yes --envs 512 --steps 64 --amp \
       --out v2/runs/lan1 --total-steps 200000000
# Ctrl-B rồi D để thoát khỏi tmux mà vẫn chạy tiếp
```

Dừng an toàn, từ máy khác hay từ chính máy đó:

```bash
touch v2/runs/lan1/STOP     # chạy nốt vòng hiện tại, lưu, rồi thoát
```

Mang kết quả về:

```bash
scp may-thue:Robotmini/v2/runs/lan1/state.pt .
python -m v2.view --brain state.pt --device cpu --yes   # xem ở nhà
```

## 4. Cỡ lô — chỗ quyết định tiền

Đây là điều quan trọng nhất trong cả trang này.

Bộ đối tia duyệt các hình khối theo từng cụm, mỗi cụm vài chục lệnh gửi
xuống GPU. Số lệnh đó **không đổi** theo số thế giới: 512 thế giới hay 64 thế
giới cũng chừng ấy lệnh. Mà mỗi lệnh có chi phí khởi động cố định vài micro
giây.

Nghĩa là: **chạy 64 thế giới thì tiền chi phí khởi động chiếm gần hết, còn
chạy 512 thế giới thì nó chia cho 512.** Cùng một giờ thuê, cỡ lô lớn có thể
cho gấp năm bảy lần số bước.

Để `--envs 0` (mặc định) thì chương trình tự chọn theo VRAM còn trống. Nhưng
hãy chạy `bench` và nhìn cột `buoc/s` tăng tới đâu thì ngừng tăng — đó mới là
cỡ lô đúng cho máy đó.

## 5. Tính tiền

Công thức, tự thay số của bạn vào:

```
số giờ = (số triệu bước) × 1.000.000 ÷ (số bước/giây đo được) ÷ 3600
tiền   = số giờ × giá mỗi giờ
```

Ước lượng của tôi cho 3090 với 512 thế giới: **30.000–100.000 bước/giây**.
Khoảng rộng vì tôi chưa đo được thật. Nếu rơi vào giữa khoảng đó, khoảng
50.000 bước/giây:

| số bước | thời gian | tiền (3090) |
|---|---|---|
| 50 triệu | ~17 phút | ~5.000 đ |
| 200 triệu | ~1,1 giờ | ~21.000 đ |
| 500 triệu | ~2,8 giờ | ~51.000 đ |

Bài học từ ảnh: những bài học tăng cường từ ảnh thường cần **100–500 triệu
bước**. Nên hãy tính ngân sách khoảng **50.000–150.000 đ cho một lần chạy tới
nơi tới chốn**. Rẻ hơn nhiều so với cảm giác ban đầu.

Thuê một giờ chạy thử trước. Nếu `bench` cho ra dưới 10.000 bước/giây thì
đừng thuê dài — báo lại con số đó, có chỗ nào đó cần sửa.

## 6. Nhìn gì trong mười phút đầu

```bash
tail -f v2/runs/lan1/status.jsonl
```

| cột | ý nghĩa | dấu hiệu tốt |
|---|---|---|
| `sps` | số bước mỗi giây | ổn định, khớp với `bench` |
| `rew` | thưởng trung bình mỗi bước | nhích lên khỏi 0 |
| `charged` | phần pin nạp được | **rời khỏi 0** — đây là cái quan trọng nhất |
| `wrong` | số lần cắm nhầm hộc | lúc đầu tăng, sau phải giảm |
| `kl` | mức đổi của chính sách | quanh 0,01–0,03 |
| `clipped` | tỉ lệ bị cắt | 0,1–0,3 |
| `ent` | độ ngẫu nhiên còn lại | giảm dần, đừng về 0 quá sớm |

Hỏng thì trông thế nào:

- `charged` đứng im ở 0 sau 20 triệu bước → giáo trình chưa đủ dễ, hạ
  `--curriculum-steps`.
- `kl` vọt lên trên 0,1 → hạ `--lr`.
- `ent` về gần 0 trong vài triệu bước đầu → tăng `--ent`.
- `sps` tụt dần → nhiều môi trường kẹt ở trạng thái chết mà không được thả
  lại, hoặc VRAM đang bị tràn ra bộ nhớ chung.

**Con số cuối cùng cần nhìn không phải `rew` mà là `wrong`.** Nếu camera có
tác dụng thật thì số lần cắm nhầm hộc phải giảm xuống gần 0, vì xe đọc được
bảng mã trước khi cắm. Nếu `wrong` vẫn cao mà `charged` vẫn lên, nghĩa là nó
đang giải bài toán bằng cách thử từng hộc như bản v1 — camera chưa được dùng
tới, và đó mới là lúc cần xem lại.
