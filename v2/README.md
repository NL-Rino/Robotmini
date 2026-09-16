# Robotmini v2 — con xe có mắt

Dự án **riêng**, không sửa gì của bản v1. Chép cả thư mục `v2/` sang máy thuê
là chạy được.

Vẫn con robot đó — tròn Ø30 cm, hai bánh vi sai, LiDAR, cảm biến vực, tiếp
điểm ở đuôi, hộc sạc có mã — nhưng thêm:

- **Một camera 64×48 RGB** trên cột cao 22 cm
- **Một servo 3 chân, chỉ ngửa lên cúi xuống** (−38°…+45°). Không có servo
  quay trái phải: muốn nhìn sang bên thì xoay cả xe.
- **Mô phỏng 3D thật**: bộ đối tia dựng hình hộp xoay, trụ đứng, sàn có lỗ
  thủng, trần có đèn. Phối cảnh, che khuất, đổ bóng đều thật.
- **Bảng mã màu gắn CAO trên thành sau mỗi hộc**, cao hơn vách hộc. Muốn đọc
  phải **ngửa camera lên**. Đọc được rồi thì biết hộc nào là của mình **trước
  khi cắm** — thứ bản v1 không thể làm.

## Chạy

```
pip install torch
python -m v2.tools.bench            # đo máy này chạy được bao nhiêu
python -m v2.view                   # mở cửa sổ, tự lái thử
python -m v2.train                  # huấn luyện
```

Cả ba lệnh đều **hỏi chọn GPU** nếu máy có nhiều card, và tự chạy bằng CPU
nếu không có card nào — chậm nhưng vẫn xem được nó làm gì.

```
python -m v2.train --device cuda:0 --yes --envs 512 --amp
python -m v2.view --brain v2/runs/thu1/state.pt
python -m v2.view --save anh.png    # máy không có màn hình
```

Dừng an toàn: tạo file `STOP` trong thư mục `--out`, hoặc bấm Ctrl-C một
lần. Đang chạy vòng nào thì chạy nốt, lưu `state.pt`, rồi thoát.

Lái tay trong cửa sổ xem: `W/S` tiến lùi, `A/D` quay, `I/K` ngửa/cúi camera,
`M` đổi giữa lái tay và để bộ não lái, `R` thả lại, `Q` thoát.

## Ba thứ chỉ bản 3D mới có

| | |
|---|---|
| **Cái bàn** | Mặt bàn ở 0,42 m. LiDAR quét ở 0,10 m nên chỉ thấy bốn cái chân; camera thấy cả mặt bàn. Xe chui được xuống gầm. |
| **Vật thấp** | Tấm thảm dày 3 cm: LiDAR không thấy gì cả, chỉ camera cúi xuống mới thấy. |
| **Bảng mã** | Ở 0,34–0,50 m, cao hơn vách hộc 0,30 m. Phải ngửa camera lên mới đọc được. |

## Cấu trúc

```
device.py     chọn GPU (hỏi khi máy có nhiều card)
params.py     mọi con số
raytrace.py   lõi đối tia 3D: hộp xoay, trụ đứng, sàn có lỗ, trần
world3d.py    sinh thế giới 3D theo lô
camera.py     dựng ảnh từ tư thế xe + góc servo
env.py        vòng lặp mô phỏng theo lô trên GPU
policy.py     mạng tích chập + GRU, 1,38 triệu tham số
ppo.py        PPO có hồi quy
train.py      vòng huấn luyện, file STOP, status.jsonl
view.py       xem và lái tay
tools/bench.py  đo thông lượng rồi quy ra tiền
tools/png.py    ghi ảnh PNG không cần thư viện ngoài
tests/          22 bài
docs/DESIGN.md      thiết kế
docs/THUE_GPU.md    hướng dẫn thuê máy, kèm cách tính tiền
```

Chi tiết: [docs/DESIGN.md](docs/DESIGN.md) · [docs/THUE_GPU.md](docs/THUE_GPU.md)
