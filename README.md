# Robotmini — mô phỏng đội xe tự hành hai bánh

Phần mô phỏng cho con xe 2 bánh vi sai tròn Ø30 cm, LiDAR Camsense, bộ não
chạy trên laptop. Đây là bản **làm lại phần mô phỏng** theo bốn thay đổi:

1. **Chạy mãi.** Không có tập, không có giới hạn số bước, không reset. Vòng
   lặp chỉ dừng khi **đứt kết nối bộ não**. Xe hết pin hay rơi xuống vực thì
   nằm im tại chỗ và thành vật cản của xe khác; thế giới vẫn chạy tiếp.
2. **Bỏ hẳn lớp cưỡng ép về sạc.** Dưới 15% pin chỉ có **một đầu vào nhấp
   nháy 0/1 liên tục** cho tới khi sạc lại. Nó chỉ nhấp nháy. Không lớp nào
   cướp quyền lái nữa — xe phớt lờ thì xe nằm đường.
3. **Muốn sạc thì phải lùi đuôi vào hộc.** Chân tiếp điện nằm ở đuôi xe.
   Cắm đầu vào thì không chạm được. Đây là tính chất vật lý của mô phỏng,
   không dạy, không có luật nào nhắc.
4. **Mỗi map thả 5 xe, mỗi xe một mã hộc sạc riêng.** Năm cái hộc nhìn từ
   ngoài giống hệt nhau. Chỉ khi đã lùi đuôi vào và chân tiếp điện chạm thì
   hộc mới phát (hoặc không phát) tín hiệu bắt tay. Cắm nhầm hộc của xe
   khác thì chạm được nhưng **không ra điện**, có cắm mạnh cũng vậy.

## Chạy

```
pip install numpy
python app.py                               # PHẦN MỀM CÓ GIAO DIỆN
```

Ba màn hình:

- **Chính** — chọn bộ não, *Xem xe chạy*, *Tạo bộ não mới*, *Huấn luyện*
- **Huấn luyện** — chạy tiến hoá ở một tiến trình riêng, có biểu đồ điểm
  chạy trực tiếp. Nút **DỪNG AN TOÀN** ghi một file `STOP`, chờ thế hệ đang
  chạy xong và lưu `state.npz`, rồi mới cho về màn hình chính. Lần sau chọn
  *Chạy tiếp* là đi tiếp từ đúng chỗ đó.
- **Xem chạy** — thả 1–5 xe ra mặt bằng và nhìn. Chấm ở đuôi xe là chân
  tiếp điện: xanh = đang có điện, đỏ = cắm rồi mà không ra điện.

Dòng lệnh, nếu cần:

```
python tests/test_sim.py                    # 32/32  mô phỏng
python tests/test_link.py                   #  7/7   xe <-> wifi <-> não
python tests/test_train.py                  # 21/21  huấn luyện
python tests/test_reward.py                 # 11/11  chống ăn gian trong sạc
python -m tools.run_fleet                   # chạy MÃI, Ctrl-C để ngắt não
python -m tools.run_fleet --view 10 --realtime
python -m train.train --out runs/thu1 --jobs 4
python -m tools.measure all                 # đo lại các con số trong docs
```

Đo được: đèn báo sáng thì **68/100** xe tự về được đúng hộc của mình, với
**0,4** lần cắm nhầm hộc mỗi lượt — chính là cơ chế mã dock đang chạy.

Não chạy ở tiến trình khác (như khi chạy thật: não trên laptop, xe qua wifi):

```
python -m link.brain_server            # cửa sổ 1
python -m tools.run_fleet --brain udp  # cửa sổ 2
```

Tắt cửa sổ 1 đi là cửa sổ 2 dừng — đúng nghĩa "chạy tới khi ngắt kết nối
bộ não".

## Bản đồ ASCII

```
chữ số = hộc sạc (x = hộc mồi nhử)   chữ cái = xe
@ = đang sạc   ? = cắm nhầm hộc   X = hết pin   ! = rơi
o = người đi lại   * = đèn gọi   ~ = vực
```

## Cấu trúc

```
app.py           PHẦN MỀM GIAO DIỆN (Tkinter, không phải cài gì thêm)
sim/
  params.py        mọi con số, kèm lý do chọn
  geometry.py      ray-cast vector hoá trên đoạn thẳng và đường tròn
  world.py         mặt bằng, hộc chữ U CÓ MÃ, vật cản, người đi lại, đèn gọi
  lidar.py         Camsense: gom đủ vòng mới xử lý, khử nhoè bằng ODOMETRY
  dock_detector.py dò hộc bằng hình học (đo độ lõm + khớp thành trong)
  sensors.py       cảm biến vực, hồng ngoại 2 kênh, TIẾP ĐIỂM ĐUÔI + MÃ HỘC
  robot.py         động lực học, odometry, va chạm, pin, đèn báo sạc
  perception.py    CHỖ DUY NHẤT dựng 48 đầu vào
  fleet.py         vòng lặp nhiều xe, chạy tới khi mất não
brain/rule_brain.py bộ luật viết tay — BẢN MẪU để xem, không phải giáo án
link/               giao thức UDP, bộ não trên laptop, phía xe
train/
  policy.py        GRU 48 -> 16 -> 2 numpy thuần, kèm bộ chuẩn hoá đầu vào
  reward.py        hàm phần thưởng
  rollout.py       một lần đánh giá + GIÁO TRÌNH NGƯỢC
  es.py            tiến hoá: đối gương, xếp hạng, AdamW
  train.py         vòng lặp, file trạng thái, file STOP
tools/run_fleet.py  trình chạy + bản đồ ASCII
tools/measure.py    đo lại: tỉ lệ về trạm, độ chính xác bộ dò, tốc độ, nhiễu
tests/            71 bài (32 mô phỏng + 7 đường truyền + 21 huấn
                  luyện + 11 chống ăn gian trong trạm sạc)
docs/SIM.md         thiết kế, 48 đầu vào, và các con số kèm lý do
```

Chi tiết: [docs/SIM.md](docs/SIM.md) — thiết kế mô phỏng.
[docs/TRAIN.md](docs/TRAIN.md) — cách dạy bộ não và vì sao bản cũ không tới đâu.

---

**Bản v2 — con xe có mắt** nằm ở [`v2/`](v2/README.md): vẫn con robot này
nhưng thêm một camera và một servo ngửa lên cúi xuống, mô phỏng 3D thật, chạy
theo lô trên GPU. Đó là một dự án riêng, không đụng gì tới bản này.
