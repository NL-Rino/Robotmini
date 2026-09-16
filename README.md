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
python tests/test_sim.py                    # 31/31  mô phỏng
python tests/test_link.py                   #  7/7   xe <-> wifi <-> não
python -m tools.run_fleet                   # chạy MÃI, Ctrl-C để ngắt não
python -m tools.run_fleet --view 10 --realtime
python -m tools.run_fleet --seconds 600 --rescue 60
```

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
tools/run_fleet.py  trình chạy + bản đồ ASCII
tests/            38 bài (31 mô phỏng + 7 đường truyền)
docs/SIM.md         thiết kế, 48 đầu vào, và các con số kèm lý do
```

Chi tiết thiết kế: [docs/SIM.md](docs/SIM.md).
