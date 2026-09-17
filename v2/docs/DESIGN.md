# Thiết kế bản v2

## 1. Vì sao phải viết lại chứ không nâng cấp v1

Bản v1 mô phỏng bằng numpy trên CPU, mỗi lần một thế giới. Máy thuê chỉ có
**8 vCPU** — giữ nguyên kiến trúc đó thì 8 nhân là nút thắt và card đồ hoạ
ngồi chơi. Muốn GPU thực sự làm việc thì **cả mô phỏng** phải chạy theo lô
trên GPU, không riêng gì mạng nơ-ron.

Đo trên CPU 4 luồng, mỗi bước môi trường:

```
vẽ ảnh + cảm biến   97%
vật lý               1%
mạng nơ-ron          2%
```

Nếu chỉ đưa mạng lên GPU thì cùng lắm nhanh hơn 2%. Phải đưa cái 97% kia lên.

## 2. Đối tia thay vì thư viện đồ hoạ

Máy thuê chỉ có dòng lệnh. OpenGL/Vulkan cần EGL và driver đồ hoạ — cài trên
máy thuê là một mớ hỗn độn, và mỗi nhà cung cấp một khác. Còn **đối tia thì
chỉ là phép tính tensor**: chạy y hệt nhau trên CPU và CUDA, không thêm một
gói nào ngoài `torch`.

Ba loại hình khối, đủ dựng cả căn phòng:

- **hộp xoay quanh trục đứng** — tường, vật cản, vách hộc, mặt bàn, bảng mã
- **trụ đứng có nắp** — người đi lại, xe khác đang đỗ
- **sàn phẳng có lỗ thủng** + **trần có đèn**

Làm **hai lượt**. Lượt một chỉ đi tìm hộp nào gần nhất — phần này chạy trên
`(B, số tia, K)` nên phải gọn hết mức, làm theo từng trục một chứ không xếp
thành vector ba chiều (xếp vector thì phải cấp phát thêm hai mảng lớn, mà ở
kích thước này tiền chuyển bộ nhớ đắt hơn tiền tính toán nhiều lần). Lượt
hai mới tính pháp tuyến, màu và bảng mã, và nó chỉ chạy cho **một** hộp đã
thắng, tức là trên `(B, số tia)`.

**Bảng mã được vẽ bằng toạ độ cắt cục bộ.** Tia chạm mặt trước của tấm bảng
thì lấy vị trí chạm dọc bề ngang, chia cho số ô, tra bảng màu. Đó là phép
dán ảnh thật, không phải một mảng màu dán cứng.

## 3. Camera và cái servo

| | |
|---|---|
| ảnh vào bộ não | 64×48 RGB |
| góc mở | 70° ngang, 52,5° dọc |
| cao | 22 cm |
| servo | −38°…+45°, **chỉ lên xuống** |

Vì sao 64×48: bảng mã rộng 25 cm. Ở 1 m nó chiếm 13 cột ảnh — hơn 4 cột mỗi
ô màu, đọc thoải mái. Ở 2,5 m còn 1,7 cột mỗi ô — bắt đầu đoán mò. Ở 4 m thì
chịu. Nghĩa là **xe buộc phải lại gần rồi ngửa lên nhìn**, chứ không đứng xa
đọc được. Đó là hành vi đáng học, và nó là hệ quả của con số 64 chứ không
phải của một luật nào.

Vì sao chỉ có một servo: đó là phần cứng thật bạn mô tả — một servo 3 chân.
Và nó làm bài toán khó hơn đúng một bậc: muốn nhìn sang bên thì phải xoay cả
xe, tức là phải đánh đổi với việc đang chạy.

## 4. Phần thưởng

Giống bản v1, và vì cùng những lý do đã ghi trong `../docs/SIM.md`:

| | |
|---|---:|
| rơi khỏi sàn | −150 |
| hết pin nằm đường | −150 |
| sống, mỗi bước | +0,02 |
| va chạm, mỗi lần | −1,5 |
| cảm biến vực kêu | −0,8 |
| nạp được điện | +150 × phần pin nạp |
| vừa cắm đúng hộc | +30 |
| cắm nhầm hộc người khác | **−4** |
| lại gần điểm đứng trước miệng hộc (chỉ khi đèn báo sáng) | +3 × mét |

**Không có số hạng nào thưởng cho việc nhìn bảng mã.** Camera phải tự chứng
minh giá trị của nó: đọc được mã thì tránh được −4 và về tới ổ sớm hơn. Nếu
thêm một khoản thưởng cho việc ngửa camera lên thì ta đã dạy nó, và sẽ không
bao giờ biết được camera có thực sự đáng hay không.

Hai cái chết vẫn cùng giá −150, vì lý do cũ: phạt rơi nhẹ hơn thì xe học
cách lao về rồi rơi; sửa xong thì nó học cách đứng im cho hết pin.

## 5. Giáo trình ngược

Giữ nguyên năm pha của bản v1 (`sap-cam`, `truoc-mieng`, `pin-yeu`,
`long-nhong`, `trong-hoc`), vì lý do vẫn thế: một chu kỳ sạc dài hàng phút,
mà phần thưởng nằm ở cuối.

Và vẫn giữ hai chi tiết đã học được:

- cái hộc dùng để đặt xe được chọn ngẫu nhiên, chỉ ~55% số lần là hộc của
  xe. Lần nào cũng đặt đúng hộc của nó thì bộ não học được rằng "cứ cắm là
  có điện" và sẽ không bao giờ nhìn lên bảng mã.
- bộ nhớ trạm được **cố tình làm lệch** 0–2,5 m. Lần nào cũng ghi đúng thì
  bộ não học cách tin vào nó, rồi ra đời là chịu.

## 6. Vì sao đổi từ ES sang PPO

Bản v1 dùng Evolution Strategies cho mạng 3.154 tham số. ES không dùng đường
lan truyền ngược, nó chỉ chấm điểm rồi xáo trọng số — với mạng nhỏ thì rẻ và
song song hoá theo nhân CPU rất tốt.

Ở đây mạng có **1.379.495 tham số** và đầu vào là ảnh. Không có gradient thật
thì số lần thử cần thiết tăng theo số chiều, mà 1,4 triệu chiều thì không
kham nổi. PPO dùng gradient thật, và phép lan truyền ngược theo lô lớn đúng
là thứ GPU sinh ra để làm.

**PPO có hồi quy**, chia lô nhỏ theo **môi trường** chứ không theo thời gian:
mỗi lô nhỏ là một nhóm môi trường với **cả chuỗi thời gian** của nó. Cắt theo
thời gian thì trạng thái GRU đứt, và bộ não sẽ học một bài toán không phải
bài toán thật.

Vì sao vẫn cần GRU dù đã có camera: đọc được bảng mã rồi thì phải **nhớ** nó
trong lúc quay xe đi lùi — lúc đó camera nhìn ra ngoài, không còn thấy bảng
mã nữa. Mạng không trí nhớ sẽ quên mất mình đang cắm vào hộc nào.

**Ảnh giữ dưới dạng uint8** trong bộ đệm. Đổi sang float32 ngay lúc thu thập
thì bộ đệm phình gấp bốn lần, mà bộ đệm là thứ ăn VRAM nhiều nhất.

## 7. Nhật ký lỗi

- **Một tiến trình treo từ phiên trước ăn 89% CPU suốt 160 phút.** Mọi phép
  đo tốc độ trong khoảng đó đều sai — tôi đo được "8 ms cho mọi phép tính
  bất kể kích thước", rồi viết lại cả bộ đối tia dựa trên con số rác đó. Sau
  khi giết tiến trình kia: 49 ms thay vì 3.735 ms, chênh **75 lần**. Bài học:
  thấy con số vô lý thì `ps aux` trước khi sửa code.
- **`--amp` được quảng cáo mà không nối vào đâu cả.** Bộ chia tỉ lệ được tạo
  ra rồi gán vào đối tượng PPO, nhưng hàm cập nhật không hề dùng tới nó. Chạy
  vẫn ra kết quả, chỉ là chậm hơn đáng lẽ — loại lỗi không ai báo gì.
- **Vị trí thử camera nằm trong lòng vật cản**, ra ảnh tối thui, suýt nữa
  tưởng bộ dựng hình hỏng.
- **Thành sau cái hộc đặt sai chỗ**, thò vào lòng hộc 11 cm: lòng hộc chỉ
  còn sâu 20 cm thay vì 31 cm, và **cắm sạc trở thành bất khả thi** — nhưng
  cả bộ kiểm thử vẫn xanh, vì mọi bài kiểm tra tiếp điểm đều đặt xe thẳng
  vào toạ độ rồi gọi hàm, không chạy qua va chạm. Giờ có một bài **lái xe
  lùi vào bằng vật lý thật**. Bài kiểm thử đi tắt qua vật lý thì không thể
  bắt được lỗi hình học.
