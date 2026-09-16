# Cách dạy bộ não — và vì sao bản cũ chạy 5.900 thế hệ mà không tới đâu

Mạng thì đúng rồi, không phải sửa. Vấn đề nằm ở **cách chấm điểm**.

---

## 1. Chẩn đoán

Bản cũ: quần thể 48, **3 tập mỗi cá thể**, mỗi cá thể chạy trên **3 tập
riêng của nó**. Tài liệu bàn giao tự nhận ra đây là chỗ đáng sửa nhất
("chấm điểm bằng 3 tập là chấm gần như ngẫu nhiên") nhưng đề xuất *tăng lên
5–8 tập* — tức là tiêu gấp đôi tính toán để mua lại một phần độ chính xác.

ES chỉ dùng **một** đại lượng duy nhất: với mỗi hướng nhiễu `eps`, hiệu điểm

```
d = f(θ + σ·eps) − f(θ − σ·eps)
```

rồi xếp hạng theo `d`. Nên câu hỏi duy nhất đáng hỏi là: trong `d` đo được,
bao nhiêu là thật và bao nhiêu là vận may.

Đo được (`python -m tools.measure crn`): **độ lệch chuẩn điểm giữa các hạt
giống là 16,9** — tức là cùng một bộ não, đổi mặt bằng và chỗ đặt xe thôi,
điểm đã nhảy chừng ấy. Với `σ = 0,05` thì chênh lệch thật giữa `θ+σ·eps` và
`θ−σ·eps` nhỏ hơn con số đó nhiều. Chấm bằng 3 tập riêng thì cái được xếp
hạng chủ yếu là mặt bằng ai dễ hơn.

Nhưng **chỗ hỏng lớn nhất không nằm ở đây.** Nó nằm ở chỗ: một chu kỳ sạc
dài hàng phút, mà mỗi lần đánh giá lại bắt đầu từ lúc xe đầy pin nằm trong
hộc. Trong hàng trăm thế hệ đầu không một cá thể nào chạy đủ xa để chạm vào
phần thưởng. Không có gì để xếp hạng thì xếp hạng kiểu gì cũng vô nghĩa.
Xem mục 2.5 — đó mới là thứ đổi cục diện.

## 2. Năm chỗ đã sửa

### 2.1 Chung số ngẫu nhiên (`common random numbers`)

Trong **một** thế hệ, mọi cá thể chạy trên **đúng** một bộ hạt giống: cùng
mặt bằng, cùng chỗ đặt xe, cùng nhiễu cảm biến, cùng đường đi của người.
Sang thế hệ sau thì đổi hạt giống.

Toàn bộ cái lợi của nó nằm gọn trong một con số — `rho`, hệ số tương quan
giữa điểm của hai cá thể khi chạy trên cùng hạt giống:

```
chấm riêng:  Var(d) = Var(f₊) + Var(f₋)
chấm chung:  Var(d) = Var(f₊) + Var(f₋) − 2·Cov = 2·Var·(1 − rho)
```

Đo được trên 4 hướng nhiễu, mỗi hướng 40 hạt giống:

```
rho = +0,44 ± 0,13      từng hướng:  −0,01   +0,85   +0,84   +0,07
```

Phân bố lệch hẳn thành hai cụm, và đó là phần đáng chú ý:

- Hướng nào mà nhiễu **không đổi hành vi mấy** thì `rho ≈ 0,85`, và chung
  hạt giống cắt nhiễu xuống còn **39%**. Đây đúng là những hướng khó xếp
  hạng nhất — hai cá thể gần như giống nhau.
- Hướng nào mà nhiễu **đổi hẳn hành vi** thì hai xe rẽ đi hai ngả ngay từ
  đầu, `rho ≈ 0`, chung hạt giống không lợi gì. Nhưng những hướng đó thì
  chênh lệch điểm cũng đã lớn sẵn, dễ xếp hạng rồi.

Trung bình cắt nhiễu xuống còn **75%**, và cắt mạnh nhất đúng chỗ cần. Miễn
phí hoàn toàn: cùng ngần ấy lần chạy. Nó cũng cho phép **giảm** số tập mỗi
cá thể chứ không phải tăng.

Điều kiện để nó chạy được: cùng bộ não + cùng hạt giống phải ra **đúng một
điểm**. Có một bài kiểm thử canh riêng việc này.

### 2.2 Xáo đối gương (`mirrored sampling`)

Với mỗi `eps`, đánh giá **cả hai phía** `θ+σ·eps` và `θ−σ·eps`. Ước lượng
gradient thành `(f₊ − f₋)/2 · eps`, tự triệt tiêu phần chung nên không cần
mốc so sánh nữa, và phương sai giảm khoảng một nửa. Cũng miễn phí.

### 2.3 Chuẩn hoá đầu vào

48 đầu vào có phân bố lệch nhau rất xa: đèn báo sạc gần như luôn bằng 0,
quạt LiDAR nhìn vào tường thì luôn quanh 0,9. Không chuẩn hoá thì mỗi trọng
số học với một tốc độ khác hẳn nhau.

Giữ trung bình và phương sai trượt của đầu vào, **dùng chung cho cả quần
thể**, đóng băng trong một thế hệ và cập nhật ở cuối thế hệ. Đóng băng là
bắt buộc: nếu mỗi cá thể tự chuẩn hoá theo riêng nó thì điểm của chúng
không so sánh được với nhau nữa.

### 2.4 AdamW, không phải Adam + weight_decay

Lỗi (l) của bản cũ. `weight_decay` cộng thẳng vào gradient thì bị Adam chia
cho `sqrt(v)` và mất tác dụng kéo trọng số về 0 → trọng số phình dần → tanh
bão hoà → bộ não trả về `(+1,+1)` với **mọi** đầu vào. Một phiên 5.900 thế
hệ tự huỷ vì đúng chỗ này.

AdamW trừ phần suy giảm **riêng**, sau khi đã chia, nên nó làm đúng việc
của nó. Có bài kiểm thử: gradient bằng 0 thì trọng số phải co lại.

Kèm theo: xếp hạng giữa (`centered rank`) có chặn — cả quần thể điểm bằng
nhau thì trả về vector 0, không sinh gradient từ hư không.

### 2.5 Giáo trình ngược — chỗ quan trọng nhất

Đây mới là thứ quyết định bài toán này có giải được hay không.

Một chu kỳ sạc đầy đủ dài hàng phút mô phỏng. Nếu mỗi lần đánh giá đều bắt
đầu từ lúc xe đầy pin trong hộc thì phải chạy hết vài nghìn bước mới tới
chỗ có phần thưởng, và trong hàng trăm thế hệ đầu **không một cá thể nào**
chạm được vào nó. ES không có gì để so sánh. Nó chỉ xáo trọng số.

Cách chữa: đặt xe vào một điểm **bất kỳ** trên chu kỳ, không phải luôn luôn
ở đầu. Năm pha:

| pha | đặt xe ở đâu | còn phải làm gì |
|---|---|---|
| `sap-cam` | lùi dang dở vào một hộc, pin cạn | còn 20 cm nữa |
| `truoc-mieng` | trước miệng một hộc, pin cạn | còn quay và lùi |
| `pin-yeu` | bất kỳ đâu, pin dưới ngưỡng | còn tìm đường về |
| `long-nhong` | bất kỳ đâu, pin còn nhiều | chỉ cần không đâm |
| `trong-hoc` | trong hộc của mình, pin đầy | học cách ra đi |

Những thế hệ đầu dành phần lớn suất cho hai pha cuối bảng, nơi phần thưởng
chỉ cách vài chục bước. Khá lên tới đâu thì đẩy dần suất về phía đầu bảng.

Hai chi tiết nhỏ nhưng quan trọng:

- **Trong mỗi pha vẫn có dải khó dễ.** Một pha chỉ có một mức khó thì hoặc
  dễ quá (không học thêm được gì) hoặc khó quá (không ai chạm tới). Đo lúc
  đầu: pha `sap-cam` với lệch góc cố định 14° thì chính sách "cứ lùi" chỉ
  ăn 39%; thêm dải khó dễ vào thì gradient mới liên tục.
- **Cái hộc trong pha `truoc-mieng`/`sap-cam` được chọn ngẫu nhiên**, chỉ
  khoảng một nửa số lần là hộc của xe. Nếu lần nào cũng đặt đúng hộc của nó
  thì bộ não sẽ học được rằng "cứ cắm là có điện", và sẽ không bao giờ nhìn
  tới tín hiệu bắt tay.

Và: **bộ nhớ trạm lúc huấn luyện được cố tình làm lệch** 0–2,5 m ngẫu
nhiên. Xe thật chạy 130 m là chỗ nhớ lệch 4 m; nếu lúc huấn luyện chỗ nhớ
lúc nào cũng đúng thì bộ não sẽ học cách tin vào nó, rồi ra đời là chịu.

---

## 3. Phần thưởng nói xe NÊN Ở ĐÂU, không nói xe PHẢI LÀM GÌ

| | |
|---|---:|
| rơi khỏi sàn | −150 |
| hết pin nằm đường | −150 |
| sống, mỗi bước | +0,02 |
| va chạm, mỗi lần | −1,5 |
| cảm biến vực kêu, mỗi bước | −0,8 |
| nạp được điện | +150 × phần pin nạp |
| vừa cắm đúng hộc | +30 |
| cắm nhầm hộc người khác | **−4** |
| tới được đèn gọi | +25 |
| lại gần điểm đứng trước miệng hộc (chỉ khi đèn báo sáng) | +3 × mét |

Ba điều đáng nói:

**Hai cái chết cùng giá.** Lỗi (g) rồi (h) của bản cũ: phạt rơi −40 mà đi
2 m về trạm được +28 thì xe học cách lao về rồi rơi; sửa xong thì nó học
cách chết rẻ hơn là đứng quay tại chỗ cho hết pin. Giờ cả hai đều −150, và
có thưởng sống mỗi bước nên đứng im cũng không rẻ.

**Cắm nhầm chỉ phạt nhẹ (−4).** Vì đó là **cách hợp lệ** để biết hộc nào là
của mình — năm cái hộc nhìn giống hệt nhau, odometry đã trôi, không cắm thử
thì không có cách nào khác. Phạt nặng là dạy xe sợ cắm, mà sợ cắm thì không
bao giờ sạc được.

**Số hạng dẫn đường dẫn tới ĐIỂM ĐỨNG TRƯỚC MIỆNG, không phải tới cái hộc.**
Dẫn thẳng tới hộc thì xe bị hút vào sườn hộc — chỗ gần nhất nhưng là ngõ
cụt, không nhìn thấu lòng hộc mà cũng không bắt được đèn hồng ngoại.

Không có số hạng nào thưởng cho việc lùi, cho việc quay đầu, hay cho việc
hỏi mã hộc. Ba việc đó để bộ não tự tìm ra — đúng như yêu cầu "cái này
không dạy". Phần thưởng nói *nên ở đâu*, chưa bao giờ nói *phải làm gì*.

---

## 4. Cấu hình cho máy của bạn

i5-7200U, 2 nhân / 4 luồng, Intel HD 620, Windows. **GPU không giúp gì**:
numpy không dùng GPU, và mô phỏng là chuỗi phép tính nhỏ nối tiếp nên riêng
tiền chuyển dữ liệu đã đắt hơn.

Mô phỏng đã được tăng tốc **1,45 lần** cho lần này (ray-cast chuyển sang
float32 và cắt vòng quét vector hoá): còn ~600 µs mỗi bước mỗi xe.

Cài đặt khuyến nghị — đây là mặc định của phần mềm:

```
quần thể     24      12 cặp đối gương
số bước/lần  400     20 giây mô phỏng
số xe        3       train nhỏ, đánh giá lớn
số tập       2       ít thôi, vì đã chung hạt giống
số luồng     3–4
số nơ-ron   16       GRU 48→16→2, 3.154 tham số
giáo trình  600      sau ngần này thế hệ thì hết dễ
```

Ước chừng **20–40 giây một thế hệ** trên máy bạn. Chạy qua đêm 10 tiếng ≈
1.000–1.800 thế hệ. Bấm **DỪNG AN TOÀN** lúc nào cũng được, hôm sau chọn
*Chạy tiếp* là đi tiếp từ đúng chỗ đó.

**Train nhỏ, đánh giá lớn.** Huấn luyện trên mặt bằng 3 xe / 3 hộc cho rẻ,
rồi xem nó chạy trên mặt bằng 5 xe. Chính sách là của từng xe và chỉ nhìn
theo hệ quy chiếu của chính nó, nên nó chuyển sang được.

Muốn nhanh hơn nữa thì hạ `số bước/lần` xuống 250 trước, đừng hạ quần thể
xuống dưới 16 — quần thể quá nhỏ thì ước lượng gradient nát.

---

## 5. Đọc bảng số lúc đang chạy

| cột | nghĩa | khi nào phải lo |
|---|---|---|
| `diem` | điểm trung bình quần thể | nhảy loạn là bình thường |
| `do rieng` | điểm trên 6 mặt bằng cố định | **đây mới là cái phải lên** |
| `sac` | tổng phần pin nạp được | 0 mãi = chưa tìm ra hộc |
| `roi` / `het pin` | số lần chết | phải về 0 trước tiên |
| `nham` | số lần cắm nhầm hộc | vài chục là bình thường |
| `sat` | tỉ lệ trọng số lớn bất thường | **> 0,25 là tanh sắp bão hoà** |
| `grad` | độ dài gradient | = 0 nghĩa là cả quần thể điểm bằng nhau |

Dấu hiệu hỏng:

- `sat` tăng đều → trọng số phình. Phần mềm sẽ in cảnh báo. Hạ `lr` hoặc
  tăng `weight_decay`.
- `grad` = 0 nhiều thế hệ liền → cả quần thể điểm y hệt nhau, `sigma` quá
  nhỏ so với địa hình. Tăng `sigma`.
- `do rieng` đứng im 200 thế hệ → tăng quần thể hoặc `sigma`.

Đường học đo được của bản này (quần thể 24, 400 bước, 3 xe):

```
thế hệ     1   điểm đo riêng     3,9
thế hệ    15                   100,4
thế hệ    30                   134,8
thế hệ    45                   160,2
thế hệ    60                   185,6
thế hệ    75                   199,8
thế hệ    90                   220,9
thế hệ   105                   221,0
thế hệ   120                   298,4
```

Phá kỷ lục ở **mọi** mốc đo. Bản cũ chạy 5.900 thế hệ mà `best.npz` vẫn ghi
`gen=1744` — tức là 4.000 thế hệ cuối không cải thiện được gì.

---

## 6. Hai file, đừng nhầm

```
state.npz   CHẠY TIẾP ĐƯỢC: trọng số, mô-men AdamW, bộ chuẩn hoá,
            số thế hệ, tiến độ giáo trình
best.npz    CHỈ là bản sao lúc phá kỷ lục, KHÔNG chạy tiếp được
```

Lỗi (k) của bản cũ: `best.npz` chỉ được ghi khi phá kỷ lục, nên đến thế hệ
6.000 nó vẫn ghi `gen=1744`, và "chạy tiếp" từ nó là quay ngược về quá khứ.
Nút *Chạy tiếp* trong phần mềm luôn dùng `state.npz`.

---

## 7. Nhật ký lỗi của chính bản này

- **Đo hai lần ước lượng gradient bằng hai vector nhiễu khác nhau** rồi so
  cosin. Đương nhiên không tương quan — chúng là tổng trên những hướng khác
  nhau. Con số đầu tiên tôi có (+0,31 so với +0,11) ra từ phép đo hỏng này.
- **Đo `Var(d)` bằng 6–10 lần lấy mẫu.** Phương sai ước lượng từ 10 mẫu thì
  sai số của chính nó đã hơn hai lần; hai lần đo ra hai kết luận ngược nhau
  là chuyện thường. Mất hai vòng đo vì chỗ này trước khi chuyển sang đo
  thẳng `rho` — đại lượng đo ổn định hơn hẳn với cùng ngần ấy tính toán.
- **Chọn một mức khó cố định cho mỗi pha giáo trình.** Pha `sap-cam` với
  lệch góc cố định 14° thì ngay cả chính sách "cứ lùi" cũng chỉ ăn 39%;
  gradient đứt quãng. Phải có dải khó dễ trong từng pha.
- **Lưu số thế hệ đang dở.** Vòng lặp kiểm tra file `STOP` ở *đầu* mỗi thế
  hệ rồi thoát, nên biến đếm đang trỏ vào thế hệ **chưa chạy**. Ghi số đó
  vào `state.npz` là lần sau chạy tiếp sẽ nhảy cóc mất một thế hệ.

Cách làm việc: muốn biết A hay B gây lỗi thì **tắt hẳn B rồi đo lại**, đừng
ngồi suy luận. Và nếu phép đo ra con số đẹp bất ngờ thì đo lại với nhiều
mẫu hơn trước khi tin nó.

## 8. Những thứ đã cân nhắc rồi bỏ

- **Học bắt chước bộ luật viết tay.** Khởi động nguội thì nhanh, nhưng bộ
  luật đó đã biết sẵn phải lùi đuôi và phải hỏi mã — dạy nó là đi ngược
  đúng cái bạn yêu cầu. Giáo trình ngược đạt cùng mục đích mà không dạy gì.
- **Chính sách tuyến tính** (kiểu ARS) chỉ 98 tham số, ES hội tụ nhanh hơn
  nhiều. Nhưng bài này cần trí nhớ: ước lượng vị trí hộc phải gộp qua nhiều
  vòng quét, và "đang trên đường về" là một trạng thái. Tuyến tính không có
  chỗ chứa. GRU 16 nơ-ron là mức rẻ nhất mà vẫn có trí nhớ.
- **Tăng số tập lên 5–8** như bản cũ đề xuất. Chung hạt giống mua được một
  phần cái đó mà không tốn gì; phần còn lại thì giáo trình ngược mua rẻ hơn
  nhiều, vì nó làm phần thưởng với tới được ngay từ thế hệ đầu.
