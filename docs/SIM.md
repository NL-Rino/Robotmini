# Thiết kế phần mô phỏng

## 1. Bốn thay đổi so với bản cũ

### 1.1 Chạy mãi, chỉ đứt não mới dừng

`FleetSim.run(link, max_seconds=None)` không có giới hạn bước. Điều kiện
thoát duy nhất là `link.any_connected()` trả về False.

| sự kiện | trước | giờ |
|---|---|---|
| hết pin | kết thúc tập | xe nằm im, thành vật cản, thế giới chạy tiếp |
| rơi khỏi sàn | kết thúc tập | như trên |
| chạm trần số bước | kết thúc tập | không còn trần |
| đứt kết nối não | — | **dừng vòng lặp** |

`max_seconds` vẫn còn nhưng mặc định `None` và chỉ dùng trong kiểm thử.

Hai kiểu đường dây:

- `LocalBrains` — não chạy ngay trong tiến trình. `disconnect()` là sự kiện
  ngắt (Ctrl-C trong `tools/run_fleet.py` gọi đúng hàm này).
- `UdpBrainLink` — não ở tiến trình khác qua UDP. Không nhận được gói trả
  lời trong `timeout` giây thì xe coi như mất não, dừng bánh. Tất cả xe mất
  não thì vòng lặp thoát. Tắt `link/brain_server.py` là mô phỏng dừng.

Mất não **từng xe một** thì các xe còn lại vẫn chạy; xe mất não bị ép ga về 0
và cờ `brain_lost` bật.

### 1.2 Đèn báo sạc thay cho lớp cưỡng ép

`sim/failsafe.py` **không còn tồn tại** (có một bài kiểm thử canh đúng việc
này). Thay vào đó, đầu vào số **39** (`batt_low_blink`):

- pin ≥ 15%: luôn bằng 0
- pin < 15%: nhấp nháy 0/1 ở 2 Hz, **liên tục**, không tự tắt, cho tới khi
  sạc lại quá 18% (có trễ 3% cho khỏi chập chờn)

Và chỉ có thế. Không có gì khác xảy ra. Lệnh ga của bộ não đi thẳng xuống
bánh xe không qua lớp nào. Bài `test_pin_thap_khong_lam_doi_lenh_ga` cho một
bộ não lái hết ga ở pin 2% và kiểm tra ga xuống bánh vẫn y nguyên.

**Hệ quả phải tính tới:** 15% bây giờ là phao cứu sinh duy nhất, nên nó phải
đủ dùng thật. Xem mục 4.

### 1.3 Phải lùi đuôi vào hộc

Chân tiếp điện nằm ở `tâm - R·hướng`, tức là **giữa đuôi xe**. Hộc có chân
tiếp điện ở giữa thành trong. Chạm được khi và chỉ khi, trong hệ quy chiếu hộc:

```
lx <= -0,31 + 0,030     (đã lùi vào tận thành trong)
|ly| <= 0,025           (không lệch ngang)
```

Thân xe tròn nên **góc quay không bị khe hộc chặn** — cái bị chặn là vị trí
chân tiếp điện. Quay 180° tại đúng chỗ đó thì chân tiếp điện chỉ ra ngoài
cửa, `in_slot` về 0 ngay. Lệch góc quá ~10° thì chân trượt ngang quá 2,5 cm
cũng mất. Lòng hộc 31 cm, thân 30 cm, nên xe chỉ còn ±0,5 cm xê dịch ngang.

Không có luật nào trong `sim/` nhắc xe phải lùi. Nó chỉ là hình học.

### 1.4 Năm xe, năm mã hộc

`make_fleet_map(seed, n_docks=5)` dựng 5 hộc mang mã 101…105 (cộng hộc mồi
nhử không mã, không phát hồng ngoại). `FleetSim` thả 5 xe, xe `i` mang mã của
hộc `i`, và **xuất phát từ trong chính hộc đó, đuôi ở phía trong, mũi hướng
ra cửa**.

Khi chân tiếp điện chạm:

| | `in_slot` | `id_signal` | có điện |
|---|:---:|:---:|:---:|
| hộc của mình | 1 | **1** | có |
| hộc của xe khác | 1 | 0 | **không** |
| hộc mồi nhử | 1 | 0 | **không** |
| chưa chạm | 0 | 0 | không |

Cắm mạnh vào hộc người khác thì `in_slot` vẫn 1 mãi và pin vẫn tụt — bài
`test_co_cam_manh_vao_hoc_nguoi_khac_cung_khong_sac_duoc` lùi hết ga 6 giây
để kiểm chứng.

Hồng ngoại **không** phân biệt hộc: mọi hộc có điện đều phát giống nhau. Nó
chỉ cho biết "đằng kia có một cái hộc thật". Cái hộc đó của ai thì phải cắm
vào mới biết.

## 2. Bốn mươi tám đầu vào

Dựng ở một chỗ duy nhất: `sim/perception.py`. Mô phỏng và robot thật đều
phải gọi vào đây.

| chỉ số | số | nội dung |
|---|---:|---|
| 0–11 | 12 | quạt LiDAR, `1 − d/8m` |
| 12–13 | 2 | vực trái / phải |
| 14–25 | 12 | 2 ứng viên hộc × [điểm, cự ly, sin/cos phương vị, sin/cos **trục ra**] |
| 26–31 | 6 | hồng ngoại: đèn gọi [thấy, sin, cos] + hộc sạc [thấy, sin, cos] |
| 32–37 | 6 | bộ nhớ trạm: [có, cự ly, sin/cos phương vị, sin/cos trục] |
| 38–40 | 3 | pin: [mức pin, **ĐÈN BÁO SẠC NHẤP NHÁY**, đang có điện] |
| 41–42 | 2 | tiếp điện: [đang cắm vào một hộc, **TÍN HIỆU ĐÚNG HỘC**] |
| 43–47 | 5 | chuyển động: [tốc độ thẳng, tốc độ quay, ga trái, ga phải, chạm] |

So với bản cũ (46): bỏ **biên an toàn** (thuộc về lớp cưỡng ép đã xoá), thêm
`batt_low_blink`, `batt_charging`, `contact_in_slot`, `contact_id_ok`.

Ga báo lại (45, 46) là **ga thực sự đã chạy**, không phải ga được yêu cầu.
Khi một lớp nào đó đè lệnh mà không báo lại thì trạng thái GRU trên laptop
sẽ trôi khỏi thực tế đúng lúc xe đang gặp chuyện.

## 3. Hình học và cảm biến

**Hộc chữ U.** Ngoài 40×40 cm, lòng 31×31 cm, vách bên 4,5 cm, thành sau
9 cm, **vát 8 cm** ở hai góc trong tại miệng (miệng loe ra tới mép ngoài
±20 cm). Không vát thì không dẫn động vi sai nào cắm nổi. Đèn hồng ngoại và
chân tiếp điện đều ở giữa thành trong.

**Chùm hồng ngoại bị chính hai vách bên bóp lại.** Trong mã không có tham số
"góc chùm" nào cả; `sensors._nearest_visible` chỉ hỏi "có nhìn thấy nhau
không" rồi hình học tự lo. Kết quả: nhìn thấy đèn chỉ trong khoảng ±33° quanh
trục. Đây là lý do "phải vòng ra đối diện cửa mới hỏi được" là hệ quả bắt
buộc chứ không phải luật viết tay.

**LiDAR Camsense.** 3000 điểm/giây, 6 Hz → 500 điểm/vòng, một vòng mất
167 ms ≈ 3,3 bước điều khiển. Khử nhoè **chỉ bằng odometry** — robot thật
không có góc thật. Bài `test_khu_nhoe_chi_dung_odometry` cho odometry lệch
0,4 rad so với sự thật và kiểm tra vòng quét nằm trong hệ odom.

**Bộ dò hộc** là thuật toán hình học, không phải mạng:

1. cắt vòng quét thành đoạn liên tục
2. **nối lại các đoạn kề nhau cách nhau dưới 28 cm** — bắt buộc phải có: khi
   nhìn thẳng vào hộc thì hai thành trong gần như song song với tia và chỉ
   được 2–3 điểm, đủ để khoảng cách nhảy vọt và cắt đoạn ngay giữa lòng hộc
3. nối hai đầu thành dây cung, đo độ **lõm**
4. co dây cung cho hết phần lồi
5. **khớp đường thẳng qua thành trong** để lấy trục
6. kiểm tra **bề rộng thành trong ~31 cm** — góc tường cũng lõm cũng đủ sâu,
   nhưng chỗ sâu nhất của nó là một *điểm*, không phải một *mặt phẳng*

Đo được (`python -m tools.measure detector`, ba mặt bằng, 180 tư thế ngẫu
nhiên):

| | |
|---|---|
| lệch vị trí trung vị, tư thế ngẫu nhiên | **1,9 cm** |
| lệch trục trung vị, tư thế ngẫu nhiên | **3,6°** |
| lệch vị trí khi nhìn trong ±15° | **0,3–0,7 cm** |
| lệch trục khi nhìn trong ±15° | **1,5–4°** |
| báo giả thật sự (cách mọi hộc trên 0,8 m) | **10%** |
| ngoài ±45° | mù hẳn |

Mù hẳn ngoài ±45° không phải khiếm khuyết: từ góc đó thì hai vách bên che
mất lòng hộc, LiDAR *không thể* nhìn thấu. Cũng chính vì thế mà "phải vòng
ra đối diện cửa" là bắt buộc chứ không phải lựa chọn.

**Odometry.** Sai số 2,5% là sai số **đường kính chung cả hai bánh** (làm sai
quãng đường), còn cái làm sai *hướng* là phần **lệch giữa hai bánh**, đặt
0,3%. Cho mỗi bánh một sai số 2,5% độc lập là sai: hai bánh lệch nhau 5%,
chia cho vệt bánh 0,235 m ra 0,21 rad mỗi mét — đi 20 m là xe tự quay đủ một
vòng trong đầu nó.

## 4. Vì sao pin lại to như thế

Đây là con số bị ràng buộc chặt nhất, và nó là **hệ quả trực tiếp** của việc
bỏ lớp cưỡng ép cộng với việc có năm cái hộc giống hệt nhau.

Đo trong mô phỏng: xe chạy 130 m thì chỗ nhớ trạm lệch tới **4 m**. Phòng
chỉ rộng 6,4 m và các hộc cách nhau 1,7 m, nên sau một chuyến dài xe **không
còn biết trong năm cái hộc cái nào là của mình**. Cách duy nhất để biết là
lùi đuôi vào và hỏi — mỗi lần hỏi hụt mất 25–30 giây.

Nên 15% phải đủ cho bốn năm lần hỏi:

```
15% × 870 giây = 130 giây ≈ 65 m đường
```

Nếu cắt 15% xuống còn vài chục giây thì bài toán thành không giải được: xe
biết phải về nhưng không còn đủ pin để tìm ra hộc nào là hộc của nó. Bản cũ
không gặp chuyện này vì lớp cưỡng ép cướp quyền lái và tự lái về — nó có bản
đồ riêng của bộ luật viết tay.

Các con số khác trong `sim/params.py` đều có ghi lý do ngay tại chỗ.

## 5. Đo được gì

Chạy lại bằng `python -m tools.measure all`.

| phép đo | kết quả |
|---|---|
| đèn báo sáng → xe **tự về được hộc của mình** | **67/100** (5 mặt bằng × 4 lượt × 5 xe) |
| số lần cắm nhầm hộc của xe khác mỗi lượt về | **0,4** |
| tốc độ | ~880 µs/bước/xe, nhanh hơn thời gian thật **11 lần** với 5 xe |

Con số 0,4 lần cắm nhầm mỗi lượt chính là cơ chế bạn yêu cầu đang chạy: năm
cái hộc giống hệt nhau, odometry đã trôi, nên xe **phải** cắm thử mới biết.
33 lượt còn lại là xe hết pin trước khi tìm ra hộc của nó — đó là bài toán
thật sau khi bỏ lớp cưỡng ép, không phải lỗi mô phỏng.

Để so sánh: bộ luật viết tay của bản cũ đạt 0,414 lần sạc mỗi tập với **một**
xe và **một** hộc, không có hộc nào giống nó để cắm nhầm.

## 6. Bộ luật viết tay dùng để làm gì

`brain/rule_brain.py` **chỉ đọc 48 đầu vào** như bộ não học được: không nhìn
toạ độ thật, không biết hộc nào mang mã nào, không biết mình đang ở đâu trên
bản đồ. Nó có mặt để chứng minh 48 đầu vào là **đủ** làm trọn quy trình, và
để nhìn mô phỏng chạy.

**Đừng dùng nó làm giáo án bắt chước cho phần cắm hộc.** Việc "phải lùi đuôi
vào" và "phải hỏi mã hộc" là để bộ não tự khám phá qua phần thưởng, đúng như
yêu cầu — không dạy trước.

Máy trạng thái: `long-nhong → ve-tram → tim-hoc → ap-mieng → chinh-truc →
lui-vao → hoi-ma → dang-sac → rut-ra`.

## 7. Nhật ký lỗi của chính bản làm lại này

Những cái đã cắn, để khỏi cắn lại:

- **Dấu ngược khi quay xe.** Quay xe một góc `+w` làm mọi hướng cố định
  ngoài thế giới **lùi** đi `−w` trong hệ quy chiếu xe, nên `psi = −axis`
  *tăng* theo `w`. Lái `w = +k·psi` biến 180° từ điểm đẩy thành **điểm hút**:
  xe đứng im ngay trước miệng hộc, mũi chúi vào trong, lắc qua lắc lại cho
  tới khi hết pin. Nhìn từ ngoài giống hệt "bộ luật dở", không giống lỗi dấu.
- **Hai thước đo cho cùng một việc.** `ap-mieng` thoát theo khoảng cách tới
  điểm đích (trộn cả lệch ngang lẫn lệch dọc) còn `chinh-truc` lại kiểm tra
  riêng lệch ngang. Hai trạng thái đá qua đá lại quanh ngưỡng 8 cm vô tận.
- **Vứt đoạn ngắn khi cắt vòng quét.** Nhìn thẳng vào hộc thì hai cạnh vát ở
  miệng chỉ được 2–3 điểm. Vứt chúng đi là mất luôn hai đầu dây cung, độ sâu
  đo được tụt từ 31 cm xuống 19 cm, và bộ dò báo miệng hộc lệch ra 12 cm.
- **Đếm theo bước thay vì theo lần.** Xe nằm kè vào tường thì mỗi bước đều
  "đang chạm"; báo cáo ra 31.724 va chạm và 31.961 lần rơi trong một phiên.
- **Bám mãi vào chỗ nhớ đã sai.** Khi odometry đã lệch 4 m, cứ bám vào toạ
  độ nhớ là nằm đường ngay tại đó. Phải bỏ bộ nhớ sau vài lần hỏi hụt và
  chuyển sang lần theo đèn hồng ngoại.
- **Cấp phát numpy trong vòng lặp nóng.** Chỗ giải va chạm dựng một
  `SegmentSet` mới cho *mỗi đoạn* bị chạm, mỗi bước vài chục mảng numpy tí
  hon.

Cách làm việc: muốn biết A hay B gây lỗi thì **tắt hẳn B rồi đo lại**, đừng
ngồi suy luận. Cả sáu lỗi trên đều tìm ra bằng cách in trạng thái ra rồi
nhìn, không cái nào đoán trúng từ đầu.
