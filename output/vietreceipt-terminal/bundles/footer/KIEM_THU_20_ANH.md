# Kiểm thử footer trên 20 ảnh hóa đơn

PaddleOCR chạy trên CPU với `PP-OCRv5_server_det` và `latin_PP-OCRv5_mobile_rec`.
Vị trí dòng mở đầu và dòng cuối được đối chiếu với ảnh và vùng footer gán tay.
Dòng tổng gồm cả “Tổng số” nếu đó là dòng mở đầu của cụm tổng trước các dòng tiền.

- **20/20** ảnh bắt đầu đúng ở dòng mở đầu cụm tổng. Cả 20 vùng đều kéo đến mép dưới ảnh.
- **19/20** ảnh có nhận ra nội dung dòng cuối. Dòng cảm ơn ở `full_03` bị cắt ở mép ảnh và chỉ đọc được một phần.
- Tỷ lệ lỗi ký tự so với bản chép là **11,35%**. Quy tắc trích số tiền phải trả khớp **17/20** ảnh.

| Ảnh | Dòng đầu footer | Dòng cuối | Lỗi ký tự | Tiền phải trả |
|---|---|---|---:|---|
| `full_01` | Tổng số | Có | 9,50% | Chưa trích được |
| `full_02` | Thành tiền | Có | 6,80% | Khớp |
| `full_03` | Tổng số (mép trái bị cắt) | Một phần | 26,56% | Chưa trích được |
| `full_04` | Tổng số (mép trái bị cắt) | Có | 10,96% | Khớp |
| `full_05` | Cộng tiền hàng | Có | 11,32% | Khớp |
| `full_06` | Tổng tiền phải thanh toán | Có | 8,49% | Khớp |
| `full_07` | Tổng tiền phải thanh toán | Có | 9,54% | Khớp |
| `full_08` | Tổng tiền phải thanh toán | Có | 9,41% | Khớp |
| `foggy_09` | Tổng cộng | Có | 14,89% | Khớp |
| `foggy_10` | Tổng tiền phải thanh toán | Có | 9,75% | Khớp |
| `foggy_11` | Số tiền | Có | 11,63% | Khớp |
| `foggy_12` | Cộng tiền hàng | Có | 11,54% | Khớp |
| `angled_13` | Tổng | Có | 7,87% | Khớp |
| `angled_14` | Cộng tiền hàng | Có | 35,25% | Chưa trích được |
| `angled_15` | Cộng tiền hàng | Có | 27,56% | Khớp |
| `angled_16` | Tổng số | Có | 6,25% | Khớp |
| `lowlight_17` | Cộng tiền hàng | Có | 13,46% | Khớp |
| `lowlight_18` | Tổng tiền phải thanh toán | Có | 10,47% | Khớp |
| `lowlight_19` | Tổng tiền | Có | 6,77% | Khớp |
| `lowlight_20` | Tổng tiền | Có | 6,21% | Khớp |

Ở `foggy_10`, phần chữ dọc ngoài tờ giấy được loại và lượt đọc đoạn sát đáy lấy lại hotline cùng website. Ở `angled_15`, lượt đọc riêng khôi phục phần lời cảm ơn bị bỏ sót. Giấy cong ở `angled_14` và `angled_15` vẫn có thể làm chữ và số bị ghép lệch hàng.

Lượt tìm bố cục thu cạnh dài về 1.000 px trên `angled_14`, `angled_15`, `lowlight_19` và `lowlight_20` khi dùng chế độ đánh giá có nhãn. Lượt đọc footer dùng ảnh gốc. Điểm tin cậy Paddle in ở terminal và tỷ lệ lỗi ký tự trong bảng là hai số đo khác nhau.
