# Đọc footer hóa đơn bằng PaddleOCR

Footer bắt đầu từ dòng tổng đầu tiên, chẳng hạn “Cộng tiền hàng” hoặc “Tổng tiền”, và kéo đến mép dưới của ảnh hóa đơn. Paddle quét ảnh để tìm dòng bắt đầu, sau đó đọc lại vùng footer. Với hóa đơn nghiêng, chương trình dùng vị trí của các dòng chữ để lọc những đoạn thuộc hàng phía trên dòng tổng. Chữ dọc nằm ngoài cột chữ trên hóa đơn được bỏ qua. Nếu còn đoạn bị bỏ sót, chương trình có thể dùng kết quả quét toàn ảnh hoặc cho Paddle đọc lại một vùng nhỏ. Console vẫn in chữ thực tế Paddle đọc được cùng điểm tin cậy.

## Chuẩn bị trên Windows

Giải nén ZIP và mở PowerShell trong thư mục có `paddle_footer.py`. Dùng Python 3.11 bản 64 bit. Các lệnh sau tạo môi trường riêng và cài phiên bản CPU của các thư viện.

```powershell
py -3.11 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r .\requirements-footer.txt
```

Lần chạy đầu, Paddle cần mạng để tải hai model `PP-OCRv5_server_det` và `latin_PP-OCRv5_mobile_rec` nếu máy chưa có sẵn. Nếu đã có model, chép hai thư mục của chúng vào `models` cạnh `paddle_footer.py` hoặc truyền `--model-dir` trỏ đến thư mục chứa trực tiếp hai model. Mỗi thư mục model cần có các file `inference.json`, `inference.pdiparams` và `inference.yml`.

## Chạy với một ảnh

```powershell
$anh = (Read-Host 'Nhập đường dẫn đến ảnh hóa đơn').Trim('"')
& $py -B .\paddle_footer.py --image $anh
```

Mỗi đoạn chữ được in cùng điểm tin cậy do Paddle trả về. Phần chữ giữ nguyên cách Paddle đọc, kể cả dấu và chữ số, để tiện so với ảnh.

```text
22,300 | Tin cậy 99.98%
```

Thêm `--output .\ket_qua.json` để lưu chữ, điểm tin cậy và tọa độ vùng footer vào JSON.

```powershell
& $py -B .\paddle_footer.py --image $anh --output .\ket_qua.json
```

## Các file chính

- `paddle_footer.py` nhận ảnh, điều phối hai lượt đọc và xuất kết quả.
- `footer_layout.py` tìm dòng tổng và tính vùng cần đọc.
- `paddle_runtime.py` nạp model PaddleOCR và chuyển kết quả về tọa độ ảnh gốc.
- `KIEM_THU_20_ANH.md` ghi kết quả đối chiếu trên bộ 20 ảnh.
