# Header checkpoint — bản gọn để ghép vào hệ thống nhóm

Giữ nguyên toàn bộ trọng số VietOCR fine-tune và config/vocab. Bản này bỏ ảnh/gold test, báo cáo và trọng số Paddle vì nhóm đã có Reader/Paddle. Không phải gói demo độc lập từ ảnh tới OCR bằng Paddle trên máy chưa cài gì.

## Thành phần

- `models/vietocr-pos/checkpoint.pth`: model fine-tune đầy đủ, SHA-256 `a4b7a6985859b8817af993870c1de5ffef8e3c3322c4e317f3964d07d317a2c2`.
- `base.yml`, `vgg-seq2seq.yml`, `checkpoint.json`: vocab/kiến trúc/preprocessing và hash.
- `checkpoint_ocr.py`: nạp model và đọc crop / đọc lại evidence.
- `checkpoint_pipeline.py`: API ưu tiên chữ checkpoint, xuất evidence và Header; không tự fallback Paddle.
- `source-goc`: source kiến trúc, schema và snapshot Reader/KIE để đối chiếu. Không copy đè toàn bộ thư mục này lên repo nhóm.

## Ghép luồng có sẵn Reader

Cần môi trường Python phù hợp, đã có Torch và các thư viện trong `requirements-integration.txt`. Bản gọn không đổi biến cấu hình hoặc cache Paddle của nhóm.

```python
from checkpoint_pipeline import CheckpointHeaderPipeline

# Nạp một lần khi khởi tạo process, không nạp cho từng dòng chữ.
engine = CheckpointHeaderPipeline()

# Reader của nhóm đã chạy một lần, trả document và metadata góc xoay.
result = engine.process(
    'hoa-don.jpg',
    evidence=reader_document,
    reader_metadata=reader_metadata,
)
primary_document = result['evidence']
header = result['header']
```

Lấy evidence + metadata từ `DocumentReader.process_with_metadata`. Thiếu metadata góc xoay của trang OCR sẽ báo lỗi để tránh crop sai; không đoán góc. Không truyền gold. evidence chính dùng chữ checkpoint với ocr_run_id mới; Reader gốc giữ trong `paddle_evidence`. `selection` ghi model/hash/nguồn, `reader_calls_in_this_adapter` bằng 0 khi truyền evidence đã có.

Nếu KIE nhóm đã được cập nhật, có thể lấy `result['evidence']` chạy KIE đó rồi test lại. Header đóng kèm dùng schema header-study-1.0, cần map vào schema hóa đơn chung. Việc đổi chữ cho PP layout/Table/Footer cần kiểm tra riêng. Không gọi nhánh tự chạy Reader khi thiếu evidence nếu máy chưa có runtime/model Paddle.

## Chỉ nhận dạng một crop

```python
from PIL import Image
from checkpoint_ocr import CheckpointRecognizer
recognizer = CheckpointRecognizer()
print(recognizer.recognize_crop(Image.open('dong-chu.png')))
```

Crop phải là một dòng chữ; không đưa cả hóa đơn vào API này. Có thể chạy `python smoke_crop.py --crop 'D:\anh\dong-chu.png'` để kiểm tra model hoặc `python verify_package.py` để kiểm hash gói.

Model giữ nguyên bản đã thử: 6/10 số hóa đơn trên bộ phát triển, chưa chứng minh tốt hơn Paddle mọi trường hợp. Bản gọn chỉ thay thành phần đóng gói, không sửa trọng số để giảm dung lượng.
