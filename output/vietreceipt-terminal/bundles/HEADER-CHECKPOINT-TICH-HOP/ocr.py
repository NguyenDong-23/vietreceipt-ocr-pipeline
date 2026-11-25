"""Chạy Reader từ ảnh/PDF thật hoặc đọc lại evidence đã lưu."""
import json  # Đọc evidence JSON.
from pathlib import Path  # Làm việc với file đầu vào.
from uuid import uuid4  # Tạo định danh riêng cho lượt OCR mới.
from setup_runtime import configure  # Cấu hình source và runtime trước khi dùng Reader.
configure()  # Đảm bảo import đúng source-goc đã đóng trong bộ test.

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".pdf": "application/pdf"}  # Các loại đầu vào Reader hỗ trợ.


def read_saved(path):  # Replay: đọc OCR có sẵn, không chạy lại model.
    return json.loads(Path(path).read_text(encoding="utf-8"))  # Trả nguyên evidence đã lưu.


class LiveReader:  # Dùng một Reader cho nhiều ảnh trong cùng một lần chạy.
    def __init__(self):  # Khởi tạo khi người dùng chọn chế độ live.
        from ai.document_reader import DocumentReader  # Source thực sự nằm trong source-goc/ai/document_reader.
        self.reader = DocumentReader()  # Tạo bộ đọc ảnh/PDF; model OCR được nạp khi cần.

    def read(self, source):  # Nhận đường dẫn ảnh hoặc PDF để chạy mới.
        source = Path(source)  # Chuẩn hóa kiểu đường dẫn.
        content_type = MIME.get(source.suffix.lower())  # Xác định loại file dựa vào đuôi.
        if content_type is None:  # Nếu không thuộc các định dạng hỗ trợ.
            raise ValueError("Chỉ nhận JPG, JPEG, PNG hoặc PDF.")  # Báo lỗi rõ ràng thay vì đoán loại file.
        return self.reader.process_with_metadata(  # Reader trả cả evidence và thông tin lượt chạy.
            source, content_type=content_type, receipt_id=uuid4(), ocr_run_id=uuid4(),  # Mỗi lần live tạo định danh mới.
        )
