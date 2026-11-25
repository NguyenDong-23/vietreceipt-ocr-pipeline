"""Chỉ trích xuất 9 trường Header; không gọi bộ trích dòng hàng hoặc Footer V1."""
from setup_runtime import configure  # Lấy hàm cấu hình source/thư viện.
configure()  # Chạy trước các import ai để dùng source đóng trong gói.

from ai.kie.v2.cells import add_reason, select  # Chọn ứng viên và ghi lý do cần review.
from ai.kie.v2.evidence import flatten  # Kiểm tra evidence rồi đổi các trang thành danh sách block chữ.
from ai.kie.v2.extraction import header_candidates, table_regions  # Tìm ứng viên Header và vị trí bảng cần tránh.
from ai.kie.v2.grouping import (  # Giữ cùng cách loại vùng bảng như luồng interval đã test trước.
    EXPERIMENT_LINE_LABELS, EXPERIMENT_LINE_STOP,  # Các nhãn cột và nhãn dừng bảng của cấu hình cũ.
    merge_receipt_item_rows, recover_merged_receipt_columns, unheaded_receipt_regions,  # Hỗ trợ nhận diện block thuộc vùng bảng.
)

HEADER_FIELDS = (  # Đây là danh sách trường Header được xuất và có thể chấm.
    "invoice_template_number", "invoice_symbol", "invoice_number", "invoice_date",  # Mẫu số, ký hiệu, số hóa đơn, ngày lập.
    "seller_name", "seller_tax_id", "seller_address", "buyer_name", "buyer_tax_id",  # Thông tin người bán/người mua.
)


def locate_table_blocks(blocks):  # Xác định block thuộc bảng để tránh lấy nhầm làm Header.
    regions, excluded = table_regions(  # Lấy vùng bảng và tập ID block bị loại khỏi Header.
        blocks, line_labels=EXPERIMENT_LINE_LABELS,  # Dùng danh sách nhãn của cấu hình interval hiện có.
        line_stop_pattern=EXPERIMENT_LINE_STOP, preserve_numeric_rows=True,  # Giữ chính sách nhận diện bảng trước đây.
    )
    prepared = []  # Gom các vùng bảng đã nhận diện theo cùng cách của cấu hình cũ.
    for kind, columns, rows in regions:  # Duyệt từng vùng bảng.
        if kind == "line":  # Nếu đây là vùng dòng hàng.
            columns = recover_merged_receipt_columns(blocks, columns, rows)  # Xác định cột ngay cả khi OCR ghép ô.
            rows = merge_receipt_item_rows(rows, columns)  # Gắn các dòng nối của cùng mặt hàng.
        prepared.append((kind, columns, rows))  # Giữ vùng đã chuẩn bị để tìm các vùng chưa có tiêu đề.
    recovered = unheaded_receipt_regions(blocks, prepared)  # Tìm vùng hàng không có tiêu đề cột.
    excluded.update(b.block_id for _, _, rows in recovered for row in rows for b in row)  # Loại thêm ID block của vùng phục hồi khỏi Header.
    return excluded  # Chỉ trả tập ID cần tránh; không trích giá/số lượng/thành tiền.


def extract_header(evidence):  # Nhận nguyên văn chữ/tọa độ Reader, trả dữ liệu Header.
    blocks = flatten(evidence)  # Kiểm tra contract nguồn và tạo các block có tọa độ.
    excluded = locate_table_blocks(blocks)  # Tránh lấy chữ bên trong bảng làm số hóa đơn hoặc tên người bán.
    candidates = header_candidates(blocks, excluded)  # Tìm các giá trị có thể thuộc từng trường Header.
    fields = {name: select(name, candidates[name]) for name in HEADER_FIELDS}  # Chọn giá trị và trạng thái cho cả 9 trường.
    number = fields["invoice_number"]  # Lấy ô số hóa đơn để áp quy tắc định dạng của bản cũ.
    if number["normalized_value"] and number["normalized_value"].endswith("-"):  # Mã kết thúc bằng dấu gạch có thể chưa đầy đủ.
        add_reason(number, "AMBIGUOUS_FORMAT")  # Yêu cầu review, không tự thêm ký tự vào mã.
    seller = fields["seller_name"]  # Lấy ô tên người bán để kiểm cách suy vai trò.
    if seller["value_status"] == "PRESENT" and any(  # Nếu tên có giá trị nhưng vai trò được suy từ chữ đầu trang.
        c.value == seller["predicted_value"] and c.blocks[0].text in c.value  # Tìm lại ứng viên đã được chọn.
        for c in candidates["seller_name"]  # Xét các ứng viên tên người bán.
    ):
        add_reason(seller, "SOURCE_ROLE_UNCLEAR")  # Giữ lý do cần xác nhận vai trò người bán.
    return {  # Lưu Header và thông tin nguồn phục vụ truy lỗi.
        "schema_version": "header-study-1.0",  # Đây là cấu trúc riêng của bộ test Header, không giả làm hóa đơn V2 đầy đủ.
        "receipt_id": evidence["receipt_id"], "ocr_run_id": evidence["ocr_run_id"],  # Giữ nguyên định danh nguồn.
        "fields": fields,  # Lưu các ô Header cùng raw_text, status, ID nguồn và giá trị chuẩn hóa.
        "excluded_table_block_ids": sorted(excluded),  # Cho người học xem block nào đã bị loại vì thuộc bảng.
        "ocr_block_count": len(blocks),  # Ghi số block Reader đã cung cấp.
    }
