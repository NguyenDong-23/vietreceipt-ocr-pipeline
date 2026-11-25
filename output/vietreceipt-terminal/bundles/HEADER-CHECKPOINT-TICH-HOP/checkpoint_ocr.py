"""Nạp checkpoint VietOCR và đọc lại đúng các khung Reader đã phát hiện.

Không nhận gold. Không sửa evidence gốc. Mỗi variant tạo ocr_run_id mới.
"""
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path
from uuid import uuid4

from setup_runtime import ROOT, configure
configure()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckpointRecognizer:
    """API cho nhóm: recognize_crop(PIL_RGB) → text + model_probability."""

    def __init__(self, model_dir=None):
        import torch
        import yaml
        from vietocr.tool.translate import build_model, process_input, translate

        self.folder = Path(model_dir or ROOT / "models/vietocr-pos").resolve()
        self.metadata = json.loads((self.folder / "checkpoint.json").read_text(encoding="utf-8"))
        self.sha256 = file_hash(self.folder / "checkpoint.pth")
        if self.sha256 != self.metadata["checkpoint_sha256"]:
            raise ValueError("Hash checkpoint không khớp cấu hình đã đóng gói.")
        for filename, expected in self.metadata["config_sha256"].items():
            if file_hash(self.folder / filename) != expected:
                raise ValueError(f"Cấu hình model đã thay đổi: {filename}")
        config = yaml.safe_load((self.folder / "base.yml").read_text(encoding="utf-8"))
        config.update(yaml.safe_load((self.folder / "vgg-seq2seq.yml").read_text(encoding="utf-8")))
        config["device"] = "cpu"
        config["cnn"]["pretrained"] = False  # Nạp toàn bộ checkpoint đã kèm; không tải CNN từ mạng.
        torch.set_num_threads(1)
        self.model, self.vocab = build_model(config)
        weights = torch.load(self.folder / "checkpoint.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)  # Sai kiến trúc/bộ ký tự phải báo lỗi.
        self.model.eval()
        self.torch, self.prepare, self.translate = torch, process_input, translate
        self.charset = config["vocab"]
        if not set("#@*&+0123456789Đđấềể").issubset(set(self.charset)):
            raise ValueError("Bộ ký tự thiếu dấu hoặc ký tự đặc biệt cần kiểm tra.")

    def recognize_crop(self, crop):
        image = crop.convert("RGB")
        tensor = self.prepare(image, 32, 32, 512)  # Giữ đúng preprocessing đã dùng khi train checkpoint.
        with self.torch.inference_mode():
            tokens, probability = self.translate(tensor, self.model, max_seq_length=128)
        value = float(probability[0])
        text = self.vocab.decode(tokens[0].tolist()).strip()
        return {
            "text": text,
            "model_probability": value if math.isfinite(value) and 0 <= value <= 1 else None,
            "decode_hit_length_limit": self.vocab.eos not in tokens[0].tolist(),
            "confidence_semantics": "uncalibrated_vietocr_mean_token_probability",
        }


def crop_polygon(image, polygon, clockwise=0):
    """Cắt từ cùng ảnh, cùng polygon chuẩn hóa 0..1; không dùng nội dung nhãn."""
    import cv2
    import numpy as np
    from PIL import Image

    width, height = image.size
    points = np.asarray([(p["x"] * width, p["y"] * height) for p in polygon], dtype=np.float32)
    w = round(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = round(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if min(w, h) < 3:
        raise ValueError("Khung chữ nhỏ hơn 3 pixel; không tự đoán nội dung.")
    target = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(np.asarray(image), cv2.getPerspectiveTransform(points, target), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(crop).rotate(-clockwise, expand=True) if clockwise else Image.fromarray(crop)


def read_page_image(source, page_index, expected_size):
    """Tái dựng đúng khung pixel của Reader; PDF có intrinsic rotation dùng rotation=0."""
    from PIL import Image, ImageOps

    source = Path(source)
    if source.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        from contextlib import closing
        with closing(pdfium.PdfDocument(source)) as document:
            with closing(document[page_index]) as page:
                page.set_rotation(0)
                scale = expected_size[0] / page.get_width()
                with closing(page.render(scale=scale)) as bitmap:
                    image = bitmap.to_pil().convert("RGB").copy()
        if any(abs(a - b) > 1 for a, b in zip(image.size, expected_size)):
            raise ValueError("Kích thước PDF render không khớp khung evidence.")
        return image.resize(expected_size) if image.size != expected_size else image
    if page_index != 0:
        raise ValueError("Ảnh đơn không có trang thứ hai.")
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB").copy()
    if image.size != expected_size:
        raise ValueError(f"Ảnh {image.size} khác khung evidence {expected_size}; dừng để tránh cắt sai tọa độ.")
    return image


def rerecognize_document(source, evidence, recognizer, *, reader_metadata=None, crop_dir=None):
    """Giữ ID/polygon của khung gốc, tạo text và confidence VietOCR có nguồn riêng."""
    from ai.kie.v2.evidence import flatten

    flatten(evidence)  # Kiểm tra contract và danh tính của evidence trước khi tạo variant.
    started = time.perf_counter()
    result = copy.deepcopy(evidence)
    parent_run = evidence["ocr_run_id"]
    run_id = str(uuid4())
    result["ocr_run_id"] = run_id
    pages = result["pages"] if result.get("schema_version") == "document-2.0" else [{"page_index": 0, "evidence": result}]
    page_metadata = (reader_metadata or {}).get("pages", [])
    changes, omitted, preserved = [], [], []
    if crop_dir:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)
    for page in pages:
        index, data = page["page_index"], page["evidence"]
        data["ocr_run_id"] = run_id
        if data["engine"]["name"] == "pdfium-text":
            preserved.append(index)  # PDF text chuẩn không cần ép qua model ảnh.
            continue
        size = (data["image"]["width_px"], data["image"]["height_px"])
        image = read_page_image(source, index, size)
        angle = page_metadata[index].get("orientation_clockwise", 0) if index < len(page_metadata) else 0
        if angle not in (0, 90, 180, 270):
            raise ValueError("Góc xoay Reader không hợp lệ.")
        new_blocks = []
        for block in data["blocks"]:
            crop = crop_polygon(image, block["polygon"], angle)
            filename = f"p{index}-{block['reading_order']:06d}.png"
            if crop_dir:
                crop.save(Path(crop_dir) / filename)
            prediction = recognizer.recognize_crop(crop)
            text = prediction["text"]
            text = "".join(c for c in text if c in "\t\r\n" or ord(c) >= 32).strip()
            record = {
                "page": index, "block_id": block["block_id"], "polygon": block["polygon"],
                "paddle_text": block["text"], "vietocr_text": text,
                "paddle_confidence": block["confidence"], "vietocr_model_probability": prediction["model_probability"],
                "different": text != block["text"], "review_required": text != block["text"],
                "decode_hit_length_limit": prediction.get("decode_hit_length_limit", False),
                "protected_value_disagreement": text != block["text"] and bool(re.search(r"[0-9@#*&+/_=]", block["text"] + text)),
                "crop": filename if crop_dir else None,
            }
            changes.append(record)
            if not text:
                omitted.append(block["block_id"])  # Không điền lại chữ Paddle rồi gắn nhãn như VietOCR đã đọc được.
                record["status"] = "EMPTY_PREDICTION_OMITTED"
                continue
            block["text"] = text
            block["confidence"] = prediction["model_probability"] if prediction["model_probability"] is not None else 0.0
            block["reading_order"] = len(new_blocks)  # Chỉ sửa thứ tự nếu có dòng rỗng bị bỏ; ID nguồn vẫn giữ.
            new_blocks.append(block)
        data["blocks"] = new_blocks
        data["engine"] = {"name": "paddle-boxes-vietocr-checkpoint", "version": "1.0"}
        data["average_confidence"] = sum(b["confidence"] for b in new_blocks) / len(new_blocks) if new_blocks else 0.0
        data["duration_ms"] = int((time.perf_counter() - started) * 1000)
    flatten(result)  # Variant phải thỏa cùng contract, không tự tạo block ID giả.
    metadata = {
        "parent_ocr_run_id": parent_run, "ocr_run_id": run_id, "checkpoint_sha256": recognizer.sha256,
        "recognizer": recognizer.metadata["name"], "source_evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        "geometry_policy": "reuse_reader_polygons_in_source_frame; do_not_generate_new_boxes",
        "confidence_semantics": "VietOCR mean token probability, not calibrated against Paddle; never copy Paddle confidence to VietOCR",
        "ocr_blocks_processed": len(changes), "changed_text_blocks": sum(r["different"] for r in changes),
        "omitted_empty_block_ids": omitted, "pdf_text_pages_preserved": preserved,
        "decode_length_limit_block_ids": [r["block_id"] for r in changes if r["decode_hit_length_limit"]],
        "seconds": time.perf_counter() - started, "review_status": "EXPERIMENTAL_PENDING_GROUP_REVIEW",
    }
    return result, metadata, changes
