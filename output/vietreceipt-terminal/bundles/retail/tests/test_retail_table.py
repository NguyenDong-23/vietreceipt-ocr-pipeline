"""Focused synthetic checks for the opt-in retail table handoff contract."""

import copy
import unittest

from ai.kie.retail_table import extract_retail_table, validate_retail_table
from tests.kie_v2_fixtures import evidence


class RetailTableTests(unittest.TestCase):
    def extract(self, entries):
        source = evidence(entries)
        return source, extract_retail_table(source, currency="VND")

    def test_merged_header_registered_price_and_multiline_product(self):
        source, result = self.extract([
            ("HÓA ĐƠN BÁN HÀNG", .20, .10, .55),
            ("Tên hàng", .03, .20),
            ("SL DVT Đơn giá Thành tiền", .23, .20, .66),
            ("8934755010156 BVS diana mỏng cánh NT", .03, .28, .65),
            ("Giá đăng ký:", .03, .34), ("17,000", .52, .34),
            ("Giá bán:", .03, .40), ("6", .24, .40),
            ("GOI", .32, .40), ("16,500", .52, .40), ("99,000", .77, .40),
            ("291038530000 con thú đồ chơi NO", .03, .48, .65),
            ("1", .24, .54), ("CAI", .32, .54),
            ("14,000", .52, .54), ("14,000", .77, .54),
            ("Tổng cộng", .03, .63), ("113,000", .77, .63),
        ])
        rows = result["line_items"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["description"]["predicted_value"], "BVS diana mỏng cánh")
        self.assertEqual(rows[0]["quantity"]["normalized_value"], 6)
        self.assertEqual(rows[0]["unit_price"]["normalized_value"], 16500)
        self.assertEqual(rows[0]["amount"]["normalized_value"], 99000)
        self.assertEqual(rows[1]["amount"]["normalized_value"], 14000)
        self.assertTrue(rows[0]["machine_needs_review"])
        validate_retail_table(result, source)

    def test_discount_is_signed_and_footer_is_not_an_item(self):
        _, result = self.extract([
            ("HÓA ĐƠN BÁN HÀNG", .20, .10, .55),
            ("Mặt hàng", .08, .20), ("Đơn giá", .48, .20),
            ("SL", .70, .20), ("T.Tiền", .84, .20),
            ("Đùi tỏi gà CP", .08, .27),
            ("1155151000000", .08, .31), ("51.900", .48, .31),
            ("1", .70, .31), ("51.900", .84, .31),
            ("Coupon giảm 50%", .08, .36),
            ("2200000005878", .08, .40), ("-25.950", .48, .40),
            ("1", .70, .40), ("-25.950", .84, .40),
            ("TỔNG TIỀN PHẢI T.TOÁN", .08, .48), ("25.950", .84, .48),
        ])
        rows = result["line_items"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["amount"]["normalized_value"], -25950)
        self.assertEqual(rows[1]["unit_price"]["normalized_value"], -25950)

    def test_no_header_one_amount_and_uncertain_ocr_number(self):
        _, result = self.extract([
            ("PHIẾU THANH TOÁN", .20, .10),
            ("Số HĐ: 123", .20, .18),
            ("Olong Milk Tea Cold (L)", .08, .32, .45),
            ("1", .63, .32), ("50,000", .82, .32),
            ("TỔNG TIỀN", .08, .39), ("50,000", .82, .39),
        ])
        self.assertEqual(len(result["line_items"]), 1)
        self.assertEqual(result["line_items"][0]["unit_price"]["value_status"], "NOT_PRESENT")
        self.assertEqual(result["line_items"][0]["amount"]["normalized_value"], 50000)

        _, result = self.extract([
            ("Tên hàng", .03, .20), ("SL DVT Đơn giá Thành tiền", .23, .20, .66),
            ("BVS diana", .03, .28),
            ("6", .24, .34), ("GOI", .32, .34),
            ("16,500", .52, .34), ("99,00", .77, .34),
            ("Tổng cộng", .03, .42), ("99,000", .77, .42),
        ])
        row = result["line_items"][0]
        self.assertEqual(row["unit_price"]["normalized_value"], 16500)
        self.assertEqual(row["amount"]["predicted_value"], "99,00")
        self.assertIsNone(row["amount"]["normalized_value"])

    def test_validator_rejects_changed_source_and_currency_is_explicit(self):
        source, result = self.extract([
            ("Trà sữa", .08, .32), ("50,000", .82, .32),
            ("Tổng", .08, .39), ("50,000", .82, .39),
        ])
        corrupted = copy.deepcopy(result)
        corrupted["line_items"][0]["amount"]["raw_text"] = "invented"
        with self.assertRaises(ValueError):
            validate_retail_table(corrupted, source)
        with self.assertRaises(ValueError):
            extract_retail_table(source, currency="USD")


if __name__ == "__main__":
    unittest.main()
