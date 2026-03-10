from __future__ import annotations

import unittest

from PIL import Image

from rebook2.benchmarking.document import _normalize_image_for_ocr


class DocumentPreparationTests(unittest.TestCase):
    def test_normalize_image_for_ocr_flattens_transparency_to_white(self) -> None:
        image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        normalized = _normalize_image_for_ocr(image)

        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
