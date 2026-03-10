from __future__ import annotations

import unittest
from PIL import Image

from extract_pdf_images import composite_page_images, normalize_rendered_image


class ExtractPdfImagesTests(unittest.TestCase):
    def test_normalize_rendered_image_flattens_transparency_to_white(self) -> None:
        image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        normalized = normalize_rendered_image(image)

        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.getpixel((0, 0)), (255, 255, 255))

    def test_composite_page_images_places_image_using_page_coordinates(self) -> None:
        page = FakePage((10, 10))
        image = FakePdfImage(Image.new("RGBA", (2, 2), (255, 0, 0, 255)), (2, 5, 4, 7))

        composite = composite_page_images(page, [image])

        self.assertEqual(composite.size, (10, 10))
        self.assertEqual(composite.getpixel((2, 3)), (255, 0, 0))
        self.assertEqual(composite.getpixel((0, 0)), (255, 255, 255))

    def test_composite_page_images_overlays_later_images_on_top(self) -> None:
        page = FakePage((6, 6))
        bottom = FakePdfImage(Image.new("RGBA", (4, 4), (255, 0, 0, 255)), (1, 1, 5, 5))
        top = FakePdfImage(Image.new("RGBA", (2, 2), (0, 0, 255, 128)), (2, 2, 4, 4))

        composite = composite_page_images(page, [bottom, top])

        self.assertEqual(composite.getpixel((2, 2)), (127, 0, 128))


class FakeBitmap:
    def __init__(self, image: Image.Image) -> None:
        self._image = image

    def to_pil(self) -> Image.Image:
        return self._image.copy()

    def close(self) -> None:
        return None


class FakePdfImage:
    def __init__(self, image: Image.Image, bounds: tuple[float, float, float, float]) -> None:
        self._image = image
        self._bounds = bounds

    def get_bitmap(self, render: bool = False, scale_to_original: bool = True) -> FakeBitmap:
        return FakeBitmap(self._image)

    def get_bounds(self) -> tuple[float, float, float, float]:
        return self._bounds


class FakePage:
    def __init__(self, size: tuple[float, float]) -> None:
        self.size = size

    def get_size(self) -> tuple[float, float]:
        return self.size


if __name__ == "__main__":
    unittest.main()
