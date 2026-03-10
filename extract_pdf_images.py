#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from PIL.Image import Image as PILImage


def normalize_rendered_image(image: PILImage) -> PILImage:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    return image.convert("RGB")


def normalize_rendered_image_rgba(image: PILImage) -> PILImage:
    return image.convert("RGBA")


def save_image(image: pdfium.PdfImage, destination: Path, raw: bool) -> None:
    if raw:
        image.extract(destination)
        return

    bitmap = image.get_bitmap(render=True, scale_to_original=True)
    try:
        rendered = normalize_rendered_image(bitmap.to_pil())
        rendered.save(destination.with_suffix(".png"))
    finally:
        bitmap.close()


def composite_page_images(page: pdfium.PdfPage, images: list[pdfium.PdfImage]) -> PILImage:
    page_width, page_height = page.get_size()
    canvas = Image.new(
        "RGBA",
        (max(1, math.ceil(page_width)), max(1, math.ceil(page_height))),
        (255, 255, 255, 255),
    )

    for image in images:
        bitmap = image.get_bitmap(render=True, scale_to_original=False)
        try:
            rendered = normalize_rendered_image_rgba(bitmap.to_pil())
            left, bottom, right, top = image.get_bounds()
            width = max(1, math.ceil(right - left))
            height = max(1, math.ceil(top - bottom))
            positioned = rendered.resize((width, height), Image.Resampling.LANCZOS)
            paste_x = int(round(left))
            paste_y = int(round(page_height - top))
            canvas.alpha_composite(positioned, (paste_x, paste_y))
        finally:
            bitmap.close()

    return canvas.convert("RGB")


def extract_images(pdf_path: Path, output_dir: Path, raw: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(str(pdf_path))
    extracted = 0

    try:
        for page_number in range(len(document)):
            page = document.get_page(page_number)

            try:
                images = list(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))

                if not raw and images:
                    composite = composite_page_images(page, images)
                    composite.save(output_dir / f"page-{page_number + 1:04d}.png")
                    extracted += 1
                    continue

                for image_number, image in enumerate(images, start=1):
                    destination = output_dir / f"page-{page_number + 1:04d}-image-{image_number:03d}"
                    save_image(image, destination, raw=raw)
                    extracted += 1
            finally:
                page.close()
    finally:
        document.close()

    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image objects from a PDF into an output directory."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF.")
    parser.add_argument("output_dir", type=Path, help="Directory for extracted image files.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Write raw embedded image streams when possible instead of rendered PNGs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not pdf_path.is_file():
        raise SystemExit(f"Input PDF not found: {pdf_path}")

    count = extract_images(pdf_path, output_dir, raw=args.raw)
    unit = "raw image(s)" if args.raw else "page composite(s)"
    print(f"Wrote {count} {unit} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
