from __future__ import annotations

from PIL import Image
from PIL.Image import Image as PILImage

from rebook2.benchmarking.types import DocumentSample, PreparedPage

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def prepare_pages(
    samples: list[DocumentSample],
    render_scale: float,
) -> list[PreparedPage]:
    pages: list[PreparedPage] = []
    for sample in samples:
        suffix = sample.path.suffix.lower()
        if suffix == ".pdf":
            pages.extend(_prepare_pdf(sample, render_scale=render_scale))
            continue
        if suffix in IMAGE_SUFFIXES:
            pages.append(_prepare_image(sample))
            continue
        raise ValueError(f"Unsupported file type: {sample.path}")
    return pages


def _prepare_image(sample: DocumentSample) -> PreparedPage:
    image = _normalize_image_for_ocr(Image.open(sample.path))
    return PreparedPage(
        sample_id=sample.id,
        source_path=sample.path,
        page_number=1,
        image=image,
        ground_truth_text=sample.ground_truth_text,
        ground_truth_path=sample.ground_truth_path,
        metadata=sample.metadata,
    )


def _prepare_pdf(sample: DocumentSample, render_scale: float) -> list[PreparedPage]:
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pypdfium2 is not installed. Run `uv sync` before benchmarking PDFs."
        ) from exc

    document = pdfium.PdfDocument(str(sample.path))
    page_numbers = sample.pages or tuple(range(1, len(document) + 1))
    prepared: list[PreparedPage] = []

    for page_number in page_numbers:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Page {page_number} is out of range for {sample.path} "
                f"(1-{len(document)})."
            )
        page = document[page_number - 1]
        bitmap = page.render(scale=render_scale)
        image = _normalize_image_for_ocr(bitmap.to_pil())
        prepared.append(
            PreparedPage(
                sample_id=sample.id,
                source_path=sample.path,
                page_number=page_number,
                image=image,
                ground_truth_text=sample.ground_truth_text,
                ground_truth_path=sample.ground_truth_path,
                metadata=sample.metadata,
            )
        )

    return prepared


def _normalize_image_for_ocr(image: PILImage) -> PILImage:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    return image.convert("RGB")
