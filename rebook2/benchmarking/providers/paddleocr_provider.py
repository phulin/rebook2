from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(self) -> None:
        try:
            import numpy as np
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "paddleocr is not installed. Run `uv sync` before benchmarking this provider."
            ) from exc

        self._np = np
        self._ocr = PaddleOCR(lang="en")

    def recognize(self, image: Image) -> OCRProviderOutput:
        image_array = self._np.asarray(image)

        if hasattr(self._ocr, "predict"):
            raw = self._ocr.predict(image_array)
        elif hasattr(self._ocr, "ocr"):
            raw = self._ocr.ocr(image_array, cls=False)
        else:
            raise RuntimeError("Unsupported PaddleOCR API: missing `predict` and `ocr`.")

        text_fragments = _collect_text_fragments(raw)
        return OCRProviderOutput(
            text="\n".join(text_fragments),
            metadata={"fragments": len(text_fragments)},
        )


def _collect_text_fragments(node: Any) -> list[str]:
    if node is None:
        return []

    if isinstance(node, str):
        stripped = node.strip()
        return [stripped] if stripped else []

    if isinstance(node, dict):
        fragments: list[str] = []
        for key, value in node.items():
            if key in {"input_path", "model_name"}:
                continue
            fragments.extend(_collect_text_fragments(value))
        return fragments

    if isinstance(node, (list, tuple)):
        fragments: list[str] = []
        for item in node:
            fragments.extend(_collect_text_fragments(item))
        return fragments

    return []
