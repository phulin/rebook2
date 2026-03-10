from __future__ import annotations

from typing import TYPE_CHECKING

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class EasyOCRProvider(OCRProvider):
    name = "easyocr"

    def __init__(self, language_codes: tuple[str, ...] = ("en",)) -> None:
        try:
            import easyocr
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "easyocr is not installed. Run `uv sync` before benchmarking this provider."
            ) from exc

        self._np = np
        self._reader = easyocr.Reader(list(language_codes))

    def recognize(self, image: Image) -> OCRProviderOutput:
        results = self._reader.readtext(
            self._np.asarray(image),
            detail=0,
            paragraph=True,
        )
        text = "\n".join(str(fragment) for fragment in results)
        return OCRProviderOutput(text=text, metadata={"fragments": len(results)})
