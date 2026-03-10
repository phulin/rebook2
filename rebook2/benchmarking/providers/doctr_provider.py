from __future__ import annotations

from typing import TYPE_CHECKING

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class DocTROCRProvider(OCRProvider):
    name = "doctr"

    def __init__(
        self,
        det_arch: str = "fast_base",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
    ) -> None:
        try:
            import numpy as np
            from doctr.models import ocr_predictor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "python-doctr is not installed. Run `uv sync` before benchmarking this provider."
            ) from exc

        self._np = np
        self._predictor = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=pretrained,
            assume_straight_pages=True,
        )

    def recognize(self, image: Image) -> OCRProviderOutput:
        image_array = self._np.asarray(image.convert("RGB"))
        result = self._predictor([image_array])

        if not result.pages:
            return OCRProviderOutput(
                text="",
                metadata={"blocks": 0, "lines": 0, "words": 0, "language": None},
            )

        page = result.pages[0]
        exported_page = page.export()
        blocks = exported_page["blocks"]
        line_count = sum(len(block["lines"]) for block in blocks)
        word_count = sum(len(line["words"]) for block in blocks for line in block["lines"])

        return OCRProviderOutput(
            text=page.render().strip(),
            metadata={
                "blocks": len(blocks),
                "lines": line_count,
                "words": word_count,
                "language": exported_page.get("language", {}).get("value"),
            },
        )
