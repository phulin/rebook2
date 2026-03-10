from __future__ import annotations

import time
from os import getenv
from typing import TYPE_CHECKING

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class MockOCRProvider(OCRProvider):
    name = "mock"

    def __init__(self) -> None:
        self.delay_seconds = float(getenv("REBOOK2_MOCK_OCR_DELAY_SECONDS", "0"))

    def recognize(self, image: Image) -> OCRProviderOutput:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return OCRProviderOutput(
            text=f"mock:{image.width}x{image.height}",
            metadata={"delay_seconds": self.delay_seconds},
        )
