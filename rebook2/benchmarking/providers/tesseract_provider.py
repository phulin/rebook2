from __future__ import annotations

import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class TesseractOCRProvider(OCRProvider):
    name = "tesseract"

    def __init__(self, language_code: str = "eng", page_segmentation_mode: int = 3) -> None:
        self.language_code = language_code
        self.page_segmentation_mode = page_segmentation_mode
        self.tesseract_path = shutil.which("tesseract")
        if not self.tesseract_path:
            raise RuntimeError(
                "The `tesseract` binary is not available on PATH. Install it before "
                "benchmarking the tesseract provider."
            )

    def recognize(self, image: Image) -> OCRProviderOutput:
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            image.save(temp_file.name, format="PNG")
            completed = subprocess.run(
                [
                    self.tesseract_path,
                    temp_file.name,
                    "stdout",
                    "--psm",
                    str(self.page_segmentation_mode),
                    "-l",
                    self.language_code,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Tesseract exited with status {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        text = completed.stdout.strip()
        return OCRProviderOutput(
            text=text,
            metadata={
                "binary_path": self.tesseract_path,
                "language_code": self.language_code,
                "page_segmentation_mode": self.page_segmentation_mode,
            },
        )
