from __future__ import annotations

from collections.abc import Callable

from rebook2.benchmarking.providers.base import OCRProvider
from rebook2.benchmarking.providers.doctr_provider import DocTROCRProvider
from rebook2.benchmarking.providers.easyocr_provider import EasyOCRProvider
from rebook2.benchmarking.providers.mock_provider import MockOCRProvider
from rebook2.benchmarking.providers.paddleocr_provider import PaddleOCRProvider
from rebook2.benchmarking.providers.tesseract_provider import TesseractOCRProvider

ProviderFactory = Callable[[], OCRProvider]

PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "doctr": DocTROCRProvider,
    "easyocr": EasyOCRProvider,
    "mock": MockOCRProvider,
    "paddleocr": PaddleOCRProvider,
    "tesseract": TesseractOCRProvider,
}

PROVIDER_DESCRIPTIONS: dict[str, str] = {
    "doctr": "docTR image-to-text adapter.",
    "easyocr": "EasyOCR image-to-text adapter.",
    "mock": "Deterministic no-op provider for framework validation.",
    "paddleocr": "PaddleOCR image-to-text adapter.",
    "tesseract": "Tesseract CLI adapter.",
}


def create_provider(name: str) -> OCRProvider:
    try:
        factory = PROVIDER_FACTORIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROVIDER_FACTORIES))
        raise ValueError(f"Unknown provider {name!r}. Known providers: {known}") from exc
    return factory()
