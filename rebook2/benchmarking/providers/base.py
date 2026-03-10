from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rebook2.benchmarking.types import OCRProviderOutput

if TYPE_CHECKING:
    from PIL.Image import Image


class OCRProvider(ABC):
    name: str

    @abstractmethod
    def recognize(self, image: Image) -> OCRProviderOutput:
        raise NotImplementedError

    def close(self) -> None:
        return None
