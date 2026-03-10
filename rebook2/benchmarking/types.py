from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentSample:
    id: str
    path: Path
    pages: tuple[int, ...] | None = None
    ground_truth_text: str | None = None
    ground_truth_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedPage:
    sample_id: str
    source_path: Path
    page_number: int
    image: Any
    ground_truth_text: str | None = None
    ground_truth_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRProviderOutput:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

