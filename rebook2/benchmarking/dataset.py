from __future__ import annotations

import json
from pathlib import Path

from rebook2.benchmarking.types import DocumentSample


def load_samples(
    inputs: list[str] | None,
    manifest_path: str | None,
    pages: tuple[int, ...] | None = None,
) -> list[DocumentSample]:
    if manifest_path and inputs:
        raise ValueError("Pass either --manifest or --input, not both.")
    if not manifest_path and not inputs:
        raise ValueError("Pass at least one --input or a --manifest.")

    if manifest_path:
        return _load_manifest(Path(manifest_path))

    assert inputs is not None
    samples: list[DocumentSample] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        samples.append(
            DocumentSample(
                id=path.stem,
                path=path,
                pages=pages,
            )
        )
    return samples


def parse_pages(raw_pages: str | None) -> tuple[int, ...] | None:
    if not raw_pages:
        return None

    page_numbers: list[int] = []
    for part in raw_pages.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {token}")
            page_numbers.extend(range(start, end + 1))
            continue

        page = int(token)
        if page < 1:
            raise ValueError(f"Page numbers are 1-based: {token}")
        page_numbers.append(page)

    deduped = tuple(dict.fromkeys(page_numbers))
    return deduped or None


def _load_manifest(path: Path) -> list[DocumentSample]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON array.")

    samples: list[DocumentSample] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be an object.")

        sample_path = Path(entry["path"]).expanduser()
        if not sample_path.is_absolute():
            sample_path = (path.parent / sample_path).resolve()

        ground_truth_path = entry.get("ground_truth_path")
        resolved_truth_path: Path | None = None
        if ground_truth_path:
            resolved_truth_path = Path(ground_truth_path).expanduser()
            if not resolved_truth_path.is_absolute():
                resolved_truth_path = (path.parent / resolved_truth_path).resolve()

        page_numbers = entry.get("pages")
        samples.append(
            DocumentSample(
                id=entry.get("id", sample_path.stem),
                path=sample_path.resolve(),
                pages=tuple(page_numbers) if page_numbers else None,
                ground_truth_text=entry.get("ground_truth_text"),
                ground_truth_path=resolved_truth_path,
                metadata=entry.get("metadata", {}),
            )
        )
    return samples

