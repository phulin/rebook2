from __future__ import annotations

import difflib
import json
import re
import statistics
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from rebook2.benchmarking.providers import create_provider
from rebook2.benchmarking.types import PreparedPage


def benchmark_providers(
    provider_names: list[str],
    pages: list[PreparedPage],
    warmup_runs: int,
    repeats: int,
    capture_text: bool,
    output_path: Path,
) -> dict[str, Any]:
    output_bundle_dir = output_path.parent / f"{output_path.stem}-outputs"
    texts_dir = output_bundle_dir / "texts"

    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "providers": provider_names,
            "page_count": len(pages),
            "warmup_runs": warmup_runs,
            "repeats": repeats,
            "capture_text": capture_text,
            "correctness_metrics_enabled": False,
        },
        "artifacts": {
            "report_path": str(output_path),
            "output_bundle_dir": str(output_bundle_dir),
            "texts_dir": str(texts_dir),
        },
        "providers": [],
    }
    page_outputs_by_provider: dict[str, dict[str, dict[str, Any]]] = {}

    for provider_name in provider_names:
        provider_start = time.perf_counter()
        provider = create_provider(provider_name)
        startup_seconds = time.perf_counter() - provider_start

        run_rows: list[dict[str, Any]] = []
        page_summaries: list[dict[str, Any]] = []
        inference_timings: list[float] = []
        provider_page_outputs: dict[str, dict[str, Any]] = {}

        try:
            for page in pages:
                for _ in range(warmup_runs):
                    provider.recognize(page.image)

                page_timings: list[float] = []
                last_output_text = ""
                for repeat_index in range(1, repeats + 1):
                    started_at = time.perf_counter()
                    output = provider.recognize(page.image)
                    elapsed_seconds = time.perf_counter() - started_at
                    page_timings.append(elapsed_seconds)
                    inference_timings.append(elapsed_seconds)
                    last_output_text = output.text

                    row = {
                        "sample_id": page.sample_id,
                        "source_path": str(page.source_path),
                        "page_number": page.page_number,
                        "repeat_index": repeat_index,
                        "elapsed_seconds": elapsed_seconds,
                        "text_characters": len(output.text),
                        "scores": {},
                        "ground_truth_available": bool(
                            page.ground_truth_text or page.ground_truth_path
                        ),
                    }
                    if capture_text:
                        row["text"] = output.text
                    if output.metadata:
                        row["provider_metadata"] = output.metadata
                    run_rows.append(row)

                page_key = _page_key(page.sample_id, page.page_number)
                text_path = _text_output_path(
                    texts_dir=texts_dir,
                    provider_name=provider_name,
                    sample_id=page.sample_id,
                    page_number=page.page_number,
                )
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(last_output_text)
                provider_page_outputs[page_key] = {
                    "sample_id": page.sample_id,
                    "source_path": str(page.source_path),
                    "page_number": page.page_number,
                    "text_path": str(text_path),
                    "text_characters": len(last_output_text),
                    "ground_truth_available": bool(
                        page.ground_truth_text or page.ground_truth_path
                    ),
                }

                page_summaries.append(
                    {
                        "sample_id": page.sample_id,
                        "source_path": str(page.source_path),
                        "page_number": page.page_number,
                        "mean_seconds": statistics.mean(page_timings),
                        "median_seconds": statistics.median(page_timings),
                        "min_seconds": min(page_timings),
                        "max_seconds": max(page_timings),
                        "text_characters": len(last_output_text),
                        "text_path": str(text_path),
                        "scores": {},
                    }
                )
        finally:
            provider.close()

        page_outputs_by_provider[provider_name] = provider_page_outputs

        report["providers"].append(
            {
                "provider": provider_name,
                "startup_seconds": startup_seconds,
                "summary": {
                    "total_runs": len(run_rows),
                    "mean_seconds": statistics.mean(inference_timings),
                    "median_seconds": statistics.median(inference_timings),
                    "min_seconds": min(inference_timings),
                    "max_seconds": max(inference_timings),
                },
                "page_summaries": page_summaries,
                "runs": run_rows,
            }
        )

    comparisons = _build_provider_comparisons(page_outputs_by_provider)
    comparisons_path = output_bundle_dir / "comparisons.json"
    comparisons_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons_path.write_text(json.dumps(comparisons, indent=2))
    report["comparisons"] = comparisons
    report["artifacts"]["comparisons_path"] = str(comparisons_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def _page_key(sample_id: str, page_number: int) -> str:
    return f"{sample_id}::page-{page_number}"


def _text_output_path(
    *,
    texts_dir: Path,
    provider_name: str,
    sample_id: str,
    page_number: int,
) -> Path:
    safe_provider = _slugify(provider_name)
    safe_sample = _slugify(sample_id)
    return texts_dir / safe_provider / safe_sample / f"page-{page_number:04d}.txt"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "item"


def _build_provider_comparisons(
    page_outputs_by_provider: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []

    shared_page_keys = sorted(set().union(*(outputs.keys() for outputs in page_outputs_by_provider.values())))
    for page_key in shared_page_keys:
        page_entries: list[tuple[str, dict[str, Any]]] = []
        for provider_name, outputs in page_outputs_by_provider.items():
            if page_key in outputs:
                page_entries.append((provider_name, outputs[page_key]))

        if len(page_entries) < 2:
            continue

        sample_id = page_entries[0][1]["sample_id"]
        source_path = page_entries[0][1]["source_path"]
        page_number = page_entries[0][1]["page_number"]
        pairwise: list[dict[str, Any]] = []

        for (left_name, left_entry), (right_name, right_entry) in combinations(page_entries, 2):
            left_text = Path(left_entry["text_path"]).read_text()
            right_text = Path(right_entry["text_path"]).read_text()
            pairwise.append(
                {
                    "providers": [left_name, right_name],
                    "exact_match": left_text == right_text,
                    "normalized_exact_match": _normalize_text(left_text)
                    == _normalize_text(right_text),
                    "similarity_ratio": round(
                        difflib.SequenceMatcher(a=left_text, b=right_text).ratio(),
                        6,
                    ),
                    "character_count_delta": len(left_text) - len(right_text),
                    "text_paths": [left_entry["text_path"], right_entry["text_path"]],
                }
            )

        mean_similarity = statistics.mean(item["similarity_ratio"] for item in pairwise)
        pages.append(
            {
                "sample_id": sample_id,
                "source_path": source_path,
                "page_number": page_number,
                "providers": [provider_name for provider_name, _ in page_entries],
                "pairwise": pairwise,
                "summary": {
                    "pair_count": len(pairwise),
                    "mean_similarity_ratio": round(mean_similarity, 6),
                    "all_exact_match": all(item["exact_match"] for item in pairwise),
                    "all_normalized_exact_match": all(
                        item["normalized_exact_match"] for item in pairwise
                    ),
                },
            }
        )

    return {
        "pages": pages,
        "summary": {
            "page_count": len(pages),
            "provider_pair_count": sum(len(page["pairwise"]) for page in pages),
            "mean_similarity_ratio": round(
                statistics.mean(
                    item["similarity_ratio"]
                    for page in pages
                    for item in page["pairwise"]
                ),
                6,
            )
            if pages
            else None,
        },
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
