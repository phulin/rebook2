from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rebook2.benchmarking.runner import benchmark_providers
from rebook2.benchmarking.types import OCRProviderOutput, PreparedPage


class FakeProvider:
    def __init__(self, text_by_page: dict[int, str]) -> None:
        self._text_by_page = text_by_page

    def recognize(self, image: object) -> OCRProviderOutput:
        page_number = getattr(image, "page_number")
        return OCRProviderOutput(text=self._text_by_page[page_number])

    def close(self) -> None:
        return None


class BenchmarkRunnerTests(unittest.TestCase):
    def test_benchmark_saves_text_outputs_and_pairwise_comparisons(self) -> None:
        pages = [
            PreparedPage(
                sample_id="sample-a",
                source_path=Path("/tmp/sample-a.png"),
                page_number=1,
                image=SimpleNamespace(page_number=1),
            ),
            PreparedPage(
                sample_id="sample-a",
                source_path=Path("/tmp/sample-a.png"),
                page_number=2,
                image=SimpleNamespace(page_number=2),
            ),
        ]

        providers = {
            "alpha": FakeProvider({1: "Hello\nworld", 2: "same text"}),
            "beta": FakeProvider({1: "Hello world", 2: "same text"}),
        }

        def create_provider(name: str) -> FakeProvider:
            return providers[name]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "benchmark.json"
            with patch("rebook2.benchmarking.runner.create_provider", side_effect=create_provider):
                report = benchmark_providers(
                    provider_names=["alpha", "beta"],
                    pages=pages,
                    warmup_runs=0,
                    repeats=1,
                    capture_text=False,
                    output_path=output_path,
                )

            self.assertTrue(output_path.exists())

            texts_dir = Path(report["artifacts"]["texts_dir"])
            comparisons_path = Path(report["artifacts"]["comparisons_path"])
            self.assertTrue(texts_dir.exists())
            self.assertTrue(comparisons_path.exists())

            alpha_page_1 = texts_dir / "alpha" / "sample-a" / "page-0001.txt"
            beta_page_1 = texts_dir / "beta" / "sample-a" / "page-0001.txt"
            self.assertEqual(alpha_page_1.read_text(), "Hello\nworld")
            self.assertEqual(beta_page_1.read_text(), "Hello world")

            saved_report = json.loads(output_path.read_text())
            self.assertEqual(saved_report["comparisons"]["summary"]["page_count"], 2)

            comparisons = json.loads(comparisons_path.read_text())
            self.assertEqual(comparisons["summary"]["provider_pair_count"], 2)

            page_1 = next(
                page
                for page in comparisons["pages"]
                if page["sample_id"] == "sample-a" and page["page_number"] == 1
            )
            pair_1 = page_1["pairwise"][0]
            self.assertFalse(pair_1["exact_match"])
            self.assertTrue(pair_1["normalized_exact_match"])

            page_2 = next(
                page
                for page in comparisons["pages"]
                if page["sample_id"] == "sample-a" and page["page_number"] == 2
            )
            pair_2 = page_2["pairwise"][0]
            self.assertTrue(pair_2["exact_match"])
            self.assertTrue(page_2["summary"]["all_exact_match"])


if __name__ == "__main__":
    unittest.main()
