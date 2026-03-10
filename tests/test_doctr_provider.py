from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from rebook2.benchmarking.providers.doctr_provider import DocTROCRProvider


class _FakePage:
    def render(self) -> str:
        return "Hello\nworld"

    def export(self) -> dict[str, object]:
        return {
            "language": {"value": "en", "confidence": 0.99},
            "blocks": [
                {
                    "lines": [
                        {"words": [{"value": "Hello"}, {"value": "world"}]},
                    ],
                }
            ],
        }


class _FakeResult:
    def __init__(self) -> None:
        self.pages = [_FakePage()]


class _FakePredictor:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, pages: object) -> _FakeResult:
        self.calls.append(pages)
        return _FakeResult()


class DocTROCRProviderTests(unittest.TestCase):
    def test_recognize_returns_rendered_text_and_counts(self) -> None:
        predictor = _FakePredictor()

        doctr_module = types.ModuleType("doctr")
        models_module = types.ModuleType("doctr.models")
        models_module.ocr_predictor = lambda **_: predictor

        with patch.dict(
            sys.modules,
            {
                "doctr": doctr_module,
                "doctr.models": models_module,
            },
        ):
            provider = DocTROCRProvider()

        image = Image.new("L", (8, 6), color=255)
        output = provider.recognize(image)

        self.assertEqual(output.text, "Hello\nworld")
        self.assertEqual(
            output.metadata,
            {"blocks": 1, "lines": 1, "words": 2, "language": "en"},
        )
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(predictor.calls[0]), 1)
        self.assertEqual(predictor.calls[0][0].shape, (6, 8, 3))


if __name__ == "__main__":
    unittest.main()
