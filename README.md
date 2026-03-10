# rebook2

Framework for evaluating OCR providers against a shared document set.

The benchmark currently focuses on speed. The data model already includes
placeholders for ground truth and correctness scores so accuracy metrics can be
added without changing the runner contract.

## What it does

- Normalizes PDF and image inputs into per-page images before OCR runs.
- Benchmarks multiple providers against the same prepared pages.
- Separates provider startup time from measured inference time.
- Saves each provider's OCR text output per page under a sibling artifacts
  directory for inspection and diffing.
- Writes a JSON report with per-run timings, page-level summaries, pairwise
  provider comparisons, and empty `scores` fields reserved for future
  correctness metrics.

## Providers

- `mock`: framework validation without OCR dependencies.
- `doctr`: real OCR adapter using `python-doctr`, loaded lazily.
- `easyocr`: real OCR adapter, loaded lazily.
- `paddleocr`: real OCR adapter, loaded lazily.
- `tesseract`: shells out to the local `tesseract` binary.

Provider dependencies are imported only when used. If they are missing, the CLI
fails with a targeted error message.

## Usage

List available providers:

```bash
python -m rebook2.main list-providers
```

Benchmark one or more files directly:

```bash
python -m rebook2.main benchmark \
  --provider mock \
  --provider doctr \
  --provider easyocr \
  --input USREPORTS-2.pdf \
  --pages 1-3 \
  --warmup-runs 1 \
  --repeats 3 \
  --output artifacts/usreports-speed.json
```

That command writes:

- `artifacts/usreports-speed.json`: main report
- `artifacts/usreports-speed-outputs/texts/<provider>/<sample>/page-XXXX.txt`:
  saved OCR output for each provider/page
- `artifacts/usreports-speed-outputs/comparisons.json`: pairwise comparisons
  between provider outputs on the same page

Benchmark from a manifest:

```bash
python -m rebook2.main benchmark \
  --provider mock \
  --manifest examples/ocr-benchmark-manifest.json
```

Validate the framework without OCR dependencies:

```bash
python -m rebook2.main benchmark \
  --provider mock \
  --input examples/mock-sample.ppm \
  --repeats 2 \
  --warmup-runs 0
```

## Manifest format

The manifest is a JSON array. Each entry supports:

```json
[
  {
    "id": "sample-name",
    "path": "relative/or/absolute/document.pdf",
    "pages": [1, 2, 3],
    "ground_truth_text": "optional future correctness text",
    "ground_truth_path": "optional future correctness file",
    "metadata": {
      "dataset": "optional tags"
    }
  }
]
```

## Notes on timing

PDF rendering happens before OCR timing starts so the current benchmark compares
provider recognition speed on a common image representation. If you later want
end-to-end pipeline timing, add a second metric that includes document
preparation.

Provider comparisons currently use provider-to-provider text agreement metrics:
exact match, whitespace-normalized exact match, character-count delta, and a
SequenceMatcher similarity ratio.
