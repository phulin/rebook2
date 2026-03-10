from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rebook2.benchmarking.dataset import load_samples, parse_pages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rebook2.main",
        description="Evaluate OCR providers on a shared document set.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-providers", help="Show supported providers.")
    list_parser.set_defaults(handler=_handle_list_providers)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a speed benchmark against one or more OCR providers.",
    )
    benchmark_parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        required=True,
        help="Provider name. Repeat the flag to benchmark multiple providers.",
    )
    benchmark_parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="Input PDF or image path. Repeat the flag for multiple files.",
    )
    benchmark_parser.add_argument(
        "--manifest",
        help="JSON manifest describing benchmark samples and optional ground truth.",
    )
    benchmark_parser.add_argument(
        "--pages",
        help="1-based PDF pages to render for every --input, for example `1,3,5-7`.",
    )
    benchmark_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup passes per prepared page before measurement.",
    )
    benchmark_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per prepared page.",
    )
    benchmark_parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="PDF rasterization scale applied before OCR timing starts.",
    )
    benchmark_parser.add_argument(
        "--capture-text",
        action="store_true",
        help="Store recognized text in the output report.",
    )
    benchmark_parser.add_argument(
        "--output",
        help="Path for the benchmark JSON report.",
    )
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _handle_list_providers(_: argparse.Namespace) -> int:
    from rebook2.benchmarking.providers import PROVIDER_DESCRIPTIONS

    for provider_name, description in sorted(PROVIDER_DESCRIPTIONS.items()):
        print(f"{provider_name}: {description}")
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    from rebook2.benchmarking.document import prepare_pages
    from rebook2.benchmarking.runner import benchmark_providers

    page_filter = parse_pages(args.pages)
    samples = load_samples(inputs=args.inputs, manifest_path=args.manifest, pages=page_filter)
    pages = prepare_pages(samples, render_scale=args.render_scale)

    output_path = Path(args.output) if args.output else _default_output_path()
    report = benchmark_providers(
        provider_names=args.providers,
        pages=pages,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        capture_text=args.capture_text,
        output_path=output_path,
    )

    print(f"Wrote benchmark report to {output_path}")
    print(f"Saved OCR text outputs to {report['artifacts']['texts_dir']}")
    print(f"Wrote provider comparisons to {report['artifacts']['comparisons_path']}")
    for provider in report["providers"]:
        summary = provider["summary"]
        print(
            f"{provider['provider']}: mean={summary['mean_seconds']:.4f}s "
            f"median={summary['median_seconds']:.4f}s "
            f"min={summary['min_seconds']:.4f}s "
            f"max={summary['max_seconds']:.4f}s "
            f"startup={provider['startup_seconds']:.4f}s"
        )
    comparison_summary = report["comparisons"]["summary"]
    if comparison_summary["page_count"] > 0:
        print(
            "comparisons: "
            f"pages={comparison_summary['page_count']} "
            f"pairs={comparison_summary['provider_pair_count']} "
            f"mean_similarity={comparison_summary['mean_similarity_ratio']:.4f}"
        )
    return 0


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / f"ocr-benchmark-{timestamp}.json"
