"""Runnable cross-precision parity audit over a spatial benchmark.

End-to-end wiring of ``vqasynth.precision_parity`` (GHOST-Q's paired
FP16-vs-quantized protocol): answer the same benchmark items twice — once at
a reference precision and once quantized — then compare the two runs
item-by-item instead of reading two aggregate accuracies side by side. The
audit reports outcome flips (grounding successes lost to quantization vs
failures rescued), decision changes, the *same-score tradeoff* flag (net
accuracy held within ±2pp while paired outcomes flipped), and an exact sign
test with Benjamini-Hochberg FDR correction over the per-category breakdown.

The pure audit logic lives in ``vqasynth.precision_parity`` (stdlib-only,
reusing the ``vqasynth.visual_credit`` / ``vqasynth.evaluation`` extractors,
unit-tested in ``tests/test_precision_parity.py``). This script only owns
I/O: loading benchmark items, driving ``vqasynth.inference`` once per
precision, and printing / persisting the report.

Prediction JSONL record shape (one per item, written by ``compare`` and
consumed by ``audit``)::

    {"id": "q042", "prediction": "The chair is about 2 meters away."}

Usage
-----
Run both precisions over a benchmark slice and audit the pair
(CUDA + model download, maintainer-run)::

    python -m experiments.precision_parity_audit.run compare \\
        --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark spatialscore \\
        --quantized-precision nf4 --limit 200 --output-dir parity_out

Audit two prediction JSONLs you already collected (no GPU needed; items come
from the benchmark or an items JSONL)::

    python -m experiments.precision_parity_audit.run audit \\
        --baseline-predictions parity_out/fp16_predictions.jsonl \\
        --quantized-predictions parity_out/nf4_predictions.jsonl \\
        --benchmark spatialscore --breakdown-by category
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vqasynth.benchmarks import BenchmarkRunner
from vqasynth.evaluation import classify_question
from vqasynth.precision_parity import audit as parity_audit
from vqasynth.precision_parity import breakdown_by, format_parity_report

# Breakdown keys supported by the audit subcommand.
_BREAKDOWN_KEYS = ("category", "question_type")


def _read_jsonl(path):
    """Yield JSON objects from a newline-delimited JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(records, path):
    """Write one JSON object per line to ``path`` (creating parent dirs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _predictions_from_jsonl(path):
    """Read a prediction JSONL into the id -> prediction dict the audit takes."""
    predictions = {}
    for record in _read_jsonl(path):
        if "id" in record and "prediction" in record:
            predictions[record["id"]] = record["prediction"]
    return predictions


def _write_report(report, breakdown, path):
    """Persist the report summary (headline metrics + breakdown) as JSON."""
    payload = {
        "paired": report.paired,
        "total": report.total,
        "baseline_accuracy": report.baseline_accuracy,
        "quantized_accuracy": report.quantized_accuracy,
        "net_delta": report.net_delta,
        "agreement": report.agreement,
        "decision_agreement": report.decision_agreement,
        "baseline_only_correct": report.baseline_only_correct,
        "quantized_only_correct": report.quantized_only_correct,
        "flip_rate": report.flip_rate,
        "same_score_tradeoff": report.same_score_tradeoff,
        "symmetry_p": report.symmetry_p,
        "significant_asymmetry": report.significant_asymmetry,
        "breakdown": breakdown,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote parity summary to {path}")


def _select_key_fn(key):
    """Map a --breakdown-by name onto an item key function."""
    if key == "category":
        return lambda item: item.get("category", "")
    if key == "question_type":
        return lambda item: item.get("question_type", "") or classify_question(
            item.get("question", "")
        )
    raise ValueError(f"Unknown breakdown key '{key}'. Valid: {_BREAKDOWN_KEYS}")


def _finish(report, items, args):
    breakdown = None
    if args.breakdown_by:
        breakdown = breakdown_by(items, report, key_fn=_select_key_fn(args.breakdown_by))
    print(format_parity_report(report, breakdown=breakdown))
    if args.output:
        _write_report(report, breakdown, args.output)
    return 0


def compare(args):
    """Run both precisions over a benchmark slice and audit the pair."""
    from vqasynth.inference import run_inference_on_benchmark

    runner = BenchmarkRunner(benchmarks=[args.benchmark])
    items = runner.load(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    print(f"auditing {len(items)} {args.benchmark} items at "
          f"{args.baseline_precision} vs {args.quantized_precision}")

    runs = {}
    for label, precision in (
        ("baseline", args.baseline_precision),
        ("quantized", args.quantized_precision),
    ):
        predictions = run_inference_on_benchmark(
            args.model, items,
            max_new_tokens=args.max_new_tokens,
            precision=precision,
        )
        runs[label] = predictions
        path = os.path.join(args.output_dir, f"{precision}_predictions.jsonl")
        _write_jsonl(
            ({"id": item_id, "prediction": pred} for item_id, pred in predictions.items()),
            path,
        )
        print(f"wrote {len(predictions)} predictions to {path}")

    report = runner.score_parity(
        args.benchmark, runs["baseline"], runs["quantized"], items=items
    )
    return _finish(report, items, args)


def run_audit(args):
    """Audit two already-collected prediction JSONLs."""
    baseline = _predictions_from_jsonl(args.baseline_predictions)
    quantized = _predictions_from_jsonl(args.quantized_predictions)

    runner = BenchmarkRunner(benchmarks=[args.benchmark] if args.benchmark else None)
    if args.benchmark:
        items = runner.load(args.benchmark)
        report = runner.score_parity(args.benchmark, baseline, quantized, items=items)
    else:
        if not args.items:
            raise SystemExit("audit needs --benchmark or --items for the ground truth")
        items = [
            {"id": record.get("id"), "question": record.get("question", ""),
             "answer": record.get("answer", record.get("gold", "")),
             "category": record.get("category", ""),
             "question_type": record.get("question_type", "")}
            for record in _read_jsonl(args.items)
        ]
        report = parity_audit(baseline, quantized, items)

    return _finish(report, items, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    compare_parser = sub.add_parser(
        "compare", help="Run baseline + quantized inference and audit the pair (CUDA)."
    )
    compare_parser.add_argument("--model", required=True, help="HuggingFace model slug.")
    compare_parser.add_argument(
        "--benchmark", required=True,
        help="Benchmark name (spatialscore, omnispatial, space10, mindcube).",
    )
    compare_parser.add_argument(
        "--baseline-precision", default="fp16",
        help="Reference precision (fp16, bf16).",
    )
    compare_parser.add_argument(
        "--quantized-precision", default="int8",
        help="Quantized precision (int8, nf4).",
    )
    compare_parser.add_argument("--limit", type=int, default=None, help="Item cap.")
    compare_parser.add_argument("--max-new-tokens", type=int, default=256)
    compare_parser.add_argument("--output-dir", default="parity_out")
    compare_parser.add_argument("--breakdown-by", choices=_BREAKDOWN_KEYS, default=None)
    compare_parser.add_argument("--output", default=None, help="Report JSON path.")
    compare_parser.set_defaults(func=compare)

    audit_parser = sub.add_parser(
        "audit", help="Audit two prediction JSONLs (no GPU needed)."
    )
    audit_parser.add_argument(
        "--baseline-predictions", required=True,
        help="Prediction JSONL from the reference-precision run.",
    )
    audit_parser.add_argument(
        "--quantized-predictions", required=True,
        help="Prediction JSONL from the quantized run.",
    )
    audit_parser.add_argument(
        "--benchmark", default=None,
        help="Benchmark name; scores pairs with its native scorer.",
    )
    audit_parser.add_argument(
        "--items", default=None,
        help="Items JSONL (id, question, answer) when --benchmark is omitted.",
    )
    audit_parser.add_argument("--breakdown-by", choices=_BREAKDOWN_KEYS, default=None)
    audit_parser.add_argument("--output", default=None, help="Report JSON path.")
    audit_parser.set_defaults(func=run_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
