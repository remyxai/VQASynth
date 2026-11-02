"""Run a VLM on the AgentVQA agentic visual-understanding benchmark.

End-to-end wiring of ``vqasynth.agentic_vqa``: register the benchmark with the
existing multi-benchmark evaluation stage (``vqasynth.benchmarks``), load
items, collect predictions, and score them through ``BenchmarkRunner`` — so
the report lands in the same shape as the SpatialScore / OmniSpatial / SpaCE-10
/ MindCube reports, with accuracy broken down by agentic domain
(``by_category``) and by source dataset (``by_subcategory``).

Adapted from *AgentVQA: A Unified Benchmark for Agentic Visual Understanding*
(https://arxiv.org/abs/2610.31494). This script only owns I/O; the loader,
scorer and registry wiring live in ``vqasynth.agentic_vqa`` and are unit-tested
in ``tests/test_agentic_vqa.py``.

Usage
-----
Score a HuggingFace VLM on the whole benchmark::

    python -m experiments.agentic_vqa_eval.run \\
        --model remyxai/SpaceThinker-Qwen2.5VL-3B --max-items 200

Only the domains VQASynth's synthetic data targets::

    python -m experiments.agentic_vqa_eval.run \\
        --model <slug> --domains "Spatial Planning,Robotics"

Score predictions you collected yourself (one JSONL record per item, keyed by
the item ids printed by ``--emit-items``)::

    python -m experiments.agentic_vqa_eval.run \\
        --predictions predictions.jsonl --emit-items items.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from vqasynth.agentic_vqa import (
    AGENTVQA_BENCHMARK_NAME,
    AGENTVQA_DOMAINS,
    AGENTVQA_DOMAIN_BY_SUBSET,
    AGENTVQA_SUBSETS,
    register_agentvqa,
    subsets_for_domains,
)
from vqasynth.benchmarks import BenchmarkRunner, format_benchmark_report

PREDICTION_KEYS = ("prediction", "pred", "answer", "output")


def _resolve_subsets(args):
    if args.domains:
        return subsets_for_domains(args.domains.split(","))
    if args.subsets:
        return [s.strip() for s in args.subsets.split(",") if s.strip()]
    return list(AGENTVQA_SUBSETS)


def _read_predictions(path):
    """Read a prediction JSONL into the {item_id: prediction} map score() takes."""
    predictions = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pred = next((record[k] for k in PREDICTION_KEYS if record.get(k)), "")
            predictions[str(record["id"])] = str(pred)
    return predictions


def _write_json(payload, path):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Benchmark report saved to {path}")


def _print_subsets():
    print(f"{'Domain':<20} Subsets")
    print("-" * 70)
    for domain in AGENTVQA_DOMAINS:
        subsets = [s for s in AGENTVQA_SUBSETS if AGENTVQA_DOMAIN_BY_SUBSET[s] == domain]
        print(f"{domain:<20} {', '.join(subsets)}")


def run_eval(args):
    register_agentvqa()

    runner = BenchmarkRunner(AGENTVQA_BENCHMARK_NAME)
    items = runner.load(
        AGENTVQA_BENCHMARK_NAME,
        subsets=_resolve_subsets(args),
        require_image=not args.include_video_items,
    )
    if not items:
        print("No AgentVQA items loaded; check --subsets / --domains.")
        return 1

    if args.max_items > 0:
        items = items[: args.max_items]

    if args.emit_items:
        # Ground truth stripped — hand these ids to whichever harness you like.
        with open(args.emit_items, "w") as f:
            for item in items:
                f.write(json.dumps({
                    "id": item["id"],
                    "question": item["question"],
                    "options": item["options"],
                    "num_images": len(item["images"]),
                    "category": item["category"],
                    "subcategory": item["subcategory"],
                }) + "\n")
        print(f"Wrote {len(items)} item records (no ground truth) to {args.emit_items}")

    if args.predictions:
        predictions = _read_predictions(args.predictions)
        unmatched = sum(1 for item in items if item["id"] not in predictions)
        if unmatched:
            print(f"Warning: {unmatched} loaded items have no prediction; they score 0.")
    else:
        from vqasynth.inference import run_inference_on_benchmark

        print(f"Running inference on {len(items)} AgentVQA items with {args.model}...")
        predictions = run_inference_on_benchmark(
            model_name=args.model,
            benchmark_items=items,
            max_new_tokens=args.max_new_tokens,
        )

    result = runner.score(AGENTVQA_BENCHMARK_NAME, items, predictions)
    report = {
        "summary": {result["benchmark"]: result["overall_accuracy"]},
        "benchmarks": {
            result["benchmark"]: {
                key: value for key, value in result.items() if key != "per_item"
            }
        },
    }
    print(format_benchmark_report(report))

    if args.output:
        _write_json({**report, "per_item": result["per_item"]}, args.output)

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="HuggingFace VLM slug to evaluate.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Prediction JSONL ({\"id\", \"prediction\"}) to score instead of running inference.",
    )
    parser.add_argument(
        "--subsets", default=None, help="Comma-separated source datasets to load (default: all 23)."
    )
    parser.add_argument(
        "--domains",
        default=None,
        help=f"Comma-separated agentic domains to load ({', '.join(AGENTVQA_DOMAINS)})."
    )
    parser.add_argument("--max-items", type=int, default=0, help="Evaluate only the first N items.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument(
        "--include-video-items",
        action="store_true",
        help="Keep items whose visual input is a video file (no embedded still).",
    )
    parser.add_argument(
        "--emit-items",
        default=None,
        help="Write the evaluated items (no ground truth) to a JSONL.",
    )
    parser.add_argument("--output", default=None, help="Path to write the full report JSON.")
    parser.add_argument(
        "--list-subsets", action="store_true", help="List domains/subsets and exit."
    )
    args = parser.parse_args(argv)

    if args.list_subsets:
        _print_subsets()
        return 0

    if not args.predictions and not args.model:
        parser.error("--model is required unless --predictions is given")

    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
