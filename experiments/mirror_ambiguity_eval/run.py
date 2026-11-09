"""Runnable RGB vs RGB-D mirror-ambiguity evaluation over MIRA-Bench items.

End-to-end wiring of ``vqasynth.mirror_ambiguity``: mirrors are the case where
the "depth is faithful" assumption behind the VGGT point-cloud stage breaks — a
reflection keeps the object's appearance but places its depth reading behind the
glass. This script scores a model's answers on the same mirror-scene questions
under two controlled inputs, (a) the RGB image and (b) the RGB image plus its
depth map, and reports the depth delta: whether adding depth helped or hurt.

The pure loader / scorer / delta logic lives in ``vqasynth.mirror_ambiguity``
(reusing the ``vqasynth.evaluation`` extractors, unit-tested in
``tests/test_mirror_ambiguity.py``). Scoring itself goes through the existing
``vqasynth.benchmarks.BenchmarkRunner`` — MIRA-Bench registers into its loader /
scorer registries, so it picks up the LLM-judge fallback and the standard
per-category report like the built-in benchmarks. This script only owns I/O.

Model inference is maintainer-run (the same split as
``experiments/visual_credit_audit``): run the VLM once per item under each
condition and collect the free-text answers into a JSONL, then feed it to
``score``.

Prediction JSONL record shape (one per item)::

    {"id": "q_00042", "pred_rgb": "Yes, the person is reflected.",
     "pred_rgbd": "No, the person is directly observed."}

Usage
-----
Emit the inference items (RGB and RGB-D image sets per item)::

    python -m experiments.mirror_ambiguity_eval.run items \\
        --output mira_items.jsonl --max-items 200

Score collected predictions and print the depth delta::

    python -m experiments.mirror_ambiguity_eval.run score \\
        --predictions mira_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vqasynth.mirror_ambiguity import (
    MIRA_BENCH_NAME,
    depth_delta,
    format_mirror_report,
    gold_option_set,
    load_mira_bench,
    register_mira_bench,
)

# JSONL keys read per modality, and the report labels they score under.
_MODALITY_KEYS = (("mirabench-rgb", "pred_rgb"), ("mirabench-rgbd", "pred_rgbd"))


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


def _load_items(args):
    """Load benchmark items with the loader flags shared by both subcommands."""
    return load_mira_bench(
        questions_json=args.questions,
        data_root=args.data_root,
        categories=args.categories,
        max_items=args.max_items,
    )


def build_items(args):
    """Emit one JSONL record per benchmark item, with the RGB / RGB-D inputs."""
    records = []
    for item in _load_items(args):
        rgb_images = list(item["images"])
        records.append(
            {
                "id": item["id"],
                "question_id": item["question_id"],
                "question": item["question"],
                "choices": item["options"],
                "gold": item["answer"],
                "gold_letters": sorted(gold_option_set(item)),
                "question_type": item["question_type"],
                "mira_question_type": item["mira_question_type"],
                "category": item["category"],
                "level": item["subcategory"],
                "requires_depth": item["requires_depth"],
                "requires_mirror_reasoning": item["requires_mirror_reasoning"],
                # Same question under both conditions; only the inputs differ.
                "images_rgb": rgb_images,
                "images_rgbd": rgb_images + list(item["depth_images"]),
            }
        )

    _write_jsonl(records, args.output)
    with_depth = sum(1 for r in records if len(r["images_rgbd"]) > len(r["images_rgb"]))
    print(f"wrote {len(records)} items to {args.output} ({with_depth} with a depth map)")
    return 0


def _judge_client(args):
    """Optional OpenAI client so BenchmarkRunner can fall back to the LLM judge."""
    if not (args.use_llm_judge and args.api_key):
        return None
    from openai import OpenAI  # deferred: only needed for the judge fallback

    return OpenAI(api_key=args.api_key)


def run_score(args):
    """Score both modalities through BenchmarkRunner and print the depth delta."""
    from vqasynth.benchmarks import (  # deferred: the loader layer pulls in `datasets`
        BenchmarkRunner,
        format_benchmark_report,
    )

    register_mira_bench()
    items = _load_items(args)
    predictions = {str(record.get("id")): record for record in _read_jsonl(args.predictions)}
    items = [item for item in items if item["id"] in predictions]
    if not items:
        raise SystemExit(
            f"no prediction ids in {args.predictions} matched the loaded benchmark items"
        )
    print(f"scoring {len(items)} items answered under both conditions")

    runner = BenchmarkRunner(
        benchmarks=MIRA_BENCH_NAME,
        llm_client=_judge_client(args),
        llm_model=args.judge_model,
    )

    report = {"summary": {}, "benchmarks": {}}
    for label, key in _MODALITY_KEYS:
        preds = {item["id"]: predictions[item["id"]].get(key, "") for item in items}
        result = runner.score(MIRA_BENCH_NAME, items, preds)
        report["benchmarks"][label] = {
            "overall_accuracy": result["overall_accuracy"],
            "total": result["total"],
            "by_category": result["by_category"],
            "by_subcategory": result["by_subcategory"],
        }
        report["summary"][label] = result["overall_accuracy"]

    print(format_benchmark_report(report))
    delta = depth_delta(items, predictions)
    print(format_mirror_report(delta))

    if args.output:
        payload = {
            "benchmark_report": report,
            "depth_delta": {
                "total": delta.total,
                "rgb_accuracy": delta.rgb_accuracy,
                "rgbd_accuracy": delta.rgbd_accuracy,
                "depth_delta": delta.depth_delta,
                "depth_harm_rate": delta.depth_harm_rate,
                "depth_help_rate": delta.depth_help_rate,
                "decision_change_rate": delta.decision_change_rate,
                "by_category": delta.by_category,
                "by_level": delta.by_level,
            },
        }
        _write_jsonl([payload], args.output)
        print(f"wrote mirror-ambiguity summary to {args.output}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_loader_flags(p):
        p.add_argument(
            "--questions", default=None,
            help="Local questions.json (defaults to the HuggingFace-cached copy).",
        )
        p.add_argument(
            "--data-root", default=None,
            help="MIRA Dataset/ root for resolving relative image paths.",
        )
        p.add_argument(
            "--categories", default=None,
            help="Comma-separated MIRA subtasks to keep (default: all).",
        )
        p.add_argument(
            "--max-items", type=int, default=0,
            help="Cap items (0 = no cap). Useful for smoke tests.",
        )

    items_parser = sub.add_parser("items", help="Emit inference items (RGB / RGB-D).")
    add_loader_flags(items_parser)
    items_parser.add_argument("--output", default="mira_items.jsonl", help="Items JSONL.")
    items_parser.set_defaults(func=build_items)

    score_parser = sub.add_parser("score", help="Score predictions + depth delta.")
    add_loader_flags(score_parser)
    score_parser.add_argument(
        "--predictions", required=True,
        help="Prediction JSONL (see module docstring).",
    )
    score_parser.add_argument("--output", default=None, help="Summary JSONL path.")
    score_parser.add_argument("--api-key", default="", help="OpenAI API key for the judge.")
    score_parser.add_argument("--judge-model", default="gpt-4o", help="LLM judge model.")
    score_parser.add_argument(
        "--use-llm-judge", action="store_true",
        help="Enable the LLM judge fallback for unextractable answers.",
    )
    score_parser.set_defaults(func=run_score)

    args = parser.parse_args(argv)
    args.categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else None
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
