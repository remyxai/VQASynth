"""Runnable Prometheus-vision judge pipeline for SpaceLLaVA / OpenSpaces.

End-to-end wiring of ``vqasynth.judge_dataset``: load an OpenSpaces-style
spatial-VQA dataset, reformat it into Prometheus-vision judge shape, hand the
resulting JSONL to the external ``prometheus-eval/prometheus-vision`` llava
eval, parse the ``[N]`` scores out of the judge output, plot a histogram, and
build a score-matched dataset ready to push to the Hugging Face Hub.

The reformat + score-parse logic lives in ``vqasynth.judge_dataset`` (pure
stdlib, unit-tested). This script only owns I/O: image materialization, JSONL
read/write, matplotlib plotting, and Hub push via ``vqasynth.datasets``.

External eval step (NOT run here — the maintainer runs it externally, per the
brief): install ``prometheus-eval/prometheus-vision`` + ``flash-attn``, fetch
``remyxai/SpaceLLaVA``, then::

    python3 -m llava.eval.model_vqa \
        --model-path /path/to/SpaceLLaVA \
        --question-file ./openspaces/sample_eval_data.jsonl \
        --answers-file ./evaluation_results.jsonl \
        --temperature 1.0 --top_p 0.9 --conv-mode vicuna_v1

Then feed ``--eval`` + ``--results`` to the ``score`` subcommand below.

Usage
-----
Build the judge-input JSONL (+ materialize images)::

    python examples/prometheus_space_judge.py build \
        --dataset remyxai/OpenSpaces --limit 1000 \
        --image-dir openspaces --output openspaces/sample_eval_data.jsonl

Parse scores + build the scored dataset (+ optional histogram / Hub push)::

    python examples/prometheus_space_judge.py score \
        --eval openspaces/sample_eval_data.jsonl \
        --results evaluation_results.jsonl \
        --image-dir openspaces \
        --histogram score_histogram.png \
        --push-to-hub <user>/SpaceJudgeDataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image

# Import the package logic we are wiring up.
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null
from vqasynth.judge_dataset import (
    build_scored_dataset,
    match_entries,
    reformat_dataset,
    score_distribution,
    write_jsonl,
)


def _load_rows(dataset_id: str, split: str, cache_dir: str | None):
    """Load an OpenSpaces-style dataset and yield its rows as plain dicts."""
    loaded = load_dataset(dataset_id, cache_dir=cache_dir)
    target = loaded[split] if split in loaded else loaded[list(loaded.keys())[0]]
    return target


def _materialize_images(rows, image_dir: str, image_ext: str) -> None:
    """Write each row's first image to ``{image_dir}/{index}.{image_ext}``.

    The judge-input JSONL references these same paths, so ``llava.eval.model_vqa``
    can resolve the ``image`` field. Mirrors the notebook's per-row ``image.save``.
    """
    os.makedirs(image_dir, exist_ok=True)
    for index, row in enumerate(rows):
        images = row.get("images")
        if not images:
            continue
        path = os.path.join(image_dir, f"{index}.{image_ext}")
        images[0].save(path)


def _read_jsonl(path: str):
    """Yield JSON objects from a newline-delimited JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _plot_histogram(distribution: dict[int, int], title: str, out_path: str) -> None:
    """Render the score distribution to a PNG (matplotlib is a project dep)."""
    import matplotlib.pyplot as plt

    buckets = list(distribution.keys())
    counts = list(distribution.values())
    total = sum(counts) or 1
    plt.figure()
    plt.bar(buckets, [c / total for c in counts], color="salmon", edgecolor="black")
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Normalized Frequency")
    plt.xticks(buckets)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"score histogram written to {out_path}")


def build(args: argparse.Namespace) -> None:
    """Reformat a spatial-VQA dataset into Prometheus-vision judge JSONL."""
    rows = _load_rows(args.dataset, args.split, args.cache_dir)
    # Drop rows with null images/messages before judging (reuses the repo's
    # existing row filter). Materializing keeps the row indices stable across
    # image materialization and JSONL so paths line up downstream.
    rows = [row for row in rows if filter_null(row)]
    _materialize_images(rows, args.image_dir, args.image_ext)

    entries = reformat_dataset(
        rows,
        image_dir=args.image_dir,
        image_ext=args.image_ext,
        limit=args.limit,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(entries, args.output)
    print(f"wrote {len(entries)} judge records to {args.output}")


def score(args: argparse.Namespace) -> None:
    """Match judge inputs with results, score, plot, and optionally push."""
    eval_entries = list(_read_jsonl(args.eval))
    result_entries = list(_read_jsonl(args.results))
    matched = match_entries(eval_entries, result_entries)
    print(f"matched {len(matched)} scored records")

    distribution = score_distribution(matched)
    print(f"score distribution: {distribution}")
    if args.histogram:
        _plot_histogram(distribution, "Score Distribution of Parsed Data", args.histogram)

    def _load_image(record):
        return Image.open(record["image"]).convert("RGB")

    loader = _load_image if not args.skip_images else None
    scored = build_scored_dataset(matched, image_loader=loader)

    if args.push_to_hub:
        dataset = DatasetDict({"train": Dataset.from_list(scored)})
        Dataloader(args.cache_dir or "").push_to_hub(dataset, args.push_to_hub.split("/")[-1])
        print(f"pushed scored dataset to {args.push_to_hub}")
    elif args.output_dataset:
        with open(args.output_dataset, "w", encoding="utf-8") as handle:
            for entry in scored:
                handle.write(json.dumps(entry) + "\n")
        print(f"wrote scored dataset to {args.output_dataset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Reformat a dataset into judge JSONL.")
    build_parser.add_argument("--dataset", default="remyxai/OpenSpaces")
    build_parser.add_argument("--split", default="train")
    build_parser.add_argument("--cache-dir", default=None)
    build_parser.add_argument("--image-dir", default="openspaces")
    build_parser.add_argument("--image-ext", default="png")
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--output", default="openspaces/sample_eval_data.jsonl")
    build_parser.set_defaults(func=build)

    score_parser = sub.add_parser("score", help="Parse scores + build scored dataset.")
    score_parser.add_argument("--eval", required=True, help="Judge-input JSONL (build output).")
    score_parser.add_argument("--results", required=True, help="llava answers JSONL.")
    score_parser.add_argument("--image-dir", default="openspaces")
    score_parser.add_argument("--skip-images", action="store_true", help="Store image paths instead of opening PIL images.")
    score_parser.add_argument("--histogram", default=None, help="Optional histogram PNG path.")
    score_parser.add_argument("--output-dataset", default=None, help="Optional scored-dataset JSONL path.")
    score_parser.add_argument("--push-to-hub", default=None, help="Optional HF Hub repo id to push to.")
    score_parser.add_argument("--cache-dir", default=None)
    score_parser.set_defaults(func=score)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
