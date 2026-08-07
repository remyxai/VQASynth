"""Runnable Visual Credit Audit over spatial-VQA predictions.

End-to-end wiring of ``vqasynth.visual_credit``: given a model's answers on a
set of spatial yes/no (or multi-choice) questions under (a) the real image and
(b) the no-image controls, compute dependence-credited correctness (D-CC) and
the correct-but-uncredited rate — i.e. how many "right" answers the image
actually earned versus how many the model would have produced with no image at
all.

The pure audit logic lives in ``vqasynth.visual_credit`` (pure stdlib, reusing
the ``vqasynth.evaluation`` extractors, unit-tested in
``tests/test_visual_credit.py``). This script only owns I/O: it reads the
prediction records, builds the audit items, groups credit by question type via
``vqasynth.evaluation.classify_question``, and prints the report.

Model inference is maintainer-run (the same split as
``experiments/prometheus_space_judge``): run the VLM once per item under each
condition (real image, text-only, blank) and collect the free-text answers into
a JSONL, then feed it to the ``audit`` subcommand. Use ``controls`` to emit the
control prompts (and optionally materialize blank images) for an items file.

Prediction JSONL record shape (one per item)::

    {"question": "Is the cup to the left of the book?",
     "gold": "Yes",
     "pred_real": "Yes, the cup is to the left.",
     "pred_text_only": "Yes.",          # optional
     "pred_blank": "No."}               # optional

Any of ``pred_text_only`` / ``pred_blank`` that are present become the
no-image controls; at least one control is required per item.

Usage
-----
Build the control prompts (and optional blank images) for an items JSONL::

    python -m experiments.visual_credit_audit.run controls \\
        --items items.jsonl --image-dir blanks --emit-images

Audit collected predictions, with a per-question-type breakdown::

    python -m experiments.visual_credit_audit.run audit \\
        --predictions predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vqasynth.evaluation import classify_question
from vqasynth.visual_credit import (
    VisualCreditItem,
    audit,
    breakdown_by,
    build_blank_control,
    build_text_only_control,
    format_credit_report,
    make_blank_image,
)

# JSONL keys the audit reads for the no-image controls, in priority order.
_CONTROL_KEYS = ("pred_text_only", "pred_blank")


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


def _items_from_predictions(records):
    """Build VisualCreditItems from prediction JSONL records.

    Each record needs ``question``, ``gold``, ``pred_real`` and at least one of
    the control keys. Raises ValueError if a record has no control prediction.
    """
    items = []
    for index, record in enumerate(records):
        controls = [
            record[key] for key in _CONTROL_KEYS if record.get(key)
        ]
        if not controls:
            raise ValueError(
                f"prediction record {index} has no control prediction "
                f"(expected one of {_CONTROL_KEYS})"
            )
        items.append(
            VisualCreditItem(
                question=record.get("question", ""),
                gold=record.get("gold", ""),
                pred_real=record.get("pred_real", ""),
                pred_controls=controls,
            )
        )
    return items


def controls(args):
    """Emit the text-only / blank control prompts for an items JSONL."""
    records = list(_read_jsonl(args.items))
    emitted = []
    for index, record in enumerate(records):
        question = record.get("question", "")
        row = {
            "id": record.get("id", index),
            "question": question,
            "gold": record.get("gold", record.get("answer", "")),
            "text_only_prompt": build_text_only_control(question),
            "blank_prompt": build_blank_control(question),
        }
        if args.emit_images:
            image_path = os.path.join(args.image_dir, f"{index}.png")
            row["blank_image"] = image_path
        emitted.append(row)

    if args.emit_images:
        os.makedirs(args.image_dir, exist_ok=True)
        for index, _ in enumerate(records):
            make_blank_image().save(os.path.join(args.image_dir, f"{index}.png"))

    _write_jsonl(emitted, args.output)
    print(f"wrote {len(emitted)} control records to {args.output}")


def run_audit(args):
    """Compute D-CC and the correct-but-uncredited rate from predictions."""
    items = _items_from_predictions(_read_jsonl(args.predictions))
    report = audit(items)
    breakdown = (
        breakdown_by(items, report, key_fn=lambda item: classify_question(item.question))
        if args.breakdown
        else None
    )
    print(format_credit_report(report, breakdown=breakdown))

    if args.output:
        payload = {
            "total": report.total,
            "accuracy": report.accuracy,
            "control_accuracy": report.control_accuracy,
            "d_cc": report.d_cc,
            "correct_but_uncredited": report.correct_but_uncredited,
            "uncredited_of_correct": report.uncredited_of_correct,
            "image_gain": report.image_gain,
            "decision_change_rate": report.decision_change_rate,
            "breakdown": breakdown,
        }
        _write_jsonl([payload], args.output)
        print(f"wrote audit summary to {args.output}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    controls_parser = sub.add_parser(
        "controls", help="Emit text-only / blank control prompts for an items file."
    )
    controls_parser.add_argument("--items", required=True, help="Items JSONL (question, gold).")
    controls_parser.add_argument(
        "--image-dir", default="blank_controls", help="Directory for materialized blank images."
    )
    controls_parser.add_argument(
        "--emit-images", action="store_true", help="Materialize solid-color blank PNGs."
    )
    controls_parser.add_argument(
        "--output", default="control_prompts.jsonl", help="Where to write the control prompts."
    )
    controls_parser.set_defaults(func=controls)

    audit_parser = sub.add_parser(
        "audit", help="Compute D-CC / correct-but-uncredited from predictions."
    )
    audit_parser.add_argument(
        "--predictions", required=True, help="Prediction JSONL (see module docstring)."
    )
    audit_parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Break credit down by question type (vqasynth.evaluation.classify_question).",
    )
    audit_parser.add_argument(
        "--output", default=None, help="Optional path to write the audit summary JSON."
    )
    audit_parser.set_defaults(func=run_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
