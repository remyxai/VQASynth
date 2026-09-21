"""Runnable Claim Grounding audit over generated CoT object claims.

End-to-end wiring of ``vqasynth.claim_grounding``: given the reasoning traces
the ``r1_reasoning`` stage writes (the ``output`` column) plus the localize
stage's ``captions`` inventory, extract the object claims each trace makes,
issue paraphrased pointing queries for them, and verify every claim by
semantic-spatial agreement — the share of paraphrases that find the object at
all (semantic support), fused via a geometric mean with the agreement of their
localizations on one image region (QIRV persistence / overlap / dominance).
Claims below threshold are the hallucination suspects a curation step would
flag or drop before the trace is used as training data.

Model inference is maintainer-run (the same split as
``experiments/visual_credit_audit``): run the pointing VLM (Molmo) once per
emitted query and collect its raw ``<point ...>`` responses into a JSONL, then
feed it to the ``audit`` subcommand. Responses are parsed with the existing
``vqasynth.localize.extract_points_and_descriptions`` (pixel coordinates);
records may instead carry pre-parsed normalized ``points`` when localize's
heavy deps (sam2/transformers) are unavailable.

Claims JSONL record shape (one per reasoning trace)::

    {"id": "row-12",
     "output": "... the man in the red hat is 2-3 feet from the pallet ...",
     "captions": ["man wearing a red hat", "stack of wooden pallets"]}

``output`` (or ``reasoning``) is the r1_reasoning stage's CoT column;
``captions`` is the localize stage's object inventory for the image and is
optional. A record may instead carry an explicit ``claim`` to target a single
mention by hand.

Prediction JSONL record shape (one per pointing query)::

    {"id": "row-12", "claim": "red hat", "query_index": 0,
     "response": "<point x=\"30\" y=\"40\" alt=\"red hat\">",
     "image_w": 640, "image_h": 480}

Usage
-----
Emit the paraphrased pointing queries for every claim in a traces file::

    python -m experiments.claim_grounding_audit.run queries \\
        --claims traces.jsonl --output pointing_queries.jsonl

Audit collected pointing responses and flag ungrounded claims::

    python -m experiments.claim_grounding_audit.run audit \\
        --predictions predictions.jsonl --output grounded.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vqasynth.claim_grounding import (
    DEFAULT_THRESHOLD,
    extract_object_claims,
    build_pointing_queries,
    format_grounding_report,
    grounding_report,
    verify_claim,
)

# Instruction prefix mirroring vqasynth.localize.MolmoCaptionLocalizer.run's
# prompt, so the responses parse with the pipeline's own <point> parser.
POINT_PROMPT_PREFIX = (
    "You are an AI assistant that localizes objects in an image. "
    "Answer with a list of <point> elements in this format: "
    '<point x="X" y="Y" alt="Object description"/>. '
    "Use normalized coordinates from 0 to 100. "
    "Only provide valid points in the specified format. "
)

# Columns the claims reader accepts for the reasoning trace.
_TEXT_KEYS = ("output", "reasoning", "text")


def build_pointing_prompt(query):
    """Wrap one paraphrased query in the localize stage's pointing format."""
    return POINT_PROMPT_PREFIX + query


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


def _molmo_parser():
    """extract_points_and_descriptions, or None if localize's deps are absent.

    localize imports sam2 + transformers + accelerate at module top, which may
    not all be present where the audit runs; callers then need pre-parsed
    ``points`` on their prediction records.
    """
    try:
        from vqasynth.localize import extract_points_and_descriptions
    except Exception as exc:  # ImportError or a transitive failure
        print(f"note: vqasynth.localize not importable ({exc}); "
              "prediction records must carry pre-parsed 'points'", file=sys.stderr)
        return None
    return extract_points_and_descriptions


def _claims_from_records(records):
    """Yield ``(record_id, claim, in_inventory)`` triples from a claims JSONL.

    Records with an explicit ``claim`` bypass extraction; otherwise claims are
    lifted from the reasoning text (``output``/``reasoning``/``text``) by
    ``vqasynth.claim_grounding.extract_object_claims`` and matched against the
    record's optional ``captions`` inventory.
    """
    for index, record in enumerate(records):
        record_id = record.get("id", index)
        if record.get("claim"):
            yield record_id, str(record["claim"]), bool(record.get("in_inventory", False))
            continue
        text = next((record[key] for key in _TEXT_KEYS if record.get(key)), None)
        if not text:
            raise ValueError(
                f"claims record {index} has no reasoning text "
                f"(expected one of {_TEXT_KEYS}) and no explicit 'claim'"
            )
        for object_claim in extract_object_claims(text, vocabulary=record.get("captions")):
            yield record_id, object_claim.claim, object_claim.in_inventory


def _points_from_record(record, parser, index):
    """One prediction record -> its localized points (see module docstring)."""
    if record.get("points") is not None:
        return [tuple(point) for point in record["points"]]
    if parser is None or not record.get("response"):
        raise ValueError(
            f"prediction record {index} needs either pre-parsed 'points' or a "
            "'response' to parse (id={record.get('id')!r}, "
            f"claim={record.get('claim')!r})"
        )
    width, height = record.get("image_w"), record.get("image_h")
    if not width or not height:
        raise ValueError(
            f"prediction record {index} with a raw 'response' must also carry "
            "'image_w' and 'image_h'"
        )
    parsed = parser(record["response"], int(width), int(height))
    return [entry["points"] for entry in parsed]


def queries(args):
    """Extract claims from a traces JSONL and emit the pointing queries."""
    emitted = []
    for record_id, claim, in_inventory in _claims_from_records(_read_jsonl(args.claims)):
        for query_index, query in enumerate(build_pointing_queries(claim)):
            emitted.append(
                {
                    "id": record_id,
                    "claim": claim,
                    "in_inventory": in_inventory,
                    "query_index": query_index,
                    "query": query,
                    "prompt": build_pointing_prompt(query),
                }
            )
    _write_jsonl(emitted, args.output)
    print(f"wrote {len(emitted)} pointing queries to {args.output}")
    return 0


def run_audit(args):
    """Verify each claim from its collected pointing responses."""
    parser = _molmo_parser()
    grouped = {}
    sizes = {}
    for index, record in enumerate(_read_jsonl(args.predictions)):
        key = (record.get("id"), record.get("claim"))
        grouped.setdefault(key, []).append((index, record))
        if record.get("image_w") and record.get("image_h"):
            sizes[key] = (int(record["image_w"]), int(record["image_h"]))

    results = []
    for (record_id, claim), records in grouped.items():
        records.sort(key=lambda item: item[1].get("query_index", 0))
        query_points = [_points_from_record(record, parser, index) for index, record in records]
        results.append(
            verify_claim(
                claim,
                query_points,
                image_size=sizes.get((record_id, claim)),
                threshold=args.threshold,
                in_inventory=bool(records[0][1].get("in_inventory", False)),
            )
        )

    report = grounding_report(results, threshold=args.threshold)
    print(format_grounding_report(report))

    if args.output:
        records = [dict(vars(result), id=key[0]) for result, key in zip(results, grouped)]
        _write_jsonl(records, args.output)
        print(f"wrote per-claim verification to {args.output}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    queries_parser = sub.add_parser(
        "queries", help="Extract object claims and emit paraphrased pointing queries."
    )
    queries_parser.add_argument(
        "--claims", required=True, help="Traces JSONL (output/reasoning + optional captions)."
    )
    queries_parser.add_argument(
        "--output", default="pointing_queries.jsonl", help="Where to write the queries."
    )
    queries_parser.set_defaults(func=queries)

    audit_parser = sub.add_parser(
        "audit", help="Verify claims from collected pointing responses."
    )
    audit_parser.add_argument(
        "--predictions", required=True, help="Prediction JSONL (see module docstring)."
    )
    audit_parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Verification score at/above which a claim counts as grounded."
    )
    audit_parser.add_argument(
        "--output", default=None, help="Optional per-claim results JSONL with the grounded flag."
    )
    audit_parser.set_defaults(func=run_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
