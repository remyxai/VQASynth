"""Prometheus-vision judge dataset builder for SpaceLLaVA / OpenSpaces outputs.

Reformats spatial-VQA rows into the judge record shape expected by
``prometheus-eval/prometheus-vision`` (consumed via ``llava.eval.model_vqa``),
then parses the ``[N]`` scores out of the judge model's feedback text.

The reformat + score-parse logic here is pure-stdlib (no torch / PIL / HF) so
it can be unit-tested with synthetic inputs the way ``tests/test_vggt_speedups``
tests the VGGT wrapper. The end-to-end wiring (loading ``remyxai/OpenSpaces``,
materializing images, plotting the histogram, pushing the scored dataset to the
Hub) lives in ``examples/prometheus_space_judge.py``.

The reference for the record shape and rubric wording is the maintainer's
``prometheus_space_judge`` Colab (issue #28).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

# --- Provenance / rubric constants (mirror the reference notebook) ------------

REFERENCE_ANSWER = (
    "An exemplary response that accurately describes spatial relationships "
    "and distances with precise unit usage and reference accuracy."
)
ORIG_INSTRUCTION = (
    "Describe the spatial relationships and distances between objects in a "
    "given scene, noting the accuracy of distances, units, and spatial "
    "relationships."
)
ORIG_CRITERIA = (
    "Evaluates the model's ability to identify image content accurately, "
    "focusing on spatial awareness and units of distance."
)

# 5-point spatial-reasoning rubric. Shared verbatim between the judge record's
# ``score rubrics`` field and the per-score ``orig_scoreN_description`` fields.
SCORE1 = (
    "The response has significant inaccuracies in distance estimates and "
    "misunderstands spatial relationships."
)
SCORE2 = (
    "The response recognizes basic spatial relationships but struggles with "
    "accurate distance estimation and appropriate unit usage."
)
SCORE3 = (
    "The response correctly estimates distances and understands spatial "
    "relationships, with minor inaccuracies or unit errors."
)
SCORE4 = (
    "The response provides accurate distance estimations and correct spatial "
    "relationships with precise unit usage, minor details notwithstanding."
)
SCORE5 = (
    "The response demonstrates excellent comprehension by accurately and "
    "precisely detailing distances, spatial relationships, and units, aligning "
    "closely with reference measurements or expert evaluations."
)

SCORE_DESCRIPTIONS = {1: SCORE1, 2: SCORE2, 3: SCORE3, 4: SCORE4, 5: SCORE5}

SCORE_RUBRICS = (
    "Evaluates the model's ability to accurately estimate distances, "
    "understand spatial relationships, and use appropriate units of "
    "measurement for distances.\n"
    f"Score 1: {SCORE1}\n"
    f"Score 2: {SCORE2}\n"
    f"Score 3: {SCORE3}\n"
    f"Score 4: {SCORE4}\n"
    f"Score 5: {SCORE5}"
)

# Task description + scoring preamble that prefixes every judge instruction.
JUDGE_PREAMBLE = (
    "###Task Description:\nAn instruction (might include an Input inside it), "
    "a response to evaluate, a reference answer that gets a score of 5, and a "
    "score rubric representing an evaluation criterion is given.\n"
    "1. Write a detailed feedback that assesses the quality of the response "
    "strictly based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 "
    "and 5. You should refer to the score rubric.\n"
    "3. The output format should look as follows: Feedback: (write a feedback "
    "for criteria) [RESULT] (an integer number between 1 and 5)\n"
    "4. Please do not generate any other opening, closing, and explanations.\n"
)

# Marker delimiting the preamble from the per-row question inside ``instruction``.
_INSTRUCTION_MARKER = "###The instruction to evaluate: "

# Matches the judge's ``[N]`` (or ``[N.M]``) score token in feedback text.
_SCORE_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")


# --- Reformat: OpenSpaces row -> Prometheus-vision judge record ---------------

def extract_text(content: Any) -> str:
    """Coerce a message ``content`` field into plain text.

    OpenSpaces rows store ``content`` either as a plain string (assistant
    turns) or as a list of ``{"text": ...}`` fragments (user turns). This
    returns the string itself, or the first non-empty fragment text — mirroring
    the notebook's ``list(set([q['text'] for q in question if q['text']]))[0]``
    extraction but deterministically (``set`` order is not stable across runs).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        seen: list[str] = []
        for fragment in content:
            text = fragment.get("text") if isinstance(fragment, Mapping) else fragment
            if text and str(text) not in seen:
                seen.append(str(text))
        return seen[0] if seen else ""
    return "" if content is None else str(content)


def iter_qa_pairs(messages: Sequence[Mapping[str, Any]]) -> Iterator[tuple[str, str]]:
    """Yield ``(question, response)`` pairs from an OpenSpaces messages list.

    Mirrors the notebook: a ``user`` turn sets the pending question, and the
    following ``assistant`` turn emits the pair (then clears it). An assistant
    turn with no preceding user turn is skipped.
    """
    question: str | None = None
    for message in messages:
        role = message.get("role")
        if role == "user":
            question = extract_text(message.get("content"))
        elif role == "assistant" and question:
            yield question, extract_text(message.get("content"))
            question = None


def build_judge_instruction(question: str, preamble: str = JUDGE_PREAMBLE) -> str:
    """Assemble the judge ``instruction`` = preamble + the user question."""
    return f"{preamble}{_INSTRUCTION_MARKER}{question}"


def build_judge_entry(
    question: str,
    response: str,
    image: str,
    *,
    reference_answer: str = REFERENCE_ANSWER,
    score_rubrics: str = SCORE_RUBRICS,
    orig_instruction: str = ORIG_INSTRUCTION,
    orig_criteria: str = ORIG_CRITERIA,
) -> dict[str, Any]:
    """Build one Prometheus-vision judge record from a single QA pair.

    The result carries the fields ``llava.eval.model_vqa`` expects
    (``image`` / ``instruction`` / ``response to evaluate`` / ``reference
    answer`` / ``score rubrics``) plus ``orig_*`` provenance copies, matching
    the notebook's ``prometheus_data`` dict.
    """
    return {
        "image": image,
        "instruction": build_judge_instruction(question),
        "response to evaluate": response,
        "reference answer": reference_answer,
        "score rubrics": score_rubrics,
        "orig_instruction": orig_instruction,
        "original_response": response,
        "orig_reference_answer": reference_answer,
        "orig_criteria": orig_criteria,
        "orig_score1_description": SCORE1,
        "orig_score2_description": SCORE2,
        "orig_score3_description": SCORE3,
        "orig_score4_description": SCORE4,
        "orig_score5_description": SCORE5,
    }


def reformat_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    image_dir: str = "openspaces",
    image_ext: str = "png",
    limit: int | None = None,
    **entry_kwargs: Any,
) -> list[dict[str, Any]]:
    """Reformat OpenSpaces-style rows into Prometheus-vision judge records.

    Each row must expose a ``messages`` list of ``{role, content}`` turns. Every
    QA pair in a row becomes its own judge record (a row may carry more than
    one). The ``image`` field is set to ``{image_dir}/{row_index}.{image_ext}``;
    the caller (see ``examples/prometheus_space_judge.py``) is responsible for
    materializing those image files from ``row['images']`` so the paths resolve
    for ``llava.eval.model_vqa``. ``limit`` caps the number of input rows.
    """
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        image_path = f"{image_dir}/{index}.{image_ext}"
        for question, response in iter_qa_pairs(row.get("messages", [])):
            entries.append(
                build_judge_entry(question, response, image_path, **entry_kwargs)
            )
    return entries


def write_jsonl(entries: Iterable[Mapping[str, Any]], path: str) -> None:
    """Write records to ``path`` as newline-delimited JSON (overwrites)."""
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


# --- Score parsing: judge feedback -> [N] score ------------------------------

def parse_score(text: str) -> float | None:
    """Extract the first ``[N]`` (or ``[N.M]``) score from judge feedback text.

    Returns ``None`` when no bracketed score is present.
    """
    match = _SCORE_RE.search(text or "")
    return float(match.group(1)) if match else None


def parse_scores(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Parse ``[N]`` scores from judge result records.

    Each record is expected to carry a ``text`` field (the judge feedback) and
    optionally a ``question_id``. Returns a list of
    ``{question_id, score, feedback}`` dicts, skipping records with no score.
    """
    parsed: list[dict[str, Any]] = []
    for record in records:
        feedback = record.get("text", "")
        score = parse_score(feedback)
        if score is None:
            continue
        parsed.append(
            {
                "question_id": record.get("question_id"),
                "score": score,
                "feedback": feedback,
            }
        )
    return parsed


def parse_scores_from_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a judge-results JSONL file and parse ``[N]`` scores from each row."""
    with open(path, "r", encoding="utf-8") as handle:
        records = (json.loads(line) for line in handle if line.strip())
        return parse_scores(records)


def score_distribution(
    scores: Iterable[Mapping[str, Any] | float],
) -> dict[int, int]:
    """Histogram of parsed scores keyed by integer bucket 1..5.

    Accepts either the records returned by :func:`parse_scores` (each with a
    ``score`` key) or a bare iterable of numbers. Each score is bucketed to the
    nearest integer and clamped to the 1..5 rubric range, so the result always
    has all five buckets (zero-filled) for stable plotting downstream.
    """
    counts: Counter[int] = Counter()
    for item in scores:
        value = item["score"] if isinstance(item, Mapping) else item
        bucket = max(1, min(5, int(round(float(value)))))
        counts[bucket] += 1
    return {bucket: counts.get(bucket, 0) for bucket in range(1, 6)}


# --- Match + rebuild: judge input + results -> scored OpenSpaces dataset -----

def match_entries(
    eval_entries: Iterable[Mapping[str, Any]],
    result_entries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Positionally zip judge-input records with judge-result records.

    Mirrors the notebook's matching step: pairs each eval row with the
    corresponding result row by line index (``llava.eval.model_vqa`` preserves
    input order in its answers file), parses the score, and returns
    ``{image, instruction, response_to_evaluate, score, feedback}``. Records
    whose feedback yields no score are dropped.
    """
    matched: list[dict[str, Any]] = []
    for eval_rec, result_rec in zip(eval_entries, result_entries):
        feedback = result_rec.get("text", "")
        score = parse_score(feedback)
        if score is None:
            continue
        matched.append(
            {
                "image": eval_rec.get("image", ""),
                "instruction": eval_rec.get("instruction", ""),
                "response_to_evaluate": eval_rec.get("response to evaluate", ""),
                "score": score,
                "feedback": feedback,
            }
        )
    return matched


def build_request_response_instruction(
    instruction: str, response_to_evaluate: str
) -> str:
    """Rebuild the OpenSpaces-style user prompt from a judge instruction.

    Mirrors the notebook: ``###Request: <question>\\n\\n###Response: <response>``
    where ``<question>`` is the instruction tail that follows the
    ``###The instruction to evaluate:`` marker. If the marker is absent the
    whole instruction is used as the question.
    """
    if _INSTRUCTION_MARKER in instruction:
        question = instruction.split(_INSTRUCTION_MARKER, 1)[1]
    else:
        question = instruction
    return f"###Request: {question}\n\n###Response: {response_to_evaluate}"


def build_scored_dataset(
    matched_records: Iterable[Mapping[str, Any]],
    *,
    image_loader: Callable[[Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenSpaces-format entries from matched judge records.

    Each output entry is ``{"images": <image>, "messages": [user, assistant]}``
    where the assistant turn carries ``[score] feedback`` (mirroring the
    notebook's ``create_dataset_openspaces_format``). ``image_loader`` maps a
    matched record to a PIL image (or any object the caller wants stored); when
    omitted the raw image path string is stored. The example script supplies a
    PIL-opening loader.
    """
    dataset: list[dict[str, Any]] = []
    for record in matched_records:
        image: Any = record.get("image", "")
        if image_loader is not None:
            image = image_loader(record)
        instruction = build_request_response_instruction(
            record.get("instruction", ""), record.get("response_to_evaluate", "")
        )
        response = f"[{record['score']:g}] {record.get('feedback', '')}"
        dataset.append(
            {
                "images": image,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return dataset
