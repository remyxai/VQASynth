"""Smoke tests for vqasynth.judge_dataset.

Verifies the Prometheus-vision judge reformat + score-parse logic against
synthetic OpenSpaces-shaped inputs — no CUDA, no HF download, no SpaceLLaVA
weights. Mirrors the dependency-free style of tests/test_vggt_speedups.py.
"""
from __future__ import annotations

import json
import pytest

from vqasynth.judge_dataset import (
    JUDGE_PREAMBLE,
    REFERENCE_ANSWER,
    SCORE_DESCRIPTIONS,
    SCORE_RUBRICS,
    build_judge_entry,
    build_judge_instruction,
    build_request_response_instruction,
    build_scored_dataset,
    extract_text,
    iter_qa_pairs,
    match_entries,
    parse_score,
    parse_scores,
    parse_scores_from_jsonl,
    reformat_dataset,
    score_distribution,
    write_jsonl,
)

QUESTION = "How far is the [A] from the [B]?"
RESPONSE = "Approximately 2 meters apart."
INSTRUCTION_MARKER = "###The instruction to evaluate: "


def _openspaces_row(question=QUESTION, response=RESPONSE, extra_pairs=()):
    """Build a synthetic OpenSpaces row: user content is a list of text
    fragments (as in the real dataset); assistant content is a plain string."""
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    messages.append({"role": "assistant", "content": response})
    for q, a in extra_pairs:
        messages.append({"role": "user", "content": [{"text": q}]})
        messages.append({"role": "assistant", "content": a})
    return {"images": [object()], "messages": messages}


# --- content coercion -------------------------------------------------------

def test_extract_text_passthrough_for_string():
    assert extract_text("plain string") == "plain string"


def test_extract_text_takes_first_fragment_from_list():
    assert extract_text([{"text": "first"}, {"text": "second"}]) == "first"


def test_extract_text_skips_empty_and_dedupes_deterministically():
    # Notebook used list(set(...))[0] (order-unstable); we dedupe stably.
    assert extract_text([{"text": ""}, {"text": "keep"}, {"text": "keep"}]) == "keep"


def test_extract_text_handles_missing_payloads():
    assert extract_text(None) == ""
    assert extract_text([]) == ""
    assert extract_text([{"no_text": True}]) == ""


# --- message pairing --------------------------------------------------------

def test_iter_qa_pairs_emits_user_then_assistant():
    pairs = list(iter_qa_pairs(_openspaces_row().get("messages")))
    assert pairs == [(QUESTION, RESPONSE)]


def test_iter_qa_pairs_supports_multiple_pairs_per_row():
    row = _openspaces_row(extra_pairs=[("second q", "second a")])
    assert list(iter_qa_pairs(row["messages"])) == [
        (QUESTION, RESPONSE),
        ("second q", "second a"),
    ]


def test_iter_qa_pairs_skips_assistant_without_preceding_user():
    messages = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": [{"text": "q"}]},
        {"role": "assistant", "content": "a"},
    ]
    assert list(iter_qa_pairs(messages)) == [("q", "a")]


# --- instruction + entry shape ---------------------------------------------

def test_build_judge_instruction_appends_question_after_marker():
    instruction = build_judge_instruction(QUESTION)
    assert instruction.startswith(JUDGE_PREAMBLE)
    assert instruction.endswith(INSTRUCTION_MARKER + QUESTION)


def test_build_judge_entry_has_required_judge_fields():
    entry = build_judge_entry(QUESTION, RESPONSE, "openspaces/0.png")
    assert entry["image"] == "openspaces/0.png"
    assert entry["instruction"].endswith(INSTRUCTION_MARKER + QUESTION)
    assert entry["response to evaluate"] == RESPONSE
    assert entry["reference answer"] == REFERENCE_ANSWER
    assert entry["score rubrics"] == SCORE_RUBRICS


def test_build_judge_entry_carries_provenance_fields():
    entry = build_judge_entry(QUESTION, RESPONSE, "openspaces/0.png")
    for key in (
        "orig_instruction",
        "original_response",
        "orig_reference_answer",
        "orig_criteria",
    ):
        assert key in entry
    # Per-score provenance descriptions match the shared rubric wording.
    for n, desc in SCORE_DESCRIPTIONS.items():
        assert entry[f"orig_score{n}_description"] == desc
        assert f"Score {n}: {desc}" in entry["score rubrics"]


def test_build_judge_entry_serializes_to_jsonl_row():
    # llava.eval.model_vqa reads one JSON object per line.
    entry = build_judge_entry(QUESTION, RESPONSE, "openspaces/0.png")
    encoded = json.loads(json.dumps(entry))
    assert encoded == entry


# --- reformat ---------------------------------------------------------------

def test_reformat_dataset_templates_image_path_per_row():
    rows = [_openspaces_row()]
    entries = reformat_dataset(rows, image_dir="imgs", image_ext="jpg")
    assert len(entries) == 1
    assert entries[0]["image"] == "imgs/0.jpg"


def test_reformat_dataset_emits_one_entry_per_qa_pair():
    rows = [_openspaces_row(extra_pairs=[("q2", "a2")])]
    entries = reformat_dataset(rows)
    assert len(entries) == 2
    assert entries[1]["response to evaluate"] == "a2"


def test_reformat_dataset_respects_limit():
    rows = [_openspaces_row(), _openspaces_row(), _openspaces_row()]
    assert len(reformat_dataset(rows, limit=2)) == 2


def test_reformat_dataset_indexed_image_paths_are_unique():
    rows = [_openspaces_row(), _openspaces_row()]
    entries = reformat_dataset(rows, image_dir="openspaces")
    assert {e["image"] for e in entries} == {"openspaces/0.png", "openspaces/1.png"}


# --- score parsing ----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Feedback: good. [RESULT] 4", None),  # bare integer, not bracketed
        ("Feedback: good. [4]", 4.0),
        ("The answer [4.5] is fine", 4.5),
        ("[5]", 5.0),
        ("no score here", None),
        ("", None),
    ],
)
def test_parse_score_extracts_bracketed_value(text, expected):
    assert parse_score(text) == expected


def test_parse_score_takes_first_bracketed_value():
    assert parse_score("first [3] then [5]") == 3.0


def test_parse_scores_carries_question_id_and_skips_unscored():
    records = [
        {"question_id": 0, "text": "[4] great"},
        {"question_id": 1, "text": "no score"},
        {"question_id": 2, "text": "[2] weak"},
    ]
    parsed = parse_scores(records)
    assert parsed == [
        {"question_id": 0, "score": 4.0, "feedback": "[4] great"},
        {"question_id": 2, "score": 2.0, "feedback": "[2] weak"},
    ]


def test_write_jsonl_and_parse_scores_round_trip(tmp_path):
    results = [
        {"question_id": 0, "text": "[5] excellent"},
        {"question_id": 1, "text": "[3] ok"},
    ]
    path = tmp_path / "evaluation_results.jsonl"
    write_jsonl(results, str(path))
    parsed = parse_scores_from_jsonl(str(path))
    assert [r["score"] for r in parsed] == [5.0, 3.0]


# --- distribution -----------------------------------------------------------

def test_score_distribution_counts_and_zero_fills_buckets():
    scores = [{"score": 1.0}, {"score": 3.0}, {"score": 3.0}, {"score": 5.0}, {"score": 4.0}]
    assert score_distribution(scores) == {1: 1, 2: 0, 3: 2, 4: 1, 5: 1}


def test_score_distribution_accepts_bare_numbers_and_clamps():
    assert score_distribution([0.3, 5.9, 3]) == {1: 1, 2: 0, 3: 1, 4: 0, 5: 1}


# --- match + rebuild --------------------------------------------------------

def test_match_entries_zips_positionally_and_drops_unscored():
    eval_entries = [
        {"image": "0.png", "instruction": "preamble" + INSTRUCTION_MARKER + "q0",
         "response to evaluate": "a0"},
        {"image": "1.png", "instruction": "preamble" + INSTRUCTION_MARKER + "q1",
         "response to evaluate": "a1"},
    ]
    result_entries = [
        {"text": "[4] good"},
        {"text": "no score"},  # dropped
    ]
    matched = match_entries(eval_entries, result_entries)
    assert len(matched) == 1
    assert matched[0]["image"] == "0.png"
    assert matched[0]["score"] == 4.0


def test_build_request_response_instruction_splits_on_marker():
    instruction = build_judge_instruction(QUESTION)
    out = build_request_response_instruction(instruction, RESPONSE)
    assert out == f"###Request: {QUESTION}\n\n###Response: {RESPONSE}"


def test_build_request_response_instruction_falls_back_without_marker():
    out = build_request_response_instruction("bare instruction", "r")
    assert out == "###Request: bare instruction\n\n###Response: r"


def test_build_scored_dataset_openspaces_shape_with_scored_response():
    matched = [{
        "image": "0.png",
        "instruction": build_judge_instruction(QUESTION),
        "response_to_evaluate": RESPONSE,
        "score": 4.0,
        "feedback": "[4] good",
    }]
    dataset = build_scored_dataset(matched)
    assert len(dataset) == 1
    entry = dataset[0]
    assert entry["images"] == "0.png"  # default: raw path stored
    roles = [m["role"] for m in entry["messages"]]
    assert roles == ["user", "assistant"]
    assert entry["messages"][0]["content"] == f"###Request: {QUESTION}\n\n###Response: {RESPONSE}"
    assert entry["messages"][1]["content"] == "[4] [4] good"


def test_build_scored_dataset_applies_image_loader():
    matched = [{"image": "0.png", "instruction": "p" + INSTRUCTION_MARKER + "q",
                "response_to_evaluate": "r", "score": 5.0, "feedback": "[5]"}]
    calls = []

    def loader(record):
        calls.append(record["image"])
        return f"<img {record['image']}>"

    dataset = build_scored_dataset(matched, image_loader=loader)
    assert dataset[0]["images"] == "<img 0.png>"
    assert calls == ["0.png"]


def test_end_to_end_reformat_then_score_round_trips(tmp_path):
    """Reformat -> write -> (simulate judge) -> match -> scored dataset."""
    rows = [_openspaces_row()]
    entries = reformat_dataset(rows, image_dir=str(tmp_path))
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(entries, str(eval_path))

    # Simulate llava.eval.model_vqa output (one answer row per question row).
    results = [{"question_id": i, "text": f"[5] perfect {i}"} for i in range(len(entries))]
    results_path = tmp_path / "results.jsonl"
    write_jsonl(results, str(results_path))

    matched = match_entries(_read_jsonl(eval_path), _read_jsonl(results_path))
    assert len(matched) == 1
    scored = build_scored_dataset(matched)
    assert scored[0]["messages"][1]["content"].startswith("[5]")


def _read_jsonl(path):
    # Local helper kept out of the module so the test owns its file reading.
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --- integration with a pre-existing vqasynth module ------------------------

def test_reformat_composes_with_filter_null():
    """Integration with vqasynth.utils.filter_null: rows whose values are None
    are dropped before reformatting into judge records, mirroring how the
    example pipeline pre-cleans OpenSpaces rows."""
    pytest.importorskip("torch")  # vqasynth.utils imports torch at module top
    from vqasynth.utils import filter_null

    good = _openspaces_row()
    bad = {"images": None, "messages": None}
    rows = [good, bad]

    kept = [row for row in rows if filter_null(row)]
    assert kept == [good]

    entries = reformat_dataset(kept)
    assert len(entries) == 1
    assert entries[0]["response to evaluate"] == RESPONSE
