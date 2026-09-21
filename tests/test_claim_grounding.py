"""Structural tests for vqasynth.claim_grounding.

Verifies the semantic-spatial agreement scoring (semantic support, the QIRV
persistence / overlap / dominance components, geometric-mean fusion) against
hand-computed small examples — no CUDA, no model download. Mirrors the
dependency-free style of tests/test_visual_credit.py.

Integration with PRE-EXISTING modules is asserted directly: the audit consumes
the reasoning traces ``vqasynth.r1_reasoning`` produces (its conversation ->
QA-pair extraction is exercised on the messages shape the pipeline writes), and
the pointing responses must parse through ``vqasynth.localize``'s own
``extract_points_and_descriptions`` before reaching the scorer — that
round-trip is exercised under the same import guard tests/test_correspondence.py
uses (localize imports sam2/transformers/accelerate at module top). The
runnable wiring lives in ``experiments.claim_grounding_audit.run`` and is
driven end-to-end through its CLI.
"""
from __future__ import annotations

import json

import pytest

from vqasynth.claim_grounding import (
    DEFAULT_RADIUS,
    POINTING_QUERIES,
    build_pointing_queries,
    claim_in_inventory,
    extract_object_claims,
    format_grounding_report,
    grounding_report,
    qirv,
    semantic_support,
    verify_claim,
)


# --- semantic support --------------------------------------------------------

def test_semantic_support_counts_localizing_queries():
    queries = [[(0.5, 0.5)], [], [(0.4, 0.6)], [], []]
    assert semantic_support(queries) == pytest.approx(2 / 5)
    assert semantic_support([[], []]) == 0.0
    assert semantic_support([]) == 0.0


# --- QIRV: persistence / overlap / dominance ---------------------------------

def _clustered():
    # 5 paraphrases, all pointing inside a 15%-of-frame neighborhood.
    return [[(0.30, 0.40)], [(0.32, 0.39)], [(0.29, 0.41)], [(0.31, 0.40)], [(0.30, 0.42)]]


def _dispersed():
    # 5 paraphrases, each pointing somewhere else entirely.
    return [[(0.1, 0.1)], [(0.9, 0.1)], [(0.1, 0.9)], [(0.9, 0.9)], [(0.5, 0.5)]]


def test_qirv_perfect_cluster_scores_one_everywhere():
    agreement = qirv(_clustered(), radius=DEFAULT_RADIUS)
    assert agreement.persistence == pytest.approx(1.0)
    assert agreement.overlap == pytest.approx(1.0)
    assert agreement.dominance == pytest.approx(1.0)
    assert agreement.score == pytest.approx(1.0)
    assert agreement.modal_point is not None


def test_qirv_dispersed_localizations_collapse():
    # Every query finds something, but never the same region twice: the paper's
    # "dispersed localizations" failure mode.
    agreement = qirv(_dispersed(), radius=DEFAULT_RADIUS)
    assert agreement.persistence == pytest.approx(1 / 5)
    assert agreement.overlap == pytest.approx(0.0)
    assert agreement.dominance == pytest.approx(0.0)   # runner-up == modal
    assert agreement.score < 0.1


def test_qirv_majority_cluster_beats_runner_up():
    # 4 of 5 queries cluster; the 5th localizes elsewhere.
    queries = [[(0.30, 0.40)], [(0.31, 0.40)], [(0.29, 0.41)], [(0.30, 0.40)], [(0.85, 0.85)]]
    agreement = qirv(queries, radius=DEFAULT_RADIUS)
    assert agreement.persistence == pytest.approx(4 / 5)
    # 6 of the 10 query pairs agree (C(4,2) inside the cluster).
    assert agreement.overlap == pytest.approx(6 / 10)
    assert agreement.dominance == pytest.approx((4 - 1) / 4)
    # Product of the three components (AND-semantics), matching reference SSAV.
    assert agreement.score == pytest.approx(0.8 * 0.6 * 0.75)


def test_qirv_empty_inputs_return_zeros():
    empty = qirv([[], [], []])
    assert (empty.persistence, empty.overlap, empty.dominance, empty.score) == (0.0, 0.0, 0.0, 0.0)
    assert qirv([]).score == 0.0


# --- fusion -------------------------------------------------------------------

def test_verify_claim_geometric_mean_fusion():
    clustered = verify_claim("red cup", _clustered())
    assert clustered.score == pytest.approx(1.0)
    assert clustered.grounded is True

    # Either branch lacking support sinks the geometric mean. Here the QIRV
    # product is 0.2 * 0.0 * 0.0 = 0.0 (dispersed => no overlap, no dominance),
    # so the fused score collapses to sqrt(1.0 * 0.0) = 0.0 — the AND-semantics
    # of the reference's product QIRV firing on a hallucinated-but-found claim.
    dispersed = verify_claim("unicorn", _dispersed())
    assert dispersed.semantic_support == pytest.approx(1.0)   # always found...
    assert dispersed.score == pytest.approx(0.0)
    assert dispersed.grounded is False

    isolated = verify_claim("ghost", [[(0.5, 0.5)], [], [], [], []])  # found once
    assert isolated.semantic_support == pytest.approx(0.2)
    assert isolated.score < 0.5
    assert isolated.grounded is False

    absent = verify_claim("ghost", [[], [], [], [], []])              # never found
    assert absent.score == 0.0
    assert absent.grounded is False


def test_verify_claim_single_query_is_unverified():
    # SSAV aggregates over paraphrases; one query carries no cross-query
    # evidence, so the claim is reported unverified rather than optimistic.
    result = verify_claim("cup", [[(0.5, 0.5)]])
    assert result.queries_total == 1
    assert result.score == 0.0
    assert result.grounded is False


def test_verify_claim_normalizes_pixel_coordinates():
    # Pixel points around (160, 128) in a 640x640 image == normalized cluster.
    pixels = [[(160, 128)], [(168, 126)], [(155, 130)], [(160, 128)], [(164, 131)]]
    result = verify_claim("chair", pixels, image_size=(640, 640))
    assert result.score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the \\[0, 1\\] frame"):
        verify_claim("chair", pixels)  # pixels without image_size


# --- claim extraction + inventory matching ------------------------------------

_COT = (
    "To determine how far the man in the red hat is from the pallet of boxes, "
    "I need to consider the spatial arrangement in the warehouse. The man is "
    "walking on the floor, and there's a visible gap between him and the "
    "pallet. A standard forklift is typically nearby."
)
_CAPTIONS = ["man wearing a red hat", "stack of wooden pallets with boxes", "warehouse floor"]


def test_extract_object_claims_finds_mentions_and_flags_suspects():
    claims = extract_object_claims(_COT, vocabulary=_CAPTIONS)
    found = {claim.claim: claim.in_inventory for claim in claims}
    # Mentions that refer to detected objects.
    assert found["man"] is True
    assert found["red hat"] is True
    assert found["pallet"] is True
    # The hallucinated mention (not in the captions inventory) is the suspect.
    assert found["standard forklift"] is False


def test_extract_object_claims_skips_geometry_and_bookkeeping():
    claims = {c.claim for c in extract_object_claims(_COT)}
    # Heads naming scene geometry / reasoning bookkeeping are not object claims.
    assert "spatial arrangement" not in claims
    assert "visible gap" not in claims
    assert "distance" not in claims


def test_extract_object_claims_handles_empty_text():
    assert extract_object_claims("") == []
    assert extract_object_claims(None) == []


def test_claim_in_inventory_matches_colloquial_mentions():
    assert claim_in_inventory("hat", ["a man wearing a red hat"]) is True
    assert claim_in_inventory("red hat", ["a man wearing a red hat"]) is True
    assert claim_in_inventory("forklift", ["stack of wooden pallets"]) is False
    assert claim_in_inventory("forklift", []) is False


# --- paraphrased queries -------------------------------------------------------

def test_build_pointing_queries_is_deterministic_and_claim_carrying():
    queries = build_pointing_queries("red hat")
    assert len(queries) == len(POINTING_QUERIES) >= 5
    # One query per template, in template order, claim substituted in each.
    assert queries == [t.format(claim="red hat") for t in POINTING_QUERIES]
    assert all("red hat" in query for query in queries)


# --- report --------------------------------------------------------------------

def _two_results():
    return [
        verify_claim("man", _clustered(), in_inventory=True),
        verify_claim("standard forklift", _dispersed(), in_inventory=False),
    ]


def test_grounding_report_flags_ungrounded_claims():
    report = grounding_report(_two_results())
    assert report.total_claims == 2
    assert report.grounded_claims == 1
    assert report.grounded_rate == pytest.approx(0.5)
    assert report.ungrounded == ["standard forklift"]
    assert len(report.per_claim) == 2


def test_grounding_report_empty_dataset():
    report = grounding_report([])
    assert report.total_claims == 0
    assert report.mean_score == 0.0


def test_format_grounding_report_contains_headline_metrics():
    text = format_grounding_report(grounding_report(_two_results()))
    assert "CLAIM GROUNDING" in text
    assert "Grounded claims" in text
    assert "standard forklift" in text


# --- integration: the PRE-EXISTING r1_reasoning stage feeds the audit -----------

def test_claims_extracted_from_r1_reasoning_output_shape():
    """The audit's claims input is the ``output`` column ``vqasynth.r1_reasoning``
    writes. Round a messages-format conversation through that stage's own QA
    extraction (constructor offline: dummy key, no calls made), then extract
    claims from a representative trace the stage would produce for it."""
    from vqasynth.r1_reasoning import R1Reasoner  # pre-existing module

    reasoner = R1Reasoner(
        api_key="sk-test", model="gpt-4",
        image_column="image", text_column="conversation",
    )
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": "How far is the man in the red hat from the pallet of boxes?"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "2-3 feet"}]},
    ]
    question, answer = reasoner._extract_qa_pairs(conversation)[0]
    assert question.startswith("How far is the man in the red hat")

    # The reasoning trace the stage appends under ``output`` mentions the same
    # objects the question named — the audit must lift exactly those claims.
    trace = (
        f"Considering {question.lower()} I observe the warehouse. "
        "The man in the red hat stands near the pallet of boxes. "
        "A standard forklift is typically nearby."
    )
    claims = {
        c.claim: c.in_inventory
        for c in extract_object_claims(trace, vocabulary=_CAPTIONS)
    }
    assert claims["red hat"] is True
    assert claims["standard forklift"] is False


# --- integration: responses parse through vqasynth.localize --------------------

def _localize_parser():
    """extract_points_and_descriptions, or skip if localize's deps are absent.

    localize imports sam2 + transformers + accelerate at module top, which may
    not all be present in a minimal test environment."""
    try:
        from vqasynth.localize import extract_points_and_descriptions
    except Exception as e:  # ImportError or a transitive failure
        pytest.skip(f"vqasynth.localize not importable in this env: {e}")
    return extract_points_and_descriptions


def _molmo_response(x_norm, y_norm, claim):
    return f'<point x="{x_norm}" y="{y_norm}" alt="{claim}">'


def test_verify_claim_consumes_localize_parsed_points():
    """The scoring path must accept exactly what the existing pointing parser
    produces: pixel coordinates + captions from Molmo ``<point>`` tags."""
    parse = _localize_parser()
    spots = [(30, 40), (32, 39), (29, 41), (31, 40), (30, 42)]
    clustered_responses = [_molmo_response(x, y, "red cup") for x, y in spots]
    query_points = [
        [entry["points"] for entry in parse(response, image_w=640, image_h=480)]
        for response in clustered_responses
    ]
    result = verify_claim("red cup", query_points, image_size=(640, 480))
    assert result.score == pytest.approx(1.0)
    assert result.grounded is True

    dispersed = [
        [entry["points"] for entry in parse(_molmo_response(x, y, "unicorn"), 640, 480)]
        for x, y in [(10, 10), (90, 10), (10, 90), (90, 90), (50, 50)]
    ]
    assert verify_claim("unicorn", dispersed, image_size=(640, 480)).grounded is False


# --- integration: the runnable wiring (experiments.claim_grounding_audit) ------

def test_run_queries_then_audit_end_to_end(tmp_path):
    """Drive the experiment CLI the maintainer would: traces JSONL -> pointing
    queries -> collected predictions -> grounded report."""
    from experiments.claim_grounding_audit import run  # the wiring under test

    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"id": "row-12", "output": _COT, "captions": _CAPTIONS}) + "\n")

    queries_path = tmp_path / "queries.jsonl"
    exit_code = run.main(["queries", "--claims", str(traces), "--output", str(queries_path)])
    assert exit_code == 0
    emitted = [json.loads(line) for line in queries_path.read_text().splitlines()]
    claims = {record["claim"] for record in emitted}
    assert "red hat" in claims and "standard forklift" in claims
    # Every prompt is the pointing-format instruction wrapped around one
    # paraphrased query for the claim.
    assert all(record["prompt"].endswith(record["query"]) for record in emitted)

    # Collected responses: "red hat" clusters, "standard forklift" disperses.
    predictions = tmp_path / "predictions.jsonl"
    rows = []
    for query in [r for r in emitted if r["claim"] == "red hat"]:
        rows.append({"id": query["id"], "claim": query["claim"],
                     "query_index": query["query_index"], "points": [[0.30, 0.40]]})
    for query in [r for r in emitted if r["claim"] == "standard forklift"]:
        rows.append({"id": query["id"], "claim": query["claim"],
                     "query_index": query["query_index"],
                     "points": [[0.1 + 0.18 * query["query_index"], 0.1]]})
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))

    output_path = tmp_path / "grounded.jsonl"
    exit_code = run.main(
        ["audit", "--predictions", str(predictions), "--output", str(output_path)]
    )
    assert exit_code == 0
    results = {
        json.loads(line)["claim"]: json.loads(line)
        for line in output_path.read_text().splitlines()
    }
    assert results["red hat"]["grounded"] is True
    assert results["standard forklift"]["grounded"] is False


def test_run_audit_requires_points_or_localize(tmp_path):
    from experiments.claim_grounding_audit import run

    predictions = tmp_path / "predictions.jsonl"
    record = {"id": "x", "claim": "cup", "query_index": 0,
              "response": '<point x="10" y="10" alt="cup">'}
    predictions.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="pre-parsed 'points'"):
        run.main(["audit", "--predictions", str(predictions)])
