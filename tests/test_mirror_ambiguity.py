"""Structural tests for vqasynth.mirror_ambiguity.

Verifies the MIRA-Bench loader, scorer and the RGB vs RGB-D depth delta against
hand-computed small examples — no CUDA, no model download, no HuggingFace
fetch. Mirrors the dependency-free style of tests/test_visual_credit.py.

Integration with the PRE-EXISTING evaluation layer the multi-benchmark stage
already ships is asserted directly: the scorer's yes/no and option paths must
agree with ``vqasynth.evaluation``'s extractors, and (guarded on the
``datasets`` stack being importable) registering the benchmark must make
``vqasynth.benchmarks.BenchmarkRunner`` score ``mirabench`` items through the
same registry contract as SpatialScore / OmniSpatial / SpaCE-10 / MindCube.
"""
from __future__ import annotations

import json

import pytest

from vqasynth.mirror_ambiguity import (
    COGNITIVE_LEVELS,
    MIRA_BENCH_NAME,
    cognitive_level,
    depth_delta,
    format_mirror_report,
    gold_option_set,
    load_mira_bench,
    normalize_mira_entry,
    register_mira_bench,
    score_mira_item,
)


def _raw_entry(**overrides):
    """One ``questions.json`` record in the MIRA annotation shape."""
    entry = {
        "question_id": "vc_001",
        "question": "Is the person directly observed or a reflection?",
        "question_type": "yes_no",
        "category": "perception",
        "subcategory": "visibility_classification",
        "choices": '["Yes", "No"]',
        "answer": "No",
        "image_path": "images/RGBD_Ben_Self_collected/rgb/scene_0001.png",
        "depth_image_path": "images/RGBD_Ben_Self_collected/depth/scene_0001.png",
        "requires_depth": False,
        "requires_mirror_reasoning": True,
    }
    entry.update(overrides)
    return entry


def _item(raw=None, **overrides):
    """A normalized item, with optional raw-annotation and item-level overrides."""
    item = normalize_mira_entry(_raw_entry(**(raw or {})), 0)
    item.update(overrides)
    return item


# --- entry normalization -----------------------------------------------------

def test_normalize_entry_maps_mira_fields_to_common_schema(tmp_path):
    item = normalize_mira_entry(_raw_entry(), 0, data_root=tmp_path)

    assert item["id"] == "vc_001"
    assert item["question_type"] == "judgment"  # yes_no -> common vocabulary
    assert item["mira_question_type"] == "yes_no"
    assert item["category"] == "visibility_classification"
    assert item["subcategory"] == "reflection_perception"
    assert item["options"] == ["Yes", "No"]  # JSON-text field decoded in place
    assert item["images"] == [
        str(tmp_path / "images/RGBD_Ben_Self_collected/rgb/scene_0001.png")
    ]
    assert item["depth_images"] == [
        str(tmp_path / "images/RGBD_Ben_Self_collected/depth/scene_0001.png")
    ]
    assert item["requires_mirror_reasoning"] is True
    assert item["source"] == "MIRA-Bench"


def test_normalize_entry_prefers_input_depth_path():
    item = normalize_mira_entry(
        _raw_entry(input_depth_image_path="depth/normalized/scene_0001.png"), 0
    )
    assert item["depth_images"] == ["depth/normalized/scene_0001.png"]


def test_normalize_entry_disambiguates_duplicate_question_ids():
    seen = set()
    first = normalize_mira_entry(_raw_entry(), 0, seen=seen)
    second = normalize_mira_entry(_raw_entry(), 1, seen=seen)
    assert first["id"] != second["id"]


def test_cognitive_level_covers_all_subtasks_with_three_levels():
    assert {len(subtasks) for subtasks in COGNITIVE_LEVELS.values()} == {4, 5}
    assert cognitive_level("visibility_classification") == "reflection_perception"
    assert cognitive_level("relative_distance_ordering") == "spatial_reasoning"
    assert cognitive_level("navigation_movement_planning") == "scene_decision_making"
    assert cognitive_level("not_a_subtask") == "other"


def test_load_mira_bench_reads_local_questions_json(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                _raw_entry(),
                _raw_entry(
                    question_id="rk_007",
                    question_type="single_choice",
                    subcategory="reachability",
                    choices='["left hand", "right hand"]',
                    answer="A",
                ),
                _raw_entry(question_id="zz_010", subcategory="not_a_subtask"),
            ]
        )
    )

    items = load_mira_bench(questions_json=str(questions))
    assert [item["id"] for item in items] == ["vc_001", "rk_007", "zz_010"]

    filtered = load_mira_bench(questions_json=str(questions), categories=["reachability"])
    assert [item["category"] for item in filtered] == ["reachability"]

    assert len(load_mira_bench(questions_json=str(questions), max_items=1)) == 1

    not_a_list = tmp_path / "not_a_list.json"
    not_a_list.write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        load_mira_bench(questions_json=str(not_a_list))


def test_load_mira_bench_gated_download_points_at_manual_path(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def unavailable(*args, **kwargs):
        raise RuntimeError("401 Client Error: gated repo")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    # the upstream release is gated, so the error must name the manual download
    with pytest.raises(RuntimeError, match="hf download"):
        load_mira_bench()


# --- scoring -----------------------------------------------------------------

def test_score_yes_no_reuses_evaluation_extractor():
    from vqasynth.evaluation import extract_yes_no  # pre-existing module

    item = _item()  # gold "No"
    gold_verdict = extract_yes_no("No")
    for prediction in ("No.", "Yes.", "Actually, it is a reflection."):
        expected = 1.0 if extract_yes_no(prediction) == gold_verdict else 0.0
        assert score_mira_item(item, prediction) == expected


def test_score_single_choice_letter_then_option_text():
    item = _item(
        raw={
            "question_type": "single_choice",
            "choices": '["reflected", "directly observed", "cannot tell"]',
            "answer": "B",
        }
    )
    assert gold_option_set(item) == {"B"}
    assert score_mira_item(item, "Answer: B") == 1.0
    assert score_mira_item(item, "(a)") == 0.0
    # No letter in the answer -> fall back to the gold option text being present.
    assert score_mira_item(item, "The person is directly observed in the scene.") == 1.0
    assert score_mira_item(item, "It is a reflection.") == 0.0


def test_score_gold_given_as_index_or_option_text():
    by_index = _item(
        raw={"question_type": "single_choice", "choices": '["left", "right"]', "answer": 1}
    )
    assert gold_option_set(by_index) == {"B"}
    assert score_mira_item(by_index, "B.") == 1.0

    by_text = _item(
        raw={
            "question_type": "single_choice",
            "choices": '["left hand", "right hand"]',
            "answer": "right hand",
        }
    )
    assert gold_option_set(by_text) == {"B"}
    assert score_mira_item(by_text, "The right hand.") == 1.0


def test_score_multi_select_uses_jaccard_partial_credit():
    item = _item(
        raw={
            "question_type": "multi_choice",
            "choices": '["mirror", "window", "person", "door"]',
            "answer": '["A", "C"]',
        }
    )
    assert gold_option_set(item) == {"A", "C"}
    assert score_mira_item(item, "A and C") == 1.0
    # Jaccard over the letters mentioned: |{A}| / |{A, C}|
    assert score_mira_item(item, "A only") == pytest.approx(0.5)
    # |{A, C}| / |{A, B, C}|
    assert score_mira_item(item, "A, B and C") == pytest.approx(2 / 3)


def test_score_count_compares_extracted_numbers():
    item = _item(raw={"question_type": "count", "answer": "3"})
    assert score_mira_item(item, "There are 3 people.") == 1.0
    assert score_mira_item(item, "I count five.") == 0.0


def test_score_unextractable_returns_none_for_judge_fallback():
    # None is the BenchmarkRunner contract for "hand this item to the LLM judge".
    assert score_mira_item(_item(raw={"question_type": "single_choice", "answer": ""}), "") is None
    assert score_mira_item(_item(raw={"answer": "Maybe"}), "perhaps") is None
    multi = _item(
        raw={
            "question_type": "multi_choice",
            "choices": '["mirror", "window", "person"]',
            "answer": '["A", "C"]',
        }
    )
    assert score_mira_item(multi, "cannot tell") is None
    axes = _item(raw={"question_type": "spatial_3d_axes", "answer": "left"})
    assert score_mira_item(axes, "somewhere nearby") is None


# --- RGB vs RGB-D depth delta ------------------------------------------------

def _delta_items():
    return [
        # correct on RGB, wrong once depth is added (depth harms)
        _item(raw={"question_id": "a", "answer": "No"}),
        # wrong on RGB, rescued by depth (depth helps); also depth-requiring
        _item(
            raw={
                "question_id": "b",
                "question_type": "single_choice",
                "subcategory": "reachability",
                "choices": '["left hand", "right hand"]',
                "answer": "B",
                "requires_depth": True,
            }
        ),
        # correct under both conditions (depth changed nothing)
        _item(raw={"question_id": "c", "answer": "Yes"}),
    ]


def _delta_predictions():
    return {
        "a": {"pred_rgb": "No.", "pred_rgbd": "Yes."},
        "b": {"pred_rgb": "A.", "pred_rgbd": "B."},
        "c": {"pred_rgb": "Yes.", "pred_rgbd": "Yes."},
    }


def test_depth_delta_matches_hand_computation():
    report = depth_delta(_delta_items(), _delta_predictions())

    assert report.total == 3
    assert report.rgb_accuracy == pytest.approx(2 / 3)  # a, c
    assert report.rgbd_accuracy == pytest.approx(2 / 3)  # b, c
    assert report.depth_delta == pytest.approx(0.0)
    assert report.depth_harm_rate == pytest.approx(1 / 3)  # a only
    assert report.depth_help_rate == pytest.approx(1 / 3)  # b only
    # only a and b changed their answer text between the two conditions
    assert report.decision_change_rate == pytest.approx(2 / 3)
    assert [(row["harmed"], row["helped"]) for row in report.per_item] == [
        (True, False),
        (False, True),
        (False, False),
    ]


def test_depth_delta_breaks_down_by_level_and_requires_depth():
    report = depth_delta(_delta_items(), _delta_predictions())

    assert set(report.by_category) == {"visibility_classification", "reachability"}
    assert report.by_category["reachability"]["depth_delta"] == pytest.approx(1.0)
    assert report.by_level["reflection_perception"]["count"] == 2
    assert report.by_requires_depth[True]["count"] == 1
    assert report.by_requires_depth[False]["count"] == 2
    # the one depth-requiring item is the one depth rescued
    assert report.by_requires_depth[True]["depth_delta"] == pytest.approx(1.0)


def test_depth_delta_skips_items_missing_a_modality():
    predictions = {"a": {"pred_rgb": "No.", "pred_rgbd": "Yes."}}
    report = depth_delta(_delta_items(), predictions)
    assert report.total == 1
    assert report.rgb_accuracy == 1.0
    assert report.rgbd_accuracy == 0.0
    assert report.depth_delta == pytest.approx(-1.0)


def test_depth_delta_empty_predictions_returns_zeros():
    report = depth_delta(_delta_items(), {})
    assert report.total == 0
    assert report.depth_delta == 0.0


def test_format_mirror_report_contains_headline_metrics():
    report = depth_delta(_delta_items(), _delta_predictions())
    text = format_mirror_report(report)
    for expected in (
        "MIRROR-INDUCED RGB-D AMBIGUITY",
        "Accuracy (RGB)",
        "Accuracy (RGB-D)",
        "Depth delta",
        "Depth harm rate",
        "By cognitive level",
    ):
        assert expected in text


# --- integration with the PRE-EXISTING vqasynth.benchmarks registries --------

@pytest.fixture
def benchmarks():
    """The pre-existing registry module, with mirabench registered for the test."""
    try:
        from vqasynth import benchmarks as module
    except ImportError as exc:  # the loader layer pulls in the `datasets` stack
        pytest.skip(f"vqasynth.benchmarks unavailable: {exc}")
    register_mira_bench()
    yield module
    module.BENCHMARK_LOADERS.pop(MIRA_BENCH_NAME, None)
    module.BENCHMARK_SCORERS.pop(MIRA_BENCH_NAME, None)


def test_register_plugs_into_loader_and_scorer_registries(benchmarks):
    assert MIRA_BENCH_NAME in benchmarks.BENCHMARK_LOADERS
    assert MIRA_BENCH_NAME in benchmarks.BENCHMARK_SCORERS
    assert benchmarks.BENCHMARK_LOADERS[MIRA_BENCH_NAME] is load_mira_bench
    assert benchmarks.BENCHMARK_SCORERS[MIRA_BENCH_NAME] is score_mira_item
    assert benchmarks.load_benchmark(MIRA_BENCH_NAME) is load_mira_bench


def test_benchmark_runner_scores_mirabench_items(benchmarks):
    items = _delta_items()
    predictions = {"a": "No.", "b": "B.", "c": "No."}

    runner = benchmarks.BenchmarkRunner(benchmarks=MIRA_BENCH_NAME)
    result = runner.score(MIRA_BENCH_NAME, items, predictions)

    assert result["benchmark"] == MIRA_BENCH_NAME
    assert result["total"] == 3
    assert result["overall_accuracy"] == pytest.approx(2 / 3)
    assert result["by_category"]["visibility_classification"]["count"] == 2
    # subcategory carries the paper's cognitive levels
    assert set(result["by_subcategory"]) == {"reflection_perception", "scene_decision_making"}
    # a list of predictions in item order is accepted too
    by_list = runner.score(MIRA_BENCH_NAME, items, ["No.", "B.", "No."])
    assert by_list["overall_accuracy"] == result["overall_accuracy"]


def test_benchmark_runner_routes_unscored_items_to_zero(benchmarks):
    items = [_item(raw={"question_type": "single_choice", "answer": ""})]
    result = benchmarks.BenchmarkRunner(benchmarks=MIRA_BENCH_NAME).score(
        MIRA_BENCH_NAME, items, {"vc_001": "no idea"}
    )
    # the scorer returned None and no judge client is configured -> 0.0
    assert result["overall_accuracy"] == 0.0
