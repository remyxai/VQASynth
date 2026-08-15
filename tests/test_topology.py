"""Tests for the inside/touching predicates added to the prompt stage.

These relations are computed directly on the fused per-object point clouds
(the same VGGT-derived clouds ``vqasynth.prompts`` consumes), so the tests
build small synthetic clouds rather than real model output. The relation
tests accept raw Nx3 arrays, so the pure geometry checks need no open3d; the
wiring checks exercise the full predicate pool, whose existing members use
the open3d API, so those are guarded on an open3d install (Docker env).
"""
import random

import numpy as np
import pytest

from vqasynth.prompts import PromptGenerator
from vqasynth.topology import TopologicalRelationGenerator, is_inside, is_touching

try:
    import open3d as o3d
except ImportError:  # Docker-only dep; relation tests do not need it
    o3d = None


def make_cloud(center, extent=0.2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-extent, extent, (n, 3)) + np.asarray(center)


@pytest.fixture
def generator():
    return PromptGenerator()


class TestRelationGeometry:
    def test_small_cloud_inside_large_cloud(self):
        outer = make_cloud([0, 0, 0], extent=1.0, seed=1)
        inner = make_cloud([0, 0, 0], extent=0.1, seed=2)
        assert is_inside(inner, outer)

    def test_disjoint_clouds_not_inside(self):
        a = make_cloud([0, 0, 0], extent=0.2, seed=3)
        b = make_cloud([5, 5, 5], extent=0.2, seed=4)
        assert not is_inside(a, b)

    def test_touching_clouds_detected(self):
        a = make_cloud([0, 0, 0], extent=0.5, seed=5)
        b = make_cloud([0.9, 0, 0], extent=0.5, seed=6)  # surfaces interpenetrate
        assert is_touching(a, b)

    def test_separated_clouds_not_touching(self):
        a = make_cloud([0, 0, 0], extent=0.2, seed=7)
        b = make_cloud([0, 0, 10], extent=0.2, seed=8)
        assert not is_touching(a, b)

    def test_empty_cloud_is_neither(self):
        cloud = make_cloud([0, 0, 0])
        assert not is_inside(np.zeros((0, 3)), cloud)
        assert not is_touching(np.zeros((0, 3)), cloud)


class TestPredicateRendering:
    def test_inside_predicate_true(self):
        gen = TopologicalRelationGenerator()
        pair_a = ("Mug", make_cloud([0, 0, 0], extent=0.1))
        pair_b = ("Sink", make_cloud([0, 0, 0], extent=1.0))
        result = gen.inside_predicate(pair_a, pair_b)
        assert "mug" in result and "sink" in result
        assert " Answer: Yes" in result or " Answer: Correct" in result or " Answer: Indeed" in result

    def test_inside_predicate_false(self):
        gen = TopologicalRelationGenerator()
        pair_a = ("Mug", make_cloud([5, 5, 5], extent=0.1))
        pair_b = ("Sink", make_cloud([0, 0, 0], extent=1.0))
        result = gen.inside_predicate(pair_a, pair_b)
        assert " Answer: " in result
        assert "Yes" not in result.split(" Answer: ")[1]

    def test_touching_predicate_mentions_both_objects(self):
        gen = TopologicalRelationGenerator()
        pair_a = ("Cup", make_cloud([0, 0, 0], extent=0.5, seed=9))
        pair_b = ("Table", make_cloud([0.9, 0, 0], extent=0.5, seed=10))
        result = gen.touching_predicate(pair_a, pair_b)
        assert "cup" in result and "table" in result
        assert " Answer: " in result


@pytest.mark.skipif(o3d is None, reason="open3d (Docker-only dep) not installed")
class TestPromptGeneratorWiring:
    @staticmethod
    def _as_geometry(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
        return cloud

    def test_topological_predicates_flagged(self, generator):
        for fn in (
            generator.topological_relations.inside_predicate,
            generator.topological_relations.touching_predicate,
        ):
            assert callable(fn)
            assert generator._is_topological(fn)

    def test_evaluate_predicates_emits_topological_prompts(self, generator):
        """The wired call site produces inside/touching Q&A for real pairs."""
        objects = [
            ("Mug", self._as_geometry(make_cloud([0, 0, 0], extent=0.1, seed=11))),
            ("Sink", self._as_geometry(make_cloud([0, 0, 0], extent=1.0, seed=12))),
            ("Towel", self._as_geometry(make_cloud([3, 0, 0], extent=0.3, seed=13))),
            ("Bench", self._as_geometry(make_cloud([3.25, 0, 0], extent=0.5, seed=14))),
        ]
        pairs = [(objects[0], objects[1]), (objects[2], objects[3])]

        # evaluate_predicates_on_pairs samples only 2 of the 12 always-on
        # variants per call, so try a few seeds: across seeds the topological
        # predicates must be reachable and render valid Q&A.
        topological = []
        for seed in range(20):
            random.seed(seed)
            results = generator.evaluate_predicates_on_pairs(
                pairs, is_canonicalized=True
            )
            topological += [
                r for r in results if "inside" in r.lower() or "touching" in r.lower()
            ]
        assert topological, "expected at least one inside/touching prompt"
        for result in topological:
            assert " Answer: " in result

    def test_degenerate_clouds_skip_topological_predicates(self, generator):
        """Flat clouds must not produce meaningless containment/contact Q&A."""
        flat = np.zeros((10, 3))
        flat[:, 0] = np.linspace(0, 1, 10)  # spans x only
        objects = [
            ("Wall art", self._as_geometry(flat)),
            ("Table", self._as_geometry(make_cloud([0, 0, 0], extent=0.5, seed=15))),
        ]

        random.seed(1)
        results = generator.evaluate_predicates_on_pairs(
            [(objects[0], objects[1])], is_canonicalized=False
        )
        assert results
        assert not [
            r for r in results if "inside" in r.lower() or "touching" in r.lower()
        ]

    def test_prompts_round_trip_into_messages(self, generator):
        objects = [
            ("Mug", self._as_geometry(make_cloud([0, 0, 0], extent=0.1, seed=16))),
            ("Sink", self._as_geometry(make_cloud([0, 0, 0], extent=1.0, seed=17))),
        ]
        random.seed(2)
        results = generator.evaluate_predicates_on_pairs(
            [(objects[0], objects[1])], is_canonicalized=True
        )
        messages = generator.create_messages_from_prompts(results[:5])
        assert messages
        assert all(m["role"] in ("user", "assistant") for m in messages)
