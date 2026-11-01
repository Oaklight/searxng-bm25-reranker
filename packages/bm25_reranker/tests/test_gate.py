"""Escalation gate tests: BM25 stays in charge unless a signal is weak."""

from __future__ import annotations

from searxng_bm25_reranker._gate import (
    GateConfig,
    best_coverage,
    load_gate,
    should_escalate,
)

DOCS = [
    "Python asyncio tutorial",
    "Cooking pasta",
    "Weather forecast",
]


def config(**overrides) -> GateConfig:
    values = {
        "enabled": True,
        "min_coverage": 0.5,
        "min_margin": 0.1,
        "min_scored": 2,
        "top_k": 10,
    }
    values.update(overrides)
    return GateConfig(**values)


class TestLoadGate:
    def test_defaults_are_disabled(self) -> None:
        gate = load_gate({})
        assert gate.enabled is False
        assert gate.min_coverage == 0.5
        assert gate.min_margin == 0.1
        assert gate.min_scored == 2
        assert gate.top_k == 10

    def test_explicit_values(self) -> None:
        gate = load_gate(
            {
                "rerankers": {
                    "escalate": {
                        "enabled": True,
                        "min_coverage": 0.7,
                        "min_margin": 0.2,
                        "min_scored": 3,
                        "top_k": 5,
                    }
                }
            }
        )
        assert gate == GateConfig(True, 0.7, 0.2, 3, 5)

    def test_invalid_values_fall_back(self) -> None:
        gate = load_gate(
            {
                "rerankers": {
                    "escalate": {
                        "enabled": "yes",
                        "min_coverage": 2.0,
                        "min_margin": -1,
                        "min_scored": 0,
                        "top_k": 0,
                    }
                }
            }
        )
        assert gate.enabled is False
        assert gate.min_coverage == 0.5
        assert gate.min_margin == 0.1
        assert gate.min_scored == 2
        assert gate.top_k == 10

    def test_non_mapping_block_is_ignored(self) -> None:
        assert load_gate({"rerankers": {"escalate": "nonsense"}}) == GateConfig()


class TestShouldEscalate:
    def test_disabled_gate_always_escalates(self) -> None:
        assert should_escalate(
            "python asyncio", DOCS, [2.0, 1.0, 0.5], [0, 1, 2], config(enabled=False)
        ) == (True, "disabled")

    def test_good_coverage_and_margin_stays_with_bm25(self) -> None:
        assert should_escalate(
            "python asyncio tutorial",
            DOCS,
            [2.0, 1.0, 0.5],
            [0, 1, 2],
            config(),
        ) == (False, "ok")

    def test_low_coverage_escalates(self) -> None:
        assert should_escalate(
            "quantum entanglement",
            DOCS,
            [2.0, 1.0, 0.5],
            [0, 1, 2],
            config(),
        ) == (True, "coverage")

    def test_flat_margin_escalates(self) -> None:
        assert should_escalate(
            "python asyncio tutorial",
            DOCS,
            [1.0, 1.0, 0.5],
            [0, 1, 2],
            config(),
        ) == (True, "margin")

    def test_too_few_scored_results_escalates(self) -> None:
        assert should_escalate(
            "python asyncio tutorial",
            DOCS,
            [1.0, None, None],
            [0, 1, 2],
            config(),
        ) == (True, "scored")

    def test_zero_top_score_escalates(self) -> None:
        assert should_escalate(
            "python asyncio tutorial",
            DOCS,
            [0.0, 0.0, 0.0],
            [0, 1, 2],
            config(),
        ) == (True, "top_score")

    def test_index_order_breaks_score_ties(self) -> None:
        assert should_escalate(
            "python asyncio tutorial",
            DOCS,
            [1.0, 1.0],
            [1, 0],
            config(min_margin=0.0),
        ) == (False, "ok")


class TestCoverage:
    def test_full_query_coverage(self) -> None:
        assert best_coverage("python asyncio tutorial", DOCS, [0]) == 1.0

    def test_partial_coverage(self) -> None:
        assert best_coverage("python pasta", DOCS, [0, 1]) == 0.5

    def test_no_coverage(self) -> None:
        assert best_coverage("quantum entanglement", DOCS, [0, 1, 2]) == 0.0

    def test_empty_query_is_not_a_reason_to_escalate(self) -> None:
        assert best_coverage("", DOCS, [0]) == 1.0

    def test_cjk_coverage(self) -> None:
        assert best_coverage("缓存穿透", ["Redis缓存穿透"], [0]) == 1.0
