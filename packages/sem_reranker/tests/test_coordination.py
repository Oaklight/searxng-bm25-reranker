"""Coordinator selection must be identical in both packages."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import searx

bm25 = pytest.importorskip("searxng_bm25_reranker")

from searxng_sem_reranker import SXNGPlugin as SemPlugin  # noqa: E402
from searxng_sem_reranker._coop import is_coordinator as sem_coordinates  # noqa: E402

import searxng_sem_reranker as sem  # noqa: E402

Bm25Plugin = bm25.SXNGPlugin
bm25_coordinates = bm25._is_coordinator


def search(*plugins: str) -> SimpleNamespace:
    return SimpleNamespace(user_plugins=list(plugins))


def set_coordinator(value: object) -> None:
    searx.settings["rerankers"] = {"coordinator": value}


@pytest.mark.parametrize(
    ("configured", "active", "bm25", "sem"),
    [
        (None, ("bm25_reranker", "sem_reranker"), False, True),
        (None, ("bm25_reranker",), True, True),
        ("sem_reranker", ("bm25_reranker", "sem_reranker"), False, True),
        ("bm25_reranker", ("bm25_reranker", "sem_reranker"), True, False),
        ("sem_reranker", ("bm25_reranker",), True, True),
        ("bm25_reranker", ("sem_reranker",), True, True),
    ],
)
def test_truth_table(configured, active, bm25, sem) -> None:
    if configured is None:
        searx.settings.pop("rerankers", None)
    else:
        set_coordinator(configured)

    request = search(*active)
    assert bm25_coordinates(request, "bm25_reranker") is bm25
    assert sem_coordinates(searx.settings, request, "sem_reranker") is sem


def test_missing_user_plugins_lets_the_running_plugin_act() -> None:
    set_coordinator("sem_reranker")
    request = SimpleNamespace()
    assert bm25_coordinates(request, "bm25_reranker") is True
    assert sem_coordinates(searx.settings, request, "sem_reranker") is True


def test_invalid_coordinator_is_treated_as_unset() -> None:
    set_coordinator("nonsense")
    request = search("bm25_reranker", "sem_reranker")
    assert bm25_coordinates(request, "bm25_reranker") is False
    assert sem_coordinates(searx.settings, request, "sem_reranker") is True


def test_non_mapping_block_is_treated_as_unset() -> None:
    searx.settings["rerankers"] = "nonsense"
    request = search("bm25_reranker", "sem_reranker")
    assert bm25_coordinates(request, "bm25_reranker") is False
    assert sem_coordinates(searx.settings, request, "sem_reranker") is True


def make_results() -> dict[str, dict]:
    return {
        "0": {"title": "alpha", "content": "first", "positions": [1]},
        "1": {"title": "beta", "content": "second", "positions": [1]},
        "2": {"title": "gamma", "content": "third", "positions": [1]},
    }


def run_both(
    order: list[str],
    coordinator: str,
    fusion: dict | None = None,
    escalate: dict | None = None,
    query: str = "alpha",
) -> dict[str, list[int]]:
    searx.settings.clear()
    searx.settings["rerankers"] = {"coordinator": coordinator}
    if escalate is not None:
        searx.settings["rerankers"]["escalate"] = escalate
    searx.settings["sem_reranker"] = {"fusion": fusion or {"rrf_k": 20}}
    results = make_results()
    request = SimpleNamespace(
        result_container=SimpleNamespace(main_results_map=results),
        search_query=SimpleNamespace(query=query),
        user_plugins=["bm25_reranker", "sem_reranker"],
    )
    plugins = {
        "bm25_reranker": Bm25Plugin(SimpleNamespace()),
        "sem_reranker": SemPlugin(SimpleNamespace()),
    }
    for plugin_id in order:
        plugins[plugin_id].post_search(object(), request)
    return {key: value["positions"] for key, value in results.items()}


@pytest.mark.parametrize("coordinator", ["sem_reranker", "bm25_reranker"])
def test_both_call_orders_produce_the_same_positions(
    monkeypatch, coordinator: str
) -> None:
    monkeypatch.setattr(sem, "score_neural", lambda q, r, c=None: [0.1, 0.9, 0.2])

    first = run_both(["bm25_reranker", "sem_reranker"], coordinator)
    second = run_both(["sem_reranker", "bm25_reranker"], coordinator)

    assert first == second
    assert sorted(value[0] for value in first.values()) == [1, 2, 3]


@pytest.mark.parametrize("coordinator", ["sem_reranker", "bm25_reranker"])
def test_both_signals_affect_the_fused_order(monkeypatch, coordinator: str) -> None:
    """The coordinator must use BM25 and neural scores, not just one of them."""
    import searxng_bm25_reranker as bm25

    monkeypatch.setattr(sem, "score_neural", lambda q, r, c=None: [0.1, 0.5, 0.9])
    monkeypatch.setattr(
        bm25, "score_results", lambda q, r, config=None: [0.9, 0.5, 0.1]
    )

    neural_heavy = run_both(
        ["bm25_reranker", "sem_reranker"],
        coordinator,
        {"rrf_k": 20, "neural_weight": 10.0, "bm25_weight": 1.0},
    )
    bm25_heavy = run_both(
        ["sem_reranker", "bm25_reranker"],
        coordinator,
        {"rrf_k": 20, "neural_weight": 1.0, "bm25_weight": 10.0},
    )

    assert min(neural_heavy, key=lambda key: neural_heavy[key][0]) == "2"
    assert min(bm25_heavy, key=lambda key: bm25_heavy[key][0]) == "0"


def test_gate_closed_does_not_call_sem(monkeypatch) -> None:
    calls: list[int] = []

    def fake_sem(q, r, c=None):
        calls.append(1)
        return [0.1, 0.9, 0.2]

    monkeypatch.setattr(sem, "score_neural", fake_sem)

    run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate={
            "enabled": True,
            "min_coverage": 0.9,
            "min_margin": 0.0,
            "min_scored": 1,
        },
        query="alpha",
    )

    assert calls == []


def test_gate_open_calls_sem(monkeypatch) -> None:
    calls: list[int] = []

    def fake_sem(q, r, c=None):
        calls.append(1)
        return [0.1, 0.9, 0.2]

    monkeypatch.setattr(sem, "score_neural", fake_sem)

    result = run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate={
            "enabled": True,
            "min_coverage": 0.5,
            "min_margin": 0.0,
            "min_scored": 1,
        },
        query="zzz",
    )

    assert calls == [1]
    assert min(result, key=lambda key: result[key][0]) == "1"


GATE_OPEN = {
    "enabled": True,
    "min_coverage": 0.5,
    "min_margin": 0.0,
    "min_scored": 1,
}


def bm25_only(monkeypatch) -> dict[str, list[int]]:
    monkeypatch.setattr(sem, "score_neural", lambda q, r, c=None: [None] * len(r))
    return run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate=GATE_OPEN,
        query="zzz",
    )


def test_inactive_sem_is_not_called(monkeypatch) -> None:
    import searxng_bm25_reranker as bm25_module

    calls: list[int] = []
    monkeypatch.setattr(sem, "score_neural", lambda *a, **k: calls.append(1))
    request = SimpleNamespace(user_plugins=["bm25_reranker"])

    assert bm25_module._neural_scores("alpha", [], request) is None
    assert calls == []


def test_missing_sem_package_keeps_bm25_order(monkeypatch) -> None:
    expected = bm25_only(monkeypatch)
    monkeypatch.setitem(sys.modules, "searxng_sem_reranker", None)

    result = run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate=GATE_OPEN,
        query="zzz",
    )

    assert result == expected


def test_sem_failure_keeps_bm25_order(monkeypatch) -> None:
    expected = bm25_only(monkeypatch)

    def fail(query, results, config=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(sem, "score_neural", fail)
    result = run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate=GATE_OPEN,
        query="zzz",
    )

    assert result == expected


def test_sem_stages_off_keeps_bm25_order(monkeypatch) -> None:
    original = sem.score_neural
    expected = bm25_only(monkeypatch)
    monkeypatch.setattr(sem, "score_neural", original)

    result = run_both(
        ["bm25_reranker", "sem_reranker"],
        "bm25_reranker",
        escalate=GATE_OPEN,
        query="zzz",
    )

    assert result == expected
