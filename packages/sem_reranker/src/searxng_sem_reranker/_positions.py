"""SearXNG position rewriting shared by the rerankers.

Only ``positions`` and ``priority`` survive ``ResultContainer.close()``;
``score`` is recalculated.  Positions must be positive integers and the list
length must stay intact because SearXNG multiplies by ``len(positions)``.
"""

from __future__ import annotations

import typing as t


def rewrite_positions(results: list[t.Any], order: list[int]) -> None:
    """Give every result a fresh, contiguous position.

    Results absent from ``order`` keep their relative order below the ranked
    ones.  The length of every ``positions`` list is preserved.
    """
    new_positions = {index: position for position, index in enumerate(order, start=1)}
    next_position = len(new_positions) + 1

    for index, result in enumerate(results):
        old = result["positions"] or []
        count = max(len(old), 1)
        if index in new_positions:
            result["positions"] = [new_positions[index]] * count
        else:
            result["positions"] = [next_position] * count
            next_position += 1
