import numpy as np

from lumina_mcp_router.index import ToolEntry, VectorIndex


def _entry(name: str, backend: str, vec: list[float]) -> ToolEntry:
    return ToolEntry(
        name=name,
        backend=backend,
        original_name=name.split("__", 1)[-1],
        description=f"desc of {name}",
        input_schema={"type": "object"},
        embedding=np.asarray(vec, dtype=np.float32),
    )


def test_empty_index_returns_no_results():
    idx = VectorIndex()
    assert idx.search([1.0, 0.0]) == []


def test_cosine_ranks_most_similar_first():
    idx = VectorIndex()
    idx.add(_entry("gsuite__send_email", "gsuite", [1.0, 0.0, 0.0]))
    idx.add(_entry("hass__toggle_light", "hass", [0.0, 1.0, 0.0]))
    idx.add(_entry("odoo__list_partners", "odoo", [0.5, 0.5, 0.0]))

    results = idx.search([0.9, 0.1, 0.0], top_k=3)
    assert len(results) == 3
    assert results[0].name == "gsuite__send_email"
    # Scores sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_clamping():
    idx = VectorIndex()
    for i in range(5):
        idx.add(_entry(f"b__t{i}", "b", [float(i), 1.0]))
    results = idx.search([1.0, 1.0], top_k=2)
    assert len(results) == 2


def test_zero_query_returns_empty():
    idx = VectorIndex()
    idx.add(_entry("b__t", "b", [1.0, 0.0]))
    assert idx.search([0.0, 0.0]) == []


def test_get_and_contains():
    idx = VectorIndex()
    e = _entry("x__y", "x", [1.0])
    idx.add(e)
    assert "x__y" in idx
    assert idx.get("x__y") is e
    assert idx.get("missing") is None
    assert len(idx) == 1


def test_clear():
    idx = VectorIndex()
    idx.add(_entry("x__y", "x", [1.0]))
    idx.clear()
    assert len(idx) == 0
