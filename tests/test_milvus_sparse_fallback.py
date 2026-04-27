from __future__ import annotations

from app.services.indexing.milvus_indexer import (
    _dense_to_sparse_fallback,
    _resolve_sparse_vector,
)


def test_dense_to_sparse_fallback_returns_non_empty() -> None:
    sparse = _dense_to_sparse_fallback([0.0, 0.0, 0.0])
    assert sparse
    assert len(sparse) == 1
    assert 0 in sparse


def test_dense_to_sparse_fallback_top_k() -> None:
    dense = [0.1, -0.8, 0.3, 0.2, -0.4]
    sparse = _dense_to_sparse_fallback(dense, top_k=2)
    assert set(sparse.keys()) == {1, 4}
    assert sparse[1] == 0.8
    assert sparse[4] == 0.4


def test_resolve_sparse_vector_prefers_existing() -> None:
    dense = [0.1, -0.8, 0.3]
    existing = {100: 0.9}
    sparse = _resolve_sparse_vector(dense, existing)
    assert sparse == existing
