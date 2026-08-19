from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.retrieval.faiss_index import FaissRetriever


IALS_MODEL_PATH = "models/ials/ials_model.npz"

INDEX_PATH = "models/retrieval/faiss.index"
METADATA_PATH = "models/retrieval/index_metadata.json"

TOP_K = 10
NUM_QUERIES = 500
RANDOM_SEED = 42


def load_item_factors() -> np.ndarray:
    with np.load(
        IALS_MODEL_PATH,
        allow_pickle=False,
    ) as data:
        item_factors = data["item_factors"]

    return np.asarray(
        item_factors,
        dtype=np.float32,
    )


def normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    result = np.zeros_like(
        vectors,
        dtype=np.float32,
    )

    mask = norms[:, 0] > 0

    result[mask] = (
        vectors[mask]
        / norms[mask]
    )

    return result


def brute_force_search(
    item_vectors: np.ndarray,
    query_vector: np.ndarray,
    k: int,
) -> tuple[list[int], list[float]]:
    """Exact Top-K inner-product search."""
    query_vector = query_vector.astype(
        np.float32
    )

    norm = np.linalg.norm(
        query_vector
    )

    if norm == 0:
        raise ValueError(
            "Query vector cannot be zero."
        )

    query_vector = (
        query_vector / norm
    )

    scores = (
        item_vectors @ query_vector
    )

    top_indices = np.argsort(
        -scores
    )[:k]

    return (
        top_indices.astype(int).tolist(),
        scores[
            top_indices
        ].astype(float).tolist(),
    )


def recall_at_k(
    expected: list[int],
    actual: list[int],
) -> float:
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set:
        return 0.0

    return len(
        expected_set & actual_set
    ) / len(expected_set)


def main() -> None:
    print("=" * 60)
    print("FAISS Retrieval Benchmark")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load vectors
    # ---------------------------------------------------------

    item_vectors = load_item_factors()

    item_vectors = normalize_rows(
        item_vectors
    )

    num_items, dimension = (
        item_vectors.shape
    )

    print(
        f"Items: {num_items:,}"
    )

    print(
        f"Dimension: {dimension}"
    )

    # ---------------------------------------------------------
    # Load FAISS
    # ---------------------------------------------------------

    retriever = FaissRetriever.load(
        INDEX_PATH,
        METADATA_PATH,
    )

    if (
        retriever.num_items
        != num_items
    ):
        raise ValueError(
            "FAISS index size does not match "
            "item factor matrix."
        )

    # ---------------------------------------------------------
    # Sample queries
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    query_indices = rng.choice(
        num_items,
        size=min(
            NUM_QUERIES,
            num_items,
        ),
        replace=False,
    )

    queries = item_vectors[
        query_indices
    ]

    # ---------------------------------------------------------
    # Brute-force benchmark
    # ---------------------------------------------------------

    brute_force_results = []

    brute_force_start = (
        time.perf_counter()
    )

    for query in queries:
        ids, _ = brute_force_search(
            item_vectors,
            query,
            TOP_K,
        )

        brute_force_results.append(
            ids
        )

    brute_force_time = (
        time.perf_counter()
        - brute_force_start
    )

    # ---------------------------------------------------------
    # FAISS benchmark
    # ---------------------------------------------------------

    faiss_results = []

    faiss_start = (
        time.perf_counter()
    )

    for query in queries:
        ids, _ = retriever.search(
            query,
            TOP_K,
        )

        faiss_results.append(
            ids
        )

    faiss_time = (
        time.perf_counter()
        - faiss_start
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    recalls = [
        recall_at_k(
            expected,
            actual,
        )
        for expected, actual in zip(
            brute_force_results,
            faiss_results,
        )
    ]

    mean_recall = (
        sum(recalls)
        / len(recalls)
    )

    exact_matches = sum(
        expected == actual
        for expected, actual in zip(
            brute_force_results,
            faiss_results,
        )
    )

    print("\n=== Results ===")

    print(
        f"Queries: "
        f"{len(queries):,}"
    )

    print(
        f"Top-K: "
        f"{TOP_K}"
    )

    print(
        f"Mean Recall@{TOP_K}: "
        f"{mean_recall:.6f}"
    )

    print(
        f"Exact Top-K matches: "
        f"{exact_matches:,} / "
        f"{len(queries):,}"
    )

    print(
        f"\nBrute-force total: "
        f"{brute_force_time:.6f} sec"
    )

    print(
        f"FAISS total: "
        f"{faiss_time:.6f} sec"
    )

    print(
        f"Brute-force/query: "
        f"{brute_force_time / len(queries) * 1000:.4f} ms"
    )

    print(
        f"FAISS/query: "
        f"{faiss_time / len(queries) * 1000:.4f} ms"
    )

    speedup = (
        brute_force_time
        / faiss_time
        if faiss_time > 0
        else float("inf")
    )

    print(
        f"Speedup: "
        f"{speedup:.2f}x"
    )


if __name__ == "__main__":
    main()