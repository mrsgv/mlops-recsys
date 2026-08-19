from __future__ import annotations

from math import log2
from typing import Iterable, Mapping, Sequence


def _validate_k(k: int) -> None:
    """Validate the Top-K parameter."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")


def _get_relevant_set(relevant_items: Iterable[int]) -> set[int]:
    """Convert relevant items to a set for efficient membership checks."""
    return set(relevant_items)


def precision_at_k(
    recommendations: Sequence[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """
    Compute Precision@K for a single user.

    Precision@K = number of relevant recommended items in top-K / K.
    """
    _validate_k(k)

    recommended = list(recommendations)[:k]
    relevant = _get_relevant_set(relevant_items)

    if k == 0:
        return 0.0

    hits = sum(
        1 for item in recommended
        if item in relevant
    )

    return hits / k


def recall_at_k(
    recommendations: Sequence[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """
    Compute Recall@K for a single user.

    Recall@K = relevant recommended items in top-K /
               total relevant items.
    """
    _validate_k(k)

    recommended = list(recommendations)[:k]
    relevant = _get_relevant_set(relevant_items)

    if not relevant:
        return 0.0

    hits = sum(
        1 for item in recommended
        if item in relevant
    )

    return hits / len(relevant)


def hit_rate_at_k(
    recommendations: Sequence[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """
    Compute Hit Rate@K for a single user.

    Returns 1.0 if at least one relevant item appears
    in the top-K recommendations, otherwise 0.0.
    """
    _validate_k(k)

    recommended = list(recommendations)[:k]
    relevant = _get_relevant_set(relevant_items)

    return float(
        any(item in relevant for item in recommended)
    )


def ndcg_at_k(
    recommendations: Sequence[int],
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """
    Compute binary-relevance NDCG@K for a single user.
    """
    _validate_k(k)

    recommended = list(recommendations)[:k]
    relevant = _get_relevant_set(relevant_items)

    if not relevant:
        return 0.0

    # Discounted cumulative gain.
    dcg = 0.0

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    # Ideal DCG.
    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_top_k(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Iterable[int]],
    k: int,
) -> dict[str, float | int]:
    """
    Evaluate Top-K recommendations across all evaluation users.

    Parameters
    ----------
    recommendations:
        Mapping:
            user_idx -> ranked recommendation list

    ground_truth:
        Mapping:
            user_idx -> relevant item(s)

    k:
        Number of recommendations considered.

    Returns
    -------
    dict
        Aggregated metrics across users:
            users_evaluated
            precision_at_k
            recall_at_k
            hit_rate_at_k
            ndcg_at_k

    Notes
    -----
    Every user in ground_truth belongs to the evaluation population.
    If a user has no recommendation entry, their recommendation list
    is treated as empty, producing zero scores for that user.
    """
    _validate_k(k)

    if not ground_truth:
        raise ValueError(
            "ground_truth must contain at least one user."
        )

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    ndcg_scores = []

    for user_idx, relevant_items in ground_truth.items():
        user_recommendations = recommendations.get(
            user_idx,
            [],
        )

        precision_scores.append(
            precision_at_k(
                user_recommendations,
                relevant_items,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                user_recommendations,
                relevant_items,
                k,
            )
        )

        hit_rate_scores.append(
            hit_rate_at_k(
                user_recommendations,
                relevant_items,
                k,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                user_recommendations,
                relevant_items,
                k,
            )
        )

    num_users = len(ground_truth)

    return {
        "users_evaluated": num_users,
        f"precision_at_{k}": (
            sum(precision_scores) / num_users
        ),
        f"recall_at_{k}": (
            sum(recall_scores) / num_users
        ),
        f"hit_rate_at_{k}": (
            sum(hit_rate_scores) / num_users
        ),
        f"ndcg_at_{k}": (
            sum(ndcg_scores) / num_users
        ),
    }