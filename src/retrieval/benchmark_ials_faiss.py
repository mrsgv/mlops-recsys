from __future__ import annotations

import time

import numpy as np

from src.models.ials_baseline import (
    IALSRecommender,
)

from src.retrieval.ials_retriever import (
    IALSFaissRetriever,
)


MODEL_PATH = (
    "models/ials/ials_model.npz"
)

TOP_K = 10
NUM_USERS = 500
SEED = 42


def main() -> None:
    rng = np.random.default_rng(
        SEED
    )

    # ---------------------------------------------------------
    # Load iALS model
    # ---------------------------------------------------------

    with np.load(
        MODEL_PATH,
        allow_pickle=False,
    ) as data:
        user_factors = data[
            "user_factors"
        ].astype(np.float32)

        item_factors = data[
            "item_factors"
        ].astype(np.float32)

    num_users = user_factors.shape[0]

    users = rng.choice(
        num_users,
        size=min(
            NUM_USERS,
            num_users,
        ),
        replace=False,
    )

    # ---------------------------------------------------------
    # Load FAISS retriever
    # ---------------------------------------------------------

    faiss_retriever = (
        IALSFaissRetriever()
    )

    # ---------------------------------------------------------
    # Compare exact raw-factor ranking vs FAISS
    # ---------------------------------------------------------

    recalls = []

    exact_matches = 0

    start = time.perf_counter()

    for user_idx in users:

        user_vector = (
            user_factors[user_idx]
        )

        scores = (
            item_factors
            @ user_vector
        )

        expected = np.argsort(
            -scores
        )[:TOP_K]

        actual, _ = (
            faiss_retriever.faiss_index.search(
                user_vector,
                TOP_K,
            )
        )

        expected_set = set(
            expected.tolist()
        )

        actual_set = set(
            actual
        )

        recall = (
            len(
                expected_set
                & actual_set
            )
            / TOP_K
        )

        recalls.append(
            recall
        )

        if list(expected) == actual:
            exact_matches += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Users tested: {len(users):,}"
    )

    print(
        f"Mean Recall@{TOP_K}: "
        f"{np.mean(recalls):.6f}"
    )

    print(
        f"Exact Top-K matches: "
        f"{exact_matches:,} / "
        f"{len(users):,}"
    )

    print(
        f"Average query time: "
        f"{elapsed / len(users) * 1000:.4f} ms"
    )


if __name__ == "__main__":
    main()