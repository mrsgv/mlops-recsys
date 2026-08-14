import pandas as pd
import numpy as np


DATA_PATH = "data/processed/video_games.parquet"
RECOMMENDATIONS_PATH = "data/predictions/svd_top10.parquet"
OUTPUT_PATH = "data/predictions/svd_evaluation.csv"

TOP_K = 10


def main():

    print("\n=== Loading Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Products: {df['item_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Recreate the exact chronological test split
    # used by the SVD baseline.
    # ---------------------------------------------------------

    print("\n=== Creating Evaluation Split ===")

    df = df.sort_values(["user_idx", "timestamp"])

    test = df.groupby("user_idx").tail(1)

    print(f"Test interactions: {len(test):,}")
    print(f"Test users: {test['user_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Load recommendations
    # ---------------------------------------------------------

    print("\n=== Loading SVD Recommendations ===")

    recommendations = pd.read_parquet(
        RECOMMENDATIONS_PATH
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Users with recommendations: "
        f"{recommendations['user_idx'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Convert recommendations into a dictionary
    # ---------------------------------------------------------

    recommendation_map = (
        recommendations
        .sort_values(["user_idx", "rank"])
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    print("\n=== Evaluating ===")

    hits = 0
    ndcg_sum = 0.0
    users_evaluated = 0

    for row in test.itertuples(index=False):

        user_idx = row.user_idx
        actual_item = row.item_idx

        if user_idx not in recommendation_map:
            continue

        recommended = recommendation_map[user_idx][
            :TOP_K
        ]

        users_evaluated += 1

        if actual_item in recommended:

            hits += 1

            rank = recommended.index(actual_item) + 1

            ndcg_sum += (
                1.0 / np.log2(rank + 1)
            )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    if users_evaluated == 0:

        print("No users could be evaluated.")
        return

    hit_rate = hits / users_evaluated

    # With one held-out relevant item per user,
    # Precision@K = Hit Rate@K / K
    precision = hit_rate / TOP_K

    # With one held-out relevant item per user,
    # Recall@K = Hit Rate@K
    recall = hit_rate

    ndcg = ndcg_sum / users_evaluated

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n=== SVD Baseline Results ===")

    print(
        f"Users evaluated: {users_evaluated:,}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.6f}"
    )

    print(
        f"Recall@{TOP_K}:    "
        f"{recall:.6f}"
    )

    print(
        f"Hit Rate@{TOP_K}:  "
        f"{hit_rate:.6f}"
    )

    print(
        f"NDCG@{TOP_K}:      "
        f"{ndcg:.6f}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "SVD",
                "factors": 50,
                "users_evaluated": users_evaluated,
                "precision_at_10": precision,
                "recall_at_10": recall,
                "hit_rate_at_10": hit_rate,
                "ndcg_at_10": ndcg,
            }
        ]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n=== Saved ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()