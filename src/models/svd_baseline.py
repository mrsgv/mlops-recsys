import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


DATA_PATH = "data/processed/video_games.parquet"
MODEL_DIR = "models/svd"
RECOMMENDATIONS_PATH = "data/predictions/svd_top10.parquet"

N_FACTORS = 50
TOP_K = 10


def main():

    print("\n=== Loading Processed Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Products: {df['item_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Chronological train/test split
    # Last interaction of every user is held out for evaluation.
    # ---------------------------------------------------------

    print("\n=== Creating Train/Test Split ===")

    df = df.sort_values(["user_idx", "timestamp"])

    test = df.groupby("user_idx").tail(1)
    train = df.drop(test.index)

    print(f"Training interactions: {len(train):,}")
    print(f"Test interactions: {len(test):,}")

    # ---------------------------------------------------------
    # Build training user-item matrix
    # ---------------------------------------------------------

    print("\n=== Building Training User-Item Matrix ===")

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    matrix = csr_matrix(
        (
            train["rating"].values,
            (train["user_idx"].values, train["item_idx"].values),
        ),
        shape=(n_users, n_items),
    )

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero entries: {matrix.nnz:,}")

    total_entries = matrix.shape[0] * matrix.shape[1]
    sparsity = 1 - (matrix.nnz / total_entries)

    print(f"Sparsity: {sparsity:.4%}")

    # ---------------------------------------------------------
    # SVD
    # ---------------------------------------------------------

    print(f"\n=== Computing SVD ({N_FACTORS} factors) ===")

    start_time = time.time()

    U, sigma, Vt = svds(matrix.astype(float), k=N_FACTORS)

    # svds returns singular values in ascending order.
    # Reverse them so largest factors come first.
    order = np.argsort(sigma)[::-1]

    sigma = sigma[order]
    U = U[:, order]
    Vt = Vt[order, :]

    elapsed = time.time() - start_time

    print(f"SVD completed in {elapsed:.2f} seconds")

    print("\n=== Factor Shapes ===")
    print(f"U:     {U.shape}")
    print(f"Sigma: {sigma.shape}")
    print(f"Vt:    {Vt.shape}")

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\n=== Generating Predictions ===")

    predictions = (U * sigma) @ Vt

    print(f"Prediction matrix shape: {predictions.shape}")

    # ---------------------------------------------------------
    # Remove items already seen during training
    # ---------------------------------------------------------

    print("\n=== Masking Previously Seen Items ===")

    train_users = train["user_idx"].to_numpy()
    train_items = train["item_idx"].to_numpy()

    predictions[train_users, train_items] = -np.inf

    # ---------------------------------------------------------
    # Generate Top-K recommendations
    # ---------------------------------------------------------

    print("\n=== Generating Top-K Recommendations ===")

    rows = []

    for user_idx in range(n_users):

        user_scores = predictions[user_idx]

        top_indices = np.argpartition(
            user_scores,
            -TOP_K
        )[-TOP_K:]

        top_indices = top_indices[
            np.argsort(user_scores[top_indices])[::-1]
        ]

        for rank, item_idx in enumerate(top_indices, start=1):

            rows.append(
                {
                    "user_idx": user_idx,
                    "item_idx": int(item_idx),
                    "rank": rank,
                    "predicted_score": float(
                        user_scores[item_idx]
                    ),
                }
            )

    recommendations = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Save factors
    # ---------------------------------------------------------

    print("\n=== Saving SVD Factors ===")

    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)

    np.save(f"{MODEL_DIR}/U.npy", U)
    np.save(f"{MODEL_DIR}/sigma.npy", sigma)
    np.save(f"{MODEL_DIR}/Vt.npy", Vt)

    # ---------------------------------------------------------
    # Save recommendations
    # ---------------------------------------------------------

    print("\n=== Saving Recommendations ===")

    recommendations.to_parquet(
        RECOMMENDATIONS_PATH,
        index=False
    )

    print("\n=== Saved ===")
    print(f"Model: {MODEL_DIR}/")
    print(
        f"Recommendations: "
        f"{RECOMMENDATIONS_PATH}"
    )

    print("\n=== Sample Recommendations ===")

    print(
        recommendations
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()