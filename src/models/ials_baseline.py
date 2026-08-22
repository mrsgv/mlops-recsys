from __future__ import annotations

import time
from pathlib import Path

import implicit
import implicit.cpu.als
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> csr_matrix:
    """
    Build the binary implicit-feedback user-item matrix.

    An observed interaction is represented by 1.0 regardless of its star
    rating, which is what makes this an implicit-feedback model.

    This is shared by training and by artifact evaluation so that a
    reloaded model is scored against exactly the same matrix it was
    trained on.
    """
    if train_df.empty:
        raise ValueError(
            "Training dataframe is empty."
        )

    if train_df["user_idx"].isna().any():
        raise ValueError(
            "Training data contains missing user_idx values."
        )

    if train_df["item_idx"].isna().any():
        raise ValueError(
            "Training data contains missing item_idx values."
        )

    users = train_df["user_idx"].to_numpy(
        dtype=np.int32
    )

    items = train_df["item_idx"].to_numpy(
        dtype=np.int32
    )

    values = np.ones(
        len(train_df),
        dtype=np.float32,
    )

    return csr_matrix(
        (
            values,
            (users, items),
        ),
        shape=(num_users, num_items),
        dtype=np.float32,
    )


def load_ials_model(
    path: str | Path,
) -> implicit.cpu.als.AlternatingLeastSquares:
    """
    Load a trained iALS model from its .npz artifact.

    ``implicit.als.AlternatingLeastSquares`` is a factory function rather
    than a class, so the concrete CPU class is used for loading.
    """
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"iALS model not found: {resolved}"
        )

    return (
        implicit.cpu.als.AlternatingLeastSquares.load(
            str(resolved)
        )
    )


class IALSRecommender:
    """
    Implicit-feedback ALS recommender.

    Training representation:
        observed user-item interaction -> 1.0

    The model is trained using the implicit library's
    AlternatingLeastSquares implementation.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 20,
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.model = (
            implicit.als.AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                random_state=random_state,
            )
        )

        self.user_item_matrix: csr_matrix | None = None

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
    ) -> IALSRecommender:
        """
        Rebuild a recommender around an already-trained artifact.

        Hyperparameters are read back from the artifact rather than
        re-declared, so an evaluation run cannot silently disagree with
        the run that produced the model.

        The training matrix must be supplied because implicit needs it to
        filter items the user already interacted with; it is not part of
        the saved artifact.
        """
        model = load_ials_model(path)

        # random_state is deliberately not read back: it only affects
        # factor initialisation during training, and implicit does not
        # guarantee the saved value round-trips as a plain integer.
        recommender = cls(
            factors=int(model.factors),
            regularization=float(
                model.regularization
            ),
            iterations=int(model.iterations),
            alpha=float(
                getattr(model, "alpha", 1.0)
            ),
        )

        recommender.model = model

        recommender.user_item_matrix = (
            build_user_item_matrix(
                train_df,
                num_users=num_users,
                num_items=num_items,
            )
        )

        return recommender

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
    ) -> float:
        """
        Fit iALS on binary user-item interactions.

        Returns
        -------
        float
            Training time in seconds.
        """
        user_item_matrix = build_user_item_matrix(
            train_df,
            num_users=num_users,
            num_items=num_items,
        )

        self.user_item_matrix = user_item_matrix

        start_time = time.perf_counter()

        # implicit expects a USER x ITEM matrix.
        self.model.fit(
            user_item_matrix,
            show_progress=True,
        )

        return time.perf_counter() - start_time

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
    ) -> tuple[list[int], list[float]]:
        """
        Generate Top-K recommendations for one user.
        """
        if self.user_item_matrix is None:
            raise RuntimeError(
                "Model must be fitted before recommendation."
            )

        if not 0 <= user_idx < self.user_item_matrix.shape[0]:
            raise ValueError(
                f"user_idx={user_idx} is outside the "
                "trained user range."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=k,
            filter_already_liked_items=True,
        )

        return (
            [int(item_id) for item_id in item_ids],
            [float(score) for score in scores],
        )

    def recommend_users(
        self,
        user_ids: list[int],
        k: int = 10,
    ) -> dict[int, list[int]]:
        """Generate Top-K item IDs for multiple users."""
        recommendations: dict[int, list[int]] = {}

        for user_idx in user_ids:
            item_ids, _ = self.recommend(
                int(user_idx),
                k,
            )

            recommendations[int(user_idx)] = item_ids

        return recommendations