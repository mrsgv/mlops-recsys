from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class NegativeSampler:
    """
    User-history-aware negative sampler.

    For every positive interaction, samples N items that the
    user has NOT interacted with in the training data.
    """

    def __init__(
        self,
        num_items: int,
        num_negatives: int = 5,
        seed: int = 42,
    ) -> None:
        if num_items <= 0:
            raise ValueError(
                "num_items must be greater than zero."
            )

        if num_negatives <= 0:
            raise ValueError(
                "num_negatives must be greater than zero."
            )

        self.num_items = num_items
        self.num_negatives = num_negatives
        self.seed = seed

        self.rng = np.random.default_rng(seed)

    @staticmethod
    def build_user_history(
        train_df: pd.DataFrame,
    ) -> dict[int, set[int]]:
        """Build user -> set of interacted training items."""
        if train_df.empty:
            raise ValueError(
                "train_df must not be empty."
            )

        required_columns = {
            "user_idx",
            "item_idx",
        }

        missing = required_columns - set(
            train_df.columns
        )

        if missing:
            raise ValueError(
                "train_df is missing required columns: "
                f"{sorted(missing)}"
            )

        return (
            train_df
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def sample_for_interactions(
        self,
        train_df: pd.DataFrame,
        user_history: Mapping[int, set[int]] | None = None,
    ) -> np.ndarray:
        """
        Sample negatives for every positive interaction.

        Returns
        -------
        np.ndarray
            Shape: (num_interactions, num_negatives)

        Every sampled negative is guaranteed not to appear
        in the corresponding user's training history.
        """
        if train_df.empty:
            raise ValueError(
                "train_df must not be empty."
            )

        if user_history is None:
            user_history = self.build_user_history(
                train_df
            )

        negatives = np.empty(
            (
                len(train_df),
                self.num_negatives,
            ),
            dtype=np.int64,
        )

        all_items = np.arange(
            self.num_items,
            dtype=np.int64,
        )

        candidate_cache: dict[
            int,
            np.ndarray,
        ] = {}

        for row_idx, user_idx in enumerate(
            train_df["user_idx"].to_numpy()
        ):
            user_idx = int(user_idx)

            if user_idx not in user_history:
                raise ValueError(
                    f"Missing interaction history for "
                    f"user {user_idx}."
                )

            if user_idx not in candidate_cache:
                seen_items = user_history[user_idx]

                if len(seen_items) >= self.num_items:
                    raise ValueError(
                        f"User {user_idx} has interacted "
                        "with every item; no negative "
                        "sample is available."
                    )

                mask = np.ones(
                    self.num_items,
                    dtype=bool,
                )

                if seen_items:
                    seen_array = np.fromiter(
                        seen_items,
                        dtype=np.int64,
                    )

                    mask[seen_array] = False

                candidate_cache[user_idx] = (
                    all_items[mask]
                )

            candidates = candidate_cache[user_idx]

            if len(candidates) < self.num_negatives:
                sampled = self.rng.choice(
                    candidates,
                    size=self.num_negatives,
                    replace=True,
                )
            else:
                sampled = self.rng.choice(
                    candidates,
                    size=self.num_negatives,
                    replace=False,
                )

            negatives[row_idx] = sampled

        return negatives