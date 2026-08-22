"""
Matrix-factorisation + FAISS candidate retrieval.

This module used to be ``ials_retriever.py`` and was named for the only
model it was believed to support. That belief was wrong: the retrieval path
is a raw inner product between a user factor row and an item factor matrix,
which is precisely what ``IndexFlatIP`` computes, so it serves *any* model
that exposes ``user_factors`` and ``item_factors``.

That was verified rather than assumed. For ALS, BPR and LMF,
``argsort(user_vector @ item_factors.T)`` with seen-item filtering
reproduces the ``implicit`` library's own ``recommend()`` top-K exactly.
BPR appends one bias column to both factor matrices and LMF two; the inner
product absorbs them, so the dimension simply differs per family and
nothing else changes.

The practical consequence is that promoting a different model family
requires no serving change at all — only a different artifact path, which
now comes from the deployment manifest instead of a module constant.

Pipeline:

    user_idx
        -> user factor row
        -> FAISS candidate search
        -> drop items seen in training
        -> Top-K
        -> item_idx + parent_asin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import FaissRetriever


MODEL_PATH = "models/promoted/model.npz"

FAISS_INDEX_PATH = (
    "models/retrieval/faiss.index"
)

FAISS_METADATA_PATH = (
    "models/retrieval/index_metadata.json"
)

ITEM_MAPPING_PATH = (
    "data/processed/item_mapping.parquet"
)

INTERACTIONS_PATH = (
    "data/processed/video_games.parquet"
)


class FactorFaissRetriever:
    """
    Serve Top-K recommendations from factor matrices and a FAISS index.

    Model-agnostic by construction: it reads factors out of the ``.npz``
    artifact and never asks which family produced them.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        faiss_index_path: str = FAISS_INDEX_PATH,
        faiss_metadata_path: str = FAISS_METADATA_PATH,
        item_mapping_path: str = ITEM_MAPPING_PATH,
        interactions_path: str = INTERACTIONS_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        # Factors are loaded first, before the index and the 45 MB
        # interaction parquet. The model artifact is the cheapest thing to
        # check and the most likely to be wrong — a factorless or missing
        # model should be reported without first paying to read everything
        # else.
        self.user_factors, self.item_factors = (
            self._load_factors(
                self.model_path
            )
        )

        self.faiss_index = FaissRetriever.load(
            faiss_index_path,
            faiss_metadata_path,
        )

        self.item_mapping = (
            pd.read_parquet(
                item_mapping_path
            )
            .sort_values("item_idx")
            .reset_index(drop=True)
        )

        self.user_history = (
            self._load_user_history(
                interactions_path
            )
        )

        self._validate_consistency()

    @staticmethod
    def _load_factors(
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read user and item factors from a saved model artifact.

        A neighbourhood model's artifact has no factor arrays, so this is
        also the check that a non-servable model was never promoted by
        mistake — it fails loudly at startup rather than at first request.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            missing = [
                key
                for key in (
                    "user_factors",
                    "item_factors",
                )
                if key not in data
            ]

            if missing:
                raise ValueError(
                    f"Model artifact {path} has no "
                    f"{' or '.join(missing)}. Only "
                    "factor-based models can be served through "
                    "the FAISS retrieval path."
                )

            user_factors = np.asarray(
                data["user_factors"],
                dtype=np.float32,
            )

            item_factors = np.asarray(
                data["item_factors"],
                dtype=np.float32,
            )

        if user_factors.ndim != 2:
            raise ValueError(
                "user_factors must be 2D."
            )

        if item_factors.ndim != 2:
            raise ValueError(
                "item_factors must be 2D."
            )

        if (
            user_factors.shape[1]
            != item_factors.shape[1]
        ):
            raise ValueError(
                "User and item factor dimensions do not match."
            )

        return user_factors, item_factors

    @staticmethod
    def _load_user_history(
        path: str,
    ) -> dict[int, set[int]]:
        """
        Load all training interaction history.

        This is used to reproduce the same seen-item filtering
        semantics as the offline evaluation.
        """
        df = pd.read_parquet(
            path,
            columns=[
                "user_idx",
                "item_idx",
            ],
        )

        return (
            df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def _validate_consistency(self) -> None:
        num_users, user_dim = (
            self.user_factors.shape
        )

        num_items, item_dim = (
            self.item_factors.shape
        )

        if user_dim != item_dim:
            raise ValueError(
                "User and item factor dimensions differ."
            )

        if (
            self.faiss_index.dimension
            != item_dim
        ):
            raise ValueError(
                "FAISS index dimension "
                f"({self.faiss_index.dimension}) does not match "
                f"the model's factor dimension ({item_dim}). "
                "Rebuild the index from the promoted model."
            )

        if (
            self.faiss_index.num_items
            != num_items
        ):
            raise ValueError(
                "FAISS index size does not match "
                "the model's item factors."
            )

        if (
            len(self.item_mapping)
            != num_items
        ):
            raise ValueError(
                "Item mapping size does not match "
                "the model's item factors."
            )

        expected_item_ids = list(
            range(num_items)
        )

        actual_item_ids = (
            self.item_mapping[
                "item_idx"
            ].tolist()
        )

        if actual_item_ids != expected_item_ids:
            raise ValueError(
                "item_idx must be contiguous "
                "from zero."
            )

        if self.item_mapping[
            "parent_asin"
        ].duplicated().any():
            raise ValueError(
                "Duplicate parent_asin values."
            )

        if not self.user_history:
            raise ValueError(
                "User interaction history is empty."
            )

        if num_users <= 0:
            raise ValueError(
                "No user factors found."
            )

    @property
    def num_users(self) -> int:
        return self.user_factors.shape[0]

    @property
    def num_items(self) -> int:
        return self.item_factors.shape[0]

    @property
    def dimension(self) -> int:
        return self.user_factors.shape[1]

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        candidate_multiplier: int = 5,
    ) -> pd.DataFrame:
        """
        Generate Top-K recommendations after filtering
        training-seen items.

        We retrieve more than K candidates from FAISS so that
        filtering does not reduce the final result below K.
        """
        if not 0 <= user_idx < self.num_users:
            raise ValueError(
                f"user_idx={user_idx} is outside the valid "
                f"range [0, {self.num_users - 1}]."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be positive."
            )

        seen_items = self.user_history.get(
            user_idx,
            set(),
        )

        target_candidates = min(
            self.num_items,
            max(
                k,
                k * candidate_multiplier,
            ),
        )

        user_vector = self.user_factors[
            user_idx
        ]

        candidate_ids, candidate_scores = (
            self.faiss_index.search(
                user_vector,
                target_candidates,
            )
        )

        filtered = []

        for item_id, score in zip(
            candidate_ids,
            candidate_scores,
        ):
            if item_id in seen_items:
                continue

            filtered.append(
                (
                    item_id,
                    score,
                )
            )

            if len(filtered) >= k:
                break

        # In a pathological case where the first candidate
        # pool is dominated by seen items, progressively expand
        # the search until enough unseen items are found.
        if len(filtered) < k:
            expanded_k = min(
                self.num_items,
                max(
                    target_candidates * 2,
                    target_candidates + 100,
                ),
            )

            while (
                expanded_k > target_candidates
                and len(filtered) < k
            ):
                candidate_ids, candidate_scores = (
                    self.faiss_index.search(
                        user_vector,
                        expanded_k,
                    )
                )

                filtered = []

                for item_id, score in zip(
                    candidate_ids,
                    candidate_scores,
                ):
                    if item_id in seen_items:
                        continue

                    filtered.append(
                        (
                            item_id,
                            score,
                        )
                    )

                    if len(filtered) >= k:
                        break

                if len(filtered) >= k:
                    break

                if expanded_k == self.num_items:
                    break

                expanded_k = min(
                    self.num_items,
                    expanded_k * 2,
                )

        if len(filtered) < k:
            raise RuntimeError(
                f"Unable to produce {k} unseen recommendations "
                f"for user {user_idx}. "
                f"Only {len(filtered)} were available."
            )

        item_ids = [
            item_id
            for item_id, _ in filtered
        ]

        scores = [
            score
            for _, score in filtered
        ]

        result = (
            self.item_mapping
            .iloc[item_ids]
            [
                [
                    "item_idx",
                    "parent_asin",
                ]
            ]
            .copy()
        )

        result["score"] = scores

        result["rank"] = range(
            1,
            len(result) + 1,
        )

        return result[
            [
                "rank",
                "item_idx",
                "parent_asin",
                "score",
            ]
        ].reset_index(drop=True)
