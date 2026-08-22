"""
MLflow ``pyfunc`` wrapper for the recommendation model.

Why a wrapper exists at all
---------------------------
The trained ``.npz`` is not the deployable thing. Serving a recommendation
needs five artifacts together: the factor matrices, the FAISS index, the
index metadata, the item mapping (to turn ``item_idx`` back into a
``parent_asin``) and the interaction history (to filter items a user has
already seen). Logging only the ``.npz`` — which is what the pipeline did
before — uploads about a fifth of a deployment and leaves the other four
paths to be hand-wired at the far end.

A ``pyfunc`` model bundles all five into a single versioned object with an
input schema and a pinned environment, so one registry version is the whole
deployable unit and ``mlflow.pyfunc.load_model`` reproduces serving exactly.

There is no new retrieval logic here. ``FactorFaissRetriever`` is the same
class the FastAPI service uses; this only re-points it at the artifact paths
MLflow unpacks at load time. Reimplementing scoring for the registry would
create a second code path that could drift from the served one.

This module imports MLflow at module scope, which is safe because the
serving image copies only ``src/serving`` and ``src/retrieval`` — nothing
here is ever imported by the container.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd


DEFAULT_K = 10

# Artifact keys: the names the model is logged with and the names
# load_context reads back. Defined once so the two ends cannot drift.
ARTIFACT_KEYS = (
    "model",
    "faiss_index",
    "faiss_metadata",
    "item_mapping",
    "interactions",
)


def build_input_example() -> pd.DataFrame:
    """
    Build the input example logged with the model.

    MLflow infers the signature from this, and it doubles as usage
    documentation on the model's registry page.
    """
    return pd.DataFrame(
        {
            "user_idx": [0, 1],
            "k": [
                DEFAULT_K,
                DEFAULT_K,
            ],
        }
    )


def build_output_example() -> pd.DataFrame:
    """Shape of the returned recommendations, for signature inference."""
    return pd.DataFrame(
        {
            "user_idx": [0],
            "rank": [1],
            "item_idx": [0],
            "parent_asin": ["B0000000AA"],
            "score": [0.0],
        }
    )


class RecommenderPythonModel(
    mlflow.pyfunc.PythonModel
):
    """
    Serve Top-K recommendations through the MLflow ``pyfunc`` interface.

    Input
    -----
    A DataFrame with a ``user_idx`` column and an optional ``k`` column.
    When ``k`` is absent, ``DEFAULT_K`` is used.

    Output
    ------
    A DataFrame of ``user_idx, rank, item_idx, parent_asin, score``, one row
    per recommendation, concatenated across the requested users.
    """

    def __init__(self) -> None:
        self.retriever: Any = None

    def load_context(
        self,
        context: Any,
    ) -> None:
        """Rebuild the serving retriever from the unpacked artifacts."""
        from src.retrieval.factor_retriever import (
            FactorFaissRetriever,
        )

        artifacts = context.artifacts

        missing = [
            key
            for key in ARTIFACT_KEYS
            if key not in artifacts
        ]

        if missing:
            raise ValueError(
                "Model is missing required artifacts: "
                f"{sorted(missing)}"
            )

        self.retriever = FactorFaissRetriever(
            model_path=artifacts["model"],
            faiss_index_path=artifacts[
                "faiss_index"
            ],
            faiss_metadata_path=artifacts[
                "faiss_metadata"
            ],
            item_mapping_path=artifacts[
                "item_mapping"
            ],
            interactions_path=artifacts[
                "interactions"
            ],
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Score every requested user.

        A per-user failure is not swallowed: an out-of-range ``user_idx``
        raises, because silently returning fewer rows than were asked for
        would make a broken request look like an empty result.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Model context was not loaded."
            )

        frame = pd.DataFrame(model_input)

        if "user_idx" not in frame.columns:
            raise ValueError(
                "Input must contain a 'user_idx' column."
            )

        default_k = int(
            (params or {}).get(
                "k",
                DEFAULT_K,
            )
        )

        if "k" in frame.columns:
            ks = (
                frame["k"]
                .fillna(default_k)
                .astype(int)
                .tolist()
            )
        else:
            ks = [default_k] * len(frame)

        outputs = []

        for user_idx, k in zip(
            frame["user_idx"].astype(int),
            ks,
        ):
            result = self.retriever.recommend(
                user_idx=int(user_idx),
                k=int(k),
            )

            result.insert(
                0,
                "user_idx",
                int(user_idx),
            )

            outputs.append(result)

        if not outputs:
            return (
                build_output_example()
                .iloc[0:0]
            )

        return pd.concat(
            outputs,
            ignore_index=True,
        )
