"""
Model family registry.

Before this module the pipeline knew about exactly one trainable model. The
model name was a constant in ``train_ials.py``, the artifact path was a
constant in ``build_index.py``, and ``select_model.py`` compared two
hardcoded files. Adding a third model meant editing all three.

Here a model family is data instead: a name, how to build it from
hyperparameters, and whether the serving path can load the result. The
sweep, selection, indexing and publishing steps all read this registry, so
a new family becomes one entry rather than a change to four steps.

Deployability
-------------
The serving path is FAISS ``IndexFlatIP`` over item factors, scored against
a user factor row (see ``src/retrieval/factor_retriever.py``). Any model
that exposes ``user_factors`` and ``item_factors`` is therefore servable
unchanged, because ranking by raw inner product over those matrices is
exactly what the index does.

This was verified per family rather than assumed: for ALS, BPR and LMF,
``argsort(user_vector @ item_factors.T)`` with seen-item filtering
reproduces the ``implicit`` library's own ``recommend()`` top-K exactly.
BPR carries one extra bias column (dimension = factors + 1) and LMF two;
the inner product absorbs both, so no special-casing is needed.

Neighbourhood models (BM25, cosine, TF-IDF) have no factors. They are
registered because they are strong, nearly free baselines worth comparing
against, but they are marked undeployable so selection reports them without
ever promoting them — the same treatment the SVD baseline gets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix


# Hyperparameter name -> abbreviation used when building a variant slug.
#
# Slugs become directory names, so they have to be short, deterministic and
# filesystem-safe while still being readable in a leaderboard.
SLUG_ABBREVIATIONS = {
    "factors": "f",
    "regularization": "r",
    "iterations": "i",
    "alpha": "a",
    "learning_rate": "lr",
    "neg_prop": "np",
    "K": "k",
    "K1": "k1",
    "B": "b",
}

# Parameters that describe the run rather than the model, and so must not
# end up in the slug or the artifact identity.
NON_MODEL_PARAMS = {
    "random_state",
}


class UnknownFamilyError(KeyError):
    """Raised when a sweep references a family that is not registered."""


@dataclass(frozen=True)
class ModelFamily:
    """
    One trainable model family.

    Attributes
    ----------
    name
        Registry key, and the prefix of every variant slug.
    build
        Maps hyperparameters to an unfitted model instance.
    load
        Reads a saved artifact back into a model instance. Needed because
        ``implicit.als.AlternatingLeastSquares`` is a factory function, not
        a class, so the concrete class differs per family.
    deployable
        Whether the FAISS serving path can load this family. False means
        "comparable but never promotable".
    deployable_reason
        Why an undeployable family cannot be served. Recorded in the
        selection output so a skipped winner is explained rather than
        silently dropped.
    description
        One line for the comparison report.
    accepts_random_state
        Whether the constructor takes ``random_state``. The neighbourhood
        models are deterministic and reject it, so the sweep injects the
        seed only where it is meaningful rather than putting an unused
        parameter in every grid entry.
    """

    name: str
    build: Callable[..., Any]
    load: Callable[[str], Any]
    deployable: bool
    description: str
    deployable_reason: str | None = None
    accepts_random_state: bool = True


def _build_als(**params: Any) -> Any:
    import implicit

    return implicit.als.AlternatingLeastSquares(
        **params
    )


def _load_als(path: str) -> Any:
    import implicit.cpu.als

    return (
        implicit.cpu.als.AlternatingLeastSquares.load(
            path
        )
    )


def _build_bpr(**params: Any) -> Any:
    from implicit.bpr import (
        BayesianPersonalizedRanking,
    )

    return BayesianPersonalizedRanking(
        **params
    )


def _load_bpr(path: str) -> Any:
    from implicit.cpu.bpr import (
        BayesianPersonalizedRanking,
    )

    return BayesianPersonalizedRanking.load(
        path
    )


def _build_lmf(**params: Any) -> Any:
    from implicit.lmf import (
        LogisticMatrixFactorization,
    )

    return LogisticMatrixFactorization(
        **params
    )


def _load_lmf(path: str) -> Any:
    from implicit.cpu.lmf import (
        LogisticMatrixFactorization,
    )

    return LogisticMatrixFactorization.load(
        path
    )


def _build_bm25(**params: Any) -> Any:
    from implicit.nearest_neighbours import (
        BM25Recommender,
    )

    return BM25Recommender(**params)


def _load_bm25(path: str) -> Any:
    from implicit.nearest_neighbours import (
        BM25Recommender,
    )

    return BM25Recommender.load(path)


def _build_cosine(**params: Any) -> Any:
    from implicit.nearest_neighbours import (
        CosineRecommender,
    )

    return CosineRecommender(**params)


def _load_cosine(path: str) -> Any:
    from implicit.nearest_neighbours import (
        CosineRecommender,
    )

    return CosineRecommender.load(path)


NO_FACTORS_REASON = (
    "Neighbourhood model: it has no item factors, so no FAISS index "
    "can be built from it."
)


FAMILIES: dict[str, ModelFamily] = {
    "als": ModelFamily(
        name="als",
        build=_build_als,
        load=_load_als,
        deployable=True,
        description=(
            "Implicit-feedback alternating least squares. Confidence "
            "weighted by alpha."
        ),
    ),
    "bpr": ModelFamily(
        name="bpr",
        build=_build_bpr,
        load=_load_bpr,
        deployable=True,
        description=(
            "Bayesian personalised ranking. Pairwise ranking loss over "
            "sampled negatives."
        ),
    ),
    "lmf": ModelFamily(
        name="lmf",
        build=_build_lmf,
        load=_load_lmf,
        deployable=True,
        description=(
            "Logistic matrix factorisation. Models interaction "
            "probability rather than confidence."
        ),
    ),
    "bm25": ModelFamily(
        name="bm25",
        build=_build_bm25,
        load=_load_bm25,
        deployable=False,
        deployable_reason=NO_FACTORS_REASON,
        accepts_random_state=False,
        description=(
            "BM25-weighted item-item nearest neighbours. Very cheap, "
            "strong on sparse catalogues."
        ),
    ),
    "cosine": ModelFamily(
        name="cosine",
        build=_build_cosine,
        load=_load_cosine,
        deployable=False,
        deployable_reason=NO_FACTORS_REASON,
        accepts_random_state=False,
        description=(
            "Cosine item-item nearest neighbours."
        ),
    ),
}


def get_family(name: str) -> ModelFamily:
    """Look up a registered family, failing with the valid options."""
    if name not in FAMILIES:
        raise UnknownFamilyError(
            f"Unknown model family '{name}'. "
            "Registered families: "
            + ", ".join(sorted(FAMILIES))
        )

    return FAMILIES[name]


def _format_value(value: Any) -> str:
    """
    Format a hyperparameter for use inside a directory name.

    Floats that hold whole numbers are written without a trailing ``.0``
    so ``alpha=40.0`` and ``alpha=40`` produce the same slug and therefore
    the same artifact directory.
    """
    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return (
            str(value)
            .replace(".", "p")
            .replace("-", "neg")
        )

    return str(value)


def variant_slug(
    family: str,
    params: dict[str, Any],
) -> str:
    """
    Build a deterministic, readable directory name for one variant.

    Example
    -------
    ``als`` with ``factors=256, alpha=40, regularization=0.1,
    iterations=20`` becomes ``als-f256-r0p1-i20-a40``.

    Sorting is by the abbreviation table's order, not by dict order, so
    the same hyperparameters always produce the same slug regardless of how
    the grid was written.
    """
    ordered = [
        key
        for key in SLUG_ABBREVIATIONS
        if key in params
        and key not in NON_MODEL_PARAMS
    ]

    extras = sorted(
        key
        for key in params
        if key not in SLUG_ABBREVIATIONS
        and key not in NON_MODEL_PARAMS
    )

    parts = [family]

    for key in ordered + extras:
        abbreviation = (
            SLUG_ABBREVIATIONS.get(key, key)
        )

        parts.append(
            f"{abbreviation}"
            f"{_format_value(params[key])}"
        )

    return "-".join(parts)


def fit_model(
    family: str,
    params: dict[str, Any],
    user_item_matrix: csr_matrix,
) -> tuple[Any, float]:
    """
    Build and fit one variant.

    Returns
    -------
    tuple
        The fitted model and its wall-clock training time in seconds.
    """
    definition = get_family(family)

    model = definition.build(**params)

    start = time.perf_counter()

    model.fit(
        user_item_matrix,
        show_progress=False,
    )

    return (
        model,
        time.perf_counter() - start,
    )


def extract_factors(
    model: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Pull user and item factors out of a fitted model.

    Returns None for families that have no factors, which is how the sweep
    decides whether a FAISS index could be built from a candidate.
    """
    user_factors = getattr(
        model,
        "user_factors",
        None,
    )

    item_factors = getattr(
        model,
        "item_factors",
        None,
    )

    if (
        user_factors is None
        or item_factors is None
    ):
        return None

    return (
        np.asarray(
            user_factors,
            dtype=np.float32,
        ),
        np.asarray(
            item_factors,
            dtype=np.float32,
        ),
    )


def recommend_batch(
    model: Any,
    user_ids: np.ndarray,
    user_item_matrix: csr_matrix,
    k: int,
) -> dict[int, list[int]]:
    """
    Generate Top-K recommendations for many users at once.

    ``implicit`` supports a batched ``recommend`` call, which is roughly two
    orders of magnitude faster than looping per user and is what makes a
    full 94k-user evaluation take seconds instead of minutes.

    Items the user already interacted with in training are filtered, which
    matches the offline evaluation protocol and the serving path.
    """
    user_ids = np.asarray(
        user_ids,
        dtype=np.int32,
    )

    item_ids, _ = model.recommend(
        user_ids,
        user_item_matrix[user_ids],
        N=k,
        filter_already_liked_items=True,
    )

    return {
        int(user_id): [
            int(item_id)
            for item_id in row
        ]
        for user_id, row in zip(
            user_ids,
            item_ids,
        )
    }


def save_model(
    model: Any,
    path: str | Path,
) -> None:
    """Save a fitted model to its ``.npz`` artifact."""
    resolved = Path(path)

    resolved.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(str(resolved))


def load_model(
    family: str,
    path: str | Path,
) -> Any:
    """Load a saved artifact back into its family's model class."""
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {resolved}"
        )

    return get_family(family).load(
        str(resolved)
    )
