"""
Model selection and promotion.

Training and evaluation produce candidates; this step decides which one is
promoted. It is deliberately a separate pipeline stage so the decision is
recorded as an artifact rather than being implied by whichever script ran
last.

Two properties decide promotion:

primary metric
    Candidates are ranked on Recall@10, the metric the project compares
    all recommenders on.

deployability
    A candidate is only promotable if the serving path can actually load
    it. That means the artifact must contain user and item factor matrices,
    because the FAISS index is an inner-product index over item factors and
    the retriever scores users with a user factor row. Any matrix
    factorisation family qualifies — ALS, BPR and LMF all do. Neighbourhood
    models (BM25, cosine) and the SVD baseline have no factors, so they are
    ranked and reported without ever being promoted.

When the best-scoring candidate is not deployable, that is reported loudly
rather than silently ignored.

Promotion
---------
Selection does not merely record a preference; it makes the winner
canonical by copying its artifact to ``models/promoted/model.npz``. That
matters because index building, the Docker image and the serving config
previously each hardcoded ``models/ials/ials_model.npz``, which meant a
different winner would have been "selected" while iALS carried on being
served. One canonical path removes the possibility of that disagreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


CANDIDATES_GLOB = (
    "models/candidates/*/evaluation.json"
)

IALS_EVALUATION_PATH = (
    "models/ials/evaluation.json"
)

SVD_EVALUATION_PATH = (
    "data/predictions/svd_evaluation.csv"
)

SELECTION_PATH = Path(
    "models/deployment/selected_model.json"
)

PROMOTED_DIR = Path(
    "models/promoted"
)

PROMOTED_MODEL_PATH = (
    PROMOTED_DIR / "model.npz"
)

PROMOTION_RECORD_PATH = (
    PROMOTED_DIR / "promotion.json"
)

PRIMARY_METRIC = "recall_at_10"


class ModelSelectionError(Exception):
    """Raised when no candidate can be promoted."""


def load_candidate_record(
    path: str | Path,
) -> dict[str, object] | None:
    """
    Load one candidate written by the sweep step.

    Returns None when the file is absent so a partially built
    ``models/candidates`` tree does not fail the pipeline.
    """
    resolved = Path(path)

    if not resolved.exists():
        return None

    record = json.loads(
        resolved.read_text()
    )

    candidate: dict[str, object] = {
        "name": record.get(
            "name",
            resolved.parent.name,
        ),
        "family": record.get("family"),
        "model_type": record.get(
            "model_type",
            record.get("family", "unknown"),
        ),
        "deployable": bool(
            record.get("deployable", False)
        ),
        "metrics": record.get("metrics", {}),
        "params": record.get("params", {}),
        "users_evaluated": record.get(
            "users_evaluated"
        ),
        "embedding_dimension": record.get(
            "embedding_dimension"
        ),
        "mlflow": record.get("mlflow", {}),
        "artifacts": record.get(
            "artifacts",
            {},
        ),
        "source": str(resolved),
    }

    if "deployable_reason" in record:
        candidate["deployable_reason"] = (
            record["deployable_reason"]
        )

    return candidate


def discover_candidates(
    pattern: str = CANDIDATES_GLOB,
    root: str | Path = ".",
) -> list[dict[str, object]]:
    """
    Find every candidate the sweep step produced.

    Discovery is a glob rather than a list of known models. That is the
    whole point: adding a model family must not require editing this file,
    otherwise the comparison silently stops covering new work.

    Results are sorted by path so the pipeline output is deterministic.
    """
    candidates = []

    for path in sorted(
        Path(root).glob(pattern)
    ):
        candidate = load_candidate_record(
            path
        )

        if candidate is not None:
            candidates.append(candidate)

    return candidates


def load_ials_candidate(
    path: str = IALS_EVALUATION_PATH,
) -> dict[str, object] | None:
    """
    Load the legacy single-model iALS evaluation output.

    Kept so a repository that has run the old ``train_ials`` step but not
    the sweep still produces a comparison instead of "no candidates".
    """
    resolved = Path(path)

    if not resolved.exists():
        return None

    record = json.loads(
        resolved.read_text()
    )

    return {
        "name": "ials",
        "family": "als",
        "model_type": record.get(
            "model_type",
            "ials",
        ),
        "deployable": bool(
            record.get("deployable", False)
        ),
        "metrics": record.get("metrics", {}),
        "params": record.get("params", {}),
        "users_evaluated": record.get(
            "users_evaluated"
        ),
        "mlflow": record.get("mlflow", {}),
        "artifacts": record.get(
            "artifacts",
            {},
        ),
        "source": str(resolved),
    }


def load_svd_candidate(
    path: str = SVD_EVALUATION_PATH,
) -> dict[str, object] | None:
    """
    Load the SVD baseline from its evaluation CSV.

    SVD is included so the promotion decision shows a comparison rather
    than a single candidate, but it has no FAISS retrieval path and so is
    never deployable.
    """
    resolved = Path(path)

    if not resolved.exists():
        return None

    with resolved.open() as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    row = rows[0]

    metrics = {
        key: float(value)
        for key, value in row.items()
        if key
        not in {
            "model",
            "factors",
            "users_evaluated",
        }
        and value not in (None, "")
    }

    return {
        "name": "svd",
        "family": "svd",
        "model_type": "svd",
        "deployable": False,
        "deployable_reason": (
            "No FAISS retrieval path: the index is built from "
            "factor matrices this artifact does not provide."
        ),
        "metrics": metrics,
        "params": {
            "factors": row.get("factors"),
        },
        "users_evaluated": int(
            float(
                row.get(
                    "users_evaluated",
                    0,
                )
            )
        ),
        "mlflow": {},
        "artifacts": {},
        "source": str(resolved),
    }


def rank_candidates(
    candidates: list[dict[str, object]],
    primary_metric: str = PRIMARY_METRIC,
) -> list[dict[str, object]]:
    """
    Rank candidates by the primary metric, best first.

    Candidates missing the primary metric sort last rather than crashing
    the pipeline; the ranking output makes the omission visible.
    """

    def sort_key(
        candidate: dict[str, object],
    ) -> tuple[int, float, str]:
        metrics = candidate.get("metrics", {})

        name = str(candidate.get("name", ""))

        if primary_metric not in metrics:
            return (1, 0.0, name)

        return (
            0,
            -float(metrics[primary_metric]),
            name,
        )

    return sorted(
        candidates,
        key=sort_key,
    )


def select_model(
    candidates: list[dict[str, object]],
    primary_metric: str = PRIMARY_METRIC,
) -> dict[str, object]:
    """
    Choose the model to promote and explain the choice.

    Raises
    ------
    ModelSelectionError
        If there are no candidates, or none of them is deployable.
    """
    if not candidates:
        raise ModelSelectionError(
            "No model candidates were found. Run training and "
            "evaluation first."
        )

    ranked = rank_candidates(
        candidates,
        primary_metric,
    )

    deployable = [
        candidate
        for candidate in ranked
        if candidate.get("deployable")
        and primary_metric
        in candidate.get("metrics", {})
    ]

    if not deployable:
        raise ModelSelectionError(
            "No deployable candidate scored on "
            f"{primary_metric}. Candidates considered: "
            + ", ".join(
                str(candidate["name"])
                for candidate in ranked
            )
        )

    selected = deployable[0]

    best_overall = ranked[0]

    notes = []

    if best_overall["name"] != selected["name"]:
        notes.append(
            f"{best_overall['name']} scored higher on "
            f"{primary_metric} but is not deployable: "
            + str(
                best_overall.get(
                    "deployable_reason",
                    "no serving path.",
                )
            )
        )

    return {
        "primary_metric": primary_metric,
        "selected": {
            "name": selected["name"],
            "family": selected.get("family"),
            "model_type": selected["model_type"],
            "params": selected.get(
                "params",
                {},
            ),
            "metrics": selected["metrics"],
            "users_evaluated": selected.get(
                "users_evaluated"
            ),
            "embedding_dimension": (
                selected.get(
                    "embedding_dimension"
                )
            ),
            "mlflow": selected.get(
                "mlflow",
                {},
            ),
            "artifacts": selected.get(
                "artifacts",
                {},
            ),
            "source": selected.get("source"),
        },
        "comparison": [
            {
                "name": candidate["name"],
                "family": candidate.get(
                    "family"
                ),
                "model_type": candidate[
                    "model_type"
                ],
                "deployable": bool(
                    candidate.get("deployable")
                ),
                primary_metric: (
                    candidate.get(
                        "metrics",
                        {},
                    ).get(primary_metric)
                ),
            }
            for candidate in ranked
        ],
        "notes": notes,
    }


def resolve_artifact(
    selected: dict[str, object],
) -> Path:
    """
    Find the artifact file belonging to the promoted candidate.

    The sweep records it under ``artifacts.model``; the legacy iALS step
    used ``artifacts.ials_model``. Both are accepted so an older
    repository state still promotes cleanly.
    """
    artifacts = selected.get(
        "artifacts",
        {},
    )

    for key in (
        "model",
        "ials_model",
    ):
        value = artifacts.get(key)

        if value:
            resolved = Path(str(value))

            if not resolved.exists():
                raise ModelSelectionError(
                    "The promoted candidate's artifact is missing: "
                    f"{resolved}. Re-run the sweep, or 'dvc pull'."
                )

            return resolved

    raise ModelSelectionError(
        "The promoted candidate records no model artifact, so it "
        "cannot be staged for serving."
    )


def promote_artifact(
    selection: dict[str, object],
    destination: Path = PROMOTED_MODEL_PATH,
    record_path: Path = PROMOTION_RECORD_PATH,
) -> dict[str, object]:
    """
    Copy the winning artifact to the canonical serving path.

    Copying rather than symlinking is deliberate: the Docker build context
    and DVC both need real bytes at a stable path, and a symlink into a
    per-variant directory would break as soon as the sweep is re-run with a
    different grid.
    """
    selected = selection["selected"]

    source = resolve_artifact(selected)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)

    record = {
        "promoted_from": str(source),
        "promoted_to": str(destination),
        "name": selected.get("name"),
        "family": selected.get("family"),
        "model_type": selected.get(
            "model_type"
        ),
        "params": selected.get("params", {}),
        "primary_metric": selection.get(
            "primary_metric"
        ),
        "metrics": selected.get("metrics", {}),
        "mlflow": selected.get("mlflow", {}),
    }

    record_path.write_text(
        json.dumps(record, indent=2) + "\n"
    )

    return record


def collect_candidates(
    candidates_glob: str = CANDIDATES_GLOB,
    ials_path: str = IALS_EVALUATION_PATH,
    svd_path: str = SVD_EVALUATION_PATH,
) -> list[dict[str, object]]:
    """
    Load every candidate whose evaluation output exists.

    Sweep candidates come first, then the two legacy single-model records.
    A sweep candidate that happens to share a name with a legacy record
    wins, so re-running the sweep supersedes stale output rather than
    competing with it.
    """
    candidates = discover_candidates(
        candidates_glob
    )

    if candidates:
        print(
            f"Discovered {len(candidates)} sweep "
            f"candidate(s) under {candidates_glob}"
        )
    else:
        print(
            f"No sweep candidates found at {candidates_glob}; "
            "falling back to legacy single-model records."
        )

    seen = {
        str(candidate["name"])
        for candidate in candidates
    }

    for loader, path in (
        (load_ials_candidate, ials_path),
        (load_svd_candidate, svd_path),
    ):
        candidate = loader(path)

        if candidate is None:
            print(
                f"No candidate found at {path}; skipping."
            )

            continue

        if str(candidate["name"]) in seen:
            print(
                f"Skipping {path}: a sweep candidate already "
                f"provides '{candidate['name']}'."
            )

            continue

        print(
            f"Loaded candidate '{candidate['name']}' "
            f"from {path}"
        )

        candidates.append(candidate)

    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the model to promote for deployment."
        )
    )

    parser.add_argument(
        "--output",
        default=str(SELECTION_PATH),
        help=(
            "Where to write the selection record."
        ),
    )

    parser.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC,
        help=(
            "Metric used to rank candidates."
        ),
    )

    parser.add_argument(
        "--promoted-model",
        default=str(PROMOTED_MODEL_PATH),
        help=(
            "Canonical path the winning artifact is copied to."
        ),
    )

    parser.add_argument(
        "--no-promote",
        action="store_true",
        help=(
            "Write the selection record without copying the winning "
            "artifact. Useful for inspecting a decision."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Model Selection")
    print("=" * 60)

    print("\n=== Collecting Candidates ===")

    candidates = collect_candidates()

    print("\n=== Selecting ===")

    selection = select_model(
        candidates,
        primary_metric=args.primary_metric,
    )

    print("\n=== Comparison ===")

    for entry in selection["comparison"]:
        score = entry[args.primary_metric]

        formatted = (
            f"{score:.6f}"
            if isinstance(score, float)
            else "n/a"
        )

        print(
            f"{entry['name']:<34} "
            f"{args.primary_metric}={formatted} "
            f"deployable={entry['deployable']}"
        )

    for note in selection["notes"]:
        print(f"\nNOTE: {note}")

    selected = selection["selected"]

    print("\n=== Selected ===")
    print(f"Model: {selected['name']}")
    print(f"Family: {selected.get('family')}")
    print(f"Params: {selected.get('params')}")

    print(
        f"MLflow run: "
        f"{selected['mlflow'].get('run_id', 'unknown')}"
    )

    if args.no_promote:
        print(
            "\nPromotion skipped (--no-promote); the canonical "
            "artifact was not updated."
        )
    else:
        promotion = promote_artifact(
            selection,
            destination=Path(
                args.promoted_model
            ),
        )

        selection["promotion"] = promotion

        print(
            f"\nPromoted artifact: "
            f"{promotion['promoted_from']} -> "
            f"{promotion['promoted_to']}"
        )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            selection,
            indent=2,
        )
        + "\n"
    )

    print(f"\nSelection written to: {output_path}")

    print("\n=== Selection Complete ===")


if __name__ == "__main__":
    main()
