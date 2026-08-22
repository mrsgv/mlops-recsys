"""
Model selection for deployment.

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
    it. Today that means iALS: the FAISS index is built from iALS item
    factors, and the retriever scores users with iALS user factors. The
    SVD baseline is comparable but not servable, so it is ranked and
    reported without ever being promoted.

When the best-scoring candidate is not deployable, that is reported loudly
rather than silently ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IALS_EVALUATION_PATH = (
    "models/ials/evaluation.json"
)

SVD_EVALUATION_PATH = (
    "data/predictions/svd_evaluation.csv"
)

SELECTION_PATH = Path(
    "models/deployment/selected_model.json"
)

PRIMARY_METRIC = "recall_at_10"


class ModelSelectionError(Exception):
    """Raised when no candidate can be promoted."""


def load_ials_candidate(
    path: str = IALS_EVALUATION_PATH,
) -> dict[str, object] | None:
    """
    Load the iALS candidate from the evaluation step's output.

    Returns None when the file is absent so selection can report a clear
    "no candidates" error rather than failing on a missing path.
    """
    resolved = Path(path)

    if not resolved.exists():
        return None

    record = json.loads(
        resolved.read_text()
    )

    return {
        "name": "ials",
        "model_type": record.get(
            "model_type",
            "ials",
        ),
        "deployable": bool(
            record.get("deployable", False)
        ),
        "metrics": record.get("metrics", {}),
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
        "model_type": "svd",
        "deployable": False,
        "deployable_reason": (
            "No FAISS retrieval path: the index is built from iALS "
            "item factors."
        ),
        "metrics": metrics,
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
    ) -> tuple[int, float]:
        metrics = candidate.get("metrics", {})

        if primary_metric not in metrics:
            return (1, 0.0)

        return (
            0,
            -float(metrics[primary_metric]),
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
            "model_type": selected["model_type"],
            "metrics": selected["metrics"],
            "users_evaluated": selected.get(
                "users_evaluated"
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
                "model_type": candidate["model_type"],
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


def collect_candidates(
    ials_path: str = IALS_EVALUATION_PATH,
    svd_path: str = SVD_EVALUATION_PATH,
) -> list[dict[str, object]]:
    """Load every candidate whose evaluation output exists."""
    candidates = []

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
            f"{entry['name']:<12} "
            f"{args.primary_metric}={formatted} "
            f"deployable={entry['deployable']}"
        )

    for note in selection["notes"]:
        print(f"\nNOTE: {note}")

    selected = selection["selected"]

    print("\n=== Selected ===")
    print(f"Model: {selected['name']}")

    print(
        f"MLflow run: "
        f"{selected['mlflow'].get('run_id', 'unknown')}"
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
