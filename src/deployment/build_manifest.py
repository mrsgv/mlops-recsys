"""
Build the deployment manifest.

The manifest is the hand-off between the training pipeline and serving. It
answers, for a running API instance: which model is this, which MLflow run
produced it, which FAISS index and dataset version does it correspond to,
and which files make up the deployment bundle.

Deliberately, it does not invent its own artifact paths. The keys under
``serving_env`` are exactly the environment variables ``src/serving/config.py``
reads, so a Cloud Run service can be configured straight from this file
without the two drifting apart.

DVC remains the versioning mechanism, so artifact identity is recorded as
the DVC content hash rather than a hash computed here. A missing pointer is
reported as a warning — or a hard failure under ``--require-dvc`` — because
an unversioned artifact cannot be reproduced later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


SELECTION_PATH = (
    "models/deployment/selected_model.json"
)

INDEX_METADATA_PATH = (
    "models/retrieval/index_metadata.json"
)

MANIFEST_PATH = Path(
    "models/deployment/deployment_manifest.json"
)

SCHEMA_VERSION = 1

# Artifact role -> (path on disk, DVC pointer that versions it).
#
# The roles match the artifact_paths keys in src/serving/config.py.
#
# The model role is deliberately named for its function, not its family.
# It used to be "ials_model" pointing at models/ials/ials_model.npz, which
# meant the manifest asserted a family the pipeline is no longer entitled to
# assume: selection can promote ALS, BPR or LMF. The selection step stages
# whichever won to this one canonical path.
ARTIFACTS = {
    "promoted_model": (
        "models/promoted/model.npz",
        "models/promoted.dvc",
    ),
    "faiss_index": (
        "models/retrieval/faiss.index",
        "models/retrieval.dvc",
    ),
    "faiss_metadata": (
        "models/retrieval/index_metadata.json",
        "models/retrieval.dvc",
    ),
    "item_mapping": (
        "data/processed/item_mapping.parquet",
        "data/processed/item_mapping.parquet.dvc",
    ),
    "interactions": (
        "data/processed/video_games.parquet",
        "data/processed/video_games.parquet.dvc",
    ),
}

RAW_DATASET = (
    "data/raw/Video_Games.csv.gz",
    "data/raw/Video_Games.csv.gz.dvc",
)

# Files Cloud Run needs. The interaction dataset is included because the
# retriever filters items each user has already seen.
BUNDLE_ROLES = [
    "promoted_model",
    "faiss_index",
    "faiss_metadata",
    "item_mapping",
    "interactions",
]


class ManifestError(Exception):
    """Raised when the manifest cannot be built."""


def read_dvc_pointer(
    path: str | Path,
) -> dict[str, object] | None:
    """
    Read the content hash out of a ``.dvc`` pointer file.

    Returns None when the pointer does not exist, which is how an
    unversioned artifact is detected.
    """
    resolved = Path(path)

    if not resolved.exists():
        return None

    document = yaml.safe_load(
        resolved.read_text()
    )

    outs = (document or {}).get("outs") or []

    if not outs:
        raise ManifestError(
            f"DVC pointer has no outputs: {resolved}"
        )

    out = outs[0]

    pointer = {
        "file": str(resolved),
        "md5": out.get("md5"),
        "size": out.get("size"),
    }

    # A directory output carries nfiles. Spark writes both the interaction
    # dataset and the item mapping as directories, which the deployment
    # bundle has to copy recursively rather than as single files.
    if "nfiles" in out:
        pointer["nfiles"] = out["nfiles"]

    return pointer


def describe_artifact(
    path: str,
    dvc_file: str,
) -> dict[str, object]:
    """Describe one artifact: where it lives and which version it is."""
    resolved = Path(path)

    pointer = read_dvc_pointer(dvc_file)

    return {
        "path": path,
        "exists": resolved.exists(),
        "is_directory": resolved.is_dir(),
        "dvc": pointer,
        # The path 'dvc add' must be given to produce THIS pointer file.
        # It is not always the artifact path: models/promoted.dvc versions
        # the whole models/promoted directory, so telling someone to run
        # 'dvc add models/promoted/model.npz' would create a different
        # pointer and leave the manifest still reporting the artifact as
        # unversioned.
        "dvc_target": (
            dvc_file[: -len(".dvc")]
            if dvc_file.endswith(".dvc")
            else dvc_file
        ),
    }


def derive_model_version(
    artifacts: dict[str, dict[str, object]],
    model_type: str = "model",
) -> str:
    """
    Derive a deployable model version from the artifact's content hash.

    Using the DVC hash keeps the version deterministic and traceable: the
    same bytes always produce the same version, and the version can be
    resolved back to an artifact through DVC. An unversioned artifact is
    labelled as such rather than given a fake version number.

    The family prefixes the version so a deployed image tag says which model
    it holds. That is why ``model_type`` is a parameter rather than the
    hardcoded ``"ials"`` it once was.
    """
    pointer = artifacts.get(
        "promoted_model",
        {},
    ).get("dvc")

    if not pointer or not pointer.get("md5"):
        return f"{model_type}-unversioned"

    return (
        f"{model_type}-"
        f"{str(pointer['md5'])[:8]}"
    )


def build_serving_env(
    model_type: str,
    model_version: str,
    artifacts: dict[str, dict[str, object]],
) -> dict[str, str]:
    """
    Build the environment block that configures a serving instance.

    These keys are read by src/serving/config.py; keeping them in the
    manifest means deployment does not hardcode artifact paths.
    """
    return {
        "MODEL_TYPE": model_type,
        "MODEL_VERSION": model_version,
        "MODEL_PATH": artifacts[
            "promoted_model"
        ]["path"],
        "FAISS_INDEX_PATH": artifacts[
            "faiss_index"
        ]["path"],
        "FAISS_METADATA_PATH": artifacts[
            "faiss_metadata"
        ]["path"],
        "ITEM_MAPPING_PATH": artifacts[
            "item_mapping"
        ]["path"],
        "INTERACTIONS_PATH": artifacts[
            "interactions"
        ]["path"],
    }


def collect_warnings(
    artifacts: dict[str, dict[str, object]],
) -> list[str]:
    """Report artifacts that are missing or unversioned."""
    warnings = []

    for role, details in artifacts.items():
        if not details["exists"]:
            warnings.append(
                f"{role} is missing at {details['path']}; "
                "run 'dvc pull' or the pipeline step that builds it."
            )

        if details["dvc"] is None:
            target = details.get(
                "dvc_target",
                details["path"],
            )

            warnings.append(
                f"{role} is not DVC-tracked, so this deployment "
                "cannot be reproduced from a version. Run "
                f"'dvc add {target} && dvc push'."
            )

    return warnings


def build_manifest(
    selection: dict[str, object],
    index_metadata: dict[str, object],
    artifacts: dict[str, dict[str, object]],
    raw_dataset: dict[str, object],
    generated_at: str,
) -> dict[str, object]:
    """Assemble the manifest from the pipeline's recorded outputs."""
    selected = selection.get("selected")

    if not selected:
        raise ManifestError(
            "Selection record contains no selected model."
        )

    model_type = str(
        selected.get(
            "model_type",
            "ials",
        )
    )

    model_version = derive_model_version(
        artifacts,
        model_type=model_type,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "model": {
            "type": model_type,
            "family": selected.get("family"),
            "candidate": selected.get("name"),
            "params": selected.get(
                "params",
                {},
            ),
            "version": model_version,
            "primary_metric": selection.get(
                "primary_metric"
            ),
            "metrics": selected.get(
                "metrics",
                {},
            ),
            "users_evaluated": selected.get(
                "users_evaluated"
            ),
            "mlflow": selected.get(
                "mlflow",
                {},
            ),
            "selection_source": selected.get(
                "source"
            ),
        },
        "retrieval": {
            "index_type": index_metadata.get(
                "index_type"
            ),
            "metric": index_metadata.get(
                "metric"
            ),
            "normalized_vectors": index_metadata.get(
                "normalized_vectors"
            ),
            "embedding_dimension": index_metadata.get(
                "embedding_dimension"
            ),
            "num_items": index_metadata.get(
                "num_items"
            ),
            "zero_vector_count": index_metadata.get(
                "zero_vector_count"
            ),
        },
        "artifacts": artifacts,
        "dataset": {
            "raw_interactions": raw_dataset,
            "processed_interactions": artifacts[
                "interactions"
            ],
            "item_mapping": artifacts[
                "item_mapping"
            ],
        },
        "bundle": [
            artifacts[role]["path"]
            for role in BUNDLE_ROLES
        ],
        "serving_env": build_serving_env(
            model_type=model_type,
            model_version=model_version,
            artifacts=artifacts,
        ),
        "warnings": collect_warnings(
            artifacts
        ),
    }


def load_json(
    path: str | Path,
    description: str,
) -> dict[str, object]:
    resolved = Path(path)

    if not resolved.exists():
        raise ManifestError(
            f"{description} not found: {resolved}. "
            "Run the pipeline steps that produce it."
        )

    return json.loads(
        resolved.read_text()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deployment manifest for the selected model."
        )
    )

    parser.add_argument(
        "--selection",
        default=SELECTION_PATH,
        help="Selection record from the select_model step.",
    )

    parser.add_argument(
        "--index-metadata",
        default=INDEX_METADATA_PATH,
        help="FAISS index metadata from the build_faiss step.",
    )

    parser.add_argument(
        "--output",
        default=str(MANIFEST_PATH),
        help="Where to write the manifest.",
    )

    parser.add_argument(
        "--require-dvc",
        action="store_true",
        help=(
            "Fail if any artifact is missing or not DVC-tracked, "
            "instead of recording a warning."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Building Deployment Manifest")
    print("=" * 60)

    selection = load_json(
        args.selection,
        "Selection record",
    )

    index_metadata = load_json(
        args.index_metadata,
        "FAISS index metadata",
    )

    artifacts = {
        role: describe_artifact(
            path,
            dvc_file,
        )
        for role, (
            path,
            dvc_file,
        ) in ARTIFACTS.items()
    }

    raw_path, raw_dvc = RAW_DATASET

    manifest = build_manifest(
        selection=selection,
        index_metadata=index_metadata,
        artifacts=artifacts,
        raw_dataset=describe_artifact(
            raw_path,
            raw_dvc,
        ),
        generated_at=(
            datetime.now(timezone.utc)
            .isoformat()
        ),
    )

    if manifest["warnings"]:
        print("\n=== Warnings ===")

        for warning in manifest["warnings"]:
            print(f"- {warning}")

        if args.require_dvc:
            raise SystemExit(
                "ERROR: --require-dvc was set and the artifacts are "
                "not fully versioned (see warnings above)."
            )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print("\n=== Manifest ===")

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(f"\nManifest written to: {output_path}")

    print("\n=== Manifest Complete ===")


if __name__ == "__main__":
    main()
