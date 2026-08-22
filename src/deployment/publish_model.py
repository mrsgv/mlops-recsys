"""
Publish the promoted model to the MLflow Model Registry.

Before this step existed, MLflow held loose ``.npz`` files attached to runs
via ``log_artifact`` and the Model Registry was completely empty — despite
the tracking server running Cloud SQL partly to host it. Promotion lived
only in ``models/deployment/selected_model.json``, a file in git, so the
tracking server had no idea which model was champion.

This step closes that gap. It logs the promoted model as a ``pyfunc``
bundle, registers it, and moves an alias to the new version. Afterwards
``models:/<name>@champion`` is a stable, server-side pointer to whatever
won, resolvable from CI, a notebook or a serving container without anyone
reading a JSON file out of the repository.

Aliases, not stages
-------------------
MLflow 3 deprecated the ``Staging``/``Production`` stage transitions in
favour of registry aliases, so promotion is expressed as
``set_registered_model_alias(name, "champion", version)``.

Three systems, three responsibilities
-------------------------------------
DVC     the bytes. The manifest already records each artifact's DVC content
        hash, which is what makes a deployment reproducible.
MLflow  the identity and lineage. Which run trained it, on what data, with
        what metrics, and which version is champion.
manifest the serving configuration. Which files to load and which env vars
        to set.

They cross-reference rather than duplicate: the registry version and model
URI are written back into the manifest, and the version is tagged with the
DVC hash.

Registering costs storage
-------------------------
Each version carries the whole bundle, dominated by the interaction parquet,
so versions are roughly a hundred megabytes. Re-registering on every
pipeline run would accumulate quickly for no benefit, so ``--only-if-better``
compares against the metric recorded on the current champion and skips
publishing when the candidate does not beat it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.deployment.pyfunc_model import (
    ARTIFACT_KEYS,
    RecommenderPythonModel,
    build_input_example,
)


SELECTION_PATH = (
    "models/deployment/selected_model.json"
)

MANIFEST_PATH = (
    "models/deployment/deployment_manifest.json"
)

REGISTERED_MODEL_NAME = (
    "video-games-recommender"
)

CHAMPION_ALIAS = "champion"

EXPERIMENT_NAME = "recsys-sweep"

MODEL_NAME = "recommender"

PRIMARY_METRIC = "recall_at_10"

# The registered model's runtime dependencies. Reusing the serving
# requirements file rather than restating the packages means the registry
# version and the Docker image can never disagree about what it takes to run
# this model.
SERVING_REQUIREMENTS = (
    "requirements-serving.txt"
)

# Only the retrieval and serving packages are shipped inside the model. The
# rest of src/ pulls in Spark and MLflow, which the model does not need at
# inference time.
CODE_PATHS = ["src"]


class PublishError(Exception):
    """Raised when the promoted model cannot be published."""


def load_json(
    path: str | Path,
    description: str,
) -> dict[str, Any]:
    resolved = Path(path)

    if not resolved.exists():
        raise PublishError(
            f"{description} not found: {resolved}. "
            "Run the pipeline steps that produce it."
        )

    return json.loads(
        resolved.read_text()
    )


def resolve_bundle(
    manifest: dict[str, Any],
) -> dict[str, str]:
    """
    Map artifact roles to on-disk paths using the manifest.

    The manifest is the authority on what the deployment consists of, so the
    bundle is read from it rather than being restated here. A role the
    manifest does not describe, or describes as missing, is a hard error:
    publishing a bundle that is missing a file would produce a registry
    version that cannot be loaded.
    """
    artifacts = manifest.get(
        "artifacts",
        {},
    )

    # Manifest role -> pyfunc artifact key. The manifest still calls the
    # model artifact by its serving role name.
    role_for_key = {
        "model": (
            "promoted_model"
            if "promoted_model" in artifacts
            else "ials_model"
        ),
        "faiss_index": "faiss_index",
        "faiss_metadata": "faiss_metadata",
        "item_mapping": "item_mapping",
        "interactions": "interactions",
    }

    bundle: dict[str, str] = {}

    for key in ARTIFACT_KEYS:
        role = role_for_key[key]

        details = artifacts.get(role)

        if not details:
            raise PublishError(
                f"The manifest does not describe the '{role}' "
                "artifact, so the model bundle is incomplete."
            )

        path = Path(str(details["path"]))

        if not path.exists():
            raise PublishError(
                f"Bundle artifact '{role}' is missing at {path}. "
                "Run 'dvc pull' or the pipeline step that builds it."
            )

        bundle[key] = str(path)

    return bundle


def build_version_tags(
    selection: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """
    Describe the version well enough to audit it from the registry alone.

    The DVC hash is included so a registry version can be tied back to
    exact bytes, which is what makes the deployment reproducible.
    """
    selected = selection.get(
        "selected",
        {},
    )

    model = manifest.get("model", {})

    artifacts = manifest.get(
        "artifacts",
        {},
    )

    model_artifact = artifacts.get(
        "promoted_model"
    ) or artifacts.get("ials_model", {})

    dvc_pointer = (
        model_artifact.get("dvc") or {}
    )

    tags = {
        "model_family": str(
            selected.get("family", "unknown")
        ),
        "candidate": str(
            selected.get("name", "unknown")
        ),
        "primary_metric": str(
            selection.get(
                "primary_metric",
                PRIMARY_METRIC,
            )
        ),
        "manifest_version": str(
            model.get("version", "unknown")
        ),
        "dvc_md5": str(
            dvc_pointer.get("md5", "untracked")
        ),
        "training_run_id": str(
            selected.get("mlflow", {}).get(
                "run_id",
                "unknown",
            )
        ),
    }

    for name, value in (
        selected.get("metrics") or {}
    ).items():
        tags[f"metric_{name}"] = (
            f"{float(value):.6f}"
        )

    return tags


def current_champion_metric(
    client: MlflowClient,
    registered_name: str,
    alias: str,
    metric: str,
) -> float | None:
    """
    Read the metric recorded on the version the alias currently points at.

    Returns None when there is no champion yet, or when the existing
    champion carries no comparable metric tag — in both cases the candidate
    should be published rather than blocked by a missing comparison.
    """
    try:
        version = (
            client.get_model_version_by_alias(
                registered_name,
                alias,
            )
        )
    except Exception:
        return None

    raw = (version.tags or {}).get(
        f"metric_{metric}"
    )

    if raw is None:
        return None

    try:
        return float(raw)
    except ValueError:
        return None


def log_and_register(
    bundle: dict[str, str],
    selection: dict[str, Any],
    tags: dict[str, str],
    registered_name: str,
    run_id: str | None,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[str, str]:
    """
    Log the pyfunc bundle and register it.

    The model is logged into the run that trained the promoted candidate
    when that run is known, so the registry version's lineage points
    directly at its training run instead of at a separate publishing run.

    Returns
    -------
    tuple
        The logged model URI and the registered version number.
    """
    selected = selection.get(
        "selected",
        {},
    )

    input_example = build_input_example()

    # The signature is inferred from a real call shape rather than declared
    # by hand, so the recorded schema cannot drift from what predict
    # actually returns.
    signature = infer_signature(
        input_example,
        pd.DataFrame(
            {
                "user_idx": [0],
                "rank": [1],
                "item_idx": [0],
                "parent_asin": ["B0000000AA"],
                "score": [0.0],
            }
        ),
    )

    if run_id:
        run_context = mlflow.start_run(
            run_id=run_id,
        )

        print(
            f"Logging into the promoted candidate's training run: "
            f"{run_id}"
        )
    else:
        mlflow.set_experiment(
            experiment_name
        )

        run_context = mlflow.start_run(
            run_name=(
                "publish-"
                + str(
                    selected.get(
                        "name",
                        "model",
                    )
                )
            ),
        )

        print(
            "No training run recorded for the promoted candidate; "
            "publishing under a new run."
        )

    with run_context:
        info = mlflow.pyfunc.log_model(
            name=MODEL_NAME,
            python_model=(
                RecommenderPythonModel()
            ),
            artifacts=bundle,
            signature=signature,
            input_example=input_example,
            code_paths=CODE_PATHS,
            pip_requirements=(
                SERVING_REQUIREMENTS
            ),
            registered_model_name=(
                registered_name
            ),
            metadata={
                "family": selected.get(
                    "family"
                ),
                "candidate": selected.get(
                    "name"
                ),
                "params": selected.get(
                    "params",
                    {},
                ),
                "metrics": selected.get(
                    "metrics",
                    {},
                ),
            },
        )

        mlflow.set_tags(
            {
                "published": "true",
                "registered_model": (
                    registered_name
                ),
            }
        )

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{registered_name}'"
    )

    if not versions:
        raise PublishError(
            "Registration reported success but no model version "
            f"exists for '{registered_name}'."
        )

    version = max(
        versions,
        key=lambda entry: int(entry.version),
    )

    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_name,
            version=version.version,
            key=key,
            value=value,
        )

    return (
        info.model_uri,
        str(version.version),
    )


def update_manifest(
    manifest_path: str | Path,
    registry_details: dict[str, Any],
) -> None:
    """
    Record the registry version in the deployment manifest.

    This is the cross-reference that lets a running service report which
    registry version it corresponds to, and lets an auditor go from a
    deployed container back to a registered model without guessing.
    """
    resolved = Path(manifest_path)

    manifest = json.loads(
        resolved.read_text()
    )

    manifest.setdefault(
        "model",
        {},
    )["registry"] = registry_details

    resolved.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the promoted model to the MLflow Model Registry."
        )
    )

    parser.add_argument(
        "--selection",
        default=SELECTION_PATH,
        help="Selection record from select_model.",
    )

    parser.add_argument(
        "--manifest",
        default=MANIFEST_PATH,
        help=(
            "Deployment manifest, read for the bundle and updated with "
            "the registry version."
        ),
    )

    parser.add_argument(
        "--registered-name",
        default=REGISTERED_MODEL_NAME,
        help="Registered model name.",
    )

    parser.add_argument(
        "--alias",
        default=CHAMPION_ALIAS,
        help=(
            "Alias moved to the new version. Use '' to register "
            "without promoting."
        ),
    )

    parser.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC,
        help=(
            "Metric compared against the current champion."
        ),
    )

    parser.add_argument(
        "--only-if-better",
        action="store_true",
        help=(
            "Skip publishing unless the candidate beats the current "
            "champion on the primary metric."
        ),
    )

    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Publish under a fresh run instead of the promoted "
            "candidate's training run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Publishing to the MLflow Model Registry")
    print("=" * 60)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    print(
        f"\nTracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    selection = load_json(
        args.selection,
        "Selection record",
    )

    manifest = load_json(
        args.manifest,
        "Deployment manifest",
    )

    selected = selection.get(
        "selected",
        {},
    )

    if not selected:
        raise PublishError(
            "Selection record contains no selected model."
        )

    candidate_metric = (
        selected.get("metrics", {}).get(
            args.primary_metric
        )
    )

    if candidate_metric is None:
        raise PublishError(
            "The promoted candidate has no "
            f"{args.primary_metric}, so it cannot be compared "
            "against the champion."
        )

    print(
        f"\nCandidate: {selected.get('name')} "
        f"(family={selected.get('family')})"
    )

    print(
        f"{args.primary_metric}: "
        f"{float(candidate_metric):.6f}"
    )

    client = MlflowClient()

    if args.alias:
        champion_metric = (
            current_champion_metric(
                client=client,
                registered_name=(
                    args.registered_name
                ),
                alias=args.alias,
                metric=args.primary_metric,
            )
        )
    else:
        champion_metric = None

    if champion_metric is None:
        print(
            f"\nNo existing '{args.alias or 'champion'}' version "
            "with a comparable metric."
        )
    else:
        print(
            f"\nCurrent {args.alias}: "
            f"{args.primary_metric}="
            f"{champion_metric:.6f}"
        )

    if (
        args.only_if_better
        and champion_metric is not None
        and float(candidate_metric)
        <= champion_metric
    ):
        print(
            "\nCandidate does not beat the current champion "
            f"({float(candidate_metric):.6f} <= "
            f"{champion_metric:.6f}). Nothing published "
            "(--only-if-better)."
        )

        print(
            "\n=== Publish Skipped ==="
        )

        return

    print("\n=== Resolving Bundle ===")

    bundle = resolve_bundle(manifest)

    for key, path in bundle.items():
        size_mb = (
            sum(
                item.stat().st_size
                for item in Path(path).rglob(
                    "*"
                )
                if item.is_file()
            )
            if Path(path).is_dir()
            else Path(path).stat().st_size
        ) / (1024 * 1024)

        print(
            f"{key:<16} {path}  "
            f"({size_mb:.1f} MB)"
        )

    print("\n=== Logging and Registering ===")

    run_id = (
        None
        if args.new_run
        else selected.get("mlflow", {}).get(
            "run_id"
        )
    )

    model_uri, version = log_and_register(
        bundle=bundle,
        selection=selection,
        tags=build_version_tags(
            selection,
            manifest,
        ),
        registered_name=(
            args.registered_name
        ),
        run_id=run_id,
    )

    print(f"Model URI: {model_uri}")

    print(
        f"Registered: "
        f"{args.registered_name} "
        f"version {version}"
    )

    alias_uri = None

    if args.alias:
        client.set_registered_model_alias(
            name=args.registered_name,
            alias=args.alias,
            version=version,
        )

        alias_uri = (
            f"models:/{args.registered_name}"
            f"@{args.alias}"
        )

        print(
            f"\nAlias '{args.alias}' now points at "
            f"version {version}."
        )

        print(f"Resolvable as: {alias_uri}")
    else:
        print(
            "\nNo alias requested; the version was registered "
            "without being promoted."
        )

    registry_details = {
        "registered_model": (
            args.registered_name
        ),
        "version": version,
        "alias": args.alias or None,
        "model_uri": model_uri,
        "alias_uri": alias_uri,
        "tracking_uri": (
            mlflow.get_tracking_uri()
        ),
        "primary_metric": (
            args.primary_metric
        ),
        "primary_metric_value": float(
            candidate_metric
        ),
        "previous_champion_metric": (
            champion_metric
        ),
    }

    update_manifest(
        args.manifest,
        registry_details,
    )

    print(
        f"\nManifest updated with the registry version: "
        f"{args.manifest}"
    )

    print("\n=== Publish Complete ===")


if __name__ == "__main__":
    main()
