"""
Tests for the deployment manifest.

The manifest is an interface between the pipeline and serving, so two
properties matter most and are tested directly:

- the ``serving_env`` block really does configure ``src/serving/config.py``
  (a renamed variable on either side would otherwise go unnoticed until a
  Cloud Run deploy silently served the wrong artifact)
- an artifact that is missing or not DVC-tracked is reported, because such a
  deployment cannot be reproduced later
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.serving.config
from src.deployment.build_manifest import (
    ARTIFACTS,
    BUNDLE_ROLES,
    ManifestError,
    build_manifest,
    build_serving_env,
    collect_warnings,
    derive_model_version,
    describe_artifact,
    read_dvc_pointer,
)


SELECTION = {
    "primary_metric": "recall_at_10",
    "selected": {
        "name": "als-f256-r0p1-i20-a40",
        "family": "als",
        "model_type": "ials",
        "params": {
            "factors": 256,
            "alpha": 40.0,
        },
        "metrics": {
            "recall_at_10": 0.038064,
            "ndcg_at_10": 0.020348,
        },
        "users_evaluated": 94_762,
        "mlflow": {
            "run_id": "abc123",
            "experiment": "ials",
        },
        "source": "models/ials/evaluation.json",
    },
}

INDEX_METADATA = {
    "index_type": "IndexFlatIP",
    "metric": "inner_product",
    "normalized_vectors": False,
    "embedding_dimension": 64,
    "num_items": 25_612,
    "zero_vector_count": 0,
}


def make_artifacts(
    md5: str | None = "ef674512f5b037460aba84f841f90e16",
    exists: bool = True,
) -> dict[str, dict[str, object]]:
    """Build an artifact description block without touching the disk."""
    return {
        role: {
            "path": path,
            "exists": exists,
            "is_directory": False,
            "dvc": (
                {
                    "file": dvc_file,
                    "md5": md5,
                    "size": 1024,
                }
                if md5
                else None
            ),
        }
        for role, (
            path,
            dvc_file,
        ) in ARTIFACTS.items()
    }


class TestReadDvcPointer(unittest.TestCase):

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.directory.name
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_missing_pointer_returns_none(self):
        self.assertIsNone(
            read_dvc_pointer(
                self.root / "absent.dvc"
            )
        )

    def test_reads_a_file_pointer(self):
        path = self.root / "model.dvc"

        path.write_text(
            "outs:\n"
            "- md5: 5f76d4fdb4399723ecf93896cb142cb3\n"
            "  size: 8906638\n"
            "  hash: md5\n"
            "  path: video_games.parquet\n"
        )

        pointer = read_dvc_pointer(path)

        self.assertEqual(
            pointer["md5"],
            "5f76d4fdb4399723ecf93896cb142cb3",
        )

        self.assertEqual(
            pointer["size"],
            8906638,
        )

        self.assertNotIn(
            "nfiles",
            pointer,
        )

    def test_records_nfiles_for_directory_outputs(self):
        # Spark writes the interaction dataset and item mapping as
        # directories, which the deployment bundle must copy recursively.
        path = self.root / "dir.dvc"

        path.write_text(
            "outs:\n"
            "- md5: 7dfc5ebe082202accaaa456bd254c03e.dir\n"
            "  size: 11588361\n"
            "  nfiles: 18\n"
            "  hash: md5\n"
            "  path: video_games.parquet\n"
        )

        self.assertEqual(
            read_dvc_pointer(path)["nfiles"],
            18,
        )

    def test_pointer_without_outputs_is_an_error(self):
        path = self.root / "broken.dvc"

        path.write_text("outs: []\n")

        with self.assertRaises(ManifestError):
            read_dvc_pointer(path)


class TestDescribeArtifact(unittest.TestCase):

    def test_reports_presence_and_directoryness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            parquet_dir = root / "data.parquet"
            parquet_dir.mkdir()

            described = describe_artifact(
                str(parquet_dir),
                str(root / "absent.dvc"),
            )

            self.assertTrue(described["exists"])

            self.assertTrue(
                described["is_directory"]
            )

            self.assertIsNone(described["dvc"])

            missing = describe_artifact(
                str(root / "nope.npz"),
                str(root / "absent.dvc"),
            )

            self.assertFalse(missing["exists"])

            self.assertFalse(
                missing["is_directory"]
            )


class TestDeriveModelVersion(unittest.TestCase):

    def test_version_comes_from_the_content_hash(self):
        # Deterministic and traceable: the same bytes always produce the
        # same version, and the version resolves back through DVC.
        self.assertEqual(
            derive_model_version(
                make_artifacts(),
                model_type="ials",
            ),
            "ials-ef674512",
        )

    def test_untracked_artifact_is_labelled_unversioned(self):
        self.assertEqual(
            derive_model_version(
                make_artifacts(md5=None),
                model_type="ials",
            ),
            "ials-unversioned",
        )


class TestCollectWarnings(unittest.TestCase):

    def test_clean_artifacts_produce_no_warnings(self):
        self.assertEqual(
            collect_warnings(
                make_artifacts()
            ),
            [],
        )

    def test_untracked_artifact_is_reported(self):
        warnings = collect_warnings(
            make_artifacts(md5=None)
        )

        self.assertEqual(
            len(warnings),
            len(ARTIFACTS),
        )

        self.assertIn(
            "not DVC-tracked",
            warnings[0],
        )

        self.assertIn(
            "dvc add",
            warnings[0],
        )

    def test_missing_artifact_is_reported(self):
        warnings = collect_warnings(
            make_artifacts(exists=False)
        )

        self.assertTrue(
            any(
                "is missing at" in warning
                for warning in warnings
            )
        )


class TestBuildManifest(unittest.TestCase):

    def build(
        self,
        md5: str | None = "ef674512f5b037460aba84f841f90e16",
    ) -> dict[str, object]:
        artifacts = make_artifacts(md5=md5)

        return build_manifest(
            selection=SELECTION,
            index_metadata=INDEX_METADATA,
            artifacts=artifacts,
            raw_dataset={
                "path": (
                    "data/raw/Video_Games.csv.gz"
                ),
                "exists": True,
                "is_directory": False,
                "dvc": {
                    "file": (
                        "data/raw/"
                        "Video_Games.csv.gz.dvc"
                    ),
                    "md5": (
                        "8a5b4585a0122b29"
                        "ba8afa496a3f383c"
                    ),
                    "size": 12801967,
                },
            },
            generated_at="2026-08-21T12:00:00+00:00",
        )

    def test_records_model_identity_and_run(self):
        manifest = self.build()

        self.assertEqual(
            manifest["model"]["type"],
            "ials",
        )

        self.assertEqual(
            manifest["model"]["version"],
            "ials-ef674512",
        )

        self.assertEqual(
            manifest["model"]["mlflow"][
                "run_id"
            ],
            "abc123",
        )

        self.assertAlmostEqual(
            manifest["model"]["metrics"][
                "recall_at_10"
            ],
            0.038064,
        )

    def test_records_retrieval_semantics(self):
        # Inner product without normalisation is what makes FAISS agree
        # with iALS scoring; a manifest that lost this would hide a
        # correctness regression.
        retrieval = self.build()["retrieval"]

        self.assertEqual(
            retrieval["index_type"],
            "IndexFlatIP",
        )

        self.assertEqual(
            retrieval["metric"],
            "inner_product",
        )

        self.assertFalse(
            retrieval["normalized_vectors"]
        )

        self.assertEqual(
            retrieval["embedding_dimension"],
            64,
        )

    def test_bundle_lists_every_file_serving_needs(self):
        manifest = self.build()

        self.assertEqual(
            manifest["bundle"],
            [
                ARTIFACTS[role][0]
                for role in BUNDLE_ROLES
            ],
        )

        # Every artifact role belongs in the bundle; a partial upload would
        # fail at container startup rather than at deploy time.
        self.assertEqual(
            set(BUNDLE_ROLES),
            set(ARTIFACTS),
        )

    def test_dataset_block_covers_raw_and_processed(self):
        dataset = self.build()["dataset"]

        self.assertIn(
            "raw_interactions",
            dataset,
        )

        self.assertIn(
            "processed_interactions",
            dataset,
        )

        self.assertIn(
            "item_mapping",
            dataset,
        )

    def test_warnings_travel_with_the_manifest(self):
        self.assertTrue(
            self.build(md5=None)["warnings"]
        )

    def test_selection_without_a_model_is_an_error(self):
        with self.assertRaises(ManifestError):
            build_manifest(
                selection={"selected": None},
                index_metadata=INDEX_METADATA,
                artifacts=make_artifacts(),
                raw_dataset={},
                generated_at="2026-08-21T12:00:00+00:00",
            )

    def test_manifest_is_json_serialisable(self):
        self.assertIsInstance(
            json.dumps(self.build()),
            str,
        )


class TestServingEnvContract(unittest.TestCase):
    """
    The manifest's serving_env block must configure the real serving app.

    src/serving/config.py reads its environment at import time, so the
    module is reloaded under a patched environment to prove the variables
    take effect rather than merely sharing a name.
    """

    def tearDown(self):
        # Restore the module to the ambient environment for other tests.
        importlib.reload(src.serving.config)

    def test_env_block_configures_serving_settings(self):
        serving_env = build_serving_env(
            model_type="ials",
            model_version="ials-ef674512",
            artifacts=make_artifacts(),
        )

        with mock.patch.dict(
            os.environ,
            serving_env,
        ):
            importlib.reload(
                src.serving.config
            )

            settings = (
                src.serving.config.Settings()
            )

            self.assertEqual(
                settings.model_type,
                "ials",
            )

            self.assertEqual(
                settings.model_version,
                "ials-ef674512",
            )

            self.assertEqual(
                settings.model_path,
                ARTIFACTS["promoted_model"][0],
            )

            self.assertEqual(
                settings.faiss_index_path,
                ARTIFACTS["faiss_index"][0],
            )

            self.assertEqual(
                settings.faiss_metadata_path,
                ARTIFACTS["faiss_metadata"][0],
            )

            self.assertEqual(
                settings.item_mapping_path,
                ARTIFACTS["item_mapping"][0],
            )

            self.assertEqual(
                settings.interactions_path,
                ARTIFACTS["interactions"][0],
            )

    def test_every_artifact_path_setting_is_covered(self):
        # If serving grows a new artifact, the manifest must learn about it
        # too; this fails the moment the two drift apart.
        settings = src.serving.config.Settings()

        self.assertEqual(
            set(settings.artifact_paths),
            set(ARTIFACTS),
        )


if __name__ == "__main__":
    unittest.main()
