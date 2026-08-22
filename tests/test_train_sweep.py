"""
Tests for the sweep's tracking and persistence contract.

These exist because of a specific incident. Run
``765681b70f16451ea79ddd222e35ef8f`` in the ``ials`` experiment is FAILED
despite training having fully succeeded — it holds the correct
``recall_at_10`` and the model reached disk. It died uploading a 30 MB
``.npz``. The lineage record was written *after* that upload, so the record
was never written, the run was marked FAILED, and all five downstream Airflow
tasks went ``upstream_failed``.

Generalising training to a sweep made the exposure worse rather than better:
the upload now sits inside a loop over model families, so one failed upload
would take out every remaining candidate.

So two properties are asserted directly, and neither is a "the file exists at
the end" check — that would pass against the broken ordering:

- the records a downstream stage reads are already on disk at the moment the
  first upload is *attempted*
- when an upload raises, the run still ends FINISHED, keeps its metrics, and
  carries ``artifacts_uploaded="false"``
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow
from mlflow.tracking import MlflowClient

from src.models.train_sweep import (
    build_candidate_record,
    track_variant,
    write_candidate,
)


METRICS = {
    "users_evaluated": 94_762,
    "precision_at_10": 0.0077372786560013515,
    "recall_at_10": 0.0773727865600135,
    "hit_rate_at_10": 0.0773727865600135,
    "ndcg_at_10": 0.04250872670585977,
}


class SweepPersistenceCase(unittest.TestCase):
    """Shared fixture: a candidate directory with a real model artifact."""

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.directory.name
        )

        self.candidate_dir = (
            self.root / "als-f128-r0p1-i20-a40"
        )

        self.candidate_dir.mkdir(
            parents=True,
        )

        self.artifact_path = (
            self.candidate_dir / "model.npz"
        )

        # The sweep saves the model before tracking begins, so the artifact
        # already exists by the time track_variant runs.
        self.artifact_path.write_bytes(
            b"factors"
        )

        self.addCleanup(
            self.directory.cleanup
        )

    @property
    def evaluation_path(self) -> Path:
        return (
            self.candidate_dir
            / "evaluation.json"
        )

    @property
    def training_run_path(self) -> Path:
        return (
            self.candidate_dir
            / "training_run.json"
        )

    def make_record(self) -> dict:
        return build_candidate_record(
            family="als",
            slug="als-f128-r0p1-i20-a40",
            params={
                "factors": 128,
                "regularization": 0.1,
                "iterations": 20,
                "alpha": 40.0,
            },
            metrics=METRICS,
            training_time_seconds=23.9,
            artifact_path=self.artifact_path,
            has_factors=True,
            embedding_dimension=128,
            mlflow_metadata={},
            num_users=94_762,
            num_items=25_612,
        )


class TestWriteCandidate(SweepPersistenceCase):

    def test_writes_both_records(self):
        write_candidate(
            self.candidate_dir,
            self.make_record(),
        )

        self.assertTrue(
            self.training_run_path.exists()
        )

        self.assertTrue(
            self.evaluation_path.exists()
        )

    def test_evaluation_carries_the_selection_contract(self):
        # select_model reads exactly these keys; losing one would make a
        # candidate silently unpromotable.
        write_candidate(
            self.candidate_dir,
            self.make_record(),
        )

        record = json.loads(
            self.evaluation_path.read_text()
        )

        for key in (
            "name",
            "family",
            "deployable",
            "metrics",
            "artifacts",
            "mlflow",
            "params",
        ):
            self.assertIn(key, record)

        self.assertTrue(
            record["deployable"]
        )

        self.assertAlmostEqual(
            record["metrics"][
                "recall_at_10"
            ],
            0.0773727865600135,
        )

    def test_neighbourhood_model_is_not_deployable(self):
        record = build_candidate_record(
            family="bm25",
            slug="bm25-k100",
            params={"K": 100},
            metrics=METRICS,
            training_time_seconds=0.3,
            artifact_path=self.artifact_path,
            has_factors=False,
            embedding_dimension=None,
            mlflow_metadata={},
            num_users=94_762,
            num_items=25_612,
        )

        self.assertFalse(
            record["deployable"]
        )

        self.assertIn(
            "deployable_reason",
            record,
        )


class TestTrackVariantOrdering(SweepPersistenceCase):
    """
    The regression guard.

    Asserting that the file exists *after* track_variant returns would pass
    against the broken ordering too. So the mock observes existence at the
    moment each upload is attempted.
    """

    def setUp(self):
        super().setUp()

        mlflow.set_tracking_uri(
            f"file:{self.root}/mlruns"
        )

        mlflow.set_experiment(
            "test-sweep-ordering"
        )

        self.addCleanup(
            mlflow.set_tracking_uri,
            None,
        )

    def test_records_are_on_disk_before_the_first_upload(self):
        observations = []

        def observe(path):
            observations.append(
                {
                    "uploading": Path(
                        path
                    ).name,
                    "evaluation_exists": (
                        self.evaluation_path.exists()
                    ),
                    "training_run_exists": (
                        self.training_run_path.exists()
                    ),
                }
            )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=observe,
        ):
            track_variant(
                record=self.make_record(),
                directory=self.candidate_dir,
                use_mlflow=True,
            )

        self.assertEqual(
            len(observations),
            3,
        )

        for observation in observations:
            self.assertTrue(
                observation[
                    "evaluation_exists"
                ],
                "evaluation.json must exist before the upload of "
                f"{observation['uploading']}",
            )

            self.assertTrue(
                observation[
                    "training_run_exists"
                ],
                "training_run.json must exist before the upload of "
                f"{observation['uploading']}",
            )

    def test_run_identity_is_recorded_before_the_first_upload(self):
        # The identity is what was lost in the incident: it was returned
        # after the upload, so a failed upload destroyed the lineage.
        seen = []

        def observe(path):
            seen.append(
                json.loads(
                    self.evaluation_path.read_text()
                )["mlflow"]
            )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=observe,
        ):
            track_variant(
                record=self.make_record(),
                directory=self.candidate_dir,
                use_mlflow=True,
            )

        self.assertTrue(seen)

        for identity in seen:
            self.assertTrue(
                identity.get("run_id"),
                "the run id must be on disk before any upload",
            )

            self.assertEqual(
                identity["run_name"],
                "als-f128-r0p1-i20-a40",
            )


class TestTrackVariantSurvivesFailure(SweepPersistenceCase):
    """
    A failed upload must degrade the run, never end it.

    Run against a real MLflow file store rather than mocks, because the
    property under test is the recorded run *status* — which only the store
    can report.
    """

    def setUp(self):
        super().setUp()

        self.tracking_uri = (
            f"file:{self.root}/mlruns"
        )

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        mlflow.set_experiment(
            "test-sweep-upload"
        )

        self.addCleanup(
            mlflow.set_tracking_uri,
            None,
        )

    def test_run_finishes_despite_upload_failure(self):
        record = self.make_record()

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=OSError(
                "Tunnel connection failed: 403 Forbidden"
            ),
        ):
            track_variant(
                record=record,
                directory=self.candidate_dir,
                use_mlflow=True,
            )

        run_id = record["mlflow"]["run_id"]

        reloaded = MlflowClient(
            tracking_uri=self.tracking_uri
        ).get_run(run_id)

        # The whole point: the sweep continues to the next candidate and the
        # DAG continues to selection.
        self.assertEqual(
            reloaded.info.status,
            "FINISHED",
        )

        self.assertEqual(
            reloaded.data.tags[
                "artifacts_uploaded"
            ],
            "false",
        )

        self.assertAlmostEqual(
            reloaded.data.metrics[
                "recall_at_10"
            ],
            0.0773727865600135,
        )

        # And the candidate remains selectable.
        self.assertTrue(
            self.evaluation_path.exists()
        )

        self.assertEqual(
            json.loads(
                self.evaluation_path.read_text()
            )["mlflow"]["run_id"],
            run_id,
        )

    def test_successful_upload_is_tagged_true(self):
        record = self.make_record()

        with mock.patch(
            "mlflow.log_artifact"
        ):
            track_variant(
                record=record,
                directory=self.candidate_dir,
                use_mlflow=True,
            )

        reloaded = MlflowClient(
            tracking_uri=self.tracking_uri
        ).get_run(
            record["mlflow"]["run_id"]
        )

        self.assertEqual(
            reloaded.data.tags[
                "artifacts_uploaded"
            ],
            "true",
        )

    def test_candidate_survives_a_tracking_failure(self):
        # If the tracking server cannot even open a run, the sweep must still
        # leave a selectable candidate behind rather than losing the training.
        record = self.make_record()

        with mock.patch(
            "mlflow.start_run",
            side_effect=OSError(
                "Connection refused"
            ),
        ):
            track_variant(
                record=record,
                directory=self.candidate_dir,
                use_mlflow=True,
            )

        self.assertTrue(
            self.evaluation_path.exists()
        )

        self.assertAlmostEqual(
            json.loads(
                self.evaluation_path.read_text()
            )["metrics"]["recall_at_10"],
            0.0773727865600135,
        )

    def test_offline_mode_still_writes_the_candidate(self):
        track_variant(
            record=self.make_record(),
            directory=self.candidate_dir,
            use_mlflow=False,
        )

        self.assertTrue(
            self.evaluation_path.exists()
        )

        self.assertTrue(
            self.training_run_path.exists()
        )


if __name__ == "__main__":
    unittest.main()
