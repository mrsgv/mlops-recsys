"""
Tests for the metadata contract that links a served model back to the
MLflow run and dataset that produced it.

These records are what the model-selection and deployment-manifest steps
read, so their shape is a real interface rather than an implementation
detail.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluation.evaluate_ials import (
    build_evaluation_record,
    compare_with_training_metrics,
    load_training_run,
    log_to_mlflow,
)
from src.models.train_ials import (
    build_training_run_record,
    log_artifact_safely,
    log_artifacts_safely,
    persist_training_outputs,
    write_training_run,
)


METRICS = {
    "precision_at_10": 0.0038,
    "recall_at_10": 0.038064,
    "hit_rate_at_10": 0.038064,
    "ndcg_at_10": 0.020348,
    "users_evaluated": 94_762,
}


class TestTrainingRunRecord(unittest.TestCase):

    def build(self):
        return build_training_run_record(
            run_id="abc123",
            tracking_uri="http://localhost:8080",
            params={
                "factors": 64,
                "alpha": 1.0,
            },
            metrics=METRICS,
            training_time_seconds=12.5,
        )

    def test_records_mlflow_identity(self):
        record = self.build()

        self.assertEqual(
            record["mlflow"]["run_id"],
            "abc123",
        )

        self.assertEqual(
            record["mlflow"]["experiment"],
            "ials",
        )

        self.assertEqual(
            record["mlflow"]["tracking_uri"],
            "http://localhost:8080",
        )

    def test_separates_metrics_from_user_count(self):
        record = self.build()

        self.assertNotIn(
            "users_evaluated",
            record["metrics"],
        )

        self.assertEqual(
            record["users_evaluated"],
            94_762,
        )

        self.assertAlmostEqual(
            record["metrics"]["recall_at_10"],
            0.038064,
        )

    def test_records_artifact_and_dataset(self):
        record = self.build()

        self.assertEqual(
            record["artifacts"]["ials_model"],
            "models/ials/ials_model.npz",
        )

        self.assertEqual(
            record["dataset"]["interactions"],
            "data/processed/video_games.parquet",
        )

    def test_is_json_serialisable(self):
        # The record is written to disk and read by later pipeline steps,
        # so anything non-serialisable would break the pipeline at runtime.
        self.assertIsInstance(
            json.dumps(self.build()),
            str,
        )


class TestLoadTrainingRun(unittest.TestCase):

    def test_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                load_training_run(
                    Path(directory)
                    / "training_run.json"
                )
            )

    def test_reads_written_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "training_run.json"
            )

            path.write_text(
                json.dumps(
                    {"model_type": "ials"}
                )
            )

            self.assertEqual(
                load_training_run(path),
                {"model_type": "ials"},
            )


class TestCompareWithTrainingMetrics(unittest.TestCase):

    def test_no_mismatch_when_identical(self):
        self.assertEqual(
            compare_with_training_metrics(
                METRICS,
                {
                    "recall_at_10": 0.038064,
                    "ndcg_at_10": 0.020348,
                },
            ),
            {},
        )

    def test_ignores_differences_within_tolerance(self):
        self.assertEqual(
            compare_with_training_metrics(
                METRICS,
                {
                    "recall_at_10": 0.038064 + 1e-9,
                },
            ),
            {},
        )

    def test_reports_real_divergence(self):
        mismatches = (
            compare_with_training_metrics(
                METRICS,
                {"recall_at_10": 0.02},
            )
        )

        self.assertIn(
            "recall_at_10",
            mismatches,
        )

        self.assertAlmostEqual(
            mismatches["recall_at_10"]["training"],
            0.02,
        )

        self.assertAlmostEqual(
            mismatches["recall_at_10"]["evaluation"],
            0.038064,
        )

    def test_ignores_metrics_absent_from_evaluation(self):
        self.assertEqual(
            compare_with_training_metrics(
                METRICS,
                {"map_at_10": 0.5},
            ),
            {},
        )


class TestEvaluationRecord(unittest.TestCase):

    def test_marks_ials_as_deployable(self):
        record = build_evaluation_record(
            metrics=METRICS,
            training_run=None,
        )

        self.assertTrue(record["deployable"])

        self.assertEqual(
            record["retrieval"],
            "faiss",
        )

        self.assertEqual(
            record["mlflow"],
            {},
        )

    def test_carries_mlflow_identity_forward(self):
        record = build_evaluation_record(
            metrics=METRICS,
            training_run={
                "mlflow": {
                    "run_id": "abc123",
                    "experiment": "ials",
                }
            },
        )

        self.assertEqual(
            record["mlflow"]["run_id"],
            "abc123",
        )

    def test_exposes_selection_metrics(self):
        record = build_evaluation_record(
            metrics=METRICS,
            training_run=None,
        )

        self.assertAlmostEqual(
            record["metrics"]["recall_at_10"],
            0.038064,
        )

        self.assertEqual(
            record["users_evaluated"],
            94_762,
        )

        self.assertEqual(
            record["top_k"],
            10,
        )


class TestWriteTrainingRun(unittest.TestCase):

    def test_writes_record_beside_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "ials"
                / "training_run.json"
            )

            with mock.patch(
                "src.models.train_ials"
                ".TRAINING_RUN_PATH",
                path,
            ):
                returned = write_training_run(
                    run_id="abc123",
                    params={"factors": 64},
                    metrics=METRICS,
                    training_time_seconds=9.0,
                )

            # The parent directory is created on demand, because training
            # may run before models/ials exists.
            self.assertTrue(path.exists())

            on_disk = json.loads(
                path.read_text()
            )

            self.assertEqual(
                on_disk["mlflow"]["run_id"],
                "abc123",
            )

            self.assertEqual(
                on_disk,
                returned,
            )

            # Later steps read this file, so it must round-trip through
            # JSON without losing the metrics.
            self.assertAlmostEqual(
                on_disk["metrics"]["recall_at_10"],
                0.038064,
            )


class TestLogToMLflow(unittest.TestCase):
    """
    Verify artifact-evaluation metrics reach the training run.

    A temporary file-backed tracking store is used, so this exercises the
    real MLflow client without needing the Cloud Run server.
    """

    def test_attaches_eval_metrics_to_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = (
                f"file:{directory}/mlruns"
            )

            mlflow.set_tracking_uri(
                tracking_uri
            )

            mlflow.set_experiment(
                "test-ials"
            )

            with mlflow.start_run() as run:
                run_id = run.info.run_id

                mlflow.log_metric(
                    "recall_at_10",
                    0.038064,
                )

            log_to_mlflow(
                run_id=run_id,
                metrics=METRICS,
                tracking_uri=tracking_uri,
            )

            client = MlflowClient(
                tracking_uri=tracking_uri
            )

            reloaded = client.get_run(run_id)

            # eval_ prefixed names so the artifact numbers never overwrite
            # the numbers training logged under the same run.
            self.assertAlmostEqual(
                reloaded.data.metrics[
                    "eval_recall_at_10"
                ],
                0.038064,
            )

            self.assertAlmostEqual(
                reloaded.data.metrics[
                    "eval_ndcg_at_10"
                ],
                0.020348,
            )

            self.assertNotIn(
                "eval_users_evaluated",
                reloaded.data.metrics,
            )

            # The original metric is untouched.
            self.assertAlmostEqual(
                reloaded.data.metrics[
                    "recall_at_10"
                ],
                0.038064,
            )

            self.assertEqual(
                reloaded.data.tags[
                    "artifact_evaluated"
                ],
                "true",
            )

            # Logging must not leave the run open.
            self.assertEqual(
                reloaded.info.status,
                "FINISHED",
            )


class FakeImplicitModel:
    """Stand-in for the implicit model, which writes a real file on save."""

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save(self, path: str) -> None:
        self.saved_to = path

        Path(path).write_bytes(b"factors")


class FakeRecommender:

    def __init__(self) -> None:
        self.model = FakeImplicitModel()


class TestLogArtifactSafely(unittest.TestCase):
    """
    An artifact upload must not be able to fail a training run.

    The model is versioned by DVC and the lineage record is on disk, so a
    failed upload is a degradation. Before this, it raised — which marked the
    run FAILED and stopped the Airflow DAG at the task after training.
    """

    def test_returns_true_when_upload_succeeds(self):
        with mock.patch(
            "mlflow.log_artifact"
        ) as log_artifact:
            self.assertTrue(
                log_artifact_safely(
                    Path("models/ials/ials_model.npz")
                )
            )

        log_artifact.assert_called_once_with(
            "models/ials/ials_model.npz"
        )

    def test_returns_false_instead_of_raising(self):
        with mock.patch(
            "mlflow.log_artifact",
            side_effect=OSError(
                "Tunnel connection failed: 403 Forbidden"
            ),
        ):
            self.assertFalse(
                log_artifact_safely(
                    Path("models/ials/ials_model.npz")
                )
            )

    def test_failure_is_reported_on_stdout(self):
        # Airflow captures task stdout, so this is how the failure becomes
        # visible without failing the task.
        with mock.patch(
            "mlflow.log_artifact",
            side_effect=RuntimeError(
                "upload timed out"
            ),
        ):
            with mock.patch(
                "builtins.print"
            ) as printed:
                log_artifact_safely(
                    Path("models/ials/ials_model.npz")
                )

        message = " ".join(
            str(call)
            for call in printed.call_args_list
        )

        self.assertIn(
            "ials_model.npz",
            message,
        )

        self.assertIn(
            "upload timed out",
            message,
        )

    def test_true_only_when_every_upload_succeeds(self):
        with mock.patch(
            "mlflow.log_artifact"
        ):
            self.assertTrue(
                log_artifacts_safely(
                    [
                        Path("a"),
                        Path("b"),
                    ]
                )
            )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=[None, OSError("boom")],
        ):
            self.assertFalse(
                log_artifacts_safely(
                    [
                        Path("a"),
                        Path("b"),
                    ]
                )
            )

    def test_every_path_is_attempted_after_a_failure(self):
        # A single unreachable upload must not hide the state of the rest.
        with mock.patch(
            "mlflow.log_artifact",
            side_effect=[
                OSError("boom"),
                None,
            ],
        ) as log_artifact:
            log_artifacts_safely(
                [
                    Path("a"),
                    Path("b"),
                ]
            )

        self.assertEqual(
            log_artifact.call_count,
            2,
        )


class TestPersistTrainingOutputs(unittest.TestCase):
    """
    Ordering contract: the lineage record reaches disk before any upload.

    ``training_run.json`` is what evaluation, selection and the deployment
    manifest read to find the run that produced the artifact. If it is written
    after the upload, a failed upload leaves no record at all.
    """

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        root = Path(self.directory.name)

        self.model_dir = root / "ials"

        self.model_path = (
            self.model_dir / "ials_model.npz"
        )

        self.run_path = (
            self.model_dir / "training_run.json"
        )

        self.patches = [
            mock.patch(
                "src.models.train_ials.MODEL_DIR",
                self.model_dir,
            ),
            mock.patch(
                "src.models.train_ials.MODEL_PATH",
                self.model_path,
            ),
            mock.patch(
                "src.models.train_ials"
                ".TRAINING_RUN_PATH",
                self.run_path,
            ),
        ]

        for patch in self.patches:
            patch.start()

        self.addCleanup(self.directory.cleanup)

        for patch in self.patches:
            self.addCleanup(patch.stop)

    def persist(self) -> bool:
        return persist_training_outputs(
            model=FakeRecommender(),
            run_id="abc123",
            params={"factors": 64},
            metrics=METRICS,
            training_time_seconds=9.0,
        )

    def test_record_survives_a_failed_upload(self):
        with mock.patch(
            "mlflow.log_artifact",
            side_effect=OSError(
                "Tunnel connection failed: 403 Forbidden"
            ),
        ):
            with mock.patch(
                "mlflow.set_tag"
            ):
                uploaded = self.persist()

        self.assertFalse(uploaded)

        # Both local outputs are intact, which is what lets the rest of the
        # DAG run against a tracking server that is unreachable.
        self.assertTrue(
            self.model_path.exists()
        )

        self.assertTrue(
            self.run_path.exists()
        )

        record = json.loads(
            self.run_path.read_text()
        )

        self.assertEqual(
            record["mlflow"]["run_id"],
            "abc123",
        )

        self.assertAlmostEqual(
            record["metrics"]["recall_at_10"],
            0.038064,
        )

    def test_record_is_on_disk_before_the_first_upload(self):
        # The regression guard: assert the file already exists at the moment
        # the first upload is attempted.
        existed_at_upload = []

        def observe(path):
            existed_at_upload.append(
                self.run_path.exists()
            )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=observe,
        ):
            with mock.patch(
                "mlflow.set_tag"
            ):
                self.persist()

        self.assertEqual(
            len(existed_at_upload),
            2,
        )

        self.assertTrue(
            all(existed_at_upload),
            "training_run.json must be written before any upload",
        )

    def test_tag_records_whether_artifacts_landed(self):
        with mock.patch(
            "mlflow.log_artifact"
        ):
            with mock.patch(
                "mlflow.set_tag"
            ) as set_tag:
                self.assertTrue(self.persist())

        set_tag.assert_called_once_with(
            "artifacts_uploaded",
            "true",
        )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=OSError("boom"),
        ):
            with mock.patch(
                "mlflow.set_tag"
            ) as set_tag:
                self.assertFalse(self.persist())

        set_tag.assert_called_once_with(
            "artifacts_uploaded",
            "false",
        )

    def test_model_directory_is_created_on_demand(self):
        # Training can run before models/ials exists.
        self.assertFalse(
            self.model_dir.exists()
        )

        with mock.patch(
            "mlflow.log_artifact"
        ):
            with mock.patch(
                "mlflow.set_tag"
            ):
                self.persist()

        self.assertTrue(
            self.model_dir.is_dir()
        )

    def test_both_artifacts_are_offered_to_mlflow(self):
        with mock.patch(
            "mlflow.log_artifact"
        ) as log_artifact:
            with mock.patch(
                "mlflow.set_tag"
            ):
                self.persist()

        self.assertEqual(
            [
                Path(call.args[0]).name
                for call in log_artifact.call_args_list
            ],
            [
                "ials_model.npz",
                "training_run.json",
            ],
        )


class TestTrainingRunSurvivesUploadFailure(unittest.TestCase):
    """
    End-to-end proof against a real MLflow store: a failed upload leaves the
    run FINISHED with its metrics, not FAILED.
    """

    def tearDown(self):
        mlflow.set_tracking_uri(None)

    def test_run_finishes_despite_upload_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            tracking_uri = (
                f"file:{root}/mlruns"
            )

            mlflow.set_tracking_uri(
                tracking_uri
            )

            mlflow.set_experiment(
                "test-ials-upload"
            )

            model_dir = root / "ials"

            with mock.patch(
                "src.models.train_ials.MODEL_DIR",
                model_dir,
            ), mock.patch(
                "src.models.train_ials.MODEL_PATH",
                model_dir / "ials_model.npz",
            ), mock.patch(
                "src.models.train_ials"
                ".TRAINING_RUN_PATH",
                model_dir / "training_run.json",
            ), mock.patch(
                "mlflow.log_artifact",
                side_effect=OSError(
                    "Tunnel connection failed: 403 Forbidden"
                ),
            ):
                with mlflow.start_run() as run:
                    run_id = run.info.run_id

                    mlflow.log_metric(
                        "recall_at_10",
                        0.038064,
                    )

                    persist_training_outputs(
                        model=FakeRecommender(),
                        run_id=run_id,
                        params={"factors": 64},
                        metrics=METRICS,
                        training_time_seconds=9.0,
                    )

            reloaded = MlflowClient(
                tracking_uri=tracking_uri
            ).get_run(run_id)

            # The whole point: an upload failure degrades the run instead of
            # failing it, so the DAG continues to evaluation.
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
                0.038064,
            )

            self.assertTrue(
                (
                    model_dir
                    / "training_run.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
