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


if __name__ == "__main__":
    unittest.main()
