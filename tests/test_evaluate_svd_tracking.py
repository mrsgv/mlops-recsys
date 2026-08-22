"""
Tests for SVD baseline tracking.

The SVD run is logged from evaluation rather than training, because
training materialises an ~18 GB dense prediction matrix. These tests pin
the payload that reaches MLflow and prove the run really lands, using a
temporary file-backed store instead of the Cloud Run server.
"""

import tempfile
import unittest

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluation.evaluate_svd import (
    EXPERIMENT_NAME,
    RUN_NAME,
    build_run_payload,
    log_to_mlflow,
)


RESULTS = {
    "users_evaluated": 94_762,
    "precision_at_10": 0.0037451721154049093,
    "recall_at_10": 0.03745172115404909,
    "hit_rate_at_10": 0.03745172115404909,
    "ndcg_at_10": 0.020084203687931408,
}


def payload():
    return build_run_payload(
        results=RESULTS,
        num_users=94_762,
        num_items=25_612,
        train_interactions=719_824,
        test_interactions=94_762,
    )


class TestBuildRunPayload(unittest.TestCase):

    def test_records_model_hyperparameters(self):
        params, _, _ = payload()

        self.assertEqual(
            params["model"],
            "SVD",
        )

        self.assertEqual(
            params["factors"],
            50,
        )

        self.assertEqual(
            params["top_k"],
            10,
        )

        self.assertEqual(
            params["feedback_type"],
            "explicit",
        )

    def test_records_dataset_shape(self):
        params, _, _ = payload()

        self.assertEqual(
            params["num_users"],
            94_762,
        )

        self.assertEqual(
            params["num_items"],
            25_612,
        )

        self.assertEqual(
            params["train_interactions"],
            719_824,
        )

    def test_metrics_are_floats_for_mlflow(self):
        # MLflow rejects non-numeric metric values, and numpy scalars have
        # bitten this project before.
        _, metrics, _ = payload()

        for name, value in metrics.items():
            self.assertIsInstance(
                value,
                float,
                f"{name} must be a float",
            )

    def test_metrics_cover_every_ranking_measure(self):
        _, metrics, _ = payload()

        for name in (
            "precision_at_10",
            "recall_at_10",
            "hit_rate_at_10",
            "ndcg_at_10",
            "users_evaluated",
        ):
            self.assertIn(name, metrics)

        self.assertAlmostEqual(
            metrics["recall_at_10"],
            0.037452,
            places=6,
        )

    def test_tags_mark_svd_as_unservable(self):
        # The FAISS index is built from iALS item factors, so SVD is
        # comparable but has no retrieval path. Model selection relies on
        # that distinction.
        _, _, tags = payload()

        self.assertEqual(
            tags["model_type"],
            "svd",
        )

        self.assertEqual(
            tags["retrieval"],
            "none",
        )

        self.assertEqual(
            tags["stage"],
            "baseline",
        )

    def test_tags_disclose_where_metrics_came_from(self):
        # This run does not retrain; the tag stops anyone reading it as a
        # fresh training run later.
        _, _, tags = payload()

        self.assertEqual(
            tags["metrics_source"],
            "stored_predictions",
        )


class TestLogToMLflow(unittest.TestCase):

    def test_run_lands_with_params_metrics_and_tags(self):
        params, metrics, tags = payload()

        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = (
                f"file:{directory}/mlruns"
            )

            run_id = log_to_mlflow(
                params=params,
                metrics=metrics,
                tags=tags,
                tracking_uri=tracking_uri,
            )

            client = MlflowClient(
                tracking_uri=tracking_uri
            )

            run = client.get_run(run_id)

            self.assertEqual(
                run.info.status,
                "FINISHED",
            )

            self.assertAlmostEqual(
                run.data.metrics["recall_at_10"],
                0.037452,
                places=6,
            )

            self.assertEqual(
                run.data.params["factors"],
                "50",
            )

            self.assertEqual(
                run.data.tags["model_type"],
                "svd",
            )

            self.assertEqual(
                run.data.tags[
                    "mlflow.runName"
                ],
                RUN_NAME,
            )

            experiment = client.get_experiment(
                run.info.experiment_id
            )

            self.assertEqual(
                experiment.name,
                EXPERIMENT_NAME,
            )

    def test_svd_uses_its_own_experiment(self):
        # A separate experiment from "ials" and "two-tower" keeps the
        # comparison readable in the UI.
        self.assertEqual(
            EXPERIMENT_NAME,
            "svd",
        )

    def tearDown(self):
        # log_to_mlflow sets a process-wide tracking URI; reset it so later
        # tests are not pointed at a deleted temporary directory.
        mlflow.set_tracking_uri(None)


if __name__ == "__main__":
    unittest.main()
