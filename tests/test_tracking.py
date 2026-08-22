"""
Tests for the shared MLflow tracking helpers.

The helpers live in one module precisely so the upload-safety contract cannot
be reimplemented — and quietly broken — per model family. These tests pin the
contract at that shared boundary; ``tests/test_train_sweep.py`` and
``tests/test_ials_lifecycle.py`` then verify each entry point actually uses
it.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow

from src.models.tracking import (
    ARTIFACTS_UPLOADED_TAG,
    log_artifact_safely,
    log_artifacts_safely,
    mark_artifacts_uploaded,
    report_upload_outcome,
    run_identity,
)


class TestLogArtifactSafely(unittest.TestCase):

    def test_returns_true_when_upload_succeeds(self):
        with mock.patch(
            "mlflow.log_artifact"
        ) as log_artifact:
            self.assertTrue(
                log_artifact_safely(
                    Path("models/promoted/model.npz")
                )
            )

        log_artifact.assert_called_once_with(
            "models/promoted/model.npz"
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
                    Path("models/promoted/model.npz")
                )
            )

    def test_accepts_a_plain_string_path(self):
        with mock.patch(
            "mlflow.log_artifact"
        ) as log_artifact:
            log_artifact_safely(
                "models/promoted/model.npz"
            )

        log_artifact.assert_called_once_with(
            "models/promoted/model.npz"
        )

    def test_failure_is_reported_on_stdout(self):
        # Airflow captures task stdout, so this is how the degradation
        # becomes visible without the task going red.
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
                    Path("model.npz")
                )

        message = " ".join(
            str(call)
            for call in printed.call_args_list
        )

        self.assertIn("model.npz", message)

        self.assertIn(
            "upload timed out",
            message,
        )

        self.assertIn(
            "RuntimeError",
            message,
        )


class TestLogArtifactsSafely(unittest.TestCase):

    def test_true_only_when_every_upload_succeeds(self):
        with mock.patch(
            "mlflow.log_artifact"
        ):
            self.assertTrue(
                log_artifacts_safely(
                    [Path("a"), Path("b")]
                )
            )

        with mock.patch(
            "mlflow.log_artifact",
            side_effect=[
                None,
                OSError("boom"),
            ],
        ):
            self.assertFalse(
                log_artifacts_safely(
                    [Path("a"), Path("b")]
                )
            )

    def test_every_path_is_attempted_after_a_failure(self):
        # A single unreachable upload must not hide the state of the rest.
        with mock.patch(
            "mlflow.log_artifact",
            side_effect=[
                OSError("boom"),
                None,
                OSError("boom"),
            ],
        ) as log_artifact:
            self.assertFalse(
                log_artifacts_safely(
                    [
                        Path("a"),
                        Path("b"),
                        Path("c"),
                    ]
                )
            )

        self.assertEqual(
            log_artifact.call_count,
            3,
        )

    def test_empty_list_is_vacuously_successful(self):
        self.assertTrue(
            log_artifacts_safely([])
        )


class TestMarkArtifactsUploaded(unittest.TestCase):

    def test_tags_true_and_false(self):
        with mock.patch(
            "mlflow.set_tag"
        ) as set_tag:
            mark_artifacts_uploaded(True)

        set_tag.assert_called_once_with(
            ARTIFACTS_UPLOADED_TAG,
            "true",
        )

        with mock.patch(
            "mlflow.set_tag"
        ) as set_tag:
            mark_artifacts_uploaded(False)

        set_tag.assert_called_once_with(
            ARTIFACTS_UPLOADED_TAG,
            "false",
        )


class TestReportUploadOutcome(unittest.TestCase):

    def test_says_nothing_when_uploads_succeeded(self):
        with mock.patch(
            "builtins.print"
        ) as printed:
            report_upload_outcome(True)

        printed.assert_not_called()

    def test_explains_a_failed_upload(self):
        with mock.patch(
            "builtins.print"
        ) as printed:
            report_upload_outcome(False)

        message = " ".join(
            str(call)
            for call in printed.call_args_list
        )

        self.assertIn("DVC", message)


class TestRunIdentity(unittest.TestCase):
    """
    Identity must be capturable the instant a run opens.

    This is the value whose loss caused the incident: it was returned only
    after the artifact upload, so an upload failure destroyed the lineage
    record along with it.
    """

    def test_captures_identity_from_an_open_run(self):
        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = (
                f"file:{directory}/mlruns"
            )

            mlflow.set_tracking_uri(
                tracking_uri
            )

            self.addCleanup(
                mlflow.set_tracking_uri,
                None,
            )

            mlflow.set_experiment(
                "test-run-identity"
            )

            with mlflow.start_run(
                run_name="als-f128"
            ) as run:
                identity = run_identity(
                    run,
                    experiment="recsys-sweep",
                    run_name="als-f128",
                )

            self.assertEqual(
                identity["run_id"],
                run.info.run_id,
            )

            self.assertEqual(
                identity["experiment"],
                "recsys-sweep",
            )

            self.assertEqual(
                identity["run_name"],
                "als-f128",
            )

            self.assertEqual(
                identity["tracking_uri"],
                tracking_uri,
            )

    def test_identity_needs_no_network_call_of_its_own(self):
        # It reads only the already-open run, so it cannot itself be the
        # thing that fails and takes the lineage with it.
        run = mock.Mock()

        run.info.run_id = "abc123"

        with mock.patch(
            "mlflow.get_tracking_uri",
            return_value="http://localhost:8080",
        ):
            identity = run_identity(
                run,
                experiment="ials",
                run_name="ials-v1-binary",
            )

        self.assertEqual(
            identity,
            {
                "run_id": "abc123",
                "experiment": "ials",
                "run_name": "ials-v1-binary",
                "tracking_uri": (
                    "http://localhost:8080"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
