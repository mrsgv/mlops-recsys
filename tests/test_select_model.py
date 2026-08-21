"""
Tests for model selection.

Selection is the promotion decision, so the tests focus on the two rules
that make it safe: rank by the primary metric, and never promote a model the
serving path cannot load.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.deployment.select_model import (
    ModelSelectionError,
    load_ials_candidate,
    load_svd_candidate,
    rank_candidates,
    select_model,
)


def make_candidate(
    name: str,
    recall: float | None,
    deployable: bool,
    reason: str | None = None,
) -> dict[str, object]:
    candidate = {
        "name": name,
        "model_type": name,
        "deployable": deployable,
        "metrics": (
            {"recall_at_10": recall}
            if recall is not None
            else {}
        ),
        "users_evaluated": 94_762,
        "mlflow": {"run_id": f"{name}-run"},
        "artifacts": {},
    }

    if reason:
        candidate["deployable_reason"] = reason

    return candidate


class TestRankCandidates(unittest.TestCase):

    def test_orders_best_first(self):
        ranked = rank_candidates(
            [
                make_candidate(
                    "svd",
                    0.037452,
                    False,
                ),
                make_candidate(
                    "ials",
                    0.038064,
                    True,
                ),
            ]
        )

        self.assertEqual(
            [
                candidate["name"]
                for candidate in ranked
            ],
            ["ials", "svd"],
        )

    def test_candidates_without_the_metric_sort_last(self):
        ranked = rank_candidates(
            [
                make_candidate(
                    "unscored",
                    None,
                    True,
                ),
                make_candidate(
                    "svd",
                    0.001,
                    False,
                ),
            ]
        )

        self.assertEqual(
            ranked[0]["name"],
            "svd",
        )

        self.assertEqual(
            ranked[-1]["name"],
            "unscored",
        )


class TestSelectModel(unittest.TestCase):

    def test_promotes_best_deployable_candidate(self):
        selection = select_model(
            [
                make_candidate(
                    "ials",
                    0.038064,
                    True,
                ),
                make_candidate(
                    "svd",
                    0.037452,
                    False,
                ),
            ]
        )

        self.assertEqual(
            selection["selected"]["name"],
            "ials",
        )

        self.assertEqual(
            selection["primary_metric"],
            "recall_at_10",
        )

        self.assertEqual(
            selection["selected"]["mlflow"][
                "run_id"
            ],
            "ials-run",
        )

        self.assertEqual(
            selection["notes"],
            [],
        )

    def test_comparison_lists_every_candidate(self):
        selection = select_model(
            [
                make_candidate(
                    "ials",
                    0.038064,
                    True,
                ),
                make_candidate(
                    "svd",
                    0.037452,
                    False,
                ),
            ]
        )

        self.assertEqual(
            [
                entry["name"]
                for entry in selection[
                    "comparison"
                ]
            ],
            ["ials", "svd"],
        )

        self.assertFalse(
            selection["comparison"][1][
                "deployable"
            ]
        )

    def test_explains_skipping_a_better_but_undeployable_model(self):
        # If the baseline ever wins on the metric, the pipeline must say so
        # out loud rather than quietly promoting the runner-up.
        selection = select_model(
            [
                make_candidate(
                    "svd",
                    0.99,
                    False,
                    reason="No FAISS retrieval path.",
                ),
                make_candidate(
                    "ials",
                    0.038064,
                    True,
                ),
            ]
        )

        self.assertEqual(
            selection["selected"]["name"],
            "ials",
        )

        self.assertEqual(
            len(selection["notes"]),
            1,
        )

        self.assertIn(
            "svd",
            selection["notes"][0],
        )

        self.assertIn(
            "No FAISS retrieval path.",
            selection["notes"][0],
        )

    def test_no_candidates_is_an_error(self):
        with self.assertRaises(
            ModelSelectionError
        ):
            select_model([])

    def test_no_deployable_candidate_is_an_error(self):
        with self.assertRaises(
            ModelSelectionError
        ):
            select_model(
                [
                    make_candidate(
                        "svd",
                        0.037452,
                        False,
                    )
                ]
            )

    def test_deployable_without_the_metric_is_an_error(self):
        with self.assertRaises(
            ModelSelectionError
        ):
            select_model(
                [
                    make_candidate(
                        "ials",
                        None,
                        True,
                    )
                ]
            )

    def test_respects_an_alternative_primary_metric(self):
        candidates = [
            {
                "name": "ials",
                "model_type": "ials",
                "deployable": True,
                "metrics": {
                    "recall_at_10": 0.01,
                    "ndcg_at_10": 0.9,
                },
            },
            {
                "name": "other",
                "model_type": "ials",
                "deployable": True,
                "metrics": {
                    "recall_at_10": 0.9,
                    "ndcg_at_10": 0.01,
                },
            },
        ]

        self.assertEqual(
            select_model(
                candidates,
                primary_metric="ndcg_at_10",
            )["selected"]["name"],
            "ials",
        )


class TestCandidateLoading(unittest.TestCase):

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.directory.name
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_missing_files_yield_no_candidate(self):
        self.assertIsNone(
            load_ials_candidate(
                str(self.root / "absent.json")
            )
        )

        self.assertIsNone(
            load_svd_candidate(
                str(self.root / "absent.csv")
            )
        )

    def test_reads_the_evaluation_step_output(self):
        path = self.root / "evaluation.json"

        path.write_text(
            json.dumps(
                {
                    "model_type": "ials",
                    "deployable": True,
                    "metrics": {
                        "recall_at_10": 0.038064
                    },
                    "users_evaluated": 94_762,
                    "mlflow": {
                        "run_id": "abc123"
                    },
                    "artifacts": {
                        "ials_model": (
                            "models/ials/"
                            "ials_model.npz"
                        )
                    },
                }
            )
        )

        candidate = load_ials_candidate(
            str(path)
        )

        self.assertTrue(
            candidate["deployable"]
        )

        self.assertAlmostEqual(
            candidate["metrics"][
                "recall_at_10"
            ],
            0.038064,
        )

        self.assertEqual(
            candidate["mlflow"]["run_id"],
            "abc123",
        )

    def test_reads_the_svd_evaluation_csv(self):
        path = self.root / "svd_evaluation.csv"

        path.write_text(
            "model,factors,users_evaluated,"
            "precision_at_10,recall_at_10,"
            "hit_rate_at_10,ndcg_at_10\n"
            "SVD,50,94762,0.003745,0.037452,"
            "0.037452,0.020084\n"
        )

        candidate = load_svd_candidate(
            str(path)
        )

        self.assertAlmostEqual(
            candidate["metrics"][
                "recall_at_10"
            ],
            0.037452,
        )

        self.assertEqual(
            candidate["users_evaluated"],
            94_762,
        )

        # SVD is comparable but has no retrieval path, so it can never be
        # promoted no matter how it scores.
        self.assertFalse(
            candidate["deployable"]
        )

        self.assertIn(
            "deployable_reason",
            candidate,
        )

    def test_empty_csv_yields_no_candidate(self):
        path = self.root / "empty.csv"

        path.write_text(
            "model,recall_at_10\n"
        )

        self.assertIsNone(
            load_svd_candidate(str(path))
        )


if __name__ == "__main__":
    unittest.main()
