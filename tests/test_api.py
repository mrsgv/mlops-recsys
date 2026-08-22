import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.serving.app import app


REQUIRED_ARTIFACTS = [
    Path(
        "models/ials/ials_model.npz"
    ),
    Path(
        "models/retrieval/faiss.index"
    ),
    Path(
        "models/retrieval/index_metadata.json"
    ),
    Path(
        "data/processed/item_mapping.parquet"
    ),
    Path(
        "data/processed/video_games.parquet"
    ),
]


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        missing = [
            str(path)
            for path in REQUIRED_ARTIFACTS
            if not path.exists()
        ]

        if missing:
            raise unittest.SkipTest(
                "Required serving artifacts missing: "
                + ", ".join(missing)
            )

        # IMPORTANT:
        # Entering the TestClient context runs the FastAPI
        # lifespan startup code, which loads iALS + FAISS.
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        # Properly execute FastAPI shutdown lifecycle.
        cls.client.__exit__(
            None,
            None,
            None,
        )

    def test_health(self):
        response = self.client.get(
            "/health"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            body["status"],
            "ok",
        )

        self.assertTrue(
            body["model_loaded"]
        )

    def test_model(self):
        response = self.client.get(
            "/model"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            body["model_type"],
            "ials",
        )

        self.assertEqual(
            body["model_version"],
            "1",
        )

        self.assertEqual(
            body["retriever"],
            "faiss",
        )

        self.assertEqual(
            body["num_users"],
            94762,
        )

        self.assertEqual(
            body["num_items"],
            25612,
        )

        self.assertEqual(
            body["embedding_dimension"],
            64,
        )

        self.assertEqual(
            body["faiss_index_type"],
            "IndexFlatIP",
        )

        self.assertFalse(
            body["normalization"]
        )

    def test_recommend(self):
        response = self.client.post(
            "/recommend",
            json={
                "user_idx": 0,
                "k": 10,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            body["user_idx"],
            0,
        )

        self.assertEqual(
            body["model"],
            "ials",
        )

        self.assertEqual(
            body["model_version"],
            "1",
        )

        self.assertEqual(
            body["k"],
            10,
        )

        self.assertEqual(
            len(
                body["recommendations"]
            ),
            10,
        )

        self.assertEqual(
            body["recommendations"][0]["rank"],
            1,
        )

        self.assertEqual(
            [
                item["rank"]
                for item
                in body["recommendations"]
            ],
            list(range(1, 11)),
        )

        # Verify recommendation schema.
        first = (
            body["recommendations"][0]
        )

        self.assertIn(
            "item_idx",
            first,
        )

        self.assertIn(
            "parent_asin",
            first,
        )

        self.assertIn(
            "score",
            first,
        )

    def test_invalid_user(self):
        response = self.client.post(
            "/recommend",
            json={
                "user_idx": 999999,
                "k": 10,
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_invalid_k(self):
        response = self.client.post(
            "/recommend",
            json={
                "user_idx": 0,
                "k": 0,
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )
    
    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIn(
            "recommendation_requests_total",
            response.text,
        )

        self.assertIn(
            "recommendation_errors_total",
            response.text,
        )

        self.assertIn(
            "recommendation_latency_seconds",
            response.text,
        )

        self.assertIn(
            "recommendation_results_total",
            response.text,
        )

    def test_metrics_change_after_recommendation(self):
        before = self.client.get("/metrics")
        self.assertEqual(before.status_code, 200)

        self.client.post(
            "/recommend",
            json={
                "user_idx": 0,
                "k": 10,
            },
        )

        after = self.client.get("/metrics")
        self.assertEqual(after.status_code, 200)

        self.assertIn(
            "recommendation_requests_total",
            after.text,
        )


if __name__ == "__main__":
    unittest.main()