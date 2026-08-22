import unittest
from pathlib import Path

import pandas as pd


KAFKA_OUTPUT_PATH = (
    "data/processed/kafka_interactions"
)

REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "rating",
    "timestamp",
    "ingested_at",
}


class TestKafkaOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not Path(KAFKA_OUTPUT_PATH).exists():
            raise unittest.SkipTest(
                f"{KAFKA_OUTPUT_PATH} not found."
            )

        cls.df = pd.read_parquet(
            KAFKA_OUTPUT_PATH
        )

    def test_required_columns_exist(self):
        self.assertTrue(
            REQUIRED_COLUMNS.issubset(
                self.df.columns
            )
        )

    def test_output_is_not_empty(self):
        self.assertGreater(
            len(self.df),
            0,
        )

    def test_required_columns_have_no_nulls(self):
        required = self.df[
            list(REQUIRED_COLUMNS)
        ]

        self.assertFalse(
            required.isna().any().any()
        )

    def test_ratings_are_valid(self):
        self.assertTrue(
            ((self.df["rating"] >= 1.0)
             & (self.df["rating"] <= 5.0))
            .all()
        )

    def test_ids_are_non_negative(self):
        self.assertTrue(
            (self.df["user_idx"] >= 0).all()
        )
        self.assertTrue(
            (self.df["item_idx"] >= 0).all()
        )


if __name__ == "__main__":
    unittest.main()