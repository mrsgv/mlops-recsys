import unittest
from pathlib import Path

import pandas as pd


ITEM_FEATURES_PATH = (
    "data/processed/video_games_items.parquet"
)

REQUIRED_COLUMNS = {
    "item_idx",
    "parent_asin",
    "metadata_found",
    "title",
    "main_category",
    "categories",
    "store",
    "brand",
    "price",
    "price_bucket",
    "title_length",
    "feature_count",
    "features_text",
    "description_text",
}


class TestItemFeatureTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not Path(ITEM_FEATURES_PATH).exists():
            raise unittest.SkipTest(
                f"{ITEM_FEATURES_PATH} not found."
            )

        cls.df = pd.read_parquet(
            ITEM_FEATURES_PATH
        )

    def test_required_columns_exist(self):
        self.assertTrue(
            REQUIRED_COLUMNS.issubset(
                self.df.columns
            )
        )

    def test_item_idx_is_unique(self):
        self.assertTrue(
            self.df["item_idx"].is_unique
        )

    def test_parent_asin_is_unique(self):
        self.assertTrue(
            self.df["parent_asin"].is_unique
        )

    def test_item_indices_are_contiguous(self):
        expected = list(
            range(len(self.df))
        )

        actual = (
            self.df
            .sort_values("item_idx")
            ["item_idx"]
            .tolist()
        )

        self.assertEqual(
            actual,
            expected,
        )

    def test_item_idx_has_no_nulls(self):
        self.assertFalse(
            self.df["item_idx"].isna().any()
        )

    def test_parent_asin_has_no_nulls(self):
        self.assertFalse(
            self.df["parent_asin"].isna().any()
        )

    def test_metadata_found_is_boolean(self):
        self.assertTrue(
            pd.api.types.is_bool_dtype(
                self.df["metadata_found"]
            )
        )

    def test_numeric_features_are_numeric(self):
        self.assertTrue(
            pd.api.types.is_numeric_dtype(
                self.df["title_length"]
            )
        )

        self.assertTrue(
            pd.api.types.is_numeric_dtype(
                self.df["feature_count"]
            )
        )


if __name__ == "__main__":
    unittest.main()