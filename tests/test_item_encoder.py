import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.item_features import ItemFeatureEncoder


class TestItemFeatureEncoder(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "item_idx": [0, 1, 2],
                "parent_asin": [
                    "A",
                    "B",
                    "C",
                ],
                "metadata_found": [True, True, True],
                "title": [
                    "Racing Game",
                    "Football Game",
                    "Adventure Game",
                ],
                "main_category": [
                    "Video Games",
                    "Video Games",
                    "Video Games",
                ],
                "categories": [
                    "PC Games Racing",
                    "PC Games Sports",
                    "PC Games Adventure",
                ],
                "store": [
                    "Store A",
                    "Store B",
                    "Store A",
                ],
                "brand": [
                    "Brand A",
                    "Brand B",
                    "Brand A",
                ],
                "price": [
                    10.0,
                    25.0,
                    50.0,
                ],
                "price_bucket": [
                    "10_to_25",
                    "25_to_50",
                    "50_to_100",
                ],
                "title_length": [
                    11,
                    13,
                    15,
                ],
                "feature_count": [
                    2,
                    3,
                    1,
                ],
                "features_text": [
                    "racing",
                    "football",
                    "adventure",
                ],
                "description_text": [
                    "racing game",
                    "football game",
                    "adventure game",
                ],
            }
        )

    def test_transform_shapes(self):
        encoder = ItemFeatureEncoder.fit(
            self.df,
            max_text_features=8,
        )

        features = encoder.transform(
            self.df
        )

        self.assertEqual(
            features["main_category"].shape,
            (3,),
        )

        self.assertEqual(
            features["brand"].shape,
            (3,),
        )

        self.assertEqual(
            features["store"].shape,
            (3,),
        )

        self.assertEqual(
            features["price_bucket"].shape,
            (3,),
        )

        self.assertEqual(
            features["numeric_features"].shape,
            (3, 3),
        )

        self.assertEqual(
            features["text_features"].shape[0],
            3,
        )

    def test_unknown_categories_map_to_zero(self):
        encoder = ItemFeatureEncoder.fit(
            self.df,
            max_text_features=8,
        )

        unknown = self.df.copy()
        unknown.loc[0, "brand"] = "UNKNOWN_BRAND"

        features = encoder.transform(
            unknown
        )

        self.assertEqual(
            features["brand"][0],
            0,
        )

    def test_save_and_load(self):
        encoder = ItemFeatureEncoder.fit(
            self.df,
            max_text_features=8,
        )

        original = encoder.transform(
            self.df
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pkl"

            encoder.save(path)

            loaded = (
                ItemFeatureEncoder.load(path)
            )

            restored = loaded.transform(
                self.df
            )

            for key in original:
                np.testing.assert_array_equal(
                    original[key],
                    restored[key],
                )
    def test_empty_text_is_handled(self):
        empty_text = self.df.copy()

        for column in [
            "title",
            "categories",
            "features_text",
            "description_text",
        ]:
            empty_text[column] = ""

        encoder = ItemFeatureEncoder.fit(
            empty_text,
            max_text_features=8,
        )

        features = encoder.transform(
            empty_text
        )

        self.assertEqual(
            features["text_features"].shape[0],
            3,
        )

if __name__ == "__main__":
    unittest.main()