import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.validate_data import (
    DataValidationError,
    check_contiguous_index,
    validate_item_mapping_frame,
    validate_processed,
    validate_processed_frame,
    validate_raw,
    validate_raw_frame,
)


def make_raw_frame(rows: int = 20_000) -> pd.DataFrame:
    """Build a raw frame large enough to clear the row-count floor."""
    return pd.DataFrame(
        {
            "user_id": [
                f"U{index % 500}"
                for index in range(rows)
            ],
            "parent_asin": [
                f"B{index % 300}"
                for index in range(rows)
            ],
            "rating": [
                1 + (index % 5)
                for index in range(rows)
            ],
            "timestamp": [
                1_600_000_000 + index
                for index in range(rows)
            ],
        }
    )


def make_processed_frame() -> pd.DataFrame:
    """Build a valid processed frame: contiguous indices, no duplicates."""
    users = []
    items = []

    for user_idx in range(4_000):
        for offset in range(3):
            users.append(user_idx)
            items.append(
                (user_idx + offset) % 3_000
            )

    return pd.DataFrame(
        {
            "user_idx": users,
            "item_idx": items,
            "rating": [
                1 + (index % 5)
                for index in range(len(users))
            ],
            "timestamp": [
                1_600_000_000 + index
                for index in range(len(users))
            ],
        }
    )


class TestCheckContiguousIndex(unittest.TestCase):

    def test_returns_cardinality(self):
        df = pd.DataFrame(
            {"item_idx": [0, 1, 2, 2, 1]}
        )

        self.assertEqual(
            check_contiguous_index(
                df,
                "item_idx",
                "Test",
            ),
            3,
        )

    def test_rejects_gap(self):
        df = pd.DataFrame(
            {"item_idx": [0, 1, 3]}
        )

        with self.assertRaises(
            DataValidationError
        ):
            check_contiguous_index(
                df,
                "item_idx",
                "Test",
            )

    def test_rejects_non_zero_start(self):
        df = pd.DataFrame(
            {"item_idx": [1, 2, 3]}
        )

        with self.assertRaises(
            DataValidationError
        ):
            check_contiguous_index(
                df,
                "item_idx",
                "Test",
            )

    def test_rejects_non_integer(self):
        df = pd.DataFrame(
            {"item_idx": ["a", "b"]}
        )

        with self.assertRaises(
            DataValidationError
        ):
            check_contiguous_index(
                df,
                "item_idx",
                "Test",
            )


class TestValidateRawFrame(unittest.TestCase):

    def test_accepts_valid_frame(self):
        report = validate_raw_frame(
            make_raw_frame()
        )

        self.assertEqual(
            report["stage"],
            "raw",
        )

        self.assertEqual(
            report["rows"],
            20_000,
        )

        self.assertEqual(
            report["users"],
            500,
        )

        self.assertEqual(
            report["items"],
            300,
        )

    def test_rejects_missing_column(self):
        df = make_raw_frame().drop(
            columns=["timestamp"]
        )

        with self.assertRaises(
            DataValidationError
        ):
            validate_raw_frame(df)

    def test_rejects_too_few_rows(self):
        with self.assertRaises(
            DataValidationError
        ):
            validate_raw_frame(
                make_raw_frame(rows=100)
            )

    def test_rejects_nulls(self):
        df = make_raw_frame()
        df.loc[0, "user_id"] = None

        with self.assertRaises(
            DataValidationError
        ):
            validate_raw_frame(df)

    def test_rejects_out_of_range_rating(self):
        df = make_raw_frame()
        df.loc[0, "rating"] = 9

        with self.assertRaises(
            DataValidationError
        ):
            validate_raw_frame(df)

    def test_rejects_non_positive_timestamp(self):
        df = make_raw_frame()
        df.loc[0, "timestamp"] = 0

        with self.assertRaises(
            DataValidationError
        ):
            validate_raw_frame(df)


class TestValidateProcessedFrame(unittest.TestCase):

    def test_accepts_valid_frame(self):
        report = validate_processed_frame(
            make_processed_frame()
        )

        self.assertEqual(
            report["stage"],
            "processed",
        )

        self.assertEqual(
            report["users"],
            4_000,
        )

        self.assertEqual(
            report["items"],
            3_000,
        )

        self.assertEqual(
            report["users_eligible_for_evaluation"],
            4_000,
        )

    def test_rejects_duplicate_interactions(self):
        df = make_processed_frame()

        duplicated = pd.concat(
            [
                df,
                df.head(1),
            ],
            ignore_index=True,
        )

        with self.assertRaises(
            DataValidationError
        ):
            validate_processed_frame(duplicated)

    def test_rejects_non_contiguous_items(self):
        df = make_processed_frame()

        df["item_idx"] = df["item_idx"] + 1

        with self.assertRaises(
            DataValidationError
        ):
            validate_processed_frame(df)


class TestValidateItemMappingFrame(unittest.TestCase):

    def make_mapping(self, num_items: int = 3):
        return pd.DataFrame(
            {
                "item_idx": list(
                    range(num_items)
                ),
                "parent_asin": [
                    f"B{index}"
                    for index in range(num_items)
                ],
            }
        )

    def test_accepts_matching_mapping(self):
        report = validate_item_mapping_frame(
            self.make_mapping(),
            num_items=3,
        )

        self.assertEqual(
            report["rows"],
            3,
        )

    def test_rejects_size_mismatch(self):
        with self.assertRaises(
            DataValidationError
        ):
            validate_item_mapping_frame(
                self.make_mapping(num_items=3),
                num_items=4,
            )

    def test_rejects_duplicate_parent_asin(self):
        mapping = self.make_mapping()

        mapping.loc[1, "parent_asin"] = "B0"

        with self.assertRaises(
            DataValidationError
        ):
            validate_item_mapping_frame(
                mapping,
                num_items=3,
            )


class TestValidateFilesOnDisk(unittest.TestCase):
    """
    Exercise the file-reading entry points the Airflow tasks call.

    The frame validators are covered above; these tests cover what the
    pipeline actually invokes: reading gzipped CSV and Parquet — including
    Parquet written as a directory, which is what Spark produces — and
    failing usefully when a file has not been pulled.
    """

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.directory.name
        )

    def tearDown(self):
        self.directory.cleanup()

    def write_raw(self) -> Path:
        path = self.root / "raw.csv.gz"

        make_raw_frame().to_csv(
            path,
            index=False,
            compression="gzip",
        )

        return path

    def write_processed(
        self,
        as_directory: bool = False,
    ) -> tuple[Path, Path]:
        df = make_processed_frame()

        interactions = (
            self.root / "video_games.parquet"
        )

        if as_directory:
            # Mimic a Spark write: a directory of part files.
            interactions.mkdir()

            half = len(df) // 2

            df.iloc[:half].to_parquet(
                interactions / "part-0.parquet",
                index=False,
            )

            df.iloc[half:].to_parquet(
                interactions / "part-1.parquet",
                index=False,
            )
        else:
            df.to_parquet(
                interactions,
                index=False,
            )

        mapping = (
            self.root / "item_mapping.parquet"
        )

        pd.DataFrame(
            {
                "item_idx": range(3_000),
                "parent_asin": [
                    f"B{index:05d}"
                    for index in range(3_000)
                ],
            }
        ).to_parquet(
            mapping,
            index=False,
        )

        return interactions, mapping

    def test_validates_gzipped_raw_csv(self):
        report = validate_raw(
            str(self.write_raw())
        )

        self.assertEqual(
            report["rows"],
            20_000,
        )

        self.assertFalse(report["sampled"])

    def test_sample_rows_limits_the_read(self):
        # The floor still applies, so a sample below it must fail rather
        # than silently pass a truncated check.
        with self.assertRaises(
            DataValidationError
        ):
            validate_raw(
                str(self.write_raw()),
                sample_rows=100,
            )

    def test_validates_processed_parquet_file(self):
        interactions, mapping = (
            self.write_processed()
        )

        report = validate_processed(
            str(interactions),
            str(mapping),
        )

        self.assertEqual(
            report["users"],
            4_000,
        )

        self.assertEqual(
            report["items"],
            3_000,
        )

        self.assertEqual(
            report["item_mapping"]["rows"],
            3_000,
        )

    def test_validates_spark_style_parquet_directory(self):
        interactions, mapping = (
            self.write_processed(
                as_directory=True,
            )
        )

        report = validate_processed(
            str(interactions),
            str(mapping),
        )

        self.assertEqual(
            report["interactions"],
            len(make_processed_frame()),
        )

    def test_missing_file_points_at_dvc_pull(self):
        _, mapping = self.write_processed()

        with self.assertRaises(
            DataValidationError
        ) as context:
            validate_processed(
                str(self.root / "absent.parquet"),
                str(mapping),
            )

        self.assertIn(
            "dvc pull",
            str(context.exception),
        )

    def test_mapping_smaller_than_item_space_fails(self):
        interactions, _ = (
            self.write_processed()
        )

        short_mapping = (
            self.root / "short_mapping.parquet"
        )

        pd.DataFrame(
            {
                "item_idx": range(2_999),
                "parent_asin": [
                    f"B{index:05d}"
                    for index in range(2_999)
                ],
            }
        ).to_parquet(
            short_mapping,
            index=False,
        )

        with self.assertRaises(
            DataValidationError
        ):
            validate_processed(
                str(interactions),
                str(short_mapping),
            )


if __name__ == "__main__":
    unittest.main()
