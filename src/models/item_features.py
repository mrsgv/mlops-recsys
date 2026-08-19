from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path
import pickle


TEXT_COLUMNS = [
    "title",
    "categories",
    "features_text",
    "description_text",
]

CATEGORICAL_COLUMNS = [
    "main_category",
    "brand",
    "store",
    "price_bucket",
]

NUMERIC_COLUMNS = [
    "price",
    "title_length",
    "feature_count",
]


def _clean_text(value: object) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Combine static textual metadata into one representation."""
    return df[TEXT_COLUMNS].fillna("").astype(str).agg(
        " ".join,
        axis=1,
    )


def build_vocabulary(
    values: Iterable[object],
) -> dict[str, int]:
    """
    Build a deterministic categorical vocabulary.

    Index 0 is reserved for unknown values.
    """
    cleaned = {
        _clean_text(value)
        for value in values
        if _clean_text(value)
    }

    vocabulary = {
        "<UNK>": 0
    }

    for idx, value in enumerate(
        sorted(cleaned),
        start=1,
    ):
        vocabulary[value] = idx

    return vocabulary


def encode_column(
    values: Iterable[object],
    vocabulary: dict[str, int],
) -> np.ndarray:
    """Encode categorical values using a vocabulary."""
    return np.asarray(
        [
            vocabulary.get(
                _clean_text(value),
                0,
            )
            for value in values
        ],
        dtype=np.int64,
    )


@dataclass
class ItemFeatureEncoder:
    vocabularies: dict[str, dict[str, int]]
    vectorizer: TfidfVectorizer
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        max_text_features: int = 256,
    ) -> "ItemFeatureEncoder":

        vocabularies = {
            column: build_vocabulary(
                df[column]
            )
            for column in CATEGORICAL_COLUMNS
        }

        text = build_combined_text(df)

        if not text.str.strip().any():
            vectorizer = TfidfVectorizer(
                max_features=1
            )

            # Fit on a harmless placeholder so that the
            # vectorizer remains serializable and transformable.
            vectorizer.fit(["placeholder"])
        else:
            n_documents = len(text)

            if n_documents < 10:
                min_df = 1
                max_df = 1.0
            else:
                min_df = 2
                max_df = 0.98

            vectorizer = TfidfVectorizer(
                max_features=max_text_features,
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=max_df,
            )

            vectorizer.fit(text)


        numeric_means = {}
        numeric_stds = {}

        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype(float)

            mean = float(values.mean())

            std = float(values.std())

            if not np.isfinite(std) or std == 0.0:
                std = 1.0

            numeric_means[column] = (
                mean if np.isfinite(mean) else 0.0
            )

            numeric_stds[column] = std

        return cls(
            vocabularies=vocabularies,
            vectorizer=vectorizer,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
        )

    def transform(
        self,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:

        outputs = {}

        for column, vocabulary in self.vocabularies.items():
            outputs[column] = encode_column(
                df[column],
                vocabulary,
            )

        text = build_combined_text(df)

        text_features = self.vectorizer.transform(
            text
        ).toarray().astype(np.float32)

        outputs["text_features"] = text_features

        numeric_features = []

        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype(float)

            values = values.fillna(
                self.numeric_means[column]
            )

            normalized = (
                values.to_numpy()
                - self.numeric_means[column]
            ) / self.numeric_stds[column]

            numeric_features.append(
                normalized.astype(np.float32)
            )

        outputs["numeric_features"] = np.column_stack(
            numeric_features
        ).astype(np.float32)

        return outputs

    def save(self, path: str | Path) -> None:
        """Persist the fitted encoder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ItemFeatureEncoder":
        """Load a previously fitted encoder."""
        path = Path(path)

        with path.open("rb") as fh:
            encoder = pickle.load(fh)

        if not isinstance(encoder, cls):
            raise TypeError(
                "Loaded object is not an ItemFeatureEncoder."
            )

        return encoder