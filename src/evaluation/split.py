"""
Common train/test splitting utilities for the recommendation system.

Evaluation protocol
-------------------
The project uses deterministic chronological leave-one-out evaluation:

- For every user with at least two interactions:
    - the chronologically latest interaction is held out as test
    - all earlier interactions are used for training
- If multiple interactions have the same timestamp, item_idx is used as
  a deterministic tie-breaker.
- Users with fewer than two interactions are excluded because they cannot
  contribute both training history and a held-out test interaction.

This module is the single source of truth for model train/test splitting.
All recommendation models must use this function.
"""

from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "rating",
    "timestamp",
}


def chronological_train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a deterministic chronological leave-one-out split.

    Ordering is:
        user_idx -> timestamp -> item_idx

    For each eligible user:
        latest interaction -> test
        all earlier interactions -> train

    Parameters
    ----------
    df:
        Processed interaction dataframe.

    Returns
    -------
    train:
        Training interactions.

    test:
        One held-out interaction per eligible user.

    Raises
    ------
    ValueError
        If required columns are missing, the dataframe is empty,
        or no user has at least two interactions.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Input dataframe is missing required columns: "
            f"{sorted(missing)}"
        )

    data = df.copy()

    # Deterministic chronological ordering.
    #
    # timestamp defines chronology.
    # item_idx breaks ties when two interactions have the same timestamp.
    data = data.sort_values(
        ["user_idx", "timestamp", "item_idx"],
        kind="mergesort",
    )

    # Only users with enough history can participate in leave-one-out
    # evaluation.
    user_counts = data.groupby("user_idx").size()

    eligible_users = user_counts[user_counts >= 2].index

    data = data[
        data["user_idx"].isin(eligible_users)
    ].copy()

    if data.empty:
        raise ValueError(
            "No users have at least two interactions; "
            "cannot create train/test split."
        )

    # Last interaction according to the deterministic ordering is test.
    test = (
        data
        .groupby("user_idx", sort=False)
        .tail(1)
    )

    # All remaining interactions are training data.
    train = data.drop(index=test.index)

    # Return deterministic dataframes.
    train = (
        train
        .sort_values(
            ["user_idx", "timestamp", "item_idx"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values(
            ["user_idx"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return train, test


def validate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate invariants of a chronological leave-one-out split.

    Raises AssertionError if any invariant is violated.
    """

    # Every test user must have exactly one held-out interaction.
    test_counts = test.groupby("user_idx").size()

    assert test_counts.eq(1).all(), (
        "Every test user must have exactly one test interaction."
    )

    # Every test user must also have training history.
    train_users = set(train["user_idx"].unique())
    test_users = set(test["user_idx"].unique())

    assert test_users.issubset(train_users), (
        "Every test user must also appear in training data."
    )

    # Train/test interaction identities must not overlap.
    train_keys = set(
        zip(
            train["user_idx"],
            train["item_idx"],
            train["timestamp"],
        )
    )

    test_keys = set(
        zip(
            test["user_idx"],
            test["item_idx"],
            test["timestamp"],
        )
    )

    assert train_keys.isdisjoint(test_keys), (
        "Training and test interactions must not overlap."
    )

    # Test must never occur before the latest training timestamp.
    #
    # Equality is allowed because item_idx is used as the deterministic
    # tie-breaker when timestamps are identical.
    latest_train_timestamp = (
        train
        .groupby("user_idx")["timestamp"]
        .max()
    )

    test_timestamp = (
        test
        .set_index("user_idx")["timestamp"]
    )

    timestamp_difference = (
        test_timestamp - latest_train_timestamp
    )

    assert (timestamp_difference >= 0).all(), (
        "A test interaction occurs before the user's latest "
        "training interaction."
    )
