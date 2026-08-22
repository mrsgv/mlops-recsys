"""
Tests for the model family registry.

The registry exists so that adding a model family does not require editing
the sweep, selection, indexing and publishing steps — the shape of coupling
that let the deployed champion sit at ``alpha=1.0`` unchallenged, because
changing it meant touching a trainer with hyperparameters frozen as module
constants.

Two properties matter most:

- slugs are deterministic, because they become artifact directory names and
  selection discovers candidates by globbing those directories
- deployability is declared per family and is honest about neighbourhood
  models, which have no factors and therefore no FAISS retrieval path
"""

import unittest

from src.models.registry import (
    FAMILIES,
    UnknownFamilyError,
    extract_factors,
    get_family,
    variant_slug,
)


class TestGetFamily(unittest.TestCase):

    def test_returns_a_registered_family(self):
        self.assertEqual(
            get_family("als").name,
            "als",
        )

    def test_unknown_family_lists_the_valid_options(self):
        with self.assertRaises(
            UnknownFamilyError
        ) as raised:
            get_family("transformer")

        message = str(raised.exception)

        self.assertIn("als", message)

        self.assertIn("bpr", message)


class TestVariantSlug(unittest.TestCase):

    def test_slug_is_readable_and_ordered(self):
        self.assertEqual(
            variant_slug(
                "als",
                {
                    "factors": 256,
                    "regularization": 0.1,
                    "iterations": 20,
                    "alpha": 40.0,
                },
            ),
            "als-f256-r0p1-i20-a40",
        )

    def test_slug_ignores_dict_ordering(self):
        # The same hyperparameters must produce the same directory however
        # the grid entry happened to be written.
        first = variant_slug(
            "als",
            {
                "alpha": 40.0,
                "factors": 256,
                "iterations": 20,
                "regularization": 0.1,
            },
        )

        second = variant_slug(
            "als",
            {
                "factors": 256,
                "regularization": 0.1,
                "iterations": 20,
                "alpha": 40.0,
            },
        )

        self.assertEqual(first, second)

    def test_whole_floats_and_ints_agree(self):
        # alpha=40 and alpha=40.0 are the same experiment, so they must not
        # produce two candidate directories.
        self.assertEqual(
            variant_slug(
                "als",
                {"alpha": 40},
            ),
            variant_slug(
                "als",
                {"alpha": 40.0},
            ),
        )

    def test_seed_is_excluded_from_the_slug(self):
        # random_state describes the run, not the model, so it must not
        # change the artifact's identity.
        self.assertEqual(
            variant_slug(
                "als",
                {
                    "factors": 64,
                    "random_state": 42,
                },
            ),
            "als-f64",
        )

    def test_slug_is_filesystem_safe(self):
        slug = variant_slug(
            "bpr",
            {
                "factors": 128,
                "learning_rate": 0.05,
                "regularization": 0.001,
            },
        )

        for character in "/\\ .:":
            self.assertNotIn(
                character,
                slug,
            )


class TestDeployabilityDeclarations(unittest.TestCase):

    def test_factor_families_are_deployable(self):
        # Verified rather than assumed: for ALS, BPR and LMF, ranking by raw
        # inner product over the saved factors reproduces implicit's own
        # recommend() top-K exactly, which is what the FAISS index computes.
        for name in ("als", "bpr", "lmf"):
            self.assertTrue(
                get_family(name).deployable,
                f"{name} should be servable",
            )

    def test_neighbourhood_families_are_not_deployable(self):
        for name in ("bm25", "cosine"):
            family = get_family(name)

            self.assertFalse(
                family.deployable,
                f"{name} has no factors",
            )

            self.assertIn(
                "no item factors",
                family.deployable_reason,
            )

    def test_undeployable_families_always_explain_themselves(self):
        # Selection reports why a better-scoring model was skipped, so the
        # reason cannot be optional in practice.
        for family in FAMILIES.values():
            if family.deployable:
                continue

            self.assertTrue(
                family.deployable_reason,
                f"{family.name} must say why it cannot be served",
            )

    def test_every_family_describes_itself(self):
        # The descriptions become the comparison table in the report.
        for family in FAMILIES.values():
            self.assertTrue(
                family.description,
            )

    def test_neighbourhood_families_reject_a_seed(self):
        # They are deterministic and their constructors do not take
        # random_state, so the sweep must not inject one.
        for name in ("bm25", "cosine"):
            self.assertFalse(
                get_family(
                    name
                ).accepts_random_state
            )


class TestExtractFactors(unittest.TestCase):

    def test_returns_none_without_factors(self):
        class NoFactors:
            pass

        self.assertIsNone(
            extract_factors(NoFactors())
        )

    def test_returns_both_matrices(self):
        import numpy as np

        class WithFactors:
            user_factors = np.zeros(
                (3, 8),
                dtype=np.float64,
            )

            item_factors = np.zeros(
                (4, 8),
                dtype=np.float64,
            )

        factors = extract_factors(
            WithFactors()
        )

        self.assertIsNotNone(factors)

        users, items = factors

        self.assertEqual(
            users.shape,
            (3, 8),
        )

        self.assertEqual(
            items.shape,
            (4, 8),
        )

        # FAISS requires float32; converting here keeps every caller from
        # having to remember.
        self.assertEqual(
            users.dtype,
            np.float32,
        )

        self.assertEqual(
            items.dtype,
            np.float32,
        )


if __name__ == "__main__":
    unittest.main()
