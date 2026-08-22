from __future__ import annotations

import argparse

from src.retrieval.factor_retriever import (
    FactorFaissRetriever,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Video Games recommendations "
            "using iALS + FAISS."
        )
    )

    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Encoded user_idx.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retriever = (
        FactorFaissRetriever()
    )

    result = retriever.recommend(
        user_idx=args.user_id,
        k=args.k,
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()