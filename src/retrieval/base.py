from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Retriever(ABC):
    """Common interface for candidate retrieval."""

    @abstractmethod
    def search(
        self,
        query_vector,
        k: int,
    ) -> tuple[list[int], list[float]]:
        """Return item IDs and scores."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the retrieval index."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Load a retrieval index."""
        raise NotImplementedError