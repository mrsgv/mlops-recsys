from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request body for Top-K recommendations."""

    user_idx: int = Field(
        ...,
        ge=0,
        description="Encoded user index.",
    )

    k: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of recommendations.",
    )


class Recommendation(BaseModel):
    """Single recommendation."""

    rank: int
    item_idx: int
    parent_asin: str
    score: float


class RecommendationResponse(BaseModel):
    """Recommendation API response."""

    user_idx: int
    model: str
    model_version: str
    k: int
    recommendations: list[
        Recommendation
    ]


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    model_loaded: bool


class ModelResponse(BaseModel):
    """Loaded model metadata."""

    model_type: str
    model_version: str
    retriever: str
    num_users: int
    num_items: int
    embedding_dimension: int
    faiss_index_type: str
    normalization: bool