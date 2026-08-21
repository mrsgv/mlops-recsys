from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from src.retrieval.ials_retriever import (
    IALSFaissRetriever,
)
from src.serving.config import settings
from src.serving.schemas import (
    HealthResponse,
    ModelResponse,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)
from src.serving.service import (
    RecommendationService,
)

RECOMMENDATION_REQUESTS = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests.",
)

RECOMMENDATION_ERRORS = Counter(
    "recommendation_errors_total",
    "Total number of failed recommendation requests.",
)

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency in seconds.",
)

RECOMMENDATION_RESULTS = Counter(
    "recommendation_results_total",
    "Total number of recommendations returned.",
)

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(
    "recommendation-api"
)


service: RecommendationService | None = None


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Load all recommendation artifacts once at application startup.

    This is important for performance: we do not reload iALS,
    FAISS, or the item mapping on every request.
    """
    global service

    logger.info(
        "Loading recommendation artifacts..."
    )

    retriever = IALSFaissRetriever(
        ials_model_path=(
            settings.ials_model_path
        ),
        faiss_index_path=(
            settings.faiss_index_path
        ),
        faiss_metadata_path=(
            settings.faiss_metadata_path
        ),
        item_mapping_path=(
            settings.item_mapping_path
        ),
        interactions_path=(
            settings.interactions_path
        ),
    )

    service = RecommendationService(
        retriever=retriever,
        model_type=settings.model_type,
        model_version=settings.model_version,
    )

    logger.info(
        "Recommendation artifacts loaded: "
        "users=%s items=%s dimension=%s",
        retriever.num_users,
        retriever.num_items,
        retriever.dimension,
    )

    yield

    logger.info(
        "Recommendation API shutting down."
    )

    service = None


app = FastAPI(
    title="Video Games Recommendation API",
    description=(
        "Top-K recommendation service using "
        "iALS + FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Basic application health check."""
    return HealthResponse(
        status="ok",
        model_loaded=(
            service is not None
        ),
    )


@app.get(
    "/model",
    response_model=ModelResponse,
)
def model_info() -> ModelResponse:
    """Return information about the loaded model."""
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model is not loaded.",
        )

    info = service.model_info

    return ModelResponse(
        model_type=info.model_type,
        model_version=info.model_version,
        retriever=info.retriever,
        num_users=info.num_users,
        num_items=info.num_items,
        embedding_dimension=(
            info.embedding_dimension
        ),
        faiss_index_type=(
            info.faiss_index_type
        ),
        normalization=(
            info.normalization
        ),
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Return Top-K recommendations for an encoded user."""

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model is not loaded.",
        )

    if request.k > settings.max_recommendations:
        raise HTTPException(
            status_code=400,
            detail=(
                f"k must be <= "
                f"{settings.max_recommendations}."
            ),
        )

    RECOMMENDATION_REQUESTS.inc()
    start_time = perf_counter()

    try:
        result = service.recommend(
            user_idx=request.user_idx,
            k=request.k,
        )
    except ValueError as exc:
        RECOMMENDATION_ERRORS.inc()

        logger.warning(
            "Invalid recommendation request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        RECOMMENDATION_ERRORS.inc()

        logger.error(
            "Recommendation generation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Recommendation generation failed.",
        ) from exc

    finally:
        RECOMMENDATION_LATENCY.observe(
            perf_counter() - start_time
        )

    RECOMMENDATION_RESULTS.inc(len(result))

    latency_ms = (
        perf_counter()
        - start_time
    ) * 1000

    logger.info(
        "recommendation_request "
        "user_idx=%s k=%s results=%s "
        "latency_ms=%.3f model=%s",
        request.user_idx,
        request.k,
        len(result),
        latency_ms,
        service.model_info.model_type,
    )

    recommendations = [
        Recommendation(
            rank=int(row.rank),
            item_idx=int(row.item_idx),
            parent_asin=str(row.parent_asin),
            score=float(row.score),
        )
        for row in result.itertuples(index=False)
    ]

    return RecommendationResponse(
        user_idx=request.user_idx,
        model=service.model_info.model_type,
        model_version=service.model_info.model_version,
        k=request.k,
        recommendations=recommendations,
    )