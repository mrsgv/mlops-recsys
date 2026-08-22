FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime library required by native ML dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements-serving.txt

# Application code required by the online serving path.
COPY src/serving ./src/serving
COPY src/retrieval ./src/retrieval

# Runtime model/retrieval artifacts.
#
# The model is copied from the canonical promoted path, not from a
# family-specific one. Selection stages whichever model won to this path, so
# the image builds identically whether ALS, BPR or LMF was promoted — the
# previous COPY of models/ials/ials_model.npz would have baked in iALS no
# matter what the pipeline actually chose.
COPY models/promoted/model.npz ./models/promoted/model.npz
COPY models/retrieval ./models/retrieval

# Runtime data required by IALSFaissRetriever.
COPY data/processed/item_mapping.parquet \
     ./data/processed/item_mapping.parquet

COPY data/processed/video_games.parquet \
     ./data/processed/video_games.parquet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]