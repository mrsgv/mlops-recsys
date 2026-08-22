#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-mlops-recsys-api:local}"
CONTAINER="${CONTAINER:-mlops-recsys-api-test}"
PORT="${PORT:-8000}"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker run -d \
    --name "$CONTAINER" \
    -p "${PORT}:8000" \
    "$IMAGE" >/dev/null

echo "Waiting for container..."

for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
        break
    fi
    sleep 2
done

curl -fsS "http://127.0.0.1:${PORT}/health"
echo

curl -fsS "http://127.0.0.1:${PORT}/model"
echo

curl -fsS -X POST \
    "http://127.0.0.1:${PORT}/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_idx":0,"k":10}'
echo

curl -fsS "http://127.0.0.1:${PORT}/metrics/"
echo

echo "Container smoke test passed."