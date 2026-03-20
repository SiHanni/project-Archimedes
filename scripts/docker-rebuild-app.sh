#!/usr/bin/env bash
# 워커·큐 소비·웹(프론트) 이미지 재빌드 후 컨테이너 재기동.
# 파이프라인/프론트 수정 후 로컬에서 자주 쓰는 용도.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> docker compose build (worker worker-consumer web)…"
docker compose build worker worker-consumer web
echo "==> docker compose up -d …"
docker compose up -d worker worker-consumer web
echo "==> done. web: http://localhost:25173 (또는 compose에 설정한 포트)"
