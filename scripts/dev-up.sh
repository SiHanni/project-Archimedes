#!/usr/bin/env bash
# 로컬 개발 스택: API(Nest --watch) · worker(uvicorn --reload) · consumer(watchfiles) 자동 반영.
# (web 은 기본 up 만; 프론트 저장 시 재빌드는 아래 --watch 사용)
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.dev.yml)
if [[ "${1:-}" == "--watch" ]]; then
  shift
  echo "==> compose up --build --watch (web 소스 변경 시 web 이미지 재빌드)…"
  "${COMPOSE[@]}" up --build --watch "$@"
else
  echo "==> compose up --build (api/worker 는 소스 마운트 + 프로세스 내 자동 재시작)…"
  "${COMPOSE[@]}" up --build "$@"
fi
