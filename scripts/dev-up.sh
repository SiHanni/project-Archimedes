#!/usr/bin/env bash
# 로컬 개발 스택: API(Nest --watch) · worker(uvicorn --reload) · consumer(watchfiles) 자동 반영.
# (web 은 기본 up 만; 프론트 저장 시 재빌드는 아래 --watch 사용)
#
# Vercel에서 로컬 API를 쓰려면 터널이 필요함:
#   ./scripts/dev-up.sh --tunnel
#   → compose 뒤에 cloudflared 를 띄우고, 출력된 VITE_API_BASE 를 Vercel에 넣고 재배포.
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.dev.yml)

WATCH=false
TUNNEL=false
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=true ;;
    --tunnel) TUNNEL=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

if $TUNNEL; then
  # 하루 한 번 / 재기동은 start-dev-day.sh 로 통일 (API 대기 + 터널)
  EXTRA=(--dev)
  $WATCH && EXTRA+=(--watch)
  exec bash "$(dirname "$0")/start-dev-day.sh" "${EXTRA[@]}"
fi

if $WATCH; then
  echo "==> compose up --build --watch (web 소스 변경 시 web 이미지 재빌드)…"
  "${COMPOSE[@]}" up --build --watch "${ARGS[@]}"
else
  echo "==> compose up --build (api/worker 는 소스 마운트 + 프로세스 내 자동 재시작)…"
  "${COMPOSE[@]}" up --build "${ARGS[@]}"
fi
