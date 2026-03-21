#!/usr/bin/env bash
# 하루 시작용: 어제 docker compose down 했어도, 이 스크립트 한 번으로
#   전체 스택(백그라운드) + API 준비 대기 + (선택) cloudflared 터널(Vercel용)
#
# 사용:
#   ./scripts/start-dev-day.sh              # dev 오버레이(핫리로드) + 터널 (기본)
#   ./scripts/start-dev-day.sh --no-dev     # docker-compose.yml 만 (이미지 그대로)
#   ./scripts/start-dev-day.sh --no-tunnel  # 도커만, 터널 없음
#   ./scripts/start-dev-day.sh --watch      # Compose watch(web 재빌드) — dev + 2.23+
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USE_DEV=true
TUNNEL=true
WATCH=false
for arg in "$@"; do
  case "$arg" in
    --no-dev) USE_DEV=false ;;
    --dev) USE_DEV=true ;;
    --no-tunnel) TUNNEL=false ;;
    --watch) WATCH=true ;;
    *)
      echo "알 수 없는 인자: $arg (지원: --no-dev, --dev, --no-tunnel, --watch)" >&2
      exit 1
      ;;
  esac
done

if $WATCH && ! $USE_DEV; then
  echo "--watch 는 dev 오버레이(--dev)와 함께만 사용됩니다. --dev 로 실행합니다." >&2
  USE_DEV=true
fi

COMPOSE=(docker compose)
if $USE_DEV; then
  COMPOSE+=(-f docker-compose.yml -f docker-compose.dev.yml)
  echo "==> Docker Compose up -d --build (dev: api/worker 소스 마운트·reload)…"
else
  COMPOSE+=(-f docker-compose.yml)
  echo "==> Docker Compose up -d --build (기본 이미지)…"
fi

if $WATCH; then
  "${COMPOSE[@]}" up -d --build --watch
else
  "${COMPOSE[@]}" up -d --build
fi

echo "==> API (http://127.0.0.1:23000/v1/legal/terms) 응답 대기…"
ok=false
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:23000/v1/legal/terms" >/dev/null 2>&1; then
    echo "API OK (약 ${i}s)"
    ok=true
    break
  fi
  sleep 1
done
if ! $ok; then
  echo "API가 120초 안에 응답하지 않습니다. 로그: docker compose logs api" >&2
  exit 1
fi

if $TUNNEL; then
  bash "$ROOT/scripts/tunnel-api.sh"
else
  echo ""
  echo "터널 생략 (--no-tunnel). Vercel 연동 시: ./scripts/tunnel-api.sh"
fi

CFILES="-f docker-compose.yml"
if $USE_DEV; then
  CFILES="-f docker-compose.yml -f docker-compose.dev.yml"
fi

echo ""
echo "==> 다음에 쓸 명령"
echo "    로그:     docker compose $CFILES logs -f"
echo "    전체 중지: docker compose $CFILES down"
echo "    웹 UI:    http://localhost:25173"
echo ""
