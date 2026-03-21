#!/usr/bin/env bash
# 고정 URL 터널 준비 상태 점검 + 안내 (Cloudflare Zero Trust 토큰은 본인이 대시보드에서 발급)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DOC="$ROOT/docs/cloudflare-stable-tunnel.md"
EXAMPLE="$ROOT/env.tunnel.example"

echo "=== Archimedes — 고정 터널(Vercel용) 체크 ==="
echo ""

if [[ ! -f "$ROOT/.env" ]]; then
  echo "[1] 프로젝트 루트에 .env 파일이 없습니다."
  echo "    아래 중 하나를 하세요:"
  echo "    cp env.tunnel.example .env   # 그다음 편집기로 값 채우기"
  echo "    또는 기존 .env 에 env.tunnel.example 내용을 붙여 넣기"
  echo ""
  echo "필수 변수:"
  echo "  CLOUDFLARE_TUNNEL_TOKEN=...     (Zero Trust → Tunnels 에서)"
  echo "  ARCHIMEDES_PUBLIC_API_BASE=https://api.본인도메인.com"
  echo ""
  echo "상세: $DOC"
  exit 1
fi

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

missing=0
if [[ -z "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "[!] CLOUDFLARE_TUNNEL_TOKEN 이 비어 있습니다."
  missing=1
fi
if [[ -z "${ARCHIMEDES_PUBLIC_API_BASE:-}" ]]; then
  echo "[!] ARCHIMEDES_PUBLIC_API_BASE 가 비어 있습니다."
  missing=1
fi

if [[ "$missing" -eq 1 ]]; then
  echo ""
  echo "템플릿: $EXAMPLE"
  echo "가이드: $DOC"
  exit 1
fi

BASE="${ARCHIMEDES_PUBLIC_API_BASE%/}"
echo "[OK] 환경 변수 형식은 갖춰졌습니다."
echo "     ARCHIMEDES_PUBLIC_API_BASE=$BASE"
echo ""

if curl -sf "${BASE}/v1/legal/terms" >/dev/null 2>&1; then
  echo "[OK] ${BASE}/v1/legal/terms 응답 확인 — 터널+API 정상일 수 있습니다."
else
  echo "[?] ${BASE}/v1/legal/terms 에 아직 연결되지 않습니다."
  echo "    - Docker API 가 떠 있는지:  curl -s http://127.0.0.1:23000/v1/legal/terms"
  echo "    - cloudflared 가 토큰으로 실행 중인지:  ./scripts/tunnel-api.sh"
  echo "    - Zero Trust Public hostname 이 ${BASE} 로 이 호스트의 23000 에 연결됐는지 대시보드 확인"
fi

echo ""
echo "Vercel Production 환경변수 (한 번 설정):"
echo "  VITE_API_BASE=${BASE}/v1"
echo ""
echo "터널 실행(백그라운드):  ./scripts/tunnel-api.sh"
echo "도커+터널 한 번에:      ./scripts/start-dev-day.sh"
echo ""
