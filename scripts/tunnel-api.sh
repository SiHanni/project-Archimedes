#!/usr/bin/env bash
# 로컬 API(기본 23000)를 HTTPS로 노출 — Vercel의 VITE_API_BASE 에 넣을 주소 출력.
#
# 모드 (우선순위):
#   1) CLOUDFLARE_TUNNEL_TOKEN + ARCHIMEDES_PUBLIC_API_BASE → 고정 URL (Zero Trust 이름 있는 터널)
#   2) ARCHIMEDES_CLOUDFLARED_CONFIG 또는 infra/cloudflared/config.yml + ARCHIMEDES_PUBLIC_API_BASE
#   3) 그 외 → Quick Tunnel (매번 trycloudflare.com 주소가 바뀜)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 선택: 프로젝트 루트 .env 에 토큰·고정 URL 넣어 두면 자동 로드
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

API_PORT="${ARCHIMEDES_TUNNEL_API_PORT:-23000}"
API_HOST="${ARCHIMEDES_TUNNEL_API_HOST:-127.0.0.1}"
LOG="${TMPDIR:-/tmp}/archimedes-cloudflared.log"
PIDFILE="${TMPDIR:-/tmp}/archimedes-cloudflared.pid"
STATE_PUBLIC="${TMPDIR:-/tmp}/archimedes-tunnel.public_base"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared 가 없습니다. 설치 예: brew install cloudflare/cloudflare/cloudflared" >&2
  exit 1
fi

# $2: fixed | quick
print_vercel_line() {
  local base="$1"
  local mode="${2:-quick}"
  base="${base%/}"
  echo "$base" >"$STATE_PUBLIC"
  echo ""
  echo "공개 API 베이스: $base"
  echo "Vercel Production → VITE_API_BASE=${base}/v1"
  if [[ "$mode" == "fixed" ]]; then
    echo "(고정 URL — Vercel env 는 최초·도메인 변경 시만 갱신)"
  else
    echo "(Quick Tunnel — 주소가 바뀌면 Vercel env + 재배포 필요)"
  fi
  echo "터널 중지:  $(cd "$(dirname "$0")" && pwd)/tunnel-api-stop.sh"
  echo ""
}

# 이미 떠 있음
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "이미 cloudflared 가 떠 있습니다 (PID $(cat "$PIDFILE")). 로그: $LOG"
  if [[ -f "$STATE_PUBLIC" ]]; then
    base=$(cat "$STATE_PUBLIC")
    if [[ "$base" == *"trycloudflare.com"* ]]; then
      print_vercel_line "$base" quick
    else
      print_vercel_line "$base" fixed
    fi
  elif u=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1) && [[ -n "${u:-}" ]]; then
    print_vercel_line "$u" quick
  else
    echo "고정 URL은 .env 의 ARCHIMEDES_PUBLIC_API_BASE 를 Vercel에 넣으면 됩니다."
  fi
  exit 0
fi

CFG=""
if [[ -n "${ARCHIMEDES_CLOUDFLARED_CONFIG:-}" && -f "${ARCHIMEDES_CLOUDFLARED_CONFIG}" ]]; then
  CFG="${ARCHIMEDES_CLOUDFLARED_CONFIG}"
elif [[ -f "$ROOT/infra/cloudflared/config.yml" ]]; then
  CFG="$ROOT/infra/cloudflared/config.yml"
fi

# --- 1) 토큰 (Zero Trust에서 발급) — 고정 호스트명 ---
if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  if [[ -z "${ARCHIMEDES_PUBLIC_API_BASE:-}" ]]; then
    echo "CLOUDFLARE_TUNNEL_TOKEN 을 쓰려면 ARCHIMEDES_PUBLIC_API_BASE 도 필요합니다." >&2
    echo "예: ARCHIMEDES_PUBLIC_API_BASE=https://api.본인도메인.com  (Zero Trust Public hostname 과 동일)" >&2
    echo "설정법: docs/cloudflare-stable-tunnel.md" >&2
    exit 1
  fi
  echo "==> Cloudflare 이름 있는 터널 (토큰) → ${ARCHIMEDES_PUBLIC_API_BASE}"
  nohup cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  print_vercel_line "$ARCHIMEDES_PUBLIC_API_BASE" fixed
  exit 0
fi

# --- 2) config.yml (tunnel login + credentials) ---
if [[ -n "${CFG}" ]]; then
  if [[ -z "${ARCHIMEDES_PUBLIC_API_BASE:-}" ]]; then
    echo "config.yml 터널 사용 시 ARCHIMEDES_PUBLIC_API_BASE 를 설정하세요 (Public hostname 의 https URL)." >&2
    echo "예: export ARCHIMEDES_PUBLIC_API_BASE=https://api.example.com" >&2
    exit 1
  fi
  echo "==> cloudflared tunnel --config $CFG"
  nohup cloudflared tunnel --config "$CFG" run >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  print_vercel_line "$ARCHIMEDES_PUBLIC_API_BASE" fixed
  exit 0
fi

# --- 3) Quick Tunnel (매번 주소 변경) ---
echo "==> cloudflared Quick Tunnel → http://${API_HOST}:${API_PORT}"
nohup cloudflared tunnel --url "http://${API_HOST}:${API_PORT}" >"$LOG" 2>&1 &
echo $! >"$PIDFILE"

URL=""
for _ in $(seq 1 20); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1 || true)
  if [[ -n "${URL:-}" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "${URL:-}" ]]; then
  echo "URL을 아직 못 읽었습니다. 로그: $LOG (PID $(cat "$PIDFILE"))" >&2
  exit 1
fi

print_vercel_line "$URL" quick
