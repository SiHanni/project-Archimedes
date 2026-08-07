#!/usr/bin/env bash
# API·워커·큐 소비·웹(프론트) 이미지 재빌드 후 컨테이너 재기동.
# 파이프라인/프론트 수정 후 로컬에서 자주 쓰는 용도.
# (일상 개발은 ./scripts/dev-up.sh 로 소스 마운트 + 자동 재시작이 더 빠름.)
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> docker compose build (api worker worker-consumer web)…"
docker compose build api worker worker-consumer web
echo "==> docker compose up -d …"
docker compose up -d api worker worker-consumer web
echo "==> done. web: http://localhost:25173 (또는 compose에 설정한 포트)"

# 터널은 기본으로 켜지 않는다 — 로컬 API 를 공개 인터넷에 노출하는 동작이라
# 재빌드의 부수효과가 되면 안 된다. 필요할 때만 명시적으로 켠다.
#
#   ./scripts/docker-rebuild-app.sh --tunnel   또는   ARCHIMEDES_TUNNEL=1 ...
#
# 터널 로직은 scripts/tunnel-api.sh 하나로 통일한다
# (고정 URL·Quick Tunnel·PID 관리·정지 포함).
WANT_TUNNEL="${ARCHIMEDES_TUNNEL:-0}"
for arg in "$@"; do
  [[ "$arg" == "--tunnel" ]] && WANT_TUNNEL=1
done

if [[ "$WANT_TUNNEL" == "1" ]]; then
  echo "==> starting API tunnel (scripts/tunnel-api.sh) …"
  ./scripts/tunnel-api.sh
else
  echo "==> 터널 생략. 필요하면 --tunnel (정지: ./scripts/tunnel-api-stop.sh)"
fi
