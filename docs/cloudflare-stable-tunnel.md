# Cloudflare로 **고정** HTTPS API URL 쓰기 (Vercel 개발용)

`trycloudflare.com` **Quick Tunnel** 은 계정 없이 쓰기 때문에 **주소가 매번 바뀝니다.**  
Vercel `VITE_API_BASE` 를 매번 고치기 싫다면 아래 중 하나를 쓰세요.

## 빠른 절차 (레포에 있는 것만)

1. **도메인**을 Cloudflare DNS에 추가해 두세요 (네임서버를 Cloudflare로).
2. 아래 **방법 A** 대로 Zero Trust에서 터널 생성 → **토큰** 복사 → **Public hostname** (`https://api.도메인` → `http://localhost:23000`).
3. 프로젝트 루트에 `.env` 가 없으면: `cp env.tunnel.example .env` 후 토큰·URL 채우기.
4. 점검: `./scripts/go-stable-tunnel.sh`
5. 매일: `./scripts/start-dev-day.sh` (도커 + 터널).  
   **Vercel**에는 최초 한 번 `VITE_API_BASE=https://api.도메인.com/v1` 만 넣으면 됩니다.

---

## 방법 A: Cloudflare Zero Trust — 이름 있는 터널 (권장, 무료 티어로 시작 가능)

1. [Cloudflare 대시보드](https://dash.cloudflare.com/) → **Zero Trust** (또는 Teams) 켜기 — 무료 플랜으로도 터널 생성 가능.
2. **Networks → Tunnels → Create a tunnel**  
   - 이름 예: `archimedes-local`
3. **Install connector** 단계에서 나오는 **토큰**을 복사 (한 번만 보여 줌).
4. 같은 화면에서 **Public hostname** 추가:
   - **Subdomain**: 예) `api`  
   - **Domain**: 본인이 Cloudflare에 넣어 둔 도메인 (예) `example.com`  
   - **Service type**: `HTTP`  
   - **URL**: `localhost:23000` (Docker API 포트)
5. 저장 후 브라우저에서 `https://api.example.com/v1/legal/terms` 가 열리는지 확인 (맥에서 도커·터널 떠 있을 때).

프로젝트 루트 `.env` (git에 안 올림):

```bash
CLOUDFLARE_TUNNEL_TOKEN=여기에_대시보드_토큰
ARCHIMEDES_PUBLIC_API_BASE=https://api.example.com
```

그다음 `./scripts/tunnel-api.sh` 또는 `./scripts/start-dev-day.sh` — 스크립트가 **고정 URL**로 `VITE_API_BASE` 안내합니다.  
**Vercel에는 한 번만** `VITE_API_BASE=https://api.example.com/v1` 넣고, 터널 토큰만 유지하면 됩니다.

---

## 방법 B: API를 클라우드에 배포 (Railway, Fly.io, VPS 등)

로컬 터널 없이 **항상 같은 `https://api.xxx.com`** — 운영에 가장 깔끔합니다. (별도 배포 파이프라인 필요)

---

## 요약

| 방식 | URL 고정 | 비고 |
|------|-----------|------|
| Quick Tunnel (`--url`) | ❌ 매번 변경 | 지금 기본 동작 |
| Zero Trust 이름 있는 터널 | ✅ 본인 도메인 | `CLOUDFLARE_TUNNEL_TOKEN` + `ARCHIMEDES_PUBLIC_API_BASE` |
| 클라우드 배포 API | ✅ | 터널 불필요 |

도메인이 Cloudflare DNS에 없으면 **도메인을 Cloudflare에 추가**하거나, 가비아/Route53 등에서 네임서버를 Cloudflare로 맞춰야 Public hostname을 붙일 수 있습니다.
