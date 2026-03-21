# Archimedes

귀금속 **5방향 이미지** → 품질 게이트 → 카드 스케일 → 세그 → 기하(v0) → Hollow → 밀도로 **참고 무게** 추정.  
스펙: `.cursor/rules/project-concept.mdc` · 구현 계획: `.cursor/rules/archimedes-implementation-plan.mdc`.  
**실행·Docker·트러블슈팅(상세)**: `.cursor/rules/archimedes-runbook.mdc`  
**기술 확장(복셀 카빙·Precision·세그·multer2)**: `.cursor/rules/archimedes-technical-extensions.mdc`  
**CV/기하 용어 정리**: `.cursor/rules/archimedes-concepts-glossary.mdc`  
**정확도·데이터 플레이북**(골든/실측 없을 때): `.cursor/rules/archimedes-accuracy-data-playbook.mdc`

## 현재 레포 상태 (스캐폴드)

- **worker**: FastAPI `/health`, Redis 큐 소비 + OpenCV 파이프라인(v0: 슬랩 AABB 부피, 카드·세그 휴리스틱).
- **api**: NestJS — `POST /v1/jobs` (multipart 5뷰), `GET /v1/jobs/:id`, 법무 초안 `/v1/legal/*`.
- **web**: React — §9.1 분기, 체크리스트, 각도별 파일 업로드, 결과·저신뢰 시 견적 숨김(§14.1).
- **infra**: Docker Compose — MySQL, Redis, MinIO, worker, worker-consumer, api, web.

로컬 개발 시 `.env`는 **선택**입니다(Compose에 기본값 포함). 호스트에서 worker만 돌릴 때는 `cp .env.example .env` 후 MinIO/MySQL 주소를 맞춥니다.

## 한 번에 기동

```bash
docker compose up --build
```

- 웹 UI: http://localhost:25173  
- API: http://localhost:23000/v1  
- Worker 헬스: http://localhost:28000/health  
- MinIO 콘솔: http://localhost:29001 (`minioadmin` / `minioadmin`)  
- (호스트에서 DB/Redis 직접 접속 시) MySQL `localhost:23306`, Redis `localhost:26379`, MinIO API `localhost:29000`

업로드 후 **worker-consumer** 로그에서 job 처리 여부를 확인합니다.

### 다음날 / PC 재부팅 후 한 번에 기동 (도커 + Vercel용 터널)

어제 `docker compose down` 했어도 **이 스크립트 한 번**이면 전체 스택을 백그라운드로 띄우고, API가 뜰 때까지 기다린 뒤 **cloudflared** 터널까지 켭니다. (출력되는 `VITE_API_BASE`를 Vercel에 넣고 재배포 — URL이 바뀌면 매번 갱신)

```bash
./scripts/start-dev-day.sh
```

- `--no-dev` : `docker-compose.yml` 만 (이미지 빌드 그대로, 마운트 없음)
- `--no-tunnel` : 도커만 (터널 없이 로컬만)
- `--watch` : Compose 2.23+ 에서 web 소스 감시 재빌드 (dev 오버레이와 함께)

`./scripts/dev-up.sh --tunnel` 은 내부적으로 위와 동일하게 `start-dev-day.sh --dev` 를 실행합니다.

### 로컬 개발: 서버 코드 저장 시 자동 반영 (재빌드 최소)

`api/` · `worker/app/` 를 수정할 때마다 이미지를 다시 빌드하지 않도록, 개발용 오버레이를 둘 수 있습니다.

```bash
./scripts/dev-up.sh
```

- **API**: `nest start --watch` + `./api/src` 마운트  
- **worker**: `uvicorn --reload` + `./worker/app` 마운트  
- **worker-consumer**: `watchfiles` 로 `app` 변경 시 프로세스 재실행  

프론트(`web/`)까지 저장할 때마다 **이미지 재빌드**까지 쓰려면(Docker Compose **2.23+**):

```bash
./scripts/dev-up.sh --watch
```

`package.json` / `package-lock.json` / `pyproject.toml` 등 **의존성을 바꾼 뒤**에는 해당 서비스만 다시 빌드하세요.

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml build api worker worker-consumer
```

### Vercel에서 로컬 API 붙일 때 (퀵 터널 — 바로 실행)

1. **cloudflared** 없으면: `brew install cloudflare/cloudflare/cloudflared`
2. **한 줄**: `./scripts/start-dev-day.sh`  
   (도커 올리고 API 뜰 때까지 기다린 뒤 터널 켬. 터널만 다시 켤 땐 `./scripts/tunnel-api.sh`)
3. 터미널에 나온 **`VITE_API_BASE=…/v1`** 를 Vercel Production에 붙여 넣고 재배포.  
   주소가 바뀌면 그때 다시 복사·재배포.

**퀵 터널이 아니라 고정 URL로 뜨면** `.env`에 `CLOUDFLARE_TUNNEL_TOKEN` / `ARCHIMEDES_PUBLIC_API_BASE` 를 넣었거나 `infra/cloudflared/config.yml` 이 있는 경우입니다. 퀵만 쓰려면 그 변수·파일을 빼면 됩니다.

종료: `./scripts/tunnel-api-stop.sh` · (옵션) **고정 URL·도메인**은 `docs/cloudflare-stable-tunnel.md`

### 코드 수정 후 통째로 재빌드 (프로덕션 이미지에 가깝게)

```bash
bash scripts/docker-rebuild-app.sh
```

API·DB·인프라까지 전부 다시 올리려면 `docker compose up --build` 를 쓰면 됩니다.

**중요**: v0 부피 코어는 뷰별 실루엣을 **서로 다른 시선**으로 해석합니다. **동일 사진을 5슬롯에 넣으면** 기하 충돌로 `ERR_VOLUME`이 날 수 있습니다. 데모·테스트 시에는 각도별로 **서로 다른 파일**을 올려 주세요.

## Worker 단독 테스트

```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check app tests
```

느린 OpenCV 스모크: `pytest -m slow`

**골든 세트**(로컬에 `golden/manifest.json` + 이미지 준비 후): `ARCHIMEDES_GOLDEN_ROOT=... pytest -m golden_manifest` — `golden/README.md` 참고.

**실측 피드백 → Hollow α 제안**: `POST /v1/jobs/{id}/feedback` 후 `python worker/scripts/calibration_suggest.py` — `jobs/README.md`.

## 알려진 한계 (다음 이터)

- 부피 코어는 **직교 슬랩 AABB v0**; 전체 **복셀 카빙·Precision 호모그래피**는 스펙 대비 미완.
- 세그는 **색/Otsu** 수준 — 주얼리 특화 모델(§15) 교체 예정.
- Phase 7: `docs/golden.md`, `jobs/` 배치 스텁; OTEL은 endpoint 설정 시에만 활성.

## 라이선스

MIT — `LICENSE`.
