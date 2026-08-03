# Archimedes

귀금속 **사진** → 품질 게이트 → 검출·분할 → **실공간 스케일 복원** → Hollow → 밀도로 **참고 무게·예상 견적** 추정.

**v2 스펙(현행 코어)**: `.cursor/rules/archimedes-v2-single-photo.mdc` — 단일 사진 + 세그멘테이션 + 단안 깊이추정, 신용카드는 **메트릭 앵커**로 재활용. 연구개발계획서 ② AI 매칭 솔루션 반영.  
제품·물성·법무·운영 정본: `.cursor/rules/project-concept.mdc` · 구현 계획: `.cursor/rules/archimedes-implementation-plan.mdc`.  
**실행·Docker·트러블슈팅(상세)**: `.cursor/rules/archimedes-runbook.mdc`  
**기술 확장(복셀 카빙·Precision·세그·multer2)**: `.cursor/rules/archimedes-technical-extensions.mdc`  
**CV/기하 용어 정리**: `.cursor/rules/archimedes-concepts-glossary.mdc`  
**정확도·데이터 플레이북**(골든/실측 없을 때): `.cursor/rules/archimedes-accuracy-data-playbook.mdc`

## 현재 레포 상태

- **worker**: FastAPI `/health`, Redis 큐 소비 + 파이프라인 2경로
  - `single` (기본): 검출 → 분할 → 깊이추정 → **카드 앵커 스케일 융합** → 2.5D 부피
  - `multiview`: 5뷰 카드 σ → 실루엣 → 복셀 카빙 (고신뢰 옵션)
- **api**: NestJS — `POST /v1/jobs` (multipart `image` 1장 **또는** 5뷰), `GET /v1/jobs/:id`,
  실측 피드백 `POST /v1/jobs/:id/feedback`, 법무 초안 `/v1/legal/*`.
- **web**: React — §9.1 분기, 촬영 체크리스트, 업로드 모드 선택, 측정값·경고 노출,
  저신뢰 시 견적 숨김(§14.1).
- **infra**: Docker Compose — MySQL, Redis, MinIO, worker, worker-consumer, api, web.

### 신용카드는 왜 필요한가

사진만으로는 물체가 몇 mm 인지 알 수 없다. **ISO/IEC 7810 ID-1**(85.60 × 53.98 mm)
규격이 고정된 신용카드를 함께 찍으면
(1) 깊이 모델 출력을 **mm 로 환산**하고, (2) 물체가 놓인 **바닥면**을 얻고,
(3) 실측 치수를 알기에 **매 job 마다 거리 오차(RMSE)를 잴 수 있다**.
카드가 없으면 깊이 모델 단독으로 진행하되 신뢰도를 낮춘다.

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

### DB 스키마가 뒤처졌을 때

`infra/mysql/init/01-schema.sql` 은 **볼륨이 비어 있을 때만** 실행됩니다.
예전에 만든 볼륨을 계속 쓰고 있다면 이후 스키마 변경이 반영되지 않으므로,
멱등 마이그레이션을 한 번 적용해 주세요.

```bash
./scripts/db-migrate.sh
```

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

**중요**: `multiview` 모드는 뷰별 실루엣을 **서로 다른 시선**으로 해석합니다. 각도별로 **서로 다른 파일**을 올려 주세요(모순되면 `ERR_VOLUME`). 단일사진 모드에는 해당하지 않습니다.

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

## 시세·견적 (계획서 Step 2-1)

`GET /v1/jobs/:id` 응답에 `quote` 가 함께 온다. 시세는 계속 변하므로 job 결과에
굳히지 않고 **조회 시점**에 계산한다.

```bash
PRICE_BACKEND=static   # 고정 표(오프라인·데모, 항상 stale 표시)
PRICE_TABLE_KRW_PER_GRAM='{"gold:24k":118000,"gold:18k":86000}'
# 또는
PRICE_BACKEND=http
PRICE_API_URL=https://.../spot   # ?metal=&purity= 로 호출, 응답에서 g당 원화를 읽음
PRICE_BUY_RATE=0.95              # 소매 시세 대비 매입률(선택)
```

노출 정책(§14.1): `confidence_tier=low` 이거나 무게 표시가 억제된 job 은
**금액을 아예 내지 않는다**. 시세를 못 가져와도 0원이 아니라 사유와 함께 생략한다.
시세만 따로 보려면 `GET /v1/pricing/spot?metal=gold&purity=18k`.

**평가 지표 (계획서 평가표 3·4)**: `python worker/scripts/evaluate_rmse.py` — 제품 형태별 중량 RMSE·MAPE·bias 와 앵커 홀드아웃 거리 RMSE.

## 백엔드 교체 (ONNX)

검출·분할·깊이는 `ARCHIMEDES_*_BACKEND` 로 갈아끼운다. 기본은 모델 없이 도는 `stub`/`heuristic`.

```bash
pip install -e ".[onnx]"
# 가중치는 이미지에 굽지 않고 볼륨으로 주입
export ARCHIMEDES_ONNX_MODEL_DIR=/models
export ARCHIMEDES_DETECTOR_BACKEND=onnx     # detector.onnx
export ARCHIMEDES_SEGMENTATION_BACKEND=onnx # segmenter.onnx
export ARCHIMEDES_DEPTH_BACKEND=onnx        # depth.onnx
export ARCHIMEDES_DEPTH_OUTPUT_KIND=metric  # metric | affine_invariant | relative
```

출력 형상이 계약과 다르면 `ERR_MODEL_UNAVAILABLE` 로 **즉시 실패**한다(조용히 틀린 값을 내지 않음).

가중치를 꽂은 뒤 계약이 맞는지 한 번에 확인:

```bash
python worker/scripts/check_models.py   # 실패 시 exit 1
```

## 학습형 잔차 보정 (§4.4)

물리식을 버리지 않고 `m_final = m_physics · exp(w·x)` 의 잔차만 학습한다.
로그 공간이라 어떤 계수에서도 무게가 음수가 되지 않고, w=0 이면 원래 값이다.

```bash
python worker/scripts/fit_residual_model.py --min-n 30 --out residual_v1.json
```

표본이 부족하면 계수를 뽑지 않고, 홀드아웃 RMSE 가 개선되지 않으면 경고한다.
**자동 반영하지 않는다** — 배포는 수동 승인(§14.4 버전 태그·롤백).

## 알려진 한계 (다음 이터)

- 깊이·검출·분할 **실제 모델 미탑재** — 스텁은 두께를 관측하지 못해 제품 기준값으로 클램프한다(결과에 명시됨).
- α·두께 계수는 **물리적 근거에서 출발한 초기값**. 실측이 쌓이면 §4.4 학습형 잔차로 이행
  (스캐폴드는 완성 — `scripts/fit_residual_model.py`, 데이터만 있으면 바로 돈다).
- 골든 실사진 없음 — `golden/README.md` 절차대로 내부 구성 필요.
- 시세 기본값은 **고정 표**(항상 `stale`). 실시간 소스는 `PRICE_BACKEND=http` 로 연결.
- OTEL은 endpoint 설정 시에만 활성.

## 라이선스

MIT — `LICENSE`.
