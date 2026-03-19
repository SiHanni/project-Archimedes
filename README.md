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

- 웹 UI: http://localhost:5173  
- API: http://localhost:3000/v1  
- Worker 헬스: http://localhost:8000/health  
- MinIO 콘솔: http://localhost:9001 (`minioadmin` / `minioadmin`)

업로드 후 **worker-consumer** 로그에서 job 처리 여부를 확인합니다.

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
