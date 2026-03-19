# 배치·캘리브레이션 작업 (Phase 7)

## 실측 튜닝 플로우 (지금 할 수 있는 것)

1. **웹/API**로 job 생성 → 처리 완료(`completed`) 후 **실측 무게(g)** 를 등록한다.

   ```http
   POST /v1/jobs/{jobId}/feedback
   Content-Type: application/json

   { "actualMassG": 5.42, "notes": "디지털 저울, 동일 프로토콜" }
   ```

2. DB에 `mass_feedback` 행이 쌓이면, **제안 배수**를 출력한다 (자동 반영 아님 — §14.4 수동 승인).

   **호스트(워커 venv)**:

   ```bash
   export MYSQL_HOST=127.0.0.1 MYSQL_USER=archimedes MYSQL_PASSWORD=archimedes MYSQL_DATABASE=archimedes
   python worker/scripts/calibration_suggest.py
   ```

   **Docker** (worker 이미지에 `scripts/` 포함):

   ```bash
   docker compose exec worker python scripts/calibration_suggest.py
   ```

3. 출력된 `suggested_alpha` 를 검토한 뒤 `worker/app/constants.py` 의 `HOLLOW_ALPHA_BETA` 를 수정하고 버전 태그를 남긴다.

## 골든 세트

회귀용 이미지·매니페스트는 저장소 루트 **`golden/`** 를 본다. (`golden/README.md`)

## 상태

- 이 디렉터리(`jobs/`) 자체는 **실행 코드 없이** 운영 절차만 두었고, 캘리브 스크립트는 **`worker/scripts/calibration_suggest.py`** 에 둔다 (pymysql·`app.constants` 재사용).
