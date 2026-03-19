# Archimedes API (NestJS)

- `POST /v1/jobs` — multipart: `front,top,left,right,back` files + `metal`, `purity`, `product_k` (+ optional `reference_weight_g`, `knows_weight`).
- `GET /v1/jobs/:id` — job status + result/error.
- `GET /v1/legal/terms`, `GET /v1/legal/privacy` — 초안 (법무 검토 전).
- `POST /v1/jobs/:id/feedback` — 실측 무게(g) 등록 (`completed` job만). 캘리브: `worker/scripts/calibration_suggest.py`.

로컬: `npm install && npm run start:dev` (MySQL·Redis·MinIO 필요).
