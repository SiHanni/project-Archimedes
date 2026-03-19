# 골든 세트 (Phase 7)

## 목적

고정 **5뷰 이미지 + 메타**에 대해 `mass_est_g`, `confidence_tier` 등이 **허용 구간** 안에 있는지 자동 검증한다.

## 레포 구조

- **절차·예시 매니페스트**: 루트 [`golden/`](../golden/README.md)
- **실사진**: PII·저작권 때문에 보통 `golden/cases/` 는 **git에 넣지 않음** (`.gitignore` 참고). `manifest.json` 은 로컬/CI 시크릿에서만 둔다.

## CI/로컬 기본

- `cd worker && pytest` — 단위 테스트만 (`golden_manifest` 제외).
- 골든 실행:

  ```bash
  export ARCHIMEDES_GOLDEN_ROOT=/절대경로/project-Archimedes/golden
  cd worker && pytest -m golden_manifest -v
  ```

## 실측과의 관계

골든은 **회귀(망가짐 방지)** 용이고, **실측 튜닝**은 `mass_feedback` + `calibration_suggest.py` 축이다. 둘 다 있으면 가장 안전하다.

## 참고

- 개념 사전: `.cursor/rules/archimedes-concepts-glossary.mdc`
- 실측 API: `POST /v1/jobs/:id/feedback` — `jobs/README.md`
