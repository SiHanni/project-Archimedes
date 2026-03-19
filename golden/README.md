# 골든 세트 (회귀 테스트용 이미지)

## 목적

같은 입력 사진·같은 메타에 대해 **부피/무게/tier**가 허용 범위 안에 있는지 CI 또는 로컬에서 자동 검증한다.

## 디렉터리 구조 (예시)

```
golden/
  README.md
  manifest.example.json   # 복사해 manifest.json 으로 사용
  cases/
    my_ring_01/
      front.jpg
      top.jpg
      left.jpg
      right.jpg
      back.jpg
```

- **저작권·개인정보**: 실사진은 기본적으로 **레포에 커밋하지 않는다**. `.gitignore`에 `golden/manifest.json`, `golden/cases/` 를 넣어도 된다.
- 내부 전용 레포/CI 시크릿에만 올리거나, S3에서 내려받도록 할 수 있다.

## manifest.json

`manifest.example.json` 을 참고한다. 각 `case` 는:

- `id`: 식별자
- `views`: 뷰 키 → **golden 루트 기준 상대 경로**
- `input`: `metal`, `purity`, `product_k`
- `expect`: `mass_est_g_min`, `mass_est_g_max`, (선택) `tier_in`

## 실행

```bash
export ARCHIMEDES_GOLDEN_ROOT="$(pwd)/golden"
cd worker
source .venv/bin/activate
pytest -m golden_manifest -v
```

기본 `pytest` 는 `golden_manifest` 를 **제외**한다 (`pyproject.toml` `addopts`).

## 참고

- 개념: `.cursor/rules/archimedes-concepts-glossary.mdc`
- 문서: `docs/golden.md`
