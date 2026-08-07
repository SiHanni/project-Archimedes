#!/usr/bin/env python3
"""
실측 피드백(`mass_feedback` + 완료된 `jobs`)으로 product_k별
**추정 무게 대비 실측 비율** 중앙값을 내고, Hollow α_k에 곱할 **제안 배수**를 출력한다.

  §6.1: V_adj = α_k V_hull,  m_est = ρ V_adj  →  동일 V_hull에서 m ∝ α_k
  따라서 **α_k' ≈ α_k × median(m_actual / m_est)** 를 1차 근사로 쓸 수 있다(수동 승인·§14.4).

사용:
  export MYSQL_HOST=127.0.0.1 MYSQL_USER=archimedes MYSQL_PASSWORD=archimedes MYSQL_DATABASE=archimedes
  python worker/scripts/calibration_suggest.py

Docker:
  docker compose exec worker python /app/scripts/calibration_suggest.py
  (이미지에 스크립트 복사 필요 — 아래 README 참고)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

# worker 패키지에서 현재 α 테이블 읽기
_WORKER = Path(__file__).resolve().parents[1]
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

from app.constants import HOLLOW_ALPHA_BETA, HOLLOW_ALPHA_BETA_DEPTH  # noqa: E402


def main() -> int:
    try:
        import pymysql
    except ImportError:
        print("Need: pip install pymysql", file=sys.stderr)
        return 1

    conn = pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "archimedes"),
        password=os.environ.get("MYSQL_PASSWORD", "archimedes"),
        database=os.environ.get("MYSQL_DATABASE", "archimedes"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    sql = """
    SELECT j.input_json, j.result_json, f.actual_mass_g
    FROM mass_feedback f
    INNER JOIN jobs j ON j.id = f.job_id
    WHERE j.status = 'completed'
      AND j.result_json IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No feedback rows joined to completed jobs. POST /v1/jobs/:id/feedback first.")
        return 0

    by_k: dict[tuple[str, bool], list[float]] = {}
    for r in rows:
        inp = r["input_json"]
        out = r["result_json"]
        if isinstance(inp, str):
            inp = json.loads(inp)
        if isinstance(out, str):
            out = json.loads(out)
        if not inp or not out:
            continue
        m_est = out.get("mass_est_g")
        m_act = float(r["actual_mass_g"])
        if m_est is None or m_est <= 0 or m_act <= 0:
            continue
        k = str(inp.get("product_k", "other")).lower()
        # ⚠️ α 표가 둘이다. v1(다뷰 복셀)은 `HOLLOW_ALPHA_BETA`, v2(단일사진 깊이)는
        #    `HOLLOW_ALPHA_BETA_DEPTH` 를 쓴다. 어느 쪽으로 계산된 결과인지 보지 않고
        #    v1 표만 읽으면 **엉뚱한 기준으로 제안**하게 된다(실측: ring 0.58 로 읽어
        #    0.264 를 제안했는데 실제 v2 기준은 0.80 → 0.364 였다).
        method = ((out.get("meta") or {}).get("reconstruction") or {}).get("method", "")
        depth_path = method in ("height_field", "measured_area_given_thickness")
        by_k.setdefault((k, depth_path), []).append(m_act / float(m_est))

    print("--- Suggested α_k multipliers (median(m_actual/m_est)), N per category ---")
    print("(Apply manually to constants / DB; do not auto-write without review §14.4)\n")
    for (k, depth_path), ratios in sorted(by_k.items()):
        table = HOLLOW_ALPHA_BETA_DEPTH if depth_path else HOLLOW_ALPHA_BETA
        name = "DEPTH(v2)" if depth_path else "VOXEL(v1)"
        med = statistics.median(ratios)
        old_a = table.get(k, table["other"])[0]
        suggested = old_a * med
        print(
            f"  product_k={k!r} [{name}]  N={len(ratios)}  median_ratio={med:.4f}  "
            f"current_alpha={old_a:.4f}  suggested_alpha≈{suggested:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
