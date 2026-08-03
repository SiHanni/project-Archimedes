#!/usr/bin/env python3
"""
실측 피드백으로 **중량 RMSE** 를 집계한다 (연구개발계획서 평가표 4번).

거리 정확도(평가표 3번)는 job 마다 `result_json.meta.scale_fusion.depth_rmse_mm`
에 이미 적재돼 있으므로 여기서 함께 집계한다 — 카드 앵커가 곧 계측기라
별도 groundtruth 수집이 필요 없다.

사용:
  export MYSQL_HOST=127.0.0.1 MYSQL_USER=archimedes MYSQL_PASSWORD=archimedes MYSQL_DATABASE=archimedes
  python worker/scripts/evaluate_rmse.py

Docker:
  docker compose exec worker python scripts/evaluate_rmse.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[1]
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

from app.eval import error_stats, group_error_stats  # noqa: E402


def _as_obj(v: object) -> dict | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    try:
        return json.loads(str(v))
    except (TypeError, ValueError):
        return None


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
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.input_json, j.result_json, f.actual_mass_g
            FROM mass_feedback f
            INNER JOIN jobs j ON j.id = f.job_id
            WHERE j.status = 'completed' AND j.result_json IS NOT NULL
            """
        )
        mass_rows = cur.fetchall()
        cur.execute(
            """
            SELECT result_json FROM jobs
            WHERE status = 'completed' AND result_json IS NOT NULL
            """
        )
        all_rows = cur.fetchall()
    conn.close()

    # ── 중량 RMSE (제품 형태별) ──
    grouped: list[tuple[str, float, float]] = []
    for r in mass_rows:
        inp = _as_obj(r["input_json"])
        out = _as_obj(r["result_json"])
        if not inp or not out:
            continue
        est = out.get("mass_est_g")
        act = float(r["actual_mass_g"])
        if est is None or est <= 0 or act <= 0:
            continue
        grouped.append((str(inp.get("product_k", "other")).lower(), float(est), act))

    print("=== 중량 정확도 (평가표 4) — RMSE g, 낮을수록 좋음 ===")
    if not grouped:
        print("  실측 피드백 없음. POST /v1/jobs/:id/feedback 으로 먼저 적재하세요.\n")
    else:
        for key, st in sorted(group_error_stats(grouped).items()):
            d = st.as_dict()
            print(
                f"  {key:<12} N={d['n']:<4} RMSE={d['rmse']:<8} MAE={d['mae']:<8} "
                f"MAPE={d['mape_pct']}%  bias={d['bias']}"
            )
        print()

    # ── 거리 정확도 (앵커 홀드아웃) ──
    depth_err: list[float] = []
    for r in all_rows:
        out = _as_obj(r["result_json"])
        if not out:
            continue
        rmse = ((out.get("meta") or {}).get("scale_fusion") or {}).get("depth_rmse_mm")
        if isinstance(rmse, (int, float)):
            depth_err.append(float(rmse))

    print("=== 거리 정확도 (평가표 3) — 카드 앵커 홀드아웃 RMSE mm ===")
    if not depth_err:
        print("  단일사진(앵커 사용) job 이 아직 없습니다.")
    else:
        # 이미 job 별 RMSE 이므로 그 자체의 분포를 본다(0 대비 오차로 재집계)
        st = error_stats(depth_err, [0.0] * len(depth_err))
        d = st.as_dict()
        print(f"  N={d['n']}  평균 {d['mae']} mm  제곱평균 {d['rmse']} mm  최대 {max(depth_err):.3f} mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
