#!/usr/bin/env python3
"""
학습형 잔차 보정 계수 추정 (`archimedes-v2-single-photo.mdc` §4.4).

`m_final = m_physics · exp(w·x)` 의 w 를 로그 공간 최소제곱으로 푼다.
물리식을 베이스로 두므로 표본이 적어도 붕괴하지 않는다(w=0 이면 원래 값).

**자동 반영하지 않는다.** 계수를 JSON 으로 출력만 하고, 배포는 수동 승인
(스펙 §14.4: 버전 태그 + 롤백 절차).

사용:
  export MYSQL_HOST=127.0.0.1 MYSQL_USER=archimedes MYSQL_PASSWORD=archimedes MYSQL_DATABASE=archimedes
  python worker/scripts/fit_residual_model.py --min-n 30 --out residual_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[1]
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

import numpy as np  # noqa: E402

from app.eval import error_stats  # noqa: E402
from app.eval.features import (  # noqa: E402
    FEATURE_NAMES,
    apply_correction,
    extract_features,
    target_log_ratio,
)


def _as_obj(v: object) -> dict | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    try:
        return json.loads(str(v))
    except (TypeError, ValueError):
        return None


def load_rows() -> list[tuple[list[float], float, float, float]]:
    """(features, target, mass_est, mass_actual)."""
    import pymysql

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
            SELECT j.result_json, f.actual_mass_g
            FROM mass_feedback f
            INNER JOIN jobs j ON j.id = f.job_id
            WHERE j.status IN ('completed', 'completed_low_confidence')
              AND j.result_json IS NOT NULL
            """
        )
        raw = cur.fetchall()
    conn.close()

    out = []
    for r in raw:
        res = _as_obj(r["result_json"])
        if not res:
            continue
        feats = extract_features(res)
        if feats is None:
            continue
        est = res.get("mass_est_g")
        act = float(r["actual_mass_g"])
        if not isinstance(est, (int, float)) or est <= 0:
            continue
        y = target_log_ratio(act, float(est))
        if y is None:
            continue
        out.append((feats, y, float(est), act))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30, help="스펙 §6.1 카테고리별 최소 표본")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1e-3, help="정칙화 — 표본이 적을 때 필수")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rows = load_rows()
    n = len(rows)
    print(f"단일사진 + 실측 표본: N={n}")
    if n < args.min_n:
        print(
            f"표본이 부족합니다 (N={n} < {args.min_n}). 이 상태로 계수를 뽑으면 과적합입니다.\n"
            f"POST /v1/jobs/:id/feedback 으로 실측을 더 모아 주세요."
        )
        return 0

    X = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    est = np.array([r[2] for r in rows], dtype=np.float64)
    act = np.array([r[3] for r in rows], dtype=np.float64)

    # 홀드아웃은 인덱스 간격으로 나눈다(시간순 편향 완화, 난수 의존 없음)
    stride = max(2, int(round(1.0 / max(args.holdout, 1e-6))))
    hold = np.arange(n) % stride == 0
    fit = ~hold
    if fit.sum() < len(FEATURE_NAMES) + 2 or hold.sum() < 3:
        fit = np.ones(n, dtype=bool)
        hold = np.zeros(n, dtype=bool)

    # ridge: (XᵀX + λI)w = Xᵀy — bias 항은 정칙화하지 않는다
    Xf, yf = X[fit], y[fit]
    lam = np.eye(X.shape[1]) * args.ridge
    lam[0, 0] = 0.0
    w = np.linalg.solve(Xf.T @ Xf + lam, Xf.T @ yf)

    print("\n--- 계수 (자동 반영 아님, §14.4 수동 승인) ---")
    for name, coef in zip(FEATURE_NAMES, w):
        print(f"  {name:<18} {coef:+.5f}")

    base = error_stats(est.tolist(), act.tolist())
    corrected = np.array(
        [apply_correction(e, f, w.tolist()) for e, f in zip(est, X)], dtype=np.float64
    )
    after = error_stats(corrected.tolist(), act.tolist())
    print(f"\n전체  보정 전 {base.as_dict()}")
    print(f"전체  보정 후 {after.as_dict()}")

    if hold.any():
        h_base = error_stats(est[hold].tolist(), act[hold].tolist())
        h_after = error_stats(corrected[hold].tolist(), act[hold].tolist())
        print(f"홀드아웃 보정 전 {h_base.as_dict()}")
        print(f"홀드아웃 보정 후 {h_after.as_dict()}")
        # 홀드아웃에서 나빠지면 반영하면 안 된다 — 이 판정이 없으면 과적합을 못 잡는다
        if h_after.rmse >= h_base.rmse:
            print("\n⚠️ 홀드아웃 RMSE 가 개선되지 않았습니다. 반영하지 마세요.")

    if args.out:
        payload = {
            "version": 1,
            "feature_names": list(FEATURE_NAMES),
            "weights": w.tolist(),
            "n_samples": int(n),
            "ridge": args.ridge,
            "rmse_before": base.rmse,
            "rmse_after": after.rmse,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
