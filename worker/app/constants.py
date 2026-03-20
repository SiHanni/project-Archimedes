"""Single source for ID-1, density, hollow defaults (see project-concept §4, §6)."""

from __future__ import annotations

# ISO/IEC 7810 ID-1 (mm)
ID1_WIDTH_MM: float = 85.60
ID1_HEIGHT_MM: float = 53.98

# Metal + purity -> rho (g/cm³) — document-level approximations
RHO_G_CM3: dict[tuple[str, str], float] = {
    ("gold", "24k"): 19.32,
    ("gold", "18k"): 15.58,
    ("gold", "14k"): 13.6,
    ("silver", "sterling"): 10.49,
    ("silver", "fine"): 10.49,
}

# product_k -> (alpha_k, beta_k mm³)
HOLLOW_ALPHA_BETA: dict[str, tuple[float, float]] = {
    # ring/other 는 layout 보정과 겹치므로 α를 약간 보수적으로
    "ring": (0.58, 0.0),
    # 체인류: Visual Hull·겹침으로 과대되기 쉬움 — α 보수적 (실측·골든으로 재튜닝)
    "necklace": (0.48, 0.0),
    "chain": (0.38, 0.0),
    "bracelet": (0.50, 0.0),
    # 얇은 형태는 Visual Hull 과대 → α 보수적 + layout_volume_mult(카드 대비 크기) 추가
    "pendant": (0.38, 0.0),
    # 귀걸이: 실물 흔히 3–5g 전후인데 Hull이 수십~수백 g로 튀기 쉬움 → α·layout 둘 다 별도(실측·골든 필수)
    "earring": (0.042, 0.0),
    "other": (0.42, 0.0),
}

# 소비자 귀금속 기준 “이 이상이면 수치 표시 의미 없음” 상한 (g) — runner sanity 게이트
SANITY_MAX_MASS_G_BY_PRODUCT: dict[str, float] = {
    "ring": 90.0,
    "necklace": 800.0,
    "chain": 800.0,
    "bracelet": 600.0,
    "pendant": 45.0,
    # 일반 데일리 귀걸이 상한을 넘기면 표시 숨김(무거운 드롭 등은 추후 서브타입으로 완화)
    "earring": 16.0,
    "other": 100.0,
}

# 카드 대비 실루엣 크기 → V_adj 추가 스케일(jewel_layout.py). 실측·골든으로 재튜닝.
LAYOUT_CORRECT_PRODUCTS: frozenset[str] = frozenset({"earring", "pendant", "ring", "other"})
# 목걸이/체인/팔찌: r_max가 이 값 이하일 때 layout 적용(펼친 긴 체인은 제외)
JEWEL_LAYOUT_CHAIN_BRACELET_MAX_RATIO: float = 0.34
# 한 축이 카드 대비 이 비율 미만이면 “가는 체인/귀걸이 오분류”로 보고 추가 부피 축소
JEWEL_LAYOUT_CHAIN_THIN_SIDE_MAX: float = 0.145
# thin 체인류: layout_mult ≈ clamp(K * r_min_side, MIN, MAX)
JEWEL_LAYOUT_CHAIN_THIN_MULT_K: float = 6.8
JEWEL_LAYOUT_CHAIN_THIN_MULT_MIN: float = 0.13
JEWEL_LAYOUT_CHAIN_THIN_MULT_MAX: float = 0.55
JEWEL_TO_CARD_WIDTH_RATIO_T1: float = 0.09
JEWEL_TO_CARD_WIDTH_RATIO_T2: float = 0.14
JEWEL_TO_CARD_WIDTH_RATIO_T3: float = 0.22
JEWEL_TO_CARD_WIDTH_RATIO_T4: float = 0.40
LAYOUT_VOL_MULT_TINY: float = 0.018
LAYOUT_VOL_MULT_SMALL: float = 0.055
LAYOUT_VOL_MULT_MED: float = 0.12
LAYOUT_VOL_MULT_LARGE: float = 0.28
LAYOUT_VOL_MULT_XL: float = 0.52
# 귀걸이 전용: 동일 ratio 버킷이라도 일반 테이블보다 훨씬 보수적으로(§7 가림·얇은 부피)
EARRING_LAYOUT_VOL_MULT_TINY: float = 0.0035
EARRING_LAYOUT_VOL_MULT_SMALL: float = 0.010
EARRING_LAYOUT_VOL_MULT_MED: float = 0.024
EARRING_LAYOUT_VOL_MULT_LARGE: float = 0.055
EARRING_LAYOUT_VOL_MULT_XL: float = 0.10

# Prior mass ranges (g) for tier demotion — soft only (§6.4)
PRIOR_MASS_G: dict[str, tuple[float, float]] = {
    "ring": (1.5, 12.0),
    "necklace": (3.0, 45.0),
    "chain": (3.0, 45.0),
    "bracelet": (4.0, 35.0),
    "pendant": (2.0, 25.0),
    # 대부분 3–5g대; 드롭·후크 무거운 편은 8–12g까지도 있음
    "earring": (0.8, 12.0),
    "other": (1.0, 80.0),
}

VOXEL_GRID_N: int = 128
VOXEL_GRID_N_COARSE: int = 64

# API `api/src/common/views.ts` VIEW_KEYS 와 동일 순서·이름 유지
VIEW_ORDER: tuple[str, ...] = ("front", "top", "left", "right", "back")

# Silhouette area gate: fraction of image area
JEWEL_AREA_FRAC_MIN: float = 0.0005
JEWEL_AREA_FRAC_MAX: float = 0.55

CARD_DILATE_PX: int = 5
