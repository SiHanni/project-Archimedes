"""Single source for ID-1, density, hollow defaults (see project-concept §4, §6)."""

from __future__ import annotations

from dataclasses import dataclass

# ISO/IEC 7810 ID-1 (mm)
ID1_WIDTH_MM: float = 85.60
ID1_HEIGHT_MM: float = 53.98


@dataclass(frozen=True)
class Material:
    """금속·함량 1행. ρ 는 **문서용 근사**이며 합금 조성에 따라 달라진다."""

    metal: str
    purity: str
    rho_g_cm3: float
    label_ko: str
    note: str = ""


# 사용자가 고르는 (금속, 함량) → ρ (g/cm³).
# ⚠️ project-concept §15.3: 운영 전 합금 규격·출처(버전)를 명문화해야 한다.
#    특히 백금 합금(Ru/Ir/Co)은 조성별 편차가 커서 아래 값은 대표값일 뿐이다.
MATERIALS: dict[tuple[str, str], Material] = {
    ("gold", "24k"): Material("gold", "24k", 19.32, "순금 24K"),
    ("gold", "22k"): Material("gold", "22k", 17.80, "22K", "합금 대표값"),
    ("gold", "18k"): Material("gold", "18k", 15.58, "18K", "합금 대표값"),
    ("gold", "14k"): Material("gold", "14k", 13.60, "14K", "합금 대표값"),
    ("gold", "10k"): Material("gold", "10k", 11.60, "10K", "합금 대표값"),
    ("silver", "fine"): Material("silver", "fine", 10.49, "순은 999"),
    ("silver", "sterling"): Material("silver", "sterling", 10.36, "실버 925", "Ag92.5/Cu7.5"),
    ("platinum", "pt999"): Material("platinum", "pt999", 21.45, "순백금 Pt999"),
    ("platinum", "pt950"): Material("platinum", "pt950", 20.10, "백금 Pt950", "Pt95/Ru5 기준"),
    ("platinum", "pt900"): Material("platinum", "pt900", 20.00, "백금 Pt900", "합금 대표값"),
}

# 입력 정규화 — 프런트·외부 호출이 보내는 표기 흔들림 흡수
METAL_ALIASES: dict[str, str] = {
    "gold": "gold", "au": "gold", "금": "gold",
    "silver": "silver", "ag": "silver", "은": "silver",
    "platinum": "platinum", "pt": "platinum", "백금": "platinum",
}
# 함량 표기는 금속마다 의미가 달라(예: "999" = 금 24K vs 은 fine) 금속별로 분리한다.
PURITY_ALIASES: dict[str, dict[str, str]] = {
    "gold": {
        "24k": "24k", "999": "24k", "1000": "24k", "pure": "24k", "순금": "24k",
        "22k": "22k", "916": "22k",
        "18k": "18k", "750": "18k",
        "14k": "14k", "585": "14k",
        "10k": "10k", "417": "10k",
    },
    "silver": {
        "sterling": "sterling", "925": "sterling", "s925": "sterling",
        "fine": "fine", "999": "fine", "1000": "fine", "pure": "fine", "순은": "fine",
    },
    "platinum": {
        "pt999": "pt999", "999": "pt999", "pt1000": "pt999", "pure": "pt999",
        "pt950": "pt950", "950": "pt950",
        "pt900": "pt900", "900": "pt900",
    },
}

# 하위 호환 (기존 호출부·테스트)
RHO_G_CM3: dict[tuple[str, str], float] = {k: m.rho_g_cm3 for k, m in MATERIALS.items()}

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

# ── v2 depth 경로 (archimedes-v2-single-photo.mdc §4) ──
#
# V_2.5D = A_proj × h_eff 는 물체를 "단면이 일정한 기둥"으로 본다.
# 실제 장신구는 단면이 둥글어(대략 타원) 부피가 π/4 ≈ 0.785 배쯤 작고,
# 체인은 링크 사이 빈 공간이 더 많다. 아래 α 는 그 **물리적 근거**에서 출발한
# 값이며, v1 의 α(예: earring 0.042)와 달리 버그 보정치가 아니다.
# 실측이 쌓이면 §4.4 학습형 잔차로 대체한다.
HOLLOW_ALPHA_BETA_DEPTH: dict[str, tuple[float, float]] = {
    "ring": (0.80, 0.0),
    "necklace": (0.65, 0.0),
    "chain": (0.60, 0.0),
    "bracelet": (0.70, 0.0),
    "pendant": (0.75, 0.0),
    "earring": (0.70, 0.0),
    "other": (0.75, 0.0),
}

# 가시 두께 h_vis 의 물리적 하한·상한 (mm).
# 하한: 접사 깊이 노이즈로 h_vis 가 0 에 붙는 것을 막는다.
# 상한: 깊이 이상치가 부피를 폭주시키는 것을 막는다.
# 둘 다 **물리적 클램프**이며 적용 시 meta 에 플래그로 남긴다(조용히 깎지 않는다).
MIN_THICKNESS_MM_BY_PRODUCT: dict[str, float] = {
    "ring": 1.0,
    "necklace": 0.8,
    "chain": 0.8,
    "bracelet": 1.0,
    "pendant": 0.8,
    "earring": 0.6,
    "other": 0.8,
}
MAX_THICKNESS_MM_BY_PRODUCT: dict[str, float] = {
    "ring": 9.0,
    "necklace": 12.0,
    "chain": 12.0,
    "bracelet": 14.0,
    "pendant": 12.0,
    "earring": 10.0,
    "other": 25.0,
}

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
