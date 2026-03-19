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
    "ring": (0.72, 0.0),
    "necklace": (0.65, 0.0),
    "chain": (0.58, 0.0),
    "bracelet": (0.68, 0.0),
    "pendant": (0.70, 0.0),
    "other": (0.65, 0.0),
}

# Prior mass ranges (g) for tier demotion — soft only (§6.4)
PRIOR_MASS_G: dict[str, tuple[float, float]] = {
    "ring": (1.5, 12.0),
    "necklace": (3.0, 45.0),
    "chain": (3.0, 45.0),
    "bracelet": (4.0, 35.0),
    "pendant": (2.0, 25.0),
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
