"""
v2 단일사진 경로 E2E — 합성 장면으로 `run_pipeline` 전 구간을 돌린다.

깊이 모델 없이(stub) 돌리므로 **퇴화 기준선**을 검증한다:
카드 앵커로 거리는 잡히지만 두께는 관측되지 않아 제품 기준값으로 클램프되고,
그 사실이 신뢰도·경고·불확실성에 정직하게 반영되는지 본다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.models.schemas import JobInputRecord
from app.pipeline.runner import run_pipeline

IMG = 2048
# ID-1 종횡비(1.586)에 맞춘 축정렬 카드 — PnP 가 정면 평면을 얻어 결정론적으로 만든다.
# 배치는 촬영 규약과 동일하게 **카드 오른쪽 · 귀금속 왼쪽**(ARCHIMEDES_OBJECT_SIDE=left).
CARD_X0, CARD_Y0, CARD_W = 900, 700, 960
CARD_H = round(CARD_W * 53.98 / 85.60)
JEWEL_CX, JEWEL_CY, JEWEL_R = 520, 1000, 110


def _scene_jpeg() -> bytes:
    img = np.full((IMG, IMG, 3), 225, np.uint8)
    cv2.rectangle(
        img, (CARD_X0, CARD_Y0), (CARD_X0 + CARD_W, CARD_Y0 + CARD_H), (250, 250, 250), -1
    )
    cv2.rectangle(img, (CARD_X0, CARD_Y0), (CARD_X0 + CARD_W, CARD_Y0 + CARD_H), (60, 60, 60), 4)
    cv2.circle(img, (JEWEL_CX, JEWEL_CY), JEWEL_R, (15, 15, 20), -1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


@pytest.fixture
def single_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ARCHIMEDES_BLUR_THRESHOLD", "3")
    monkeypatch.setenv("ARCHIMEDES_DETECTOR_BACKEND", "stub")
    monkeypatch.setenv("ARCHIMEDES_SEGMENTATION_BACKEND", "heuristic")
    monkeypatch.setenv("ARCHIMEDES_DEPTH_BACKEND", "stub")
    # 산출물 인코딩(풀해상도 PNG)은 무겁다 — 여기서는 파이프라인 로직만 본다.
    # 산출물 자체는 test_visualize.py 가 따로 검증한다.
    monkeypatch.setenv("ARCHIMEDES_SAVE_SEG_ASSETS", "0")
    return Settings()


def _run(settings: Settings, **kwargs) -> dict:
    inp = JobInputRecord(image="uploads/j/main.jpg", **kwargs)
    return run_pipeline("t-single", inp, {"front": _scene_jpeg()}, settings)


def test_single_photo_runs_end_to_end(single_settings: Settings) -> None:
    out = _run(single_settings, metal="gold", purity="18k", product_k="ring")

    assert out["meta"]["capture_mode"] == "single"
    assert out["meta"]["volume_model"] == "height_field"
    assert out["meta"]["scale_fusion"]["anchor_used"] is True
    assert out["mass_est_g"] > 0


def test_scale_comes_from_the_card_not_a_guess(single_settings: Settings) -> None:
    """
    카드 폭 960px, ID-1 85.60mm, 폴백 f=1.15×2048 → 카드 거리 ≈ 210mm.
    반지름 110px 원 → 실반경 ≈ 9.8mm, 면적 ≈ 302mm².
    스케일이 사진에서 **측정**되는지 수치로 고정한다.
    """
    out = _run(single_settings, product_k="ring")
    fusion = out["meta"]["scale_fusion"]
    rec = out["meta"]["reconstruction"]

    f_px = 1.15 * IMG
    expected_distance = f_px * 85.60 / CARD_W
    expected_radius_mm = JEWEL_R * expected_distance / f_px
    expected_area = np.pi * expected_radius_mm**2

    assert fusion["card_distance_mm"] == pytest.approx(expected_distance, rel=0.05)
    assert rec["area_proj_mm2"] == pytest.approx(expected_area, rel=0.1)


def test_stub_depth_is_honest_about_unknown_thickness(single_settings: Settings) -> None:
    """
    상수 깊이 스텁은 두께를 못 본다. 조용히 그럴듯한 값을 내지 말고
    (1) 최소 두께로 클램프하고 (2) 플래그·경고를 남기고 (3) 신뢰도를 낮춰야 한다.
    """
    out = _run(single_settings, product_k="ring")

    assert out["meta"]["reconstruction"]["thickness_clamp"] == "min"
    assert out["meta"]["scale_fusion"]["ill_conditioned"] is True
    assert out["confidence_tier"] == "low"

    warnings = " ".join(out["meta"]["sanity"]["warnings"])
    assert "두께" in warnings
    assert "EXIF" in warnings  # 폴백 K 도 알려야 한다


def test_mass_is_plausible_for_a_ring(single_settings: Settings) -> None:
    """
    무게 = 면적 × 두께 × α_k × ρ 가 실제로 그렇게 곱해지는지.

    ⚠️ 기대값을 숫자로 박지 않는다. α_k 는 실측 피드백으로 바뀌는 값이라
    (2026-08-07: ring 0.80 → 0.36) 숫자를 박아 두면 **보정할 때마다 테스트가
    깨진다.** 상수에서 유도해 사슬 자체를 검증한다.
    """
    from app.constants import HOLLOW_ALPHA_BETA_DEPTH, MATERIALS

    out = _run(single_settings, metal="gold", purity="18k", product_k="ring")
    rec = out["meta"]["reconstruction"]
    alpha = HOLLOW_ALPHA_BETA_DEPTH["ring"][0]
    rho = MATERIALS[("gold", "18k")].rho_g_cm3

    expected = rec["area_proj_mm2"] * rec["h_mean_mm"] * alpha * rho / 1000.0
    assert out["mass_est_g"] == pytest.approx(expected, rel=0.02)
    assert out["meta"]["hollow"]["table"] == "depth"
    assert out["meta"]["hollow"]["alpha_k"] == pytest.approx(alpha)
    # 소비자 반지가 물리적으로 가질 수 있는 범위 안이어야 한다
    assert 0.3 < out["mass_est_g"] < 20.0


def test_density_choice_changes_mass_proportionally(single_settings: Settings) -> None:
    """같은 사진이면 무게비 = 밀도비. 백금이 실제로 반영되는지."""
    gold = _run(single_settings, metal="gold", purity="18k", product_k="ring")
    plat = _run(single_settings, metal="platinum", purity="pt999", product_k="ring")
    assert plat["mass_est_g"] / gold["mass_est_g"] == pytest.approx(21.45 / 15.58, rel=1e-3)


def test_uncertainty_widens_when_thickness_is_assumed(single_settings: Settings) -> None:
    """
    §15.1: 범위는 tier 상수가 아니라 **측정된 불확실성**에서 나와야 한다.
    두께를 가정했으면 범위가 눈에 띄게 넓어져야 한다.
    """
    out = _run(single_settings, product_k="ring")
    sigma = out["meta"]["uncertainty"]["volume_relative_sigma"]
    assert sigma > 0.4  # 두께 가정 항(0.5)이 지배

    rng = out["mass_range"]
    assert rng is not None
    width = (rng["max_g"] - rng["min_g"]) / (2 * rng["estimate_g"])
    assert width == pytest.approx(min(0.75, np.hypot(0.30, sigma)), rel=1e-3)


def test_require_anchor_rejects_when_card_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """카드 없는 사진 + ARCHIMEDES_REQUIRE_ANCHOR=1 → 명시적 실패."""
    monkeypatch.setenv("ARCHIMEDES_BLUR_THRESHOLD", "3")
    monkeypatch.setenv("ARCHIMEDES_DEPTH_BACKEND", "stub")
    monkeypatch.setenv("ARCHIMEDES_REQUIRE_ANCHOR", "1")
    settings = Settings()

    img = np.full((IMG, IMG, 3), 225, np.uint8)
    cv2.circle(img, (IMG // 2, IMG // 2), 160, (15, 15, 20), -1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok

    from app.pipeline.exceptions import PipelineError

    inp = JobInputRecord(image="uploads/j/main.jpg", product_k="ring")
    with pytest.raises(PipelineError) as ei:
        run_pipeline("t-noanchor", inp, {"front": buf.tobytes()}, settings)
    assert ei.value.code in ("ERR_SCALE_UNRESOLVED", "ERR_CARD_NOT_FOUND")


def test_plated_products_refuse_to_report_mass(single_settings: Settings) -> None:
    """
    몸체 전체가 금이 아닌 제품은 부피와 금 함량이 무관하다(§6.2).

    실측 계기: "순금 0.005g" 기념 골드바. 0.005g/19.32 = 0.259mm³ 를
    53×19mm 에 펴면 0.26μm — 금박 두께다. 몸체(수지)를 순금으로 치면
    4.06g 이 나와 실제와 800배 차이. 숫자를 내면 거짓말이 된다.
    """
    out = _run(single_settings, metal="gold", purity="24k", product_k="plated")

    sanity = out["meta"]["sanity"]
    assert sanity["volume_unmeasurable"] is True
    assert sanity["suppress_mass_display"] is True
    assert out["mass_range"] is None
    assert out["confidence_tier"] == "low"

    warning = sanity["warnings"][0]
    assert "부피로 금 함량을 알 수 없" in warning
    assert "제품 표기" in warning


def test_solid_products_still_report_mass(single_settings: Settings) -> None:
    """억제가 일반 제품까지 막아 버리면 안 된다."""
    out = _run(single_settings, metal="gold", purity="24k", product_k="ring")
    assert out["meta"]["sanity"]["volume_unmeasurable"] is False
    assert out["mass_range"] is not None


def test_plated_with_declared_label_reports_that_weight(single_settings: Settings) -> None:
    """
    몸체가 금이 아닌 제품은 부피로 못 재지만 함유량이 **각인돼 있다**.
    표기값을 받으면 그걸 쓰고, 측정값이 아님을 출처로 밝힌다.
    """
    out = _run(
        single_settings,
        metal="gold",
        purity="24k",
        product_k="plated",
        declared_gold_g=0.005,
    )

    sanity = out["meta"]["sanity"]
    assert sanity["mass_source"] == "declared_label"
    assert sanity["volume_unmeasurable"] is False
    assert sanity["suppress_mass_display"] is False
    assert out["mass_est_g"] == pytest.approx(0.005)
    # 표기값은 제조사 스펙이라 우리 측정 불확실성을 얹지 않는다
    assert out["meta"]["uncertainty"]["volume_relative_sigma"] == 0.0
    # low 가 아니어야 §14.1 견적 게이팅을 통과한다
    assert out["confidence_tier"] != "low"
    assert "제품 표기" in sanity["warnings"][0]


def test_plated_without_label_still_refuses(single_settings: Settings) -> None:
    """표기값을 안 주면 종전대로 억제한다 — 몸체 부피를 금으로 치면 안 된다."""
    out = _run(single_settings, metal="gold", purity="24k", product_k="plated")
    assert out["meta"]["sanity"]["mass_source"] == "measured_volume"
    assert out["meta"]["sanity"]["suppress_mass_display"] is True


def test_declared_far_below_measured_flags_non_solid_body(single_settings: Settings) -> None:
    """
    표기값과 측정 부피를 대조하면 **몸체가 순금인지 자동 판정**할 수 있다.
    순금이면 두 값이 비슷해야 하는데, 측정이 몇 배나 크면 몸체가 금이 아니다.

    실측 계기: "FINE GOLD 0.05g" 각인 바를 goldbar 로 재면 6.74g.
    0.05g 을 그 면적에 펴면 2.9μm — 따로 제련한 순금 박을 봉입한 제품.
    """
    out = _run(
        single_settings,
        metal="gold",
        purity="24k",
        product_k="goldbar",
        reference_thickness_mm=1.0,
        declared_gold_g=0.05,
    )
    sanity = out["meta"]["sanity"]
    assert sanity["body_not_solid_gold"] is True
    assert sanity["measured_over_declared_ratio"] > 3.0
    assert out["mass_est_g"] == pytest.approx(0.05)
    assert any("몸체 전체가 순금은 아닌" in w for w in sanity["warnings"])


def test_declared_matching_measured_is_not_flagged(single_settings: Settings) -> None:
    """표기와 측정이 비슷하면 속이 꽉 찬 순금이다 — 경고를 남발하면 안 된다."""
    baseline = _run(single_settings, metal="gold", purity="18k", product_k="ring")
    out = _run(
        single_settings,
        metal="gold",
        purity="18k",
        product_k="ring",
        declared_gold_g=baseline["mass_est_g"],
    )
    assert out["meta"]["sanity"]["body_not_solid_gold"] is False
