"""금속·함량 해석과 입력 스키마(단일사진/다뷰) 회귀."""

from __future__ import annotations

import pytest

from app.constants import MATERIALS
from app.models.schemas import JobInputRecord, JobViews
from app.pipeline.exceptions import PipelineError
from app.pipeline.hollow import mass_g, resolve_material


def test_platinum_supported() -> None:
    """계획서 요구: 금·은·**백금** + 함량 입력."""
    assert resolve_material("platinum", "pt950").rho_g_cm3 == pytest.approx(20.10)
    assert resolve_material("platinum", "950").rho_g_cm3 == pytest.approx(20.10)
    assert resolve_material("백금", "pt999").rho_g_cm3 == pytest.approx(21.45)


def test_purity_alias_is_metal_scoped() -> None:
    """'999' 는 금이면 24K, 은이면 fine, 백금이면 Pt999 — 금속별로 달라야 한다."""
    assert resolve_material("gold", "999").purity == "24k"
    assert resolve_material("silver", "999").purity == "fine"
    assert resolve_material("platinum", "999").purity == "pt999"


def test_sterling_is_not_fine_silver() -> None:
    """925 는 순은(10.49)이 아니라 10.36 — 이전에는 둘 다 10.49 였다."""
    sterling = resolve_material("silver", "925").rho_g_cm3
    fine = resolve_material("silver", "fine").rho_g_cm3
    assert sterling < fine
    assert sterling == pytest.approx(10.36)


def test_unsupported_material_raises_instead_of_silent_gold_fallback() -> None:
    """
    이전 구현은 미지원 조합을 **말없이 18K 금**으로 계산했다.
    백금·22K 요청이 통째로 다른 밀도로 나가는 심각한 오류라 명시적 실패로 바꿨다.
    """
    for metal, purity in (("palladium", "950"), ("gold", "13k"), ("silver", "pt950")):
        with pytest.raises(PipelineError) as ei:
            resolve_material(metal, purity)
        assert ei.value.code == "ERR_UNSUPPORTED_MATERIAL"


def test_mass_unit_conversion() -> None:
    """1 cm³ = 1000 mm³."""
    assert mass_g(1000.0, "gold", "18k") == pytest.approx(15.58)
    assert mass_g(1000.0, "platinum", "pt999") == pytest.approx(21.45)


def test_density_ordering_matches_domain() -> None:
    """§6.4: 같은 부피면 24K > 18K > 14K > 10K."""
    r = [MATERIALS[("gold", k)].rho_g_cm3 for k in ("24k", "22k", "18k", "14k", "10k")]
    assert r == sorted(r, reverse=True)


# ───────────────────────── 입력 스키마 ─────────────────────────


def test_legacy_five_view_row_infers_multiview() -> None:
    """기존 DB 행(views 만 있음)이 그대로 동작해야 한다."""
    rec = JobInputRecord(
        views=JobViews(front="a", top="b", left="c", right="d", back="e"),
    )
    assert rec.capture_mode == "multiview"
    assert list(rec.image_keys()) == ["front", "top", "left", "right", "back"]


def test_single_photo_input() -> None:
    rec = JobInputRecord(image="uploads/x/main.jpg", metal="platinum", purity="pt950")
    assert rec.capture_mode == "single"
    assert rec.image_keys() == {"front": "uploads/x/main.jpg"}


def test_mode_requires_matching_payload() -> None:
    with pytest.raises(ValueError):
        JobInputRecord(capture_mode="single")
    with pytest.raises(ValueError):
        JobInputRecord(capture_mode="multiview", image="only-one.jpg")
