from app.pipeline import hollow


def test_mass_unit_mm3_to_g() -> None:
    # 1 cm³ gold 18k ~ 15.58 g => 1000 mm³
    m = hollow.mass_g(1000.0, "gold", "18k")
    assert 15.0 < m < 16.5
