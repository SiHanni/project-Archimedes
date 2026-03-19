import pytest

from app.config import Settings


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Relaxed thresholds for unit tests (CI / no GPU)."""
    monkeypatch.setenv("ARCHIMEDES_MIN_SHORT_EDGE", "800")
    monkeypatch.setenv("ARCHIMEDES_BLUR_THRESHOLD", "5")
    monkeypatch.setenv("ARCHIMEDES_SCALE_MISMATCH", "0.25")
    monkeypatch.setenv("ARCHIMEDES_VOXEL_PENALTY_RATIO", "0.5")
    monkeypatch.setenv("ARCHIMEDES_USE_VOXEL_CARVE", "0")
    return Settings()
