"""각인 판독 — 무게·순도 파싱 (계획서 Step 1 확장)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.ocr import DON_TO_GRAM, LabelReading, parse_label, read_label


class FakeReader:
    """방향마다 다른 결과를 주는 가짜 OCR — 회전 병합을 검증한다."""

    name = "fake"

    def __init__(self, per_call: list[list[tuple[str, float]]]) -> None:
        self._per_call = per_call
        self.calls = 0

    def read(self, bgr):
        i = min(self.calls, len(self._per_call) - 1)
        self.calls += 1
        return self._per_call[i]


def test_parses_grams_from_engraving() -> None:
    r = parse_label([("FINE GOLD", 0.95), ("0.05g", 0.92)])
    assert r.weight_g == pytest.approx(0.05)
    assert r.weight_confidence == pytest.approx(0.92)
    assert r.weight_source_text == "0.05g"


def test_unit_conversion() -> None:
    assert parse_label([("500mg", 0.9)]).weight_g == pytest.approx(0.5)
    assert parse_label([("1돈", 0.9)]).weight_g == pytest.approx(DON_TO_GRAM)
    assert parse_label([("3,75 g", 0.9)]).weight_g == pytest.approx(3.75)


def test_bare_numbers_are_not_weights() -> None:
    """각인에는 일련번호·연도가 함께 있다 — 단위 없는 숫자를 무게로 읽으면 안 된다."""
    r = parse_label([("6900", 0.99), ("0020", 0.95), ("FINE GOLD", 0.97)])
    assert r.weight_g is None


def test_absurd_values_rejected() -> None:
    assert parse_label([("9999g", 0.99)]).weight_g is None


def test_highest_confidence_wins() -> None:
    r = parse_label([("1g", 0.55), ("0.05g", 0.97)])
    assert r.weight_g == pytest.approx(0.05)


def test_purity_from_fineness_and_karat() -> None:
    assert parse_label([("999", 0.9)]).purity == "24k"
    assert parse_label([("585", 0.9)]).purity == "14k"
    assert parse_label([("18K", 0.9)]).purity == "18k"


def test_rotation_merge_finds_upside_down_text() -> None:
    """
    실측: 금괴가 뒤집혀 놓여 0.05g 이 `6900` 으로 읽혔다.
    촬영 규약으로 위치는 고정해도 **방향**은 강요할 수 없다 → 네 방향 시도.
    """
    reader = FakeReader([
        [("6900", 0.99), ("FINE GOLD", 0.95)],   # 0도 — 무게 못 읽음
        [("0.05g", 0.96), ("FINE GOLD", 0.94)],  # 180도 — 읽힘
        [],
        [],
    ])
    img = np.zeros((40, 80, 3), np.uint8)
    r = read_label(reader, img)
    assert reader.calls == 4
    assert r.weight_g == pytest.approx(0.05)
    assert "6900" in r.texts and "0.05g" in r.texts, "검수용으로 모든 방향 텍스트를 남긴다"


def test_reader_failure_does_not_break_analysis() -> None:
    class Boom:
        name = "boom"

        def read(self, bgr):
            raise RuntimeError("model exploded")

    assert read_label(Boom(), np.zeros((10, 10, 3), np.uint8)) == LabelReading()


def test_no_text_returns_empty_reading() -> None:
    reader = FakeReader([[]])
    r = read_label(reader, np.zeros((10, 10, 3), np.uint8))
    assert r.weight_g is None and r.texts == []
