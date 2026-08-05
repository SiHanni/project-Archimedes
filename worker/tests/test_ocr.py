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


def test_truncated_fragment_does_not_beat_full_reading():
    """
    앞자리가 떨어져 나간 조각이 신뢰도만 높다고 이기면 안 된다.

    실측(도련님 08:25 사진): 온전한 "0.05g"(0.72)와 잘린 "05g"(0.988)가 함께
    잡혀 신뢰도만 보고 5 g 으로 읽었다 — 정답의 100배.
    """
    from app.pipeline.ocr import parse_label

    items = [
        ("999", 0.99), ("666", 0.6), ("05g", 0.988), ("FINE", 0.95),
        ("0.05g", 0.72), ("GOD", 0.8), ("FINE GO#D", 0.6),
    ]
    r = parse_label(items)
    assert r.weight_g == 0.05
    assert r.weight_source_text == "0.05g"
    assert r.purity == "24k"


def test_full_reading_still_wins_on_confidence_when_no_fragment():
    """조각이 없으면 종전대로 신뢰도가 가장 높은 것을 쓴다."""
    from app.pipeline.ocr import parse_label

    r = parse_label([("3.75g", 0.6), ("1.2 g", 0.95)])
    assert r.weight_g == 1.2
    assert r.weight_source_text == "1.2 g"


def test_truncated_decimal_is_rejected_when_alone():
    """
    온전한 판독이 없어도 조각을 그냥 쓰면 안 된다.

    "05g" 는 소수점을 놓친 조각인데 무엇이 떨어졌는지 알 수 없다
    ("0.05" 일 수도 "0.5" 일 수도 있다). 복원하지 말고 버린다 —
    100배 틀린 값보다 "못 읽음"이 낫다.
    """
    from app.pipeline.ocr import parse_label

    assert parse_label([("05g", 0.99), ("999", 0.9)]).weight_g is None
    assert parse_label([("005 g", 0.99)]).weight_g is None
    # 정상 표기는 영향 없어야 한다
    assert parse_label([("5g", 0.9)]).weight_g == 5.0
    assert parse_label([("0.5g", 0.9)]).weight_g == 0.5


def test_rotations_are_judged_together_not_per_rotation():
    """
    회전별로 따로 뽑아 신뢰도로 고르면 조각 필터가 무력해진다.

    실측: 한 방향은 "05g"(0.988), 다른 방향은 "0.05g" 를 냈는데 방향별 최고를
    고르는 바람에 5 g 이 채택됐다.
    """
    from app.pipeline.ocr import read_label

    class TwoFaceReader:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def read(self, bgr):
            self.calls += 1
            # 첫 방향은 잘린 조각(고신뢰), 두 번째 방향은 온전한 판독(저신뢰)
            return [("05g", 0.988)] if self.calls == 1 else [("0.05g", 0.70)]

    r = read_label(TwoFaceReader(), np.zeros((40, 40, 3), np.uint8))
    assert r.weight_g == 0.05
    assert r.weight_source_text == "0.05g"
