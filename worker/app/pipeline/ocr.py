"""
각인 판독 (계획서 Step 1 확장 — 라벨 자동 추출).

골드바·골드카드는 **함유량이 제품에 각인돼 있다**("FINE GOLD 999 / 0.05g").
두께가 마이크로미터라 부피로는 원리적으로 못 재는 제품이라도, 각인을 읽으면
정확한 값을 얻는다. 측정할 수 없는 것을 추정하지 말고 **적혀 있는 것을 읽는다.**

실측(도련님 금괴 사진): 원본 컬러 크롭에서 `0.05g`(0.92) `FINE GOLD`(0.95).
CLAHE·업스케일 전처리는 오히려 인식률을 떨어뜨렸다 — 각인이 금색 위 금색이라
대비를 키우면 문자 경계가 뭉개진다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

# 한 돈 = 3.75 g (거래 관행 단위, project-concept §6.4)
DON_TO_GRAM = 3.75

# 숫자 + 단위. OCR 이 g 를 9 로 읽는 일이 잦아 뒤에 오는 잡음은 관대하게 본다.
_WEIGHT_RE = re.compile(
    r"(?<![\d.])(\d{1,4}(?:[.,]\d{1,3})?)\s*(mg|g|그램|돈|don)\b",
    re.IGNORECASE,
)
# 순도 각인: 999 / 995 / 916 / 750 / 585 / 24K …
_PURITY_KARAT_RE = re.compile(r"\b(10|14|18|22|24)\s*k\b", re.IGNORECASE)
_PURITY_FINENESS_RE = re.compile(r"\b(999|995|990|916|750|585|417)\b")

_FINENESS_TO_PURITY = {
    "999": "24k", "995": "24k", "990": "24k",
    "916": "22k", "750": "18k", "585": "14k", "417": "10k",
}


@dataclass
class LabelReading:
    """각인에서 읽어 낸 것."""

    texts: list[str] = field(default_factory=list)
    weight_g: float | None = None
    weight_confidence: float = 0.0
    weight_source_text: str | None = None
    purity: str | None = None
    purity_source_text: str | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "texts": self.texts[:12],
            "weight_g": self.weight_g,
            "weight_confidence": round(self.weight_confidence, 3),
            "weight_source_text": self.weight_source_text,
            "purity": self.purity,
            "purity_source_text": self.purity_source_text,
        }


@runtime_checkable
class OcrReader(Protocol):
    name: str

    def read(self, bgr: np.ndarray) -> list[tuple[str, float]]:
        """(텍스트, 신뢰도) 목록."""
        ...


class StubOcrReader:
    """모델 없이 도는 기본값 — 아무것도 읽지 않는다."""

    name = "stub"

    def read(self, bgr: np.ndarray) -> list[tuple[str, float]]:
        return []


class RapidOcrReader:
    """
    RapidOCR(ONNX Runtime) — 가중치가 패키지에 포함돼 별도 다운로드가 없다.

    `pip install -e ".[ocr]"`
    """

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine: Any = None

    def _get(self) -> Any:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, bgr: np.ndarray) -> list[tuple[str, float]]:
        result, _ = self._get()(bgr)
        if not result:
            return []
        out: list[tuple[str, float]] = []
        for row in result:
            try:
                out.append((str(row[1]), float(row[2])))
            except (IndexError, TypeError, ValueError):
                continue
        return out


def _to_grams(value: float, unit: str) -> float | None:
    u = unit.lower()
    if u == "mg":
        return value / 1000.0
    if u in ("g", "그램"):
        return value
    if u in ("돈", "don"):
        return value * DON_TO_GRAM
    return None


def parse_label(items: list[tuple[str, float]]) -> LabelReading:
    """
    OCR 결과 → 무게·순도.

    무게는 **가장 신뢰도 높은 매치**를 쓴다. 여러 숫자가 잡혀도 단위가 붙은
    것만 후보로 본다 — 각인에는 연도·일련번호 같은 숫자도 함께 있다.
    """
    reading = LabelReading(texts=[t for t, _ in items])

    best_conf = 0.0
    for text, conf in items:
        m = _WEIGHT_RE.search(text)
        if not m:
            continue
        raw = m.group(1).replace(",", ".")
        try:
            grams = _to_grams(float(raw), m.group(2))
        except ValueError:
            continue
        # 소비자 귀금속 범위 밖은 각인이 아니라 다른 숫자로 본다
        if grams is None or not (0.0001 <= grams <= 2000.0):
            continue
        if conf > best_conf:
            best_conf = conf
            reading.weight_g = grams
            reading.weight_confidence = conf
            reading.weight_source_text = text

    for text, _conf in items:
        km = _PURITY_KARAT_RE.search(text)
        if km:
            reading.purity = f"{km.group(1)}k"
            reading.purity_source_text = text
            break
        fm = _PURITY_FINENESS_RE.search(text)
        if fm:
            reading.purity = _FINENESS_TO_PURITY.get(fm.group(1))
            reading.purity_source_text = text
            break

    return reading


def read_label(reader: OcrReader, bgr: np.ndarray, *, try_rotations: bool = True) -> LabelReading:
    """
    각인을 읽는다. 예외는 삼킨다 — 판독 실패가 분석 실패가 되면 안 된다.

    물체가 어느 방향으로 놓일지 알 수 없으므로 **네 방향을 모두 시도**하고
    결과를 합친다. 실측: 금괴가 뒤집혀 놓여 `0.05g` 이 `6900` 으로 읽혔다.
    (촬영 규약으로 물체 **위치**는 고정할 수 있지만 **방향**까지 강요할 수는 없다)
    """
    import cv2

    rotations = (
        [(0, None), (180, cv2.ROTATE_180), (90, cv2.ROTATE_90_CLOCKWISE),
         (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]
        if try_rotations
        else [(0, None)]
    )

    merged: list[tuple[str, float]] = []
    best: LabelReading | None = None
    try:
        for _deg, rot in rotations:
            img = bgr if rot is None else cv2.rotate(bgr, rot)
            items = reader.read(img)
            if not items:
                continue
            merged.extend(items)
            got = parse_label(items)
            # 무게를 읽은 것 중 신뢰도가 가장 높은 방향을 채택
            if got.weight_g and (best is None or got.weight_confidence > best.weight_confidence):
                best = got
    except Exception as e:  # noqa: BLE001
        log.warning("label OCR failed: %s", e)
        return LabelReading()

    if best is not None:
        # 텍스트는 모든 방향에서 본 것을 합쳐 남긴다(디버깅·검수용)
        best.texts = _dedupe(merged)
        return best
    combined = parse_label(merged)
    combined.texts = _dedupe(merged)
    return combined


def _dedupe(items: list[tuple[str, float]]) -> list[str]:
    seen: dict[str, float] = {}
    for text, conf in items:
        if conf > seen.get(text, -1.0):
            seen[text] = conf
    return [t for t, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)]
