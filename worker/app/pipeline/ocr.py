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
# 정수부가 0 으로 시작하고 소수점이 없다 = 소수점을 놓친 조각
_TRUNCATED_DECIMAL_RE = re.compile(r"^0\d")

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

    candidates: list[tuple[str, float, float]] = []  # (원문, 신뢰도, 그램)
    for text, conf in items:
        m = _WEIGHT_RE.search(text)
        if not m:
            continue
        raw = m.group(1).replace(",", ".")
        # "05" · "005" 처럼 **앞자리 0 뒤에 바로 숫자**가 오는 표기는 실제 각인에
        # 없다. 소수점을 놓친 조각이다(실측: "0.05g" → "05g" → 5 g, 정답의 100배).
        # 무엇이 떨어졌는지 알 수 없으므로 복원하지 않고 버린다 — 100배 틀린 값을
        # 내놓느니 부피 측정으로 내려가는 편이 낫다.
        if _TRUNCATED_DECIMAL_RE.match(raw):
            continue
        try:
            grams = _to_grams(float(raw), m.group(2))
        except ValueError:
            continue
        # 소비자 귀금속 범위 밖은 각인이 아니라 다른 숫자로 본다
        if grams is None or not (0.0001 <= grams <= 2000.0):
            continue
        candidates.append((text, conf, grams))

    # **잘린 조각을 버린다.** OCR 은 같은 각인을 여러 번, 여러 방향에서 읽는데
    # 앞자리가 떨어져 나간 조각이 오히려 더 높은 신뢰도를 받는 일이 있다.
    # 실측(도련님 사진): 온전한 "0.05g"(낮은 신뢰도)와 잘린 "05g"(0.988)가 함께
    # 잡혀 신뢰도만 보고 **5 g** 으로 읽었다 — 정답의 100배다.
    # 어떤 후보의 원문이 다른 후보 원문에 통째로 들어가면 그건 조각이다.
    trimmed = [
        c
        for c in candidates
        if not any(c[0] != o[0] and c[0] in o[0] for o in candidates)
    ]
    for text, conf, grams in trimmed or candidates:
        if conf > reading.weight_confidence:
            reading.weight_confidence = conf
            reading.weight_g = grams
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
    try:
        for _deg, rot in rotations:
            img = bgr if rot is None else cv2.rotate(bgr, rot)
            items = reader.read(img)
            if items:
                merged.extend(items)
    except Exception as e:  # noqa: BLE001
        log.warning("label OCR failed: %s", e)
        return LabelReading()

    # **모든 방향을 합쳐 한 번에 판정한다.** 방향별로 따로 뽑아 신뢰도로 고르면
    # 조각 필터가 무력해진다 — 실측(도련님 08:25 사진)에서 한 방향은 잘린
    # "05g"(0.988)를, 다른 방향은 온전한 "0.05g"를 냈는데, 방향별 최고를 고르는
    # 바람에 둘이 만나지 못하고 5 g 이 채택됐다. 정답의 100배다.
    combined = parse_label(merged)
    combined.texts = _dedupe(merged)
    return combined


def _dedupe(items: list[tuple[str, float]]) -> list[str]:
    seen: dict[str, float] = {}
    for text, conf in items:
        if conf > seen.get(text, -1.0):
            seen[text] = conf
    return [t for t, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)]
