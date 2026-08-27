"""
에라토스테네스 — **기준물 없이 누끼만** 따는 경로 (계획서 Step 1: 세미-오토 라벨링).

## 왜 크기·무게가 없는가

카드(ID-1) 같은 **크기를 아는 물체**가 화면에 없으면 절대 크기는 원리적으로 안
나온다. 단안 스케일 모호성이다. 대안을 셋 다 실측해 봤고 셋 다 실패했다.

1. **metric depth 모델(Apple Depth Pro)로 절대 거리**
   카드 PnP 실측이 정답. 6장에서 예측/정답 배율 2.4·17.5·9.8·3.6·10.9·17.2 —
   중앙값 10.3배 과대. 상수배로 보정해도 잔차 ±77%. 접사(8~22cm)가 학습 분포
   밖이라 모델이 "1m 거리의 책상"으로 본다. (초점거리 추정만은 중앙값 1.09배로 쓸 만했다)

2. **주변 지물의 크기 사전값으로 σ 유추**
   RT-DETR 로 키보드·마우스를 잡아 σ 를 냈더니 카드 실측 대비 **4~6배** 어긋났다.
   배경 물건은 귀금속보다 훨씬 멀어 같은 mm/px 가 성립하지 않는다.

3. **상대깊이로 "같은 거리인지" 걸러내기**
   affine-invariant 깊이는 단조라 값이 비슷하면 거리도 비슷할 것으로 봤는데,
   실측에서 **동일 평면의 카드가 깊이차 0.12~0.28 로 탈락**하고 배경 키보드가
   0.04 로 통과했다. 이 스케일에서는 판별력이 없다.

그래서 이 경로는 **크기를 주장하지 않는다.** 정확한 값이 필요하면 카드를 함께
찍는 아르키메데스 경로를 쓴다. 여기서 내는 것은 **외곽선**이고, 그것만으로도
계획서 Step 1 의 오토라벨링 데이터셋을 만들 수 있다.

## 어떻게 찾는가

카드가 없으니 "카드 옆"이라는 탐색 제약을 못 쓴다. 대신:

- **프레임 테두리에 닿지 않는** 성분만 본다 (배경은 언제나 화면 밖으로 이어진다)
- 명도(Otsu)와 **채도** 두 후보를 만들고 `local_lab_contrast` 로 고른다
  (밝은 바닥에서는 명도 Otsu 가 그림자를 물체로 잡는다 — 실측 반지 사진)
- **화면 중앙에 가까울수록** 가점 (사용자는 찍고 싶은 것을 가운데 둔다)
- GrabCut 으로 색 분포를 넓혀 물체 전체를 덮는다
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.constants import JEWEL_AREA_FRAC_MAX, JEWEL_AREA_FRAC_MIN
from app.pipeline.appearance import chroma_foreground, local_lab_contrast
from app.pipeline.exceptions import PipelineError
from app.pipeline.jewel_mask import touches_frame_border
from app.pipeline.matting import refine_with_grabcut
from app.pipeline.non_metal import drop_non_metal

log = logging.getLogger(__name__)

# 중앙에서 이 비율(프레임 대각 대비) 이상 벗어난 성분은 가점을 잃는다
_CENTER_FALLOFF = 0.35
# 대비가 최고값의 이 비율 안이면 '엇비슷하다'고 보고 작은 쪽을 택한다
_CONTRAST_TIE_RATIO = 0.9
# 카드를 빼서 물체가 원래의 이 비율 밑으로 줄면 카드 오검출로 본다.
# 진짜 카드는 물체와 겹치지 않으므로 빼도 물체 크기가 그대로다.
_CARD_SHRINK_LIMIT = 0.6


@dataclass
class OutlineResult:
    mask: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _center_bias(cx: float, cy: float, shape: tuple[int, int]) -> float:
    """화면 중앙에 가까울수록 1, 멀수록 0.3 까지 떨어진다."""
    h, w = shape[0], shape[1]
    diag = float(np.hypot(h, w))
    d = float(np.hypot(cx - w / 2.0, cy - h / 2.0)) / max(diag, 1e-6)
    return float(max(0.3, 1.0 - d / _CENTER_FALLOFF))


def _pick_component(fg: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """
    전경 → 물체 하나.

    프레임 테두리에 닿는 성분(=배경)을 버리고, 남은 것 중 **면적 × 중앙가점**이
    가장 큰 것을 고른다. 카드가 없어 ROI 로 자를 수 없으므로 이 둘이 유일한 단서다.
    """
    cleaned = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n <= 1:
        return None

    best_i, best_score = -1, -1.0
    for i in range(1, n):
        if touches_frame_border(stats[i], shape):
            continue
        area = float(stats[i, cv2.CC_STAT_AREA])
        frac = area / float(shape[0] * shape[1])
        if not (JEWEL_AREA_FRAC_MIN <= frac <= JEWEL_AREA_FRAC_MAX):
            continue
        score = area * _center_bias(float(centroids[i][0]), float(centroids[i][1]), shape)
        if score > best_score:
            best_score, best_i = score, i
    if best_i < 0:
        return None
    return ((labels == best_i).astype(np.uint8)) * 255


# 틈을 이을 때 쓰는 커널 = 선 두께의 이 배수. 크게 잡으면 반지 구멍까지 메운다.
_GAP_BRIDGE_STROKES = 3.0


def bridge_stroke_gaps(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    물체의 **선 두께**만큼만 틈을 잇는다. (마스크, 사용한 커널 px)

    반지처럼 가는 고리는 한쪽 호가 그늘지면 그 구간의 채도·명도가 떨어져 마스크가
    끊긴다(실측: 도련님 반지 11시 방향 호가 빠졌다). 색만 보는 방법으로는 못 살린다.

    고정 크기로 닫으면 위험하다 — 크게 잡으면 **반지 구멍까지 메워** 면적이 몇 배가
    된다. 그래서 커널을 물체 자신의 선 두께에서 만든다. 거리변환의 최대값이
    선 반두께이므로, 그 3배면 가는 고리의 끊김은 잇고 구멍(반지름 수백 px)은
    건드리지 못한다.

    꽉 찬 덩어리(골드바)에서는 선 두께가 물체 반지름과 같아 커널이 커지지만,
    닫을 틈이 없으므로 결과가 바뀌지 않는다.
    """
    if int(cv2.countNonZero(mask)) < 64:
        return mask, 0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    half_stroke = float(dist.max())
    if half_stroke <= 1.0:
        return mask, 0

    ys, xs = np.where(mask > 0)
    extent = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 1)
    k = round(_GAP_BRIDGE_STROKES * 2.0 * half_stroke)
    # 물체 전체 크기의 1/4 을 넘는 커널은 형태를 바꿔 버린다
    k = max(3, min(k, extent // 4) | 1)
    if k < 3:
        return mask, 0

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # 이어 붙인 결과가 원본의 1.6배를 넘으면 형태를 바꾼 것이다 — 되돌린다
    if int(cv2.countNonZero(closed)) > int(cv2.countNonZero(mask)) * 1.6:
        return mask, 0
    return closed, k


def _card_exclusion(bgr: np.ndarray, settings: Any) -> tuple[np.ndarray | None, bool]:
    """
    사진에 신용카드가 있으면 그 영역을 지운다. (제외 마스크, 카드 있었나)

    이 경로는 "기준물 없이"가 전제지만, 사용자가 카드를 같이 찍어 올릴 수 있다.
    그때 카드는 프레임에서 가장 큰 **테두리에 안 닿는** 덩어리라 그대로 물체로
    뽑힌다(실측: 마스크가 화면의 16~27%). 귀금속보다 훨씬 크므로 반드시 뺀다.
    """
    from app.pipeline.card import try_compute_card_geometry
    from app.pipeline.jewel_mask import card_dilate_px

    card = try_compute_card_geometry(bgr, "front", settings, None)
    if card is None:
        return None, False
    h, w = bgr.shape[:2]
    filled = np.zeros((h, w), np.uint8)
    cv2.fillPoly(filled, [np.asarray(card.quad_px, dtype=np.int32)], 255)
    d = card_dilate_px(card)
    return cv2.dilate(filled, np.ones((d, d), np.uint8)), True


# 모델 마스크가 화면의 이 비율을 넘으면 물체가 아니라 배경을 잡은 것으로 본다
_MATTE_AREA_FRAC_MAX = 0.92
# 모델 마스크가 이보다 작으면 잡음으로 본다
_MATTE_AREA_FRAC_MIN = 0.0002


def _matte_outline(bgr: np.ndarray, settings: Any) -> OutlineResult | None:
    """
    학습 기반 누끼(BiRefNet). 모델이 없거나 결과가 터무니없으면 None.

    **이 경로가 본선이다.** 색 임계값 경로는 모델을 못 쓸 때의 폴백으로만 남긴다
    — 실측 근거는 `backends/matte.py` 머리말.
    """
    model_dir = getattr(settings, "matte_model_dir", None)
    if not model_dir:
        return None
    from app.pipeline.backends.matte import BiRefNetMatter

    try:
        matter = BiRefNetMatter(model_dir, getattr(settings, "matte_model_file", "model.onnx"))
        mask, meta = matter.mask_refined(bgr)
    except Exception as e:  # noqa: BLE001 — 모델 실패가 분석 실패가 되면 안 된다
        log.warning("matte backend failed, falling back to appearance: %s", e)
        return None

    frac = float(meta["area_frac"])
    if not (_MATTE_AREA_FRAC_MIN <= frac <= _MATTE_AREA_FRAC_MAX):
        log.warning("matte area %.4f out of range — falling back to appearance", frac)
        return None

    # 모델은 "두드러진 물체"를 잡지 "귀금속"을 잡지 않는다. 케이스·저울·서류가
    # 딸려 오므로 색으로 한 번 더 거른다 — 근거는 `pipeline/non_metal.py` 머리말.
    mask, nm_meta = drop_non_metal(mask, bgr)
    meta.update(nm_meta)
    h, w = bgr.shape[:2]
    meta["area_frac"] = round(float(cv2.countNonZero(mask)) / float(h * w), 6)

    meta.update({"scale_available": False, "card_present": False, "matting": "birefnet"})
    log.info("outline via birefnet frac=%.5f", frac)
    return OutlineResult(mask=mask, meta=meta)


def extract_outline(bgr: np.ndarray, settings: Any = None) -> OutlineResult:
    """기준물 없이 귀금속 외곽선을 딴다. 크기는 내지 않는다."""
    h, w = bgr.shape[:2]

    learned = _matte_outline(bgr, settings)
    if learned is not None:
        return learned

    exclude, had_card = _card_exclusion(bgr, settings)

    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(otsu.mean()) > 127:
        otsu = 255 - otsu
    foregrounds = (("otsu", otsu), ("chroma", chroma_foreground(bgr)))

    def _collect(mask_out: np.ndarray | None) -> list[tuple[str, np.ndarray, float]]:
        keep = None if mask_out is None else cv2.bitwise_not(mask_out)
        found: list[tuple[str, np.ndarray, float]] = []
        for name, fg in foregrounds:
            sel = fg if keep is None else cv2.bitwise_and(fg, keep)
            picked = _pick_component(sel, (h, w))
            if picked is not None:
                found.append((name, picked, local_lab_contrast(bgr, picked)))
        return found

    candidates = _collect(exclude)

    # 카드를 뺐더니 **아무것도 안 남으면** 그건 카드가 아니었다.
    #
    # 이 경로는 "기준물 없이"가 전제라 카드 검출은 어디까지나 보조다. 그런데
    # 카드가 없는 사진에서도 검출기가 무언가를 카드라고 우기는 일이 있고, 그게
    # 하필 물체 위를 덮으면 물체가 통째로 지워진다 — 실측(도련님 사진): 카드가
    # 없는데 화면의 24.4% 를 카드로 잡았고 그 영역이 금괴를 덮어 "귀금속을 찾지
    # 못했습니다"가 났다. 정작 Otsu 는 금괴를 2.2% 로 잘 잡고 있었다.
    #
    # 카드를 빼서 물체가 남으면 그 카드는 진짜였고(반지·금괴 사진),
    # 빼서 아무것도 안 남으면 가짜였다. 이 한 줄로 둘을 가른다.
    if exclude is not None:
        # 카드를 빼기 **전후를 비교**한다. 빼서 물체가 크게 쪼그라들면 그건
        # 카드가 아니라 **물체 자신**을 지운 것이다.
        #
        # 종전에는 "빼서 **아무것도** 안 남으면"만 봤는데, 조각이라도 남으면
        # 그대로 통과했다 — 실측(흰 배경 반지 제품컷): 반지(면적 19%)를 카드로
        # 오인해(16.7%, 중심이 반지와 5px 차이) 본체를 지우고 위쪽 조각 1.17%
        # 만 남겼다. 남은 게 원래의 6% 였는데도 "뭔가 남았으니 OK"로 지나갔다.
        without = _collect(None)
        best_with = max((int(cv2.countNonZero(c[1])) for c in candidates), default=0)
        best_without = max((int(cv2.countNonZero(c[1])) for c in without), default=0)
        if best_without > 0 and best_with < best_without * _CARD_SHRINK_LIMIT:
            log.info(
                "card exclusion shrank object %d → %d px — 오검출로 보고 무시한다",
                best_without, best_with,
            )
            exclude, had_card = None, False
            candidates = without

    if not candidates:
        raise PipelineError(
            "ERR_SILHOUETTE_AREA",
            "사진에서 귀금속을 찾지 못했습니다. 무늬 없는 바닥 위에 물체 하나만 두고, "
            "화면 가운데에 크게 나오도록 다시 찍어 주세요.",
            error_severity="soft",
            suggested_action="retake_photo",
        )

    # 대비가 엇비슷하면 **작은 쪽**을 쓴다. 금속의 반사광은 물체와 색이 같아
    # 대비가 거의 똑같이 나오는데, 반사까지 삼킨 마스크는 언제나 더 크다
    # (실측 real4: otsu 117.2 vs chroma 116.3 으로 사실상 동점인데 otsu 쪽이
    # 금괴 주변 반사를 통째로 물고 있었다).
    top = max(c[2] for c in candidates)
    close = [c for c in candidates if c[2] >= top * _CONTRAST_TIE_RATIO]
    source, mask, contrast = min(close, key=lambda c: int(cv2.countNonZero(c[1])))

    # 카드 경로에는 "카드 반경 2배 밖은 배경"이라는 울타리가 있었는데 여기엔 없다.
    # 없으면 GrabCut 이 책상 반사광까지 먹는다(실측 real4: 마스크가 화면의 19%,
    # 금괴 주변 반사가 통째로 딸려왔다). 씨앗 주변으로 울타리를 친다.
    ys, xs = np.where(mask > 0)
    seed_span = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 8)
    near_seed = cv2.dilate(
        mask, np.ones((seed_span | 1, seed_span | 1), np.uint8)
    )
    fence = cv2.bitwise_not(near_seed)
    exclude = fence if exclude is None else cv2.bitwise_or(exclude, fence)

    mask, matte = refine_with_grabcut(bgr, mask, exclude=exclude)
    mask, bridged_px = bridge_stroke_gaps(mask)
    if bridged_px:
        matte["bridged_gap_px"] = bridged_px

    frac = float(np.count_nonzero(mask)) / float(h * w)
    meta: dict[str, Any] = {
        "backend": "outline_no_reference",
        "appearance_source": source,
        "appearance_contrast": round(contrast, 2),
        "appearance_candidates": {n: round(c, 2) for n, _m, c in candidates},
        "area_frac": round(frac, 6),
        # 이 경로는 크기를 주장하지 않는다 — 소비 측이 착각하지 않도록 명시한다
        "scale_available": False,
        "card_present": had_card,
    }
    meta.update(matte)
    log.info("outline source=%s frac=%.5f contrast=%.1f", source, frac, contrast)
    return OutlineResult(mask=mask, meta=meta)
