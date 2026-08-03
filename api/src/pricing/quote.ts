/**
 * 견적 산출 — 순수 함수 (연구개발계획서 Step 2-1 "중량 + 실시간 금시세 → 예상 견적").
 *
 * 부작용·I/O 가 없어야 테스트·검증이 가능하므로 시세 조회(`PricingService`)와 분리한다.
 *
 * 노출 정책은 `project-concept.mdc` §14.1 을 따른다.
 * - `confidence_tier=low` → 금액을 **아예 내지 않는다**. ±30% 구간 견적을 그대로
 *   보여 주면 체감 신뢰도가 무너진다(§1.5: 한돈 85만 원 기준 51만 원 폭).
 * - 무게 표시가 억제된 job(`suppress_mass_display`)도 금액을 내지 않는다.
 */

export type QuoteInput = {
  massEstG: number;
  massRange?: { min_g: number; estimate_g: number; max_g: number } | null;
  confidenceTier: string;
  suppressMassDisplay?: boolean;
};

export type SpotPrice = {
  /** 순물질 1g 당 원화 */
  krwPerGram: number;
  metal: string;
  purity: string;
  source: string;
  asOf: string;
  stale: boolean;
};

export type Quote = {
  currency: 'KRW';
  krwPerGram: number;
  /** 매입률(0~1). 소매 시세 대비 실제 매입가 비율 — 운영 정책값 */
  buyRate: number;
  estimate: number;
  min: number | null;
  max: number | null;
  source: string;
  asOf: string;
  stale: boolean;
  disclaimer: string;
};

export type QuoteSuppressed = {
  suppressed: true;
  reason: 'low_confidence' | 'mass_suppressed' | 'no_price';
  message: string;
};

export const QUOTE_DISCLAIMER =
  '실시간 시세 × 추정 무게로 계산한 참고값입니다. 감정·매입 확정 금액이 아니며 업체·상태에 따라 달라집니다.';

const SUPPRESS_MESSAGE: Record<QuoteSuppressed['reason'], string> = {
  low_confidence:
    '신뢰도가 낮아 금액을 표시하지 않습니다. 안내에 맞게 다시 촬영하면 금액까지 확인할 수 있어요.',
  mass_suppressed: '추정 무게가 비현실적이라 금액을 계산하지 않았습니다.',
  no_price: '시세를 가져오지 못해 금액을 계산하지 못했습니다.',
};

function round10(v: number): number {
  return Math.round(v / 10) * 10;
}

export function buildQuote(
  input: QuoteInput,
  spot: SpotPrice | null,
  buyRate = 1.0,
): Quote | QuoteSuppressed {
  if (input.suppressMassDisplay) {
    return { suppressed: true, reason: 'mass_suppressed', message: SUPPRESS_MESSAGE.mass_suppressed };
  }
  // §14.1 신뢰도 게이팅 — 낮으면 금액을 아예 숨긴다
  if (input.confidenceTier === 'low') {
    return { suppressed: true, reason: 'low_confidence', message: SUPPRESS_MESSAGE.low_confidence };
  }
  if (!spot || !Number.isFinite(spot.krwPerGram) || spot.krwPerGram <= 0) {
    return { suppressed: true, reason: 'no_price', message: SUPPRESS_MESSAGE.no_price };
  }

  const effective = spot.krwPerGram * buyRate;
  const range = input.massRange;
  return {
    currency: 'KRW',
    krwPerGram: spot.krwPerGram,
    buyRate,
    estimate: round10(input.massEstG * effective),
    // 범위는 무게 불확실성을 그대로 물려받는다. 없으면 단일값만 낸다.
    min: range ? round10(range.min_g * effective) : null,
    max: range ? round10(range.max_g * effective) : null,
    source: spot.source,
    asOf: spot.asOf,
    stale: spot.stale,
    disclaimer: QUOTE_DISCLAIMER,
  };
}

export function isSuppressed(q: Quote | QuoteSuppressed): q is QuoteSuppressed {
  return (q as QuoteSuppressed).suppressed === true;
}
