/**
 * 견적 노출 정책 회귀 (project-concept §14.1).
 *
 * 순수 함수라 빌드 산출물(dist)만 있으면 외부 의존 없이 검증된다.
 * 실행: npm test  (nest build → node --test)
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildQuote, isSuppressed, QUOTE_DISCLAIMER } from '../dist/pricing/quote.js';

const spot = {
  krwPerGram: 100000,
  metal: 'gold',
  purity: '18k',
  source: 'test',
  asOf: '2026-01-01T00:00:00.000Z',
  stale: false,
};

const base = {
  massEstG: 5,
  massRange: { min_g: 4, estimate_g: 5, max_g: 6 },
  confidenceTier: 'high',
};

test('시세 × 무게로 금액과 범위를 낸다', () => {
  const q = buildQuote(base, spot);
  assert.equal(isSuppressed(q), false);
  assert.equal(q.estimate, 500000);
  assert.equal(q.min, 400000);
  assert.equal(q.max, 600000);
  assert.equal(q.disclaimer, QUOTE_DISCLAIMER);
});

test('§14.1 — tier low 면 금액을 아예 내지 않는다', () => {
  const q = buildQuote({ ...base, confidenceTier: 'low' }, spot);
  assert.equal(isSuppressed(q), true);
  assert.equal(q.reason, 'low_confidence');
  assert.equal('estimate' in q, false, '억제 시 금액 필드가 새어 나가면 안 된다');
});

test('무게 표시가 억제된 job 은 금액도 내지 않는다', () => {
  const q = buildQuote({ ...base, suppressMassDisplay: true }, spot);
  assert.equal(q.reason, 'mass_suppressed');
});

test('시세를 못 가져오면 0원이 아니라 억제한다', () => {
  for (const bad of [null, { ...spot, krwPerGram: 0 }, { ...spot, krwPerGram: NaN }]) {
    const q = buildQuote(base, bad);
    assert.equal(isSuppressed(q), true, `bad spot: ${JSON.stringify(bad)}`);
    assert.equal(q.reason, 'no_price');
  }
});

test('매입률이 금액에 곱해지고 원본 시세는 그대로 노출된다', () => {
  const q = buildQuote(base, spot, 0.9);
  assert.equal(q.estimate, 450000);
  assert.equal(q.krwPerGram, 100000, '표시용 시세는 매입률 적용 전 값');
  assert.equal(q.buyRate, 0.9);
});

test('범위가 없으면 단일 금액만 낸다', () => {
  const q = buildQuote({ ...base, massRange: null }, spot);
  assert.equal(q.estimate, 500000);
  assert.equal(q.min, null);
  assert.equal(q.max, null);
});

test('금액은 10원 단위로 반올림한다', () => {
  const q = buildQuote({ ...base, massEstG: 5.1234, massRange: null }, { ...spot, krwPerGram: 12345 });
  assert.equal(q.estimate % 10, 0);
});

test('묵은 시세 표시가 그대로 전달된다', () => {
  const q = buildQuote(base, { ...spot, stale: true });
  assert.equal(q.stale, true);
});
