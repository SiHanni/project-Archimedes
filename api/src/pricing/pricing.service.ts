import { Injectable, Logger } from '@nestjs/common';
import type { SpotPrice } from './quote';

/**
 * 시세 조회 (연구개발계획서 Step 2-1 "실시간 금시세 API").
 *
 * 시세는 **외부 소스**다(`project-concept.mdc` §1.2). 여기서는 교체 가능한
 * 어댑터만 두고, 실패하면 조용히 0 을 쓰지 않고 `null` 을 돌려 견적을 생략한다.
 *
 * 백엔드
 * - `static` (기본): `PRICE_TABLE_KRW_PER_GRAM` 환경변수의 JSON. 시세가 고정이라
 *   데모·오프라인 개발용이며 항상 `stale=true` 로 표시된다.
 * - `http`: `PRICE_API_URL` 에 GET. 응답에서 g당 원화를 찾아 쓴다.
 *
 * 캐시: 같은 (metal, purity) 를 `PRICE_TTL_SECONDS` 동안 재사용해 외부 호출을 줄인다.
 */
@Injectable()
export class PricingService {
  private readonly log = new Logger(PricingService.name);
  private cache = new Map<string, { value: SpotPrice; expiresAt: number }>();

  private get backend(): string {
    return (process.env.PRICE_BACKEND || 'static').trim().toLowerCase();
  }

  private get ttlMs(): number {
    return Math.max(0, parseInt(process.env.PRICE_TTL_SECONDS || '60', 10)) * 1000;
  }

  /** 소매 시세 대비 매입률. 운영 정책값이며 1.0 이면 시세 그대로. */
  get buyRate(): number {
    const n = parseFloat(process.env.PRICE_BUY_RATE || '1');
    return Number.isFinite(n) && n > 0 && n <= 1 ? n : 1;
  }

  async getSpot(metal: string, purity: string): Promise<SpotPrice | null> {
    const key = `${metal}:${purity}`.toLowerCase();
    const hit = this.cache.get(key);
    if (hit && hit.expiresAt > Date.now()) return hit.value;

    let value: SpotPrice | null = null;
    try {
      value =
        this.backend === 'http'
          ? await this.fetchHttp(metal, purity)
          : this.fromStaticTable(metal, purity);
    } catch (e) {
      // 시세를 못 가져오는 것은 분석 실패가 아니다 — 견적만 생략하고 job 은 살린다
      this.log.warn(`spot price lookup failed for ${key}: ${String(e)}`);
      value = null;
    }

    if (value) this.cache.set(key, { value, expiresAt: Date.now() + this.ttlMs });
    return value;
  }

  /**
   * `PRICE_TABLE_KRW_PER_GRAM` 예:
   *   {"gold:24k":118000,"gold:18k":86000,"silver:sterling":1400,"platinum:pt950":52000}
   */
  private fromStaticTable(metal: string, purity: string): SpotPrice | null {
    const raw = process.env.PRICE_TABLE_KRW_PER_GRAM;
    if (!raw) return null;
    let table: Record<string, number>;
    try {
      table = JSON.parse(raw);
    } catch {
      this.log.warn('PRICE_TABLE_KRW_PER_GRAM is not valid JSON');
      return null;
    }
    const key = `${metal}:${purity}`.toLowerCase();
    const krwPerGram = table[key];
    if (typeof krwPerGram !== 'number' || !(krwPerGram > 0)) return null;
    return {
      krwPerGram,
      metal,
      purity,
      source: 'static-table',
      // ⚠️ 조회 시각을 쓰면 안 된다. 고정 표는 며칠 전 값일 수 있는데 `asOf` 에
      // now() 를 넣으면 **오늘 시세인 척**하게 된다. 표를 채운 날짜를 그대로 밝힌다.
      asOf: staticTableAsOf(),
      // 고정 표는 실시간이 아니다 — UI 가 "참고용"임을 알 수 있게 항상 표시
      stale: true,
    };
  }

  private async fetchHttp(metal: string, purity: string): Promise<SpotPrice | null> {
    const base = process.env.PRICE_API_URL;
    if (!base) return null;

    const url = new URL(base);
    url.searchParams.set('metal', metal);
    url.searchParams.set('purity', purity);

    const headers: Record<string, string> = { accept: 'application/json' };
    if (process.env.PRICE_API_KEY) {
      headers.authorization = `Bearer ${process.env.PRICE_API_KEY}`;
    }

    const timeoutMs = Math.max(
      500,
      parseInt(process.env.PRICE_TIMEOUT_MS || '3000', 10),
    );
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const res = await fetch(url, { headers, signal: ac.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as Record<string, unknown>;
      const krwPerGram = pickNumber(body, [
        'krwPerGram',
        'krw_per_gram',
        'pricePerGram',
        'price_per_gram',
        'value',
      ]);
      if (krwPerGram === null || krwPerGram <= 0) {
        throw new Error('response has no positive per-gram price');
      }
      const asOf =
        typeof body.asOf === 'string'
          ? body.asOf
          : typeof body.as_of === 'string'
            ? body.as_of
            : new Date().toISOString();
      return {
        krwPerGram,
        metal,
        purity,
        source: typeof body.source === 'string' ? body.source : url.host,
        asOf,
        stale: isStale(asOf),
      };
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * 고정 표가 **언제 기준인지**. `PRICE_TABLE_AS_OF`(YYYY-MM-DD 또는 ISO)를 쓴다.
 *
 * 지정이 없으면 조회 시각으로 떨어지는데, 그러면 며칠 전 값이 오늘 시세로
 * 보고된다. 시세는 매일 움직이므로 이건 그냥 틀린 정보다.
 */
function staticTableAsOf(): string {
  const raw = (process.env.PRICE_TABLE_AS_OF || '').trim();
  if (raw) {
    const d = new Date(raw.length === 10 ? `${raw}T00:00:00+09:00` : raw);
    if (!Number.isNaN(d.getTime())) return d.toISOString();
  }
  return new Date().toISOString();
}

function pickNumber(obj: Record<string, unknown>, keys: string[]): number | null {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
    if (typeof v === 'string') {
      const n = parseFloat(v);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

/** 15분 넘게 묵은 시세는 "실시간"이라고 말하지 않는다 */
function isStale(asOf: string, maxAgeMs = 15 * 60 * 1000): boolean {
  const t = Date.parse(asOf);
  if (Number.isNaN(t)) return true;
  return Date.now() - t > maxAgeMs;
}
