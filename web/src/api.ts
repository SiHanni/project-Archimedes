const API_BASE_KEY = 'archimedes.apiBase';

/**
 * API 주소를 **런타임에** 정한다.
 *
 * 우선순위: `?api=` 쿼리 → localStorage → 빌드 시 `VITE_API_BASE` → `/api`
 *
 * 왜 런타임인가: 로컬 API 를 cloudflared Quick Tunnel 로 노출하는데 이 터널은
 * 끊기면 **주소가 바뀐다**. 빌드 시점에 주소를 구우면 바뀔 때마다 Vercel 환경변수
 * 갱신 + 재배포가 필요하고, 그 사이 사용자는 "Load Failed" 만 본다.
 * 쿼리로 한 번 넘겨 주면 localStorage 에 남아 재배포 없이 계속 쓴다.
 */
function resolveApiBase(): string {
  const strip = (v: string) => v.replace(/\/+$/, '');
  try {
    const fromQuery = new URLSearchParams(window.location.search).get('api');
    if (fromQuery) {
      const v = strip(fromQuery);
      window.localStorage.setItem(API_BASE_KEY, v);
      return v;
    }
    const saved = window.localStorage.getItem(API_BASE_KEY);
    if (saved) return strip(saved);
  } catch {
    // localStorage 차단(프라이빗 모드 등) — 빌드 기본값으로 계속 간다
  }
  return strip(import.meta.env.VITE_API_BASE || '/api');
}

export const apiBase = resolveApiBase();

/** fetch 자체가 실패한 경우(네트워크·CORS·터널 다운)를 사용자 말로 옮긴다 */
class ApiUnreachableError extends Error {
  constructor(base: string) {
    super(
      `API 서버에 연결할 수 없습니다. (${base})\n` +
        '로컬 서버가 꺼져 있거나 터널 주소가 바뀌었을 수 있어요. ' +
        '새 주소를 받으셨다면 이 페이지 주소 뒤에 ?api=새주소/v1 을 붙여 한 번만 열어 주세요.',
    );
    this.name = 'ApiUnreachableError';
  }
}

export async function postJob(form: FormData): Promise<{ id: string; status: string }> {
  let r: Response;
  try {
    r = await fetch(`${apiBase}/jobs`, { method: 'POST', body: form });
  } catch {
    throw new ApiUnreachableError(apiBase);
  }
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

export async function getJob(id: string): Promise<JobDto> {
  let r: Response;
  try {
    r = await fetch(`${apiBase}/jobs/${id}`);
  } catch {
    throw new ApiUnreachableError(apiBase);
  }
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export type JobSanityMeta = {
  suppress_mass_display?: boolean;
  implausible_mass?: boolean;
  sanity_mass_cap_g?: number;
  used_card_fallback_views?: string[];
  warnings?: string[];
  raw_mass_est_g?: number;
  volume_unmeasurable?: boolean;
  /** measured_volume | declared_label — 표기값 기반이면 그 사실을 UI 가 밝혀야 한다 */
  mass_source?: string;
  body_not_solid_gold?: boolean;
  measured_mass_g?: number | null;
  measured_over_declared_ratio?: number | null;
};

/** worker `runner` — soft/hard 등급과 재촬영 유도 */
export type JobWorkflowMeta = {
  error_severity?: 'none' | 'soft' | 'hard';
  suggested_action?: 'retry_one_view' | 'retake_photo' | 'continue_low_confidence' | string;
  retry_views?: string[];
  degraded_reasons?: string[];
};

/** worker `scale_fusion.ScaleFusionResult.as_meta()` */
export type JobScaleFusionMeta = {
  method?: string;
  anchor_used?: boolean;
  ill_conditioned?: boolean;
  card_distance_mm?: number | null;
  depth_rmse_mm?: number | null;
};

/** worker `reconstruct.Reconstruction.as_meta()` */
export type JobReconstructionMeta = {
  method?: string;
  area_proj_mm2?: number;
  length_mm?: number;
  width_mm?: number;
  h_mean_mm?: number;
  thickness_clamp?: string | null;
};

/** api `pricing/quote.ts` — 시세×무게 견적 (§14.1 게이팅으로 숨겨질 수 있음) */
export type JobQuote =
  | {
      suppressed: true;
      reason: 'low_confidence' | 'mass_suppressed' | 'no_price';
      message: string;
    }
  | {
      currency: 'KRW';
      krwPerGram: number;
      buyRate: number;
      estimate: number;
      min: number | null;
      max: number | null;
      source: string;
      asOf: string;
      stale: boolean;
      disclaimer: string;
    };

export function isQuoteSuppressed(
  q: JobQuote,
): q is Extract<JobQuote, { suppressed: true }> {
  return (q as { suppressed?: boolean }).suppressed === true;
}

/** worker `visualize` — 오토 라벨링·검수 산출물 */
export type JobSegmentationMeta = {
  backend?: string;
  placement_mode?: string;
  object_side?: string;
  area_frac?: number;
  /** 이미지 없이도 라벨로 쓸 수 있는 픽셀 좌표 폴리곤 */
  polygon_xy?: number[][];
  polygon_points?: number;
  image_width?: number;
  image_height?: number;
  assets?: string[];
};

/** 세그 산출물은 API 가 S3 를 프록시해 준다 */
export function assetUrl(jobId: string, name: string): string {
  return `${apiBase}/jobs/${jobId}/assets/${name}`;
}

export type JobDto = {
  id: string;
  status: string;
  algorithmVersion?: string | null;
  /**
   * soft 에러 job 은 분석 결과가 없다. 그때도 재촬영 안내가 필요하므로
   * 서버가 workflow 를 **result 밖**으로 따로 내려준다.
   */
  workflow?: JobWorkflowMeta | null;
  result: {
    mass_est_g?: number;
    confidence_tier?: string;
    confidence_pct?: number;
    mass_range?: { min_g: number; estimate_g: number; max_g: number } | null;
    V_hull_mm3?: number;
    V_adj_mm3?: number;
    algorithm_version?: string;
    meta?: {
      capture_mode?: string;
      sanity?: JobSanityMeta;
      segmentation?: JobSegmentationMeta;
      /** worker `ocr` — 제품 각인에서 읽어 낸 함유량·순도 */
      label_ocr?: {
        texts?: string[];
        weight_g?: number | null;
        weight_confidence?: number;
        weight_source_text?: string | null;
        purity?: string | null;
      };
      workflow?: JobWorkflowMeta;
      scale_fusion?: JobScaleFusionMeta;
      reconstruction?: JobReconstructionMeta;
    };
  } | null;
  error: {
    code: string;
    message: string;
    retryViews?: string[];
    errorSeverity?: 'soft' | 'hard';
    suggestedAction?: string | null;
  } | null;
  quote?: JobQuote | null;
};
