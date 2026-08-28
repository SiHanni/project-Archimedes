const API_BASE_KEY = 'archimedes.apiBase';

/**
 * API 주소를 **런타임에** 정한다.
 *
 * 우선순위: `?api=` 쿼리 → localStorage(임시 주소) → 빌드 시 `VITE_API_BASE`
 *
 * ## 왜 런타임인가
 *
 * 임시 터널로 API 를 노출하던 시절, 터널이 끊기면 주소가 바뀌었다. 그때마다
 * 재배포하는 대신 `?api=` 로 한 번 넘겨 주면 localStorage 에 남아 계속 쓸 수 있다.
 *
 * ## ⚠️ 저장된 주소는 **버려질 수 있어야 한다**
 *
 * 실사고 — 임시 주소를 저장해 둔 브라우저가 **그 뒤로 계속 옛 서버를 쳤다.**
 * 그 서버는 옛 코드를 돌고 있어서, 정식 서버에서는 정상 처리되는 사진이
 * 화면에서는 "사진이 너무 작습니다"로 떨어졌다. 같은 요청인데 결과가 갈리니
 * 원인을 찾기 어려웠다.
 *
 * 그래서 저장된 주소는 **빌드 기본값과 다를 때만** 쓰고, 그 주소가 죽으면
 * (`clearSavedApiBase()`) 즉시 버리고 기본값으로 돌아간다. 기본값은 이제
 * 바뀌지 않는 정식 도메인이라 저장해 둘 이유가 없다.
 */
const BUILD_DEFAULT = (import.meta.env.VITE_API_BASE || '/api').replace(/\/+$/, '');

function resolveApiBase(): string {
  const strip = (v: string) => v.replace(/\/+$/, '');
  try {
    const fromQuery = new URLSearchParams(window.location.search).get('api');
    if (fromQuery) {
      const v = strip(fromQuery);
      if (v === BUILD_DEFAULT) window.localStorage.removeItem(API_BASE_KEY);
      else window.localStorage.setItem(API_BASE_KEY, v);
      return v;
    }
    const saved = window.localStorage.getItem(API_BASE_KEY);
    if (saved && strip(saved) !== BUILD_DEFAULT) return strip(saved);
  } catch {
    // localStorage 차단(프라이빗 모드 등) — 빌드 기본값으로 계속 간다
  }
  return BUILD_DEFAULT;
}

/** 저장된 임시 주소를 버린다. 그 주소가 죽었을 때 호출한다. */
export function clearSavedApiBase(): boolean {
  try {
    const saved = window.localStorage.getItem(API_BASE_KEY);
    if (saved && saved.replace(/\/+$/, '') !== BUILD_DEFAULT) {
      window.localStorage.removeItem(API_BASE_KEY);
      return true;
    }
  } catch {
    // 무시
  }
  return false;
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
    // 저장해 둔 임시 주소가 죽은 경우 — 버리고 정식 주소로 한 번 더 시도한다.
    // 이걸 안 하면 사용자는 페이지를 아무리 새로 고쳐도 죽은 서버만 계속 친다.
    if (clearSavedApiBase()) {
      try {
        r = await fetch(`${BUILD_DEFAULT}/jobs`, { method: 'POST', body: form });
        window.location.reload();
      } catch {
        throw new ApiUnreachableError(apiBase);
      }
    } else {
      throw new ApiUnreachableError(apiBase);
    }
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
      /**
       * 카메라 ↔ 피사체 거리. 카드 앵커가 푼 바닥 평면에서 나오므로
       * 기준물이 없으면 전부 null 이다(단안 스케일 모호성).
       */
      distance?: {
        object_mm?: number | null;
        card_mm?: number | null;
        source?: string | null;
        /** 거리 추정 모드(outline)에서 채워지는 값들 */
        range_mm?: [number, number];
        relative_sigma?: number;
        assumed_long_mm?: number;
        size_source?: 'user_input' | 'product_prior';
        focal_source?: string;
      };
      reconstruction?: JobReconstructionMeta;
      /** outline(에라토스테네스) 경로가 채우는 원본 크기 */
      image_size?: { width: number; height: number };
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
