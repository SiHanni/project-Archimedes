/** Docker nginx 프록시(`/api`→`/v1`) 또는 Vercel 빌드 시 `VITE_API_BASE`(터널 `/v1`) */
export const apiBase = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '');

export async function postJob(form: FormData): Promise<{ id: string; status: string }> {
  const r = await fetch(`${apiBase}/jobs`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

export async function getJob(id: string): Promise<JobDto> {
  const r = await fetch(`${apiBase}/jobs/${id}`);
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
