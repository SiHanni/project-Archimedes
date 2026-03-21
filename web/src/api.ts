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
};

export type JobWorkflowMeta = {
  error_severity?: 'none' | 'soft' | 'hard';
  suggested_action?: 'retry_one_view' | 'continue_low_confidence' | string;
  retry_views?: string[];
  degraded_reasons?: string[];
};

export type JobDto = {
  id: string;
  status: string;
  result: {
    mass_est_g: number;
    confidence_tier: string;
    confidence_pct: number;
    mass_range?: { min_g: number; estimate_g: number; max_g: number } | null;
    V_hull_mm3: number;
    V_adj_mm3: number;
    algorithm_version: string;
    meta?: { sanity?: JobSanityMeta; workflow?: JobWorkflowMeta };
  } | null;
  error: {
    code: string;
    message: string;
    retryViews?: string[];
    errorSeverity?: 'soft' | 'hard';
    suggestedAction?: string | null;
  } | null;
};
