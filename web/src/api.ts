const rawApiBase = import.meta.env.VITE_API_BASE as string | undefined;
const prefix = (rawApiBase || "/api").replace(/\/+$/, "");

export async function postJob(form: FormData): Promise<{ id: string; status: string }> {
  const r = await fetch(`${prefix}/jobs`, {
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
  const r = await fetch(`${prefix}/jobs/${id}`);
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
    meta?: {
      capture_mode?: string;
      sanity?: JobSanityMeta;
      scale_fusion?: JobScaleFusionMeta;
      reconstruction?: JobReconstructionMeta;
    };
  } | null;
  error: { code: string; message: string } | null;
};
