const prefix = '/api';

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

export type JobDto = {
  id: string;
  status: string;
  result: {
    mass_est_g: number;
    confidence_tier: string;
    confidence_pct: number;
    mass_range?: { min_g: number; estimate_g: number; max_g: number };
    V_hull_mm3: number;
    V_adj_mm3: number;
    algorithm_version: string;
  } | null;
  error: { code: string; message: string } | null;
};
