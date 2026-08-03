import { useCallback, useState } from 'react';
import { getJob, postJob, type JobDto } from '../api';

const VIEWS = ['front', 'top', 'left', 'right', 'back'] as const;
const VIEW_LABEL: Record<(typeof VIEWS)[number], string> = {
  front: '정면',
  top: '상단',
  left: '좌측',
  right: '우측',
  back: '후면',
};

/** worker `constants.MATERIALS` 와 동일해야 함 (금속별로 함량 표기가 다름) */
const METALS = [
  { value: 'gold', label: '금' },
  { value: 'silver', label: '은' },
  { value: 'platinum', label: '백금' },
] as const;

const PURITY_BY_METAL: Record<string, { value: string; label: string }[]> = {
  gold: [
    { value: '24k', label: '24K (순금)' },
    { value: '22k', label: '22K' },
    { value: '18k', label: '18K' },
    { value: '14k', label: '14K' },
    { value: '10k', label: '10K' },
  ],
  silver: [
    { value: 'sterling', label: '925 (실버)' },
    { value: 'fine', label: '999 (순은)' },
  ],
  platinum: [
    { value: 'pt950', label: 'Pt950' },
    { value: 'pt900', label: 'Pt900' },
    { value: 'pt999', label: 'Pt999 (순백금)' },
  ],
};

export function HomePage() {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const [knows, setKnows] = useState<'yes' | 'no' | null>(null);
  const [refWeight, setRefWeight] = useState('');
  const [captureMode, setCaptureMode] = useState<'single' | 'multiview'>('single');
  const [singleFile, setSingleFile] = useState<File | undefined>();
  const [files, setFiles] = useState<Partial<Record<(typeof VIEWS)[number], File>>>({});
  const [metal, setMetal] = useState('gold');
  const [purity, setPurity] = useState('18k');
  const [product, setProduct] = useState('ring');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<JobDto | null>(null);

  const onFile = (v: (typeof VIEWS)[number], f: File | undefined) => {
    setFiles((prev) => ({ ...prev, [v]: f }));
  };

  /** 금속을 바꾸면 함량 표기 체계가 달라지므로 첫 항목으로 재설정한다 (예: 금 18k → 백금 pt950) */
  const onMetalChange = (next: string) => {
    setMetal(next);
    const first = PURITY_BY_METAL[next]?.[0]?.value;
    if (first) setPurity(first);
  };

  const poll = useCallback(async (id: string) => {
    for (let i = 0; i < 120; i++) {
      const j = await getJob(id);
      setJob(j);
      if (j.status === 'completed' || j.status === 'failed') return;
      await new Promise((r) => setTimeout(r, 1500));
    }
    setErr('시간 초과 — 나중에 job id로 다시 조회해 보세요.');
  }, []);

  const submit = async () => {
    setErr(null);
    setBusy(true);
    setJob(null);
    try {
      const fd = new FormData();
      if (captureMode === 'single') {
        if (!singleFile) throw new Error('사진을 선택하세요.');
        fd.append('image', singleFile);
      } else {
        for (const v of VIEWS) {
          if (!files[v]) throw new Error(`${VIEW_LABEL[v]} 이미지를 선택하세요.`);
        }
        for (const v of VIEWS) {
          fd.append(v, files[v]!);
        }
      }
      fd.append('capture_mode', captureMode);
      fd.append('metal', metal);
      fd.append('purity', purity);
      fd.append('product_k', product);
      if (knows === 'yes' && refWeight.trim()) {
        fd.append('reference_weight_g', refWeight.trim());
      }
      if (knows) fd.append('knows_weight', knows);
      const { id } = await postJob(fd);
      await poll(id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  /** 서버 meta 없이 올라온 구버전/비정상 JSON 대비 — 일반 귀금속에서 거의 안 나오는 무게는 숫자 숨김 */
  const CLIENT_ABSURD_MASS_G = 350;

  const tier = job?.result?.confidence_tier;
  const hideQuote = tier === 'low';
  const sanity = job?.result?.meta?.sanity;
  const fusion = job?.result?.meta?.scale_fusion;
  const recon = job?.result?.meta?.reconstruction;
  const massG = job?.result?.mass_est_g;
  const hideMass =
    Boolean(sanity?.suppress_mass_display) ||
    (typeof massG === 'number' && massG > CLIENT_ABSURD_MASS_G);

  return (
    <div>
      {step === 0 && (
        <section>
          <h2>귀금속의 무게를 알고 계신가요?</h2>
          <p style={{ color: '#64748b', fontSize: '0.92rem' }}>
            저울 위에 올려 찍은 사진은 분석에 맞지 않을 수 있어요. 평평한 곳에 내려놓고 촬영·업로드해 주세요. (§9.1)
          </p>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem' }}>
            <button type="button" onClick={() => { setKnows('no'); setStep(1); }}>
              아니요 / 잘 모르겠어요
            </button>
            <button type="button" onClick={() => { setKnows('yes'); setStep(1); }}>
              네, 알고 있어요
            </button>
          </div>
        </section>
      )}

      {step === 1 && (
        <section>
          <h2>촬영·업로드 전 체크</h2>
          <ul style={{ color: '#334155' }}>
            <li>
              <strong>신용카드를 함께 찍어 주세요.</strong> 카드는 크기가 규격(85.6×54mm)으로 정해져 있어
              사진 속 실제 크기를 재는 <strong>기준자</strong>가 됩니다. 없으면 정확도가 크게 떨어집니다.
            </li>
            <li>카드와 귀금속을 <strong>같은 바닥 면</strong>에 나란히 둡니다.</li>
            <li>카드가 살짝 <strong>비스듬히</strong> 보이게 찍으면 거리 추정이 더 정확해집니다.</li>
            <li>저울 위에 올린 상태로 찍지 않습니다.</li>
          </ul>
          {knows === 'yes' && (
            <label style={{ display: 'block', marginTop: '1rem' }}>
              참고 무게 (g, 선택)
              <input
                type="number"
                step="0.01"
                value={refWeight}
                onChange={(e) => setRefWeight(e.target.value)}
                style={{ display: 'block', marginTop: '0.35rem', width: '100%', maxWidth: 280 }}
              />
            </label>
          )}
          <button type="button" style={{ marginTop: '1rem' }} onClick={() => setStep(2)}>
            다음
          </button>
        </section>
      )}

      {step === 2 && (
        <section>
          <h2>이미지 업로드</h2>
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', fontSize: '0.92rem' }}>
            <label>
              <input
                type="radio"
                checked={captureMode === 'single'}
                onChange={() => setCaptureMode('single')}
              />{' '}
              사진 1장 <span style={{ color: '#64748b' }}>(기본)</span>
            </label>
            <label>
              <input
                type="radio"
                checked={captureMode === 'multiview'}
                onChange={() => setCaptureMode('multiview')}
              />{' '}
              5방향 <span style={{ color: '#64748b' }}>(고신뢰)</span>
            </label>
          </div>
          <div style={{ display: 'grid', gap: '0.75rem', marginBottom: '1rem' }}>
            {captureMode === 'single' ? (
              <label style={{ display: 'block' }}>
                귀금속 + 신용카드가 함께 나온 사진
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setSingleFile(e.target.files?.[0])}
                  style={{ display: 'block', marginTop: '0.25rem' }}
                />
              </label>
            ) : (
              VIEWS.map((v) => (
                <label key={v} style={{ display: 'block' }}>
                  {VIEW_LABEL[v]}
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => onFile(v, e.target.files?.[0])}
                    style={{ display: 'block', marginTop: '0.25rem' }}
                  />
                </label>
              ))
            )}
          </div>
          <div style={{ display: 'grid', gap: '0.5rem', maxWidth: 320, marginBottom: '1rem' }}>
            <label>
              금속
              <select value={metal} onChange={(e) => onMetalChange(e.target.value)} style={{ display: 'block', width: '100%' }}>
                {METALS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </label>
            <label>
              함량
              <select value={purity} onChange={(e) => setPurity(e.target.value)} style={{ display: 'block', width: '100%' }}>
                {(PURITY_BY_METAL[metal] ?? []).map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </label>
            <label>
              형태
              <select value={product} onChange={(e) => setProduct(e.target.value)} style={{ display: 'block', width: '100%' }}>
                <option value="ring">반지</option>
                <option value="necklace">목걸이</option>
                <option value="chain">체인</option>
                <option value="bracelet">팔찌</option>
                <option value="pendant">펜던트</option>
                <option value="earring">귀걸이</option>
                <option value="other">기타</option>
              </select>
            </label>
            <p style={{ fontSize: '0.82rem', color: '#64748b', margin: 0 }}>
              귀걸이는 반드시 <strong>귀걸이</strong>로 선택하세요. (일반 금 귀걸이는 흔히 <strong>약 3–5g</strong> 전후입니다.) 다른 형태로 고르면
              보정이 맞지 않아 수치가 크게 어긋날 수 있어요.
            </p>
          </div>
          <button type="button" disabled={busy} onClick={() => void submit()}>
            {busy ? '처리 중…' : '분석 요청'}
          </button>
        </section>
      )}

      {err && (
        <p style={{ color: '#b91c1c', marginTop: '1rem', whiteSpace: 'pre-wrap' }}>{err}</p>
      )}

      {job && (
        <section style={{ marginTop: '1.5rem', padding: '1rem', background: '#fff', borderRadius: 8 }}>
          <h3>결과</h3>
          <p>상태: <strong>{job.status}</strong></p>
          {job.error && (
            <p style={{ color: '#b91c1c' }}>
              {job.error.code}: {job.error.message}
            </p>
          )}
          {job.result && (
            <>
              {hideMass ? (
                <p style={{ color: '#b45309' }}>
                  추정 무게가 <strong>비현실적으로 크게</strong> 나와 숫자는 표시하지 않습니다.
                  카드 옆 바닥에 나란히 두고(카드 위 X), 귀금속이 선명히 보이게 다시 촬영해 주세요.
                  {sanity?.sanity_mass_cap_g != null && (
                    <span> (서버 참고 상한 약 {sanity.sanity_mass_cap_g} g)</span>
                  )}
                  {!sanity?.suppress_mass_display && typeof massG === 'number' && (
                    <span style={{ display: 'block', marginTop: '0.35rem', fontSize: '0.85rem' }}>
                      (내부 추정값이 비정상적으로 큽니다. 앱·워커를 최신으로 올렸는지도 확인해 주세요.)
                    </span>
                  )}
                </p>
              ) : (
                <p>
                  추정 무게: <strong>{job.result.mass_est_g.toFixed(3)} g</strong>
                </p>
              )}
              <p>신뢰도 등급: {job.result.confidence_tier} ({job.result.confidence_pct}%)</p>
              {!hideMass && job.result.mass_range && (
                <p>
                  범위: {job.result.mass_range.min_g.toFixed(2)} ~ {job.result.mass_range.max_g.toFixed(2)} g
                </p>
              )}
              {(fusion || recon) && (
                <div
                  style={{
                    marginTop: '0.75rem',
                    padding: '0.6rem 0.75rem',
                    background: '#f8fafc',
                    borderRadius: 6,
                    fontSize: '0.85rem',
                    color: '#334155',
                  }}
                >
                  <strong style={{ display: 'block', marginBottom: '0.35rem' }}>측정값</strong>
                  {fusion?.card_distance_mm != null && (
                    <div>
                      카드까지 거리: {fusion.card_distance_mm.toFixed(0)} mm
                      {fusion.depth_rmse_mm != null && (
                        <span style={{ color: '#64748b' }}>
                          {' '}(거리 오차 ±{fusion.depth_rmse_mm.toFixed(1)} mm)
                        </span>
                      )}
                    </div>
                  )}
                  {recon?.length_mm != null && recon?.width_mm != null && (
                    <div>
                      실제 크기: {recon.length_mm.toFixed(1)} × {recon.width_mm.toFixed(1)} mm
                    </div>
                  )}
                  {recon?.h_mean_mm != null && (
                    <div>
                      평균 두께: {recon.h_mean_mm.toFixed(1)} mm
                      {recon.thickness_clamp && (
                        <span style={{ color: '#b45309' }}> (관측 실패 — 기준값 가정)</span>
                      )}
                    </div>
                  )}
                  {fusion?.anchor_used === false && (
                    <div style={{ color: '#b45309' }}>기준물(신용카드) 없이 추정한 값입니다.</div>
                  )}
                </div>
              )}
              {sanity?.warnings && sanity.warnings.length > 0 && (
                <ul style={{ color: '#92400e', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  {sanity.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
              <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
                {!hideMass && (
                  <>
                    V_hull={job.result.V_hull_mm3} mm³, V_adj={job.result.V_adj_mm3} mm³,{' '}
                  </>
                )}
                {job.result.algorithm_version}
              </p>
              {hideQuote ? (
                <p style={{ marginTop: '0.75rem', color: '#92400e' }}>
                  신뢰도가 낮아 참고 시세·원화 견적은 표시하지 않습니다. (§14.1)
                </p>
              ) : (
                <p style={{ marginTop: '0.75rem', color: '#64748b' }}>
                  시세 연동 전 — 금액 견적 UI는 추후 연결합니다.
                </p>
              )}
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '1rem' }}>
                본 결과는 감정·법적 효력이 없는 참고 추정입니다.
              </p>
            </>
          )}
        </section>
      )}
    </div>
  );
}
